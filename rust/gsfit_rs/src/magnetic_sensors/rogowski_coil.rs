//! `magnetics/rogowski_coil(i1)`: the Rogowski coils, with the custom key `gaps(i2)`

use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use geo::{Contains, Coord, LineString, Point, Polygon};
use imas_rs::ids::magnetics::{MagneticsRogowski, MagneticsRogowskiGaps, Rphiz0dStatic, SignalFlt1dValidity};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::{EquilibriumConstraints0dIpLike, EquilibriumGreensRogowskiCoil};
use ndarray::{Array1, Array2, Axis, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Number of points along each gap at which the poloidal field is integrated, by the trapezium rule
const N_GAP_POINTS: usize = 50;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append a Rogowski coil to `magnetics/rogowski_coil`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the Rogowski coil, unique within the Rogowski coils, e.g. `"INIVC000"`
    /// * `r` - major radius of each point of the coil's contour (1d array), [metre]
    /// * `z` - height of each point of the coil's contour (1d array), [metre]
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `current` - the measured current (1d array), [ampere]
    /// * `gaps_name` - (optional) short identifier for each gap in the coil's winding, unique within the coil
    /// * `gaps_r_start` - (optional) major radius of the start of each gap (1d array), [metre]
    /// * `gaps_r_end` - (optional) major radius of the end of each gap (1d array), [metre]
    /// * `gaps_z_start` - (optional) height of the start of each gap (1d array), [metre]
    /// * `gaps_z_end` - (optional) height of the end of each gap (1d array), [metre]
    ///
    /// Each gap is the straight line from its start to its end; the documentation of `rogowski_coil(i1)/gaps(i2)/r_start` says
    /// which way it runs. A coil with no gaps leaves the gap arguments out.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, r, z, time, current, gaps_name=Vec::new(), gaps_r_start=None, gaps_r_end=None, gaps_z_start=None, gaps_z_end=None))]
    pub fn add_rogowski_coil(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        current: PyReadonlyArray1<f64>,
        gaps_name: Vec<String>,
        gaps_r_start: Option<PyReadonlyArray1<f64>>,
        gaps_r_end: Option<PyReadonlyArray1<f64>>,
        gaps_z_start: Option<PyReadonlyArray1<f64>>,
        gaps_z_end: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let position_r: Array1<f64> = r.to_owned_array();
        let position_z: Array1<f64> = z.to_owned_array();
        let current_time: Array1<f64> = time.to_owned_array();
        let current_data: Array1<f64> = current.to_owned_array();

        // Fallback when a gap argument is not supplied: no gaps
        let gaps_r_start_ndarray: Array1<f64> = match gaps_r_start {
            Some(gaps_r_start_readonly) => gaps_r_start_readonly.to_owned_array(),
            None => Array1::zeros(0),
        };
        let gaps_r_end_ndarray: Array1<f64> = match gaps_r_end {
            Some(gaps_r_end_readonly) => gaps_r_end_readonly.to_owned_array(),
            None => Array1::zeros(0),
        };
        let gaps_z_start_ndarray: Array1<f64> = match gaps_z_start {
            Some(gaps_z_start_readonly) => gaps_z_start_readonly.to_owned_array(),
            None => Array1::zeros(0),
        };
        let gaps_z_end_ndarray: Array1<f64> = match gaps_z_end {
            Some(gaps_z_end_readonly) => gaps_z_end_readonly.to_owned_array(),
            None => Array1::zeros(0),
        };

        // Run the Rust method
        self.add_rogowski_coil_rs(
            name,
            position_r,
            position_z,
            current_time,
            current_data,
            gaps_name,
            gaps_r_start_ndarray,
            gaps_r_end_ndarray,
            gaps_z_start_ndarray,
            gaps_z_end_ndarray,
        )
        .map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append a Rogowski coil to `magnetics/rogowski_coil`. This does the work of the Python method `add_rogowski_coil`, so that
    /// Rust code can add Rogowski coils too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the Rogowski coil, unique within the Rogowski coils, e.g. `"INIVC000"`
    /// * `position_r` - major radius of each point of the coil's contour, [metre]
    /// * `position_z` - height of each point of the coil's contour, [metre]
    /// * `current_time` - the experimental timebase, [second]
    /// * `current_data` - the measured current, [ampere]
    /// * `gaps_name` - short identifier for each gap in the coil's winding, unique within the coil; empty for none
    /// * `gaps_r_start` - major radius of the start of each gap, [metre]
    /// * `gaps_r_end` - major radius of the end of each gap, [metre]
    /// * `gaps_z_start` - height of the start of each gap, [metre]
    /// * `gaps_z_end` - height of the end of each gap, [metre]
    ///
    /// # Errors
    /// Why the Rogowski coil cannot be added: an empty or repeated name, a contour of fewer than 3 points or with a point without a
    /// positive radius, a current without one value per time, or a gap which is unnamed, repeated, without a positive radius at
    /// both ends, or of no length.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rogowski_coil_rs(
        &mut self,
        name: &str,
        position_r: Array1<f64>,
        position_z: Array1<f64>,
        current_time: Array1<f64>,
        current_data: Array1<f64>,
        gaps_name: Vec<String>,
        gaps_r_start: Array1<f64>,
        gaps_r_end: Array1<f64>,
        gaps_z_start: Array1<f64>,
        gaps_z_end: Array1<f64>,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("Rogowski coil: `name` is empty; a Rogowski coil needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device"
        let n_rogowski_coil: usize = self.magnetics_ids.rogowski_coil.len();
        for i_rogowski_coil in 0..n_rogowski_coil {
            if self.magnetics_ids.rogowski_coil[i_rogowski_coil].name == name {
                return Err(format!(
                    "Rogowski coil `{name}`: a Rogowski coil with this name already exists, at `magnetics/rogowski_coil({i_rogowski_coil})`"
                ));
            }
        }

        let n_position: usize = position_r.len();
        if position_z.len() != n_position {
            return Err(format!(
                "Rogowski coil `{name}`: `r` has {n_position} point(s) but `z` has {}; the contour needs one `z` per `r`",
                position_z.len()
            ));
        }

        // The coil measures the current inside its contour, so the contour has to enclose an area
        if n_position < 3 {
            return Err(format!(
                "Rogowski coil `{name}`: the contour has {n_position} point(s); at least 3 are needed to enclose a current"
            ));
        }

        let mut positions: Vec<Rphiz0dStatic> = Vec::with_capacity(n_position);
        for i_position in 0..n_position {
            let is_usable_position: bool = position_r[i_position] > 0.0 && position_r[i_position].is_finite() && position_z[i_position].is_finite();
            if !is_usable_position {
                return Err(format!(
                    "Rogowski coil `{name}`: contour point {i_position} has r = {} and z = {} metre; r must be positive, and both finite",
                    position_r[i_position], position_z[i_position]
                ));
            }

            positions.push(Rphiz0dStatic {
                r: position_r[i_position],
                z: position_z[i_position],
                ..Default::default()
            });
        }

        if current_time.len() != current_data.len() {
            return Err(format!(
                "Rogowski coil `{name}`: `time` has {} point(s) but `current` has {}; a signal needs one value per time",
                current_time.len(),
                current_data.len()
            ));
        }

        if current_time.is_empty() {
            return Err(format!("Rogowski coil `{name}`: `time` is empty; the current needs at least one time"));
        }

        let n_gap: usize = gaps_name.len();
        if gaps_r_start.len() != n_gap || gaps_r_end.len() != n_gap || gaps_z_start.len() != n_gap || gaps_z_end.len() != n_gap {
            return Err(format!(
                "Rogowski coil `{name}`: `gaps_name`, `gaps_r_start`, `gaps_r_end`, `gaps_z_start` and `gaps_z_end` have {n_gap}, {}, {}, {} \
                 and {} gap(s); each gap needs one of each",
                gaps_r_start.len(),
                gaps_r_end.len(),
                gaps_z_start.len(),
                gaps_z_end.len()
            ));
        }

        let mut gaps: Vec<MagneticsRogowskiGaps> = Vec::with_capacity(n_gap);
        for i_gap in 0..n_gap {
            let gap_name: &str = &gaps_name[i_gap];
            if gap_name.is_empty() {
                return Err(format!("Rogowski coil `{name}`: gap {i_gap} has an empty name; a gap needs a short identifier"));
            }

            for j_gap in 0..i_gap {
                if gaps_name[j_gap] == gap_name {
                    return Err(format!("Rogowski coil `{name}`: gaps {j_gap} and {i_gap} are both named `{gap_name}`"));
                }
            }

            let gap_r_start: f64 = gaps_r_start[i_gap];
            let gap_r_end: f64 = gaps_r_end[i_gap];
            let gap_z_start: f64 = gaps_z_start[i_gap];
            let gap_z_end: f64 = gaps_z_end[i_gap];
            let is_usable_gap: bool =
                gap_r_start > 0.0 && gap_r_start.is_finite() && gap_r_end > 0.0 && gap_r_end.is_finite() && gap_z_start.is_finite() && gap_z_end.is_finite();
            if !is_usable_gap {
                return Err(format!(
                    "Rogowski coil `{name}`, gap `{gap_name}`: it runs from (r, z) = ({gap_r_start}, {gap_z_start}) to ({gap_r_end}, {gap_z_end}) \
                     metre; r must be positive, and both finite"
                ));
            }

            // A gap with no length would leave nothing out of the winding
            if gap_r_start == gap_r_end && gap_z_start == gap_z_end {
                return Err(format!(
                    "Rogowski coil `{name}`, gap `{gap_name}`: it starts and ends at (r, z) = ({gap_r_start}, {gap_z_start}) metre, so has no length"
                ));
            }

            gaps.push(MagneticsRogowskiGaps {
                name: gap_name.to_string(),
                r_start: gap_r_start,
                r_end: gap_r_end,
                z_start: gap_z_start,
                z_end: gap_z_end,
            });
        }

        self.magnetics_ids.rogowski_coil.push(MagneticsRogowski {
            name: name.to_string(),
            position: positions,
            current: SignalFlt1dValidity {
                data: current_data,
                time: current_time,
                ..Default::default()
            },
            gaps,
            ..Default::default()
        });

        Ok(())
    }
}

/// Rust only methods, copying the Rogowski coils into the equilibrium IDS
impl Equilibrium {
    /// Copy the Rogowski coils' currents from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/rogowski_coil(i1)`, a custom key. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the Rogowski coils in `magnetics/rogowski_coil`
    ///
    /// # Errors
    /// A Rogowski coil in the fit settings, `code/parameters_parsed/constraints/rogowski_coil`, is not in the magnetics IDS, or a Rogowski coil's
    /// measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_rogowski_coil(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let rogowski_coils: &Vec<MagneticsRogowski> = &magnetics.magnetics_ids.rogowski_coil;
        let n_rogowski_coil: usize = rogowski_coils.len();

        // Only the Rogowski coils in the fit settings, `code/parameters_parsed/constraints/rogowski_coil`, are copied, and each of them has to
        // be in the magnetics IDS. Checked before anything is copied
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.rogowski_coil(..).name.to_vec();
        let settings_weights: Array1<f64> = self.equilibrium_ids.code.parameters_parsed.constraints.rogowski_coil(..).weight.to_array();
        let n_settings: usize = settings_names.len();
        for i_settings in 0..n_settings {
            let settings_name: &str = &settings_names[i_settings];
            if !rogowski_coils
                .iter()
                .any(|magnetics_sensor: &MagneticsRogowski| magnetics_sensor.name == settings_name)
            {
                return Err(format!(
                    "`{settings_name}` is in the fit settings, `code/parameters_parsed/constraints/rogowski_coil`, but not in the magnetics IDS, `magnetics/rogowski_coil`"
                ));
            }
        }

        // Each Rogowski coil's current is on the experimental timebase, so it is interpolated onto the times of the time-slices
        // `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut current_measured: Array2<f64> = Array2::from_elem((n_time, n_rogowski_coil), f64::NAN); // shape = (n_time, n_rogowski_coil)
        for i_rogowski_coil in 0..n_rogowski_coil {
            let rogowski_coil_name: &str = &rogowski_coils[i_rogowski_coil].name;
            if !settings_names.iter().any(|settings_name: &String| settings_name.as_str() == rogowski_coil_name) {
                continue;
            }
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(
                rogowski_coils[i_rogowski_coil].current.time.clone(),
                rogowski_coils[i_rogowski_coil].current.data.clone(),
            )
            .map_err(|error: interpolation::Error| {
                format!("cannot interpolate `magnetics/rogowski_coil({i_rogowski_coil})/current` ({rogowski_coil_name}): {error:?}")
            })?;
            let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
                format!(
                    "cannot interpolate `magnetics/rogowski_coil({i_rogowski_coil})/current` ({rogowski_coil_name}) onto the reconstruction times: {error:?}"
                )
            })?;
            current_measured.slice_mut(s![.., i_rogowski_coil]).assign(&measured);
        }

        // Rebuilt from the Rogowski coils in the fit settings, in the order of the magnetics IDS, so that calling this twice replaces them
        // instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight`, found by
        // its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraints0dIpLike> =
                std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.rogowski_coil);
            let mut rogowski_coil_constraints: Vec<EquilibriumConstraints0dIpLike> = Vec::with_capacity(n_settings);
            for i_rogowski_coil in 0..n_rogowski_coil {
                let rogowski_coil_name: &str = &rogowski_coils[i_rogowski_coil].name;
                let Some(i_settings) = settings_names
                    .iter()
                    .position(|settings_name: &String| settings_name.as_str() == rogowski_coil_name)
                else {
                    continue;
                };
                let mut constraint: EquilibriumConstraints0dIpLike = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraints0dIpLike| settings_constraint.name == rogowski_coil_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => EquilibriumConstraints0dIpLike {
                        name: rogowski_coil_name.to_string(),
                        weight: settings_weights[i_settings],
                        ..Default::default()
                    },
                };
                constraint.measured = current_measured[[i_time, i_rogowski_coil]];
                constraint.source = format!("magnetics/rogowski_coil({i_rogowski_coil})/current");
                constraint.time_measurement = times_to_reconstruct[i_time];
                rogowski_coil_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.rogowski_coil = rogowski_coil_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the Rogowski coils' Greens tables
impl Equilibrium {
    /// Calculate the Rogowski coils' Greens tables, `equilibrium/greens/rogowski_coil(i1)`, one for each coil in the fit settings,
    /// `code/parameters_parsed/constraints/rogowski_coil`, in the order of `magnetics/rogowski_coil`. This does part of the work of the
    /// Python method `greens_with_magnetics`; see it for what is stored, and for the checks on `pf_active`, `pf_passive` and the fit
    /// settings.
    ///
    /// Each table is the current the coil measures: the current inside its contour, `position`, less `integral(b . d_l) / mu_0`
    /// along each of its gaps, from the gap's start to its end. A grid cell, or a conductor's filament, is inside when its centre
    /// is. The integral along a gap is the trapezium rule over `N_GAP_POINTS` points.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the Rogowski coils in `magnetics/rogowski_coil`
    /// * `pf_active` - the PF coils, in the order of `greens/grid/pf_active`
    /// * `pf_passive` - the passive conductors, in the order of `greens/grid/pf_passive`
    pub fn greens_with_magnetics_rogowski_coil(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) {
        let rogowski_coils: &Vec<MagneticsRogowski> = &magnetics.magnetics_ids.rogowski_coil;
        let n_rogowski_coil: usize = rogowski_coils.len();

        // Only the Rogowski coils in the fit settings, `code/parameters_parsed/constraints/rogowski_coil`, which `copy_from_magnetics_ids` copies,
        // have Greens tables. `greens_with_magnetics` checked that each of them is in the magnetics IDS
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.rogowski_coil(..).name.to_vec();

        // The grid. `time_slice(0)` because the grid is the same on every time-slice, and `profiles_2d(0)` because GSFit solves
        // on a single rectangular (R, Z) grid. `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives
        // the flattened grid
        let n_r: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_r as usize;
        let n_z: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_z as usize;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let grid_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let grid_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());
        let n_grid: usize = grid_r.len();

        // The PF coils' filaments, which are the same for every Rogowski coil
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();
        let mut coil_filaments_r: Vec<Array1<f64>> = Vec::with_capacity(n_coil);
        let mut coil_filaments_z: Vec<Array1<f64>> = Vec::with_capacity(n_coil);
        for i_coil in 0..n_coil {
            coil_filaments_r.push(pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array());
            coil_filaments_z.push(pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array());
        }

        // The passive conductors' filaments, which are the same for every Rogowski coil
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut loop_filaments: Vec<PassiveFilaments> = Vec::with_capacity(n_loop);
        let mut n_dof_total: usize = 0;
        for i_loop in 0..n_loop {
            loop_filaments.push(passive_filaments(&pf_passive_loops[i_loop].element));
            n_dof_total += pf_passive_loops[i_loop].dof.len();
        }

        // The trapezium rule's weight for each point along a gap: half at the gap's two ends
        let mut gap_point_weights: Array1<f64> = Array1::ones(N_GAP_POINTS);
        gap_point_weights[0] = 0.5;
        gap_point_weights[N_GAP_POINTS - 1] = 0.5;

        let mut greens_rogowski_coils: Vec<EquilibriumGreensRogowskiCoil> = Vec::with_capacity(n_rogowski_coil);
        for i_rogowski_coil in 0..n_rogowski_coil {
            if !settings_names
                .iter()
                .any(|settings_name: &String| *settings_name == rogowski_coils[i_rogowski_coil].name)
            {
                continue;
            }
            // The coil's contour, which `Polygon` closes
            let positions: &Vec<Rphiz0dStatic> = &rogowski_coils[i_rogowski_coil].position;
            let n_position: usize = positions.len();
            let mut contour_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_position);
            for i_position in 0..n_position {
                contour_coordinates.push(Coord {
                    x: positions[i_position].r,
                    y: positions[i_position].z,
                });
            }
            let contour: Polygon = Polygon::new(
                LineString::new(contour_coordinates),
                vec![], // No holes
            );

            // The points along each gap, from its start to its end, and the step between them
            let gaps: &Vec<MagneticsRogowskiGaps> = &rogowski_coils[i_rogowski_coil].gaps;
            let n_gap: usize = gaps.len();
            let mut gaps_r: Vec<Array1<f64>> = Vec::with_capacity(n_gap);
            let mut gaps_z: Vec<Array1<f64>> = Vec::with_capacity(n_gap);
            let mut gaps_d_r: Vec<f64> = Vec::with_capacity(n_gap);
            let mut gaps_d_z: Vec<f64> = Vec::with_capacity(n_gap);
            for i_gap in 0..n_gap {
                let gap_r: Array1<f64> = Array1::linspace(gaps[i_gap].r_start, gaps[i_gap].r_end, N_GAP_POINTS);
                let gap_z: Array1<f64> = Array1::linspace(gaps[i_gap].z_start, gaps[i_gap].z_end, N_GAP_POINTS);
                gaps_d_r.push(gap_r[1] - gap_r[0]);
                gaps_d_z.push(gap_z[1] - gap_z[0]);
                gaps_r.push(gap_r);
                gaps_z.push(gap_z);
            }

            // With the plasma: the grid cells inside the contour, less `integral(b . d_l) / mu_0` along the gaps
            let mut g_plasma_inside: Array1<f64> = Array1::zeros(n_grid); // 1 for a grid cell inside the contour; shape = [n_z * n_r]
            for i_grid in 0..n_grid {
                if contour.contains(&Point::new(grid_r[i_grid], grid_z[i_grid])) {
                    g_plasma_inside[i_grid] = 1.0;
                }
            }
            let mut g_plasma_gaps: Array1<f64> = Array1::zeros(n_grid); // shape = [n_z * n_r]
            let mut g_d_plasma_d_z_gaps: Array1<f64> = Array1::zeros(n_grid); // shape = [n_z * n_r]
            for i_gap in 0..n_gap {
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    gaps_r[i_gap].clone(),
                    gaps_z[i_gap].clone(),
                    grid_r.clone(),
                    grid_z.clone(),
                    grid_r.clone() * 0.0, // TODO: this should be d_r not 0.0
                    grid_z.clone() * 0.0,
                );
                let g_br: Array2<f64> = greens_calculator.b_r(); // shape = [N_GAP_POINTS, n_z * n_r]
                let g_bz: Array2<f64> = greens_calculator.b_z(); // shape = [N_GAP_POINTS, n_z * n_r]
                let g_d_br_d_z: Array2<f64> = greens_calculator.d_b_r_d_z(); // shape = [N_GAP_POINTS, n_z * n_r]
                let g_d_bz_d_z: Array2<f64> = greens_calculator.d_b_z_d_z(); // shape = [N_GAP_POINTS, n_z * n_r]
                for i_gap_point in 0..N_GAP_POINTS {
                    let weight: f64 = gap_point_weights[i_gap_point];
                    g_plasma_gaps = g_plasma_gaps + &g_br.row(i_gap_point) * weight * gaps_d_r[i_gap] + &g_bz.row(i_gap_point) * weight * gaps_d_z[i_gap];
                    g_d_plasma_d_z_gaps =
                        g_d_plasma_d_z_gaps + &g_d_br_d_z.row(i_gap_point) * weight * gaps_d_r[i_gap] + &g_d_bz_d_z.row(i_gap_point) * weight * gaps_d_z[i_gap];
                }
            }
            let g_plasma: Array1<f64> = g_plasma_inside - g_plasma_gaps / MU_0; // shape = [n_z * n_r]
            // The current inside the contour does not change with a small shift, so only the gaps contribute
            let g_d_plasma_d_z: Array1<f64> = Array1::zeros(n_grid) - g_d_plasma_d_z_gaps / MU_0; // shape = [n_z * n_r]

            // With each PF coil: its filaments inside the contour, less `integral(b . d_l) / mu_0` along the gaps
            let mut g_pf_active: Array1<f64> = Array1::from_elem(n_coil, f64::NAN); // shape = [n_coil]
            for i_coil in 0..n_coil {
                let n_filament: usize = coil_filaments_r[i_coil].len();
                let mut g_inside: f64 = 0.0;
                for i_filament in 0..n_filament {
                    if contour.contains(&Point::new(coil_filaments_r[i_coil][i_filament], coil_filaments_z[i_coil][i_filament])) {
                        g_inside += 1.0;
                    }
                }

                let mut g_gaps: f64 = 0.0;
                for i_gap in 0..n_gap {
                    let greens_calculator: Greens = Greens::sensor_to_conductor(
                        gaps_r[i_gap].clone(),
                        gaps_z[i_gap].clone(),
                        coil_filaments_r[i_coil].clone(),
                        coil_filaments_z[i_coil].clone(),
                        coil_filaments_r[i_coil].clone() * 0.0, // TODO: should this be NaN instead?
                        coil_filaments_z[i_coil].clone() * 0.0,
                    );
                    let g_br: Array1<f64> = greens_calculator.b_r().sum_axis(Axis(1)); // summed over the coil's filaments; shape = [N_GAP_POINTS]
                    let g_bz: Array1<f64> = greens_calculator.b_z().sum_axis(Axis(1)); // summed over the coil's filaments; shape = [N_GAP_POINTS]
                    for i_gap_point in 0..N_GAP_POINTS {
                        let weight: f64 = gap_point_weights[i_gap_point];
                        g_gaps += g_br[i_gap_point] * weight * gaps_d_r[i_gap] + g_bz[i_gap_point] * weight * gaps_d_z[i_gap];
                    }
                }

                g_pf_active[i_coil] = g_inside - g_gaps / MU_0;
            }

            // With each passive degree of freedom: the loop's filaments inside the contour, weighted by the degree of freedom's
            // current distribution, less `integral(b . d_l) / mu_0` along the gaps
            let mut g_pf_passive: Array1<f64> = Array1::from_elem(n_dof_total, f64::NAN); // shape = [n_dof_total]
            let mut i_dof_total: usize = 0;
            for i_loop in 0..n_loop {
                let n_filament: usize = loop_filaments[i_loop].r.len();
                let mut filaments_inside: Array1<f64> = Array1::zeros(n_filament); // 1 for a filament inside the contour; shape = [n_filament]
                for i_filament in 0..n_filament {
                    if contour.contains(&Point::new(loop_filaments[i_loop].r[i_filament], loop_filaments[i_loop].z[i_filament])) {
                        filaments_inside[i_filament] = 1.0;
                    }
                }

                // The field at the points along each gap, from each filament. These depend on the gap and the loop, not on the
                // degree of freedom, so they are calculated once for all of them
                let mut gaps_g_br_filaments: Vec<Array2<f64>> = Vec::with_capacity(n_gap);
                let mut gaps_g_bz_filaments: Vec<Array2<f64>> = Vec::with_capacity(n_gap);
                for i_gap in 0..n_gap {
                    let greens_calculator: Greens = Greens::sensor_to_conductor(
                        gaps_r[i_gap].clone(),
                        gaps_z[i_gap].clone(),
                        loop_filaments[i_loop].r.clone(),
                        loop_filaments[i_loop].z.clone(),
                        loop_filaments[i_loop].r.clone() * 0.0, // TODO: should this be NaN instead?
                        loop_filaments[i_loop].z.clone() * 0.0,
                    );
                    gaps_g_br_filaments.push(greens_calculator.b_r()); // shape = [N_GAP_POINTS, n_filament]
                    gaps_g_bz_filaments.push(greens_calculator.b_z()); // shape = [N_GAP_POINTS, n_filament]
                }

                let n_dof: usize = pf_passive_loops[i_loop].dof.len();
                for i_dof in 0..n_dof {
                    let current_distribution: &Array1<f64> = &pf_passive_loops[i_loop].dof[i_dof].current_distribution;
                    let g_inside: f64 = (&filaments_inside * current_distribution).sum();

                    let mut g_gaps: f64 = 0.0;
                    for i_gap in 0..n_gap {
                        let g_br: Array1<f64> = (&gaps_g_br_filaments[i_gap] * current_distribution).sum_axis(Axis(1)); // shape = [N_GAP_POINTS]
                        let g_bz: Array1<f64> = (&gaps_g_bz_filaments[i_gap] * current_distribution).sum_axis(Axis(1)); // shape = [N_GAP_POINTS]
                        for i_gap_point in 0..N_GAP_POINTS {
                            let weight: f64 = gap_point_weights[i_gap_point];
                            g_gaps += g_br[i_gap_point] * weight * gaps_d_r[i_gap] + g_bz[i_gap_point] * weight * gaps_d_z[i_gap];
                        }
                    }

                    g_pf_passive[i_dof_total] = g_inside - g_gaps / MU_0;
                    i_dof_total += 1;
                }
            }

            greens_rogowski_coils.push(EquilibriumGreensRogowskiCoil {
                name: rogowski_coils[i_rogowski_coil].name.clone(),
                plasma: g_plasma
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_rogowski_coil: failed to reshape `g_plasma` into (n_z, n_r)"),
                d_plasma_d_z: g_d_plasma_d_z
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_rogowski_coil: failed to reshape `g_d_plasma_d_z` into (n_z, n_r)"),
                pf_active: g_pf_active,
                pf_passive: g_pf_passive,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the Rogowski coils' tables instead of repeating them
        self.equilibrium_ids.greens.rogowski_coil = greens_rogowski_coils;
    }
}
