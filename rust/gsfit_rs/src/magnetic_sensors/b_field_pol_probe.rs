//! `magnetics/b_field_pol_probe(i1)`: the poloidal field probes

use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use imas_rs::ids::magnetics::{MagneticsBpolProbe, Rphiz0dStatic, SignalFlt1dValidity};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::{EquilibriumConstraints0dOneLike, EquilibriumGreensBFieldPolProbe};
use ndarray::{Array1, Array2, Axis, array, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append a poloidal field probe to `magnetics/b_field_pol_probe`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the probe, unique within the poloidal field probes, e.g. `"P101"`
    /// * `r` - major radius of the probe's centre, [metre]
    /// * `z` - height of the probe's centre, [metre]
    /// * `poloidal_angle` - angle of the probe's normal, counter-clockwise from `grad(R)`: GSFit's COCOS 13, not the data
    ///   dictionary's COCOS 17; see the `Magnetics` documentation, [radian]
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `field` - the measured field along the probe's normal (1d array), [tesla]
    pub fn add_b_field_pol_probe(
        &mut self,
        name: &str,
        r: f64,
        z: f64,
        poloidal_angle: f64,
        time: PyReadonlyArray1<f64>,
        field: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let field_time: Array1<f64> = time.to_owned_array();
        let field_data: Array1<f64> = field.to_owned_array();

        // Run the Rust method
        self.add_b_field_pol_probe_rs(name, r, z, poloidal_angle, field_time, field_data)
            .map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append a poloidal field probe to `magnetics/b_field_pol_probe`. This does the work of the Python method
    /// `add_b_field_pol_probe`, so that Rust code can add probes too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the probe, unique within the poloidal field probes, e.g. `"P101"`
    /// * `r` - major radius of the probe's centre, [metre]
    /// * `z` - height of the probe's centre, [metre]
    /// * `poloidal_angle` - angle of the probe's normal, counter-clockwise from `grad(R)` (COCOS 13), [radian]
    /// * `field_time` - the experimental timebase, [second]
    /// * `field_data` - the measured field along the probe's normal, [tesla]
    ///
    /// # Errors
    /// Why the probe cannot be added: an empty or repeated name, a position without a positive radius, an angle which is not
    /// finite, or a field without one value per time.
    pub fn add_b_field_pol_probe_rs(
        &mut self,
        name: &str,
        r: f64,
        z: f64,
        poloidal_angle: f64,
        field_time: Array1<f64>,
        field_data: Array1<f64>,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("poloidal field probe: `name` is empty; a probe needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device"
        let n_b_field_pol_probe: usize = self.magnetics_ids.b_field_pol_probe.len();
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            if self.magnetics_ids.b_field_pol_probe[i_b_field_pol_probe].name == name {
                return Err(format!(
                    "poloidal field probe `{name}`: a probe with this name already exists, at `magnetics/b_field_pol_probe({i_b_field_pol_probe})`"
                ));
            }
        }

        let is_usable_position: bool = r > 0.0 && r.is_finite() && z.is_finite();
        if !is_usable_position {
            return Err(format!(
                "poloidal field probe `{name}`: r = {r} and z = {z} metre; r must be positive, and both finite"
            ));
        }

        // The angle is stored as given, in GSFit's COCOS 13, so it is not checked against the data dictionary's [0, 2 * PI]
        if !poloidal_angle.is_finite() {
            return Err(format!(
                "poloidal field probe `{name}`: poloidal_angle = {poloidal_angle} radian; it must be finite"
            ));
        }

        if field_time.len() != field_data.len() {
            return Err(format!(
                "poloidal field probe `{name}`: `time` has {} point(s) but `field` has {}; a signal needs one value per time",
                field_time.len(),
                field_data.len()
            ));
        }

        if field_time.is_empty() {
            return Err(format!("poloidal field probe `{name}`: `time` is empty; the field needs at least one time"));
        }

        self.magnetics_ids.b_field_pol_probe.push(MagneticsBpolProbe {
            name: name.to_string(),
            position: Rphiz0dStatic { r, z, ..Default::default() },
            poloidal_angle,
            field: SignalFlt1dValidity {
                data: field_data,
                time: field_time,
                ..Default::default()
            },
            ..Default::default()
        });

        Ok(())
    }
}

/// Rust only methods, copying the poloidal field probes into the equilibrium IDS
impl Equilibrium {
    /// Copy the poloidal field probes' fields from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/b_field_pol_probe(i1)`. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the poloidal field probes in `magnetics/b_field_pol_probe`
    ///
    /// # Errors
    /// A probe in the fit settings, `code/parameters_parsed/constraints/b_field_pol_probe`, is not in the magnetics IDS, or a probe's
    /// measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_b_field_pol_probe(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let b_field_pol_probes: &Vec<MagneticsBpolProbe> = &magnetics.magnetics_ids.b_field_pol_probe;
        let n_b_field_pol_probe: usize = b_field_pol_probes.len();

        // Only the poloidal field probes in the fit settings, `code/parameters_parsed/constraints/b_field_pol_probe`, are copied, and each of them has to
        // be in the magnetics IDS. Checked before anything is copied
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.b_field_pol_probe(..).name.to_vec();
        let settings_weights: Array1<f64> = self.equilibrium_ids.code.parameters_parsed.constraints.b_field_pol_probe(..).weight.to_array();
        let n_settings: usize = settings_names.len();
        for i_settings in 0..n_settings {
            let settings_name: &str = &settings_names[i_settings];
            if !b_field_pol_probes
                .iter()
                .any(|magnetics_sensor: &MagneticsBpolProbe| magnetics_sensor.name == settings_name)
            {
                return Err(format!(
                    "`{settings_name}` is in the fit settings, `code/parameters_parsed/constraints/b_field_pol_probe`, but not in the magnetics IDS, `magnetics/b_field_pol_probe`"
                ));
            }
        }

        // Each probe's field is on the experimental timebase, so it is interpolated onto the times of the time-slices
        // `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut field_measured: Array2<f64> = Array2::from_elem((n_time, n_b_field_pol_probe), f64::NAN); // shape = (n_time, n_b_field_pol_probe)
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            let b_field_pol_probe_name: &str = &b_field_pol_probes[i_b_field_pol_probe].name;
            if !settings_names
                .iter()
                .any(|settings_name: &String| settings_name.as_str() == b_field_pol_probe_name)
            {
                continue;
            }
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(
                b_field_pol_probes[i_b_field_pol_probe].field.time.clone(),
                b_field_pol_probes[i_b_field_pol_probe].field.data.clone(),
            )
            .map_err(|error: interpolation::Error| {
                format!("cannot interpolate `magnetics/b_field_pol_probe({i_b_field_pol_probe})/field` ({b_field_pol_probe_name}): {error:?}")
            })?;
            let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
                format!("cannot interpolate `magnetics/b_field_pol_probe({i_b_field_pol_probe})/field` ({b_field_pol_probe_name}) onto the reconstruction times: {error:?}")
            })?;
            field_measured.slice_mut(s![.., i_b_field_pol_probe]).assign(&measured);
        }

        // Rebuilt from the poloidal field probes in the fit settings, in the order of the magnetics IDS, so that calling this twice replaces them
        // instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight`, found by
        // its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraints0dOneLike> =
                std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.b_field_pol_probe);
            let mut b_field_pol_probe_constraints: Vec<EquilibriumConstraints0dOneLike> = Vec::with_capacity(n_settings);
            for i_b_field_pol_probe in 0..n_b_field_pol_probe {
                let b_field_pol_probe_name: &str = &b_field_pol_probes[i_b_field_pol_probe].name;
                let Some(i_settings) = settings_names
                    .iter()
                    .position(|settings_name: &String| settings_name.as_str() == b_field_pol_probe_name)
                else {
                    continue;
                };
                let mut constraint: EquilibriumConstraints0dOneLike = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraints0dOneLike| settings_constraint.name == b_field_pol_probe_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => EquilibriumConstraints0dOneLike {
                        name: b_field_pol_probe_name.to_string(),
                        weight: settings_weights[i_settings],
                        ..Default::default()
                    },
                };
                constraint.measured = field_measured[[i_time, i_b_field_pol_probe]];
                constraint.source = format!("magnetics/b_field_pol_probe({i_b_field_pol_probe})/field");
                constraint.time_measurement = times_to_reconstruct[i_time];
                b_field_pol_probe_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.b_field_pol_probe = b_field_pol_probe_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the poloidal field probes' Greens tables
impl Equilibrium {
    /// Calculate the poloidal field probes' Greens tables, `equilibrium/greens/b_field_pol_probe(i1)`, one for each probe in the fit
    /// settings, `code/parameters_parsed/constraints/b_field_pol_probe`, in the order of `magnetics/b_field_pol_probe`. This does part of
    /// the work of the Python method `greens_with_magnetics`; see it for what is stored, and for the checks on `pf_active`, `pf_passive`
    /// and the fit settings.
    ///
    /// Each table is the field along the probe's normal, `b_r * cos(poloidal_angle) + b_z * sin(poloidal_angle)`, with GSFit's
    /// COCOS 13 `poloidal_angle`.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the poloidal field probes in `magnetics/b_field_pol_probe`
    /// * `pf_active` - the PF coils, in the order of `greens/grid/pf_active`
    /// * `pf_passive` - the passive conductors, in the order of `greens/grid/pf_passive`
    pub fn greens_with_magnetics_b_field_pol_probe(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) {
        let b_field_pol_probes: &Vec<MagneticsBpolProbe> = &magnetics.magnetics_ids.b_field_pol_probe;
        let n_b_field_pol_probe: usize = b_field_pol_probes.len();

        // Only the poloidal field probes in the fit settings, `code/parameters_parsed/constraints/b_field_pol_probe`, which `copy_from_magnetics_ids` copies,
        // have Greens tables. `greens_with_magnetics` checked that each of them is in the magnetics IDS
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.b_field_pol_probe(..).name.to_vec();

        // The grid. `time_slice(0)` because the grid is the same on every time-slice, and `profiles_2d(0)` because GSFit solves
        // on a single rectangular (R, Z) grid. `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives
        // the flattened grid
        let n_r: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_r as usize;
        let n_z: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_z as usize;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let grid_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let grid_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // The PF coils
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();

        // The passive conductors' filaments, which are the same for every probe
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut loop_filaments: Vec<PassiveFilaments> = Vec::with_capacity(n_loop);
        let mut n_dof_total: usize = 0;
        for i_loop in 0..n_loop {
            loop_filaments.push(passive_filaments(&pf_passive_loops[i_loop].element));
            n_dof_total += pf_passive_loops[i_loop].dof.len();
        }

        let mut greens_b_field_pol_probes: Vec<EquilibriumGreensBFieldPolProbe> = Vec::with_capacity(n_b_field_pol_probe);
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            if !settings_names
                .iter()
                .any(|settings_name: &String| *settings_name == b_field_pol_probes[i_b_field_pol_probe].name)
            {
                continue;
            }
            let probe_r: f64 = b_field_pol_probes[i_b_field_pol_probe].position.r;
            let probe_z: f64 = b_field_pol_probes[i_b_field_pol_probe].position.z;
            let probe_angle: f64 = b_field_pol_probes[i_b_field_pol_probe].poloidal_angle;

            // With the plasma
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                array![probe_r],
                array![probe_z],
                grid_r.clone(),
                grid_z.clone(),
                grid_r.clone() * 0.0, // TODO: this should be d_r not 0.0
                grid_z.clone() * 0.0,
            );
            let g_br: Array1<f64> = greens_calculator.b_r().sum_axis(Axis(0)); // shape = [n_z * n_r]
            let g_bz: Array1<f64> = greens_calculator.b_z().sum_axis(Axis(0)); // shape = [n_z * n_r]
            let g_plasma: Array1<f64> = g_br * probe_angle.cos() + g_bz * probe_angle.sin(); // shape = [n_z * n_r]
            let g_d_br_d_z: Array1<f64> = greens_calculator.d_b_r_d_z().sum_axis(Axis(0)); // shape = [n_z * n_r]
            let g_d_bz_d_z: Array1<f64> = greens_calculator.d_b_z_d_z().sum_axis(Axis(0)); // shape = [n_z * n_r]
            let g_d_plasma_d_z: Array1<f64> = g_d_br_d_z * probe_angle.cos() + g_d_bz_d_z * probe_angle.sin(); // shape = [n_z * n_r]

            // With each PF coil, summed over the coil's filaments
            let mut g_pf_active: Array1<f64> = Array1::from_elem(n_coil, f64::NAN); // shape = [n_coil]
            for i_coil in 0..n_coil {
                let coil_r: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array();
                let coil_z: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array();
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![probe_r],
                    array![probe_z],
                    coil_r.clone(),
                    coil_z.clone(),
                    coil_r.clone() * 0.0, // TODO: should this be NaN instead?
                    coil_z.clone() * 0.0,
                );
                let g_br: f64 = greens_calculator.b_r().sum();
                let g_bz: f64 = greens_calculator.b_z().sum();
                g_pf_active[i_coil] = g_br * probe_angle.cos() + g_bz * probe_angle.sin();
            }

            // With each passive degree of freedom: the loop's filaments, weighted by the degree of freedom's current distribution
            let mut g_pf_passive: Array1<f64> = Array1::from_elem(n_dof_total, f64::NAN); // shape = [n_dof_total]
            let mut i_dof_total: usize = 0;
            for i_loop in 0..n_loop {
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![probe_r],
                    array![probe_z],
                    loop_filaments[i_loop].r.clone(),
                    loop_filaments[i_loop].z.clone(),
                    loop_filaments[i_loop].r.clone() * 0.0, // TODO: should this be NaN instead?
                    loop_filaments[i_loop].z.clone() * 0.0,
                );
                let g_br_filaments: Array2<f64> = greens_calculator.b_r(); // shape = [1, n_filament]
                let g_bz_filaments: Array2<f64> = greens_calculator.b_z(); // shape = [1, n_filament]

                let n_dof: usize = pf_passive_loops[i_loop].dof.len();
                for i_dof in 0..n_dof {
                    let current_distribution: &Array1<f64> = &pf_passive_loops[i_loop].dof[i_dof].current_distribution;
                    let g_br: f64 = (&g_br_filaments * current_distribution).sum();
                    let g_bz: f64 = (&g_bz_filaments * current_distribution).sum();
                    g_pf_passive[i_dof_total] = g_br * probe_angle.cos() + g_bz * probe_angle.sin();
                    i_dof_total += 1;
                }
            }

            greens_b_field_pol_probes.push(EquilibriumGreensBFieldPolProbe {
                name: b_field_pol_probes[i_b_field_pol_probe].name.clone(),
                plasma: g_plasma
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_b_field_pol_probe: failed to reshape `g_plasma` into (n_z, n_r)"),
                d_plasma_d_z: g_d_plasma_d_z
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_b_field_pol_probe: failed to reshape `g_d_plasma_d_z` into (n_z, n_r)"),
                pf_active: g_pf_active,
                pf_passive: g_pf_passive,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the probes' tables instead of repeating them
        self.equilibrium_ids.greens.b_field_pol_probe = greens_b_field_pol_probes;
    }
}
