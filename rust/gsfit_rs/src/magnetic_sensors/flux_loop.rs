//! `magnetics/flux_loop(i1)`: the flux loops

use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use imas_rs::ids::magnetics::{IdentifierStatic, MagneticsFluxLoop, Rphiz0dStatic, SignalFlt1dValidity};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::{EquilibriumConstraints0d, EquilibriumGreensFluxLoop};
use ndarray::{Array1, Array2, Axis, array, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append a flux loop to `magnetics/flux_loop`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the flux loop, unique within the flux loops, e.g. `"L001"`
    /// * `r` - major radius of the loop, [metre]
    /// * `z` - height of the loop, [metre]
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `flux` - the measured poloidal flux, GSFit's `psi` (1d array), [weber]
    ///
    /// The loop goes all the way around the machine, so it is a `poloidal_flux` loop with one `position`.
    pub fn add_flux_loop(&mut self, name: &str, r: f64, z: f64, time: PyReadonlyArray1<f64>, flux: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let flux_time: Array1<f64> = time.to_owned_array();
        let flux_data: Array1<f64> = flux.to_owned_array();

        // Run the Rust method
        self.add_flux_loop_rs(name, r, z, flux_time, flux_data).map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append a flux loop to `magnetics/flux_loop`. This does the work of the Python method `add_flux_loop`, so that Rust code
    /// can add flux loops too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the flux loop, unique within the flux loops, e.g. `"L001"`
    /// * `r` - major radius of the loop, [metre]
    /// * `z` - height of the loop, [metre]
    /// * `flux_time` - the experimental timebase, [second]
    /// * `flux_data` - the measured poloidal flux, GSFit's `psi`, [weber]
    ///
    /// # Errors
    /// Why the flux loop cannot be added: an empty or repeated name, a position without a positive radius, or a flux without one
    /// value per time.
    pub fn add_flux_loop_rs(&mut self, name: &str, r: f64, z: f64, flux_time: Array1<f64>, flux_data: Array1<f64>) -> Result<(), String> {
        if name.is_empty() {
            return Err("flux loop: `name` is empty; a flux loop needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device"
        let n_flux_loop: usize = self.magnetics_ids.flux_loop.len();
        for i_flux_loop in 0..n_flux_loop {
            if self.magnetics_ids.flux_loop[i_flux_loop].name == name {
                return Err(format!(
                    "flux loop `{name}`: a flux loop with this name already exists, at `magnetics/flux_loop({i_flux_loop})`"
                ));
            }
        }

        let is_usable_position: bool = r > 0.0 && r.is_finite() && z.is_finite();
        if !is_usable_position {
            return Err(format!("flux loop `{name}`: r = {r} and z = {z} metre; r must be positive, and both finite"));
        }

        if flux_time.len() != flux_data.len() {
            return Err(format!(
                "flux loop `{name}`: `time` has {} point(s) but `flux` has {}; a signal needs one value per time",
                flux_time.len(),
                flux_data.len()
            ));
        }

        if flux_time.is_empty() {
            return Err(format!("flux loop `{name}`: `time` is empty; the flux needs at least one time"));
        }

        self.magnetics_ids.flux_loop.push(MagneticsFluxLoop {
            name: name.to_string(),
            r#type: IdentifierStatic {
                name: "poloidal_flux".to_string(),
                index: 1,
                description: "Poloidal flux loop".to_string(),
            },
            position: vec![Rphiz0dStatic { r, z, ..Default::default() }],
            flux: SignalFlt1dValidity {
                data: flux_data,
                time: flux_time,
                ..Default::default()
            },
            ..Default::default()
        });

        Ok(())
    }
}

/// Rust only methods, copying the flux loops into the equilibrium IDS
impl Equilibrium {
    /// Copy the flux loops' fluxes from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/flux_loop(i1)`. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the flux loops in `magnetics/flux_loop`
    ///
    /// # Errors
    /// A flux loop in the fit settings, `code/parameters_parsed/constraints/flux_loop`, is not in the magnetics IDS, or a flux loop's
    /// measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_flux_loop(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let flux_loops: &Vec<MagneticsFluxLoop> = &magnetics.magnetics_ids.flux_loop;
        let n_flux_loop: usize = flux_loops.len();

        // Only the flux loops in the fit settings, `code/parameters_parsed/constraints/flux_loop`, are copied, and each of them has to
        // be in the magnetics IDS. Checked before anything is copied
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.flux_loop.flux_loop(..).name.to_vec();
        let settings_weights: Array1<f64> = self
            .equilibrium_ids
            .code
            .parameters_parsed
            .constraints
            .flux_loop
            .flux_loop(..)
            .weight
            .to_array();
        let n_settings: usize = settings_names.len();
        for i_settings in 0..n_settings {
            let settings_name: &str = &settings_names[i_settings];
            if !flux_loops
                .iter()
                .any(|magnetics_sensor: &MagneticsFluxLoop| magnetics_sensor.name == settings_name)
            {
                return Err(format!(
                    "`{settings_name}` is in the fit settings, `code/parameters_parsed/constraints/flux_loop`, but not in the magnetics IDS, `magnetics/flux_loop`"
                ));
            }
        }

        // Each flux loop's flux is on the experimental timebase, so it is interpolated onto the times of the time-slices
        // `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut flux_measured: Array2<f64> = Array2::from_elem((n_time, n_flux_loop), f64::NAN); // shape = (n_time, n_flux_loop)
        for i_flux_loop in 0..n_flux_loop {
            let flux_loop_name: &str = &flux_loops[i_flux_loop].name;
            if !settings_names.iter().any(|settings_name: &String| settings_name.as_str() == flux_loop_name) {
                continue;
            }
            let interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(flux_loops[i_flux_loop].flux.time.clone(), flux_loops[i_flux_loop].flux.data.clone()).map_err(
                    |error: interpolation::Error| format!("cannot interpolate `magnetics/flux_loop({i_flux_loop})/flux` ({flux_loop_name}): {error:?}"),
                )?;
            let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
                format!("cannot interpolate `magnetics/flux_loop({i_flux_loop})/flux` ({flux_loop_name}) onto the reconstruction times: {error:?}")
            })?;
            flux_measured.slice_mut(s![.., i_flux_loop]).assign(&measured);
        }

        // Rebuilt from the flux loops in the fit settings, in the order of the magnetics IDS, so that calling this twice replaces them
        // instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight`, found by
        // its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraints0d> = std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.flux_loop);
            let mut flux_loop_constraints: Vec<EquilibriumConstraints0d> = Vec::with_capacity(n_settings);
            for i_flux_loop in 0..n_flux_loop {
                let flux_loop_name: &str = &flux_loops[i_flux_loop].name;
                let Some(i_settings) = settings_names
                    .iter()
                    .position(|settings_name: &String| settings_name.as_str() == flux_loop_name)
                else {
                    continue;
                };
                let mut constraint: EquilibriumConstraints0d = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraints0d| settings_constraint.name == flux_loop_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => EquilibriumConstraints0d {
                        name: flux_loop_name.to_string(),
                        weight: settings_weights[i_settings],
                        ..Default::default()
                    },
                };
                constraint.measured = flux_measured[[i_time, i_flux_loop]];
                constraint.source = format!("magnetics/flux_loop({i_flux_loop})/flux");
                constraint.time_measurement = times_to_reconstruct[i_time];
                flux_loop_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.flux_loop = flux_loop_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the flux loops' Greens tables
impl Equilibrium {
    /// Calculate the flux loops' Greens tables, `equilibrium/greens/flux_loop(i1)`, one for each flux loop in the fit settings,
    /// `code/parameters_parsed/constraints/flux_loop`, in the order of `magnetics/flux_loop`. This does part of the work of the Python
    /// method `greens_with_magnetics`; see it for what is stored, and for the checks on `pf_active`, `pf_passive` and the fit settings.
    ///
    /// Each table is the poloidal flux through the loop, GSFit's `psi`. The loop goes all the way around the machine, at its one
    /// `position`.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the flux loops in `magnetics/flux_loop`
    /// * `pf_active` - the PF coils, in the order of `greens/grid/pf_active`
    /// * `pf_passive` - the passive conductors, in the order of `greens/grid/pf_passive`
    pub fn greens_with_magnetics_flux_loop(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) {
        let flux_loops: &Vec<MagneticsFluxLoop> = &magnetics.magnetics_ids.flux_loop;
        let n_flux_loop: usize = flux_loops.len();

        // Only the flux loops in the fit settings, `code/parameters_parsed/constraints/flux_loop`, which `copy_from_magnetics_ids` copies,
        // have Greens tables. `greens_with_magnetics` checked that each of them is in the magnetics IDS
        let settings_names: Vec<String> = self.equilibrium_ids.code.parameters_parsed.constraints.flux_loop.flux_loop(..).name.to_vec();

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

        // The passive conductors' filaments, which are the same for every flux loop
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut loop_filaments: Vec<PassiveFilaments> = Vec::with_capacity(n_loop);
        let mut n_dof_total: usize = 0;
        for i_loop in 0..n_loop {
            loop_filaments.push(passive_filaments(&pf_passive_loops[i_loop].element));
            n_dof_total += pf_passive_loops[i_loop].dof.len();
        }

        let mut greens_flux_loops: Vec<EquilibriumGreensFluxLoop> = Vec::with_capacity(n_flux_loop);
        for i_flux_loop in 0..n_flux_loop {
            if !settings_names
                .iter()
                .any(|settings_name: &String| *settings_name == flux_loops[i_flux_loop].name)
            {
                continue;
            }
            // `Magnetics.add_flux_loop` gives every flux loop exactly one position
            let flux_loop_r: f64 = flux_loops[i_flux_loop].position[0].r;
            let flux_loop_z: f64 = flux_loops[i_flux_loop].position[0].z;

            // With the plasma
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                array![flux_loop_r],
                array![flux_loop_z],
                grid_r.clone(),
                grid_z.clone(),
                grid_r.clone() * 0.0, // TODO: should I set these to NaN?
                grid_z.clone() * 0.0,
            );
            let g_plasma: Array1<f64> = greens_calculator.psi().sum_axis(Axis(0)); // shape = [n_z * n_r]
            let g_d_plasma_d_z: Array1<f64> = greens_calculator.d_psi_d_z().sum_axis(Axis(0)); // shape = [n_z * n_r]

            // With each PF coil, summed over the coil's filaments
            let mut g_pf_active: Array1<f64> = Array1::from_elem(n_coil, f64::NAN); // shape = [n_coil]
            for i_coil in 0..n_coil {
                let coil_r: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array();
                let coil_z: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array();
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![flux_loop_r],
                    array![flux_loop_z],
                    coil_r.clone(),
                    coil_z.clone(),
                    coil_r.clone() * 0.0, // TODO: should I set these to NaN?
                    coil_z.clone() * 0.0,
                );
                g_pf_active[i_coil] = greens_calculator.psi().sum();
            }

            // With each passive degree of freedom: the loop's filaments, weighted by the degree of freedom's current distribution
            let mut g_pf_passive: Array1<f64> = Array1::from_elem(n_dof_total, f64::NAN); // shape = [n_dof_total]
            let mut i_dof_total: usize = 0;
            for i_loop in 0..n_loop {
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![flux_loop_r],
                    array![flux_loop_z],
                    loop_filaments[i_loop].r.clone(),
                    loop_filaments[i_loop].z.clone(),
                    loop_filaments[i_loop].r.clone() * 0.0, // TODO: should I set these to NaN?
                    loop_filaments[i_loop].z.clone() * 0.0,
                );
                let g_psi_filaments: Array2<f64> = greens_calculator.psi(); // shape = [1, n_filament]

                let n_dof: usize = pf_passive_loops[i_loop].dof.len();
                for i_dof in 0..n_dof {
                    let current_distribution: &Array1<f64> = &pf_passive_loops[i_loop].dof[i_dof].current_distribution;
                    g_pf_passive[i_dof_total] = (&g_psi_filaments * current_distribution).sum();
                    i_dof_total += 1;
                }
            }

            greens_flux_loops.push(EquilibriumGreensFluxLoop {
                name: flux_loops[i_flux_loop].name.clone(),
                plasma: g_plasma
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_flux_loop: failed to reshape `g_plasma` into (n_z, n_r)"),
                d_plasma_d_z: g_d_plasma_d_z
                    .into_shape_with_order((n_z, n_r))
                    .expect("greens_with_magnetics_flux_loop: failed to reshape `g_d_plasma_d_z` into (n_z, n_r)"),
                pf_active: g_pf_active,
                pf_passive: g_pf_passive,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the flux loops' tables instead of repeating them
        self.equilibrium_ids.greens.flux_loop = greens_flux_loops;
    }
}
