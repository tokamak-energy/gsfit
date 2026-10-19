//! `magnetics/diamagnetic_flux(i1)`: the diamagnetic flux. The data dictionary allows one for each method of calculating it, but GSFit
//! accepts only one, `magnetics/diamagnetic_flux(0)`, as the equilibrium IDS has a single diamagnetic flux constraint. The signals of
//! several diamagnetic loops are averaged into one before it is added

use crate::equilibrium::Equilibrium;
use crate::magnetics::Magnetics;
use imas_rs::EMPTY_INT;
use imas_rs::EquilibriumConstraints0dB0Like;
use imas_rs::ids::magnetics::MagneticsMethodDistinct;
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Add the diamagnetic flux to `magnetics/diamagnetic_flux`, as `magnetics/diamagnetic_flux(0)`.
    ///
    /// GSFit accepts only one diamagnetic flux, as the equilibrium IDS has a single diamagnetic flux constraint,
    /// `time_slice(itime)/constraints/diamagnetic_flux`. A machine with several diamagnetic loops averages their signals into one before
    /// adding it.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the diamagnetic flux is calculated with, e.g. `"DIALOOP"`
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `data` - the diamagnetic flux, positive counter-clockwise viewed from above (1d array), [weber]
    pub fn add_diamagnetic_flux(&mut self, method_name: &str, time: PyReadonlyArray1<f64>, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let diamagnetic_flux_time: Array1<f64> = time.to_owned_array();
        let diamagnetic_flux_data: Array1<f64> = data.to_owned_array();

        // Run the Rust method
        self.add_diamagnetic_flux_rs(method_name, diamagnetic_flux_time, diamagnetic_flux_data)
            .map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Add the diamagnetic flux to `magnetics/diamagnetic_flux`. This does the work of the Python method `add_diamagnetic_flux`,
    /// so that Rust code can add the diamagnetic flux too; see it for what is stored.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the diamagnetic flux is calculated with, e.g. `"DIALOOP"`
    /// * `diamagnetic_flux_time` - the experimental timebase, [second]
    /// * `diamagnetic_flux_data` - the diamagnetic flux, positive counter-clockwise viewed from above, [weber]
    ///
    /// # Errors
    /// Why the diamagnetic flux cannot be added: one has been added already, an empty method name, or data without one value per time.
    pub fn add_diamagnetic_flux_rs(&mut self, method_name: &str, diamagnetic_flux_time: Array1<f64>, diamagnetic_flux_data: Array1<f64>) -> Result<(), String> {
        if method_name.is_empty() {
            return Err("diamagnetic flux: `method_name` is empty; a diamagnetic flux needs the name of its method".to_string());
        }

        // GSFit accepts a single diamagnetic flux, as the equilibrium IDS has a single diamagnetic flux constraint
        if !self.magnetics_ids.diamagnetic_flux.is_empty() {
            return Err(format!(
                "diamagnetic flux `{method_name}`: `magnetics/diamagnetic_flux` already holds a diamagnetic flux, `{}`, and GSFit accepts only one, \
                 as the equilibrium IDS has a single diamagnetic flux constraint: average the diamagnetic loops' signals into one before adding it",
                self.magnetics_ids.diamagnetic_flux[0].method_name
            ));
        }

        if diamagnetic_flux_time.len() != diamagnetic_flux_data.len() {
            return Err(format!(
                "diamagnetic flux `{method_name}`: `time` has {} point(s) but `data` has {}; a signal needs one value per time",
                diamagnetic_flux_time.len(),
                diamagnetic_flux_data.len()
            ));
        }

        if diamagnetic_flux_time.is_empty() {
            return Err(format!(
                "diamagnetic flux `{method_name}`: `time` is empty; the diamagnetic flux needs at least one time"
            ));
        }

        self.magnetics_ids.diamagnetic_flux.push(MagneticsMethodDistinct {
            method_name: method_name.to_string(),
            data: diamagnetic_flux_data,
            time: diamagnetic_flux_time,
        });

        Ok(())
    }
}

/// Rust only methods, copying the diamagnetic flux into the equilibrium IDS
impl Equilibrium {
    /// Copy the diamagnetic flux from the magnetics IDS, `magnetics/diamagnetic_flux(0)`, onto every time-slice of the equilibrium IDS,
    /// as `equilibrium/time_slice(itime)/constraints/diamagnetic_flux`. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the diamagnetic flux in `magnetics/diamagnetic_flux`
    ///
    /// # Errors
    /// The diamagnetic flux is in the fit settings but not in the magnetics IDS, the magnetics IDS holds more than one diamagnetic flux,
    /// or the diamagnetic flux cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_diamagnetic_flux(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();

        // The diamagnetic flux is only copied if it is in the fit settings, `code/parameters_parsed/constraints/diamagnetic_flux`, whose
        // `include` is required, so is unset only when they are left out. Without them the constraint is left unset. Assigned, so that
        // calling this twice replaces it
        let settings_include: i32 = self.equilibrium_ids.code.parameters_parsed.constraints.diamagnetic_flux.include;
        if settings_include == EMPTY_INT {
            for i_time in 0..n_time {
                self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = EquilibriumConstraints0dB0Like::default();
            }
            return Ok(());
        }

        // The data dictionary has a single diamagnetic flux constraint, so GSFit accepts a single diamagnetic flux, which has to be there
        let diamagnetic_fluxes: &Vec<MagneticsMethodDistinct> = &magnetics.magnetics_ids.diamagnetic_flux;
        let n_diamagnetic_flux: usize = diamagnetic_fluxes.len();
        if n_diamagnetic_flux == 0 {
            return Err(
                "the diamagnetic flux is in the fit settings, `code/parameters_parsed/constraints/diamagnetic_flux`, but not in the magnetics IDS, `magnetics/diamagnetic_flux`"
                    .to_string(),
            );
        }
        if n_diamagnetic_flux > 1 {
            return Err(format!(
                "the magnetics IDS holds {n_diamagnetic_flux} diamagnetic fluxes, `magnetics/diamagnetic_flux`, but GSFit accepts only one, as the \
                 equilibrium IDS has a single diamagnetic flux constraint: average the diamagnetic loops' signals into one before adding it"
            ));
        }
        let diamagnetic_flux: &MagneticsMethodDistinct = &diamagnetic_fluxes[0];

        // The diamagnetic flux is on the experimental timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new`
        // allocated
        let interpolator: interpolation::Dim1Linear =
            interpolation::Dim1Linear::new(diamagnetic_flux.time.clone(), diamagnetic_flux.data.clone()).map_err(|error: interpolation::Error| {
                format!(
                    "cannot interpolate `magnetics/diamagnetic_flux(0)/data` ({}): {error:?}",
                    diamagnetic_flux.method_name
                )
            })?;
        let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
            format!(
                "cannot interpolate `magnetics/diamagnetic_flux(0)/data` ({}) onto the reconstruction times: {error:?}",
                diamagnetic_flux.method_name
            )
        })?;

        // Assigned, so that calling this twice replaces the diamagnetic flux. It keeps the constraint `Equilibrium::new` made from the fit
        // settings, with its `weight`
        for i_time in 0..n_time {
            let mut constraint: EquilibriumConstraints0dB0Like = self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux.clone();
            constraint.measured = measured[i_time];
            constraint.source = "magnetics/diamagnetic_flux(0)/data".to_string();
            constraint.time_measurement = times_to_reconstruct[i_time];
            self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = constraint;
        }

        Ok(())
    }
}
