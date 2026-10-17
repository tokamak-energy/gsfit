//! `magnetics/diamagnetic_flux(i1)`: the diamagnetic flux, one for each method of calculating it

use crate::equilibrium::Equilibrium;
use crate::magnetics::Magnetics;
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
    /// Append a diamagnetic flux to `magnetics/diamagnetic_flux`, one for each method of calculating it.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the diamagnetic flux is calculated with, unique within the diamagnetic fluxes,
    ///   e.g. `"DIALOOP"`
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `data` - the diamagnetic flux, positive counter-clockwise viewed from above (1d array), [weber]
    ///
    /// The data dictionary asks for the generally recommended method to be added first.
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
    /// Append a diamagnetic flux to `magnetics/diamagnetic_flux`. This does the work of the Python method `add_diamagnetic_flux`,
    /// so that Rust code can add diamagnetic fluxes too; see it for what is stored.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the diamagnetic flux is calculated with, unique within the diamagnetic fluxes,
    ///   e.g. `"DIALOOP"`
    /// * `diamagnetic_flux_time` - the experimental timebase, [second]
    /// * `diamagnetic_flux_data` - the diamagnetic flux, positive counter-clockwise viewed from above, [weber]
    ///
    /// # Errors
    /// Why the diamagnetic flux cannot be added: an empty or repeated method name, or data without one value per time.
    pub fn add_diamagnetic_flux_rs(&mut self, method_name: &str, diamagnetic_flux_time: Array1<f64>, diamagnetic_flux_data: Array1<f64>) -> Result<(), String> {
        if method_name.is_empty() {
            return Err("diamagnetic flux: `method_name` is empty; a diamagnetic flux needs the name of its method".to_string());
        }

        let n_diamagnetic_flux: usize = self.magnetics_ids.diamagnetic_flux.len();
        for i_diamagnetic_flux in 0..n_diamagnetic_flux {
            if self.magnetics_ids.diamagnetic_flux[i_diamagnetic_flux].method_name == method_name {
                return Err(format!(
                    "diamagnetic flux `{method_name}`: a diamagnetic flux with this method name already exists, at \
                     `magnetics/diamagnetic_flux({i_diamagnetic_flux})`"
                ));
            }
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
    /// Copy the diamagnetic flux from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/diamagnetic_flux`. This does part of the work of the Python method `copy_from_magnetics_ids`;
    /// see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the diamagnetic flux in `magnetics/diamagnetic_flux`
    ///
    /// # Errors
    /// The diamagnetic flux is in the fit settings but not in the magnetics IDS, or it cannot be interpolated onto the reconstruction times,
    /// e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_diamagnetic_flux(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();

        // The diamagnetic flux is only copied if it is in the fit settings, `code/parameters_parsed/constraints/diamagnetic_flux`, whose
        // `name` is required, so is empty only when they are left out. Without them the constraint is left unset. Assigned, so that calling
        // this twice replaces it
        let settings_name: String = self
            .equilibrium_ids
            .code
            .parameters_parsed
            .constraints
            .diamagnetic_flux
            .diamagnetic_flux
            .name
            .clone();
        let settings_weight: f64 = self.equilibrium_ids.code.parameters_parsed.constraints.diamagnetic_flux.diamagnetic_flux.weight;
        if settings_name.is_empty() {
            for i_time in 0..n_time {
                self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = EquilibriumConstraints0dB0Like::default();
            }
            return Ok(());
        }

        // The data dictionary has one diamagnetic flux constraint, whereas the magnetics IDS has one diamagnetic flux for each method of
        // calculating it. The one the fit settings name is copied, and it has to be there
        let diamagnetic_flux_methods: &Vec<MagneticsMethodDistinct> = &magnetics.magnetics_ids.diamagnetic_flux;
        let Some(i_method) = diamagnetic_flux_methods
            .iter()
            .position(|diamagnetic_flux_method: &MagneticsMethodDistinct| diamagnetic_flux_method.method_name == settings_name)
        else {
            return Err(format!(
                "`{settings_name}` is in the fit settings, `code/parameters_parsed/constraints/diamagnetic_flux`, but not in the magnetics IDS, `magnetics/diamagnetic_flux(:)/method_name`"
            ));
        };

        // The diamagnetic flux is on the experimental timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new`
        // allocated
        let interpolator: interpolation::Dim1Linear =
            interpolation::Dim1Linear::new(diamagnetic_flux_methods[i_method].time.clone(), diamagnetic_flux_methods[i_method].data.clone()).map_err(
                |error: interpolation::Error| format!("cannot interpolate `magnetics/diamagnetic_flux({i_method})/data` ({settings_name}): {error:?}"),
            )?;
        let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
            format!("cannot interpolate `magnetics/diamagnetic_flux({i_method})/data` ({settings_name}) onto the reconstruction times: {error:?}")
        })?;

        // Assigned, so that calling this twice replaces the diamagnetic flux. It keeps the constraint `Equilibrium::new` made from the fit
        // settings, with its `weight`
        for i_time in 0..n_time {
            let settings_constraint: &EquilibriumConstraints0dB0Like = &self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux;
            let mut constraint: EquilibriumConstraints0dB0Like = if settings_constraint.name == settings_name {
                settings_constraint.clone()
            } else {
                EquilibriumConstraints0dB0Like {
                    name: settings_name.clone(),
                    weight: settings_weight,
                    ..Default::default()
                }
            };
            constraint.measured = measured[i_time];
            constraint.source = format!("magnetics/diamagnetic_flux({i_method})/data");
            constraint.time_measurement = times_to_reconstruct[i_time];
            self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = constraint;
        }

        Ok(())
    }
}
