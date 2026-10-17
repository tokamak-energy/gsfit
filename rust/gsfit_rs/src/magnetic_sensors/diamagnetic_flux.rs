//! `magnetics/diamagnetic_flux(i1)`: the diamagnetic flux, one for each method of calculating it

use crate::magnetics::Magnetics;
use crate::plasma::Plasma;
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
impl Plasma {
    /// Copy the diamagnetic flux from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/diamagnetic_flux`. This does part of the work of the Python method `copy_from_magnetics_ids`;
    /// see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the diamagnetic flux in `magnetics/diamagnetic_flux`
    ///
    /// # Errors
    /// Why the diamagnetic flux cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_diamagnetic_flux(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();

        // The data dictionary has one diamagnetic flux constraint, whereas the magnetics IDS has one diamagnetic flux for each method of
        // calculating it, the generally recommended method first. So that one is copied
        let diamagnetic_flux_methods: &Vec<MagneticsMethodDistinct> = &magnetics.magnetics_ids.diamagnetic_flux;

        // Without one, the constraint is left unset. Assigned, so that calling this twice replaces it
        if diamagnetic_flux_methods.is_empty() {
            for i_time in 0..n_time {
                self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = EquilibriumConstraints0dB0Like::default();
            }
            return Ok(());
        }

        // The diamagnetic flux is on the experimental timebase, so it is interpolated onto the times of the time-slices `Plasma::new`
        // allocated
        let method_name: &str = &diamagnetic_flux_methods[0].method_name;
        let interpolator: interpolation::Dim1Linear =
            interpolation::Dim1Linear::new(diamagnetic_flux_methods[0].time.clone(), diamagnetic_flux_methods[0].data.clone())
                .map_err(|error: interpolation::Error| format!("cannot interpolate `magnetics/diamagnetic_flux(0)/data` ({method_name}): {error:?}"))?;
        let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
            format!("cannot interpolate `magnetics/diamagnetic_flux(0)/data` ({method_name}) onto the reconstruction times: {error:?}")
        })?;

        // Assigned, so that calling this twice replaces the diamagnetic flux
        for i_time in 0..n_time {
            self.equilibrium_ids.time_slice[i_time].constraints.diamagnetic_flux = EquilibriumConstraints0dB0Like {
                name: method_name.to_string(),
                measured: measured[i_time],
                source: "magnetics/diamagnetic_flux(0)/data".to_string(),
                time_measurement: times_to_reconstruct[i_time],
                ..Default::default()
            };
        }

        Ok(())
    }
}
