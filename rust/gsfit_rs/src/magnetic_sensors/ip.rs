//! `magnetics/ip(i1)`: the plasma current, one for each method of calculating it

use crate::equilibrium::Equilibrium;
use crate::magnetics::Magnetics;
use imas_rs::EquilibriumConstraints0dIpLike;
use imas_rs::ids::magnetics::MagneticsMethodDistinct;
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append a plasma current to `magnetics/ip`, one for each method of calculating it.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the plasma current is calculated with, unique within the plasma currents
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `data` - the plasma current, positive counter-clockwise viewed from above (1d array), [ampere]
    ///
    /// The data dictionary asks for the generally recommended method to be added first.
    pub fn add_ip(&mut self, method_name: &str, time: PyReadonlyArray1<f64>, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let ip_time: Array1<f64> = time.to_owned_array();
        let ip_data: Array1<f64> = data.to_owned_array();

        // Run the Rust method
        self.add_ip_rs(method_name, ip_time, ip_data).map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append a plasma current to `magnetics/ip`. This does the work of the Python method `add_ip`, so that Rust code can add
    /// plasma currents too; see it for what is stored.
    ///
    /// # Arguments
    /// * `method_name` - name of the method the plasma current is calculated with, unique within the plasma currents
    /// * `ip_time` - the experimental timebase, [second]
    /// * `ip_data` - the plasma current, positive counter-clockwise viewed from above, [ampere]
    ///
    /// # Errors
    /// Why the plasma current cannot be added: an empty or repeated method name, or data without one value per time.
    pub fn add_ip_rs(&mut self, method_name: &str, ip_time: Array1<f64>, ip_data: Array1<f64>) -> Result<(), String> {
        if method_name.is_empty() {
            return Err("plasma current: `method_name` is empty; a plasma current needs the name of its method".to_string());
        }

        let n_ip: usize = self.magnetics_ids.ip.len();
        for i_ip in 0..n_ip {
            if self.magnetics_ids.ip[i_ip].method_name == method_name {
                return Err(format!(
                    "plasma current `{method_name}`: a plasma current with this method name already exists, at `magnetics/ip({i_ip})`"
                ));
            }
        }

        if ip_time.len() != ip_data.len() {
            return Err(format!(
                "plasma current `{method_name}`: `time` has {} point(s) but `data` has {}; a signal needs one value per time",
                ip_time.len(),
                ip_data.len()
            ));
        }

        if ip_time.is_empty() {
            return Err(format!(
                "plasma current `{method_name}`: `time` is empty; the plasma current needs at least one time"
            ));
        }

        self.magnetics_ids.ip.push(MagneticsMethodDistinct {
            method_name: method_name.to_string(),
            data: ip_data,
            time: ip_time,
        });

        Ok(())
    }
}

/// Rust only methods, copying the plasma current into the equilibrium IDS
impl Equilibrium {
    /// Copy the plasma current from the magnetics IDS onto every time-slice of the equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/ip`. This does part of the work of the Python method `copy_from_magnetics_ids`;
    /// see it for what is stored.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the plasma current in `magnetics/ip`
    ///
    /// # Errors
    /// Why the plasma current cannot be interpolated onto the reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_magnetics_ids_ip(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();

        // The data dictionary has one plasma current constraint, whereas the magnetics IDS has one plasma current for each method of
        // calculating it, the generally recommended method first. So that one is copied
        let ip_methods: &Vec<MagneticsMethodDistinct> = &magnetics.magnetics_ids.ip;

        // Without one, the constraint is left unset. Assigned, so that calling this twice replaces it
        if ip_methods.is_empty() {
            for i_time in 0..n_time {
                self.equilibrium_ids.time_slice[i_time].constraints.ip = EquilibriumConstraints0dIpLike::default();
            }
            return Ok(());
        }

        // The plasma current is on the experimental timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new`
        // allocated
        let method_name: &str = &ip_methods[0].method_name;
        let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(ip_methods[0].time.clone(), ip_methods[0].data.clone())
            .map_err(|error: interpolation::Error| format!("cannot interpolate `magnetics/ip(0)/data` ({method_name}): {error:?}"))?;
        let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
            format!("cannot interpolate `magnetics/ip(0)/data` ({method_name}) onto the reconstruction times: {error:?}")
        })?;

        // Assigned, so that calling this twice replaces the plasma current
        for i_time in 0..n_time {
            self.equilibrium_ids.time_slice[i_time].constraints.ip = EquilibriumConstraints0dIpLike {
                name: method_name.to_string(),
                measured: measured[i_time],
                source: "magnetics/ip(0)/data".to_string(),
                time_measurement: times_to_reconstruct[i_time],
                ..Default::default()
            };
        }

        Ok(())
    }
}
