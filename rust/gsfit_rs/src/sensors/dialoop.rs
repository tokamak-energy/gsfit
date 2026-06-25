use crate::Plasma;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{
    create_empty_sensor_data, SensorsDynamic, SensorsStatic
};

use data_tree::{AddDataTreeGetters, DataTree};
use ndarray::{Array1, Array2, Array3, Axis, s};
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass(module = "gsfit_rs", skip_from_py_object)]
pub struct Dialoop {
    pub results: DataTree,
}

impl Default for Dialoop {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl Dialoop {

    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_sensor(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: PyReadonlyArray1<f64>,
        measured: PyReadonlyArray1<f64>,
    ) {
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("comment", fit_settings_comment);

        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("expected_value", fit_settings_expected_value);

        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include", fit_settings_include);

        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("weight", fit_settings_weight);

        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("time", time.to_owned_array());

        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("value", measured.to_owned_array());
    }

    // ✅ CORE FUNCTION (this is the important one)
    pub fn calculate_sensor_values(
        &mut self,
        _coils: PyRef<Coils>,
        _passives: PyRef<Passives>,
        plasma: PyRef<Plasma>
    ) {
        self.calculate_sensor_values_rs(&plasma);
    }
}

impl Dialoop {

    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        let n_time: usize = times_to_reconstruct.len();

        // Vector of boolean's to say if we use the sensor or not
        let include: Vec<bool> = self.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();

        // Convert from boolean to indices
        let include_indices: Vec<usize> = include
            .iter()
            .enumerate()
            .filter(|(_, include)| **include) // Filter for `true` values
            .map(|(i, _)| i) // Extract the indices
            .collect(); // Collect into a Vec

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();
        let n_sensors_all: usize = sensor_names_all.len();
        // Down select sensor names
        let sensor_names: Vec<String> = include_indices.iter().map(|&index| sensor_names_all[index].clone()).collect();
        let n_sensors: usize = sensor_names.len();

        // If there are no sensors selected, return empty data
        if n_sensors == 0 {
            let (static_data_empty, dynamic_data_empty): (SensorsStatic, SensorsDynamic) = create_empty_sensor_data();
            let static_data_empty_vs_time: Vec<SensorsStatic> = vec![static_data_empty; n_time];
            let dynamic_data_empty_vs_time: Vec<SensorsDynamic> = vec![dynamic_data_empty; n_time];
            return (static_data_empty_vs_time, dynamic_data_empty_vs_time);
        }

        // Fit settings
        // Weight
        let fit_settings_weight: Array1<f64> = self.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let fit_settings_weight: Array1<f64> = fit_settings_weight.select(Axis(0), &include_indices);
        // Expected value
        let fit_settings_expected_value: Array1<f64> = self.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let fit_settings_expected_value: Array1<f64> = fit_settings_expected_value.select(Axis(0), &include_indices);

        // Diamagnetic loops do not use Green's functions: the response is computed directly from
        // the ff' source function inside the GS solver (see `gs_solution.rs`). The Green's arrays
        // below are therefore left empty.
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid: Array2::zeros((0, n_sensors)),
            greens_with_pf: Array2::zeros((0, n_sensors)),
            greens_with_passives: Array2::zeros((0, n_sensors)),
            greens_d_sensor_dz: Array2::zeros((0, n_sensors)),
            fit_settings_weight,
            fit_settings_expected_value,
            geometry_r: Array1::from_elem(n_sensors, f64::NAN), // dialoop is integrated; no single (r, z) position
            geometry_z: Array1::from_elem(n_sensors, f64::NAN), // dialoop is integrated; no single (r, z) position
        };

        // Time dependent
        // Interpolate all sensors to `times_to_reconstruct`
        let mut measured: Array2<f64> = Array2::from_elem((n_sensors_all, n_time), f64::NAN);
        for i_sensor in 0..n_sensors_all {
            // Sensor names
            let sensor_name: &str = &sensor_names_all[i_sensor];

            // Measured values
            let experimental_time: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("time").unwrap_array1();
            let experimental_values: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("value").unwrap_array1();

            // Create the interpolator
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(experimental_time.clone(), experimental_values.clone())
                .expect("Dialoop.split_into_static_and_dynamic: Can't make interpolator");

            // Do the interpolation
            let measured_this_sensor: Array1<f64> = interpolator
                .interpolate_array1(times_to_reconstruct)
                .expect("Dialoop.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_sensor, ..]).assign(&measured_this_sensor);

            // Store in self
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("b")
                .get_or_insert("measured")
                .insert("value", measured_this_sensor);
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("b")
                .get_or_insert("measured")
                .insert("time", times_to_reconstruct.clone());
        }

        // MDSplus is "Sensor-Major", but we want to rearrange the data to be "Time-Major"
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            // Select time-slice and the sensors we use in reconstruction
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            // Create new `SensorsDynamic` instance and store
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        let results_static_time_dependent: Vec<SensorsStatic> = vec![results_static.clone(); n_time];

        // Return the static and dynamic results
        (results_static_time_dependent, results_dynamic)
    }

    pub fn calculate_sensor_values_rs(&mut self, plasma: &Plasma) {

        let psi_n_2d: Array3<f64> =
            plasma.results.get("two_d").get("psi_n").unwrap_array3();

        let mask_2d: Array3<f64> =
            plasma.results.get("two_d").get("mask").unwrap_array3();

        let flat_r: Array1<f64> =
            plasma.results.get("grid").get("flat").get("r").unwrap_array1();

        let d_area: f64 =
            plasma.results.get("grid").get("d_area").unwrap_f64();

        let time: Array1<f64> =
            plasma.results.get("time").unwrap_array1();

        let psi_a_vs_time: Array1<f64> =
            plasma.results.get("global").get("psi_a").unwrap_array1();

        let psi_b_vs_time: Array1<f64> =
            plasma.results.get("global").get("psi_b").unwrap_array1();

        let ff_coeffs: Array2<f64> =
            plasma.results
                .get("source_functions")
                .get("ff_prime")
                .get("coefficients")
                .unwrap_array2();

        let n_time = time.len();

        for sensor_name in self.results.keys() {

            let mut values = Array1::zeros(n_time);

            for t in 0..n_time {

                let psi_n_flat = Array1::from_iter(
                    psi_n_2d.slice(s![t, .., ..]).iter().copied()
                );

                let mask_flat = Array1::from_iter(
                    mask_2d.slice(s![t, .., ..]).iter().copied()
                );

                let ff_dof = ff_coeffs.slice(s![t, ..]).to_owned();

                // The diamagnetic loop measures the toroidal (diamagnetic) flux,
                // integrated over the poloidal cross-section (Moret Eq. 41):
                //
                //     Phi_t = integral( (f - f_boundary) / R ) dR dZ
                //
                // where `f = R * B_phi` is the toroidal flux function and `f_boundary = R0 * B_phi0`
                // is the vacuum value. Inside the plasma, gsfit reconstructs `f` from the ff' source
                // function exactly as in `epp_bt_2d`:
                //
                //     f - f_boundary = (psi_b - psi_a) * integral(ff', psi_n)
                //
                // so the vacuum term (and hence `i_rod` / `R0 * B_phi0`) cancels.
                let f_minus_f_boundary: Array1<f64> = (psi_b_vs_time[t] - psi_a_vs_time[t])
                    * &plasma
                        .ff_prime_source_function
                        .source_function_integral(&psi_n_flat, &ff_dof);

                let integrand =
                    &mask_flat * &f_minus_f_boundary / &flat_r;

                values[t] = integrand.sum() * d_area;
            }

            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("value", values.clone());

            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("time", time.clone());
        }
    }
}