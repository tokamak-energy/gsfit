use crate::Plasma;
use crate::coils::Coils;
use crate::greens::greens;
use crate::greens::greens_magnetic_field;
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::SensorsStatic;
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_interp::interp1d::Interp1D;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::SensorsDynamic;

const PI: f64 = std::f64::consts::PI;

#[derive(Clone)]
#[pyclass]
pub struct IsofluxBoundary {
    pub results: NestedDict,
}

/// Python accessible methods
#[pymethods]
impl IsofluxBoundary {
    #[new]
    pub fn new() -> Self {
        Self { results: NestedDict::new() }
    }

    /// Data structure:
    ///
    /// # Examples
    ///
    /// ```
    /// [probe_name]["b"]["calculated"]                                 = Array1<f64>;  shape=[n_time]
    /// [probe_name]["b"]["measured"]                                   = Array1<f64>;  shape=[n_time]
    /// [probe_name]["fit_settings"]["comment"]                         = str
    /// [probe_name]["fit_settings"]["expected_value"]                  = f64
    /// [probe_name]["fit_settings"]["include"]                         = bool
    /// [probe_name]["fit_settings"]["weight"]                          = f64
    /// [probe_name]["geometry"]["angle_pol"]                           = f64
    /// [probe_name]["geometry"]["r"]                                   = f64
    /// [probe_name]["geometry"]["z"]                                   = f64
    /// [probe_name]["greens"]["d_plasma_d_z"]                          = Array1;  shape=[n_z * n_r]
    /// [probe_name]["greens"]["passives"][passive_name][dof_name]      = f64
    /// [probe_name]["greens"]["pf"][pf_name]                           = f64
    /// [probe_name]["greens"]["plasma"]                                = Array1;  shape=[n_z * n_r]
    /// ```
    ///
    pub fn add_sensor(
        &mut self,
        name: &str,
        fit_settings_comment: String,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: &Bound<'_, PyArray1<f64>>,
        location_1_r: &Bound<'_, PyArray1<f64>>,
        location_1_z: &Bound<'_, PyArray1<f64>>,
        times_to_reconstruct: &Bound<'_, PyArray1<f64>>,
    ) {
        // Convert into Rust data types
        let location_1_r_ndarray: Array1<f64> = Array1::from(unsafe { location_1_r.as_array() }.to_vec());
        let location_1_z_ndarray: Array1<f64> = Array1::from(unsafe { location_1_z.as_array() }.to_vec());
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let times_to_reconstruct_ndarray: Array1<f64> = Array1::from(unsafe { times_to_reconstruct.as_array() }.to_vec());
        let n_time: usize = times_to_reconstruct_ndarray.len();

        // Fit settings
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("comment", fit_settings_comment);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include", fit_settings_include);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("weight", fit_settings_weight);
        self.results.get_or_insert(name).get_or_insert("fit_settings").insert("expected_value", 1.0f64);

        // Results
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .get_or_insert("location_1")
            .get_or_insert("r")
            .insert("measured_experimental", location_1_r_ndarray.clone());
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .get_or_insert("location_1")
            .get_or_insert("z")
            .insert("measured_experimental", location_1_z_ndarray.clone());
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .insert("time_experimental", time_ndarray.clone());

        // Interpolate all sensors to `times_to_reconstruct`
        // location_1_r
        let interpolator = Interp1D::builder(location_1_r_ndarray)
            .x(time_ndarray.clone())
            .build()
            .expect("Isoflux.greens_with_coils: Can't make Interp1D for location_1_r");
        let location_1_r_measured: Array1<f64> = interpolator
            .interp_array(&times_to_reconstruct_ndarray)
            .expect("Isoflux.greens_with_coils: Can't do interpolation for location_1_r");
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .get_or_insert("location_1")
            .get_or_insert("r")
            .insert("measured", location_1_r_measured.clone());
        // location_1_z
        let interpolator = Interp1D::builder(location_1_z_ndarray)
            .x(time_ndarray.clone())
            .build()
            .expect("Isoflux.greens_with_coils: Can't make Interp1D for location_1_r");
        let location_1_z_measured: Array1<f64> = interpolator
            .interp_array(&times_to_reconstruct_ndarray)
            .expect("Isoflux.greens_with_coils: Can't do interpolation for location_1_r");
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .get_or_insert("location_1")
            .get_or_insert("z")
            .insert("measured", location_1_z_measured.clone());

        // Add time
        self.results
            .get_or_insert(name)
            .get_or_insert("isoflux_geometry")
            .insert("time", times_to_reconstruct_ndarray);

        // Add a time-dependent "include"
        let mut include_dynamic: Vec<bool> = vec![fit_settings_include; n_time];
        for i_time in 0..n_time {
            if location_1_r_measured[i_time].is_nan() || location_1_z_measured[i_time].is_nan() {
                include_dynamic[i_time] = false;
            }
        }
        println!("include_dynamic = {:?}", include_dynamic);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include_dynamic", include_dynamic);
    }

    ///
    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &*coils;

        for sensor_name in self.results.keys() {
            // Get the isoflux locations
            let location_1_r: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("r")
                .get("measured")
                .unwrap_array1();
            let location_1_z: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("z")
                .get("measured")
                .unwrap_array1();

            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("isoflux_geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            for pf_coil_name in coils_local.results.get("pf").keys() {
                let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

                let mut g_vs_time: Array1<f64> = Array1::zeros(n_time);
                for i_time in 0..n_time {
                    // Calculate the Green's at location 1
                    let g_full_location_1: Array2<f64> = greens(
                        Array1::from_vec(vec![location_1_r[i_time]]),
                        Array1::from_vec(vec![location_1_z[i_time]]),
                        coil_r.clone(),
                        coil_z.clone(),
                        coil_r.clone() * 0.0,
                        coil_r.clone() * 0.0,
                    );

                    // Sum over all the current sources
                    g_vs_time[i_time] = g_full_location_1.sum();
                }

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, g_vs_time); // shape = [n_time]
            }
        }
    }

    ///
    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &*passives;

        for sensor_name in self.results.keys() {
            // Get the isoflux locations
            let location_1_r: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("r")
                .get("measured")
                .unwrap_array1();
            let location_1_z: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("z")
                .get("measured")
                .unwrap_array1();

            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("isoflux_geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Calculate Greens with each passive degree of freedom
            for passive_name in passives_local.results.keys() {
                let _tmp: NestedDictAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();
                let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
                let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

                // Loop over all degrees of freedom
                for dof_name in dof_names {
                    let mut g_vs_time: Array1<f64> = Array1::zeros(n_time);
                    for i_time in 0..n_time {
                        // Location 1
                        let g_full_location_1: Array2<f64> = greens(
                            Array1::from_vec(vec![location_1_r[i_time]]), // by convention (r, z) are "sensors"
                            Array1::from_vec(vec![location_1_z[i_time]]),
                            passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
                            passive_z.clone(),
                            passive_r.clone() * 0.0,
                            passive_z.clone() * 0.0,
                        );

                        // Current distribution
                        let current_distribution: Array1<f64> = passives
                            .results
                            .get(&passive_name)
                            .get("dof")
                            .get(&dof_name)
                            .get("current_distribution")
                            .unwrap_array1();

                        let g_with_dof_full_location_1: Array2<f64> = g_full_location_1 * &current_distribution; // shape = [n_r * n_z, n_filament]

                        // Sum over all filaments
                        g_vs_time[i_time] = g_with_dof_full_location_1.sum();
                    }

                    // Store
                    self.results
                        .get_or_insert(&sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(&passive_name)
                        .insert(&dof_name, g_vs_time); // shape = [n_time]
                }
            }
        }
    }

    fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &*plasma;

        let n_r: usize = plasma_local.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma_local.results.get("grid").get("n_z").unwrap_usize();

        let plasma_r: Array1<f64> = plasma_local.results.get("grid").get("flat").get("r").unwrap_array1(); // shape = n_z * n_r
        let plasma_z: Array1<f64> = plasma_local.results.get("grid").get("flat").get("z").unwrap_array1();

        for sensor_name in self.results.keys() {
            // Get the isoflux locations
            let location_1_r: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("r")
                .get("measured")
                .unwrap_array1();
            let location_1_z: Array1<f64> = self
                .results
                .get(&sensor_name)
                .get("isoflux_geometry")
                .get("location_1")
                .get("z")
                .get("measured")
                .unwrap_array1();

            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("isoflux_geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            let mut g_with_plasma: Array2<f64> = Array2::zeros([n_time, n_z * n_r]);
            let mut g_d_plasma_d_z: Array2<f64> = Array2::zeros([n_time, n_z * n_r]);
            for i_time in 0..n_time {
                // Plasma componet
                let g_full_location_1: Array2<f64> = greens(
                    Array1::from_vec(vec![location_1_r[i_time]]), // sensor
                    Array1::from_vec(vec![location_1_z[i_time]]),
                    plasma_r.clone(), // current source
                    plasma_z.clone(),
                    plasma_r.clone() * 0.0, // TODO: I don't like the dr_prime and dz_prime
                    plasma_r.clone() * 0.0,
                );

                // Sensors Green's function
                let g_with_plasma_location_1: Array1<f64> = g_full_location_1.sum_axis(Axis(0)); // g_br_full_location_1.shape = [1, n_z * n_r];  g_br_location_1.shape = [n_z * n_r]

                // Store greens
                let g_with_plasma_now: Array1<f64> = g_with_plasma_location_1;
                g_with_plasma.slice_mut(s![i_time, ..]).assign(&g_with_plasma_now);

                // Vertical stability
                // location_1
                let (g_br_full_location_1, _g_bz_full_location_1): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
                    Array1::from_vec(vec![location_1_r[i_time]]), // sensors
                    Array1::from_vec(vec![location_1_r[i_time]]),
                    plasma_r.clone(), // current sources
                    plasma_z.clone(),
                );

                let g_d_plasma_d_z_location_1: Array1<f64> = -2.0 * PI * location_1_r[i_time].clone() * g_br_full_location_1.sum_axis(Axis(0)); // shape = n_r * n_z

                g_d_plasma_d_z.slice_mut(s![i_time, ..]).assign(&g_d_plasma_d_z_location_1);
            }

            // Store
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma); // shape = [n_time, n_z * n_r]

            // TODO: should I reshape to Array2 [n_z, n_r] ????
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z); // shape = [n_time, n_z * n_r]
        }
    }

    /// Calculate the sensor values
    pub fn calculate_sensor_values(&mut self, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let plasma_rs: &Plasma = &*plasma;

        // Run the Rust method
        self.calculate_sensor_values_rust(plasma_rs);
    }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Isoflux>");
        string_output += &format!("║  {:<74} ║\n", version);

        // n_sensors = self.results
        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }

    /// Get Array1<f64> and return a numpy.ndarray
    pub fn get_array1(&self, keys: Vec<String>, py: Python) -> Py<PyArray1<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array1: Array1<f64> = result_accumulator.unwrap_array1();
        return array1.into_pyarray(py).into();
    }

    /// Get Array2<f64> and return a numpy.ndarray
    pub fn get_array2(&self, keys: Vec<String>, py: Python) -> Py<PyArray2<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array2: Array2<f64> = result_accumulator.unwrap_array2();
        return array2.into_pyarray(py).into();
    }

    /// Get Array3<f64> and return a numpy.ndarray
    pub fn get_array3(&self, keys: Vec<String>, py: Python) -> Py<PyArray3<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array3: Array3<f64> = result_accumulator.unwrap_array3();
        return array3.into_pyarray(py).into();
    }

    /// Get Vec<bool> and return a Python list[bool]
    pub fn get_bool(&self, keys: Vec<String>) -> bool {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<bool>, and return as a Python list
        let bool_value: bool = result_accumulator.unwrap_bool();

        return bool_value;
    }

    /// Get f64 value and return a f64
    pub fn get_f64(&self, keys: Vec<String>) -> f64 {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap the f64, and return
        return result_accumulator.unwrap_f64();
    }

    /// Get usize value and return a int
    pub fn get_usize(&self, keys: Vec<String>) -> usize {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap the f64, and return
        return result_accumulator.unwrap_usize();
    }

    /// Get Vec<bool> and return a Python list[bool]
    pub fn get_vec_bool(&self, keys: Vec<String>, py: Python) -> Py<PyList> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<bool>, and return as a Python list
        let vec_bool: Vec<bool> = result_accumulator.unwrap_vec_bool();
        let result: Py<PyList> = PyList::new(py, vec_bool).unwrap().into();
        return result;
    }

    /// Get Vec<usize> and return a Python list[int]
    pub fn get_vec_usize(&self, keys: Vec<String>, py: Python) -> Py<PyList> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<usize>, and return as a Python list of integers
        let vec_usize: Vec<usize> = result_accumulator.unwrap_vec_usize();
        let vec_i64: Vec<i64> = vec_usize.into_iter().map(|x| x as i64).collect();
        let result: Py<PyList> = PyList::new(py, vec_i64).unwrap().into();
        return result;
    }

    /// Get the keys from results and return a Python list of strings
    #[pyo3(signature = (key_path=None))]
    pub fn keys(&self, py: Python, key_path: Option<&Bound<'_, PyList>>) -> Py<PyList> {
        let keys: Vec<String> = if let Some(key_path) = key_path {
            if key_path.len() == 0 {
                self.results.keys()
            } else {
                // Convert PyList to Vec<String> and traverse NestedDictAccumulator
                let keys: Vec<String> = key_path.extract().expect("Failed to extract key_path as Vec<String>");
                let mut result_accumulator = self.results.get(&keys[0]);
                // Skip the first key and traverse the rest
                for key in &keys[1..] {
                    result_accumulator = result_accumulator.get(key);
                }
                result_accumulator.keys()
            }
        } else {
            // if `sub_keys = None` (when not supplied)
            self.results.keys()
        };
        let result: Py<PyList> = PyList::new(py, keys).unwrap().into();
        return result;
    }

    /// Print keys to screen, to be used within Python
    pub fn print_keys(&self) {
        self.results.print_keys();
    }
}

// Rust only methods
impl IsofluxBoundary {
    /// For IsofluxBoundary sensors the static data is actually time-dependent.
    /// TODO: consider renaming `SensorsStatic`. Perhaps `GeometricGreens` ?
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        // Define empty data arrays
        let results_static_empty: SensorsStatic = SensorsStatic {
            greens_with_grid: Array2::zeros((0, 0)),
            greens_with_pf: Array2::zeros((0, 0)),
            greens_with_passives: Array2::zeros((0, 0)),
            greens_d_sensor_dz: Array2::zeros((0, 0)),
            fit_settings_weight: Array1::zeros(0),
            fit_settings_expected_value: Array1::zeros(0),
        };
        let results_dynamic_empty: SensorsDynamic = SensorsDynamic { measured: Array1::zeros(0) };

        // Number of time-slices to reconstruct
        let n_time: usize = times_to_reconstruct.len();

        // Create the time-dependent data structures
        let mut results_static: Vec<SensorsStatic> = Vec::with_capacity(n_time);
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();

        'time_loop: for i_time in 0..n_time {
            let mut include: Vec<bool> = Vec::new();
            for sensor_name in sensor_names_all.clone() {
                let include_dynamic: Vec<bool> = self.results.get(&sensor_name).get("fit_settings").get("include_dynamic").unwrap_vec_bool();
                println!("include_dynamic = {:?}", include_dynamic);

                if include_dynamic[i_time] == true {
                    include.push(true)
                } else {
                    include.push(false)
                }
            }

            // Convert from boolean to indices
            let include_indices: Vec<usize> = include
                .iter()
                .enumerate()
                .filter(|(_, include)| **include) // Filter for `true` values
                .map(|(i, _)| i) // Extract the indices
                .collect(); // Collect into a Vec

            // If there are no sensors at this time-slice then we should exit
            if include_indices.is_empty() {
                results_static.push(results_static_empty.clone());
                results_dynamic.push(results_dynamic_empty.clone());
                continue 'time_loop; // Go to next time-slice
            }

            // Down select sensor names
            let sensor_names: Vec<String> = include_indices.iter().map(|&index| sensor_names_all[index].clone()).collect();
            let n_sensors: usize = sensor_names.len();

            // Now do the static data
            // Fit settings
            // Weight
            let fit_settings_weight: Array1<f64> = self.results.get("*").get("fit_settings").get("weight").unwrap_array1();
            let fit_settings_weight: Array1<f64> = fit_settings_weight.select(Axis(0), &include_indices);

            // Expected value
            let fit_settings_expected_value: Array1<f64> = self.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
            let fit_settings_expected_value: Array1<f64> = fit_settings_expected_value.select(Axis(0), &include_indices);

            // Greens
            // With PF coils (TIMING: accessing greens_with_pf with unwrap_array3() takes ~50microseconds)
            let greens_with_pf: Array3<f64> = self.results.get("*").get("greens").get("pf").get("*").unwrap_array3(); // shape = [n_time, n_pf, n_sensors]
            let greens_with_pf: Array3<f64> = greens_with_pf.select(Axis(2), &include_indices); // downselect to only the sensors needed; shape = [n_time, n_pf, n_sensors]

            // With plasma
            let greens_with_grid: Array3<f64> = self.results.get("*").get("greens").get("plasma").unwrap_array3(); // shape = [n_time, n_z * n_r, n_sensors]
            let greens_with_grid: Array3<f64> = greens_with_grid.select(Axis(2), &include_indices); // downselect to only the sensors needed; shape = [n_time, n_z * n_r, n_sensors]

            // With d_sensor_dz (for vertical stability)
            let greens_d_sensor_dz: Array3<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array3(); // shape = [n_time, n_z * n_r, n_sensors]
            let greens_d_sensor_dz: Array3<f64> = greens_d_sensor_dz.select(Axis(2), &include_indices); // downselect to only the sensors needed; shape = [n_time, n_z * n_r, n_sensors]

            // With passives
            let passive_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").keys();
            let n_passives: usize = passive_names.len();

            // Count the number of degrees of freedom
            let mut n_dof_total: usize = 0;
            for passive_name in &passive_names {
                let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys();
                n_dof_total += dof_names.len();
            }

            // With passives
            let mut greens_with_passives: Array3<f64> = Array3::zeros((n_time, n_dof_total, n_sensors)) * f64::NAN;
            for i_sensor in 0..n_sensors {
                let mut i_dof_total: usize = 0;
                for i_passive in 0..n_passives {
                    let passive_name: &str = &passive_names[i_passive];
                    let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
                    for dof_name in dof_names {
                        let greens_with_passives_tmp: Array1<f64> = self
                            .results
                            .get(&sensor_names[i_sensor])
                            .get("greens")
                            .get("passives")
                            .get(&passive_name)
                            .get(&dof_name)
                            .unwrap_array1(); // shape = [n_time]
                        greens_with_passives.slice_mut(s![.., i_dof_total, i_sensor]).assign(&greens_with_passives_tmp);

                        // Keep count
                        i_dof_total += 1;
                    }
                }
            }

            // Create the `SensorsStatic` data
            let results_static_this_time_slice: SensorsStatic = SensorsStatic {
                greens_with_grid: greens_with_grid.slice(s![i_time, .., ..]).to_owned(), // shape = [n_z * n_r, n_sensors]
                greens_with_pf: greens_with_pf.slice(s![i_time, .., ..]).to_owned(),
                greens_with_passives: greens_with_passives.slice(s![i_time, .., ..]).to_owned(),
                greens_d_sensor_dz: greens_d_sensor_dz.slice(s![i_time, .., ..]).to_owned(),
                fit_settings_weight: fit_settings_weight.clone(),
                fit_settings_expected_value: fit_settings_expected_value.clone(),
            };
            results_static.push(results_static_this_time_slice);

            // The measured sensor values are = 0.0
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: Array1::zeros(n_sensors),
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        // Return the static and dynamic results
        return (results_static, results_dynamic);
    }

    ///
    pub fn calculate_sensor_values_rust(&mut self, _plasma: &Plasma) {
        for sensor_name in self.results.keys() {
            println!("sensor_name = {}", sensor_name);
        }
    }
}
