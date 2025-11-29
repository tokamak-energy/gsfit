use crate::Plasma;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, Axis, s};
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass]
pub struct Pressure {
    pub results: DataTree,
}

/// Python accessible methods
#[pymethods]
impl Pressure {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    /// Data structure:
    ///
    /// # Examples
    ///
    /// ```ignore
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
        geometry_r: f64,
        geometry_z: f64,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: PyReadonlyArray1<f64>,
        measured: PyReadonlyArray1<f64>,
    ) {
        // Convert into Rust data types
        let time_ndarray: Array1<f64> = time.to_owned_array();
        let measured_ndarray: Array1<f64> = measured.to_owned_array();

        // Geometry
        self.results.get_or_insert(name).get_or_insert("geometry").insert("r", geometry_r);
        self.results.get_or_insert(name).get_or_insert("geometry").insert("z", geometry_z);

        // Fit settings
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

        // Measurements
        self.results.get_or_insert(name).get_or_insert("b").insert("time_experimental", time_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .insert("measured_experimental", measured_ndarray);
    }

    ///
    pub fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &*coils;

        for sensor_name in self.results.keys() {
            for pf_coil_name in coils_local.results.get("pf").keys() {
                // Store - zero
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, 0.0);
            }
        }
    }

    ///
    pub fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &*passives;

        for sensor_name in self.results.keys() {
            // Calculate Greens with each passive degree of freedom
            for passive_name in passives_local.results.keys() {
                let _tmp: DataTreeAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();

                for dof_name in dof_names {
                    // Store
                    self.results
                        .get_or_insert(&sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(&passive_name)
                        .insert(&dof_name, 0.0);
                }
            }
        }
    }

    ///
    pub fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &*plasma;

        let n_r: usize = plasma_local.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma_local.results.get("grid").get("n_z").unwrap_usize();

        for sensor_name in self.results.keys() {
            // Create zero array
            let g_d_plasma_d_z: Array1<f64> = Array1::from_elem(n_z * n_r, 0.0);

            // Store
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
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
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Pressure>");
        string_output += &format!("║  {:<74} ║\n", version);

        // n_sensors = self.results
        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}

// Rust only methods
impl Pressure {
    /// This splits the Pressure into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent ojbects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        // Define empty data arrays
        let results_static_empty: SensorsStatic = SensorsStatic {
            greens_with_grid: Array2::zeros((0, 0)),
            greens_with_pf: Array2::zeros((0, 0)),
            greens_with_passives: Array2::zeros((0, 0)),
            greens_d_sensor_dz: Array2::zeros((0, 0)),
            fit_settings_weight: Array1::zeros(0),
            fit_settings_expected_value: Array1::zeros(0),
            geometry_r: Array1::zeros(0),
            geometry_z: Array1::zeros(0),
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

                if include_dynamic[i_time] == true {
                    include.push(true)
                } else {
                    include.push(false)
                }
            }

            // Convert from Vec<bool> to Vec of indices
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
                geometry_r: Array1::zeros(n_sensors), // used for pressure location
                geometry_z: Array1::zeros(n_sensors), // used for pressure location
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

    /// Calculate the sensor values
    pub fn calculate_sensor_values_rust(&mut self, plasma: &Plasma) {
        for sensor_name in self.results.keys() {}
    }
}
