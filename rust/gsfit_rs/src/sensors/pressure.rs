use crate::Plasma;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
use crate::source_functions::SourceFunctionTraits;
use crate::python_pickling_methods::{data_tree_to_py_dict, py_dict_to_data_tree};
use crate::sensors::static_and_dynamic_data_types::create_empty_sensor_data;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use std::f64::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};
use ndarray_stats::QuantileExt;
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, AddDataTreeGetters)]
#[pyclass(module = "gsfit_rs", skip_from_py_object)]
pub struct Pressure {
    pub results: DataTree,
}

impl Default for Pressure {
    fn default() -> Self {
        Self::new()
    }
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
    #[allow(clippy::too_many_arguments)]
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

        let n_time: usize = time_ndarray.len();

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

        // time-dependent "include"
        let mut include_dynamic: Vec<bool> = vec![fit_settings_include; n_time];
        for i_time in 0..n_time {
            if measured_ndarray[i_time].is_nan() {
                include_dynamic[i_time] = false;
            }
        }
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include_dynamic", include_dynamic);

        // Measurements
        self.results
            .get_or_insert(name)
            .get_or_insert("pressure")
            .get_or_insert("experimental")
            .insert("time", time_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("pressure")
            .get_or_insert("experimental")
            .insert("value", measured_ndarray);
    }

    ///
    pub fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &coils;

        for sensor_name in &self.results.keys() {
            for pf_coil_name in &coils_local.results.get("pf").keys() {
                // Store - zero
                self.results
                    .get_or_insert(sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(pf_coil_name, 0.0);
            }
        }
    }

    ///
    pub fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &passives;

        for sensor_name in &self.results.keys() {
            // Calculate Greens with each passive degree of freedom
            for passive_name in &passives_local.results.keys() {
                let _tmp: DataTreeAccumulator<'_> = passives_local.results.get(passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();

                for dof_name in &dof_names {
                    // Store
                    self.results
                        .get_or_insert(sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(passive_name)
                        .insert(dof_name, 0.0);
                }
            }
        }
    }

    /// Greens function with plasma
    ///
    /// # Arguments
    /// * `plasma` - The plasma object from Python
    ///
    /// # Returns
    /// * None
    ///
    pub fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &plasma;

        let n_r: usize = plasma_local.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma_local.results.get("grid").get("n_z").unwrap_usize();

        for sensor_name in &self.results.keys() {
            // Create zero array
            let g_plasma: Array1<f64> = Array1::from_elem(n_z * n_r, 0.0);
            let g_d_plasma_d_z: Array1<f64> = Array1::from_elem(n_z * n_r, 0.0);

            // Store
            self.results.get_or_insert(sensor_name).get_or_insert("greens").insert("plasma", g_plasma);
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
        }
    }

    /// Calculate the sensor values
    pub fn calculate_sensor_values(&mut self, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let plasma_rs: &Plasma = &plasma;

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

        string_output
    }

    /// Python pickling method
    pub fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let state_dict: Bound<'_, PyDict> = PyDict::new(py);
        let results_dict: Py<PyDict> = data_tree_to_py_dict(py, &self.results).expect("Failed to convert DataTree to PyDict");
        state_dict.set_item("results", results_dict).expect("Failed to add `results` key to dictionary");
        state_dict
            .set_item("version", env!("CARGO_PKG_VERSION"))
            .expect("Failed to add `version` key to dictionary");
        let system_time_now: SystemTime = SystemTime::now();
        let datetime: f64 = system_time_now
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .expect("Failed to get current datetime");
        state_dict
            .set_item("pickled_datetime", datetime)
            .expect("Failed to add `pickled_datetime` key to dictionary");
        Ok(state_dict.into())
    }

    /// Python unpickling method
    pub fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        let current_version: &str = env!("CARGO_PKG_VERSION");
        let pickled_version: String = state
            .get_item("version")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .expect("Pickled Pressure object has no version information");
        if pickled_version != current_version {
            eprintln!(
                "Warning: Unpickling Pressure object created with gsfit_rs v{}, but current version is v{}",
                pickled_version, current_version
            );
        }
        let results_dict: Bound<'_, PyAny> = state
            .get_item("results")
            .expect("Missing 'results' key in pickled data")
            .ok_or_else(|| PyTypeError::new_err("Missing 'results' key in pickled data"))
            .expect("Failed to get `results` from pickled data");
        let results_dict_bound: &Bound<'_, PyDict> = results_dict.cast::<PyDict>().expect("Failed to downcast `results` to PyDict");
        self.results = py_dict_to_data_tree(results_dict_bound).expect("Failed to convert PyDict to DataTree");
        Ok(())
    }
}

// Rust only methods
impl Pressure {
    /// This splits the Pressure into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent objects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        // Number of time-slices to reconstruct
        let n_time: usize = times_to_reconstruct.len();

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();
        let n_sensors_all: usize = sensor_names_all.len();

        // Interpolate all sensors to `times_to_reconstruct`
        let mut measured: Array2<f64> = Array2::from_elem((n_sensors_all, n_time), f64::NAN);
        for i_sensor in 0..n_sensors_all {
            let sensor_name: &str = &sensor_names_all[i_sensor];

            // Measured values
            let experimental_time: Array1<f64> = self.results.get(sensor_name).get("pressure").get("experimental").get("time").unwrap_array1();
            let experimental_values: Array1<f64> = self.results.get(sensor_name).get("pressure").get("experimental").get("value").unwrap_array1();

            // Create the interpolator
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(experimental_time.clone(), experimental_values.clone())
                .expect("Pressure.split_into_static_and_dynamic: Can't make interpolator");

            // Do the interpolation
            let measured_this_sensor: Array1<f64> = interpolator
                .interpolate_array1(times_to_reconstruct)
                .expect("Pressure.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_sensor, ..]).assign(&measured_this_sensor);

            // Store in self
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("pressure")
                .get_or_insert("measured")
                .insert("value", measured_this_sensor);
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("pressure")
                .get_or_insert("measured")
                .insert("time", times_to_reconstruct.clone());
        }

        // Create the time-dependent data structures
        let mut results_static: Vec<SensorsStatic> = Vec::with_capacity(n_time);
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);

        'time_loop: for i_time in 0..n_time {
            let mut include: Vec<bool> = Vec::new();
            for sensor_name in &sensor_names_all {
                let include_dynamic: Vec<bool> = self.results.get(sensor_name).get("fit_settings").get("include_dynamic").unwrap_vec_bool();
                include.push(include_dynamic[i_time]);
            }

            // Convert from Vec<bool> to Vec of indices
            let include_indices: Vec<usize> = include
                .iter()
                .enumerate()
                .filter(|(_, include)| **include) // Filter for `true` values
                .map(|(i, _)| i) // Extract the indices
                .collect(); // Collect into a Vec

            // If there are no sensors at this time-slice then push empty and continue
            if include_indices.is_empty() {
                let (static_data_empty, dynamic_data_empty): (SensorsStatic, SensorsDynamic) = create_empty_sensor_data();
                results_static.push(static_data_empty);
                results_dynamic.push(dynamic_data_empty);
                continue 'time_loop; // Go to next time-slice
            }

            // Down select sensor names
            let sensor_names: Vec<String> = include_indices.iter().map(|&index| sensor_names_all[index].clone()).collect();
            let n_sensors: usize = sensor_names.len();

            // Fit settings
            // Weight
            let fit_settings_weight: Array1<f64> = self.results.get("*").get("fit_settings").get("weight").unwrap_array1();
            let fit_settings_weight: Array1<f64> = fit_settings_weight.select(Axis(0), &include_indices);
            // Expected value
            let fit_settings_expected_value: Array1<f64> = self.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
            let fit_settings_expected_value: Array1<f64> = fit_settings_expected_value.select(Axis(0), &include_indices);

            // Greens
            // With PF coils
            let greens_with_pf: Array2<f64> = self.results.get("*").get("greens").get("pf").get("*").unwrap_array2(); // shape = [n_pf, n_sensors]
            let greens_with_pf: Array2<f64> = greens_with_pf.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_pf, n_sensors]
            // With plasma
            let greens_with_grid: Array2<f64> = self.results.get("*").get("greens").get("plasma").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
            let greens_with_grid: Array2<f64> = greens_with_grid.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]
            // With d_sensor_dz (for vertical stability)
            let greens_d_sensor_dz: Array2<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
            let greens_d_sensor_dz: Array2<f64> = greens_d_sensor_dz.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]

            // Geometry
            let geometry_r: Array1<f64> = self.results.get("*").get("geometry").get("r").unwrap_array1();
            let geometry_r: Array1<f64> = geometry_r.select(Axis(0), &include_indices);
            let geometry_z: Array1<f64> = self.results.get("*").get("geometry").get("z").unwrap_array1();
            let geometry_z: Array1<f64> = geometry_z.select(Axis(0), &include_indices);

            // With passives
            let passive_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").keys();
            let n_passives: usize = passive_names.len();

            // Count the number of degrees of freedom
            let mut n_dof_total: usize = 0;
            for passive_name in &passive_names {
                let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys();
                n_dof_total += dof_names.len();
            }

            let mut greens_with_passives: Array2<f64> = Array2::from_elem((n_dof_total, n_sensors), f64::NAN);
            for i_sensor in 0..n_sensors {
                let mut i_dof_total: usize = 0;
                for i_passive in 0..n_passives {
                    let passive_name: &str = &passive_names[i_passive];
                    let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
                    for dof_name in &dof_names {
                        greens_with_passives[(i_dof_total, i_sensor)] = self
                            .results
                            .get(&sensor_names[i_sensor])
                            .get("greens")
                            .get("passives")
                            .get(passive_name)
                            .get(dof_name)
                            .unwrap_f64();

                        // Keep count
                        i_dof_total += 1;
                    }
                }
            }

            // Create the `SensorsStatic` data
            let results_static_this_time_slice: SensorsStatic = SensorsStatic {
                greens_with_grid,
                greens_with_pf,
                greens_with_passives,
                greens_d_sensor_dz,
                fit_settings_weight,
                fit_settings_expected_value,
                geometry_r,
                geometry_z,
            };
            results_static.push(results_static_this_time_slice);

            // Select time-slice and the sensors we use in reconstruction
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        // Return the static and dynamic results
        (results_static, results_dynamic)
    }

    /// Calculate the sensor values
    pub fn calculate_sensor_values_rust(&mut self, plasma: &Plasma) {
        let time: Array1<f64> = plasma.results.get("time").unwrap_array1();
        let n_time: usize = time.len();

        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let psi_2d_vs_time: Array3<f64> = plasma.results.get("two_d").get("psi").unwrap_array3();
        let br_2d_vs_time: Array3<f64> = plasma.results.get("two_d").get("br").unwrap_array3();
        let bz_2d_vs_time: Array3<f64> = plasma.results.get("two_d").get("bz").unwrap_array3();
        let d_bz_d_z_2d_vs_time: Array3<f64> = plasma.results.get("two_d").get("d_bz_d_z").unwrap_array3();
        let p_prime_dof_values_vs_time: Array2<f64> = plasma.results.get("source_functions").get("p_prime").get("coefficients").unwrap_array2();

        let psi_a_vs_time: Array1<f64> = plasma.results.get("global").get("psi_a").unwrap_array1();
        let psi_b_vs_time: Array1<f64> = plasma.results.get("global").get("psi_b").unwrap_array1();

        let sensor_names: Vec<String> = self.results.keys();
        if sensor_names.is_empty() {
            println!("Warning: Pressure::calculate_sensor_values_rust called with no sensors. Was `pressure_sensors` overwritten by `setup_objects()`?");
            return;
        }

        for sensor_name in &sensor_names {
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

            // Find the value of psi_n at the location of the pressure sensor
            let sensor_r: f64 = self.results.get(sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(sensor_name).get("geometry").get("z").unwrap_f64();

            if sensor_r < r[0] || sensor_r > r[r.len() - 1] || sensor_z < z[0] || sensor_z > z[z.len() - 1] {
                panic!(
                    "Pressure sensor `{sensor_name}` is outside the plasma grid. sensor_r = {sensor_r}, sensor_z = {sensor_z}, min(r) = {}, max(r) = {}, min(z) = {}, max(z) = {}",
                    r[0],
                    r[r.len() - 1],
                    z[0],
                    z[z.len() - 1]
                );
            }

            // Find the nearest grid point to the sensor location
            let i_r_nearest: usize = (&r - sensor_r).abs().argmin().expect("unwrapping i_r_nearest");
            let i_z_nearest: usize = (&z - sensor_z).abs().argmin().expect("unwrapping i_z_nearest");

            // Find the four corner grid points surrounding the pressure sensor
            let i_r_nearest_left: usize;
            let i_r_nearest_right: usize;
            let i_z_nearest_lower: usize;
            let i_z_nearest_upper: usize;
            if sensor_r > r[i_r_nearest] {
                i_r_nearest_left = i_r_nearest;
                i_r_nearest_right = i_r_nearest + 1;
            } else {
                i_r_nearest_left = i_r_nearest - 1;
                i_r_nearest_right = i_r_nearest;
            }
            if sensor_z > z[i_z_nearest] {
                i_z_nearest_lower = i_z_nearest;
                i_z_nearest_upper = i_z_nearest + 1;
            } else {
                i_z_nearest_lower = i_z_nearest - 1;
                i_z_nearest_upper = i_z_nearest;
            }

            for i_time in 0..n_time {
                let psi_a: f64 = psi_a_vs_time[i_time];
                let psi_b: f64 = psi_b_vs_time[i_time];
                let psi_2d: Array2<f64> = psi_2d_vs_time.slice(s![i_time, .., ..]).to_owned();
                let br_2d: Array2<f64> = br_2d_vs_time.slice(s![i_time, .., ..]).to_owned();
                let bz_2d: Array2<f64> = bz_2d_vs_time.slice(s![i_time, .., ..]).to_owned();
                let d_bz_d_z_2d: Array2<f64> = d_bz_d_z_2d_vs_time.slice(s![i_time, .., ..]).to_owned();

                // Bicubic interpolation of psi at the sensor location
                let f: ArrayView2<f64> = psi_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let mut d_f_d_r: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
                let mut d_f_d_z: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
                let mut d2_f_d_r_d_z: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

                // d(psi)/d(r): bz = 1 / (2 * pi * r) * d(psi)/d(r)
                d_f_d_r[(0, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d_f_d_r[(1, 0)] = bz_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d_f_d_r[(0, 1)] = bz_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                d_f_d_r[(1, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                // d(psi)/d(z): br = -1 / (2 * pi * r) * d(psi)/d(z)
                d_f_d_z[(0, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d_f_d_z[(1, 0)] = -br_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d_f_d_z[(0, 1)] = -br_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                d_f_d_z[(1, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                // d^2(psi)/(d(r)*d(z)): d(bz)/d(z) = 1 / (2 * pi * r) * d^2(psi)/(d(r)*d(z))
                d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                let bicubic_interpolator: BicubicInterpolator =
                    BicubicInterpolator::new(d_r, d_z, f.view(), d_f_d_r.view(), d_f_d_z.view(), d2_f_d_r_d_z.view());
                let x: f64 = (sensor_r - r[i_r_nearest_left]) / d_r;
                let y: f64 = (sensor_z - z[i_z_nearest_lower]) / d_z;
                let psi_at_sensor: f64 = bicubic_interpolator.interpolate(x, y);

                let psi_n_at_sensor: f64 = (psi_at_sensor - psi_a) / (psi_b - psi_a);

                // If sensor is outside the plasma boundary, leave sensor_values[i_time] as NaN
                if !(0.0..=1.0).contains(&psi_n_at_sensor) {
                    continue;
                }

                // Analytically integrate p'(psi_n) with boundary condition p(psi_n = 1) = 0.
                // Note: source_function_integral returns an integral from psi_n = 1 to psi_n,
                // i.e. p(psi_n) = integral_{1}^{psi_n} p'(x) dx, scaled by (psi_b - psi_a).
                let p_prime_dof_values: Array1<f64> = p_prime_dof_values_vs_time.row(i_time).to_owned();
                sensor_values[i_time] = plasma.p_prime_source_function
                    .source_function_integral(&Array1::from_vec(vec![psi_n_at_sensor]), &p_prime_dof_values)[0]
                    * (psi_b - psi_a);
            }

            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("pressure")
                .get_or_insert("calculated")
                .insert("value", sensor_values);
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("pressure")
                .get_or_insert("calculated")
                .insert("time", time.clone());
        }
    }
}
