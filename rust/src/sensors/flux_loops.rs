use crate::Plasma;
use crate::coils::Coils;
use crate::greens::greens;
use crate::greens::greens_magnetic_field;
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_interp::interp1d::Interp1D;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

const PI: f64 = std::f64::consts::PI;

#[derive(Clone)]
#[pyclass]
pub struct FluxLoops {
    pub results: NestedDict,
}

/// Python accessible methods
#[pymethods]
impl FluxLoops {
    #[new]
    pub fn new() -> Self {
        Self { results: NestedDict::new() }
    }

    /// Data structure:
    ///
    /// # Examples
    ///
    /// ```
    /// [probe_name]["psi"]["calculated"]                                 = Array1<f64>;  shape=[n_time]
    /// [probe_name]["psi"]["measured"]                                   = Array1<f64>;  shape=[n_time]
    /// [probe_name]["fit_settings"]["comment"]                         = str
    /// [probe_name]["fit_settings"]["expected_value"]                  = f64
    /// [probe_name]["fit_settings"]["include"]                         = bool
    /// [probe_name]["fit_settings"]["weight"]                          = f64
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
        time: &Bound<'_, PyArray1<f64>>,
        measured: &Bound<'_, PyArray1<f64>>,
    ) {
        // Convert to Rust data types
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let measured_ndarray: Array1<f64> = Array1::from(unsafe { measured.as_array() }.to_vec());

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
        self.results.get_or_insert(name).get_or_insert("psi").insert("time_experimental", time_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("psi")
            .insert("measured_experimental", measured_ndarray);
    }

    ///
    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &*coils;

        for sensor_name in self.results.keys() {
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            for pf_coil_name in coils_local.results.get("pf").keys() {
                let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

                let g_full: Array2<f64> = greens(
                    Array1::from_vec(vec![sensor_r]),
                    Array1::from_vec(vec![sensor_z]),
                    coil_r.clone(),
                    coil_z.clone(),
                    coil_r.clone() * 0.0,
                    coil_r.clone() * 0.0,
                );

                // Sum over all the current sources
                let g: f64 = g_full.sum();

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, g);
            }
        }
    }

    ///
    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &*passives;

        for sensor_name in self.results.keys() {
            // Get variables out of self
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            // Calculate Greens with each passive degree of freedom
            for passive_name in passives_local.results.keys() {
                let _tmp: NestedDictAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();
                let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
                let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

                for dof_name in dof_names {
                    let g_full: Array2<f64> = greens(
                        Array1::from_vec(vec![sensor_r]), // by convention (r, z) are "sensors"
                        Array1::from_vec(vec![sensor_z]),
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

                    let g_with_dof_full: Array2<f64> = g_full * &current_distribution; // shape = [n_r * n_z, n_filament]

                    // Sum over all filaments
                    let g: f64 = g_with_dof_full.sum(); // shape = [n_r * n_z]

                    // Store
                    self.results
                        .get_or_insert(&sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(&passive_name)
                        .insert(&dof_name, g);
                }
            }
        }
    }

    fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &*plasma;

        let plasma_r: Array1<f64> = plasma_local.results.get("grid").get("flat").get("r").unwrap_array1();
        let plasma_z: Array1<f64> = plasma_local.results.get("grid").get("flat").get("z").unwrap_array1();

        for sensor_name in self.results.keys() {
            // Get variables out of self
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            let g_full: Array2<f64> = greens(
                Array1::from_vec(vec![sensor_r]), // sensor
                Array1::from_vec(vec![sensor_z]),
                plasma_r.clone(), // current source
                plasma_z.clone(),
                plasma_r.clone() * 0.0, // TODO: I don't like the dr_prime and dz_prime
                plasma_r.clone() * 0.0,
            );

            // Sensors Green's function
            let g_with_plasma: Array1<f64> = g_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]

            // Store greens
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma);

            // Vertical stability
            let (g_br_full, _g_bz_full): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
                Array1::from_vec(vec![sensor_r]), // sensors
                Array1::from_vec(vec![sensor_z]),
                plasma_r.clone(), // current sources
                plasma_z.clone(),
            );

            let g_d_plasma_d_z: Array1<f64> = -2.0 * PI * sensor_r.clone() * g_br_full.sum_axis(Axis(0));

            // Store
            // TODO: should I reshape to Array2 [n_z, n_r] ????
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
        }
    }

    fn calculate_sensor_values(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let coils_rs: &Coils = &*coils;
        let passives_rs: &Passives = &*passives;
        let plasma_rs: &Plasma = &*plasma;

        // Run the Rust method
        self.calculate_sensor_values_rust(coils_rs, passives_rs, plasma_rs);
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.FluxLoops>");
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
impl FluxLoops {
    /// This splits the FluxLoops into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent ojbects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (SensorsStatic, Vec<SensorsDynamic>) {
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
        let greens_with_grid: Array2<f64> = greens_with_grid.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_pf, n_sensors]
        // With d_sensor_dz (for vertical stability)
        let greens_d_sensor_dz: Array2<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
        let greens_d_sensor_dz: Array2<f64> = greens_d_sensor_dz.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]

        // With passives
        let passive_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_dof_total, n_sensors));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        for i_sensor in 0..n_sensors {
            let mut i_dof_total: usize = 0;
            for i_passive in 0..n_passives {
                let passive_name: &str = &passive_names[i_passive];
                let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
                for dof_name in dof_names {
                    greens_with_passives[[i_dof_total, i_sensor]] = self
                        .results
                        .get(&sensor_names[i_sensor])
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .unwrap_f64();

                    // Store the name
                    // dof_names_total[i_dof_total] = dof_name;

                    // Keep count
                    i_dof_total += 1;
                }
            }
        }

        // Create the `FluxLoopsStatic` data
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid,
            greens_with_pf,
            greens_with_passives,
            greens_d_sensor_dz,
            fit_settings_weight,
            fit_settings_expected_value,
        };

        // Time dependent
        // Interpolate all sensors to `times_to_reconstruct`
        let n_time: usize = times_to_reconstruct.len();
        let mut measured: Array2<f64> = Array2::zeros((n_sensors_all, n_time));
        for i_sensor in 0..n_sensors_all {
            // Sensor names
            let sensor_name: &str = &sensor_names_all[i_sensor];

            // Measured values
            let time_experimental: Array1<f64> = self.results.get(sensor_name).get("psi").get("time_experimental").unwrap_array1();
            let measured_experimental: Array1<f64> = self.results.get(sensor_name).get("psi").get("measured_experimental").unwrap_array1();

            // Create the interpolator
            let interpolator = Interp1D::builder(measured_experimental)
                .x(time_experimental.clone())
                .build()
                .expect("FluxLoops.split_into_static_and_dynamic: Can't make Interp1D");

            // Do the interpolation
            let measured_this_coil: Array1<f64> = interpolator
                .interp_array(&times_to_reconstruct)
                .expect("FluxLoops.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_sensor, ..]).assign(&measured_this_coil);

            // Store in self
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("psi")
                .insert("measured", measured_this_coil);
        }

        // MDSplus is "Sensor-Major", but we want to rearrange the data to be "Time-Major"
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            // Select time-slide and the sensors we use in reconstruction
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            // Create new `SensorsDynamic` instance and store
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        return (results_static, results_dynamic);
    }

    pub fn calculate_sensor_values_rust(&mut self, coils: &Coils, passives: &Passives, plasma: &Plasma) {
        for sensor_name in self.results.keys() {
            // Coils
            let g_with_coils: Array1<f64> = self.results.get(&sensor_name).get("greens").get("pf").get("*").unwrap_array1(); // shape = [n_pf]
            let coil_currents: Array2<f64> = coils.results.get("pf").get("*").get("i").get("measured").unwrap_array2(); // shape = [n_time, n_pf]
            let n_time: usize = coil_currents.len_of(Axis(0));

            // Plasma
            let g_with_plasma: Array1<f64> = self.results.get(&sensor_name).get("greens").get("plasma").unwrap_array1(); // shape = [n_z*n_r]
            let j_2d: Array3<f64> = plasma.results.get("two_d").get("j").unwrap_array3(); // shape = [n_time, n_z, n_r]
            let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();

            // Loop over time
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                // PF coil
                let sensor_values_from_coils: f64 = g_with_coils.dot(&coil_currents.slice(s![i_time, ..]));

                // Passives
                // TODO: this looks slow!
                let mut sensor_values_from_passives: f64 = 0.0;
                let passive_names: Vec<String> = self.results.get(&sensor_name).get("greens").get("passives").keys();
                for passive_name in passive_names {
                    let dof_names: Vec<String> = self.results.get(&sensor_name).get("greens").get("passives").get(&passive_name).keys();
                    for dof_name in dof_names {
                        let g_with_passive: f64 = self
                            .results
                            .get(&sensor_name)
                            .get("greens")
                            .get("passives")
                            .get(&passive_name)
                            .get(&dof_name)
                            .unwrap_f64();
                        let passive_dof_value: f64 = passives.results.get(&passive_name).get("dof").get(&dof_name).get("calculated").unwrap_array1()[i_time];
                        sensor_values_from_passives += g_with_passive * passive_dof_value;
                    }
                }

                // Plasma
                let j_2d_flat: Array1<f64> = Array1::from_iter(j_2d.slice(s![i_time, .., ..]).iter().copied());
                let sensor_values_from_plasma: f64 = (&g_with_plasma * j_2d_flat).sum() * d_area;

                // Total
                sensor_values[i_time] = sensor_values_from_coils + sensor_values_from_passives + sensor_values_from_plasma;
            }

            // Store this sensor
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("psi")
                .insert("calculated", sensor_values);
        }
    }
}
