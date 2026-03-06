use crate::coils::Coils;
use crate::greens::greens_b;
use crate::greens::greens_d_b_d_z;
use crate::passives::PassiveGeometryAll;
use crate::passives::Passives;
use crate::plasma::Plasma;
use crate::python_pickling_methods::{data_tree_to_py_dict, py_dict_to_data_tree};
use crate::sensors::static_and_dynamic_data_types::create_empty_sensor_data;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, Axis, s};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, AddDataTreeGetters)]
#[pyclass(module = "gsfit_rs")]
pub struct BpProbes {
    pub results: DataTree,
}

/// Python accessible methods
#[pymethods]
impl BpProbes {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    /// Add a bp-probe sensor (Mirnov coil)
    ///
    /// # Arguments
    /// * `name` - Name of the sensor
    /// * `geometry_angle_pol` - Poloidal angle of the sensor, radian
    /// * `geometry_r` - R coordinate of the sensor, meter
    /// * `geometry_z` - Z coordinate of the sensor, meter
    /// * `fit_settings_comment` - Comment for the fit settings
    /// * `fit_settings_expected_value` - Expected value for the fit settings
    /// * `fit_settings_include` - Include in the fit settings
    /// * `fit_settings_weight` - Weight in the fit settings
    /// * `time` - Time array of the measured data, second
    /// * `measured` - Measured data array, tesla
    ///
    /// # Returns
    /// None
    ///
    pub fn add_sensor(
        &mut self,
        name: &str,
        geometry_angle_pol: f64,
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
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("angle_pol", geometry_angle_pol);
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

        // Experimental measurements
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("time", time_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("value", measured_ndarray);
    }

    /// Greens with coils
    pub fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &*coils;

        // Run the Rust method
        self.greens_with_coils_rs(coils_local.to_owned());
    }

    /// Greens with passives
    pub fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &*passives;

        // Run the Rust method
        self.greens_with_passives_rs(passives_local.to_owned());
    }

    /// Greens with plasma
    pub fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &*plasma;

        // Run the Rust method
        self.greens_with_plasma_rs(plasma_local.to_owned());
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let coils_rs: &Coils = &*coils;
        let passives_rs: &Passives = &*passives;
        let plasma_rs: &Plasma = &*plasma;

        // Run the Rust method
        self.calculate_sensor_values_rs(coils_rs, passives_rs, plasma_rs);
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values_vacuum(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>) {
        // TODO: this should be combined with `calculate_sensor_values`

        let passives_geometry: PassiveGeometryAll = passives.get_all_passive_filament_geometry();
        let passives_r: Array1<f64> = passives_geometry.r;
        let passives_z: Array1<f64> = passives_geometry.z;

        let passive_currents: Array2<f64> = passives.get_passive_filament_currents_from_simulated();
        for sensor_name in &self.results.keys() {
            let sensor_r: f64 = self.results.get(sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(sensor_name).get("geometry").get("z").unwrap_f64();
            let sensor_angle_pol: f64 = self.results.get(sensor_name).get("geometry").get("angle_pol").unwrap_f64();
            let sensor_r_array: Array1<f64> = Array1::from_vec(vec![sensor_r]);
            let sensor_z_array: Array1<f64> = Array1::from_vec(vec![sensor_z]);

            // Calculate Greens:  g_br.shape() = (1, n_passives);  g_bz.shape() = (1, n_passives)
            let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens_b(sensor_r_array, sensor_z_array, passives_r.clone(), passives_z.clone());
            let g_br: Array1<f64> = g_br.sum_axis(Axis(0));
            let g_bz: Array1<f64> = g_bz.sum_axis(Axis(0));
            let g: Array1<f64> = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin(); // shape = (n_passives)

            // Coils
            let g_with_coils: Array1<f64> = self.results.get(sensor_name).get("greens").get("pf").get("*").unwrap_array1(); // shape = (n_pf)
            let coil_currents: Array2<f64> = coils.results.get("pf").get("*").get("i").get("simulated").get("value").unwrap_array2(); // shape = (n_time, n_pf)
            // let n_time: usize = coil_currents.len_of(Axis(0));

            let pf_names: Vec<String> = coils.results.get("pf").keys();
            let simulated_time: Array1<f64> = coils.results.get("pf").get(&pf_names[0]).get("i").get("simulated").get("time").unwrap_array1();
            let n_time: usize = simulated_time.len();

            // Loop over time
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                // PF coil
                let sensor_value_from_coils: f64 = g_with_coils.dot(&coil_currents.slice(s![i_time, ..]));

                // Passives
                let sensor_value_from_passives: f64 = g.dot(&passive_currents.slice(s![i_time, ..]));

                // Total
                sensor_values[i_time] = sensor_value_from_coils + sensor_value_from_passives;
            }

            // Store simulated sensor values
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("simulated")
                .insert("value", sensor_values);
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("simulated")
                .insert("time", simulated_time.clone());
        }
    }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.BpProbes>");
        string_output += &format!("║  {:<74} ║\n", version);

        // n_sensors = self.results
        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }

    /// Python pickling method
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Create a Python dictionary, which will be pickled
        let state_dict: Bound<'_, PyDict> = PyDict::new(py);

        // Store the self.results DataTree under "results" key
        let results_dict: Py<PyDict> = data_tree_to_py_dict(py, &self.results).expect("Failed to convert DataTree to PyDict");
        state_dict.set_item("results", results_dict).expect("Failed to add `results` key to dictionary");

        // Store `gsfit_rs` version
        state_dict
            .set_item("version", env!("CARGO_PKG_VERSION"))
            .expect("Failed to add `version` key to dictionary");

        // Store current datetime as Unix timestamp (seconds since epoch)
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
    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        // Extract the "results" key and convert back to DataTree
        let results_dict: Bound<'_, PyAny> = state
            .get_item("results")
            .expect("Missing 'results' key in pickled data")
            .ok_or_else(|| PyTypeError::new_err("Missing 'results' key in pickled data"))
            .expect("Failed to get `results` from pickled data");
        let results_dict_bound: &Bound<'_, PyDict> = results_dict.downcast::<PyDict>().expect("Failed to downcast `results` to PyDict");

        // Insert into self
        self.results = py_dict_to_data_tree(results_dict_bound).expect("Failed to convert PyDict to DataTree");

        // Return Ok to signal successful completion, no "data" returned
        Ok(())
    }
}

// Rust only methods
impl BpProbes {
    /// This splits the BpProbes into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent objects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    /// TODO: change `SensorsStatic` to Vec<SensorsStatic> to be consistent with other sensor types.
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
                    greens_with_passives[(i_dof_total, i_sensor)] = self
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

        // Create the `BpProbesStatic` data
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid,
            greens_with_pf,
            greens_with_passives,
            greens_d_sensor_dz,
            fit_settings_weight,
            fit_settings_expected_value,
            geometry_r: Array1::zeros(n_sensors), // not used for BpProbes
            geometry_z: Array1::zeros(n_sensors), // not used for BpProbes
        };

        // Time dependent
        // Interpolate all sensors to `times_to_reconstruct`
        let mut measured: Array2<f64> = Array2::zeros((n_sensors_all, n_time));
        for i_sensor in 0..n_sensors_all {
            // Sensor names
            let sensor_name: &str = &sensor_names_all[i_sensor];

            // Measured values
            let experimental_time: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("time").unwrap_array1();
            let experimental_values: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("value").unwrap_array1();

            // Create the interpolator
            let interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(experimental_time.clone(), experimental_values.clone()).expect("Can't make interpolator");

            // Do the interpolation
            let measured_this_sensor: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).expect("Can't do interpolation");

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
            // Select time-slide and the sensors we use in reconstruction
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            // Create new `SensorsDynamic` instance and store
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        let results_static_time_dependent: Vec<SensorsStatic> = vec![results_static.clone(); n_time];
        // Return the static and dynamic results
        return (results_static_time_dependent, results_dynamic);
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values_rs(&mut self, coils: &Coils, passives: &Passives, plasma: &Plasma) {
        for sensor_name in self.results.keys() {
            // Coils
            let g_with_coils: Array1<f64> = self.results.get(&sensor_name).get("greens").get("pf").get("*").unwrap_array1(); // shape = [n_pf]
            let coil_currents: Array2<f64> = coils.results.get("pf").get("*").get("i").get("measured").get("value").unwrap_array2(); // shape = [n_time, n_pf]
            // let n_time: usize = coil_currents.len_of(Axis(0));

            // Plasma
            let g_with_plasma: Array1<f64> = self.results.get(&sensor_name).get("greens").get("plasma").unwrap_array1(); // shape = [n_z*n_r]
            let j_2d: Array3<f64> = plasma.results.get("two_d").get("j").unwrap_array3(); // shape = [n_time, n_z, n_r]
            let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
            let time: Array1<f64> = plasma.results.get("time").unwrap_array1();
            let n_time: usize = time.len();

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
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("value", sensor_values);
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("time", time.clone());
        }
    }

    pub fn add_sensor_rs(
        &mut self,
        name: &str,
        geometry_angle_pol: f64,
        geometry_r: f64,
        geometry_z: f64,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: Array1<f64>,
        measured: Array1<f64>,
    ) {
        // Geometry
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("angle_pol", geometry_angle_pol);
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
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("time", time);
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("value", measured);
    }

    pub fn greens_with_coils_rs(&mut self, coils: Coils) {
        for sensor_name in self.results.keys() {
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            for pf_coil_name in coils.results.get("pf").keys() {
                let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

                let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                    Array1::from_vec(vec![sensor_r]),
                    Array1::from_vec(vec![sensor_z]),
                    coil_r.clone(),
                    coil_z.clone(),
                );

                // Sum over all the current sources
                let g_br: f64 = g_br_full.sum();
                let g_bz: f64 = g_bz_full.sum();

                // Sensors Green's function
                let g: f64 = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, g);
            }
        }
    }

    pub fn greens_with_plasma_rs(&mut self, plasma: Plasma) {
        let plasma_r: Array1<f64> = plasma.results.get("grid").get("flat").get("r").unwrap_array1();
        let plasma_z: Array1<f64> = plasma.results.get("grid").get("flat").get("z").unwrap_array1();

        for sensor_name in self.results.keys() {
            // Get variables out of self
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                Array1::from_vec(vec![sensor_r]), // sensor
                Array1::from_vec(vec![sensor_z]),
                plasma_r.clone(), // current source
                plasma_z.clone(),
            );

            let g_br: Array1<f64> = g_br_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]
            let g_bz: Array1<f64> = g_bz_full.sum_axis(Axis(0));

            // Sensors Green's function
            let g_with_plasma: Array1<f64> = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

            // Store
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma); // shape = [(n_z * n_r)]

            // Vertical stability
            let (g_d_plasma_br_d_z_full, g_d_plasma_bz_d_z_full): (Array2<f64>, Array2<f64>) = greens_d_b_d_z(
                Array1::from_vec(vec![sensor_r]), // sensor
                Array1::from_vec(vec![sensor_z]),
                plasma_r.clone(), // current source
                plasma_z.clone(),
            );

            let g_d_plasma_br_d_z: Array1<f64> = g_d_plasma_br_d_z_full.sum_axis(Axis(0)); // g_d_plasma_br_d_z_full.shape = [1, n_z * n_r];  g_d_plasma_br_d_z.shape = [n_z * n_r]
            let g_d_plasma_bz_d_z: Array1<f64> = g_d_plasma_bz_d_z_full.sum_axis(Axis(0));

            // Sensors Green's function
            let g_d_plasma_d_z: Array1<f64> = g_d_plasma_br_d_z * sensor_angle_pol.cos() + g_d_plasma_bz_d_z * sensor_angle_pol.sin();

            // Store
            // TODO: should I reshape to Array2 [n_z, n_r] ????
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
        }
    }

    pub fn greens_with_passives_rs(&mut self, passives: Passives) {
        // Loop over sensors
        for sensor_name in self.results.keys() {
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();

            // Calculate Greens with each passive degree of freedom
            for passive_name in passives.results.keys() {
                let _tmp: DataTreeAccumulator<'_> = passives.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();
                let passive_r: Array1<f64> = passives.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
                let passive_z: Array1<f64> = passives.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

                for dof_name in dof_names {
                    let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                        Array1::from_vec(vec![sensor_r]), // by convention (r, z) are "sensors"
                        Array1::from_vec(vec![sensor_z]),
                        passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
                        passive_z.clone(),
                    );

                    // Current distribution
                    let current_distribution: Array1<f64> = passives
                        .results
                        .get(&passive_name)
                        .get("dof")
                        .get(&dof_name)
                        .get("current_distribution")
                        .unwrap_array1();

                    let g_br_with_dof_full: Array2<f64> = g_br_full * &current_distribution; // shape = [n_passive_dof, n_filament]
                    let g_bz_with_dof_full: Array2<f64> = g_bz_full * current_distribution; // shape = [n_passive_dof, n_filament]

                    // Sum over all filaments
                    let g_br: f64 = g_br_with_dof_full.sum(); // shape = [n_passive_dof]
                    let g_bz: f64 = g_bz_with_dof_full.sum(); // shape = [n_passive_dof]

                    // Calculate Green's function
                    let g: f64 = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

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
}
