use crate::greens::Greens;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use crate::plasma::Plasma;
use crate::python_pickling_methods::{data_tree_to_py_dict, py_dict_to_data_tree};
use crate::sensors::static_and_dynamic_data_types::create_empty_sensor_data;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use ndarray::{Array1, Array2, Array3, Axis, array, s};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, AddDataTreeGetters)]
#[pyclass(module = "gsfit_rs", skip_from_py_object)]
pub struct BpProbes {
    pub results: DataTree,
}

impl Default for BpProbes {
    fn default() -> Self {
        Self::new()
    }
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
    /// * `geometry_angle_pol` - Poloidal angle of the sensor, [radian]
    /// * `geometry_r` - R coordinate of the sensor, [metre]
    /// * `geometry_z` - Z coordinate of the sensor, [metre]
    /// * `fit_settings_comment` - Comment for the fit settings
    /// * `fit_settings_expected_value` - Expected value for the fit settings
    /// * `fit_settings_include` - Include in the fit settings
    /// * `fit_settings_weight` - Weight in the fit settings
    /// * `time` - Time array of the measured data, [second]
    /// * `measured` - Measured data array, [tesla]
    ///
    /// # Returns
    /// None
    #[allow(clippy::too_many_arguments)]
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

    /// Greens with the PF coils
    pub fn greens_with_pf_active(&mut self, pf_active: PyRef<PfActive>) {
        // Run the Rust method
        self.greens_with_pf_active_rs(&pf_active);
    }

    /// Greens with the passive conductors
    pub fn greens_with_pf_passive(&mut self, pf_passive: PyRef<PfPassive>) {
        // Run the Rust method
        self.greens_with_pf_passive_rs(&pf_passive);
    }

    /// Greens with plasma
    pub fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &plasma;

        // Run the Rust method
        self.greens_with_plasma_rs(plasma_local);
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values(&mut self, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let plasma_rs: &Plasma = &plasma;

        // Run the Rust method
        self.calculate_sensor_values_rs(plasma_rs);
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values_vacuum(&mut self, pf_active: PyRef<PfActive>, pf_passive: PyRef<PfPassive>) {
        // TODO: this should be combined with `calculate_sensor_values`

        // Every passive filament, and its current, which `solve_circuit_equations` simulated, on the timebase it simulated the
        // PF coil currents on
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut passives_r_vec: Vec<f64> = Vec::new();
        let mut passives_z_vec: Vec<f64> = Vec::new();
        let mut passive_currents_vec: Vec<Array1<f64>> = Vec::new();
        for i_loop in 0..n_loop {
            let filaments: PassiveFilaments = passive_filaments(&pf_passive_loops[i_loop].element);
            passives_r_vec.extend(filaments.r.iter());
            passives_z_vec.extend(filaments.z.iter());
            let n_element: usize = pf_passive_loops[i_loop].element.len();
            for i_element in 0..n_element {
                passive_currents_vec.push(pf_passive_loops[i_loop].element[i_element].current.clone());
            }
        }
        let passives_r: Array1<f64> = Array1::from_vec(passives_r_vec);
        let passives_z: Array1<f64> = Array1::from_vec(passives_z_vec);
        let n_passive_filaments: usize = passive_currents_vec.len();
        let n_time_passive: usize = if n_passive_filaments > 0 { passive_currents_vec[0].len() } else { 0 };
        let mut passive_currents: Array2<f64> = Array2::from_elem((n_time_passive, n_passive_filaments), f64::NAN); // shape = (n_time, n_passive_filaments)
        for i_filament in 0..n_passive_filaments {
            passive_currents.slice_mut(s![.., i_filament]).assign(&passive_currents_vec[i_filament]);
        }

        for sensor_name in &self.results.keys() {
            let sensor_r: f64 = self.results.get(sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(sensor_name).get("geometry").get("z").unwrap_f64();
            let sensor_angle_pol: f64 = self.results.get(sensor_name).get("geometry").get("angle_pol").unwrap_f64();

            let greens_calculator: Greens = Greens::sensor_to_conductor(
                array![sensor_r],
                array![sensor_z],
                passives_r.clone(),
                passives_z.clone(),
                passives_r.clone() * 0.0, // TODO: should this be NaN instead?
                passives_z.clone() * 0.0,
            );

            // Calculate Greens
            let g_br_matrix: Array2<f64> = greens_calculator.b_r(); // shape() = (1, n_passives)
            let g_bz_matrix: Array2<f64> = greens_calculator.b_z(); // shape() = (1, n_passives)
            let g_br: Array1<f64> = g_br_matrix.sum_axis(Axis(0));
            let g_bz: Array1<f64> = g_bz_matrix.sum_axis(Axis(0));
            let g_with_passives: Array1<f64> = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin(); // shape = (n_passives)

            // The PF coils' currents, which share one timebase, e.g. the one `solve_circuit_equations` simulated them on,
            // and this sensor's Greens with each coil, found by name
            let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
            let n_pf: usize = pf_active_coils.len();
            let simulated_time: Array1<f64> = pf_active_coils[0].current.time.clone();
            let n_time: usize = simulated_time.len();
            let mut g_with_coils: Array1<f64> = Array1::from_elem(n_pf, f64::NAN); // shape = (n_pf)
            let mut coil_currents: Array2<f64> = Array2::from_elem((n_time, n_pf), f64::NAN); // shape = (n_time, n_pf)
            for i_pf in 0..n_pf {
                g_with_coils[i_pf] = self
                    .results
                    .get(sensor_name)
                    .get("greens")
                    .get("pf")
                    .get(&pf_active_coils[i_pf].name)
                    .unwrap_f64();
                coil_currents.slice_mut(s![.., i_pf]).assign(&pf_active_coils[i_pf].current.data);
            }

            // Loop over time
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                // PF coil
                let sensor_value_from_coils: f64 = g_with_coils.dot(&coil_currents.slice(s![i_time, ..]));

                // Passives
                let sensor_value_from_passives: f64 = g_with_passives.dot(&passive_currents.slice(s![i_time, ..]));

                // Total
                sensor_values[i_time] = sensor_value_from_coils + sensor_value_from_passives;
            }

            // Store simulated sensor values
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("b")
                .get_or_insert("simulated")
                .insert("value", sensor_values);
            self.results
                .get_or_insert(sensor_name)
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

        string_output
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
        let results_dict_bound: &Bound<'_, PyDict> = results_dict.cast::<PyDict>().expect("Failed to downcast `results` to PyDict");

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
    /// TODO: change `SensorsStatic` to Vec<Arc<SensorsStatic>> to be consistent with other sensor types.
    ///
    /// The Greens with the PF coils are gathered in the order of `pf_active_names`, and those with the passive conductors' degrees of
    /// freedom in the order of `pf_passive_dof_names`, each a (passive conductor name, degree of freedom name) pair
    pub fn split_into_static_and_dynamic(
        &mut self,
        times_to_reconstruct: &Array1<f64>,
        pf_active_names: &[String],
        pf_passive_dof_names: &[(String, String)],
    ) -> (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) {
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
            let static_data_empty_vs_time: Vec<Arc<SensorsStatic>> = vec![Arc::new(static_data_empty); n_time];
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
        // In the order of `pf_active_names`, the order of the PF coil currents, for the sensors needed. Found by name,
        // because the sensors' own store sorts them by name
        let n_pf: usize = pf_active_names.len();
        let mut greens_with_pf: Array2<f64> = Array2::from_elem((n_pf, n_sensors), f64::NAN); // shape = [n_pf, n_sensors]
        for i_pf in 0..n_pf {
            for i_sensor in 0..n_sensors {
                greens_with_pf[[i_pf, i_sensor]] = self
                    .results
                    .get(&sensor_names[i_sensor])
                    .get("greens")
                    .get("pf")
                    .get(&pf_active_names[i_pf])
                    .unwrap_f64();
            }
        }
        // With plasma
        let greens_with_grid: Array2<f64> = self.results.get("*").get("greens").get("plasma").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
        let greens_with_grid: Array2<f64> = greens_with_grid.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_pf, n_sensors]
        // With d_sensor_dz (for vertical stability)
        let greens_d_sensor_dz: Array2<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
        let greens_d_sensor_dz: Array2<f64> = greens_d_sensor_dz.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]

        // With passives
        // In the order of `pf_passive_dof_names`, the order of the passive degrees of freedom, for the sensors needed. Found by
        // name, because the sensors' own store sorts them by name
        let n_dof_total: usize = pf_passive_dof_names.len();
        let mut greens_with_passives: Array2<f64> = Array2::from_elem((n_dof_total, n_sensors), f64::NAN); // shape = [n_dof_total, n_sensors]
        for i_dof_total in 0..n_dof_total {
            let (passive_name, dof_name): &(String, String) = &pf_passive_dof_names[i_dof_total];
            for i_sensor in 0..n_sensors {
                greens_with_passives[(i_dof_total, i_sensor)] = self
                    .results
                    .get(&sensor_names[i_sensor])
                    .get("greens")
                    .get("passives")
                    .get(passive_name)
                    .get(dof_name)
                    .unwrap_f64();
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
            geometry_r: Array1::from_elem(n_sensors, f64::NAN), // not used for BpProbes
            geometry_z: Array1::from_elem(n_sensors, f64::NAN), // not used for BpProbes
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
            let interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(experimental_time.clone(), experimental_values.clone()).expect("Can't make interpolator");

            // Do the interpolation
            let measured_this_sensor: Array1<f64> = interpolator.interpolate_array1(times_to_reconstruct).expect("Can't do interpolation");

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

        // These Green's tables are fixed geometry, so every time-slice gets an `Arc` handle
        // to the same copy rather than 480 identical copies of it
        let results_static_time_dependent: Vec<Arc<SensorsStatic>> = vec![Arc::new(results_static); n_time];

        // Return the static and dynamic results
        (results_static_time_dependent, results_dynamic)
    }

    /// Calculate sensor values
    pub fn calculate_sensor_values_rs(&mut self, plasma: &Plasma) {
        // The passive degrees of freedom, in the order of `greens/pf_passive`, which every time-slice's `passive_dof_values` follows
        let mut pf_passive_dof_names: Vec<(String, String)> = Vec::new();
        let n_pf_passive: usize = plasma.equilibrium_ids.greens.pf_passive.len();
        for i_pf_passive in 0..n_pf_passive {
            let n_dof: usize = plasma.equilibrium_ids.greens.pf_passive[i_pf_passive].dof.len();
            for i_dof in 0..n_dof {
                pf_passive_dof_names.push((
                    plasma.equilibrium_ids.greens.pf_passive[i_pf_passive].name.clone(),
                    plasma.equilibrium_ids.greens.pf_passive[i_pf_passive].dof[i_dof].name.clone(),
                ));
            }
        }
        let n_dof_total: usize = pf_passive_dof_names.len();

        for sensor_name in self.results.keys() {
            // The PF coil currents, which `Plasma.copy_from_pf_active_ids` copied onto every time-slice, and this sensor's
            // Greens with each of those coils, found by name
            let pf_current_names: Vec<String> = plasma.equilibrium_ids.time_slice[0].constraints.pf_current(..).name.to_vec();
            let n_pf: usize = pf_current_names.len();
            let mut g_with_coils: Array1<f64> = Array1::from_elem(n_pf, f64::NAN); // shape = [n_pf]
            for i_pf in 0..n_pf {
                g_with_coils[i_pf] = self.results.get(&sensor_name).get("greens").get("pf").get(&pf_current_names[i_pf]).unwrap_f64();
            }
            let coil_currents: Array2<f64> = plasma.equilibrium_ids.time_slice(..).constraints.pf_current(..).measured.to_array(); // shape = [n_time, n_pf]
            // let n_time: usize = coil_currents.len_of(Axis(0));

            // This sensor's Greens with each passive degree of freedom, found by name
            let mut g_with_passives: Array1<f64> = Array1::from_elem(n_dof_total, f64::NAN); // shape = [n_dof_total]
            for i_dof_total in 0..n_dof_total {
                let (passive_name, dof_name): &(String, String) = &pf_passive_dof_names[i_dof_total];
                g_with_passives[i_dof_total] = self
                    .results
                    .get(&sensor_name)
                    .get("greens")
                    .get("passives")
                    .get(passive_name)
                    .get(dof_name)
                    .unwrap_f64();
            }

            // Plasma
            let g_with_plasma: Array1<f64> = self.results.get(&sensor_name).get("greens").get("plasma").unwrap_array1(); // shape = [n_z*n_r]
            // `time_slice[0]` because the grid is the same on every time-slice, and `profiles_2d[0]`
            // because GSFit solves on a single rectangular (R, Z) grid
            let d_area: f64 = plasma.equilibrium_ids.time_slice[0].profiles_2d[0].grid.d_area;
            let time: Array1<f64> = plasma.equilibrium_ids.time_slice(..).time.to_array();
            let n_time: usize = time.len();

            // Loop over time
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                // PF coil
                let sensor_values_from_coils: f64 = g_with_coils.dot(&coil_currents.slice(s![i_time, ..]));

                // Passives, with the degrees of freedom the reconstruction fitted, which are NaN on a time-slice which failed
                let passive_dof_values: &Array1<f64> = &plasma.equilibrium_ids.time_slice[i_time].passive_dof_values;
                let mut sensor_values_from_passives: f64 = 0.0;
                for i_dof_total in 0..n_dof_total {
                    sensor_values_from_passives += g_with_passives[i_dof_total] * passive_dof_values[i_dof_total];
                }

                // Plasma
                // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid
                let j_2d: &Array2<f64> = &plasma.equilibrium_ids.time_slice[i_time].profiles_2d[0].j_phi;
                let j_2d_flat: Array1<f64> = Array1::from_iter(j_2d.iter().copied());
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

    #[allow(clippy::too_many_arguments)]
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

    pub fn greens_with_pf_active_rs(&mut self, pf_active: &PfActive) {
        // The PF coils, in the order of `pf_active/coil`
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();

        for sensor_name in self.results.keys() {
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            for i_coil in 0..n_coil {
                let pf_coil_name: &str = &pf_active_coils[i_coil].name;
                let coil_r: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array();
                let coil_z: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array();

                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![sensor_r],
                    array![sensor_z],
                    coil_r.clone(),
                    coil_z.clone(),
                    coil_r.clone() * 0.0, // TODO: should this be NaN instead?
                    coil_z.clone() * 0.0,
                );

                let g_b_r_matrix: Array2<f64> = greens_calculator.b_r(); // shape() = (1, n_coils)
                let g_b_z_matrix: Array2<f64> = greens_calculator.b_z(); // shape() = (1, n_coils)

                // Sum over all the current sources
                let g_br: f64 = g_b_r_matrix.sum();
                let g_bz: f64 = g_b_z_matrix.sum();

                // Sensors Green's function
                let g: f64 = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(pf_coil_name, g);
            }
        }
    }

    pub fn greens_with_plasma_rs(&mut self, plasma: &Plasma) {
        // `time_slice(0)` because the grid is the same on every time-slice, and `profiles_2d(0)`
        // because GSFit solves on a single rectangular (R, Z) grid. `profiles_2d/r` and `/z` are the
        // (R, Z) mesh, so iterating them row-major gives the flattened grid the Green's tables are
        // indexed by
        let mesh_r: &Array2<f64> = &plasma.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &plasma.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let plasma_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let plasma_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        for sensor_name in self.results.keys() {
            // Get variables out of self
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

            let greens_calculator: Greens = Greens::sensor_to_conductor(
                array![sensor_r],
                array![sensor_z],
                plasma_r.clone(),
                plasma_z.clone(),
                plasma_r.clone() * 0.0, // TODO: this should be d_r not 0.0
                plasma_z.clone() * 0.0,
            );

            let g_b_r_matrix: Array2<f64> = greens_calculator.b_r(); // shape() = (1, n_z*n_r)
            let g_b_z_matrix: Array2<f64> = greens_calculator.b_z(); // shape() = (1, n_z*n_r)

            let g_br: Array1<f64> = g_b_r_matrix.sum_axis(Axis(0)); // g_b_r_matrix.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]
            let g_bz: Array1<f64> = g_b_z_matrix.sum_axis(Axis(0));

            // Sensors Green's function
            let g_with_plasma: Array1<f64> = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

            // Store
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma); // shape = [(n_z * n_r)]

            // Same `greens_calculator` as above: the sensor and the grid have not changed
            let g_d_b_r_d_z_matrix: Array2<f64> = greens_calculator.d_b_r_d_z(); // shape() = (1, n_z*n_r)
            let g_d_b_z_d_z_matrix: Array2<f64> = greens_calculator.d_b_z_d_z(); // shape() = (1, n_z*n_r)

            // Vertical stability
            let g_d_plasma_br_d_z: Array1<f64> = g_d_b_r_d_z_matrix.sum_axis(Axis(0)); // g_d_b_r_d_z_matrix.shape = [1, n_z * n_r];  g_d_plasma_br_d_z.shape = [n_z * n_r]
            let g_d_plasma_bz_d_z: Array1<f64> = g_d_b_z_d_z_matrix.sum_axis(Axis(0));

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

    pub fn greens_with_pf_passive_rs(&mut self, pf_passive: &PfPassive) {
        // The passive conductors, in the order of `pf_passive/loop`
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();

        // Loop over sensors
        for sensor_name in self.results.keys() {
            let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
            let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();
            let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();

            // Calculate Greens with each passive degree of freedom
            for i_loop in 0..n_loop {
                let passive_name: &str = &pf_passive_loops[i_loop].name;
                let filaments: PassiveFilaments = passive_filaments(&pf_passive_loops[i_loop].element);
                let passive_r: Array1<f64> = filaments.r;
                let passive_z: Array1<f64> = filaments.z;

                // These Green's tables depend on the sensor and this passive's filament geometry,
                // not on the degree of freedom, so they are built once and re-used for every one
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![sensor_r],
                    array![sensor_z],
                    passive_r.clone(),
                    passive_z.clone(),
                    passive_r.clone() * 0.0, // TODO: should this be NaN instead?
                    passive_z.clone() * 0.0,
                );

                let g_br_matrix: Array2<f64> = greens_calculator.b_r(); // shape() = (1, n_z*n_r)
                let g_bz_matrix: Array2<f64> = greens_calculator.b_z(); // shape() = (1, n_z*n_r)

                let n_dof: usize = pf_passive_loops[i_loop].dof.len();
                for i_dof in 0..n_dof {
                    let dof_name: &str = &pf_passive_loops[i_loop].dof[i_dof].name;

                    // Current distribution
                    let current_distribution: &Array1<f64> = &pf_passive_loops[i_loop].dof[i_dof].current_distribution;

                    let g_br_with_dof_full: Array2<f64> = &g_br_matrix * current_distribution; // shape = [n_passive_dof, n_filament]
                    let g_bz_with_dof_full: Array2<f64> = &g_bz_matrix * current_distribution; // shape = [n_passive_dof, n_filament]

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
                        .get_or_insert(passive_name)
                        .insert(dof_name, g);
                }
            }
        }
    }
}
