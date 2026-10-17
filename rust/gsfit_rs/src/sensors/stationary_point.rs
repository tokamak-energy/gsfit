use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use crate::python_pickling_methods::{data_tree_to_py_dict, py_dict_to_data_tree};
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
pub struct StationaryPoint {
    pub results: DataTree,
}

impl Default for StationaryPoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Python accessible methods
#[pymethods]
impl StationaryPoint {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_sensor(
        &mut self,
        name: &str,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: PyReadonlyArray1<f64>,
        mag_axis_r: PyReadonlyArray1<f64>,
        mag_axis_z: PyReadonlyArray1<f64>,
        times_to_reconstruct: PyReadonlyArray1<f64>,
    ) {
        // Convert into Rust data types
        let time_ndarray: Array1<f64> = time.to_owned_array();
        let mag_axis_r_ndarray: Array1<f64> = mag_axis_r.to_owned_array();
        let mag_axis_z_ndarray: Array1<f64> = mag_axis_z.to_owned_array();
        let times_to_reconstruct_ndarray: Array1<f64> = times_to_reconstruct.to_owned_array();
        let n_time: usize = times_to_reconstruct_ndarray.len();

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
            .get_or_insert("geometry")
            .insert("time_experimental", time_ndarray.clone());
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .get_or_insert("r")
            .insert("measured_experimental", mag_axis_r_ndarray.clone());
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .get_or_insert("z")
            .insert("measured_experimental", mag_axis_z_ndarray.clone());

        // Interpolate the geometry to `times_to_reconstruct`
        // geometry_r
        let interpolator = interpolation::Dim1Linear::new(time_ndarray.clone(), mag_axis_r_ndarray.clone())
            .expect("StationaryPoint.add_sensor: Can't make interpolator for mag_axis_r");
        let geometry_r_measured: Array1<f64> = interpolator
            .interpolate_array1(&times_to_reconstruct_ndarray)
            .expect("StationaryPoint.add_sensor: Can't do interpolation for geometry_r");
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .get_or_insert("r")
            .insert("measured", geometry_r_measured);
        // geometry_z
        let interpolator = interpolation::Dim1Linear::new(time_ndarray.clone(), mag_axis_z_ndarray.clone())
            .expect("StationaryPoint.add_sensor: Can't make interpolator for mag_axis_z");
        let geometry_z_measured: Array1<f64> = interpolator
            .interpolate_array1(&times_to_reconstruct_ndarray)
            .expect("StationaryPoint.add_sensor: Can't do interpolation for geometry_z");
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .get_or_insert("z")
            .insert("measured", geometry_z_measured);

        // Store time
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("time", times_to_reconstruct_ndarray);

        // Add a time-dependent "include"
        let include_dynamic: Vec<bool> = vec![fit_settings_include; n_time]; // we are including all sensors
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include_dynamic", include_dynamic);
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
    pub fn greens_with_plasma(&mut self, equilibrium: PyRef<Equilibrium>) {
        // Change Python type into Rust
        let equilibrium_local: &Equilibrium = &equilibrium;

        // Run the Rust method
        self.greens_with_plasma_rs(equilibrium_local);
    }

    // /// Calculate sensor values
    // pub fn calculate_sensor_values(&mut self, equilibrium: PyRef<Equilibrium>) {
    //     // Convert Python types into Rust
    //     let equilibrium_rs: &Equilibrium = &equilibrium;

    //     // Run the Rust method
    //     self.calculate_sensor_values_rust(equilibrium_rs);
    // }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.StationaryPoint>");
        string_output += &format!("║  {:<74} ║\n", version);

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
            .expect("Pickled StationaryPoint object has no version information");
        if pickled_version != current_version {
            eprintln!(
                "Warning: Unpickling StationaryPoint object created with gsfit_rs v{}, but current version is v{}",
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
impl StationaryPoint {
    /// This splits the StationaryPoint into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent objects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    /// For Isoflux sensors the static data is actually time-dependent.
    /// TODO: consider renaming `SensorsStatic`. Perhaps `SensorsGeometry` ?
    ///
    /// The Greens with the PF coils are gathered in the order of `pf_active_names`, and those with the passive conductors' degrees of
    /// freedom in the order of `pf_passive_dof_names`, each a (passive conductor name, degree of freedom name) pair
    pub fn split_into_static_and_dynamic(
        &mut self,
        times_to_reconstruct: &Array1<f64>,
        pf_active_names: &[String],
        pf_passive_dof_names: &[(String, String)],
    ) -> (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) {
        // Define empty data arrays
        let results_static_empty: Arc<SensorsStatic> = Arc::new(SensorsStatic {
            names: Vec::new(),
            greens_with_grid: Array2::zeros((0, 0)),
            greens_with_pf: Array2::zeros((0, 0)),
            greens_with_passives: Array2::zeros((0, 0)),
            greens_d_sensor_dz: Array2::zeros((0, 0)),
            fit_settings_weight: Array1::zeros(0),
            fit_settings_expected_value: Array1::zeros(0),
            geometry_r: Array1::zeros(0), // not used for StationaryPoint
            geometry_z: Array1::zeros(0), // not used for StationaryPoint
        });
        let results_dynamic_empty: SensorsDynamic = SensorsDynamic { measured: Array1::zeros(0) };

        // Number of time-slices to reconstruct
        let n_time: usize = times_to_reconstruct.len();

        // Create the time-dependent data structures
        let mut results_static: Vec<Arc<SensorsStatic>> = Vec::with_capacity(n_time);
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();

        'time_loop: for i_time in 0..n_time {
            let mut include: Vec<bool> = Vec::new();
            for sensor_name in sensor_names_all.clone() {
                let include_dynamic: Vec<bool> = self.results.get(&sensor_name).get("fit_settings").get("include_dynamic").unwrap_vec_bool();

                include.push(include_dynamic[i_time]);
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
                results_static.push(Arc::clone(&results_static_empty));
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
            // With PF coils, at this time
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
                        .unwrap_array1()[i_time];
                }
            }

            // With plasma
            let greens_with_grid: Array3<f64> = self.results.get("*").get("greens").get("plasma").unwrap_array3(); // shape = [n_time, n_z * n_r, n_sensors]
            let greens_with_grid: Array3<f64> = greens_with_grid.select(Axis(2), &include_indices); // downselect to only the sensors needed; shape = [n_time, n_z * n_r, n_sensors]

            // With d_sensor_dz (for vertical stability)
            let greens_d_sensor_dz: Array3<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array3(); // shape = [n_time, n_z * n_r, n_sensors]
            let greens_d_sensor_dz: Array3<f64> = greens_d_sensor_dz.select(Axis(2), &include_indices); // downselect to only the sensors needed; shape = [n_time, n_z * n_r, n_sensors]

            // With passives
            // In the order of `pf_passive_dof_names`, the order of the passive degrees of freedom, for the sensors needed. Found by
            // name, because the sensors' own store sorts them by name
            let n_dof_total: usize = pf_passive_dof_names.len();
            let mut greens_with_passives: Array3<f64> = Array3::from_elem((n_time, n_dof_total, n_sensors), f64::NAN);
            for i_dof_total in 0..n_dof_total {
                let (passive_name, dof_name): &(String, String) = &pf_passive_dof_names[i_dof_total];
                for i_sensor in 0..n_sensors {
                    let greens_with_passives_tmp: Array1<f64> = self
                        .results
                        .get(&sensor_names[i_sensor])
                        .get("greens")
                        .get("passives")
                        .get(passive_name)
                        .get(dof_name)
                        .unwrap_array1(); // shape = [n_time]
                    greens_with_passives.slice_mut(s![.., i_dof_total, i_sensor]).assign(&greens_with_passives_tmp);
                }
            }

            // Create the `SensorsStatic` data
            let results_static_this_time_slice: SensorsStatic = SensorsStatic {
                names: sensor_names.clone(),
                greens_with_grid: greens_with_grid.slice(s![i_time, .., ..]).to_owned(), // shape = [n_z * n_r, n_sensors]
                greens_with_pf,
                greens_with_passives: greens_with_passives.slice(s![i_time, .., ..]).to_owned(),
                greens_d_sensor_dz: greens_d_sensor_dz.slice(s![i_time, .., ..]).to_owned(),
                fit_settings_weight: fit_settings_weight.clone(),
                fit_settings_expected_value: fit_settings_expected_value.clone(),
                geometry_r: Array1::zeros(n_sensors), // not used for StationaryPoint
                geometry_z: Array1::zeros(n_sensors), // not used for StationaryPoint
            };
            results_static.push(Arc::new(results_static_this_time_slice));

            // The measured sensor values are = 0.0
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: Array1::zeros(n_sensors),
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        // Return the static and dynamic results
        (results_static, results_dynamic)
    }

    pub fn greens_with_pf_active_rs(&mut self, pf_active: &PfActive) {
        // The PF coils, in the order of `pf_active/coil`
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();

        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            for i_coil in 0..n_coil {
                let pf_coil_name: &str = &pf_active_coils[i_coil].name;
                let mut g_vs_time: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                for i_time in 0..n_time {
                    let coil_r: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array();
                    let coil_z: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array();
                    let n_filaments: usize = coil_r.len();

                    let mag_axis_r_local: Array1<f64> = array![mag_axis_r[i_time]];
                    let mag_axis_z_local: Array1<f64> = array![mag_axis_z[i_time]];

                    let greens_calculator: Greens = Greens::sensor_to_conductor(
                        mag_axis_r_local.clone(),
                        mag_axis_z_local.clone(),
                        coil_r.clone(),
                        coil_z.clone(),
                        Array1::zeros(n_filaments), // TODO: should this be NaN?
                        Array1::zeros(n_filaments),
                    );

                    let g_b_r_matrix: Array2<f64> = greens_calculator.b_r(); // shape = [n_sensor, n_coil_filament]
                    // let g_b_z_matrix: Array2<f64> = greens_calculator.b_z();  // shape = [n_sensor, n_coil_filament]

                    // Sum over all the current sources
                    let g_br: f64 = g_b_r_matrix.sum();

                    // Sensors Green's function
                    g_vs_time[i_time] = g_br;
                }
                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(pf_coil_name, g_vs_time); // shape = [n_time]
            }
        }
    }

    pub fn greens_with_plasma_rs(&mut self, equilibrium: &Equilibrium) {
        // `time_slice(0)` because the grid is the same on every time-slice, and `profiles_2d(0)`
        // because GSFit solves on a single rectangular (R, Z) grid. `profiles_2d/r` and `/z` are the
        // (R, Z) mesh, so iterating them row-major gives the flattened grid the Green's tables are
        // indexed by
        let mesh_r: &Array2<f64> = &equilibrium.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &equilibrium.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let plasma_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let plasma_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());
        let n_r: usize = equilibrium.equilibrium_ids.code.parameters_parsed.grid.n_r as usize;
        let n_z: usize = equilibrium.equilibrium_ids.code.parameters_parsed.grid.n_z as usize;

        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            let mut g_with_plasma: Array2<f64> = Array2::from_elem((n_time, n_z * n_r), f64::NAN);
            let mut g_d_plasma_d_z: Array2<f64> = Array2::from_elem((n_time, n_z * n_r), f64::NAN);
            for i_time in 0..n_time {
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    array![mag_axis_r[i_time]], // sensor
                    array![mag_axis_z[i_time]],
                    plasma_r.clone(), // current source
                    plasma_z.clone(),
                    Array1::zeros(n_z * n_r), // TODO: should this be NaN?
                    Array1::zeros(n_z * n_r),
                );

                let g_br_full: Array2<f64> = greens_calculator.b_r(); // shape = [n_sensor, n_z * n_r]
                // let g_bz_full: Array2<f64> = greens_calculator.b_z();  // shape = [n_sensor, n_z * n_r]
                let g_br: Array1<f64> = g_br_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]

                // Sensors Green's function
                g_with_plasma.slice_mut(s![i_time, ..]).assign(&g_br);

                // Vertical stability
                let g_d_plasma_br_d_z_full: Array2<f64> = greens_calculator.d_b_r_d_z(); // shape = [n_sensor, n_z * n_r]

                let g_d_plasma_br_d_z: Array1<f64> = g_d_plasma_br_d_z_full.sum_axis(Axis(0)); // g_d_plasma_br_d_z_full.shape = [1, n_z * n_r];  g_d_plasma_br_d_z.shape = [n_z * n_r]

                // Sensors Green's function
                g_d_plasma_d_z.slice_mut(s![i_time, ..]).assign(&g_d_plasma_br_d_z);
            }
            // Store
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma); // shape = [(n_z * n_r)]
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z); // shape = [n_time, n_z * n_r]
        }
    }

    pub fn greens_with_pf_passive_rs(&mut self, pf_passive: &PfPassive) {
        // The passive conductors, in the order of `pf_passive/loop`
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();

        // Loop over sensors
        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            // Calculate Greens with each passive degree of freedom
            for i_loop in 0..n_loop {
                let passive_name: &str = &pf_passive_loops[i_loop].name;
                let filaments: PassiveFilaments = passive_filaments(&pf_passive_loops[i_loop].element);
                let passive_r: Array1<f64> = filaments.r;
                let passive_z: Array1<f64> = filaments.z;

                let n_dof: usize = pf_passive_loops[i_loop].dof.len();
                for i_dof in 0..n_dof {
                    let dof_name: &str = &pf_passive_loops[i_loop].dof[i_dof].name;
                    let mut g_vs_time: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                    for i_time in 0..n_time {
                        let greens_calculator: Greens = Greens::sensor_to_conductor(
                            array![mag_axis_r[i_time]], // sensor
                            array![mag_axis_z[i_time]],
                            passive_r.clone(), // current source
                            passive_z.clone(),
                            Array1::zeros(passive_r.len()), // TODO: should this be NaN?
                            Array1::zeros(passive_r.len()),
                        );

                        let g_br_full: Array2<f64> = greens_calculator.b_r(); // shape = [n_sensor, n_passive_filament]

                        // Current distribution
                        let current_distribution: &Array1<f64> = &pf_passive_loops[i_loop].dof[i_dof].current_distribution;

                        let g_br_with_dof_full: Array2<f64> = g_br_full * current_distribution; // shape = [n_passive_dof, n_filament]

                        // Sum over all filaments
                        let g_br: f64 = g_br_with_dof_full.sum(); // shape = [n_passive_dof]

                        // Calculate Green's function
                        g_vs_time[i_time] = g_br;
                    }
                    // Store
                    self.results
                        .get_or_insert(&sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(passive_name)
                        .insert(dof_name, g_vs_time); // shape = [n_time]
                }
            }
        }
    }
}
