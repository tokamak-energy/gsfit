use crate::Plasma;
use crate::coils::Coils;
use crate::greens::greens_b;
use crate::greens::greens_d_b_d_z;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_interp::interp1d::Interp1D;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass]
pub struct MagneticAxis {
    pub results: DataTree,
}

/// Python accessible methods
#[pymethods]
impl MagneticAxis {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    pub fn add_sensor(
        &mut self,
        name: &str,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: &Bound<'_, PyArray1<f64>>,
        mag_axis_r: &Bound<'_, PyArray1<f64>>,
        mag_axis_z: &Bound<'_, PyArray1<f64>>,
        times_to_reconstruct: &Bound<'_, PyArray1<f64>>,
    ) {
        // Convert into Rust data types
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let mag_axis_r_ndarray: Array1<f64> = Array1::from(unsafe { mag_axis_r.as_array() }.to_vec());
        let mag_axis_z_ndarray: Array1<f64> = Array1::from(unsafe { mag_axis_z.as_array() }.to_vec());
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
        let interpolator = Interp1D::builder(mag_axis_r_ndarray)
            .x(time_ndarray.clone())
            .build()
            .expect("MagneticAxis.add_sensor: Can't make Interp1D for mag_axis_r");
        let geometry_r_measured: Array1<f64> = interpolator
            .interp_array(&times_to_reconstruct_ndarray)
            .expect("MagneticAxis.add_sensor: Can't do interpolation for geometry_r");
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .get_or_insert("r")
            .insert("measured", geometry_r_measured);
        // geometry_z
        let interpolator = Interp1D::builder(mag_axis_z_ndarray)
            .x(time_ndarray.clone())
            .build()
            .expect("MagneticAxis.add_sensor: Can't make Interp1D for mag_axis_z");
        let geometry_z_measured: Array1<f64> = interpolator
            .interp_array(&times_to_reconstruct_ndarray)
            .expect("MagneticAxis.add_sensor: Can't do interpolation for mag_axis_z");
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

    // /// Calculate sensor values
    // pub fn calculate_sensor_values(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>, plasma: PyRef<Plasma>) {
    //     // Convert Python types into Rust
    //     let coils_rs: &Coils = &*coils;
    //     let passives_rs: &Passives = &*passives;
    //     let plasma_rs: &Plasma = &*plasma;

    //     // Run the Rust method
    //     self.calculate_sensor_values_rust(coils_rs, passives_rs, plasma_rs);
    // }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.MagneticAxis>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}

// Rust only methods
impl MagneticAxis {
    /// This splits the MagneticAxis into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent ojbects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    /// For Isoflux sensors the static data is actually time-dependent.
    /// TODO: consider renaming `SensorsStatic`. Perhaps `SensorsGeometry` ?
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

    pub fn greens_with_coils_rs(&mut self, coils: Coils) {
        // There should be only be one `sensor_name` in MagneticAxis
        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            for pf_coil_name in coils.results.get("pf").keys() {
                let mut g_vs_time: Array1<f64> = Array1::zeros(n_time);
                for i_time in 0..n_time {
                    let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                    let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

                    let (g_br_full, _g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                        Array1::from_vec(vec![mag_axis_r[i_time]]),
                        Array1::from_vec(vec![mag_axis_z[i_time]]),
                        coil_r.clone(),
                        coil_z.clone(),
                    );

                    // Sum over all the current sources
                    let g_br: f64 = g_br_full.sum();

                    // Sensors Green's function
                    g_vs_time[i_time] = g_br;
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

    pub fn greens_with_plasma_rs(&mut self, plasma: Plasma) {
        let plasma_r: Array1<f64> = plasma.results.get("grid").get("flat").get("r").unwrap_array1();
        let plasma_z: Array1<f64> = plasma.results.get("grid").get("flat").get("z").unwrap_array1();
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();

        // There should be only be one `sensor_name` in MagneticAxis
        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            self.results.print_keys();

            let mut g_with_plasma: Array2<f64> = Array2::zeros([n_time, n_z * n_r]);
            let mut g_d_plasma_d_z: Array2<f64> = Array2::zeros([n_time, n_z * n_r]);
            for i_time in 0..n_time {
                let (g_br_full, _g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                    Array1::from_vec(vec![mag_axis_r[i_time]]), // sensor
                    Array1::from_vec(vec![mag_axis_z[i_time]]),
                    plasma_r.clone(), // current source
                    plasma_z.clone(),
                );

                let g_br: Array1<f64> = g_br_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]

                // Sensors Green's function
                g_with_plasma.slice_mut(s![i_time, ..]).assign(&g_br);

                // Vertical stability
                let (g_d_plasma_br_d_z_full, _g_d_plasma_bz_d_z_full): (Array2<f64>, Array2<f64>) = greens_d_b_d_z(
                    Array1::from_vec(vec![mag_axis_r[i_time]]), // sensor
                    Array1::from_vec(vec![mag_axis_z[i_time]]),
                    plasma_r.clone(), // current source
                    plasma_z.clone(),
                );

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

    pub fn greens_with_passives_rs(&mut self, passives: Passives) {
        // Loop over sensors
        for sensor_name in self.results.keys() {
            // Get time
            let times_to_reconstruct: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("time").unwrap_array1();
            let n_time: usize = times_to_reconstruct.len();

            // Time-dependent magnetic axis
            let mag_axis_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").get("measured").unwrap_array1();
            let mag_axis_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").get("measured").unwrap_array1();

            // Calculate Greens with each passive degree of freedom
            for passive_name in passives.results.keys() {
                let _tmp: DataTreeAccumulator<'_> = passives.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();
                let passive_r: Array1<f64> = passives.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
                let passive_z: Array1<f64> = passives.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

                for dof_name in dof_names {
                    let mut g_vs_time: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                    for i_time in 0..n_time {
                        let (g_br_full, _g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
                            Array1::from_vec(vec![mag_axis_r[i_time]]), // by convention (r, z) are "sensors"
                            Array1::from_vec(vec![mag_axis_z[i_time]]),
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
                        .get_or_insert(&passive_name)
                        .insert(&dof_name, g_vs_time); // shape = [n_time]
                }
            }
        }
    }
}
