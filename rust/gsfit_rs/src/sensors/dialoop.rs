use crate::Plasma;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, Axis, s};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

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

/// Python accessible methods
#[pymethods]
impl Dialoop {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    /// Data structure:
    ///
    /// # Examples
    ///
    /// ```ignore
    /// [probe_name]["b"]["calculated"]["value"]                        = Array1<f64>;  shape=[n_time]
    /// [probe_name]["b"]["calculated"]["time"]                         = Array1<f64>;  shape=[n_time]
    /// [probe_name]["b"]["experimental"]["value"]                      = Array1<f64>;  shape=[n_time_experimental]
    /// [probe_name]["b"]["experimental"]["time"]                       = Array1<f64>;  shape=[n_time_experimental]
    /// [probe_name]["b"]["measured"]["value"]                          = Array1<f64>;  shape=[n_time]
    /// [probe_name]["b"]["measured"]["time"]                           = Array1<f64>;  shape=[n_time]
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

    // ///
    // pub fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
    //     // Change Python type into Rust
    //     let coils_local: &Coils = &coils;

    //     for sensor_name in self.results.keys() {
    //         let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
    //         let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
    //         let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

    //         for pf_coil_name in coils_local.results.get("pf").keys() {
    //             let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
    //             let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

    //             let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
    //                 Array1::from_vec(vec![sensor_r]),
    //                 Array1::from_vec(vec![sensor_z]),
    //                 coil_r.clone(),
    //                 coil_z.clone(),
    //             );

    //             // Sum over all the current sources
    //             let g_br: f64 = g_br_full.sum();
    //             let g_bz: f64 = g_bz_full.sum();

    //             // Sensors Green's function
    //             let g = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

    //             // Store
    //             self.results
    //                 .get_or_insert(&sensor_name)
    //                 .get_or_insert("greens")
    //                 .get_or_insert("pf")
    //                 .insert(&pf_coil_name, g);
    //         }
    //     }
    // }

    // ///
    // pub fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
    //     // Change Python type into Rust
    //     let passives_local: &Passives = &passives;

    //     for sensor_name in self.results.keys() {
    //         let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
    //         let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();
    //         let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();

    //         // Calculate Greens with each passive degree of freedom
    //         for passive_name in passives_local.results.keys() {
    //             let _tmp: DataTreeAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
    //             let dof_names: Vec<String> = _tmp.keys();
    //             let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
    //             let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

    //             for dof_name in dof_names {
    //                 let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
    //                     Array1::from_vec(vec![sensor_r]), // by convention (r, z) are "sensors"
    //                     Array1::from_vec(vec![sensor_z]),
    //                     passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
    //                     passive_z.clone(),
    //                 );

    //                 // Current distribution
    //                 let current_distribution: Array1<f64> = passives
    //                     .results
    //                     .get(&passive_name)
    //                     .get("dof")
    //                     .get(&dof_name)
    //                     .get("current_distribution")
    //                     .unwrap_array1();

    //                 let g_br_with_dof_full: Array2<f64> = g_br_full * &current_distribution; // shape = [n_r * n_z, n_filament]
    //                 let g_bz_with_dof_full: Array2<f64> = g_bz_full * current_distribution; // shape = [n_r * n_z, n_filament]

    //                 // Sum over all filaments
    //                 let g_br: f64 = g_br_with_dof_full.sum(); // shape = [n_r * n_z]
    //                 let g_bz: f64 = g_bz_with_dof_full.sum(); // shape = [n_r * n_z]

    //                 // Calculate Green's function
    //                 let g: f64 = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

    //                 // Store
    //                 self.results
    //                     .get_or_insert(&sensor_name)
    //                     .get_or_insert("greens")
    //                     .get_or_insert("passives")
    //                     .get_or_insert(&passive_name)
    //                     .insert(&dof_name, g);
    //             }
    //         }
    //     }
    // }

    // ///
    // pub fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
    //     // Change Python type into Rust
    //     let plasma_local: &Plasma = &plasma;

    //     let plasma_r: Array1<f64> = plasma_local.results.get("grid").get("flat").get("r").unwrap_array1();
    //     let plasma_z: Array1<f64> = plasma_local.results.get("grid").get("flat").get("z").unwrap_array1();

    //     for sensor_name in self.results.keys() {
    //         // Get variables out of self
    //         let sensor_angle_pol: f64 = self.results.get(&sensor_name).get("geometry").get("angle_pol").unwrap_f64();
    //         let sensor_r: f64 = self.results.get(&sensor_name).get("geometry").get("r").unwrap_f64();
    //         let sensor_z: f64 = self.results.get(&sensor_name).get("geometry").get("z").unwrap_f64();

    //         let (g_br_full, g_bz_full): (Array2<f64>, Array2<f64>) = greens_b(
    //             Array1::from_vec(vec![sensor_r]), // sensor
    //             Array1::from_vec(vec![sensor_z]),
    //             plasma_r.clone(), // current source
    //             plasma_z.clone(),
    //         );

    //         let g_br: Array1<f64> = g_br_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]
    //         let g_bz: Array1<f64> = g_bz_full.sum_axis(Axis(0));

    //         // Sensors Green's function
    //         let g_with_plasma: Array1<f64> = g_br * sensor_angle_pol.cos() + g_bz * sensor_angle_pol.sin();

    //         // Store
    //         self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma);

    //         // Vertical stability
    //         let (g_d_plasma_br_d_z_full, g_d_plasma_bz_d_z_full): (Array2<f64>, Array2<f64>) = greens_d_b_dz(
    //             Array1::from_vec(vec![sensor_r]), // sensor
    //             Array1::from_vec(vec![sensor_z]),
    //             plasma_r.clone(), // current source
    //             plasma_z.clone(),
    //         );

    //         let g_d_plasma_br_d_z: Array1<f64> = g_d_plasma_br_d_z_full.sum_axis(Axis(0)); // g_br_full.shape = [1, n_z * n_r];  g_br.shape = [n_z * n_r]
    //         let g_d_plasma_bz_d_z: Array1<f64> = g_d_plasma_bz_d_z_full.sum_axis(Axis(0));

    //         // Sensors Green's function
    //         let g_d_plasma_d_z: Array1<f64> = g_d_plasma_br_d_z * sensor_angle_pol.cos() + g_d_plasma_bz_d_z * sensor_angle_pol.sin();

    //         // Store
    //         // TODO: should I reshape to Array2 [n_z, n_r] ????
    //         self.results
    //             .get_or_insert(&sensor_name)
    //             .get_or_insert("greens")
    //             .insert("d_plasma_d_z", g_d_plasma_d_z);
    //     }
    // }

    /// Calculate the sensor values
    pub fn calculate_sensor_values(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let coils_rs: &Coils = &coils;
        let passives_rs: &Passives = &passives;
        let plasma_rs: &Plasma = &plasma;

        // Run the Rust method
        self.calculate_sensor_values_rs(coils_rs, passives_rs, plasma_rs);
    }

    /// Calculate the vacuum sensor values.
    ///
    /// The diamagnetic flux is a plasma-only quantity (the vacuum toroidal field is subtracted),
    /// so the vacuum contribution is always zero.
    pub fn calculate_sensor_values_vacuum(&mut self, coils: PyRef<Coils>, _passives: PyRef<Passives>) {
        // The time-base is taken from the simulated coil currents (matching `FluxLoops`)
        let pf_names: Vec<String> = coils.results.get("pf").keys();
        let simulated_time: Array1<f64> = coils.results.get("pf").get(&pf_names[0]).get("i").get("simulated").get("time").unwrap_array1();
        let n_time: usize = simulated_time.len();

        for sensor_name in self.results.keys() {
            // Vacuum diamagnetic flux is zero
            let sensor_values: Array1<f64> = Array1::zeros(n_time);

            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("value", sensor_values);
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("b")
                .get_or_insert("calculated")
                .insert("time", simulated_time.clone());
        }
    }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Dialoop>");
        string_output += &format!("║  {:<74} ║\n", version);

        // n_sensors = self.results
        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}

// Rust only methods
impl Dialoop {
    /// This splits the Dialoop into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent ojbects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    ///
    /// Note: `Dialoop` has no Green's functions computed (it is not yet used as a constraint in the
    /// reconstruction), so there is no static object to return; only the Vec of time-dependent objects
    /// is produced. This mirrors `Coils::split_into_static_and_dynamic`.
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> Vec<SensorsDynamic> {
        let n_time: usize = times_to_reconstruct.len();

        // Sensor names
        let sensor_names: Vec<String> = self.results.keys();
        let n_sensors: usize = sensor_names.len();

        // Time dependent
        // Interpolate all sensors to `times_to_reconstruct`
        let mut measured: Array2<f64> = Array2::from_elem((n_sensors, n_time), f64::NAN);
        for i_sensor in 0..n_sensors {
            // Sensor name
            let sensor_name: &str = &sensor_names[i_sensor];

            // Experimental measurements (stored on the experimental time-base by `add_sensor`)
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
            // Create new `SensorsDynamic` instance and store
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured.slice(s![.., i_time]).to_owned(),
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        // Return the dynamic results
        return results_dynamic;
    }

    /// Calculate the sensor values
    pub fn calculate_sensor_values_rs(&mut self, _coils: &Coils, _passives: &Passives, _plasma: &Plasma) {
        // TODO: implement the diamagnetic flux calculation from the reconstructed equilibrium
        for _sensor_name in self.results.keys() {}
    }
}
