use crate::Plasma;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{create_empty_sensor_data, SensorsDynamic, SensorsStatic};
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
    /// [probe_name]["geometry"]["r"]                                   = Array1<f64>;  shape=[n_path_points]
    /// [probe_name]["geometry"]["z"]                                   = Array1<f64>;  shape=[n_path_points]
    /// [probe_name]["greens"]["d_plasma_d_z"]                          = Array1;  shape=[n_z * n_r]
    /// [probe_name]["greens"]["passives"][passive_name][dof_name]      = f64
    /// [probe_name]["greens"]["pf"][pf_name]                           = f64
    /// [probe_name]["greens"]["plasma"]                                = Array1;  shape=[n_z * n_r]
    /// ```
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
        // Convert into Rust data types
        let r_ndarray: Array1<f64> = r.to_owned_array();
        let z_ndarray: Array1<f64> = z.to_owned_array();
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

        // Geometry (path of integration)
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("r", r_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("z", z_ndarray);
    }

    /// Compute the Green's functions with the PF coils
    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &coils;

        for sensor_name in self.results.keys() {
            for pf_coil_name in coils_local.results.get("pf").keys() {
                let coil_r: Array1<f64> = coils_local.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                let coil_z: Array1<f64> = coils_local.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();

                // TODO: Calculate the diamagnetic flux response to each PF coil
                // For now, store zero (dialoop Green's with coils not yet implemented)
                let g: f64 = 0.0;

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, g);
            }
        }
    }

    /// Compute the Green's functions with the passives
    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &passives;

        for sensor_name in self.results.keys() {
            // Calculate Greens with each passive degree of freedom
            for passive_name in passives_local.results.keys() {
                let dof_accumulator: DataTreeAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = dof_accumulator.keys();

                for dof_name in dof_names {
                    // TODO: Calculate the diamagnetic flux response to each passive DOF
                    // For now, store zero (dialoop Green's with passives not yet implemented)
                    let g: f64 = 0.0;

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

    /// Compute the Green's functions with the plasma
    fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &plasma;

        let plasma_r: Array1<f64> = plasma_local.results.get("grid").get("flat").get("r").unwrap_array1();
        let plasma_z: Array1<f64> = plasma_local.results.get("grid").get("flat").get("z").unwrap_array1();

        for sensor_name in self.results.keys() {
            // TODO: Calculate the diamagnetic flux Green's function over the plasma domain
            // The diamagnetic flux is integrated as:
            //   Φ_t = (μ₀ / 2π r₀ B_φ0) ∫_Ωp G_g(ψ(r,z)) r dr dz
            // where G_g is a primitive of the source function base function
            // For now, store zero array (dialoop Green's with plasma not yet implemented)
            let g_with_plasma: Array1<f64> = Array1::zeros(plasma_r.len());
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma);

            // Vertical stability: derivative with respect to z
            let g_d_plasma_d_z: Array1<f64> = Array1::zeros(plasma_r.len());
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
        }
    }

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
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        let n_time: usize = times_to_reconstruct.len();

        // Vector of booleans to say if we use the sensor or not
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
        let greens_with_grid: Array2<f64> = greens_with_grid.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]
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

        let mut greens_with_passives: Array2<f64> = Array2::from_elem((n_dof_total, n_sensors), f64::NAN);

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
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

                    // Store the name
                    // dof_names_total[i_dof_total] = dof_name;

                    // Keep count
                    i_dof_total += 1;
                }
            }
        }

        // Create the `DialoopStatic` data
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid,
            greens_with_pf,
            greens_with_passives,
            greens_d_sensor_dz,
            fit_settings_weight,
            fit_settings_expected_value,
            geometry_r: Array1::zeros(n_sensors), // Dialoop is integrated; no single (r, z) position
            geometry_z: Array1::zeros(n_sensors),
        };
        let results_static_vs_time: Vec<SensorsStatic> = vec![results_static; n_time];

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

        // Return the static and dynamic results
        return (results_static_vs_time, results_dynamic);
    }

    /// Calculate the sensor values
    pub fn calculate_sensor_values_rs(&mut self, _coils: &Coils, _passives: &Passives, _plasma: &Plasma) {
        // TODO: Implement the diamagnetic flux calculation from the reconstructed equilibrium
        // using the Green's functions computed by greens_with_coils(), greens_with_passives(), and greens_with_plasma()
        for _sensor_name in self.results.keys() {}
    }
}
