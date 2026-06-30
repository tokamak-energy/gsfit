use crate::Plasma;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic, create_empty_sensor_data};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use ndarray::{Array1, Array2, Array3, Axis, s};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

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

/// Diamagnetic flux loop ("DIALOOP").
///
/// # Physics
///
/// The diamagnetic flux loop measures the difference between the actual toroidal flux and the
/// vacuum toroidal field, integrated over the poloidal cross-section (Moret Eq. 41):
///
/// ```text
///     Phi_t = integral( (f - f_vac) / R ) dR dZ
/// ```
///
/// where `f = R * B_phi` is the toroidal flux function (the poloidal-current function) and
/// `f_vac = R0 * B_phi0 = MU_0 * i_rod / (2 * PI)` is the vacuum value. Outside the plasma
/// `f = f_vac`, so the integrand is zero there and the integral is taken over the plasma mask.
///
/// Crucially, `f` depends on the **poloidal** currents (the ff' source function), *not* on the
/// toroidal currents. The Green's tables in gsfit relate toroidal currents to `psi`/`B_p`, so the
/// diamagnetic loop deliberately uses **no Green's functions** (no coils, no passives, no plasma
/// grid Green's, no vertical-stabilisation terms).
///
/// In gsfit, `f` is reconstructed from the ff' source function exactly as in `epp_bt_2d`:
///
/// ```text
///     f = sqrt( f_vac^2 + 2 * (psi_b - psi_a) * G(psi_n) )
/// ```
///
/// where `G(psi_n) = sum_i ff'_dof[i] * ff'_integral_i(psi_n)` is the integral of the ff' source
/// function.
#[pymethods]
impl Dialoop {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    /// Add a single diamagnetic loop sensor.
    ///
    /// The loop path `(r, z)` is stored for reference only; the diamagnetic flux is an integral
    /// over the poloidal cross-section enclosed by the loop and does not use the path directly.
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
        // Loop-path geometry (stored for reference only)
        self.results.get_or_insert(name).get_or_insert("geometry").insert("r_path", r.to_owned_array());
        self.results.get_or_insert(name).get_or_insert("geometry").insert("z_path", z.to_owned_array());

        // Fit settings
        self.results.get_or_insert(name).get_or_insert("fit_settings").insert("comment", fit_settings_comment);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("expected_value", fit_settings_expected_value);
        self.results.get_or_insert(name).get_or_insert("fit_settings").insert("include", fit_settings_include);
        self.results.get_or_insert(name).get_or_insert("fit_settings").insert("weight", fit_settings_weight);

        // Measurements
        self.results.get_or_insert(name).get_or_insert("b").get_or_insert("experimental").insert("time", time.to_owned_array());
        self.results
            .get_or_insert(name)
            .get_or_insert("b")
            .get_or_insert("experimental")
            .insert("value", measured.to_owned_array());
    }

    /// Calculate the sensor values from a reconstructed `Plasma` (Python entry-point).
    pub fn calculate_sensor_values(&mut self, plasma: PyRef<Plasma>) {
        let plasma_rs: &Plasma = &plasma;
        self.calculate_sensor_values_rs(plasma_rs);
    }

    /// Print to screen, to be used within Python
    pub fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Dialoop>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║  {:<74} ║\n", format!("n_sensors = {}", n_sensors));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}

// Rust only methods
impl Dialoop {
    /// Split the `Dialoop` into:
    /// 1.) Static (non time-dependent) data. Sensors are down-selected here based on `["fit_settings"]["include"]`.
    /// 2.) A `Vec` of time-dependent objects (one per time-slice we want to reconstruct).
    ///
    /// Diamagnetic loops do not use Green's functions: the response is computed directly from the
    /// ff' source function inside the GS solver (see `gs_solution.rs`). The Green's arrays below are
    /// therefore left empty.
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (Vec<SensorsStatic>, Vec<SensorsDynamic>) {
        let n_time: usize = times_to_reconstruct.len();

        // Vector of boolean's to say if we use the sensor or not
        let include: Vec<bool> = self.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();

        // Convert from boolean to indices
        let include_indices: Vec<usize> = include
            .iter()
            .enumerate()
            .filter(|(_, include)| **include)
            .map(|(i, _)| i)
            .collect();

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();
        let n_sensors_all: usize = sensor_names_all.len();
        let n_sensors: usize = include_indices.len();

        // Time dependent: interpolate all sensors to `times_to_reconstruct`.
        // Note, this is done *before* the early return below, because the equilibrium
        // post-processor always reads `["b"]["measured"]["value"]` whenever any dialoop sensor
        // exists.
        //
        // Only sensors that are actually included are interpolated. For excluded sensors we store
        // `NaN` (the post-processor ignores them), so an un-included dialoop does not force its
        // experimental time-range onto `times_to_reconstruct` (which would otherwise panic with an
        // out-of-bounds interpolation error).
        let mut measured: Array2<f64> = Array2::from_elem((n_sensors_all, n_time), f64::NAN);
        for i_sensor in 0..n_sensors_all {
            let sensor_name: &str = &sensor_names_all[i_sensor];

            let measured_this_sensor: Array1<f64> = if include[i_sensor] {
                let experimental_time: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("time").unwrap_array1();
                let experimental_values: Array1<f64> = self.results.get(sensor_name).get("b").get("experimental").get("value").unwrap_array1();

                let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(experimental_time.clone(), experimental_values.clone())
                    .expect("Dialoop.split_into_static_and_dynamic: Can't make interpolator");

                interpolator
                    .interpolate_array1(times_to_reconstruct)
                    .expect("Dialoop.split_into_static_and_dynamic: Can't do interpolation")
            } else {
                // Excluded sensor: do not require the experimental data to cover the reconstruct times
                Array1::from_elem(n_time, f64::NAN)
            };

            measured.slice_mut(s![i_sensor, ..]).assign(&measured_this_sensor);

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

        // If there are no sensors selected, return empty data
        if n_sensors == 0 {
            let (static_data_empty, dynamic_data_empty): (SensorsStatic, SensorsDynamic) = create_empty_sensor_data();
            let static_data_empty_vs_time: Vec<SensorsStatic> = vec![static_data_empty; n_time];
            let dynamic_data_empty_vs_time: Vec<SensorsDynamic> = vec![dynamic_data_empty; n_time];
            return (static_data_empty_vs_time, dynamic_data_empty_vs_time);
        }

        // Fit settings
        let fit_settings_weight: Array1<f64> = self.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let fit_settings_weight: Array1<f64> = fit_settings_weight.select(Axis(0), &include_indices);
        let fit_settings_expected_value: Array1<f64> = self.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let fit_settings_expected_value: Array1<f64> = fit_settings_expected_value.select(Axis(0), &include_indices);

        // No Green's functions for the diamagnetic loop (see method docstring).
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid: Array2::zeros((0, n_sensors)),
            greens_with_pf: Array2::zeros((0, n_sensors)),
            greens_with_passives: Array2::zeros((0, n_sensors)),
            greens_d_sensor_dz: Array2::zeros((0, n_sensors)),
            fit_settings_weight,
            fit_settings_expected_value,
            geometry_r: Array1::from_elem(n_sensors, f64::NAN), // dialoop is integrated; no single (r, z) position
            geometry_z: Array1::from_elem(n_sensors, f64::NAN), // dialoop is integrated; no single (r, z) position
        };

        // MDSplus is "Sensor-Major", but we want the data to be "Time-Major"
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        let results_static_time_dependent: Vec<SensorsStatic> = vec![results_static.clone(); n_time];

        (results_static_time_dependent, results_dynamic)
    }

    /// Calculate the diamagnetic flux for each sensor and time-slice from a reconstructed plasma.
    ///
    /// Uses the exact toroidal flux function `f = sqrt(f_vac^2 + 2*(psi_b - psi_a)*G)`, consistent
    /// with `epp_bt_2d` in `plasma.rs`, and integrates `(f - f_vac) / R` over the plasma mask
    /// (Moret Eq. 41).
    pub fn calculate_sensor_values_rs(&mut self, plasma: &Plasma) {
        let psi_n_2d: Array3<f64> = plasma.results.get("two_d").get("psi_n").unwrap_array3(); // shape = [n_time, n_z, n_r]
        let mask_2d: Array3<f64> = plasma.results.get("two_d").get("mask").unwrap_array3(); // shape = [n_time, n_z, n_r]
        let flat_r: Array1<f64> = plasma.results.get("grid").get("flat").get("r").unwrap_array1(); // shape = [n_z * n_r]
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let time: Array1<f64> = plasma.results.get("time").unwrap_array1();
        let psi_a_vs_time: Array1<f64> = plasma.results.get("global").get("psi_a").unwrap_array1();
        let psi_b_vs_time: Array1<f64> = plasma.results.get("global").get("psi_b").unwrap_array1();
        let i_rod_vs_time: Array1<f64> = plasma.results.get("global").get("i_rod").unwrap_array1();
        let ff_coeffs: Array2<f64> = plasma.results.get("source_functions").get("ff_prime").get("coefficients").unwrap_array2(); // shape = [n_time, n_dof]

        let n_time: usize = time.len();

        for sensor_name in self.results.keys() {
            let mut values: Array1<f64> = Array1::zeros(n_time);

            for i_time in 0..n_time {
                let psi_n_flat: Array1<f64> = Array1::from_iter(psi_n_2d.slice(s![i_time, .., ..]).iter().copied());
                let mask_flat: Array1<f64> = Array1::from_iter(mask_2d.slice(s![i_time, .., ..]).iter().copied());
                let ff_dof: Array1<f64> = ff_coeffs.slice(s![i_time, ..]).to_owned();

                // Vacuum toroidal flux function: f_vac = R0 * B_phi0 = MU_0 * i_rod / (2 * PI)
                let f_vac: f64 = MU_0 * i_rod_vs_time[i_time] / (2.0 * PI);

                // G(psi_n) = sum_i ff'_dof[i] * ff'_integral_i(psi_n)
                let g_integral: Array1<f64> = plasma.ff_prime_source_function.source_function_integral(&psi_n_flat, &ff_dof);

                // f = sqrt( f_vac^2 + 2*(psi_b - psi_a)*G ), then (f - f_vac)
                let d_psi: f64 = psi_b_vs_time[i_time] - psi_a_vs_time[i_time];
                let f_squared: Array1<f64> = 2.0 * d_psi * &g_integral + f_vac * f_vac;
                let f_minus_f_vac: Array1<f64> = f_squared.mapv(f64::sqrt) - f_vac;

                // Phi_t = integral( mask * (f - f_vac) / R ) dA
                let integrand: Array1<f64> = &mask_flat * &f_minus_f_vac / &flat_r;
                values[i_time] = integrand.sum() * d_area;
            }

            self.results.get_or_insert(&sensor_name).get_or_insert("b").get_or_insert("calculated").insert("value", values.clone());
            self.results.get_or_insert(&sensor_name).get_or_insert("b").get_or_insert("calculated").insert("time", time.clone());
        }
    }
}
