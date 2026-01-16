use crate::greens::greens_psi;
use crate::material_properties::copper_resistivity;
use crate::sensors::SensorsDynamic;
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use interpolation;
use ndarray::{Array1, Array2, Array3, s};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::f64::consts::PI;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass]
pub struct Coils {
    pub results: DataTree,
}

#[pymethods]
impl Coils {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }

    pub fn add_pf_coil(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        d_r: PyReadonlyArray1<f64>,
        d_z: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        measured: PyReadonlyArray1<f64>,
    ) {
        // Change Python types into Rust types
        let r_ndarray: Array1<f64> = r.to_owned_array();
        let z_ndarray: Array1<f64> = z.to_owned_array();
        let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
        let d_z_ndarray: Array1<f64> = d_z.to_owned_array();
        let time_ndarray: Array1<f64> = time.to_owned_array();
        let measured_ndarray: Array1<f64> = measured.to_owned_array();

        self.add_pf_coil_rs(name, &r_ndarray, &z_ndarray, &d_r_ndarray, &d_z_ndarray, &time_ndarray, &measured_ndarray)
    }

    // TODO: I'm not convinced about this implementation.
    // Is there a nicer way of doing this?
    pub fn set_pf_voltage(&mut self, name: &str, time: PyReadonlyArray1<f64>, measured_voltage: PyReadonlyArray1<f64>) {
        // Change Python types into Rust types

        let time_ndarray: Array1<f64> = time.to_owned_array();
        let voltages_ndarray: Array1<f64> = measured_voltage.to_owned_array();

        // Store the PF coil voltages
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("v")
            .insert("time_experimental", time_ndarray);
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("v")
            .insert("measured_experimental", voltages_ndarray);

        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .insert("driven_by", "voltage".to_string());
    }

    pub fn add_tf_coil(&mut self, time: PyReadonlyArray1<f64>, measured: PyReadonlyArray1<f64>) {
        // Change Python types into Rust types
        let time_ndarray: Array1<f64> = time.to_owned_array();
        let measured_ndarray: Array1<f64> = measured.to_owned_array();

        // Store the rod current
        self.results
            .get_or_insert("tf")
            .get_or_insert("rod_i")
            .insert("time_experimental", time_ndarray);
        self.results
            .get_or_insert("tf")
            .get_or_insert("rod_i")
            .insert("measured_experimental", measured_ndarray);
    }

    pub fn greens_with_self(&mut self) {
        for coil_name in self.results.get("pf").keys() {
            let coil_r: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("z").unwrap_array1();
            let coil_d_r: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("d_r").unwrap_array1();
            let coil_d_z: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("d_z").unwrap_array1();

            for other_coil_name in self.results.get("pf").keys() {
                let other_coil_r: Array1<f64> = self.results.get("pf").get(&other_coil_name).get("geometry").get("r").unwrap_array1();
                let other_coil_z: Array1<f64> = self.results.get("pf").get(&other_coil_name).get("geometry").get("z").unwrap_array1();

                let greens_filament_matrix: Array2<f64> =
                    greens_psi(coil_r.clone(), coil_z.clone(), other_coil_r, other_coil_z, coil_d_r.clone(), coil_d_z.clone());
                // println!("coil_name={}, other_coil_name={}", coil_name, other_coil_name);
                // println!("greens_filament_matrix={:?}", greens_filament_matrix);

                let g: f64 = greens_filament_matrix.sum();

                // Store greens
                self.results
                    .get_or_insert("pf")
                    .get_or_insert(&coil_name)
                    .get_or_insert("greens")
                    .insert(&other_coil_name, g);
            }
        }
    }

    fn calculate_coil_resistance(&mut self) {
        for coil_name in self.results.get("pf").keys() {
            // Calculate the cross sectional area
            let d_r: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("d_r").unwrap_array1();
            let d_z: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("d_z").unwrap_array1();
            let area: Array1<f64> = 4.0 * &d_r * &d_z;

            // Length of the coil
            let r: Array1<f64> = self.results.get("pf").get(&coil_name).get("geometry").get("r").unwrap_array1();
            let length: Array1<f64> = 2.0 * PI * r;

            // Resistivity of copper
            let temperature_in_kelvin: f64 = 293.15; // 20 degrees C
            let resistivity_copper_20c: f64 = copper_resistivity(temperature_in_kelvin); // ~ 1.68e-8 ohm * meter

            // Resistance of coil
            let resistance: f64 = (resistivity_copper_20c * length / area).sum();

            // Store resistance
            self.results.get_or_insert("pf").get_or_insert(&coil_name).insert("resistance", resistance);
        }
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Coils>");
        string_output += &format!("║  {:<74} ║\n", version);

        // PF names
        let pf_coil_names: Vec<String> = self.results.get("pf").keys();
        let n_pf_coils: usize = pf_coil_names.len();
        string_output += &format!("║  {:<74} ║\n", format!("n_pf_coils={}", n_pf_coils));
        string_output += &format!("║  {:<74} ║\n", format!("pf_coil_names=[{}]", pf_coil_names.join(", ")));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}

/// Rust only methods (either because we want to keep the methods private
/// or more likely because we the methods are incompatible with Python)
impl Coils {
    pub fn add_pf_coil_rs(
        &mut self,
        name: &str,
        r: &Array1<f64>,
        z: &Array1<f64>,
        d_r: &Array1<f64>,
        d_z: &Array1<f64>,
        time: &Array1<f64>,
        measured: &Array1<f64>,
    ) {
        // Store the PF coils
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("r", r.to_owned()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("z", z.to_owned()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("d_r", d_r.to_owned()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("d_z", d_z.to_owned()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("i")
            .insert("time_experimental", time.to_owned()); // Array1<f64>; shape = (n_time)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("i")
            .insert("measured_experimental", measured.to_owned()); // Array1<f64>; shape = (n_time)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .insert("driven_by", "current".to_string());
    }

    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> Vec<SensorsDynamic> {
        // TF coil
        let time_experimental: Array1<f64> = self.results.get("tf").get("rod_i").get("time_experimental").unwrap_array1();
        let measured_experimental: Array1<f64> = self.results.get("tf").get("rod_i").get("measured_experimental").unwrap_array1();

        // Create the interpolator
        let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(time_experimental.clone(), measured_experimental.clone())
            .expect("Coils.split_into_static_and_dynamic: Can't make interpolator, has the TF coil been added?");

        // Do the interpolation
        let measured_tf: Array1<f64> = interpolator
            .interpolate_array1(times_to_reconstruct)
            .expect("Coils.split_into_static_and_dynamic: Can't do TF interpolation");

        // Store in self
        self.results.get_or_insert("tf").get_or_insert("rod_i").insert("measured", measured_tf);

        // PF coils
        let coil_names: Vec<String> = self.results.get("pf").keys();
        let n_coils: usize = coil_names.len();

        // Time dependent
        // Interpolate to `times_to_reconstruct`
        let n_time: usize = times_to_reconstruct.len();
        let mut measured: Array2<f64> = Array2::zeros((n_coils, n_time));
        for i_coil in 0..n_coils {
            // Coils
            let coil_name: &String = &coil_names[i_coil];
            let time_experimental: Array1<f64> = self.results.get("pf").get(coil_name).get("i").get("time_experimental").unwrap_array1();
            let measured_experimental: Array1<f64> = self.results.get("pf").get(coil_name).get("i").get("measured_experimental").unwrap_array1();

            // Create the interpolator
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(time_experimental.clone(), measured_experimental.clone())
                .expect("Coils.split_into_static_and_dynamic: Can't make interpolator");

            // Do the interpolation
            let measured_this_coil: Array1<f64> = interpolator
                .interpolate_array1(times_to_reconstruct)
                .expect("Coils.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_coil, ..]).assign(&measured_this_coil);

            // Store in self
            self.results
                .get_or_insert("pf")
                .get_or_insert(coil_name)
                .get_or_insert("i")
                .insert("measured", measured_this_coil);
        }

        // MDSplus is "Coil-Major", but we want to rearrange the data to be "Time-Major"
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured.slice(s![.., i_time]).to_owned(),
            };

            results_dynamic.push(results_dynamic_this_time_slice);
        }

        // Return the dynamic results
        return results_dynamic;
    }
}
