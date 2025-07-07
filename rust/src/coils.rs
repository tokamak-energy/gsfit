use crate::greens::greens;
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use crate::sensors::SensorsDynamic;
use ndarray::{Array1, Array2, Array3, s};
use ndarray_interp::interp1d::Interp1D;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone)]
#[pyclass]
pub struct Coils {
    pub results: NestedDict,
}

#[pymethods]
impl Coils {
    #[new]
    pub fn new() -> Self {
        Self { results: NestedDict::new() }
    }

    pub fn add_pf_coil(
        &mut self,
        name: &str,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        d_r: &Bound<'_, PyArray1<f64>>,
        d_z: &Bound<'_, PyArray1<f64>>,
        time: &Bound<'_, PyArray1<f64>>,
        measured: &Bound<'_, PyArray1<f64>>,
    ) {
        // Change Python types into Rust types
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let d_r_ndarray: Array1<f64> = Array1::from(unsafe { d_r.as_array() }.to_vec());
        let d_z_ndarray: Array1<f64> = Array1::from(unsafe { d_z.as_array() }.to_vec());
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let measured_ndarray: Array1<f64> = Array1::from(unsafe { measured.as_array() }.to_vec());

        // Store the PF coils
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("r", r_ndarray); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("z", z_ndarray); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("d_r", d_r_ndarray); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("d_z", d_z_ndarray); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("i")
            .insert("time_experimental", time_ndarray); // Array1<f64>; shape = (n_time)
        self.results
            .get_or_insert("pf")
            .get_or_insert(name)
            .get_or_insert("i")
            .insert("measured_experimental", measured_ndarray); // Array1<f64>; shape = (n_time)
    }

    pub fn add_tf_coil(&mut self, time: &Bound<'_, PyArray1<f64>>, measured: &Bound<'_, PyArray1<f64>>) {
        // Change Python types into Rust types
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let measured_ndarray: Array1<f64> = Array1::from(unsafe { measured.as_array() }.to_vec());

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
            let coil_r = self.results.get("pf").get(&coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z = self.results.get("pf").get(&coil_name).get("geometry").get("z").unwrap_array1();
            let coil_d_r = self.results.get("pf").get(&coil_name).get("geometry").get("d_r").unwrap_array1();
            let coil_d_z = self.results.get("pf").get(&coil_name).get("geometry").get("d_z").unwrap_array1();

            for other_coil_name in self.results.get("pf").keys() {
                let other_coil_r = self.results.get("pf").get(&other_coil_name).get("geometry").get("r").unwrap_array1();
                let other_coil_z = self.results.get("pf").get(&other_coil_name).get("geometry").get("z").unwrap_array1();

                let greens_filament_matrix: Array2<f64> =
                    greens(coil_r.clone(), coil_z.clone(), other_coil_r, other_coil_z, coil_d_r.clone(), coil_d_z.clone());

                let g: f64 = greens_filament_matrix.sum();

                // Store greens
                self.results
                    .get_or_insert("pf")
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .get_or_insert(&coil_name)
                    .insert(&other_coil_name, g);
            }
        }
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Coils>");
        string_output += &format!("║  {:<74} ║\n", version);

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

/// Rust only methods (either because we want to keep the methods private
/// or more likely because we the methods are incompatible with Python)
impl Coils {
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> Vec<SensorsDynamic> {
        // TF coil
        let time_experimental: Array1<f64> = self.results.get("tf").get("rod_i").get("time_experimental").unwrap_array1();
        let measured_experimental: Array1<f64> = self.results.get("tf").get("rod_i").get("measured_experimental").unwrap_array1();

        // Create the interpolator
        let interpolator = Interp1D::builder(measured_experimental)
            .x(time_experimental.clone())
            .build()
            .expect("Coils.split_into_static_and_dynamic: Can't make Interp1D");

        // Do the interpolation
        let measured_tf: Array1<f64> = interpolator
            .interp_array(&times_to_reconstruct)
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
            let interpolator = Interp1D::builder(measured_experimental)
                .x(time_experimental.clone())
                .build()
                .expect("Coils.split_into_static_and_dynamic: Can't make Interp1D");

            // Do the interpolation
            let measured_this_coil: Array1<f64> = interpolator
                .interp_array(&times_to_reconstruct)
                .expect("Coils.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_coil, ..]).assign(&measured_this_coil);

            // Store in self
            self.results
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
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
