use crate::grad_shafranov::GsSolution;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use lapack::*;
use ndarray::{Array1, Array2, Array3, s};
use ndarray_linalg::Norm;
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

// Global constants
const PI: f64 = std::f64::consts::PI;

#[derive(Clone)]
#[pyclass]
pub struct Passives {
    pub results: NestedDict,
}

/// Python accessible methods
#[pymethods]
impl Passives {
    #[new]
    pub fn new() -> Self {
        Self { results: NestedDict::new() }
    }

    ///
    /// [passive_name]["geometry"]["r"]                           = np.ndarray;  shape=(388,)
    /// [passive_name]["geometry"]["z"]                           = np.ndarray;  shape=(388,)
    /// [passive_name]["geometry"]["dr"]                          = np.ndarray;  shape=(388,)
    /// [passive_name]["geometry"]["dz"]                          = np.ndarray;  shape=(388,)
    /// [passive_name]["geometry"]["resistivity"]                 = float64
    /// [passive_name]["dof"][dof_name]["current_distribution"]   = np.ndarray;  shape=(388,)
    /// [passive_name]["dof"][dof_name]["cvalue"]                 = np.ndarray;  shape=(388,)
    ///
    pub fn add_passive(
        &mut self,
        name: &str,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        d_r: &Bound<'_, PyArray1<f64>>,
        d_z: &Bound<'_, PyArray1<f64>>,
        angle_1: &Bound<'_, PyArray1<f64>>,
        angle_2: &Bound<'_, PyArray1<f64>>,
        resistivity: f64,
        current_distribution_type: &str,
        n_dof: usize,
        regularisations: &Bound<'_, PyArray2<f64>>,
        regularisations_weight: &Bound<'_, PyArray1<f64>>,
    ) {
        // Change Python types into Rust types
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let d_r_ndarray: Array1<f64> = Array1::from(unsafe { d_r.as_array() }.to_vec());
        let d_z_ndarray: Array1<f64> = Array1::from(unsafe { d_z.as_array() }.to_vec());
        let angle_1_ndarray: Array1<f64> = Array1::from(unsafe { angle_1.as_array() }.to_vec());
        let angle_2_ndarray: Array1<f64> = Array1::from(unsafe { angle_2.as_array() }.to_vec());
        let regularisations_ndarray: Array2<f64> = Array2::from(unsafe { regularisations.as_array() }.to_owned());
        let regularisations_weight_ndarray: Array1<f64> = Array1::from(unsafe { regularisations_weight.as_array() }.to_owned());

        // Check sizes match
        // regularisation can either be empty or the same size as the number of degrees of freedom
        // assert!(regularisation_weight_ndarray.len() == 0 || regularisation_weight_ndarray.len() == n_dof);
        // if regularisation_weight_ndarray.len() == 0 {
        //     regularisation_weight_ndarray = Array1::zeros(n_dof);
        // }

        let n_filaments_this_passive: usize = r_ndarray.len();

        // Store the passive
        self.results.get_or_insert(name).get_or_insert("geometry").insert("r", r_ndarray.clone()); // Array1<f64>; shape = (n_filaments)
        self.results.get_or_insert(name).get_or_insert("geometry").insert("z", z_ndarray.clone()); // Array1<f64>; shape = (n_filaments)
        self.results.get_or_insert(name).get_or_insert("geometry").insert("d_r", d_r_ndarray.clone()); // Array1<f64>; shape = (n_filaments)
        self.results.get_or_insert(name).get_or_insert("geometry").insert("d_z", d_z_ndarray.clone()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("angle_1", angle_1_ndarray.clone()); // Array1<f64>; shape = (n_filaments)
        self.results
            .get_or_insert(name)
            .get_or_insert("geometry")
            .insert("angle_2", angle_2_ndarray.clone()); // Array1<f64>; shape = (n_filaments)

        // TODO: this is WRONG for angled filamanets!!!
        let area: Array1<f64> = &d_r_ndarray * &d_z_ndarray;

        self.results.get_or_insert(name).get_or_insert("geometry").insert("area", area.clone()); // Array1<f64>; shape = (n_filaments)

        self.results.get_or_insert(name).insert("resistivity", resistivity); // float64

        // Store the regularisations and regularisations_weight
        self.results.get_or_insert(name).insert("regularisations", regularisations_ndarray);
        self.results
            .get_or_insert(name)
            .insert("regularisations_weight", regularisations_weight_ndarray);

        if current_distribution_type == "constant_current_density" {
            let current_distribution: Array1<f64> = Array1::ones(n_filaments_this_passive) * area;

            self.results
                .get_or_insert(name)
                .get_or_insert("dof")
                .get_or_insert("constant_current_density")
                .insert("current_distribution", current_distribution);
        } else if current_distribution_type == "eig" {
            // Mutual inductance matrix
            let mut mutual_inductance_matrix: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                &r_ndarray,
                &z_ndarray,
                &d_r_ndarray,
                &d_z_ndarray,
                &angle_1_ndarray,
                &angle_2_ndarray,
                &r_ndarray,
                &z_ndarray,
                &d_r_ndarray,
                &d_z_ndarray,
                &angle_1_ndarray,
                &angle_2_ndarray,
            );

            // Store the mutual inductance matrix
            self.results
                .get_or_insert(name)
                .insert("mutual_inductance_matrix", mutual_inductance_matrix.clone());

            // Resistance matrix
            let length: Array1<f64> = 2.0 * PI * r_ndarray;
            let mut resistance_matrix: Array2<f64> = Array2::eye(n_filaments_this_passive) * resistivity * length / area;

            // Store the resistance matrix
            self.results.get_or_insert(name).insert("resistance_matrix", resistance_matrix.clone());

            // Initial lapack run figures out the optimal work array size
            let itype: Vec<i32> = vec![1];
            let jobz: u8 = b'V';
            let uplo: u8 = b'U';
            let n: i32 = n_filaments_this_passive as i32;
            let a: &mut [f64] = mutual_inductance_matrix.as_slice_mut().unwrap();
            let lda: i32 = n;
            let b: &mut [f64] = resistance_matrix.as_slice_mut().unwrap();
            let ldb: i32 = n;
            let mut _tmp: Vec<f64> = vec![0.0; n as usize];
            let w: &mut [f64] = &mut _tmp;
            let mut _tmp: Vec<f64> = vec![0.0; 1];
            let work: &mut [f64] = &mut _tmp;
            let lwork: i32 = -1;
            let mut _tmp: Vec<i32> = vec![0; 1];
            let iwork: &mut [i32] = &mut _tmp;
            let liwork: i32 = -1;
            let mut _tmp: i32 = -1;
            let info: &mut i32 = &mut _tmp;
            unsafe {
                dsygvd(
                    // `dsygvd` is a "symmetric" eigenvalue/eigenvector solver.
                    // It uses "divide and conquer" for the eigenvalues
                    // TODO: might be better using a direct solver which doesn't use divide and conquer
                    &itype, // &[i32]
                    jobz,   // u8
                    uplo,   // u8
                    n,      // i32
                    a,      // &mut [f64]
                    lda,    // i32
                    b,      // &mut [f64]
                    ldb,    // i32
                    w,      // &mut [f64]
                    work,   // &mut [f64]
                    lwork,  // i32
                    iwork,  // &mut [i32]
                    liwork, // i32
                    info,   // &mut i32
                );
            }

            // Allocate "work" arrays
            let lwork: i32 = work[0] as i32;
            let liwork: i32 = iwork[0];
            let mut _tmp: Vec<f64> = vec![0.0; lwork as usize];
            let work: &mut [f64] = &mut _tmp;
            let mut _tmp: Vec<i32> = vec![0; liwork as usize];
            let iwork: &mut [i32] = &mut _tmp;
            unsafe {
                dsygvd(
                    &itype, // &[i32]
                    jobz,   // u8
                    uplo,   // u8
                    n,      // i32
                    a,      // &mut [f64]
                    lda,    // i32
                    b,      // &mut [f64]
                    ldb,    // i32
                    w,      // &mut [f64]
                    work,   // &mut [f64]
                    lwork,  // i32
                    iwork,  // &mut [i32]
                    liwork, // i32
                    info,   // &mut i32
                );
            }

            // Eigenvalues are stored from smallest to largest
            let eigenvectors_unnormalised: Array2<f64> = Array2::from_shape_vec((n_filaments_this_passive, n_filaments_this_passive), a.to_vec()).unwrap();
            let eigenvalues: Array1<f64> = Array1::from_vec(w.to_vec());

            // Normalize eigenvectors using "2-norm"
            // This will panic if the eigenvector is itself zero, which should never happen, so let's keep the panic in place as it's the correct behaviour
            let mut eigenvectors: Array2<f64> = Array2::from_elem((n_filaments_this_passive, n_filaments_this_passive), f64::NAN);
            for i_eig in 0..n_filaments_this_passive {
                let eigenvector_unnormalised: Array1<f64> = eigenvectors_unnormalised.slice(s![i_eig, ..]).to_owned();
                let norm: f64 = eigenvector_unnormalised.norm_l2();
                let eigenvector_normalised: Array1<f64> = eigenvector_unnormalised / norm;
                eigenvectors.slice_mut(s![i_eig, ..]).assign(&eigenvector_normalised);
            }

            for i_eig in 0..n_dof {
                let dof_name: String = format!("eig_{:02}", i_eig + 1);

                let current_distribution_now: Array1<f64> = eigenvectors.slice(s![n_filaments_this_passive - i_eig - 1, ..]).to_owned() * 100.0;

                // Store the current_distribution
                self.results
                    .get_or_insert(name)
                    .get_or_insert("dof")
                    .get_or_insert(&dof_name)
                    .insert("current_distribution", current_distribution_now);

                // // Store the regularisation weight
                // self.results
                //     .get_or_insert(name)
                //     .get_or_insert("dof")
                //     .get_or_insert(&dof_name)
                //     .insert("regularisation_weight", regularisations_weight_ndarray[i_eig]);

                // Store time-constant (tau)
                self.results
                    .get_or_insert(name)
                    .get_or_insert("dof")
                    .get_or_insert(&dof_name)
                    .insert("time_constant", eigenvalues[n_filaments_this_passive - i_eig - 1].to_owned());
            }
        } else {
            panic!("Unknown option current_distribution_type={current_distribution_type}");
        }
    }

    pub fn greens_with_self(&mut self) {
        //
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║ {:<75} ║\n", " <gsfit_rs.Coils>");
        string_output += &format!("║  {:<74} ║\n", version);

        // // n_sensors = self.results
        // let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        // let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();
        // string_output += &format!("║ {:<75} ║\n", format!(" n_r = {}, n_z = {}", n_r.to_string(), n_z.to_string()));

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
impl Passives {
    pub fn equilibrium_post_processor(&mut self, gs_solutions: &Vec<GsSolution>) {
        let n_time: usize = gs_solutions.len();

        // Get sizes
        // TODO: BUG: this assumes that time-slice 0 has converged!!
        let passive_dof_values_single_time_slice: Array1<f64> = gs_solutions[0].passive_dof_values.to_owned(); // shape = [n_passive_dof]
        let n_dofs: usize = passive_dof_values_single_time_slice.len();

        // Allocate arrays for results
        let mut passive_dof_values: Array2<f64> = Array2::from_elem((n_time, n_dofs), f64::NAN);

        // Loop over time, collecting results
        for i_time in 0..n_time {
            if !gs_solutions[i_time].psi_b.is_nan() {
                // skip non-converged solutions
                passive_dof_values.slice_mut(s![i_time, ..]).assign(&gs_solutions[i_time].passive_dof_values);
            }
        }

        // Passive currents
        let mut i_dof: usize = 0;
        let passive_names: Vec<String> = self.results.keys();
        for passive_name in passive_names {
            let dof_names: Vec<String> = self.results.get(&passive_name).get("dof").keys();
            for dof_name in dof_names {
                let calculated_value: Array1<f64> = passive_dof_values.slice(s![.., i_dof]).to_owned();
                // Assign to Passive implementation
                self.results
                    .get_or_insert(&passive_name)
                    .get_or_insert("dof")
                    .get_or_insert(&dof_name)
                    .insert("calculated", calculated_value);

                // Increment counter
                i_dof = i_dof + 1;
            }
        }
    }
}
