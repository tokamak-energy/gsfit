use crate::grad_shafranov::GsSolution;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use crate::python_pickling_methods::{data_tree_to_py_dict, py_dict_to_data_tree};
use data_tree::{AddDataTreeGetters, DataTree, DataTreeAccumulator};
use lapack::*;
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_linalg::Norm;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::{SystemTime, UNIX_EPOCH};

const PI: f64 = std::f64::consts::PI;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass(module = "gsfit_rs")]
pub struct Passives {
    pub results: DataTree,
}

/// Python accessible methods
#[pymethods]
impl Passives {
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
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
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        d_r: PyReadonlyArray1<f64>,
        d_z: PyReadonlyArray1<f64>,
        angle_1: PyReadonlyArray1<f64>,
        angle_2: PyReadonlyArray1<f64>,
        resistivity: f64,
        current_distribution_type: &str,
        n_dof: usize,
        regularisations: PyReadonlyArray2<f64>,
        regularisations_weight: PyReadonlyArray1<f64>,
    ) {
        // Change Python types into Rust types
        let r_ndarray: Array1<f64> = r.to_owned_array();
        let z_ndarray: Array1<f64> = z.to_owned_array();
        let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
        let d_z_ndarray: Array1<f64> = d_z.to_owned_array();
        let angle_1_ndarray: Array1<f64> = angle_1.to_owned_array();
        let angle_2_ndarray: Array1<f64> = angle_2.to_owned_array();
        let regularisations_ndarray: Array2<f64> = regularisations.to_owned_array();
        let regularisations_weight_ndarray: Array1<f64> = regularisations_weight.to_owned_array();
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

/// Rust only methods (either because we want to keep the methods private
/// or more likely because we the methods are incompatible with Python)
impl Passives {
    pub fn equilibrium_post_processor(&mut self, gs_solutions: &Vec<GsSolution>) {
        let n_time: usize = gs_solutions.len();
        if n_time == 0 {
            println!("Passives.equilibrium_post_processor: no time slices to process, returning");
            return;
        }

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
                i_dof += 1;
            }
        }
    }

    pub fn get_n_passive_filaments(&self) -> usize {
        let mut n_passive_filaments: usize = 0;
        for passive_name in &self.results.keys() {
            let r: Array1<f64> = self.results.get(passive_name).get("geometry").get("r").unwrap_array1();
            n_passive_filaments += r.len();
        }
        return n_passive_filaments;
    }

    pub fn greens_with_self(&self) -> Array2<f64> {
        // Collect the passive conductor locations
        let mut passive_locations_r: Vec<f64> = Vec::new();
        let mut passive_locations_z: Vec<f64> = Vec::new();
        let mut passive_locations_d_r: Vec<f64> = Vec::new();
        let mut passive_locations_d_z: Vec<f64> = Vec::new();
        let mut passive_locations_angle_1: Vec<f64> = Vec::new();
        let mut passive_locations_angle_2: Vec<f64> = Vec::new();
        for passive_name in &self.results.keys() {
            let r: Array1<f64> = self.results.get(passive_name).get("geometry").get("r").unwrap_array1();
            passive_locations_r.extend(r.iter());
            let z: Array1<f64> = self.results.get(passive_name).get("geometry").get("z").unwrap_array1();
            passive_locations_z.extend(z.iter());
            let d_r: Array1<f64> = self.results.get(passive_name).get("geometry").get("d_r").unwrap_array1();
            passive_locations_d_r.extend(d_r.iter());
            let d_z: Array1<f64> = self.results.get(passive_name).get("geometry").get("d_z").unwrap_array1();
            passive_locations_d_z.extend(d_z.iter());
            let angle_1: Array1<f64> = self.results.get(passive_name).get("geometry").get("angle_1").unwrap_array1();
            passive_locations_angle_1.extend(angle_1.iter());
            let angle_2: Array1<f64> = self.results.get(passive_name).get("geometry").get("angle_2").unwrap_array1();
            passive_locations_angle_2.extend(angle_2.iter());
        }

        let passive_locations_r: Array1<f64> = Array1::from_vec(passive_locations_r);
        let passive_locations_z: Array1<f64> = Array1::from_vec(passive_locations_z);
        let passive_locations_d_r: Array1<f64> = Array1::from_vec(passive_locations_d_r);
        let passive_locations_d_z: Array1<f64> = Array1::from_vec(passive_locations_d_z);
        let passive_locations_angle_1: Array1<f64> = Array1::from_vec(passive_locations_angle_1);
        let passive_locations_angle_2: Array1<f64> = Array1::from_vec(passive_locations_angle_2);

        let g_psi: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
            &passive_locations_r,
            &passive_locations_z,
            &passive_locations_d_r,
            &passive_locations_d_z,
            &passive_locations_angle_1,
            &passive_locations_angle_2,
            &passive_locations_r,
            &passive_locations_z,
            &passive_locations_d_r,
            &passive_locations_d_z,
            &passive_locations_angle_1,
            &passive_locations_angle_2,
        );

        return g_psi;
    }

    pub fn get_all_passive_filament_geometry(&self) -> PassiveGeometryAll {
        // Collect the passive conductor locations
        let mut passive_locations_r: Vec<f64> = Vec::new();
        let mut passive_locations_z: Vec<f64> = Vec::new();
        let mut passive_locations_d_r: Vec<f64> = Vec::new();
        let mut passive_locations_d_z: Vec<f64> = Vec::new();
        let mut passive_locations_angle_1: Vec<f64> = Vec::new();
        let mut passive_locations_angle_2: Vec<f64> = Vec::new();
        let mut passive_resistivity: Vec<f64> = Vec::new();
        for passive_name in &self.results.keys() {
            let r: Array1<f64> = self.results.get(passive_name).get("geometry").get("r").unwrap_array1();
            passive_locations_r.extend(r.iter());
            let z: Array1<f64> = self.results.get(passive_name).get("geometry").get("z").unwrap_array1();
            passive_locations_z.extend(z.iter());
            let d_r: Array1<f64> = self.results.get(passive_name).get("geometry").get("d_r").unwrap_array1();
            passive_locations_d_r.extend(d_r.iter());
            let d_z: Array1<f64> = self.results.get(passive_name).get("geometry").get("d_z").unwrap_array1();
            passive_locations_d_z.extend(d_z.iter());
            let angle_1: Array1<f64> = self.results.get(passive_name).get("geometry").get("angle_1").unwrap_array1();
            passive_locations_angle_1.extend(angle_1.iter());
            let angle_2: Array1<f64> = self.results.get(passive_name).get("geometry").get("angle_2").unwrap_array1();
            passive_locations_angle_2.extend(angle_2.iter());
            let resistivity: f64 = self.results.get(passive_name).get("resistivity").unwrap_f64();
            for _i in 0..r.len() {
                passive_resistivity.push(resistivity);
            }
        }

        let passive_geometry_all = PassiveGeometryAll {
            r: Array1::from_vec(passive_locations_r),
            z: Array1::from_vec(passive_locations_z),
            d_r: Array1::from_vec(passive_locations_d_r),
            d_z: Array1::from_vec(passive_locations_d_z),
            angle_1: Array1::from_vec(passive_locations_angle_1),
            angle_2: Array1::from_vec(passive_locations_angle_2),
            resistivity: Array1::from_vec(passive_resistivity),
        };

        return passive_geometry_all;
    }

    pub fn get_passive_filament_currents_from_simulated(&self) -> Array2<f64> {
        let passive_names: Vec<String> = self.results.keys();
        let n_passive_filaments: usize = self.get_n_passive_filaments();
        let n_time: usize = self
            .results
            .get(&passive_names[0])
            .get("i_filaments")
            .get("simulated")
            .get("time")
            .unwrap_array1()
            .len();

        // Collect the passive filament currents
        let mut passive_currents: Array2<f64> = Array2::from_elem((n_time, n_passive_filaments), f64::NAN);
        let mut i_start: usize = 0;
        let mut i_end: usize;
        for passive_name in &passive_names {
            let passive_currents_local: Array2<f64> = self.results.get(passive_name).get("i_filaments").get("simulated").get("value").unwrap_array2();

            i_end = i_start + passive_currents_local.len_of(Axis(1));
            passive_currents.slice_mut(s![.., i_start..i_end]).assign(&passive_currents_local);
            i_start = i_end;
        }

        // Return the passive filament currents
        passive_currents
    }
}

pub struct PassiveGeometryAll {
    pub r: Array1<f64>,
    pub z: Array1<f64>,
    pub d_r: Array1<f64>,
    pub d_z: Array1<f64>,
    pub angle_1: Array1<f64>,
    pub angle_2: Array1<f64>,
    pub resistivity: Array1<f64>,
}
