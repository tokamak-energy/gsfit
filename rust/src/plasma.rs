use crate::coils::Coils;
use crate::grad_shafranov::GsSolution;
use crate::greens::{d_greens_magnetic_field_dz, greens, greens_magnetic_field};
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use crate::passives::Passives;
use crate::source_functions::SourceFunctionTraits;
use crate::source_functions::{EfitPolynomial, LiuqePolynomial};
use contour::ContourBuilder;
use core::f64;
use geo::Area;
use geo::Centroid;
use geo::Line;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array, Array1, Array2, Array3, Axis, s};
use ndarray_interp::interp1d::Interp1D;
use ndarray_interp::interp2d::Interp2D;
use ndarray_stats::QuantileExt;
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::result;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;
const PI: f64 = std::f64::consts::PI;

// use log::{debug, error, info};
// env_logger::init(); // in the "new" method

#[derive(Clone)]
#[pyclass]
pub struct Plasma {
    pub results: NestedDict,
    pub p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
}

/// Python accessible methods
#[pymethods]
impl Plasma {
    #[new]
    pub fn new(
        n_r: usize,
        n_z: usize,
        r_min: f64,
        r_max: f64,
        z_min: f64,
        z_max: f64,
        psi_n: &Bound<'_, PyArray1<f64>>,
        limit_pts_r: &Bound<'_, PyArray1<f64>>,
        limit_pts_z: &Bound<'_, PyArray1<f64>>,
        vessel_r: &Bound<'_, PyArray1<f64>>,
        vessel_z: &Bound<'_, PyArray1<f64>>,
        p_prime_source_function: Py<PyAny>,  // This is a big ugly!
        ff_prime_source_function: Py<PyAny>, // This is a big ugly!
    ) -> Self {
        // Change Python types into Rust types
        let psi_n_ndarray: Array1<f64> = Array1::from(unsafe { psi_n.as_array() }.to_vec());
        let limit_pts_r_ndarray: Array1<f64> = Array1::from(unsafe { limit_pts_r.as_array() }.to_vec());
        let limit_pts_z_ndarray: Array1<f64> = Array1::from(unsafe { limit_pts_z.as_array() }.to_vec());
        let vessel_r_ndarray: Array1<f64> = Array1::from(unsafe { vessel_r.as_array() }.to_vec());
        let vessel_z_ndarray: Array1<f64> = Array1::from(unsafe { vessel_z.as_array() }.to_vec());

        // Extract an object from p_prime_source_function which contains the common traits
        let p_prime_source_function_arc: Arc<dyn SourceFunctionTraits + Send + Sync> = Python::with_gil(|py| {
            p_prime_source_function
                .extract::<Py<PyAny>>(py)
                .and_then(|obj| {
                    if let Ok(efit) = obj.extract::<PyRef<EfitPolynomial>>(py) {
                        Ok(Arc::new(EfitPolynomial {
                            n_dof: efit.n_dof,
                            regularisations: efit.regularisations.clone(),
                        }) as Arc<dyn SourceFunctionTraits + Send + Sync>)
                    } else if let Ok(liuqe) = obj.extract::<PyRef<LiuqePolynomial>>(py) {
                        Ok(Arc::new(LiuqePolynomial {
                            n_dof: liuqe.n_dof,
                            regularisations: liuqe.regularisations.clone(),
                        }) as Arc<dyn SourceFunctionTraits + Send + Sync>)
                    } else {
                        Err(pyo3::exceptions::PyTypeError::new_err(
                            "p_prime_source_function must implement SourceFunctionTraits",
                        ))
                    }
                })
                .expect("Failed to extract p_prime_source_function")
        });

        // Extract an object from p_prime_source_function which contains the common traits
        let ff_prime_source_function_arc: Arc<dyn SourceFunctionTraits + Send + Sync> = Python::with_gil(|py| {
            ff_prime_source_function
                .extract::<Py<PyAny>>(py)
                .and_then(|obj| {
                    if let Ok(efit) = obj.extract::<PyRef<EfitPolynomial>>(py) {
                        Ok(Arc::new(EfitPolynomial {
                            n_dof: efit.n_dof,
                            regularisations: efit.regularisations.clone(),
                        }) as Arc<dyn SourceFunctionTraits + Send + Sync>)
                    } else if let Ok(liuqe) = obj.extract::<PyRef<LiuqePolynomial>>(py) {
                        Ok(Arc::new(LiuqePolynomial {
                            n_dof: liuqe.n_dof,
                            regularisations: liuqe.regularisations.clone(),
                        }) as Arc<dyn SourceFunctionTraits + Send + Sync>)
                    } else {
                        Err(pyo3::exceptions::PyTypeError::new_err(
                            "ff_prime_source_function must implement SourceFunctionTraits",
                        ))
                    }
                })
                .expect("Failed to extract ff_prime_source_function")
        });

        // Create storage
        let mut results: NestedDict = NestedDict::new();

        // Create (r, z) grids
        let r: Array1<f64> = Array::linspace(r_min, r_max, n_r);
        let z: Array1<f64> = Array::linspace(z_min, z_max, n_z);

        // Grid spacing
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let d_area: f64 = d_r * d_z;

        // 2d (r, z) mesh
        let mut mesh_r: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
        let mut mesh_z: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                mesh_r[[i_z, i_r]] = r[i_r];
                mesh_z[[i_z, i_r]] = z[i_z];
            }
        }

        // Flattended 2d mesh
        let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
        let flat_z: Array1<f64> = mesh_z.flatten().to_owned();

        // Calculate the Greens grid-grid
        let d_r_flat: Array1<f64> = &r * 0.0 + d_r;
        let d_z_flat: Array1<f64> = &r * 0.0 + d_z;
        let g_grid_grid_psi: Array2<f64> = greens(
            flat_r.clone(),
            flat_z.clone(),
            r.clone(),
            0.0 * r.clone() + z[0],
            d_r_flat, // TODO: I don't like thsee variables
            d_z_flat,
        );
        let (g_grid_grid_br, g_grid_grid_bz): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
            flat_r.clone(), // sensors
            flat_z.clone(),
            r.clone(), // current sources
            0.0 * r.clone() + z[0],
        );

        let (mut g_d_grid_grid_br_d_z, mut g_d_grid_grid_bz_d_z): (Array2<f64>, Array2<f64>) = d_greens_magnetic_field_dz(
            flat_r.clone(), // sensors
            flat_z.clone(),
            r.clone(), // current sources
            0.0 * r.clone() + z[0],
        );

        // TODO: we could change this to set the self-values to 0.0, instead of doing "if" statement, which is slow
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d_grid_grid_br_d_z[[i_rz, i_r]].is_nan() {
                    g_d_grid_grid_br_d_z[[i_rz, i_r]] = 0.0; // TODO: this can be improved
                    g_d_grid_grid_bz_d_z[[i_rz, i_r]] = 0.0;
                }
            }
        }

        // Store values
        results.get_or_insert("greens").get_or_insert("grid_grid").insert("br", g_grid_grid_br); // Array2<f64>; shape = (n_z * n_r, n_r)
        results.get_or_insert("greens").get_or_insert("grid_grid").insert("bz", g_grid_grid_bz); // Array2<f64>; shape = (n_z * n_r, n_r)
        results.get_or_insert("greens").get_or_insert("grid_grid").insert("psi", g_grid_grid_psi); // Array2<f64>; shape = (n_z * n_r, n_r)
        results
            .get_or_insert("greens")
            .get_or_insert("grid_grid")
            .insert("d_br_d_z", g_d_grid_grid_br_d_z); // Array2<f64>; shape = (n_z * n_r, n_r)
        results
            .get_or_insert("greens")
            .get_or_insert("grid_grid")
            .insert("d_bz_d_z", g_d_grid_grid_bz_d_z); // Array2<f64>; shape = (n_z * n_r, n_r)
        results.get_or_insert("grid").insert("d_area", d_area); // f64
        results.get_or_insert("grid").get_or_insert("flat").insert("r", flat_r); // Array1<f64>; shape = (n_z * n_r)
        results.get_or_insert("grid").get_or_insert("flat").insert("z", flat_z); // Array1<f64>; shape = (n_z * n_r)
        results.get_or_insert("grid").get_or_insert("mesh").insert("r", mesh_r); // Array2<f64>; shape = (n_z,  n_r)
        results.get_or_insert("grid").get_or_insert("mesh").insert("z", mesh_z); // Array2<f64>; shape = (n_z,  n_r)
        results.get_or_insert("grid").insert("r", r); // Array1<f64>; shape = (n_r)
        results.get_or_insert("grid").insert("z", z); // Array1<f64>; shape = (n_z)
        results.get_or_insert("grid").insert("n_r", n_r); // usize
        results.get_or_insert("grid").insert("n_z", n_z); // usize
        results.get_or_insert("limiter").get_or_insert("limit_pts").insert("r", limit_pts_r_ndarray); // Array1<f64>; shape = (n_limit_pts)
        results.get_or_insert("limiter").get_or_insert("limit_pts").insert("z", limit_pts_z_ndarray); // Array1<f64>; shape = (n_limit_pts)
        results.get_or_insert("vessel").insert("r", vessel_r_ndarray); // Array1<f64>; shape = (n_vessel_pts)
        results.get_or_insert("vessel").insert("z", vessel_z_ndarray); // Array1<f64>; shape = (n_vessel_pts)
        results.get_or_insert("profiles").insert("psi_n", psi_n_ndarray); // Array1<f64>; shape = (n_psi_n)

        Self {
            results,
            p_prime_source_function: p_prime_source_function_arc,
            ff_prime_source_function: ff_prime_source_function_arc,
        }
    }

    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python types into Rust types
        let coils_local: &Coils = &*coils;

        // Get variables out of self
        let flat_r: Array1<f64> = self.results.get("grid").get("flat").get("r").unwrap_array1();
        let flat_z: Array1<f64> = self.results.get("grid").get("flat").get("z").unwrap_array1();
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        for coil_name in coils_local.results.get("pf").keys() {
            // Coils
            let coil_r: Array1<f64> = coils_local.results.get("pf").get(&coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z: Array1<f64> = coils_local.results.get("pf").get(&coil_name).get("geometry").get("z").unwrap_array1();
            let d_r: Array1<f64> = &coil_r * 0.0;
            let d_z: Array1<f64> = &coil_z * 0.0;

            // Greens function for flux
            let g_grid_coil_all_filaments: Array2<f64> = greens(
                flat_r.clone(),
                flat_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
                d_r.clone(), // TODO: can be improved
                d_z.clone(),
            ); // shape = (n_z * n_r, n_filaments)

            // sum over all filaments and convert into shape = (n_z, n_r)
            let g_grid_coil: Array2<f64> = g_grid_coil_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape array into (n_z, n_r)")
                .to_owned();

            // Greens function for d_psi_d_z
            // (needed for calculating correction to psi from the vertical sabilisation "delta_z")
            let (g_br_grid_coil_all_filaments, _g_br_grid_coil_all_filaments): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
                flat_r.clone(), // "sensors"
                flat_z.clone(),
                coil_r.clone(), // "current sources"
                coil_z.clone(),
            ); // shape = (n_r * n_z, n_filaments)

            // d_psi_d_z = -2 * pi * r * br (same equation as in "flux_loops.rs")
            // sum over all filaments and convert into shape = (n_z, n_r)
            let g_d_psi_d_z_coil_tmp: Array1<f64> = -2.0 * PI * flat_r.clone() * (g_br_grid_coil_all_filaments.sum_axis(Axis(1)));
            let g_d_psi_d_z_coil: Array2<f64> = g_d_psi_d_z_coil_tmp
                .to_shape((n_z, n_r))
                .expect("Failed to reshape array into (n_z, n_r)")
                .to_owned();

            // Greens function for br and bz
            let (g_br_grid_coil_all_filaments, g_bz_grid_coil_all_filaments): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
                // shape = (n_z * n_r, n_filaments)
                flat_r.clone(),
                flat_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
            );

            // Greens function for d_br_d_z and d_bz_d_z
            let (g_d_br_grid_coil_all_filaments_d_z, g_d_bz_grid_coil_all_filaments_d_z): (Array2<f64>, Array2<f64>) = d_greens_magnetic_field_dz(
                // shape = (n_z * n_r, n_filaments)
                flat_r.clone(),
                flat_z.clone(),
                coil_r,
                coil_z,
            );

            // sum over all filaments and convert into shape = (n_z, n_r)
            let g_br_grid_coil: Array2<f64> = g_br_grid_coil_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils.g_br_grid_coil: Failed to reshape array into (n_z, n_r)")
                .to_owned();
            let g_bz_grid_coil: Array2<f64> = g_bz_grid_coil_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils.g_bz_grid_coil: Failed to reshape array into (n_z, n_r)")
                .to_owned();
            let g_d_br_grid_coil_d_z: Array2<f64> = g_d_br_grid_coil_all_filaments_d_z
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils.g_d_br_grid_coil_d_z: Failed to reshape array into (n_z, n_r)")
                .to_owned();
            let g_d_bz_grid_coil_d_z: Array2<f64> = g_d_bz_grid_coil_all_filaments_d_z
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils.g_d_bz_grid_coil_d_z: Failed to reshape array into (n_z, n_r)")
                .to_owned();

            // Store results
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("br", g_br_grid_coil); // Array2<f64>; shape = (n_z, n_r)
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("bz", g_bz_grid_coil); // Array2<f64>; shape = (n_z, n_r)
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("d_br_d_z", g_d_br_grid_coil_d_z); // Array2<f64>; shape = (n_z, n_r)
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("d_bz_d_z", g_d_bz_grid_coil_d_z); // Array2<f64>; shape = (n_z, n_r)
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("psi", g_grid_coil); // Array2<f64>; shape = (n_z, n_r)
            self.results
                .get_or_insert("greens")
                .get_or_insert("pf")
                .get_or_insert(&coil_name)
                .insert("d_psi_d_z", g_d_psi_d_z_coil); // Array2<f64>; shape = (n_z, n_r)
        }
    }

    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python types into Rust types
        let passives_local: &Passives = &*passives;

        // Get variables out of self
        let flat_r: Array1<f64> = self.results.get("grid").get("flat").get("r").unwrap_array1();
        let flat_z: Array1<f64> = self.results.get("grid").get("flat").get("z").unwrap_array1();

        // Calculate Greens with each passive degree of freedom
        // let passive_names: Vec<String> = ;
        for passive_name in passives_local.results.keys() {
            let _tmp: NestedDictAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
            let dof_names: Vec<String> = _tmp.keys();
            let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
            let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

            for dof_name in dof_names {
                // Current distribution
                let current_distribution: Array1<f64> = passives
                    .results
                    .get(&passive_name)
                    .get("dof")
                    .get(&dof_name)
                    .get("current_distribution")
                    .unwrap_array1();

                // Green's table
                let g_filaments: Array2<f64> = greens(
                    flat_r.clone(), // by convention (r, z) are "sensors"
                    flat_z.clone(),
                    passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
                    passive_z.clone(),
                    flat_r.clone() * 0.0, // d_r=0; as there will not be any points which coincide
                    flat_r.clone() * 0.0, // d_z=0; as there will not be any points which coincide
                );

                // Green's with degrees of freedom
                let g_filaments_with_dof: Array2<f64> = g_filaments * &current_distribution; // shape = [n_r * n_z, n_filament]

                // Sum over all filaments
                let g_with_dof: Array1<f64> = g_filaments_with_dof.sum_axis(Axis(1)); // shape = [n_r * n_z]

                // Store
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("psi", g_with_dof);

                // Green's functions for BR and BZ
                let (g_br_filaments, g_bz_filaments): (Array2<f64>, Array2<f64>) = greens_magnetic_field(
                    flat_r.clone(), // by convention (r, z) are "sensors"
                    flat_z.clone(),
                    passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
                    passive_z.clone(),
                );

                // Green's functions for d_br_d_z and d_bz_d_z
                let (d_g_br_filaments_d_z, d_g_bz_filaments_d_z): (Array2<f64>, Array2<f64>) = d_greens_magnetic_field_dz(
                    flat_r.clone(), // by convention (r, z) are "sensors"
                    flat_z.clone(),
                    passive_r.clone(), // by convention (r_prime, z_prime) are "current sources"
                    passive_z.clone(),
                );

                // Apply the current_distribution
                let g_br_filaments_with_dof: Array2<f64> = &g_br_filaments * &current_distribution; // shape = [n_r * n_z]
                let g_bz_filaments_with_dof: Array2<f64> = g_bz_filaments * &current_distribution; // shape = [n_r * n_z]
                let d_g_br_filaments_with_dof_d_z: Array2<f64> = d_g_br_filaments_d_z * &current_distribution; // shape = [n_r * n_z]
                let d_g_bz_filaments_with_dof_d_z: Array2<f64> = d_g_bz_filaments_d_z * &current_distribution; // shape = [n_r * n_z]

                // Sum over all filaments
                let g_br: Array1<f64> = g_br_filaments_with_dof.sum_axis(Axis(1)); // shape = [n_r * n_z]
                let g_bz: Array1<f64> = g_bz_filaments_with_dof.sum_axis(Axis(1)); // shape = [n_r * n_z]
                let g_d_br_d_z: Array1<f64> = d_g_br_filaments_with_dof_d_z.sum_axis(Axis(1)); // shape = [n_r * n_z]
                let g_d_bz_d_z: Array1<f64> = d_g_bz_filaments_with_dof_d_z.sum_axis(Axis(1)); // shape = [n_r * n_z]

                // Store
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("br", g_br);
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("bz", g_bz);
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("d_br_d_z", g_d_br_d_z);
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("d_bz_d_z", g_d_bz_d_z);

                // >> d(psi)/d(z) (needed for calculating correction to psi from the vertical sabilisation "delta_z")
                // (needed for calculating correction to psi from the vertical sabilisation "delta_z")
                // d_psi_d_z = -2 * pi * r * br (same equation as in "flux_loops.rs")
                // apply current_distribtion and sum over all filaments
                let g_br_filaments_with_dof: Array2<f64> = g_br_filaments * &current_distribution; // shape = [n_r * n_z, n_filament]
                let g_d_psi_d_z_coil: Array1<f64> = -2.0 * PI * flat_r.clone() * (g_br_filaments_with_dof.sum_axis(Axis(1)));

                // Store
                self.results
                    .get_or_insert("greens")
                    .get_or_insert("passives")
                    .get_or_insert(&passive_name)
                    .get_or_insert(&dof_name)
                    .insert("d_psi_d_z", g_d_psi_d_z_coil); // shape = [n_r * n_z]
            }
        }
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Plasma>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();
        string_output += &format!("║  {:<74} ║\n", format!(" n_r = {}, n_z = {}", n_r.to_string(), n_z.to_string()));

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

    // /// Get the keys from results and return a Python list of strings
    // pub fn keys(&self, py: Python) -> Py<PyList> {
    //     let keys: Vec<String> = self.results.keys();
    //     let result: Py<PyList> = PyList::new(py, keys).unwrap().into();
    //     return result;
    // }

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
impl Plasma {
    pub fn get_greens_passive_grid(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("psi")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn get_d_psi_d_z_passive(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("d_psi_d_z")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn get_greens_passive_grid_br(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("br")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn get_greens_passive_grid_bz(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("bz")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn get_greens_passive_grid_d_br_d_z(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("d_br_d_z")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn get_greens_passive_grid_d_bz_d_z(&self) -> Array2<f64> {
        // Get grid sizes
        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Passives
        let passive_names: Vec<String> = self.results.get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_dof_total));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        let mut i_dof_total: usize = 0;
        for i_passive in 0..n_passives {
            let passive_name: &str = &passive_names[i_passive];
            let dof_names: Vec<String> = self.results.get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
            for dof_name in dof_names {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(
                    &self
                        .results
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .get("d_bz_d_z")
                        .unwrap_array1(),
                );

                // Keep count
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    pub fn equilibrium_post_processor(&mut self, gs_solutions: &mut Vec<GsSolution>, coils: &Coils, plasma: &Plasma) {
        println!("equilibrium_post_processor: starting");

        let n_time: usize = gs_solutions.len();
        let psi_n: Array1<f64> = self.results.get("profiles").get("psi_n").unwrap_array1();
        let n_psi_n: usize = psi_n.len();

        let n_r: usize = self.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = self.results.get("grid").get("n_z").unwrap_usize();

        // Get the mesh (note, the [0] is because the mesh is the same for all time slices)
        let time: Array1<f64> = plasma.results.get("time").unwrap_array1();
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let r_mesh: Array2<f64> = plasma.results.get("grid").get("mesh").get("r").unwrap_array2();
        let z_mesh: Array2<f64> = plasma.results.get("grid").get("mesh").get("z").unwrap_array2();
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();

        // Allocate arrays for results which we already have in `gs_solutions`
        // Two-d
        let mut j_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut psi_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut psi_n_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        // let mut psi_2d_coils: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut br_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut bz_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut mask_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        // Fit values
        let n_p_prime: usize = plasma.p_prime_source_function.source_function_n_dof();
        let n_ff_prime: usize = plasma.ff_prime_source_function.source_function_n_dof();
        let mut p_prime_dof_values: Array2<f64> = Array2::from_elem((n_time, n_p_prime), f64::NAN);
        let mut ff_prime_dof_values: Array2<f64> = Array2::from_elem((n_time, n_ff_prime), f64::NAN);
        // Global quantities
        // Boundary point
        let mut bounding_r: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut bounding_z: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Axis poloidal flux
        let mut psi_a: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Boundary poloidal flux
        let mut psi_b: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Radial current centroid
        let mut r_cur: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Vertical current centroid
        let mut z_cur: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Plasma current
        let mut ip: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Vertical displacement
        let mut delta_z: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Minor radius
        let mut r_minor: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Geometric radius
        let mut r_geo: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Geometric height
        let mut z_geo: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Elongation
        let mut elongation: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // X-points
        let mut xpt_upper_r: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut xpt_upper_z: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut xpt_lower_r: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut xpt_lower_z: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Number of iterations
        let mut n_iter: Vec<usize> = Vec::with_capacity(n_time); // Could also have been Array1<usize> ?
        let mut gs_error: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Magnetic axis
        let mut r_mag: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut z_mag: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        // Total toroidal flux
        let mut flux_tor: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

        let mut xpt_diverted: Vec<bool> = Vec::with_capacity(n_time);

        let mut boundary_nbnd: Vec<usize> = Vec::with_capacity(n_time);
        let mut vec_array_boundary_r: Vec<Array1<f64>> = Vec::with_capacity(n_time);
        let mut vec_array_boundary_z: Vec<Array1<f64>> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            let tmp_boundary_r: Array1<f64>;
            let tmp_boundary_z: Array1<f64>;
            if gs_solutions[i_time].xpt_diverted {
                (tmp_boundary_r, tmp_boundary_z) = epp_clean_boundary(&gs_solutions[i_time]);
                // Reassign the boundary to the GS solution
                // TODO: how slow is this? could I always do this in the iteration?
                gs_solutions[i_time].boundary_r = tmp_boundary_r.clone();
                gs_solutions[i_time].boundary_z = tmp_boundary_z.clone();
            } else {
                tmp_boundary_r = gs_solutions[i_time].boundary_r.clone();
                tmp_boundary_z = gs_solutions[i_time].boundary_z.clone();
            }

            boundary_nbnd.push(tmp_boundary_r.len());
            vec_array_boundary_r.push(tmp_boundary_r);
            vec_array_boundary_z.push(tmp_boundary_z);
        }

        let max_n_boundary: usize = boundary_nbnd.iter().max().unwrap().clone();

        let mut boundary_r: Array2<f64> = Array2::from_elem((n_time, max_n_boundary), f64::NAN);
        let mut boundary_z: Array2<f64> = Array2::from_elem((n_time, max_n_boundary), f64::NAN);

        for i_time in 0..n_time {
            boundary_r.slice_mut(s![i_time, 0..boundary_nbnd[i_time]]).assign(&vec_array_boundary_r[i_time]);
            boundary_z.slice_mut(s![i_time, 0..boundary_nbnd[i_time]]).assign(&vec_array_boundary_z[i_time]);
        }

        // Loop over time, and perform post-processing on `gs_solutions`
        let mut p_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut bt_2d: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
        let mut w_mhd: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut plasma_volume: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut beta_n: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut beta_p_1: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut beta_p_2: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut beta_p_3: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut beta_t: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut bt_vac_at_r_geo: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut li_1: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut li_2: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut li_3: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut p_1d: Array1<f64> = Array1::from_elem(n_time, f64::NAN); // total pressure
        let mut q0: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut q95: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        let mut f_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut ff_prime_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut p_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut p_prime_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut psi_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut mid_plane_p_profile: Array2<f64> = Array2::from_elem((n_time, n_r), f64::NAN);
        let mut area_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut area_prime_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut volume_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut volume_prime_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut flux_tor_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut q_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut rho_tor_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);
        let mut rho_pol_profile: Array2<f64> = Array2::from_elem((n_time, n_psi_n), f64::NAN);

        let i_rod: Array1<f64> = coils.results.get("tf").get("rod_i").get("measured").unwrap_array1();

        'time_loop: for i_time in 0..n_time {
            // Skip time-slices which didn't converge
            if gs_solutions[i_time].psi_a.is_nan() {
                continue 'time_loop;
            }

            // Two-d
            br_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].br_2d);
            bz_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].bz_2d);
            j_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].j_2d);
            mask_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].mask);
            psi_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].psi_2d);
            psi_n_2d.slice_mut(s![i_time, .., ..]).assign(&gs_solutions[i_time].psi_n_2d);

            // Fit values
            p_prime_dof_values.slice_mut(s![i_time, ..]).assign(&gs_solutions[i_time].p_prime_dof_values);
            ff_prime_dof_values.slice_mut(s![i_time, ..]).assign(&gs_solutions[i_time].ff_prime_dof_values);

            // Global
            bounding_r[i_time] = gs_solutions[i_time].bounding_r;
            bounding_z[i_time] = gs_solutions[i_time].bounding_z;
            psi_a[i_time] = gs_solutions[i_time].psi_a;
            psi_b[i_time] = gs_solutions[i_time].psi_b;
            ip[i_time] = gs_solutions[i_time].ip;
            delta_z[i_time] = gs_solutions[i_time].delta_z;
            r_cur[i_time] = d_area * (&r_mesh * &gs_solutions[i_time].j_2d).sum() / ip[i_time];
            z_cur[i_time] = d_area * (&z_mesh * &gs_solutions[i_time].j_2d).sum() / ip[i_time];
            // Minor radius
            r_minor[i_time] = (gs_solutions[i_time].boundary_r.max().unwrap().to_owned() - gs_solutions[i_time].boundary_r.min().unwrap().to_owned()) / 2.0;
            // Geometric radius
            r_geo[i_time] = (gs_solutions[i_time].boundary_r.max().unwrap().to_owned() + gs_solutions[i_time].boundary_r.min().unwrap().to_owned()) / 2.0;
            // Geometric radius
            z_geo[i_time] = (gs_solutions[i_time].boundary_z.max().unwrap().to_owned() + gs_solutions[i_time].boundary_z.min().unwrap().to_owned()) / 2.0;
            // elongation
            let plasma_height: f64 = gs_solutions[i_time].boundary_z.max().unwrap().to_owned() - gs_solutions[i_time].boundary_z.min().unwrap().to_owned();
            elongation[i_time] = plasma_height / (2.0 * r_minor[i_time]);

            // diverted
            xpt_diverted.push(gs_solutions[i_time].xpt_diverted);

            // x-points
            xpt_upper_r[i_time] = gs_solutions[i_time].xpt_upper_r;
            xpt_upper_z[i_time] = gs_solutions[i_time].xpt_upper_z;
            xpt_lower_r[i_time] = gs_solutions[i_time].xpt_lower_r;
            xpt_lower_z[i_time] = gs_solutions[i_time].xpt_lower_z;

            // Number of iterations
            n_iter.push(gs_solutions[i_time].n_iter);
            gs_error[i_time] = gs_solutions[i_time].gs_error_calculated;

            // Magnetic axis
            r_mag[i_time] = gs_solutions[i_time].r_mag;
            z_mag[i_time] = gs_solutions[i_time].z_mag;

            // Pressure on 2D grid
            let p_2d_this_time_slice: Array2<f64> = epp_p_2d(&gs_solutions[i_time], &r, &z);
            p_2d.slice_mut(s![i_time, .., ..]).assign(&p_2d_this_time_slice);

            // Stored energy
            w_mhd[i_time] = epp_w_mhd(&p_2d_this_time_slice, &r, d_area);

            plasma_volume[i_time] = epp_plasma_volume(&gs_solutions[i_time], r_geo[i_time]);

            // Plasma beta
            (beta_p_1[i_time], beta_p_2[i_time], beta_p_3[i_time]) = epp_beta_p(w_mhd[i_time], ip[i_time], r_mag[i_time], r_geo[i_time], plasma_volume[i_time]);

            let bt_vac_at_r_geo_this_time: f64 = epp_bt_vac_at_r_geo(i_rod[i_time], r_geo[i_time]);
            bt_vac_at_r_geo[i_time] = bt_vac_at_r_geo_this_time;

            // Internal inductance: TODO: add li(1)
            (li_1[i_time], li_2[i_time], li_3[i_time]) = epp_li(
                ip[i_time],
                &r,
                d_area,
                r_mag[i_time],
                r_geo[i_time],
                &br_2d.slice(s![i_time, .., ..]).to_owned(),
                &bz_2d.slice(s![i_time, .., ..]).to_owned(),
                &mask_2d.slice(s![i_time, .., ..]).to_owned(),
            );

            let beta_t_this_time_slice: f64 = epp_beta(w_mhd[i_time], bt_vac_at_r_geo_this_time, plasma_volume[i_time]);
            beta_t[i_time] = beta_t_this_time_slice;

            let beta_n_this_time_slice: f64 = epp_beta_n(beta_t_this_time_slice, r_minor[i_time], bt_vac_at_r_geo[i_time], ip[i_time]);
            beta_n[i_time] = beta_n_this_time_slice;

            // total pressure
            // TODO: it "could" be better to do integral over flux surfaces ?
            p_1d[i_time] = p_2d.slice(s![i_time, .., ..]).sum();

            // Profiles
            let f_profile_local: Array1<f64> = epp_f_profile(&gs_solutions[i_time], &psi_n, psi_a[i_time], psi_b[i_time], i_rod[i_time]);
            f_profile.slice_mut(s![i_time, ..]).assign(&f_profile_local);

            let ff_prime_profile_local: Array1<f64> = epp_ff_prime_profile(&gs_solutions[i_time], &psi_n);
            ff_prime_profile.slice_mut(s![i_time, ..]).assign(&ff_prime_profile_local);

            let p_profile_local: Array1<f64> = epp_p_profile(&gs_solutions[i_time], &psi_n, psi_a[i_time], psi_b[i_time]);
            p_profile.slice_mut(s![i_time, ..]).assign(&p_profile_local);

            let p_prime_profile_this_time: Array1<f64> = epp_p_prime_profile(&gs_solutions[i_time], &psi_n);
            p_prime_profile.slice_mut(s![i_time, ..]).assign(&p_prime_profile_this_time);

            let psi_profile_this_time: Array1<f64> = &psi_n * (psi_b[i_time] - psi_a[i_time]) + psi_a[i_time];
            let d_psi: f64 = psi_profile_this_time[1] - psi_profile_this_time[0];
            psi_profile.slice_mut(s![i_time, ..]).assign(&psi_profile_this_time);

            // Mid-plane profiles
            let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
            let mid_plane_p_profile_this_time: Array1<f64> = epp_mid_plane_p_profile(
                &gs_solutions[i_time],
                &r,
                i_z_centre,
                psi_a[i_time],
                psi_b[i_time],
                &psi_n_2d.slice(s![i_time, .., ..]).to_owned(),
                &mask_2d.slice(s![i_time, .., ..]).to_owned(),
            );
            mid_plane_p_profile.slice_mut(s![i_time, ..]).assign(&mid_plane_p_profile_this_time);

            let (bt_2d_this_time, bt_vac_this_time): (Array2<f64>, Array2<f64>) = epp_bt_2d(&gs_solutions[i_time], &r, &z, i_rod[i_time]);
            bt_2d.slice_mut(s![i_time, .., ..]).assign(&bt_2d_this_time);

            let (volume_profile_this_time, volume_prime_profile_this_time, area_profile_this_time, area_prime_profile_this_time): (
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
            ) = epp_vol_profile(&gs_solutions[i_time], &psi_n, &r, &z, d_psi);
            area_profile.slice_mut(s![i_time, ..]).assign(&area_profile_this_time);
            area_prime_profile.slice_mut(s![i_time, ..]).assign(&area_prime_profile_this_time);
            volume_profile.slice_mut(s![i_time, ..]).assign(&volume_profile_this_time);
            volume_prime_profile.slice_mut(s![i_time, ..]).assign(&volume_prime_profile_this_time);

            let flux_surfaces: Vec<FluxSurface> = epp_flux_surfaces(&gs_solutions[i_time], &psi_n, &r, &z, d_psi);

            let q_profile_this_time: Array1<f64> = epp_q_profile(&gs_solutions[i_time], &flux_surfaces, &f_profile_local, &r, &z);
            q_profile.slice_mut(s![i_time, ..]).assign(&q_profile_this_time);

            let flux_tor_profile_this_time_slice: Array1<f64> = epp_flux_toroidal_profile(&q_profile_this_time, &psi_profile_this_time);
            flux_tor_profile.slice_mut(s![i_time, ..]).assign(&flux_tor_profile_this_time_slice);

            // TODO: this is **VERY** hacky, and **SHOULD** be improved!!
            // set f_profile to the vacuum profile, then calculate the vacuum q-profile, then the vacuum toroidal flux
            let f_profile_vacuum: Array1<f64> = 0.0 * &f_profile_local + MU_0 * i_rod[i_time] / (2.0 * PI);
            let q_profile_vacuum: Array1<f64> = epp_q_profile(&gs_solutions[i_time], &flux_surfaces, &f_profile_vacuum, &r, &z);
            let flux_tor_profile_vacuum: Array1<f64> = epp_flux_toroidal_profile(&q_profile_vacuum, &psi_profile_this_time);
            flux_tor[i_time] = flux_tor_profile_this_time_slice.last().unwrap().to_owned() - flux_tor_profile_vacuum.last().unwrap().to_owned();

            let rho_tor_profile_this_time_slice: Array1<f64> = epp_rho_tor_profile(&flux_tor_profile_this_time_slice);
            rho_tor_profile.slice_mut(s![i_time, ..]).assign(&rho_tor_profile_this_time_slice);

            let q95_this_time: f64 = epp_q95(&q_profile_this_time, &psi_n);
            q95[i_time] = q95_this_time;

            q0[i_time] = q_profile_this_time[0];

            rho_pol_profile.slice_mut(s![i_time, ..]).assign(&psi_n.clone().mapv(|x| x.sqrt()));
        }

        let v_loop: Array1<f64> = epp_v_loop(&psi_b, &time);

        // Do the assignments
        // Global
        self.results.get_or_insert("global").insert("beta_n", beta_n);
        self.results.get_or_insert("global").insert("beta_p_1", beta_p_1);
        self.results.get_or_insert("global").insert("beta_p_2", beta_p_2);
        self.results.get_or_insert("global").insert("beta_p_3", beta_p_3);
        self.results.get_or_insert("global").insert("beta_t", beta_t);
        self.results.get_or_insert("global").insert("bt_vac_at_r_geo", bt_vac_at_r_geo);
        self.results.get_or_insert("global").insert("elongation", elongation);
        self.results.get_or_insert("global").insert("gs_error", gs_error);
        self.results.get_or_insert("global").insert("i_rod", i_rod);
        self.results.get_or_insert("global").insert("ip", ip);
        self.results.get_or_insert("global").insert("psi_a", psi_a);
        self.results.get_or_insert("global").insert("psi_b", psi_b);
        self.results.get_or_insert("global").insert("delta_z", delta_z);
        self.results.get_or_insert("global").insert("li_1", li_1);
        self.results.get_or_insert("global").insert("li_2", li_2);
        self.results.get_or_insert("global").insert("li_3", li_3);
        self.results.get_or_insert("global").insert("p", p_1d);
        self.results.get_or_insert("global").insert("q0", q0);
        self.results.get_or_insert("global").insert("q95", q95);
        self.results.get_or_insert("global").insert("r_minor", r_minor);
        self.results.get_or_insert("global").insert("r_cur", r_cur);
        self.results.get_or_insert("global").insert("z_cur", z_cur);
        self.results.get_or_insert("global").insert("r_geo", r_geo);
        self.results.get_or_insert("global").insert("z_geo", z_geo);
        self.results.get_or_insert("global").insert("r_mag", r_mag);
        self.results.get_or_insert("global").insert("z_mag", z_mag);
        self.results.get_or_insert("global").insert("phi_dia", flux_tor);
        self.results.get_or_insert("global").insert("n_iter", n_iter);
        self.results.get_or_insert("global").insert("plasma_volume", plasma_volume);
        self.results.get_or_insert("global").insert("v_loop", v_loop);
        self.results.get_or_insert("global").insert("w_mhd", w_mhd);
        self.results.get_or_insert("global").insert("xpt_diverted", xpt_diverted);

        // Plasma boundary
        self.results.get_or_insert("p_boundary").insert("bounding_r", bounding_r);
        self.results.get_or_insert("p_boundary").insert("bounding_z", bounding_z);
        self.results.get_or_insert("p_boundary").insert("nbnd", boundary_nbnd);
        self.results.get_or_insert("p_boundary").insert("rbnd", boundary_r);
        self.results.get_or_insert("p_boundary").insert("zbnd", boundary_z);

        // Profiles (psi_n is already inside "profiles")
        self.results.get_or_insert("profiles").insert("area", area_profile);
        self.results.get_or_insert("profiles").insert("area_prime", area_prime_profile);
        self.results.get_or_insert("profiles").insert("f", f_profile);
        self.results.get_or_insert("profiles").insert("ff_prime", ff_prime_profile);
        self.results.get_or_insert("profiles").insert("flux_tor", flux_tor_profile);
        self.results.get_or_insert("profiles").insert("p", p_profile);
        self.results.get_or_insert("profiles").insert("p_prime", p_prime_profile);
        self.results.get_or_insert("profiles").insert("psi", psi_profile);
        self.results.get_or_insert("profiles").insert("q", q_profile);
        self.results.get_or_insert("profiles").insert("rho_pol", rho_pol_profile);
        self.results.get_or_insert("profiles").insert("rho_tor", rho_tor_profile);
        self.results.get_or_insert("profiles").insert("vol", volume_profile);
        self.results.get_or_insert("profiles").insert("vol_prime", volume_prime_profile);

        // Mid-plane profiles
        self.results
            .get_or_insert("profiles")
            .get_or_insert("mid_plane")
            .insert("p", mid_plane_p_profile);
        self.results.get_or_insert("profiles").get_or_insert("mid_plane").insert("r", r.clone());

        // Source functions
        self.results
            .get_or_insert("source_functions")
            .get_or_insert("ff_prime")
            .insert("coefficients", ff_prime_dof_values);
        self.results
            .get_or_insert("source_functions")
            .get_or_insert("p_prime")
            .insert("coefficients", p_prime_dof_values);

        // Two-d
        self.results.get_or_insert("two_d").insert("br", br_2d);
        self.results.get_or_insert("two_d").insert("bt", bt_2d);
        self.results.get_or_insert("two_d").insert("bz", bz_2d);
        self.results.get_or_insert("two_d").insert("j", j_2d);
        self.results.get_or_insert("two_d").insert("mask", mask_2d);
        self.results.get_or_insert("two_d").insert("p", p_2d.clone());
        self.results.get_or_insert("two_d").insert("psi", psi_2d);
        self.results.get_or_insert("two_d").insert("psi_n", psi_n_2d);

        // x-points
        self.results.get_or_insert("xpoints").get_or_insert("upper").insert("r", xpt_upper_r);
        self.results.get_or_insert("xpoints").get_or_insert("upper").insert("z", xpt_upper_z);
        self.results.get_or_insert("xpoints").get_or_insert("lower").insert("r", xpt_lower_r);
        self.results.get_or_insert("xpoints").get_or_insert("lower").insert("z", xpt_lower_z);
    }
}

fn epp_v_loop(psi_b: &Array1<f64>, time: &Array1<f64>) -> Array1<f64> {
    // v_loop = - d(psi_b)/d(time)

    // Note: when the time-slice is "user_defined", the time-vecor can have variable time steps
    let n_time: usize = time.len();
    let mut v_loop: Array1<f64> = Array1::from_elem(psi_b.len(), f64::NAN);

    // Exit if we only have one time-slice
    if n_time == 1 {
        return v_loop;
    }

    // forward/backward differences for the first time point
    v_loop[0] = -(psi_b[1] - psi_b[0]) / (time[1] - time[0]);
    // Central differencing for the rest
    for i_time in 1..n_time - 1 {
        let d_psi_b: f64 = -(psi_b[i_time + 1] - psi_b[i_time - 1]);
        let d_time: f64 = time[i_time + 1] - time[i_time - 1];
        v_loop[i_time] = d_psi_b / d_time;
    }
    // forward/backward difference for the last time point
    v_loop[n_time - 1] = -(psi_b[n_time - 1] - psi_b[n_time - 2]) / (time[n_time - 1] - time[n_time - 2]);

    return v_loop;
}

fn epp_beta(w_mhd: f64, bt_vac_at_r_geo: f64, plasma_volume: f64) -> f64 {
    let p_vol_int: f64 = (2.0 / 3.0) * w_mhd;
    let p_vol_avg: f64 = p_vol_int / plasma_volume;

    let beta_t: f64 = 2.0 * MU_0 * p_vol_avg * 100.0 / bt_vac_at_r_geo.powi(2);

    return beta_t;
}

fn epp_beta_n(beta: f64, r_minor: f64, bt_vac_at_r_geo: f64, ip: f64) -> f64 {
    let beta_n: f64 = beta * r_minor * bt_vac_at_r_geo / (ip / 1e6);
    return beta_n;
}

fn epp_beta_p(w_mhd: f64, ip: f64, r_mag: f64, r_geo: f64, plasma_volume: f64) -> (f64, f64, f64) {
    let p_vol_int: f64 = w_mhd * 2.0 / 3.0;

    let beta_p_1: f64 = f64::NAN;

    let beta_p_2: f64 = 4.0 * p_vol_int / (MU_0 * ip * ip * r_mag);

    let beta_p_3: f64 = 4.0 * p_vol_int / (MU_0 * ip * ip * r_geo);

    return (beta_p_1, beta_p_2, beta_p_3);
}

fn epp_bt_2d(gs_solution: &GsSolution, r: &Array1<f64>, z: &Array1<f64>, i_rod: f64) -> (Array2<f64>, Array2<f64>) {
    let n_r: usize = r.len();
    let n_z: usize = z.len();

    let ff_prime_dof_values: Array1<f64> = gs_solution.ff_prime_dof_values.to_owned();
    let psi_a: f64 = gs_solution.psi_a;
    let psi_b: f64 = gs_solution.psi_b;
    let mask: Array2<f64> = gs_solution.mask.to_owned(); // shape = (n_z, n_r)
    let psi_n_2d: Array2<f64> = gs_solution.psi_n_2d.to_owned(); // shape = (n_z, n_r)

    let mut bt_2d_now: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut bt_vac_2d_now: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);

    // BT vacuum
    let bt_vac_vs_r: Array1<f64> = MU_0 * i_rod / (2.0 * PI * r);
    for i_z in 0..n_z {
        bt_2d_now.slice_mut(s![i_z, ..]).assign(&bt_vac_vs_r);
        bt_vac_2d_now.slice_mut(s![i_z, ..]).assign(&bt_vac_vs_r);
    }

    // Outside the plasma:
    // BT(R) = i_rod * MU_0 / (2 * PI * R)
    // Inside the plasma:
    // BT(R) = f/R
    // where `f` comes from the integral of `ff_prime`
    // f = 1/2*int(ff_prime) + constant_of_integration
    // `constant_of_integration` set so that BT at plasma boundary = vacuum BT
    let f_boundary: f64 = i_rod * MU_0 / (2.0 * PI);

    // d(psi)/d(psi_n)
    let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            if mask[[i_z, i_r]] > 0.99 {
                // Integrate the source function
                let f_unnormalise: f64 = gs_solution
                    .ff_prime_source_function
                    .source_function_integral(&Array1::from_vec(vec![psi_n_2d[[i_z, i_r]]]), &ff_prime_dof_values)[0];

                // Calculate the poloidal flux function, f
                let f_at_this_rz: f64 = f_unnormalise / d_psi_d_psi_n + f_boundary;

                // Toroidal field
                bt_2d_now[[i_z, i_r]] = f_at_this_rz / r[i_r];
            }
        }
    }

    return (bt_2d_now, bt_vac_2d_now);
}

fn epp_bt_vac_at_r_geo(i_rod: f64, r_geo: f64) -> f64 {
    let bt_vac_at_r_geo: f64 = MU_0 * i_rod / (2.0 * PI * r_geo);
    return bt_vac_at_r_geo;
}

fn epp_clean_boundary(gs_solution: &GsSolution) -> (Array1<f64>, Array1<f64>) {
    let boundary_r: Array1<f64> = gs_solution.boundary_r.to_owned();
    let boundary_z: Array1<f64> = gs_solution.boundary_z.to_owned();
    let mag_axis_r: f64 = gs_solution.r_mag;
    let mag_axis_z: f64 = gs_solution.z_mag;
    let xpt_r: f64 = gs_solution.bounding_r;
    let xpt_z: f64 = gs_solution.bounding_z;

    // Draw a horizontal line, starting at the magnetic axis, ending +1m away on the LFS
    // TODO: improve the +1m, this could fail for large tokamaks!!
    let line_along_mag_axis: Line = Line::new((mag_axis_r, mag_axis_z), (mag_axis_r + 1.0, mag_axis_z)); // Horizontal line

    // Find outboard mid-plane
    // Find the intersection of the boundary with the horizontal line
    // let mut intersection_points: Vec<(f64, f64)> = Vec::new();
    let mut index_outboard: usize = 0;
    let n_points: usize = boundary_r.len();
    for i_point in 0..n_points - 1 {
        let r1 = boundary_r[i_point];
        let z1 = boundary_z[i_point];
        let r2 = boundary_r[i_point + 1];
        let z2 = boundary_z[i_point + 1];

        // Use the geo crate to calculate the intersection
        let boundary_line: Line = Line::new((r1, z1), (r2, z2));

        // this only "unwraps" when the lines intersect!
        if let Some(intersection_between_boundary_and_horizontal_line) = line_intersection(boundary_line, line_along_mag_axis) {
            if let LineIntersection::SinglePoint { intersection, .. } = intersection_between_boundary_and_horizontal_line {
                let r: f64 = intersection.x;
                if r > mag_axis_r {
                    index_outboard = i_point;
                    break;
                }
            }
        }
    }

    // Find the point nearest to the x-point
    // TODO: WRONG! this should be either above or below the x-point, depending on the x-point z-sign
    let mut nearest_distance: f64 = f64::MAX;
    // let mut index_nearest_to_xpt: usize = 0;
    let n_points: usize = boundary_r.len();
    let mut point_nearest_to_xpt_r: f64 = f64::NAN;
    let mut point_nearest_to_xpt_z: f64 = f64::NAN;
    for i_point in 0..n_points {
        let r: f64 = boundary_r[i_point];
        let z: f64 = boundary_z[i_point];
        let distance: f64 = ((r - xpt_r).powi(2) + (z - xpt_z).powi(2)).sqrt();
        if distance < nearest_distance {
            nearest_distance = distance;
            point_nearest_to_xpt_r = r;
            point_nearest_to_xpt_z = z;
        }
    }

    // Go in one direction to the x-point
    let mut tmp_r: Vec<f64> = boundary_r.clone().to_vec();
    let mut tmp_z: Vec<f64> = boundary_z.clone().to_vec();
    let mut final_direction_1_r: Vec<f64> = Vec::new();
    let mut final_direction_1_z: Vec<f64> = Vec::new();
    final_direction_1_r.push(tmp_r[index_outboard]);
    final_direction_1_z.push(tmp_z[index_outboard]);
    tmp_r.remove(index_outboard);
    tmp_z.remove(index_outboard);
    'looping_over_points: for _i_point in 0..n_points {
        // Find the index of the nearest point
        let mut nearest_distance: f64 = f64::MAX;
        let mut index: usize = 0;
        for i_point_min in 0..tmp_r.len() {
            let r: f64 = tmp_r[i_point_min];
            let z: f64 = tmp_z[i_point_min];
            let distance: f64 = ((r - final_direction_1_r.last().unwrap()).powi(2) + (z - final_direction_1_z.last().unwrap()).powi(2)).sqrt();
            if distance < nearest_distance {
                nearest_distance = distance;
                index = i_point_min;
            }
        }

        // Test if this point is the x-point
        if tmp_r.len() == 0 {
            // some sort of exception handeling??
            return (Array1::from_elem(0, f64::NAN), Array1::from_elem(0, f64::NAN));
        }
        if tmp_r[index] == point_nearest_to_xpt_r && tmp_z[index] == point_nearest_to_xpt_z {
            break 'looping_over_points;
        }

        // Add this point to the final list
        final_direction_1_r.push(tmp_r[index]);
        final_direction_1_z.push(tmp_z[index]);
        tmp_r.remove(index);
        tmp_z.remove(index);
    }

    // Go in other direction to the x-point
    let mut final_direction_2_r: Vec<f64> = Vec::new();
    let mut final_direction_2_z: Vec<f64> = Vec::new();
    final_direction_2_r.push(final_direction_1_r[0]);
    final_direction_2_z.push(final_direction_1_z[0]);
    'looping_over_points: for _i_point in 0..n_points {
        // Find the index of the nearest point
        let mut nearest_distance: f64 = f64::MAX;
        let mut index: usize = 0;
        for i_point_min in 0..tmp_r.len() {
            let r: f64 = tmp_r[i_point_min];
            let z: f64 = tmp_z[i_point_min];
            let distance: f64 = ((r - final_direction_2_r.last().unwrap()).powi(2) + (z - final_direction_2_z.last().unwrap()).powi(2)).sqrt();
            if distance < nearest_distance {
                nearest_distance = distance;
                index = i_point_min;
            }
        }

        // Test if this point is the x-point
        if tmp_r[index] == point_nearest_to_xpt_r && tmp_z[index] == point_nearest_to_xpt_z {
            break 'looping_over_points;
        }

        // Add this point to the final list
        final_direction_2_r.push(tmp_r[index]);
        final_direction_2_z.push(tmp_z[index]);
        tmp_r.remove(index);
        tmp_z.remove(index);
    }

    final_direction_2_r.reverse();
    final_direction_2_z.reverse();
    final_direction_1_r.extend(final_direction_2_r);
    final_direction_1_z.extend(final_direction_2_z);
    let boundary_r_clean: Array1<f64> = final_direction_1_r.into();
    let boundary_z_clean: Array1<f64> = final_direction_1_z.into();

    return (boundary_r_clean, boundary_z_clean);
}

fn epp_f_profile(gs_solution: &GsSolution, psi_n: &Array1<f64>, psi_a: f64, psi_b: f64, i_rod: f64) -> Array1<f64> {
    let n_psi_n: usize = psi_n.len();

    let mut f_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);

    let ff_prime_dof_values: Array1<f64> = gs_solution.ff_prime_dof_values.to_owned();

    // f(psi_n) = R * BT(R)
    // BT(R) = i_rod * MU_0 / (2 * PI * R)
    // f_boundary = i_rod * MU_0 / (2 * PI)
    // Constant of integration set so that BT at plasma boundary = vacuum BT
    let f_boundary: f64 = i_rod * MU_0 / (2.0 * PI);

    // d(psi)/d(psi_n)
    let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

    for i_psi_n in 0..n_psi_n {
        // Integrate the source function
        let f_local: f64 = gs_solution
            .ff_prime_source_function
            .source_function_integral(&Array1::from_vec(vec![psi_n[i_psi_n]]), &ff_prime_dof_values)[0];

        // Apply the mask, and store pressure
        f_profile[i_psi_n] = f_local / d_psi_d_psi_n + f_boundary;
    }

    return f_profile;
}

fn epp_ff_prime_profile(gs_solution: &GsSolution, psi_n: &Array1<f64>) -> Array1<f64> {
    let ff_prime_dof_values: Array1<f64> = gs_solution.ff_prime_dof_values.to_owned();
    let ff_prime_local: Array1<f64> = gs_solution.ff_prime_source_function.source_function_value(psi_n, &ff_prime_dof_values);
    return ff_prime_local;
}

fn epp_flux_surfaces(gs_solution: &GsSolution, psi_n: &Array1<f64>, r: &Array1<f64>, z: &Array1<f64>, d_psi: f64) -> Vec<FluxSurface> {
    // Sizes and grid variables
    let n_psi_n: usize = psi_n.len();
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    let flux_surface_empty = FluxSurface {
        r: Array1::from_elem(0, f64::NAN),
        z: Array1::from_elem(0, f64::NAN),
    };
    let mut flux_surfaces: Vec<FluxSurface> = vec![flux_surface_empty; n_psi_n];
    let mut volume_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    let mut area_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);

    // Create an empty contour grid
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true) // x dim., y dim., smoothing
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    let psi_2d: Array2<f64> = gs_solution.psi_2d.to_owned();
    let psi_2d_flattened: Vec<f64> = psi_2d.iter().cloned().collect();

    let psi_a: f64 = gs_solution.psi_a;
    let psi_b: f64 = gs_solution.psi_b;

    // Creat the plasma boundary polygon
    let boundary_r: Array1<f64> = gs_solution.boundary_r.to_owned();
    let boundary_z: Array1<f64> = gs_solution.boundary_z.to_owned();
    let n_boundary_points: usize = boundary_r.len();
    let mut boundary_polygon_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_boundary_points);
    for i_boundary_point in 0..n_boundary_points {
        boundary_polygon_coordinates.push(Coord {
            x: boundary_r[i_boundary_point],
            y: boundary_z[i_boundary_point],
        });
    }
    let boundary_polygon_coordinates: Vec<Coord<f64>> = boundary_r.iter().zip(boundary_z.iter()).map(|(&x, &y)| Coord { x, y }).collect();
    let boundary_polygon: Polygon = Polygon::new(
        LineString::from(boundary_polygon_coordinates),
        vec![], // No holes
    );

    // Add on the last closed flux surface
    let flux_surface_last_closed: FluxSurface = FluxSurface {
        r: boundary_r.clone(),
        z: boundary_z.clone(),
    };
    flux_surfaces[n_psi_n - 1] = flux_surface_last_closed;

    // Loop over psi_n
    // Set the volume at the magnetic axis (psiN=0) to be zero
    volume_profile[0] = 0.0;
    area_profile[0] = 0.0;
    'psi_n_loop: for i_psi_n in 1..n_psi_n {
        let psi_local: f64 = psi_n[i_psi_n] * (psi_b - psi_a) + psi_a;

        let flux_surface_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&psi_2d_flattened, &[psi_local]).expect("Plasma: boundary_contours_tmp");

        let flux_surface_contours: &geo_types::MultiPolygon = flux_surface_contours_tmp[0].geometry(); // The [0] is because I have only supplied one threshold

        // Loop over all contours and find the one which is inside (r_cur, z_cur)
        let n_contour: usize = flux_surface_contours.iter().count();

        'contour_loop: for i_contour in 0..n_contour {
            let fs_contour: &Polygon = flux_surface_contours.iter().nth(i_contour).expect("find_boundary: boundary_contour");

            // Test if all the points are inside the plasma boundary
            for coord in fs_contour.exterior() {
                let fs_r: f64 = coord.x;
                let fs_z: f64 = coord.y;
                let point: Point = Point::new(fs_r, fs_z);

                let inside_boundary: bool = boundary_polygon.contains(&point);
                if !inside_boundary {
                    // Not a valid contour, so try the next contour
                    continue 'contour_loop;
                }
            }

            // Store the flux surface
            let fs_r: Array1<f64> = fs_contour.exterior().coords().map(|coord| coord.x).collect::<Array1<f64>>();
            let fs_z: Array1<f64> = fs_contour.exterior().coords().map(|coord| coord.y).collect::<Array1<f64>>();
            let flux_surface = FluxSurface { r: fs_r, z: fs_z };
            flux_surfaces[i_psi_n] = flux_surface;

            // // Calculate the area of the contour
            // let area: f64 = fs_contour.unsigned_area();

            // let mass_centroid: Point = fs_contour.centroid().unwrap();
            // let mass_centroid_r: f64 = mass_centroid.x();

            // // Calculate the volume
            // area_profile[i_psi_n] = area;
            // volume_profile[i_psi_n] = 2.0 * PI * mass_centroid_r * area;

            // Go to the next psi_n
            continue 'psi_n_loop;
        }
    }

    // let mut volume_prime_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    // volume_prime_profile[0] = (volume_profile[0] - volume_profile[1]) / d_psi;
    // for i_psi_n in 1..n_psi_n - 1 {
    //     volume_prime_profile[i_psi_n] = (volume_profile[i_psi_n - 1] - volume_profile[i_psi_n + 1]) / (2.0 * d_psi);
    // }
    // volume_prime_profile[n_psi_n - 1] = (volume_profile[n_psi_n - 2] - volume_profile[n_psi_n - 1]) / d_psi;

    // let mut area_prime_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    // area_prime_profile[0] = (volume_profile[0] - volume_profile[1]) / d_psi;
    // for i_psi_n in 1..n_psi_n - 1 {
    //     area_prime_profile[i_psi_n] = (volume_profile[i_psi_n - 1] - volume_profile[i_psi_n + 1]) / (2.0 * d_psi);
    // }
    // area_prime_profile[n_psi_n - 1] = (volume_profile[n_psi_n - 2] - volume_profile[n_psi_n - 1]) / d_psi;

    return flux_surfaces;
}

fn epp_flux_toroidal_profile(q_profile: &Array1<f64>, psi_profile: &Array1<f64>) -> Array1<f64> {
    let n_psi_n: usize = psi_profile.len();

    let mut flux_toroidal_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    flux_toroidal_profile[0] = 0.0; // no toroidal flux at the magnetic axis
    for i_psi_n in 1..n_psi_n {
        let avg_y: f64 = (q_profile[i_psi_n] + q_profile[i_psi_n - 1]) / 2.0;
        let dx: f64 = psi_profile[i_psi_n] - psi_profile[i_psi_n - 1];
        flux_toroidal_profile[i_psi_n] = flux_toroidal_profile[i_psi_n - 1] - avg_y * dx;
    }

    return flux_toroidal_profile;
}

fn epp_li(ip: f64, r: &Array1<f64>, d_area: f64, r_mag: f64, r_geo: f64, b_r: &Array2<f64>, b_z: &Array2<f64>, mask: &Array2<f64>) -> (f64, f64, f64) {
    let dims: &[usize] = b_r.shape();
    let n_z: usize = dims[0];
    let n_r: usize = dims[1];

    let mut bp_sq_vol_int: f64 = 0.0;
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            let bp_sq: f64 = b_r[[i_z, i_r]].powi(2) + b_z[[i_z, i_r]].powi(2);
            bp_sq_vol_int += bp_sq * mask[[i_z, i_r]] * 2.0 * PI * r[i_r] * d_area;
        }
    }

    let li_1: f64 = f64::NAN;
    let li_2: f64 = 2.0 * bp_sq_vol_int / (MU_0.powi(2) * ip.powi(2) * r_mag);
    let li_3: f64 = 2.0 * bp_sq_vol_int / (MU_0.powi(2) * ip.powi(2) * r_geo);

    return (li_1, li_2, li_3);
}

fn epp_mid_plane_p_profile(
    gs_solution: &GsSolution,
    r: &Array1<f64>,
    i_z_centre: usize,
    psi_a: f64,
    psi_b: f64,
    psi_n_2d: &Array2<f64>,
    mask_2d: &Array2<f64>,
) -> Array1<f64> {
    let n_r: usize = r.len();

    let mut p_profile: Array1<f64> = Array1::from_elem(n_r, f64::NAN);

    let p_prime_dof_values: Array1<f64> = gs_solution.p_prime_dof_values.to_owned();

    // d(psi)/d(psi_n)
    let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

    // TODO: change this to a slice
    for i_r in 0..n_r {
        let psi_n_here: f64 = psi_n_2d[[i_z_centre, i_r]];

        let pressure_local: f64 = gs_solution
            .p_prime_source_function
            .source_function_integral(&Array1::from_vec(vec![psi_n_here]), &p_prime_dof_values)[0];

        // Apply the mask, and store pressure
        p_profile[i_r] = pressure_local * mask_2d[[i_z_centre, i_r]] / d_psi_d_psi_n;
    }

    return p_profile;
}

fn epp_p_2d(gs_solution: &GsSolution, r: &Array1<f64>, z: &Array1<f64>) -> Array2<f64> {
    // TODO: We might want to do some 2D interpolation
    let n_r: usize = r.len();
    let n_z: usize = z.len();

    let mut p_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);

    let psi_n_2d: Array2<f64> = gs_solution.psi_n_2d.to_owned(); // shape = (n_z, n_r)
    let mask_2d: Array2<f64> = gs_solution.mask.to_owned(); // shape = (n_z, n_r)

    let psi_a: f64 = gs_solution.psi_a;
    let psi_b: f64 = gs_solution.psi_b;

    // d(psi)/d(psi_n)
    let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

    let p_prime_dof_values: Array1<f64> = gs_solution.p_prime_dof_values.to_owned();
    let n_p_prime_dof: usize = gs_solution.p_prime_source_function.source_function_n_dof();

    for i_r in 0..n_r {
        for i_z in 0..n_z {
            let psi_n: f64 = psi_n_2d[(i_z, i_r)];

            let pressure_local_ndarray: Array1<f64> = gs_solution
                .p_prime_source_function
                .source_function_integral(&Array1::from_vec(vec![psi_n]), &p_prime_dof_values);
            let pressure_local: f64 = pressure_local_ndarray[0];

            // Apply the mask, and store pressure
            p_2d[(i_z, i_r)] = pressure_local * mask_2d[(i_z, i_r)] / d_psi_d_psi_n;
        }
    }

    return p_2d;
}

fn epp_hessian_matrix(gs_solution: &GsSolution, r: &Array1<f64>, z: &Array1<f64>, i_r: usize, i_z: usize) -> (Array2<f64>, f64, f64) {
    // TODO: Perhaps I should 2D interpolate the Hessian matrix?
    let psi: Array2<f64> = gs_solution.psi_2d.to_owned(); // shape = (n_z, n_r)

    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    let c: f64 = -2.0 * psi[[i_z, i_r]] + psi[[i_z, i_r + 1]] + psi[[i_z, i_r - 1]];
    let d: f64 = -2.0 * psi[[i_z, i_r]] + psi[[i_z + 1, i_r]] + psi[[i_z - 1, i_r]];
    let e: f64 = psi[[i_z, i_r]] - psi[[i_z, i_r + 1]] + psi[[i_z + 1, i_r + 1]] - psi[[i_z + 1, i_r]];

    let mut hessian_matrix: Array2<f64> = Array2::from_elem((2, 2), f64::NAN);
    hessian_matrix[[0, 0]] = c / d_r.powi(2);
    hessian_matrix[[0, 1]] = e / (d_r * d_z);
    hessian_matrix[[1, 0]] = e / (d_r * d_z);
    hessian_matrix[[1, 1]] = d / d_z.powi(2);

    // Calculate determinant and trace (as it's only 2x2 lets not use a library)
    let hessian_determinant: f64 = hessian_matrix[[0, 0]] * hessian_matrix[[1, 1]] - hessian_matrix[[0, 1]] * hessian_matrix[[1, 0]];
    let hessian_trace: f64 = hessian_matrix[[0, 0]] + hessian_matrix[[1, 1]];

    return (hessian_matrix, hessian_determinant, hessian_trace);
}

fn epp_p_profile(gs_solution: &GsSolution, psi_n: &Array1<f64>, psi_a: f64, psi_b: f64) -> Array1<f64> {
    // TODO: We might want to do some 2D interpolation
    let n_psi_n: usize = psi_n.len();

    let p_prime_dof_values: Array1<f64> = gs_solution.p_prime_dof_values.to_owned();

    // d(psi)/d(psi_n)
    let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

    // Integrate the source function
    let p_profile: Array1<f64> = gs_solution.p_prime_source_function.source_function_integral(psi_n, &p_prime_dof_values) / d_psi_d_psi_n;

    return p_profile;
}

fn epp_plasma_volume(gs_solution: &GsSolution, r_geo: f64) -> f64 {
    let boundary_r: Array1<f64> = gs_solution.boundary_r.to_owned();
    let boundary_z: Array1<f64> = gs_solution.boundary_z.to_owned();
    let n_boundary_points = boundary_r.len();
    // Collect the coordinates
    let mut boundary_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_boundary_points);
    for i_boundary_point in 0..n_boundary_points {
        boundary_coordinates.push(Coord {
            x: boundary_r[i_boundary_point],
            y: boundary_z[i_boundary_point],
        });
    }

    // Construct the contour
    let boundary_contour: Polygon = Polygon::new(
        LineString::new(boundary_coordinates),
        vec![], // No holes
    );

    let cross_sectional_area: f64 = boundary_contour.unsigned_area();

    let volume: f64 = 2.0 * PI * r_geo * cross_sectional_area;

    return volume;
}

fn epp_p_prime_profile(gs_solution: &GsSolution, psi_n: &Array1<f64>) -> Array1<f64> {
    let p_prime_dof_values: Array1<f64> = gs_solution.p_prime_dof_values.to_owned();
    let p_prime_local: Array1<f64> = gs_solution.p_prime_source_function.source_function_value(psi_n, &p_prime_dof_values);
    return p_prime_local;
}

fn epp_q_profile(gs_solution: &GsSolution, flux_surfaces: &Vec<FluxSurface>, f_profile: &Array1<f64>, r: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
    // g3 = <1/R**2> = (2.0 / vol_prime) * integral(1 / (Bp * R**2) d_ell)
    // where: vol_prime = d(V)/d(psi)
    // where: <1/R**2> is notation for the flux surface average

    let n_psi_n: usize = flux_surfaces.len();
    let br: Array2<f64> = gs_solution.br_2d.to_owned();
    let bz: Array2<f64> = gs_solution.bz_2d.to_owned();
    let bp: Array2<f64> = (br.mapv(|x| x.powi(2)) + bz.mapv(|x| x.powi(2))).mapv(f64::sqrt);

    let bp_interpolator = Interp2D::builder(bp)
        .x(z.clone())
        .y(r.clone())
        .build()
        .expect("find_boundary: Can't make Interp2D");

    let mut q_profile: Array1<f64> = Array1::zeros(n_psi_n);
    'fs_loop: for i_psi_n in 0..n_psi_n {
        let fs_r: Array1<f64> = flux_surfaces[i_psi_n].r.clone();
        let fs_z: Array1<f64> = flux_surfaces[i_psi_n].z.clone();
        let fs_n: usize = fs_r.len();

        if fs_n < 2 {
            continue 'fs_loop;
        }

        // TODO: temporary fix for invalid LCFS!!
        let invalid_lcfs: bool = fs_z
            .abs()
            .max()
            .map(|&fs_z_val| fs_z_val > *z.max().expect("can't unwrap max"))
            .expect("can't unwrap max");
        if invalid_lcfs {
            continue 'fs_loop;
        }

        let mut ell: Array1<f64> = Array1::from_elem(fs_n, f64::NAN);
        ell[0] = 0.0;
        for i_fs in 1..fs_n {
            ell[i_fs] = ell[i_fs - 1] + (fs_r[i_fs] - fs_r[i_fs - 1]).hypot(fs_z[i_fs] - fs_z[i_fs - 1]);
        }
        let mut integrand: Array1<f64> = Array1::from_elem(fs_n, f64::NAN);
        // TODO: this **COULD** be wrong because I am calculating the integrand at the boundary point.
        // But the ell variable is not consistent, since it's between boundary points.
        // Look up "midpoint integral approximation" ??
        for i_fs in 0..fs_n {
            let bp_here: f64 = bp_interpolator
                .interp_scalar(fs_z[i_fs], fs_r[i_fs])
                .expect("possible_bounding_psi: error, limiter");

            integrand[i_fs] = f_profile[i_psi_n] / (2.0 * PI * bp_here * fs_r[i_fs].powi(2));
        }

        // Perform the integration
        for i_fs in 1..fs_n {
            q_profile[i_psi_n] += 0.5 * (ell[i_fs] - ell[i_fs - 1]) * (integrand[i_fs] + integrand[i_fs - 1]);
        }
    }

    // Central safety factor
    let q0: f64 = epp_q_axis(gs_solution, r, z, f_profile);
    q_profile[0] = q0;

    return q_profile;
}

fn epp_q_axis(gs_solution: &GsSolution, r: &Array1<f64>, z: &Array1<f64>, f_profile: &Array1<f64>) -> f64 {
    // TODO: this works ok (ish). I think I will need to do a 2D interpolation for the Hessian matrix
    // I could do this by calculating the Hessian matrix at each point in the grid, and then doing 2D interpolation.
    // Or I could do 2D interpolation on psi, which is used to calculate the Hessian matrix?

    let r_mag: f64 = gs_solution.r_mag;
    let z_mag: f64 = gs_solution.z_mag;

    // Find the nearest point to the magnetic axis
    let mut index_r_mag: usize = 0;
    let mut index_z_mag: usize = 0;
    let mut min_distance: f64 = f64::MAX;
    for i_r in 0..r.len() {
        for i_z in 0..z.len() {
            let distance: f64 = ((r[i_r] - r_mag).powi(2) + (z[i_z] - z_mag).powi(2)).sqrt();
            if distance < min_distance {
                min_distance = distance;
                index_r_mag = i_r;
                index_z_mag = i_z;
            }
        }
    }

    let (_hessian_matrix, hessian_determinant, hessian_trace): (Array2<f64>, f64, f64) = epp_hessian_matrix(gs_solution, r, z, index_r_mag, index_z_mag);

    let j_phi: f64 = gs_solution.j_2d[[index_z_mag, index_r_mag]];
    let q_axis: f64 = hessian_trace.abs() / hessian_determinant.sqrt() * f_profile[0] / (MU_0 * r_mag.powi(2) * j_phi);

    // q_axis = abs(tri(H(psiN=0))) / sqrt(det(H(psiN=0))) * f_profile(psiN=0) / (mu0 * r_mag**2 * j_phi)

    return q_axis;
}

fn epp_q95(q_profile: &Array1<f64>, psi_n: &Array1<f64>) -> f64 {
    // q95 = q(psi_n=0.95)
    let q95: f64 = Interp1D::builder(q_profile.to_owned())
        .x(psi_n.clone())
        .build()
        .expect("find_boundary: Can't make Interp1D")
        .interp_scalar(0.95)
        .expect("possible_bounding_psi: error, limiter");
    return q95;
}

fn epp_rho_tor_profile(flux_tor_profile: &Array1<f64>) -> Array1<f64> {
    let n_psi_n: usize = flux_tor_profile.len();

    let flux_tor_max: Result<&f64, ndarray_stats::errors::MinMaxError> = flux_tor_profile.max();
    if flux_tor_max.is_err() {
        let rho_tor: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
        return rho_tor;
    }
    let rho_tor: Array1<f64> = (flux_tor_profile / flux_tor_max.unwrap().to_owned()).mapv(|x| x.sqrt());
    return rho_tor;
}

fn epp_vol_profile(
    gs_solution: &GsSolution,
    psi_n: &Array1<f64>,
    r: &Array1<f64>,
    z: &Array1<f64>,
    d_psi: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    // Sizes and grid variables
    let n_psi_n: usize = psi_n.len();
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    let mut volume_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    let mut area_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);

    // Create an empty contour grid
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true) // x dim., y dim., smoothing
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    let psi_2d: Array2<f64> = gs_solution.psi_2d.to_owned();
    let psi_2d_flattened: Vec<f64> = psi_2d.iter().cloned().collect();

    let psi_a: f64 = gs_solution.psi_a;
    let psi_b: f64 = gs_solution.psi_b;

    // Creat the plasma boundary polygon
    let boundary_r: Array1<f64> = gs_solution.boundary_r.to_owned();
    let boundary_z: Array1<f64> = gs_solution.boundary_z.to_owned();
    let n_boundary_points: usize = boundary_r.len();
    let mut boundary_polygon_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_boundary_points);
    for i_boundary_point in 0..n_boundary_points {
        boundary_polygon_coordinates.push(Coord {
            x: boundary_r[i_boundary_point],
            y: boundary_z[i_boundary_point],
        });
    }
    let boundary_polygon_coordinates: Vec<Coord<f64>> = boundary_r.iter().zip(boundary_z.iter()).map(|(&x, &y)| Coord { x, y }).collect();
    let boundary_polygon: Polygon = Polygon::new(
        LineString::from(boundary_polygon_coordinates),
        vec![], // No holes
    );

    // Loop over psi_n
    // Set the volume at the magnetic axis (psiN=0) to be zero
    volume_profile[0] = 0.0;
    area_profile[0] = 0.0;
    // Do the last closed flux surface
    // (LCFS fails because the contour is "on" the boundary and is removed)
    let area: f64 = boundary_polygon.unsigned_area();
    let mass_centroid: Point = boundary_polygon.centroid().unwrap();
    let mass_centroid_r: f64 = mass_centroid.x();
    area_profile[n_psi_n - 1] = area;
    volume_profile[n_psi_n - 1] = 2.0 * PI * mass_centroid_r * area;

    // Don't do the first or last points
    'psi_n_loop: for i_psi_n in 1..n_psi_n - 1 {
        let psi_local: f64 = psi_n[i_psi_n] * (psi_b - psi_a) + psi_a;

        let flux_surface_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&psi_2d_flattened, &[psi_local]).expect("Plasma: boundary_contours_tmp");

        let flux_surface_contours: &geo_types::MultiPolygon = flux_surface_contours_tmp[0].geometry(); // The [0] is because I have only supplied one threshold

        // Loop over all contours and find the one which is inside (r_cur, z_cur)
        let n_contour: usize = flux_surface_contours.iter().count();

        'contour_loop: for i_contour in 0..n_contour {
            let fs_contour: &Polygon = flux_surface_contours.iter().nth(i_contour).expect("find_boundary: boundary_contour");

            // Test if all the points are inside the plasma boundary
            for coord in fs_contour.exterior() {
                let fs_r: f64 = coord.x;
                let fs_z: f64 = coord.y;
                let point: Point = Point::new(fs_r, fs_z);

                let inside_boundary: bool = boundary_polygon.contains(&point);
                if !inside_boundary {
                    // Not a valid contour, so try the next contour
                    continue 'contour_loop;
                }
            }

            // Calculate the area of the contour
            let area: f64 = fs_contour.unsigned_area();

            let mass_centroid: Point = fs_contour.centroid().unwrap();
            let mass_centroid_r: f64 = mass_centroid.x();

            // Calculate the volume
            area_profile[i_psi_n] = area;
            volume_profile[i_psi_n] = 2.0 * PI * mass_centroid_r * area;

            // Go to the next psi_n
            continue 'psi_n_loop;
        }
    }

    // Take derivatives
    let mut volume_prime_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    volume_prime_profile[0] = (volume_profile[0] - volume_profile[1]) / d_psi;
    for i_psi_n in 1..n_psi_n - 1 {
        volume_prime_profile[i_psi_n] = (volume_profile[i_psi_n - 1] - volume_profile[i_psi_n + 1]) / (2.0 * d_psi);
    }
    volume_prime_profile[n_psi_n - 1] = (volume_profile[n_psi_n - 2] - volume_profile[n_psi_n - 1]) / d_psi;

    let mut area_prime_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    area_prime_profile[0] = (volume_profile[0] - volume_profile[1]) / d_psi;
    for i_psi_n in 1..n_psi_n - 1 {
        area_prime_profile[i_psi_n] = (volume_profile[i_psi_n - 1] - volume_profile[i_psi_n + 1]) / (2.0 * d_psi);
    }
    area_prime_profile[n_psi_n - 1] = (volume_profile[n_psi_n - 2] - volume_profile[n_psi_n - 1]) / d_psi;

    return (volume_profile, volume_prime_profile, area_profile, area_prime_profile);
}

fn epp_w_mhd(p_2d: &Array2<f64>, r: &Array1<f64>, d_area: f64) -> f64 {
    let dims: &[usize] = p_2d.shape();
    let n_z: usize = dims[0];
    let n_r: usize = dims[1];

    let mut w_mhd: f64 = 0.0;
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            w_mhd += (3.0 / 2.0) * p_2d[[i_z, i_r]] * 2.0 * PI * r[i_r] * d_area;
        }
    }

    return w_mhd;
}

#[derive(Clone)]
pub struct FluxSurface {
    pub r: Array1<f64>,
    pub z: Array1<f64>,
}
