use crate::coils::Coils;
use crate::greens::Greens;
use crate::passives::Passives;
use crate::source_functions::SharedSourceFunction;
use crate::source_functions::extract_source_function;
use data_tree::DataTreeAccumulator;
use imas_rs::EquilibriumProfiles2dGrid;
use imas_rs::python::PyEquilibrium;
use imas_rs::{
    Equilibrium, EquilibriumGreensGridGrid, EquilibriumGreensPfActive, EquilibriumGreensPfPassive, EquilibriumGreensPfPassiveDof, EquilibriumProfiles2d,
};
use ndarray::{Array1, Array2, ArrayView2, Axis, MeshIndex, meshgrid, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

#[derive(Clone)]
#[pyclass(skip_from_py_object)]
pub struct Plasma {
    /// The equilibrium IDS. Empty until `initialise_equilibrium_ids` is called, which happens once
    /// the reconstruction times are known
    pub equilibrium_ids: Equilibrium,
    pub p_prime_source_function: SharedSourceFunction,
    pub ff_prime_source_function: SharedSourceFunction,
}

// Python accessible methods
#[pymethods]
impl Plasma {
    /// Create a new Plasma instance
    ///
    /// # Arguments
    /// * `n_r` - number of radial points, [dimensionless]
    /// * `n_z` - number of vertical points, [dimensionless]
    /// * `r_min` - minimum radial coordinate, [metre]
    /// * `r_max` - maximum radial coordinate, [metre]
    /// * `z_min` - minimum vertical coordinate, [metre]
    /// * `z_max` - maximum vertical coordinate, [metre]
    /// * `psi_n` - normalized poloidal flux points (1d array), [dimensionless]
    /// * `p_prime_source_function` - pressure source function (a Rust implementation, initialised in Python)
    /// * `ff_prime_source_function` - ff_prime source function (a Rust implementation, initialised in Python)
    /// * `initial_guess_ip` - initial total plasma current, [ampere]
    /// * `initial_guess_cur_r` - radial centre of the initial current distribution, [metre]
    /// * `initial_guess_cur_z` - vertical centre of the initial current distribution, [metre]
    /// * `initial_guess_minor_radius` - radial semi-axis of the initial current distribution, [metre]
    /// * `initial_guess_elongation` - elongation of the initial current distribution, [dimensionless]
    /// * `n_iter_max` - maximum number of Picard iterations, [dimensionless]
    /// * `n_iter_min` - minimum number of Picard iterations before the convergence test may pass, [dimensionless]
    /// * `n_iter_no_vertical_feedback` - number of initial iterations with the vertical feedback off, [dimensionless]
    /// * `gs_error` - Grad-Shafranov deviation below which the solution is taken as converged, [mixed]
    /// * `use_anderson_mixing` - whether Anderson mixing is applied
    /// * `anderson_mixing_from_previous_iter` - fraction of the previous iteration mixed in, [dimensionless]
    /// * `times_to_reconstruct` - the times the equilibrium will be solved at (1d array), [second].
    ///   One equilibrium time-slice is allocated per time, so that the IDS is fully formed before
    ///   the Green's tables are built
    ///
    /// # Returns
    /// * `self` - a new instance of the Plasma struct
    ///
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_r: usize,
        n_z: usize,
        r_min: f64,
        r_max: f64,
        z_min: f64,
        z_max: f64,
        psi_n: PyReadonlyArray1<f64>,
        p_prime_source_function: &Bound<'_, PyAny>,  // Any Python object, because Python doesn't know about types
        ff_prime_source_function: &Bound<'_, PyAny>, // Any Python object, because Python doesn't know about types
        initial_guess_ip: f64,
        initial_guess_cur_r: f64,
        initial_guess_cur_z: f64,
        initial_guess_minor_radius: f64,
        initial_guess_elongation: f64,
        n_iter_max: usize,
        n_iter_min: usize,
        n_iter_no_vertical_feedback: usize,
        gs_error: f64,
        use_anderson_mixing: bool,
        anderson_mixing_from_previous_iter: f64,
        times_to_reconstruct: PyReadonlyArray1<f64>,
    ) -> Self {
        // Change Python types into Rust types
        let psi_n_ndarray: Array1<f64> = psi_n.to_owned_array();

        // `p_prime_source_function` and `ff_prime_source_function` come into Rust as a Python PyAny type, so we have no idea what type they are.
        // In `extract_source_function` we attempt to convert into a known Rust type and panic if we can't.
        // This is a "bit ugly", but it is a known limitation of PyO3 because of how Python types work.
        let p_prime_source_function_arc: SharedSourceFunction = extract_source_function(p_prime_source_function);
        let ff_prime_source_function_arc: SharedSourceFunction = extract_source_function(ff_prime_source_function);

        // Create storage

        // Create (r, z) grids
        let r: Array1<f64> = Array1::linspace(r_min, r_max, n_r);
        let z: Array1<f64> = Array1::linspace(z_min, z_max, n_z);

        // Grid spacing
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let d_area: f64 = d_r * d_z;

        // Check that the R grid doesn't go negative
        // Note, we allow cells to touch the axis (R=0), which would be excluded by `r_min - d_r / 2.0 <= 0.0`
        // but we do not allow cells to go negative.
        if r_min - d_r / 2.0 < 0.0 {
            panic!("plasma.new: r_min - d_r / 2.0 < 0.0; the radial grid must not go negative");
        }

        // 2d (r, z) mesh
        let (mesh_z_view, mesh_r_view): (ArrayView2<f64>, ArrayView2<f64>) = meshgrid((&z, &r), MeshIndex::IJ);
        let mesh_z: Array2<f64> = mesh_z_view.to_owned(); // shape = (n_z, n_r)
        let mesh_r: Array2<f64> = mesh_r_view.to_owned(); // shape = (n_z, n_r)

        // Flatten 2d mesh
        let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
        let flat_z: Array1<f64> = mesh_z.flatten().to_owned();

        // Calculate the grid-grid Greens
        let flat_d_r: Array1<f64> = &r * 0.0 + d_r;
        let flat_d_z: Array1<f64> = &r * 0.0 + d_z;
        let greens_calculator: Greens = Greens::sensor_to_conductor(flat_r.clone(), flat_z.clone(), r.clone(), 0.0 * r.clone() + z[0], flat_d_r, flat_d_z);
        let g_psi: Array2<f64> = greens_calculator.psi();
        let g_br: Array2<f64> = greens_calculator.b_r();
        let g_bz: Array2<f64> = greens_calculator.b_z();
        let mut g_d_br_d_z: Array2<f64> = greens_calculator.d_b_r_d_z();
        let mut g_d_bz_d_z: Array2<f64> = greens_calculator.d_b_z_d_z();

        // Set the self-values to 0.0
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d_br_d_z[(i_rz, i_r)].is_nan() {
                    g_d_br_d_z[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                    g_d_bz_d_z[(i_rz, i_r)] = 0.0;
                }
            }
        }

        // d2_g_d_r2
        let mut g_d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2();
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d2_psi_d_r2[(i_rz, i_r)].is_nan() {
                    g_d2_psi_d_r2[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
            }
        }

        // d2_g_d_r_d_z
        let mut g_d2_psi_d_r_d_z: Array2<f64> = greens_calculator.d2_psi_d_r_d_z();
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d2_psi_d_r_d_z[(i_rz, i_r)].is_nan() {
                    g_d2_psi_d_r_d_z[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
            }
        }

        // d_g_d_r and d_g_d_z
        let mut g_d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r();
        let mut g_d_psi_d_z: Array2<f64> = greens_calculator.d_psi_d_z();
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d_psi_d_r[(i_rz, i_r)].is_nan() {
                    g_d_psi_d_r[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
                if g_d_psi_d_z[(i_rz, i_r)].is_nan() {
                    g_d_psi_d_z[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
            }
        }

        // Second and third z-derivatives (needed by `calculate_psi_and_derivatives` and its `delta_z` shifts)
        let mut g_d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2();
        let mut g_d3_psi_d_r2_d_z: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z();
        let mut g_d3_psi_d_r_d_z2: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2();
        let mut g_d3_psi_d_z3: Array2<f64> = greens_calculator.d3_psi_d_z3();
        for i_r in 0..n_r {
            for i_rz in 0..n_r * n_z {
                if g_d2_psi_d_z2[(i_rz, i_r)].is_nan() {
                    g_d2_psi_d_z2[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
                if g_d3_psi_d_r2_d_z[(i_rz, i_r)].is_nan() {
                    g_d3_psi_d_r2_d_z[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
                if g_d3_psi_d_r_d_z2[(i_rz, i_r)].is_nan() {
                    g_d3_psi_d_r_d_z2[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
                if g_d3_psi_d_z3[(i_rz, i_r)].is_nan() {
                    g_d3_psi_d_z3[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
                }
            }
        }

        // Store the plasma grid-to-grid Greens tables in the equilibrium IDS. They are geometry, so
        // they are the same for every time-slice and hang off the IDS root rather than off
        // `time_slice`
        let greens_grid_grid: EquilibriumGreensGridGrid = EquilibriumGreensGridGrid {
            psi: Some(g_psi),
            br: Some(g_br),
            bz: Some(g_bz),
            d_br_d_z: Some(g_d_br_d_z),
            d_bz_d_z: Some(g_d_bz_d_z),
            d_psi_d_r: Some(g_d_psi_d_r),
            d_psi_d_z: Some(g_d_psi_d_z),
            d2_psi_d_r2: Some(g_d2_psi_d_r2),
            d2_psi_d_r_d_z: Some(g_d2_psi_d_r_d_z),
            d2_psi_d_z2: Some(g_d2_psi_d_z2),
            d3_psi_d_r2_d_z: Some(g_d3_psi_d_r2_d_z),
            d3_psi_d_r_d_z2: Some(g_d3_psi_d_r_d_z2),
            d3_psi_d_z3: Some(g_d3_psi_d_z3),
        };

        // The equilibrium IDS. `initialise_equilibrium_ids` fills in the time-slices below; what is
        // set here is the machine-level data which does not depend on them
        let mut equilibrium_ids: Equilibrium = Equilibrium::default();

        // Store values
        equilibrium_ids.code.grid.n_r = Some(n_r as i32);
        equilibrium_ids.code.grid.n_z = Some(n_z as i32);
        equilibrium_ids.code.grid.r_min = Some(r_min);
        equilibrium_ids.code.grid.r_max = Some(r_max);
        equilibrium_ids.code.grid.z_min = Some(z_min);
        equilibrium_ids.code.grid.z_max = Some(z_max);

        equilibrium_ids.code.initial_guess.ip = Some(initial_guess_ip);
        equilibrium_ids.code.initial_guess.cur_r = Some(initial_guess_cur_r);
        equilibrium_ids.code.initial_guess.cur_z = Some(initial_guess_cur_z);
        equilibrium_ids.code.initial_guess.minor_radius = Some(initial_guess_minor_radius);
        equilibrium_ids.code.initial_guess.elongation = Some(initial_guess_elongation);

        equilibrium_ids.code.numerics.iterations.n_max = Some(n_iter_max as i32);
        equilibrium_ids.code.numerics.iterations.n_min = Some(n_iter_min as i32);
        equilibrium_ids.code.numerics.iterations.n_no_vertical_feedback = Some(n_iter_no_vertical_feedback as i32);
        equilibrium_ids.code.numerics.grad_shafranov_deviation_tolerance = Some(gs_error);
        // The data dictionary has no boolean base type, so the flag is stored as 0 or 1
        equilibrium_ids.code.numerics.anderson_mixing.r#use = Some(use_anderson_mixing as i32);
        equilibrium_ids.code.numerics.anderson_mixing.mixing_from_previous_iter = Some(anderson_mixing_from_previous_iter);

        equilibrium_ids.greens.grid_grid = greens_grid_grid;

        let mut plasma: Self = Self {
            equilibrium_ids,
            p_prime_source_function: p_prime_source_function_arc,
            ff_prime_source_function: ff_prime_source_function_arc,
        };

        // The equilibrium IDS is allocated here, rather than once the solve starts, because the
        // grid it carries is read before then: the Green's tables are built from it
        plasma.initialise_equilibrium_ids(&times_to_reconstruct.to_owned_array(), &r, &z, &mesh_r, &mesh_z, &psi_n_ndarray, d_area);

        return plasma;
    }

    /// Calculate the Greens function with coils
    /// The Greens tables are stored within self. Example data structure:
    /// `greens/pf_active(i)/br` on the equilibrium IDS
    ///
    /// # Arguments
    /// * `coils` - The Coils object (a Rust implementation, initialised in Python)
    ///
    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Unpack from self
        let n_r: usize = self.equilibrium_ids.code.grid.n_r.unwrap() as usize;
        let n_z: usize = self.equilibrium_ids.code.grid.n_z.unwrap() as usize;
        let mesh_r: &Array2<f64> = self.equilibrium_ids.time_slice(0).profiles_2d(0).r.as_ref().unwrap();
        let mesh_z: &Array2<f64> = self.equilibrium_ids.time_slice(0).profiles_2d(0).z.as_ref().unwrap();
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables for the equilibrium IDS. Built in the same loop, and therefore the same
        // order, as the DataTree ones: both iterate `coils.results.get("pf").keys()`, which is
        // sorted, and the solver's `pf` wildcard read sorts too, so the two agree column for column
        let mut greens_pf_active: Vec<EquilibriumGreensPfActive> = Vec::with_capacity(coils.results.get("pf").keys().len());

        for coil_name in &coils.results.get("pf").keys() {
            // Coils
            let coil_r: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("z").unwrap_array1();
            let n_coil_filaments: usize = coil_r.len();

            // Greens function for flux
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                flat_r.clone(),
                flat_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
                Array1::from_elem(n_coil_filaments, f64::NAN), // grid should not overlap with coils
                Array1::from_elem(n_coil_filaments, f64::NAN),
            );

            // Greens function for psi, br, bz, and derivatives
            let g_psi_filaments: Array2<f64> = greens_calculator.psi(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_psi_d_r_filaments: Array2<f64> = greens_calculator.d_psi_d_r(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_psi_d_z_filaments: Array2<f64> = greens_calculator.d_psi_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let g_br_all_filaments: Array2<f64> = greens_calculator.b_r(); // shape = (n_z * n_r, n_coil_filaments)
            let g_bz_all_filaments: Array2<f64> = greens_calculator.b_z(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_br_d_z_all_filaments: Array2<f64> = greens_calculator.d_b_r_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_bz_d_z_all_filaments: Array2<f64> = greens_calculator.d_b_z_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_r2_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_r2(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_r_d_z_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_r_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_z2_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_z2(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_r2_d_z_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_r_d_z2_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_z3_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_z3(); // shape = (n_z * n_r, n_coil_filaments)

            // sum over all filaments and convert into shape = (n_z, n_r)
            let g_psi: Array2<f64> = g_psi_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_psi_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_psi_d_r: Array2<f64> = g_d_psi_d_r_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_d_psi_d_r_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_psi_d_z: Array2<f64> = g_d_psi_d_z_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_d_psi_d_z_filaments` into (n_z, n_r)")
                .to_owned();
            let g_br: Array2<f64> = g_br_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_br_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_bz: Array2<f64> = g_bz_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_bz_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_br_d_z: Array2<f64> = g_d_br_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_d_br_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_bz_d_z: Array2<f64> = g_d_bz_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `g_d_bz_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_r2: Array2<f64> = d2_g_d_r2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d2_g_d_r2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_r_d_z: Array2<f64> = d2_g_d_r_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d2_g_d_r_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_z2: Array2<f64> = d2_g_d_z2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d2_g_d_z2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_r2_d_z: Array2<f64> = d3_g_d_r2_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d3_g_d_r2_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_r_d_z2: Array2<f64> = d3_g_d_r_d_z2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d3_g_d_r_d_z2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_z3: Array2<f64> = d3_g_d_z3_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("plasma.greens_with_coils: Failed to reshape `d3_g_d_z3_all_filaments` into (n_z, n_r)")
                .to_owned();

            // Store in the equilibrium IDS
            greens_pf_active.push(EquilibriumGreensPfActive {
                name: Some(coil_name.clone()),
                psi: Some(g_psi),
                br: Some(g_br),
                bz: Some(g_bz),
                d_br_d_z: Some(g_d_br_d_z),
                d_bz_d_z: Some(g_d_bz_d_z),
                d_psi_d_r: Some(g_d_psi_d_r),
                d_psi_d_z: Some(g_d_psi_d_z),
                d2_psi_d_r2: Some(g_d2_psi_d_r2),
                d2_psi_d_r_d_z: Some(g_d2_psi_d_r_d_z),
                d2_psi_d_z2: Some(g_d2_psi_d_z2),
                d3_psi_d_r2_d_z: Some(g_d3_psi_d_r2_d_z),
                d3_psi_d_r_d_z2: Some(g_d3_psi_d_r_d_z2),
                d3_psi_d_z3: Some(g_d3_psi_d_z3),
            });
        }

        self.equilibrium_ids.greens.pf_active = greens_pf_active;
    }

    /// Calculate the Greens function with passives
    /// The Greens tables are stored within self. Example data structure:
    /// `greens/pf_passive(i)/dof(j)/psi` on the equilibrium IDS
    /// Note: when adding a passive to the `passives` implementation we selected how to represent the
    /// passive degrees of freedom through `current_distribution_type` (e.g. `constant_current_density` or `eig`)
    ///
    /// # Arguments
    /// * `passives` - The Passives object (a Rust implementation, initialised in Python)
    ///
    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python types into Rust types
        let passives_local: &Passives = &passives;

        // Get variables out of self
        // `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives the
        // flattened grid
        let mesh_r: &Array2<f64> = self.equilibrium_ids.time_slice(0).profiles_2d(0).r.as_ref().unwrap();
        let mesh_z: &Array2<f64> = self.equilibrium_ids.time_slice(0).profiles_2d(0).z.as_ref().unwrap();
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables for the equilibrium IDS. Built in the same nested loop, and therefore the
        // same order, as the DataTree ones. Both walk sorted key lists, and so does
        // `get_greens_passive_grid`, so the degree-of-freedom columns agree one for one
        let mut greens_pf_passive: Vec<EquilibriumGreensPfPassive> = Vec::with_capacity(passives_local.results.keys().len());

        // Calculate Greens with each passive degree of freedom
        // let passive_names: Vec<String> = ;
        for passive_name in passives_local.results.keys() {
            let _tmp: DataTreeAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
            let dof_names: Vec<String> = _tmp.keys();
            let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
            let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

            // Green's tables. These depend only on the grid and this passive's filament geometry,
            // so they are built once per passive; the degree of freedom is applied to them below
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                flat_r.clone(),
                flat_z.clone(),
                passive_r.clone(),
                passive_z.clone(),
                passive_r.clone() * f64::NAN, // d_r=0; as there will not be any points which coincide; using NaN as safety - if we get NaN's we know we have a problem
                passive_z.clone() * f64::NAN, // d_z=0; as there will not be any points which coincide; using NaN as safety - if we get NaN's we know we have a problem
            );

            // Green's functions for `psi`, `b_r`, `b_z`, and derivatives
            let g_psi_filaments: Array2<f64> = greens_calculator.psi(); // shape = [n_r * n_z, n_filament]
            let g_br_filaments: Array2<f64> = greens_calculator.b_r(); // shape = [n_r * n_z, n_filament]
            let g_bz_filaments: Array2<f64> = greens_calculator.b_z(); // shape = [n_r * n_z, n_filament]
            let d_g_br_filaments_d_z: Array2<f64> = greens_calculator.d_b_r_d_z(); // shape = [n_r * n_z, n_filament]
            let d_g_bz_filaments_d_z: Array2<f64> = greens_calculator.d_b_z_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d_psi_d_r_coil_filaments: Array2<f64> = greens_calculator.d_psi_d_r(); // shape = [n_r * n_z, n_filament]
            let g_d_psi_d_z_coil_filaments: Array2<f64> = greens_calculator.d_psi_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_r2_filaments: Array2<f64> = greens_calculator.d2_psi_d_r2(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_r_d_z_filaments: Array2<f64> = greens_calculator.d2_psi_d_r_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_z2_filaments: Array2<f64> = greens_calculator.d2_psi_d_z2(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_r2_d_z_filaments: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_r_d_z2_filaments: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_z3_filaments: Array2<f64> = greens_calculator.d3_psi_d_z3(); // shape = [n_r * n_z, n_filament]

            // The current distribution belonging to each degree of freedom, collected up-front so that
            // all of them can be applied to a Green's table in a single pass over it
            let n_dof: usize = dof_names.len();
            let mut current_distributions: Vec<Array1<f64>> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                current_distributions.push(
                    passives_local
                        .results
                        .get(&passive_name)
                        .get("dof")
                        .get(&dof_names[i_dof])
                        .get("current_distribution")
                        .unwrap_array1(),
                );
            }

            // Apply the current distributions and sum over the filaments; one column per degree of freedom
            let g_psi_with_dof: Array2<f64> = apply_current_distributions(&g_psi_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_br_with_dof: Array2<f64> = apply_current_distributions(&g_br_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_bz_with_dof: Array2<f64> = apply_current_distributions(&g_bz_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_br_d_z_with_dof: Array2<f64> = apply_current_distributions(&d_g_br_filaments_d_z, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_bz_d_z_with_dof: Array2<f64> = apply_current_distributions(&d_g_bz_filaments_d_z, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_psi_d_r_with_dof: Array2<f64> = apply_current_distributions(&g_d_psi_d_r_coil_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_psi_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d_psi_d_z_coil_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_r2_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_r2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_r_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_r_d_z_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_z2_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_z2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_r2_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_r2_d_z_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_r_d_z2_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_r_d_z2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_z3_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_z3_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]

            let mut greens_dof: Vec<EquilibriumGreensPfPassiveDof> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                greens_dof.push(EquilibriumGreensPfPassiveDof {
                    name: Some(dof_names[i_dof].clone()),
                    psi: Some(g_psi_with_dof.column(i_dof).to_owned()),
                    br: Some(g_br_with_dof.column(i_dof).to_owned()),
                    bz: Some(g_bz_with_dof.column(i_dof).to_owned()),
                    d_br_d_z: Some(g_d_br_d_z_with_dof.column(i_dof).to_owned()),
                    d_bz_d_z: Some(g_d_bz_d_z_with_dof.column(i_dof).to_owned()),
                    d_psi_d_r: Some(g_d_psi_d_r_with_dof.column(i_dof).to_owned()),
                    d_psi_d_z: Some(g_d_psi_d_z_with_dof.column(i_dof).to_owned()),
                    d2_psi_d_r2: Some(g_d2_psi_d_r2_with_dof.column(i_dof).to_owned()),
                    d2_psi_d_r_d_z: Some(g_d2_psi_d_r_d_z_with_dof.column(i_dof).to_owned()),
                    d2_psi_d_z2: Some(g_d2_psi_d_z2_with_dof.column(i_dof).to_owned()),
                    d3_psi_d_r2_d_z: Some(g_d3_psi_d_r2_d_z_with_dof.column(i_dof).to_owned()),
                    d3_psi_d_r_d_z2: Some(g_d3_psi_d_r_d_z2_with_dof.column(i_dof).to_owned()),
                    d3_psi_d_z3: Some(g_d3_psi_d_z3_with_dof.column(i_dof).to_owned()),
                });
            }

            greens_pf_passive.push(EquilibriumGreensPfPassive {
                name: Some(passive_name),
                dof: greens_dof,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the tables instead of
        // silently doubling them up
        self.equilibrium_ids.greens.pf_passive = greens_pf_passive;
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Plasma>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_r: usize = self.equilibrium_ids.code.grid.n_r.unwrap() as usize;
        let n_z: usize = self.equilibrium_ids.code.grid.n_z.unwrap() as usize;
        string_output += &format!("║  {:<74} ║\n", format!(" n_r = {}, n_z = {}", n_r, n_z));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }

    /// The equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`.
    ///
    /// The IDS is copied into the returned object, so it is a snapshot: changes made on the
    /// Rust side afterwards are not seen by it. A borrow is not possible here, because
    /// `imas_rs` cannot name `Plasma` without the two crates depending on each other.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        return PyEquilibrium::new(self.equilibrium_ids.clone());
    }
}

// Rust only methods - either because we want to keep the methods private
// or more likely because the methods are incompatible with Python
impl Plasma {
    /// Pre-allocate the equilibrium IDS with one empty time-slice per reconstruction time.
    ///
    /// Every leaf is unset (`None`); the Grad-Shafranov solver fills them in. Called once the
    /// reconstruction times are known, which is later than `Plasma::new`.
    ///
    /// Only the time slices are (re)allocated. Anything already on the IDS - `code`,
    /// `vacuum_toroidal_field`, ... - is left alone, so this does not discard data set before the
    /// reconstruction runs.
    /// Allocate one equilibrium time-slice per reconstruction time, and store the grid on each.
    ///
    /// The equilibrium IDS has no IDS-level grid: `grid`, `grid_type` and the (R, Z) positions all
    /// live inside `time_slice/profiles_2d`, so the grid can only be stored once the time-slices
    /// exist. That is why this runs from `Plasma::new`, before anything reads the grid.
    ///
    /// # Arguments
    /// * `times_to_reconstruct` - the times to allocate a time-slice for [second]
    /// * `r`, `z` - the grid axes [metre]
    /// * `mesh_r`, `mesh_z` - the (R, Z) mesh, shape `(n_z, n_r)` [metre]
    /// * `psi_norm` - the normalised poloidal flux grid the source functions are defined on
    /// * `d_area` - area of one grid cell [metre ** 2]
    #[allow(clippy::too_many_arguments)]
    fn initialise_equilibrium_ids(
        &mut self,
        times_to_reconstruct: &Array1<f64>,
        r: &Array1<f64>,
        z: &Array1<f64>,
        mesh_r: &Array2<f64>,
        mesh_z: &Array2<f64>,
        psi_norm: &Array1<f64>,
        d_area: f64,
    ) {
        self.equilibrium_ids.allocate_time_slices(times_to_reconstruct);

        for time_slice in self.equilibrium_ids.time_slice.iter_mut() {
            // GSFit solves on a single rectangular (R, Z) grid, so there is exactly one entry in
            // this array of structures
            let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();

            // `rectangular` is index 1 of the data dictionary's poloidal plane coordinates
            // enumeration: "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)"
            profiles_2d.grid_type.name = Some("rectangular".to_string());
            profiles_2d.grid_type.index = Some(1);
            profiles_2d.grid_type.description = Some("Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".to_string());

            profiles_2d.grid.dim1 = Some(r.to_owned());
            profiles_2d.grid.dim2 = Some(z.to_owned());
            profiles_2d.grid.d_area = Some(d_area);
            profiles_2d.r = Some(mesh_r.to_owned());
            profiles_2d.z = Some(mesh_z.to_owned());
            time_slice.profiles_2d = vec![profiles_2d];

            // The psi_norm grid the source functions are defined on
            time_slice.profiles_1d.psi_norm = Some(psi_norm.to_owned());
        }
    }

    /// Gather one Green's quantity, for every passive degree of freedom, into a matrix over the grid.
    ///
    /// The degrees of freedom are laid out in the order they appear in `greens/pf_passive`, which is
    /// the order the Green's tables were built in.
    ///
    /// # Arguments
    /// * `select` - picks the quantity out of one degree of freedom's Green's tables
    ///
    /// # Returns
    /// * `greens_with_passives` - shape `(n_z * n_r, n_dof_total)`
    fn greens_passive_grid(&self, select: fn(&EquilibriumGreensPfPassiveDof) -> &Option<Array1<f64>>) -> Array2<f64> {
        // Unpack from self
        let n_dof_total: usize = self.equilibrium_ids.greens.pf_passive.iter().map(|pf_passive| pf_passive.dof.len()).sum();
        let n_r: usize = self.equilibrium_ids.code.grid.n_r.unwrap() as usize;
        let n_z: usize = self.equilibrium_ids.code.grid.n_z.unwrap() as usize;

        let mut greens_with_passives: Array2<f64> = Array2::from_elem((n_z * n_r, n_dof_total), f64::NAN);

        let mut i_dof_total: usize = 0;
        for pf_passive in &self.equilibrium_ids.greens.pf_passive {
            for dof in &pf_passive.dof {
                greens_with_passives.slice_mut(s![.., i_dof_total]).assign(select(dof).as_ref().unwrap());
                i_dof_total += 1;
            }
        }

        return greens_with_passives;
    }

    /// Green's table for the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.psi);
    }

    /// Green's table for the second radial derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d2_psi_d_r2(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d2_psi_d_r2);
    }

    /// Green's table for the second vertical derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d2_psi_d_z2(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d2_psi_d_z2);
    }

    /// Green's table for the third derivative of the poloidal flux, twice by R and once by Z, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d3_psi_d_r2_d_z(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d3_psi_d_r2_d_z);
    }

    /// Green's table for the third derivative of the poloidal flux, once by R and twice by Z, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d3_psi_d_r_d_z2(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d3_psi_d_r_d_z2);
    }

    /// Green's table for the third vertical derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d3_psi_d_z3(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d3_psi_d_z3);
    }

    /// Green's table for the radial derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d_psi_d_r(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d_psi_d_r);
    }

    /// Green's table for the vertical derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d_psi_d_z(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d_psi_d_z);
    }

    /// Green's table for the mixed second derivative of the poloidal flux, for every passive degree of freedom
    ///
    /// # Returns
    /// * shape `(n_z * n_r, n_dof_total)`
    pub fn get_greens_passive_grid_d2_psi_d_r_d_z(&self) -> Array2<f64> {
        return self.greens_passive_grid(|dof| &dof.d2_psi_d_r_d_z);
    }
}

/// Apply every degree of freedom's current distribution to a Green's table, and sum over the passive
/// filaments.
///
/// The straightforward way to write this is one pass per degree of freedom:
/// ```ignore
/// let with_dof: Array2<f64> = &greens_filaments * &current_distribution;  // a table-sized temporary
/// let g: Array1<f64> = with_dof.sum_axis(Axis(1));
/// ```
/// which re-reads the whole `[n_grid, n_filament]` table once per degree of freedom, and allocates
/// and streams a table-sized temporary each time. For ST40's inner vessel that is 15 degrees of
/// freedom against a 56 MB table, for each of 13 tables.
///
/// This does all the degrees of freedom in a single pass over the table instead: one grid point's
/// row is loaded, and every degree of freedom is applied to it while it is still in cache. Nothing
/// table-sized is allocated.
///
/// The arithmetic is unchanged. The products are formed in the same order and handed to the same
/// `ndarray` summation that `sum_axis` used, so the answer is identical bit for bit.
///
/// # Arguments
/// * `greens_filaments` - Green's table between the grid and this passive's filaments, shape = [n_grid, n_filament]
/// * `current_distributions` - one current distribution per degree of freedom, each of length `n_filament`, [dimensionless]
///
/// # Returns
/// * `greens_with_dof` - shape = [n_grid, n_dof]
fn apply_current_distributions(greens_filaments: &Array2<f64>, current_distributions: &[Array1<f64>]) -> Array2<f64> {
    let n_grid: usize = greens_filaments.nrows();
    let n_filament: usize = greens_filaments.ncols();
    let n_dof: usize = current_distributions.len();

    let mut greens_with_dof: Array2<f64> = Array2::from_elem((n_grid, n_dof), f64::NAN);
    if n_dof == 0 || n_filament == 0 || n_grid == 0 {
        return greens_with_dof;
    }

    // Both arrays are freshly allocated, and so are contiguous and row-major
    let greens_filaments_flat: &[f64] = greens_filaments
        .as_slice()
        .expect("plasma.apply_current_distributions: `greens_filaments` must be contiguous");
    let mut current_distributions_flat: Vec<&[f64]> = Vec::with_capacity(n_dof);
    for i_dof in 0..n_dof {
        current_distributions_flat.push(
            current_distributions[i_dof]
                .as_slice()
                .expect("plasma.apply_current_distributions: `current_distribution` must be contiguous"),
        );
    }

    // One rayon task per block of grid points, rather than per grid point, so that the work in a
    // task is large compared with the cost of handing it to a worker thread
    const N_GRID_PER_TASK: usize = 64;

    {
        let greens_with_dof_flat: &mut [f64] = greens_with_dof
            .as_slice_mut()
            .expect("plasma.apply_current_distributions: `greens_with_dof` must be contiguous");

        greens_filaments_flat
            .par_chunks(N_GRID_PER_TASK * n_filament)
            .zip(greens_with_dof_flat.par_chunks_mut(N_GRID_PER_TASK * n_dof))
            .for_each(|(greens_filaments_block, greens_with_dof_block): (&[f64], &mut [f64])| {
                let n_grid_in_block: usize = greens_filaments_block.len() / n_filament;

                // Scratch space for one grid point's row, re-used by every degree of freedom and
                // every grid point in this block
                let mut products: Array1<f64> = Array1::from_elem(n_filament, f64::NAN);

                for i_grid_in_block in 0..n_grid_in_block {
                    let greens_filaments_row: &[f64] = &greens_filaments_block[i_grid_in_block * n_filament..(i_grid_in_block + 1) * n_filament];

                    for i_dof in 0..n_dof {
                        let current_distribution: &[f64] = current_distributions_flat[i_dof];
                        for i_filament in 0..n_filament {
                            products[i_filament] = greens_filaments_row[i_filament] * current_distribution[i_filament];
                        }

                        // `Array1::sum()` on contiguous data is the same eight-accumulator reduction
                        // that `sum_axis(Axis(1))` applied to each row, so this is bit-identical
                        greens_with_dof_block[i_grid_in_block * n_dof + i_dof] = products.sum();
                    }
                }
            });
    }

    return greens_with_dof;
}
