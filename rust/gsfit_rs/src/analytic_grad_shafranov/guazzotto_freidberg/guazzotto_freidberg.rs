use super::Configuration;
use super::coils_fit::{BoundaryCoilFit, fit_boundary_coils, plasma_current_density, plasma_current_filaments};
use super::eigenvalue::{flux_solution, solve_alpha};
use super::flux::{FluxDerivatives, FluxSolution};
use super::matching_geometry::{MatchingGeometry, matching_geometry};
use crate::coils::Coils;
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::grad_shafranov::CONVERGENCE_STATUS_CONVERGED;
use crate::grad_shafranov::contour_tree_nodes as grad_shafranov_contour_tree_nodes;
use crate::greens::Greens;
use crate::plasma_geometry::{BoundaryContour, MagneticAxis, StationaryPoint, find_boundary, find_magnetic_axis, find_stationary_points_using_winding_number};
use crate::sensors::{BpProbes, FluxLoops};
use crate::source_functions::{EfitPolynomial, SharedSourceFunction};
use crate::wall::{Wall, limiter_points, vacuum_vessel_outline};
use imas_rs::ids::wall::Wall as WallIds;
use imas_rs::python::PyEquilibrium;
use imas_rs::{Equilibrium, EquilibriumContourTreeNode, EquilibriumProfiles2d, EquilibriumTimeSlice};
use ndarray::{Array1, Array2, array};
use numpy::borrow::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Number of points in `profiles_1d/psi_norm`, evenly spaced from the magnetic axis to the boundary
const N_PSI_NORM: usize = 65;

/// Maximum number of Newton iterations when locating the magnetic axis
const N_ITER_MAGNETIC_AXIS_MAX: usize = 50;

/// Newton step, in the normalised coordinates `x`, `y`, below which the magnetic axis is converged
const MAGNETIC_AXIS_TOLERANCE: f64 = 1.0e-12;

/// The data dictionary's `boundary/type`
const BOUNDARY_TYPE_LIMITED: i32 = 0;

/// The data dictionary's `contour_tree/node/critical_type`
const CRITICAL_TYPE_MAXIMUM: i32 = 2;

/// Guazzotto–Freidberg analytic Grad–Shafranov equilibrium.
///
/// Reference: L. Guazzotto & J. P. Freidberg, "Simple, general, realistic, robust,
/// analytic tokamak equilibria. Part 1", J. Plasma Phys. 87 (2021) 905870303.
///
/// The equilibrium is solved analytically; a [`Coils`] object is then produced whose
/// PF-coil currents (placed on the `(R, Z)` grid boundary) reproduce the vacuum field
/// consistent with the analytic solution. This yields a self-consistent free-boundary
/// test case that GSFit can be asked to reconstruct.
///
/// # Flux convention
///
/// GF write the flux per radian, `Psi = R * A_phi`. Everything stored here follows the IMAS
/// convention instead, the total poloidal flux `psi = 2 * pi * R * A_phi` [weber], so `psi_0` and the
/// equilibrium IDS are both `2 * pi` times GF's values, and `dpressure_dpsi` and `f_df_dpsi` are
/// `1 / (2 * pi)` times GF's `p'` and `ff'`.
///
/// # Normalised flux
///
/// GF's normalised flux, `psi_hat = psi / psi_0`, is 1 at `(x, y) = (0, 0)` and 0 on the plasma
/// boundary. It is **not** the IMAS `psi_norm`, which is 0 at the magnetic axis and 1 on the
/// boundary.
#[derive(Clone)]
#[pyclass(module = "gsfit_rs.analytic_grad_shafranov.guazzotto_freidberg", skip_from_py_object)]
pub struct GuazzottoFreidberg {
    // Inputs
    /// Boundary topology and its shape parameters
    #[pyo3(get)]
    configuration: Configuration,
    /// Inverse aspect ratio, `a / r_geo` [dimensionless]
    #[pyo3(get)]
    eps: f64,
    /// Profile parameter, approximately the poloidal beta [dimensionless]
    #[pyo3(get)]
    nu: f64,
    /// Geometric major radius [metre]
    #[pyo3(get)]
    r_geo: f64,
    /// Vacuum toroidal field at `r_geo` [tesla]
    #[pyo3(get)]
    bt_vac_at_r_geo: f64,
    /// Pressure normalisation, the pressure where `psi_hat = 1` [pascal]
    #[pyo3(get)]
    p_axis: f64,
    /// Number of radial grid points [dimensionless]
    #[pyo3(get)]
    n_r: usize,
    /// Number of vertical grid points [dimensionless]
    #[pyo3(get)]
    n_z: usize,
    /// Minimum radius of the grid [metre]
    #[pyo3(get)]
    r_min: f64,
    /// Maximum radius of the grid [metre]
    #[pyo3(get)]
    r_max: f64,
    /// Minimum vertical position of the grid [metre]
    #[pyo3(get)]
    z_min: f64,
    /// Maximum vertical position of the grid [metre]
    #[pyo3(get)]
    z_max: f64,
    /// Number of terms in the `C_n(x)` / `S_n(x)` series (the paper's `M`) [dimensionless]
    #[pyo3(get)]
    n_radial_expansion: usize,

    // Results of `solve`, NaN or empty until it has run
    /// The eigenvalue [dimensionless]
    #[pyo3(get)]
    alpha: f64,
    /// Beta at `psi_hat = 1`, `2 * mu_0 * p_axis / bt_vac_at_r_geo ** 2` (GF Eq. 6.4) [dimensionless]
    #[pyo3(get)]
    beta_0: f64,
    /// Diamagnetic change in the toroidal field (GF Eq. 6.4) [tesla]
    #[pyo3(get)]
    bt_diamagnetic_shift: f64,
    /// Flux normalisation, `psi = psi_0 * psi_hat`: GF's `Psi_0` (Eq. 6.5) times `2 * pi` [weber]
    #[pyo3(get)]
    psi_0: f64,
    /// Analytic flux, `psi_0 * psi_hat`, on every grid cell, shape `(n_z, n_r)` [weber].
    /// Outside the model box `|x| < 1`, `y_lower < y < y_upper` the truncated series is unphysical and
    /// can diverge; it is kept there anyway, so that it can be plotted for comparison
    psi_analytic: Array2<f64>,
    /// 1 inside the analytic plasma, which is the model box where `psi_hat > 0`, and 0 elsewhere;
    /// shape `(n_z, n_r)` [dimensionless]
    mask: Array2<f64>,

    // Result of `get_coils`
    /// The free-boundary equilibrium, post-processed; empty until `get_coils` has run
    equilibrium_ids: Equilibrium,
}

#[pymethods]
impl GuazzottoFreidberg {
    /// Construct a Guazzotto–Freidberg equilibrium (does not solve; call `solve`).
    ///
    /// # Configuration
    /// * `configuration` - boundary topology and its shape parameters:
    ///   `SymmetricLimited`, `AntisymmetricLimited`, `DoubleNull`, or `SingleNull`
    ///
    /// # Shape / profile inputs (dimensionless)
    /// * `eps` - inverse aspect ratio, `a / r_geo`
    /// * `nu` - profile parameter, approximately the poloidal beta
    ///
    /// # Dimensional inputs
    /// * `r_geo` - geometric major radius [metre]
    /// * `bt_vac_at_r_geo` - vacuum toroidal field at `r_geo` [tesla]
    /// * `p_axis` - pressure on the magnetic axis [pascal]
    ///
    /// # `(R, Z)` grid (PF coils are placed on its boundary)
    /// * `n_r` - number of radial grid points
    /// * `n_z` - number of vertical grid points
    /// * `r_min` - minimum radius [metre]
    /// * `r_max` - maximum radius [metre]
    /// * `z_min` - minimum vertical position [metre]
    /// * `z_max` - maximum vertical position [metre]
    ///
    /// # Numerical controls
    /// * `n_radial_expansion` - number of terms in the `C_n(x)` / `S_n(x)` series (the paper's `M`)
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        configuration: Configuration,
        eps: f64,
        nu: f64,
        r_geo: f64,
        bt_vac_at_r_geo: f64,
        p_axis: f64,
        n_r: usize,
        n_z: usize,
        r_min: f64,
        r_max: f64,
        z_min: f64,
        z_max: f64,
        n_radial_expansion: usize,
    ) -> Self {
        GuazzottoFreidberg {
            configuration,
            eps,
            nu,
            r_geo,
            bt_vac_at_r_geo,
            p_axis,
            n_r,
            n_z,
            r_min,
            r_max,
            z_min,
            z_max,
            n_radial_expansion,
            alpha: f64::NAN,
            beta_0: f64::NAN,
            bt_diamagnetic_shift: f64::NAN,
            psi_0: f64::NAN,
            psi_analytic: Array2::zeros((0, 0)),
            mask: Array2::zeros((0, 0)),
            equilibrium_ids: Equilibrium::default(),
        }
    }

    /// Solve the eigenvalue `alpha` and evaluate the analytic flux and plasma mask over the `(R, Z)` grid.
    pub fn solve(&mut self) {
        let eps: f64 = self.eps;
        let nu: f64 = self.nu;
        let r_geo: f64 = self.r_geo;
        let bt_vac_at_r_geo: f64 = self.bt_vac_at_r_geo;
        let p_axis: f64 = self.p_axis;
        let n_r: usize = self.n_r;
        let n_z: usize = self.n_z;

        // 1. Topology-specific matching geometry and the eigenvalue alpha.
        let geom: MatchingGeometry = matching_geometry(&self.configuration, eps).expect("GuazzottoFreidberg.solve");
        self.alpha = solve_alpha(eps, nu, &geom, self.n_radial_expansion, 1.0, 3.0);

        // 2. Final expansion coefficients and flux solution.
        let flux: FluxSolution = self.flux_solution(&geom);

        // 3. Dimensional scalars (GF Eqs. 6.4, 6.5). GF's `Psi_0` is per radian, so it is multiplied
        //    by `2 * pi` to give the total flux
        self.beta_0 = 2.0 * MU_0 * p_axis / (bt_vac_at_r_geo * bt_vac_at_r_geo);
        self.bt_diamagnetic_shift = bt_vac_at_r_geo * 0.5 * self.beta_0 * (1.0 + eps * eps) * (1.0 - nu) / nu;
        self.psi_0 = 2.0 * PI * eps * bt_vac_at_r_geo * r_geo * r_geo / self.alpha * (self.beta_0 / nu).sqrt();

        // 4. Guard: the requested grid must contain the model plasma. The GF model surface spans
        //    R in [R0(1 - eps), R0(1 + eps)] and Z in [a * y_lower, a * y_upper] (maxima / X-points).
        let a_minor: f64 = eps * r_geo;
        let plasma_r_inner: f64 = r_geo * (1.0 - eps);
        let plasma_r_outer: f64 = r_geo * (1.0 + eps);
        let plasma_z_lower: f64 = a_minor * geom.y_lower();
        let plasma_z_upper: f64 = a_minor * geom.y_upper();
        assert!(
            self.r_min <= plasma_r_inner && self.r_max >= plasma_r_outer && self.z_min <= plasma_z_lower && self.z_max >= plasma_z_upper,
            "GuazzottoFreidberg.solve: plasma (R in [{plasma_r_inner:.4}, {plasma_r_outer:.4}] m, Z in [{plasma_z_lower:.4}, {plasma_z_upper:.4}] m) is not contained in the requested grid (R in [{:.4}, {:.4}] m, Z in [{:.4}, {:.4}] m); enlarge the r_min/r_max/z_min/z_max bounds",
            self.r_min,
            self.r_max,
            self.z_min,
            self.z_max,
        );

        // 5. The analytic flux on every cell, and the plasma mask. The plasma lies inside the
        //    model box |x| < 1, y_lower < y < y_upper, where the C_n / S_n series is reliable. Outside
        //    it the truncated series is unphysical (it can diverge near |x| = 1 / eps_hat or oscillate
        //    beyond the maxima / X-points), so those points are never part of the plasma. The box also
        //    cuts off the private flux region below an X-point, where psi_hat > 0 too.
        let (r, z): (Array1<f64>, Array1<f64>) = self.grid();
        let mut psi_analytic: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut mask: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let (x, y): (f64, f64) = self.normalised_coordinates(r[i_r], z[i_z]);
                let psi_hat: f64 = flux.eval_all(x, y).psi_hat;
                psi_analytic[(i_z, i_r)] = self.psi_0 * psi_hat;
                if geom.is_in_model_box(x, y) && psi_hat > 0.0 {
                    mask[(i_z, i_r)] = 1.0;
                }
            }
        }
        self.psi_analytic = psi_analytic;
        self.mask = mask;
    }

    /// Fit the PF-coil currents on the `(R, Z)` grid boundary so their vacuum field, added to the
    /// plasma's own field, reproduces the analytic flux at the `control_points`. Then store the
    /// resulting free-boundary equilibrium in the equilibrium IDS, and post-process it. Requires
    /// `solve` first.
    ///
    /// The IDS holds the total flux, plasma plus coils, which is valid in the vacuum as well as
    /// inside the plasma. For a diverted plasma the magnetic axis, the X-points, the boundary flux
    /// and the plasma mask are found in that flux, as the Grad-Shafranov solver finds them; for a
    /// limited one they are the analytic ones. See `build_equilibrium_ids`.
    ///
    /// # Arguments
    /// * `coil_regularisation_weight` - Tikhonov weight `lambda^2` on the coil-current magnitude
    /// * `control_points` - `(n_control_point, 2)` array of `(r, z)` points where the flux is matched [metre]
    /// * `wall` - the wall: its limiter and vacuum vessel bound a diverted plasma, and the post-processor traces
    ///   the scrape-off layer legs up to it
    ///
    /// # Returns
    /// * `coils` - one single-filament PF coil per boundary node, carrying the fitted current
    pub fn get_coils(&mut self, coil_regularisation_weight: f64, control_points: PyReadonlyArray2<f64>, wall: PyRef<Wall>) -> Coils {
        self.assert_solved("get_coils");

        let control_points_ndarray: Array2<f64> = control_points.to_owned_array();
        assert!(
            control_points_ndarray.ncols() == 2,
            "GuazzottoFreidberg.get_coils: control_points must have shape (n_control_point, 2)"
        );
        let control_r: Array1<f64> = control_points_ndarray.column(0).to_owned();
        let control_z: Array1<f64> = control_points_ndarray.column(1).to_owned();
        let n_control: usize = control_r.len();

        // Re-derive the flux solution (alpha is known, so this is cheap) to evaluate the target flux.
        let geom: MatchingGeometry = matching_geometry(&self.configuration, self.eps).expect("GuazzottoFreidberg.get_coils");
        let flux: FluxSolution = self.flux_solution(&geom);
        let (r, z): (Array1<f64>, Array1<f64>) = self.grid();

        // Target flux at the control points: psi_0 * psi_hat inside the model box, 0 on the LCFS and outside.
        let mut psi_control: Array1<f64> = Array1::zeros(n_control);
        for i_control in 0..n_control {
            let (x, y): (f64, f64) = self.normalised_coordinates(control_r[i_control], control_z[i_control]);
            if geom.is_in_model_box(x, y) {
                psi_control[i_control] = self.psi_0 * flux.eval_all(x, y).psi_hat;
            }
        }

        // Fit the boundary coil currents and recompute the total flux (plasma + coils).
        let j_phi_2d: Array2<f64> = self.plasma_current_density();
        let fit: BoundaryCoilFit = fit_boundary_coils(
            &r,
            &z,
            &j_phi_2d,
            &self.mask,
            &control_r,
            &control_z,
            &psi_control,
            coil_regularisation_weight,
        );

        // Assemble the Coils object: one single-filament PF coil per boundary node.
        let mut coils: Coils = Coils::new();
        let n_coil: usize = fit.coil_r.len();
        let time_single: Array1<f64> = Array1::from_elem(1, 0.0);
        for i_coil in 0..n_coil {
            let name: String = format!("GF_BND_{i_coil:04}");
            let r_single: Array1<f64> = Array1::from_elem(1, fit.coil_r[i_coil]);
            let z_single: Array1<f64> = Array1::from_elem(1, fit.coil_z[i_coil]);
            let d_r_single: Array1<f64> = Array1::from_elem(1, fit.coil_d_r[i_coil]);
            let d_z_single: Array1<f64> = Array1::from_elem(1, fit.coil_d_z[i_coil]);
            let current_single: Array1<f64> = Array1::from_elem(1, fit.coil_current[i_coil]);
            coils.add_pf_coil_rs(&name, &r_single, &z_single, &d_r_single, &d_z_single, &time_single, &current_single);
        }

        // Store the free-boundary equilibrium, and post-process it exactly as a reconstruction is
        self.equilibrium_ids = self.build_equilibrium_ids(&flux, &geom, &r, &z, fit, j_phi_2d, &wall.wall_ids);
        let (p_prime_source_function, ff_prime_source_function): (SharedSourceFunction, SharedSourceFunction) = source_functions();
        equilibrium_post_processor(&mut self.equilibrium_ids, &wall.wall_ids, &p_prime_source_function, &ff_prime_source_function);

        coils
    }

    /// Fill the `experimental` measurement of each BP probe (tesla) and flux loop (weber) with the
    /// field the analytic plasma current plus `coils` produce at that sensor, mutating the passed
    /// `bp_probes` / `flux_loops` in place. Requires `solve` first. The sensors' geometry must
    /// already be set (e.g. via `add_sensor`).
    ///
    /// Each coil filament carries its coil's current at the first time of its experimental
    /// timebase.
    ///
    /// The values are physical: flux loops read `Greens::psi` (= `2 pi R A_phi`, weber) and BP
    /// probes read `B_r cos(angle_pol) + B_z sin(angle_pol)` (tesla), mirroring GSFit's own
    /// `calculate_sensor_values`.
    ///
    /// # Arguments
    /// * `bp_probes` - BP probes with geometry already set; their measurements are overwritten
    /// * `flux_loops` - flux loops with geometry already set; their measurements are overwritten
    /// * `coils` - the PF coils, typically the ones `get_coils` returned
    pub fn get_sensor_values(&self, mut bp_probes: PyRefMut<BpProbes>, mut flux_loops: PyRefMut<FluxLoops>, coils: PyRef<Coils>) {
        self.assert_solved("get_sensor_values");

        // Plasma current filaments + the coils are the free-boundary field the sensors see.
        let (r, z): (Array1<f64>, Array1<f64>) = self.grid();
        let j_phi_2d: Array2<f64> = self.plasma_current_density();
        let (cell_r, cell_z, cell_current): (Array1<f64>, Array1<f64>, Array1<f64>) = plasma_current_filaments(&r, &z, &j_phi_2d, &self.mask);
        let n_cell: usize = cell_r.len();

        // Every filament of every PF coil, each carrying its coil's current
        let mut coil_r_vec: Vec<f64> = Vec::new();
        let mut coil_z_vec: Vec<f64> = Vec::new();
        let mut coil_current_vec: Vec<f64> = Vec::new();
        for coil_name in coils.results.get("pf").keys() {
            let filament_r: Array1<f64> = coils.results.get("pf").get(&coil_name).get("geometry").get("r").unwrap_array1();
            let filament_z: Array1<f64> = coils.results.get("pf").get(&coil_name).get("geometry").get("z").unwrap_array1();
            let coil_current: f64 = coils.results.get("pf").get(&coil_name).get("i").get("experimental").get("value").unwrap_array1()[0];
            let n_filament: usize = filament_r.len();
            for i_filament in 0..n_filament {
                coil_r_vec.push(filament_r[i_filament]);
                coil_z_vec.push(filament_z[i_filament]);
                coil_current_vec.push(coil_current);
            }
        }
        let coil_r: Array1<f64> = Array1::from(coil_r_vec);
        let coil_z: Array1<f64> = Array1::from(coil_z_vec);
        let coil_current: Array1<f64> = Array1::from(coil_current_vec);
        let n_coil: usize = coil_r.len();

        // Sensors sit away from the current sources, so treat sources as point filaments (d_r = d_z = 0),
        // matching GSFit's own sensor path.
        let cell_d_r: Array1<f64> = Array1::zeros(n_cell);
        let cell_d_z: Array1<f64> = Array1::zeros(n_cell);
        let coil_d_r: Array1<f64> = Array1::zeros(n_coil);
        let coil_d_z: Array1<f64> = Array1::zeros(n_coil);

        // --- Flux loops: measured psi = Greens::psi() @ (plasma cells + coils)  [weber] ---
        let flux_loop_names: Vec<String> = flux_loops.results.keys();
        let n_flux_loop: usize = flux_loop_names.len();
        if n_flux_loop > 0 {
            let mut flux_loop_r: Array1<f64> = Array1::zeros(n_flux_loop);
            let mut flux_loop_z: Array1<f64> = Array1::zeros(n_flux_loop);
            for i_flux_loop in 0..n_flux_loop {
                flux_loop_r[i_flux_loop] = flux_loops.results.get(&flux_loop_names[i_flux_loop]).get("geometry").get("r").unwrap_f64();
                flux_loop_z[i_flux_loop] = flux_loops.results.get(&flux_loop_names[i_flux_loop]).get("geometry").get("z").unwrap_f64();
            }
            let psi_from_cells: Array1<f64> = Greens::sensor_to_conductor(
                flux_loop_r.clone(),
                flux_loop_z.clone(),
                cell_r.clone(),
                cell_z.clone(),
                cell_d_r.clone(),
                cell_d_z.clone(),
            )
            .psi()
            .dot(&cell_current);
            let psi_from_coils: Array1<f64> = Greens::sensor_to_conductor(
                flux_loop_r.clone(),
                flux_loop_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
                coil_d_r.clone(),
                coil_d_z.clone(),
            )
            .psi()
            .dot(&coil_current);
            for i_flux_loop in 0..n_flux_loop {
                let value: Array1<f64> = Array1::from_elem(1, psi_from_cells[i_flux_loop] + psi_from_coils[i_flux_loop]);
                flux_loops
                    .results
                    .get_or_insert(&flux_loop_names[i_flux_loop])
                    .get_or_insert("psi")
                    .get_or_insert("experimental")
                    .insert("value", value);
            }
        }

        // --- BP probes: measured B = B_r cos(angle_pol) + B_z sin(angle_pol)  [tesla] ---
        let bp_probe_names: Vec<String> = bp_probes.results.keys();
        let n_bp_probe: usize = bp_probe_names.len();
        if n_bp_probe > 0 {
            let mut bp_probe_r: Array1<f64> = Array1::zeros(n_bp_probe);
            let mut bp_probe_z: Array1<f64> = Array1::zeros(n_bp_probe);
            let mut bp_probe_angle_pol: Array1<f64> = Array1::zeros(n_bp_probe);
            for i_bp_probe in 0..n_bp_probe {
                bp_probe_r[i_bp_probe] = bp_probes.results.get(&bp_probe_names[i_bp_probe]).get("geometry").get("r").unwrap_f64();
                bp_probe_z[i_bp_probe] = bp_probes.results.get(&bp_probe_names[i_bp_probe]).get("geometry").get("z").unwrap_f64();
                bp_probe_angle_pol[i_bp_probe] = bp_probes.results.get(&bp_probe_names[i_bp_probe]).get("geometry").get("angle_pol").unwrap_f64();
            }
            let greens_cells: Greens = Greens::sensor_to_conductor(
                bp_probe_r.clone(),
                bp_probe_z.clone(),
                cell_r.clone(),
                cell_z.clone(),
                cell_d_r.clone(),
                cell_d_z.clone(),
            );
            let greens_coils: Greens = Greens::sensor_to_conductor(
                bp_probe_r.clone(),
                bp_probe_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
                coil_d_r.clone(),
                coil_d_z.clone(),
            );
            let b_r_total: Array1<f64> = greens_cells.b_r().dot(&cell_current) + greens_coils.b_r().dot(&coil_current);
            let b_z_total: Array1<f64> = greens_cells.b_z().dot(&cell_current) + greens_coils.b_z().dot(&coil_current);
            for i_bp_probe in 0..n_bp_probe {
                let value: f64 = b_r_total[i_bp_probe] * bp_probe_angle_pol[i_bp_probe].cos() + b_z_total[i_bp_probe] * bp_probe_angle_pol[i_bp_probe].sin();
                let value_array: Array1<f64> = Array1::from_elem(1, value);
                bp_probes
                    .results
                    .get_or_insert(&bp_probe_names[i_bp_probe])
                    .get_or_insert("b")
                    .get_or_insert("experimental")
                    .insert("value", value_array);
            }
        }
    }

    /// GF's model surface, the shape whose position, slope and curvature the analytic boundary is
    /// matched to at the matching points: the red curve in GF's figures. Elsewhere the analytic
    /// boundary is close to the model surface, but not on it. Does not need `solve`.
    ///
    /// The upper half is sampled uniformly in GF's angle-like parameter `theta`, from the outer
    /// midplane over the top to the inner midplane, and the lower half likewise back to the outer
    /// midplane. A half with an X-point is two ellipses, each sampled with half of the half's points.
    ///
    /// # Arguments
    /// * `n_point_per_half` - number of points on each of the upper and lower halves; must be even
    ///
    /// # Returns
    /// * `(r, z)` - the closed outline, anticlockwise from the outer midplane; its last point repeats
    ///   the first, so each is `2 * n_point_per_half + 1` long [metre]
    fn model_surface<'py>(&self, py: Python<'py>, n_point_per_half: usize) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let geom: MatchingGeometry = matching_geometry(&self.configuration, self.eps).expect("GuazzottoFreidberg.model_surface");
        let (upper_x, upper_y): (Array1<f64>, Array1<f64>) = geom.upper_half.outline(true, n_point_per_half);
        let (lower_x, lower_y): (Array1<f64>, Array1<f64>) = geom.lower_half.outline(false, n_point_per_half);

        let n_point: usize = 2 * n_point_per_half + 1;
        let mut r: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut z: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        for i_point in 0..n_point_per_half {
            (r[i_point], z[i_point]) = self.physical_coordinates(upper_x[i_point], upper_y[i_point]);
            (r[n_point_per_half + i_point], z[n_point_per_half + i_point]) = self.physical_coordinates(lower_x[i_point], lower_y[i_point]);
        }
        r[n_point - 1] = r[0];
        z[n_point - 1] = z[0];
        (r.into_pyarray(py), z.into_pyarray(py))
    }

    /// The analytic flux, `psi_0 * psi_hat`, on every grid cell, shape `(n_z, n_r)` [weber].
    ///
    /// Outside the model box the truncated series is unphysical and can diverge; use `mask` to
    /// select the plasma.
    #[getter]
    fn psi_analytic<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.psi_analytic.clone().into_pyarray(py)
    }

    /// 1 inside the analytic plasma and 0 elsewhere, shape `(n_z, n_r)` [dimensionless].
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mask.clone().into_pyarray(py)
    }

    /// The equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`. Empty until
    /// `get_coils` has run.
    ///
    /// The IDS is copied into the returned object, so it is a snapshot: changes made on the Rust
    /// side afterwards are not seen by it.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        PyEquilibrium::new(self.equilibrium_ids.clone())
    }
}

// Rust only methods
impl GuazzottoFreidberg {
    /// Panic with a helpful message when a method needing `solve` is called before it.
    fn assert_solved(&self, method_name: &str) {
        assert!(!self.alpha.is_nan(), "GuazzottoFreidberg.{method_name}: call solve() before {method_name}()");
    }

    /// The grid axes, `r` and `z` [metre].
    fn grid(&self) -> (Array1<f64>, Array1<f64>) {
        let r: Array1<f64> = Array1::linspace(self.r_min, self.r_max, self.n_r);
        let z: Array1<f64> = Array1::linspace(self.z_min, self.z_max, self.n_z);
        (r, z)
    }

    /// GF's normalised coordinates of a point: `x = (R ** 2 / r_geo ** 2 - 1 - eps ** 2) / (2 * eps)`, `y = Z / a`.
    fn normalised_coordinates(&self, r: f64, z: f64) -> (f64, f64) {
        let x: f64 = (r * r / (self.r_geo * self.r_geo) - 1.0 - self.eps * self.eps) / (2.0 * self.eps);
        let y: f64 = z / (self.eps * self.r_geo);
        (x, y)
    }

    /// The inverse of `normalised_coordinates`: `(R, Z)` [metre] of a point `(x, y)`.
    fn physical_coordinates(&self, x: f64, y: f64) -> (f64, f64) {
        let r: f64 = self.r_geo * (1.0 + self.eps * self.eps + 2.0 * self.eps * x).sqrt();
        let z: f64 = self.eps * self.r_geo * y;
        (r, z)
    }

    /// The flux solution for the solved `alpha`.
    fn flux_solution(&self, geom: &MatchingGeometry) -> FluxSolution {
        flux_solution(self.alpha, self.eps, self.nu, geom, self.n_radial_expansion).expect("GuazzottoFreidberg: singular constraint system")
    }

    /// The analytic plasma's toroidal current density on the grid, shape `(n_z, n_r)` [ampere / metre ** 2].
    fn plasma_current_density(&self) -> Array2<f64> {
        let (r, _z): (Array1<f64>, Array1<f64>) = self.grid();
        let psi_hat_2d: Array2<f64> = &self.psi_analytic / self.psi_0;
        plasma_current_density(
            &r,
            &psi_hat_2d,
            &self.mask,
            self.psi_0,
            self.p_axis,
            self.r_geo,
            self.bt_vac_at_r_geo,
            self.bt_diamagnetic_shift,
        )
    }

    /// The analytic magnetic axis, the maximum of `psi_hat`, found by Newton iteration on
    /// `grad(psi_hat) = 0` from the plasma's highest grid cell.
    ///
    /// # Returns
    /// * `(x, y, psi_hat)` - the axis in normalised coordinates, and the normalised flux there [dimensionless]
    fn magnetic_axis(&self, flux: &FluxSolution) -> (f64, f64, f64) {
        // Start from the highest cell inside the plasma
        let (r, z): (Array1<f64>, Array1<f64>) = self.grid();
        let mut psi_max: f64 = f64::NEG_INFINITY;
        let mut r_start: f64 = f64::NAN;
        let mut z_start: f64 = f64::NAN;
        for i_z in 0..self.n_z {
            for i_r in 0..self.n_r {
                if self.mask[(i_z, i_r)] > 0.0 && self.psi_analytic[(i_z, i_r)] > psi_max {
                    psi_max = self.psi_analytic[(i_z, i_r)];
                    r_start = r[i_r];
                    z_start = z[i_z];
                }
            }
        }
        let (mut x, mut y): (f64, f64) = self.normalised_coordinates(r_start, z_start);

        // The map from (R, Z) to (x, y) is one-to-one, so the stationary point is the same in both
        let mut converged: bool = false;
        'newton_loop: for _i_iter in 0..N_ITER_MAGNETIC_AXIS_MAX {
            let derivatives: FluxDerivatives = flux.eval_all(x, y);
            let hessian_determinant: f64 =
                derivatives.d2_psi_hat_d_x2 * derivatives.d2_psi_hat_d_y2 - derivatives.d2_psi_hat_d_x_d_y * derivatives.d2_psi_hat_d_x_d_y;
            let step_x: f64 = (derivatives.d2_psi_hat_d_y2 * derivatives.d_psi_hat_d_x - derivatives.d2_psi_hat_d_x_d_y * derivatives.d_psi_hat_d_y) / hessian_determinant;
            let step_y: f64 = (derivatives.d2_psi_hat_d_x2 * derivatives.d_psi_hat_d_y - derivatives.d2_psi_hat_d_x_d_y * derivatives.d_psi_hat_d_x) / hessian_determinant;
            x -= step_x;
            y -= step_y;
            if step_x.hypot(step_y) < MAGNETIC_AXIS_TOLERANCE {
                converged = true;
                break 'newton_loop;
            }
        }
        assert!(
            converged,
            "GuazzottoFreidberg.magnetic_axis: Newton iteration did not converge within {N_ITER_MAGNETIC_AXIS_MAX} iterations"
        );

        (x, y, flux.eval_all(x, y).psi_hat)
    }

    /// Build the equilibrium IDS for the free-boundary equilibrium: one time-slice, at `time = 0`.
    ///
    /// Filled with everything the equilibrium post-processor reads, the same as the Grad-Shafranov
    /// solver leaves it. The flux and its derivatives are the total, plasma plus coils.
    ///
    /// For a diverted plasma the magnetic axis, the X-points, the boundary flux and the mask are
    /// found in the total flux, by the same functions the Grad-Shafranov solver uses, so that they
    /// agree with the flux the post-processor traces the boundary in. The analytic ones do not quite:
    /// the total flux differs from the analytic flux by the coil fit's residual, and near an X-point,
    /// where the flux is flat, that is as large as the flux itself, so the traced boundary would
    /// not pass through the analytic X-points. A limited GF plasma touches no wall and has no
    /// X-point, so there is nothing in the flux to find its boundary by, and the analytic one is
    /// kept: `psi_hat = 0`, with the analytic mask.
    ///
    /// The source functions are one-degree-of-freedom `EfitPolynomial`s, whose basis `1 - psi_norm` is
    /// exactly the GF profile shape, so the post-processor's `pressure`, `f`, ... are the analytic
    /// profiles. See `source_functions`.
    #[allow(clippy::too_many_arguments)]
    fn build_equilibrium_ids(
        &self,
        flux: &FluxSolution,
        geom: &MatchingGeometry,
        r: &Array1<f64>,
        z: &Array1<f64>,
        fit: BoundaryCoilFit,
        j_phi_2d: Array2<f64>,
        wall_ids: &WallIds,
    ) -> Equilibrium {
        let n_r: usize = self.n_r;
        let n_z: usize = self.n_z;
        let d_area: f64 = (r[1] - r[0]) * (z[1] - z[0]);

        // Analytic magnetic axis
        let (axis_x, axis_y, axis_psi_hat): (f64, f64, f64) = self.magnetic_axis(flux);
        let (mag_r_analytic, mag_z_analytic): (f64, f64) = self.physical_coordinates(axis_x, axis_y);
        let psi_a_analytic: f64 = self.psi_0 * axis_psi_hat;

        // Magnetic axis, boundary and stationary points
        let mag_r: f64;
        let mag_z: f64;
        let psi_a: f64;
        let psi_b: f64;
        let boundary_type: i32;
        let bounding_r: f64;
        let bounding_z: f64;
        let mask_2d: Array2<f64>;
        let contour_tree_nodes: Vec<EquilibriumContourTreeNode>;
        if !geom.xpts().0.is_empty() {
            // Diverted: found in the total flux, as `EquilibriumSolver::update_psi_and_geometry` finds them
            let (limit_pts_r, limit_pts_z): (Array1<f64>, Array1<f64>) = limiter_points(wall_ids).expect("GuazzottoFreidberg: limiter points");
            let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(wall_ids).expect("GuazzottoFreidberg: vacuum vessel");
            let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
                r.view(),
                z.view(),
                fit.flux_total.psi.view(),
                fit.flux_total.d_psi_d_r.view(),
                fit.flux_total.d_psi_d_z.view(),
                fit.flux_total.d2_psi_d_r2.view(),
                fit.flux_total.d2_psi_d_r_d_z.view(),
                fit.flux_total.d2_psi_d_z2.view(),
            );
            // The search for the magnetic axis starts from the analytic one
            let magnetic_axis: MagneticAxis = find_magnetic_axis(&stationary_points, mag_r_analytic, mag_z_analytic, &vessel_r, &vessel_z)
                .expect("GuazzottoFreidberg: no magnetic axis found in the total flux");
            let plasma_boundary: BoundaryContour = find_boundary(
                r,
                z,
                &fit.flux_total.psi,
                &fit.flux_total.d_psi_d_r,
                &fit.flux_total.d_psi_d_z,
                &fit.flux_total.d2_psi_d_r_d_z,
                &stationary_points,
                &limit_pts_r,
                &limit_pts_z,
                &vessel_r,
                &vessel_z,
                magnetic_axis.r,
                magnetic_axis.z,
            )
            .expect("GuazzottoFreidberg: no plasma boundary found in the total flux");
            mag_r = magnetic_axis.r;
            mag_z = magnetic_axis.z;
            psi_a = magnetic_axis.psi;
            psi_b = plasma_boundary.bounding_psi;
            boundary_type = plasma_boundary.xpt_diverted as i32;
            bounding_r = plasma_boundary.bounding_r;
            bounding_z = plasma_boundary.bounding_z;
            mask_2d = plasma_boundary.mask.expect("GuazzottoFreidberg: plasma boundary has no mask");
            contour_tree_nodes = grad_shafranov_contour_tree_nodes(&stationary_points);
        } else {
            // Limited: the analytic boundary, `psi_hat = 0`. The inboard mid-plane, `(x, y) = (-1, 0)`, is
            // one of GF's matching points on the boundary
            mag_r = mag_r_analytic;
            mag_z = mag_z_analytic;
            psi_a = psi_a_analytic;
            psi_b = 0.0;
            boundary_type = BOUNDARY_TYPE_LIMITED;
            (bounding_r, bounding_z) = self.physical_coordinates(-1.0, 0.0);
            mask_2d = self.mask.clone();
            contour_tree_nodes = vec![EquilibriumContourTreeNode {
                critical_type: CRITICAL_TYPE_MAXIMUM,
                r: mag_r,
                z: mag_z,
                psi: psi_a,
                ..Default::default()
            }];
        }

        // The IDS layout, the same as a reconstruction's: one time-slice, holding one rectangular grid
        let mut equilibrium_ids: Equilibrium = Equilibrium::default();
        equilibrium_ids.allocate_time_slices(&array![0.0]);
        equilibrium_ids.code.grid.n_r = n_r as i32;
        equilibrium_ids.code.grid.n_z = n_z as i32;
        equilibrium_ids.code.grid.r_min = self.r_min;
        equilibrium_ids.code.grid.r_max = self.r_max;
        equilibrium_ids.code.grid.z_min = self.z_min;
        equilibrium_ids.code.grid.z_max = self.z_max;
        // An analytic solution is always usable
        equilibrium_ids.code.output_flag = array![0];
        // `r0 * b0` is the vacuum `f`, which is what the post-processor recovers the rod current from
        equilibrium_ids.vacuum_toroidal_field.r0 = self.r_geo;
        equilibrium_ids.vacuum_toroidal_field.b0 = array![self.bt_vac_at_r_geo];

        let time_slice: &mut EquilibriumTimeSlice = &mut equilibrium_ids.time_slice[0];
        // The normalised flux the source functions are evaluated on
        time_slice.profiles_1d.psi_norm = Array1::linspace(0.0, 1.0, N_PSI_NORM);

        // `EfitPolynomial` coefficients which reproduce the GF profiles. With `psi = psi_a * (1 - psi_norm)`:
        // p' = 2 * p_axis * psi / psi_0 ** 2 = (2 * p_axis * psi_a / psi_0 ** 2) * (1 - psi_norm)
        // ff' = 2 * r_geo ** 2 * b_0 * d_b * psi / psi_0 ** 2 = (2 * r_geo ** 2 * b_0 * d_b * psi_a / psi_0 ** 2) * (1 - psi_norm)
        // `psi_a` is the analytic one, as these are the analytic profiles
        let psi_0_squared: f64 = self.psi_0 * self.psi_0;
        time_slice.source_functions.p_prime.coefficients = array![2.0 * self.p_axis * psi_a_analytic / psi_0_squared];
        time_slice.source_functions.ff_prime.coefficients =
            array![2.0 * self.r_geo * self.r_geo * self.bt_vac_at_r_geo * self.bt_diamagnetic_shift * psi_a_analytic / psi_0_squared];

        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice.global_quantities.psi_magnetic_axis = psi_a;
        time_slice.global_quantities.ip = j_phi_2d.sum() * d_area;
        time_slice.boundary.psi = psi_b;
        time_slice.boundary.bounding.r = bounding_r;
        time_slice.boundary.bounding.z = bounding_z;
        time_slice.boundary.r#type = boundary_type;
        time_slice.contour_tree.node = contour_tree_nodes;
        time_slice.convergence.result.name = "converged".to_string();
        time_slice.convergence.result.index = CONVERGENCE_STATUS_CONVERGED;
        time_slice.convergence.result.description = "analytic Guazzotto-Freidberg equilibrium".to_string();

        // `rectangular` is index 1 of the data dictionary's poloidal plane coordinates enumeration
        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid_type.name = "rectangular".to_string();
        profiles_2d.grid_type.index = 1;
        profiles_2d.grid_type.description = "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".to_string();
        profiles_2d.grid.dim1 = r.to_owned();
        profiles_2d.grid.dim2 = z.to_owned();
        profiles_2d.grid.d_area = d_area;
        profiles_2d.r = Array2::from_shape_fn((n_z, n_r), |(_i_z, i_r)| r[i_r]);
        profiles_2d.z = Array2::from_shape_fn((n_z, n_r), |(i_z, _i_r)| z[i_z]);
        profiles_2d.psi_norm = &mask_2d * (&fit.flux_total.psi - psi_a) / (psi_b - psi_a);
        profiles_2d.psi = fit.flux_total.psi;
        profiles_2d.d_psi_d_r = fit.flux_total.d_psi_d_r;
        profiles_2d.d_psi_d_z = fit.flux_total.d_psi_d_z;
        profiles_2d.d2_psi_d_r2 = fit.flux_total.d2_psi_d_r2;
        profiles_2d.d2_psi_d_r_d_z = fit.flux_total.d2_psi_d_r_d_z;
        profiles_2d.d2_psi_d_z2 = fit.flux_total.d2_psi_d_z2;
        profiles_2d.psi_coils = fit.psi_coils;
        profiles_2d.mask = mask_2d;
        profiles_2d.j_phi = j_phi_2d;
        time_slice.profiles_2d = vec![profiles_2d];

        equilibrium_ids
    }
}

/// The p' and ff' source functions the equilibrium IDS's coefficients are written for.
///
/// Both are a one-degree-of-freedom `EfitPolynomial`, whose single basis function is `1 - psi_norm`.
/// Neither is fitted, so neither has a regularisation.
fn source_functions() -> (SharedSourceFunction, SharedSourceFunction) {
    let p_prime_source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
        n_dof: 1,
        regularisations: Array2::zeros((0, 1)),
        dof_values: Array1::zeros(0),
        exact: false,
    });
    let ff_prime_source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
        n_dof: 1,
        regularisations: Array2::zeros((0, 1)),
        dof_values: Array1::zeros(0),
        exact: false,
    });
    (p_prime_source_function, ff_prime_source_function)
}
