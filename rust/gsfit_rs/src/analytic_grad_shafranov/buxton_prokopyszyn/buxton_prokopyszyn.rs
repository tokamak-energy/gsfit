//! The user-facing class: an analytic Grad-Shafranov equilibrium with the paper's EFIT-polynomial `p'` and `ff'` of any order, each
//! with a constant term (Solov'ev-like), given by their coefficients in physical units, which also fills an Equilibrium IDS and
//! post-processes it as a reconstruction is.
//!
//! The code uses the paper's notation. The flux is the total poloidal flux, `psi = 2 pi R A_phi` [weber], and
//! `psi_N = (psi_a - psi) / (psi_a - psi_b)` is 0 on the magnetic axis and 1 on the LCFS. Here `psi_b = 0`, and `psi_a > psi_b` for a
//! positive plasma current (GSFit's sign convention), with `B_R = -(d psi / d Z) / (2 pi R)` and `B_Z = (d psi / d R) / (2 pi R)`.
//!
//! The solver works in the paper's variables: `(1 - psi_N)`, 1 on the magnetic axis and 0 on the LCFS, so that
//! `psi = (psi_a - psi_b) (1 - psi_N)`, at points `(R, Z)` with `Z` measured from `z_geo`, the height the series is built about; the
//! wavenumbers `k_j` and `k_max` [1 / metre]; and `R_c**2 = R_geo**2 + a_minor**2`.

use super::configuration::{BoundaryConditions, Configuration, Point, Points, RadialScale};
use super::diagnostics::{Closure, EdgeRays, closure_check, coefficient_sizes, edge_targets, gs_residual_ratio, is_growing};
use super::flux_geometry::{PlasmaQuadrature, find_stationary_point, inside_polygon, lcfs_contour, plasma_quadrature};
use super::gsfit_units::{NormalisedProfiles, normalised_profiles, nu_on_axis, psi_a_minus_psi_b_of};
use super::lattice::{FluxValues, Lattice, LatticeSeries, series_evaluate, series_power_sizes, series_scale, series_values};
use super::perturbation::{AxisInRadial, CBarMethod, FitRegularisation, PerturbationOptions, PerturbationSeries, find_axis, solve_perturbation};
use super::profiles::{ProfileShapes, is_broad_profile, profile_integral, profile_value};
use super::radial_series::{N_RADIAL_MIN, N_RADIAL_TAIL, RadialPair, RadialVariable, RadialVariableKind, ValidRange, c_and_s, radial_terms_needed};
use super::reference_operator::ReferenceOperator;
use super::settings_schema::{BuxtonProkopyszynSettings, BuxtonProkopyszynSettingsGrid, BuxtonProkopyszynSettingsNumerics, BuxtonProkopyszynSettingsProfiles};
use super::source_function::source_function_and_dofs;
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::grad_shafranov::CONVERGENCE_STATUS_CONVERGED;
use crate::greens_storage::GreensTables;
use crate::source_functions::SharedSourceFunction;
use faer::c64;
use imas_rs::ids::equilibrium::Code;
use imas_rs::python::{PyEquilibrium, PyParametersInput};
use imas_rs::{Equilibrium, EquilibriumContourTreeNode, EquilibriumLimiterUnit, EquilibriumLimiterUnitOutline, EquilibriumProfiles2d, EquilibriumTimeSlice};
use ndarray::{Array1, Array2, ArrayD, IxDyn, array};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;
use std::ffi::CString;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// A solution whose GS residual (`gs_residual`, relative to the largest value of the right-hand side) is above this is not a
/// converged solution, and `solve` warns [dimensionless]
const GS_RESIDUAL_WARNING: f64 = 1.0e-6;

/// LCFS points within this many `a_minor` of an X-point are left out of the boundary misfit, and keep their weight in Lawson's
/// weighting: the first-order distance `|1 - psi_N| / |grad (1 - psi_N)|` fails where the gradient vanishes [dimensionless]
const X_POINT_EXCLUSION_RADIUS: f64 = 0.05;

/// Lawson's weighting: the largest ratio of two LCFS points' weights; smaller weights are raised to it, so that no point drops out of
/// the fits [dimensionless]
const LAWSON_WEIGHT_RATIO_MAX: f64 = 1.0e3;

/// Points of a model surface's outline from which the targets of the GS residual's rays up to the LCFS are chosen, equally spaced in
/// arclength; user LCFS points are used as given [dimensionless]
const EDGE_OUTLINE_N_POINT: usize = 513;

/// Gauss-Legendre nodes in each direction of each half of the plasma for its current: within 1e-14 of 256 nodes on the elongated D
/// (limited), with an edge current, and on an ST40 single null [dimensionless]
const N_PLASMA_CURRENT_NODE: usize = 128;

/// About the number of points of the analytic LCFS polygon that `is_inside` and the IDS's mask use [dimensionless]
const N_BOUNDARY_POLYGON_POINT: usize = 512;

/// The data dictionary's `boundary/type`
const BOUNDARY_TYPE_LIMITED: i32 = 0;
const BOUNDARY_TYPE_DIVERTED: i32 = 1;

/// The data dictionary's `contour_tree/node/critical_type`
const CRITICAL_TYPE_SADDLE: i32 = 1;
const CRITICAL_TYPE_MAXIMUM: i32 = 2;

/// The data dictionary's convergence status of a failed solve, as the Grad-Shafranov solver's
const CONVERGENCE_STATUS_UNCONVERGED: i32 = 10;

/// The fit of orders >= 1: `Auto` tries the plain least squares, then Tikhonov
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FitChoice {
    Auto,
    Given(FitRegularisation),
}

/// The physical scale of the solution
#[derive(Clone, Debug)]
struct PhysicalScale {
    /// The factor the given coefficients are multiplied by: 1, or the one that gives the given plasma current [dimensionless]
    coefficient_scale: f64,
    /// `[c_0_p_prime, c_1_p_prime, ...]`, the given ones times `coefficient_scale` [pascal / weber]
    p_prime_coefficients: Vec<f64>,
    /// `[c_0_ff_prime, c_1_ff_prime, ...]`, likewise [tesla ** 2 * metre ** 2 / weber]
    ff_prime_coefficients: Vec<f64>,
    /// The paper's `psi_a - psi_b` [weber]
    psi_a_minus_psi_b: f64,
    /// Pressure on the magnetic axis [pascal]
    p_axis: f64,
}

impl PhysicalScale {
    /// NaN everywhere, until the solve sets it
    fn unset() -> PhysicalScale {
        PhysicalScale {
            coefficient_scale: f64::NAN,
            p_prime_coefficients: Vec::new(),
            ff_prime_coefficients: Vec::new(),
            psi_a_minus_psi_b: f64::NAN,
            p_axis: f64::NAN,
        }
    }
}

/// The flux and its derivatives in `(R, Z)`
struct PsiDerivatives {
    /// [weber]
    psi: Array1<f64>,
    /// [weber / metre]
    d_psi_d_r: Array1<f64>,
    /// [weber / metre]
    d_psi_d_z: Array1<f64>,
    /// [weber / metre ** 2]
    d2_psi_d_r2: Array1<f64>,
    /// [weber / metre ** 2]
    d2_psi_d_r_d_z: Array1<f64>,
    /// [weber / metre ** 2]
    d2_psi_d_z2: Array1<f64>,
}

/// One solve's solution, physical scale and diagnostics
#[derive(Clone)]
struct Solution {
    /// The number of `cosh` terms' wavenumbers it was solved with [dimensionless]
    n_wave_cosh: usize,
    series: PerturbationSeries,
    /// `(1 - psi_N)`, the partial sum of the series divided by its value on the magnetic axis, 1 there and 0 on the LCFS [dimensionless]
    one_minus_psi_n: LatticeSeries,
    fit_used: FitRegularisation,
    /// The number of Taylor terms kept, orders `0..n_taylor - 1`: the partial sum with the smallest GS residual [dimensionless]
    n_taylor: usize,
    /// The change of `(1 - psi_N)` on the plasma from the last `N_RADIAL_TAIL` powers of its radial series [dimensionless]
    radial_tail: f64,
    /// The fewest terms of the radial series that meet `radial_tolerance` (`radial_series::radial_terms_needed`); `None` without the
    /// check, or when its last powers do not decay [dimensionless]
    n_radial_needed: Option<usize>,
    /// The GS residual of every partial sum [dimensionless]
    order_gs_residuals: Array1<f64>,
    /// [dimensionless]
    coefficient_sizes: Array1<f64>,
    is_unstable: bool,
    /// The magnetic axis, `Z` from `z_geo` [metre]
    mag: Point,
    /// The paper's `k_max`, with the linear part's `c_bar` (see `gsfit_units`) [1 / metre]
    k_max: f64,
    /// The paper's GS residual: the largest difference between the two sides of the Grad-Shafranov equation inside the LCFS, over
    /// the largest value of its right-hand side [dimensionless]
    gs_residual: f64,
    /// The rays up to the LCFS along which the GS residual was also measured [dimensionless]
    n_edge_ray: usize,
    /// The X-points, `Z` from `z_geo` [metre]
    x_points: Vec<Point>,
    /// `1 - psi_N` at the X-point, 0 when the separatrix is the LCFS [dimensionless]
    x_point_flux: f64,
    /// The analytic LCFS, `Z` from `z_geo`, closed [metre]
    boundary_polygon: Points,
    closure: Closure,
    /// Largest distance from the given shape to the analytic LCFS [metre]
    boundary_misfit: f64,
    /// The relative weight of each given LCFS point's row in the fits (empty for a model surface) [dimensionless]
    boundary_weights: Vec<f64>,
    /// The distance of each given LCFS point from the analytic LCFS, `|1 - psi_N| / |grad (1 - psi_N)|`; NaN within
    /// `X_POINT_EXCLUSION_RADIUS` of an X-point (empty for a model surface) [metre]
    boundary_deviations: Vec<f64>,
    /// The boundary misfit of each solve of the weighting, the first with every point weighted 1 [metre]
    weighting_misfits: Vec<f64>,
    /// Why Lawson's weighting stopped (empty without it, `n_boundary_lawson_iteration_max = 0`)
    weighting_stop: String,
    scale: PhysicalScale,
    /// The warnings of the solve
    warnings: Vec<String>,
    /// Why the plasma current could not set the scale (the trial scale's current is not a positive number, as when the LCFS does not
    /// close): raised by `solve`
    scale_error: Option<String>,
}

/// The analytic problem: the shape, the profiles, the scale, and the solver's options, as resolved from the inputs
#[derive(Clone)]
struct Problem {
    /// The shape, moved down by `z_geo` (the solver works relative to it)
    configuration: Configuration,
    is_up_down_symmetric: bool,
    /// The paper's `R_geo` [metre]
    r_geo: f64,
    /// The paper's `a_minor` [metre]
    a_minor: f64,
    /// The major radius at which `b0` is given [metre]
    r0: f64,
    /// The vacuum toroidal field at `r0` [tesla]
    b0: f64,
    /// `[c_0_p_prime, c_1_p_prime, ...]` as given, at least two [pascal / weber]
    p_prime_coefficients: Vec<f64>,
    /// `[c_0_ff_prime, c_1_ff_prime, ...]` as given, at least two [tesla ** 2 * metre ** 2 / weber]
    ff_prime_coefficients: Vec<f64>,
    /// The plasma current that sets the factor the coefficients are multiplied by [ampere]
    ip: Option<f64>,
    /// The profiles `P` and `Q`: `mu0 R_c**2 p'` and `ff'` over `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` (see `gsfit_units`)
    shapes: ProfileShapes,
    /// `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` of the given coefficients, the bracket of the paper's `k_max**2`, which they were
    /// divided by [tesla ** 2 * metre ** 2 / weber]
    linear_source_at_r_c: f64,
    /// Number of equally spaced wavenumbers [dimensionless]
    n_wave: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    k_spacing: f64,
    /// Number of `cosh` terms' wavenumbers [dimensionless]
    n_wave_cosh: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    k_spacing_cosh: f64,
    radial_variable: RadialVariableKind,
    /// Number of terms in each radial power series [dimensionless]
    n_radial: usize,
    /// The largest change of `(1 - psi_N)` on the plasma from the last `N_RADIAL_TAIL` powers of its radial series, above which the solve
    /// warns; 0 for no check [dimensionless]
    radial_tolerance: f64,
    /// The choice of `c_bar`
    c_bar_method: CBarMethod,
    /// Most terms of the Taylor series [dimensionless]
    n_taylor_max: usize,
    /// [dimensionless]
    taylor_tolerance: f64,
    exact_x_points: bool,
    /// Most re-solves of Lawson's weighting of the given LCFS points; 0 for every point weighted 1 [dimensionless]
    n_boundary_lawson_iteration_max: usize,
    fit_regularisation: FitChoice,
    /// [dimensionless]
    fit_threshold: f64,
    verbose: bool,
    /// The configuration's height about which the series is built, from which `Z` is measured [metre]
    z_geo: f64,
}

impl Problem {
    /// A point `(R, Z)` in the machine, with `Z` measured from `z_geo` as the solver measures it
    fn point_from_machine(&self, r: f64, z: f64) -> Point {
        Point { r, z: z - self.z_geo }
    }

    /// `(R, Z)` in the machine of a point whose `Z` is measured from `z_geo` [metre]
    fn machine_coordinates(&self, point: Point) -> (f64, f64) {
        (point.r, point.z + self.z_geo)
    }

    /// The physical scale of the solution (`k_max` solved) with the given coefficients multiplied by `coefficient_scale`: the
    /// coefficients, `psi_a - psi_b` (from `k_max`) and `p_axis`. Multiplying the coefficients by a factor multiplies `psi_a - psi_b` and
    /// the plasma current by it, and `p_axis` by its square.
    fn physical_scale(&self, k_max: f64, coefficient_scale: f64) -> PhysicalScale {
        let n_pressure: usize = self.p_prime_coefficients.len();
        let mut p_prime_coefficients: Vec<f64> = Vec::with_capacity(n_pressure);
        for i_pressure in 0..n_pressure {
            p_prime_coefficients.push(coefficient_scale * self.p_prime_coefficients[i_pressure]);
        }
        let n_ff: usize = self.ff_prime_coefficients.len();
        let mut ff_prime_coefficients: Vec<f64> = Vec::with_capacity(n_ff);
        for i_ff in 0..n_ff {
            ff_prime_coefficients.push(coefficient_scale * self.ff_prime_coefficients[i_ff]);
        }
        let psi_a_minus_psi_b: f64 = coefficient_scale * psi_a_minus_psi_b_of(k_max, self.linear_source_at_r_c);
        // p(axis) = (psi_a - psi_b) int_0^1 p' dpsi_N, which includes the constant term's
        let p_axis: f64 = psi_a_minus_psi_b * profile_integral(1.0, &p_prime_coefficients);
        PhysicalScale {
            coefficient_scale,
            p_prime_coefficients,
            ff_prime_coefficients,
            psi_a_minus_psi_b,
            p_axis,
        }
    }

    /// `p'(psi_N) = c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1)`, with `p'`'s coefficients times `coefficient_scale`
    /// [pascal / weber]
    fn p_prime(&self, scale: &PhysicalScale, psi_n: f64) -> f64 {
        profile_value(1.0 - psi_n, &scale.p_prime_coefficients)
    }

    /// `ff'(psi_N)`, likewise [tesla ** 2 * metre ** 2 / weber]
    fn ff_prime(&self, scale: &PhysicalScale, psi_n: f64) -> f64 {
        profile_value(1.0 - psi_n, &scale.ff_prime_coefficients)
    }

    /// The toroidal current density `J_phi = 2 pi (R p' + ff' / (mu0 R))` at `R` and `psi_N` [ampere / metre ** 2]
    fn current_density(&self, scale: &PhysicalScale, r: f64, psi_n: f64) -> f64 {
        2.0 * PI * (r * self.p_prime(scale, psi_n) + self.ff_prime(scale, psi_n) / (MU_0 * r))
    }

    /// `p(psi_N) = (psi_a - psi_b) int_{psi_N}^1 p' dpsi_N`, zero on the LCFS [pascal]
    fn pressure(&self, scale: &PhysicalScale, psi_n: f64) -> f64 {
        scale.psi_a_minus_psi_b * profile_integral(1.0 - psi_n, &scale.p_prime_coefficients)
    }

    /// `f = R B_phi`, `f**2 = (r0 b0)**2 + 2 (psi_a - psi_b) int_{psi_N}^1 ff' dpsi_N`, with the sign of `b0` [tesla * metre]
    fn f(&self, scale: &PhysicalScale, psi_n: f64) -> f64 {
        let integral: f64 = profile_integral(1.0 - psi_n, &scale.ff_prime_coefficients);
        let f_squared: f64 = (self.r0 * self.b0).powi(2) + 2.0 * scale.psi_a_minus_psi_b * integral;
        f_squared.sqrt().copysign(self.b0)
    }

    /// The range of `R` the LCFS searches use: where the radial series can be evaluated, with `R**2` within `R_c**2 -+ 1.5` times the
    /// plasma's half-width in `R**2`
    fn search_range(&self, lattice: &Lattice) -> ValidRange {
        r_squared_range(&lattice.radial, 1.5)
    }

    /// The plasma current, `I_p = int J_phi dR dZ` over the LCFS, by Gauss-Legendre quadrature half by half as the LCFS is traced
    /// (`flux_geometry::plasma_quadrature`): about 1e-14 relative, edge currents included [ampere]
    fn plasma_current(&self, solution: &Solution, scale: &PhysicalScale, n_node: usize) -> f64 {
        let lattice: &Lattice = &solution.series.lattice;
        let one_minus_psi_n: &LatticeSeries = &solution.one_minus_psi_n;
        let value_function = |r: &[f64], z: &[f64]| -> Array1<f64> { series_values(lattice, one_minus_psi_n, r, z) };
        let range: ValidRange = self.search_range(lattice);
        let quadrature: PlasmaQuadrature = plasma_quadrature(
            &value_function,
            solution.mag,
            &solution.x_points,
            n_node,
            range.r_minimum,
            range.r_maximum,
            3.0 * self.a_minor,
        );
        let values_at_nodes: Array1<f64> = value_function(
            quadrature.node_r.as_slice().expect("contiguous"),
            quadrature.node_z.as_slice().expect("contiguous"),
        );
        let n_quadrature: usize = quadrature.weight.len();
        let mut total: f64 = 0.0;
        for i_quadrature in 0..n_quadrature {
            let current: f64 = self.current_density(scale, quadrature.node_r[i_quadrature], 1.0 - values_at_nodes[i_quadrature]);
            total += quadrature.weight[i_quadrature] * current;
        }
        total
    }

    /// The LCFS, `psi_N = 1`, `Z` from `z_geo`, closed (last point = first) [metre]
    fn boundary_points(&self, lattice: &Lattice, one_minus_psi_n: &LatticeSeries, mag: Point, x_points: &[Point], n_point: usize) -> Points {
        let value_function = |r: &[f64], z: &[f64]| -> Array1<f64> { series_values(lattice, one_minus_psi_n, r, z) };
        let range: ValidRange = self.search_range(lattice);
        lcfs_contour(&value_function, mag, x_points, n_point, range.r_minimum, range.r_maximum, 3.0 * self.a_minor)
    }

    /// The number of given LCFS points, whose `(1 - psi_N) = 0` rows come first in the conditions (`Configuration::conditions`); 0 for a model
    /// surface [dimensionless]
    fn n_lcfs_point(&self) -> usize {
        match &self.configuration {
            Configuration::LcfsPoints { boundary_r, .. } => boundary_r.len(),
            _ => 0,
        }
    }

    /// `solve_once` with every given LCFS point weighted 1, then, with `n_boundary_lawson_iteration_max > 0`, re-solved with the weights
    /// raised where the analytic LCFS deviates most from the points, keeping the solution with the smallest boundary misfit.
    ///
    /// Lawson's iteration for the smallest largest deviation: the weights `u = w**2` of the squared rows become `u d / mean(u d)`, `d`
    /// being each point's distance from the last solution's LCFS (points next to an X-point keep their weight); the weights are then
    /// kept within `LAWSON_WEIGHT_RATIO_MAX` of the largest and normalised to `mean(w**2) = 1`. It stops when a re-solve fails, or its
    /// LCFS does not close, or it does not lower the misfit, or after `n_boundary_lawson_iteration_max` re-solves.
    ///
    /// # Errors
    /// The first solve's (see `solve_once`)
    fn solve_weighted(&self) -> Result<Solution, String> {
        let n_lcfs_point: usize = self.n_lcfs_point();
        let weights_start: Vec<f64> = vec![1.0; n_lcfs_point];
        let mut solution: Solution = self.solve_once(&weights_start)?;
        let mut weighting_misfits: Vec<f64> = vec![solution.boundary_misfit];
        let mut weighting_stop: String = String::new();
        if self.n_boundary_lawson_iteration_max > 0 {
            weighting_stop = format!("after {} re-solves (n_boundary_lawson_iteration_max)", self.n_boundary_lawson_iteration_max);
            'lawson_loop: for i_iteration in 0..self.n_boundary_lawson_iteration_max {
                // u d / mean(u d) at the points with a deviation; the others keep their weight
                let mut weights_squared: Vec<f64> = vec![f64::NAN; n_lcfs_point];
                let mut deviation_total: f64 = 0.0;
                let mut n_deviation: usize = 0;
                for i_point in 0..n_lcfs_point {
                    weights_squared[i_point] = solution.boundary_weights[i_point] * solution.boundary_weights[i_point];
                    if solution.boundary_deviations[i_point].is_finite() {
                        deviation_total += weights_squared[i_point] * solution.boundary_deviations[i_point];
                        n_deviation += 1;
                    }
                }
                if n_deviation == 0 || !(deviation_total > 0.0) {
                    weighting_stop = "no LCFS point has a deviation".to_string();
                    break 'lawson_loop;
                }
                let deviation_mean: f64 = deviation_total / n_deviation as f64;
                let mut weight_largest: f64 = 0.0;
                for i_point in 0..n_lcfs_point {
                    if solution.boundary_deviations[i_point].is_finite() {
                        weights_squared[i_point] *= solution.boundary_deviations[i_point] / deviation_mean;
                    }
                    weight_largest = weight_largest.max(weights_squared[i_point].sqrt());
                }
                // Within LAWSON_WEIGHT_RATIO_MAX of the largest weight, then mean(w**2) = 1
                let mut weights: Vec<f64> = vec![f64::NAN; n_lcfs_point];
                let mut weight_squared_total: f64 = 0.0;
                for i_point in 0..n_lcfs_point {
                    weights[i_point] = weights_squared[i_point].sqrt().max(weight_largest / LAWSON_WEIGHT_RATIO_MAX);
                    weight_squared_total += weights[i_point] * weights[i_point];
                }
                let weight_scale: f64 = (n_lcfs_point as f64 / weight_squared_total).sqrt();
                for i_point in 0..n_lcfs_point {
                    weights[i_point] *= weight_scale;
                }
                let candidate: Solution = match self.solve_once(&weights) {
                    Ok(candidate) => candidate,
                    Err(error) => {
                        weighting_stop = format!("re-solve {} failed: {error}", i_iteration + 1);
                        break 'lawson_loop;
                    }
                };
                weighting_misfits.push(candidate.boundary_misfit);
                // A closed LCFS, a scale and a GS residual no worse than the warning's are needed; a valid solution then beats an invalid
                // one, and otherwise needs a smaller misfit
                let is_valid: bool =
                    candidate.closure.closed && candidate.scale_error.is_none() && candidate.gs_residual <= GS_RESIDUAL_WARNING.max(solution.gs_residual);
                if !is_valid {
                    weighting_stop = format!("re-solve {}: its LCFS did not close, or its scale or its GS residual failed", i_iteration + 1);
                    break 'lawson_loop;
                }
                let is_solution_valid: bool = solution.closure.closed && solution.scale_error.is_none();
                if is_solution_valid && !(candidate.boundary_misfit < solution.boundary_misfit) {
                    weighting_stop = format!("re-solve {} did not lower the misfit", i_iteration + 1);
                    break 'lawson_loop;
                }
                solution = candidate;
            }
        }
        solution.weighting_misfits = weighting_misfits;
        solution.weighting_stop = weighting_stop;
        Ok(solution)
    }

    /// One solve with the options and the given LCFS points' weights (see `BuxtonProkopyszyn.solve`).
    ///
    /// # Arguments
    /// * `boundary_weights` - the relative weight of each given LCFS point's row in the fits, `n_lcfs_point` of them [dimensionless]
    ///
    /// # Errors
    /// No partial sum of the series has a positive `k_max**2`, no order-0 eigenvalue, or a plasma current that cannot be set
    fn solve_once(&self, boundary_weights: &[f64]) -> Result<Solution, String> {
        let r_geo: f64 = self.r_geo;
        let a_minor: f64 = self.a_minor;
        let n_wave_cosh: usize = self.n_wave_cosh;
        let mut warnings: Vec<String> = Vec::new();
        let n_point_extra: usize = 10 * (self.n_wave - 1 + n_wave_cosh);
        let mut conditions: BoundaryConditions = self.configuration.conditions(r_geo, a_minor, n_point_extra)?;
        if !self.exact_x_points {
            conditions.is_exact = vec![false; conditions.n_condition()];
        }
        // The given LCFS points' rows come first, in the given order (`Configuration::conditions`)
        let n_boundary_weight: usize = boundary_weights.len();
        for i_boundary_weight in 0..n_boundary_weight {
            conditions.weight[i_boundary_weight] *= boundary_weights[i_boundary_weight];
        }
        let interior: Points = self.configuration.interior_points(r_geo, a_minor)?;
        let radial: Arc<RadialVariable> = Arc::new(RadialVariable::new(r_geo, a_minor, self.radial_variable, self.n_radial));
        // The plasma's vertical extent, from the conditions (on the upper half only for an up-down symmetric shape)
        let n_condition: usize = conditions.n_condition();
        let mut z_top: f64 = f64::NEG_INFINITY;
        let mut z_minimum: f64 = f64::INFINITY;
        for i_condition in 0..n_condition {
            z_top = z_top.max(conditions.z[i_condition]);
            z_minimum = z_minimum.min(conditions.z[i_condition]);
        }
        let z_bottom: f64 = if self.is_up_down_symmetric { -z_top } else { z_minimum };
        // The targets of the rays along which the GS residual is also measured, up to the LCFS: up to the given LCFS when choosing the
        // order of a series, up to the solution's own LCFS for the residual reported (and compared between the fits)
        let edge_outline: Points = self.configuration.outline(r_geo, a_minor, EDGE_OUTLINE_N_POINT)?;
        let configuration_x_points: Vec<Point> = self.configuration.x_points(r_geo, a_minor)?;
        let edge: Points = edge_targets(&edge_outline, &configuration_x_points);
        let n_edge_ray: usize = edge.r.len();
        // Where the radial series can be evaluated, which bounds the search for the solution's own LCFS along the rays
        let valid: ValidRange = radial.r_range_valid();
        let rays_to_given: EdgeRays = EdgeRays {
            targets: &edge,
            to_own_lcfs: false,
            r_minimum: f64::NEG_INFINITY,
            r_maximum: f64::INFINITY,
        };
        let rays_to_own: EdgeRays = EdgeRays {
            targets: &edge,
            to_own_lcfs: true,
            r_minimum: valid.r_minimum,
            r_maximum: valid.r_maximum,
        };
        // The magnetic axis is searched for within 90 % of the way from R_c**2 to the plasma's edges in R**2
        let axis_search_low: f64 = radial.of_r_squared(radial.r_c_squared - 0.9 * radial.r_squared_half_width);
        let axis_search_high: f64 = radial.of_r_squared(radial.r_c_squared + 0.9 * radial.r_squared_half_width);

        // The fits to try: `Auto` is the plain least squares and then, if that series has not reached rounding, Tikhonov, which is kept
        // if its series' GS residual is three times smaller (the second series reuses the first's operator and its cached matrices);
        // with `cosh` terms, whose plain fits are ill-conditioned, Tikhonov alone
        let fits: Vec<(FitRegularisation, f64)> = match self.fit_regularisation {
            FitChoice::Given(regularisation) => vec![(regularisation, self.fit_threshold)],
            FitChoice::Auto => {
                if n_wave_cosh > 0 {
                    vec![(FitRegularisation::Tikhonov, self.fit_threshold)]
                } else {
                    vec![(FitRegularisation::None, 0.0), (FitRegularisation::Tikhonov, self.fit_threshold)]
                }
            }
        };
        let n_fit: usize = fits.len();
        let mut kept: Option<Solution> = None;
        'fit_loop: for i_fit in 0..n_fit {
            if i_fit > 0 {
                if let Some(solution) = &kept {
                    if solution.gs_residual <= 1.0e-12 {
                        break 'fit_loop;
                    }
                }
            }
            let options: PerturbationOptions = PerturbationOptions {
                n_wave: self.n_wave,
                k_spacing: self.k_spacing,
                n_wave_cosh,
                k_spacing_cosh: self.k_spacing_cosh,
                is_up_down_symmetric: self.is_up_down_symmetric,
                n_taylor_max: self.n_taylor_max,
                c_bar_method: self.c_bar_method,
                taylor_tolerance: self.taylor_tolerance,
                fit_regularisation: fits[i_fit].0,
                fit_threshold: fits[i_fit].1,
                verbose: self.verbose,
            };
            let reference_operator: Option<&ReferenceOperator> = kept.as_ref().map(|solution: &Solution| &solution.series.operator);
            let series_fit: PerturbationSeries = solve_perturbation(&radial, &self.shapes, &conditions, &interior, &options, reference_operator)?;
            let lattice_fit: &Lattice = &series_fit.lattice;
            // The GS residual of every partial sum; the best one is kept
            let n_taylor_all: usize = series_fit.n_taylor();
            let mut order_gs_residuals: Array1<f64> = Array1::from_elem(n_taylor_all, f64::NAN);
            let mut partial_sums: Vec<LatticeSeries> = Vec::with_capacity(n_taylor_all);
            let mut axes: Vec<Point> = Vec::with_capacity(n_taylor_all);
            for i_order in 0..n_taylor_all {
                // The partial sum of orders 0..i_order
                let partial_sum: LatticeSeries = series_fit.one_minus_psi_n_sum(i_order + 1, 1.0);
                let axis: AxisInRadial = find_axis(lattice_fit, &partial_sum, axis_search_low, axis_search_high);
                let r_axis: f64 = radial.to_r(axis.radial);
                let value_at_axis: f64 = series_values(lattice_fit, &partial_sum, &[r_axis], &[axis.z])[0];
                partial_sums.push(series_scale(&partial_sum, 1.0 / value_at_axis));
                axes.push(Point { r: r_axis, z: axis.z });
                let k_max_squared: f64 = series_fit.k_max_squared(i_order + 1, 1.0);
                if k_max_squared > 0.0 && value_at_axis > 0.0 {
                    order_gs_residuals[i_order] = gs_residual_ratio(
                        lattice_fit,
                        &partial_sums[i_order],
                        1.0,
                        axes[i_order],
                        z_bottom,
                        z_top,
                        k_max_squared.sqrt(),
                        &self.shapes,
                        41,
                        Some(&rays_to_given),
                    );
                }
            }
            let mut i_best: Option<usize> = None;
            let mut residual_best: f64 = f64::INFINITY;
            for i_order in 0..n_taylor_all {
                if order_gs_residuals[i_order].is_finite() && order_gs_residuals[i_order] < residual_best {
                    residual_best = order_gs_residuals[i_order];
                    i_best = Some(i_order);
                }
            }
            let i_best: usize = match i_best {
                Some(i_order) => i_order,
                None => {
                    if i_fit == 0 {
                        return Err("BuxtonProkopyszyn.solve: no partial sum of the series has a positive k_max**2".to_string());
                    }
                    continue 'fit_loop;
                }
            };
            let k_max_fit: f64 = series_fit.k_max_squared(i_best + 1, 1.0).sqrt();
            let gs_residual_fit: f64 = gs_residual_ratio(
                lattice_fit,
                &partial_sums[i_best],
                1.0,
                axes[i_best],
                z_bottom,
                z_top,
                k_max_fit,
                &self.shapes,
                61,
                Some(&rays_to_own),
            );
            let is_better: bool = match &kept {
                None => true,
                Some(solution) => 3.0 * gs_residual_fit < solution.gs_residual,
            };
            if !is_better {
                continue 'fit_loop;
            }
            let sizes: Array1<f64> = coefficient_sizes(&series_fit.one_minus_psi_n_orders);
            let mut is_unstable: bool = is_growing(&sizes);
            if n_wave_cosh > 0 {
                // The coefficient sums of a series with `cosh` terms are dominated by the cosh terms beyond the plasma's height; its
                // stability is that of its values on the plasma (the LCFS points and the interior points)
                let mut plasma_r: Vec<f64> = conditions.r.clone();
                plasma_r.extend(interior.r.iter());
                let mut plasma_z: Vec<f64> = conditions.z.clone();
                plasma_z.extend(interior.z.iter());
                let n_taylor_series: usize = series_fit.one_minus_psi_n_orders.len();
                let mut plasma_sizes: Array1<f64> = Array1::from_elem(n_taylor_series, f64::NAN);
                for i_order in 0..n_taylor_series {
                    let values: Array1<f64> = series_values(lattice_fit, &series_fit.one_minus_psi_n_orders[i_order], &plasma_r, &plasma_z);
                    // NaN propagates: a value that cannot be evaluated is not a small one
                    let mut largest: f64 = 0.0;
                    for value in values.iter() {
                        if value.is_nan() || value.abs() > largest {
                            largest = value.abs();
                        }
                    }
                    plasma_sizes[i_order] = largest;
                }
                is_unstable = is_growing(&plasma_sizes);
            }
            let one_minus_psi_n: LatticeSeries = partial_sums[i_best].clone();
            kept = Some(Solution {
                n_wave_cosh,
                series: series_fit,
                one_minus_psi_n,
                fit_used: fits[i_fit].0,
                n_taylor: i_best + 1,
                radial_tail: f64::NAN,
                n_radial_needed: None,
                order_gs_residuals,
                coefficient_sizes: sizes,
                is_unstable,
                mag: axes[i_best],
                k_max: k_max_fit,
                gs_residual: gs_residual_fit,
                n_edge_ray,
                x_points: Vec::new(),
                x_point_flux: f64::NAN,
                boundary_polygon: Points {
                    r: Array1::zeros(0),
                    z: Array1::zeros(0),
                },
                closure: Closure {
                    closed: false,
                    reaches_box_edge: true,
                    n_maximum: 0,
                    monotonic: false,
                },
                boundary_misfit: f64::NAN,
                boundary_weights: Vec::new(),
                boundary_deviations: Vec::new(),
                weighting_misfits: Vec::new(),
                weighting_stop: String::new(),
                scale: PhysicalScale::unset(),
                warnings: Vec::new(),
                scale_error: None,
            });
        }
        let mut solution: Solution = kept.expect("solve_once: the first fit is always kept");

        // The radial series' truncation: the change of (1 - psi_N) from the last N_RADIAL_TAIL powers, and with radial_tolerance the
        // fewest terms that meet it
        let power_sizes: Array1<f64> = series_power_sizes(&solution.one_minus_psi_n);
        let n_radial: usize = power_sizes.len();
        let mut radial_tail: f64 = 0.0;
        for i_radial in (n_radial - N_RADIAL_TAIL)..n_radial {
            radial_tail += power_sizes[i_radial];
        }
        solution.radial_tail = radial_tail;
        if self.radial_tolerance > 0.0 {
            solution.n_radial_needed = radial_terms_needed(&power_sizes, self.radial_tolerance);
            // NaN propagates: a tail that cannot be evaluated is not a small one
            if !(radial_tail <= self.radial_tolerance) {
                let needed: String = match solution.n_radial_needed {
                    Some(n_radial_needed) => format!("about {n_radial_needed} terms would meet it"),
                    None => "its last powers do not decay".to_string(),
                };
                warnings.push(format!(
                    "BuxtonProkopyszyn: the last {N_RADIAL_TAIL} powers of the radial series (n_radial = {n_radial}) change 1 - psi_N by {radial_tail:.1e}, above radial_tolerance = {:.1e}; {needed}",
                    self.radial_tolerance
                ));
            }
        }

        // Physical scalars of the given coefficients, which a given plasma current rescales below, once the X-points are known
        solution.scale = self.physical_scale(solution.k_max, 1.0);

        // The X-points, and (1 - psi_N) at them
        let lattice: Lattice = solution.series.lattice.clone();
        let one_minus_psi_n: LatticeSeries = solution.one_minus_psi_n.clone();
        let flux_function = |r: &[f64], z: &[f64]| -> FluxValues { series_evaluate(&lattice, &one_minus_psi_n, r, z) };
        let n_x_point: usize = configuration_x_points.len();
        let mut x_points: Vec<Point> = Vec::with_capacity(n_x_point);
        for i_x_point in 0..n_x_point {
            let x_point: Point = find_stationary_point(flux_function, configuration_x_points[i_x_point], a_minor);
            let x_point_flux: f64 = series_values(&lattice, &one_minus_psi_n, &[x_point.r], &[x_point.z])[0];
            // The largest; NaN propagates
            if i_x_point == 0 || x_point_flux.is_nan() || x_point_flux > solution.x_point_flux {
                solution.x_point_flux = x_point_flux;
            }
            x_points.push(x_point);
        }
        solution.x_points = x_points;
        // A given plasma current: I_p is proportional to the coefficients' common factor, so the given coefficients' current sets it
        // exactly. If it cannot, the error is kept for `solve`, which raises it (and Lawson's weighting, which stops)
        if let Some(ip) = self.ip {
            let current_given: f64 = self.plasma_current(&solution, &solution.scale, N_PLASMA_CURRENT_NODE);
            if current_given.is_finite() && current_given != 0.0 {
                solution.scale = self.physical_scale(solution.k_max, ip / current_given);
            } else {
                solution.scale_error = Some(format!(
                    "BuxtonProkopyszyn: the given coefficients' plasma current is {current_given}, so the plasma current cannot set their scale (an LCFS that does not close, for example)"
                ));
            }
        }
        // p' fixes the pressure only up to a constant, here zero on the LCFS, so a negative pressure on the axis is the profiles', not a
        // failure of the solution; its sign is the same at every coefficient scale
        if !(solution.scale.p_axis >= 0.0) {
            warnings.push(format!(
                "BuxtonProkopyszyn: the profiles give a negative pressure on the magnetic axis, {:.6e} Pa, with zero pressure on the LCFS",
                solution.scale.p_axis
            ));
        }
        solution.boundary_polygon = self.boundary_points(&lattice, &one_minus_psi_n, solution.mag, &solution.x_points, N_BOUNDARY_POLYGON_POINT);
        // Closed, nested surfaces with one maximum, inside the region where the radial series can be evaluated, with R**2 within
        // R_c**2 -+ 1.3 times the plasma's half-width in R**2
        let value_function = |r: &[f64], z: &[f64]| -> Array1<f64> { series_values(&lattice, &one_minus_psi_n, r, z) };
        let closure_range: ValidRange = r_squared_range(&radial, 1.3);
        solution.closure = closure_check(
            &value_function,
            solution.mag,
            closure_range.r_minimum,
            closure_range.r_maximum,
            z_bottom,
            z_top,
            &solution.x_points,
            &solution.boundary_polygon,
        );
        if !solution.closure.closed {
            warnings.push(format!(
                "BuxtonProkopyszyn: the LCFS is not a closed set of nested surfaces (closed {}, reaches_box_edge {}, n_maximum {}, monotonic {}); try more wavenumbers (n_wave) or more LCFS points",
                solution.closure.closed, solution.closure.reaches_box_edge, solution.closure.n_maximum, solution.closure.monotonic
            ));
        }
        if solution.is_unstable {
            warnings.push("BuxtonProkopyszyn: the series' coefficient sums grow (unstable); try a neighbouring n_wave".to_string());
        }
        if !(solution.gs_residual <= GS_RESIDUAL_WARNING) {
            warnings.push(format!(
                "BuxtonProkopyszyn: the series has a GS residual of {:.1e}, above {GS_RESIDUAL_WARNING:.0e}: it is not a converged solution; try a neighbouring n_wave",
                solution.gs_residual
            ));
        }
        // The boundary misfit: largest distance from the given LCFS to the analytic one, |1 - psi_N| / |grad (1 - psi_N)|
        let outline: Points = self.configuration.outline(r_geo, a_minor, 257)?;
        let flux_outline: FluxValues = series_evaluate(
            &lattice,
            &one_minus_psi_n,
            outline.r.as_slice().expect("contiguous"),
            outline.z.as_slice().expect("contiguous"),
        );
        let n_outline: usize = outline.r.len();
        let exclusion_radius: f64 = X_POINT_EXCLUSION_RADIUS * a_minor;
        let mut boundary_misfit: f64 = f64::NEG_INFINITY;
        let mut n_kept: usize = 0;
        for i_outline in 0..n_outline {
            // The first-order distance fails where the gradient vanishes, at the X-points
            let mut is_kept: bool = true;
            for x_point in &solution.x_points {
                is_kept &= (outline.r[i_outline] - x_point.r).hypot(outline.z[i_outline] - x_point.z) > exclusion_radius;
            }
            if !is_kept {
                continue;
            }
            // NaN propagates: a distance that cannot be evaluated is not a small one
            let distance: f64 = flux_outline.one_minus_psi_n[i_outline].abs() / flux_outline.d_d_r[i_outline].hypot(flux_outline.d_d_z[i_outline]);
            if distance.is_nan() || distance > boundary_misfit {
                boundary_misfit = distance;
            }
            n_kept += 1;
        }
        solution.boundary_misfit = if n_kept > 0 { boundary_misfit } else { f64::NAN };
        // The distance of each given LCFS point from the analytic LCFS, by which Lawson's weighting raises its weight
        let n_lcfs_point: usize = self.n_lcfs_point();
        let flux_points: FluxValues = series_evaluate(&lattice, &one_minus_psi_n, &conditions.r[..n_lcfs_point], &conditions.z[..n_lcfs_point]);
        let mut boundary_deviations: Vec<f64> = vec![f64::NAN; n_lcfs_point];
        'point_loop: for i_point in 0..n_lcfs_point {
            for x_point in &solution.x_points {
                if (conditions.r[i_point] - x_point.r).hypot(conditions.z[i_point] - x_point.z) <= exclusion_radius {
                    continue 'point_loop;
                }
            }
            boundary_deviations[i_point] = flux_points.one_minus_psi_n[i_point].abs() / flux_points.d_d_r[i_point].hypot(flux_points.d_d_z[i_point]);
        }
        solution.boundary_weights = boundary_weights.to_vec();
        solution.boundary_deviations = boundary_deviations;
        solution.warnings = warnings;
        Ok(solution)
    }
}

/// Analytic Grad-Shafranov equilibrium whose `p'` and `ff'` are the paper's EFIT polynomials of any order, each with a constant term,
/// `c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1)`: `c_0` is the constant (Solov'ev) term, the profile's value on the LCFS, where
/// the current density then does not vanish (Solov'ev's equilibrium has constants only); `c_1` is the coefficient of `1 - psi_N`, `c_2`
/// that of `(1 - psi_N) psi_N`, and so on (P. F. Buxton and A. P. K. Prokopyszyn).
///
/// `[0, c_1]` for both is exactly Guazzotto and Freidberg's linear solution (J. Plasma Phys. 87 (2021) 905870303). The flux is a
/// cosine series in `Z` at `n_wave` equally spaced wavenumbers `k_j`, whose coefficients are power series in `R**2 - R_c**2` (or
/// `ln(R**2 / R_c**2)`), each generated by an explicit recursion; the profile nonlinearity is summed as a Taylor series in `lambda`
/// about Guazzotto and Freidberg's solution. The shape is matched to `2 n_wave - 1` shape terms by least squares on
/// `7 + 10 (n_wave - 1)` LCFS conditions, optionally with `2 n_wave_cosh` terms `cosh(|k_j| Z) X_j(R**2)` (`k_j**2 < 0`) and
/// `10 n_wave_cosh` more conditions, which spherical double nulls need. Given LCFS points can be weighted by Lawson's iteration for the
/// smallest largest deviation (`n_boundary_lawson_iteration_max > 0`), which holds the LCFS where the flux gradient is small, as on the
/// inboard side near the beta limit.
///
/// The solution is analytic only where the Taylor series converges. When it does not, `solve` keeps the partial sum with the smallest
/// GS residual and warns (`gs_residual` above 1e-6): such an equilibrium is one for a numerical Grad-Shafranov solver.
///
/// Construct it from its settings, then call `solve`, which also fills `equilibrium_ids` on the `(R, Z)` grid and post-processes it.
/// The equilibrium is a fixed-boundary one: the flux outside the LCFS is the analytic continuation of the series (not a vacuum field).
/// The grid must lie radially where the series can be evaluated (`valid_r_range`), since the post-processor needs finite fields
/// everywhere; `psi(r, z)` and the other methods return NaN outside that range.
///
/// # Settings
///
/// It is constructed with its settings, `equilibrium.xml` with each file it links to written in its link's place, as an IMAS
/// `parameters_input`, as GSFit's `Equilibrium` is. Their XML is written to `equilibrium_ids`' `code/parameters`, and read into
/// `buxton_prokopyszyn_settings`, whose structs are generated from the settings' XML schema,
/// `python/analytic_grad_shafranov/buxton_prokopyszyn/settings_schema/xsd/equilibrium/equilibrium.xsd`.
#[pyclass(module = "gsfit_rs.analytic_grad_shafranov.buxton_prokopyszyn")]
pub struct BuxtonProkopyszyn {
    /// The Equilibrium IDS: `code/parameters` holds the settings from construction, and `solve` fills and post-processes the rest
    equilibrium_ids: Equilibrium,
    /// What the solve works from and makes, which is not IDS data: the problem resolved from the settings, and the solution
    buxton_prokopyszyn_temporary_storage: BuxtonProkopyszynTemporaryStorage,
    /// The settings, the XML of `code/parameters` read into structs
    buxton_prokopyszyn_settings: BuxtonProkopyszynSettings,
}

/// What `BuxtonProkopyszyn` holds while it solves, alongside its IDS
struct BuxtonProkopyszynTemporaryStorage {
    /// The analytic problem, resolved from the settings when constructed
    problem: Problem,
    /// The result of `solve`; `None` until it has run
    solution: Option<Solution>,
}

/// The values of a numpy array of any shape, flattened, and its shape
fn flatten(array: &PyReadonlyArrayDyn<'_, f64>) -> (Vec<f64>, Vec<usize>) {
    let view: ndarray::ArrayViewD<f64> = array.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    let mut values: Vec<f64> = Vec::with_capacity(view.len());
    for value in view.iter() {
        values.push(*value);
    }
    (values, shape)
}

/// `values` with `shape`, as a numpy array
fn reshape<'py>(py: Python<'py>, values: Vec<f64>, shape: &[usize]) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let array: ArrayD<f64> = ArrayD::from_shape_vec(IxDyn(shape), values).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Emit Python `UserWarning`s
fn emit_warnings(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    for warning in warnings {
        let message: CString = CString::new(warning.as_str()).unwrap_or_else(|_| CString::new("BuxtonProkopyszyn: warning").expect("a C string"));
        PyErr::warn(py, py.get_type::<PyUserWarning>().as_any(), &message, 1)?;
    }
    Ok(())
}

#[pymethods]
impl BuxtonProkopyszyn {
    /// Construct the analytic equilibrium from its settings; `solve` solves it.
    ///
    /// The settings are read into `buxton_prokopyszyn_settings`, and the analytic problem is resolved from them. Every setting is
    /// described, in the paper's notation, in the XML schemas in
    /// `python/analytic_grad_shafranov/buxton_prokopyszyn/settings_schema/xsd/equilibrium/`. The coefficients fix the equilibrium:
    /// `psi_a - psi_b`, the plasma current and `p_axis` follow from them. With `global_quantities/ip`, both lists are multiplied by the one
    /// factor that gives that current instead (`coefficient_scale`, the paper's `s`), which leaves the flux surfaces unchanged (only the
    /// ratios of the coefficients shape them).
    ///
    /// # Arguments
    /// * `parameters_input` - the settings, as a `gsfit_rs.dd_support.ParametersInput`, whose XML is `equilibrium.xml` with each file
    ///   it links to written in its link's place; it is written to `equilibrium_ids`' `code/parameters`, and read into
    ///   `buxton_prokopyszyn_settings`
    ///
    /// # Errors
    /// A `ValueError` if the XML cannot be read into `buxton_prokopyszyn_settings`: it is not well-formed, it is not `equilibrium.xml`,
    /// it holds an element which is not in the XML schema the structs were generated from, or lacks one, or a value cannot be read as
    /// its type. Or when the settings cannot be solved: a shape the configuration rejects; an unknown option; `n_wave < 4`, `k_spacing`
    /// outside `(0, 1)`, `k_spacing_cosh <= 0` with `cosh` terms, `cosh` terms on an up-down asymmetric shape, `n_radial < 20`, a
    /// negative or non-finite `radial_tolerance`, or `n_taylor_max < 1`;
    /// coefficients that are empty or not finite, or all zero; a non-positive or non-finite `r0`; a zero or non-finite `b0` or `ip`; a grid with
    /// fewer than 4 points either way, `r_min <= 0`, an empty extent, or radially outside `valid_r_range`; `n_psi_norm < 3`;
    /// `n_boundary_lawson_iteration_max > 0` without `LcfsPoints`; or profiles whose linear part's
    /// value at `R_c = (r_geo**2 + a_minor**2)**0.5`, `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` (the bracket of the paper's
    /// `k_max**2`), vanishes.
    #[new]
    pub fn new(py: Python<'_>, parameters_input: &PyParametersInput) -> PyResult<Self> {
        let error = |message: String| PyValueError::new_err(format!("BuxtonProkopyszyn: {message}"));
        // The settings, read into `buxton_prokopyszyn_settings`
        let parameters_value: &str = &parameters_input.inner.parameters_value;
        let settings: BuxtonProkopyszynSettings =
            BuxtonProkopyszynSettings::from_xml(parameters_value).map_err(|message: String| error(format!("the settings cannot be read: {message}")))?;
        let configuration: Configuration = Configuration::from_settings(&settings.configuration).map_err(error)?;
        let profiles: &BuxtonProkopyszynSettingsProfiles = &settings.profiles;
        let p_prime_coefficients: Vec<f64> = profiles.p_prime_coefficients.to_vec();
        let ff_prime_coefficients: Vec<f64> = profiles.ff_prime_coefficients.to_vec();
        let r0: f64 = settings.vacuum_toroidal_field.r0;
        let b0: f64 = settings.vacuum_toroidal_field.b0;
        let ip: Option<f64> = settings.global_quantities.ip;
        let grid: &BuxtonProkopyszynSettingsGrid = &settings.grid;
        let n_r: usize = grid.n_r;
        let n_z: usize = grid.n_z;
        let r_min: f64 = grid.r_min;
        let r_max: f64 = grid.r_max;
        let z_min: f64 = grid.z_min;
        let z_max: f64 = grid.z_max;
        let n_psi_norm: usize = grid.n_psi_norm;
        let numerics: &BuxtonProkopyszynSettingsNumerics = &settings.numerics;
        let n_wave: usize = numerics.wavenumbers.n_wave;
        let k_spacing: f64 = numerics.wavenumbers.k_spacing;
        let n_wave_cosh: usize = numerics.wavenumbers.n_wave_cosh;
        let k_spacing_cosh: f64 = numerics.wavenumbers.k_spacing_cosh;
        let radial_variable: &str = &numerics.radial_series.radial_variable;
        let n_radial: usize = numerics.radial_series.n_radial;
        let radial_tolerance: f64 = numerics.radial_series.radial_tolerance;
        let c_bar_method: &str = &numerics.taylor_series.c_bar_method;
        let n_taylor_max: usize = numerics.taylor_series.n_taylor_max;
        let taylor_tolerance: f64 = numerics.taylor_series.taylor_tolerance;
        let fit_regularisation_method: &str = &numerics.boundary_fitting.fit_regularisation_method;
        let fit_threshold: f64 = numerics.boundary_fitting.fit_threshold;
        let exact_x_points: bool = numerics.boundary_fitting.exact_x_points;
        let n_boundary_lawson_iteration_max: usize = numerics.boundary_fitting.n_boundary_lawson_iteration_max;
        let verbose: bool = numerics.verbose;
        configuration.validate().map_err(error)?;
        // Lawson's weighting re-weights the given LCFS points
        if n_boundary_lawson_iteration_max > 0 && !matches!(configuration, Configuration::LcfsPoints { .. }) {
            return Err(error(format!(
                "n_boundary_lawson_iteration_max = {n_boundary_lawson_iteration_max}: Lawson's weighting needs LcfsPoints; use 0 with a model surface"
            )));
        }
        if let Some(ip) = ip {
            if !(ip.is_finite() && ip != 0.0) {
                return Err(error(format!("ip must be finite and non-zero, not {ip}")));
            }
        }
        if !(r0.is_finite() && r0 > 0.0) {
            return Err(error(format!("r0 must be finite and positive, not {r0}")));
        }
        if !(b0.is_finite() && b0 != 0.0) {
            return Err(error(format!("b0 must be finite and non-zero, not {b0}")));
        }
        if n_r < 4 || n_z < 4 || !(r_max > r_min) || !(z_max > z_min) || r_min <= 0.0 {
            return Err(error("the grid needs n_r, n_z >= 4, 0 < r_min < r_max and z_min < z_max".to_string()));
        }
        if n_psi_norm < 3 {
            return Err(error("n_psi_norm must be at least 3".to_string()));
        }
        // The solver works relative to z_geo
        let z_geo: f64 = configuration.z_geo();
        let configuration_centred: Configuration = configuration.shifted_down(z_geo);
        let scale: RadialScale = configuration_centred.scale();
        let r_geo_resolved: f64 = scale.r_geo;
        let a_minor_resolved: f64 = scale.a_minor;
        if !(r_geo_resolved.is_finite() && r_geo_resolved > 0.0) {
            return Err(error(format!("r_geo must be positive and finite, not {r_geo_resolved}")));
        }
        if !(a_minor_resolved > 0.0 && a_minor_resolved < r_geo_resolved) {
            return Err(error(format!(
                "a_minor must lie between 0 and r_geo = {r_geo_resolved}, not {a_minor_resolved}"
            )));
        }
        let r_c_squared: f64 = r_geo_resolved * r_geo_resolved + a_minor_resolved * a_minor_resolved;
        // The coefficients, each list padded to [c_0, c_1] at least, so that both have a linear term
        let mut p_prime_coefficients_padded: Vec<f64> = p_prime_coefficients;
        let mut ff_prime_coefficients_padded: Vec<f64> = ff_prime_coefficients;
        if p_prime_coefficients_padded.is_empty() || ff_prime_coefficients_padded.is_empty() {
            return Err(error(
                "p_prime_coefficients and ff_prime_coefficients need at least one value each, c_0".to_string(),
            ));
        }
        if p_prime_coefficients_padded.len() == 1 {
            p_prime_coefficients_padded.push(0.0);
        }
        if ff_prime_coefficients_padded.len() == 1 {
            ff_prime_coefficients_padded.push(0.0);
        }
        let mut is_any_nonzero: bool = false;
        for coefficients in [&p_prime_coefficients_padded, &ff_prime_coefficients_padded] {
            let n_coefficient: usize = coefficients.len();
            for i_coefficient in 0..n_coefficient {
                if !coefficients[i_coefficient].is_finite() {
                    return Err(error(format!("the coefficients must be finite, not {}", coefficients[i_coefficient])));
                }
                is_any_nonzero |= coefficients[i_coefficient] != 0.0;
            }
        }
        if !is_any_nonzero {
            return Err(error("the coefficients are all zero: there is no plasma current".to_string()));
        }
        if n_wave < 4 {
            return Err(error(format!("n_wave must be at least 4 wavenumbers, not {n_wave}")));
        }
        if !(k_spacing > 0.0 && k_spacing < 1.0) {
            return Err(error(format!("k_spacing must lie between 0 and 1 (in units of k_max), not {k_spacing}")));
        }
        if n_wave_cosh > 0 && !(k_spacing_cosh.is_finite() && k_spacing_cosh > 0.0) {
            return Err(error(format!(
                "k_spacing_cosh must be positive and finite with cosh terms, not {k_spacing_cosh}"
            )));
        }
        if n_radial < N_RADIAL_MIN {
            return Err(error(format!("n_radial must be at least {N_RADIAL_MIN} terms, not {n_radial}")));
        }
        if !(radial_tolerance.is_finite() && radial_tolerance >= 0.0) {
            return Err(error(format!("radial_tolerance must be zero or positive and finite, not {radial_tolerance:?}")));
        }
        if n_taylor_max < 1 {
            return Err(error("n_taylor_max must be at least 1 (order 0 alone)".to_string()));
        }
        let c_bar_method_resolved: CBarMethod = match c_bar_method {
            "auto" => {
                if is_broad_profile(&p_prime_coefficients_padded, &ff_prime_coefficients_padded) {
                    CBarMethod::SlopeAtBoundary
                } else {
                    CBarMethod::ValueAtAxis
                }
            }
            "slope_at_boundary" => CBarMethod::SlopeAtBoundary,
            "value_at_axis" => CBarMethod::ValueAtAxis,
            "least_squares" => CBarMethod::LeastSquares,
            _ => {
                return Err(error(format!(
                    "c_bar_method must be auto, slope_at_boundary, value_at_axis or least_squares, not {c_bar_method:?}"
                )));
            }
        };
        // The profiles P and Q: the source functions divided by their linear part's value at R_c, which must have the sign of its p'
        // part (a ValueError here, rather than when solving)
        let normalised: NormalisedProfiles =
            normalised_profiles(&p_prime_coefficients_padded, &ff_prime_coefficients_padded, r_c_squared, c_bar_method_resolved).map_err(error)?;
        let radial_variable_resolved: RadialVariableKind = resolve_radial_variable(radial_variable, a_minor_resolved / r_geo_resolved).map_err(error)?;
        // The post-processor needs finite fields on the whole grid, and the radial series can only be evaluated in its valid range
        let (r_valid_min, r_valid_max): (f64, f64) = valid_r_range_of(r_geo_resolved, a_minor_resolved, radial_variable_resolved);
        if r_min < r_valid_min || r_max > r_valid_max {
            return Err(error(format!(
                "the grid's radial extent, {r_min:.4} <= R <= {r_max:.4} m, must lie where the radial series can be evaluated, {r_valid_min:.4} <= R <= {r_valid_max:.4} m (radial variable {}); psi(r, z) can still be evaluated anywhere, NaN outside that range",
                radial_variable_resolved.name()
            )));
        }
        let fit_regularisation_resolved: FitChoice = match fit_regularisation_method {
            "auto" => FitChoice::Auto,
            "none" => FitChoice::Given(FitRegularisation::None),
            "tikhonov" => FitChoice::Given(FitRegularisation::Tikhonov),
            "tsvd" => FitChoice::Given(FitRegularisation::Tsvd),
            _ => {
                return Err(error(format!(
                    "fit_regularisation_method must be auto, none, tikhonov or tsvd, not {fit_regularisation_method:?}"
                )));
            }
        };
        let is_up_down_symmetric: bool = configuration_centred.is_up_down_symmetric();
        if n_wave_cosh > 0 && !is_up_down_symmetric {
            return Err(error(
                "cosh terms (n_wave_cosh > 0) are implemented for up-down symmetric shapes only".to_string(),
            ));
        }
        // Guazzotto and Freidberg's linear profiles: only c_1 (the others zero) in each, so the series ends at order 0
        let mut is_linear: bool = true;
        for coefficients in [&p_prime_coefficients_padded, &ff_prime_coefficients_padded] {
            let n_coefficient: usize = coefficients.len();
            for i_coefficient in 0..n_coefficient {
                is_linear &= i_coefficient == 1 || coefficients[i_coefficient] == 0.0;
            }
        }
        if n_wave_cosh > 0 && !is_linear && k_spacing_cosh < 0.45 {
            emit_warnings(
                py,
                &["BuxtonProkopyszyn: cosh terms with nonlinear profiles and k_spacing_cosh < 0.45: the series grows x2 per order with the spacing 1/3; use 0.5".to_string()],
            )?;
        }
        let problem: Problem = Problem {
            configuration: configuration_centred,
            is_up_down_symmetric,
            r_geo: r_geo_resolved,
            a_minor: a_minor_resolved,
            r0,
            b0,
            p_prime_coefficients: p_prime_coefficients_padded,
            ff_prime_coefficients: ff_prime_coefficients_padded,
            ip,
            shapes: normalised.shapes,
            linear_source_at_r_c: normalised.linear_source_at_r_c,
            n_wave,
            k_spacing,
            n_wave_cosh,
            k_spacing_cosh,
            radial_variable: radial_variable_resolved,
            n_radial,
            radial_tolerance,
            c_bar_method: c_bar_method_resolved,
            n_taylor_max,
            taylor_tolerance,
            exact_x_points,
            n_boundary_lawson_iteration_max,
            fit_regularisation: fit_regularisation_resolved,
            fit_threshold,
            verbose,
            z_geo,
        };
        // The IDS records the settings it is made from in `code/parameters`, as GSFit's `Equilibrium` does; `solve` fills the rest
        let equilibrium_ids: Equilibrium = Equilibrium {
            code: Code {
                parameters: parameters_value.to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(BuxtonProkopyszyn {
            equilibrium_ids,
            buxton_prokopyszyn_temporary_storage: BuxtonProkopyszynTemporaryStorage { problem, solution: None },
            buxton_prokopyszyn_settings: settings,
        })
    }

    /// Solve the normalised problem, then the physical scalars and the diagnostics; then fill the Equilibrium IDS on the grid and
    /// post-process it.
    ///
    /// With `n_boundary_lawson_iteration_max > 0` the solve is repeated with the LCFS points reweighted (Lawson's iteration), and the
    /// one with the smallest boundary misfit kept (`options_used["weighting_misfits"]` and `["weighting_stop"]` say how it went).
    pub fn solve(&mut self, py: Python<'_>) -> PyResult<()> {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let solution: Solution = problem.solve_weighted().map_err(PyRuntimeError::new_err)?;
        // The warnings before a scale error, so that it does not lose them
        emit_warnings(py, &solution.warnings)?;
        if let Some(scale_error) = &solution.scale_error {
            return Err(PyRuntimeError::new_err(scale_error.clone()));
        }
        self.equilibrium_ids = self.build_equilibrium_ids(&solution).map_err(PyRuntimeError::new_err)?;
        self.buxton_prokopyszyn_temporary_storage.solution = Some(solution);
        let (p_prime_source_function, ff_prime_source_function): (SharedSourceFunction, SharedSourceFunction) = self.source_functions();
        // It has no magnetic sensors to reconstruct, and so no Greens tables
        equilibrium_post_processor(
            &mut self.equilibrium_ids,
            &GreensTables::default(),
            &p_prime_source_function,
            &ff_prime_source_function,
        );
        Ok(())
    }

    /// The wavenumbers of the series in units of `k_max`, to two decimals, in order of `k**2`: the `cosh` terms' imaginary ones,
    /// `k_j = i m k_spacing_cosh k_max` (`k_j**2 < 0`, `cosh(|k_j| Z) = cos(k_j Z)`), `m = n_wave_cosh..1`, then the paper's
    /// `k_j = (j - 1) k_spacing k_max`, `j = 1..n_wave`, e.g. `k = [1.0i, 0.5i, 0.0, 0.33, 0.67, 1.0, 1.33] * k_max`
    fn __repr__(&self) -> String {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let n_wave_cosh: usize = problem.n_wave_cosh;
        let n_wave: usize = problem.n_wave;
        let mut wavenumbers: Vec<String> = Vec::with_capacity(n_wave_cosh + n_wave);
        // The cosh terms' imaginary wavenumbers, the largest first
        for i_wave_cosh in 0..n_wave_cosh {
            // |k_j| / k_max of the cosh term m = n_wave_cosh - i_wave_cosh, rounded to two decimals [dimensionless]
            let k_over_k_max: f64 = (100.0 * ((n_wave_cosh - i_wave_cosh) as f64) * problem.k_spacing_cosh).round() / 100.0;
            wavenumbers.push(format!("{k_over_k_max:?}i"));
        }
        for i_wave in 0..n_wave {
            // k_j / k_max, rounded to two decimals [dimensionless]
            let k_over_k_max: f64 = (100.0 * (i_wave as f64) * problem.k_spacing).round() / 100.0;
            wavenumbers.push(format!("{k_over_k_max:?}"));
        }
        format!("k = [{}] * k_max", wavenumbers.join(", "))
    }

    /// The Equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`; empty until `solve` has run. The IDS is copied into
    /// the returned object, so it is a snapshot.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        PyEquilibrium::new(self.equilibrium_ids.clone())
    }

    /// The paper's `k_max`, `k_max**2 = (2 pi)**2 (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime) / (psi_a - psi_b)` for the linear part's
    /// `c_bar` (`options_used["c_bar_method"]`): the sum of the orders' `k_max,l**2` kept [1 / metre]
    #[getter]
    fn k_max(&self) -> PyResult<f64> {
        Ok(self.solved()?.k_max)
    }

    /// The paper's `psi_a - psi_b`, with `psi_b = 0` [weber]
    #[getter]
    fn psi_a_minus_psi_b(&self) -> PyResult<f64> {
        Ok(self.solved()?.scale.psi_a_minus_psi_b)
    }

    /// The pressure on the magnetic axis [pascal]
    #[getter]
    fn p_axis(&self) -> PyResult<f64> {
        Ok(self.solved()?.scale.p_axis)
    }

    /// The factor the given coefficients were multiplied by: 1, or the one that gives `global_quantities/ip` [dimensionless]
    #[getter]
    fn coefficient_scale(&self) -> PyResult<f64> {
        Ok(self.solved()?.scale.coefficient_scale)
    }

    /// Guazzotto and Freidberg's `nu` of the profiles, an output only (the solution does not use it): the share of the toroidal
    /// current density at `R_c = (r_geo**2 + a_minor**2)**0.5` carried by `p'`, with both profiles at their values on the magnetic axis
    /// (`c_0 + c_1`), `mu0 R_c**2 p'(axis) / (mu0 R_c**2 p'(axis) + ff'(axis))`. It is their `nu` for their linear profiles, 0 without
    /// `p'`, 1 without `ff'`, above 1 for a diamagnetic `ff'`, negative where `p'` opposes the total; infinite where the two cancel,
    /// NaN where both vanish on the axis
    /// [dimensionless]
    #[getter]
    fn nu(&self) -> f64 {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let r_c_squared: f64 = problem.r_geo * problem.r_geo + problem.a_minor * problem.a_minor;
        nu_on_axis(&problem.p_prime_coefficients, &problem.ff_prime_coefficients, r_c_squared)
    }

    /// The paper's `R_geo`: the model surface's, or the middle of the radial extent of the LCFS points and the X-point [metre]
    #[getter]
    fn r_geo(&self) -> f64 {
        self.buxton_prokopyszyn_temporary_storage.problem.r_geo
    }

    /// The height about which the series is built: the model surface's `z_geo`, or the middle of the vertical extent of the LCFS points
    /// and the X-point [metre]
    #[getter]
    fn z_geo(&self) -> f64 {
        self.buxton_prokopyszyn_temporary_storage.problem.z_geo
    }

    /// The paper's `a_minor`: the model surface's, or half the radial extent of the LCFS points and the X-point [metre]
    #[getter]
    fn a_minor(&self) -> f64 {
        self.buxton_prokopyszyn_temporary_storage.problem.a_minor
    }

    /// The magnetic axis' major radius [metre]
    #[getter]
    fn mag_r(&self) -> PyResult<f64> {
        let solution: &Solution = self.solved()?;
        Ok(self.buxton_prokopyszyn_temporary_storage.problem.machine_coordinates(solution.mag).0)
    }

    /// The magnetic axis' height in the machine [metre]
    #[getter]
    fn mag_z(&self) -> PyResult<f64> {
        let solution: &Solution = self.solved()?;
        Ok(self.buxton_prokopyszyn_temporary_storage.problem.machine_coordinates(solution.mag).1)
    }

    /// The paper's GS residual: the largest difference between the two sides of the Grad-Shafranov equation inside the LCFS, over the
    /// largest value of its right-hand side, on a 61 x 61 grid and along rays from the magnetic axis up to the solution's own LCFS
    /// [dimensionless]
    #[getter]
    fn gs_residual(&self) -> PyResult<f64> {
        Ok(self.solved()?.gs_residual)
    }

    /// Largest distance from the given shape (model surface or points) to the analytic LCFS [metre]
    #[getter]
    fn boundary_misfit(&self) -> PyResult<f64> {
        Ok(self.solved()?.boundary_misfit)
    }

    /// The relative weight of each given LCFS point's row in the kept solution's fits: all 1 with `n_boundary_lawson_iteration_max = 0`,
    /// else Lawson's, normalised to `mean(w**2) = 1`; empty for a model surface [dimensionless]
    #[getter]
    fn boundary_weights_used<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.boundary_weights.clone()).into_pyarray(py))
    }

    /// The distance of each given LCFS point from the analytic LCFS, `|1 - psi_N| / |grad (1 - psi_N)|`, in the given order; NaN within
    /// 0.05 `a_minor` of an X-point; empty for a model surface [metre]
    #[getter]
    fn boundary_deviations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.boundary_deviations.clone()).into_pyarray(py))
    }

    /// `1 - psi_N` at the X-point, 0 when the separatrix is the LCFS (about 1e-16 with exact X-point conditions) [dimensionless]
    #[getter]
    fn x_point_flux(&self) -> PyResult<f64> {
        Ok(self.solved()?.x_point_flux)
    }

    /// The LCFS closes, with nested surfaces and one maximum, inside the region the series can be evaluated
    #[getter]
    fn is_closed(&self) -> PyResult<bool> {
        Ok(self.solved()?.closure.closed)
    }

    /// The coefficient sums of the series grow (an unreliable case; try a neighbouring `n_wave`)
    #[getter]
    fn is_unstable(&self) -> PyResult<bool> {
        Ok(self.solved()?.is_unstable)
    }

    /// The number of Taylor terms kept, orders `0..n_taylor - 1`: the partial sum with the smallest GS residual (the paper's
    /// `n_Taylor`) [dimensionless]
    #[getter]
    fn n_taylor(&self) -> PyResult<usize> {
        Ok(self.solved()?.n_taylor)
    }

    /// The fit of orders >= 1 kept: `none`, `tikhonov` or `tsvd`
    #[getter]
    fn fit_used(&self) -> PyResult<String> {
        Ok(self.solved()?.fit_used.name().to_string())
    }

    /// The GS residual of every partial sum of the series kept [dimensionless]
    #[getter]
    fn order_gs_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.solved()?.order_gs_residuals.clone().into_pyarray(py))
    }

    /// `sum |d_{j,m}|` of every order of the series, the paper's `(1 - psi_N)_l` [dimensionless]
    #[getter]
    fn coefficient_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.solved()?.coefficient_sizes.clone().into_pyarray(py))
    }

    /// The paper's `k_max,l**2` of every order [1 / metre ** 2]
    #[getter]
    fn k_max_squared_orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.series.k_max_squared_orders.clone()).into_pyarray(py))
    }

    /// The series' coefficients in `lambda` of the magnetic axis' shift from order 0's, in the scaled radial variable:
    /// `(R**2 - R_c**2) / (2 R_geo a_minor)` (`r_squared`), or `ln(R**2 / R_c**2)` scaled to -1 and 1 at the plasma's innermost and
    /// outermost `R` (`log_r_squared`) [dimensionless]
    #[getter]
    fn axis_radial_orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.series.axis_radial_orders.clone()).into_pyarray(py))
    }

    /// ... and in `Z` [metre]
    #[getter]
    fn axis_z_orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.series.axis_z_orders.clone()).into_pyarray(py))
    }

    /// The number of distinct non-zero wavenumbers in every order of the series [dimensionless]
    #[getter]
    fn n_wavenumbers(&self) -> PyResult<Vec<usize>> {
        Ok(self.solved()?.series.n_wavenumbers.clone())
    }

    /// The largest multiple of the wavenumber spacing (and of the `cosh` terms' spacing) in every order of the series [dimensionless]
    #[getter]
    fn order_half_widths(&self) -> PyResult<Vec<Vec<usize>>> {
        let solution: &Solution = self.solved()?;
        let mut half_widths: Vec<Vec<usize>> = Vec::with_capacity(solution.series.one_minus_psi_n_orders.len());
        for order in &solution.series.one_minus_psi_n_orders {
            half_widths.push(order.half_widths.clone());
        }
        Ok(half_widths)
    }

    /// The largest source the growth cut-offs dropped (wavenumbers far above `k_max`, whose radial functions grow too fast across the
    /// plasma), relative to `k_max,0**2` times the size of order 0, the size of the terms of its equation [dimensionless]
    #[getter]
    fn dropped_source(&self) -> PyResult<f64> {
        Ok(self.solved()?.series.dropped_source)
    }

    /// The terms of one order of the Taylor series at one point, for looking at its sums; with `verbose = True` only.
    ///
    /// Order `i_taylor`'s `(1 - psi_N)_l` at `(R, Z)` is `sum_j sum_m terms[j, m]`: `terms[j, m]` is the radial power `m` of the
    /// wavenumber `k_j = k_over_k_max[j] k_max`, with `k_j` and `-k_j` together, `cos(k_j (Z - z_geo))` and `sin(k_j (Z - z_geo))`
    /// (`cosh` and `sinh` for an imaginary `k_j`). The sum of the orders `0..n - 1` (`lambda = 1`), divided by its value on its own
    /// magnetic axis, is the partial sum whose GS residual is `order_gs_residuals[n - 1]`.
    ///
    /// The terms are also split as the paper's sum over the wavenumbers,
    /// `cos(k_j Z') [A_j C_j + B_j S_j + P_j] + sin(k_j Z') [A~_j C_j + B~_j S_j + P~_j]`, `Z' = Z - z_geo`: `C_j` is 1 with zero slope
    /// at `R_c` and `S_j` 0 with unit slope there, as order 0's terms are built; `A_j` and `B_j` are the value and slope at `R_c` of the
    /// radial function of `cos(k_j Z')`, and `P_j`, zero with zero slope at `R_c`, is the rest: the particular part of orders >= 1,
    /// rounding at order 0. Likewise for `sin(k_j Z')`; `terms` is the sum of the six parts.
    ///
    /// # Arguments
    /// * `i_taylor` - the order, below `len(k_max_squared_orders)`; 0 is Guazzotto and Freidberg's solution of the profiles' linear part
    ///   [dimensionless]
    /// * `r` - major radius [metre]
    /// * `z` - height [metre]
    ///
    /// # Returns
    /// * dict: `k_over_k_max`, the wavenumbers in units of order 0's `k_max`, complex, imaginary for the `cosh` terms [dimensionless];
    ///   `terms`, `(n_k, n_radial)` [dimensionless]; its parts `terms_a` (`cos(k_j Z') A_j C_j`), `terms_b` (`cos(k_j Z') B_j S_j`),
    ///   `terms_a_tilde` (`sin(k_j Z') A~_j C_j`), `terms_b_tilde` (`sin(k_j Z') B~_j S_j`), `terms_particular` (`cos(k_j Z') P_j`) and
    ///   `terms_particular_tilde` (`sin(k_j Z') P~_j`), each `(n_k, n_radial)` [dimensionless]; `radial`, the radial variable at `R`,
    ///   -1 and +1 at the plasma's innermost and outermost `R` [dimensionless]; and `k_max`, order 0's [1 / metre]
    ///
    /// # Errors
    /// Not solved, constructed without `verbose`, or an order beyond those computed
    fn order_terms<'py>(&self, py: Python<'py>, i_taylor: usize, r: f64, z: f64) -> PyResult<Bound<'py, PyDict>> {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        if !problem.verbose {
            return Err(PyValueError::new_err(
                "BuxtonProkopyszyn.order_terms: construct with verbose=True to look at the series' terms",
            ));
        }
        let solution: &Solution = self.solved()?;
        let n_order: usize = solution.series.one_minus_psi_n_orders.len();
        if i_taylor >= n_order {
            return Err(PyValueError::new_err(format!(
                "BuxtonProkopyszyn.order_terms: i_taylor must be below {n_order}, the orders computed, not {i_taylor}"
            )));
        }
        let lattice: &Lattice = &solution.series.lattice;
        let operator: &ReferenceOperator = &solution.series.operator;
        let series: &LatticeSeries = &solution.series.one_minus_psi_n_orders[i_taylor];
        let point: Point = problem.point_from_machine(r, z);
        let radial_variable: &RadialVariable = &lattice.radial;
        let radial: f64 = radial_variable.of_r(point.r);
        // R = R_c, where C_j has unit value and S_j unit slope in (R**2 - R_c**2) / (2 R_geo a_minor), in the radial variable, and the
        // radial variable's derivative there in (R**2 - R_c**2) / (2 R_geo a_minor)
        let r_c: f64 = radial_variable.r_c_squared.sqrt();
        let radial_at_r_c: f64 = radial_variable.of_r(r_c);
        let d_radial_at_r_c: f64 = radial_variable.derivatives_of_radial(r_c).d_radial_d_r * radial_variable.r_squared_half_width / (2.0 * r_c);
        // The lattice points j and -j are the rows centre + i_k and centre - i_k
        let n_row: usize = series.coefficients.nrows();
        let n_radial: usize = series.coefficients.ncols();
        let centre: usize = (n_row - 1) / 2;
        let n_k: usize = centre + 1;
        let wavenumbers: Vec<c64> = lattice.box_wavenumbers(&series.half_widths);
        let i_unit: c64 = c64::new(0.0, 1.0);
        let zero: c64 = c64::new(0.0, 0.0);
        let mut k_over_k_max: Array1<c64> = Array1::from_elem(n_k, c64::new(f64::NAN, f64::NAN));
        let mut terms: Array2<f64> = Array2::from_elem((n_k, n_radial), f64::NAN);
        // The parts of the terms, [0] of cos(k_j Z') and [1] of sin(k_j Z'): the C_j, S_j and particular parts; m = 0 has no sine
        let mut terms_c: [Array2<f64>; 2] = [Array2::zeros((n_k, n_radial)), Array2::zeros((n_k, n_radial))];
        let mut terms_s: [Array2<f64>; 2] = [Array2::zeros((n_k, n_radial)), Array2::zeros((n_k, n_radial))];
        let mut terms_particular: [Array2<f64>; 2] = [Array2::zeros((n_k, n_radial)), Array2::zeros((n_k, n_radial))];
        for i_k in 0..n_k {
            let wavenumber: c64 = wavenumbers[centre + i_k];
            k_over_k_max[i_k] = wavenumber * (1.0 / operator.k_max_0);
            let phase: c64 = wavenumber * point.z;
            let cos_phase: c64 = phase.cos();
            let sin_phase: c64 = phase.sin();
            let mut power: f64 = 1.0;
            for i_radial in 0..n_radial {
                let plus: c64 = series.coefficients[[centre + i_k, i_radial]];
                let minus: c64 = series.coefficients[[centre - i_k, i_radial]];
                // d_j exp(i k_j Z') + d_{-j} exp(-i k_j Z') = (d_j + d_{-j}) cos(k_j Z') + i (d_j - d_{-j}) sin(k_j Z'); j = 0 once
                let vertical: c64 = if i_k == 0 {
                    plus * cos_phase
                } else {
                    (plus + minus) * cos_phase + i_unit * (plus - minus) * sin_phase
                };
                terms[[i_k, i_radial]] = vertical.re * power;
                power *= radial;
            }

            // C_j and S_j of the wavenumber, as order 0's terms are built
            let pair: RadialPair = c_and_s(&lattice.radial, operator.k_max_squared_minus_k_squared(wavenumber), operator.p_prime_term);
            let n_part: usize = if i_k == 0 { 1 } else { 2 };
            for i_part in 0..n_part {
                // The radial function of cos(k_j Z'), d_j + d_{-j}, or of sin(k_j Z'), i (d_j - d_{-j})
                let mut radial_function: Array1<c64> = Array1::from_elem(n_radial, zero);
                for i_radial in 0..n_radial {
                    let plus: c64 = series.coefficients[[centre + i_k, i_radial]];
                    let minus: c64 = series.coefficients[[centre - i_k, i_radial]];
                    radial_function[i_radial] = if i_k == 0 {
                        plus
                    } else if i_part == 0 {
                        plus + minus
                    } else {
                        i_unit * (plus - minus)
                    };
                }
                // A_j and B_j: its value and slope at R_c in (R**2 - R_c**2) / (2 R_geo a_minor)
                let mut amplitude_c: c64 = zero;
                let mut amplitude_s: c64 = zero;
                let mut power_at_r_c: f64 = 1.0;
                for i_radial in 0..n_radial {
                    amplitude_c += radial_function[i_radial] * power_at_r_c;
                    if i_radial + 1 < n_radial {
                        amplitude_s += radial_function[i_radial + 1] * (((i_radial + 1) as f64) * power_at_r_c);
                    }
                    power_at_r_c *= radial_at_r_c;
                }
                amplitude_s *= d_radial_at_r_c;
                let vertical: c64 = if i_part == 0 { cos_phase } else { sin_phase };
                let mut power: f64 = 1.0;
                for i_radial in 0..n_radial {
                    let part_c: c64 = amplitude_c * pair.c[i_radial];
                    let part_s: c64 = amplitude_s * pair.s[i_radial];
                    let part_particular: c64 = radial_function[i_radial] - part_c - part_s;
                    terms_c[i_part][[i_k, i_radial]] = (vertical * part_c).re * power;
                    terms_s[i_part][[i_k, i_radial]] = (vertical * part_s).re * power;
                    terms_particular[i_part][[i_k, i_radial]] = (vertical * part_particular).re * power;
                    power *= radial;
                }
            }
        }
        let [terms_a, terms_a_tilde]: [Array2<f64>; 2] = terms_c;
        let [terms_b, terms_b_tilde]: [Array2<f64>; 2] = terms_s;
        let [terms_particular_cos, terms_particular_tilde]: [Array2<f64>; 2] = terms_particular;
        let result: Bound<'py, PyDict> = PyDict::new(py);
        result.set_item("k_over_k_max", k_over_k_max.into_pyarray(py))?;
        result.set_item("terms", terms.into_pyarray(py))?;
        result.set_item("terms_a", terms_a.into_pyarray(py))?;
        result.set_item("terms_b", terms_b.into_pyarray(py))?;
        result.set_item("terms_a_tilde", terms_a_tilde.into_pyarray(py))?;
        result.set_item("terms_b_tilde", terms_b_tilde.into_pyarray(py))?;
        result.set_item("terms_particular", terms_particular_cos.into_pyarray(py))?;
        result.set_item("terms_particular_tilde", terms_particular_tilde.into_pyarray(py))?;
        result.set_item("radial", radial)?;
        result.set_item("k_max", operator.k_max_0)?;
        Ok(result)
    }

    /// `[c_0_p_prime, c_1_p_prime, ...]` of the solution: the given ones (padded to two) times `coefficient_scale`. GSFit's
    /// `EfitPolynomial` DOFs are `[c_1_p_prime, ...]`, for a GSFit run with the exact profiles when `c_0` is zero [pascal / weber]
    #[getter]
    fn p_prime_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.scale.p_prime_coefficients.clone()).into_pyarray(py))
    }

    /// `[c_0_ff_prime, c_1_ff_prime, ...]` of the solution, likewise [tesla ** 2 * metre ** 2 / weber]
    #[getter]
    fn ff_prime_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.solved()?.scale.ff_prime_coefficients.clone()).into_pyarray(py))
    }

    /// Every option as `solve` resolved it, the fit used, and the `gsfit_rs` version: what a stored benchmark needs to be regenerated
    /// exactly
    #[getter]
    fn options_used<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let solution: &Solution = self.solved()?;
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let options: Bound<'py, PyDict> = PyDict::new(py);
        options.set_item("configuration", problem.configuration.name())?;
        options.set_item("r_geo", problem.r_geo)?;
        options.set_item("z_geo", problem.z_geo)?;
        options.set_item("a_minor", problem.a_minor)?;
        options.set_item("p_prime_coefficients", problem.p_prime_coefficients.clone())?;
        options.set_item("ff_prime_coefficients", problem.ff_prime_coefficients.clone())?;
        options.set_item("ip", problem.ip)?;
        options.set_item("r0", problem.r0)?;
        options.set_item("b0", problem.b0)?;
        options.set_item("n_wave", problem.n_wave)?;
        options.set_item("k_spacing", problem.k_spacing)?;
        options.set_item("n_wave_cosh", solution.n_wave_cosh)?;
        options.set_item("k_spacing_cosh", problem.k_spacing_cosh)?;
        options.set_item("exact_x_points", problem.exact_x_points)?;
        options.set_item("n_boundary_lawson_iteration_max", problem.n_boundary_lawson_iteration_max)?;
        options.set_item("weighting_misfits", solution.weighting_misfits.clone())?;
        options.set_item("weighting_stop", solution.weighting_stop.clone())?;
        options.set_item("c_bar_method", problem.c_bar_method.name())?;
        options.set_item("radial_variable", problem.radial_variable.name())?;
        options.set_item("n_radial", problem.n_radial)?;
        options.set_item("radial_tolerance", problem.radial_tolerance)?;
        options.set_item("radial_tail", solution.radial_tail)?;
        options.set_item("n_radial_needed", solution.n_radial_needed)?;
        options.set_item("n_taylor_max", problem.n_taylor_max)?;
        options.set_item("taylor_tolerance", problem.taylor_tolerance)?;
        let fit_regularisation_method: &str = match problem.fit_regularisation {
            FitChoice::Auto => "auto",
            FitChoice::Given(regularisation) => regularisation.name(),
        };
        options.set_item("fit_regularisation_method", fit_regularisation_method)?;
        options.set_item("fit_threshold", problem.fit_threshold)?;
        options.set_item("fit_used", solution.fit_used.name())?;
        options.set_item("interior_penalty", solution.series.interior_penalty)?;
        options.set_item("interior_penalty_choice", solution.series.interior_penalty_choice.clone())?;
        options.set_item("n_gs_residual_edge_ray", solution.n_edge_ray)?;
        options.set_item("n_plasma_current_node", N_PLASMA_CURRENT_NODE)?;
        options.set_item("n_taylor", solution.n_taylor)?;
        options.set_item("gsfit_rs_version", env!("CARGO_PKG_VERSION"))?;
        Ok(options)
    }

    /// The warnings `solve` raised (also emitted as Python `UserWarning`s)
    #[getter]
    fn warnings(&self) -> PyResult<Vec<String>> {
        Ok(self.solved()?.warnings.clone())
    }

    /// # Arguments
    /// * `r` - major radius, `R > 0`, any shape [metre]
    /// * `z` - vertical position, same shape [metre]
    ///
    /// # Returns
    /// * the total poloidal flux `psi`, `psi_a` on the magnetic axis and `psi_b = 0` on the LCFS [weber]
    fn psi<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let derivatives: PsiDerivatives = self.psi_derivatives_of(&r_values, &z_values)?;
        reshape(py, derivatives.psi.to_vec(), &shape)
    }

    /// The paper's `psi_N`, 0 on the magnetic axis and 1 on the LCFS [dimensionless]
    fn psi_n<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let psi_n: Vec<f64> = self.psi_n_of(&r_values, &z_values)?;
        reshape(py, psi_n, &shape)
    }

    /// # Returns
    /// * `psi` [weber]; `d_psi_d_r`, `d_psi_d_z` [weber / metre]; `d2_psi_d_r2`, `d2_psi_d_r_d_z`, `d2_psi_d_z2` [weber / metre ** 2]
    fn psi_derivatives<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyDict>> {
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let derivatives: PsiDerivatives = self.psi_derivatives_of(&r_values, &z_values)?;
        let result: Bound<'py, PyDict> = PyDict::new(py);
        result.set_item("psi", reshape(py, derivatives.psi.to_vec(), &shape)?)?;
        result.set_item("d_psi_d_r", reshape(py, derivatives.d_psi_d_r.to_vec(), &shape)?)?;
        result.set_item("d_psi_d_z", reshape(py, derivatives.d_psi_d_z.to_vec(), &shape)?)?;
        result.set_item("d2_psi_d_r2", reshape(py, derivatives.d2_psi_d_r2.to_vec(), &shape)?)?;
        result.set_item("d2_psi_d_r_d_z", reshape(py, derivatives.d2_psi_d_r_d_z.to_vec(), &shape)?)?;
        result.set_item("d2_psi_d_z2", reshape(py, derivatives.d2_psi_d_z2.to_vec(), &shape)?)?;
        Ok(result)
    }

    /// The poloidal field of the plasma's own analytic flux, GSFit's convention.
    ///
    /// # Returns
    /// * `(b_r, b_z)`, `B_R = -(d psi / d Z) / (2 pi R)`, `B_Z = (d psi / d R) / (2 pi R)` [tesla]
    fn b_field<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArrayDyn<'py, f64>,
        z: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let derivatives: PsiDerivatives = self.psi_derivatives_of(&r_values, &z_values)?;
        let n_point: usize = r_values.len();
        let mut b_r: Vec<f64> = Vec::with_capacity(n_point);
        let mut b_z: Vec<f64> = Vec::with_capacity(n_point);
        for i_point in 0..n_point {
            b_r.push(-derivatives.d_psi_d_z[i_point] / (2.0 * PI * r_values[i_point]));
            b_z.push(derivatives.d_psi_d_r[i_point] / (2.0 * PI * r_values[i_point]));
        }
        Ok((reshape(py, b_r, &shape)?, reshape(py, b_z, &shape)?))
    }

    /// Inside the analytic LCFS, shaped like `r`
    fn is_inside<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let inside: Vec<bool> = self.is_inside_of(&r_values, &z_values)?;
        let array: ArrayD<bool> = ArrayD::from_shape_vec(IxDyn(&shape), inside).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    /// The toroidal field `B_phi = f(psi_N) / R` inside the LCFS and `r0 b0 / R` outside [tesla]
    fn b_field_phi<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let inside: Vec<bool> = self.is_inside_of(&r_values, &z_values)?;
        let psi_n: Vec<f64> = self.psi_n_of(&r_values, &z_values)?;
        let n_point: usize = r_values.len();
        let mut b_field_phi: Vec<f64> = Vec::with_capacity(n_point);
        for i_point in 0..n_point {
            let psi_n_point: f64 = if inside[i_point] { psi_n[i_point] } else { 1.0 };
            b_field_phi.push(self.buxton_prokopyszyn_temporary_storage.problem.f(&solution.scale, psi_n_point) / r_values[i_point]);
        }
        reshape(py, b_field_phi, &shape)
    }

    /// The toroidal current density `J_phi = 2 pi (R p' + ff' / (mu0 R))` inside the LCFS, 0 outside [ampere / metre ** 2]
    fn j_phi<'py>(&self, py: Python<'py>, r: PyReadonlyArrayDyn<'py, f64>, z: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (r_values, shape): (Vec<f64>, Vec<usize>) = flatten(&r);
        let (z_values, _): (Vec<f64>, Vec<usize>) = flatten(&z);
        let inside: Vec<bool> = self.is_inside_of(&r_values, &z_values)?;
        let psi_n: Vec<f64> = self.psi_n_of(&r_values, &z_values)?;
        let n_point: usize = r_values.len();
        let mut j_phi: Vec<f64> = Vec::with_capacity(n_point);
        for i_point in 0..n_point {
            j_phi.push(if inside[i_point] {
                self.buxton_prokopyszyn_temporary_storage
                    .problem
                    .current_density(&solution.scale, r_values[i_point], psi_n[i_point])
            } else {
                0.0
            });
        }
        reshape(py, j_phi, &shape)
    }

    /// `p(psi_N) = (psi_a - psi_b) int_{psi_N}^1 p' dpsi_N`, zero on the LCFS [pascal]
    fn pressure<'py>(&self, py: Python<'py>, psi_n: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (psi_n_values, shape): (Vec<f64>, Vec<usize>) = flatten(&psi_n);
        let mut values: Vec<f64> = Vec::with_capacity(psi_n_values.len());
        for psi_n_value in &psi_n_values {
            values.push(self.buxton_prokopyszyn_temporary_storage.problem.pressure(&solution.scale, *psi_n_value));
        }
        reshape(py, values, &shape)
    }

    /// `f = R B_phi` at `psi_N`, `f**2 = (r0 b0)**2 + 2 (psi_a - psi_b) int_{psi_N}^1 ff' dpsi_N`, with the sign of `b0` [tesla * metre]
    fn f<'py>(&self, py: Python<'py>, psi_n: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (psi_n_values, shape): (Vec<f64>, Vec<usize>) = flatten(&psi_n);
        let mut values: Vec<f64> = Vec::with_capacity(psi_n_values.len());
        for psi_n_value in &psi_n_values {
            values.push(self.buxton_prokopyszyn_temporary_storage.problem.f(&solution.scale, *psi_n_value));
        }
        reshape(py, values, &shape)
    }

    /// `p' = dp / dpsi = c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1)` at `psi_N`, with `p_prime_coefficients` [pascal / weber]
    fn p_prime<'py>(&self, py: Python<'py>, psi_n: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (psi_n_values, shape): (Vec<f64>, Vec<usize>) = flatten(&psi_n);
        let mut values: Vec<f64> = Vec::with_capacity(psi_n_values.len());
        for psi_n_value in &psi_n_values {
            values.push(self.buxton_prokopyszyn_temporary_storage.problem.p_prime(&solution.scale, *psi_n_value));
        }
        reshape(py, values, &shape)
    }

    /// `ff' = f df / dpsi = c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1)` at `psi_N`, with `ff_prime_coefficients`
    /// [tesla ** 2 * metre ** 2 / weber]
    fn ff_prime<'py>(&self, py: Python<'py>, psi_n: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let solution: &Solution = self.solved()?;
        let (psi_n_values, shape): (Vec<f64>, Vec<usize>) = flatten(&psi_n);
        let mut values: Vec<f64> = Vec::with_capacity(psi_n_values.len());
        for psi_n_value in &psi_n_values {
            values.push(self.buxton_prokopyszyn_temporary_storage.problem.ff_prime(&solution.scale, *psi_n_value));
        }
        reshape(py, values, &shape)
    }

    /// The analytic LCFS, closed.
    ///
    /// # Arguments
    /// * `n_point` - about the number of points [dimensionless]
    ///
    /// # Returns
    /// * `(boundary_r, boundary_z)` [metre]
    fn boundary<'py>(&self, py: Python<'py>, n_point: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let solution: &Solution = self.solved()?;
        let boundary: Points = self.buxton_prokopyszyn_temporary_storage.problem.boundary_points(
            &solution.series.lattice,
            &solution.one_minus_psi_n,
            solution.mag,
            &solution.x_points,
            n_point,
        );
        let (boundary_r, boundary_z): (Array1<f64>, Array1<f64>) = self.to_machine(&boundary);
        Ok((boundary_r.into_pyarray(py), boundary_z.into_pyarray(py)))
    }

    /// The shape the LCFS was fitted to: the model surface, or the user's points (and their mirror images, and the X-points), closed.
    ///
    /// # Arguments
    /// * `n_point` - about the number of points [dimensionless]
    ///
    /// # Returns
    /// * `(r, z)` [metre]
    fn model_surface<'py>(&self, py: Python<'py>, n_point: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let outline: Points = problem
            .configuration
            .outline(problem.r_geo, problem.a_minor, n_point)
            .map_err(PyValueError::new_err)?;
        let (outline_r, outline_z): (Array1<f64>, Array1<f64>) = self.to_machine(&outline);
        Ok((outline_r.into_pyarray(py), outline_z.into_pyarray(py)))
    }

    /// `(x_point_r, x_point_z)` of every X-point: the upper and lower one of a double null, the one of a single null [metre]
    fn x_points(&self) -> PyResult<Vec<(f64, f64)>> {
        let solution: &Solution = self.solved()?;
        let mut x_points: Vec<(f64, f64)> = Vec::with_capacity(solution.x_points.len());
        for x_point in &solution.x_points {
            x_points.push(self.buxton_prokopyszyn_temporary_storage.problem.machine_coordinates(*x_point));
        }
        Ok(x_points)
    }

    /// `(x_point_r, x_point_z)` of the first X-point (the upper one of a double null, the one of a single null), or `None` [metre]
    fn x_point(&self) -> PyResult<Option<(f64, f64)>> {
        Ok(self.x_points()?.first().copied())
    }

    /// The plasma current, `I_p = int J_phi dR dZ` over the LCFS, by Gauss-Legendre quadrature half by half as the LCFS is traced: about
    /// 1e-14 relative, edge currents included.
    ///
    /// # Arguments
    /// * `n_node` - Gauss-Legendre nodes in each direction of each half [dimensionless]
    ///
    /// # Returns
    /// * `I_p` [ampere]
    fn plasma_current(&self, n_node: usize) -> PyResult<f64> {
        let solution: &Solution = self.solved()?;
        Ok(self
            .buxton_prokopyszyn_temporary_storage
            .problem
            .plasma_current(solution, &solution.scale, n_node))
    }

    /// The radial extent where the radial power series can be evaluated for the settings' shape, which the IDS grid must lie in
    /// (`psi(r, z)` and the other methods are NaN outside it): `|R**2 - R_c**2| <= 0.98 R_c**2`, 98 % of the radius of convergence, with
    /// `r_squared`; with `log_r_squared`, `ln(R**2 / R_c**2)` within 1.1 half-widths of the middle of the plasma's range of it. Only the
    /// settings' `configuration` and `numerics/radial_series/radial_variable` are used, so the grid can be chosen from it.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings, as the constructor takes them
    ///
    /// # Returns
    /// * `(r_min, r_max)` [metre]
    ///
    /// # Errors
    /// A `ValueError` if the XML cannot be read into the settings' structs, or the configuration rejects the shape
    #[staticmethod]
    fn valid_r_range(parameters_input: &PyParametersInput) -> PyResult<(f64, f64)> {
        let error = |message: String| PyValueError::new_err(format!("BuxtonProkopyszyn: {message}"));
        let settings: BuxtonProkopyszynSettings = BuxtonProkopyszynSettings::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|message: String| error(format!("the settings cannot be read: {message}")))?;
        let configuration: Configuration = Configuration::from_settings(&settings.configuration).map_err(error)?;
        configuration.validate().map_err(error)?;
        let scale: RadialScale = configuration.scale();
        let kind: RadialVariableKind = resolve_radial_variable(&settings.numerics.radial_series.radial_variable, scale.a_minor / scale.r_geo).map_err(error)?;
        Ok(valid_r_range_of(scale.r_geo, scale.a_minor, kind))
    }
}

// Rust only methods
impl BuxtonProkopyszyn {
    /// The solution, or a `RuntimeError` naming `solve` when it has not run
    fn solved(&self) -> PyResult<&Solution> {
        self.buxton_prokopyszyn_temporary_storage
            .solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("BuxtonProkopyszyn: call solve() first"))
    }

    /// Points whose `Z` is measured from `z_geo` to `(R, Z)` in the machine [metre]
    fn to_machine(&self, points: &Points) -> (Array1<f64>, Array1<f64>) {
        let z_geo: f64 = self.buxton_prokopyszyn_temporary_storage.problem.z_geo;
        (points.r.clone(), points.z.mapv(|z: f64| z + z_geo))
    }

    /// The flux and its derivatives in `(R, Z)`
    fn psi_derivatives_of(&self, r: &[f64], z: &[f64]) -> PyResult<PsiDerivatives> {
        let solution: &Solution = self.solved()?;
        Ok(psi_derivatives(&self.buxton_prokopyszyn_temporary_storage.problem, solution, r, z))
    }

    /// `psi_N` at points `(R, Z)` in the machine [dimensionless]
    fn psi_n_of(&self, r: &[f64], z: &[f64]) -> PyResult<Vec<f64>> {
        let solution: &Solution = self.solved()?;
        let z_relative: Vec<f64> = relative_heights(&self.buxton_prokopyszyn_temporary_storage.problem, z);
        let flux: FluxValues = evaluate_masked(solution, r, &z_relative, true);
        Ok(flux.one_minus_psi_n.mapv(|value: f64| 1.0 - value).to_vec())
    }

    /// Inside the analytic LCFS polygon
    fn is_inside_of(&self, r: &[f64], z: &[f64]) -> PyResult<Vec<bool>> {
        let solution: &Solution = self.solved()?;
        Ok(is_inside(&self.buxton_prokopyszyn_temporary_storage.problem, solution, r, z))
    }

    /// The source functions the IDS's coefficients are written for: GSFit's `EfitPolynomial` without constant terms, the EFIT
    /// polynomial with a constant term otherwise (see `source_function`)
    fn source_functions(&self) -> (SharedSourceFunction, SharedSourceFunction) {
        let is_with_constant: bool = self.buxton_prokopyszyn_temporary_storage.problem.shapes.has_constant_terms();
        let solution: &Solution = self.buxton_prokopyszyn_temporary_storage.solution.as_ref().expect("source_functions: solved");
        let (p_prime_source_function, _): (SharedSourceFunction, Array1<f64>) =
            source_function_and_dofs(&solution.scale.p_prime_coefficients, is_with_constant);
        let (ff_prime_source_function, _): (SharedSourceFunction, Array1<f64>) =
            source_function_and_dofs(&solution.scale.ff_prime_coefficients, is_with_constant);
        (p_prime_source_function, ff_prime_source_function)
    }

    /// Build the Equilibrium IDS: one time-slice, filled with everything the equilibrium post-processor reads, as the Grad-Shafranov
    /// solver leaves it, from the analytic solution.
    ///
    /// * `profiles_2d(0)`: the analytic flux and its derivatives on the `(R, Z)` grid (NaN where `R` is outside the radial series'
    ///   valid range), the mask (inside the analytic LCFS), `psi_norm` and `j_phi`.
    /// * `global_quantities`: the analytic magnetic axis, `psi_magnetic_axis = psi_a`, and the plasma current by Gauss-Legendre
    ///   quadrature (not the grid's sum, which is first order where `j_phi` jumps at the LCFS).
    /// * `boundary`: `psi = 0`; diverted with the bounding point at the X-point (the first of a double null), or limited with it at
    ///   the innermost point of the analytic LCFS.
    /// * `contour_tree`: the axis and the X-points.
    /// * `source_functions`: the GSFit DOFs of the analytic `p'` and `ff'` (the constants appended when there are any).
    /// * `limiter`: the analytic equilibrium has no wall, so `unit(0)`, the vacuum vessel the post-processor traces the scrape-off
    ///   layer up to, is the grid's boundary.
    fn build_equilibrium_ids(&self, solution: &Solution) -> Result<Equilibrium, String> {
        let problem: &Problem = &self.buxton_prokopyszyn_temporary_storage.problem;
        let grid: &BuxtonProkopyszynSettingsGrid = &self.buxton_prokopyszyn_settings.grid;
        let n_r: usize = grid.n_r;
        let n_z: usize = grid.n_z;
        // The grid ends exactly at `r_max` and `z_max`: `Array1::linspace`'s last point, `r_min + (n_r - 1) * d_r`, can round past
        // `r_max`, which can be the edge of the radial series' valid range, beyond which the flux is NaN
        let mut r: Array1<f64> = Array1::linspace(grid.r_min, grid.r_max, n_r);
        r[n_r - 1] = grid.r_max;
        let mut z: Array1<f64> = Array1::linspace(grid.z_min, grid.z_max, n_z);
        z[n_z - 1] = grid.z_max;
        let d_area: f64 = (r[1] - r[0]) * (z[1] - z[0]);
        let scale: &PhysicalScale = &solution.scale;

        // The flux on the grid, (n_z, n_r) flattened row-major
        let mut grid_r: Vec<f64> = Vec::with_capacity(n_r * n_z);
        let mut grid_z: Vec<f64> = Vec::with_capacity(n_r * n_z);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                grid_r.push(r[i_r]);
                grid_z.push(z[i_z]);
            }
        }
        let derivatives: PsiDerivatives = psi_derivatives(problem, solution, &grid_r, &grid_z);
        let inside: Vec<bool> = is_inside(problem, solution, &grid_r, &grid_z);
        let psi_a: f64 = scale.psi_a_minus_psi_b;
        let psi_b: f64 = 0.0;
        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_z2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut mask_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut psi_norm_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut j_phi_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let i_point: usize = i_z * n_r + i_r;
                psi_2d[(i_z, i_r)] = derivatives.psi[i_point];
                d_psi_d_r_2d[(i_z, i_r)] = derivatives.d_psi_d_r[i_point];
                d_psi_d_z_2d[(i_z, i_r)] = derivatives.d_psi_d_z[i_point];
                d2_psi_d_r2_2d[(i_z, i_r)] = derivatives.d2_psi_d_r2[i_point];
                d2_psi_d_r_d_z_2d[(i_z, i_r)] = derivatives.d2_psi_d_r_d_z[i_point];
                d2_psi_d_z2_2d[(i_z, i_r)] = derivatives.d2_psi_d_z2[i_point];
                if inside[i_point] && derivatives.psi[i_point].is_finite() {
                    let psi_norm: f64 = (derivatives.psi[i_point] - psi_a) / (psi_b - psi_a);
                    mask_2d[(i_z, i_r)] = 1.0;
                    psi_norm_2d[(i_z, i_r)] = psi_norm;
                    j_phi_2d[(i_z, i_r)] = problem.current_density(scale, r[i_r], psi_norm);
                }
            }
        }

        // The magnetic axis, the boundary's bounding point, and the contour tree
        let (mag_r, mag_z): (f64, f64) = problem.machine_coordinates(solution.mag);
        let mut contour_tree_nodes: Vec<EquilibriumContourTreeNode> = vec![EquilibriumContourTreeNode {
            critical_type: CRITICAL_TYPE_MAXIMUM,
            r: mag_r,
            z: mag_z,
            psi: psi_a,
            ..Default::default()
        }];
        let boundary_type: i32;
        let bounding_r: f64;
        let bounding_z: f64;
        if !solution.x_points.is_empty() {
            for x_point in &solution.x_points {
                let (x_point_r, x_point_z): (f64, f64) = problem.machine_coordinates(*x_point);
                let x_point_psi: f64 =
                    scale.psi_a_minus_psi_b * series_values(&solution.series.lattice, &solution.one_minus_psi_n, &[x_point.r], &[x_point.z])[0];
                contour_tree_nodes.push(EquilibriumContourTreeNode {
                    critical_type: CRITICAL_TYPE_SADDLE,
                    r: x_point_r,
                    z: x_point_z,
                    psi: x_point_psi,
                    ..Default::default()
                });
            }
            boundary_type = BOUNDARY_TYPE_DIVERTED;
            (bounding_r, bounding_z) = problem.machine_coordinates(solution.x_points[0]);
        } else {
            // A limited analytic plasma touches no wall: the bounding point is the innermost point of its LCFS (NaN if none was found)
            boundary_type = BOUNDARY_TYPE_LIMITED;
            let n_boundary: usize = solution.boundary_polygon.r.len();
            if n_boundary > 0 {
                let mut i_innermost: usize = 0;
                for i_boundary in 0..n_boundary {
                    if solution.boundary_polygon.r[i_boundary] < solution.boundary_polygon.r[i_innermost] {
                        i_innermost = i_boundary;
                    }
                }
                (bounding_r, bounding_z) = problem.machine_coordinates(Point {
                    r: solution.boundary_polygon.r[i_innermost],
                    z: solution.boundary_polygon.z[i_innermost],
                });
            } else {
                (bounding_r, bounding_z) = (f64::NAN, f64::NAN);
            }
        }

        // The IDS layout, the same as a reconstruction's: one time-slice, holding one rectangular grid, with the settings it was
        // constructed with kept in `code/parameters`
        let mut equilibrium_ids: Equilibrium = Equilibrium::default();
        equilibrium_ids.code.parameters = self.equilibrium_ids.code.parameters.clone();
        // The vacuum vessel: the grid's boundary, anticlockwise from its bottom-left corner
        let vessel_r: Array1<f64> = array![grid.r_min, grid.r_max, grid.r_max, grid.r_min, grid.r_min];
        let vessel_z: Array1<f64> = array![grid.z_min, grid.z_min, grid.z_max, grid.z_max, grid.z_min];
        equilibrium_ids.limiter.unit.push(EquilibriumLimiterUnit {
            name: "grid_boundary".to_string(),
            outline: EquilibriumLimiterUnitOutline { r: vessel_r, z: vessel_z },
        });
        equilibrium_ids.allocate_time_slices(&array![self.buxton_prokopyszyn_settings.time]);
        // A converged analytic solution is usable
        equilibrium_ids.code.output_flag = array![if solution.closure.closed { 0 } else { -1 }];
        // `r0 * b0` is the vacuum `f`, which is what the post-processor recovers the rod current from
        equilibrium_ids.vacuum_toroidal_field.r0 = problem.r0;
        equilibrium_ids.vacuum_toroidal_field.b0 = array![problem.b0];

        let time_slice: &mut EquilibriumTimeSlice = &mut equilibrium_ids.time_slice[0];
        // The normalised flux the source functions are evaluated on, exactly 1 at the end (also `boundary.psi_norm`):
        // `Array1::linspace`'s last point, `(1 / (n - 1)) * (n - 1)`, can round to 0.9999999999999999
        let mut psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, grid.n_psi_norm);
        psi_norm[grid.n_psi_norm - 1] = 1.0;
        time_slice.profiles_1d.psi_norm = psi_norm;
        let is_with_constant: bool = problem.shapes.has_constant_terms();
        let (_, p_prime_dofs): (SharedSourceFunction, Array1<f64>) = source_function_and_dofs(&scale.p_prime_coefficients, is_with_constant);
        let (_, ff_prime_dofs): (SharedSourceFunction, Array1<f64>) = source_function_and_dofs(&scale.ff_prime_coefficients, is_with_constant);
        time_slice.source_functions.p_prime.coefficients = p_prime_dofs;
        time_slice.source_functions.ff_prime.coefficients = ff_prime_dofs;

        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice.global_quantities.psi_magnetic_axis = psi_a;
        time_slice.global_quantities.ip = problem.plasma_current(solution, scale, N_PLASMA_CURRENT_NODE);
        time_slice.boundary.psi = psi_b;
        time_slice.boundary.bounding.r = bounding_r;
        time_slice.boundary.bounding.z = bounding_z;
        time_slice.boundary.r#type = boundary_type;
        time_slice.contour_tree.node = contour_tree_nodes;
        if solution.closure.closed {
            time_slice.convergence.result.name = "converged".to_string();
            time_slice.convergence.result.index = CONVERGENCE_STATUS_CONVERGED;
            time_slice.convergence.result.description = format!(
                "analytic Buxton-Prokopyszyn equilibrium ({} Taylor terms, GS residual {:.1e})",
                solution.n_taylor, solution.gs_residual
            );
        } else {
            // The post-processor skips a time-slice which did not converge
            time_slice.convergence.result.name = "unconverged".to_string();
            time_slice.convergence.result.index = CONVERGENCE_STATUS_UNCONVERGED;
            time_slice.convergence.result.description = "analytic Buxton-Prokopyszyn equilibrium: the LCFS is not a closed set of nested surfaces".to_string();
        }

        // `rectangular` is index 1 of the data dictionary's poloidal plane coordinates enumeration
        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid_type.name = "rectangular".to_string();
        profiles_2d.grid_type.index = 1;
        profiles_2d.grid_type.description = "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".to_string();
        profiles_2d.grid.dim1 = r.clone();
        profiles_2d.grid.dim2 = z.clone();
        profiles_2d.grid.d_area = d_area;
        profiles_2d.r = Array2::from_shape_fn((n_z, n_r), |(_i_z, i_r)| r[i_r]);
        profiles_2d.z = Array2::from_shape_fn((n_z, n_r), |(i_z, _i_r)| z[i_z]);
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;
        profiles_2d.d_psi_d_z = d_psi_d_z_2d;
        profiles_2d.d2_psi_d_r2 = d2_psi_d_r2_2d;
        profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;
        profiles_2d.d2_psi_d_z2 = d2_psi_d_z2_2d;
        profiles_2d.mask = mask_2d;
        profiles_2d.psi_norm = psi_norm_2d;
        profiles_2d.j_phi = j_phi_2d;
        time_slice.profiles_2d = vec![profiles_2d];
        Ok(equilibrium_ids)
    }
}

/// The radial variable: `"r_squared"`, `"log_r_squared"`, or `"auto"` (`log_r_squared` when `a_minor / r_geo >= 0.5`)
fn resolve_radial_variable(radial_variable: &str, a_minor_over_r_geo: f64) -> Result<RadialVariableKind, String> {
    match radial_variable {
        "auto" => {
            if a_minor_over_r_geo >= 0.5 {
                Ok(RadialVariableKind::LogRSquared)
            } else {
                Ok(RadialVariableKind::RSquared)
            }
        }
        "r_squared" => Ok(RadialVariableKind::RSquared),
        "log_r_squared" => Ok(RadialVariableKind::LogRSquared),
        _ => Err(format!("radial_variable must be auto, r_squared or log_r_squared, not {radial_variable:?}")),
    }
}

/// The radial extent where the radial power series can be evaluated, `(r_min, r_max)` [metre]
fn valid_r_range_of(r_geo: f64, a_minor: f64, kind: RadialVariableKind) -> (f64, f64) {
    let valid: ValidRange = RadialVariable::valid_range(r_geo, a_minor, kind);
    (valid.r_minimum, valid.r_maximum)
}

/// Where the radial series can be evaluated, with `R**2` within `R_c**2 -+ factor` times the plasma's half-width in `R**2`
fn r_squared_range(radial: &RadialVariable, factor: f64) -> ValidRange {
    let valid: ValidRange = radial.r_range_valid();
    let r_squared_low: f64 = radial.r_c_squared - factor * radial.r_squared_half_width;
    let r_squared_high: f64 = radial.r_c_squared + factor * radial.r_squared_half_width;
    ValidRange {
        r_minimum: valid.r_minimum.max(r_squared_low.max(0.0).sqrt()),
        r_maximum: valid.r_maximum.min(r_squared_high.sqrt()),
    }
}

/// Heights in the machine measured from `z_geo`, as the solver measures them [metre]
fn relative_heights(problem: &Problem, z: &[f64]) -> Vec<f64> {
    let n_point: usize = z.len();
    let mut z_relative: Vec<f64> = Vec::with_capacity(n_point);
    for i_point in 0..n_point {
        z_relative.push(z[i_point] - problem.z_geo);
    }
    z_relative
}

/// Whether points `(R, Z)` in the machine lie inside the analytic LCFS polygon
fn is_inside(problem: &Problem, solution: &Solution, r: &[f64], z: &[f64]) -> Vec<bool> {
    let z_relative: Vec<f64> = relative_heights(problem, z);
    inside_polygon(r, &z_relative, &solution.boundary_polygon.r, &solution.boundary_polygon.z)
}

/// `(1 - psi_N)` and its derivatives at points `(R, Z)`, `Z` from `z_geo`, NaN where `R` is outside the radial series' valid range; with
/// `values_only`, the values only (the derivatives are not computed, and left zero inside the range)
fn evaluate_masked(solution: &Solution, r: &[f64], z: &[f64], values_only: bool) -> FluxValues {
    let lattice: &Lattice = &solution.series.lattice;
    let valid: ValidRange = lattice.radial.r_range_valid();
    let n_point: usize = r.len();
    let mut valid_points: Vec<usize> = Vec::with_capacity(n_point);
    let mut r_valid: Vec<f64> = Vec::with_capacity(n_point);
    let mut z_valid: Vec<f64> = Vec::with_capacity(n_point);
    for i_point in 0..n_point {
        if r[i_point] >= valid.r_minimum && r[i_point] <= valid.r_maximum {
            valid_points.push(i_point);
            r_valid.push(r[i_point]);
            z_valid.push(z[i_point]);
        }
    }
    let n_valid: usize = valid_points.len();
    let flux_valid: FluxValues = if values_only {
        let mut values: FluxValues = FluxValues::zeros(n_valid);
        values.one_minus_psi_n = series_values(lattice, &solution.one_minus_psi_n, &r_valid, &z_valid);
        values
    } else {
        series_evaluate(lattice, &solution.one_minus_psi_n, &r_valid, &z_valid)
    };
    let mut flux: FluxValues = FluxValues {
        one_minus_psi_n: Array1::from_elem(n_point, f64::NAN),
        d_d_r: Array1::from_elem(n_point, f64::NAN),
        d_d_z: Array1::from_elem(n_point, f64::NAN),
        d2_d_r2: Array1::from_elem(n_point, f64::NAN),
        d2_d_r_d_z: Array1::from_elem(n_point, f64::NAN),
        d2_d_z2: Array1::from_elem(n_point, f64::NAN),
    };
    for i_valid in 0..n_valid {
        let i_point: usize = valid_points[i_valid];
        flux.one_minus_psi_n[i_point] = flux_valid.one_minus_psi_n[i_valid];
        flux.d_d_r[i_point] = flux_valid.d_d_r[i_valid];
        flux.d_d_z[i_point] = flux_valid.d_d_z[i_valid];
        flux.d2_d_r2[i_point] = flux_valid.d2_d_r2[i_valid];
        flux.d2_d_r_d_z[i_point] = flux_valid.d2_d_r_d_z[i_valid];
        flux.d2_d_z2[i_point] = flux_valid.d2_d_z2[i_valid];
    }
    flux
}

/// The flux and its derivatives at points `(R, Z)` in the machine, `psi = (psi_a - psi_b) (1 - psi_N)`, NaN where `R` is outside the
/// radial series' valid range
fn psi_derivatives(problem: &Problem, solution: &Solution, r: &[f64], z: &[f64]) -> PsiDerivatives {
    let z_relative: Vec<f64> = relative_heights(problem, z);
    let flux: FluxValues = evaluate_masked(solution, r, &z_relative, false);
    let psi_a_minus_psi_b: f64 = solution.scale.psi_a_minus_psi_b;
    PsiDerivatives {
        psi: flux.one_minus_psi_n.mapv(|value: f64| psi_a_minus_psi_b * value),
        d_psi_d_r: flux.d_d_r.mapv(|value: f64| psi_a_minus_psi_b * value),
        d_psi_d_z: flux.d_d_z.mapv(|value: f64| psi_a_minus_psi_b * value),
        d2_psi_d_r2: flux.d2_d_r2.mapv(|value: f64| psi_a_minus_psi_b * value),
        d2_psi_d_r_d_z: flux.d2_d_r_d_z.mapv(|value: f64| psi_a_minus_psi_b * value),
        d2_psi_d_z2: flux.d2_d_z2.mapv(|value: f64| psi_a_minus_psi_b * value),
    }
}
