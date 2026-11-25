//! Order 0 of the series: Guazzotto and Freidberg's eigenvalue problem, the paper's Eq. order_0 with the boundary conditions,
//! `M(k_max) u = 0`, for the linear part of the profiles.
//!
//! The shape terms with more conditions than terms make it a least-squares eigenproblem (the method of particular solutions with
//! interior normalisation, Betcke and Trefethen 2005): `k_max` minimises the tension, `min_z |Q_B z|`, of the weighted boundary rows of
//! `M(k_max)`, in coordinates `z` that are orthonormal on the boundary and interior points together.

use super::configuration::{BoundaryConditions, Points};
use super::lattice::{LatticeSeries, series_scale, series_values, series_values_many, value_at_centre};
use super::numerics::{SingularValueDecomposition, minimize_scalar_bounded, rank_above, singular_values, svd_full, svd_thin};
use super::radial_series::RadialVariable;
use super::shape_basis::{ShapeBasis, condition_matrix, flux_from_coefficients, shape_basis};
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use std::sync::Arc;

/// Relative singular-value cut-off of the least-squares frame: combinations of the basis smaller than this, relative to the largest,
/// at every boundary and interior point are dropped, which keeps the eigenvector's coefficients small [dimensionless]
const FRAME_CUTOFF: f64 = 1.0e-13;

/// The most function evaluations of the bounded minimiser [dimensionless]
const N_MINIMISER_EVALUATION_MAX: usize = 500;

/// The least-squares eigenfunction, normalised to 1 at `(R_c, 0)`, must stay below this at the interior points (and above zero): the
/// fundamental mode peaks at its magnetic axis, which the Shafranov shift moves away from `R_c` (on ST40's TEST36 boundary its value on
/// the axis passes 1.5 times that at `R_c` at `beta_p(3) = 7.2`), while a mode whose `(R_c, 0)` lies near a node peaks far higher
/// [dimensionless]
const EIGENFUNCTION_PROBE_MAX: f64 = 4.0;

/// The order-0 eigen-solution
pub struct EigenSolution {
    /// Order 0's `k_max,0` [1 / metre]
    pub k_max_0: f64,
    /// The paper's amplitudes `u` of the shape terms [dimensionless]
    pub coefficients: Array1<f64>,
}

/// The shape options the eigenproblem's basis is built with
#[derive(Clone, Copy, Debug)]
pub struct BasisOptions {
    /// Number of equally spaced wavenumbers, `n_wave >= 4` [dimensionless]
    pub n_wave: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    pub k_spacing: f64,
    /// Number of `cosh(|k_j| Z)` terms' wavenumbers [dimensionless]
    pub n_wave_cosh: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    pub k_spacing_cosh: f64,
    pub is_up_down_symmetric: bool,
}

/// The orthonormal coordinates of the least-squares eigenproblem
struct LeastSquaresFrame {
    /// `Q_B = U[:n_least_squares]` [dimensionless]
    boundary_frame: Array2<f64>,
    /// The map `Z V Sigma^-1` from `z` to the coefficients `u` [dimensionless]
    to_coefficients: Array2<f64>,
}

/// Orthonormal coordinates for the least-squares eigenproblem: with `[W A_B; A_I] Z = U Sigma V^T` (`A_B` the boundary conditions
/// fitted by least squares, `W` their weights, `A_I` the basis at the interior points, `Z` a basis of the null space of the exact
/// conditions `A_E`), `u = Z V Sigma^-1 z` satisfies `A_E u = 0` and makes `[W A_B; A_I] u = U z` orthonormal in `z`.
fn least_squares_frame(shape: &ShapeBasis, conditions: &BoundaryConditions, row_weights: &Array1<f64>, interior: &Points) -> LeastSquaresFrame {
    let matrix_all: Array2<f64> = condition_matrix(&shape.lattice, &shape.basis, conditions);
    let n_term: usize = shape.basis.len();
    let n_condition: usize = conditions.n_condition();
    let mut exact_rows: Vec<usize> = Vec::new();
    let mut least_squares_rows: Vec<usize> = Vec::new();
    for i_condition in 0..n_condition {
        if conditions.is_exact[i_condition] {
            exact_rows.push(i_condition);
        } else {
            least_squares_rows.push(i_condition);
        }
    }
    let mut null_basis: Array2<f64> = Array2::eye(n_term);
    let n_exact: usize = exact_rows.len();
    if n_exact > 0 {
        let mut matrix_exact: Array2<f64> = Array2::from_elem((n_exact, n_term), f64::NAN);
        for i_exact in 0..n_exact {
            matrix_exact.row_mut(i_exact).assign(&matrix_all.row(exact_rows[i_exact]));
        }
        let svd_exact: SingularValueDecomposition = svd_full(matrix_exact.view());
        let rank: usize = rank_above(&svd_exact.s, FRAME_CUTOFF);
        null_basis = svd_exact.v.slice(s![.., rank..]).to_owned();
    }
    let n_least_squares: usize = least_squares_rows.len();
    let mut weighted: Array2<f64> = Array2::from_elem((n_least_squares, n_term), f64::NAN);
    for i_least_squares in 0..n_least_squares {
        let i_condition: usize = least_squares_rows[i_least_squares];
        for i_term in 0..n_term {
            weighted[[i_least_squares, i_term]] = matrix_all[[i_condition, i_term]] * row_weights[i_condition];
        }
    }
    let matrix: Array2<f64> = weighted.dot(&null_basis);
    let n_interior: usize = interior.r.len();
    let mut basis_references: Vec<&LatticeSeries> = Vec::with_capacity(n_term);
    for i_term in 0..n_term {
        basis_references.push(&shape.basis[i_term]);
    }
    let interior_values: Vec<Array1<f64>> = series_values_many(
        &shape.lattice,
        &basis_references,
        interior.r.as_slice().expect("contiguous"),
        interior.z.as_slice().expect("contiguous"),
    );
    let mut interior_matrix: Array2<f64> = Array2::from_elem((n_interior, n_term), f64::NAN);
    for i_term in 0..n_term {
        interior_matrix.column_mut(i_term).assign(&interior_values[i_term]);
    }
    let interior_reduced: Array2<f64> = interior_matrix.dot(&null_basis);
    let n_null: usize = null_basis.ncols();
    let mut stacked: Array2<f64> = Array2::from_elem((n_least_squares + n_interior, n_null), f64::NAN);
    stacked.slice_mut(s![..n_least_squares, ..]).assign(&matrix);
    stacked.slice_mut(s![n_least_squares.., ..]).assign(&interior_reduced);
    let svd: SingularValueDecomposition = svd_thin(stacked.view());
    let n_kept: usize = rank_above(&svd.s, FRAME_CUTOFF);
    let boundary_frame: Array2<f64> = svd.u.slice(s![..n_least_squares, ..n_kept]).to_owned();
    let mut scaled_v: Array2<f64> = svd.v.slice(s![.., ..n_kept]).to_owned();
    for i_kept in 0..n_kept {
        let singular: f64 = svd.s[i_kept];
        scaled_v.column_mut(i_kept).mapv_inplace(|value: f64| value / singular);
    }
    LeastSquaresFrame {
        boundary_frame,
        to_coefficients: null_basis.dot(&scaled_v),
    }
}

/// The `k_max` of order one for the plasma's size, `R_c / (R_geo a_minor)`, the unit of the search for `k_max,0` [1 / metre]
fn k_max_unit(radial: &RadialVariable) -> f64 {
    2.0 * radial.r_c_squared.sqrt() / radial.r_squared_half_width
}

/// Row weights of the least-squares fits: every condition row of `M(k_max)` normalised to unit length at a typical `k_max`, the middle
/// of the search (`2 R_c / (R_geo a_minor)`), times the condition's own relative weight.
///
/// # Arguments
/// * `radial` - the radial variable
/// * `p_prime_linear_part` - `P_bar`, the linear part of the profile `P` [dimensionless]
/// * `conditions` - boundary conditions
/// * `options` - the shape options
///
/// # Returns
/// * one weight per condition [dimensionless]
pub fn condition_row_weights(
    radial: &Arc<RadialVariable>,
    p_prime_linear_part: f64,
    conditions: &BoundaryConditions,
    options: &BasisOptions,
) -> Result<Array1<f64>, String> {
    let shape: ShapeBasis = shape_basis(
        radial,
        p_prime_linear_part,
        2.0 * k_max_unit(radial),
        options.n_wave,
        options.k_spacing,
        options.n_wave_cosh,
        options.k_spacing_cosh,
        0.0,
        options.is_up_down_symmetric,
    )?;
    let matrix: Array2<f64> = condition_matrix(&shape.lattice, &shape.basis, conditions);
    let n_condition: usize = conditions.n_condition();
    let mut row_weights: Array1<f64> = Array1::from_elem(n_condition, f64::NAN);
    for i_condition in 0..n_condition {
        let row_norm: f64 = matrix.row(i_condition).dot(&matrix.row(i_condition)).sqrt();
        row_weights[i_condition] = conditions.weight[i_condition] / row_norm;
    }
    Ok(row_weights)
}

/// The least-squares `k_max,0` and amplitudes: the first local minimum of the tension below 0.5 whose eigenfunction is positive and below
/// `EIGENFUNCTION_PROBE_MAX` at the interior points.
///
/// # Arguments
/// * `radial` - the radial variable
/// * `p_prime_linear_part` - `P_bar`, the linear part of the profile `P` [dimensionless]
/// * `conditions` - boundary conditions
/// * `row_weights` - one weight per condition [dimensionless]
/// * `interior` - interior points [metre]
/// * `options` - the shape options
///
/// # Returns
/// * `k_max,0` and the amplitudes, with `(1 - psi_N)_0 = 1` at `(R_c, 0)`
///
/// # Errors
/// No physical eigenvalue for `R_c / (R_geo a_minor) <= k_max <= 3 R_c / (R_geo a_minor)` (the message lists the local minima of the
/// tension that were rejected)
pub fn solve_least_squares_eigenvalue(
    radial: &Arc<RadialVariable>,
    p_prime_linear_part: f64,
    conditions: &BoundaryConditions,
    row_weights: &Array1<f64>,
    interior: &Points,
    options: &BasisOptions,
) -> Result<EigenSolution, String> {
    let k_max_unit: f64 = k_max_unit(radial);
    let k_max_low: f64 = k_max_unit;
    let k_max_high: f64 = 3.0 * k_max_unit;
    let n_scan: usize = 201;
    let basis_at = |k_max: f64| -> ShapeBasis {
        shape_basis(
            radial,
            p_prime_linear_part,
            k_max,
            options.n_wave,
            options.k_spacing,
            options.n_wave_cosh,
            options.k_spacing_cosh,
            0.0,
            options.is_up_down_symmetric,
        )
        .expect("solve_least_squares_eigenvalue: the shape terms")
    };
    let tension = |k_max: f64| -> f64 {
        let shape: ShapeBasis = basis_at(k_max);
        let frame: LeastSquaresFrame = least_squares_frame(&shape, conditions, row_weights, interior);
        let singular: Array1<f64> = singular_values(frame.boundary_frame.view());
        singular[singular.len() - 1]
    };
    let k_maxes: Array1<f64> = Array1::linspace(k_max_low, k_max_high, n_scan);
    let tensions: Vec<f64> = (0..n_scan).into_par_iter().map(|i_scan: usize| tension(k_maxes[i_scan])).collect();
    // The local minima rejected, for the error
    let mut rejected: Vec<String> = Vec::new();
    for i_scan in 1..n_scan - 1 {
        if !(tensions[i_scan] < tensions[i_scan - 1] && tensions[i_scan] <= tensions[i_scan + 1] && tensions[i_scan] < 0.5) {
            continue;
        }
        let k_max_root: f64 = minimize_scalar_bounded(
            tension,
            k_maxes[i_scan - 1],
            k_maxes[i_scan + 1],
            1.0e-13 * k_max_unit,
            N_MINIMISER_EVALUATION_MAX,
        );
        let shape: ShapeBasis = basis_at(k_max_root);
        let frame: LeastSquaresFrame = least_squares_frame(&shape, conditions, row_weights, interior);
        // The last right singular vector of the full SVD: a null vector also with fewer rows than columns
        let svd: SingularValueDecomposition = svd_full(frame.boundary_frame.view());
        let n_kept: usize = svd.v.ncols();
        let coefficients_raw: Array1<f64> = frame.to_coefficients.dot(&svd.v.column(n_kept - 1));
        let series: LatticeSeries = flux_from_coefficients(&shape.basis, &coefficients_raw);
        let centre_value: f64 = value_at_centre(&shape.lattice, &series);
        let coefficients: Array1<f64> = coefficients_raw.mapv(|value: f64| value / centre_value);
        let probe: Array1<f64> = series_values(
            &shape.lattice,
            &series_scale(&series, 1.0 / centre_value),
            interior.r.as_slice().expect("contiguous"),
            interior.z.as_slice().expect("contiguous"),
        );
        // NaN propagates, and rejects the root
        let mut probe_min: f64 = f64::INFINITY;
        let mut probe_max: f64 = f64::NEG_INFINITY;
        for value in probe.iter() {
            if value.is_nan() || *value < probe_min {
                probe_min = *value;
            }
            if value.is_nan() || *value > probe_max {
                probe_max = *value;
            }
        }
        rejected.push(format!("{k_max_root:.4} / m, [{probe_min:.3}, {probe_max:.3}]"));
        if probe_min > 0.0 && probe_max < EIGENFUNCTION_PROBE_MAX {
            return Ok(EigenSolution {
                k_max_0: k_max_root,
                coefficients,
            });
        }
    }
    Err(format!(
        "solve_least_squares_eigenvalue: no physical eigenvalue found in the scan range (the local minima of the tension below 0.5, as \
         k_max, [smallest, largest] eigenfunction at the interior points: {})",
        if rejected.is_empty() { "none".to_string() } else { rejected.join("; ") }
    ))
}
