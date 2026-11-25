//! The reference linear operator of the perturbation series, the left-hand side of every order (the paper's Eqs. order_0 and
//! order_1) with order 0's `k_max,0`,
//!
//! ```text
//! L_0 X = 4 R**2 d2X/d(R**2)**2 + d2X/dZ**2 + k_max,0**2 [1 + (P_bar / R_c**2) (R**2 - R_c**2)] X,
//! ```
//!
//! with `P_bar` the linear part, `c_bar`, of the profile `P` (`profiles`; `P_bar + Q_bar = 1`), so that
//! `k_max,0**2 P_bar / R_c**2 = (2 pi)**2 mu0 c_bar_p_prime / (psi_a - psi_b)`, and its bounded (minimum-norm) right inverse on lattice
//! series. A source `cos(k_j Z) s(R**2)` has the particular solution `cos(k_j Z) X_j(R**2)`, with `X_j` solving the radial equation
//! `4 R**2 X_j'' + (k_max,0**2 - k_j**2 + p_prime_term (R**2 - R_c**2)) X_j = s`. Wavenumbers above `k_max,0` grow towards the inboard
//! and outboard edges.

use super::lattice::{Lattice, LatticeSeries, box_offsets};
use super::radial_series::{GROWTH_MAX, GROWTH_MAX_EXTENDED, ModeProjection, log_growth, min_norm_particular_extended, solve_radial_ode};
use faer::c64;
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::prelude::*;
use std::collections::HashMap;

/// Growth from the midplane to the top, in e-folds, above which a wavenumber with an imaginary part (`cosh`) is dropped: its terms are
/// `exp(|Im k_j| z_top)` larger at the top than their size, so their rounding is amplified as much [dimensionless]
pub const GROWTH_MAX_Z: f64 = 22.0;

/// Source of a strongly hyperbolic wavenumber, relative to the source's largest wavenumber's, below which it is dropped rather than
/// solved in double-double arithmetic [dimensionless]
const EXTENDED_SOURCE_MIN: f64 = 1.0e-13;

/// The left-hand side of every order, with order 0's `k_max,0`, and what its minimum-norm inverse has met so far.
#[derive(Clone)]
pub struct ReferenceOperator {
    /// Order 0's `k_max,0` [1 / metre]
    pub k_max_0: f64,
    /// `(2 pi)**2 mu0 c_bar_p_prime / (psi_a - psi_b)` of order 0, `k_max,0**2 P_bar / R_c**2` [1 / metre ** 4]
    pub p_prime_term: f64,
    /// For the wavenumbers solved in double precision, by lattice point, the wavenumber's index in `projections`, which holds their
    /// homogeneous solutions and QR factors
    projection_index: HashMap<Vec<i64>, usize>,
    projections: Vec<ModeProjection>,
    /// For the wavenumbers solved in double-double arithmetic, by lattice point, the minimum-norm Green's matrix (column `q` the solution
    /// for the source `radial**q`) [metre ** 2]
    greens: HashMap<Vec<i64>, Array2<f64>>,
}

/// The minimum-norm particular solution of `L_0 X = source`, and what the growth cut-offs dropped
pub struct ParticularSolution {
    /// The solution, in the source's box [dimensionless]
    pub series: LatticeSeries,
    /// The largest `sum_m |s_{j,m}|` of a dropped wavenumber [1 / metre ** 2]
    pub dropped_size: f64,
}

impl ReferenceOperator {
    /// # Arguments
    /// * `k_max_0` - order 0's `k_max,0` [1 / metre]
    /// * `p_prime_linear_part` - `P_bar`, the linear part of the profile `P` [dimensionless]
    /// * `r_c_squared` - `R_c**2` [metre ** 2]
    pub fn new(k_max_0: f64, p_prime_linear_part: f64, r_c_squared: f64) -> ReferenceOperator {
        ReferenceOperator {
            k_max_0,
            p_prime_term: k_max_0 * k_max_0 * p_prime_linear_part / r_c_squared,
            projection_index: HashMap::new(),
            projections: Vec::new(),
            greens: HashMap::new(),
        }
    }

    /// `k_max,0**2 - k_j**2`, complex for complex `k_j` [1 / metre ** 2]
    pub fn k_max_squared_minus_k_squared(&self, wavenumber: c64) -> c64 {
        -(wavenumber * wavenumber) + self.k_max_0 * self.k_max_0
    }
}

/// The minimum-norm particular solution of `L_0 X = source`, lattice point by lattice point.
///
/// Wavenumbers whose radial functions grow by at most `GROWTH_MAX` e-folds across the plasma are solved in double precision; hyperbolic
/// wavenumbers with a real `k_max**2 - k_j**2` that grow by up to `GROWTH_MAX_EXTENDED` e-folds in double-double arithmetic (their
/// projection cancels terms `exp(growth)` times larger), unless their source is below `EXTENDED_SOURCE_MIN` of the largest wavenumber's;
/// stronger ones get no particular solution, and neither, with `cosh` terms, do wavenumbers whose `|Im k_j| z_top` exceeds
/// `GROWTH_MAX_Z` (they grow by that many e-folds from the midplane to the top) or whose `k_max**2 - k_j**2` is complex and which grow by
/// more than `GROWTH_MAX`. The largest dropped source is returned, so that callers can check it is negligible.
///
/// What depends on the wavenumber only is computed once and kept in the operator: in double precision its homogeneous solutions and
/// their QR factors, so that a solution is one recursion for the actual source and one projection; in double-double arithmetic, whose
/// projections are accurate to far below double precision, the minimum-norm Green's matrix, the solution being linear in the source.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `operator` - the reference operator, which holds the cached projections and Green's matrices
/// * `source` - the right-hand side [1 / metre ** 2]
pub fn particular_solution(lattice: &Lattice, operator: &mut ReferenceOperator, source: &LatticeSeries) -> ParticularSolution {
    let n_radial: usize = lattice.n_radial;
    let wavenumbers: Vec<c64> = lattice.box_wavenumbers(&source.half_widths);
    let n_row: usize = wavenumbers.len();
    let p_prime_term: f64 = operator.p_prime_term;
    let mut k_max_squared_minus_k_squared_all: Vec<c64> = Vec::with_capacity(n_row);
    for i_row in 0..n_row {
        k_max_squared_minus_k_squared_all.push(operator.k_max_squared_minus_k_squared(wavenumbers[i_row]));
    }
    let growth: Vec<f64> = (0..n_row)
        .into_par_iter()
        .map(|i_row: usize| log_growth(&lattice.radial, k_max_squared_minus_k_squared_all[i_row], p_prime_term))
        .collect();
    let mut magnitudes: Vec<f64> = vec![0.0; n_row];
    let mut is_nonzero: Vec<bool> = vec![false; n_row];
    let mut magnitude_max: f64 = 0.0;
    for i_row in 0..n_row {
        for i_radial in 0..n_radial {
            let coefficient: c64 = source.coefficients[[i_row, i_radial]];
            magnitudes[i_row] += coefficient.norm();
            is_nonzero[i_row] |= coefficient.re != 0.0 || coefficient.im != 0.0;
        }
        // NaN propagates: a source that cannot be evaluated is not a small one
        if magnitudes[i_row].is_nan() || magnitudes[i_row] > magnitude_max {
            magnitude_max = magnitudes[i_row];
        }
    }
    let mut is_active: Vec<bool> = vec![false; n_row];
    let mut is_extended: Vec<bool> = vec![false; n_row];
    let mut dropped_size: f64 = 0.0;
    for i_row in 0..n_row {
        let is_within_z: bool = !lattice.is_complex || wavenumbers[i_row].im.abs() * lattice.z_top <= GROWTH_MAX_Z;
        is_active[i_row] = is_nonzero[i_row] && is_within_z && growth[i_row] <= GROWTH_MAX;
        // Double-double solutions only for wavenumbers whose source is not negligible beside the largest wavenumber's: they are costly to
        // set up
        let is_significant: bool = magnitudes[i_row] > EXTENDED_SOURCE_MIN * magnitude_max;
        is_extended[i_row] = is_significant
            && is_within_z
            && growth[i_row] > GROWTH_MAX
            && growth[i_row] <= GROWTH_MAX_EXTENDED
            && k_max_squared_minus_k_squared_all[i_row].im == 0.0;
        let is_dropped: bool = is_nonzero[i_row] && !is_active[i_row] && !is_extended[i_row];
        if is_dropped && (magnitudes[i_row].is_nan() || magnitudes[i_row] > dropped_size) {
            dropped_size = magnitudes[i_row];
        }
    }
    let mut result: Array2<c64> = Array2::from_elem((n_row, n_radial), c64::new(0.0, 0.0));

    // The lattice points of the rows, and the projections and Green's matrices still missing
    let keys: Vec<Vec<i64>> = box_offsets(&source.half_widths);
    let mut missing_double: Vec<usize> = Vec::new();
    let mut missing_extended: Vec<usize> = Vec::new();
    for i_row in 0..n_row {
        if is_active[i_row] && !operator.projection_index.contains_key(&keys[i_row]) {
            missing_double.push(i_row);
        }
        if is_extended[i_row] && !operator.greens.contains_key(&keys[i_row]) {
            missing_extended.push(i_row);
        }
    }
    if !missing_double.is_empty() {
        let radial: &super::radial_series::RadialVariable = &lattice.radial;
        let plasma_heights: Option<&Array1<f64>> = lattice.plasma_heights.as_ref();
        let new_projections: Vec<ModeProjection> = missing_double
            .par_iter()
            .map(|i_row: &usize| {
                // Wavenumbers with an imaginary part b are amplified by up to cosh(b half_height) over the plasma's column at R**2
                match plasma_heights {
                    Some(heights) => {
                        let imaginary_part: f64 = wavenumbers[*i_row].im.abs();
                        let weight_factors: Array1<f64> = heights.mapv(|height: f64| (imaginary_part * height).cosh());
                        ModeProjection::new(radial, k_max_squared_minus_k_squared_all[*i_row], p_prime_term, Some(weight_factors.view()))
                    }
                    None => ModeProjection::new(radial, k_max_squared_minus_k_squared_all[*i_row], p_prime_term, None),
                }
            })
            .collect();
        let n_missing: usize = missing_double.len();
        for i_missing in 0..n_missing {
            let index: usize = operator.projections.len();
            operator.projections.push(new_projections[i_missing].clone());
            operator.projection_index.insert(keys[missing_double[i_missing]].clone(), index);
        }
    }
    let n_missing_extended: usize = missing_extended.len();
    if n_missing_extended > 0 {
        // Rows are the minimum-norm solutions for the sources radial**q: the Green's matrix maps source to solution coefficients. The
        // rows of every missing wavenumber in one call, which sets up the wavenumbers' projections in parallel
        let mut unit_sources: Array2<f64> = Array2::zeros((n_missing_extended * n_radial, n_radial));
        let mut stacked_k_max_squared_minus_k_squared: Vec<f64> = Vec::with_capacity(n_missing_extended * n_radial);
        for i_missing in 0..n_missing_extended {
            for i_radial in 0..n_radial {
                unit_sources[[i_missing * n_radial + i_radial, i_radial]] = 1.0;
                stacked_k_max_squared_minus_k_squared.push(k_max_squared_minus_k_squared_all[missing_extended[i_missing]].re);
            }
        }
        let stacked: Array2<f64> = min_norm_particular_extended(&lattice.radial, unit_sources.view(), &stacked_k_max_squared_minus_k_squared, p_prime_term);
        for i_missing in 0..n_missing_extended {
            let rows: ArrayView2<f64> = stacked.slice(s![i_missing * n_radial..(i_missing + 1) * n_radial, ..]);
            operator.greens.insert(keys[missing_extended[i_missing]].clone(), rows.t().to_owned());
        }
    }

    // Double precision: the forced solutions of the actual sources, then their projections
    let mut active_rows: Vec<usize> = Vec::new();
    for i_row in 0..n_row {
        if is_active[i_row] {
            active_rows.push(i_row);
        }
    }
    let operator_view: &ReferenceOperator = operator;
    let solved_rows: Vec<Array1<c64>> = active_rows
        .par_iter()
        .map(|i_row: &usize| {
            let zero: c64 = c64::new(0.0, 0.0);
            let forced: Array1<c64> = solve_radial_ode(
                &lattice.radial,
                source.coefficients.row(*i_row),
                k_max_squared_minus_k_squared_all[*i_row],
                p_prime_term,
                zero,
                zero,
            );
            let mode: usize = operator_view.projection_index[&keys[*i_row]];
            operator_view.projections[mode].project(&lattice.radial, forced.view())
        })
        .collect();
    let n_active: usize = active_rows.len();
    for i_active in 0..n_active {
        result.row_mut(active_rows[i_active]).assign(&solved_rows[i_active]);
    }
    // Double-double: the Green's matrices applied to the actual sources
    for i_row in 0..n_row {
        if !is_extended[i_row] {
            continue;
        }
        let greens_matrix: &Array2<f64> = &operator_view.greens[&keys[i_row]];
        for i_radial in 0..n_radial {
            let mut total_re: f64 = 0.0;
            let mut total_im: f64 = 0.0;
            for j_radial in 0..n_radial {
                total_re += greens_matrix[[i_radial, j_radial]] * source.coefficients[[i_row, j_radial]].re;
                total_im += greens_matrix[[i_radial, j_radial]] * source.coefficients[[i_row, j_radial]].im;
            }
            result[[i_row, i_radial]] = c64::new(total_re, total_im);
        }
    }
    ParticularSolution {
        series: LatticeSeries {
            half_widths: source.half_widths.clone(),
            coefficients: result,
            support: source.support,
        },
        dropped_size,
    }
}
