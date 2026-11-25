//! The homogeneous shape terms, the paper's `cos(k_j Z) C_j` and `cos(k_j Z) S_j`, and their values under the boundary conditions.
//!
//! * The paper's `n_wave >= 4` equally spaced wavenumbers: `k_j = (j - 1) k_spacing k_max`, `j = 1..n_wave`; only `C_j` at `k_j = k_max`
//!   (`j = 4` for the paper's `k_spacing = 1/3`), as in the paper and in Guazzotto and Freidberg's solution, `C_j` and `S_j` at the
//!   wavenumbers below it, and `C_j` and `S_j` at those above it, whose radial functions grow towards the inboard and outboard edges:
//!   `2 n_wave - 1` terms on the one-dimensional lattice of the spacing `k_spacing k_max`. Neighbouring wavenumbers stay
//!   `k_spacing k_max` apart, which keeps them far from linearly dependent.
//! * Optionally, the paper's terms with `k_j**2 < 0`, for the top and bottom of the plasma, where a spherical tokamak's X-points are:
//!   `cosh(|k_j| Z) C_j` and `cosh(|k_j| Z) S_j` at `|k_j| = n k_spacing_cosh k_max`, `n = 1..n_wave_cosh`, whose radial functions
//!   oscillate. The lattice then gains an imaginary axis, the spacing `i k_spacing_cosh k_max`: it is two-dimensional.
//! * Up-down asymmetric shapes (single null, different upper and lower shapes) add the odd terms `sin(k_j Z) C_j` and `sin(k_j Z) S_j`
//!   at the same wavenumbers, but none at `k_1 = 0` (where the sine vanishes), again without `S_j` at `k_max`, for `4 (n_wave - 1)` terms;
//!   and `sinh(|k_j| Z)` with the `cosh` terms.
//!
//! `C_j` has `X = 1`, `dX/d(radial) = 0` and `S_j` has `X = 0`, `dX/d(radial) = 1` at `R = R_c` (`radial_series::c_and_s`).

use super::configuration::BoundaryConditions;
use super::lattice::{FluxValues, Lattice, LatticeSeries, Parity, series_add_assign, series_evaluate, series_evaluate_many, series_from_term, series_scale};
use super::radial_series::{RadialPair, RadialVariable, c_and_s};
use super::reference_operator::ReferenceOperator;
use faer::c64;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::sync::Arc;

/// The radial function of a shape term
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadialKind {
    /// The paper's `C_j`: unit value at `R_c`
    C,
    /// The paper's `S_j`: unit slope at `R_c`
    S,
}

/// One homogeneous shape term: its lattice point, radial function and parity
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeTerm {
    /// The lattice point [dimensionless]
    pub key: Vec<i64>,
    pub radial_kind: RadialKind,
    pub parity: Parity,
}

/// The shape terms, and the wavenumber spacings of the lattice in units of `k_max`
pub struct BasisTerms {
    /// One real spacing, `k_spacing`, and an imaginary one, `i k_spacing_cosh`, with the `cosh` terms [dimensionless]
    pub spacing_factors: Vec<c64>,
    pub terms: Vec<ShapeTerm>,
}

/// The shape terms at a trial eigenvalue, with the lattice and operator they belong to
pub struct ShapeBasis {
    pub lattice: Lattice,
    pub operator: ReferenceOperator,
    pub basis: Vec<LatticeSeries>,
}

fn term(key: Vec<i64>, radial_kind: RadialKind, parity: Parity) -> ShapeTerm {
    ShapeTerm { key, radial_kind, parity }
}

/// The shape terms and the wavenumber spacings.
///
/// `C_j` and `S_j` at the wavenumbers `k_j = n k_spacing k_max`, `n = j - 1 = 0..n_wave - 1`, but only `C_j` at `k_j = k_max` (with the
/// paper's spacing `k_max / 3`, `n = 3`: `2 n_wave - 1` terms; with a spacing that does not divide `k_max`, `2 n_wave`). The order: the
/// wavenumber at `k_max`, those below it downwards, then those above it upwards.
///
/// # Arguments
/// * `n_wave` - number of equally spaced wavenumbers, `n_wave >= 4` [dimensionless]
/// * `k_spacing` - their spacing in units of `k_max`, `0 < k_spacing < 1` [dimensionless]
/// * `n_wave_cosh` - number of `cosh(|k_j| Z)` terms' wavenumbers, `|k_j| = n k_spacing_cosh k_max`, `n = 1..n_wave_cosh` [dimensionless]
/// * `k_spacing_cosh` - their spacing in units of `k_max` [dimensionless]
/// * `is_up_down_symmetric` - false adds the odd (`sin`, `sinh`) terms
///
/// # Errors
/// An invalid number of wavenumbers or spacing
pub fn basis_terms(n_wave: usize, k_spacing: f64, n_wave_cosh: usize, k_spacing_cosh: f64, is_up_down_symmetric: bool) -> Result<BasisTerms, String> {
    let mut terms: Vec<ShapeTerm> = Vec::new();
    if n_wave < 4 {
        return Err(format!("basis_terms: needs n_wave >= 4 wavenumbers, not {n_wave}"));
    }
    if n_wave_cosh > 0 && k_spacing_cosh <= 0.0 {
        return Err(format!("basis_terms: needs k_spacing_cosh > 0 with cosh terms, not {k_spacing_cosh}"));
    }
    if !(k_spacing > 0.0 && k_spacing < 1.0) {
        return Err(format!("basis_terms: k_spacing, in units of k_max, must lie in (0, 1), not {k_spacing}"));
    }
    // The wavenumber at k_max, if the spacing divides k_max, and the last one below it
    let n_per_k_max: f64 = 1.0 / k_spacing;
    let mut i_at_k_max: i64 = -1;
    if (n_per_k_max - n_per_k_max.round()).abs() < 1.0e-9 {
        i_at_k_max = n_per_k_max.round() as i64;
    }
    let i_below: i64 = if i_at_k_max >= 0 { i_at_k_max - 1 } else { n_per_k_max.floor() as i64 };
    // The last wavenumber, `n = n_wave - 1` [dimensionless]
    let i_harmonic_last: i64 = n_wave as i64 - 1;
    let mut real_harmonics: Vec<i64> = Vec::new();
    let mut real_kinds: Vec<RadialKind> = Vec::new();
    if 0 <= i_at_k_max && i_at_k_max <= i_harmonic_last {
        real_harmonics.push(i_at_k_max);
        real_kinds.push(RadialKind::C);
    }
    let i_below_start: i64 = i_below.min(i_harmonic_last);
    for i_harmonic in (0..=i_below_start).rev() {
        real_harmonics.push(i_harmonic);
        real_kinds.push(RadialKind::C);
        real_harmonics.push(i_harmonic);
        real_kinds.push(RadialKind::S);
    }
    let i_first_above: i64 = if i_at_k_max >= 0 { i_at_k_max + 1 } else { i_below + 1 };
    for i_harmonic in i_first_above..=i_harmonic_last {
        real_harmonics.push(i_harmonic);
        real_kinds.push(RadialKind::C);
        real_harmonics.push(i_harmonic);
        real_kinds.push(RadialKind::S);
    }
    let n_real_even: usize = real_harmonics.len();
    let mut real_parities: Vec<Parity> = vec![Parity::Cos; n_real_even];
    if !is_up_down_symmetric {
        // The odd partners of every term with a non-zero wavenumber
        for i_real in 0..n_real_even {
            if real_harmonics[i_real] > 0 {
                real_harmonics.push(real_harmonics[i_real]);
                real_kinds.push(real_kinds[i_real]);
                real_parities.push(Parity::Sin);
            }
        }
    }
    let n_real: usize = real_harmonics.len();
    if n_wave_cosh == 0 {
        for i_real in 0..n_real {
            terms.push(term(vec![real_harmonics[i_real]], real_kinds[i_real], real_parities[i_real]));
        }
        return Ok(BasisTerms {
            spacing_factors: vec![c64::new(k_spacing, 0.0)],
            terms,
        });
    }
    for i_real in 0..n_real {
        terms.push(term(vec![real_harmonics[i_real], 0], real_kinds[i_real], real_parities[i_real]));
    }
    for i_wave_cosh in 1..=(n_wave_cosh as i64) {
        terms.push(term(vec![0, i_wave_cosh], RadialKind::C, Parity::Cos));
        terms.push(term(vec![0, i_wave_cosh], RadialKind::S, Parity::Cos));
    }
    if !is_up_down_symmetric {
        for i_wave_cosh in 1..=(n_wave_cosh as i64) {
            terms.push(term(vec![0, i_wave_cosh], RadialKind::C, Parity::Sin));
            terms.push(term(vec![0, i_wave_cosh], RadialKind::S, Parity::Sin));
        }
    }
    Ok(BasisTerms {
        spacing_factors: vec![c64::new(k_spacing, 0.0), c64::new(0.0, k_spacing_cosh)],
        terms,
    })
}

/// `cos(k_j Z) X_j` or `sin(k_j Z) X_j` for every term (`cosh(|k_j| Z) X_j`, `sinh(|k_j| Z) X_j` for an imaginary `k_j`), with
/// `X_j = C_j` or `S_j`.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `operator` - the operator whose homogeneous solutions are wanted
/// * `terms` - the terms
pub fn homogeneous_terms(lattice: &Lattice, operator: &ReferenceOperator, terms: &[ShapeTerm]) -> Vec<LatticeSeries> {
    let radial: &RadialVariable = &lattice.radial;
    let p_prime_term: f64 = operator.p_prime_term;
    // A term on one lattice axis, real or imaginary, has a real k_j**2 and real radial functions
    let pairs: Vec<RadialPair> = terms
        .par_iter()
        .map(|shape_term: &ShapeTerm| {
            let wavenumber: c64 = lattice.wavenumber(&shape_term.key);
            c_and_s(radial, operator.k_max_squared_minus_k_squared(wavenumber), p_prime_term)
        })
        .collect();
    let n_term: usize = terms.len();
    let mut series: Vec<LatticeSeries> = Vec::with_capacity(n_term);
    for i_term in 0..n_term {
        let radial_coefficients: &Array1<c64> = match terms[i_term].radial_kind {
            RadialKind::C => &pairs[i_term].c,
            RadialKind::S => &pairs[i_term].s,
        };
        series.push(series_from_term(lattice, &terms[i_term].key, radial_coefficients.view(), terms[i_term].parity).expect("homogeneous_terms: a valid term"));
    }
    series
}

/// The shape terms at a trial `k_max`, with the lattice and operator they belong to.
///
/// # Arguments
/// * `radial` - the radial variable
/// * `p_prime_linear_part` - `P_bar`, the linear part of the profile `P` [dimensionless]
/// * `k_max` - trial `k_max` [1 / metre]
/// * `n_wave` - number of equally spaced wavenumbers, `n_wave >= 4` [dimensionless]
/// * `k_spacing` - their spacing in units of `k_max` [dimensionless]
/// * `n_wave_cosh` - number of `cosh(|k_j| Z)` terms' wavenumbers [dimensionless]
/// * `k_spacing_cosh` - their spacing in units of `k_max` [dimensionless]
/// * `z_top` - largest `|Z|` of the plasma, for the growth cut-off of the `cosh` terms [metre]
/// * `is_up_down_symmetric` - false adds the odd terms, on a lattice with complex coefficients
#[allow(clippy::too_many_arguments)]
pub fn shape_basis(
    radial: &Arc<RadialVariable>,
    p_prime_linear_part: f64,
    k_max: f64,
    n_wave: usize,
    k_spacing: f64,
    n_wave_cosh: usize,
    k_spacing_cosh: f64,
    z_top: f64,
    is_up_down_symmetric: bool,
) -> Result<ShapeBasis, String> {
    let basis_terms_found: BasisTerms = basis_terms(n_wave, k_spacing, n_wave_cosh, k_spacing_cosh, is_up_down_symmetric)?;
    let n_base: usize = basis_terms_found.spacing_factors.len();
    let mut wavenumber_spacings: Vec<c64> = Vec::with_capacity(n_base);
    for i_base in 0..n_base {
        wavenumber_spacings.push(basis_terms_found.spacing_factors[i_base] * k_max);
    }
    let lattice: Lattice = Lattice::new(wavenumber_spacings, Arc::clone(radial), z_top, is_up_down_symmetric);
    let operator: ReferenceOperator = ReferenceOperator::new(k_max, p_prime_linear_part, radial.r_c_squared);
    let basis: Vec<LatticeSeries> = homogeneous_terms(&lattice, &operator, &basis_terms_found.terms);
    Ok(ShapeBasis { lattice, operator, basis })
}

/// The paper's `M(k_max)`: `M[i, n] = condition_i(T_n)`, shape `(n_condition, n_term)` [1 / metre ** a, a the condition's order]
pub fn condition_matrix(lattice: &Lattice, basis: &[LatticeSeries], conditions: &BoundaryConditions) -> Array2<f64> {
    let n_term: usize = basis.len();
    let mut basis_references: Vec<&LatticeSeries> = Vec::with_capacity(n_term);
    for i_term in 0..n_term {
        basis_references.push(&basis[i_term]);
    }
    let flux_list: Vec<FluxValues> = series_evaluate_many(lattice, &basis_references, &conditions.r, &conditions.z);
    let mut matrix: Array2<f64> = Array2::from_elem((conditions.n_condition(), n_term), f64::NAN);
    for i_term in 0..n_term {
        matrix.column_mut(i_term).assign(&conditions.residuals(&flux_list[i_term]));
    }
    matrix
}

/// `condition_i(series)` for every condition [1 / metre ** a, a the condition's order]
pub fn condition_residuals(lattice: &Lattice, series: &LatticeSeries, conditions: &BoundaryConditions) -> Array1<f64> {
    let flux: FluxValues = series_evaluate(lattice, series, &conditions.r, &conditions.z);
    conditions.residuals(&flux)
}

/// `sum_n coefficients[n] basis[n]`, the paper's amplitudes `u` times the shape terms [dimensionless]
pub fn flux_from_coefficients(basis: &[LatticeSeries], coefficients: &Array1<f64>) -> LatticeSeries {
    let n_term: usize = basis.len();
    let mut total: LatticeSeries = series_scale(&basis[0], coefficients[0]);
    for i_term in 1..n_term {
        series_add_assign(&mut total, &basis[i_term], coefficients[i_term]);
    }
    total
}
