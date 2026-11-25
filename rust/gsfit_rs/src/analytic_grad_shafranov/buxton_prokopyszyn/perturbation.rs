//! The paper's perturbation series about Guazzotto and Freidberg's solution, order by order.
//!
//! The profiles `P` and `Q` of `profiles` are split into a linear part, `P_bar (1 - psi_N)` and `Q_bar (1 - psi_N)` (the paper's
//! `c_bar`, so that `P_bar + Q_bar = 1`), and a remainder multiplied by `lambda` (the paper's Eq. lambda_equation, 1 at the end):
//!
//! ```text
//! L (1 - psi_N) = -lambda k_max**2 sum_n [remainder_constant[n] + remainder_slope[n] (R**2 - R_c**2)] (1 - psi_N)**n,
//! L = 4 R**2 d2/d(R**2)**2 + d2/dZ**2 + k_max**2 [1 + (P_bar / R_c**2) (R**2 - R_c**2)],
//! ```
//!
//! with `remainder_constant[n] = (P_n - P_bar delta_n1) + (Q_n - Q_bar delta_n1)` and `remainder_slope[n] = (P_n - P_bar delta_n1) / R_c**2`,
//! `P_n` and `Q_n` being the coefficients of `(1 - psi_N)**n` of the profiles. The constant terms of the profiles, `P_0` and `Q_0`
//! (Solov'ev's), are the `n = 0` power of the remainder. With the paper's Taylor series `(1 - psi_N) = sum_l lambda**l (1 - psi_N)_l`
//! and `k_max**2 = sum_l lambda**l k_max,l**2`, order `l >= 1` is
//!
//! ```text
//! L_0 (1 - psi_N)_l = -k_max,l**2 [1 + (P_bar / R_c**2) (R**2 - R_c**2)] (1 - psi_N)_0
//!                     - sum_{i=1}^{l-1} k_max,i**2 [1 + (P_bar / R_c**2) (R**2 - R_c**2)] (1 - psi_N)_{l-i}
//!                     - sum_{i=0}^{l-1} k_max,i**2 sum_n [remainder_constant[n] + remainder_slope[n] (R**2 - R_c**2)] [(1 - psi_N)**n]_{l-1-i},
//! ```
//!
//! `L_0` being `L` with `k_max,0` (`reference_operator`): a linear problem whose source is a finite lattice series; since
//! `[(1 - psi_N)**0]_l = delta_l0` (the unit series), the constant terms add the `n = 0` term with `k_max,(l-1)**2` only to the source of
//! order `l`. Its solution is the minimum-norm particular solution plus homogeneous terms fitted to the boundary conditions, with
//! `k_max,l**2` fitted with them, and the multiple of `(1 - psi_N)_0` from `(1 - psi_N) = 1` at the moving magnetic axis, which moves in
//! `R` and, for an up-down asymmetric shape, in `Z` too (the paper's three conditions of each order). With constant terms the problem at
//! `lambda = 1` is inhomogeneous, not an eigenproblem, and the series is the Taylor expansion in `lambda` of the branch that starts at
//! Guazzotto and Freidberg's solution.
//!
//! The paper's choices of `c_bar`: `slope_at_boundary` (`c_bar = sum_{i >= 1} c_i`, the slope at the LCFS; best for broad profiles),
//! `value_at_axis` (`c_bar = c_0 + c_1`, the value on the magnetic axis, which keeps the source on the axis fixed; needed for peaked
//! profiles, whose slope at the boundary meets an obstruction on the magnetic axis, and for profiles dominated by their constant terms,
//! whose slope at the LCFS can vanish), and `least_squares` (`c_bar = 3 int_0^1 S (1 - psi_N) d(1 - psi_N)` for the profile `S`, the
//! least-squares line through the origin, which makes the remainder as small as a line can in the mean over `(1 - psi_N)`).

use super::configuration::{BoundaryConditions, ConditionKind, Point, Points};
use super::eigenvalue::{BasisOptions, EigenSolution, condition_row_weights, solve_least_squares_eigenvalue};
use super::flux_geometry::{find_stationary_point, solve_2x2};
use super::lattice::{
    FluxValues, Lattice, LatticeSeries, Parity, midplane_coefficients, polynomial_value, series_add, series_add_assign, series_count, series_evaluate,
    series_evaluate_many, series_from_term, series_multiply_linear, series_product, series_scale, series_size, series_taylor, series_trim, series_values,
    series_zero, value_at_centre,
};
use super::numerics::{SingularValueDecomposition, pinv, rank_above, svd_full, svd_thin};
use super::profiles::{ProfileShapes, power_coefficients, profile_value};
use super::radial_series::RadialVariable;
use super::reference_operator::{ParticularSolution, ReferenceOperator, particular_solution};
use super::shape_basis::{RadialKind, ShapeBasis, ShapeTerm, condition_residuals, flux_from_coefficients, homogeneous_terms, shape_basis};
use faer::c64;
use ndarray::{Array1, Array2, Array3, s};
use std::sync::Arc;

/// Lattice points smaller than this, relative to `(1 - psi_N)_0`, are dropped from every order [dimensionless]
const PRUNE_TOLERANCE: f64 = 1.0e-17;

/// Weight of the interior values in the boundary-fitted least squares of orders >= 1. It is kept small: a larger weight holds the
/// corrections of orders >= 1 small inside the plasma, which biases the LCFS fit wherever they are not small (on an ST40 single null
/// with order-2 profiles, 1e-3 puts the LCFS 2 to 13 mm off, 1e-5 0.1 to 0.3 mm). The gauge directions of the fit are bounded by the
/// Tikhonov fit of `fit_regularisation = Auto` [dimensionless]
pub const REGULARISATION: f64 = 1.0e-5;

/// The same with cosh terms: the corrections that keep a spherical double null's corners on the LCFS are not small inside
/// [dimensionless]
pub const REGULARISATION_COSH: f64 = 1.0e-5;

/// The same with constant terms in the profiles: the corrections of orders >= 1 are then not small inside either (for Solov'ev's
/// profile the remainder is as large as the linear part) [dimensionless]
pub const REGULARISATION_CONSTANT_TERMS: f64 = 1.0e-5;

/// A series whose coefficient sums exceed this times those of order 0 is not continued [dimensionless]
const SERIES_SIZE_MAX: f64 = 1.0e3;

/// The paper's choice of `c_bar`, the coefficient of a profile's linear part, `c_bar (1 - psi_N)`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CBarMethod {
    /// `c_bar = sum_{i >= 1} c_i`, the slope at the boundary; broad profiles
    SlopeAtBoundary,
    /// `c_bar = c_0 + c_1`, the value on the magnetic axis; peaked profiles and those dominated by their constant terms
    ValueAtAxis,
    /// `c_bar = 3 int_0^1 S (1 - psi_N) d(1 - psi_N)`, the least-squares line `c_bar (1 - psi_N)` through the profile `S`
    LeastSquares,
}

impl CBarMethod {
    /// Its name in the options, the input `c_bar_method`
    pub fn name(&self) -> &'static str {
        match self {
            CBarMethod::SlopeAtBoundary => "slope_at_boundary",
            CBarMethod::ValueAtAxis => "value_at_axis",
            CBarMethod::LeastSquares => "least_squares",
        }
    }

    /// The coefficient `c_bar` of a profile's linear part, `c_bar (1 - psi_N)`.
    ///
    /// # Arguments
    /// * `coefficients` - `[c_0, c_1, ...]` [any unit]
    ///
    /// # Returns
    /// * the slope at the boundary, `sum_{i >= 1} c_i` (`SlopeAtBoundary`), the value on the magnetic axis, `c_0 + c_1`
    ///   (`ValueAtAxis`), or the least-squares slope, `3 int_0^1 S (1 - psi_N) d(1 - psi_N) = 3 sum_n a_n / (n + 2)` with `a_n` the
    ///   coefficients of `(1 - psi_N)**n` (`LeastSquares`) [same unit]
    pub fn linear_part(&self, coefficients: &[f64]) -> f64 {
        match self {
            CBarMethod::SlopeAtBoundary => power_coefficients(coefficients)[1],
            CBarMethod::ValueAtAxis => profile_value(1.0, coefficients),
            CBarMethod::LeastSquares => {
                let power: Array1<f64> = power_coefficients(coefficients);
                let n_power: usize = power.len();
                let mut slope: f64 = 0.0;
                for i_power in 0..n_power {
                    slope += 3.0 * power[i_power] / (i_power as f64 + 2.0);
                }
                slope
            }
        }
    }
}

/// The boundary fit of the series' orders >= 1
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitRegularisation {
    /// Plain least squares
    None,
    /// Tikhonov, `mu / sigma_max = fit_threshold`
    Tikhonov,
    /// Truncated SVD at `fit_threshold`
    Tsvd,
}

impl FitRegularisation {
    pub fn name(&self) -> &'static str {
        match self {
            FitRegularisation::None => "none",
            FitRegularisation::Tikhonov => "tikhonov",
            FitRegularisation::Tsvd => "tsvd",
        }
    }
}

/// The linear part of the profiles, `P_bar (1 - psi_N)` and `Q_bar (1 - psi_N)`, and the coefficients of the remainder,
/// `remainder_constant[n] + remainder_slope[n] (R**2 - R_c**2)` for `(1 - psi_N)**n`, `n = 0..n_power_max`; `n = 0` carries the constant
/// terms.
#[derive(Clone, Debug)]
pub struct LinearPartAndRemainder {
    /// `P_bar`, the linear part of `P`, `mu0 R_c**2 c_bar_p_prime / (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime)` [dimensionless]
    pub p_prime_linear_part: f64,
    /// [dimensionless]
    pub remainder_constant: Array1<f64>,
    /// [1 / metre ** 2]
    pub remainder_slope: Array1<f64>,
    /// The highest power of `(1 - psi_N)` in the profiles [dimensionless]
    pub n_power_max: usize,
}

impl LinearPartAndRemainder {
    /// # Arguments
    /// * `shapes` - the profiles `P` and `Q`
    /// * `r_c_squared` - `R_c**2` [metre ** 2]
    /// * `c_bar_method` - the choice of `c_bar`
    pub fn new(shapes: &ProfileShapes, r_c_squared: f64, c_bar_method: CBarMethod) -> LinearPartAndRemainder {
        let p_prime_power: Array1<f64> = power_coefficients(&shapes.p_prime);
        let ff_prime_power: Array1<f64> = power_coefficients(&shapes.ff_prime);
        // Index n is the coefficient of (1 - psi_N)**n, padded to a common length
        let n_power_profile: usize = p_prime_power.len().max(ff_prime_power.len());
        let mut p_prime_padded: Array1<f64> = Array1::zeros(n_power_profile);
        p_prime_padded.slice_mut(s![..p_prime_power.len()]).assign(&p_prime_power);
        let mut ff_prime_padded: Array1<f64> = Array1::zeros(n_power_profile);
        ff_prime_padded.slice_mut(s![..ff_prime_power.len()]).assign(&ff_prime_power);
        let p_prime_linear_part: f64 = c_bar_method.linear_part(&shapes.p_prime);
        let ff_prime_linear_part: f64 = c_bar_method.linear_part(&shapes.ff_prime);
        let mut remainder_constant: Array1<f64> = Array1::zeros(n_power_profile);
        let mut remainder_slope: Array1<f64> = Array1::zeros(n_power_profile);
        for i_power_profile in 0..n_power_profile {
            let mut p_prime_remainder: f64 = p_prime_padded[i_power_profile];
            let mut ff_prime_remainder: f64 = ff_prime_padded[i_power_profile];
            if i_power_profile == 1 {
                p_prime_remainder -= p_prime_linear_part;
                ff_prime_remainder -= ff_prime_linear_part;
            }
            remainder_constant[i_power_profile] = p_prime_remainder + ff_prime_remainder;
            remainder_slope[i_power_profile] = p_prime_remainder / r_c_squared;
        }
        LinearPartAndRemainder {
            p_prime_linear_part,
            remainder_constant,
            remainder_slope,
            n_power_max: n_power_profile - 1,
        }
    }
}

/// The paper's Taylor series `(1 - psi_N) = sum_l lambda**l (1 - psi_N)_l`, `k_max**2 = sum_l lambda**l k_max,l**2`, with
/// `(1 - psi_N)_0 = 1` on its magnetic axis.
#[derive(Clone)]
pub struct PerturbationSeries {
    pub lattice: Lattice,
    pub operator: ReferenceOperator,
    /// `(1 - psi_N)_l` [dimensionless]
    pub one_minus_psi_n_orders: Vec<LatticeSeries>,
    /// `k_max,l**2` [1 / metre ** 2]
    pub k_max_squared_orders: Vec<f64>,
    /// Coefficients in `lambda` of the magnetic axis' shift from order 0's, in the scaled radial variable [dimensionless]
    pub axis_radial_orders: Vec<f64>,
    /// ... and in `Z` [metre]
    pub axis_z_orders: Vec<f64>,
    /// Number of distinct non-zero wavenumbers in every order [dimensionless]
    pub n_wavenumbers: Vec<usize>,
    /// Largest source dropped by the growth cut-offs, relative to `k_max,0**2` times the size of `(1 - psi_N)_0`, the size of the terms
    /// of order 0's equation [dimensionless]
    pub dropped_source: f64,
    /// Weight of the interior values in the fits of orders >= 1, and why it was chosen [dimensionless]
    pub interior_penalty: f64,
    pub interior_penalty_choice: String,
}

impl PerturbationSeries {
    /// Number of orders computed, the terms `l = 0..n_taylor - 1` of the Taylor series [dimensionless]
    pub fn n_taylor(&self) -> usize {
        self.k_max_squared_orders.len()
    }

    /// `sum_{l < n_taylor} lambda**l (1 - psi_N)_l` [dimensionless]
    pub fn one_minus_psi_n_sum(&self, n_taylor: usize, lambda: f64) -> LatticeSeries {
        let mut total: LatticeSeries = self.one_minus_psi_n_orders[0].clone();
        for i_order in 1..n_taylor {
            series_add_assign(&mut total, &self.one_minus_psi_n_orders[i_order], lambda.powi(i_order as i32));
        }
        total
    }

    /// `k_max**2 = sum_{l < n_taylor} lambda**l k_max,l**2` [1 / metre ** 2]
    pub fn k_max_squared(&self, n_taylor: usize, lambda: f64) -> f64 {
        let mut total: f64 = 0.0;
        for i_order in 0..n_taylor {
            total += self.k_max_squared_orders[i_order] * lambda.powi(i_order as i32);
        }
        total
    }
}

/// The options of one perturbation series
#[derive(Clone, Copy, Debug)]
pub struct PerturbationOptions {
    /// Number of equally spaced wavenumbers, `n_wave >= 4` [dimensionless]
    pub n_wave: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    pub k_spacing: f64,
    /// Number of `cosh(|k_j| Z)` terms' wavenumbers [dimensionless]
    pub n_wave_cosh: usize,
    /// Their spacing in units of `k_max` [dimensionless]
    pub k_spacing_cosh: f64,
    pub is_up_down_symmetric: bool,
    /// Most terms of the Taylor series, the orders `0..n_taylor_max - 1` [dimensionless]
    pub n_taylor_max: usize,
    pub c_bar_method: CBarMethod,
    /// Stop early once an order adds less than this to `(1 - psi_N)` and to `k_max**2`, relative to order 0 [dimensionless]
    pub taylor_tolerance: f64,
    pub fit_regularisation: FitRegularisation,
    /// [dimensionless]
    pub fit_threshold: f64,
    pub verbose: bool,
}

/// The magnetic axis in the scaled radial variable and `Z`
#[derive(Clone, Copy, Debug)]
pub struct AxisInRadial {
    /// The scaled radial variable [dimensionless]
    pub radial: f64,
    /// [metre]
    pub z: f64,
}

/// Coefficients in `lambda` of `shift_radial(lambda)**a shift_z(lambda)**b`, for the composition of a two-variable Taylor series with the
/// axis' path.
///
/// # Arguments
/// * `shift_radial` - coefficients of the axis' shift in the scaled radial variable, `shift_radial[0] = 0` [dimensionless]
/// * `shift_z` - coefficients of its shift in `Z`, `shift_z[0] = 0`, as many [metre]
/// * `n_degree` - number of powers `a`, `b` [dimensionless]
///
/// # Returns
/// * `T[a, b, l]`, the coefficient of `lambda**l`, shape `(n_degree, n_degree, len(shift_radial))` [metre ** b]
fn displacement_powers(shift_radial: &Array1<f64>, shift_z: &Array1<f64>, n_degree: usize) -> Array3<f64> {
    let n_taylor: usize = shift_radial.len();
    let mut powers_radial: Array2<f64> = Array2::zeros((n_degree, n_taylor));
    let mut powers_z: Array2<f64> = Array2::zeros((n_degree, n_taylor));
    powers_radial[[0, 0]] = 1.0;
    powers_z[[0, 0]] = 1.0;
    // The product of the power series `a` and `b` in `lambda`, truncated at `n_taylor` terms
    let convolve_truncated = |a: &Array1<f64>, b: &Array1<f64>| -> Array1<f64> {
        let mut result: Array1<f64> = Array1::zeros(n_taylor);
        for i_order in 0..n_taylor {
            for j_order in 0..=i_order {
                result[i_order] += a[j_order] * b[i_order - j_order];
            }
        }
        result
    };
    for i_degree in 1..n_degree {
        let previous_radial: Array1<f64> = powers_radial.row(i_degree - 1).to_owned();
        powers_radial.row_mut(i_degree).assign(&convolve_truncated(&previous_radial, shift_radial));
        let previous_z: Array1<f64> = powers_z.row(i_degree - 1).to_owned();
        powers_z.row_mut(i_degree).assign(&convolve_truncated(&previous_z, shift_z));
    }
    let mut has_shift_z: bool = false;
    for i_order in 0..n_taylor {
        has_shift_z |= shift_z[i_order] != 0.0;
    }
    let mut products: Array3<f64> = Array3::zeros((n_degree, n_degree, n_taylor));
    for i_a in 0..n_degree {
        products.slice_mut(s![i_a, 0, ..]).assign(&powers_radial.row(i_a));
        if has_shift_z {
            for i_b in 1..n_degree {
                // shift_radial**a shift_z**b starts at lambda**(a + b)
                if i_a + i_b < n_taylor {
                    let radial_row: Array1<f64> = powers_radial.row(i_a).to_owned();
                    let z_row: Array1<f64> = powers_z.row(i_b).to_owned();
                    products.slice_mut(s![i_a, i_b, ..]).assign(&convolve_truncated(&radial_row, &z_row));
                }
            }
        }
    }
    products
}

/// The magnetic axis, the maximum of `(1 - psi_N)`: on the midplane from its power series (`find_midplane_axis`), then, for an up-down
/// asymmetric series, by Newton's method in `(R, Z)` from there.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - `(1 - psi_N)`, or a multiple of it [dimensionless]
/// * `radial_low` - midplane scan start, in the scaled radial variable [dimensionless]
/// * `radial_high` - midplane scan end [dimensionless]
pub fn find_axis(lattice: &Lattice, series: &LatticeSeries, radial_low: f64, radial_high: f64) -> AxisInRadial {
    let radial_axis: f64 = find_midplane_axis(&midplane_coefficients(series), radial_low, radial_high);
    if lattice.is_up_down_symmetric {
        return AxisInRadial { radial: radial_axis, z: 0.0 };
    }
    let flux_function = |r: &[f64], z: &[f64]| -> FluxValues { series_evaluate(lattice, series, r, z) };
    let axis: Point = find_stationary_point(
        flux_function,
        Point {
            r: lattice.radial.to_r(radial_axis),
            z: 0.0,
        },
        lattice.radial.a_minor,
    );
    AxisInRadial {
        radial: lattice.radial.of_r(axis.r),
        z: axis.z,
    }
}

/// The maximum of `(1 - psi_N)` on the midplane from its power series in the scaled radial variable: a scan, then a bounded Newton
/// iteration on its derivative being zero.
///
/// # Arguments
/// * `midplane` - power-series coefficients of `(1 - psi_N)` on the midplane [dimensionless]
/// * `radial_low` - scan start [dimensionless]
/// * `radial_high` - scan end [dimensionless]
pub fn find_midplane_axis(midplane: &Array1<f64>, radial_low: f64, radial_high: f64) -> f64 {
    let n_scan: usize = 2001;
    let radial_scan: Array1<f64> = Array1::linspace(radial_low, radial_high, n_scan);
    let mut i_maximum: usize = 0;
    let mut value_maximum: f64 = f64::NEG_INFINITY;
    for i_scan in 0..n_scan {
        let value: f64 = polynomial_value(midplane, radial_scan[i_scan]);
        if value > value_maximum {
            value_maximum = value;
            i_maximum = i_scan;
        }
    }
    let mut radial_axis: f64 = radial_scan[i_maximum];
    let n_coefficient: usize = midplane.len();
    let mut derivative_1: Array1<f64> = Array1::zeros(n_coefficient.saturating_sub(1).max(1));
    for i_coefficient in 1..n_coefficient {
        derivative_1[i_coefficient - 1] = midplane[i_coefficient] * (i_coefficient as f64);
    }
    let mut derivative_2: Array1<f64> = Array1::zeros(n_coefficient.saturating_sub(2).max(1));
    for i_coefficient in 2..n_coefficient {
        derivative_2[i_coefficient - 2] = midplane[i_coefficient] * (i_coefficient as f64) * ((i_coefficient - 1) as f64);
    }
    let d_radial_scan: f64 = (radial_high - radial_low) / (n_scan as f64 - 1.0);
    let n_newton_max: usize = 50;
    'newton_loop: for _i_newton in 0..n_newton_max {
        let step_raw: f64 = polynomial_value(&derivative_1, radial_axis) / polynomial_value(&derivative_2, radial_axis);
        let step: f64 = step_raw.min(d_radial_scan).max(-d_radial_scan);
        radial_axis -= step;
        if step.abs() < 1.0e-15 {
            break 'newton_loop;
        }
    }
    radial_axis
}

/// The free homogeneous terms of the boundary-fitted fits of orders >= 1: `C_j` and `S_j` at the paper's wavenumbers and the next two,
/// `k_j = n k_spacing k_max`, `n = 0..n_wave + 1`, with their odd (`sin`) partners for an up-down asymmetric shape.
///
/// # Arguments
/// * `n_wave` - number of equally spaced wavenumbers, `n_wave >= 4` [dimensionless]
/// * `n_base` - number of lattice axes: 1, or 2 with an imaginary (`cosh`) axis [dimensionless]
/// * `is_up_down_symmetric` - false adds the `sin` terms
pub fn free_term_keys(n_wave: usize, n_base: usize, is_up_down_symmetric: bool) -> Vec<ShapeTerm> {
    let mut keys: Vec<ShapeTerm> = Vec::new();
    for i_harmonic in 0..n_wave + 2 {
        let mut key: Vec<i64> = vec![0; n_base];
        key[0] = i_harmonic as i64;
        keys.push(ShapeTerm {
            key: key.clone(),
            radial_kind: RadialKind::C,
            parity: Parity::Cos,
        });
        keys.push(ShapeTerm {
            key,
            radial_kind: RadialKind::S,
            parity: Parity::Cos,
        });
    }
    if !is_up_down_symmetric {
        for i_harmonic in 1..n_wave + 2 {
            let mut key: Vec<i64> = vec![0; n_base];
            key[0] = i_harmonic as i64;
            keys.push(ShapeTerm {
                key: key.clone(),
                radial_kind: RadialKind::C,
                parity: Parity::Sin,
            });
            keys.push(ShapeTerm {
                key,
                radial_kind: RadialKind::S,
                parity: Parity::Sin,
            });
        }
    }
    keys
}

/// The fit of orders >= 1 as a linear map: the amplitudes `z` (free terms and `k_max,l**2`) that an order adds for a function `q` with
/// least-squares data `D_L(q)` (the rows of `fit_matrix`) and exact data `C(q)` (the rows of `exact_matrix`) are `z = -K [D_L(q); C(q)]`.
/// In the scaled amplitudes `z' = W z` (`W` the largest value of each term on the plasma), with `z' = z'_p + N w`, `z'_p` the
/// minimum-norm solution of the exact rows `E W^-1 z' = -C(q)` and `N` their null space, `w` minimises `|B w - r|**2` (plus a
/// penalty), with `B = A W^-1 N` and `r = -D_L(q) - A W^-1 z'_p`:
///
/// * `None`: plain least squares;
/// * `Tsvd`: truncated SVD, the singular values of `B` below `threshold` times the largest dropped;
/// * `Tikhonov`: `+ mu**2 |z'_penalised|**2`, `mu` being `threshold` times the largest singular value of `B`, which bounds the scaled
///   amplitudes of the free terms (not `k_max,l**2`).
///
/// # Arguments
/// * `fit_matrix` - `A`, the weighted least-squares rows of the free terms and of the solution that `k_max,l**2` multiplies, one column
///   each [dimensionless]
/// * `exact_matrix` - `E`, their exact rows (an X-point's), possibly none [dimensionless]
/// * `column_scales` - `W`, the largest value of each term on the plasma [dimensionless]
/// * `is_penalised` - the amplitudes the Tikhonov penalty applies to
/// * `regularisation` - `None`, `Tsvd` or `Tikhonov`
/// * `threshold` - the relative singular-value cut-off (`Tsvd`) or `mu / sigma_max` (`Tikhonov`); unused for `None` [dimensionless]
///
/// # Returns
/// * `K`, shape `(n_unknown, n_least_squares + n_exact)` [dimensionless]
fn fit_operator(
    fit_matrix: &Array2<f64>,
    exact_matrix: &Array2<f64>,
    column_scales: &Array1<f64>,
    is_penalised: &[bool],
    regularisation: FitRegularisation,
    threshold: f64,
) -> Array2<f64> {
    let n_least_squares: usize = fit_matrix.nrows();
    let n_unknown: usize = fit_matrix.ncols();
    let n_exact: usize = exact_matrix.nrows();
    let mut scaled_fit: Array2<f64> = fit_matrix.clone();
    for i_unknown in 0..n_unknown {
        let column_scale: f64 = column_scales[i_unknown];
        scaled_fit.column_mut(i_unknown).mapv_inplace(|value: f64| value / column_scale);
    }
    let mut exact_pseudo_inverse: Array2<f64> = Array2::zeros((n_unknown, n_exact));
    let mut exact_null: Array2<f64> = Array2::eye(n_unknown);
    if n_exact > 0 {
        let mut scaled_exact: Array2<f64> = exact_matrix.clone();
        for i_unknown in 0..n_unknown {
            let column_scale: f64 = column_scales[i_unknown];
            scaled_exact.column_mut(i_unknown).mapv_inplace(|value: f64| value / column_scale);
        }
        let svd_exact: SingularValueDecomposition = svd_full(scaled_exact.view());
        let rank_exact: usize = rank_above(&svd_exact.s, 1.0e-13);
        exact_pseudo_inverse = pinv(scaled_exact.view(), 1.0e-13);
        exact_null = svd_exact.v.slice(s![.., rank_exact..]).to_owned();
    }
    let n_null: usize = exact_null.ncols();
    let reduced: Array2<f64> = scaled_fit.dot(&exact_null);
    let svd: SingularValueDecomposition = svd_thin(reduced.view());
    let n_singular: usize = svd.s.len();
    let mut penalty: Array2<f64> = Array2::zeros((n_unknown, n_unknown));
    for i_unknown in 0..n_unknown {
        if is_penalised[i_unknown] {
            penalty[[i_unknown, i_unknown]] = 1.0;
        }
    }
    // w = reduced_inverse r + penalty_inverse (-mu D z'_p), D the diagonal of the penalised amplitudes
    let mut reduced_inverse: Array2<f64> = Array2::zeros((n_null, n_least_squares));
    let mut penalty_inverse: Array2<f64> = Array2::zeros((n_null, n_unknown));
    let mut mu: f64 = 0.0;
    match regularisation {
        FitRegularisation::None | FitRegularisation::Tsvd => {
            let mut cutoff: f64 = f64::EPSILON * (n_least_squares.max(n_null) as f64);
            if regularisation == FitRegularisation::Tsvd {
                cutoff = cutoff.max(threshold);
            }
            for i_singular in 0..n_singular {
                if svd.s[i_singular] > cutoff * svd.s[0] {
                    let inverse_singular: f64 = 1.0 / svd.s[i_singular];
                    for i_null in 0..n_null {
                        let factor: f64 = svd.v[[i_null, i_singular]] * inverse_singular;
                        for i_least_squares in 0..n_least_squares {
                            reduced_inverse[[i_null, i_least_squares]] += factor * svd.u[[i_least_squares, i_singular]];
                        }
                    }
                }
            }
        }
        FitRegularisation::Tikhonov => {
            mu = threshold * svd.s[0];
            let penalty_rows: Array2<f64> = penalty.dot(&exact_null).mapv(|value: f64| mu * value);
            let mut stacked: Array2<f64> = Array2::zeros((n_least_squares + n_unknown, n_null));
            stacked.slice_mut(s![..n_least_squares, ..]).assign(&reduced);
            stacked.slice_mut(s![n_least_squares.., ..]).assign(&penalty_rows);
            let stacked_inverse: Array2<f64> = pinv(stacked.view(), 1.0e-15);
            reduced_inverse = stacked_inverse.slice(s![.., ..n_least_squares]).to_owned();
            penalty_inverse = stacked_inverse.slice(s![.., n_least_squares..]).to_owned();
        }
    }
    // z' = -[N R_inv] D_L(q) - [(I - N (R_inv A W^-1 + mu P_inv D)) E^+] C(q)
    let mut operator_scaled: Array2<f64> = Array2::from_elem((n_unknown, n_least_squares + n_exact), f64::NAN);
    operator_scaled.slice_mut(s![.., ..n_least_squares]).assign(&exact_null.dot(&reduced_inverse));
    if n_exact > 0 {
        let inner: Array2<f64> = reduced_inverse.dot(&scaled_fit) + penalty_inverse.dot(&penalty).mapv(|value: f64| mu * value);
        let complement: Array2<f64> = Array2::eye(n_unknown) - exact_null.dot(&inner);
        operator_scaled
            .slice_mut(s![.., n_least_squares..])
            .assign(&complement.dot(&exact_pseudo_inverse));
    }
    for i_unknown in 0..n_unknown {
        let column_scale: f64 = column_scales[i_unknown];
        operator_scaled.row_mut(i_unknown).mapv_inplace(|value: f64| value / column_scale);
    }
    operator_scaled
}

/// The median: the middle value, or the mean of the two middle ones
fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|first: &f64, second: &f64| first.partial_cmp(second).unwrap_or(std::cmp::Ordering::Equal));
    let n_value: usize = sorted.len();
    if n_value % 2 == 1 {
        sorted[n_value / 2]
    } else {
        0.5 * (sorted[n_value / 2 - 1] + sorted[n_value / 2])
    }
}

/// Linear interpolation of `(xp, fp)`, `xp` increasing, clamped to the end values outside
fn interpolate_linear(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n_point: usize = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n_point - 1] {
        return fp[n_point - 1];
    }
    let mut i_right: usize = 1;
    'search_loop: for i_point in 1..n_point {
        i_right = i_point;
        if xp[i_point] > x {
            break 'search_loop;
        }
    }
    let i_left: usize = i_right - 1;
    let slope: f64 = (fp[i_right] - fp[i_left]) / (xp[i_right] - xp[i_left]);
    slope * (x - xp[i_left]) + fp[i_left]
}

/// The perturbation series, its first `n_taylor_max` terms at most.
///
/// The `n_wave` equally spaced wavenumbers (`2 n_wave - 1` terms) with more conditions than terms. Order 0 is the least-squares
/// eigenproblem; from order 1 on, the amplitudes `A_j C_j + B_j S_j` at the paper's wavenumbers and the next two and `k_max,l**2` are
/// fitted jointly to the weighted conditions by least squares, with `REGULARISATION` times the interior values as a penalty, and the
/// higher wavenumbers keep the minimum-norm particular solution.
///
/// # Arguments
/// * `radial` - the radial variable of the power series
/// * `shapes` - the profiles `P` and `Q`
/// * `conditions` - the boundary conditions, at least `2 n_wave - 1`
/// * `interior` - interior points of the least-squares normalisation [metre]
/// * `options` - the options
/// * `reference_operator` - the operator of an earlier series on the same radial variable; if it has the same `k_max,0` and
///   `p_prime_term` (the same order 0), it is reused with the particular-solution matrices it has cached
///
/// # Errors
/// No order-0 eigenvalue, or invalid options
pub fn solve_perturbation(
    radial: &Arc<RadialVariable>,
    shapes: &ProfileShapes,
    conditions: &BoundaryConditions,
    interior: &Points,
    options: &PerturbationOptions,
    reference_operator: Option<&ReferenceOperator>,
) -> Result<PerturbationSeries, String> {
    let r_c_squared: f64 = radial.r_c_squared;
    let linear_part_and_remainder: LinearPartAndRemainder = LinearPartAndRemainder::new(shapes, r_c_squared, options.c_bar_method);
    let p_prime_linear_part: f64 = linear_part_and_remainder.p_prime_linear_part;
    let n_power_max: usize = linear_part_and_remainder.n_power_max;
    let n_wave: usize = options.n_wave;
    let n_taylor_max: usize = options.n_taylor_max;
    let n_condition: usize = conditions.n_condition();
    let basis_options: BasisOptions = BasisOptions {
        n_wave,
        k_spacing: options.k_spacing,
        n_wave_cosh: options.n_wave_cosh,
        k_spacing_cosh: options.k_spacing_cosh,
        is_up_down_symmetric: options.is_up_down_symmetric,
    };

    // Order 0: Guazzotto and Freidberg's eigenvalue problem for the linear part
    let mut row_weights: Array1<f64> = condition_row_weights(radial, p_prime_linear_part, conditions, &basis_options)?;
    let eigen: EigenSolution = solve_least_squares_eigenvalue(radial, p_prime_linear_part, conditions, &row_weights, interior, &basis_options)?;
    let k_max_0: f64 = eigen.k_max_0;
    let mut z_top: f64 = 0.0;
    for i_condition in 0..n_condition {
        z_top = z_top.max(conditions.z[i_condition].abs());
    }
    let shape: ShapeBasis = shape_basis(
        radial,
        p_prime_linear_part,
        k_max_0,
        n_wave,
        options.k_spacing,
        options.n_wave_cosh,
        options.k_spacing_cosh,
        z_top,
        options.is_up_down_symmetric,
    )?;
    let mut lattice: Lattice = shape.lattice;
    let basis: Vec<LatticeSeries> = shape.basis;
    let mut operator: ReferenceOperator = shape.operator;
    if let Some(reference) = reference_operator {
        if reference.k_max_0 == operator.k_max_0 && reference.p_prime_term == operator.p_prime_term {
            operator = reference.clone();
        }
    }
    if options.n_wave_cosh > 0 {
        // The plasma's half-height at the radial quadrature nodes, from the LCFS points of the conditions (the upper half, single valued
        // in R**2), which weights the projections of the wavenumbers with imaginary parts by their amplification over the plasma
        let mut condition_r_squared: Vec<f64> = Vec::with_capacity(n_condition);
        let mut condition_z: Vec<f64> = Vec::with_capacity(n_condition);
        for i_condition in 0..n_condition {
            condition_r_squared.push(conditions.r[i_condition] * conditions.r[i_condition]);
            condition_z.push(conditions.z[i_condition].abs());
        }
        let order: Vec<usize> = super::configuration::argsort_stable(&condition_r_squared);
        let mut sorted_r_squared: Vec<f64> = Vec::with_capacity(n_condition);
        let mut sorted_z: Vec<f64> = Vec::with_capacity(n_condition);
        for i_condition in 0..n_condition {
            sorted_r_squared.push(condition_r_squared[order[i_condition]]);
            sorted_z.push(condition_z[order[i_condition]]);
        }
        let heights: Array1<f64> = radial
            .quadrature_r_squared
            .mapv(|r_squared: f64| interpolate_linear(r_squared, &sorted_r_squared, &sorted_z));
        lattice.plasma_heights = Some(heights);
    }
    let order_0_raw: LatticeSeries = flux_from_coefficients(&basis, &eigen.coefficients);
    // First 1 at (R_c, 0), as Guazzotto and Freidberg, which fixes the sign; then 1 at its magnetic axis. The axis is searched for
    // within 90 % of the way from R_c**2 to the plasma's edges in R**2
    let order_0_centred: LatticeSeries = series_scale(&order_0_raw, 1.0 / value_at_centre(&lattice, &order_0_raw));
    let r_squared_search: f64 = 0.9 * radial.r_squared_half_width;
    let axis_0: AxisInRadial = find_axis(
        &lattice,
        &order_0_centred,
        radial.of_r_squared(r_c_squared - r_squared_search),
        radial.of_r_squared(r_c_squared + r_squared_search),
    );
    let n_degree: usize = n_taylor_max + 2;
    let order_0: LatticeSeries = series_scale(
        &order_0_centred,
        1.0 / series_taylor(&lattice, &order_0_centred, axis_0.radial, axis_0.z, 1)[[0, 0]],
    );
    let size_0: f64 = series_size(&order_0);
    let tolerance: f64 = PRUNE_TOLERANCE * size_0;

    let mut conditions_kind_is_value: Vec<bool> = vec![false; n_condition];
    for i_condition in 0..n_condition {
        conditions_kind_is_value[i_condition] = conditions.kind[i_condition] == ConditionKind::Value && !conditions.is_exact[i_condition];
    }
    if options.n_wave_cosh > 0 {
        // With cosh terms the rows' lengths in the basis (the eigenproblem's weights make them 1) vary by x100 along the LCFS (the cosh
        // terms' radial functions are large next to R = 0), which made the fits of orders >= 1 ignore the inboard side of a spherical
        // tokamak: there they weight the (1 - psi_N) = 0 rows by 1 / |grad (1 - psi_N)_0|, the gradient made dimensionless as the
        // conditions are, so that each residual is a displacement of the LCFS, scaled so that the median row has unit length in the
        // basis as before (the other rows keep their weights)
        let flux_0: FluxValues = series_evaluate(&lattice, &order_0, &conditions.r, &conditions.z);
        let mut value_rows: Vec<usize> = Vec::new();
        let mut gradients: Vec<f64> = Vec::new();
        for i_condition in 0..n_condition {
            if conditions_kind_is_value[i_condition] {
                value_rows.push(i_condition);
                let length_r: f64 = radial.r_geo * radial.a_minor / conditions.r[i_condition];
                gradients.push((length_r * flux_0.d_d_r[i_condition]).hypot(radial.a_minor * flux_0.d_d_z[i_condition]));
            }
        }
        let n_value: usize = value_rows.len();
        if n_value > 0 {
            let gradient_floor: f64 = 0.1 * median(&gradients);
            let mut value_gradients: Vec<f64> = Vec::with_capacity(n_value);
            let mut inverse_lengths: Vec<f64> = Vec::with_capacity(n_value);
            for i_value in 0..n_value {
                value_gradients.push(gradients[i_value].max(gradient_floor));
                // Row i's length in the basis is 1 / row_weights[i]; with the weight c / g_i it is c / (g_i row_weights[i])
                inverse_lengths.push(1.0 / (value_gradients[i_value] * row_weights[value_rows[i_value]]));
            }
            let scale: f64 = 1.0 / median(&inverse_lengths);
            for i_value in 0..n_value {
                row_weights[value_rows[i_value]] = scale / value_gradients[i_value];
            }
        }
    }
    // The interior penalty: the default, or the one for cosh terms or for constant terms in the profiles, where the corrections of
    // orders >= 1 are not small inside
    let mut interior_penalty: f64 = REGULARISATION;
    let mut interior_penalty_choice: String = "the default".to_string();
    if options.n_wave_cosh > 0 {
        interior_penalty = REGULARISATION_COSH;
        interior_penalty_choice = "cosh terms: their corner corrections are not small inside".to_string();
    } else if shapes.has_constant_terms() {
        interior_penalty = REGULARISATION_CONSTANT_TERMS;
        interior_penalty_choice = "constant terms: the corrections of orders >= 1 are not small inside".to_string();
    }

    // The solution k_max,l**2 multiplies: L_0 X = -[1 + (P_bar / R_c**2) (R**2 - R_c**2)] (1 - psi_N)_0
    let linear_slope: f64 = p_prime_linear_part / r_c_squared;
    // The size of the terms of order 0's equation, which the dropped sources are measured against [1 / metre ** 2]
    let source_size_0: f64 = k_max_0 * k_max_0 * size_0;
    let mut dropped_source: f64 = 0.0;
    let k_max_squared_solution: ParticularSolution =
        particular_solution(&lattice, &mut operator, &series_multiply_linear(&lattice, &order_0, -1.0, -linear_slope));
    dropped_source = dropped_source.max(k_max_squared_solution.dropped_size / source_size_0);
    let k_max_squared_series: LatticeSeries = k_max_squared_solution.series;

    // The free terms C_j, S_j at the paper's wavenumbers and the next two, and their weighted rows; the exact conditions (an X-point's)
    // are imposed exactly, `z = z_p + N w` with `z_p` the minimum-norm solution of the exact rows and `N` their null space
    let mut exact_rows: Vec<usize> = Vec::new();
    let mut least_squares_rows: Vec<usize> = Vec::new();
    for i_condition in 0..n_condition {
        if conditions.is_exact[i_condition] {
            exact_rows.push(i_condition);
        } else {
            least_squares_rows.push(i_condition);
        }
    }
    let n_least_squares: usize = least_squares_rows.len();
    let n_exact: usize = exact_rows.len();
    let n_interior: usize = interior.r.len();
    let interior_r: &[f64] = interior.r.as_slice().expect("contiguous");
    let interior_z: &[f64] = interior.z.as_slice().expect("contiguous");
    let free_terms: Vec<LatticeSeries> = homogeneous_terms(&lattice, &operator, &free_term_keys(n_wave, lattice.n_base, options.is_up_down_symmetric));
    let n_free: usize = free_terms.len();
    let mut fit_matrix: Array2<f64> = Array2::from_elem((n_least_squares + n_interior, n_free + 1), f64::NAN);
    let mut exact_matrix: Array2<f64> = Array2::from_elem((n_exact, n_free + 1), f64::NAN);
    // The largest value of each term on the plasma (its interior points and the LCFS points of the conditions), which scales its
    // amplitude
    let mut plasma_r: Vec<f64> = interior_r.to_vec();
    plasma_r.extend_from_slice(&conditions.r);
    let mut plasma_z: Vec<f64> = interior_z.to_vec();
    plasma_z.extend_from_slice(&conditions.z);
    let mut column_scales: Array1<f64> = Array1::from_elem(n_free + 1, f64::NAN);
    let mut all_terms: Vec<&LatticeSeries> = Vec::with_capacity(n_free + 1);
    for i_free in 0..n_free {
        all_terms.push(&free_terms[i_free]);
    }
    all_terms.push(&k_max_squared_series);
    let condition_flux: Vec<FluxValues> = series_evaluate_many(&lattice, &all_terms, &conditions.r, &conditions.z);
    let interior_flux: Vec<FluxValues> = series_evaluate_many(&lattice, &all_terms, interior_r, interior_z);
    let plasma_flux: Vec<FluxValues> = series_evaluate_many(&lattice, &all_terms, &plasma_r, &plasma_z);
    for i_column in 0..=n_free {
        let residuals: Array1<f64> = conditions.residuals(&condition_flux[i_column]);
        for i_least_squares in 0..n_least_squares {
            let i_condition: usize = least_squares_rows[i_least_squares];
            fit_matrix[[i_least_squares, i_column]] = row_weights[i_condition] * residuals[i_condition];
        }
        for i_interior in 0..n_interior {
            fit_matrix[[n_least_squares + i_interior, i_column]] = interior_penalty * interior_flux[i_column].one_minus_psi_n[i_interior];
        }
        for i_exact in 0..n_exact {
            exact_matrix[[i_exact, i_column]] = residuals[exact_rows[i_exact]];
        }
        let mut largest: f64 = f64::NEG_INFINITY;
        for value in plasma_flux[i_column].one_minus_psi_n.iter() {
            largest = largest.max(value.abs());
        }
        column_scales[i_column] = largest;
    }
    // The penalty bounds the free amplitudes, not k_max,l**2
    let mut is_penalised: Vec<bool> = vec![true; n_free + 1];
    is_penalised[n_free] = false;
    let fit: Array2<f64> = fit_operator(
        &fit_matrix,
        &exact_matrix,
        &column_scales,
        &is_penalised,
        options.fit_regularisation,
        options.fit_threshold,
    );

    // [(1 - psi_N)**n]_l for n = 0..n_power_max, as power_cache[n][l]; [(1 - psi_N)**0]_l = delta_l0, so power_cache[0] holds the unit
    // series, l = 0, only
    let mut power_cache: Vec<Vec<LatticeSeries>> = vec![Vec::new(); n_power_max + 1];
    let mut unit_radial: Array1<c64> = Array1::from_elem(lattice.n_radial, c64::new(0.0, 0.0));
    unit_radial[0] = c64::new(1.0, 0.0);
    let zero_key: Vec<i64> = vec![0; lattice.n_base];
    power_cache[0].push(series_from_term(&lattice, &zero_key, unit_radial.view(), Parity::Cos)?);
    if n_power_max >= 1 {
        power_cache[1].push(order_0.clone());
    }
    for i_power_profile in 2..=n_power_max {
        let product: LatticeSeries = series_product(&lattice, &power_cache[i_power_profile - 1][0], &order_0);
        power_cache[i_power_profile].push(series_trim(&product, tolerance));
    }

    // Taylor coefficients of (1 - psi_N)_l about order 0's axis, in the scaled radial variable and Z; the Hessian of (1 - psi_N)_0 there
    let mut taylors: Vec<Array2<f64>> = vec![series_taylor(&lattice, &order_0, axis_0.radial, axis_0.z, n_degree)];
    let hessian: [[f64; 2]; 2] = [[2.0 * taylors[0][[2, 0]], taylors[0][[1, 1]]], [taylors[0][[1, 1]], 2.0 * taylors[0][[0, 2]]]];

    let mut one_minus_psi_n_orders: Vec<LatticeSeries> = vec![order_0.clone()];
    let mut k_max_squared_orders: Vec<f64> = vec![k_max_0 * k_max_0];
    let mut axis_radial_orders: Vec<f64> = vec![0.0];
    let mut axis_z_orders: Vec<f64> = vec![0.0];
    let mut n_wavenumbers: Vec<usize> = vec![series_count(&order_0)];
    let zero_half_widths: Vec<usize> = vec![0; lattice.n_base];

    for i_order in 1..n_taylor_max {
        // The source of order l, without the k_max,l**2 term. The product with a linear function is linear in the series, so each
        // linear function multiplies, once, the sum of the series it multiplies
        let mut orders_sum: LatticeSeries = series_zero(&lattice, &zero_half_widths);
        for i_previous in 1..i_order {
            series_add_assign(&mut orders_sum, &one_minus_psi_n_orders[i_order - i_previous], k_max_squared_orders[i_previous]);
        }
        let mut source: LatticeSeries = series_multiply_linear(&lattice, &orders_sum, -1.0, -linear_slope);
        for i_power_profile in 0..=n_power_max {
            let remainder_constant: f64 = linear_part_and_remainder.remainder_constant[i_power_profile];
            let remainder_slope: f64 = linear_part_and_remainder.remainder_slope[i_power_profile];
            if remainder_constant == 0.0 && remainder_slope == 0.0 {
                continue;
            }
            let mut power_sum: LatticeSeries = series_zero(&lattice, &zero_half_widths);
            for i_previous in 0..i_order {
                // The constant terms (n = 0): [(1 - psi_N)**0]_l = delta_l0 enters with k_max,(l-1)**2 only
                if i_power_profile == 0 && i_previous < i_order - 1 {
                    continue;
                }
                series_add_assign(
                    &mut power_sum,
                    &power_cache[i_power_profile][i_order - 1 - i_previous],
                    k_max_squared_orders[i_previous],
                );
            }
            let term: LatticeSeries = series_multiply_linear(&lattice, &power_sum, -remainder_constant, -remainder_slope);
            series_add_assign(&mut source, &term, 1.0);
        }
        let source_solution: ParticularSolution = particular_solution(&lattice, &mut operator, &source);
        dropped_source = dropped_source.max(source_solution.dropped_size / source_size_0);
        let source_series: LatticeSeries = source_solution.series;
        let source_conditions: Array1<f64> = condition_residuals(&lattice, &source_series, conditions);
        let source_interior: Array1<f64> = series_values(&lattice, &source_series, interior_r, interior_z);
        let mut source_data: Array1<f64> = Array1::from_elem(n_least_squares + n_interior + n_exact, f64::NAN);
        for i_least_squares in 0..n_least_squares {
            let i_condition: usize = least_squares_rows[i_least_squares];
            source_data[i_least_squares] = row_weights[i_condition] * source_conditions[i_condition];
        }
        for i_interior in 0..n_interior {
            source_data[n_least_squares + i_interior] = interior_penalty * source_interior[i_interior];
        }
        for i_exact in 0..n_exact {
            source_data[n_least_squares + n_interior + i_exact] = source_conditions[exact_rows[i_exact]];
        }
        let fitted: Array1<f64> = fit.dot(&source_data).mapv(|value: f64| -value);
        let k_max_squared_order: f64 = fitted[n_free];
        let mut order_series: LatticeSeries = series_add(&source_series, &k_max_squared_series, k_max_squared_order);
        for i_free in 0..n_free {
            series_add_assign(&mut order_series, &free_terms[i_free], fitted[i_free]);
        }

        // The paper's third condition, order by order: (1 - psi_N) = 1 and its gradient zero at the moving axis, whose shift from order
        // 0's is sum_l lambda**l (axis_radial_orders[l], axis_z_orders[l]). The value condition fixes the multiple of (1 - psi_N)_0
        // added (the shift drops out, since the gradient of (1 - psi_N)_0 is zero at its axis), and the gradient conditions are the
        // Hessian of (1 - psi_N)_0 times the shift plus what is known
        let mut taylor_order: Array2<f64> = series_taylor(&lattice, &order_series, axis_0.radial, axis_0.z, n_degree);
        let mut shift_radial: Array1<f64> = Array1::zeros(i_order + 1);
        let mut shift_z: Array1<f64> = Array1::zeros(i_order + 1);
        for i_previous in 1..i_order {
            shift_radial[i_previous] = axis_radial_orders[i_previous];
            shift_z[i_previous] = axis_z_orders[i_previous];
        }
        let powers: Array3<f64> = displacement_powers(&shift_radial, &shift_z, n_degree);
        let mut known_value: f64 = 0.0;
        for i_previous in 0..i_order {
            let mut total: f64 = 0.0;
            for i_a in 0..n_degree {
                for i_b in 0..n_degree {
                    total += taylors[i_previous][[i_a, i_b]] * powers[[i_a, i_b, i_order - i_previous]];
                }
            }
            known_value += total;
        }
        let axis_multiple: f64 = -(taylor_order[[0, 0]] + known_value);
        series_add_assign(&mut order_series, &order_0, axis_multiple);
        taylor_order = &taylor_order + &taylors[0].mapv(|value: f64| axis_multiple * value);
        taylors.push(taylor_order);
        let mut known_gradient: [f64; 2] = [0.0, 0.0];
        for i_previous in 0..=i_order {
            let mut total_radial: f64 = 0.0;
            let mut total_z: f64 = 0.0;
            for i_a in 0..n_degree - 1 {
                for i_b in 0..n_degree {
                    // d/d(shift_radial) of shift_radial**(a + 1) shift_z**b, against the powers of shift_radial**a shift_z**b
                    total_radial += ((i_a + 1) as f64) * taylors[i_previous][[i_a + 1, i_b]] * powers[[i_a, i_b, i_order - i_previous]];
                }
            }
            for i_a in 0..n_degree {
                for i_b in 0..n_degree - 1 {
                    total_z += ((i_b + 1) as f64) * taylors[i_previous][[i_a, i_b + 1]] * powers[[i_a, i_b, i_order - i_previous]];
                }
            }
            known_gradient[0] += total_radial;
            known_gradient[1] += total_z;
        }
        let displacement: [f64; 2] = solve_2x2(hessian, known_gradient);
        let axis_radial_order: f64 = -displacement[0];
        let axis_z_order: f64 = -displacement[1];

        let order_trimmed: LatticeSeries = series_trim(&order_series, tolerance);
        let size_order: f64 = series_size(&order_trimmed);
        n_wavenumbers.push(series_count(&order_trimmed));
        if options.verbose {
            println!(
                "order {i_order:2}: k_max_squared = {k_max_squared_order:.6e} / m**2, multiple of order 0 = {axis_multiple:.3e}, axis shift = ({axis_radial_order:.3e}, {axis_z_order:.3e} m), n_wavenumber = {}, half_widths = {:?}, size = {size_order:.2e}",
                n_wavenumbers[n_wavenumbers.len() - 1],
                order_trimmed.half_widths
            );
        }
        one_minus_psi_n_orders.push(order_trimmed);
        k_max_squared_orders.push(k_max_squared_order);
        axis_radial_orders.push(axis_radial_order);
        axis_z_orders.push(axis_z_order);
        if options.taylor_tolerance > 0.0
            && size_order < options.taylor_tolerance * size_0
            && k_max_squared_order.abs() < options.taylor_tolerance * k_max_squared_orders[0]
        {
            break;
        }
        // A diverging series (coefficient sums SERIES_SIZE_MAX times those of order 0) is not continued
        if !size_order.is_finite() || size_order > SERIES_SIZE_MAX * size_0 {
            break;
        }

        // The power cache: [(1 - psi_N)**n]_l = sum_i [(1 - psi_N)**(n - 1)]_i (1 - psi_N)_{l-i}
        if i_order + 1 < n_taylor_max {
            if n_power_max >= 1 {
                power_cache[1].push(one_minus_psi_n_orders[i_order].clone());
            }
            for i_power_profile in 2..=n_power_max {
                let mut total: LatticeSeries = series_zero(&lattice, &zero_half_widths);
                for i_split in 0..=i_order {
                    let product: LatticeSeries =
                        series_product(&lattice, &power_cache[i_power_profile - 1][i_split], &one_minus_psi_n_orders[i_order - i_split]);
                    series_add_assign(&mut total, &product, 1.0);
                }
                power_cache[i_power_profile].push(series_trim(&total, tolerance));
            }
        }
    }

    Ok(PerturbationSeries {
        lattice,
        operator,
        one_minus_psi_n_orders,
        k_max_squared_orders,
        axis_radial_orders,
        axis_z_orders,
        n_wavenumbers,
        dropped_source,
        interior_penalty,
        interior_penalty_choice,
    })
}
