//! The source functions: the paper's EFIT polynomials, with a constant term each, written in `(1 - psi_N)`.
//!
//! The source functions are
//!
//! ```text
//! p'(psi_N)  = c_0_p_prime  + sum_{i >= 1} c_i_p_prime  (1 - psi_N) psi_N**(i - 1)     [pascal / weber],
//! ff'(psi_N) = c_0_ff_prime + sum_{i >= 1} c_i_ff_prime (1 - psi_N) psi_N**(i - 1)     [tesla ** 2 * metre ** 2 / weber],
//! ```
//!
//! `c_0` being the constant (Solov'ev) term, the profile's value on the LCFS, `c_1` the coefficient of `1 - psi_N`, `c_2` that of
//! `(1 - psi_N) psi_N`, and so on. GSFit's `EfitPolynomial` basis is the terms `i >= 1`. As `psi_N = 1 - (1 - psi_N)`, one list
//! `[c_0, c_1, ...]` is a polynomial in `(1 - psi_N)`, whose value on the magnetic axis is `c_0 + c_1`.
//!
//! The Grad-Shafranov equation is solved for `(1 - psi_N)`, with `(2 pi)**2 / (psi_a - psi_b) = k_max**2 / (mu0 R_c**2 c_bar_p_prime +
//! c_bar_ff_prime)` (the definition of `k_max`, with `c_bar` in place of `c_1`):
//!
//! ```text
//! 4 R**2 d2(1 - psi_N)/d(R**2)**2 + d2(1 - psi_N)/dZ**2 = -k_max**2 [(R**2 / R_c**2) P(1 - psi_N) + Q(1 - psi_N)],
//! ```
//!
//! with the profiles `P = mu0 R_c**2 p' / (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime)` and `Q = ff' / (mu0 R_c**2 c_bar_p_prime +
//! c_bar_ff_prime)` (see `gsfit_units`), whose linear parts add up to one.

use super::numerics::binomial;
use ndarray::Array1;

/// The two profiles of the Grad-Shafranov equation in `(1 - psi_N)`, each as its coefficients `[c_0, c_1, ...]`
#[derive(Clone, Debug)]
pub struct ProfileShapes {
    /// `mu0 R_c**2 p'` divided by `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` [dimensionless]
    pub p_prime: Vec<f64>,
    /// `ff'` divided by `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` [dimensionless]
    pub ff_prime: Vec<f64>,
}

impl ProfileShapes {
    /// Whether either profile has a constant term, `c_0 != 0`: the current density then does not vanish on the LCFS
    pub fn has_constant_terms(&self) -> bool {
        self.p_prime[0] != 0.0 || self.ff_prime[0] != 0.0
    }
}

/// A profile's coefficients as a polynomial in `(1 - psi_N)`:
/// `c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1) = sum_n power_coefficients[n] (1 - psi_N)**n`.
///
/// # Arguments
/// * `coefficients` - `[c_0, c_1, ...]`, at least one [any unit]
///
/// # Returns
/// * `n = 0..max(n_dof, 1)` for `n_dof = coefficients.len() - 1` EFIT terms: `[c_0, sum_{i >= 1} c_i, ...]`, the value on the LCFS and
///   the slope there, then the higher powers [same unit]
pub fn power_coefficients(coefficients: &[f64]) -> Array1<f64> {
    let n_dof: usize = coefficients.len() - 1;
    let mut power: Array1<f64> = Array1::zeros(n_dof.max(1) + 1);
    power[0] = coefficients[0];
    // c_i (1 - psi_N) psi_N**(i - 1) = c_i sum_n binom(i - 1, n) (-1)**n (1 - psi_N)**(n + 1)
    for i_dof in 0..n_dof {
        for i_power in 0..=i_dof {
            let sign: f64 = if i_power % 2 == 0 { 1.0 } else { -1.0 };
            power[i_power + 1] += coefficients[i_dof + 1] * binomial(i_dof, i_power) * sign;
        }
    }
    power
}

/// # Arguments
/// * `one_minus_psi_n` - `(1 - psi_N)`, 1 on the magnetic axis, 0 on the LCFS [dimensionless]
/// * `coefficients` - `[c_0, c_1, ...]` [any unit]
///
/// # Returns
/// * `c_0 + sum_{i >= 1} c_i (1 - psi_N) psi_N**(i - 1)` [same unit]
pub fn profile_value(one_minus_psi_n: f64, coefficients: &[f64]) -> f64 {
    let n_coefficient: usize = coefficients.len();
    let mut value: f64 = coefficients[0];
    for i_coefficient in 1..n_coefficient {
        value += coefficients[i_coefficient] * one_minus_psi_n * (1.0 - one_minus_psi_n).powi(i_coefficient as i32 - 1);
    }
    value
}

/// Integral of a profile from the LCFS, `int_0^(1 - psi_N) profile d(1 - psi_N)`, which includes `c_0 (1 - psi_N)`. With the physical
/// coefficients, `p = (psi_a - psi_b) int_0^(1 - psi_N) p' d(1 - psi_N)` and
/// `f**2 = f_vacuum**2 + 2 (psi_a - psi_b) int_0^(1 - psi_N) ff' d(1 - psi_N)`.
///
/// # Arguments
/// * `one_minus_psi_n` - `(1 - psi_N)` [dimensionless]
/// * `coefficients` - `[c_0, c_1, ...]` [any unit]
///
/// # Returns
/// * the integral [same unit]
pub fn profile_integral(one_minus_psi_n: f64, coefficients: &[f64]) -> f64 {
    let power: Array1<f64> = power_coefficients(coefficients);
    let n_power: usize = power.len();
    let mut integral: f64 = 0.0;
    for i_power in 0..n_power {
        integral += power[i_power] * one_minus_psi_n.powi(i_power as i32 + 1) / (i_power as f64 + 1.0);
    }
    integral
}

/// The right-hand side of the Grad-Shafranov equation in `(1 - psi_N)`, divided by `-k_max**2`: `(R**2 / R_c**2) P + Q`.
///
/// # Arguments
/// * `r` - major radius [metre]
/// * `one_minus_psi_n` - `(1 - psi_N)` [dimensionless]
/// * `r_c_squared` - `R_c**2 = R_geo**2 + a_minor**2` [metre ** 2]
/// * `shapes` - the profiles
pub fn source_term(r: f64, one_minus_psi_n: f64, r_c_squared: f64, shapes: &ProfileShapes) -> f64 {
    (r * r / r_c_squared) * profile_value(one_minus_psi_n, &shapes.p_prime) + profile_value(one_minus_psi_n, &shapes.ff_prime)
}

/// Residual of the Grad-Shafranov equation in `(1 - psi_N)`,
/// `d2(1 - psi_N)/dR**2 - d(1 - psi_N)/dR / R + d2(1 - psi_N)/dZ**2 + k_max**2 [(R**2 / R_c**2) P + Q]`, which is
/// `4 R**2 d2(1 - psi_N)/d(R**2)**2 + d2(1 - psi_N)/dZ**2` plus the right-hand side; zero for an exact solution.
///
/// # Arguments
/// * `r` - major radius [metre]
/// * `one_minus_psi_n` - `(1 - psi_N)` [dimensionless]
/// * `d_d_r` - `d(1 - psi_N)/dR` [1 / metre]
/// * `d2_d_r2` - `d2(1 - psi_N)/dR**2` [1 / metre ** 2]
/// * `d2_d_z2` - `d2(1 - psi_N)/dZ**2` [1 / metre ** 2]
/// * `k_max` - the paper's `k_max` [1 / metre]
/// * `r_c_squared` - `R_c**2` [metre ** 2]
/// * `shapes` - the profiles
///
/// # Returns
/// * the residual [1 / metre ** 2]
#[allow(clippy::too_many_arguments)]
pub fn grad_shafranov_residual(
    r: f64,
    one_minus_psi_n: f64,
    d_d_r: f64,
    d2_d_r2: f64,
    d2_d_z2: f64,
    k_max: f64,
    r_c_squared: f64,
    shapes: &ProfileShapes,
) -> f64 {
    d2_d_r2 - d_d_r / r + d2_d_z2 + k_max * k_max * source_term(r, one_minus_psi_n, r_c_squared, shapes)
}

/// Whether both profiles are "broad": each one's slope at the LCFS, `sum_{i >= 1} c_i`, is at least its value on the axis, `c_0 + c_1`,
/// in units of that value. A profile that vanishes on the axis (`p' = 0` everywhere, for one) counts as broad, as the value on the axis
/// gives it no linear part. Broad profiles take `c_bar` from the slope at the boundary; the others from the value on the magnetic axis
/// (see `perturbation::CBarMethod`).
///
/// # Arguments
/// * `p_prime_coefficients` - `[c_0_p_prime, c_1_p_prime, ...]` [pascal / weber]
/// * `ff_prime_coefficients` - `[c_0_ff_prime, c_1_ff_prime, ...]` [tesla ** 2 * metre ** 2 / weber]
pub fn is_broad_profile(p_prime_coefficients: &[f64], ff_prime_coefficients: &[f64]) -> bool {
    let mut is_broad: bool = true;
    for coefficients in [p_prime_coefficients, ff_prime_coefficients] {
        let value_on_axis: f64 = profile_value(1.0, coefficients);
        let slope_at_lcfs: f64 = power_coefficients(coefficients)[1];
        is_broad &= value_on_axis == 0.0 || slope_at_lcfs / value_on_axis >= 1.0;
    }
    is_broad
}
