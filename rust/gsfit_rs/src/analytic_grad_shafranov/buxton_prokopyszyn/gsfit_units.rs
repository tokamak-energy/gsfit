//! Conversion between the coefficients of the source functions and the profiles the Grad-Shafranov equation is solved with.
//!
//! GSFit (IMAS) uses the total poloidal flux `psi = 2 pi R A_phi` [weber], and its source functions are polynomials in
//! `psi_N = (psi_a - psi) / (psi_a - psi_b)`: the paper's EFIT polynomials with a constant term each (see `profiles`), with the
//! coefficients `[c_0_p_prime, c_1_p_prime, ...]` [pascal / weber] and `[c_0_ff_prime, c_1_ff_prime, ...]`
//! [tesla ** 2 * metre ** 2 / weber].
//!
//! The source functions' linear part (the paper's `c_bar`, `perturbation::CBarMethod`), `c_bar_p_prime (1 - psi_N)` and
//! `c_bar_ff_prime (1 - psi_N)`, gives `mu0 R**2 p' + ff'` the value `(mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime) (1 - psi_N)` at `R_c`,
//! whose bracket is that of the paper's `k_max**2`:
//!
//! ```text
//! k_max**2 = (2 pi)**2 (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime) / (psi_a - psi_b).
//! ```
//!
//! Both profiles are divided by the bracket, which leaves the profiles `P` and `Q` of `profiles`, whose linear parts add up to one, and
//! `psi_a - psi_b` follows from `k_max`. Multiplying every coefficient by one factor multiplies the bracket and `psi_a - psi_b` by it and
//! leaves `P`, `Q` and the flux surfaces unchanged: the plasma current scales with it. The bracket must not vanish, but its sign need
//! not be that of `mu0 R_c**2 c_bar_p_prime`: `p'` fixes the pressure only up to a constant.

use super::perturbation::CBarMethod;
use super::profiles::{ProfileShapes, profile_value};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// The profiles of the Grad-Shafranov equation in `(1 - psi_N)`, and the bracket the source functions were divided by
#[derive(Clone, Debug)]
pub struct NormalisedProfiles {
    /// `P` and `Q`: `mu0 R_c**2 p'` and `ff'` divided by `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime`
    pub shapes: ProfileShapes,
    /// `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime`, the linear part of `mu0 R**2 p' + ff'` at `R_c` per unit `(1 - psi_N)`
    /// [tesla ** 2 * metre ** 2 / weber]
    pub linear_source_at_r_c: f64,
}

/// The profiles `P` and `Q` of the paper's coefficients.
///
/// # Arguments
/// * `p_prime_coefficients` - `[c_0_p_prime, c_1_p_prime, ...]` [pascal / weber]
/// * `ff_prime_coefficients` - `[c_0_ff_prime, c_1_ff_prime, ...]` [tesla ** 2 * metre ** 2 / weber]
/// * `r_c_squared` - `R_c**2 = R_geo**2 + a_minor**2` [metre ** 2]
/// * `c_bar_method` - the choice of `c_bar`, the linear part of the source functions
///
/// # Errors
/// The linear part's current density at `R_c`, `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime`, vanishes or is not finite
pub fn normalised_profiles(
    p_prime_coefficients: &[f64],
    ff_prime_coefficients: &[f64],
    r_c_squared: f64,
    c_bar_method: CBarMethod,
) -> Result<NormalisedProfiles, String> {
    let p_prime_linear: f64 = MU_0 * r_c_squared * c_bar_method.linear_part(p_prime_coefficients);
    let ff_prime_linear: f64 = c_bar_method.linear_part(ff_prime_coefficients);
    let linear_source_at_r_c: f64 = p_prime_linear + ff_prime_linear;
    if !(linear_source_at_r_c != 0.0 && linear_source_at_r_c.is_finite()) {
        return Err(format!(
            "the linear part of the source functions (c_bar_method = {}) at R_c = {:.4} m, mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime = {linear_source_at_r_c:.6e} T**2 m**2 / Wb, is zero or not finite (mu0 R_c**2 c_bar_p_prime = {p_prime_linear:.6e}, c_bar_ff_prime = {ff_prime_linear:.6e})",
            c_bar_method.name(),
            r_c_squared.sqrt()
        ));
    }
    let n_p_prime: usize = p_prime_coefficients.len();
    let mut p_prime: Vec<f64> = Vec::with_capacity(n_p_prime);
    for i_p_prime in 0..n_p_prime {
        p_prime.push(MU_0 * r_c_squared * p_prime_coefficients[i_p_prime] / linear_source_at_r_c);
    }
    let n_ff_prime: usize = ff_prime_coefficients.len();
    let mut ff_prime: Vec<f64> = Vec::with_capacity(n_ff_prime);
    for i_ff_prime in 0..n_ff_prime {
        ff_prime.push(ff_prime_coefficients[i_ff_prime] / linear_source_at_r_c);
    }
    Ok(NormalisedProfiles {
        shapes: ProfileShapes { p_prime, ff_prime },
        linear_source_at_r_c,
    })
}

/// `psi_a - psi_b = (2 pi)**2 (mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime) / k_max**2`, the definition of `k_max` with `c_bar`.
///
/// # Arguments
/// * `k_max` - the paper's `k_max` [1 / metre]
/// * `linear_source_at_r_c` - `mu0 R_c**2 c_bar_p_prime + c_bar_ff_prime` [tesla ** 2 * metre ** 2 / weber]
///
/// # Returns
/// * `psi_a - psi_b`, with the sign of the bracket [weber]
pub fn psi_a_minus_psi_b_of(k_max: f64, linear_source_at_r_c: f64) -> f64 {
    (2.0 * PI).powi(2) * linear_source_at_r_c / (k_max * k_max)
}

/// Guazzotto and Freidberg's `nu` of the source functions on the magnetic axis: the share of the toroidal current density at `R_c`
/// carried by `p'`, with both at their values on the axis (`c_0 + c_1`), `mu0 R_c**2 p'(axis) / (mu0 R_c**2 p'(axis) + ff'(axis))`.
/// For their linear source functions it is their `nu`; it is 0 without `p'`, 1 without `ff'`, above 1 for a diamagnetic `ff'`, infinite
/// where the two cancel, and NaN when both vanish on the axis. An output only: the solution does not use it.
///
/// # Arguments
/// * `p_prime_coefficients` - `[c_0_p_prime, c_1_p_prime, ...]` [pascal / weber]
/// * `ff_prime_coefficients` - `[c_0_ff_prime, c_1_ff_prime, ...]` [tesla ** 2 * metre ** 2 / weber]
/// * `r_c_squared` - `R_c**2` [metre ** 2]
///
/// # Returns
/// * `nu` [dimensionless]
pub fn nu_on_axis(p_prime_coefficients: &[f64], ff_prime_coefficients: &[f64], r_c_squared: f64) -> f64 {
    let p_prime_on_axis: f64 = MU_0 * r_c_squared * profile_value(1.0, p_prime_coefficients);
    p_prime_on_axis / (p_prime_on_axis + profile_value(1.0, ff_prime_coefficients))
}
