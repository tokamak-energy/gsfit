//! The series which the flux from conductors of finite size, `finite_size_to_point.rs`, and the mutual inductance between them,
//! `finite_size_to_finite_size.rs`, share: the checks of the conductors, how small each series' expansion parameter has to be, the
//! coefficients of the Green's function of a filament expanded about the point, the two combinations of the elliptic integrals which the
//! multipole series starts from, and the moments of a parallelogram.
//!
//! The comments use the symbols, and the section and equation numbers, of the note
//! `documentation/flux_from_parallelogram_conductor/note/flux_from_parallelogram_conductor.pdf`.

use crate::settings::settings_equilibrium::SettingsEquilibriumNumericsGreensFiniteSize;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// The two sides of a parallelogram are taken as parallel, so that it has no area, when the sine of the angle between them is below this.
const PARALLEL_SIDES_TOLERANCE: f64 = 1.0e-12;

/// Check that conductors can be averaged over: each is described by one value in each of the six arrays, has an area, and is wholly at
/// a positive major radius, which the Green's function of a filament needs.
///
/// # Arguments
/// * `oblique_r` - R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_z` - Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_length_alpha` - Length of each conductor's side which is inclined at `oblique_alpha`, shape = [n_conductor], [metre]
/// * `oblique_length_beta` - Length of each conductor's side which is inclined at `oblique_beta`, shape = [n_conductor], [metre]
/// * `oblique_alpha` - Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
/// * `oblique_beta` - Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
/// * `name_suffix` - What the names of the six arrays end with where this is called from, `""` or `"_prime"`: the errors name the arrays
///
/// # Errors
/// * the six arrays do not have the same length
/// * a conductor has no area: a side without a positive length, or two sides which are parallel
/// * a conductor is not wholly at a positive major radius
pub(super) fn check_conductors(
    oblique_r: &Array1<f64>,
    oblique_z: &Array1<f64>,
    oblique_length_alpha: &Array1<f64>,
    oblique_length_beta: &Array1<f64>,
    oblique_alpha: &Array1<f64>,
    oblique_beta: &Array1<f64>,
    name_suffix: &str,
) -> Result<(), String> {
    let n_conductor: usize = oblique_r.len();
    if oblique_z.len() != n_conductor
        || oblique_length_alpha.len() != n_conductor
        || oblique_length_beta.len() != n_conductor
        || oblique_alpha.len() != n_conductor
        || oblique_beta.len() != n_conductor
    {
        return Err(format!(
            "`oblique_r{name_suffix}`, `oblique_z{name_suffix}`, `oblique_length_alpha{name_suffix}`, `oblique_length_beta{name_suffix}`, \
             `oblique_alpha{name_suffix}` and `oblique_beta{name_suffix}` have {}, {}, {}, {}, {} and {} conductor(s); each conductor needs one of each",
            n_conductor,
            oblique_z.len(),
            oblique_length_alpha.len(),
            oblique_length_beta.len(),
            oblique_alpha.len(),
            oblique_beta.len()
        ));
    }

    for i_conductor in 0..n_conductor {
        // The directions of the two sides which meet at the reference corner
        let side_alpha_unit_r: f64 = oblique_alpha[i_conductor].cos();
        let side_alpha_unit_z: f64 = oblique_alpha[i_conductor].sin();
        let side_beta_unit_r: f64 = -oblique_beta[i_conductor].sin();
        let side_beta_unit_z: f64 = oblique_beta[i_conductor].cos();

        // A parallelogram with an area has two sides of positive length which are not parallel
        let sine_between_sides: f64 = (side_alpha_unit_r * side_beta_unit_z - side_alpha_unit_z * side_beta_unit_r).abs();
        let has_area: bool = oblique_length_alpha[i_conductor] > 0.0
            && oblique_length_alpha[i_conductor].is_finite()
            && oblique_length_beta[i_conductor] > 0.0
            && oblique_length_beta[i_conductor].is_finite()
            && sine_between_sides >= PARALLEL_SIDES_TOLERANCE;
        if !has_area {
            return Err(format!(
                "conductor {i_conductor} has no area: `oblique_length_alpha{name_suffix}` = {} metre, `oblique_length_beta{name_suffix}` = {} metre, \
                 `oblique_alpha{name_suffix}` = {} radian and `oblique_beta{name_suffix}` = {} radian; the lengths must be positive, and the two sides \
                 not parallel",
                oblique_length_alpha[i_conductor], oblique_length_beta[i_conductor], oblique_alpha[i_conductor], oblique_beta[i_conductor]
            ));
        }

        // The innermost of the four corners
        let corner_r_min: f64 = oblique_r[i_conductor]
            + (oblique_length_alpha[i_conductor] * side_alpha_unit_r).min(0.0)
            + (oblique_length_beta[i_conductor] * side_beta_unit_r).min(0.0);
        let is_usable_position: bool = corner_r_min > 0.0 && corner_r_min.is_finite() && oblique_z[i_conductor].is_finite();
        if !is_usable_position {
            return Err(format!(
                "conductor {i_conductor} has its reference corner at `oblique_r{name_suffix}` = {} and `oblique_z{name_suffix}` = {} metre, and its \
                 innermost corner at r = {corner_r_min} metre; the whole conductor must be at a positive major radius",
                oblique_r[i_conductor], oblique_z[i_conductor]
            ));
        }
    }

    Ok(())
}

/// What `expansion_parameter_thresholds` returns: how small each series' expansion parameter has to be before the series is used.
pub(super) struct ExpansionParameterThresholds {
    /// The largest `half_diagonal / min(distance, centre_r)` at which the far-field series is used, [dimensionless]
    pub(super) far_field_ratio_max: f64,
    /// The largest `t / (1 - t)`, with `t = (distance + half_diagonal) / (2 * r)`, at which the near-field series is used, [dimensionless]
    pub(super) near_field_ratio_max: f64,
}

/// How small each series' expansion parameter has to be for its first neglected term to be below the relative tolerance (section 5 of the
/// note).
///
/// The far-field series' first neglected term is of the order `n_multipole + 2`, with `n_multipole` rounded down to an even number, because
/// the odd orders add nothing. The near-field series has two truncations, whose first neglected terms are of the orders `n_thin_ring + 1`
/// and `2 * n_elliptic + 2`. Neither threshold is above 1/2.
///
/// # Arguments
/// * `settings_greens_finite_size` - The orders of the two series, and the relative accuracy wanted
///
/// # Returns
/// * `thresholds` - `far_field_ratio_max` and `near_field_ratio_max`, [dimensionless]
///
/// # Errors
/// * the relative tolerance is not greater than 0 and at most 1
pub(super) fn expansion_parameter_thresholds(
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<ExpansionParameterThresholds, String> {
    // A tolerance which is negative, above 1, or not a number would otherwise give the largest thresholds, 1/2, without any warning
    let relative_tolerance: f64 = settings_greens_finite_size.relative_tolerance;
    let is_fraction: bool = relative_tolerance > 0.0 && relative_tolerance <= 1.0;
    if !is_fraction {
        return Err(format!("`relative_tolerance` = {relative_tolerance}; it must be greater than 0 and at most 1"));
    }

    let n_multipole_even: usize = 2 * (settings_greens_finite_size.n_multipole / 2);
    let far_field_ratio_max: f64 = relative_tolerance.powf(1.0 / ((n_multipole_even + 2) as f64)).min(0.5);
    let near_field_ratio_max: f64 = relative_tolerance
        .powf(1.0 / ((settings_greens_finite_size.n_thin_ring + 1) as f64))
        .min(relative_tolerance.powf(1.0 / ((2 * settings_greens_finite_size.n_elliptic + 2) as f64)))
        .min(0.5);

    Ok(ExpansionParameterThresholds {
        far_field_ratio_max,
        near_field_ratio_max,
    })
}

/// What `maxwell_series_coefficients` returns: the two tables of coefficients of the Green's function of a filament expanded about the
/// point. Both are symmetric, and are zero where the truncation keeps no term.
pub(super) struct MaxwellSeriesCoefficients {
    /// The coefficients of the terms with the logarithm, `P_ab` in the note, shape = [degree_max + 1, degree_max + 1], [dimensionless]
    pub(super) table_p: Array2<f64>,
    /// The coefficients of the terms without the logarithm, `Q_ab` in the note, shape = [degree_max + 1, degree_max + 1], [dimensionless]
    pub(super) table_q: Array2<f64>,
}

/// The coefficients of the Green's function of a filament expanded about the point: the tables `P_ab` and `Q_ab` of the note, Eq. (35) to
/// (37).
///
/// With `zeta = ((r' - r) + i * (z' - z)) / (2 * r)`, the position of the source `(r', z')` from the point `(r, z)` in units of `2 * r`:
///
/// ```text
/// G / (MU_0 * r) = sum over i_a, i_b of  zeta ** i_a * conj(zeta) ** i_b * (table_p[(i_a, i_b)] * ln(4 / |zeta|) + table_q[(i_a, i_b)])
/// ```
///
/// These are pure numbers, the same for every conductor and every point. The lowest orders are Maxwell's series for the mutual
/// inductance of two nearby coaxial circles, Eq. (38) of the note.
///
/// # Arguments
/// * `n_elliptic` - The highest power of `1 - k ** 2` kept in the series of the elliptic integrals, Eq. (23) of the note
/// * `n_thin_ring` - The highest order kept in the expansion about the thin ring, Eq. (31) and (32) of the note
///
/// # Returns
/// * `coefficients` - The two tables, `table_p` and `table_q`, each of shape = [degree_max + 1, degree_max + 1] with
///   `degree_max = 2 * n_elliptic + n_thin_ring`, [dimensionless]
pub(super) fn maxwell_series_coefficients(n_elliptic: usize, n_thin_ring: usize) -> MaxwellSeriesCoefficients {
    let degree_max: usize = 2 * n_elliptic + n_thin_ring;
    // The terms are added up, so the tables start from zero, which is what is left where the truncation keeps no term
    let mut table_p: Array2<f64> = Array2::zeros((degree_max + 1, degree_max + 1));
    let mut table_q: Array2<f64> = Array2::zeros((degree_max + 1, degree_max + 1));

    // The series of the elliptic integrals, Eq. (23) to (25): (2 - k ** 2) * K - 2 * E = sum over n of chi_n * q ** n * (ln(4 / sqrt(q)) - gamma_n)
    let mut chi: f64 = 1.0;
    let mut gamma: f64 = 2.0;
    let mut c_previous: f64 = 1.0; // c_(n - 1) = ((2 * n - 2)! / (2 ** (2 * n - 2) * (n - 1)! ** 2)) ** 2
    let mut kappa_previous: f64 = 0.0; // kappa_(n - 1) = sum over j = 1 .. n - 1 of 1 / (j * (2 * j - 1))
    let mut binomial: Array1<f64> = Array1::from_elem(n_thin_ring + 1, f64::NAN);
    let mut binomial_derivative: Array1<f64> = Array1::from_elem(n_thin_ring + 1, f64::NAN);
    for i_elliptic in 0..n_elliptic + 1 {
        let order_elliptic: f64 = i_elliptic as f64; // `n` in the note
        if i_elliptic >= 1 {
            chi = c_previous / (4.0 * order_elliptic.powi(2));
            gamma = kappa_previous - 1.0 / order_elliptic;
            c_previous *= ((2.0 * order_elliptic - 1.0) / (2.0 * order_elliptic)).powi(2);
            kappa_previous += 1.0 / (order_elliptic * (2.0 * order_elliptic - 1.0));
        }

        // The expansion about the thin ring, Eq. (31) to (34): |1 + zeta| ** (1 - 2 * n) = sum over j, l of binomial[j] * binomial[l] *
        // zeta ** j * conj(zeta) ** l, and the same with the derivative of the binomial coefficient with respect to its upper argument,
        // `nu`, for the factor ln|1 + zeta|
        let nu: f64 = 0.5 - order_elliptic;
        binomial[0] = 1.0;
        binomial_derivative[0] = 0.0;
        for i_thin_ring in 1..n_thin_ring + 1 {
            let order_thin_ring: f64 = i_thin_ring as f64; // `j` in the note
            binomial[i_thin_ring] = binomial[i_thin_ring - 1] * (nu - order_thin_ring + 1.0) / order_thin_ring;
            binomial_derivative[i_thin_ring] =
                (binomial_derivative[i_thin_ring - 1] * (nu - order_thin_ring + 1.0) + binomial[i_thin_ring - 1]) / order_thin_ring;
        }

        // The term in `zeta ** i_a * conj(zeta) ** i_b`, Eq. (36) and (37)
        for i_thin_ring in 0..n_thin_ring + 1 {
            for j_thin_ring in 0..n_thin_ring + 1 - i_thin_ring {
                let i_a: usize = i_elliptic + i_thin_ring;
                let i_b: usize = i_elliptic + j_thin_ring;
                table_p[(i_a, i_b)] += chi * binomial[i_thin_ring] * binomial[j_thin_ring];
                table_q[(i_a, i_b)] += chi
                    * (-gamma * binomial[i_thin_ring] * binomial[j_thin_ring]
                        + 0.5 * (binomial_derivative[i_thin_ring] * binomial[j_thin_ring] + binomial[i_thin_ring] * binomial_derivative[j_thin_ring]));
            }
        }
    }

    MaxwellSeriesCoefficients { table_p, table_q }
}

/// What `elliptic_phi_theta` returns: two combinations of the complete elliptic integrals `K` and `E` of the parameter `k_sq`.
pub(super) struct EllipticPhiTheta {
    /// `(2 - k_sq) * K - 2 * E`, `Phi` in the note, [dimensionless]
    pub(super) phi: f64,
    /// `(2 - k_sq) * E - 2 * (1 - k_sq) * K`, `Theta` in the note, [dimensionless]
    pub(super) theta: f64,
}

/// The two combinations of the complete elliptic integrals which the flux of a filament and its vertical derivative need, Eq. (72) to (75)
/// of the note: `phi = (2 - k_sq) * K - 2 * E` and `theta = (2 - k_sq) * E - 2 * (1 - k_sq) * K`.
///
/// Both are of the order `k_sq ** 2` when `k_sq` is small, where forming them from `K` and `E` loses accuracy. They are calculated from
/// the arithmetic-geometric mean (appendix B of the note), with `phi` as a sum of positive terms, which keeps their relative accuracy at
/// every `k_sq`.
///
/// # Arguments
/// * `k_sq` - The parameter of the elliptic integrals, `k ** 2`, between 0 and 1, [dimensionless]
/// * `q` - The complementary parameter, `1 - k ** 2`, given separately so that it keeps its accuracy when `k ** 2` is close to 1,
///   [dimensionless]
///
/// # Returns
/// * `phi_theta` - `phi` and `theta`, [dimensionless]. At `q = 0`, which is the point on the filament, `phi` is infinite
pub(super) fn elliptic_phi_theta(k_sq: f64, q: f64) -> EllipticPhiTheta {
    if q == 0.0 {
        return EllipticPhiTheta {
            phi: f64::INFINITY,
            theta: 1.0,
        };
    }

    // x_n, y_n and z_n of Eq. (181) to (183), and S = sum over n >= 1 of 2 ** n * z_n ** 2, Eq. (185)
    let mut mean_arithmetic: f64 = 1.0;
    let mut mean_geometric: f64 = q.sqrt();
    let mut mean_half_difference: f64 = k_sq.sqrt();
    let mut sum_s: f64 = 0.0;
    let mut power_of_two: f64 = 1.0;

    // The arithmetic-geometric mean converges quadratically: 8 iterations are enough for `q > 1e-20`
    let n_iteration_max: usize = 40;
    'iteration_loop: for _i_iteration in 0..n_iteration_max {
        let mean_arithmetic_next: f64 = 0.5 * (mean_arithmetic + mean_geometric);
        mean_geometric = (mean_arithmetic * mean_geometric).sqrt();
        mean_half_difference = mean_half_difference.powi(2) / (4.0 * mean_arithmetic_next); // the same as `(x_n - y_n) / 2`, without the cancellation
        mean_arithmetic = mean_arithmetic_next;
        power_of_two *= 2.0;
        sum_s += power_of_two * mean_half_difference.powi(2);
        if mean_half_difference <= 1.0e-17 * mean_arithmetic {
            break 'iteration_loop;
        }
    }

    // Eq. (184), (186) and (187)
    let elliptic_k: f64 = PI / (2.0 * mean_arithmetic);
    let phi: f64 = elliptic_k * sum_s;
    let theta: f64 = 0.5 * elliptic_k * (k_sq.powi(2) - (2.0 - k_sq) * sum_s);

    EllipticPhiTheta { phi, theta }
}

/// The moments of a parallelogram about its centre, Eq. (71) of the note: the averages over the parallelogram of
/// `delta_r ** i * delta_z ** j`, where `(delta_r, delta_z)` is the position of a point of the parallelogram from its centre.
///
/// A point of the parallelogram is `lambda_alpha * side_alpha + lambda_beta * side_beta`, with `lambda_alpha` and `lambda_beta` uniformly
/// distributed between -1/2 and 1/2. The moments with `i + j` odd are zero, because a parallelogram is symmetric about its centre.
///
/// # Arguments
/// * `side_alpha_r` - R component of the side which is inclined at `oblique_alpha`, [metre]
/// * `side_alpha_z` - Z component of the side which is inclined at `oblique_alpha`, [metre]
/// * `side_beta_r` - R component of the side which is inclined at `oblique_beta`, [metre]
/// * `side_beta_z` - Z component of the side which is inclined at `oblique_beta`, [metre]
/// * `n_multipole` - The highest order, `i + j`, of the moments
///
/// # Returns
/// * `moment` - `moment[(i, j)]` is the average of `delta_r ** i * delta_z ** j`, for `i + j <= n_multipole`; it is NaN above that,
///   shape = [n_multipole + 1, n_multipole + 1], [metre ** (i + j)]
pub(super) fn parallelogram_moments(side_alpha_r: f64, side_alpha_z: f64, side_beta_r: f64, side_beta_z: f64, n_multipole: usize) -> Array2<f64> {
    let n_series: usize = n_multipole + 1;

    // The average of `lambda_alpha ** m`, Eq. (70), which is the same for `lambda_beta`, and the powers of the sides' components
    let mut moment_one_side: Array1<f64> = Array1::from_elem(n_series, f64::NAN);
    let mut side_alpha_r_power: Array1<f64> = Array1::from_elem(n_series, f64::NAN);
    let mut side_alpha_z_power: Array1<f64> = Array1::from_elem(n_series, f64::NAN);
    let mut side_beta_r_power: Array1<f64> = Array1::from_elem(n_series, f64::NAN);
    let mut side_beta_z_power: Array1<f64> = Array1::from_elem(n_series, f64::NAN);
    for i_power in 0..n_series {
        moment_one_side[i_power] = if i_power % 2 == 0 {
            0.5_f64.powi(i_power as i32) / ((i_power + 1) as f64)
        } else {
            0.0
        };
        side_alpha_r_power[i_power] = side_alpha_r.powi(i_power as i32);
        side_alpha_z_power[i_power] = side_alpha_z.powi(i_power as i32);
        side_beta_r_power[i_power] = side_beta_r.powi(i_power as i32);
        side_beta_z_power[i_power] = side_beta_z.powi(i_power as i32);
    }

    let mut moment: Array2<f64> = Array2::from_elem((n_series, n_series), f64::NAN);
    for i_series in 0..n_series {
        for j_series in 0..n_series - i_series {
            if (i_series + j_series) % 2 == 1 {
                moment[(i_series, j_series)] = 0.0;
            } else {
                // `delta_r ** i_series * delta_z ** j_series`, with `delta_r = lambda_alpha * side_alpha_r + lambda_beta * side_beta_r` and
                // `delta_z = lambda_alpha * side_alpha_z + lambda_beta * side_beta_z`, expanded with the binomial theorem and averaged
                let mut moment_sum: f64 = 0.0;
                let mut binomial_i: f64 = 1.0; // binomial(i_series, i_power)
                for i_power in 0..i_series + 1 {
                    let mut binomial_j: f64 = 1.0; // binomial(j_series, j_power)
                    for j_power in 0..j_series + 1 {
                        moment_sum += binomial_i
                            * binomial_j
                            * side_alpha_r_power[i_power]
                            * side_beta_r_power[i_series - i_power]
                            * side_alpha_z_power[j_power]
                            * side_beta_z_power[j_series - j_power]
                            * moment_one_side[i_power + j_power]
                            * moment_one_side[i_series + j_series - i_power - j_power];
                        binomial_j = binomial_j * ((j_series - j_power) as f64) / ((j_power + 1) as f64);
                    }
                    binomial_i = binomial_i * ((i_series - i_power) as f64) / ((i_power + 1) as f64);
                }
                moment[(i_series, j_series)] = moment_sum;
            }
        }
    }

    moment
}

#[cfg(test)]
mod tests {
    use super::*;
    use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
    use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind

    #[test]
    fn the_tables_of_coefficients_start_with_maxwells_series() {
        // With these orders every term up to the third degree is kept
        let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(2, 3);
        let table_p: &Array2<f64> = &coefficients.table_p;
        let table_q: &Array2<f64> = &coefficients.table_q;
        let degree_max: usize = 7;
        assert_eq!(table_p.dim(), (degree_max + 1, degree_max + 1));
        assert_eq!(table_q.dim(), (degree_max + 1, degree_max + 1));

        // Eq. (38) of the note, written in `zeta` and its complex conjugate
        assert_eq!(table_p[(0, 0)], 1.0);
        assert_eq!(table_p[(1, 0)], 1.0 / 2.0);
        assert_eq!(table_p[(2, 0)], -1.0 / 8.0);
        assert_eq!(table_p[(1, 1)], 1.0 / 2.0);
        assert_eq!(table_p[(3, 0)], 1.0 / 16.0);
        assert_eq!(table_p[(2, 1)], -3.0 / 16.0);
        assert_eq!(table_q[(0, 0)], -2.0);
        assert_eq!(table_q[(1, 0)], -1.0 / 2.0);
        assert_eq!(table_q[(2, 0)], 1.0 / 4.0);
        assert_eq!(table_q[(1, 1)], 1.0 / 4.0);
        assert!((table_q[(3, 0)] + 7.0 / 48.0).abs() < 1.0e-16);
        assert_eq!(table_q[(2, 1)], 1.0 / 16.0);

        // Both tables are symmetric, and zero where the truncation keeps no term: `i_a` and `i_b` both above `n_elliptic`, with their sum
        // above `2 * n_elliptic + n_thin_ring`, or their difference above `n_thin_ring`
        for i_a in 0..degree_max + 1 {
            for i_b in 0..degree_max + 1 {
                assert_eq!(table_p[(i_a, i_b)], table_p[(i_b, i_a)]);
                assert_eq!(table_q[(i_a, i_b)], table_q[(i_b, i_a)]);
            }
        }
        assert_eq!(table_p[(4, 0)], 0.0);
        assert_eq!(table_q[(4, 0)], 0.0);
        assert_eq!(table_p[(4, 4)], 0.0);
        assert_eq!(table_q[(4, 4)], 0.0);
        assert!(table_p[(5, 2)] != 0.0);
        assert!(table_q[(5, 2)] != 0.0);
    }

    #[test]
    fn phi_and_theta_are_the_two_combinations_of_the_elliptic_integrals() {
        // Where `k_sq` is not small, forming them from `K` and `E` is accurate
        let n_k_sq: usize = 5;
        let k_sq_all: [f64; 5] = [0.25, 0.5, 0.75, 0.9375, 0.99609375];
        for i_k_sq in 0..n_k_sq {
            let k_sq: f64 = k_sq_all[i_k_sq];
            let elliptic_k: f64 = ellpk(1.0 - k_sq);
            let elliptic_e: f64 = ellpe(k_sq);
            let phi: f64 = (2.0 - k_sq) * elliptic_k - 2.0 * elliptic_e;
            let theta: f64 = (2.0 - k_sq) * elliptic_e - 2.0 * (1.0 - k_sq) * elliptic_k;

            let phi_theta: EllipticPhiTheta = elliptic_phi_theta(k_sq, 1.0 - k_sq);

            assert!((phi_theta.phi / phi - 1.0).abs() < 1.0e-13, "k_sq = {k_sq}: phi");
            assert!((phi_theta.theta / theta - 1.0).abs() < 1.0e-13, "k_sq = {k_sq}: theta");
        }

        // Where `k_sq` is small, both are of the order `k_sq ** 2`, Eq. (179) and (180) of the note
        let k_sq: f64 = 1.0e-8;
        let phi_theta: EllipticPhiTheta = elliptic_phi_theta(k_sq, 1.0 - k_sq);
        assert!((phi_theta.phi / (PI * k_sq.powi(2) / 16.0 * (1.0 + 0.75 * k_sq)) - 1.0).abs() < 1.0e-14);
        assert!((phi_theta.theta / (3.0 * PI * k_sq.powi(2) / 16.0 * (1.0 + 0.25 * k_sq)) - 1.0).abs() < 1.0e-14);

        // On the filament
        let phi_theta: EllipticPhiTheta = elliptic_phi_theta(1.0, 0.0);
        assert_eq!(phi_theta.phi, f64::INFINITY);
        assert_eq!(phi_theta.theta, 1.0);
    }

    #[test]
    fn the_moments_of_a_parallelogram_are_its_averages() {
        let side_alpha_r: f64 = 0.03;
        let side_alpha_z: f64 = 0.01;
        let side_beta_r: f64 = -0.02;
        let side_beta_z: f64 = 0.04;
        let n_multipole: usize = 4;

        let moment: Array2<f64> = parallelogram_moments(side_alpha_r, side_alpha_z, side_beta_r, side_beta_z, n_multipole);

        // The examples below Eq. (71) of the note
        assert_eq!(moment.dim(), (n_multipole + 1, n_multipole + 1));
        assert_eq!(moment[(0, 0)], 1.0);
        assert!((moment[(2, 0)] / ((side_alpha_r.powi(2) + side_beta_r.powi(2)) / 12.0) - 1.0).abs() < 1.0e-15);
        assert!((moment[(1, 1)] / ((side_alpha_r * side_alpha_z + side_beta_r * side_beta_z) / 12.0) - 1.0).abs() < 1.0e-15);
        assert!((moment[(0, 2)] / ((side_alpha_z.powi(2) + side_beta_z.powi(2)) / 12.0) - 1.0).abs() < 1.0e-15);
        // A parallelogram is symmetric about its centre
        assert_eq!(moment[(1, 0)], 0.0);
        assert_eq!(moment[(0, 1)], 0.0);
        assert_eq!(moment[(2, 1)], 0.0);
        // Above the highest order nothing is calculated
        assert!(moment[(3, 2)].is_nan());

        // A rectangle, whose two directions are independent: the average of `x ** 4` over a width `w` is `w ** 4 / 80`
        let width: f64 = 0.03;
        let height: f64 = 0.05;
        let moment: Array2<f64> = parallelogram_moments(width, 0.0, 0.0, height, n_multipole);
        assert!((moment[(4, 0)] / (width.powi(4) / 80.0) - 1.0).abs() < 1.0e-15);
        assert!((moment[(2, 2)] / (width.powi(2) / 12.0 * height.powi(2) / 12.0) - 1.0).abs() < 1.0e-15);
        assert!((moment[(0, 4)] / (height.powi(4) / 80.0) - 1.0).abs() < 1.0e-15);
    }
}
