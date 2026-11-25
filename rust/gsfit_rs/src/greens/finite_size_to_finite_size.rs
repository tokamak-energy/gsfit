//! The mutual inductance between toroidal conductors of finite size.
//!
//! Each conductor has a parallelogram cross-section, and carries a uniform current density. The mutual inductance of two of them is the
//! average over both cross-sections of the Green's function of a filament, `G`:
//!
//! ```text
//! M = (1 / (A * A')) * integral over the first cross-section, integral over the second, of  G(R, Z; R', Z')
//! ```
//!
//! Nothing is discretised: the average is calculated analytically, as the flux at a point is in `finite_size_to_point.rs`.
//! The derivation is in the note `documentation/flux_from_parallelogram_conductor/note/flux_from_parallelogram_conductor.pdf`, in its
//! section on the mutual inductance, and the comments here use its symbols.
//! * `mutual_inductance_parallelogram_far_field` - the far-field series, for two conductors which are apart: the Taylor series of `G`
//!   about the two centres, averaged over both cross-sections
//! * `mutual_inductance_parallelogram_near_field` - the near-field formula, for two conductors which are close together, which touch,
//!   which overlap, or which are the same conductor: a sum over the 16 pairs of an edge of one conductor and an edge of the other of
//!   double integrals which are known in closed form
//! * `pair_series_coefficients` - the near-field formula's coefficients, which depend on the orders of the series only
//! * `mutual_inductance_parallelogram` - chooses between the two for each pair, and halves a conductor where neither converges fast
//!   enough
//!
//! A prime marks what belongs to the conductor which carries the current, as it marks the source's coordinates, `(R', Z')`; the conductor
//! through which the flux is taken has the same symbols without the prime. The symbols of the note, and their names here:
//!
//! | In the note | Here |
//! | --- | --- |
//! | `M` | `mutual_inductance` |
//! | `A`, `A'` | `conductors_area` |
//! | `N_e`, `N_t`, `N_m`, `epsilon` | `n_elliptic`, `n_thin_ring`, `n_multipole`, `relative_tolerance` |
//! | The far-field series | |
//! | `(R_c, Z_c)`, `(R'_c, Z'_c)` | `centre_r`, `centre_z`, `centre_r_prime`, `centre_z_prime` |
//! | `H`, `H'` | `half_diagonal`, `half_diagonal_prime` |
//! | `d_c`, `h_c = Z_c - Z'_c` | `distance`, `h` |
//! | `delta_R`, `delta_Z`, `delta'_R`, `delta'_Z` | `delta_r`, `delta_z`, `delta_r_prime`, `delta_z_prime` |
//! | `delta_h = delta_Z - delta'_Z` | `delta_h` |
//! | `G_ijm`, the Taylor coefficients of `G` in `delta_R^i delta'_R^j delta_h^m` | `derivative[i_r][i_r_prime][i_h] / (i_r! * i_r_prime! * i_h!)` |
//! | `T = -(1 / h) dG/dh`, and its Taylor coefficients in `delta_h` | `series_t` |
//! | the Taylor coefficients of `G` in `delta_h` | `series_g` |
//! | the moments `<delta_R^i delta_Z^j>`, `<delta'_R^i delta'_Z^j>` | `moment_scaled[i_r][i_z] * (i_r! * i_z!)`, and the same with `_prime` |
//! | `u^2`, `d^2` | `u_sq`, `d_sq`; as polynomials in `delta_h`: `polynomial_u_sq`, `polynomial_d_sq` |
//! | `Phi`, `Theta` | `phi`, `theta` |
//! | The near-field formula | |
//! | `t = d_max / (2 R_min)`, the expansion parameter of the pair | `t`; `distance_sq_max` and `vertex_r_min` where the choice is made, `distance_max` and `corner_r_min` in the formula |
//! | `R_bar`, the mean major radius of the two centres | `centre_r_mean` |
//! | `rho = ((R' - R) + i (Z' - Z)) / (2 R_bar)`, the position of a point of the second conductor from a point of the first | `rho` |
//! | `xi = (R - R_bar) / R_bar`, `xi' = (R' - R_bar) / R_bar`, and `xi'_e'`, which is `xi'` at the middle of an edge | `xi`, `xi_prime`, `xi_middle` |
//! | `D = a + b` | `degree` |
//! | `U_mab`, `W_mab` | `kernel_plain`, `kernel_ln` |
//! | `e`, `e'`, the edges of the first and of the second conductor | `i_edge`, `i_edge_prime` |
//! | `exp(i phi_e)`, `exp(i phi_e')`, and `exp(i phi_ee')` with `phi_ee' = phi_e' - phi_e` | `phase`, `phase_prime`, `phase_relative` |
//! | `sigma = -rho exp(-i phi_e) = s - i d_e` | `sigma`, `s`, `edge_distance` |
//! | `s'`, the position along `e'` from its middle | `position` |
//! | `I_(a,b)` and `J~_(a,b)`, the integrals along `e` without and with the complex logarithm | `integral_plain`, `integral_ln` |
//! | `X^m_(a,b)`, and `Y~^m_(a,b)` with the complex logarithm, added up over the edges `e` | `sum_plain_plus`, `sum_plain_minus`, `sum_ln_plus`, `sum_ln_minus` |
//! | `Y^m_(a,b) = (Y~^m_(a,b) + conj(Y~^m_(b,a))) / 2`, with the real logarithm, added up over the edges `e` | `double_ln_plus`, `double_ln_minus` |
//! | the two tables of numbers which multiply the brackets without and with the logarithm | `coefficient_plain`, `coefficient_ln` |
//! | `P_ab`, `Q_ab` | `table_p`, `table_q` |

use super::finite_size_series::{
    EllipticPhiTheta, ExpansionParameterThresholds, MaxwellSeriesCoefficients, check_conductors, elliptic_phi_theta, expansion_parameter_thresholds,
    maxwell_series_coefficients, parallelogram_moments,
};
use super::geometry::Conductor;
use crate::settings::settings_equilibrium::SettingsEquilibriumNumericsGreensFiniteSize;
use faer::c64;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// The number of vertices of a parallelogram
const N_VERTEX: usize = 4;

/// The number of edges of a parallelogram
const N_EDGE: usize = 4;

/// The number of ends of an edge
const N_END: usize = 2;

/// The number of pairs of a corner of the second conductor and an edge of the first, each of which has one table of integrals along the edge
const N_CORNER_EDGE: usize = N_EDGE * N_EDGE;

/// A term which is smaller than this, compared with the largest term, cannot change a sum in double precision.
const NEGLIGIBLE_TERM: f64 = 1.0e-17;

/// How fast the terms of the orders which the truncation keeps only partly can grow from one order to the next, in units of `t`. Measured:
/// see `mutual_inductance_parallelogram_near_field`.
const PARTIAL_ORDER_GROWTH: f64 = 2.5;

/// The mutual inductance between toroidal conductors of finite size: between each conductor of one set and each conductor of another.
///
/// Each conductor has a parallelogram cross-section, and carries a uniform current density. It is described as the IMAS data dictionary
/// describes it, `pf_passive/loop(i1)/element(i2)/geometry/oblique`: one of its corners, the reference corner, the lengths of the two sides
/// which meet there, and their inclinations. The mutual inductance is the average over both cross-sections of the Green's function of a
/// filament, with the calculation chosen for each pair by `mutual_inductance_parallelogram`. Two conductors can be anywhere: apart,
/// touching, overlapping, or the same conductor, which gives its self-inductance.
///
/// With `is_symmetric` the two sets are the same, and only the pairs with `i_conductor_prime >= i_conductor` are calculated; the others are
/// copied from them, so the matrix is exactly symmetric.
///
/// # Arguments
/// * `oblique_r` - R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_z` - Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_length_alpha` - Length of each conductor's side which is inclined at `oblique_alpha`, shape = [n_conductor], [metre]
/// * `oblique_length_beta` - Length of each conductor's side which is inclined at `oblique_beta`, shape = [n_conductor], [metre]
/// * `oblique_alpha` - Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
/// * `oblique_beta` - Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
/// * `oblique_r_prime` - R coordinate of the reference corner of each conductor of the second set, shape = [n_conductor_prime], [metre]
/// * `oblique_z_prime` - Z coordinate of the reference corner of each conductor of the second set, shape = [n_conductor_prime], [metre]
/// * `oblique_length_alpha_prime` - Length of the side which is inclined at `oblique_alpha_prime`, of each conductor of the second set,
///   shape = [n_conductor_prime], [metre]
/// * `oblique_length_beta_prime` - Length of the side which is inclined at `oblique_beta_prime`, of each conductor of the second set,
///   shape = [n_conductor_prime], [metre]
/// * `oblique_alpha_prime` - Inclination of the first side of each conductor of the second set, counter-clockwise from the R axis,
///   shape = [n_conductor_prime], [radian]
/// * `oblique_beta_prime` - Inclination of the second side of each conductor of the second set, counter-clockwise from the Z axis,
///   shape = [n_conductor_prime], [radian]
/// * `is_symmetric` - The two sets are the same conductors, so each pair is calculated once; `Greens::conductor_to_self`
/// * `settings_greens_finite_size` - The orders of the series, the relative accuracy wanted, and how many times a conductor may
///   be halved: `equilibrium_settings/numerics/greens/finite_size`
///
/// # Returns
/// * `mutual_inductance` - The mutual inductance between each conductor of the first set and each conductor of the second,
///   shape = [n_conductor, n_conductor_prime], [henry]
///
/// # Errors
/// * the six arrays which describe a set of conductors do not have the same length
/// * a conductor has no area: a side without a positive length, or two sides which are parallel
/// * a conductor is not wholly at a positive major radius
/// * the relative tolerance is not greater than 0 and at most 1
/// * neither calculation converges fast enough for a pair of pieces, one of which has already been halved `n_subdivision_max` times
/// * `is_symmetric`, but the two sets are not the same conductors, to the last bit
#[allow(clippy::too_many_arguments)]
pub fn mutual_inductance_finite_size_to_finite_size(
    oblique_r: &Array1<f64>,
    oblique_z: &Array1<f64>,
    oblique_length_alpha: &Array1<f64>,
    oblique_length_beta: &Array1<f64>,
    oblique_alpha: &Array1<f64>,
    oblique_beta: &Array1<f64>,
    oblique_r_prime: &Array1<f64>,
    oblique_z_prime: &Array1<f64>,
    oblique_length_alpha_prime: &Array1<f64>,
    oblique_length_beta_prime: &Array1<f64>,
    oblique_alpha_prime: &Array1<f64>,
    oblique_beta_prime: &Array1<f64>,
    is_symmetric: bool,
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<Array2<f64>, String> {
    check_conductors(oblique_r, oblique_z, oblique_length_alpha, oblique_length_beta, oblique_alpha, oblique_beta, "")?;
    check_conductors(
        oblique_r_prime,
        oblique_z_prime,
        oblique_length_alpha_prime,
        oblique_length_beta_prime,
        oblique_alpha_prime,
        oblique_beta_prime,
        "_prime",
    )?;
    let n_conductor: usize = oblique_r.len();
    let n_conductor_prime: usize = oblique_r_prime.len();

    // What every pair shares: how small each calculation's expansion parameter has to be, and the near-field formula's tables of
    // coefficients. The far-field series' terms of odd order are zero, so an odd `n_multipole` is rounded down
    let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(settings_greens_finite_size)?;
    let coefficients: PairSeriesCoefficients = pair_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring);
    let n_multipole: usize = 2 * (settings_greens_finite_size.n_multipole / 2);

    // What depends on one conductor alone, its shape, is calculated once for each conductor, not once for each pair
    let mut conductors_shape: Vec<ConductorShape> = Vec::with_capacity(n_conductor);
    for i_conductor in 0..n_conductor {
        conductors_shape.push(ConductorShape::new(
            oblique_length_alpha[i_conductor],
            oblique_length_beta[i_conductor],
            oblique_alpha[i_conductor],
            oblique_beta[i_conductor],
            n_multipole,
        ));
    }
    let mut conductors_shape_prime: Vec<ConductorShape> = Vec::with_capacity(n_conductor_prime);
    for i_conductor_prime in 0..n_conductor_prime {
        conductors_shape_prime.push(ConductorShape::new(
            oblique_length_alpha_prime[i_conductor_prime],
            oblique_length_beta_prime[i_conductor_prime],
            oblique_alpha_prime[i_conductor_prime],
            oblique_beta_prime[i_conductor_prime],
            n_multipole,
        ));
    }

    // With `is_symmetric`, the two sets have to be the same conductors: the same number, and every value equal to the last bit
    if is_symmetric {
        let mut is_same_set: bool = n_conductor == n_conductor_prime;
        if is_same_set {
            'conductor_loop: for i_conductor in 0..n_conductor {
                let is_same_conductor: bool = oblique_r[i_conductor].to_bits() == oblique_r_prime[i_conductor].to_bits()
                    && oblique_z[i_conductor].to_bits() == oblique_z_prime[i_conductor].to_bits()
                    && oblique_length_alpha[i_conductor].to_bits() == oblique_length_alpha_prime[i_conductor].to_bits()
                    && oblique_length_beta[i_conductor].to_bits() == oblique_length_beta_prime[i_conductor].to_bits()
                    && oblique_alpha[i_conductor].to_bits() == oblique_alpha_prime[i_conductor].to_bits()
                    && oblique_beta[i_conductor].to_bits() == oblique_beta_prime[i_conductor].to_bits();
                if !is_same_conductor {
                    is_same_set = false;
                    break 'conductor_loop;
                }
            }
        }
        if !is_same_set {
            return Err("`is_symmetric`, but the two sets are not the same conductors".to_string());
        }
    }

    // Each conductor of the first set is one task, which calculates one row of the matrix: all of it, or, when the two sets are the same,
    // from the diagonal on. The work arrays are allocated once for each thread
    let rows: Vec<Array1<f64>> = (0..n_conductor)
        .into_par_iter()
        .map_init(
            || -> PairWorkArrays { PairWorkArrays::new(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring, n_multipole) },
            |work_arrays: &mut PairWorkArrays, i_conductor: usize| -> Result<Array1<f64>, String> {
                let i_conductor_prime_start: usize = if is_symmetric { i_conductor } else { 0 };
                let mut row: Array1<f64> = Array1::from_elem(n_conductor_prime, f64::NAN);
                for i_conductor_prime in i_conductor_prime_start..n_conductor_prime {
                    let pair: MutualInductanceParallelogram = mutual_inductance_parallelogram(
                        oblique_r[i_conductor],
                        oblique_z[i_conductor],
                        &conductors_shape[i_conductor],
                        oblique_r_prime[i_conductor_prime],
                        oblique_z_prime[i_conductor_prime],
                        &conductors_shape_prime[i_conductor_prime],
                        &thresholds,
                        &coefficients,
                        n_multipole,
                        settings_greens_finite_size.n_subdivision_max,
                        work_arrays,
                    )
                    .map_err(|error: String| format!("conductor {i_conductor} of the first set and conductor {i_conductor_prime} of the second: {error}"))?;
                    row[i_conductor_prime] = pair.mutual_inductance;
                }

                Ok(row)
            },
        )
        .collect::<Result<Vec<Array1<f64>>, String>>()?;

    let mut mutual_inductance: Array2<f64> = Array2::from_elem((n_conductor, n_conductor_prime), f64::NAN);
    for i_conductor in 0..n_conductor {
        let i_conductor_prime_start: usize = if is_symmetric { i_conductor } else { 0 };
        for i_conductor_prime in i_conductor_prime_start..n_conductor_prime {
            mutual_inductance[(i_conductor, i_conductor_prime)] = rows[i_conductor][i_conductor_prime];
            if is_symmetric {
                mutual_inductance[(i_conductor_prime, i_conductor)] = rows[i_conductor][i_conductor_prime];
            }
        }
    }

    Ok(mutual_inductance)
}

/// `ConductorModel::FiniteSize` between conductors: their self- and mutual inductances.
pub(super) struct FiniteSizeToFiniteSize {
    /// The first set of conductors, the rows
    pub(super) conductor_1: Conductor,
    /// The second set of conductors, the columns
    pub(super) conductor_2: Conductor,
    /// The two sets are the same, from `conductor_to_self`: each pair is calculated once
    pub(super) is_symmetric: bool,
    /// The orders of the series and their accuracy
    pub(super) settings_greens_finite_size: SettingsEquilibriumNumericsGreensFiniteSize,
}

impl FiniteSizeToFiniteSize {
    /// The mutual inductances.
    ///
    /// # Returns
    /// * `mutual_inductance` - shape = [n_conductor_1, n_conductor_2], [henry]
    ///
    /// # Errors
    /// * as `mutual_inductance_finite_size_to_finite_size`
    pub(super) fn psi(&self) -> Result<Array2<f64>, String> {
        mutual_inductance_finite_size_to_finite_size(
            &self.conductor_1.r,
            &self.conductor_1.z,
            &self.conductor_1.length_alpha,
            &self.conductor_1.length_beta,
            &self.conductor_1.alpha,
            &self.conductor_1.beta,
            &self.conductor_2.r,
            &self.conductor_2.z,
            &self.conductor_2.length_alpha,
            &self.conductor_2.length_beta,
            &self.conductor_2.alpha,
            &self.conductor_2.beta,
            self.is_symmetric,
            &self.settings_greens_finite_size,
        )
    }
}

/// What `mutual_inductance_parallelogram` returns: the mutual inductance of one pair of conductors, and how it was calculated.
struct MutualInductanceParallelogram {
    /// The mutual inductance of the pair, [henry]
    mutual_inductance: f64,
    /// How many pairs of pieces were summed with the far-field series, [count]
    #[allow(dead_code, reason = "how the pair was calculated is only read by the tests")]
    n_piece_pair_far_field: usize,
    /// How many pairs of pieces were summed with the near-field formula, [count]
    #[allow(dead_code, reason = "how the pair was calculated is only read by the tests")]
    n_piece_pair_near_field: usize,
}

/// The shape of a parallelogram: what the calculations need to know about a conductor, or about one of the pieces which it is cut into,
/// apart from where it is. It depends on one conductor alone, so it is built once for each conductor, not once for each pair.
struct ConductorShape {
    /// Length of the side which is inclined at `oblique_alpha`, [metre]
    length_alpha: f64,
    /// Length of the side which is inclined at `oblique_beta`, [metre]
    length_beta: f64,
    /// R component of the side which is inclined at `oblique_alpha`, [metre]
    side_alpha_r: f64,
    /// Z component of the side which is inclined at `oblique_alpha`, [metre]
    side_alpha_z: f64,
    /// R component of the side which is inclined at `oblique_beta`, [metre]
    side_beta_r: f64,
    /// Z component of the side which is inclined at `oblique_beta`, [metre]
    side_beta_z: f64,
    /// R component of half of the longer side, along which the parallelogram is halved: the position of the second half's reference
    /// corner from the first half's, which is the parallelogram's own, [metre]
    half_longer_side_r: f64,
    /// Z component of half of the longer side, [metre]
    half_longer_side_z: f64,
    /// The distance from the centre to the furthest vertex, `H` in the note, [metre]
    half_diagonal: f64,
    /// R coordinate of each of the four vertices, from the reference corner, [metre]
    vertices_r: [f64; N_VERTEX],
    /// Z coordinate of each of the four vertices, from the reference corner, [metre]
    vertices_z: [f64; N_VERTEX],
    /// The moments about the centre, each over the factorials of its two powers: `moment_scaled[i_r][i_z]` is
    /// `<delta_R^i_r delta_Z^i_z> / (i_r! * i_z!)` in the note's symbols, shape = [n_multipole + 1, n_multipole + 1],
    /// [metre ** (i_r + i_z)]. They are nested vectors, because the far-field series takes their rows in its innermost loop
    moment_scaled: Vec<Vec<f64>>,
}

impl ConductorShape {
    /// The shape of a conductor, from the lengths and the inclinations of the two sides which meet at its reference corner.
    ///
    /// # Arguments
    /// * `oblique_length_alpha` - Length of the side which is inclined at `oblique_alpha`, [metre]
    /// * `oblique_length_beta` - Length of the side which is inclined at `oblique_beta`, [metre]
    /// * `oblique_alpha` - Inclination of the first side, counter-clockwise from the R axis, [radian]
    /// * `oblique_beta` - Inclination of the second side, counter-clockwise from the Z axis, [radian]
    /// * `n_multipole` - The highest order of the far-field series, which is the highest order of the moments
    ///
    /// # Returns
    /// * `shape` - The conductor's sides, half-diagonal, vertices and moments
    fn new(oblique_length_alpha: f64, oblique_length_beta: f64, oblique_alpha: f64, oblique_beta: f64, n_multipole: usize) -> Self {
        ConductorShape::from_sides(
            oblique_length_alpha,
            oblique_length_beta,
            oblique_length_alpha * oblique_alpha.cos(),
            oblique_length_alpha * oblique_alpha.sin(),
            -oblique_length_beta * oblique_beta.sin(),
            oblique_length_beta * oblique_beta.cos(),
            n_multipole,
        )
    }

    /// The shape of the two halves of this parallelogram, when it is halved along its longer side: along the side which is inclined at
    /// `oblique_alpha`, when the two are equally long.
    ///
    /// # Arguments
    /// * `n_multipole` - The highest order of the far-field series, which is the highest order of the moments
    ///
    /// # Returns
    /// * `shape` - The sides, half-diagonal, vertices and moments of each half
    fn halved(&self, n_multipole: usize) -> Self {
        if self.length_alpha >= self.length_beta {
            ConductorShape::from_sides(
                0.5 * self.length_alpha,
                self.length_beta,
                0.5 * self.side_alpha_r,
                0.5 * self.side_alpha_z,
                self.side_beta_r,
                self.side_beta_z,
                n_multipole,
            )
        } else {
            ConductorShape::from_sides(
                self.length_alpha,
                0.5 * self.length_beta,
                self.side_alpha_r,
                self.side_alpha_z,
                0.5 * self.side_beta_r,
                0.5 * self.side_beta_z,
                n_multipole,
            )
        }
    }

    /// The shape of a parallelogram, from the two sides which meet at its reference corner.
    ///
    /// # Arguments
    /// * `length_alpha` - Length of the side which is inclined at `oblique_alpha`, [metre]
    /// * `length_beta` - Length of the side which is inclined at `oblique_beta`, [metre]
    /// * `side_alpha_r` - R component of the side which is inclined at `oblique_alpha`, [metre]
    /// * `side_alpha_z` - Z component of the side which is inclined at `oblique_alpha`, [metre]
    /// * `side_beta_r` - R component of the side which is inclined at `oblique_beta`, [metre]
    /// * `side_beta_z` - Z component of the side which is inclined at `oblique_beta`, [metre]
    /// * `n_multipole` - The highest order of the far-field series, which is the highest order of the moments
    ///
    /// # Returns
    /// * `shape` - The parallelogram's sides, half-diagonal, vertices and moments
    fn from_sides(length_alpha: f64, length_beta: f64, side_alpha_r: f64, side_alpha_z: f64, side_beta_r: f64, side_beta_z: f64, n_multipole: usize) -> Self {
        // The longer of the two half-diagonals
        let half_diagonal: f64 = (0.5 * side_alpha_r + 0.5 * side_beta_r)
            .hypot(0.5 * side_alpha_z + 0.5 * side_beta_z)
            .max((0.5 * side_alpha_r - 0.5 * side_beta_r).hypot(0.5 * side_alpha_z - 0.5 * side_beta_z));

        // The moments, each over the factorials of its two powers
        let moment: Array2<f64> = parallelogram_moments(side_alpha_r, side_alpha_z, side_beta_r, side_beta_z, n_multipole);
        let n_series: usize = n_multipole + 1;
        let mut moment_scaled: Vec<Vec<f64>> = vec![vec![f64::NAN; n_series]; n_series];
        let mut factorial_r: f64 = 1.0;
        for i_r in 0..n_series {
            if i_r >= 1 {
                factorial_r *= i_r as f64;
            }
            let mut factorial_z: f64 = 1.0;
            for i_z in 0..n_series {
                if i_z >= 1 {
                    factorial_z *= i_z as f64;
                }
                moment_scaled[i_r][i_z] = moment[(i_r, i_z)] / (factorial_r * factorial_z);
            }
        }

        // The side along which the parallelogram is halved, as in `halved`
        let is_alpha_longer: bool = length_alpha >= length_beta;

        ConductorShape {
            length_alpha,
            length_beta,
            side_alpha_r,
            side_alpha_z,
            side_beta_r,
            side_beta_z,
            half_longer_side_r: if is_alpha_longer { 0.5 * side_alpha_r } else { 0.5 * side_beta_r },
            half_longer_side_z: if is_alpha_longer { 0.5 * side_alpha_z } else { 0.5 * side_beta_z },
            half_diagonal,
            vertices_r: [0.0, side_alpha_r, side_alpha_r + side_beta_r, side_beta_r],
            vertices_z: [0.0, side_alpha_z, side_alpha_z + side_beta_z, side_beta_z],
            moment_scaled,
        }
    }
}

/// One of the pieces which a conductor is cut into: a parallelogram with the conductor's two inclinations.
#[derive(Clone, Copy)]
struct Piece {
    /// R coordinate of the piece's reference corner, [metre]
    oblique_r: f64,
    /// Z coordinate of the piece's reference corner, [metre]
    oblique_z: f64,
    /// How many times the conductor was halved to make the piece
    i_subdivision: usize,
}

/// A piece of each of the two conductors of a pair, which are still to be summed.
struct PiecePair {
    /// The piece of the first conductor
    piece: Piece,
    /// The piece of the second conductor
    piece_prime: Piece,
    /// The fraction of the first conductor's current which its piece carries, times the same for the second, [dimensionless]
    current_fraction: f64,
}

/// The work arrays of a pair of conductors. They are allocated once for each thread, and used again for every pair.
struct PairWorkArrays {
    /// The pairs of pieces which are still to be summed
    piece_pairs: Vec<PiecePair>,
    /// The shape of the first conductor's pieces after 1, 2, ... halvings, as far as they have been needed
    subdivisions_shape: Vec<ConductorShape>,
    /// The shape of the second conductor's pieces after 1, 2, ... halvings, as far as they have been needed
    subdivisions_shape_prime: Vec<ConductorShape>,
    /// The work arrays of the far-field series
    far_field: FarFieldPairWorkArrays,
    /// The work arrays of the near-field formula
    near_field: NearFieldPairWorkArrays,
}

impl PairWorkArrays {
    /// The work arrays of a pair of conductors, for the orders of the settings.
    fn new(n_elliptic: usize, n_thin_ring: usize, n_multipole: usize) -> Self {
        PairWorkArrays {
            piece_pairs: Vec::new(),
            subdivisions_shape: Vec::new(),
            subdivisions_shape_prime: Vec::new(),
            far_field: FarFieldPairWorkArrays::new(n_multipole),
            near_field: NearFieldPairWorkArrays::new(n_elliptic, n_thin_ring),
        }
    }
}

/// The mutual inductance of one pair of parallelogram conductors, each with a uniform current density, with the calculation chosen for
/// the pair.
///
/// With `half_diagonal` and `half_diagonal_prime` the distances from each conductor's centre to its furthest vertex, `distance` the
/// distance between the two centres, and `centre_r` and `centre_r_prime` the major radii of the two centres:
/// * the far-field series is used when its expansion parameter, the largest of `(half_diagonal + half_diagonal_prime) / distance`,
///   `half_diagonal / centre_r` and `half_diagonal_prime / centre_r_prime`, is at most `far_field_ratio_max`;
/// * otherwise the near-field formula is used when its expansion parameter, `t / (1 - t)`, is at most `near_field_ratio_max`, where `t` is
///   the largest distance between a vertex of one conductor and a vertex of the other, over twice the smallest major radius of the eight
///   vertices;
/// * otherwise the conductor whose longer side is the longer of the two is halved along that side, and each half is paired with the
///   other conductor and treated in the same way. This is exact, because the current density is uniform: the mutual inductance is the
///   mean of the two halves' mutual inductances. When the two longer sides are equally long, the first conductor is halved.
///
/// Conductors which touch, overlap, or are the same conductor always end in the near-field formula. Higher orders mean larger
/// thresholds, and fewer pieces.
///
/// # Arguments
/// * `oblique_r` - R coordinate of the first conductor's reference corner, [metre]
/// * `oblique_z` - Z coordinate of the first conductor's reference corner, [metre]
/// * `shape` - The first conductor's shape
/// * `oblique_r_prime` - R coordinate of the second conductor's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the second conductor's reference corner, [metre]
/// * `shape_prime` - The second conductor's shape
/// * `thresholds` - How small each calculation's expansion parameter has to be before it is used, from `expansion_parameter_thresholds`
/// * `coefficients` - The near-field formula's tables of coefficients, from `pair_series_coefficients`
/// * `n_multipole` - The highest order of the far-field series, which is even
/// * `n_subdivision_max` - How many times a conductor may be halved
/// * `work_arrays` - The work arrays, which are overwritten
///
/// The two conductors have already been checked by `mutual_inductance_finite_size_to_finite_size`.
///
/// # Returns
/// * `pair` - The mutual inductance, `mutual_inductance` [henry], and how many pairs of pieces were summed with each calculation,
///   `n_piece_pair_far_field` and `n_piece_pair_near_field` [count]
///
/// # Errors
/// * neither calculation converges fast enough for a pair of pieces, one of which has already been halved `n_subdivision_max` times
#[allow(clippy::too_many_arguments)]
fn mutual_inductance_parallelogram(
    oblique_r: f64,
    oblique_z: f64,
    shape: &ConductorShape,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    shape_prime: &ConductorShape,
    thresholds: &ExpansionParameterThresholds,
    coefficients: &PairSeriesCoefficients,
    n_multipole: usize,
    n_subdivision_max: usize,
    work_arrays: &mut PairWorkArrays,
) -> Result<MutualInductanceParallelogram, String> {
    // Each conductor may be halved `n_subdivision_max` times, which cuts the pair into at most `4 ** n_subdivision_max` pairs of pieces,
    // and fewer than twice as many pairs are looked at on the way to them. The count stops at the largest integer when
    // `n_subdivision_max` is large
    let n_piece_pair_max: usize = 2_usize.saturating_pow(u32::try_from(n_subdivision_max).unwrap_or(u32::MAX).saturating_mul(2).saturating_add(1));

    // The pairs of pieces which are still to be summed, starting with the two whole conductors. The last pair is taken from the list, and
    // it is either summed with one of the two calculations, or replaced with two pairs, in which one of its pieces is halved
    work_arrays.subdivisions_shape.clear();
    work_arrays.subdivisions_shape_prime.clear();
    work_arrays.piece_pairs.clear();
    work_arrays.piece_pairs.push(PiecePair {
        piece: Piece {
            oblique_r,
            oblique_z,
            i_subdivision: 0,
        },
        piece_prime: Piece {
            oblique_r: oblique_r_prime,
            oblique_z: oblique_z_prime,
            i_subdivision: 0,
        },
        current_fraction: 1.0,
    });

    let mut mutual_inductance: f64 = 0.0;
    let mut n_piece_pair_far_field: usize = 0;
    let mut n_piece_pair_near_field: usize = 0;
    'piece_pair_loop: for _i_piece_pair in 0..n_piece_pair_max {
        let Some(piece_pair): Option<PiecePair> = work_arrays.piece_pairs.pop() else {
            break 'piece_pair_loop;
        };
        let piece: Piece = piece_pair.piece;
        let piece_prime: Piece = piece_pair.piece_prime;

        // The shapes of the two pieces: the whole conductors', or the ones which their halvings have made
        let piece_shape: &ConductorShape = if piece.i_subdivision == 0 {
            shape
        } else {
            &work_arrays.subdivisions_shape[piece.i_subdivision - 1]
        };
        let piece_shape_prime: &ConductorShape = if piece_prime.i_subdivision == 0 {
            shape_prime
        } else {
            &work_arrays.subdivisions_shape_prime[piece_prime.i_subdivision - 1]
        };

        // The two centres, and the far-field series' expansion parameter: the two pieces' sizes over the distance between their centres,
        // or a piece's size over its own major radius if that is larger
        let centre_r: f64 = piece.oblique_r + (0.5 * piece_shape.side_alpha_r + 0.5 * piece_shape.side_beta_r);
        let centre_z: f64 = piece.oblique_z + (0.5 * piece_shape.side_alpha_z + 0.5 * piece_shape.side_beta_z);
        let centre_r_prime: f64 = piece_prime.oblique_r + (0.5 * piece_shape_prime.side_alpha_r + 0.5 * piece_shape_prime.side_beta_r);
        let centre_z_prime: f64 = piece_prime.oblique_z + (0.5 * piece_shape_prime.side_alpha_z + 0.5 * piece_shape_prime.side_beta_z);
        let h: f64 = centre_z - centre_z_prime;
        let distance: f64 = (centre_r - centre_r_prime).hypot(h);
        let is_far_field: bool = piece_shape.half_diagonal + piece_shape_prime.half_diagonal <= thresholds.far_field_ratio_max * distance
            && piece_shape.half_diagonal <= thresholds.far_field_ratio_max * centre_r
            && piece_shape_prime.half_diagonal <= thresholds.far_field_ratio_max * centre_r_prime;

        if is_far_field {
            mutual_inductance += piece_pair.current_fraction
                * mutual_inductance_parallelogram_far_field(
                    centre_r,
                    centre_r_prime,
                    h,
                    &piece_shape.moment_scaled,
                    &piece_shape_prime.moment_scaled,
                    n_multipole,
                    &mut work_arrays.far_field,
                );
            n_piece_pair_far_field += 1;
        } else {
            // The near-field formula's expansion parameter, `t / (1 - t)`, where `t` is the largest distance between a vertex of one piece
            // and a vertex of the other, over twice the smallest major radius of the eight vertices
            let mut vertices_r: [f64; N_VERTEX] = [f64::NAN; N_VERTEX];
            let mut vertices_z: [f64; N_VERTEX] = [f64::NAN; N_VERTEX];
            let mut vertices_r_prime: [f64; N_VERTEX] = [f64::NAN; N_VERTEX];
            let mut vertices_z_prime: [f64; N_VERTEX] = [f64::NAN; N_VERTEX];
            let mut vertex_r_min: f64 = f64::INFINITY;
            for i_vertex in 0..N_VERTEX {
                vertices_r[i_vertex] = piece.oblique_r + piece_shape.vertices_r[i_vertex];
                vertices_z[i_vertex] = piece.oblique_z + piece_shape.vertices_z[i_vertex];
                vertices_r_prime[i_vertex] = piece_prime.oblique_r + piece_shape_prime.vertices_r[i_vertex];
                vertices_z_prime[i_vertex] = piece_prime.oblique_z + piece_shape_prime.vertices_z[i_vertex];
                vertex_r_min = vertex_r_min.min(vertices_r[i_vertex]).min(vertices_r_prime[i_vertex]);
            }
            let mut distance_sq_max: f64 = 0.0;
            for i_vertex in 0..N_VERTEX {
                for i_vertex_prime in 0..N_VERTEX {
                    distance_sq_max = distance_sq_max.max(
                        (vertices_r[i_vertex] - vertices_r_prime[i_vertex_prime]).powi(2) + (vertices_z[i_vertex] - vertices_z_prime[i_vertex_prime]).powi(2),
                    );
                }
            }
            let t: f64 = distance_sq_max.sqrt() / (2.0 * vertex_r_min);
            let is_near_field: bool = t <= thresholds.near_field_ratio_max * (1.0 - t);

            if is_near_field {
                mutual_inductance += piece_pair.current_fraction
                    * mutual_inductance_parallelogram_near_field(
                        piece.oblique_r,
                        piece.oblique_z,
                        piece_shape.side_alpha_r,
                        piece_shape.side_alpha_z,
                        piece_shape.side_beta_r,
                        piece_shape.side_beta_z,
                        piece_prime.oblique_r,
                        piece_prime.oblique_z,
                        piece_shape_prime.side_alpha_r,
                        piece_shape_prime.side_alpha_z,
                        piece_shape_prime.side_beta_r,
                        piece_shape_prime.side_beta_z,
                        coefficients,
                        &mut work_arrays.near_field,
                    );
                n_piece_pair_near_field += 1;
            } else {
                // Halve the piece whose longer side is the longer of the two, along that side. Each half carries half of its current
                let is_first_halved: bool =
                    piece_shape.length_alpha.max(piece_shape.length_beta) >= piece_shape_prime.length_alpha.max(piece_shape_prime.length_beta);
                let i_subdivision_halved: usize = if is_first_halved { piece.i_subdivision } else { piece_prime.i_subdivision };
                if i_subdivision_halved == n_subdivision_max {
                    return Err(format!(
                        "neither the far-field series nor the near-field formula converges fast enough for a pair of pieces of the two conductors, \
                         one of which has already been halved {n_subdivision_max} time(s); increase `n_subdivision_max`, or the orders \
                         `n_elliptic`, `n_thin_ring` and `n_multipole`, or `relative_tolerance`"
                    ));
                }

                if is_first_halved {
                    let first_half: Piece = Piece {
                        oblique_r: piece.oblique_r,
                        oblique_z: piece.oblique_z,
                        i_subdivision: piece.i_subdivision + 1,
                    };
                    let second_half: Piece = Piece {
                        oblique_r: piece.oblique_r + piece_shape.half_longer_side_r,
                        oblique_z: piece.oblique_z + piece_shape.half_longer_side_z,
                        i_subdivision: piece.i_subdivision + 1,
                    };
                    // The shape of the halves is built the first time that the first conductor is halved this many times
                    if piece.i_subdivision == work_arrays.subdivisions_shape.len() {
                        let halves_shape: ConductorShape = piece_shape.halved(n_multipole);
                        work_arrays.subdivisions_shape.push(halves_shape);
                    }
                    work_arrays.piece_pairs.push(PiecePair {
                        piece: second_half,
                        piece_prime,
                        current_fraction: 0.5 * piece_pair.current_fraction,
                    });
                    work_arrays.piece_pairs.push(PiecePair {
                        piece: first_half,
                        piece_prime,
                        current_fraction: 0.5 * piece_pair.current_fraction,
                    });
                } else {
                    // The same for the second conductor
                    let first_half_prime: Piece = Piece {
                        oblique_r: piece_prime.oblique_r,
                        oblique_z: piece_prime.oblique_z,
                        i_subdivision: piece_prime.i_subdivision + 1,
                    };
                    let second_half_prime: Piece = Piece {
                        oblique_r: piece_prime.oblique_r + piece_shape_prime.half_longer_side_r,
                        oblique_z: piece_prime.oblique_z + piece_shape_prime.half_longer_side_z,
                        i_subdivision: piece_prime.i_subdivision + 1,
                    };
                    if piece_prime.i_subdivision == work_arrays.subdivisions_shape_prime.len() {
                        let halves_shape_prime: ConductorShape = piece_shape_prime.halved(n_multipole);
                        work_arrays.subdivisions_shape_prime.push(halves_shape_prime);
                    }
                    work_arrays.piece_pairs.push(PiecePair {
                        piece,
                        piece_prime: second_half_prime,
                        current_fraction: 0.5 * piece_pair.current_fraction,
                    });
                    work_arrays.piece_pairs.push(PiecePair {
                        piece,
                        piece_prime: first_half_prime,
                        current_fraction: 0.5 * piece_pair.current_fraction,
                    });
                }
            }
        }
    }

    Ok(MutualInductanceParallelogram {
        mutual_inductance,
        n_piece_pair_far_field,
        n_piece_pair_near_field,
    })
}

/// The arrays of the far-field series: its work arrays, which are allocated once and overwritten for every pair of pieces, and the
/// factorials, which are the same for every pair.
struct FarFieldPairWorkArrays {
    /// The Taylor coefficients in `delta_h` of the Green's function of a filament, along the vertical line through the two centres,
    /// shape = [n_multipole + 1], [henry / metre ** i_h]
    series_g: Vec<f64>,
    /// The Taylor coefficients in `delta_h` of `T = -(1 / h) dG/dh`, along the same line, shape = [n_multipole + 1],
    /// [henry / metre ** (i_h + 2)]
    series_t: Vec<f64>,
    /// The derivatives of the Green's function of a filament at the two centres: `derivative[i_r][i_r_prime][i_h]` is differentiated
    /// `i_r` times with respect to the first major radius, `i_r_prime` times with respect to the second, and `i_h` times with respect to
    /// the height of the first above the second. It is used for `i_r + i_r_prime + i_h <= n_multipole`,
    /// shape = [n_multipole + 1, n_multipole + 1, n_multipole + 1], [henry / metre ** (i_r + i_r_prime + i_h)]. They are nested
    /// vectors, because the sum takes their rows in its innermost loops
    derivative: Vec<Vec<Vec<f64>>>,
    /// The factorials, `factorial[i] = i!`, shape = [n_multipole + 1]
    factorial: Vec<f64>,
}

impl FarFieldPairWorkArrays {
    /// The arrays of the far-field series, for the orders up to `n_multipole`.
    fn new(n_multipole: usize) -> Self {
        let n_series: usize = n_multipole + 1;

        let mut factorial: Vec<f64> = vec![f64::NAN; n_series];
        factorial[0] = 1.0;
        for i_series in 1..n_series {
            factorial[i_series] = factorial[i_series - 1] * (i_series as f64);
        }

        FarFieldPairWorkArrays {
            series_g: vec![f64::NAN; n_series],
            series_t: vec![f64::NAN; n_series],
            derivative: vec![vec![vec![f64::NAN; n_series]; n_series]; n_series],
            factorial,
        }
    }
}

/// The far-field series: the mutual inductance of two parallelograms, each with a uniform current density, which are apart.
///
/// A point of the first parallelogram is at `(centre_r + delta_r, centre_z + delta_z)`, and a point of the second at
/// `(centre_r_prime + delta_r_prime, centre_z_prime + delta_z_prime)`. The Green's function of a filament, `G`, depends on the two heights
/// only through their difference, `h + delta_h`, with `h = centre_z - centre_z_prime` and `delta_h = delta_z - delta_z_prime`, so its
/// Taylor series about the two centres has three indices:
///
/// ```text
/// G = sum over i_r, i_r_prime, i_h of
///     derivative[i_r][i_r_prime][i_h] * (delta_r ** i_r / i_r!) * (delta_r_prime ** i_r_prime / i_r_prime!) * (delta_h ** i_h / i_h!)
/// ```
///
/// where `derivative[i_r][i_r_prime][i_h]` is `G` at the two centres, differentiated `i_r` times with respect to the first major radius,
/// `i_r_prime` times with respect to the second, and `i_h` times with respect to the difference of the heights. The mutual inductance is
/// the average of this over both parallelograms. The two are independent, so the average of each term is a sum of products of the two
/// parallelograms' own moments, one for each term of the binomial expansion of `(delta_z - delta_z_prime) ** i_h`. A binomial
/// coefficient over `i_h!` is `1 / (i_z! * (i_h - i_z)!)`, so each moment is divided by the factorials of its own two powers, and the
/// moments are kept in that form:
///
/// ```text
/// mutual_inductance = sum over i_r + i_r_prime + i_h even, at most n_multipole, of  derivative[i_r][i_r_prime][i_h] * moment_pair
/// moment_pair = sum over i_z = 0 .. i_h of  (-1) ** (i_h - i_z) * moment_scaled[i_r][i_z] * moment_scaled_prime[i_r_prime][i_h - i_z]
/// moment_scaled[i_r][i_z] = (the average over the first parallelogram of  delta_r ** i_r * delta_z ** i_z) / (i_r! * i_z!)
/// ```
///
/// The term of order 0 is the mutual inductance of two filaments at the centres. The terms of odd order are zero, because a
/// parallelogram is symmetric about its centre.
///
/// Every derivative comes from one pair of elliptic integrals, at the two centres, by recurrences. With
/// `T = -(1 / h) * d(G)/d(h) = MU_0 * u * Theta / (2 * d ** 2)`, the Green's function obeys six equations in the two major radii,
/// `r` and `r_prime`, and the difference of the heights, `h`. Each recurrence is one of them, differentiated, and they are used in this
/// order:
/// 1. along the vertical line through the two centres, the Taylor series in `delta_h` of `G` and of `T`, from
///    `d(G)/d(h) = -h * T` and `d(u ** 2 * d ** 2 * T)/d(h) = -3 * h * G`, whose coefficients are polynomials in `delta_h`. These two
///    equations are singular only where `G` itself is. The derivatives with respect to `h` are the Taylor coefficients times `i_h!`;
/// 2. on the same line, the first derivative with respect to the first major radius, from
///    `2 * r * d(G)/d(r) = G + (r_prime ** 2 + h ** 2 - r ** 2) * T`;
/// 3. the higher derivatives with respect to the first major radius, from the equation which `G` obeys in the first point's
///    coordinates, `r * (d2(G)/d(r2) + d2(G)/d(h2)) - d(G)/d(r) = 0`;
/// 4. the first derivative with respect to the second major radius, from Euler's theorem,
///    `r * d(G)/d(r) + r_prime * d(G)/d(r_prime) + h * d(G)/d(h) = G`, because `G` is homogeneous of degree 1 in `(r, r_prime, h)`;
/// 5. the higher derivatives with respect to the second major radius, from the equation which `G` obeys in the second point's
///    coordinates, `r_prime * (d2(G)/d(r_prime2) + d2(G)/d(h2)) - d(G)/d(r_prime) = 0`.
///
/// The order matters. The recurrences of steps 3 and 5 are singular where a major radius is zero: rounding excites a second solution of
/// them, whose terms in the series grow like `(delta_r / centre_r) ** i_r` and `(delta_r_prime / centre_r_prime) ** i_r_prime`. A
/// conductor at a positive major radius is narrower than its centre's major radius, so these terms decrease, and the sum keeps its
/// accuracy. The derivatives with respect to `h` are all made in step 1, which has no such solution: the heights of the conductors are
/// not limited by their major radii.
///
/// The series converges when the two centres are further apart than the sum of the two half-diagonals. Its relative error is below
/// `0.6 * ratio ** (n_multipole + 2)`, where `ratio` is the largest of `(half_diagonal + half_diagonal_prime) / distance`,
/// `half_diagonal / centre_r` and `half_diagonal_prime / centre_r_prime`.
///
/// # Arguments
/// * `centre_r` - R coordinate of the first parallelogram's centre, [metre]
/// * `centre_r_prime` - R coordinate of the second parallelogram's centre, [metre]
/// * `h` - Height of the first parallelogram's centre above the second's, `centre_z - centre_z_prime`, [metre]
/// * `moment_scaled` - The first parallelogram's moments about its centre, each over the factorials of its two powers,
///   shape = [n_multipole + 1, n_multipole + 1], [metre ** (i_r + i_z)]
/// * `moment_scaled_prime` - The same for the second parallelogram, shape = [n_multipole + 1, n_multipole + 1], [metre ** (i_r + i_z)]
/// * `n_multipole` - The highest order kept, which is even; 0 is two filaments
/// * `work_arrays` - The series' arrays, for this `n_multipole`; the work arrays are overwritten
///
/// # Returns
/// * `mutual_inductance` - The mutual inductance of the two parallelograms, [henry]
fn mutual_inductance_parallelogram_far_field(
    centre_r: f64,
    centre_r_prime: f64,
    h: f64,
    moment_scaled: &[Vec<f64>],
    moment_scaled_prime: &[Vec<f64>],
    n_multipole: usize,
    work_arrays: &mut FarFieldPairWorkArrays,
) -> f64 {
    let n_series: usize = n_multipole + 1;
    let series_g: &mut [f64] = &mut work_arrays.series_g;
    let series_t: &mut [f64] = &mut work_arrays.series_t;
    let derivative: &mut [Vec<Vec<f64>>] = &mut work_arrays.derivative;
    let factorial: &[f64] = &work_arrays.factorial;

    // Along the vertical line through the two centres, where the height of the first point above the second is `h + delta_h`, these are
    // quadratics in `delta_h`, lowest power first: u ** 2 and d ** 2. Their product is a quartic
    const N_COEFFICIENT_QUADRATIC: usize = 3;
    const N_COEFFICIENT_QUARTIC: usize = 5;
    let polynomial_u_sq: [f64; N_COEFFICIENT_QUADRATIC] = [(centre_r + centre_r_prime).powi(2) + h.powi(2), 2.0 * h, 1.0];
    let polynomial_d_sq: [f64; N_COEFFICIENT_QUADRATIC] = [(centre_r - centre_r_prime).powi(2) + h.powi(2), 2.0 * h, 1.0];
    let mut polynomial_u_sq_d_sq: [f64; N_COEFFICIENT_QUARTIC] = [0.0; N_COEFFICIENT_QUARTIC];
    for i_power in 0..N_COEFFICIENT_QUADRATIC {
        for j_power in 0..N_COEFFICIENT_QUADRATIC {
            polynomial_u_sq_d_sq[i_power + j_power] += polynomial_u_sq[i_power] * polynomial_d_sq[j_power];
        }
    }

    // The elliptic integrals at the two centres, as the pair `phi` and `theta`
    let phi_theta: EllipticPhiTheta = elliptic_phi_theta(4.0 * centre_r * centre_r_prime / polynomial_u_sq[0], polynomial_d_sq[0] / polynomial_u_sq[0]);

    // Step 1. The Taylor coefficients in `delta_h`, along the vertical line, of the Green's function and of its vertical derivative, as
    // the pair
    //     G = MU_0 * u * phi / 2,  and  T = -(1 / h) * d(G)/d(h) = MU_0 * u * theta / (2 * d ** 2)
    // which obey two differential equations with polynomial coefficients, where the height difference is `h + delta_h`:
    //     d(G)/d(delta_h) = -(h + delta_h) * T,  and  d(u ** 2 * d ** 2 * T)/d(delta_h) = -3 * (h + delta_h) * G
    series_g[0] = 0.5 * MU_0 * polynomial_u_sq[0].sqrt() * phi_theta.phi;
    series_t[0] = 0.5 * MU_0 * polynomial_u_sq[0].sqrt() * phi_theta.theta / polynomial_d_sq[0];
    for i_h in 0..n_series - 1 {
        // The coefficients of `delta_h ** i_h` in `(h + delta_h) * T` and in `(h + delta_h) * G`
        let mut h_times_t: f64 = h * series_t[i_h];
        let mut h_times_g: f64 = h * series_g[i_h];
        if i_h >= 1 {
            h_times_t += series_t[i_h - 1];
            h_times_g += series_g[i_h - 1];
        }
        series_g[i_h + 1] = -h_times_t / ((i_h + 1) as f64);
        // The coefficient of `delta_h ** (i_h + 1)` in `u ** 2 * d ** 2 * T`, and then the division by the quartic
        let mut value_t: f64 = -3.0 * h_times_g / ((i_h + 1) as f64);
        for i_power in 1..N_COEFFICIENT_QUARTIC {
            if i_h + 1 >= i_power {
                value_t -= polynomial_u_sq_d_sq[i_power] * series_t[i_h + 1 - i_power];
            }
        }
        series_t[i_h + 1] = value_t / polynomial_u_sq_d_sq[0];
    }
    for i_h in 0..n_series {
        derivative[0][0][i_h] = factorial[i_h] * series_g[i_h];
    }

    // Step 2. On the vertical line, the first derivative with respect to the first point's major radius, from
    //     2 * r * d(G)/d(r) = G + (r_prime ** 2 + (h + delta_h) ** 2 - r ** 2) * T
    // in which `r_prime ** 2 - r ** 2` is formed as a product, which keeps its accuracy when the two major radii are close
    for i_h in 0..n_series - 1 {
        let mut value_d_r: f64 = series_g[i_h] + (h.powi(2) - (centre_r + centre_r_prime) * (centre_r - centre_r_prime)) * series_t[i_h];
        if i_h >= 1 {
            value_d_r += 2.0 * h * series_t[i_h - 1];
        }
        if i_h >= 2 {
            value_d_r += series_t[i_h - 2];
        }
        derivative[1][0][i_h] = factorial[i_h] * value_d_r / (2.0 * centre_r);
    }

    // Step 3. The higher derivatives with respect to the first point's major radius, from the equation which the Green's function obeys
    // in the first point's coordinates,
    //     r * (d2(G)/d(r2) + d2(G)/d(h2)) - d(G)/d(r) = 0
    // differentiated `i_r` times with respect to `r` and `i_h` times with respect to `h`:
    //     r * (derivative[i_r + 2][0][i_h] + derivative[i_r][0][i_h + 2])
    //         + (i_r - 1) * derivative[i_r + 1][0][i_h] + i_r * derivative[i_r - 1][0][i_h + 2] = 0
    for i_r in 0..n_multipole.saturating_sub(1) {
        for i_h in 0..n_multipole - 1 - i_r {
            let mut value_r: f64 = ((i_r as f64) - 1.0) * derivative[i_r + 1][0][i_h];
            if i_r >= 1 {
                value_r += (i_r as f64) * derivative[i_r - 1][0][i_h + 2];
            }
            derivative[i_r + 2][0][i_h] = -derivative[i_r][0][i_h + 2] - value_r / centre_r;
        }
    }

    // Step 4. The first derivative with respect to the second point's major radius, from Euler's theorem: the Green's function is
    // homogeneous of degree 1 in `(r, r_prime, h)`,
    //     r * d(G)/d(r) + r_prime * d(G)/d(r_prime) + h * d(G)/d(h) = G
    // differentiated `i_r` times with respect to `r` and `i_h` times with respect to `h`:
    //     r * derivative[i_r + 1][0][i_h] + r_prime * derivative[i_r][1][i_h] + h * derivative[i_r][0][i_h + 1]
    //         + (i_r + i_h - 1) * derivative[i_r][0][i_h] = 0
    for i_r in 0..n_multipole {
        for i_h in 0..n_multipole - i_r {
            derivative[i_r][1][i_h] =
                -(centre_r * derivative[i_r + 1][0][i_h] + h * derivative[i_r][0][i_h + 1] + (((i_r + i_h) as f64) - 1.0) * derivative[i_r][0][i_h])
                    / centre_r_prime;
        }
    }

    // Step 5. The higher derivatives with respect to the second point's major radius, from the equation which the Green's function obeys
    // in the second point's coordinates,
    //     r_prime * (d2(G)/d(r_prime2) + d2(G)/d(h2)) - d(G)/d(r_prime) = 0
    // differentiated `i_r_prime` times with respect to `r_prime`, as the first point's equation is in step 3. The first major radius is
    // not in this equation, so the recurrence is the same for every `i_r`
    for i_r_prime in 0..n_multipole.saturating_sub(1) {
        for i_r in 0..n_multipole - 1 - i_r_prime {
            for i_h in 0..n_multipole - 1 - i_r_prime - i_r {
                let mut value_r_prime: f64 = ((i_r_prime as f64) - 1.0) * derivative[i_r][i_r_prime + 1][i_h];
                if i_r_prime >= 1 {
                    value_r_prime += (i_r_prime as f64) * derivative[i_r][i_r_prime - 1][i_h + 2];
                }
                derivative[i_r][i_r_prime + 2][i_h] = -derivative[i_r][i_r_prime][i_h + 2] - value_r_prime / centre_r_prime;
            }
        }
    }

    // Sum the derivatives against the averages over both parallelograms. The term of order 0 is the mutual inductance of two filaments
    // at the centres, and the others are small corrections to it. The corrections are added up on their own, and to the first term at
    // the end, so that only the last addition is rounded at the size of the whole. This also makes the result the same to the last bit,
    // for almost every pair, when the two parallelograms are exchanged
    let mut correction: f64 = 0.0;
    for i_r in 0..n_series {
        let moment_scaled_row: &[f64] = &moment_scaled[i_r];
        // A parallelogram's moments of odd order are zero, which leaves every second term of `moment_pair`: `i_z` has the parity of
        // `i_r`, and then the second parallelogram's moment is of even order too, because `i_r + i_r_prime + i_h` is even
        let i_z_start: usize = i_r % 2;
        for i_r_prime in 0..n_series - i_r {
            let moment_scaled_prime_row: &[f64] = &moment_scaled_prime[i_r_prime];
            let derivative_row: &[f64] = &derivative[i_r][i_r_prime];
            for i_h in 0..n_series - i_r - i_r_prime {
                let order: usize = i_r + i_r_prime + i_h;
                if order.is_multiple_of(2) && order >= 2 {
                    // The terms which are left all have the same sign, `(-1) ** (i_h - i_z)`, as their `i_z` has the parity of `i_r`
                    let sign: f64 = if (i_h + i_r).is_multiple_of(2) { 1.0 } else { -1.0 };
                    let n_term: usize = (i_h + 2 - i_z_start) / 2;
                    let mut moment_pair: f64 = 0.0;
                    for i_term in 0..n_term {
                        let i_z: usize = i_z_start + 2 * i_term;
                        moment_pair += moment_scaled_row[i_z] * moment_scaled_prime_row[i_h - i_z];
                    }
                    correction += sign * derivative_row[i_h] * moment_pair;
                }
            }
        }
    }

    derivative[0][0][0] + correction
}

/// What `pair_series_coefficients` returns: everything which the near-field formula for a pair of conductors needs and which depends on
/// the orders of the series only.
///
/// The two tables hold one number for each `(i_radial, i_a, i_b)` with `i_a >= i_b` and `i_radial + i_a + i_b <= n_order_max`; both are
/// symmetric in `i_a` and `i_b`. The entry is at `i_radial * n_half_triangle + half_rows_start[i_a + i_b] + i_b`: for each `i_radial`, the
/// rows of equal degree `i_a + i_b` follow each other, and a row holds `i_b = 0, 1, ..., (i_a + i_b) / 2`.
struct PairSeriesCoefficients {
    /// The highest order kept in the expansion about the thin ring, and in the expansion of the major radius about `centre_r_mean`
    n_thin_ring: usize,
    /// The highest total order of a term, `i_radial + i_a + i_b`: `2 * n_elliptic + 2 * n_thin_ring`
    n_order_max: usize,
    /// The coefficient of the double integral without the logarithm of `xi_prime ** i_radial * rho ** i_a * conj(rho) ** i_b` times the two
    /// distances, shape = [(n_thin_ring + 1) * n_half_triangle], [dimensionless]
    coefficient_plain: Vec<f64>,
    /// The coefficient of the same double integral with the factor `ln|rho|`, which is subtracted,
    /// shape = [(n_thin_ring + 1) * n_half_triangle], [dimensionless]
    coefficient_ln: Vec<f64>,
    /// How many `(i_a, i_b)` with `i_a >= i_b` and `i_a + i_b <= n_order_max` there are: the length of the tables for one `i_radial`
    n_half_triangle: usize,
    /// Where the row of each degree `i_a + i_b` starts in the tables for one `i_radial`, shape = [n_order_max + 2]; the last one is
    /// `n_half_triangle`
    half_rows_start: Vec<usize>,
    /// `binomial[(n, k)]` is the binomial coefficient of `n` over `k`, shape = [n_thin_ring + 1, n_thin_ring + 1], [dimensionless]
    binomial: Array2<f64>,
    /// A term which is smaller than this, compared with the largest term, is left out: `NEGLIGIBLE_TERM`. With zero nothing is left out
    negligible_term: f64,
}

/// The coefficients of the near-field formula for the mutual inductance of a pair of conductors. They are pure numbers, which depend on
/// the orders of the series only.
///
/// This is the note's "Near each other: one series in the positions of both points" and "Two divergence theorems". Every length is in
/// units of `2 * centre_r_mean`, where `centre_r_mean` is the mean major radius of the two conductors' centres. With `rho` the position of
/// a point of the second conductor from a point of the first, as a complex number, and `xi = (r - centre_r_mean) / centre_r_mean` for the
/// point of the first conductor, the Green's function of a filament expanded about the point of the first conductor, and that point's
/// major radius expanded about `centre_r_mean`, is
///
/// ```text
/// G / (MU_0 * centre_r_mean) = sum over m, a, b of  xi ** m * rho ** a * conj(rho) ** b * (kernel_plain - kernel_ln * ln|rho|)
/// kernel_plain = binomial_radial[m] * (table_p[(a, b)] * ln(4) + table_q[(a, b)]) + binomial_radial_derivative[m] * table_p[(a, b)]
/// kernel_ln    = binomial_radial[m] * table_p[(a, b)]
/// ```
///
/// where `binomial_radial[m]` is the binomial coefficient of `1 - a - b` over `m`, `binomial_radial_derivative[m]` is its derivative with
/// respect to its upper argument, and `table_p` and `table_q` are the tables of `maxwell_series_coefficients`.
///
/// The integral of a term over the second conductor is a sum over its edges of the distance from the point of the first conductor to the
/// edge's line times the integral along the edge, divided by `a + b + 2`: the term is homogeneous of degree `a + b` about that point.
/// Seen from a point of such an edge, where the radial offset is `xi_prime`, `xi = xi_prime - (rho + conj(rho))`, so `xi ** m` is a sum
/// over `j` of terms with `xi_prime ** (m - j)` and `j` more powers of `rho` and `conj(rho)`, each of which is, with the distance,
/// homogeneous of degree `a + b + j + 1` about that point; its integral over the first conductor is a sum over the first conductor's edges
/// in the same way, divided by `a + b + j + 3`. With the logarithm each division also subtracts its reciprocal from the logarithm. So
///
/// ```text
/// integral over both conductors of G / (MU_0 * centre_r_mean) = sum over the 16 pairs of edges, and over n, a, b of
///     coefficient_plain[(n, a, b)] * K[(n, a, b)] - coefficient_ln[(n, a, b)] * K_ln[(n, a, b)]
/// ```
///
/// where `K[(n, a, b)]` is the double integral over the pair of edges of `xi_prime ** n * rho ** a * conj(rho) ** b` times the two
/// distances, from the point of each edge to the line of the other, and `K_ln[(n, a, b)]` is the same with the factor `ln|rho|`.
///
/// The truncated series is not symmetric in its two points. The same series expanded about the point of the second conductor has
/// `xi_prime ** m` and `(-rho) ** a * conj(-rho) ** b`. For the same two divergence theorems `xi_prime` is first written as
/// `xi + (rho + conj(rho))`, and then `xi` as before; the two binomial sums collapse into a finite difference of `1 / (a + b + j + 2)`,
/// which is `j! / ((a + b + 2) * (a + b + 3) * ... * (a + b + j + 2))` with alternating sign. The coefficients here are the mean of the
/// two expansions, so that the mutual inductance does not depend on which of the two conductors is the first.
///
/// # Arguments
/// * `n_elliptic` - The highest power of `1 - k ** 2` kept in the series of the elliptic integrals
/// * `n_thin_ring` - The highest order kept in the expansion about the thin ring, and in the expansion of the major radius about
///   `centre_r_mean`
///
/// # Returns
/// * `coefficients` - The two tables of coefficients, `coefficient_plain` and `coefficient_ln`, how they are laid out, and the binomial
///   coefficients which turn the powers of `xi_prime` into powers of the position along an edge
fn pair_series_coefficients(n_elliptic: usize, n_thin_ring: usize) -> PairSeriesCoefficients {
    let maxwell_coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(n_elliptic, n_thin_ring);
    let table_p: &Array2<f64> = &maxwell_coefficients.table_p;
    let table_q: &Array2<f64> = &maxwell_coefficients.table_q;
    let degree_max: usize = 2 * n_elliptic + n_thin_ring;
    let n_order_max: usize = degree_max + n_thin_ring;
    let ln_4: f64 = 4.0_f64.ln();

    // binomial[(n, k)] = the binomial coefficient of `n` over `k`, from Pascal's triangle
    let mut binomial: Array2<f64> = Array2::zeros((n_thin_ring + 1, n_thin_ring + 1));
    for i_power in 0..n_thin_ring + 1 {
        binomial[(i_power, 0)] = 1.0;
        for j_power in 1..i_power + 1 {
            binomial[(i_power, j_power)] = binomial[(i_power - 1, j_power - 1)] + binomial[(i_power - 1, j_power)];
        }
    }

    // Where the row of each degree starts: the row of the degree `i_degree` holds `i_b = 0, ..., i_degree / 2`
    let mut half_rows_start: Vec<usize> = vec![0; n_order_max + 2];
    for i_degree in 0..n_order_max + 1 {
        half_rows_start[i_degree + 1] = half_rows_start[i_degree] + i_degree / 2 + 1;
    }
    let n_half_triangle: usize = half_rows_start[n_order_max + 1];

    // The terms are added up, so the tables start from zero, which is what is left where the truncation keeps no term
    let mut coefficient_plain: Vec<f64> = vec![0.0; (n_thin_ring + 1) * n_half_triangle];
    let mut coefficient_ln: Vec<f64> = vec![0.0; (n_thin_ring + 1) * n_half_triangle];
    let mut binomial_radial: Vec<f64> = vec![f64::NAN; n_thin_ring + 1];
    let mut binomial_radial_derivative: Vec<f64> = vec![f64::NAN; n_thin_ring + 1];
    for i_degree in 0..degree_max + 1 {
        let degree: f64 = i_degree as f64;

        // The radial series of the terms of this degree: with `r = centre_r_mean * (1 + xi)`,
        // (1 + xi) ** (1 - degree)               = sum over m of  binomial_radial[m] * xi ** m
        // (1 + xi) ** (1 - degree) * ln(1 + xi)  = sum over m of  binomial_radial_derivative[m] * xi ** m
        let nu: f64 = 1.0 - degree;
        binomial_radial[0] = 1.0;
        binomial_radial_derivative[0] = 0.0;
        for i_radial in 1..n_thin_ring + 1 {
            let order_radial: f64 = i_radial as f64;
            binomial_radial[i_radial] = binomial_radial[i_radial - 1] * (nu - order_radial + 1.0) / order_radial;
            binomial_radial_derivative[i_radial] =
                (binomial_radial_derivative[i_radial - 1] * (nu - order_radial + 1.0) + binomial_radial[i_radial - 1]) / order_radial;
        }

        for i_b in 0..i_degree + 1 {
            let i_a: usize = i_degree - i_b;
            if table_p[(i_a, i_b)] == 0.0 && table_q[(i_a, i_b)] == 0.0 {
                continue;
            }
            for i_radial in 0..n_thin_ring + 1 {
                // The term `xi ** i_radial * rho ** i_a * conj(rho) ** i_b * (kernel_plain - kernel_ln * ln|rho|)` of the Green's function
                let kernel_plain: f64 =
                    binomial_radial[i_radial] * (table_p[(i_a, i_b)] * ln_4 + table_q[(i_a, i_b)]) + binomial_radial_derivative[i_radial] * table_p[(i_a, i_b)];
                let kernel_ln: f64 = binomial_radial[i_radial] * table_p[(i_a, i_b)];

                // `j_radial` of the `i_radial` factors `xi = xi_prime - (rho + conj(rho))` give `-(rho + conj(rho))`, of which `i_power`
                // give `rho`
                // Expanded about the point of the second conductor, the first binomial sum gives the divisions by `a + b + j' + 2` for
                // `j' <= j_radial`, which add up to `finite_difference = j! / ((a + b + 2) * ... * (a + b + j + 2))`, and `harmonic` is
                // the sum of the reciprocals of those factors
                let mut finite_difference: f64 = 1.0 / (degree + 2.0);
                let mut harmonic: f64 = 1.0 / (degree + 2.0);
                for j_radial in 0..i_radial + 1 {
                    let order_exchanged: f64 = j_radial as f64;
                    if j_radial >= 1 {
                        finite_difference *= order_exchanged / (degree + 2.0 + order_exchanged);
                        harmonic += 1.0 / (degree + 2.0 + order_exchanged);
                    }

                    // Expanded about the point of the first conductor: the divisions by `a + b + 2` and `a + b + j + 3`, and what they
                    // subtract from the logarithm
                    let sign: f64 = if j_radial % 2 == 0 { 1.0 } else { -1.0 };
                    let weight: f64 = sign * binomial[(i_radial, j_radial)] / ((degree + 2.0) * (degree + order_exchanged + 3.0));
                    let ln_constant: f64 = 1.0 / (degree + 2.0) + 1.0 / (degree + order_exchanged + 3.0);

                    // Expanded about the point of the second conductor
                    let sign_prime: f64 = if (i_degree + j_radial) % 2 == 0 { 1.0 } else { -1.0 };
                    let weight_prime: f64 = sign_prime * binomial[(i_radial, j_radial)] * finite_difference / (degree + order_exchanged + 3.0);
                    let ln_constant_prime: f64 = harmonic + 1.0 / (degree + order_exchanged + 3.0);

                    // The mean of the two, added to the term with `i_power` more powers of `rho` and `j_radial - i_power` more powers of
                    // `conj(rho)`. The tables are symmetric, and only their half with at least as many powers of `rho` is kept
                    let term_plain: f64 =
                        0.5 * ((kernel_plain + kernel_ln * ln_constant) * weight + (kernel_plain + kernel_ln * ln_constant_prime) * weight_prime);
                    let term_ln: f64 = 0.5 * kernel_ln * (weight + weight_prime);
                    for i_power in 0..j_radial + 1 {
                        let j_a: usize = i_a + i_power;
                        let j_b: usize = i_b + j_radial - i_power;
                        if j_a >= j_b {
                            let i_entry: usize = (i_radial - j_radial) * n_half_triangle + half_rows_start[j_a + j_b] + j_b;
                            coefficient_plain[i_entry] += term_plain * binomial[(j_radial, i_power)];
                            coefficient_ln[i_entry] += term_ln * binomial[(j_radial, i_power)];
                        }
                    }
                }
            }
        }
    }

    PairSeriesCoefficients {
        n_thin_ring,
        n_order_max,
        coefficient_plain,
        coefficient_ln,
        n_half_triangle,
        half_rows_start,
        binomial,
        negligible_term: NEGLIGIBLE_TERM,
    }
}

/// The work arrays of the near-field formula for a pair of conductors. They are allocated once for each thread, and overwritten for every
/// pair; nothing in them is carried from one pair to the next.
///
/// They are vectors, because they are indexed in the innermost loops, where a `Vec` is quicker to index than an `Array1`. A table
/// over `(i_a, i_b)` is stored as the rows of equal degree `i_a + i_b`, one after the other, each with `i_b = 0, 1, ..., i_a + i_b`: the
/// entry is at `rows_start[i_a + i_b] + i_b`.
struct NearFieldPairWorkArrays {
    /// Where the row of each degree starts in a table over `(i_a, i_b)`, shape = [n_order_max + 5]
    rows_start: Vec<usize>,
    /// The length of a table over `(i_a, i_b)` up to the degree `n_order_max + 3`
    n_triangle: usize,
    /// `reciprocal[k] = 1 / k`, shape = [n_order_max + 5]; the first is not used
    reciprocal: Vec<f64>,
    /// The powers of `sigma` at the start of an edge, shape = [n_order_max + 5], [dimensionless]
    sigma_start_power: Vec<c64>,
    /// The powers of `sigma` at the end of an edge, shape = [n_order_max + 5], [dimensionless]
    sigma_end_power: Vec<c64>,
    /// `difference[k] = (sigma_end ** (k + 1) - sigma_start ** (k + 1)) / length`, shape = [n_order_max + 4], [dimensionless]
    difference: Vec<c64>,
    /// The powers of the direction of an edge of the second conductor, `phase_prime_power[k] = exp(i * k * phi_e')`,
    /// shape = [n_order_max + 1], [dimensionless]
    phase_prime_power: Vec<c64>,
    /// The powers of `phase_relative` for a pair of edges, negative powers too: `phase_relative_power[n_order_max + 4 + k]` is
    /// `exp(i * k * phi_ee')` for `k` from `-(n_order_max + 4)` to `n_order_max + 4`, shape = [2 * n_order_max + 9], [dimensionless]
    phase_relative_power: Vec<c64>,
    /// The integrals along an edge of the first conductor of `sigma ** i_a * conj(sigma) ** i_b`, seen from a corner of the second: one
    /// table for each pair of a corner and an edge, shape = [N_CORNER_EDGE * n_triangle], [dimensionless]
    integral_plain: Vec<c64>,
    /// The same integrals with the factor `ln(sigma)`, the complex logarithm, shape = [N_CORNER_EDGE * n_triangle], [dimensionless]
    integral_ln: Vec<c64>,
    /// For one edge of the second conductor, the sum over the edges of the first of `phase_relative ** (i_b - i_a + 1) * integral_plain`.
    /// Two tables: first seen from the start and from the end of the edge, and then for the even and for the odd powers of `position`,
    /// shape = [N_END * n_triangle], [dimensionless]
    boundary_plain_plus: Vec<c64>,
    /// The same with `phase_relative ** (i_b - i_a - 1)`, shape = [N_END * n_triangle], [dimensionless]
    boundary_plain_minus: Vec<c64>,
    /// The same as `boundary_plain_plus` with `integral_ln`, which has the complex logarithm, and after its two tables one more for each
    /// edge of the first conductor whose line the edge of the second crosses, shape = [(N_END + N_EDGE) * n_triangle], [dimensionless]
    boundary_ln_plus: Vec<c64>,
    /// The same as `boundary_ln_plus` with `phase_relative ** (i_b - i_a - 1)`, shape = [(N_END + N_EDGE) * n_triangle], [dimensionless]
    boundary_ln_minus: Vec<c64>,
    /// For one edge of the second conductor, the sum over the edges of the first of `phase_relative ** (i_b - i_a + 1)` times the double
    /// integral of `position ** i_radial * sigma ** i_a * conj(sigma) ** i_b`. Two tables, for an even and for an odd `i_radial`: the one
    /// which is being calculated, and the one before it, shape = [2 * n_triangle], [dimensionless]
    sum_plain_plus: Vec<c64>,
    /// The same with `phase_relative ** (i_b - i_a - 1)`, shape = [2 * n_triangle], [dimensionless]
    sum_plain_minus: Vec<c64>,
    /// The same as `sum_plain_plus` with the factor `ln(sigma)`, the complex logarithm, shape = [2 * n_triangle], [dimensionless]
    sum_ln_plus: Vec<c64>,
    /// The same as `sum_plain_minus` with the factor `ln(sigma)`, the complex logarithm, shape = [2 * n_triangle], [dimensionless]
    sum_ln_minus: Vec<c64>,
    /// One row of `sum_ln_plus` with the real logarithm, `ln|sigma|`, in place of the complex one, shape = [n_order_max + 3], [dimensionless]
    double_ln_plus: Vec<c64>,
    /// One row of `sum_ln_minus` with the real logarithm, shape = [n_order_max + 3], [dimensionless]
    double_ln_minus: Vec<c64>,
    /// The coefficients of the double integrals without the logarithm for one edge of the second conductor, with the powers of the position
    /// along the edge in place of the powers of `xi_prime`, laid out as `PairSeriesCoefficients::coefficient_plain`,
    /// shape = [(n_thin_ring + 1) * n_half_triangle], [dimensionless]
    edge_coefficient_plain: Vec<f64>,
    /// The same for the double integrals with the logarithm, shape = [(n_thin_ring + 1) * n_half_triangle], [dimensionless]
    edge_coefficient_ln: Vec<f64>,
}

impl NearFieldPairWorkArrays {
    /// The work arrays of the near-field formula for a pair of conductors, for the orders `n_elliptic` and `n_thin_ring` of the series.
    fn new(n_elliptic: usize, n_thin_ring: usize) -> Self {
        let n_order_max: usize = 2 * n_elliptic + 2 * n_thin_ring;

        // The integrals along an edge are needed up to the degree `n_order_max + 3`: the two distances raise the degree of a term by 2, and
        // each integration by parts along the other edge needs one degree more
        let mut rows_start: Vec<usize> = vec![0; n_order_max + 5];
        for i_degree in 0..n_order_max + 4 {
            rows_start[i_degree + 1] = rows_start[i_degree] + i_degree + 1;
        }
        let n_triangle: usize = rows_start[n_order_max + 4];
        let mut reciprocal: Vec<f64> = vec![f64::NAN; n_order_max + 5];
        for i_divisor in 1..n_order_max + 5 {
            reciprocal[i_divisor] = 1.0 / (i_divisor as f64);
        }

        // How many `(i_a, i_b)` with `i_a >= i_b` and `i_a + i_b <= n_order_max` there are
        let mut n_half_triangle: usize = 0;
        for i_degree in 0..n_order_max + 1 {
            n_half_triangle += i_degree / 2 + 1;
        }

        let complex_nan: c64 = c64::new(f64::NAN, f64::NAN);
        NearFieldPairWorkArrays {
            rows_start,
            n_triangle,
            reciprocal,
            sigma_start_power: vec![complex_nan; n_order_max + 5],
            sigma_end_power: vec![complex_nan; n_order_max + 5],
            difference: vec![complex_nan; n_order_max + 4],
            phase_prime_power: vec![complex_nan; n_order_max + 1],
            phase_relative_power: vec![complex_nan; 2 * n_order_max + 9],
            integral_plain: vec![complex_nan; N_CORNER_EDGE * n_triangle],
            integral_ln: vec![complex_nan; N_CORNER_EDGE * n_triangle],
            boundary_plain_plus: vec![complex_nan; N_END * n_triangle],
            boundary_plain_minus: vec![complex_nan; N_END * n_triangle],
            boundary_ln_plus: vec![complex_nan; (N_END + N_EDGE) * n_triangle],
            boundary_ln_minus: vec![complex_nan; (N_END + N_EDGE) * n_triangle],
            sum_plain_plus: vec![complex_nan; 2 * n_triangle],
            sum_plain_minus: vec![complex_nan; 2 * n_triangle],
            sum_ln_plus: vec![complex_nan; 2 * n_triangle],
            sum_ln_minus: vec![complex_nan; 2 * n_triangle],
            double_ln_plus: vec![complex_nan; n_order_max + 3],
            double_ln_minus: vec![complex_nan; n_order_max + 3],
            edge_coefficient_plain: vec![f64::NAN; (n_thin_ring + 1) * n_half_triangle],
            edge_coefficient_ln: vec![f64::NAN; (n_thin_ring + 1) * n_half_triangle],
        }
    }
}

/// The near-field formula: the mutual inductance between two toroidal conductors with parallelogram cross-sections, each with a uniform
/// current density, which are close together compared with their major radius.
///
/// The two conductors can be apart, touching, overlapping, one inside the other, or the same conductor, which gives the self-inductance.
/// The two sides of each can be given in either order.
///
/// Every length is in units of `2 * centre_r_mean`, where `centre_r_mean` is the mean major radius of the two conductors' centres. With
/// the coefficients of `pair_series_coefficients`, the note's "The result" is
///
/// ```text
/// area * area_prime * M = MU_0 * centre_r_mean * (2 * centre_r_mean) ** 4 * sum over the 16 pairs of edges (e, e'), and over n, a, b of
///     coefficient_plain[(n, a, b)] * K[(n, a, b)] - coefficient_ln[(n, a, b)] * K_ln[(n, a, b)]
/// K[(n, a, b)] = integral along e', integral along e, of  d_e * d_e' * xi_prime ** n * rho ** a * conj(rho) ** b,   K_ln: with the factor ln|rho|
/// ```
///
/// where `e` is an edge of the first conductor and `e'` an edge of the second, `rho` is the position of the point of `e'` from the point of
/// `e`, `xi_prime` is the radial offset of the point of `e'`, `d_e` is the distance from the point of `e'` to the line of `e`, and `d_e'` is
/// the distance from the point of `e` to the line of `e'`, each positive on the inner side of the edge.
///
/// The double integrals are those of the note's "The double integrals over a pair of edges". For one pair of edges,
/// `sigma = -rho * conj(phase) = s - i * d_e` is the position of the point of `e` from the point of `e'`, rotated so that `e` lies along
/// the real axis, and `position` is the distance along `e'` from its middle. With `phase = exp(i * phi_e)`,
/// `phase_prime = exp(i * phi_e')` and `phase_relative = phase_prime * conj(phase)`:
///
/// ```text
/// rho ** a * conj(rho) ** b = (-1) ** (a + b) * phase ** (a - b) * sigma ** a * conj(sigma) ** b
/// d_e * d_e' = (conj(phase_relative) * sigma ** 2 - 2 * re(phase_relative) * sigma * conj(sigma) + phase_relative * conj(sigma) ** 2) / 4
/// xi_prime = xi_middle + 2 * re(phase_prime) * position
/// ```
///
/// so that `K` and `K_ln` are sums of the double integrals
///
/// ```text
/// X[(m, a, b)] = integral along e' of  position ** m * (integral along e of  sigma ** a * conj(sigma) ** b)
/// Y[(m, a, b)] = the same with the factor ln|sigma|
/// ```
///
/// which are polynomials and logarithms of the positions of the edges' ends. They are calculated by integration by parts: along `e`,
/// which gives the integrals along `e` seen from one point, `integral_plain` and `integral_ln`, and along `e'`, which needs those seen
/// from the two ends of `e'` only. The logarithm in `integral_ln` is the complex one, `ln(sigma)`, whose derivative along `e'` is a single
/// term. With `Y~` for `Y` with the complex logarithm, `Y[(m, a, b)] = (Y~[(m, a, b)] + conj(Y~[(m, b, a)])) / 2`, and with `[f]` for `f`
/// at the end of an edge minus `f` at its start:
///
/// ```text
/// (a + 1) * integral_plain[(a, b)] = [sigma ** (a + 1) * conj(sigma) ** b] - b * integral_plain[(a + 1, b - 1)]
/// (a + 1) * integral_ln[(a, b)]    = [sigma ** (a + 1) * conj(sigma) ** b * ln(sigma)] - b * integral_ln[(a + 1, b - 1)] - integral_plain[(a, b)]
/// phase_relative * (a + 1) * X[(m, a, b)]  = -[position ** m * integral_plain[(a + 1, b)]] + m * X[(m - 1, a + 1, b)]
///                                            - conj(phase_relative) * b * X[(m, a + 1, b - 1)]
/// phase_relative * (a + 1) * Y~[(m, a, b)] = the same with `integral_ln` and `Y~`, - phase_relative * X[(m, a, b)]
/// ```
///
/// The last two are not used for one pair of edges at a time. Multiplied by a power of `phase_relative`, which rotates `sigma` from the
/// frame of `e` into the frame of `e'`, they have real coefficients, which are the same for every edge `e`. So for each `e'` the edges `e`
/// are added up first, and the recurrences are used once, for the sums:
///
/// ```text
/// sum_plain_plus[(m, a, b)]  = sum over e of  phase_relative ** (b - a + 1) * X[(m, a, b)]
/// sum_plain_minus[(m, a, b)] = sum over e of  phase_relative ** (b - a - 1) * X[(m, a, b)]
/// sum_ln_plus, sum_ln_minus:           the same with `Y~` in place of `X`
/// boundary_plain_plus, boundary_plain_minus, boundary_ln_plus, boundary_ln_minus:
///                                      the same with `integral_plain[(a, b)]` and `integral_ln[(a, b)]`, seen from one end of e'
/// (a + 1) * sum_plain_plus[(m, a, b)] = -[position ** m * boundary_plain_plus[(a + 1, b)]] + m * sum_plain_plus[(m - 1, a + 1, b)]
///                                       - b * sum_plain_plus[(m, a + 1, b - 1)]
/// (a + 1) * sum_ln_plus[(m, a, b)]    = the same with `boundary_ln_plus` and `sum_ln_plus`, - sum_plain_plus[(m, a, b)]
/// ```
///
/// and the same for the two with `minus`. `sum_plain_minus[(m, b, a)]` is the complex conjugate of `sum_plain_plus[(m, a, b)]`. The
/// product of the two distances and the power of `phase` then leave, for each `e'`:
///
/// ```text
/// sum over e of  the integral along e', integral along e, of  d_e * d_e' * position ** m * rho ** a * conj(rho) ** b =
///     (-1) ** (a + b) * phase_prime ** (a - b) / 4 * (sum_plain_plus[(m, a + 2, b)] - sum_plain_plus[(m, a + 1, b + 1)]
///                                                     - sum_plain_minus[(m, a + 1, b + 1)] + sum_plain_minus[(m, a, b + 2)])
/// ```
///
/// and the same with the factor `ln|rho|` and with `double_ln_plus` and `double_ln_minus`, which have the real logarithm:
///
/// ```text
/// double_ln_plus[(m, a, b)]  = (sum_ln_plus[(m, a, b)] + conj(sum_ln_minus[(m, b, a)])) / 2
/// double_ln_minus[(m, a, b)] = (sum_ln_minus[(m, a, b)] + conj(sum_ln_plus[(m, b, a)])) / 2
/// ```
///
/// The complex logarithm has to be continuous over the pair of edges. Where `e'` stays on one side of the line of `e`, `sigma` stays in one
/// half plane, and the branch whose cut is in the other half plane is continuous. Where `e'` crosses the line of `e` it is cut into two
/// parts, each with its own branch. At the crossing the two branches differ by a whole turn for the points of `e` with a negative `s`, and
/// not at all for the others, so what the crossing adds to `-[position ** m * integral_ln[(a, b)]]` is `2 * pi * i * position ** m` times
/// the integral of `s ** (a + b)` over the part of `e` with a negative `s`, which needs no logarithm. The choice of the branch cancels in
/// `Y`. Where `e'` lies on the line of `e` the distance `d_e` is zero, and the pair of edges adds nothing.
///
/// The series converges when the pair is small compared with its major radius: with `t = (the largest distance between a corner of one
/// conductor and a corner of the other) / (2 * the smallest major radius of the eight corners)` it needs `t < 1/2`, and with
/// `n_thin_ring = 2 * n_elliptic` its relative error is below `0.01 * t ** (n_thin_ring + 1)` (measured). At the lowest order,
/// `n_elliptic = n_thin_ring = 0`, it is `MU_0 * centre_r_mean * (ln(8 * centre_r_mean) - 2 - <<ln d>>)`, with `<<ln d>>` the logarithm of
/// the geometric mean distance of the two cross-sections.
///
/// Two kinds of terms are left out because they cannot change the sum in double precision, which the tests check against the whole sum:
/// * the terms whose total order `n + a + b` is above `n_order`. A term of the total order `k` is of the size `t ** k`. The truncation
///   keeps every term up to the total order `n_thin_ring`, and some of the terms of each higher order, up to
///   `2 * n_elliptic + 2 * n_thin_ring`; what it keeps of those orders is measured to fall like `(PARTIAL_ORDER_GROWTH * t)` from one order
///   to the next, starting from `t ** (n_thin_ring + 1)`.
/// * for each edge `e'`, the powers of `position` above `n_radial`. They are measured to be bounded by `radial_ratio ** i_radial` times the
///   power 0, with `radial_ratio` the largest change of `xi_prime` along the edge over `1 - t - |xi_middle|`. For a vertical edge
///   `xi_prime` is constant, and only the power 0 is left.
///
/// The rounding error is a few times 1e-16 of the mutual inductance, except where two edges are long compared with the distance between
/// their lines: the product of the two distances is a difference of double integrals which are larger than it by the square of that
/// ratio, and the rounding error grows by as much. For two thin rectangles of ST40's inboard wall, one on top of the other, which are
/// together 15 times as long as wide, it is 2e-14 (measured), and for a rectangle 240 times as long as wide 1e-12.
///
/// # Arguments
/// * `oblique_r` - R coordinate of the first conductor's reference corner, [metre]
/// * `oblique_z` - Z coordinate of the first conductor's reference corner, [metre]
/// * `side_alpha_r` - R component of the first conductor's first side, `oblique_length_alpha * cos(oblique_alpha)`, [metre]
/// * `side_alpha_z` - Z component of the first conductor's first side, `oblique_length_alpha * sin(oblique_alpha)`, [metre]
/// * `side_beta_r` - R component of the first conductor's second side, `-oblique_length_beta * sin(oblique_beta)`, [metre]
/// * `side_beta_z` - Z component of the first conductor's second side, `oblique_length_beta * cos(oblique_beta)`, [metre]
/// * `oblique_r_prime` - R coordinate of the second conductor's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the second conductor's reference corner, [metre]
/// * `side_alpha_r_prime` - R component of the second conductor's first side, [metre]
/// * `side_alpha_z_prime` - Z component of the second conductor's first side, [metre]
/// * `side_beta_r_prime` - R component of the second conductor's second side, [metre]
/// * `side_beta_z_prime` - Z component of the second conductor's second side, [metre]
/// * `coefficients` - The coefficients of the double integrals, from `pair_series_coefficients`, whose orders are the orders of the series
/// * `work` - The work arrays, from `NearFieldPairWorkArrays::new` with the same orders, which are overwritten
///
/// The two conductors have already been checked: each has an area, and is wholly at a positive major radius.
///
/// # Returns
/// * `mutual_inductance` - The mutual inductance between the two conductors, [henry]. Exchanging the two conductors changes it by its
///   rounding only
#[allow(clippy::too_many_arguments)]
fn mutual_inductance_parallelogram_near_field(
    oblique_r: f64,
    oblique_z: f64,
    side_alpha_r: f64,
    side_alpha_z: f64,
    side_beta_r: f64,
    side_beta_z: f64,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    side_alpha_r_prime: f64,
    side_alpha_z_prime: f64,
    side_beta_r_prime: f64,
    side_beta_z_prime: f64,
    coefficients: &PairSeriesCoefficients,
    work: &mut NearFieldPairWorkArrays,
) -> f64 {
    const N_CONDUCTOR: usize = 2;
    let n_thin_ring: usize = coefficients.n_thin_ring;
    let n_order_max: usize = coefficients.n_order_max;
    let n_half_triangle: usize = coefficients.n_half_triangle;
    let half_rows_start: &[usize] = &coefficients.half_rows_start;
    let binomial: &Array2<f64> = &coefficients.binomial;

    // The two conductors: index 0 is the first, and index 1 is the second, whose names end with `_prime`
    let conductors_oblique_r: [f64; N_CONDUCTOR] = [oblique_r, oblique_r_prime];
    let conductors_side_alpha_r: [f64; N_CONDUCTOR] = [side_alpha_r, side_alpha_r_prime];
    let conductors_side_alpha_z: [f64; N_CONDUCTOR] = [side_alpha_z, side_alpha_z_prime];
    let conductors_side_beta_r: [f64; N_CONDUCTOR] = [side_beta_r, side_beta_r_prime];
    let conductors_side_beta_z: [f64; N_CONDUCTOR] = [side_beta_z, side_beta_z_prime];

    // The four edges of each conductor, counter-clockwise, each as the position of its start from the reference corner and its vector. They
    // are built from the two sides themselves, not from differences of the corners' positions, which would be rounded
    let mut conductors_area: [f64; N_CONDUCTOR] = [f64::NAN; N_CONDUCTOR];
    let mut conductors_centre_r: [f64; N_CONDUCTOR] = [f64::NAN; N_CONDUCTOR];
    let mut conductors_edges_start_r: [[f64; N_EDGE]; N_CONDUCTOR] = [[f64::NAN; N_EDGE]; N_CONDUCTOR];
    let mut conductors_edges_start_z: [[f64; N_EDGE]; N_CONDUCTOR] = [[f64::NAN; N_EDGE]; N_CONDUCTOR];
    let mut conductors_edges_r: [[f64; N_EDGE]; N_CONDUCTOR] = [[f64::NAN; N_EDGE]; N_CONDUCTOR];
    let mut conductors_edges_z: [[f64; N_EDGE]; N_CONDUCTOR] = [[f64::NAN; N_EDGE]; N_CONDUCTOR];
    for i_conductor in 0..N_CONDUCTOR {
        let area_signed: f64 = conductors_side_alpha_r[i_conductor] * conductors_side_beta_z[i_conductor]
            - conductors_side_alpha_z[i_conductor] * conductors_side_beta_r[i_conductor];

        // Going along the first side and then the second has to be counter-clockwise, so the two sides are exchanged if it is not
        let is_counter_clockwise: bool = area_signed > 0.0;
        let first_r: f64 = if is_counter_clockwise {
            conductors_side_alpha_r[i_conductor]
        } else {
            conductors_side_beta_r[i_conductor]
        };
        let first_z: f64 = if is_counter_clockwise {
            conductors_side_alpha_z[i_conductor]
        } else {
            conductors_side_beta_z[i_conductor]
        };
        let second_r: f64 = if is_counter_clockwise {
            conductors_side_beta_r[i_conductor]
        } else {
            conductors_side_alpha_r[i_conductor]
        };
        let second_z: f64 = if is_counter_clockwise {
            conductors_side_beta_z[i_conductor]
        } else {
            conductors_side_alpha_z[i_conductor]
        };

        conductors_area[i_conductor] = area_signed.abs();
        conductors_centre_r[i_conductor] =
            conductors_oblique_r[i_conductor] + 0.5 * (conductors_side_alpha_r[i_conductor] + conductors_side_beta_r[i_conductor]);
        conductors_edges_start_r[i_conductor] = [0.0, first_r, first_r + second_r, second_r];
        conductors_edges_start_z[i_conductor] = [0.0, first_z, first_z + second_z, second_z];
        conductors_edges_r[i_conductor] = [first_r, second_r, -first_r, -second_r];
        conductors_edges_z[i_conductor] = [first_z, second_z, -first_z, -second_z];
    }

    // The major radius about which the major radii of the two conductors' points are expanded: the middle of the pair. From here on every
    // length is in units of `scale`, twice that major radius
    let centre_r_mean: f64 = 0.5 * (conductors_centre_r[0] + conductors_centre_r[1]);
    let scale: f64 = 2.0 * centre_r_mean;

    // The first conductor's reference corner from the second's is formed first, because that difference is exact
    let corner_delta_r: f64 = oblique_r - oblique_r_prime;
    let corner_delta_z: f64 = oblique_z - oblique_z_prime;

    // The expansion parameter of the pair: the largest distance between a point of one conductor and a point of the other, which is
    // between two corners, over twice the smallest major radius, which is at a corner
    let mut distance_max: f64 = 0.0;
    let mut corner_r_min: f64 = f64::INFINITY;
    for i_edge in 0..N_EDGE {
        corner_r_min = corner_r_min
            .min(oblique_r + conductors_edges_start_r[0][i_edge])
            .min(oblique_r_prime + conductors_edges_start_r[1][i_edge]);
        for i_edge_prime in 0..N_EDGE {
            let distance: f64 = (corner_delta_r + (conductors_edges_start_r[0][i_edge] - conductors_edges_start_r[1][i_edge_prime]))
                .hypot(corner_delta_z + (conductors_edges_start_z[0][i_edge] - conductors_edges_start_z[1][i_edge_prime]));
            distance_max = distance_max.max(distance);
        }
    }
    let t: f64 = distance_max / (2.0 * corner_r_min);

    // The highest total order which can change the sum: `order_bound` bounds what the orders above `i_order` add, compared with the sum
    let mut n_order: usize = n_order_max;
    let mut order_bound: f64 = 1.0;
    'order_loop: for i_order in 0..n_order_max {
        order_bound *= if i_order <= n_thin_ring { t } else { PARTIAL_ORDER_GROWTH * t };
        if order_bound < coefficients.negligible_term {
            n_order = i_order;
            break 'order_loop;
        }
    }

    // The two distances raise the degree of a term by 2: the double integrals are needed up to `i_radial + i_a + i_b = n_total`, and the
    // integrals along an edge up to the degree `n_total + 1`
    let n_total: usize = n_order + 2;

    // A distance from the line of an edge which is below the rounding of the largest coordinate is taken as zero
    let coordinate_max: f64 = oblique_r.abs().max(oblique_z.abs()).max(oblique_r_prime.abs()).max(oblique_z_prime.abs());
    let distance_tolerance: f64 = 8.0 * f64::EPSILON * coordinate_max / scale;

    let NearFieldPairWorkArrays {
        rows_start,
        n_triangle,
        reciprocal,
        sigma_start_power,
        sigma_end_power,
        difference,
        phase_prime_power,
        phase_relative_power,
        integral_plain,
        integral_ln,
        boundary_plain_plus,
        boundary_plain_minus,
        boundary_ln_plus,
        boundary_ln_minus,
        sum_plain_plus,
        sum_plain_minus,
        sum_ln_plus,
        sum_ln_minus,
        double_ln_plus,
        double_ln_minus,
        edge_coefficient_plain,
        edge_coefficient_ln,
    } = work;
    let n_triangle: usize = *n_triangle;
    let rows_start: &[usize] = rows_start;
    let reciprocal: &[f64] = reciprocal;

    // The length of a table over `(i_a, i_b)` up to the degree `n_total + 1`, and where the power 0 is in `phase_relative_power`
    let n_entry: usize = rows_start[n_total + 2];
    let i_power_zero: usize = n_order_max + 4;
    let complex_zero: c64 = c64::new(0.0, 0.0);

    // The edges of the first conductor: the direction of each, `phase = exp(i * phi_e)`, and its length
    let mut edges_phase: [c64; N_EDGE] = [c64::new(f64::NAN, f64::NAN); N_EDGE];
    let mut edges_length: [f64; N_EDGE] = [f64::NAN; N_EDGE];
    for i_edge in 0..N_EDGE {
        let edge_r: f64 = conductors_edges_r[0][i_edge];
        let edge_z: f64 = conductors_edges_z[0][i_edge];
        edges_phase[i_edge] = c64::new(edge_r, edge_z) / edge_r.hypot(edge_z);
        edges_length[i_edge] = edge_r.hypot(edge_z) / scale;
    }

    // Each corner of the second conductor, which is the start of its edge `i_edge_prime`, seen from each edge of the first conductor:
    // `sigma` for the start of `e`, rotated so that `e` lies along the real axis, is `s - i * edge_distance`, where `edge_distance` is the
    // distance from the corner to the line of `e`, positive when the corner is on the inner side of `e`
    let mut corners_s_start: [[f64; N_EDGE]; N_EDGE] = [[f64::NAN; N_EDGE]; N_EDGE];
    let mut corners_edge_distance: [[f64; N_EDGE]; N_EDGE] = [[f64::NAN; N_EDGE]; N_EDGE];
    for i_edge_prime in 0..N_EDGE {
        for i_edge in 0..N_EDGE {
            let sigma: c64 = c64::new(
                corner_delta_r + (conductors_edges_start_r[0][i_edge] - conductors_edges_start_r[1][i_edge_prime]),
                corner_delta_z + (conductors_edges_start_z[0][i_edge] - conductors_edges_start_z[1][i_edge_prime]),
            ) * edges_phase[i_edge].conj()
                / scale;
            corners_s_start[i_edge_prime][i_edge] = sigma.re;
            corners_edge_distance[i_edge_prime][i_edge] = if sigma.im.abs() <= distance_tolerance { 0.0 } else { -sigma.im };
        }
    }

    // The integrals along each edge of the first conductor seen from each corner of the second are calculated when they are first needed,
    // and used again for the other edge of the second conductor which meets that corner. `corners_branch_sign` is the branch of the
    // logarithm with which each table was calculated, and zero while it has not been calculated
    let mut corners_branch_sign: [f64; N_CORNER_EDGE] = [0.0; N_CORNER_EDGE];

    // The sum over the edges `e'` of the second conductor, each with the edges `e` of the first
    let mut edges_sum: f64 = 0.0;
    for i_edge_prime in 0..N_EDGE {
        // The edge `e'`: its direction, its length, and `xi_prime` at its middle
        let i_edge_prime_next: usize = (i_edge_prime + 1) % N_EDGE;
        let edge_prime_r: f64 = conductors_edges_r[1][i_edge_prime];
        let edge_prime_z: f64 = conductors_edges_z[1][i_edge_prime];
        let phase_prime: c64 = c64::new(edge_prime_r, edge_prime_z) / edge_prime_r.hypot(edge_prime_z);
        let length_prime: f64 = edge_prime_r.hypot(edge_prime_z) / scale;
        let xi_middle: f64 = ((oblique_r_prime - centre_r_mean) + (conductors_edges_start_r[1][i_edge_prime] + 0.5 * edge_prime_r)) / centre_r_mean;
        phase_prime_power[0] = c64::new(1.0, 0.0);
        for i_power in 1..n_order + 1 {
            phase_prime_power[i_power] = phase_prime_power[i_power - 1] * phase_prime;
        }

        // Along `e'`, at the distance `position` from its middle, `xi_prime = xi_middle + 2 * re(phase_prime) * position`. The powers of
        // `position` above `n_radial` cannot change the sum
        let xi_change_max: f64 = length_prime * phase_prime.re.abs();
        let radial_ratio: f64 = if t + xi_middle.abs() < 1.0 {
            xi_change_max / (1.0 - t - xi_middle.abs())
        } else {
            f64::INFINITY
        };
        let n_radial_max: usize = n_thin_ring.min(n_order);
        let mut n_radial: usize = 0;
        let mut radial_bound: f64 = radial_ratio;
        'radial_loop: for i_radial in 1..n_radial_max + 1 {
            if radial_bound < coefficients.negligible_term {
                break 'radial_loop;
            }
            n_radial = i_radial;
            radial_bound *= radial_ratio;
        }

        // The coefficients for this edge, of the double integrals with `position ** i_radial` in place of `xi_prime ** j_radial`:
        // xi_prime ** j_radial = sum over i_radial of  binomial[(j_radial, i_radial)] * xi_middle ** (j_radial - i_radial)
        //                                              * (2 * re(phase_prime) * position) ** i_radial
        // A coefficient of the total order `j_radial + i_a + i_b` above `n_order` is left out
        for i_radial in 0..n_radial + 1 {
            let n_entry_edge: usize = half_rows_start[n_order - i_radial + 1];
            let edge_plain: &mut [f64] = &mut edge_coefficient_plain[i_radial * n_half_triangle..i_radial * n_half_triangle + n_entry_edge];
            let edge_ln: &mut [f64] = &mut edge_coefficient_ln[i_radial * n_half_triangle..i_radial * n_half_triangle + n_entry_edge];
            edge_plain.fill(0.0);
            edge_ln.fill(0.0);
            for j_radial in i_radial..n_radial_max + 1 {
                let factor: f64 = binomial[(j_radial, i_radial)] * xi_middle.powi((j_radial - i_radial) as i32) * (2.0 * phase_prime.re).powi(i_radial as i32);
                let n_entry_kept: usize = half_rows_start[n_order - j_radial + 1];
                let pair_plain: &[f64] = &coefficients.coefficient_plain[j_radial * n_half_triangle..j_radial * n_half_triangle + n_entry_kept];
                let pair_ln: &[f64] = &coefficients.coefficient_ln[j_radial * n_half_triangle..j_radial * n_half_triangle + n_entry_kept];
                for i_entry in 0..n_entry_kept {
                    edge_plain[i_entry] += factor * pair_plain[i_entry];
                    edge_ln[i_entry] += factor * pair_ln[i_entry];
                }
            }
        }

        // The sums over the edges `e` of the integrals along `e` seen from the start of `e'`, in the first table of each of the four, and
        // seen from its end, in the second
        for i_end in 0..N_END {
            boundary_plain_plus[i_end * n_triangle..i_end * n_triangle + n_entry].fill(complex_zero);
            boundary_plain_minus[i_end * n_triangle..i_end * n_triangle + n_entry].fill(complex_zero);
            boundary_ln_plus[i_end * n_triangle..i_end * n_triangle + n_entry].fill(complex_zero);
            boundary_ln_minus[i_end * n_triangle..i_end * n_triangle + n_entry].fill(complex_zero);
        }
        let mut n_crossing: usize = 0;
        let mut crossings_position: [f64; N_EDGE] = [f64::NAN; N_EDGE];
        for i_edge in 0..N_EDGE {
            // The edge `e` of the first conductor, and the distances from the two ends of `e'` to its line
            let length: f64 = edges_length[i_edge];
            let ends_edge_distance: [f64; N_END] = [corners_edge_distance[i_edge_prime][i_edge], corners_edge_distance[i_edge_prime_next][i_edge]];

            // If `e'` lies on the line of `e`, its points are at no distance from that line, and this pair of edges adds nothing
            if ends_edge_distance[0] == 0.0 && ends_edge_distance[1] == 0.0 {
                continue;
            }

            // The direction of `e'` in the frame in which `e` lies along the real axis, and its powers: along `e'`, `sigma` changes at the
            // rate `-phase_relative`
            let phase_relative: c64 = phase_prime * edges_phase[i_edge].conj();
            phase_relative_power[i_power_zero] = c64::new(1.0, 0.0);
            for i_power in 1..n_total + 3 {
                let power: c64 = phase_relative_power[i_power_zero + i_power - 1] * phase_relative;
                phase_relative_power[i_power_zero + i_power] = power;
                phase_relative_power[i_power_zero - i_power] = power.conj();
            }

            for i_end in 0..N_END {
                // This end of `e'` is a corner of the second conductor. The points of `e'` near it are on the inner side of `e`
                // (`branch_sign = 1`: the imaginary part of `sigma` is negative or zero, and its argument is between -pi and 0) or on its
                // outer side (`branch_sign = -1`: the argument is between 0 and pi)
                let i_corner: usize = if i_end == 0 { i_edge_prime } else { i_edge_prime_next };
                let edge_distance: f64 = ends_edge_distance[i_end];
                let is_inner_side: bool = edge_distance > 0.0 || (edge_distance == 0.0 && ends_edge_distance[1 - i_end] > 0.0);
                let branch_sign: f64 = if is_inner_side { 1.0 } else { -1.0 };

                // The integrals along `e` seen from this corner, which are those of the flux at a point:
                // integral_plain[(i_a, i_b)] = integral of sigma ** i_a * conj(sigma) ** i_b
                // integral_ln[(i_a, i_b)]    = the same with the factor ln(sigma), the complex logarithm
                // The table of the corner is used again if it was calculated with this branch of the logarithm
                let i_table: usize = i_corner * N_EDGE + i_edge;
                if corners_branch_sign[i_table] != branch_sign {
                    corners_branch_sign[i_table] = branch_sign;
                    let s_start: f64 = corners_s_start[i_corner][i_edge];
                    let s_end: f64 = s_start + length;
                    let sigma_start: c64 = c64::new(s_start, -edge_distance);
                    let sigma_end: c64 = c64::new(s_end, -edge_distance);

                    // The ends of `e` are told apart by which is further from the corner, so that a logarithm is only ever taken of a
                    // ratio which is at most 1: ln(sigma) at the far end, and ln(sigma at the near end / sigma at the far end), whose
                    // real part is formed without cancellation, from |sigma_end| ** 2 - |sigma_start| ** 2 = length * (s_start + s_end),
                    // and whose imaginary part is the angle which `e` subtends at the corner.
                    // At an end of `e` which is at the corner the ratio is zero, but its logarithm is only ever multiplied by zero, so it
                    // is set to zero
                    let start_abs_sq: f64 = s_start.powi(2) + edge_distance.powi(2);
                    let end_abs_sq: f64 = s_end.powi(2) + edge_distance.powi(2);
                    let is_end_far: bool = end_abs_sq >= start_abs_sq;
                    let far_abs_sq: f64 = if is_end_far { end_abs_sq } else { start_abs_sq };
                    let near_abs_sq: f64 = if is_end_far { start_abs_sq } else { end_abs_sq };
                    let s_far: f64 = if is_end_far { s_end } else { s_start };
                    let ratio_sq: f64 = near_abs_sq / far_abs_sq;
                    let ln_near_over_far: f64 = if ratio_sq == 0.0 {
                        0.0
                    } else if ratio_sq < 0.5 {
                        0.5 * ratio_sq.ln()
                    } else {
                        0.5 * (-(length * (s_start + s_end)).abs() / far_abs_sq).ln_1p()
                    };
                    // With `[f]` for `f` at the end of `e` minus `f` at its start:
                    // [f * ln(sigma)] = [f] * ln_far + f_near * ln_near_minus_far, where the sign of the bracket is in `ln_near_minus_far`
                    let sign_near: f64 = if is_end_far { -1.0 } else { 1.0 };

                    // The angle from the start of `e` to its end, seen from the corner. On the line of `e` it is zero when both ends are on
                    // the same side of the corner, and half a turn when the corner is between them, with the sign of the branch
                    let angle: f64 = branch_sign * (edge_distance.abs() * length).atan2(s_start * s_end + edge_distance.powi(2));
                    let ln_near_minus_far: c64 = c64::new(sign_near * ln_near_over_far, angle);

                    // ln(sigma) at the far end, on the branch whose cut is a quarter of a turn away from every value which `sigma` takes
                    let argument_far: f64 = if branch_sign > 0.0 {
                        s_far.atan2(edge_distance) - 0.5 * PI
                    } else {
                        (-s_far).atan2(-edge_distance) + 0.5 * PI
                    };
                    let ln_far: c64 = c64::new(0.5 * far_abs_sq.ln(), argument_far);

                    // The powers of `sigma` at the two ends, and `difference[k] = [sigma ** (k + 1)] / length`, which is formed as the sum
                    // of `sigma_end ** j * sigma_start ** (k - j)`, without cancellation
                    sigma_start_power[0] = c64::new(1.0, 0.0);
                    sigma_end_power[0] = c64::new(1.0, 0.0);
                    difference[0] = c64::new(1.0, 0.0);
                    for i_power in 1..n_total + 3 {
                        sigma_start_power[i_power] = sigma_start_power[i_power - 1] * sigma_start;
                        sigma_end_power[i_power] = sigma_end_power[i_power - 1] * sigma_end;
                        if i_power <= n_total + 1 {
                            difference[i_power] = sigma_end * difference[i_power - 1] + sigma_start_power[i_power];
                        }
                    }
                    let sigma_start_power: &[c64] = &sigma_start_power[..];
                    let sigma_end_power: &[c64] = &sigma_end_power[..];
                    let difference: &[c64] = &difference[..];
                    let sigma_near_power: &[c64] = if is_end_far { sigma_start_power } else { sigma_end_power };

                    let table_plain: &mut [c64] = &mut integral_plain[i_table * n_triangle..(i_table + 1) * n_triangle];
                    let table_ln: &mut [c64] = &mut integral_ln[i_table * n_triangle..(i_table + 1) * n_triangle];
                    for i_degree in 0..n_total + 2 {
                        // The terms with `i_a + i_b = i_degree`, from `i_b = 0`, where the integrals are elementary, by integration by
                        // parts along `e`, which steps from (i_a + 1, i_b - 1) to (i_a, i_b)
                        let row_plain: &mut [c64] = &mut table_plain[rows_start[i_degree]..rows_start[i_degree + 1]];
                        let row_ln: &mut [c64] = &mut table_ln[rows_start[i_degree]..rows_start[i_degree + 1]];
                        for i_b in 0..i_degree + 1 {
                            let i_a: usize = i_degree - i_b;
                            // boundary = [sigma ** (i_a + 1) * conj(sigma) ** i_b], and the same with the logarithm
                            let mut boundary: c64 = difference[i_a] * sigma_end_power[i_b].conj();
                            if i_b >= 1 {
                                boundary += sigma_start_power[i_a + 1] * difference[i_b - 1].conj();
                            }
                            boundary *= length;
                            let boundary_ln: c64 = boundary * ln_far + sigma_near_power[i_a + 1] * sigma_near_power[i_b].conj() * ln_near_minus_far;

                            // Without the logarithm the (i_b, i_a) integral is the complex conjugate of the (i_a, i_b) one, so the
                            // recurrence is only used where it damps the rounding errors, `i_b <= i_a`
                            if i_a >= i_b {
                                let mut value_plain: c64 = boundary;
                                if i_b >= 1 {
                                    value_plain -= row_plain[i_b - 1] * (i_b as f64);
                                }
                                value_plain *= reciprocal[i_a + 1];
                                row_plain[i_b] = value_plain;
                                row_plain[i_a] = value_plain.conj();
                            }

                            // With the complex logarithm there is no such symmetry, and every (i_a, i_b) is needed
                            let mut value_ln: c64 = boundary_ln - row_plain[i_b];
                            if i_b >= 1 {
                                value_ln -= row_ln[i_b - 1] * (i_b as f64);
                            }
                            row_ln[i_b] = value_ln * reciprocal[i_a + 1];
                        }
                    }
                }

                // Rotated into the frame of `e'`, and added to the sums over the edges `e`. The recurrences along `e'` need these sums
                // for `i_a >= 1`, and those without the logarithm for `i_a > i_b` only
                let table_plain: &[c64] = &integral_plain[i_table * n_triangle..(i_table + 1) * n_triangle];
                let table_ln: &[c64] = &integral_ln[i_table * n_triangle..(i_table + 1) * n_triangle];
                let end_plain_plus: &mut [c64] = &mut boundary_plain_plus[i_end * n_triangle..(i_end + 1) * n_triangle];
                let end_plain_minus: &mut [c64] = &mut boundary_plain_minus[i_end * n_triangle..(i_end + 1) * n_triangle];
                let end_ln_plus: &mut [c64] = &mut boundary_ln_plus[i_end * n_triangle..(i_end + 1) * n_triangle];
                let end_ln_minus: &mut [c64] = &mut boundary_ln_minus[i_end * n_triangle..(i_end + 1) * n_triangle];
                for i_degree in 1..n_total + 2 {
                    for i_b in 0..i_degree {
                        // phase_relative ** (i_b - i_a + 1) and phase_relative ** (i_b - i_a - 1), with `i_a = i_degree - i_b`
                        let i_entry: usize = rows_start[i_degree] + i_b;
                        let weight_plus: c64 = phase_relative_power[i_power_zero + 2 * i_b + 1 - i_degree];
                        let weight_minus: c64 = phase_relative_power[i_power_zero + 2 * i_b - 1 - i_degree];
                        if 2 * i_b < i_degree {
                            end_plain_plus[i_entry] += weight_plus * table_plain[i_entry];
                            end_plain_minus[i_entry] += weight_minus * table_plain[i_entry];
                        }
                        end_ln_plus[i_entry] += weight_plus * table_ln[i_entry];
                        end_ln_minus[i_entry] += weight_minus * table_ln[i_entry];
                    }
                }
            }

            // If `e'` crosses the line of `e`, the branch of the logarithm changes at the crossing, where `sigma` is real: for the points
            // of `e` with a negative `s` the logarithm on the inner side is that on the outer side minus `2 * pi * i`. The table of the
            // crossing holds what that adds to `-[position ** i_radial * integral_ln]`, without the power of `position`, rotated into the
            // frame of `e'` as the sums over the edges `e` are
            if ends_edge_distance[0] * ends_edge_distance[1] < 0.0 {
                let fraction: f64 = ends_edge_distance[0] / (ends_edge_distance[0] - ends_edge_distance[1]);
                let s_start: f64 = corners_s_start[i_edge_prime][i_edge] - fraction * length_prime * phase_relative.re;
                if s_start < 0.0 {
                    // The part of `e` with a negative `s`: from `s_start` to `s_stop`, of the length `width`
                    let is_whole_edge: bool = s_start + length < 0.0;
                    let s_stop: f64 = if is_whole_edge { s_start + length } else { 0.0 };
                    let width: f64 = if is_whole_edge { length } else { -s_start };
                    let turn: f64 = if ends_edge_distance[0] > 0.0 { 2.0 * PI } else { -2.0 * PI };
                    crossings_position[n_crossing] = (fraction - 0.5) * length_prime;
                    let i_table_crossing: usize = (N_END + n_crossing) * n_triangle;
                    let crossing_ln_plus: &mut [c64] = &mut boundary_ln_plus[i_table_crossing..i_table_crossing + n_triangle];
                    let crossing_ln_minus: &mut [c64] = &mut boundary_ln_minus[i_table_crossing..i_table_crossing + n_triangle];

                    // The integral of `s ** i_degree` over that part is `width * power_sum / (i_degree + 1)`, with `power_sum` the sum of
                    // `s_stop ** j * s_start ** (i_degree - j)`, whose terms all have the same sign
                    let mut s_start_power: f64 = 1.0;
                    let mut power_sum: f64 = 1.0;
                    for i_degree in 1..n_total + 2 {
                        s_start_power *= s_start;
                        power_sum = s_stop * power_sum + s_start_power;
                        let jump: c64 = c64::new(0.0, turn * width * power_sum * reciprocal[i_degree + 1]);
                        for i_b in 0..i_degree {
                            let i_entry: usize = rows_start[i_degree] + i_b;
                            crossing_ln_plus[i_entry] = jump * phase_relative_power[i_power_zero + 2 * i_b + 1 - i_degree];
                            crossing_ln_minus[i_entry] = jump * phase_relative_power[i_power_zero + 2 * i_b - 1 - i_degree];
                        }
                    }
                    n_crossing += 1;
                }
            }
        }

        // `position` at the start of `e'` is minus `position` at its end, `position_end`, so with `f` for any of the four sums
        // -[position ** i_radial * f] = position_end ** i_radial * (f at the start - f at the end)     for an even `i_radial`
        //                             = position_end ** i_radial * (-f at the start - f at the end)    for an odd `i_radial`
        // From here on the first table of each of the four is for the even powers of `position`, and the second for the odd powers
        for i_entry in rows_start[1]..n_entry {
            let plain_plus_start: c64 = boundary_plain_plus[i_entry];
            let plain_plus_end: c64 = boundary_plain_plus[n_triangle + i_entry];
            boundary_plain_plus[i_entry] = plain_plus_start - plain_plus_end;
            boundary_plain_plus[n_triangle + i_entry] = -(plain_plus_start + plain_plus_end);
            let plain_minus_start: c64 = boundary_plain_minus[i_entry];
            let plain_minus_end: c64 = boundary_plain_minus[n_triangle + i_entry];
            boundary_plain_minus[i_entry] = plain_minus_start - plain_minus_end;
            boundary_plain_minus[n_triangle + i_entry] = -(plain_minus_start + plain_minus_end);
            let ln_plus_start: c64 = boundary_ln_plus[i_entry];
            let ln_plus_end: c64 = boundary_ln_plus[n_triangle + i_entry];
            boundary_ln_plus[i_entry] = ln_plus_start - ln_plus_end;
            boundary_ln_plus[n_triangle + i_entry] = -(ln_plus_start + ln_plus_end);
            let ln_minus_start: c64 = boundary_ln_minus[i_entry];
            let ln_minus_end: c64 = boundary_ln_minus[n_triangle + i_entry];
            boundary_ln_minus[i_entry] = ln_minus_start - ln_minus_end;
            boundary_ln_minus[n_triangle + i_entry] = -(ln_minus_start + ln_minus_end);
        }

        // The powers of `position`, one after the other: the double integrals of each power, from those of the power below, and then the
        // terms of that power
        let position_end: f64 = 0.5 * length_prime;
        let mut edge_prime_sum: f64 = 0.0;
        for i_radial in 0..n_radial + 1 {
            let order_radial: f64 = i_radial as f64;
            let i_parity: usize = i_radial % 2;
            let weight_ends: f64 = position_end.powi(i_radial as i32);
            let mut crossings_weight: [f64; N_EDGE] = [f64::NAN; N_EDGE];
            for i_crossing in 0..n_crossing {
                crossings_weight[i_crossing] = crossings_position[i_crossing].powi(i_radial as i32);
            }

            // The tables of this power, which are calculated here, and of the power below
            let (even_plain_plus, odd_plain_plus): (&mut [c64], &mut [c64]) = sum_plain_plus.split_at_mut(n_triangle);
            let (even_plain_minus, odd_plain_minus): (&mut [c64], &mut [c64]) = sum_plain_minus.split_at_mut(n_triangle);
            let (even_ln_plus, odd_ln_plus): (&mut [c64], &mut [c64]) = sum_ln_plus.split_at_mut(n_triangle);
            let (even_ln_minus, odd_ln_minus): (&mut [c64], &mut [c64]) = sum_ln_minus.split_at_mut(n_triangle);
            let (level_plain_plus, below_plain_plus): (&mut [c64], &mut [c64]) = if i_parity == 0 {
                (even_plain_plus, odd_plain_plus)
            } else {
                (odd_plain_plus, even_plain_plus)
            };
            let (level_plain_minus, below_plain_minus): (&mut [c64], &mut [c64]) = if i_parity == 0 {
                (even_plain_minus, odd_plain_minus)
            } else {
                (odd_plain_minus, even_plain_minus)
            };
            let (level_ln_plus, below_ln_plus): (&mut [c64], &mut [c64]) = if i_parity == 0 {
                (even_ln_plus, odd_ln_plus)
            } else {
                (odd_ln_plus, even_ln_plus)
            };
            let (level_ln_minus, below_ln_minus): (&mut [c64], &mut [c64]) = if i_parity == 0 {
                (even_ln_minus, odd_ln_minus)
            } else {
                (odd_ln_minus, even_ln_minus)
            };
            let ends_plain_plus: &[c64] = &boundary_plain_plus[i_parity * n_triangle..(i_parity + 1) * n_triangle];
            let ends_plain_minus: &[c64] = &boundary_plain_minus[i_parity * n_triangle..(i_parity + 1) * n_triangle];
            let ends_ln_plus: &[c64] = &boundary_ln_plus[i_parity * n_triangle..(i_parity + 1) * n_triangle];
            let ends_ln_minus: &[c64] = &boundary_ln_minus[i_parity * n_triangle..(i_parity + 1) * n_triangle];

            // The double integrals, added up over the edges `e`, by integration by parts along `e'`: with `[f]` for `f` at the end of `e'`
            // minus `f` at its start,
            // (i_a + 1) * sum_plain_plus[(i_radial, i_a, i_b)] =
            //     -[position ** i_radial * boundary_plain_plus[(i_a + 1, i_b)]] + i_radial * sum_plain_plus[(i_radial - 1, i_a + 1, i_b)]
            //     - i_b * sum_plain_plus[(i_radial, i_a + 1, i_b - 1)]
            // and the same for `sum_ln_plus`, with `- sum_plain_plus[(i_radial, i_a, i_b)]` added to the right-hand side, and for the two
            // with `minus`
            for i_degree in 0..n_total - i_radial + 1 {
                let i_row: usize = rows_start[i_degree];
                let i_row_above: usize = rows_start[i_degree + 1];
                let i_row_above_end: usize = rows_start[i_degree + 2];
                let row_plain_plus: &mut [c64] = &mut level_plain_plus[i_row..i_row_above];
                let row_plain_minus: &mut [c64] = &mut level_plain_minus[i_row..i_row_above];
                let row_ln_plus: &mut [c64] = &mut level_ln_plus[i_row..i_row_above];
                let row_ln_minus: &mut [c64] = &mut level_ln_minus[i_row..i_row_above];
                let row_ends_plain_plus: &[c64] = &ends_plain_plus[i_row_above..i_row_above_end];
                let row_ends_plain_minus: &[c64] = &ends_plain_minus[i_row_above..i_row_above_end];
                let row_ends_ln_plus: &[c64] = &ends_ln_plus[i_row_above..i_row_above_end];
                let row_ends_ln_minus: &[c64] = &ends_ln_minus[i_row_above..i_row_above_end];
                let row_below_plain_plus: &[c64] = &below_plain_plus[i_row_above..i_row_above_end];
                let row_below_plain_minus: &[c64] = &below_plain_minus[i_row_above..i_row_above_end];
                let row_below_ln_plus: &[c64] = &below_ln_plus[i_row_above..i_row_above_end];
                let row_below_ln_minus: &[c64] = &below_ln_minus[i_row_above..i_row_above_end];
                for i_b in 0..i_degree + 1 {
                    let i_a: usize = i_degree - i_b;

                    // Without the logarithm, `sum_plain_minus[(i_b, i_a)]` is the complex conjugate of `sum_plain_plus[(i_a, i_b)]`, so the
                    // recurrence is only used where it damps the rounding errors, `i_b <= i_a`
                    if i_a >= i_b {
                        let mut value_plain_plus: c64 = row_ends_plain_plus[i_b] * weight_ends;
                        let mut value_plain_minus: c64 = row_ends_plain_minus[i_b] * weight_ends;
                        if i_radial >= 1 {
                            value_plain_plus += row_below_plain_plus[i_b] * order_radial;
                            value_plain_minus += row_below_plain_minus[i_b] * order_radial;
                        }
                        if i_b >= 1 {
                            value_plain_plus -= row_plain_plus[i_b - 1] * (i_b as f64);
                            value_plain_minus -= row_plain_minus[i_b - 1] * (i_b as f64);
                        }
                        value_plain_plus *= reciprocal[i_a + 1];
                        value_plain_minus *= reciprocal[i_a + 1];
                        row_plain_plus[i_b] = value_plain_plus;
                        row_plain_minus[i_b] = value_plain_minus;
                        if i_a > i_b {
                            row_plain_plus[i_a] = value_plain_minus.conj();
                            row_plain_minus[i_a] = value_plain_plus.conj();
                        }
                    }

                    // With the complex logarithm there is no such symmetry, and every (i_a, i_b) is needed
                    let mut value_ln_plus: c64 = row_ends_ln_plus[i_b] * weight_ends;
                    let mut value_ln_minus: c64 = row_ends_ln_minus[i_b] * weight_ends;
                    for i_crossing in 0..n_crossing {
                        let i_entry: usize = (N_END + i_crossing) * n_triangle + i_row_above + i_b;
                        value_ln_plus += boundary_ln_plus[i_entry] * crossings_weight[i_crossing];
                        value_ln_minus += boundary_ln_minus[i_entry] * crossings_weight[i_crossing];
                    }
                    if i_radial >= 1 {
                        value_ln_plus += row_below_ln_plus[i_b] * order_radial;
                        value_ln_minus += row_below_ln_minus[i_b] * order_radial;
                    }
                    if i_b >= 1 {
                        value_ln_plus -= row_ln_plus[i_b - 1] * (i_b as f64);
                        value_ln_minus -= row_ln_minus[i_b - 1] * (i_b as f64);
                    }
                    row_ln_plus[i_b] = (value_ln_plus - row_plain_plus[i_b]) * reciprocal[i_a + 1];
                    row_ln_minus[i_b] = (value_ln_minus - row_plain_minus[i_b]) * reciprocal[i_a + 1];
                }
            }

            // The terms with this power of `position`. The term of `(i_radial, i_a, i_b)` has the double integrals of the degree
            // `i_a + i_b + 2`, because the product of the two distances is
            // (conj(phase_relative) * sigma ** 2 - 2 * re(phase_relative) * sigma * conj(sigma) + phase_relative * conj(sigma) ** 2) / 4
            // whose powers of `phase_relative` are in the four sums. `rho = -sigma * phase`, whose powers of `phase` are in the four sums
            // too, up to `phase_prime ** (i_a - i_b)`. The (i_b, i_a) term is the complex conjugate of the (i_a, i_b) term
            let level_edge_plain: &[f64] = &edge_coefficient_plain[i_radial * n_half_triangle..(i_radial + 1) * n_half_triangle];
            let level_edge_ln: &[f64] = &edge_coefficient_ln[i_radial * n_half_triangle..(i_radial + 1) * n_half_triangle];
            for i_degree in 0..n_order - i_radial + 1 {
                let i_row: usize = rows_start[i_degree + 2];
                let i_row_end: usize = rows_start[i_degree + 3];
                let row_plain_plus: &[c64] = &level_plain_plus[i_row..i_row_end];
                let row_plain_minus: &[c64] = &level_plain_minus[i_row..i_row_end];
                let row_ln_plus: &[c64] = &level_ln_plus[i_row..i_row_end];
                let row_ln_minus: &[c64] = &level_ln_minus[i_row..i_row_end];

                // The real logarithm is the mean of the complex logarithm and its conjugate, in which the choice of the branch cancels,
                // which is why each part of `e'` could have its own
                let n_double: usize = i_degree / 2 + 3;
                for i_double in 0..n_double {
                    double_ln_plus[i_double] = (row_ln_plus[i_double] + row_ln_minus[i_degree + 2 - i_double].conj()) * 0.5;
                    double_ln_minus[i_double] = (row_ln_minus[i_double] + row_ln_plus[i_degree + 2 - i_double].conj()) * 0.5;
                }

                let sign: f64 = if i_degree % 2 == 0 { 1.0 } else { -1.0 };
                let row_edge_plain: &[f64] = &level_edge_plain[half_rows_start[i_degree]..half_rows_start[i_degree + 1]];
                let row_edge_ln: &[f64] = &level_edge_ln[half_rows_start[i_degree]..half_rows_start[i_degree + 1]];
                let mut row_sum: f64 = 0.0;
                for i_b in 0..i_degree / 2 + 1 {
                    let i_a: usize = i_degree - i_b;
                    if row_edge_plain[i_b] == 0.0 && row_edge_ln[i_b] == 0.0 {
                        continue;
                    }
                    let multiplicity: f64 = if i_a == i_b { 1.0 } else { 2.0 };
                    let distances_plain: c64 = row_plain_plus[i_b] - row_plain_plus[i_b + 1] - row_plain_minus[i_b + 1] + row_plain_minus[i_b + 2];
                    let distances_ln: c64 = double_ln_plus[i_b] - double_ln_plus[i_b + 1] - double_ln_minus[i_b + 1] + double_ln_minus[i_b + 2];
                    let term: c64 = phase_prime_power[i_a - i_b] * (distances_plain * row_edge_plain[i_b] - distances_ln * row_edge_ln[i_b]);
                    row_sum += multiplicity * term.re;
                }
                edge_prime_sum += sign * row_sum;
            }
        }
        edges_sum += 0.25 * edge_prime_sum;
    }

    MU_0 * centre_r_mean * scale.powi(4) * edges_sum / (conductors_area[0] * conductors_area[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::time::Instant;

    /// A conductor of the tables of reference values, as the IMAS data dictionary describes it: the lengths are in [metre], and the
    /// inclinations in [radian]
    struct ReferenceConductor {
        oblique_r: f64,
        oblique_z: f64,
        oblique_length_alpha: f64,
        oblique_length_beta: f64,
        oblique_alpha: f64,
        oblique_beta: f64,
    }

    /// A pair of conductors of `documentation/flux_from_parallelogram_conductor/verification/mutual_inductance_reference.json`
    struct ReferencePair {
        /// The pair's name in that file, which starts with what kind of pair it is: `self`, `touching`, `overlapping`, `near` or `far`
        name: &'static str,
        /// The mutual inductance of the pair, [henry]
        mutual_inductance: f64,
        /// The conductor through which the flux is taken
        conductor: ReferenceConductor,
        /// The conductor which carries the current
        conductor_prime: ReferenceConductor,
    }

    /// The 16 pairs of `documentation/flux_from_parallelogram_conductor/verification/mutual_inductance_reference.json`, whose mutual
    /// inductances were calculated to 30 digits by two integrals which do not use the series. The values are the file's
    /// `mutual_inductance`, which is for the positions, lengths and inclinations rounded to doubles, and they are rounded to doubles
    /// themselves. Every number is written with as many digits as it takes to read it back as the same double, which is at most 17
    const N_REFERENCE_PAIR: usize = 16;
    #[rustfmt::skip]
    const REFERENCE_PAIRS: [ReferencePair; N_REFERENCE_PAIR] = [
        // 0: self-inductance of a thin tall rectangle, 3.2351 mm x 24.0734 mm, an element of the inner wall
        ReferencePair {
            name: "self_thin_tall",
            mutual_inductance: 6.470065692853491e-07,
            conductor: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 1: self-inductance of a 12.5 mm square centred at R = 0.41 m, the grid cell of the GSFit paper
        ReferencePair {
            name: "self_square",
            mutual_inductance: 2.2541991198640103e-06,
            conductor: ReferenceConductor { oblique_r: 0.40375, oblique_z: -0.00625, oblique_length_alpha: 0.0125, oblique_length_beta: 0.0125,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.40375, oblique_z: -0.00625, oblique_length_alpha: 0.0125, oblique_length_beta: 0.0125,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 2: self-inductance of a sheared, rotated parallelogram, 34 mm x 36 mm, with an interior angle of 58.5 degrees, near R = 0.33 m, Z = 0.62 m
        ReferencePair {
            name: "self_sheared_rotated",
            mutual_inductance: 1.4233877727589298e-06,
            conductor: ReferenceConductor { oblique_r: 0.33, oblique_z: 0.62, oblique_length_alpha: 0.034, oblique_length_beta: 0.036,
                oblique_alpha: 0.3, oblique_beta: -0.25 },
            conductor_prime: ReferenceConductor { oblique_r: 0.33, oblique_z: 0.62, oblique_length_alpha: 0.034, oblique_length_beta: 0.036,
                oblique_alpha: 0.3, oblique_beta: -0.25 },
        },
        // 3: self-inductance of a rectangle close to the axis, 20 mm x 30 mm, with its inner side at R = 0.03 m
        ReferencePair {
            name: "self_near_axis",
            mutual_inductance: 7.106060702803194e-08,
            conductor: ReferenceConductor { oblique_r: 0.03, oblique_z: 0.1, oblique_length_alpha: 0.02, oblique_length_beta: 0.03,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.03, oblique_z: 0.1, oblique_length_alpha: 0.02, oblique_length_beta: 0.03,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 4: two thin tall rectangles stacked: they share a whole short side
        ReferencePair {
            name: "touching_stacked_thin_tall",
            mutual_inductance: 4.035710710492987e-07,
            conductor: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0240734, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 5: two rectangles, 30 mm x 20 mm and 25 mm x 12 mm, one on top of the other and displaced sideways: they share 13 mm of a side
        ReferencePair {
            name: "touching_part_of_side",
            mutual_inductance: 2.237355500768218e-06,
            conductor: ReferenceConductor { oblique_r: 0.52, oblique_z: 0.0, oblique_length_alpha: 0.03, oblique_length_beta: 0.02,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.537, oblique_z: 0.02, oblique_length_alpha: 0.025, oblique_length_beta: 0.012,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 6: two rectangles, 31.25 mm x 20 mm and 18 mm x 26 mm, which touch at one vertex only (diagonally opposite)
        ReferencePair {
            name: "touching_vertex",
            mutual_inductance: 1.3013836136301075e-06,
            conductor: ReferenceConductor { oblique_r: 0.375, oblique_z: 0.0, oblique_length_alpha: 0.03125, oblique_length_beta: 0.02,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.40625, oblique_z: 0.02, oblique_length_alpha: 0.018, oblique_length_beta: 0.026,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 7: two 22 mm x 30 mm rectangles which would share a 30 mm side, the second rotated by 5 degrees about the shared reference corner: they
        // share one vertex, and their nearest sides open at 5 degrees
        ReferencePair {
            name: "touching_rotated_5_degrees",
            mutual_inductance: 1.682966902992877e-06,
            conductor: ReferenceConductor { oblique_r: 0.45, oblique_z: 0.2, oblique_length_alpha: 0.022, oblique_length_beta: 0.03,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.45, oblique_z: 0.2, oblique_length_alpha: 0.03, oblique_length_beta: 0.022,
                oblique_alpha: 1.6580627893946132, oblique_beta: 1.6580627893946132 },
        },
        // 8: a 30 mm x 20 mm rectangle and a 25 mm x 18 mm rectangle rotated by 20 degrees, which overlap partially: their edges cross twice, and one
        // vertex of each is inside the other
        ReferencePair {
            name: "overlapping_rotated_20_degrees",
            mutual_inductance: 1.2501908167258603e-06,
            conductor: ReferenceConductor { oblique_r: 0.3, oblique_z: -0.4, oblique_length_alpha: 0.03, oblique_length_beta: 0.02,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.315, oblique_z: -0.392, oblique_length_alpha: 0.025, oblique_length_beta: 0.018,
                oblique_alpha: 0.3490658503988659, oblique_beta: 0.3490658503988659 },
        },
        // 9: a 12 mm x 9 mm rectangle wholly inside a 40 mm x 35 mm rectangle
        ReferencePair {
            name: "overlapping_small_inside_large",
            mutual_inductance: 3.040185189733014e-06,
            conductor: ReferenceConductor { oblique_r: 0.6, oblique_z: -0.02, oblique_length_alpha: 0.04, oblique_length_beta: 0.035,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.611, oblique_z: -0.008, oblique_length_alpha: 0.012, oblique_length_beta: 0.009,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 10: two thin tall rectangles, one above the other, with a gap of one element length (24.0734 mm) between them
        ReferencePair {
            name: "near_thin_tall_one_length_gap",
            mutual_inductance: 2.5834510310833e-07,
            conductor: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0481468, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 11: two sheared parallelograms, 32 mm x 28 mm and 30 mm x 28 mm, side by side with a gap of about 7 mm between parallel sides
        ReferencePair {
            name: "near_sheared_quarter_gap",
            mutual_inductance: 2.829314056433516e-06,
            conductor: ReferenceConductor { oblique_r: 0.7, oblique_z: 0.3, oblique_length_alpha: 0.032, oblique_length_beta: 0.028,
                oblique_alpha: 0.2, oblique_beta: -0.3 },
            conductor_prime: ReferenceConductor { oblique_r: 0.7392, oblique_z: 0.3079, oblique_length_alpha: 0.03, oblique_length_beta: 0.028,
                oblique_alpha: 0.2, oblique_beta: -0.3 },
        },
        // 12: two thin tall rectangles, one above the other, 10 element lengths (0.240734 m) apart
        ReferencePair {
            name: "far_thin_tall_10_lengths",
            mutual_inductance: 3.696452699155757e-08,
            conductor: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.240734, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 13: two thin tall rectangles, one above the other, 30 element lengths (0.722202 m) apart
        ReferencePair {
            name: "far_thin_tall_30_lengths",
            mutual_inductance: 2.6737556214668998e-09,
            conductor: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.0, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.15357, oblique_z: 0.722202, oblique_length_alpha: 0.0032351, oblique_length_beta: 0.0240734,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 14: a sheared, rotated parallelogram, 34 mm x 36 mm, and a 20 mm x 35 mm rectangle, about 0.5 m apart
        ReferencePair {
            name: "far_different_shapes",
            mutual_inductance: 2.5953656601410346e-07,
            conductor: ReferenceConductor { oblique_r: 0.33, oblique_z: 0.62, oblique_length_alpha: 0.034, oblique_length_beta: 0.036,
                oblique_alpha: 0.3, oblique_beta: -0.25 },
            conductor_prime: ReferenceConductor { oblique_r: 0.72, oblique_z: 0.31, oblique_length_alpha: 0.02, oblique_length_beta: 0.035,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
        },
        // 15: a 20 mm x 30 mm rectangle close to the axis (inner side at R = 0.05 m) and a sheared 30 mm x 40 mm parallelogram at R = 0.8 m
        ReferencePair {
            name: "far_near_axis_to_outer",
            mutual_inductance: 8.602801548229873e-09,
            conductor: ReferenceConductor { oblique_r: 0.05, oblique_z: -0.01, oblique_length_alpha: 0.02, oblique_length_beta: 0.03,
                oblique_alpha: 0.0, oblique_beta: 0.0 },
            conductor_prime: ReferenceConductor { oblique_r: 0.8, oblique_z: 0.1, oblique_length_alpha: 0.03, oblique_length_beta: 0.04,
                oblique_alpha: 0.1, oblique_beta: 0.25 },
        },
    ];

    /// The far-field series of one of `REFERENCE_PAIRS`, at each of `FAR_FIELD_SERIES_ORDERS`
    struct FarFieldSeriesCase {
        /// Which of `REFERENCE_PAIRS`
        i_pair: usize,
        /// The series, truncated at each of `FAR_FIELD_SERIES_ORDERS`, [henry]
        mutual_inductance: [f64; N_FAR_FIELD_SERIES_ORDER],
    }

    /// The far-field series of five of `REFERENCE_PAIRS`, truncated at five orders, one of them odd. The fourth pair has a conductor close
    /// to the axis, and the last pair is so close together that the series converges slowly. The values were calculated with the library
    /// which accompanies the note, in `documentation/flux_from_parallelogram_conductor/code/`
    const N_FAR_FIELD_SERIES_ORDER: usize = 5;
    const FAR_FIELD_SERIES_ORDERS: [usize; N_FAR_FIELD_SERIES_ORDER] = [0, 2, 7, 12, 20];
    const N_FAR_FIELD_SERIES_CASE: usize = 5;
    #[rustfmt::skip]
    const FAR_FIELD_SERIES_CASES: [FarFieldSeriesCase; N_FAR_FIELD_SERIES_CASE] = [
        // `far_thin_tall_10_lengths`
        FarFieldSeriesCase { i_pair: 12,
            mutual_inductance: [3.6828266627533156e-08, 3.6964134472199834e-08, 3.696452698681972e-08, 3.696452699155756e-08, 3.696452699155756e-08] },
        // `far_thin_tall_30_lengths`
        FarFieldSeriesCase { i_pair: 13,
            mutual_inductance: [2.671155601819959e-09, 2.673753465955912e-09, 2.6737556214657955e-09, 2.6737556214668994e-09, 2.6737556214668994e-09] },
        // `far_different_shapes`
        FarFieldSeriesCase { i_pair: 14,
            mutual_inductance: [2.595077124060151e-07, 2.5953638299626815e-07, 2.5953656601296605e-07, 2.595365660141034e-07, 2.595365660141034e-07] },
        // `far_near_axis_to_outer`
        FarFieldSeriesCase { i_pair: 15,
            mutual_inductance: [8.526109378782629e-09, 8.602826419705173e-09, 8.602801548222203e-09, 8.602801548229876e-09, 8.602801548229876e-09] },
        // `near_thin_tall_one_length_gap`
        FarFieldSeriesCase { i_pair: 10,
            mutual_inductance: [2.5394004134909203e-07, 2.5813103627034877e-07, 2.5834298950750917e-07, 2.5834509744371316e-07, 2.5834510310493606e-07] },
    ];

    /// The settings of the tests: the orders and the tolerance of ST40's settings, and higher orders with a smaller tolerance
    const N_REFERENCE_SETTING: usize = 2;
    const REFERENCE_SETTINGS: [SettingsEquilibriumNumericsGreensFiniteSize; N_REFERENCE_SETTING] = [
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 6,
            n_thin_ring: 12,
            n_multipole: 12,
            relative_tolerance: 1.0e-9,
            n_subdivision_max: 30,
        },
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 10,
            n_thin_ring: 20,
            n_multipole: 20,
            relative_tolerance: 1.0e-12,
            n_subdivision_max: 30,
        },
    ];

    /// Settings with which the near-field formula is never chosen, as its threshold is the tolerance itself, and with which the far-field
    /// series is of a low order, so that a pair which is apart has to be halved until the far-field series converges fast enough
    const SETTINGS_FAR_FIELD_ONLY: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
        n_elliptic: 0,
        n_thin_ring: 0,
        n_multipole: 4,
        relative_tolerance: 1.0e-9,
        n_subdivision_max: 30,
    };

    /// The mutual inductance of one pair of conductors, through the function which takes two sets
    fn mutual_inductance_of_one_pair(
        conductor: &ReferenceConductor,
        conductor_prime: &ReferenceConductor,
        settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
    ) -> Result<f64, String> {
        let mutual_inductance: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            &array![conductor_prime.oblique_r],
            &array![conductor_prime.oblique_z],
            &array![conductor_prime.oblique_length_alpha],
            &array![conductor_prime.oblique_length_beta],
            &array![conductor_prime.oblique_alpha],
            &array![conductor_prime.oblique_beta],
            false,
            settings_greens_finite_size,
        )?;

        Ok(mutual_inductance[(0, 0)])
    }

    /// The mutual inductance of one pair of conductors, and how many pairs of pieces were summed with each calculation
    fn mutual_inductance_and_pieces_of_one_pair(
        conductor: &ReferenceConductor,
        conductor_prime: &ReferenceConductor,
        settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
    ) -> Result<MutualInductanceParallelogram, String> {
        let n_multipole: usize = 2 * (settings_greens_finite_size.n_multipole / 2);
        let shape: ConductorShape = ConductorShape::new(
            conductor.oblique_length_alpha,
            conductor.oblique_length_beta,
            conductor.oblique_alpha,
            conductor.oblique_beta,
            n_multipole,
        );
        let shape_prime: ConductorShape = ConductorShape::new(
            conductor_prime.oblique_length_alpha,
            conductor_prime.oblique_length_beta,
            conductor_prime.oblique_alpha,
            conductor_prime.oblique_beta,
            n_multipole,
        );
        let mut work_arrays: PairWorkArrays = PairWorkArrays::new(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring, n_multipole);

        mutual_inductance_parallelogram(
            conductor.oblique_r,
            conductor.oblique_z,
            &shape,
            conductor_prime.oblique_r,
            conductor_prime.oblique_z,
            &shape_prime,
            &expansion_parameter_thresholds(settings_greens_finite_size)?,
            &pair_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring),
            n_multipole,
            settings_greens_finite_size.n_subdivision_max,
            &mut work_arrays,
        )
    }

    /// The far-field series of a whole pair of conductors, truncated at `n_multipole`, which may be odd
    fn far_field_series_of_one_pair(conductor: &ReferenceConductor, conductor_prime: &ReferenceConductor, n_multipole: usize) -> f64 {
        // An odd order adds nothing to the even order below it
        let n_multipole_even: usize = 2 * (n_multipole / 2);
        let shape: ConductorShape = ConductorShape::new(
            conductor.oblique_length_alpha,
            conductor.oblique_length_beta,
            conductor.oblique_alpha,
            conductor.oblique_beta,
            n_multipole_even,
        );
        let shape_prime: ConductorShape = ConductorShape::new(
            conductor_prime.oblique_length_alpha,
            conductor_prime.oblique_length_beta,
            conductor_prime.oblique_alpha,
            conductor_prime.oblique_beta,
            n_multipole_even,
        );
        let centre_r: f64 = conductor.oblique_r + (0.5 * shape.side_alpha_r + 0.5 * shape.side_beta_r);
        let centre_z: f64 = conductor.oblique_z + (0.5 * shape.side_alpha_z + 0.5 * shape.side_beta_z);
        let centre_r_prime: f64 = conductor_prime.oblique_r + (0.5 * shape_prime.side_alpha_r + 0.5 * shape_prime.side_beta_r);
        let centre_z_prime: f64 = conductor_prime.oblique_z + (0.5 * shape_prime.side_alpha_z + 0.5 * shape_prime.side_beta_z);
        let mut work_arrays: FarFieldPairWorkArrays = FarFieldPairWorkArrays::new(n_multipole_even);

        mutual_inductance_parallelogram_far_field(
            centre_r,
            centre_r_prime,
            centre_z - centre_z_prime,
            &shape.moment_scaled,
            &shape_prime.moment_scaled,
            n_multipole_even,
            &mut work_arrays,
        )
    }

    #[test]
    fn the_far_field_series_agrees_with_the_reference_values() {
        for i_case in 0..N_FAR_FIELD_SERIES_CASE {
            let case: &FarFieldSeriesCase = &FAR_FIELD_SERIES_CASES[i_case];
            let pair: &ReferencePair = &REFERENCE_PAIRS[case.i_pair];
            for i_order in 0..N_FAR_FIELD_SERIES_ORDER {
                let n_multipole: usize = FAR_FIELD_SERIES_ORDERS[i_order];
                let mutual_inductance: f64 = far_field_series_of_one_pair(&pair.conductor, &pair.conductor_prime, n_multipole);
                let relative_difference: f64 = (mutual_inductance / case.mutual_inductance[i_order] - 1.0).abs();
                assert!(
                    relative_difference < 5.0e-15,
                    "`{}`, order {n_multipole}: {mutual_inductance:e} henry differs from the reference value by {relative_difference:e}",
                    pair.name
                );

                // The two conductors exchanged
                let mutual_inductance_exchanged: f64 = far_field_series_of_one_pair(&pair.conductor_prime, &pair.conductor, n_multipole);
                assert!(
                    (mutual_inductance_exchanged / mutual_inductance - 1.0).abs() < 1.0e-15,
                    "`{}`, order {n_multipole}: the two conductors exchanged",
                    pair.name
                );
            }
        }
    }

    #[test]
    fn the_mutual_inductance_of_conductors_which_are_apart_agrees_with_the_high_precision_values() {
        for i_setting in 0..N_REFERENCE_SETTING {
            let settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[i_setting];
            for i_pair in 0..N_REFERENCE_PAIR {
                let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
                if pair.name.starts_with("far") {
                    let mutual_inductance: f64 = mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, settings_greens_finite_size)
                        .expect("the far-field series converges for every pair of pieces");
                    let relative_difference: f64 = (mutual_inductance / pair.mutual_inductance - 1.0).abs();
                    assert!(
                        relative_difference < settings_greens_finite_size.relative_tolerance,
                        "setting {i_setting}, `{}`: {mutual_inductance:e} henry differs from the high-precision value by {relative_difference:e}",
                        pair.name
                    );
                }
            }
        }
    }

    #[test]
    fn the_mutual_inductance_of_conductors_which_are_close_agrees_with_the_high_precision_values() {
        for i_setting in 0..N_REFERENCE_SETTING {
            let settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[i_setting];
            for i_pair in 0..N_REFERENCE_PAIR {
                let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
                if !pair.name.starts_with("far") {
                    let mutual_inductance: f64 = mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, settings_greens_finite_size)
                        .expect("one of the two calculations converges for every pair of pieces");
                    let relative_difference: f64 = (mutual_inductance / pair.mutual_inductance - 1.0).abs();
                    assert!(
                        relative_difference < settings_greens_finite_size.relative_tolerance,
                        "setting {i_setting}, `{}`: {mutual_inductance:e} henry differs from the high-precision value by {relative_difference:e}",
                        pair.name
                    );
                }
            }
        }
    }

    #[test]
    fn a_pair_is_halved_until_the_far_field_series_converges() {
        for i_pair in 0..N_REFERENCE_PAIR {
            let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
            if pair.name.starts_with("far") {
                let halved: MutualInductanceParallelogram =
                    mutual_inductance_and_pieces_of_one_pair(&pair.conductor, &pair.conductor_prime, &SETTINGS_FAR_FIELD_ONLY)
                        .expect("the far-field series converges for every pair of pieces");

                // The whole pair is too large for a series of this order, and the near-field formula is never chosen
                assert!(halved.n_piece_pair_far_field > 1, "`{}`", pair.name);
                assert_eq!(halved.n_piece_pair_near_field, 0, "`{}`", pair.name);

                // A series of a high order has converged for the whole pair
                let whole: f64 = far_field_series_of_one_pair(&pair.conductor, &pair.conductor_prime, 20);
                let relative_difference: f64 = (halved.mutual_inductance / whole - 1.0).abs();
                assert!(
                    relative_difference < SETTINGS_FAR_FIELD_ONLY.relative_tolerance,
                    "`{}`: {} pairs of pieces differ from the whole pair by {relative_difference:e}",
                    pair.name,
                    halved.n_piece_pair_far_field
                );
                assert!((halved.mutual_inductance / pair.mutual_inductance - 1.0).abs() < SETTINGS_FAR_FIELD_ONLY.relative_tolerance);
            }
        }
    }

    #[test]
    fn exchanging_the_two_conductors_gives_the_same_mutual_inductance() {
        for i_pair in 0..N_REFERENCE_PAIR {
            let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
            if pair.name.starts_with("far") {
                // Both ways round the same pieces are summed with the same series, unless the two conductors' longer sides are equal,
                // so the two differ by rounding
                for i_setting in 0..N_REFERENCE_SETTING {
                    let mutual_inductance: f64 = mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &REFERENCE_SETTINGS[i_setting])
                        .expect("the far-field series converges for every pair of pieces");
                    let mutual_inductance_exchanged: f64 =
                        mutual_inductance_of_one_pair(&pair.conductor_prime, &pair.conductor, &REFERENCE_SETTINGS[i_setting])
                            .expect("the far-field series converges for every pair of pieces");
                    assert!(
                        (mutual_inductance_exchanged / mutual_inductance - 1.0).abs() < 1.0e-14,
                        "setting {i_setting}, `{}`",
                        pair.name
                    );
                }

                // When they are equal, as the two thin tall rectangles' are, the first conductor is halved first, so the pieces are not
                // the same both ways round, and the two differ by less than the tolerance
                let mutual_inductance: f64 = mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &SETTINGS_FAR_FIELD_ONLY)
                    .expect("the far-field series converges for every pair of pieces");
                let mutual_inductance_exchanged: f64 = mutual_inductance_of_one_pair(&pair.conductor_prime, &pair.conductor, &SETTINGS_FAR_FIELD_ONLY)
                    .expect("the far-field series converges for every pair of pieces");
                assert!(
                    (mutual_inductance_exchanged / mutual_inductance - 1.0).abs() < SETTINGS_FAR_FIELD_ONLY.relative_tolerance,
                    "`{}`, halved",
                    pair.name
                );
            }
        }
    }

    /// Five conductors which are apart from each other: the conductors of the pairs `far_different_shapes` and `far_near_axis_to_outer`,
    /// and the first conductor of `far_thin_tall_10_lengths`
    fn conductors_which_are_apart() -> [&'static ReferenceConductor; 5] {
        [
            &REFERENCE_PAIRS[14].conductor,
            &REFERENCE_PAIRS[14].conductor_prime,
            &REFERENCE_PAIRS[15].conductor,
            &REFERENCE_PAIRS[15].conductor_prime,
            &REFERENCE_PAIRS[12].conductor,
        ]
    }

    #[test]
    fn the_same_set_twice_gives_an_exactly_symmetric_matrix() {
        let n_conductor: usize = 5;
        let conductors: [&ReferenceConductor; 5] = conductors_which_are_apart();
        let mut oblique_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut oblique_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut oblique_length_alpha: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut oblique_length_beta: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut oblique_alpha: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut oblique_beta: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        for i_conductor in 0..n_conductor {
            oblique_r[i_conductor] = conductors[i_conductor].oblique_r;
            oblique_z[i_conductor] = conductors[i_conductor].oblique_z;
            oblique_length_alpha[i_conductor] = conductors[i_conductor].oblique_length_alpha;
            oblique_length_beta[i_conductor] = conductors[i_conductor].oblique_length_beta;
            oblique_alpha[i_conductor] = conductors[i_conductor].oblique_alpha;
            oblique_beta[i_conductor] = conductors[i_conductor].oblique_beta;
        }

        let mutual_inductance: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
            &oblique_r,
            &oblique_z,
            &oblique_length_alpha,
            &oblique_length_beta,
            &oblique_alpha,
            &oblique_beta,
            &oblique_r.clone(),
            &oblique_z.clone(),
            &oblique_length_alpha.clone(),
            &oblique_length_beta.clone(),
            &oblique_alpha.clone(),
            &oblique_beta.clone(),
            true,
            &REFERENCE_SETTINGS[0],
        )
        .expect("one of the two calculations converges for every pair of pieces");

        // Every entry, the diagonal of self-inductances included, is the same, to rounding, as the pair on its own, which is calculated in
        // full, and the matrix is exactly symmetric
        assert_eq!(mutual_inductance.dim(), (n_conductor, n_conductor));
        for i_conductor in 0..n_conductor {
            for i_conductor_prime in 0..n_conductor {
                assert_eq!(
                    mutual_inductance[(i_conductor, i_conductor_prime)].to_bits(),
                    mutual_inductance[(i_conductor_prime, i_conductor)].to_bits(),
                    "conductors {i_conductor} and {i_conductor_prime}"
                );
                let pair: f64 = mutual_inductance_of_one_pair(conductors[i_conductor], conductors[i_conductor_prime], &REFERENCE_SETTINGS[0])
                    .expect("one of the two calculations converges for every pair of pieces");
                assert!(
                    (mutual_inductance[(i_conductor, i_conductor_prime)] / pair - 1.0).abs() < 1.0e-14,
                    "conductors {i_conductor} and {i_conductor_prime}"
                );
            }
        }
    }

    #[test]
    fn each_pair_of_two_different_sets_fills_one_entry() {
        // Two conductors, and three others
        let conductors: [&ReferenceConductor; 5] = conductors_which_are_apart();
        let n_conductor: usize = 2;
        let n_conductor_prime: usize = 3;

        let mutual_inductance: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
            &array![conductors[0].oblique_r, conductors[1].oblique_r],
            &array![conductors[0].oblique_z, conductors[1].oblique_z],
            &array![conductors[0].oblique_length_alpha, conductors[1].oblique_length_alpha],
            &array![conductors[0].oblique_length_beta, conductors[1].oblique_length_beta],
            &array![conductors[0].oblique_alpha, conductors[1].oblique_alpha],
            &array![conductors[0].oblique_beta, conductors[1].oblique_beta],
            &array![conductors[2].oblique_r, conductors[3].oblique_r, conductors[4].oblique_r],
            &array![conductors[2].oblique_z, conductors[3].oblique_z, conductors[4].oblique_z],
            &array![
                conductors[2].oblique_length_alpha,
                conductors[3].oblique_length_alpha,
                conductors[4].oblique_length_alpha
            ],
            &array![
                conductors[2].oblique_length_beta,
                conductors[3].oblique_length_beta,
                conductors[4].oblique_length_beta
            ],
            &array![conductors[2].oblique_alpha, conductors[3].oblique_alpha, conductors[4].oblique_alpha],
            &array![conductors[2].oblique_beta, conductors[3].oblique_beta, conductors[4].oblique_beta],
            false,
            &REFERENCE_SETTINGS[0],
        )
        .expect("the far-field series converges for every pair of pieces");

        assert_eq!(mutual_inductance.dim(), (n_conductor, n_conductor_prime));
        for i_conductor in 0..n_conductor {
            for i_conductor_prime in 0..n_conductor_prime {
                let pair: f64 = mutual_inductance_of_one_pair(conductors[i_conductor], conductors[n_conductor + i_conductor_prime], &REFERENCE_SETTINGS[0])
                    .expect("the far-field series converges for every pair of pieces");
                assert_eq!(
                    mutual_inductance[(i_conductor, i_conductor_prime)],
                    pair,
                    "conductors {i_conductor} and {i_conductor_prime}"
                );
            }
        }
    }

    #[test]
    fn arrays_of_different_lengths_are_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_PAIRS[14].conductor;
        let conductor_prime: &ReferenceConductor = &REFERENCE_PAIRS[14].conductor_prime;

        // The first set has two reference corners, and one of everything else
        let error: String = mutual_inductance_finite_size_to_finite_size(
            &array![conductor.oblique_r, conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            &array![conductor_prime.oblique_r],
            &array![conductor_prime.oblique_z],
            &array![conductor_prime.oblique_length_alpha],
            &array![conductor_prime.oblique_length_beta],
            &array![conductor_prime.oblique_alpha],
            &array![conductor_prime.oblique_beta],
            false,
            &REFERENCE_SETTINGS[0],
        )
        .expect_err("the first set's arrays have different lengths");
        assert!(error.contains("`oblique_r`"), "{error}");

        // The second set has two inclinations of its second side
        let error: String = mutual_inductance_finite_size_to_finite_size(
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            &array![conductor_prime.oblique_r],
            &array![conductor_prime.oblique_z],
            &array![conductor_prime.oblique_length_alpha],
            &array![conductor_prime.oblique_length_beta],
            &array![conductor_prime.oblique_alpha],
            &array![conductor_prime.oblique_beta, conductor_prime.oblique_beta],
            false,
            &REFERENCE_SETTINGS[0],
        )
        .expect_err("the second set's arrays have different lengths");
        assert!(error.contains("`oblique_beta_prime`"), "{error}");
    }

    #[test]
    fn a_conductor_with_no_area_is_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_PAIRS[14].conductor;

        // A side without a positive length, in either set
        let conductor_without_length: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.0,
            oblique_length_beta: 0.01,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        let error: String = mutual_inductance_of_one_pair(&conductor_without_length, conductor, &REFERENCE_SETTINGS[0])
            .expect_err("a side of the first conductor has no length");
        assert!(error.contains("conductor 0 has no area: `oblique_length_alpha` ="), "{error}");
        let error: String = mutual_inductance_of_one_pair(conductor, &conductor_without_length, &REFERENCE_SETTINGS[0])
            .expect_err("a side of the second conductor has no length");
        assert!(error.contains("conductor 0 has no area: `oblique_length_alpha_prime` ="), "{error}");

        // Two sides which are parallel: the second side is inclined from the Z axis, so a quarter of a turn makes it lie along the first
        let conductor_with_parallel_sides: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.01,
            oblique_length_beta: 0.01,
            oblique_alpha: 0.3,
            oblique_beta: 0.3 - 0.5 * std::f64::consts::PI,
        };
        assert!(mutual_inductance_of_one_pair(&conductor_with_parallel_sides, conductor, &REFERENCE_SETTINGS[0]).is_err());
        assert!(mutual_inductance_of_one_pair(conductor, &conductor_with_parallel_sides, &REFERENCE_SETTINGS[0]).is_err());

        // A length which is not a number
        let conductor_without_number: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.01,
            oblique_length_beta: f64::NAN,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        assert!(mutual_inductance_of_one_pair(&conductor_without_number, conductor, &REFERENCE_SETTINGS[0]).is_err());
    }

    #[test]
    fn a_conductor_which_is_not_wholly_at_a_positive_major_radius_is_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_PAIRS[14].conductor;

        // The reference corner is at a positive major radius, but the first side crosses the axis
        let conductor_across_the_axis: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.01,
            oblique_z: 0.0,
            oblique_length_alpha: 0.02,
            oblique_length_beta: 0.01,
            oblique_alpha: std::f64::consts::PI,
            oblique_beta: 0.0,
        };
        let error: String =
            mutual_inductance_of_one_pair(&conductor_across_the_axis, conductor, &REFERENCE_SETTINGS[0]).expect_err("the first conductor crosses the axis");
        assert!(error.contains("positive major radius"), "{error}");
        assert!(error.contains("`oblique_r` ="), "{error}");
        let error: String =
            mutual_inductance_of_one_pair(conductor, &conductor_across_the_axis, &REFERENCE_SETTINGS[0]).expect_err("the second conductor crosses the axis");
        assert!(error.contains("`oblique_r_prime` ="), "{error}");

        // A height which is not finite
        let conductor_at_infinity: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: f64::INFINITY,
            oblique_length_alpha: 0.02,
            oblique_length_beta: 0.01,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        assert!(mutual_inductance_of_one_pair(conductor, &conductor_at_infinity, &REFERENCE_SETTINGS[0]).is_err());
    }

    #[test]
    fn a_tolerance_which_is_not_a_fraction_is_an_error() {
        let pair: &ReferencePair = &REFERENCE_PAIRS[14];
        let n_tolerance: usize = 4;
        let relative_tolerances: [f64; 4] = [0.0, -1.0e-9, 1.5, f64::NAN];
        for i_tolerance in 0..n_tolerance {
            let settings_greens_finite_size: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
                relative_tolerance: relative_tolerances[i_tolerance],
                ..REFERENCE_SETTINGS[0].clone()
            };
            let error: String = mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &settings_greens_finite_size)
                .expect_err("the tolerance is not a fraction");
            assert!(error.contains("`relative_tolerance`"), "{error}");
        }
    }

    #[test]
    fn a_conductor_which_may_not_be_halved_often_enough_is_an_error() {
        // With these settings the pair has to be halved until the far-field series converges
        let pair: &ReferencePair = &REFERENCE_PAIRS[14];
        assert!(mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &SETTINGS_FAR_FIELD_ONLY).is_ok());

        let settings_too_few: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
            n_subdivision_max: 1,
            ..SETTINGS_FAR_FIELD_ONLY.clone()
        };
        let error: String =
            mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &settings_too_few).expect_err("one halving of each conductor is not enough");
        assert!(error.contains("conductor 0 of the first set and conductor 0 of the second"), "{error}");
        assert!(error.contains("halved 1 time(s)"), "{error}");

        // With the orders of ST40's settings the same pair needs no halving at all
        let settings_no_halving: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
            n_subdivision_max: 0,
            ..REFERENCE_SETTINGS[0].clone()
        };
        assert!(mutual_inductance_of_one_pair(&pair.conductor, &pair.conductor_prime, &settings_no_halving).is_ok());
    }

    /// How long the mutual inductances between the elements of ST40's inner vacuum vessel take, and how many of the pairs take each
    /// calculation. Nothing is checked: it prints what it measures, so it is run on its own, optimised, with `--ignored --nocapture`.
    /// The number of threads is rayon's, which `RAYON_NUM_THREADS` sets
    #[test]
    #[ignore = "a timing, which prints what it measures"]
    fn timing_of_the_mutual_inductances_of_st40s_inner_vacuum_vessel() {
        // ST40's passive conductors as the pf_passive IDS holds them. The loop named `IVC` is the inner vacuum vessel, with 388 elements:
        // after its name it holds `"oblique_r": [ ... ]`, and the same for the other five
        let path: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../documentation/flux_from_parallelogram_conductor/code/st40_pf_passive_oblique.json"
        );
        let json: String = std::fs::read_to_string(path).expect("the passive conductors of ST40 can be read");
        let i_loop_name: usize = json.find("\"name\": \"IVC\"").expect("there is a loop named IVC");
        let n_key: usize = 6;
        let keys: [&str; 6] = [
            "oblique_r",
            "oblique_z",
            "oblique_length_alpha",
            "oblique_length_beta",
            "oblique_alpha",
            "oblique_beta",
        ];
        let mut keys_values: Vec<Array1<f64>> = Vec::with_capacity(n_key);
        for i_key in 0..n_key {
            let list_opening: String = format!("\"{}\": [", keys[i_key]);
            let i_list_start: usize = i_loop_name + json[i_loop_name..].find(&list_opening).expect("the loop has this list") + list_opening.len();
            let i_list_end: usize = i_list_start + json[i_list_start..].find(']').expect("the list is closed");
            let mut values: Vec<f64> = Vec::new();
            for number in json[i_list_start..i_list_end].split(',') {
                values.push(number.trim().parse::<f64>().expect("the list holds numbers"));
            }
            keys_values.push(Array1::from_vec(values));
        }
        let oblique_r: &Array1<f64> = &keys_values[0];
        let oblique_z: &Array1<f64> = &keys_values[1];
        let oblique_length_alpha: &Array1<f64> = &keys_values[2];
        let oblique_length_beta: &Array1<f64> = &keys_values[3];
        let oblique_alpha: &Array1<f64> = &keys_values[4];
        let oblique_beta: &Array1<f64> = &keys_values[5];
        let n_conductor: usize = oblique_r.len();
        let n_pair: usize = n_conductor * (n_conductor + 1) / 2;
        println!(
            "{n_conductor} conductors, {n_pair} pairs with `i_conductor_prime >= i_conductor`, {} thread(s)",
            rayon::current_num_threads()
        );

        let settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[0];

        // The whole matrix: the fastest of a few repeats, as the machine is shared
        let n_repeat: usize = 3;
        let mut duration_min: f64 = f64::INFINITY;
        let mut n_entry_finite: usize = 0;
        let mut mutual_inductance_sum: f64 = f64::NAN;
        for _i_repeat in 0..n_repeat {
            let timing_start: Instant = Instant::now();
            let mutual_inductance: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                oblique_r,
                oblique_z,
                oblique_length_alpha,
                oblique_length_beta,
                oblique_alpha,
                oblique_beta,
                oblique_r,
                oblique_z,
                oblique_length_alpha,
                oblique_length_beta,
                oblique_alpha,
                oblique_beta,
                true,
                settings_greens_finite_size,
            )
            .expect("one of the two calculations converges for every pair of pieces");
            duration_min = duration_min.min(timing_start.elapsed().as_secs_f64());
            n_entry_finite = 0;
            mutual_inductance_sum = 0.0;
            for i_conductor in 0..n_conductor {
                for i_conductor_prime in 0..n_conductor {
                    if mutual_inductance[(i_conductor, i_conductor_prime)].is_finite() {
                        n_entry_finite += 1;
                        mutual_inductance_sum += mutual_inductance[(i_conductor, i_conductor_prime)];
                    }
                }
            }
        }
        println!(
            "orders ({}, {}, {}), tolerance {:e}: the matrix takes {:.4} second; {n_entry_finite} of its {} entries are finite, and their sum is {:.17e} henry",
            settings_greens_finite_size.n_elliptic,
            settings_greens_finite_size.n_thin_ring,
            settings_greens_finite_size.n_multipole,
            settings_greens_finite_size.relative_tolerance,
            duration_min,
            n_conductor * n_conductor,
            mutual_inductance_sum
        );

        // Which calculation each pair takes, which is not timed
        let mut n_pair_far_field: usize = 0;
        let mut n_pair_near_field: usize = 0;
        let mut n_pair_halved: usize = 0;
        let mut n_piece_pair_far_field: usize = 0;
        let mut n_piece_pair_near_field: usize = 0;
        let mut n_piece_pair_of_one_pair_max: usize = 0;
        let mut pairs_far_field_i_conductor: Vec<usize> = Vec::new();
        let mut pairs_far_field_i_conductor_prime: Vec<usize> = Vec::new();
        for i_conductor in 0..n_conductor {
            let conductor: ReferenceConductor = ReferenceConductor {
                oblique_r: oblique_r[i_conductor],
                oblique_z: oblique_z[i_conductor],
                oblique_length_alpha: oblique_length_alpha[i_conductor],
                oblique_length_beta: oblique_length_beta[i_conductor],
                oblique_alpha: oblique_alpha[i_conductor],
                oblique_beta: oblique_beta[i_conductor],
            };
            for i_conductor_prime in i_conductor..n_conductor {
                let conductor_prime: ReferenceConductor = ReferenceConductor {
                    oblique_r: oblique_r[i_conductor_prime],
                    oblique_z: oblique_z[i_conductor_prime],
                    oblique_length_alpha: oblique_length_alpha[i_conductor_prime],
                    oblique_length_beta: oblique_length_beta[i_conductor_prime],
                    oblique_alpha: oblique_alpha[i_conductor_prime],
                    oblique_beta: oblique_beta[i_conductor_prime],
                };
                let pair: MutualInductanceParallelogram = mutual_inductance_and_pieces_of_one_pair(&conductor, &conductor_prime, settings_greens_finite_size)
                    .expect("one of the two calculations converges for every pair of pieces");
                let n_piece_pair: usize = pair.n_piece_pair_far_field + pair.n_piece_pair_near_field;
                if n_piece_pair > 1 {
                    n_pair_halved += 1;
                } else if pair.n_piece_pair_far_field == 1 {
                    n_pair_far_field += 1;
                    pairs_far_field_i_conductor.push(i_conductor);
                    pairs_far_field_i_conductor_prime.push(i_conductor_prime);
                } else {
                    n_pair_near_field += 1;
                }
                n_piece_pair_far_field += pair.n_piece_pair_far_field;
                n_piece_pair_near_field += pair.n_piece_pair_near_field;
                n_piece_pair_of_one_pair_max = n_piece_pair_of_one_pair_max.max(n_piece_pair);
            }
        }
        println!(
            "    pairs: {n_pair_far_field} far-field series, {n_pair_near_field} near-field formula, {n_pair_halved} halved (into at most \
             {n_piece_pair_of_one_pair_max} pairs of pieces); pairs of pieces: {n_piece_pair_far_field} far-field series, {n_piece_pair_near_field} \
             near-field formula"
        );

        // The accuracy of the first settings on this geometry: each entry against the same entry with the second settings, whose orders are
        // much higher and whose tolerance is 1000 times smaller. Compared with the entry itself, and with the geometric mean of the two
        // self-inductances of its row and column, which is the size that matters where the matrix is used
        let mut matrices: Vec<Array2<f64>> = Vec::with_capacity(N_REFERENCE_SETTING);
        for i_setting in 0..N_REFERENCE_SETTING {
            let timing_start: Instant = Instant::now();
            let mutual_inductance: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                oblique_r,
                oblique_z,
                oblique_length_alpha,
                oblique_length_beta,
                oblique_alpha,
                oblique_beta,
                oblique_r,
                oblique_z,
                oblique_length_alpha,
                oblique_length_beta,
                oblique_alpha,
                oblique_beta,
                true,
                &REFERENCE_SETTINGS[i_setting],
            )
            .expect("one of the two calculations converges for every pair of pieces");
            println!(
                "    orders ({}, {}, {}), tolerance {:e}: the matrix takes {:.4} second",
                REFERENCE_SETTINGS[i_setting].n_elliptic,
                REFERENCE_SETTINGS[i_setting].n_thin_ring,
                REFERENCE_SETTINGS[i_setting].n_multipole,
                REFERENCE_SETTINGS[i_setting].relative_tolerance,
                timing_start.elapsed().as_secs_f64()
            );
            matrices.push(mutual_inductance);
        }
        let mut relative_difference_max: f64 = 0.0;
        let mut relative_difference_diagonal_max: f64 = 0.0;
        let mut difference_over_self_inductances_max: f64 = 0.0;
        for i_conductor in 0..n_conductor {
            for i_conductor_prime in 0..n_conductor {
                let difference: f64 = (matrices[0][(i_conductor, i_conductor_prime)] - matrices[1][(i_conductor, i_conductor_prime)]).abs();
                let relative_difference: f64 = difference / matrices[1][(i_conductor, i_conductor_prime)].abs();
                relative_difference_max = relative_difference_max.max(relative_difference);
                if i_conductor == i_conductor_prime {
                    relative_difference_diagonal_max = relative_difference_diagonal_max.max(relative_difference);
                }
                let self_inductances_mean: f64 = (matrices[1][(i_conductor, i_conductor)] * matrices[1][(i_conductor_prime, i_conductor_prime)]).sqrt();
                difference_over_self_inductances_max = difference_over_self_inductances_max.max(difference / self_inductances_mean);
            }
        }
        println!(
            "    the first settings against the second: largest relative difference {relative_difference_max:.2e} (of the self-inductances \
             {relative_difference_diagonal_max:.2e}); largest difference over the geometric mean of the two self-inductances \
             {difference_over_self_inductances_max:.2e}"
        );

        // The far-field series on its own, for the pairs which take it, at three orders, on one thread
        let n_order: usize = 3;
        let orders_n_multipole: [usize; 3] = [8, 12, 20];
        for i_order in 0..n_order {
            let n_multipole: usize = orders_n_multipole[i_order];
            let mut conductors_shape: Vec<ConductorShape> = Vec::with_capacity(n_conductor);
            let mut conductors_centre_r: Vec<f64> = Vec::with_capacity(n_conductor);
            let mut conductors_centre_z: Vec<f64> = Vec::with_capacity(n_conductor);
            for i_conductor in 0..n_conductor {
                let shape: ConductorShape = ConductorShape::new(
                    oblique_length_alpha[i_conductor],
                    oblique_length_beta[i_conductor],
                    oblique_alpha[i_conductor],
                    oblique_beta[i_conductor],
                    n_multipole,
                );
                conductors_centre_r.push(oblique_r[i_conductor] + (0.5 * shape.side_alpha_r + 0.5 * shape.side_beta_r));
                conductors_centre_z.push(oblique_z[i_conductor] + (0.5 * shape.side_alpha_z + 0.5 * shape.side_beta_z));
                conductors_shape.push(shape);
            }
            let mut work_arrays: FarFieldPairWorkArrays = FarFieldPairWorkArrays::new(n_multipole);
            let mut duration_min: f64 = f64::INFINITY;
            let mut mutual_inductance_sum: f64 = f64::NAN;
            for _i_repeat in 0..n_repeat {
                let timing_start: Instant = Instant::now();
                mutual_inductance_sum = 0.0;
                for i_pair in 0..n_pair_far_field {
                    let i_conductor: usize = pairs_far_field_i_conductor[i_pair];
                    let i_conductor_prime: usize = pairs_far_field_i_conductor_prime[i_pair];
                    mutual_inductance_sum += mutual_inductance_parallelogram_far_field(
                        conductors_centre_r[i_conductor],
                        conductors_centre_r[i_conductor_prime],
                        conductors_centre_z[i_conductor] - conductors_centre_z[i_conductor_prime],
                        &conductors_shape[i_conductor].moment_scaled,
                        &conductors_shape[i_conductor_prime].moment_scaled,
                        n_multipole,
                        &mut work_arrays,
                    );
                }
                duration_min = duration_min.min(timing_start.elapsed().as_secs_f64());
            }
            println!(
                "    far-field series of order {n_multipole}: {:.2} microsecond for each pair (sum {:.17e} henry)",
                1.0e6 * duration_min / (n_pair_far_field as f64),
                mutual_inductance_sum
            );
        }
    }
}

#[cfg(test)]
mod tests_near_field {
    use super::*;
    use std::time::Instant;

    /// A pair of conductors of the table of reference values. Each conductor is its reference corner and its two sides:
    /// `[oblique_r, oblique_z, side_alpha_r, side_alpha_z, side_beta_r, side_beta_z]`, all in [metre]
    struct ReferencePair {
        conductor: [f64; 6],
        conductor_prime: [f64; 6],
        /// The mutual inductance with each of the orders of `REFERENCE_ORDERS_N_ELLIPTIC` and `REFERENCE_ORDERS_N_THIN_RING`, [henry]
        mutual_inductance: [f64; N_REFERENCE_ORDER],
    }

    /// A pair of conductors of the table of high-precision values, described as in `ReferencePair`
    struct HighPrecisionPair {
        conductor: [f64; 6],
        conductor_prime: [f64; 6],
        /// The mutual inductance, [henry]
        mutual_inductance: f64,
    }

    // The mutual inductance of 14 pairs of conductors with four pairs of orders: the lowest, low, the default and high. The values were
    // calculated with the code which accompanies the note, in `documentation/flux_from_parallelogram_conductor/code/`, which sums every
    // term. Every number is written with as many digits as it takes to read it back as the same double, which is at most 17
    const N_REFERENCE_ORDER: usize = 4;
    const REFERENCE_ORDERS_N_ELLIPTIC: [usize; N_REFERENCE_ORDER] = [0, 2, 6, 8];
    const REFERENCE_ORDERS_N_THIN_RING: [usize; N_REFERENCE_ORDER] = [0, 4, 12, 16];
    const N_REFERENCE_PAIR: usize = 14;
    #[rustfmt::skip]
    const REFERENCE_PAIRS: [ReferencePair; N_REFERENCE_PAIR] = [
        // self: a thin tall rectangle, an element of the inboard wall
        ReferencePair { conductor: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734], conductor_prime: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734],
            mutual_inductance: [6.463859118425479e-07, 6.470065690788878e-07, 6.470065692853485e-07, 6.470065692853486e-07] },
        // self: a square, the grid cell of the GSFit paper
        ReferencePair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125],
            mutual_inductance: [2.254070461812775e-06, 2.254199119864153e-06, 2.25419911986401e-06, 2.25419911986401e-06] },
        // self: a sheared and rotated parallelogram
        ReferencePair { conductor: [0.33, 0.05, 0.032878044949658264, 0.012001423260940797, 0.005442887103599504, 0.021316073277634184], conductor_prime: [0.33, 0.05, 0.032878044949658264, 0.012001423260940797, 0.005442887103599504, 0.021316073277634184],
            mutual_inductance: [1.5064510850830105e-06, 1.5070874085922848e-06, 1.5070874084361976e-06, 1.5070874084361972e-06] },
        // self: a sheared and rotated parallelogram whose two sides are in clockwise order
        ReferencePair { conductor: [0.52, -0.31, -0.0203901214681414, 0.005024235913493632, -0.01324022363849412, 0.0313160737960981], conductor_prime: [0.52, -0.31, -0.0203901214681414, 0.005024235913493632, -0.01324022363849412, 0.0313160737960981],
            mutual_inductance: [2.428484410105687e-06, 2.4289979422296238e-06, 2.428997942230654e-06, 2.428997942230654e-06] },
        // two thin tall rectangles which share a whole side
        ReferencePair { conductor: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734], conductor_prime: [0.15357, 0.0240734, 0.0032351, 0.0, -0.0, 0.0240734],
            mutual_inductance: [4.000837690056891e-07, 4.0357105547599114e-07, 4.035710710493005e-07, 4.0357107104929994e-07] },
        // two squares which share part of a side
        ReferencePair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.41625, -0.00125, 0.0125, 0.0, -0.0, 0.0125],
            mutual_inductance: [1.8356552291966404e-06, 1.8360252084841975e-06, 1.836025208464956e-06, 1.836025208464956e-06] },
        // two squares which touch at one corner
        ReferencePair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.41625, 0.00625, 0.0125, 0.0, -0.0, 0.0125],
            mutual_inductance: [1.695144003781951e-06, 1.6958547902126197e-06, 1.6958547901933304e-06, 1.6958547901933304e-06] },
        // two squares which overlap, the second rotated by 20 degrees: edges which cross
        ReferencePair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.4094886953317308, -0.004442810769794971, 0.011746157759823857, 0.004275251791570859, -0.004275251791570859, 0.011746157759823857],
            mutual_inductance: [2.1691757947342844e-06, 2.169354453668545e-06, 2.1693544536678217e-06, 2.1693544536678217e-06] },
        // a square inside a 40 mm square
        ReferencePair { conductor: [0.41, -0.012, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.4, -0.02, 0.04, 0.0, -0.0, 0.04],
            mutual_inductance: [1.794839675577003e-06, 1.7945379516213826e-06, 1.7945379512575555e-06, 1.7945379512575546e-06] },
        // a thin tall rectangle and itself rotated by 5 degrees about its reference corner
        ReferencePair { conductor: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734], conductor_prime: [0.15357, 0.0, 0.003222789467796606, 0.0002819575433629489, -0.002098135057461474, 0.023981793445041824],
            mutual_inductance: [6.359903714490272e-07, 6.36628531849557e-07, 6.366285320674697e-07, 6.366285320674699e-07] },
        // two squares 4 mm apart
        ReferencePair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.42025, -0.00325, 0.0125, 0.0, -0.0, 0.0125],
            mutual_inductance: [1.731297536310239e-06, 1.7317468928211154e-06, 1.7317468927723707e-06, 1.7317468927723707e-06] },
        // two sheared parallelograms which share a whole side
        ReferencePair { conductor: [0.33, 0.05, 0.032878044949658264, 0.012001423260940797, 0.005442887103599504, 0.021316073277634184], conductor_prime: [0.33544288710359954, 0.07131607327763419, 0.032878044949658264, 0.012001423260940797, 0.005442887103599504, 0.021316073277634184],
            mutual_inductance: [1.234857339008548e-06, 1.2367426081094802e-06, 1.2367426084983878e-06, 1.236742608498392e-06] },
        // the clockwise parallelogram and a rectangle which overlaps it
        ReferencePair { conductor: [0.52, -0.31, -0.0203901214681414, 0.005024235913493632, -0.01324022363849412, 0.0313160737960981], conductor_prime: [0.49, -0.3, 0.029850124958340774, 0.0029950024994048444, -0.001996668332936563, 0.019900083305560517],
            mutual_inductance: [2.3933549451149485e-06, 2.3931904058984458e-06, 2.3931904058876084e-06, 2.3931904058876084e-06] },
        // a sheared parallelogram and a square 3 mm from it, described clockwise
        ReferencePair { conductor: [0.33, 0.05, 0.032878044949658264, 0.012001423260940797, 0.005442887103599504, 0.021316073277634184], conductor_prime: [0.372, 0.06, 7.347880794884119e-19, 0.012, 0.012, 7.347880794884119e-19],
            mutual_inductance: [1.2011195115137032e-06, 1.204085673529935e-06, 1.204085677087191e-06, 1.2040856770871939e-06] },
    ];

    // The ten pairs of `documentation/flux_from_parallelogram_conductor/verification/mutual_inductance_reference.json` whose `t` is at
    // most 0.2, each labelled with its name and its description in that file. Their values were calculated to 30 digits by two integrals
    // which do not use the series, and are rounded here to doubles
    const N_HIGH_PRECISION_PAIR: usize = 10;
    #[rustfmt::skip]
    const HIGH_PRECISION_PAIRS: [HighPrecisionPair; N_HIGH_PRECISION_PAIR] = [
        // `self_thin_tall`: self-inductance of a thin tall rectangle, 3.2351 mm x 24.0734 mm, an element of the inner wall
        HighPrecisionPair { conductor: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734], conductor_prime: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734],
            mutual_inductance: 6.470065692853491e-07 },
        // `self_square`: self-inductance of a 12.5 mm square centred at R = 0.41 m, the grid cell of the GSFit paper
        HighPrecisionPair { conductor: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125], conductor_prime: [0.40375, -0.00625, 0.0125, 0.0, -0.0, 0.0125],
            mutual_inductance: 2.2541991198640103e-06 },
        // `self_sheared_rotated`: self-inductance of a sheared, rotated parallelogram, 34 mm x 36 mm, with an interior angle of 58.5 degrees, near R = 0.33 m, Z = 0.62 m
        HighPrecisionPair { conductor: [0.33, 0.62, 0.03248144063027061, 0.010047687026485546, 0.008906542533162825, 0.034880847181583206], conductor_prime: [0.33, 0.62, 0.03248144063027061, 0.010047687026485546, 0.008906542533162825, 0.034880847181583206],
            mutual_inductance: 1.4233877727589298e-06 },
        // `touching_stacked_thin_tall`: two thin tall rectangles stacked: they share a whole short side
        HighPrecisionPair { conductor: [0.15357, 0.0, 0.0032351, 0.0, -0.0, 0.0240734], conductor_prime: [0.15357, 0.0240734, 0.0032351, 0.0, -0.0, 0.0240734],
            mutual_inductance: 4.035710710492987e-07 },
        // `touching_part_of_side`: two rectangles, 30 mm x 20 mm and 25 mm x 12 mm, one on top of the other and displaced sideways: they share 13 mm of a side
        HighPrecisionPair { conductor: [0.52, 0.0, 0.03, 0.0, -0.0, 0.02], conductor_prime: [0.537, 0.02, 0.025, 0.0, -0.0, 0.012],
            mutual_inductance: 2.237355500768218e-06 },
        // `touching_vertex`: two rectangles, 31.25 mm x 20 mm and 18 mm x 26 mm, which touch at one vertex only (diagonally opposite)
        HighPrecisionPair { conductor: [0.375, 0.0, 0.03125, 0.0, -0.0, 0.02], conductor_prime: [0.40625, 0.02, 0.018, 0.0, -0.0, 0.026],
            mutual_inductance: 1.3013836136301075e-06 },
        // `touching_rotated_5_degrees`: two 22 mm x 30 mm rectangles which would share a 30 mm side, the second rotated by 5 degrees about the shared reference corner: they share one vertex, and their nearest sides open at 5 degrees
        HighPrecisionPair { conductor: [0.45, 0.2, 0.022, 0.0, -0.0, 0.03], conductor_prime: [0.45, 0.2, -0.002614672282429747, 0.029885840942752366, -0.0219162833580184, -0.001917426340448481],
            mutual_inductance: 1.682966902992877e-06 },
        // `overlapping_rotated_20_degrees`: a 30 mm x 20 mm rectangle and a 25 mm x 18 mm rectangle rotated by 20 degrees, which overlap partially: their edges cross twice, and one vertex of each is inside the other
        HighPrecisionPair { conductor: [0.3, -0.4, 0.03, 0.0, -0.0, 0.02], conductor_prime: [0.315, -0.392, 0.023492315519647713, 0.008550503583141718, -0.006156362579862036, 0.01691446717414635],
            mutual_inductance: 1.2501908167258603e-06 },
        // `overlapping_small_inside_large`: a 12 mm x 9 mm rectangle wholly inside a 40 mm x 35 mm rectangle
        HighPrecisionPair { conductor: [0.6, -0.02, 0.04, 0.0, -0.0, 0.035], conductor_prime: [0.611, -0.008, 0.012, 0.0, -0.0, 0.009],
            mutual_inductance: 3.040185189733014e-06 },
        // `near_sheared_quarter_gap`: two sheared parallelograms, 32 mm x 28 mm and 30 mm x 28 mm, side by side with a gap of about 7 mm between parallel sides
        HighPrecisionPair { conductor: [0.7, 0.3, 0.03136213049091973, 0.006357418585441959, 0.008274565786517507, 0.02674942169551697], conductor_prime: [0.7392, 0.3079, 0.029401997335237247, 0.005960079923851836, 0.008274565786517507, 0.02674942169551697],
            mutual_inductance: 2.829314056433516e-06 },
    ];

    /// How far two values of a mutual inductance which differ by their rounding only can be apart, compared with them. The product of the
    /// two distances is a difference of double integrals, which are larger than it where two edges are long compared with the distance
    /// between their lines: the rounding error is about 1e-16 times the square of that ratio. Measured: up to 1.6e-14 for two thin tall
    /// rectangles of the inboard wall, one on top of the other, which are 15 times as long as wide together, and up to 4e-15 for the
    /// other pairs of these tests
    const ROUNDING_TOLERANCE: f64 = 3.0e-14;

    /// The mutual inductance of a pair of conductors which are described as in `ReferencePair`
    fn mutual_inductance_of_pair(
        conductor: &[f64; 6],
        conductor_prime: &[f64; 6],
        coefficients: &PairSeriesCoefficients,
        work: &mut NearFieldPairWorkArrays,
    ) -> f64 {
        mutual_inductance_parallelogram_near_field(
            conductor[0],
            conductor[1],
            conductor[2],
            conductor[3],
            conductor[4],
            conductor[5],
            conductor_prime[0],
            conductor_prime[1],
            conductor_prime[2],
            conductor_prime[3],
            conductor_prime[4],
            conductor_prime[5],
            coefficients,
            work,
        )
    }

    #[test]
    fn the_mutual_inductance_agrees_with_the_reference_values() {
        for i_order in 0..N_REFERENCE_ORDER {
            let n_elliptic: usize = REFERENCE_ORDERS_N_ELLIPTIC[i_order];
            let n_thin_ring: usize = REFERENCE_ORDERS_N_THIN_RING[i_order];
            let coefficients: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(n_elliptic, n_thin_ring);
            for i_pair in 0..N_REFERENCE_PAIR {
                let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
                let mutual_inductance: f64 = mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients, &mut work);
                let relative_difference: f64 = (mutual_inductance / pair.mutual_inductance[i_order] - 1.0).abs();
                assert!(
                    relative_difference < ROUNDING_TOLERANCE,
                    "orders ({n_elliptic}, {n_thin_ring}), pair {i_pair}: {mutual_inductance:e} henry differs from the reference value by {relative_difference:e}"
                );
            }
        }
    }

    #[test]
    fn the_mutual_inductance_agrees_with_the_high_precision_values() {
        // With these orders the series has converged for every `t` up to 0.2, and what is left is the rounding
        let coefficients: PairSeriesCoefficients = pair_series_coefficients(8, 16);
        let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(8, 16);
        for i_pair in 0..N_HIGH_PRECISION_PAIR {
            let pair: &HighPrecisionPair = &HIGH_PRECISION_PAIRS[i_pair];
            let mutual_inductance: f64 = mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients, &mut work);
            let relative_difference: f64 = (mutual_inductance / pair.mutual_inductance - 1.0).abs();
            assert!(
                relative_difference < ROUNDING_TOLERANCE,
                "pair {i_pair}: {mutual_inductance:e} henry differs from the high-precision value by {relative_difference:e}"
            );
        }
    }

    #[test]
    fn exchanging_the_two_conductors_changes_only_the_rounding() {
        for i_order in 0..N_REFERENCE_ORDER {
            let n_elliptic: usize = REFERENCE_ORDERS_N_ELLIPTIC[i_order];
            let n_thin_ring: usize = REFERENCE_ORDERS_N_THIN_RING[i_order];
            let coefficients: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(n_elliptic, n_thin_ring);
            for i_pair in 0..N_REFERENCE_PAIR {
                let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
                let mutual_inductance: f64 = mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients, &mut work);
                let mutual_inductance_exchanged: f64 = mutual_inductance_of_pair(&pair.conductor_prime, &pair.conductor, &coefficients, &mut work);
                let relative_difference: f64 = (mutual_inductance_exchanged / mutual_inductance - 1.0).abs();
                assert!(
                    relative_difference < ROUNDING_TOLERANCE,
                    "orders ({n_elliptic}, {n_thin_ring}), pair {i_pair}: exchanging the conductors changes the mutual inductance by {relative_difference:e}"
                );
            }
        }
    }

    #[test]
    fn the_same_conductor_described_from_another_corner_gives_the_same_mutual_inductance() {
        let coefficients: PairSeriesCoefficients = pair_series_coefficients(6, 12);
        let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(6, 12);

        // A sheared and rotated parallelogram: its reference corner and its two sides
        let corner_r: f64 = 0.33;
        let corner_z: f64 = 0.05;
        let side_alpha_r: f64 = 0.035 * 0.35_f64.cos();
        let side_alpha_z: f64 = 0.035 * 0.35_f64.sin();
        let side_beta_r: f64 = -0.022 * (-0.25_f64).sin();
        let side_beta_z: f64 = 0.022 * (-0.25_f64).cos();

        // The same parallelogram from each of its four corners, counter-clockwise and clockwise
        let n_description: usize = 8;
        let descriptions: [[f64; 6]; 8] = [
            [corner_r, corner_z, side_alpha_r, side_alpha_z, side_beta_r, side_beta_z],
            [corner_r, corner_z, side_beta_r, side_beta_z, side_alpha_r, side_alpha_z],
            [
                corner_r + side_alpha_r,
                corner_z + side_alpha_z,
                side_beta_r,
                side_beta_z,
                -side_alpha_r,
                -side_alpha_z,
            ],
            [
                corner_r + side_alpha_r,
                corner_z + side_alpha_z,
                -side_alpha_r,
                -side_alpha_z,
                side_beta_r,
                side_beta_z,
            ],
            [
                corner_r + side_alpha_r + side_beta_r,
                corner_z + side_alpha_z + side_beta_z,
                -side_alpha_r,
                -side_alpha_z,
                -side_beta_r,
                -side_beta_z,
            ],
            [
                corner_r + side_alpha_r + side_beta_r,
                corner_z + side_alpha_z + side_beta_z,
                -side_beta_r,
                -side_beta_z,
                -side_alpha_r,
                -side_alpha_z,
            ],
            [
                corner_r + side_beta_r,
                corner_z + side_beta_z,
                -side_beta_r,
                -side_beta_z,
                side_alpha_r,
                side_alpha_z,
            ],
            [
                corner_r + side_beta_r,
                corner_z + side_beta_z,
                side_alpha_r,
                side_alpha_z,
                -side_beta_r,
                -side_beta_z,
            ],
        ];

        // Its self-inductance, and its mutual inductance with a square which overlaps it
        let other: [f64; 6] = [0.345, 0.055, 0.012, 0.004, -0.004, 0.012];
        let self_inductance: f64 = mutual_inductance_of_pair(&descriptions[0], &descriptions[0], &coefficients, &mut work);
        let mutual_inductance: f64 = mutual_inductance_of_pair(&descriptions[0], &other, &coefficients, &mut work);
        for i_description in 1..n_description {
            for j_description in 0..n_description {
                let self_inductance_described: f64 =
                    mutual_inductance_of_pair(&descriptions[i_description], &descriptions[j_description], &coefficients, &mut work);
                let relative_difference: f64 = (self_inductance_described / self_inductance - 1.0).abs();
                assert!(
                    relative_difference < ROUNDING_TOLERANCE,
                    "descriptions {i_description} and {j_description}: the self-inductance changes by {relative_difference:e}"
                );
            }
            let mutual_inductance_described: f64 = mutual_inductance_of_pair(&descriptions[i_description], &other, &coefficients, &mut work);
            let relative_difference: f64 = (mutual_inductance_described / mutual_inductance - 1.0).abs();
            assert!(
                relative_difference < ROUNDING_TOLERANCE,
                "description {i_description}: the mutual inductance with another conductor changes by {relative_difference:e}"
            );
        }
    }

    #[test]
    fn a_conductor_with_itself_is_the_mean_of_the_four_pairs_of_its_halves() {
        // The current density is uniform, so a conductor is its two halves, each with half of the current: its self-inductance is the mean
        // of the self-inductances of the two halves and, twice, of their mutual inductance. The halves share a whole side. With orders
        // at which the series has converged, the two agree to the rounding
        let coefficients: PairSeriesCoefficients = pair_series_coefficients(8, 16);
        let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(8, 16);
        let n_conductor: usize = 3;
        for i_conductor in 0..n_conductor {
            let whole: &[f64; 6] = &REFERENCE_PAIRS[i_conductor].conductor;
            let self_inductance: f64 = mutual_inductance_of_pair(whole, whole, &coefficients, &mut work);

            // Halved along its first side, and along its second
            let n_halving: usize = 2;
            for i_halving in 0..n_halving {
                let mut half: [f64; 6] = *whole;
                let mut half_other: [f64; 6] = *whole;
                let i_side: usize = 2 + 2 * i_halving;
                half[i_side] = 0.5 * whole[i_side];
                half[i_side + 1] = 0.5 * whole[i_side + 1];
                half_other[i_side] = half[i_side];
                half_other[i_side + 1] = half[i_side + 1];
                half_other[0] = whole[0] + half[i_side];
                half_other[1] = whole[1] + half[i_side + 1];

                let mean_of_halves: f64 = 0.25
                    * (mutual_inductance_of_pair(&half, &half, &coefficients, &mut work)
                        + mutual_inductance_of_pair(&half, &half_other, &coefficients, &mut work)
                        + mutual_inductance_of_pair(&half_other, &half, &coefficients, &mut work)
                        + mutual_inductance_of_pair(&half_other, &half_other, &coefficients, &mut work));
                let relative_difference: f64 = (mean_of_halves / self_inductance - 1.0).abs();
                assert!(
                    relative_difference < ROUNDING_TOLERANCE,
                    "conductor {i_conductor}, halving {i_halving}: the mean of the halves differs from the whole by {relative_difference:e}"
                );
            }
        }
    }

    #[test]
    fn the_terms_which_are_left_out_cannot_change_the_sum() {
        // With `negligible_term = 0` every term of the truncated series is summed, as in the reference values
        for i_order in 0..N_REFERENCE_ORDER {
            let n_elliptic: usize = REFERENCE_ORDERS_N_ELLIPTIC[i_order];
            let n_thin_ring: usize = REFERENCE_ORDERS_N_THIN_RING[i_order];
            let coefficients: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            let mut coefficients_every_term: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            coefficients_every_term.negligible_term = 0.0;
            let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(n_elliptic, n_thin_ring);
            for i_pair in 0..N_REFERENCE_PAIR {
                let pair: &ReferencePair = &REFERENCE_PAIRS[i_pair];
                let mutual_inductance: f64 = mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients, &mut work);
                let mutual_inductance_every_term: f64 = mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients_every_term, &mut work);
                let relative_difference: f64 = (mutual_inductance / mutual_inductance_every_term - 1.0).abs();
                assert!(
                    relative_difference < 5.0e-15,
                    "orders ({n_elliptic}, {n_thin_ring}), pair {i_pair}: the terms which are left out change the sum by {relative_difference:e}"
                );
            }
        }
    }

    #[test]
    fn the_lowest_order_is_the_geometric_mean_distance_formula() {
        // A rectangle of the width `width` and the height `height` with itself: Maxwell's formula for the logarithm of the geometric mean
        // distance of a rectangle from itself (A Treatise on Electricity and Magnetism, article 692)
        let coefficients: PairSeriesCoefficients = pair_series_coefficients(0, 0);
        let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(0, 0);
        let width: f64 = 0.0125;
        let height: f64 = 0.02;
        let centre_r: f64 = 0.41;
        let rectangle: [f64; 6] = [centre_r - 0.5 * width, -0.5 * height, width, 0.0, 0.0, height];
        let mutual_inductance: f64 = mutual_inductance_of_pair(&rectangle, &rectangle, &coefficients, &mut work);

        let ln_geometric_mean_distance: f64 = (width.powi(2) + height.powi(2)).sqrt().ln()
            - (width.powi(2) / (6.0 * height.powi(2))) * (1.0 + height.powi(2) / width.powi(2)).sqrt().ln()
            - (height.powi(2) / (6.0 * width.powi(2))) * (1.0 + width.powi(2) / height.powi(2)).sqrt().ln()
            + (2.0 * width / (3.0 * height)) * (height / width).atan()
            + (2.0 * height / (3.0 * width)) * (width / height).atan()
            - 25.0 / 12.0;
        let expected: f64 = MU_0 * centre_r * ((8.0 * centre_r).ln() - 2.0 - ln_geometric_mean_distance);
        let relative_difference: f64 = (mutual_inductance / expected - 1.0).abs();
        assert!(
            relative_difference < 1.0e-14,
            "{mutual_inductance:e} henry differs from the geometric-mean-distance formula by {relative_difference:e}"
        );
    }

    #[test]
    fn the_coefficients_start_with_the_two_divisions_of_the_lowest_term() {
        // The lowest term of the Green's function is `ln(4) - 2 - ln|rho|`; the two divergence theorems divide it by 2 and by 3, and
        // subtract 1/2 + 1/3 from the logarithm
        let coefficients: PairSeriesCoefficients = pair_series_coefficients(2, 4);
        assert!((coefficients.coefficient_ln[0] - 1.0 / 6.0).abs() < 1.0e-16);
        assert!((coefficients.coefficient_plain[0] - (4.0_f64.ln() - 2.0 + 5.0 / 6.0) / 6.0).abs() < 1.0e-16);
        assert_eq!(coefficients.n_order_max, 12);
        assert_eq!(coefficients.coefficient_plain.len(), 5 * coefficients.n_half_triangle);
    }

    /// How long the mutual inductance of one pair of conductors takes, with four pairs of orders, for three pairs of the shapes of ST40's
    /// inner wall: a conductor with itself, two conductors which share a side, and two whose edges cross. Nothing is checked: it prints
    /// what it measures, so it is run on its own, optimised, with `--ignored --nocapture`
    #[test]
    #[ignore = "a timing, which prints what it measures"]
    fn timing_of_the_mutual_inductance_of_one_pair() {
        let n_timed_order: usize = 4;
        let timed_orders_n_elliptic: [usize; 4] = [2, 4, 6, 8];
        let timed_orders_n_thin_ring: [usize; 4] = [4, 8, 12, 16];
        let n_timed_pair: usize = 5;
        let timed_pairs_name: [&str; 5] = [
            "a thin tall rectangle with itself",
            "two thin tall rectangles which share a side",
            "a thin tall rectangle and itself rotated by 5 degrees",
            "two squares, the second rotated by 20 degrees",
            "two sheared parallelograms which share a side",
        ];
        let timed_pairs_i_reference: [usize; 5] = [0, 4, 9, 7, 11];
        let n_repetition: usize = 200;

        for i_order in 0..n_timed_order {
            let n_elliptic: usize = timed_orders_n_elliptic[i_order];
            let n_thin_ring: usize = timed_orders_n_thin_ring[i_order];
            let start_coefficients: Instant = Instant::now();
            let coefficients: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            let seconds_coefficients: f64 = start_coefficients.elapsed().as_secs_f64();
            let mut coefficients_every_term: PairSeriesCoefficients = pair_series_coefficients(n_elliptic, n_thin_ring);
            coefficients_every_term.negligible_term = 0.0;
            let mut work: NearFieldPairWorkArrays = NearFieldPairWorkArrays::new(n_elliptic, n_thin_ring);
            println!(
                "orders ({n_elliptic}, {n_thin_ring}): the coefficients take {:.1} microsecond",
                1.0e6 * seconds_coefficients
            );
            for i_pair in 0..n_timed_pair {
                let pair: &ReferencePair = &REFERENCE_PAIRS[timed_pairs_i_reference[i_pair]];
                let mut sum: f64 = 0.0;
                let start: Instant = Instant::now();
                for _i_repetition in 0..n_repetition {
                    sum += mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients, &mut work);
                }
                let microsecond_per_pair: f64 = 1.0e6 * start.elapsed().as_secs_f64() / (n_repetition as f64);
                let mut sum_every_term: f64 = 0.0;
                let start_every_term: Instant = Instant::now();
                for _i_repetition in 0..n_repetition {
                    sum_every_term += mutual_inductance_of_pair(&pair.conductor, &pair.conductor_prime, &coefficients_every_term, &mut work);
                }
                let microsecond_per_pair_every_term: f64 = 1.0e6 * start_every_term.elapsed().as_secs_f64() / (n_repetition as f64);
                println!(
                    "    {:55} {microsecond_per_pair:9.1} microsecond per pair; with every term {microsecond_per_pair_every_term:9.1}; they differ by {:.1e}",
                    timed_pairs_name[i_pair],
                    sum / sum_every_term - 1.0
                );
            }
        }
    }
}
