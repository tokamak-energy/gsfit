//! `ConductorModel::FiniteSize` at points: the poloidal flux at points from toroidal conductors of finite size, which is the mutual
//! inductance between each conductor and a filament at each point, and its derivatives with respect to the point's position, up to the
//! third.
//!
//! Each conductor has a parallelogram cross-section, and carries a uniform current density. The flux per ampere is the average over the
//! cross-section of the Green's function of a filament. Nothing is discretised: the average is calculated from two analytic series, which
//! are derived in the note `documentation/flux_from_parallelogram_conductor/note/flux_from_parallelogram_conductor.pdf`. The comments here
//! use the note's symbols, and its section and equation numbers.
//! * `greens_psi_parallelogram_near_field` - the near-field series (section 3 of the note), for a point inside, on, or near a conductor
//! * `greens_psi_parallelogram_far_field` - the far-field series (section 4), for a point away from a conductor
//! * `greens_psi_parallelogram` - chooses between the two for each point, and halves the conductor where neither converges fast enough
//!   (section 5)
//!
//! The symbols of the note, and their names here:
//!
//! | In the note | Here |
//! | --- | --- |
//! | `k^2`, `q = 1 - k^2` | `k_sq`, `q` |
//! | `d^2`, `u^2`, `w^2`, `h` | `d_sq`, `u_sq`, `w_sq`, `h`; as polynomials in `delta_R`: `polynomial_d_sq`, `polynomial_u_sq`, `polynomial_w_sq` |
//! | `Phi`, `Theta` | `phi`, `theta`; their Taylor coefficients in `delta_R`: `series_phi`, `series_theta` |
//! | `x_n`, `y_n`, `z_n`, `S` | `mean_arithmetic`, `mean_geometric`, `mean_half_difference`, `sum_s` |
//! | `chi_n`, `gamma_n`, `c_(n-1)`, `kappa_(n-1)` | `chi`, `gamma`, `c_previous`, `kappa_previous` |
//! | `B_(n,j)`, and its derivative with respect to its upper argument | `binomial`, `binomial_derivative` |
//! | `P_ab`, `Q_ab` | `table_p`, `table_q`, indexed with `i_a`, `i_b` |
//! | `N_e`, `N_t`, `N_m`, `epsilon` | `n_elliptic`, `n_thin_ring`, `n_multipole`, `relative_tolerance` |
//! | `sigma` at the start and at the end of an edge, `s_1`, `s_2` | `sigma_start`, `sigma_end`, `s_start`, `s_end` |
//! | `d_e`, `exp(i phi_e)`, the edge's length `s_2 - s_1` | `edge_distance`, `phase`, `length` |
//! | `I_(a,b)`, `J_(a,b)`, `I_(m,-1)` | `integral_plain`, `integral_ln`, `integral_inverse` |
//! | `eta_ab` | `multiplicity` |
//! | `L_(alpha R)`, `L_(alpha Z)`, `L_(beta R)`, `L_(beta Z)` | `side_alpha_r`, `side_alpha_z`, `side_beta_r`, `side_beta_z` |
//! | `(R_c, Z_c)`, `H`, `d_c`, `t` | `centre_r`, `centre_z`, `half_diagonal`, `distance`, `t` |
//! | `delta_R`, `delta_Z` | `delta_r`, `delta_z` |
//! | `G_ij`, the moments `<delta_R^i delta_Z^j>`, the average of `lambda_alpha^m` | `taylor`, `moment`, `moment_one_side` |

use super::finite_size_series::{
    EllipticPhiTheta, ExpansionParameterThresholds, MaxwellSeriesCoefficients, check_conductors, elliptic_phi_theta, expansion_parameter_thresholds,
    maxwell_series_coefficients, parallelogram_moments,
};
use super::geometry::{Conductor, Sensor};
use crate::settings::settings_equilibrium::SettingsEquilibriumNumericsGreensFiniteSize;
use faer::c64;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::sync::OnceLock;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// The number of edges of a parallelogram
const N_EDGE: usize = 4;

/// The poloidal flux at points from toroidal conductors of finite size, per ampere in each conductor: the mutual inductance between each
/// conductor and a filament at each point.
///
/// Each conductor has a parallelogram cross-section, and carries a uniform current density. It is described as the IMAS data dictionary
/// describes it, `pf_passive/loop(i1)/element(i2)/geometry/oblique`: one of its corners, the reference corner, the lengths of the two sides
/// which meet there, and their inclinations. The flux is the average over the cross-section of the Green's function of a filament, Eq. (1)
/// of the note `documentation/flux_from_parallelogram_conductor/note/flux_from_parallelogram_conductor.pdf`, with the series chosen for each
/// point and each conductor by `greens_psi_parallelogram`. A point can be anywhere: inside a conductor, on its boundary, or outside it.
///
/// # Arguments
/// * `r` - R coordinate of the points, shape = [n_point], [metre]
/// * `z` - Z coordinate of the points, shape = [n_point], [metre]
/// * `oblique_r_prime` - R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_z_prime` - Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_length_alpha_prime` - Length of each conductor's side which is inclined at `oblique_alpha_prime`, shape = [n_conductor], [metre]
/// * `oblique_length_beta_prime` - Length of each conductor's side which is inclined at `oblique_beta_prime`, shape = [n_conductor], [metre]
/// * `oblique_alpha_prime` - Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
/// * `oblique_beta_prime` - Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
/// * `settings_greens_finite_size` - The orders of the two series, the relative accuracy wanted, and how many times a conductor may
///   be halved: `equilibrium_settings/numerics/greens/finite_size`
///
/// # Returns
/// * `g_psi` - The flux at each point per ampere in each conductor, shape = [n_point, n_conductor], [henry]
///
/// # Errors
/// * `r` and `z`, or the six descriptions of the conductors, do not have the same length
/// * a point is not at a positive major radius, `r <= 0`
/// * a conductor has no area: a side without a positive length, or two sides which are parallel
/// * a conductor is not wholly at a positive major radius
/// * the relative tolerance is not greater than 0 and at most 1
/// * neither series converges fast enough for a piece of a conductor which has already been halved `n_subdivision_max` times
#[allow(clippy::too_many_arguments)]
pub fn mutual_inductance_finite_size_to_point_source(
    r: &Array1<f64>,
    z: &Array1<f64>,
    oblique_r_prime: &Array1<f64>,
    oblique_z_prime: &Array1<f64>,
    oblique_length_alpha_prime: &Array1<f64>,
    oblique_length_beta_prime: &Array1<f64>,
    oblique_alpha_prime: &Array1<f64>,
    oblique_beta_prime: &Array1<f64>,
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<Array2<f64>, String> {
    let n_point: usize = r.len();
    if z.len() != n_point {
        return Err(format!("`r` and `z` have {} and {} point(s); each point needs one of each", n_point, z.len()));
    }

    // The Green's function of a filament needs the points, and the whole of every conductor, at a positive major radius
    for i_point in 0..n_point {
        let is_usable_point: bool = r[i_point] > 0.0 && r[i_point].is_finite() && z[i_point].is_finite();
        if !is_usable_point {
            return Err(format!(
                "point {i_point} is at r = {} and z = {} metre; r must be positive, and both finite",
                r[i_point], z[i_point]
            ));
        }
    }
    check_conductors(
        oblique_r_prime,
        oblique_z_prime,
        oblique_length_alpha_prime,
        oblique_length_beta_prime,
        oblique_alpha_prime,
        oblique_beta_prime,
        "_prime",
    )?;
    let n_conductor: usize = oblique_r_prime.len();

    // What every point and every conductor share: how small each series' expansion parameter has to be, and the near-field series' two
    // tables of coefficients
    let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(settings_greens_finite_size)?;
    let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring);

    // Each conductor is one task, which calculates the flux from that conductor at every point: one whole column of `g_psi`. The tasks are
    // large, and independent of each other
    let conductors_g_psi: Vec<Array1<f64>> = (0..n_conductor)
        .into_par_iter()
        .map(|i_conductor: usize| -> Result<Array1<f64>, String> {
            let greens_psi: GreensPsiParallelogram = greens_psi_parallelogram(
                r,
                z,
                oblique_r_prime[i_conductor],
                oblique_z_prime[i_conductor],
                oblique_length_alpha_prime[i_conductor],
                oblique_length_beta_prime[i_conductor],
                oblique_alpha_prime[i_conductor],
                oblique_beta_prime[i_conductor],
                &thresholds,
                &coefficients,
                settings_greens_finite_size,
            )
            .map_err(|error: String| format!("conductor {i_conductor}: {error}"))?;

            Ok(greens_psi.g_psi)
        })
        .collect::<Result<Vec<Array1<f64>>, String>>()?;

    let mut g_psi: Array2<f64> = Array2::from_elem((n_point, n_conductor), f64::NAN);
    for i_conductor in 0..n_conductor {
        g_psi.column_mut(i_conductor).assign(&conductors_g_psi[i_conductor]);
    }

    Ok(g_psi)
}

/// What `greens_psi_parallelogram` returns: the flux at the points from one conductor, and how it was calculated at each of them.
struct GreensPsiParallelogram {
    /// The flux at each point per ampere in the conductor, shape = [n_point], [henry]
    g_psi: Array1<f64>,
    /// How many pieces of the conductor were summed with the near-field series for each point, shape = [n_point], [count]
    #[allow(dead_code, reason = "how each point was calculated is only read by the tests")]
    n_piece_near_field: Array1<usize>,
    /// How many pieces of the conductor were summed with the far-field series for each point, shape = [n_point], [count]
    #[allow(dead_code, reason = "how each point was calculated is only read by the tests")]
    n_piece_far_field: Array1<usize>,
}

/// One of the pieces which a conductor is cut into: a parallelogram with the conductor's two inclinations. Its shape is
/// `subdivisions_shape[i_subdivision]` in `greens_psi_parallelogram`.
struct Piece {
    /// R coordinate of the piece's reference corner, [metre]
    oblique_r: f64,
    /// Z coordinate of the piece's reference corner, [metre]
    oblique_z: f64,
    /// How many times the conductor was halved to make the piece
    i_subdivision: usize,
    /// The fraction of the conductor's current which the piece carries, [dimensionless]
    current_fraction: f64,
}

/// The flux at points from one parallelogram conductor which carries 1 ampere, uniformly distributed, with the series chosen for each point.
///
/// This is section 5 of the note. With `half_diagonal` the distance from the conductor's centre to its furthest corner, `distance` the
/// distance from the point to its centre, and `centre_r` the major radius of its centre:
/// * the far-field series is used when its expansion parameter, `half_diagonal / min(distance, centre_r)`, Eq. (89), is at most
///   `far_field_ratio_max`;
/// * otherwise the near-field series is used when its expansion parameter, `t / (1 - t)` with `t = (distance + half_diagonal) / (2 * r)`,
///   Eq. (66) and (67), is at most `near_field_ratio_max`;
/// * otherwise the conductor is halved along its longer side, and each half is treated in the same way. This is exact, because the current
///   density is uniform: each half is a parallelogram which carries half of the current.
///
/// Higher orders of the series mean larger thresholds, and fewer pieces.
///
/// # Arguments
/// * `r` - R coordinate of the points, shape = [n_point], [metre]
/// * `z` - Z coordinate of the points, shape = [n_point], [metre]
/// * `oblique_r_prime` - R coordinate of the conductor's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the conductor's reference corner, [metre]
/// * `oblique_length_alpha_prime` - Length of the conductor's side which is inclined at `oblique_alpha_prime`, [metre]
/// * `oblique_length_beta_prime` - Length of the conductor's side which is inclined at `oblique_beta_prime`, [metre]
/// * `oblique_alpha_prime` - Inclination of the conductor's first side, counter-clockwise from the R axis, [radian]
/// * `oblique_beta_prime` - Inclination of the conductor's second side, counter-clockwise from the Z axis, [radian]
/// * `thresholds` - How small each series' expansion parameter has to be before the series is used, from `expansion_parameter_thresholds`
/// * `coefficients` - The near-field series' two tables of coefficients, from `maxwell_series_coefficients`
/// * `settings_greens_finite_size` - The orders of the two series, and how many times the conductor may be halved
///
/// The points and the conductor have already been checked by `mutual_inductance_finite_size_to_point_source`.
///
/// # Returns
/// * `greens_psi` - The flux at each point per ampere in the conductor, `g_psi` [henry], and how many pieces of the conductor were summed
///   with each series, `n_piece_near_field` and `n_piece_far_field` [count], all of shape = [n_point]
///
/// # Errors
/// * neither series converges fast enough for a piece of the conductor which has already been halved `n_subdivision_max` times
#[allow(clippy::too_many_arguments)]
fn greens_psi_parallelogram(
    r: &Array1<f64>,
    z: &Array1<f64>,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    oblique_length_alpha_prime: f64,
    oblique_length_beta_prime: f64,
    oblique_alpha_prime: f64,
    oblique_beta_prime: f64,
    thresholds: &ExpansionParameterThresholds,
    coefficients: &MaxwellSeriesCoefficients,
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<GreensPsiParallelogram, String> {
    let n_point: usize = r.len();
    let n_multipole: usize = settings_greens_finite_size.n_multipole;
    let n_subdivision_max: usize = settings_greens_finite_size.n_subdivision_max;
    // The highest degree, `a + b`, of the near-field series' terms
    let degree_max: usize = 2 * settings_greens_finite_size.n_elliptic + settings_greens_finite_size.n_thin_ring;

    // The directions of the two sides which meet at the reference corner
    let side_alpha_unit_r: f64 = oblique_alpha_prime.cos();
    let side_alpha_unit_z: f64 = oblique_alpha_prime.sin();
    let side_beta_unit_r: f64 = -oblique_beta_prime.sin();
    let side_beta_unit_z: f64 = oblique_beta_prime.cos();

    // The shape of the pieces after each number of halvings: `subdivisions_shape[0]` is the whole conductor. Every piece which was made by
    // the same number of halvings has the same shape, because it is always the longer side which is halved, so a shape is built only once,
    // the first time that a piece of it is needed
    let mut subdivisions_shape: Vec<ParallelogramShape> = vec![ParallelogramShape::new(
        oblique_length_alpha_prime,
        oblique_length_beta_prime,
        side_alpha_unit_r,
        side_alpha_unit_z,
        side_beta_unit_r,
        side_beta_unit_z,
        degree_max,
        n_multipole,
    )];

    // A conductor which is halved `n_subdivision_max` times is cut into at most `2 ** n_subdivision_max` pieces, and fewer than twice as many
    // pieces are looked at on the way to them. The count stops at the largest integer when `n_subdivision_max` is large
    let n_piece_max: usize = 2_usize.saturating_pow(u32::try_from(n_subdivision_max).unwrap_or(u32::MAX).saturating_add(1));

    // The work arrays of the two series, and the list of the pieces, are allocated once and used again for every point
    let mut near_field_work_arrays: NearFieldWorkArrays = NearFieldWorkArrays::new(degree_max);
    let mut far_field_work_arrays: FarFieldWorkArrays = FarFieldWorkArrays::new(n_multipole);
    let mut pieces: Vec<Piece> = Vec::new();

    let mut g_psi: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
    let mut n_piece_near_field: Array1<usize> = Array1::zeros(n_point);
    let mut n_piece_far_field: Array1<usize> = Array1::zeros(n_point);

    for i_point in 0..n_point {
        // The pieces which are still to be summed for this point, starting with the whole conductor. The last piece is taken from the list,
        // and it is either summed with one of the two series, or replaced with its two halves
        pieces.clear();
        pieces.push(Piece {
            oblique_r: oblique_r_prime,
            oblique_z: oblique_z_prime,
            i_subdivision: 0,
            current_fraction: 1.0,
        });

        let mut g_psi_sum: f64 = 0.0;
        'piece_loop: for _i_piece in 0..n_piece_max {
            let Some(piece): Option<Piece> = pieces.pop() else {
                break 'piece_loop;
            };
            let shape: &ParallelogramShape = &subdivisions_shape[piece.i_subdivision];

            // The piece's centre
            let half_alpha_r: f64 = 0.5 * shape.side_alpha_r;
            let half_alpha_z: f64 = 0.5 * shape.side_alpha_z;
            let half_beta_r: f64 = 0.5 * shape.side_beta_r;
            let half_beta_z: f64 = 0.5 * shape.side_beta_z;
            let centre_r: f64 = piece.oblique_r + (half_alpha_r + half_beta_r);
            let centre_z: f64 = piece.oblique_z + (half_alpha_z + half_beta_z);

            // The far-field series' expansion parameter, Eq. (89): the piece's size over its distance from the point, or over its own major
            // radius if that is smaller
            let distance: f64 = (r[i_point] - centre_r).hypot(z[i_point] - centre_z);
            let is_far_field: bool = shape.half_diagonal <= thresholds.far_field_ratio_max * distance.min(centre_r);

            // The near-field series' expansion parameter, Eq. (66) and (67): `t / (1 - t)`, where `t` is the distance from the point to the
            // furthest point of the piece over the distance from the point to its own mirror image in the axis, `2 * r`
            let t: f64 = (distance + shape.half_diagonal) / (2.0 * r[i_point]);
            let is_near_field: bool = t <= thresholds.near_field_ratio_max * (1.0 - t);

            if is_far_field {
                g_psi_sum += piece.current_fraction
                    * greens_psi_parallelogram_far_field(
                        r[i_point],
                        z[i_point],
                        centre_r,
                        centre_z,
                        &shape.moment,
                        n_multipole,
                        &mut far_field_work_arrays,
                    );
                n_piece_far_field[i_point] += 1;
            } else if is_near_field {
                g_psi_sum += piece.current_fraction
                    * greens_psi_parallelogram_near_field(
                        r[i_point],
                        z[i_point],
                        piece.oblique_r,
                        piece.oblique_z,
                        shape,
                        coefficients,
                        &mut near_field_work_arrays,
                    );
                n_piece_near_field[i_point] += 1;
            } else {
                if piece.i_subdivision == n_subdivision_max {
                    return Err(format!(
                        "point {i_point}, at r = {} and z = {} metre: neither series converges to `relative_tolerance` = {} for a piece of the \
                         conductor which has already been halved {n_subdivision_max} time(s); increase `n_subdivision_max`, or the orders \
                         `n_elliptic`, `n_thin_ring` and `n_multipole`",
                        r[i_point], z[i_point], settings_greens_finite_size.relative_tolerance
                    ));
                }

                // Halve the piece along its longer side; each half carries half of its current
                let is_alpha_longer: bool = shape.length_alpha >= shape.length_beta;
                let second_half_oblique_r: f64 = if is_alpha_longer {
                    piece.oblique_r + half_alpha_r
                } else {
                    piece.oblique_r + half_beta_r
                };
                let second_half_oblique_z: f64 = if is_alpha_longer {
                    piece.oblique_z + half_alpha_z
                } else {
                    piece.oblique_z + half_beta_z
                };
                if piece.i_subdivision + 1 == subdivisions_shape.len() {
                    // This is the first piece to be halved this many times, so the shape of its halves has not been built yet
                    let halves_length_alpha: f64 = if is_alpha_longer { 0.5 * shape.length_alpha } else { shape.length_alpha };
                    let halves_length_beta: f64 = if is_alpha_longer { shape.length_beta } else { 0.5 * shape.length_beta };
                    subdivisions_shape.push(ParallelogramShape::new(
                        halves_length_alpha,
                        halves_length_beta,
                        side_alpha_unit_r,
                        side_alpha_unit_z,
                        side_beta_unit_r,
                        side_beta_unit_z,
                        degree_max,
                        n_multipole,
                    ));
                }
                pieces.push(Piece {
                    oblique_r: second_half_oblique_r,
                    oblique_z: second_half_oblique_z,
                    i_subdivision: piece.i_subdivision + 1,
                    current_fraction: 0.5 * piece.current_fraction,
                });
                pieces.push(Piece {
                    oblique_r: piece.oblique_r,
                    oblique_z: piece.oblique_z,
                    i_subdivision: piece.i_subdivision + 1,
                    current_fraction: 0.5 * piece.current_fraction,
                });
            }
        }

        g_psi[i_point] = g_psi_sum;
    }

    Ok(GreensPsiParallelogram {
        g_psi,
        n_piece_near_field,
        n_piece_far_field,
    })
}

/// The shape of a parallelogram: everything which the two series need to know about a piece of a conductor, apart from where it is and
/// where the point is. It is built once, and used for every point and for every piece of the same shape.
struct ParallelogramShape {
    /// Length of the side which is inclined at `oblique_alpha`, [metre]
    length_alpha: f64,
    /// Length of the side which is inclined at `oblique_beta`, [metre]
    length_beta: f64,
    /// R component of the side which is inclined at `oblique_alpha`, `L_(alpha R)` in the note, [metre]
    side_alpha_r: f64,
    /// Z component of the side which is inclined at `oblique_alpha`, `L_(alpha Z)` in the note, [metre]
    side_alpha_z: f64,
    /// R component of the side which is inclined at `oblique_beta`, `L_(beta R)` in the note, [metre]
    side_beta_r: f64,
    /// Z component of the side which is inclined at `oblique_beta`, `L_(beta Z)` in the note, [metre]
    side_beta_z: f64,
    /// The distance from the centre to the furthest corner, `H` in the note, [metre]
    half_diagonal: f64,
    /// The area, `A` in the note, [metre ** 2]
    area: f64,
    /// R coordinate of the start of each of the four edges, from the reference corner, going around the parallelogram counter-clockwise, [metre]
    edges_start_r: [f64; N_EDGE],
    /// Z coordinate of the start of each of the four edges, from the reference corner, [metre]
    edges_start_z: [f64; N_EDGE],
    /// The length of each edge, [metre]
    edges_length: [f64; N_EDGE],
    /// The direction of each edge, `exp(i * phi_e)` in the note, [dimensionless]
    edges_phase: [c64; N_EDGE],
    /// The powers of each edge's direction, `edges_phase_power[i_edge][i_power] = exp(i * i_power * phi_e)`: one vector for each edge,
    /// shape = [N_EDGE][degree_max + 1], [dimensionless]
    edges_phase_power: Vec<Vec<c64>>,
    /// The moments about the centre, `<delta_R^i delta_Z^j>` in the note, from `parallelogram_moments`, shape = [n_multipole + 1, n_multipole + 1],
    /// [metre ** (i + j)]
    moment: Array2<f64>,
}

impl ParallelogramShape {
    /// The shape of a parallelogram, from the lengths and the directions of the two sides which meet at its reference corner.
    ///
    /// # Arguments
    /// * `length_alpha` - Length of the side which is inclined at `oblique_alpha`, [metre]
    /// * `length_beta` - Length of the side which is inclined at `oblique_beta`, [metre]
    /// * `side_alpha_unit_r` - R component of the direction of the first side, `cos(oblique_alpha)`, [dimensionless]
    /// * `side_alpha_unit_z` - Z component of the direction of the first side, `sin(oblique_alpha)`, [dimensionless]
    /// * `side_beta_unit_r` - R component of the direction of the second side, `-sin(oblique_beta)`, [dimensionless]
    /// * `side_beta_unit_z` - Z component of the direction of the second side, `cos(oblique_beta)`, [dimensionless]
    /// * `degree_max` - The highest degree of the near-field series' terms, `2 * n_elliptic + n_thin_ring`
    /// * `n_multipole` - The highest order of the far-field series
    ///
    /// # Returns
    /// * `shape` - The parallelogram's sides, half-diagonal, area, edges and moments
    #[allow(clippy::too_many_arguments)]
    fn new(
        length_alpha: f64,
        length_beta: f64,
        side_alpha_unit_r: f64,
        side_alpha_unit_z: f64,
        side_beta_unit_r: f64,
        side_beta_unit_z: f64,
        degree_max: usize,
        n_multipole: usize,
    ) -> Self {
        // The two sides which meet at the reference corner, Eq. (7) and (8) of the note
        let side_alpha_r: f64 = length_alpha * side_alpha_unit_r;
        let side_alpha_z: f64 = length_alpha * side_alpha_unit_z;
        let side_beta_r: f64 = length_beta * side_beta_unit_r;
        let side_beta_z: f64 = length_beta * side_beta_unit_z;
        let area_signed: f64 = side_alpha_r * side_beta_z - side_alpha_z * side_beta_r;

        // The longer of the two half-diagonals
        let half_diagonal: f64 = (0.5 * side_alpha_r + 0.5 * side_beta_r)
            .hypot(0.5 * side_alpha_z + 0.5 * side_beta_z)
            .max((0.5 * side_alpha_r - 0.5 * side_beta_r).hypot(0.5 * side_alpha_z - 0.5 * side_beta_z));

        // Going along the first side and then the second has to be counter-clockwise, so the two sides are exchanged if it is not
        let is_counter_clockwise: bool = area_signed > 0.0;
        let first_r: f64 = if is_counter_clockwise { side_alpha_r } else { side_beta_r };
        let first_z: f64 = if is_counter_clockwise { side_alpha_z } else { side_beta_z };
        let second_r: f64 = if is_counter_clockwise { side_beta_r } else { side_alpha_r };
        let second_z: f64 = if is_counter_clockwise { side_beta_z } else { side_alpha_z };

        // The four edges, counter-clockwise, each as the position of its start from the reference corner and its vector. They are built from
        // the two sides themselves, not from differences of the corners' positions, which would be rounded (section 8.5 of the note)
        let edges_start_r: [f64; N_EDGE] = [0.0, first_r, first_r + second_r, second_r];
        let edges_start_z: [f64; N_EDGE] = [0.0, first_z, first_z + second_z, second_z];
        let edges_r: [f64; N_EDGE] = [first_r, second_r, -first_r, -second_r];
        let edges_z: [f64; N_EDGE] = [first_z, second_z, -first_z, -second_z];

        // Each edge's length, its direction, and the powers of its direction
        let mut edges_length: [f64; N_EDGE] = [f64::NAN; N_EDGE];
        let mut edges_phase: [c64; N_EDGE] = [c64::new(f64::NAN, f64::NAN); N_EDGE];
        let mut edges_phase_power: Vec<Vec<c64>> = vec![vec![c64::new(f64::NAN, f64::NAN); degree_max + 1]; N_EDGE];
        for i_edge in 0..N_EDGE {
            edges_length[i_edge] = edges_r[i_edge].hypot(edges_z[i_edge]);
            edges_phase[i_edge] = c64::new(edges_r[i_edge], edges_z[i_edge]) / edges_length[i_edge];
            for i_power in 0..degree_max + 1 {
                edges_phase_power[i_edge][i_power] = edges_phase[i_edge].powi(i_power as i32);
            }
        }

        let moment: Array2<f64> = parallelogram_moments(side_alpha_r, side_alpha_z, side_beta_r, side_beta_z, n_multipole);

        ParallelogramShape {
            length_alpha,
            length_beta,
            side_alpha_r,
            side_alpha_z,
            side_beta_r,
            side_beta_z,
            half_diagonal,
            area: area_signed.abs(),
            edges_start_r,
            edges_start_z,
            edges_length,
            edges_phase,
            edges_phase_power,
            moment,
        }
    }
}

/// The work arrays of the near-field series. They are allocated once, and overwritten for every edge of every piece.
///
/// They are vectors, which are indexed in the innermost loops, where a vector is quicker to index than an array of one dimension.
struct NearFieldWorkArrays {
    /// The powers of `sigma` at the start of an edge, shape = [degree_max + 2], [dimensionless]
    sigma_start_power: Vec<c64>,
    /// The powers of `sigma` at the end of an edge, shape = [degree_max + 2], [dimensionless]
    sigma_end_power: Vec<c64>,
    /// `difference[k] = (sigma_end ** (k + 1) - sigma_start ** (k + 1)) / length`, shape = [degree_max + 1], [dimensionless]
    difference: Vec<c64>,
}

impl NearFieldWorkArrays {
    /// The work arrays of the near-field series, for terms up to the degree `degree_max`.
    fn new(degree_max: usize) -> Self {
        NearFieldWorkArrays {
            sigma_start_power: vec![c64::new(f64::NAN, f64::NAN); degree_max + 2],
            sigma_end_power: vec![c64::new(f64::NAN, f64::NAN); degree_max + 2],
            difference: vec![c64::new(f64::NAN, f64::NAN); degree_max + 1],
        }
    }
}

/// The near-field series: the flux at a point from a parallelogram which carries 1 ampere, uniformly distributed, for a point inside the
/// parallelogram, on one of its edges, at one of its corners, or outside it but near it.
///
/// This is Eq. (56) of the note (section 3.6). The Green's function of a filament is expanded about the point, in powers of
/// `zeta = ((r' - r) + i * (z' - z)) / (2 * r)`, the position of the source `(r', z')` from the point `(r, z)`, with the tables `table_p` and
/// `table_q` of `maxwell_series_coefficients`. The integral of each term over the parallelogram is exact, a sum over the four edges:
///
/// ```text
/// g_psi = (MU_0 * r / area) * sum over the edges of  edge_distance * sum over i_a >= i_b of  (multiplicity / (i_a + i_b + 2))
///         * Re{ phase ** (i_a - i_b) * [ (table_p * (ln(4) + 1 / (i_a + i_b + 2)) + table_q) * integral_plain - table_p * integral_ln ] }
/// ```
///
/// with every length in units of `2 * r`. `integral_plain` and `integral_ln` are the integrals along an edge of `sigma ** i_a *
/// conj(sigma) ** i_b`, and of the same with the factor `ln|sigma|`, where `sigma = zeta / phase` is `zeta` rotated so that the edge
/// lies along the real axis. They follow from recurrences, and need one logarithm for each corner and one angle for each edge.
///
/// The series converges when the parallelogram and the point are close together compared with the point's major radius: with
/// `t = (distance from the point to the parallelogram's centre + half_diagonal) / (2 * r)` it converges for `t < 1/2`, and with
/// `n_thin_ring = 2 * n_elliptic` its error falls like `(t / (1 - t)) ** (n_thin_ring + 1)` (section 3.7 of the note).
///
/// # Arguments
/// * `r` - R coordinate of the point, [metre]
/// * `z` - Z coordinate of the point, [metre]
/// * `oblique_r_prime` - R coordinate of the parallelogram's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the parallelogram's reference corner, [metre]
/// * `shape` - The parallelogram's shape: its edges and its area
/// * `coefficients` - The two tables of coefficients, from `maxwell_series_coefficients`, whose orders are the orders of the series
/// * `work_arrays` - The work arrays, which are overwritten
///
/// # Returns
/// * `g_psi` - The flux at the point per ampere in the parallelogram, [henry]
fn greens_psi_parallelogram_near_field(
    r: f64,
    z: f64,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    shape: &ParallelogramShape,
    coefficients: &MaxwellSeriesCoefficients,
    work_arrays: &mut NearFieldWorkArrays,
) -> f64 {
    let table_p: &Array2<f64> = &coefficients.table_p;
    let table_q: &Array2<f64> = &coefficients.table_q;
    let degree_max: usize = table_p.nrows() - 1;
    let ln_4: f64 = 4.0_f64.ln();

    // The sum over the edges, of the edge's distance from the point times its integrals. Every length is in units of `2 * r`
    let two_r: f64 = 2.0 * r;
    let mut edges_sum: f64 = 0.0;
    for i_edge in 0..N_EDGE {
        // The edge's two ends from the point, rotated so that the edge lies along the real axis, Eq. (44): `sigma = s - i * d_e`, where `s`
        // runs along the edge from `s_start` to `s_end`, and `d_e` is the distance from the point to the edge's line, positive when the
        // point is on the inner side of the edge. The start is the reference corner from the point, plus the edge's start from the
        // reference corner
        let phase: c64 = shape.edges_phase[i_edge];
        let phase_power: &[c64] = &shape.edges_phase_power[i_edge];
        let length: f64 = shape.edges_length[i_edge] / two_r;
        let zeta_start: c64 = c64::new(
            (oblique_r_prime - r) + shape.edges_start_r[i_edge],
            (oblique_z_prime - z) + shape.edges_start_z[i_edge],
        ) / two_r;
        let zeta_start_rotated: c64 = zeta_start * phase.conj();
        let s_start: f64 = zeta_start_rotated.re;
        let s_end: f64 = s_start + length;
        let edge_distance: f64 = -zeta_start_rotated.im;
        let sigma_start: c64 = c64::new(s_start, -edge_distance);
        let sigma_end: c64 = c64::new(s_end, -edge_distance);

        // The edge's ends are told apart by which is further from the point, so that a logarithm is only ever taken of a ratio which is at
        // most 1: ln|sigma| at the far end, and ln(|sigma| at the near end / |sigma| at the far end). When the two distances are similar,
        // the ratio is formed without cancellation, from |sigma_end| ** 2 - |sigma_start| ** 2 = length * (s_start + s_end).
        // At a corner which is at the point the ratio is zero, but its logarithm is only ever multiplied by zero, so it is set to zero
        let start_abs_sq: f64 = s_start.powi(2) + edge_distance.powi(2);
        let end_abs_sq: f64 = s_end.powi(2) + edge_distance.powi(2);
        let is_end_far: bool = end_abs_sq >= start_abs_sq;
        let far_abs_sq: f64 = if is_end_far { end_abs_sq } else { start_abs_sq };
        let near_abs_sq: f64 = if is_end_far { start_abs_sq } else { end_abs_sq };
        let ln_far: f64 = 0.5 * far_abs_sq.ln();
        let ratio_sq: f64 = near_abs_sq / far_abs_sq;
        let ln_near_over_far: f64 = if ratio_sq == 0.0 {
            0.0
        } else if ratio_sq < 0.5 {
            0.5 * ratio_sq.ln()
        } else {
            0.5 * (-(length * (s_start + s_end)).abs() / far_abs_sq).ln_1p()
        };
        // With `[f]` for `f` at the end of the edge minus `f` at its start: `[f * ln|sigma|] = [f] * ln_far + sign_near * f_near * ln_near_over_far`
        let sign_near: f64 = if is_end_far { -1.0 } else { 1.0 };

        // The powers of `sigma` at the two ends, and `difference[k] = [sigma ** (k + 1)] / length`, which is formed as the sum of
        // `sigma_end ** j * sigma_start ** (k - j)`, without cancellation (section 8.5 of the note)
        work_arrays.sigma_start_power[0] = c64::new(1.0, 0.0);
        work_arrays.sigma_end_power[0] = c64::new(1.0, 0.0);
        work_arrays.difference[0] = c64::new(1.0, 0.0);
        for i_power in 1..degree_max + 2 {
            work_arrays.sigma_start_power[i_power] = work_arrays.sigma_start_power[i_power - 1] * sigma_start;
            work_arrays.sigma_end_power[i_power] = work_arrays.sigma_end_power[i_power - 1] * sigma_end;
            if i_power <= degree_max {
                work_arrays.difference[i_power] = sigma_end * work_arrays.difference[i_power - 1] + work_arrays.sigma_start_power[i_power];
            }
        }
        let sigma_start_power: &[c64] = &work_arrays.sigma_start_power;
        let sigma_end_power: &[c64] = &work_arrays.sigma_end_power;
        let difference: &[c64] = &work_arrays.difference;
        let sigma_near_power: &[c64] = if is_end_far { sigma_start_power } else { sigma_end_power };

        // I_(0, -1), Eq. (54): the integral of 1 / conj(sigma) along the edge, ln(d_end / d_start) - i * omega_e, where omega_e is the angle
        // which the edge subtends at the point, formed without cancellation
        let mut integral_inverse: c64 = c64::new(
            sign_near * ln_near_over_far,
            -(edge_distance * length).atan2(s_start * s_end + edge_distance.powi(2)),
        );

        let mut edge_sum: f64 = 0.0;
        for i_degree in 0..degree_max + 1 {
            // The terms with `i_a + i_b = i_degree`. They start at `i_b = 0`, where the integrals are elementary: I_(D, 0), Eq. (52),
            // I_(D + 1, -1), Eq. (53), and J_(D, 0), Eq. (51)
            let inverse_degree_plus_one: f64 = 1.0 / ((i_degree + 1) as f64);
            let mut integral_plain: c64 = difference[i_degree] * length * inverse_degree_plus_one;
            integral_inverse = integral_plain - c64::new(0.0, 2.0 * edge_distance) * integral_inverse;
            let mut integral_ln: c64 = (difference[i_degree] * length * ln_far + sigma_near_power[i_degree + 1] * sign_near * ln_near_over_far
                - (integral_plain + integral_inverse) * 0.5)
                * inverse_degree_plus_one;

            for i_b in 0..i_degree / 2 + 1 {
                let i_a: usize = i_degree - i_b;
                if i_b >= 1 {
                    // Step from (i_a + 1, i_b - 1) to (i_a, i_b), by integration by parts, Eq. (49) and (51).
                    // `boundary = [sigma ** (i_a + 1) * conj(sigma) ** i_b]`
                    let boundary: c64 = (difference[i_a] * sigma_end_power[i_b].conj() + sigma_start_power[i_a + 1] * difference[i_b - 1].conj()) * length;
                    let boundary_ln: c64 = boundary * ln_far + sigma_near_power[i_a + 1] * sign_near * sigma_near_power[i_b].conj() * ln_near_over_far;
                    let inverse_a_plus_one: f64 = 1.0 / ((i_a + 1) as f64);
                    let integral_plain_previous: c64 = integral_plain;
                    integral_plain = (boundary - integral_plain_previous * (i_b as f64)) * inverse_a_plus_one;
                    integral_ln = (boundary_ln - (integral_plain + integral_plain_previous) * 0.5 - integral_ln * (i_b as f64)) * inverse_a_plus_one;
                }

                // The term of Eq. (56), where the truncation of the two tables keeps it. The (i_b, i_a) term is its complex conjugate
                let coefficient_p: f64 = table_p[(i_a, i_b)];
                let coefficient_q: f64 = table_q[(i_a, i_b)];
                if coefficient_p != 0.0 || coefficient_q != 0.0 {
                    let multiplicity: f64 = if i_a == i_b { 1.0 } else { 2.0 };
                    let degree_plus_two: f64 = (i_degree + 2) as f64;
                    let term: c64 = phase_power[i_a - i_b]
                        * (integral_plain * (coefficient_p * (ln_4 + 1.0 / degree_plus_two) + coefficient_q) - integral_ln * coefficient_p);
                    edge_sum += (multiplicity / degree_plus_two) * term.re;
                }
            }
        }

        edges_sum += edge_distance * edge_sum;
    }

    // The area, in units of `2 * r` squared
    let area_scaled: f64 = shape.area / two_r.powi(2);

    MU_0 * r * edges_sum / area_scaled
}

/// The work arrays of the far-field series. They are allocated once, and overwritten for every piece.
///
/// The series are vectors, which are indexed in the innermost loops, where a vector is quicker to index than an array of one dimension.
struct FarFieldWorkArrays {
    /// The Taylor coefficients in `delta_r` of `Phi`, shape = [n_multipole + 1]
    series_phi: Vec<f64>,
    /// The Taylor coefficients in `delta_r` of `Theta`, shape = [n_multipole + 1]
    series_theta: Vec<f64>,
    /// The Taylor coefficients in `delta_r` of `u`, shape = [n_multipole + 1]
    series_u: Vec<f64>,
    /// The Taylor coefficients in `delta_r` of `1 / d ** 2`, shape = [n_multipole + 1]
    series_inverse_d_sq: Vec<f64>,
    /// The Taylor coefficients in `delta_r` of `Theta / d ** 2`, shape = [n_multipole + 1]
    series_theta_over_d_sq: Vec<f64>,
    /// The Taylor coefficients of the Green's function of a filament about the centre, `G_ij` in the note: `taylor[(i, j)]` multiplies
    /// `delta_r ** i * delta_z ** j`, and is used for `i + j <= n_multipole`, shape = [n_multipole + 1, n_multipole + 1]
    taylor: Array2<f64>,
}

impl FarFieldWorkArrays {
    /// The work arrays of the far-field series, for the orders up to `n_multipole`.
    fn new(n_multipole: usize) -> Self {
        let n_series: usize = n_multipole + 1;
        FarFieldWorkArrays {
            series_phi: vec![f64::NAN; n_series],
            series_theta: vec![f64::NAN; n_series],
            series_u: vec![f64::NAN; n_series],
            series_inverse_d_sq: vec![f64::NAN; n_series],
            series_theta_over_d_sq: vec![f64::NAN; n_series],
            taylor: Array2::from_elem((n_series, n_series), f64::NAN),
        }
    }
}

/// The far-field series: the flux at a point from a parallelogram which carries 1 ampere, uniformly distributed, for a point away from the
/// parallelogram.
///
/// This is Eq. (69) of the note (section 4): the multipole expansion about the parallelogram's centre. The Green's function of a filament
/// is expanded in a Taylor series in the position of the source from the centre, `(delta_r, delta_z)`, and averaged over the parallelogram:
///
/// ```text
/// g_psi = sum over i + j even, i + j <= n_multipole, of  taylor[(i, j)] * moment[(i, j)]
/// ```
///
/// The term `i = j = 0` is a point filament at the centre. Every Taylor coefficient comes from one pair of elliptic integrals, at the
/// centre, by recurrences (section 4.2 of the note).
///
/// The series converges when the point is further from the parallelogram's centre than any point of the parallelogram is, at the rate
/// `(half_diagonal / distance from the point to the centre) ** (n_multipole + 2)` (section 4.3 of the note). At the centre itself it is
/// infinite.
///
/// # Arguments
/// * `r` - R coordinate of the point, [metre]
/// * `z` - Z coordinate of the point, [metre]
/// * `centre_r` - R coordinate of the parallelogram's centre, [metre]
/// * `centre_z` - Z coordinate of the parallelogram's centre, [metre]
/// * `moment` - The parallelogram's moments about its centre, from `parallelogram_moments`, shape = [n_multipole + 1, n_multipole + 1],
///   [metre ** (i + j)]
/// * `n_multipole` - The highest order kept; 0 is a point filament, and an odd order adds nothing to the even order below it
/// * `work_arrays` - The work arrays, which are overwritten
///
/// # Returns
/// * `g_psi` - The flux at the point per ampere in the parallelogram, [henry]
fn greens_psi_parallelogram_far_field(
    r: f64,
    z: f64,
    centre_r: f64,
    centre_z: f64,
    moment: &Array2<f64>,
    n_multipole: usize,
    work_arrays: &mut FarFieldWorkArrays,
) -> f64 {
    greens_taylor_coefficients(r, z, centre_r, centre_z, n_multipole, work_arrays);
    let n_series: usize = n_multipole + 1;
    let taylor: &Array2<f64> = &work_arrays.taylor;

    // Sum the Taylor coefficients against the moments of the parallelogram, Eq. (69). The odd moments are zero
    let mut g_psi: f64 = 0.0;
    for i_series in 0..n_series {
        for j_series in 0..n_series - i_series {
            if (i_series + j_series) % 2 == 0 {
                g_psi += moment[(i_series, j_series)] * taylor[(i_series, j_series)];
            }
        }
    }

    g_psi
}

/// The Taylor coefficients of the Green's function of a filament about a centre, in the position of the source from the centre,
/// `(delta_r, delta_z)`: `G_ij` of the note, Eq. (69), for `i + j <= n_order`. They are written into `work_arrays.taylor`, where
/// `taylor[(i, j)]` multiplies `delta_r ** i * delta_z ** j`; the other entries are left as they were.
///
/// This is section 4.2 of the note. Every coefficient comes from one pair of elliptic integrals, at the centre: the first two columns from
/// the Taylor series along the horizontal line through the centre, and the others from the equation which the Green's function obeys in
/// the source's position.
///
/// # Arguments
/// * `r` - R coordinate of the point, [metre]
/// * `z` - Z coordinate of the point, [metre]
/// * `centre_r` - R coordinate of the centre, [metre]
/// * `centre_z` - Z coordinate of the centre, [metre]
/// * `n_order` - The highest order, `i + j`, of the coefficients; the work arrays must have been made for at least this order
/// * `work_arrays` - The work arrays, which are overwritten
fn greens_taylor_coefficients(r: f64, z: f64, centre_r: f64, centre_z: f64, n_order: usize, work_arrays: &mut FarFieldWorkArrays) {
    let n_series: usize = n_order + 1;
    let series_phi: &mut [f64] = &mut work_arrays.series_phi;
    let series_theta: &mut [f64] = &mut work_arrays.series_theta;
    let series_u: &mut [f64] = &mut work_arrays.series_u;
    let series_inverse_d_sq: &mut [f64] = &mut work_arrays.series_inverse_d_sq;
    let series_theta_over_d_sq: &mut [f64] = &mut work_arrays.series_theta_over_d_sq;
    let taylor: &mut Array2<f64> = &mut work_arrays.taylor;

    // Along the horizontal line through the centre, with the source at `r' = centre_r + delta_r`, these are quadratics in `delta_r`, lowest
    // power first, Eq. (82) to (84): u ** 2, d ** 2 and w ** 2 = r' ** 2 - r ** 2 - h ** 2
    const N_COEFFICIENT_QUADRATIC: usize = 3;
    let h: f64 = z - centre_z;
    let polynomial_u_sq: [f64; N_COEFFICIENT_QUADRATIC] = [(r + centre_r).powi(2) + h.powi(2), 2.0 * (r + centre_r), 1.0];
    let polynomial_d_sq: [f64; N_COEFFICIENT_QUADRATIC] = [(r - centre_r).powi(2) + h.powi(2), -2.0 * (r - centre_r), 1.0];
    let polynomial_w_sq: [f64; N_COEFFICIENT_QUADRATIC] = [centre_r.powi(2) - r.powi(2) - h.powi(2), 2.0 * centre_r, 1.0];

    // The polynomials which multiply the two derivatives in Eq. (79) and (80): 2 * r' * u ** 2 for `theta`, which is a cubic, and
    // 2 * r' * d ** 2 * u ** 2 for `phi`, which is a quintic
    const N_COEFFICIENT_THETA: usize = 4;
    const N_COEFFICIENT_PHI: usize = 6;
    let polynomial_theta: [f64; N_COEFFICIENT_THETA] = [
        2.0 * centre_r * polynomial_u_sq[0],
        2.0 * (polynomial_u_sq[0] + centre_r * polynomial_u_sq[1]),
        2.0 * (polynomial_u_sq[1] + centre_r),
        2.0,
    ];
    let mut polynomial_phi: [f64; N_COEFFICIENT_PHI] = [f64::NAN; N_COEFFICIENT_PHI];
    for i_power in 0..N_COEFFICIENT_PHI {
        let mut coefficient: f64 = 0.0;
        for j_power in 0..N_COEFFICIENT_QUADRATIC {
            if i_power >= j_power && i_power - j_power < N_COEFFICIENT_THETA {
                coefficient += polynomial_theta[i_power - j_power] * polynomial_d_sq[j_power];
            }
        }
        polynomial_phi[i_power] = coefficient;
    }

    // The elliptic integrals at the centre, as the pair `phi` and `theta`, Eq. (72) and (73)
    let phi_theta: EllipticPhiTheta = elliptic_phi_theta(4.0 * r * centre_r / polynomial_u_sq[0], polynomial_d_sq[0] / polynomial_u_sq[0]);
    series_phi[0] = phi_theta.phi;
    series_theta[0] = phi_theta.theta;

    // Their Taylor coefficients in `delta_r`, from the two differential equations with polynomial coefficients, Eq. (79) and (80):
    // 2 * r' * d ** 2 * u ** 2 * phi' = -w ** 2 * (d ** 2 * phi + u ** 2 * theta),  and  2 * r' * u ** 2 * theta' = -3 * w ** 2 * (phi + theta)
    for i_series in 0..n_series - 1 {
        let mut rhs_theta: f64 = 0.0;
        let mut rhs_phi: f64 = 0.0;
        for i_power in 0..N_COEFFICIENT_QUADRATIC {
            if i_series >= i_power {
                rhs_theta -= 3.0 * polynomial_w_sq[i_power] * (series_phi[i_series - i_power] + series_theta[i_series - i_power]);
                // The coefficient of `delta_r ** (i_series - i_power)` in `d ** 2 * phi + u ** 2 * theta`
                for j_power in 0..N_COEFFICIENT_QUADRATIC {
                    if i_series - i_power >= j_power {
                        rhs_phi -= polynomial_w_sq[i_power]
                            * (polynomial_d_sq[j_power] * series_phi[i_series - i_power - j_power]
                                + polynomial_u_sq[j_power] * series_theta[i_series - i_power - j_power]);
                    }
                }
            }
        }
        for i_power in 1..N_COEFFICIENT_THETA {
            if i_series >= i_power {
                rhs_theta -= polynomial_theta[i_power] * ((i_series - i_power + 1) as f64) * series_theta[i_series - i_power + 1];
            }
        }
        for i_power in 1..N_COEFFICIENT_PHI {
            if i_series >= i_power {
                rhs_phi -= polynomial_phi[i_power] * ((i_series - i_power + 1) as f64) * series_phi[i_series - i_power + 1];
            }
        }
        series_theta[i_series + 1] = rhs_theta / (polynomial_theta[0] * ((i_series + 1) as f64));
        series_phi[i_series + 1] = rhs_phi / (polynomial_phi[0] * ((i_series + 1) as f64));
    }

    // The Taylor coefficients in `delta_r` of `u`, from `u ** 2 * f' = f * (u ** 2)' / 2`, and of `1 / d ** 2`
    series_u[0] = polynomial_u_sq[0].sqrt();
    series_inverse_d_sq[0] = 1.0 / polynomial_d_sq[0];
    for i_series in 0..n_series - 1 {
        let mut value_u: f64 = (0.5 - (i_series as f64)) * polynomial_u_sq[1] * series_u[i_series];
        let mut value_inverse_d_sq: f64 = -polynomial_d_sq[1] * series_inverse_d_sq[i_series];
        if i_series >= 1 {
            value_u += (2.0 - (i_series as f64)) * series_u[i_series - 1];
            value_inverse_d_sq -= series_inverse_d_sq[i_series - 1];
        }
        series_u[i_series + 1] = value_u / (((i_series + 1) as f64) * polynomial_u_sq[0]);
        series_inverse_d_sq[i_series + 1] = value_inverse_d_sq / polynomial_d_sq[0];
    }

    // The first two columns of the Taylor coefficients are the Green's function and its vertical derivative along the horizontal line,
    // Eq. (85) and (86), which are products of series: G = MU_0 * u * phi / 2,  and  d(G)/d(z') = MU_0 * h * u * theta / (2 * d ** 2)
    for i_series in 0..n_series {
        let mut theta_over_d_sq: f64 = 0.0;
        for j_series in 0..i_series + 1 {
            theta_over_d_sq += series_inverse_d_sq[j_series] * series_theta[i_series - j_series];
        }
        series_theta_over_d_sq[i_series] = theta_over_d_sq;
    }
    for i_series in 0..n_series {
        let mut taylor_0: f64 = 0.0;
        for j_series in 0..i_series + 1 {
            taylor_0 += 0.5 * MU_0 * series_u[j_series] * series_phi[i_series - j_series];
        }
        taylor[(i_series, 0)] = taylor_0;
        if i_series < n_order {
            let mut taylor_1: f64 = 0.0;
            for j_series in 0..i_series + 1 {
                taylor_1 += 0.5 * MU_0 * h * series_u[j_series] * series_theta_over_d_sq[i_series - j_series];
            }
            taylor[(i_series, 1)] = taylor_1;
        }
    }

    // The other columns, Eq. (88), from the equation which the Green's function obeys in the source's position:
    // r' * (d2(G)/d(r'2) + d2(G)/d(z'2)) - d(G)/d(r') = 0
    for j_series in 0..n_series.saturating_sub(2) {
        for i_series in 0..n_order - j_series - 1 {
            let mut value: f64 = centre_r * ((i_series + 2) as f64) * ((i_series + 1) as f64) * taylor[(i_series + 2, j_series)]
                + ((i_series + 1) as f64) * ((i_series as f64) - 1.0) * taylor[(i_series + 1, j_series)];
            if i_series >= 1 {
                value += ((j_series + 2) as f64) * ((j_series + 1) as f64) * taylor[(i_series - 1, j_series + 2)];
            }
            taylor[(i_series, j_series + 2)] = -value / (centre_r * ((j_series + 2) as f64) * ((j_series + 1) as f64));
        }
    }
}

/// `ConductorModel::FiniteSize` at points: the flux, and its derivatives, from conductors of finite size.
///
/// `psi` alone is calculated on its own. The derivatives are calculated together, all eight with the flux, the first time one of them is
/// asked for, and kept, so asking for the others does not calculate them again.
pub(super) struct FiniteSizeToPoint {
    /// The points, after the safety shift of `Greens::sensor_to_conductor`, which moves any point which is on an edge or at a corner of a
    /// conductor by 1 nm
    pub(super) sensor: Sensor,
    /// The conductors
    pub(super) conductor: Conductor,
    /// The orders of the series and their accuracy
    pub(super) settings_greens_finite_size: SettingsEquilibriumNumericsGreensFiniteSize,
    /// All nine tables, calculated the first time a derivative is asked for
    pub(super) tables: OnceLock<Result<FiniteSizeToPointTables, String>>,
}

impl FiniteSizeToPoint {
    /// The flux per ampere, on its own.
    ///
    /// # Returns
    /// * `g_psi` - shape = [n_sensor, n_conductor], [henry]
    ///
    /// # Errors
    /// * as `mutual_inductance_finite_size_to_point_source`
    pub(super) fn psi(&self) -> Result<Array2<f64>, String> {
        mutual_inductance_finite_size_to_point_source(
            &self.sensor.r,
            &self.sensor.z,
            &self.conductor.r,
            &self.conductor.z,
            &self.conductor.length_alpha,
            &self.conductor.length_beta,
            &self.conductor.alpha,
            &self.conductor.beta,
            &self.settings_greens_finite_size,
        )
    }

    /// All nine tables, calculated once.
    ///
    /// # Returns
    /// * `tables` - the flux and its eight derivatives, each of shape = [n_sensor, n_conductor]
    ///
    /// # Errors
    /// * as `mutual_inductance_finite_size_to_point_source_with_derivatives`, every time it is asked for
    pub(super) fn tables(&self) -> Result<&FiniteSizeToPointTables, String> {
        let tables: &Result<FiniteSizeToPointTables, String> = self.tables.get_or_init(|| {
            mutual_inductance_finite_size_to_point_source_with_derivatives(
                &self.sensor.r,
                &self.sensor.z,
                &self.conductor.r,
                &self.conductor.z,
                &self.conductor.length_alpha,
                &self.conductor.length_beta,
                &self.conductor.alpha,
                &self.conductor.beta,
                &self.settings_greens_finite_size,
            )
        });

        tables.as_ref().map_err(|error: &String| error.clone())
    }
}

/// All nine Green's tables of conductors of finite size at points: the flux per ampere in each conductor, and its derivatives with respect
/// to the point's position up to the third. Each table has shape = [n_point, n_conductor].
pub struct FiniteSizeToPointTables {
    /// The flux, [henry]
    pub psi: Array2<f64>,
    /// `d(psi)/d(r)`, [henry / metre]
    pub d_psi_d_r: Array2<f64>,
    /// `d(psi)/d(z)`, [henry / metre]
    pub d_psi_d_z: Array2<f64>,
    /// `d2(psi)/d(r)2`, [henry / metre ** 2]
    pub d2_psi_d_r2: Array2<f64>,
    /// `d2(psi)/d(r)d(z)`, [henry / metre ** 2]
    pub d2_psi_d_r_d_z: Array2<f64>,
    /// `d2(psi)/d(z)2`, [henry / metre ** 2]
    pub d2_psi_d_z2: Array2<f64>,
    /// `d3(psi)/d(r)2d(z)`, [henry / metre ** 3]
    pub d3_psi_d_r2_d_z: Array2<f64>,
    /// `d3(psi)/d(r)d(z)2`, [henry / metre ** 3]
    pub d3_psi_d_r_d_z2: Array2<f64>,
    /// `d3(psi)/d(z)3`, [henry / metre ** 3]
    pub d3_psi_d_z3: Array2<f64>,
}

/// The poloidal flux at points from toroidal conductors of finite size, per ampere in each conductor, and its derivatives with respect to
/// the point's position, up to the third: the nine Green's tables which GSFit's grid and sensors need.
///
/// The conductors, the points, the checks and the two series are those of `mutual_inductance_finite_size_to_point_source`, and so are the
/// pieces a conductor is cut into, except for a point inside a piece which has to be cut (see below). The thresholds of the series are
/// those of `expansion_parameter_thresholds_derivatives`, so that the third derivatives are within the relative tolerance too, and the
/// far-field series is used with the lowest even order which is enough for each piece and point. The derivatives are exact derivatives of
/// the two series:
/// * the far-field series: the Taylor coefficients of the Green's function in the source's position, `G_ij`, give the derivatives in the
///   point's position too. The Green's function depends on `z - z'` only, so `d/d(z)` is `-d/d(z')`; and it is homogeneous of degree 1 in
///   `(r, z, r', z')` together, so Euler's theorem, `r * d(G)/d(r) + r' * d(G)/d(r') + (z - z') * d(G)/d(z) = G`, gives `d/d(r)` from
///   the derivatives in the source's position. Each derivative is a recurrence on the table of the one before, and the table of `G_ij` is
///   calculated three orders further, to `n_multipole + 3`, so that every derivative is summed to the order `n_multipole`, as the flux is.
/// * the near-field series: the integral over the parallelogram is a closed form, Eq. (56) of the note, which is an exact function of the
///   point's position. It is evaluated with every quantity as its Taylor polynomial in the point's position, to the third degree, which
///   gives its derivatives exactly; inside the conductor too, where the derivatives of the closed form are finite.
///
/// A piece which contains the point, and for which neither series converges fast enough, is not halved: halving could put the point on an
/// edge, or at a corner, of the halves, where the derivatives of each half jump or are infinite, and only their sum is not. Instead it is
/// cut into a piece of half its size centred on the point, the ring around that, and strips: the point is always at the centre of the piece
/// which contains it, and away from the edges of all the others.
///
/// # Arguments
/// * `r` - R coordinate of the points, shape = [n_point], [metre]
/// * `z` - Z coordinate of the points, shape = [n_point], [metre]
/// * `oblique_r_prime` - R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_z_prime` - Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
/// * `oblique_length_alpha_prime` - Length of each conductor's side which is inclined at `oblique_alpha_prime`, shape = [n_conductor], [metre]
/// * `oblique_length_beta_prime` - Length of each conductor's side which is inclined at `oblique_beta_prime`, shape = [n_conductor], [metre]
/// * `oblique_alpha_prime` - Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
/// * `oblique_beta_prime` - Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
/// * `settings_greens_finite_size` - The orders of the two series, the relative accuracy wanted, and how many times a conductor may
///   be halved: `equilibrium_settings/numerics/greens/finite_size`
///
/// # Returns
/// * `tables` - The flux at each point per ampere in each conductor [henry], and its eight derivatives [henry / metre ** n], each of shape =
///   [n_point, n_conductor]
///
/// # Errors
/// * `r` and `z`, or the six descriptions of the conductors, do not have the same length
/// * a point is not at a positive major radius, `r <= 0`
/// * a conductor has no area: a side without a positive length, or two sides which are parallel
/// * a conductor is not wholly at a positive major radius
/// * the relative tolerance is not greater than 0 and at most 1
/// * a point is on an edge of a conductor, where the second derivatives jump, or at one of its corners, where they are infinite
/// * neither series converges fast enough for a piece of a conductor which has already been cut `n_subdivision_max` times
#[allow(clippy::too_many_arguments)]
pub fn mutual_inductance_finite_size_to_point_source_with_derivatives(
    r: &Array1<f64>,
    z: &Array1<f64>,
    oblique_r_prime: &Array1<f64>,
    oblique_z_prime: &Array1<f64>,
    oblique_length_alpha_prime: &Array1<f64>,
    oblique_length_beta_prime: &Array1<f64>,
    oblique_alpha_prime: &Array1<f64>,
    oblique_beta_prime: &Array1<f64>,
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<FiniteSizeToPointTables, String> {
    let n_point: usize = r.len();
    if z.len() != n_point {
        return Err(format!("`r` and `z` have {} and {} point(s); each point needs one of each", n_point, z.len()));
    }

    // The Green's function of a filament needs the points, and the whole of every conductor, at a positive major radius
    for i_point in 0..n_point {
        let is_usable_point: bool = r[i_point] > 0.0 && r[i_point].is_finite() && z[i_point].is_finite();
        if !is_usable_point {
            return Err(format!(
                "point {i_point} is at r = {} and z = {} metre; r must be positive, and both finite",
                r[i_point], z[i_point]
            ));
        }
    }
    check_conductors(
        oblique_r_prime,
        oblique_z_prime,
        oblique_length_alpha_prime,
        oblique_length_beta_prime,
        oblique_alpha_prime,
        oblique_beta_prime,
        "_prime",
    )?;
    let n_conductor: usize = oblique_r_prime.len();

    // What every point and every conductor share: how small each series' expansion parameter has to be, and the near-field series' two
    // tables of coefficients
    let thresholds: ExpansionParameterThresholdsDerivatives = expansion_parameter_thresholds_derivatives(settings_greens_finite_size)?;
    let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring);

    // Each conductor is one task, which calculates the flux and its derivatives from that conductor at every point: one whole column of
    // each table. The tasks are large, and independent of each other
    let conductors_derivatives: Vec<GreensDerivativesParallelogram> = (0..n_conductor)
        .into_par_iter()
        .map(|i_conductor: usize| -> Result<GreensDerivativesParallelogram, String> {
            greens_derivatives_parallelogram(
                r,
                z,
                oblique_r_prime[i_conductor],
                oblique_z_prime[i_conductor],
                oblique_length_alpha_prime[i_conductor],
                oblique_length_beta_prime[i_conductor],
                oblique_alpha_prime[i_conductor],
                oblique_beta_prime[i_conductor],
                &thresholds,
                &coefficients,
                settings_greens_finite_size,
            )
            .map_err(|error: String| format!("conductor {i_conductor}: {error}"))
        })
        .collect::<Result<Vec<GreensDerivativesParallelogram>, String>>()?;

    let mut tables: FiniteSizeToPointTables = FiniteSizeToPointTables {
        psi: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d_psi_d_r: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d_psi_d_z: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d2_psi_d_r2: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d2_psi_d_r_d_z: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d2_psi_d_z2: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d3_psi_d_r2_d_z: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d3_psi_d_r_d_z2: Array2::from_elem((n_point, n_conductor), f64::NAN),
        d3_psi_d_z3: Array2::from_elem((n_point, n_conductor), f64::NAN),
    };
    for i_conductor in 0..n_conductor {
        let conductor_derivatives: &GreensDerivativesParallelogram = &conductors_derivatives[i_conductor];
        tables.psi.column_mut(i_conductor).assign(&conductor_derivatives.psi);
        tables.d_psi_d_r.column_mut(i_conductor).assign(&conductor_derivatives.d_psi_d_r);
        tables.d_psi_d_z.column_mut(i_conductor).assign(&conductor_derivatives.d_psi_d_z);
        tables.d2_psi_d_r2.column_mut(i_conductor).assign(&conductor_derivatives.d2_psi_d_r2);
        tables.d2_psi_d_r_d_z.column_mut(i_conductor).assign(&conductor_derivatives.d2_psi_d_r_d_z);
        tables.d2_psi_d_z2.column_mut(i_conductor).assign(&conductor_derivatives.d2_psi_d_z2);
        tables.d3_psi_d_r2_d_z.column_mut(i_conductor).assign(&conductor_derivatives.d3_psi_d_r2_d_z);
        tables.d3_psi_d_r_d_z2.column_mut(i_conductor).assign(&conductor_derivatives.d3_psi_d_r_d_z2);
        tables.d3_psi_d_z3.column_mut(i_conductor).assign(&conductor_derivatives.d3_psi_d_z3);
    }

    Ok(tables)
}

/// The highest order of the derivatives of the flux which the Green's tables hold
const DERIVATIVE_ORDER_MAX: usize = 3;

/// A point whose position in a piece, as a fraction of each side, is closer than this to 0 or to 1, is taken to be on the piece's edge,
/// [dimensionless]
const POINT_ON_EDGE_FRACTION: f64 = 1.0e-10;

/// What `expansion_parameter_thresholds_derivatives` returns: how small each series' expansion parameter has to be before the series is
/// used for the flux and its derivatives.
struct ExpansionParameterThresholdsDerivatives {
    /// For each even order of the far-field series up to `n_multipole`, `0, 2, 4, ...`, the largest `half_diagonal / min(distance, centre_r)`
    /// at which that order is enough; the far-field series is used up to the last of them, [dimensionless]
    far_field_orders_ratio_max: Vec<f64>,
    /// The largest `t / (1 - t)`, with `t = (distance + half_diagonal) / (2 * r)`, at which the near-field series is used, [dimensionless]
    near_field_ratio_max: f64,
}

/// How small each series' expansion parameter has to be for the derivatives of the flux, up to the third, to be within the relative
/// tolerance.
///
/// The near-field series' threshold is that of `expansion_parameter_thresholds`, for the flux: its terms are powers of the distance from
/// the point over `2 * r`, so the derivatives of the terms which are left out grow by powers of the same small ratio, and the derivatives
/// are as accurate as the flux.
///
/// The far-field series' terms are powers of the size of the parallelogram over the distance from its centre, and the derivatives of the
/// terms which are left out grow by powers of their order instead. The first term which is left out by the order `n` (even) is of the order
/// `n + 2 = k`; its third derivative, relative to the third derivative of the flux, is larger than the term is relative to the flux by about
/// the binomial coefficient `(k + 3)! / (k! * 3!)`, so the tolerance is divided by that. The order `n` is then enough up to the ratio
/// `(relative_tolerance / ((k + 3)! / (k! * 3!))) ** (1 / k)`, which grows with `n`; a point far from a piece needs a lower order than one
/// near it.
///
/// # Arguments
/// * `settings_greens_finite_size` - The orders of the two series, and the relative accuracy wanted
///
/// # Returns
/// * `thresholds` - `far_field_orders_ratio_max` and `near_field_ratio_max`, [dimensionless]
///
/// # Errors
/// * the relative tolerance is not greater than 0 and at most 1
fn expansion_parameter_thresholds_derivatives(
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<ExpansionParameterThresholdsDerivatives, String> {
    let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(settings_greens_finite_size)?;

    let n_order_even: usize = settings_greens_finite_size.n_multipole / 2 + 1;
    let mut far_field_orders_ratio_max: Vec<f64> = vec![f64::NAN; n_order_even];
    for i_order_even in 0..n_order_even {
        let order_left_out: usize = 2 * i_order_even + 2;
        let mut growth: f64 = 1.0;
        for i_derivative in 1..DERIVATIVE_ORDER_MAX + 1 {
            growth *= ((order_left_out + i_derivative) as f64) / (i_derivative as f64);
        }
        far_field_orders_ratio_max[i_order_even] = (settings_greens_finite_size.relative_tolerance / growth)
            .powf(1.0 / (order_left_out as f64))
            .min(0.5);
    }

    Ok(ExpansionParameterThresholdsDerivatives {
        far_field_orders_ratio_max,
        near_field_ratio_max: thresholds.near_field_ratio_max,
    })
}

/// The flux at one point from one piece or conductor, per ampere, and its eight derivatives with respect to the point's position
#[derive(Clone, Copy, Debug)]
struct PsiDerivatives {
    /// The flux, [henry]
    psi: f64,
    /// `d(psi)/d(r)`, [henry / metre]
    d_psi_d_r: f64,
    /// `d(psi)/d(z)`, [henry / metre]
    d_psi_d_z: f64,
    /// `d2(psi)/d(r)2`, [henry / metre ** 2]
    d2_psi_d_r2: f64,
    /// `d2(psi)/d(r)d(z)`, [henry / metre ** 2]
    d2_psi_d_r_d_z: f64,
    /// `d2(psi)/d(z)2`, [henry / metre ** 2]
    d2_psi_d_z2: f64,
    /// `d3(psi)/d(r)2d(z)`, [henry / metre ** 3]
    d3_psi_d_r2_d_z: f64,
    /// `d3(psi)/d(r)d(z)2`, [henry / metre ** 3]
    d3_psi_d_r_d_z2: f64,
    /// `d3(psi)/d(z)3`, [henry / metre ** 3]
    d3_psi_d_z3: f64,
}

impl PsiDerivatives {
    /// All nine are zero, to be summed into
    fn zero() -> Self {
        PsiDerivatives {
            psi: 0.0,
            d_psi_d_r: 0.0,
            d_psi_d_z: 0.0,
            d2_psi_d_r2: 0.0,
            d2_psi_d_r_d_z: 0.0,
            d2_psi_d_z2: 0.0,
            d3_psi_d_r2_d_z: 0.0,
            d3_psi_d_r_d_z2: 0.0,
            d3_psi_d_z3: 0.0,
        }
    }

    /// The flux and its derivatives from its Taylor polynomial in the point's position: a coefficient times the factorials of its powers
    fn from_taylor(taylor: &Taylor) -> Self {
        let coefficient: &[f64; N_TAYLOR_COEFFICIENT] = &taylor.coefficient;
        PsiDerivatives {
            psi: coefficient[TAYLOR_1],
            d_psi_d_r: coefficient[TAYLOR_R],
            d_psi_d_z: coefficient[TAYLOR_Z],
            d2_psi_d_r2: 2.0 * coefficient[TAYLOR_RR],
            d2_psi_d_r_d_z: coefficient[TAYLOR_RZ],
            d2_psi_d_z2: 2.0 * coefficient[TAYLOR_ZZ],
            d3_psi_d_r2_d_z: 2.0 * coefficient[TAYLOR_RRZ],
            d3_psi_d_r_d_z2: 2.0 * coefficient[TAYLOR_RZZ],
            d3_psi_d_z3: 6.0 * coefficient[TAYLOR_ZZZ],
        }
    }

    /// Add `factor` times `other`
    fn add_scaled(&mut self, other: &PsiDerivatives, factor: f64) {
        self.psi += factor * other.psi;
        self.d_psi_d_r += factor * other.d_psi_d_r;
        self.d_psi_d_z += factor * other.d_psi_d_z;
        self.d2_psi_d_r2 += factor * other.d2_psi_d_r2;
        self.d2_psi_d_r_d_z += factor * other.d2_psi_d_r_d_z;
        self.d2_psi_d_z2 += factor * other.d2_psi_d_z2;
        self.d3_psi_d_r2_d_z += factor * other.d3_psi_d_r2_d_z;
        self.d3_psi_d_r_d_z2 += factor * other.d3_psi_d_r_d_z2;
        self.d3_psi_d_z3 += factor * other.d3_psi_d_z3;
    }

    /// Whether all nine are finite
    fn is_finite(&self) -> bool {
        self.psi.is_finite()
            && self.d_psi_d_r.is_finite()
            && self.d_psi_d_z.is_finite()
            && self.d2_psi_d_r2.is_finite()
            && self.d2_psi_d_r_d_z.is_finite()
            && self.d2_psi_d_z2.is_finite()
            && self.d3_psi_d_r2_d_z.is_finite()
            && self.d3_psi_d_r_d_z2.is_finite()
            && self.d3_psi_d_z3.is_finite()
    }
}

/// What `greens_derivatives_parallelogram` returns: the flux at the points from one conductor and its derivatives, and how they were
/// calculated at each point. Every array has shape = [n_point].
struct GreensDerivativesParallelogram {
    /// The flux at each point per ampere in the conductor, [henry]
    psi: Array1<f64>,
    /// `d(psi)/d(r)`, [henry / metre]
    d_psi_d_r: Array1<f64>,
    /// `d(psi)/d(z)`, [henry / metre]
    d_psi_d_z: Array1<f64>,
    /// `d2(psi)/d(r)2`, [henry / metre ** 2]
    d2_psi_d_r2: Array1<f64>,
    /// `d2(psi)/d(r)d(z)`, [henry / metre ** 2]
    d2_psi_d_r_d_z: Array1<f64>,
    /// `d2(psi)/d(z)2`, [henry / metre ** 2]
    d2_psi_d_z2: Array1<f64>,
    /// `d3(psi)/d(r)2d(z)`, [henry / metre ** 3]
    d3_psi_d_r2_d_z: Array1<f64>,
    /// `d3(psi)/d(r)d(z)2`, [henry / metre ** 3]
    d3_psi_d_r_d_z2: Array1<f64>,
    /// `d3(psi)/d(z)3`, [henry / metre ** 3]
    d3_psi_d_z3: Array1<f64>,
    /// How many pieces of the conductor were summed with the near-field series for each point, [count]
    #[allow(dead_code, reason = "how each point was calculated is only read by the tests")]
    n_piece_near_field: Array1<usize>,
    /// How many pieces of the conductor were summed with the far-field series for each point, [count]
    #[allow(dead_code, reason = "how each point was calculated is only read by the tests")]
    n_piece_far_field: Array1<usize>,
}

/// Where the shape of a piece is kept, in `greens_derivatives_parallelogram`
#[derive(Clone, Copy)]
enum PieceShape {
    /// `subdivisions_shape[i_subdivision]`: the shape after `i_subdivision` halvings of the whole conductor, which is shared by every point
    Subdivision(usize),
    /// `point_shapes[i_shape]`: a shape made for one point, by cutting the piece which contains it
    Point(usize),
}

/// A piece cut out of a piece which contains the point, as the ranges of the fractions of the two sides of that piece which it spans: the
/// points `corner + lambda_alpha * side_alpha + lambda_beta * side_beta` of the piece with `lambda_alpha` between `lambda_alpha_low` and
/// `lambda_alpha_high`, and `lambda_beta` between `lambda_beta_low` and `lambda_beta_high`
struct CutRange {
    /// [dimensionless]
    lambda_alpha_low: f64,
    /// [dimensionless]
    lambda_alpha_high: f64,
    /// [dimensionless]
    lambda_beta_low: f64,
    /// [dimensionless]
    lambda_beta_high: f64,
}

/// One of the pieces which a conductor is cut into for the derivatives: a parallelogram with the conductor's two inclinations
struct PieceDerivatives {
    /// R coordinate of the piece's reference corner, [metre]
    oblique_r: f64,
    /// Z coordinate of the piece's reference corner, [metre]
    oblique_z: f64,
    /// Where the piece's shape is kept
    shape: PieceShape,
    /// How many times the conductor was cut to make the piece
    i_subdivision: usize,
    /// The fraction of the conductor's current which the piece carries, [dimensionless]
    current_fraction: f64,
}

/// The flux and its derivatives at points from one parallelogram conductor which carries 1 ampere, uniformly distributed, with the series
/// chosen for each point.
///
/// The series are chosen as in `greens_psi_parallelogram`, with the thresholds of `expansion_parameter_thresholds_derivatives`, and the
/// far-field series is used with the lowest even order which is enough for the piece and the point, up to `n_multipole`. A piece for
/// which neither series converges fast enough is halved along its longer side, unless the point is inside it: it is then cut into a piece
/// of half its size centred on the point, the four pieces of the ring around that, and the strips which are left of the piece, so that
/// the point is never on an edge, or at a corner, of any piece of the conductor.
///
/// # Arguments
/// * `r` - R coordinate of the points, shape = [n_point], [metre]
/// * `z` - Z coordinate of the points, shape = [n_point], [metre]
/// * `oblique_r_prime` - R coordinate of the conductor's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the conductor's reference corner, [metre]
/// * `oblique_length_alpha_prime` - Length of the conductor's side which is inclined at `oblique_alpha_prime`, [metre]
/// * `oblique_length_beta_prime` - Length of the conductor's side which is inclined at `oblique_beta_prime`, [metre]
/// * `oblique_alpha_prime` - Inclination of the conductor's first side, counter-clockwise from the R axis, [radian]
/// * `oblique_beta_prime` - Inclination of the conductor's second side, counter-clockwise from the Z axis, [radian]
/// * `thresholds` - How small each series' expansion parameter has to be before the series is used, and which order of the far-field
///   series is enough, from `expansion_parameter_thresholds_derivatives`
/// * `coefficients` - The near-field series' two tables of coefficients, from `maxwell_series_coefficients`
/// * `settings_greens_finite_size` - The orders of the two series, and how many times the conductor may be cut
///
/// The points and the conductor have already been checked by `mutual_inductance_finite_size_to_point_source_with_derivatives`.
///
/// # Returns
/// * `greens_derivatives` - The flux at each point per ampere in the conductor and its derivatives, and how many pieces of the conductor
///   were summed with each series, all of shape = [n_point]
///
/// # Errors
/// * a point is on an edge of the conductor, or at one of its corners
/// * the flux or one of its derivatives is not finite at a point
/// * neither series converges fast enough for a piece of the conductor which has already been cut `n_subdivision_max` times
#[allow(clippy::too_many_arguments)]
fn greens_derivatives_parallelogram(
    r: &Array1<f64>,
    z: &Array1<f64>,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    oblique_length_alpha_prime: f64,
    oblique_length_beta_prime: f64,
    oblique_alpha_prime: f64,
    oblique_beta_prime: f64,
    thresholds: &ExpansionParameterThresholdsDerivatives,
    coefficients: &MaxwellSeriesCoefficients,
    settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
) -> Result<GreensDerivativesParallelogram, String> {
    let n_point: usize = r.len();
    let n_multipole: usize = settings_greens_finite_size.n_multipole;
    let n_order_even: usize = thresholds.far_field_orders_ratio_max.len();
    let n_subdivision_max: usize = settings_greens_finite_size.n_subdivision_max;
    // The highest degree, `a + b`, of the near-field series' terms
    let degree_max: usize = 2 * settings_greens_finite_size.n_elliptic + settings_greens_finite_size.n_thin_ring;

    // The directions of the two sides which meet at the reference corner
    let side_alpha_unit_r: f64 = oblique_alpha_prime.cos();
    let side_alpha_unit_z: f64 = oblique_alpha_prime.sin();
    let side_beta_unit_r: f64 = -oblique_beta_prime.sin();
    let side_beta_unit_z: f64 = oblique_beta_prime.cos();

    // The shapes of the pieces made by halving the whole conductor, as in `greens_psi_parallelogram`, which every point shares, and the
    // shapes made for one point by cutting the piece which contains it
    let mut subdivisions_shape: Vec<ParallelogramShape> = vec![ParallelogramShape::new(
        oblique_length_alpha_prime,
        oblique_length_beta_prime,
        side_alpha_unit_r,
        side_alpha_unit_z,
        side_beta_unit_r,
        side_beta_unit_z,
        degree_max,
        n_multipole,
    )];
    let mut point_shapes: Vec<ParallelogramShape> = Vec::new();

    // A cut makes at most eleven pieces, of which at most one is cut again in the same way; the others are halved. The count stops at the
    // largest integer when `n_subdivision_max` is large
    let n_piece_max: usize = 16_usize.saturating_mul(2_usize.saturating_pow(u32::try_from(n_subdivision_max).unwrap_or(u32::MAX).saturating_add(1)));

    // The work arrays of the two series, and the list of the pieces, are allocated once and used again for every point
    let mut near_field_work_arrays: NearFieldTaylorWorkArrays = NearFieldTaylorWorkArrays::new(degree_max);
    let mut far_field_work_arrays: FarFieldDerivativesWorkArrays = FarFieldDerivativesWorkArrays::new(n_multipole);
    let mut pieces: Vec<PieceDerivatives> = Vec::new();

    let mut greens_derivatives: GreensDerivativesParallelogram = GreensDerivativesParallelogram {
        psi: Array1::from_elem(n_point, f64::NAN),
        d_psi_d_r: Array1::from_elem(n_point, f64::NAN),
        d_psi_d_z: Array1::from_elem(n_point, f64::NAN),
        d2_psi_d_r2: Array1::from_elem(n_point, f64::NAN),
        d2_psi_d_r_d_z: Array1::from_elem(n_point, f64::NAN),
        d2_psi_d_z2: Array1::from_elem(n_point, f64::NAN),
        d3_psi_d_r2_d_z: Array1::from_elem(n_point, f64::NAN),
        d3_psi_d_r_d_z2: Array1::from_elem(n_point, f64::NAN),
        d3_psi_d_z3: Array1::from_elem(n_point, f64::NAN),
        n_piece_near_field: Array1::zeros(n_point),
        n_piece_far_field: Array1::zeros(n_point),
    };

    for i_point in 0..n_point {
        // The pieces which are still to be summed for this point, starting with the whole conductor. The last piece is taken from the list,
        // and it is either summed with one of the two series, or replaced with the pieces it is cut into
        pieces.clear();
        point_shapes.clear();
        pieces.push(PieceDerivatives {
            oblique_r: oblique_r_prime,
            oblique_z: oblique_z_prime,
            shape: PieceShape::Subdivision(0),
            i_subdivision: 0,
            current_fraction: 1.0,
        });

        let mut derivatives_sum: PsiDerivatives = PsiDerivatives::zero();
        'piece_loop: for _i_piece in 0..n_piece_max {
            let Some(piece): Option<PieceDerivatives> = pieces.pop() else {
                break 'piece_loop;
            };
            let shape: &ParallelogramShape = match piece.shape {
                PieceShape::Subdivision(i_subdivision) => &subdivisions_shape[i_subdivision],
                PieceShape::Point(i_shape) => &point_shapes[i_shape],
            };
            // What the cuts below need of the shape, copied so that more shapes can be made
            let length_alpha: f64 = shape.length_alpha;
            let length_beta: f64 = shape.length_beta;
            let side_alpha_r: f64 = shape.side_alpha_r;
            let side_alpha_z: f64 = shape.side_alpha_z;
            let side_beta_r: f64 = shape.side_beta_r;
            let side_beta_z: f64 = shape.side_beta_z;

            // The piece's centre
            let half_alpha_r: f64 = 0.5 * side_alpha_r;
            let half_alpha_z: f64 = 0.5 * side_alpha_z;
            let half_beta_r: f64 = 0.5 * side_beta_r;
            let half_beta_z: f64 = 0.5 * side_beta_z;
            let centre_r: f64 = piece.oblique_r + (half_alpha_r + half_beta_r);
            let centre_z: f64 = piece.oblique_z + (half_alpha_z + half_beta_z);

            // The two series' expansion parameters, as in `greens_psi_parallelogram`. The far-field series is used with the lowest even
            // order which is enough, if any is
            let distance: f64 = (r[i_point] - centre_r).hypot(z[i_point] - centre_z);
            let mut n_multipole_used: Option<usize> = None;
            'order_loop: for i_order_even in 0..n_order_even {
                if shape.half_diagonal <= thresholds.far_field_orders_ratio_max[i_order_even] * distance.min(centre_r) {
                    n_multipole_used = Some(2 * i_order_even);
                    break 'order_loop;
                }
            }
            let t: f64 = (distance + shape.half_diagonal) / (2.0 * r[i_point]);
            let is_near_field: bool = t <= thresholds.near_field_ratio_max * (1.0 - t);

            if let Some(n_multipole_used) = n_multipole_used {
                let derivatives: PsiDerivatives = greens_derivatives_parallelogram_far_field(
                    r[i_point],
                    z[i_point],
                    centre_r,
                    centre_z,
                    &shape.moment,
                    n_multipole_used,
                    &mut far_field_work_arrays,
                );
                derivatives_sum.add_scaled(&derivatives, piece.current_fraction);
                greens_derivatives.n_piece_far_field[i_point] += 1;
            } else if is_near_field {
                let taylor: Taylor = greens_derivatives_parallelogram_near_field(
                    r[i_point],
                    z[i_point],
                    piece.oblique_r,
                    piece.oblique_z,
                    shape,
                    coefficients,
                    &mut near_field_work_arrays,
                )
                .map_err(|error: String| format!("point {i_point}, at r = {} and z = {} metre: {error}", r[i_point], z[i_point]))?;
                derivatives_sum.add_scaled(&PsiDerivatives::from_taylor(&taylor), piece.current_fraction);
                greens_derivatives.n_piece_near_field[i_point] += 1;
            } else {
                if piece.i_subdivision == n_subdivision_max {
                    return Err(format!(
                        "point {i_point}, at r = {} and z = {} metre: neither series converges to `relative_tolerance` = {} for a piece of the \
                         conductor which has already been cut {n_subdivision_max} time(s); increase `n_subdivision_max`, or the orders \
                         `n_elliptic`, `n_thin_ring` and `n_multipole`",
                        r[i_point], z[i_point], settings_greens_finite_size.relative_tolerance
                    ));
                }

                // The point's position in the piece, as fractions of the two sides from the reference corner: `(lambda_alpha,
                // lambda_beta)` with `point = corner + lambda_alpha * side_alpha + lambda_beta * side_beta`
                let point_from_corner_r: f64 = r[i_point] - piece.oblique_r;
                let point_from_corner_z: f64 = z[i_point] - piece.oblique_z;
                let area_signed: f64 = side_alpha_r * side_beta_z - side_alpha_z * side_beta_r;
                let lambda_alpha: f64 = (point_from_corner_r * side_beta_z - point_from_corner_z * side_beta_r) / area_signed;
                let lambda_beta: f64 = (side_alpha_r * point_from_corner_z - side_alpha_z * point_from_corner_r) / area_signed;
                let is_inside: bool = lambda_alpha > 0.0 && lambda_alpha < 1.0 && lambda_beta > 0.0 && lambda_beta < 1.0;

                if is_inside {
                    let fraction_to_edge_min: f64 = lambda_alpha.min(1.0 - lambda_alpha).min(lambda_beta).min(1.0 - lambda_beta);
                    if fraction_to_edge_min < POINT_ON_EDGE_FRACTION {
                        return Err(format!(
                            "point {i_point}, at r = {} and z = {} metre, is on an edge of the conductor, to within rounding: the second \
                             derivatives of the flux jump there",
                            r[i_point], z[i_point]
                        ));
                    }

                    // The largest piece centred on the point, `[lambda_alpha_low, lambda_alpha_high] x [lambda_beta_low,
                    // lambda_beta_high]`, which reaches one edge of the piece in each direction, or both when the point is half way
                    let lambda_alpha_low: f64 = if lambda_alpha <= 0.5 { 0.0 } else { 2.0 * lambda_alpha - 1.0 };
                    let lambda_alpha_high: f64 = if lambda_alpha <= 0.5 { 2.0 * lambda_alpha } else { 1.0 };
                    let lambda_beta_low: f64 = if lambda_beta <= 0.5 { 0.0 } else { 2.0 * lambda_beta - 1.0 };
                    let lambda_beta_high: f64 = if lambda_beta <= 0.5 { 2.0 * lambda_beta } else { 1.0 };
                    let quarter_alpha: f64 = 0.25 * (lambda_alpha_high - lambda_alpha_low);
                    let quarter_beta: f64 = 0.25 * (lambda_beta_high - lambda_beta_low);

                    // The pieces, as ranges of the two fractions: the piece of half the size centred on the point, the ring around it, and
                    // the strips left of the piece, of which there are at most two. Neighbouring pieces share the same numbers at their
                    // common edge
                    let n_cut_max: usize = 9;
                    let mut cuts: Vec<CutRange> = Vec::with_capacity(n_cut_max);
                    cuts.push(CutRange {
                        lambda_alpha_low: lambda_alpha_low + quarter_alpha,
                        lambda_alpha_high: lambda_alpha_high - quarter_alpha,
                        lambda_beta_low: lambda_beta_low + quarter_beta,
                        lambda_beta_high: lambda_beta_high - quarter_beta,
                    });
                    cuts.push(CutRange {
                        lambda_alpha_low,
                        lambda_alpha_high: lambda_alpha_low + quarter_alpha,
                        lambda_beta_low,
                        lambda_beta_high,
                    });
                    cuts.push(CutRange {
                        lambda_alpha_low: lambda_alpha_high - quarter_alpha,
                        lambda_alpha_high,
                        lambda_beta_low,
                        lambda_beta_high,
                    });
                    cuts.push(CutRange {
                        lambda_alpha_low: lambda_alpha_low + quarter_alpha,
                        lambda_alpha_high: lambda_alpha_high - quarter_alpha,
                        lambda_beta_low,
                        lambda_beta_high: lambda_beta_low + quarter_beta,
                    });
                    cuts.push(CutRange {
                        lambda_alpha_low: lambda_alpha_low + quarter_alpha,
                        lambda_alpha_high: lambda_alpha_high - quarter_alpha,
                        lambda_beta_low: lambda_beta_high - quarter_beta,
                        lambda_beta_high,
                    });
                    if lambda_alpha_low > 0.0 {
                        cuts.push(CutRange {
                            lambda_alpha_low: 0.0,
                            lambda_alpha_high: lambda_alpha_low,
                            lambda_beta_low: 0.0,
                            lambda_beta_high: 1.0,
                        });
                    }
                    if lambda_alpha_high < 1.0 {
                        cuts.push(CutRange {
                            lambda_alpha_low: lambda_alpha_high,
                            lambda_alpha_high: 1.0,
                            lambda_beta_low: 0.0,
                            lambda_beta_high: 1.0,
                        });
                    }
                    if lambda_beta_low > 0.0 {
                        cuts.push(CutRange {
                            lambda_alpha_low,
                            lambda_alpha_high,
                            lambda_beta_low: 0.0,
                            lambda_beta_high: lambda_beta_low,
                        });
                    }
                    if lambda_beta_high < 1.0 {
                        cuts.push(CutRange {
                            lambda_alpha_low,
                            lambda_alpha_high,
                            lambda_beta_low: lambda_beta_high,
                            lambda_beta_high: 1.0,
                        });
                    }

                    let n_cut: usize = cuts.len();
                    for i_cut in 0..n_cut {
                        let cut: &CutRange = &cuts[i_cut];
                        let cut_fraction_alpha: f64 = cut.lambda_alpha_high - cut.lambda_alpha_low;
                        let cut_fraction_beta: f64 = cut.lambda_beta_high - cut.lambda_beta_low;
                        point_shapes.push(ParallelogramShape::new(
                            cut_fraction_alpha * length_alpha,
                            cut_fraction_beta * length_beta,
                            side_alpha_unit_r,
                            side_alpha_unit_z,
                            side_beta_unit_r,
                            side_beta_unit_z,
                            degree_max,
                            n_multipole,
                        ));
                        pieces.push(PieceDerivatives {
                            oblique_r: piece.oblique_r + (cut.lambda_alpha_low * side_alpha_r + cut.lambda_beta_low * side_beta_r),
                            oblique_z: piece.oblique_z + (cut.lambda_alpha_low * side_alpha_z + cut.lambda_beta_low * side_beta_z),
                            shape: PieceShape::Point(point_shapes.len() - 1),
                            i_subdivision: piece.i_subdivision + 1,
                            current_fraction: piece.current_fraction * cut_fraction_alpha * cut_fraction_beta,
                        });
                    }
                } else {
                    // Halve the piece along its longer side; each half carries half of its current
                    let is_alpha_longer: bool = length_alpha >= length_beta;
                    let second_half_oblique_r: f64 = if is_alpha_longer {
                        piece.oblique_r + half_alpha_r
                    } else {
                        piece.oblique_r + half_beta_r
                    };
                    let second_half_oblique_z: f64 = if is_alpha_longer {
                        piece.oblique_z + half_alpha_z
                    } else {
                        piece.oblique_z + half_beta_z
                    };
                    let halves_length_alpha: f64 = if is_alpha_longer { 0.5 * length_alpha } else { length_alpha };
                    let halves_length_beta: f64 = if is_alpha_longer { length_beta } else { 0.5 * length_beta };
                    let halves_shape: PieceShape = match piece.shape {
                        PieceShape::Subdivision(i_subdivision) => {
                            if i_subdivision + 1 == subdivisions_shape.len() {
                                // This is the first piece to be halved this many times, so the shape of its halves has not been built yet
                                subdivisions_shape.push(ParallelogramShape::new(
                                    halves_length_alpha,
                                    halves_length_beta,
                                    side_alpha_unit_r,
                                    side_alpha_unit_z,
                                    side_beta_unit_r,
                                    side_beta_unit_z,
                                    degree_max,
                                    n_multipole,
                                ));
                            }
                            PieceShape::Subdivision(i_subdivision + 1)
                        }
                        PieceShape::Point(_i_shape) => {
                            point_shapes.push(ParallelogramShape::new(
                                halves_length_alpha,
                                halves_length_beta,
                                side_alpha_unit_r,
                                side_alpha_unit_z,
                                side_beta_unit_r,
                                side_beta_unit_z,
                                degree_max,
                                n_multipole,
                            ));
                            PieceShape::Point(point_shapes.len() - 1)
                        }
                    };
                    pieces.push(PieceDerivatives {
                        oblique_r: second_half_oblique_r,
                        oblique_z: second_half_oblique_z,
                        shape: halves_shape,
                        i_subdivision: piece.i_subdivision + 1,
                        current_fraction: 0.5 * piece.current_fraction,
                    });
                    pieces.push(PieceDerivatives {
                        oblique_r: piece.oblique_r,
                        oblique_z: piece.oblique_z,
                        shape: halves_shape,
                        i_subdivision: piece.i_subdivision + 1,
                        current_fraction: 0.5 * piece.current_fraction,
                    });
                }
            }
        }
        if !pieces.is_empty() {
            return Err(format!(
                "point {i_point}, at r = {} and z = {} metre: the conductor was cut into more than {n_piece_max} pieces",
                r[i_point], z[i_point]
            ));
        }
        if !derivatives_sum.is_finite() {
            return Err(format!(
                "point {i_point}, at r = {} and z = {} metre: the flux or one of its derivatives is not finite",
                r[i_point], z[i_point]
            ));
        }

        greens_derivatives.psi[i_point] = derivatives_sum.psi;
        greens_derivatives.d_psi_d_r[i_point] = derivatives_sum.d_psi_d_r;
        greens_derivatives.d_psi_d_z[i_point] = derivatives_sum.d_psi_d_z;
        greens_derivatives.d2_psi_d_r2[i_point] = derivatives_sum.d2_psi_d_r2;
        greens_derivatives.d2_psi_d_r_d_z[i_point] = derivatives_sum.d2_psi_d_r_d_z;
        greens_derivatives.d2_psi_d_z2[i_point] = derivatives_sum.d2_psi_d_z2;
        greens_derivatives.d3_psi_d_r2_d_z[i_point] = derivatives_sum.d3_psi_d_r2_d_z;
        greens_derivatives.d3_psi_d_r_d_z2[i_point] = derivatives_sum.d3_psi_d_r_d_z2;
        greens_derivatives.d3_psi_d_z3[i_point] = derivatives_sum.d3_psi_d_z3;
    }

    Ok(greens_derivatives)
}

/// The work arrays of the far-field series with the derivatives. They are allocated once, and overwritten for every piece.
///
/// Each table holds the Taylor coefficients in the source's position about the centre, `(delta_r, delta_z)`, of a derivative of the Green's
/// function of a filament with respect to the point's position: `taylor_r[(i, j)]` multiplies `delta_r ** i * delta_z ** j` in `d(G)/d(r)`.
/// Each has shape = [n_multipole + 4, n_multipole + 4].
struct FarFieldDerivativesWorkArrays {
    /// The work arrays of the Taylor coefficients of the Green's function itself, made for the order `n_multipole + 3`
    far_field: FarFieldWorkArrays,
    /// `d(G)/d(r)`, for `i + j <= n_multipole + 2`
    taylor_r: Array2<f64>,
    /// `d(G)/d(z)`, for `i + j <= n_multipole + 2`
    taylor_z: Array2<f64>,
    /// `d2(G)/d(r)2`, for `i + j <= n_multipole + 1`
    taylor_rr: Array2<f64>,
    /// `d2(G)/d(r)d(z)`, for `i + j <= n_multipole + 1`
    taylor_rz: Array2<f64>,
    /// `d2(G)/d(z)2`, for `i + j <= n_multipole + 1`
    taylor_zz: Array2<f64>,
    /// `d3(G)/d(r)2d(z)`, for `i + j <= n_multipole`
    taylor_rrz: Array2<f64>,
    /// `d3(G)/d(r)d(z)2`, for `i + j <= n_multipole`
    taylor_rzz: Array2<f64>,
    /// `d3(G)/d(z)3`, for `i + j <= n_multipole`
    taylor_zzz: Array2<f64>,
}

impl FarFieldDerivativesWorkArrays {
    /// The work arrays of the far-field series with the derivatives, for the orders up to `n_multipole`.
    fn new(n_multipole: usize) -> Self {
        let n_order: usize = n_multipole + DERIVATIVE_ORDER_MAX;
        let n_series: usize = n_order + 1;
        FarFieldDerivativesWorkArrays {
            far_field: FarFieldWorkArrays::new(n_order),
            taylor_r: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_z: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_rr: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_rz: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_zz: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_rrz: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_rzz: Array2::from_elem((n_series, n_series), f64::NAN),
            taylor_zzz: Array2::from_elem((n_series, n_series), f64::NAN),
        }
    }
}

/// The far-field series with the derivatives: the flux at a point from a parallelogram which carries 1 ampere, uniformly distributed, and
/// its derivatives with respect to the point's position, for a point away from the parallelogram.
///
/// The flux is that of `greens_psi_parallelogram_far_field`, Eq. (69) of the note. The Taylor coefficients of the Green's function in the
/// source's position from the centre, `G_ij`, are calculated to the order `n_multipole + 3`, and those of each derivative follow from them.
/// With `h = z - centre_z`, and a function `F` of `(r, z, r', z')` which depends on `z - z'` only and is homogeneous of degree `p`, whose
/// Taylor coefficients are `F_ij`:
///
/// ```text
/// d(F)/d(z):  -(j + 1) * F_(i, j + 1)
/// d(F)/d(r):  ((p - i - j) * F_ij - centre_r * (i + 1) * F_(i + 1, j) + h * (j + 1) * F_(i, j + 1)) / r
/// ```
///
/// The first is `d/d(z) = -d/d(z')`. The second is Euler's theorem, `r * d(F)/d(r) + r' * d(F)/d(r') + (z - z') * d(F)/d(z) = p * F`, with
/// `r' = centre_r + delta_r` and `z - z' = h - delta_z`. Both derivatives are homogeneous of degree `p - 1`, and each needs one order more
/// of `F` than it gives; the Green's function has `p = 1`. Every derivative is then summed against the moments to the order `n_multipole`,
/// as the flux is.
///
/// # Arguments
/// * `r` - R coordinate of the point, [metre]
/// * `z` - Z coordinate of the point, [metre]
/// * `centre_r` - R coordinate of the parallelogram's centre, [metre]
/// * `centre_z` - Z coordinate of the parallelogram's centre, [metre]
/// * `moment` - The parallelogram's moments about its centre, from `parallelogram_moments`, of at least the order `n_multipole`,
///   [metre ** (i + j)]
/// * `n_multipole` - The highest order kept; 0 is a point filament, and an odd order adds nothing to the even order below it
/// * `work_arrays` - The work arrays, made for at least `n_multipole`, which are overwritten
///
/// # Returns
/// * `derivatives` - The flux at the point per ampere in the parallelogram [henry], and its derivatives [henry / metre ** n]
fn greens_derivatives_parallelogram_far_field(
    r: f64,
    z: f64,
    centre_r: f64,
    centre_z: f64,
    moment: &Array2<f64>,
    n_multipole: usize,
    work_arrays: &mut FarFieldDerivativesWorkArrays,
) -> PsiDerivatives {
    let n_order: usize = n_multipole + DERIVATIVE_ORDER_MAX;
    greens_taylor_coefficients(r, z, centre_r, centre_z, n_order, &mut work_arrays.far_field);
    let taylor: &Array2<f64> = &work_arrays.far_field.taylor;
    let h: f64 = z - centre_z;

    // The first derivatives, of degree 0, for `i + j <= n_order - 1`
    for i_series in 0..n_order {
        for j_series in 0..n_order - i_series {
            let degree_minus_order: f64 = 1.0 - ((i_series + j_series) as f64);
            work_arrays.taylor_r[(i_series, j_series)] = (degree_minus_order * taylor[(i_series, j_series)]
                - centre_r * ((i_series + 1) as f64) * taylor[(i_series + 1, j_series)]
                + h * ((j_series + 1) as f64) * taylor[(i_series, j_series + 1)])
                / r;
            work_arrays.taylor_z[(i_series, j_series)] = -((j_series + 1) as f64) * taylor[(i_series, j_series + 1)];
        }
    }

    // The second derivatives, of degree -1, for `i + j <= n_order - 2`
    for i_series in 0..n_order - 1 {
        for j_series in 0..n_order - 1 - i_series {
            let degree_minus_order: f64 = -((i_series + j_series) as f64);
            work_arrays.taylor_rr[(i_series, j_series)] = (degree_minus_order * work_arrays.taylor_r[(i_series, j_series)]
                - centre_r * ((i_series + 1) as f64) * work_arrays.taylor_r[(i_series + 1, j_series)]
                + h * ((j_series + 1) as f64) * work_arrays.taylor_r[(i_series, j_series + 1)])
                / r;
            work_arrays.taylor_rz[(i_series, j_series)] = -((j_series + 1) as f64) * work_arrays.taylor_r[(i_series, j_series + 1)];
            work_arrays.taylor_zz[(i_series, j_series)] = -((j_series + 1) as f64) * work_arrays.taylor_z[(i_series, j_series + 1)];
        }
    }

    // The third derivatives, for `i + j <= n_order - 3 = n_multipole`
    for i_series in 0..n_order - 2 {
        for j_series in 0..n_order - 2 - i_series {
            work_arrays.taylor_rrz[(i_series, j_series)] = -((j_series + 1) as f64) * work_arrays.taylor_rr[(i_series, j_series + 1)];
            work_arrays.taylor_rzz[(i_series, j_series)] = -((j_series + 1) as f64) * work_arrays.taylor_rz[(i_series, j_series + 1)];
            work_arrays.taylor_zzz[(i_series, j_series)] = -((j_series + 1) as f64) * work_arrays.taylor_zz[(i_series, j_series + 1)];
        }
    }

    // Sum the Taylor coefficients against the moments of the parallelogram, Eq. (69), as `greens_psi_parallelogram_far_field` does. The
    // odd moments are zero
    let n_series: usize = n_multipole + 1;
    let mut derivatives: PsiDerivatives = PsiDerivatives::zero();
    for i_series in 0..n_series {
        for j_series in 0..n_series - i_series {
            if (i_series + j_series) % 2 == 0 {
                let moment_ij: f64 = moment[(i_series, j_series)];
                derivatives.psi += moment_ij * taylor[(i_series, j_series)];
                derivatives.d_psi_d_r += moment_ij * work_arrays.taylor_r[(i_series, j_series)];
                derivatives.d_psi_d_z += moment_ij * work_arrays.taylor_z[(i_series, j_series)];
                derivatives.d2_psi_d_r2 += moment_ij * work_arrays.taylor_rr[(i_series, j_series)];
                derivatives.d2_psi_d_r_d_z += moment_ij * work_arrays.taylor_rz[(i_series, j_series)];
                derivatives.d2_psi_d_z2 += moment_ij * work_arrays.taylor_zz[(i_series, j_series)];
                derivatives.d3_psi_d_r2_d_z += moment_ij * work_arrays.taylor_rrz[(i_series, j_series)];
                derivatives.d3_psi_d_r_d_z2 += moment_ij * work_arrays.taylor_rzz[(i_series, j_series)];
                derivatives.d3_psi_d_z3 += moment_ij * work_arrays.taylor_zzz[(i_series, j_series)];
            }
        }
    }

    derivatives
}

/// The work arrays of the near-field series with the derivatives: those of `NearFieldWorkArrays`, as Taylor polynomials in the point's
/// position. They are allocated once, and overwritten for every edge of every piece.
struct NearFieldTaylorWorkArrays {
    /// The powers of `sigma` at the start of an edge, shape = [degree_max + 2], [dimensionless]
    sigma_start_power: Vec<TaylorComplex>,
    /// The powers of `sigma` at the end of an edge, shape = [degree_max + 2], [dimensionless]
    sigma_end_power: Vec<TaylorComplex>,
    /// `difference[k] = (sigma_end ** (k + 1) - sigma_start ** (k + 1)) / length`, shape = [degree_max + 1], [dimensionless]
    difference: Vec<TaylorComplex>,
}

impl NearFieldTaylorWorkArrays {
    /// The work arrays of the near-field series with the derivatives, for terms up to the degree `degree_max`.
    fn new(degree_max: usize) -> Self {
        let not_a_number: TaylorComplex = TaylorComplex::constant(c64::new(f64::NAN, f64::NAN));
        NearFieldTaylorWorkArrays {
            sigma_start_power: vec![not_a_number; degree_max + 2],
            sigma_end_power: vec![not_a_number; degree_max + 2],
            difference: vec![not_a_number; degree_max + 1],
        }
    }
}

/// The near-field series with the derivatives: the flux at a point from a parallelogram which carries 1 ampere, uniformly distributed, as
/// its Taylor polynomial in the point's position to the third degree, for a point inside the parallelogram or near it.
///
/// This is the closed form of `greens_psi_parallelogram_near_field`, Eq. (56) of the note, with every quantity which depends on the point's
/// position, `(r, z)`, written as its Taylor polynomial in `(delta_r, delta_z)`, so that the derivatives of the flux are those of the closed
/// form, exactly. The constant terms are calculated with the same operations as `greens_psi_parallelogram_near_field`, so they are the
/// flux it gives. Which end of an edge is the further from the point is decided by the constant terms, and the formula for that choice,
/// which is the same function on either side of the choice, is differentiated.
///
/// The closed form is a smooth function of the point's position, except where the point crosses an edge of the parallelogram, where the
/// second derivatives jump, and at its corners, where they are infinite.
///
/// # Arguments
/// * `r` - R coordinate of the point, [metre]
/// * `z` - Z coordinate of the point, [metre]
/// * `oblique_r_prime` - R coordinate of the parallelogram's reference corner, [metre]
/// * `oblique_z_prime` - Z coordinate of the parallelogram's reference corner, [metre]
/// * `shape` - The parallelogram's shape: its edges and its area
/// * `coefficients` - The two tables of coefficients, from `maxwell_series_coefficients`, whose orders are the orders of the series
/// * `work_arrays` - The work arrays, which are overwritten
///
/// # Returns
/// * `taylor` - The Taylor polynomial of the flux at the point per ampere in the parallelogram, [henry], and [henry / metre ** n] for the
///   coefficient of a term of degree n
///
/// # Errors
/// * the point is at a corner of the parallelogram, or on one of its edges
fn greens_derivatives_parallelogram_near_field(
    r: f64,
    z: f64,
    oblique_r_prime: f64,
    oblique_z_prime: f64,
    shape: &ParallelogramShape,
    coefficients: &MaxwellSeriesCoefficients,
    work_arrays: &mut NearFieldTaylorWorkArrays,
) -> Result<Taylor, String> {
    let table_p: &Array2<f64> = &coefficients.table_p;
    let table_q: &Array2<f64> = &coefficients.table_q;
    let degree_max: usize = table_p.nrows() - 1;
    let ln_4: f64 = 4.0_f64.ln();

    // The point's position, as the two variables of the Taylor polynomials
    let r_taylor: Taylor = Taylor::variable_r(r);
    let z_taylor: Taylor = Taylor::variable_z(z);

    // The sum over the edges, of the edge's distance from the point times its integrals. Every length is in units of `2 * r`
    let two_r: Taylor = 2.0 * r_taylor;
    let mut edges_sum: Taylor = Taylor::constant(0.0);
    for i_edge in 0..N_EDGE {
        // The edge's two ends from the point, rotated so that the edge lies along the real axis, Eq. (44), as in
        // `greens_psi_parallelogram_near_field`
        let phase: c64 = shape.edges_phase[i_edge];
        let phase_power: &[c64] = &shape.edges_phase_power[i_edge];
        let length: Taylor = shape.edges_length[i_edge] / two_r;
        let zeta_start: TaylorComplex = TaylorComplex::new(
            (oblique_r_prime - r_taylor) + shape.edges_start_r[i_edge],
            (oblique_z_prime - z_taylor) + shape.edges_start_z[i_edge],
        )
        .divide_by_real(two_r);
        let zeta_start_rotated: TaylorComplex = zeta_start.multiply_by_constant(phase.conj());
        let s_start: Taylor = zeta_start_rotated.re;
        let s_end: Taylor = s_start + length;
        let edge_distance: Taylor = -zeta_start_rotated.im;
        let sigma_start: TaylorComplex = TaylorComplex::new(s_start, -edge_distance);
        let sigma_end: TaylorComplex = TaylorComplex::new(s_end, -edge_distance);

        // The closed form is not smooth at a corner or on an edge, so neither has derivatives
        let start_abs_sq: Taylor = s_start * s_start + edge_distance * edge_distance;
        let end_abs_sq: Taylor = s_end * s_end + edge_distance * edge_distance;
        if start_abs_sq.value() == 0.0 || end_abs_sq.value() == 0.0 {
            return Err("the point is at a corner of the conductor, where the second derivatives of the flux are infinite".to_string());
        }
        if edge_distance.value() == 0.0 && s_start.value() < 0.0 && s_end.value() > 0.0 {
            return Err("the point is on an edge of the conductor, where the second derivatives of the flux jump".to_string());
        }

        // The logarithms of the distances to the two ends, as in `greens_psi_parallelogram_near_field`. Where the two distances are similar,
        // `|sigma_near| ** 2 - |sigma_far| ** 2` is written as the function which it is for the end which is chosen as the far one, so that
        // it is differentiated as that function
        let is_end_far: bool = end_abs_sq.value() >= start_abs_sq.value();
        let far_abs_sq: Taylor = if is_end_far { end_abs_sq } else { start_abs_sq };
        let near_abs_sq: Taylor = if is_end_far { start_abs_sq } else { end_abs_sq };
        let ln_far: Taylor = 0.5 * far_abs_sq.ln();
        let ratio_sq: Taylor = near_abs_sq / far_abs_sq;
        let ln_near_over_far: Taylor = if ratio_sq.value() < 0.5 {
            0.5 * ratio_sq.ln()
        } else {
            let near_minus_far_abs_sq: Taylor = if is_end_far {
                -(length * (s_start + s_end))
            } else {
                length * (s_start + s_end)
            };
            0.5 * (near_minus_far_abs_sq / far_abs_sq).ln_1p()
        };
        let sign_near: f64 = if is_end_far { -1.0 } else { 1.0 };

        // The powers of `sigma` at the two ends, and `difference[k] = [sigma ** (k + 1)] / length`, without cancellation
        work_arrays.sigma_start_power[0] = TaylorComplex::constant(c64::new(1.0, 0.0));
        work_arrays.sigma_end_power[0] = TaylorComplex::constant(c64::new(1.0, 0.0));
        work_arrays.difference[0] = TaylorComplex::constant(c64::new(1.0, 0.0));
        for i_power in 1..degree_max + 2 {
            work_arrays.sigma_start_power[i_power] = work_arrays.sigma_start_power[i_power - 1] * sigma_start;
            work_arrays.sigma_end_power[i_power] = work_arrays.sigma_end_power[i_power - 1] * sigma_end;
            if i_power <= degree_max {
                work_arrays.difference[i_power] = sigma_end * work_arrays.difference[i_power - 1] + work_arrays.sigma_start_power[i_power];
            }
        }
        let sigma_start_power: &[TaylorComplex] = &work_arrays.sigma_start_power;
        let sigma_end_power: &[TaylorComplex] = &work_arrays.sigma_end_power;
        let difference: &[TaylorComplex] = &work_arrays.difference;
        let sigma_near_power: &[TaylorComplex] = if is_end_far { sigma_start_power } else { sigma_end_power };

        // I_(0, -1), Eq. (54): ln(d_end / d_start) - i * omega_e
        let mut integral_inverse: TaylorComplex = TaylorComplex::new(
            sign_near * ln_near_over_far,
            -Taylor::atan2(edge_distance * length, s_start * s_end + edge_distance * edge_distance),
        );

        let mut edge_sum: Taylor = Taylor::constant(0.0);
        for i_degree in 0..degree_max + 1 {
            // The terms with `i_a + i_b = i_degree`, from I_(D, 0), Eq. (52), I_(D + 1, -1), Eq. (53), and J_(D, 0), Eq. (51)
            let inverse_degree_plus_one: f64 = 1.0 / ((i_degree + 1) as f64);
            let mut integral_plain: TaylorComplex = difference[i_degree] * length * inverse_degree_plus_one;
            integral_inverse = integral_plain - integral_inverse.multiply_by_imaginary(2.0 * edge_distance);
            let mut integral_ln: TaylorComplex = (difference[i_degree] * length * ln_far + sigma_near_power[i_degree + 1] * sign_near * ln_near_over_far
                - (integral_plain + integral_inverse) * 0.5)
                * inverse_degree_plus_one;

            for i_b in 0..i_degree / 2 + 1 {
                let i_a: usize = i_degree - i_b;
                if i_b >= 1 {
                    // Step from (i_a + 1, i_b - 1) to (i_a, i_b), by integration by parts, Eq. (49) and (51)
                    let boundary: TaylorComplex =
                        (difference[i_a] * sigma_end_power[i_b].conj() + sigma_start_power[i_a + 1] * difference[i_b - 1].conj()) * length;
                    let boundary_ln: TaylorComplex =
                        boundary * ln_far + sigma_near_power[i_a + 1] * sign_near * sigma_near_power[i_b].conj() * ln_near_over_far;
                    let inverse_a_plus_one: f64 = 1.0 / ((i_a + 1) as f64);
                    let integral_plain_previous: TaylorComplex = integral_plain;
                    integral_plain = (boundary - integral_plain_previous * (i_b as f64)) * inverse_a_plus_one;
                    integral_ln = (boundary_ln - (integral_plain + integral_plain_previous) * 0.5 - integral_ln * (i_b as f64)) * inverse_a_plus_one;
                }

                // The term of Eq. (56), where the truncation of the two tables keeps it. The (i_b, i_a) term is its complex conjugate
                let coefficient_p: f64 = table_p[(i_a, i_b)];
                let coefficient_q: f64 = table_q[(i_a, i_b)];
                if coefficient_p != 0.0 || coefficient_q != 0.0 {
                    let multiplicity: f64 = if i_a == i_b { 1.0 } else { 2.0 };
                    let degree_plus_two: f64 = (i_degree + 2) as f64;
                    let term: TaylorComplex = (integral_plain * (coefficient_p * (ln_4 + 1.0 / degree_plus_two) + coefficient_q) - integral_ln * coefficient_p)
                        .multiply_by_constant(phase_power[i_a - i_b]);
                    edge_sum += (multiplicity / degree_plus_two) * term.re;
                }
            }
        }

        edges_sum += edge_distance * edge_sum;
    }

    // The area, in units of `2 * r` squared
    let area_scaled: Taylor = shape.area / (two_r * two_r);

    Ok(MU_0 * r_taylor * edges_sum / area_scaled)
}

/// The number of coefficients of a Taylor polynomial in two variables to the third degree
const N_TAYLOR_COEFFICIENT: usize = 10;
/// The index of the coefficient of 1 in `Taylor::coefficient`
const TAYLOR_1: usize = 0;
/// The index of the coefficient of `delta_r`
const TAYLOR_R: usize = 1;
/// The index of the coefficient of `delta_z`
const TAYLOR_Z: usize = 2;
/// The index of the coefficient of `delta_r ** 2`
const TAYLOR_RR: usize = 3;
/// The index of the coefficient of `delta_r * delta_z`
const TAYLOR_RZ: usize = 4;
/// The index of the coefficient of `delta_z ** 2`
const TAYLOR_ZZ: usize = 5;
/// The index of the coefficient of `delta_r ** 2 * delta_z`
const TAYLOR_RRZ: usize = 7;
/// The index of the coefficient of `delta_r * delta_z ** 2`
const TAYLOR_RZZ: usize = 8;
/// The index of the coefficient of `delta_z ** 3`
const TAYLOR_ZZZ: usize = 9;

/// A function of the point's position, as its Taylor polynomial in `(delta_r, delta_z)`, the change of the point's position, truncated
/// after the third degree. The coefficients multiply, in order: 1, `delta_r`, `delta_z`, `delta_r ** 2`, `delta_r * delta_z`,
/// `delta_z ** 2`, `delta_r ** 3`, `delta_r ** 2 * delta_z`, `delta_r * delta_z ** 2` and `delta_z ** 3`.
///
/// Sums, products, quotients and functions of these polynomials are those of the functions they describe, to the third degree, so a
/// calculation done with them gives its result's derivatives with respect to the point's position up to the third, exactly: the coefficient
/// of `delta_r ** a * delta_z ** b` is the derivative divided by `a! * b!`. The constant term of every operation is the operation on the
/// constant terms, so it is the value the same calculation gives with numbers.
#[derive(Clone, Copy)]
struct Taylor {
    /// The coefficients, in the order above
    coefficient: [f64; N_TAYLOR_COEFFICIENT],
}

impl Taylor {
    /// A constant: its derivatives are zero
    fn constant(value: f64) -> Self {
        let mut coefficient: [f64; N_TAYLOR_COEFFICIENT] = [0.0; N_TAYLOR_COEFFICIENT];
        coefficient[TAYLOR_1] = value;
        Taylor { coefficient }
    }

    /// The point's major radius, `r + delta_r`
    fn variable_r(r: f64) -> Self {
        let mut variable: Taylor = Taylor::constant(r);
        variable.coefficient[TAYLOR_R] = 1.0;
        variable
    }

    /// The point's height, `z + delta_z`
    fn variable_z(z: f64) -> Self {
        let mut variable: Taylor = Taylor::constant(z);
        variable.coefficient[TAYLOR_Z] = 1.0;
        variable
    }

    /// The value at the point, the constant term
    fn value(&self) -> f64 {
        self.coefficient[TAYLOR_1]
    }

    /// A function of this polynomial, from the function's Taylor coefficients at this polynomial's value, `function_taylor[k] =
    /// function^(k)(value) / k!`: `f(value + variation) = sum over k of function_taylor[k] * variation ** k`
    fn compose(self, function_taylor: [f64; DERIVATIVE_ORDER_MAX + 1]) -> Taylor {
        let mut variation: Taylor = self;
        variation.coefficient[TAYLOR_1] = 0.0;
        let variation_sq: Taylor = variation * variation;
        let variation_cube: Taylor = variation_sq * variation;
        let mut result: Taylor = Taylor::constant(function_taylor[0]);
        for i_coefficient in 1..N_TAYLOR_COEFFICIENT {
            result.coefficient[i_coefficient] = function_taylor[1] * variation.coefficient[i_coefficient]
                + function_taylor[2] * variation_sq.coefficient[i_coefficient]
                + function_taylor[3] * variation_cube.coefficient[i_coefficient];
        }
        result
    }

    /// The natural logarithm
    fn ln(self) -> Taylor {
        let value: f64 = self.value();
        let inverse: f64 = 1.0 / value;
        self.compose([value.ln(), inverse, -0.5 * inverse.powi(2), inverse.powi(3) / 3.0])
    }

    /// `ln(1 + self)`, with its constant term from `ln_1p`
    fn ln_1p(self) -> Taylor {
        let value: f64 = self.value();
        let inverse: f64 = 1.0 / (1.0 + value);
        self.compose([value.ln_1p(), inverse, -0.5 * inverse.powi(2), inverse.powi(3) / 3.0])
    }

    /// The angle of `(x, y)` from the x axis, with its constant term from `atan2`. The change of the angle from its value at the point has
    /// the tangent `t = (x_0 * y - y_0 * x) / (x_0 * x + y_0 * y)`, which has no constant term, and the change is `t - t ** 3 / 3`
    fn atan2(y: Taylor, x: Taylor) -> Taylor {
        let y_value: f64 = y.value();
        let x_value: f64 = x.value();
        let mut tangent: Taylor = (y * x_value - x * y_value) / (x * x_value + y * y_value);
        tangent.coefficient[TAYLOR_1] = 0.0;
        let tangent_cube: Taylor = tangent * tangent * tangent;
        let mut angle: Taylor = tangent - tangent_cube * (1.0 / 3.0);
        angle.coefficient[TAYLOR_1] = y_value.atan2(x_value);
        angle
    }
}

impl std::ops::Add for Taylor {
    type Output = Taylor;
    fn add(self, other: Taylor) -> Taylor {
        let mut sum: Taylor = self;
        for i_coefficient in 0..N_TAYLOR_COEFFICIENT {
            sum.coefficient[i_coefficient] += other.coefficient[i_coefficient];
        }
        sum
    }
}

impl std::ops::AddAssign for Taylor {
    fn add_assign(&mut self, other: Taylor) {
        for i_coefficient in 0..N_TAYLOR_COEFFICIENT {
            self.coefficient[i_coefficient] += other.coefficient[i_coefficient];
        }
    }
}

impl std::ops::Add<f64> for Taylor {
    type Output = Taylor;
    fn add(self, other: f64) -> Taylor {
        let mut sum: Taylor = self;
        sum.coefficient[TAYLOR_1] += other;
        sum
    }
}

impl std::ops::Sub for Taylor {
    type Output = Taylor;
    fn sub(self, other: Taylor) -> Taylor {
        let mut difference: Taylor = self;
        for i_coefficient in 0..N_TAYLOR_COEFFICIENT {
            difference.coefficient[i_coefficient] -= other.coefficient[i_coefficient];
        }
        difference
    }
}

impl std::ops::Sub<Taylor> for f64 {
    type Output = Taylor;
    fn sub(self, other: Taylor) -> Taylor {
        Taylor::constant(self) - other
    }
}

impl std::ops::Neg for Taylor {
    type Output = Taylor;
    fn neg(self) -> Taylor {
        let mut negative: Taylor = self;
        for i_coefficient in 0..N_TAYLOR_COEFFICIENT {
            negative.coefficient[i_coefficient] = -negative.coefficient[i_coefficient];
        }
        negative
    }
}

impl std::ops::Mul for Taylor {
    type Output = Taylor;
    /// The product, truncated after the third degree
    fn mul(self, other: Taylor) -> Taylor {
        let f: &[f64; N_TAYLOR_COEFFICIENT] = &self.coefficient;
        let g: &[f64; N_TAYLOR_COEFFICIENT] = &other.coefficient;
        Taylor {
            coefficient: [
                f[0] * g[0],
                f[0] * g[1] + f[1] * g[0],
                f[0] * g[2] + f[2] * g[0],
                f[0] * g[3] + f[1] * g[1] + f[3] * g[0],
                f[0] * g[4] + f[1] * g[2] + f[2] * g[1] + f[4] * g[0],
                f[0] * g[5] + f[2] * g[2] + f[5] * g[0],
                f[0] * g[6] + f[1] * g[3] + f[3] * g[1] + f[6] * g[0],
                f[0] * g[7] + f[1] * g[4] + f[2] * g[3] + f[3] * g[2] + f[4] * g[1] + f[7] * g[0],
                f[0] * g[8] + f[1] * g[5] + f[2] * g[4] + f[4] * g[2] + f[5] * g[1] + f[8] * g[0],
                f[0] * g[9] + f[2] * g[5] + f[5] * g[2] + f[9] * g[0],
            ],
        }
    }
}

impl std::ops::Mul<f64> for Taylor {
    type Output = Taylor;
    fn mul(self, other: f64) -> Taylor {
        let mut product: Taylor = self;
        for i_coefficient in 0..N_TAYLOR_COEFFICIENT {
            product.coefficient[i_coefficient] *= other;
        }
        product
    }
}

impl std::ops::Mul<Taylor> for f64 {
    type Output = Taylor;
    fn mul(self, other: Taylor) -> Taylor {
        other * self
    }
}

impl std::ops::Div for Taylor {
    type Output = Taylor;
    /// The quotient `h = f / g`, from `h * g = f`, one coefficient at a time in order of degree
    fn div(self, other: Taylor) -> Taylor {
        let f: &[f64; N_TAYLOR_COEFFICIENT] = &self.coefficient;
        let g: &[f64; N_TAYLOR_COEFFICIENT] = &other.coefficient;
        let mut h: [f64; N_TAYLOR_COEFFICIENT] = [f64::NAN; N_TAYLOR_COEFFICIENT];
        h[0] = f[0] / g[0];
        h[1] = (f[1] - h[0] * g[1]) / g[0];
        h[2] = (f[2] - h[0] * g[2]) / g[0];
        h[3] = (f[3] - h[0] * g[3] - h[1] * g[1]) / g[0];
        h[4] = (f[4] - h[0] * g[4] - h[1] * g[2] - h[2] * g[1]) / g[0];
        h[5] = (f[5] - h[0] * g[5] - h[2] * g[2]) / g[0];
        h[6] = (f[6] - h[0] * g[6] - h[1] * g[3] - h[3] * g[1]) / g[0];
        h[7] = (f[7] - h[0] * g[7] - h[1] * g[4] - h[2] * g[3] - h[3] * g[2] - h[4] * g[1]) / g[0];
        h[8] = (f[8] - h[0] * g[8] - h[1] * g[5] - h[2] * g[4] - h[4] * g[2] - h[5] * g[1]) / g[0];
        h[9] = (f[9] - h[0] * g[9] - h[2] * g[5] - h[5] * g[2]) / g[0];
        Taylor { coefficient: h }
    }
}

impl std::ops::Div<Taylor> for f64 {
    type Output = Taylor;
    fn div(self, other: Taylor) -> Taylor {
        Taylor::constant(self) / other
    }
}

/// A complex function of the point's position, as the Taylor polynomials of its real and imaginary parts
#[derive(Clone, Copy)]
struct TaylorComplex {
    /// The real part
    re: Taylor,
    /// The imaginary part
    im: Taylor,
}

impl TaylorComplex {
    /// From its real and imaginary parts
    fn new(re: Taylor, im: Taylor) -> Self {
        TaylorComplex { re, im }
    }

    /// A constant
    fn constant(value: c64) -> Self {
        TaylorComplex {
            re: Taylor::constant(value.re),
            im: Taylor::constant(value.im),
        }
    }

    /// The complex conjugate
    fn conj(self) -> TaylorComplex {
        TaylorComplex { re: self.re, im: -self.im }
    }

    /// The product with a complex constant
    fn multiply_by_constant(self, other: c64) -> TaylorComplex {
        TaylorComplex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// The product with `i` times a real polynomial, `i * imaginary * self`
    fn multiply_by_imaginary(self, imaginary: Taylor) -> TaylorComplex {
        TaylorComplex {
            re: -(imaginary * self.im),
            im: imaginary * self.re,
        }
    }

    /// The quotient by a real polynomial
    fn divide_by_real(self, other: Taylor) -> TaylorComplex {
        TaylorComplex {
            re: self.re / other,
            im: self.im / other,
        }
    }
}

impl std::ops::Add for TaylorComplex {
    type Output = TaylorComplex;
    fn add(self, other: TaylorComplex) -> TaylorComplex {
        TaylorComplex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for TaylorComplex {
    type Output = TaylorComplex;
    fn sub(self, other: TaylorComplex) -> TaylorComplex {
        TaylorComplex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Mul for TaylorComplex {
    type Output = TaylorComplex;
    fn mul(self, other: TaylorComplex) -> TaylorComplex {
        TaylorComplex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Mul<Taylor> for TaylorComplex {
    type Output = TaylorComplex;
    fn mul(self, other: Taylor) -> TaylorComplex {
        TaylorComplex {
            re: self.re * other,
            im: self.im * other,
        }
    }
}

impl std::ops::Mul<f64> for TaylorComplex {
    type Output = TaylorComplex;
    fn mul(self, other: f64) -> TaylorComplex {
        TaylorComplex {
            re: self.re * other,
            im: self.im * other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::point_to_point::PointToPoint;
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;
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

    /// One point of the table of reference values
    struct ReferenceCase {
        /// Which of `REFERENCE_CONDUCTORS` carries the current
        i_conductor: usize,
        /// R coordinate of the point, [metre]
        r: f64,
        /// Z coordinate of the point, [metre]
        z: f64,
        /// The flux at the point per ampere in the conductor, with each of `REFERENCE_SETTINGS`, [henry]
        g_psi: [f64; N_REFERENCE_SETTING],
        /// How many pieces of the conductor are summed with the near-field series, with each of `REFERENCE_SETTINGS`
        n_piece_near_field: [usize; N_REFERENCE_SETTING],
        /// How many pieces of the conductor are summed with the far-field series, with each of `REFERENCE_SETTINGS`
        n_piece_far_field: [usize; N_REFERENCE_SETTING],
    }

    /// The settings of the table of reference values: high orders, with which most points take one piece; low orders; and the lowest
    /// orders, with which a conductor is cut into as many as a thousand pieces
    const N_REFERENCE_SETTING: usize = 3;
    const REFERENCE_SETTINGS: [SettingsEquilibriumNumericsGreensFiniteSize; N_REFERENCE_SETTING] = [
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 10,
            n_thin_ring: 20,
            n_multipole: 20,
            relative_tolerance: 1.0e-12,
            n_subdivision_max: 30,
        },
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 2,
            n_thin_ring: 4,
            n_multipole: 4,
            relative_tolerance: 1.0e-4,
            n_subdivision_max: 30,
        },
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 0,
            n_thin_ring: 0,
            n_multipole: 1,
            relative_tolerance: 1.0e-2,
            n_subdivision_max: 30,
        },
    ];

    const N_REFERENCE_CONDUCTOR: usize = 4;
    const REFERENCE_CONDUCTORS: [ReferenceConductor; N_REFERENCE_CONDUCTOR] = [
        // 0: a thin tall rectangle, an element of the inboard wall
        ReferenceConductor {
            oblique_r: 0.15357,
            oblique_z: 0.0,
            oblique_length_alpha: 0.0032351,
            oblique_length_beta: 0.0240734,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        },
        // 1: a square, the grid cell of the GSFit paper
        ReferenceConductor {
            oblique_r: 0.40375,
            oblique_z: -0.00625,
            oblique_length_alpha: 0.0125,
            oblique_length_beta: 0.0125,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        },
        // 2: a sheared parallelogram, the fourth element of ST40's upper divertor passive plate
        ReferenceConductor {
            oblique_r: 0.31726463700234186,
            oblique_z: 0.6797142857142857,
            oblique_length_alpha: 0.034393442622950854,
            oblique_length_beta: 0.036087264455791655,
            oblique_alpha: 0.0,
            oblique_beta: -0.6103591265479971,
        },
        // 3: a sheared and rotated parallelogram, whose two sides are in clockwise order
        ReferenceConductor {
            oblique_r: 0.52,
            oblique_z: -0.31,
            oblique_length_alpha: 0.021,
            oblique_length_beta: 0.034,
            oblique_alpha: 2.9,
            oblique_beta: 0.4,
        },
    ];

    /// The flux from each of `REFERENCE_CONDUCTORS` at nine points, with each of `REFERENCE_SETTINGS`, and how many pieces of the conductor
    /// are summed with each series. The values were calculated, one point at a time, with the library which accompanies the note, in
    /// `documentation/flux_from_parallelogram_conductor/code/`. Every number is written with as many digits as it takes to read it back as
    /// the same double, which is at most 17
    const N_REFERENCE_CASE: usize = 36;
    #[rustfmt::skip]
    const REFERENCE_CASES: [ReferenceCase; N_REFERENCE_CASE] = [
        // Conductor 0: a thin tall rectangle, an element of the inboard wall
        // at the centre
        ReferenceCase { i_conductor: 0, r: 0.15518755, z: 0.0120367, n_piece_near_field: [1, 1, 24], n_piece_far_field: [0, 0, 144],
            g_psi: [6.894410135164899e-07, 6.894410135019046e-07, 6.894173960568285e-07] },
        // inside, off the centre
        ReferenceCase { i_conductor: 0, r: 0.15421702, z: 0.01685138, n_piece_near_field: [1, 1, 17], n_piece_far_field: [0, 0, 139],
            g_psi: [6.642304036349318e-07, 6.642304029608111e-07, 6.636643167703735e-07] },
        // on the first side, a quarter of the way along it
        ReferenceCase { i_conductor: 0, r: 0.15437877500000002, z: 0.0, n_piece_near_field: [1, 1, 11], n_piece_far_field: [0, 0, 87],
            g_psi: [5.604815341056683e-07, 5.604815309087624e-07, 5.602235388726705e-07] },
        // at the reference corner
        ReferenceCase { i_conductor: 0, r: 0.15357, z: 0.0, n_piece_near_field: [1, 1, 11], n_piece_far_field: [0, 0, 93],
            g_psi: [5.511582407636707e-07, 5.51158235057152e-07, 5.50825834461452e-07] },
        // at the corner opposite the reference corner
        ReferenceCase { i_conductor: 0, r: 0.1568051, z: 0.0240734, n_piece_near_field: [1, 1, 9], n_piece_far_field: [0, 0, 92],
            g_psi: [5.590240175502911e-07, 5.590240215633396e-07, 5.5930844069456e-07] },
        // just outside the first side
        ReferenceCase { i_conductor: 0, r: 0.15518755, z: -0.00120367, n_piece_near_field: [1, 1, 8], n_piece_far_field: [0, 0, 85],
            g_psi: [5.336157745409827e-07, 5.336157735587561e-07, 5.335606819475157e-07] },
        // 3 half-diagonals from the centre
        ReferenceCase { i_conductor: 0, r: 0.18305434742478216, z: 0.03550857967161247, n_piece_near_field: [1, 1, 0], n_piece_far_field: [0, 0, 5],
            g_psi: [3.467247708923833e-07, 3.4672479044649937e-07, 3.4671242288009477e-07] },
        // far away, at a corner of the grid
        ReferenceCase { i_conductor: 0, r: 1.01, z: -1.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [1.6481421757676413e-08, 1.6481421757679626e-08, 1.647995099764262e-08] },
        // near the axis, at the inner edge of the grid
        ReferenceCase { i_conductor: 0, r: 0.01, z: 0.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [1.2588390884311391e-09, 1.2588391534154905e-09, 1.262463395996048e-09] },
        // Conductor 1: a square, the grid cell of the GSFit paper
        // at the centre
        ReferenceCase { i_conductor: 1, r: 0.41, z: 0.0, n_piece_near_field: [1, 1, 16], n_piece_far_field: [0, 0, 16],
            g_psi: [2.386081204993199e-06, 2.3860812049932003e-06, 2.3860221985486727e-06] },
        // inside, off the centre
        ReferenceCase { i_conductor: 1, r: 0.40625, z: 0.0024999999999999988, n_piece_near_field: [1, 1, 11], n_piece_far_field: [0, 0, 73],
            g_psi: [2.269211789880396e-06, 2.269211789883636e-06, 2.2653251573328795e-06] },
        // on the first side, a quarter of the way along it
        ReferenceCase { i_conductor: 1, r: 0.406875, z: -0.00625, n_piece_near_field: [1, 1, 10], n_piece_far_field: [0, 0, 82],
            g_psi: [2.1269149981263717e-06, 2.1269149981163157e-06, 2.12483318107106e-06] },
        // at the reference corner
        ReferenceCase { i_conductor: 1, r: 0.40375, z: -0.00625, n_piece_near_field: [1, 1, 7], n_piece_far_field: [0, 0, 87],
            g_psi: [2.0090103180318125e-06, 2.009010318031766e-06, 2.0062560428959747e-06] },
        // at the corner opposite the reference corner
        ReferenceCase { i_conductor: 1, r: 0.41625, z: 0.00625, n_piece_near_field: [1, 1, 8], n_piece_far_field: [0, 0, 81],
            g_psi: [2.049259018363471e-06, 2.0492590183636865e-06, 2.052255109214486e-06] },
        // just outside the first side
        ReferenceCase { i_conductor: 1, r: 0.41, z: -0.006875000000000001, n_piece_near_field: [1, 1, 10], n_piece_far_field: [0, 0, 78],
            g_psi: [2.1301167404342068e-06, 2.130116740434178e-06, 2.130017207311008e-06] },
        // 3 half-diagonals from the centre
        ReferenceCase { i_conductor: 1, r: 0.4302809411437405, z: 0.017082401070228216, n_piece_near_field: [1, 1, 0], n_piece_far_field: [0, 0, 18],
            g_psi: [1.502183216680692e-06, 1.5021832156413509e-06, 1.5021729594152042e-06] },
        // far away, at a corner of the grid
        ReferenceCase { i_conductor: 1, r: 1.01, z: -1.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [1.1234032824086926e-07, 1.1234032824086966e-07, 1.1233206647802599e-07] },
        // near the axis, at the inner edge of the grid
        ReferenceCase { i_conductor: 1, r: 0.01, z: 0.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [4.81532888994526e-10, 4.815328889932799e-10, 4.815515586132322e-10] },
        // Conductor 2: a sheared parallelogram, the fourth element of ST40's upper divertor passive plate
        // at the centre
        ReferenceCase { i_conductor: 2, r: 0.34480327868852456, z: 0.6945, n_piece_near_field: [1, 1, 58], n_piece_far_field: [0, 0, 980],
            g_psi: [1.506512604652171e-06, 1.5065126046432583e-06, 1.5064532986198696e-06] },
        // inside, off the centre
        ReferenceCase { i_conductor: 2, r: 0.3386220140515222, z: 0.7004142857142858, n_piece_near_field: [1, 1, 50], n_piece_far_field: [0, 0, 790],
            g_psi: [1.4295136388338884e-06, 1.4295136393008033e-06, 1.4294854877407043e-06] },
        // on the first side, a quarter of the way along it
        ReferenceCase { i_conductor: 2, r: 0.3258629976580796, z: 0.6797142857142857, n_piece_near_field: [1, 1, 21], n_piece_far_field: [0, 0, 485],
            g_psi: [1.1885725222711374e-06, 1.1885725298071368e-06, 1.188448369539522e-06] },
        // at the reference corner
        ReferenceCase { i_conductor: 2, r: 0.31726463700234186, z: 0.6797142857142857, n_piece_near_field: [1, 1, 5], n_piece_far_field: [0, 0, 296],
            g_psi: [1.0486605785075823e-06, 1.0486606039583502e-06, 1.0483639930158767e-06] },
        // at the corner opposite the reference corner
        ReferenceCase { i_conductor: 2, r: 0.37234192037470726, z: 0.7092857142857143, n_piece_near_field: [1, 1, 10], n_piece_far_field: [0, 0, 268],
            g_psi: [1.1787730694467682e-06, 1.1787730534076749e-06, 1.1788463875801795e-06] },
        // just outside the first side
        ReferenceCase { i_conductor: 2, r: 0.33342716627634655, z: 0.6782357142857143, n_piece_near_field: [1, 1, 24], n_piece_far_field: [0, 0, 581],
            g_psi: [1.2516847336485564e-06, 1.251684733655725e-06, 1.2516528655710967e-06] },
        // 3 half-diagonals from the centre
        ReferenceCase { i_conductor: 2, r: 0.4165230782847982, z: 0.754908753849082, n_piece_near_field: [1, 1, 0], n_piece_far_field: [0, 2, 18],
            g_psi: [7.242960358599697e-07, 7.242959378970325e-07, 7.240568024084041e-07] },
        // far away, at a corner of the grid
        ReferenceCase { i_conductor: 2, r: 1.01, z: -1.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [3.0259103357355514e-08, 3.0259103357609e-08, 3.023897181861787e-08] },
        // near the axis, at the inner edge of the grid
        ReferenceCase { i_conductor: 2, r: 0.01, z: 0.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [5.032238251241837e-11, 5.032238246981758e-11, 5.033166576424754e-11] },
        // Conductor 3: a sheared and rotated parallelogram, whose two sides are in clockwise order
        // at the centre
        ReferenceCase { i_conductor: 3, r: 0.5031848274466822, z: -0.29182984514520416, n_piece_near_field: [1, 1, 46], n_piece_far_field: [0, 0, 316],
            g_psi: [2.5917873162711103e-06, 2.5917873162716245e-06, 2.5916572220129153e-06] },
        // inside, off the centre
        ReferenceCase { i_conductor: 3, r: 0.5066538191594259, z: -0.2870739011600326, n_piece_near_field: [1, 1, 39], n_piece_far_field: [0, 0, 320],
            g_psi: [2.5284542364224785e-06, 2.5284542363593394e-06, 2.5304479689432795e-06] },
        // on the first side, a quarter of the way along it
        ReferenceCase { i_conductor: 3, r: 0.5149024696329647, z: -0.3087439410216266, n_piece_near_field: [1, 1, 14], n_piece_far_field: [0, 0, 207],
            g_psi: [2.1978322918530565e-06, 2.197832292838279e-06, 2.1984418089761564e-06] },
        // at the reference corner
        ReferenceCase { i_conductor: 3, r: 0.52, z: -0.31, n_piece_near_field: [1, 1, 10], n_piece_far_field: [0, 0, 183],
            g_psi: [2.0637223778680813e-06, 2.063722378553487e-06, 2.0644710346293625e-06] },
        // at the corner opposite the reference corner
        ReferenceCase { i_conductor: 3, r: 0.4863696548933645, z: -0.27365969029040826, n_piece_near_field: [1, 1, 8], n_piece_far_field: [0, 0, 189],
            g_psi: [1.9712072547778055e-06, 1.9712072540108405e-06, 1.970094432163752e-06] },
        // just outside the first side
        ReferenceCase { i_conductor: 3, r: 0.510466950447854, z: -0.3090536857330581, n_piece_near_field: [1, 1, 22], n_piece_far_field: [0, 0, 219],
            g_psi: [2.223408237715552e-06, 2.223408238871976e-06, 2.222701019149161e-06] },
        // 3 half-diagonals from the centre
        ReferenceCase { i_conductor: 3, r: 0.5599902133024351, z: -0.24398332869118172, n_piece_near_field: [1, 1, 0], n_piece_far_field: [0, 0, 16],
            g_psi: [1.3688457421562325e-06, 1.3688457173641357e-06, 1.3691594438961622e-06] },
        // far away, at a corner of the grid
        ReferenceCase { i_conductor: 3, r: 1.01, z: -1.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [2.5841270740046455e-07, 2.5841270739701093e-07, 2.5828622856806195e-07] },
        // near the axis, at the inner edge of the grid
        ReferenceCase { i_conductor: 3, r: 0.01, z: 0.0, n_piece_near_field: [0, 0, 0], n_piece_far_field: [1, 1, 1],
            g_psi: [2.540265549896544e-10, 2.5402655498900967e-10, 2.539244690130782e-10] },
    ];

    /// One of the reference values of `documentation/flux_from_parallelogram_conductor/verification/high_precision_reference.json`
    struct HighPrecisionCase {
        /// Which of `HIGH_PRECISION_CONDUCTORS` carries the current
        i_conductor: usize,
        /// R coordinate of the point, [metre]
        r: f64,
        /// Z coordinate of the point, [metre]
        z: f64,
        /// The flux at the point per ampere in the conductor, [henry]
        g_psi: f64,
    }

    /// The four conductors of `documentation/flux_from_parallelogram_conductor/verification/high_precision_reference.json`, each labelled
    /// with its name in that file, with their positions, lengths and inclinations rounded to doubles
    const N_HIGH_PRECISION_CONDUCTOR: usize = 4;
    const HIGH_PRECISION_CONDUCTORS: [ReferenceConductor; N_HIGH_PRECISION_CONDUCTOR] = [
        // 0: `rectangle_grid_cell`, rectangle, the grid cell of the GSFit paper
        ReferenceConductor {
            oblique_r: 0.40375,
            oblique_z: -0.00625,
            oblique_length_alpha: 0.0125,
            oblique_length_beta: 0.0125,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        },
        // 1: `thin_tall_inner_wall`, thin tall element of the inner wall
        ReferenceConductor {
            oblique_r: 0.1536,
            oblique_z: 5e-06,
            oblique_length_alpha: 0.00324,
            oblique_length_beta: 0.02407,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        },
        // 2: `sheared_divertor_plate`, sheared element of a divertor plate
        ReferenceConductor {
            oblique_r: 0.3,
            oblique_z: 0.6,
            oblique_length_alpha: 0.035,
            oblique_length_beta: 0.03,
            oblique_alpha: 0.25,
            oblique_beta: -0.6,
        },
        // 3: `large_sheared_near_axis`, large sheared element close to the axis
        ReferenceConductor {
            oblique_r: 0.17190131270061476,
            oblique_z: -0.025908263021115024,
            oblique_length_alpha: 0.046,
            oblique_length_beta: 0.039,
            oblique_alpha: 0.3,
            oblique_beta: 0.2,
        },
    ];

    /// Thirteen of the 56 reference values of that file, which were calculated to 40 digits by two integrals which do not use the series.
    /// Each is labelled with the names of its conductor and of its sensor in that file. The values are the file's `g_psi_float64_inputs`,
    /// which are for the positions, lengths and inclinations rounded to doubles, and they are rounded to doubles themselves
    const N_HIGH_PRECISION_CASE: usize = 13;
    #[rustfmt::skip]
    const HIGH_PRECISION_CASES: [HighPrecisionCase; N_HIGH_PRECISION_CASE] = [
        // `rectangle_grid_cell`, `centre`
        HighPrecisionCase { i_conductor: 0, r: 0.41, z: 0.0, g_psi: 2.3860812049931994e-06 },
        // `rectangle_grid_cell`, `vertex`
        HighPrecisionCase { i_conductor: 0, r: 0.40375, z: -0.00625, g_psi: 2.0090103180318125e-06 },
        // `rectangle_grid_cell`, `far_grid_point_1`
        HighPrecisionCase { i_conductor: 0, r: 0.01, z: 0.0, g_psi: 4.815328889945259e-10 },
        // `thin_tall_inner_wall`, `edge_mid_point`
        HighPrecisionCase { i_conductor: 1, r: 0.15684, z: 0.01204, g_psi: 6.753413134452516e-07 },
        // `thin_tall_inner_wall`, `just_outside_edge`
        HighPrecisionCase { i_conductor: 1, r: 0.157164, z: 0.0084295, g_psi: 6.601175074135182e-07 },
        // `thin_tall_inner_wall`, `outside_8_half_diagonal_direction_1`
        HighPrecisionCase { i_conductor: 1, r: 0.24647849986481682, z: 0.045351952846136585, g_psi: 2.1861419307565625e-07 },
        // `sheared_divertor_plate`, `interior_off_centre`
        HighPrecisionCase { i_conductor: 2, r: 0.3186398788932703, z: 0.6190638756278849, g_psi: 1.3827421432015072e-06 },
        // `sheared_divertor_plate`, `vertex`
        HighPrecisionCase { i_conductor: 2, r: 0.3, z: 0.6, g_psi: 9.844850957242247e-07 },
        // `sheared_divertor_plate`, `outside_3_half_diagonal_direction_1`
        HighPrecisionCase { i_conductor: 2, r: 0.4111664592500676, z: 0.6480074586724766, g_psi: 6.960490697473502e-07 },
        // `sheared_divertor_plate`, `far_grid_point_2`
        HighPrecisionCase { i_conductor: 2, r: 1.01, z: -1.0, g_psi: 2.993861959970286e-08 },
        // `large_sheared_near_axis`, `centre`
        HighPrecisionCase { i_conductor: 3, r: 0.19, z: 0.0, g_psi: 6.31298032027175e-07 },
        // `large_sheared_near_axis`, `outside_1p5_half_diagonal_direction_2`
        HighPrecisionCase { i_conductor: 3, r: 0.17543066724616685, z: -0.04511137147389276, g_psi: 3.3669904304251556e-07 },
        // `large_sheared_near_axis`, `far_grid_point_1`
        HighPrecisionCase { i_conductor: 3, r: 0.01, z: 0.0, g_psi: 1.0387935398875145e-09 },
    ];

    /// The flux from one conductor at one point
    fn g_psi_at_one_point(
        r: f64,
        z: f64,
        conductor: &ReferenceConductor,
        settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
    ) -> Result<f64, String> {
        let g_psi: Array2<f64> = mutual_inductance_finite_size_to_point_source(
            &array![r],
            &array![z],
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            settings_greens_finite_size,
        )?;

        Ok(g_psi[(0, 0)])
    }

    #[test]
    fn the_flux_agrees_with_the_reference_values_and_takes_the_same_pieces() {
        for i_setting in 0..N_REFERENCE_SETTING {
            let settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[i_setting];
            let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(settings_greens_finite_size).expect("the tolerance is a fraction");
            let coefficients: MaxwellSeriesCoefficients =
                maxwell_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring);

            for i_case in 0..N_REFERENCE_CASE {
                let case: &ReferenceCase = &REFERENCE_CASES[i_case];
                let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[case.i_conductor];
                let greens_psi: GreensPsiParallelogram = greens_psi_parallelogram(
                    &array![case.r],
                    &array![case.z],
                    conductor.oblique_r,
                    conductor.oblique_z,
                    conductor.oblique_length_alpha,
                    conductor.oblique_length_beta,
                    conductor.oblique_alpha,
                    conductor.oblique_beta,
                    &thresholds,
                    &coefficients,
                    settings_greens_finite_size,
                )
                .expect("a series converges for every piece");

                assert_eq!(
                    greens_psi.n_piece_near_field[0], case.n_piece_near_field[i_setting],
                    "setting {i_setting}, case {i_case}: pieces summed with the near-field series"
                );
                assert_eq!(
                    greens_psi.n_piece_far_field[0], case.n_piece_far_field[i_setting],
                    "setting {i_setting}, case {i_case}: pieces summed with the far-field series"
                );
                let relative_difference: f64 = (greens_psi.g_psi[0] / case.g_psi[i_setting] - 1.0).abs();
                assert!(
                    relative_difference < 5.0e-15,
                    "setting {i_setting}, case {i_case}: g_psi = {:e} henry differs from the reference value by {relative_difference:e}",
                    greens_psi.g_psi[0]
                );
            }
        }
    }

    #[test]
    fn the_flux_agrees_with_the_high_precision_values() {
        for i_case in 0..N_HIGH_PRECISION_CASE {
            let case: &HighPrecisionCase = &HIGH_PRECISION_CASES[i_case];
            let conductor: &ReferenceConductor = &HIGH_PRECISION_CONDUCTORS[case.i_conductor];
            let g_psi: f64 = g_psi_at_one_point(case.r, case.z, conductor, &REFERENCE_SETTINGS[0]).expect("a series converges for every piece");

            let relative_difference: f64 = (g_psi / case.g_psi - 1.0).abs();
            assert!(
                relative_difference < 5.0e-15,
                "case {i_case}: g_psi = {g_psi:e} henry differs from the high-precision value by {relative_difference:e}"
            );
        }
    }

    #[test]
    fn each_series_on_its_own_agrees_with_the_reference_values() {
        // The sheared parallelogram, and five points: just outside its left side, 2 centimetre below it, 3.4 half-diagonals from its centre,
        // near the axis, and far away
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[2];
        let n_point: usize = 5;
        let r: [f64; 5] = [0.33, 0.36, 0.45, 0.01, 0.9];
        let z: [f64; 5] = [0.70, 0.66, 0.75, 0.1, -0.5];

        // The near-field series at the first three points, with increasing orders; the first row is the GSFit paper's two approximations
        let n_point_near_field: usize = 3;
        let n_order_near_field: usize = 5;
        let orders_n_elliptic: [usize; 5] = [0, 0, 1, 3, 10];
        let orders_n_thin_ring: [usize; 5] = [0, 1, 0, 5, 20];
        #[rustfmt::skip]
        let g_psi_near_field_reference: [[f64; 3]; 5] = [
            [1.2486832729449505e-06, 1.0404175728532402e-06, 7.99789185173388e-07], // n_elliptic = 0, n_thin_ring = 0
            [1.2832979102386989e-06, 1.0090644100869846e-06, 6.409675631430424e-07], // n_elliptic = 0, n_thin_ring = 1
            [1.249299547459536e-06, 1.042278730519715e-06, 8.107908318144504e-07], // n_elliptic = 1, n_thin_ring = 0
            [1.284254539360502e-06, 1.0129222597131124e-06, 6.599858047378373e-07], // n_elliptic = 3, n_thin_ring = 5
            [1.2842545394222543e-06, 1.0129222597233456e-06, 6.599858250206783e-07], // n_elliptic = 10, n_thin_ring = 20
        ];
        for i_order in 0..n_order_near_field {
            let degree_max: usize = 2 * orders_n_elliptic[i_order] + orders_n_thin_ring[i_order];
            let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(orders_n_elliptic[i_order], orders_n_thin_ring[i_order]);
            let shape: ParallelogramShape = ParallelogramShape::new(
                conductor.oblique_length_alpha,
                conductor.oblique_length_beta,
                conductor.oblique_alpha.cos(),
                conductor.oblique_alpha.sin(),
                -conductor.oblique_beta.sin(),
                conductor.oblique_beta.cos(),
                degree_max,
                0,
            );
            let mut work_arrays: NearFieldWorkArrays = NearFieldWorkArrays::new(degree_max);
            for i_point in 0..n_point_near_field {
                let g_psi: f64 = greens_psi_parallelogram_near_field(
                    r[i_point],
                    z[i_point],
                    conductor.oblique_r,
                    conductor.oblique_z,
                    &shape,
                    &coefficients,
                    &mut work_arrays,
                );
                let relative_difference: f64 = (g_psi / g_psi_near_field_reference[i_order][i_point] - 1.0).abs();
                assert!(
                    relative_difference < 5.0e-15,
                    "near-field series, orders {i_order}, point {i_point}: g_psi = {g_psi:e} henry differs from the reference by {relative_difference:e}"
                );
            }
        }

        // The far-field series at the last four points, with increasing orders; the first row is a point filament
        let n_point_far_field: usize = 4;
        let n_order_far_field: usize = 4;
        let orders_n_multipole: [usize; 4] = [0, 2, 7, 20];
        #[rustfmt::skip]
        let g_psi_far_field_reference: [[f64; 4]; 4] = [
            [1.0284341213034292e-06, 6.568542692178301e-07, 7.228099184950966e-11, 5.436694685042713e-08], // n_multipole = 0
            [1.0107033687649943e-06, 6.599469789310139e-07, 7.226325784728414e-11, 5.439642082931257e-08], // n_multipole = 2
            [1.0127867838387967e-06, 6.599857951647028e-07, 7.226327007163266e-11, 5.439642793730046e-08], // n_multipole = 7
            [1.0129222226603528e-06, 6.599858250206766e-07, 7.226327007173776e-11, 5.439642793730074e-08], // n_multipole = 20
        ];
        let side_alpha_r: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.cos();
        let side_alpha_z: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.sin();
        let side_beta_r: f64 = -conductor.oblique_length_beta * conductor.oblique_beta.sin();
        let side_beta_z: f64 = conductor.oblique_length_beta * conductor.oblique_beta.cos();
        let centre_r: f64 = conductor.oblique_r + 0.5 * (side_alpha_r + side_beta_r);
        let centre_z: f64 = conductor.oblique_z + 0.5 * (side_alpha_z + side_beta_z);
        for i_order in 0..n_order_far_field {
            let n_multipole: usize = orders_n_multipole[i_order];
            let moment: Array2<f64> = parallelogram_moments(side_alpha_r, side_alpha_z, side_beta_r, side_beta_z, n_multipole);
            let mut work_arrays: FarFieldWorkArrays = FarFieldWorkArrays::new(n_multipole);
            // An odd order adds nothing to the even order below it
            let moment_odd: Array2<f64> = parallelogram_moments(side_alpha_r, side_alpha_z, side_beta_r, side_beta_z, n_multipole + 1);
            let mut work_arrays_odd: FarFieldWorkArrays = FarFieldWorkArrays::new(n_multipole + 1);
            for i_point in 0..n_point_far_field {
                let i_point_all: usize = n_point - n_point_far_field + i_point;
                let g_psi: f64 = greens_psi_parallelogram_far_field(r[i_point_all], z[i_point_all], centre_r, centre_z, &moment, n_multipole, &mut work_arrays);
                let relative_difference: f64 = (g_psi / g_psi_far_field_reference[i_order][i_point] - 1.0).abs();
                assert!(
                    relative_difference < 5.0e-15,
                    "far-field series, order {n_multipole}, point {i_point}: g_psi = {g_psi:e} henry differs from the reference by {relative_difference:e}"
                );

                if n_multipole.is_multiple_of(2) {
                    let g_psi_odd: f64 = greens_psi_parallelogram_far_field(
                        r[i_point_all],
                        z[i_point_all],
                        centre_r,
                        centre_z,
                        &moment_odd,
                        n_multipole + 1,
                        &mut work_arrays_odd,
                    );
                    assert_eq!(g_psi_odd, g_psi, "far-field series, order {n_multipole} + 1, point {i_point}");
                }
            }
        }
    }

    #[test]
    fn the_same_parallelogram_described_from_another_corner_gives_the_same_flux() {
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[2];
        let side_alpha_r: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.cos();
        let side_alpha_z: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.sin();
        let side_beta_r: f64 = -conductor.oblique_length_beta * conductor.oblique_beta.sin();
        let side_beta_z: f64 = conductor.oblique_length_beta * conductor.oblique_beta.cos();

        // From the opposite corner, with both sides reversed
        let conductor_reversed: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r + side_alpha_r + side_beta_r,
            oblique_z: conductor.oblique_z + side_alpha_z + side_beta_z,
            oblique_length_alpha: conductor.oblique_length_alpha,
            oblique_length_beta: conductor.oblique_length_beta,
            oblique_alpha: conductor.oblique_alpha + PI,
            oblique_beta: conductor.oblique_beta + PI,
        };
        // From the same corner, with the two sides exchanged, which puts them in clockwise order
        let conductor_exchanged: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r,
            oblique_z: conductor.oblique_z,
            oblique_length_alpha: conductor.oblique_length_beta,
            oblique_length_beta: conductor.oblique_length_alpha,
            oblique_alpha: conductor.oblique_beta + 0.5 * PI,
            oblique_beta: conductor.oblique_alpha - 0.5 * PI,
        };

        for i_setting in 0..N_REFERENCE_SETTING {
            for i_case in 0..N_REFERENCE_CASE {
                let case: &ReferenceCase = &REFERENCE_CASES[i_case];
                if case.i_conductor == 2 {
                    let g_psi: f64 = g_psi_at_one_point(case.r, case.z, conductor, &REFERENCE_SETTINGS[i_setting]).expect("a series converges");
                    let g_psi_reversed: f64 =
                        g_psi_at_one_point(case.r, case.z, &conductor_reversed, &REFERENCE_SETTINGS[i_setting]).expect("a series converges");
                    let g_psi_exchanged: f64 =
                        g_psi_at_one_point(case.r, case.z, &conductor_exchanged, &REFERENCE_SETTINGS[i_setting]).expect("a series converges");

                    assert!(
                        (g_psi_reversed / g_psi - 1.0).abs() < 1.0e-13,
                        "setting {i_setting}, case {i_case}: both sides reversed"
                    );
                    assert!(
                        (g_psi_exchanged / g_psi - 1.0).abs() < 1.0e-13,
                        "setting {i_setting}, case {i_case}: the sides exchanged"
                    );
                }
            }
        }
    }

    #[test]
    fn a_conductor_is_the_mean_of_its_two_halves() {
        // The thin tall rectangle, and its lower and upper halves
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[0];
        let half_lower: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r,
            oblique_z: conductor.oblique_z,
            oblique_length_alpha: conductor.oblique_length_alpha,
            oblique_length_beta: 0.5 * conductor.oblique_length_beta,
            oblique_alpha: conductor.oblique_alpha,
            oblique_beta: conductor.oblique_beta,
        };
        let half_upper: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r,
            oblique_z: conductor.oblique_z + 0.5 * conductor.oblique_length_beta,
            oblique_length_alpha: conductor.oblique_length_alpha,
            oblique_length_beta: 0.5 * conductor.oblique_length_beta,
            oblique_alpha: conductor.oblique_alpha,
            oblique_beta: conductor.oblique_beta,
        };

        for i_case in 0..N_REFERENCE_CASE {
            let case: &ReferenceCase = &REFERENCE_CASES[i_case];
            if case.i_conductor == 0 {
                let g_psi: f64 = g_psi_at_one_point(case.r, case.z, conductor, &REFERENCE_SETTINGS[0]).expect("a series converges");
                let g_psi_lower: f64 = g_psi_at_one_point(case.r, case.z, &half_lower, &REFERENCE_SETTINGS[0]).expect("a series converges");
                let g_psi_upper: f64 = g_psi_at_one_point(case.r, case.z, &half_upper, &REFERENCE_SETTINGS[0]).expect("a series converges");

                // The three are summed with different pieces, so they agree to the relative tolerance of the settings, not to rounding
                assert!((0.5 * (g_psi_lower + g_psi_upper) / g_psi - 1.0).abs() < 1.0e-12, "case {i_case}");
            }
        }
    }

    #[test]
    fn each_conductor_fills_one_column() {
        let n_point: usize = 5;
        let r: Array1<f64> = array![0.155, 0.41, 0.34, 0.9, 0.01];
        let z: Array1<f64> = array![0.01, 0.0, 0.69, -0.5, 0.2];
        let mut oblique_r: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        let mut oblique_z: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        let mut oblique_length_alpha: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        let mut oblique_length_beta: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        let mut oblique_alpha: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        let mut oblique_beta: Array1<f64> = Array1::from_elem(N_REFERENCE_CONDUCTOR, f64::NAN);
        for i_conductor in 0..N_REFERENCE_CONDUCTOR {
            oblique_r[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_r;
            oblique_z[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_z;
            oblique_length_alpha[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_length_alpha;
            oblique_length_beta[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_length_beta;
            oblique_alpha[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_alpha;
            oblique_beta[i_conductor] = REFERENCE_CONDUCTORS[i_conductor].oblique_beta;
        }

        let g_psi: Array2<f64> = mutual_inductance_finite_size_to_point_source(
            &r,
            &z,
            &oblique_r,
            &oblique_z,
            &oblique_length_alpha,
            &oblique_length_beta,
            &oblique_alpha,
            &oblique_beta,
            &REFERENCE_SETTINGS[1],
        )
        .expect("a series converges for every piece");

        assert_eq!(g_psi.dim(), (n_point, N_REFERENCE_CONDUCTOR));
        for i_point in 0..n_point {
            for i_conductor in 0..N_REFERENCE_CONDUCTOR {
                let g_psi_alone: f64 =
                    g_psi_at_one_point(r[i_point], z[i_point], &REFERENCE_CONDUCTORS[i_conductor], &REFERENCE_SETTINGS[1]).expect("a series converges");
                assert_eq!(g_psi[(i_point, i_conductor)], g_psi_alone, "point {i_point}, conductor {i_conductor}");
            }
        }
    }

    #[test]
    fn the_thresholds_are_where_the_first_neglected_term_is_the_tolerance() {
        // Section 5 of the note: with these settings the thresholds are 0.28 and 0.27
        let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(&REFERENCE_SETTINGS[0]).expect("the tolerance is a fraction");
        assert!((thresholds.far_field_ratio_max - 1.0e-12_f64.powf(1.0 / 22.0)).abs() < 1.0e-15);
        assert!((thresholds.near_field_ratio_max - 1.0e-12_f64.powf(1.0 / 21.0)).abs() < 1.0e-15);
        assert!((thresholds.far_field_ratio_max - 0.28).abs() < 0.005);
        assert!((thresholds.near_field_ratio_max - 0.27).abs() < 0.005);

        // An odd order of the far-field series adds nothing to the even order below it
        let thresholds_even: ExpansionParameterThresholds = expansion_parameter_thresholds(&REFERENCE_SETTINGS[1]).expect("the tolerance is a fraction");
        let thresholds_odd: ExpansionParameterThresholds = expansion_parameter_thresholds(&SettingsEquilibriumNumericsGreensFiniteSize {
            n_multipole: 5,
            ..REFERENCE_SETTINGS[1].clone()
        })
        .expect("the tolerance is a fraction");
        assert_eq!(thresholds_odd.far_field_ratio_max, thresholds_even.far_field_ratio_max);
        // The elliptic series is the one which is truncated first here
        assert!((thresholds_even.near_field_ratio_max - 1.0e-4_f64.powf(1.0 / 5.0)).abs() < 1.0e-15);

        // Neither threshold is above 1/2
        let thresholds_loose: ExpansionParameterThresholds = expansion_parameter_thresholds(&SettingsEquilibriumNumericsGreensFiniteSize {
            relative_tolerance: 1.0,
            ..REFERENCE_SETTINGS[0].clone()
        })
        .expect("the tolerance is a fraction");
        assert_eq!(thresholds_loose.far_field_ratio_max, 0.5);
        assert_eq!(thresholds_loose.near_field_ratio_max, 0.5);
    }

    #[test]
    fn a_tolerance_which_is_not_a_fraction_is_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[1];
        let n_tolerance: usize = 4;
        let relative_tolerances: [f64; 4] = [0.0, -1.0e-9, 1.5, f64::NAN];
        for i_tolerance in 0..n_tolerance {
            let settings_greens_finite_size: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
                relative_tolerance: relative_tolerances[i_tolerance],
                ..REFERENCE_SETTINGS[0].clone()
            };

            assert!(expansion_parameter_thresholds(&settings_greens_finite_size).is_err());
            let error: String = g_psi_at_one_point(0.5, 0.1, conductor, &settings_greens_finite_size).expect_err("the tolerance is not a fraction");
            assert!(error.contains("`relative_tolerance`"), "{error}");
        }
    }

    #[test]
    fn a_point_which_is_not_at_a_positive_major_radius_is_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[1];

        assert!(g_psi_at_one_point(0.5, 0.1, conductor, &REFERENCE_SETTINGS[0]).is_ok());
        // On the axis, and beyond it
        assert!(g_psi_at_one_point(0.0, 0.1, conductor, &REFERENCE_SETTINGS[0]).is_err());
        assert!(g_psi_at_one_point(-0.5, 0.1, conductor, &REFERENCE_SETTINGS[0]).is_err());
        // Not a number
        assert!(g_psi_at_one_point(f64::NAN, 0.1, conductor, &REFERENCE_SETTINGS[0]).is_err());
        assert!(g_psi_at_one_point(0.5, f64::INFINITY, conductor, &REFERENCE_SETTINGS[0]).is_err());

        // The error says which point it is
        let error: String = mutual_inductance_finite_size_to_point_source(
            &array![0.5, 0.6, -0.1],
            &array![0.1, 0.1, 0.1],
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            &REFERENCE_SETTINGS[0],
        )
        .expect_err("the last point is beyond the axis");
        assert!(error.contains("point 2"), "{error}");
    }

    #[test]
    fn a_conductor_with_no_area_is_an_error() {
        // A side without a positive length
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.0,
            oblique_length_beta: 0.01,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        let error: String = g_psi_at_one_point(0.5, 0.1, &conductor, &REFERENCE_SETTINGS[0]).expect_err("the first side has no length");
        assert!(error.contains("conductor 0 has no area"), "{error}");
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.01,
            oblique_length_beta: -0.01,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        assert!(g_psi_at_one_point(0.5, 0.1, &conductor, &REFERENCE_SETTINGS[0]).is_err());

        // Two sides which are parallel: the second side is inclined from the Z axis, so a quarter of a turn makes it lie along the first
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.4,
            oblique_z: 0.0,
            oblique_length_alpha: 0.01,
            oblique_length_beta: 0.01,
            oblique_alpha: 0.3,
            oblique_beta: 0.3 - 0.5 * PI,
        };
        let error: String = g_psi_at_one_point(0.5, 0.1, &conductor, &REFERENCE_SETTINGS[0]).expect_err("the two sides are parallel");
        assert!(error.contains("conductor 0 has no area"), "{error}");
    }

    #[test]
    fn a_conductor_which_is_not_wholly_at_a_positive_major_radius_is_an_error() {
        // The reference corner is at a positive major radius, but the first side crosses the axis
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.01,
            oblique_z: 0.0,
            oblique_length_alpha: 0.02,
            oblique_length_beta: 0.01,
            oblique_alpha: PI,
            oblique_beta: 0.0,
        };
        let error: String = g_psi_at_one_point(0.5, 0.1, &conductor, &REFERENCE_SETTINGS[0]).expect_err("the conductor crosses the axis");
        assert!(error.contains("positive major radius"), "{error}");

        // The same conductor, moved outwards until it is clear of the axis
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.03,
            oblique_z: 0.0,
            oblique_length_alpha: 0.02,
            oblique_length_beta: 0.01,
            oblique_alpha: PI,
            oblique_beta: 0.0,
        };
        assert!(g_psi_at_one_point(0.5, 0.1, &conductor, &REFERENCE_SETTINGS[0]).is_ok());
    }

    #[test]
    fn a_conductor_which_may_not_be_halved_often_enough_is_an_error() {
        // With the lowest orders, the centre of the square takes 32 pieces, the smallest of which are made by 8 halvings
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[1];
        let case: &ReferenceCase = &REFERENCE_CASES[9];
        let n_subdivision_needed: usize = 8;

        let settings_enough: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
            n_subdivision_max: n_subdivision_needed,
            ..REFERENCE_SETTINGS[2].clone()
        };
        let g_psi: f64 = g_psi_at_one_point(case.r, case.z, conductor, &settings_enough).expect("8 halvings are enough");
        assert!((g_psi / case.g_psi[2] - 1.0).abs() < 5.0e-15);

        let settings_too_few: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
            n_subdivision_max: n_subdivision_needed - 1,
            ..REFERENCE_SETTINGS[2].clone()
        };
        let error: String = g_psi_at_one_point(case.r, case.z, conductor, &settings_too_few).expect_err("7 halvings are not enough");
        assert!(error.contains("conductor 0: point 0"), "{error}");
        assert!(error.contains("halved 7 time(s)"), "{error}");

        // A point far away needs no halving at all
        let settings_no_halving: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
            n_subdivision_max: 0,
            ..REFERENCE_SETTINGS[2].clone()
        };
        assert!(g_psi_at_one_point(1.01, -1.0, conductor, &settings_no_halving).is_ok());
    }

    #[test]
    fn arrays_of_different_lengths_are_an_error() {
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[1];

        // Two points, with one height
        let result: Result<Array2<f64>, String> = mutual_inductance_finite_size_to_point_source(
            &array![0.5, 0.6],
            &array![0.1],
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            &REFERENCE_SETTINGS[0],
        );
        assert!(result.is_err());

        // One conductor, with two inclinations of its second side
        let result: Result<Array2<f64>, String> = mutual_inductance_finite_size_to_point_source(
            &array![0.5, 0.6],
            &array![0.1, 0.1],
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta, conductor.oblique_beta],
            &REFERENCE_SETTINGS[0],
        );
        assert!(result.is_err());
    }

    /// How long the flux from ST40's passive conductors takes on GSFit's grid, and how many of the pairs of a point and a conductor take
    /// each series. Nothing is checked: it prints what it measures, so it is run on its own, optimised, with `--ignored --nocapture`.
    /// The number of threads is rayon's, which `RAYON_NUM_THREADS` sets
    #[test]
    #[ignore = "a timing, which prints what it measures"]
    fn timing_of_the_flux_from_st40s_passive_conductors_on_the_grid() {
        // ST40's passive conductors as the pf_passive IDS holds them: 8 loops, with 538 elements between them. Each loop holds
        // `"oblique_r": [ ... ]`, and the same for the other five
        let path: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../documentation/flux_from_parallelogram_conductor/code/st40_pf_passive_oblique.json"
        );
        let json: String = std::fs::read_to_string(path).expect("the passive conductors of ST40 can be read");
        let n_key: usize = 6;
        let keys: [&str; 6] = [
            "oblique_r",
            "oblique_z",
            "oblique_length_alpha",
            "oblique_length_beta",
            "oblique_alpha",
            "oblique_beta",
        ];
        let mut keys_values: Vec<Vec<f64>> = vec![Vec::new(); n_key];
        for i_key in 0..n_key {
            let list_opening: String = format!("\"{}\": [", keys[i_key]);
            for (i_list_opening, _) in json.match_indices(&list_opening) {
                let i_list_start: usize = i_list_opening + list_opening.len();
                let i_list_end: usize = i_list_start + json[i_list_start..].find(']').expect("the list is closed");
                for number in json[i_list_start..i_list_end].split(',') {
                    keys_values[i_key].push(number.trim().parse::<f64>().expect("the list holds numbers"));
                }
            }
        }
        let oblique_r: Array1<f64> = Array1::from_vec(keys_values[0].clone());
        let oblique_z: Array1<f64> = Array1::from_vec(keys_values[1].clone());
        let oblique_length_alpha: Array1<f64> = Array1::from_vec(keys_values[2].clone());
        let oblique_length_beta: Array1<f64> = Array1::from_vec(keys_values[3].clone());
        let oblique_alpha: Array1<f64> = Array1::from_vec(keys_values[4].clone());
        let oblique_beta: Array1<f64> = Array1::from_vec(keys_values[5].clone());
        let n_conductor: usize = oblique_r.len();

        // GSFit's grid for ST40
        let n_r: usize = 81;
        let n_z: usize = 161;
        let grid_r: Array1<f64> = Array1::linspace(0.01, 1.01, n_r);
        let grid_z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);
        let n_point: usize = n_r * n_z;
        let mut r: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut z: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                r[i_z * n_r + i_r] = grid_r[i_r];
                z[i_z * n_r + i_r] = grid_z[i_z];
            }
        }
        let n_pair: usize = n_point * n_conductor;
        println!(
            "{n_conductor} conductors, {n_point} points, {n_pair} pairs, {} thread(s)",
            rayon::current_num_threads()
        );

        let n_setting: usize = 2;
        let settings_all: [SettingsEquilibriumNumericsGreensFiniteSize; 2] = [
            SettingsEquilibriumNumericsGreensFiniteSize {
                n_elliptic: 6,
                n_thin_ring: 12,
                n_multipole: 12,
                relative_tolerance: 1.0e-9,
                n_subdivision_max: 30,
            },
            SettingsEquilibriumNumericsGreensFiniteSize {
                n_elliptic: 2,
                n_thin_ring: 4,
                n_multipole: 4,
                relative_tolerance: 1.0e-4,
                n_subdivision_max: 30,
            },
        ];
        for i_setting in 0..n_setting {
            let settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize = &settings_all[i_setting];

            // The fastest of a few repeats, as the machine is shared
            let n_repeat: usize = 3;
            let mut duration_min: f64 = f64::INFINITY;
            let mut g_psi_sum: f64 = f64::NAN;
            for _i_repeat in 0..n_repeat {
                let timing_start: Instant = Instant::now();
                let g_psi: Array2<f64> = mutual_inductance_finite_size_to_point_source(
                    &r,
                    &z,
                    &oblique_r,
                    &oblique_z,
                    &oblique_length_alpha,
                    &oblique_length_beta,
                    &oblique_alpha,
                    &oblique_beta,
                    settings_greens_finite_size,
                )
                .expect("a series converges for every piece");
                duration_min = duration_min.min(timing_start.elapsed().as_secs_f64());
                g_psi_sum = g_psi.sum();
            }

            // Which series each pair takes, which is not timed
            let thresholds: ExpansionParameterThresholds = expansion_parameter_thresholds(settings_greens_finite_size).expect("the tolerance is a fraction");
            let coefficients: MaxwellSeriesCoefficients =
                maxwell_series_coefficients(settings_greens_finite_size.n_elliptic, settings_greens_finite_size.n_thin_ring);
            let mut n_pair_far_field: usize = 0;
            let mut n_pair_near_field: usize = 0;
            let mut n_pair_halved: usize = 0;
            let mut n_piece_near_field: usize = 0;
            let mut n_piece_far_field: usize = 0;
            let mut n_piece_of_one_pair_max: usize = 0;
            for i_conductor in 0..n_conductor {
                let greens_psi: GreensPsiParallelogram = greens_psi_parallelogram(
                    &r,
                    &z,
                    oblique_r[i_conductor],
                    oblique_z[i_conductor],
                    oblique_length_alpha[i_conductor],
                    oblique_length_beta[i_conductor],
                    oblique_alpha[i_conductor],
                    oblique_beta[i_conductor],
                    &thresholds,
                    &coefficients,
                    settings_greens_finite_size,
                )
                .expect("a series converges for every piece");
                for i_point in 0..n_point {
                    let n_piece: usize = greens_psi.n_piece_near_field[i_point] + greens_psi.n_piece_far_field[i_point];
                    if n_piece > 1 {
                        n_pair_halved += 1;
                    } else if greens_psi.n_piece_far_field[i_point] == 1 {
                        n_pair_far_field += 1;
                    } else {
                        n_pair_near_field += 1;
                    }
                    n_piece_near_field += greens_psi.n_piece_near_field[i_point];
                    n_piece_far_field += greens_psi.n_piece_far_field[i_point];
                    n_piece_of_one_pair_max = n_piece_of_one_pair_max.max(n_piece);
                }
            }

            println!(
                "orders ({}, {}, {}), tolerance {:e}: {:.3} second, {:.1} nanosecond for each pair; sum of g_psi = {:.17e} henry",
                settings_greens_finite_size.n_elliptic,
                settings_greens_finite_size.n_thin_ring,
                settings_greens_finite_size.n_multipole,
                settings_greens_finite_size.relative_tolerance,
                duration_min,
                1.0e9 * duration_min / (n_pair as f64),
                g_psi_sum
            );
            println!(
                "    pairs: {n_pair_far_field} far-field series, {n_pair_near_field} near-field series, {n_pair_halved} halved (into at most \
                 {n_piece_of_one_pair_max} pieces); pieces: {n_piece_far_field} far-field series, {n_piece_near_field} near-field series"
            );
        }
    }

    /// A point of the poloidal plane, [metre]
    struct PointRZ {
        r: f64,
        z: f64,
    }

    /// The point `corner + lambda_alpha * side_alpha + lambda_beta * side_beta` of a conductor
    fn point_in_conductor(conductor: &ReferenceConductor, lambda_alpha: f64, lambda_beta: f64) -> PointRZ {
        let side_alpha_r: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.cos();
        let side_alpha_z: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.sin();
        let side_beta_r: f64 = -conductor.oblique_length_beta * conductor.oblique_beta.sin();
        let side_beta_z: f64 = conductor.oblique_length_beta * conductor.oblique_beta.cos();
        PointRZ {
            r: conductor.oblique_r + lambda_alpha * side_alpha_r + lambda_beta * side_beta_r,
            z: conductor.oblique_z + lambda_alpha * side_alpha_z + lambda_beta * side_beta_z,
        }
    }

    /// The flux and its derivatives from one conductor at one point
    fn derivatives_at_one_point(
        r: f64,
        z: f64,
        conductor: &ReferenceConductor,
        settings_greens_finite_size: &SettingsEquilibriumNumericsGreensFiniteSize,
    ) -> Result<PsiDerivatives, String> {
        let tables: FiniteSizeToPointTables = mutual_inductance_finite_size_to_point_source_with_derivatives(
            &array![r],
            &array![z],
            &array![conductor.oblique_r],
            &array![conductor.oblique_z],
            &array![conductor.oblique_length_alpha],
            &array![conductor.oblique_length_beta],
            &array![conductor.oblique_alpha],
            &array![conductor.oblique_beta],
            settings_greens_finite_size,
        )?;

        Ok(PsiDerivatives {
            psi: tables.psi[(0, 0)],
            d_psi_d_r: tables.d_psi_d_r[(0, 0)],
            d_psi_d_z: tables.d_psi_d_z[(0, 0)],
            d2_psi_d_r2: tables.d2_psi_d_r2[(0, 0)],
            d2_psi_d_r_d_z: tables.d2_psi_d_r_d_z[(0, 0)],
            d2_psi_d_z2: tables.d2_psi_d_z2[(0, 0)],
            d3_psi_d_r2_d_z: tables.d3_psi_d_r2_d_z[(0, 0)],
            d3_psi_d_r_d_z2: tables.d3_psi_d_r_d_z2[(0, 0)],
            d3_psi_d_z3: tables.d3_psi_d_z3[(0, 0)],
        })
    }

    /// The number of orders of the flux and its derivatives: the flux, and the first, second and third derivatives
    const N_DERIVATIVE_ORDER: usize = 4;

    /// The largest difference between two sets of the flux and its derivatives in each order, relative to the largest of the reference's
    /// values of that order: the flux, the first derivatives, the second, and the third
    fn relative_differences_by_order(derivatives: &PsiDerivatives, reference: &PsiDerivatives) -> [f64; N_DERIVATIVE_ORDER] {
        let values: [f64; 9] = [
            derivatives.psi,
            derivatives.d_psi_d_r,
            derivatives.d_psi_d_z,
            derivatives.d2_psi_d_r2,
            derivatives.d2_psi_d_r_d_z,
            derivatives.d2_psi_d_z2,
            derivatives.d3_psi_d_r2_d_z,
            derivatives.d3_psi_d_r_d_z2,
            derivatives.d3_psi_d_z3,
        ];
        let reference_values: [f64; 9] = [
            reference.psi,
            reference.d_psi_d_r,
            reference.d_psi_d_z,
            reference.d2_psi_d_r2,
            reference.d2_psi_d_r_d_z,
            reference.d2_psi_d_z2,
            reference.d3_psi_d_r2_d_z,
            reference.d3_psi_d_r_d_z2,
            reference.d3_psi_d_z3,
        ];
        // The first and the last value of each order
        let orders_start: [usize; N_DERIVATIVE_ORDER] = [0, 1, 3, 6];
        let orders_end: [usize; N_DERIVATIVE_ORDER] = [1, 3, 6, 9];

        let mut relative_differences: [f64; N_DERIVATIVE_ORDER] = [f64::NAN; N_DERIVATIVE_ORDER];
        for i_order in 0..N_DERIVATIVE_ORDER {
            let mut scale: f64 = 0.0;
            let mut difference_max: f64 = 0.0;
            for i_value in orders_start[i_order]..orders_end[i_order] {
                scale = scale.max(reference_values[i_value].abs());
                difference_max = difference_max.max((values[i_value] - reference_values[i_value]).abs());
            }
            relative_differences[i_order] = difference_max / scale;
        }

        relative_differences
    }

    /// The settings of ST40's reconstructions, `numerics/greens/finite_size`
    const ST40_SETTINGS: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
        n_elliptic: 6,
        n_thin_ring: 12,
        n_multipole: 12,
        relative_tolerance: 1.0e-9,
        n_subdivision_max: 30,
    };

    /// Settings of higher orders and a smaller tolerance than any of the others, as a reference
    const REFERENCE_SETTINGS_DERIVATIVES: SettingsEquilibriumNumericsGreensFiniteSize = SettingsEquilibriumNumericsGreensFiniteSize {
        n_elliptic: 12,
        n_thin_ring: 24,
        n_multipole: 30,
        relative_tolerance: 1.0e-15,
        n_subdivision_max: 30,
    };

    /// The points of the accuracy tests around a conductor: inside it, at its centre and off it, and outside it, at increasing distances
    /// from its centre in units of its half-diagonal, in eight directions
    fn points_around_conductor(conductor: &ReferenceConductor) -> Vec<PointRZ> {
        let mut points: Vec<PointRZ> = Vec::new();
        points.push(point_in_conductor(conductor, 0.5, 0.5));
        points.push(point_in_conductor(conductor, 0.3, 0.7));
        points.push(point_in_conductor(conductor, 0.9, 0.15));

        let side_alpha_r: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.cos();
        let side_alpha_z: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.sin();
        let side_beta_r: f64 = -conductor.oblique_length_beta * conductor.oblique_beta.sin();
        let side_beta_z: f64 = conductor.oblique_length_beta * conductor.oblique_beta.cos();
        let half_diagonal: f64 = (0.5 * side_alpha_r + 0.5 * side_beta_r)
            .hypot(0.5 * side_alpha_z + 0.5 * side_beta_z)
            .max((0.5 * side_alpha_r - 0.5 * side_beta_r).hypot(0.5 * side_alpha_z - 0.5 * side_beta_z));
        let centre: PointRZ = point_in_conductor(conductor, 0.5, 0.5);

        let n_distance: usize = 12;
        let distances: [f64; 12] = [1.05, 1.3, 1.7, 2.2, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 30.0];
        let n_direction: usize = 8;
        for i_distance in 0..n_distance {
            for i_direction in 0..n_direction {
                let angle: f64 = 0.3 + 2.0 * PI * (i_direction as f64) / (n_direction as f64);
                let r: f64 = centre.r + distances[i_distance] * half_diagonal * angle.cos();
                let z: f64 = centre.z + distances[i_distance] * half_diagonal * angle.sin();
                if r > 0.0 {
                    points.push(PointRZ { r, z });
                }
            }
        }

        points
    }

    /// How close the flux and its derivatives come to their reference values with ST40's settings, around each of the reference conductors
    /// and GSFit's grid cells: printed for each order, and checked against the relative tolerance
    #[test]
    fn the_derivatives_with_st40s_settings_are_within_the_tolerance() {
        let mut conductors: Vec<ReferenceConductor> = Vec::new();
        for i_conductor in 0..N_REFERENCE_CONDUCTOR {
            conductors.push(ReferenceConductor {
                ..REFERENCE_CONDUCTORS[i_conductor]
            });
        }
        for i_conductor in 0..N_HIGH_PRECISION_CONDUCTOR {
            conductors.push(ReferenceConductor {
                ..HIGH_PRECISION_CONDUCTORS[i_conductor]
            });
        }
        // A cell of GSFit's 241 x 481 grid of ST40, and a cell of its 81 x 161 grid next to the axis, which has to be cut
        let cell_small: f64 = 1.0 / 240.0;
        conductors.push(ReferenceConductor {
            oblique_r: 0.5 - 0.5 * cell_small,
            oblique_z: -0.5 * cell_small,
            oblique_length_alpha: cell_small,
            oblique_length_beta: cell_small,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        });
        let cell_large: f64 = 1.0 / 80.0;
        conductors.push(ReferenceConductor {
            oblique_r: 0.01 + 0.5 * cell_large - cell_large,
            oblique_z: -0.5 * cell_large,
            oblique_length_alpha: cell_large,
            oblique_length_beta: cell_large,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        });
        let n_conductor: usize = conductors.len();

        let mut relative_differences_max: [f64; N_DERIVATIVE_ORDER] = [0.0; N_DERIVATIVE_ORDER];
        for i_conductor in 0..n_conductor {
            let mut relative_differences_conductor_max: [f64; N_DERIVATIVE_ORDER] = [0.0; N_DERIVATIVE_ORDER];
            let points: Vec<PointRZ> = points_around_conductor(&conductors[i_conductor]);
            let n_point: usize = points.len();
            for i_point in 0..n_point {
                let derivatives: PsiDerivatives =
                    derivatives_at_one_point(points[i_point].r, points[i_point].z, &conductors[i_conductor], &ST40_SETTINGS).expect("a series converges");
                let reference: PsiDerivatives =
                    derivatives_at_one_point(points[i_point].r, points[i_point].z, &conductors[i_conductor], &REFERENCE_SETTINGS_DERIVATIVES)
                        .expect("a series converges");
                let relative_differences: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&derivatives, &reference);
                for i_order in 0..N_DERIVATIVE_ORDER {
                    relative_differences_conductor_max[i_order] = relative_differences_conductor_max[i_order].max(relative_differences[i_order]);
                }
            }
            println!(
                "conductor {i_conductor}: {n_point} points, largest relative difference of the flux and its first, second and third \
                 derivatives: {:.2e}, {:.2e}, {:.2e}, {:.2e}",
                relative_differences_conductor_max[0],
                relative_differences_conductor_max[1],
                relative_differences_conductor_max[2],
                relative_differences_conductor_max[3]
            );
            for i_order in 0..N_DERIVATIVE_ORDER {
                relative_differences_max[i_order] = relative_differences_max[i_order].max(relative_differences_conductor_max[i_order]);
            }
        }

        for i_order in 0..N_DERIVATIVE_ORDER {
            assert!(
                relative_differences_max[i_order] < ST40_SETTINGS.relative_tolerance,
                "order {i_order}: the largest relative difference from the reference is {:e}",
                relative_differences_max[i_order]
            );
        }
    }

    /// The Grad-Shafranov operator applied to the flux, `d2(psi)/d(r)2 - d(psi)/d(r) / r + d2(psi)/d(z)2`, and its derivative with respect to
    /// z
    struct GradShafranovOperator {
        /// [henry / metre ** 2]
        delta_star: f64,
        /// [henry / metre ** 3]
        delta_star_d_z: f64,
    }

    fn grad_shafranov_operator(r: f64, derivatives: &PsiDerivatives) -> GradShafranovOperator {
        GradShafranovOperator {
            delta_star: derivatives.d2_psi_d_r2 - derivatives.d_psi_d_r / r + derivatives.d2_psi_d_z2,
            delta_star_d_z: derivatives.d3_psi_d_r2_d_z - derivatives.d2_psi_d_r_d_z / r + derivatives.d3_psi_d_z3,
        }
    }

    #[test]
    fn the_derivatives_obey_the_grad_shafranov_equation_inside_and_outside() {
        // A cell of GSFit's 241 x 481 grid of ST40, a cell of its 81 x 161 grid next to the axis, which has to be cut around a point inside
        // it, and the reference conductors
        let cell_small: f64 = 1.0 / 240.0;
        let cell_large: f64 = 1.0 / 80.0;
        let mut conductors: Vec<ReferenceConductor> = vec![
            ReferenceConductor {
                oblique_r: 0.5 - 0.5 * cell_small,
                oblique_z: -0.5 * cell_small,
                oblique_length_alpha: cell_small,
                oblique_length_beta: cell_small,
                oblique_alpha: 0.0,
                oblique_beta: 0.0,
            },
            ReferenceConductor {
                oblique_r: 0.01 - 0.5 * cell_large,
                oblique_z: -0.5 * cell_large,
                oblique_length_alpha: cell_large,
                oblique_length_beta: cell_large,
                oblique_alpha: 0.0,
                oblique_beta: 0.0,
            },
        ];
        for i_conductor in 0..N_REFERENCE_CONDUCTOR {
            conductors.push(ReferenceConductor {
                ..REFERENCE_CONDUCTORS[i_conductor]
            });
        }
        let n_conductor: usize = conductors.len();

        // Inside: the centre, and off it; outside: next to the conductor, and far from it, as fractions of the two sides
        let n_point: usize = 6;
        let points_lambda_alpha: [f64; 6] = [0.5, 0.3, 0.85, 2.0, -0.6, 25.0];
        let points_lambda_beta: [f64; 6] = [0.5, 0.8, 0.1, 0.5, 1.4, -30.0];
        let points_is_inside: [bool; 6] = [true, true, true, false, false, false];

        let n_setting: usize = 4;
        let settings_all: [SettingsEquilibriumNumericsGreensFiniteSize; 4] = [
            ST40_SETTINGS.clone(),
            REFERENCE_SETTINGS[0].clone(),
            REFERENCE_SETTINGS[1].clone(),
            REFERENCE_SETTINGS[2].clone(),
        ];
        for i_setting in 0..n_setting {
            // The low orders satisfy the equation only to their own accuracy
            let accuracy: f64 = (10.0 * settings_all[i_setting].relative_tolerance).max(1.0e-10);
            let mut difference_second_max: f64 = 0.0;
            let mut difference_third_max: f64 = 0.0;
            for i_conductor in 0..n_conductor {
                let conductor: &ReferenceConductor = &conductors[i_conductor];
                let area: f64 = conductor.oblique_length_alpha * conductor.oblique_length_beta * (conductor.oblique_alpha - conductor.oblique_beta).cos().abs();
                for i_point in 0..n_point {
                    let point: PointRZ = point_in_conductor(conductor, points_lambda_alpha[i_point], points_lambda_beta[i_point]);
                    if point.r <= 0.0 {
                        continue;
                    }
                    let derivatives: PsiDerivatives =
                        derivatives_at_one_point(point.r, point.z, conductor, &settings_all[i_setting]).expect("a series converges");
                    let operator: GradShafranovOperator = grad_shafranov_operator(point.r, &derivatives);

                    // Delta* psi = -2 * pi * MU_0 * r * j, with the current density of 1 ampere over the area inside, and zero outside
                    let current_term: f64 = 2.0 * PI * MU_0 * point.r / area;
                    let expected: f64 = if points_is_inside[i_point] { -current_term } else { 0.0 };
                    let scale_second: f64 = derivatives.d2_psi_d_r2.abs().max(derivatives.d2_psi_d_z2.abs()).max(current_term);
                    let scale_third: f64 = derivatives
                        .d3_psi_d_r2_d_z
                        .abs()
                        .max(derivatives.d3_psi_d_r_d_z2.abs())
                        .max(derivatives.d3_psi_d_z3.abs());
                    let difference_second: f64 = (operator.delta_star - expected).abs() / scale_second;
                    let difference_third: f64 = operator.delta_star_d_z.abs() / scale_third;
                    difference_second_max = difference_second_max.max(difference_second);
                    difference_third_max = difference_third_max.max(difference_third);
                    if i_setting == 0 && i_conductor == 0 && i_point == 0 {
                        // The centre of a grid cell, the reason for the derivatives: the current density of the cell itself
                        assert!(
                            (operator.delta_star / expected - 1.0).abs() < 1.0e-11,
                            "the centre of a grid cell: Delta* psi = {:e}, expected {expected:e}",
                            operator.delta_star
                        );
                    }
                    assert!(
                        difference_second < accuracy,
                        "setting {i_setting}, conductor {i_conductor}, point {i_point}: Delta* psi = {:e}, expected {expected:e}, relative \
                         difference {difference_second:e}",
                        operator.delta_star
                    );
                    assert!(
                        difference_third < accuracy,
                        "setting {i_setting}, conductor {i_conductor}, point {i_point}: d(Delta* psi)/d(z) = {:e}, relative to the third \
                         derivatives {difference_third:e}",
                        operator.delta_star_d_z
                    );
                }
            }
            println!(
                "setting {i_setting}: the Grad-Shafranov equation and its z derivative hold to {difference_second_max:.2e} and \
                 {difference_third_max:.2e} of the second and third derivatives"
            );
        }
    }

    #[test]
    fn the_near_and_far_field_series_agree_where_both_converge() {
        // High orders, so that both series converge to rounding
        let n_elliptic: usize = 12;
        let n_thin_ring: usize = 24;
        let n_multipole: usize = 30;
        let degree_max: usize = 2 * n_elliptic + n_thin_ring;
        let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(n_elliptic, n_thin_ring);
        let mut near_field_work_arrays: NearFieldTaylorWorkArrays = NearFieldTaylorWorkArrays::new(degree_max);
        let mut far_field_work_arrays: FarFieldDerivativesWorkArrays = FarFieldDerivativesWorkArrays::new(n_multipole);

        let mut relative_differences_max: [f64; N_DERIVATIVE_ORDER] = [0.0; N_DERIVATIVE_ORDER];
        let mut n_point_compared: usize = 0;
        for i_conductor in 0..N_REFERENCE_CONDUCTOR {
            let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[i_conductor];
            let shape: ParallelogramShape = ParallelogramShape::new(
                conductor.oblique_length_alpha,
                conductor.oblique_length_beta,
                conductor.oblique_alpha.cos(),
                conductor.oblique_alpha.sin(),
                -conductor.oblique_beta.sin(),
                conductor.oblique_beta.cos(),
                degree_max,
                n_multipole,
            );
            let centre: PointRZ = point_in_conductor(conductor, 0.5, 0.5);

            let n_distance: usize = 3;
            let distances: [f64; 3] = [3.0, 4.0, 6.0];
            let n_direction: usize = 5;
            for i_distance in 0..n_distance {
                for i_direction in 0..n_direction {
                    let angle: f64 = 0.7 + 2.0 * PI * (i_direction as f64) / (n_direction as f64);
                    let r: f64 = centre.r + distances[i_distance] * shape.half_diagonal * angle.cos();
                    let z: f64 = centre.z + distances[i_distance] * shape.half_diagonal * angle.sin();

                    // Both series converge to rounding with these orders when their expansion parameters are at most 1/3
                    let distance: f64 = (r - centre.r).hypot(z - centre.z);
                    let t: f64 = (distance + shape.half_diagonal) / (2.0 * r);
                    let is_both_converging: bool = t / (1.0 - t) <= 1.0 / 3.0 && shape.half_diagonal <= distance.min(centre.r) / 3.0;
                    if !is_both_converging {
                        continue;
                    }
                    n_point_compared += 1;

                    let far_field: PsiDerivatives =
                        greens_derivatives_parallelogram_far_field(r, z, centre.r, centre.z, &shape.moment, n_multipole, &mut far_field_work_arrays);
                    let near_field: PsiDerivatives = PsiDerivatives::from_taylor(
                        &greens_derivatives_parallelogram_near_field(
                            r,
                            z,
                            conductor.oblique_r,
                            conductor.oblique_z,
                            &shape,
                            &coefficients,
                            &mut near_field_work_arrays,
                        )
                        .expect("the point is outside"),
                    );
                    let relative_differences: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&far_field, &near_field);
                    for i_order in 0..N_DERIVATIVE_ORDER {
                        relative_differences_max[i_order] = relative_differences_max[i_order].max(relative_differences[i_order]);
                    }
                }
            }
        }

        println!("near- and far-field series at {n_point_compared} points, largest relative difference of each order: {relative_differences_max:?}");
        assert!(n_point_compared >= 30, "{n_point_compared} points compared");
        for i_order in 0..N_DERIVATIVE_ORDER {
            assert!(
                relative_differences_max[i_order] < 1.0e-12,
                "order {i_order}: the two series differ by {:e}",
                relative_differences_max[i_order]
            );
        }
    }

    #[test]
    fn the_constant_terms_are_the_flux_of_each_series() {
        // The constant term of the near-field series with the derivatives is the flux of `greens_psi_parallelogram_near_field`, and the flux
        // of the far-field series with the derivatives is that of `greens_psi_parallelogram_far_field`, to the last bit
        let settings: &SettingsEquilibriumNumericsGreensFiniteSize = &ST40_SETTINGS;
        let degree_max: usize = 2 * settings.n_elliptic + settings.n_thin_ring;
        let coefficients: MaxwellSeriesCoefficients = maxwell_series_coefficients(settings.n_elliptic, settings.n_thin_ring);
        let mut near_field_work_arrays: NearFieldWorkArrays = NearFieldWorkArrays::new(degree_max);
        let mut near_field_taylor_work_arrays: NearFieldTaylorWorkArrays = NearFieldTaylorWorkArrays::new(degree_max);
        let mut far_field_work_arrays: FarFieldWorkArrays = FarFieldWorkArrays::new(settings.n_multipole);
        let mut far_field_derivatives_work_arrays: FarFieldDerivativesWorkArrays = FarFieldDerivativesWorkArrays::new(settings.n_multipole);

        for i_case in 0..N_REFERENCE_CASE {
            let case: &ReferenceCase = &REFERENCE_CASES[i_case];
            let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[case.i_conductor];
            let shape: ParallelogramShape = ParallelogramShape::new(
                conductor.oblique_length_alpha,
                conductor.oblique_length_beta,
                conductor.oblique_alpha.cos(),
                conductor.oblique_alpha.sin(),
                -conductor.oblique_beta.sin(),
                conductor.oblique_beta.cos(),
                degree_max,
                settings.n_multipole,
            );
            let centre: PointRZ = point_in_conductor(conductor, 0.5, 0.5);

            let g_psi_near_field: f64 = greens_psi_parallelogram_near_field(
                case.r,
                case.z,
                conductor.oblique_r,
                conductor.oblique_z,
                &shape,
                &coefficients,
                &mut near_field_work_arrays,
            );
            // A point on an edge or at a corner has no derivatives
            if let Ok(taylor) = greens_derivatives_parallelogram_near_field(
                case.r,
                case.z,
                conductor.oblique_r,
                conductor.oblique_z,
                &shape,
                &coefficients,
                &mut near_field_taylor_work_arrays,
            ) {
                assert_eq!(taylor.value(), g_psi_near_field, "case {i_case}: near-field series");
            }

            // The far-field series converges only away from the conductor, but it gives the same number anywhere
            if (case.r - centre.r).hypot(case.z - centre.z) > shape.half_diagonal {
                let g_psi_far_field: f64 = greens_psi_parallelogram_far_field(
                    case.r,
                    case.z,
                    centre.r,
                    centre.z,
                    &shape.moment,
                    settings.n_multipole,
                    &mut far_field_work_arrays,
                );
                let derivatives: PsiDerivatives = greens_derivatives_parallelogram_far_field(
                    case.r,
                    case.z,
                    centre.r,
                    centre.z,
                    &shape.moment,
                    settings.n_multipole,
                    &mut far_field_derivatives_work_arrays,
                );
                assert_eq!(derivatives.psi, g_psi_far_field, "case {i_case}: far-field series");
            }
        }
    }

    #[test]
    fn the_flux_with_its_derivatives_agrees_with_the_reference_values() {
        for i_setting in 0..N_REFERENCE_SETTING {
            let settings: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[i_setting];
            for i_case in 0..N_REFERENCE_CASE {
                let case: &ReferenceCase = &REFERENCE_CASES[i_case];
                let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[case.i_conductor];
                match derivatives_at_one_point(case.r, case.z, conductor, settings) {
                    Ok(derivatives) => {
                        // The two calculations choose their series with different thresholds, so they agree to the relative tolerance
                        let relative_difference: f64 = (derivatives.psi / case.g_psi[i_setting] - 1.0).abs();
                        assert!(
                            relative_difference < settings.relative_tolerance.max(5.0e-15),
                            "setting {i_setting}, case {i_case}: g_psi = {:e} henry differs from the reference value by {relative_difference:e}",
                            derivatives.psi
                        );
                    }
                    Err(error) => {
                        // Only the points on an edge or at a corner have no derivatives
                        assert!(
                            error.contains("on an edge") || error.contains("at a corner"),
                            "setting {i_setting}, case {i_case}: {error}"
                        );
                        assert!(
                            i_case % 9 == 2 || i_case % 9 == 3 || i_case % 9 == 4,
                            "setting {i_setting}, case {i_case} is not on an edge or at a corner: {error}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_point_on_an_edge_or_at_a_corner_has_no_derivatives() {
        // A rectangle whose corners and edges' mid-points are exact in binary
        let conductor: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.5,
            oblique_z: 0.25,
            oblique_length_alpha: 0.0625,
            oblique_length_beta: 0.125,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        let n_point: usize = 8;
        let r: [f64; 8] = [0.53125, 0.5625, 0.53125, 0.5, 0.5, 0.5625, 0.5625, 0.5];
        let z: [f64; 8] = [0.25, 0.3125, 0.375, 0.3125, 0.25, 0.25, 0.375, 0.375];
        for i_point in 0..n_point {
            let error: String = derivatives_at_one_point(r[i_point], z[i_point], &conductor, &ST40_SETTINGS).expect_err("on an edge or at a corner");
            let message_expected: &str = if i_point < 4 { "on an edge" } else { "at a corner" };
            assert!(error.contains(message_expected), "point {i_point}: {error}");
            assert!(error.contains("conductor 0: point 0"), "point {i_point}: {error}");
        }

        // Just inside an edge, the derivatives are those of the inside
        let derivatives: PsiDerivatives = derivatives_at_one_point(0.53125, 0.25 + 1.0e-9, &conductor, &ST40_SETTINGS).expect("inside");
        let operator: GradShafranovOperator = grad_shafranov_operator(0.53125, &derivatives);
        let current_term: f64 = 2.0 * PI * MU_0 * 0.53125 / (0.0625 * 0.125);
        assert!((operator.delta_star / -current_term - 1.0).abs() < 1.0e-6);

        // A conductor next to the axis, which has to be cut for a point on its edge
        let conductor_near_axis: ReferenceConductor = ReferenceConductor {
            oblique_r: 0.03125,
            oblique_z: 0.0,
            oblique_length_alpha: 0.0625,
            oblique_length_beta: 0.0625,
            oblique_alpha: 0.0,
            oblique_beta: 0.0,
        };
        let error: String = derivatives_at_one_point(0.03125, 0.03125, &conductor_near_axis, &ST40_SETTINGS).expect_err("on an edge");
        assert!(error.contains("on an edge") || error.contains("at a corner"), "{error}");
        assert!(derivatives_at_one_point(0.0625, 0.03125, &conductor_near_axis, &ST40_SETTINGS).is_ok());
    }

    #[test]
    fn the_derivatives_agree_with_finite_differences() {
        let settings: &SettingsEquilibriumNumericsGreensFiniteSize = &REFERENCE_SETTINGS[0];

        let n_point: usize = 5;
        let points_lambda_alpha: [f64; 5] = [0.5, 0.3, 1.3, 3.5, f64::NAN];
        let points_lambda_beta: [f64; 5] = [0.5, 0.7, 0.4, 0.5, f64::NAN];

        let mut relative_differences_max: [f64; N_DERIVATIVE_ORDER] = [0.0; N_DERIVATIVE_ORDER];
        for i_conductor in 0..N_REFERENCE_CONDUCTOR {
            let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[i_conductor];
            let size: f64 = conductor.oblique_length_alpha.min(conductor.oblique_length_beta);
            for i_point in 0..n_point {
                // The last point is far away, at a corner of GSFit's grid
                let point: PointRZ = if i_point == n_point - 1 {
                    PointRZ { r: 1.01, z: -1.0 }
                } else {
                    point_in_conductor(conductor, points_lambda_alpha[i_point], points_lambda_beta[i_point])
                };
                let step: f64 = if i_point == n_point - 1 { 1.0e-3 } else { 1.0e-3 * size };

                // Richardson's extrapolation of the central differences with the steps `step` and `step / 2`
                let n_step: usize = 2;
                let steps: [f64; 2] = [step, 0.5 * step];
                let mut differences_r: Vec<PsiDerivatives> = Vec::with_capacity(n_step);
                let mut differences_z: Vec<PsiDerivatives> = Vec::with_capacity(n_step);
                for i_step in 0..n_step {
                    let plus_r: PsiDerivatives = derivatives_at_one_point(point.r + steps[i_step], point.z, conductor, settings).expect("a series converges");
                    let minus_r: PsiDerivatives = derivatives_at_one_point(point.r - steps[i_step], point.z, conductor, settings).expect("a series converges");
                    let plus_z: PsiDerivatives = derivatives_at_one_point(point.r, point.z + steps[i_step], conductor, settings).expect("a series converges");
                    let minus_z: PsiDerivatives = derivatives_at_one_point(point.r, point.z - steps[i_step], conductor, settings).expect("a series converges");
                    let mut difference_r: PsiDerivatives = PsiDerivatives::zero();
                    difference_r.add_scaled(&plus_r, 0.5 / steps[i_step]);
                    difference_r.add_scaled(&minus_r, -0.5 / steps[i_step]);
                    let mut difference_z: PsiDerivatives = PsiDerivatives::zero();
                    difference_z.add_scaled(&plus_z, 0.5 / steps[i_step]);
                    difference_z.add_scaled(&minus_z, -0.5 / steps[i_step]);
                    differences_r.push(difference_r);
                    differences_z.push(difference_z);
                }
                let mut difference_r: PsiDerivatives = PsiDerivatives::zero();
                difference_r.add_scaled(&differences_r[1], 4.0 / 3.0);
                difference_r.add_scaled(&differences_r[0], -1.0 / 3.0);
                let mut difference_z: PsiDerivatives = PsiDerivatives::zero();
                difference_z.add_scaled(&differences_z[1], 4.0 / 3.0);
                difference_z.add_scaled(&differences_z[0], -1.0 / 3.0);

                // Each derivative from the differences of the one below it
                let derivatives: PsiDerivatives = derivatives_at_one_point(point.r, point.z, conductor, settings).expect("a series converges");
                let finite_differences: PsiDerivatives = PsiDerivatives {
                    psi: derivatives.psi,
                    d_psi_d_r: difference_r.psi,
                    d_psi_d_z: difference_z.psi,
                    d2_psi_d_r2: difference_r.d_psi_d_r,
                    d2_psi_d_r_d_z: difference_z.d_psi_d_r,
                    d2_psi_d_z2: difference_z.d_psi_d_z,
                    d3_psi_d_r2_d_z: difference_z.d2_psi_d_r2,
                    d3_psi_d_r_d_z2: difference_z.d2_psi_d_r_d_z,
                    d3_psi_d_z3: difference_z.d2_psi_d_z2,
                };
                let relative_differences: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&finite_differences, &derivatives);
                for i_order in 0..N_DERIVATIVE_ORDER {
                    relative_differences_max[i_order] = relative_differences_max[i_order].max(relative_differences[i_order]);
                }

                // The mixed derivative, from the other order of differentiation
                let relative_difference_mixed: f64 =
                    (difference_r.d_psi_d_z - derivatives.d2_psi_d_r_d_z).abs() / derivatives.d2_psi_d_r2.abs().max(derivatives.d2_psi_d_z2.abs());
                assert!(
                    relative_difference_mixed < 1.0e-8,
                    "conductor {i_conductor}, point {i_point}: {relative_difference_mixed:e}"
                );
            }
        }

        println!("finite differences, largest relative difference of each order: {relative_differences_max:?}");
        for i_order in 1..N_DERIVATIVE_ORDER {
            assert!(
                relative_differences_max[i_order] < 1.0e-8,
                "order {i_order}: the finite differences differ by {:e}",
                relative_differences_max[i_order]
            );
        }
    }

    #[test]
    fn a_point_filament_has_the_derivatives_of_the_reference_values() {
        // The far-field series of the order 0 is a point filament at the centre. The reference values are the derivatives of the flux of a
        // filament at (0.6, 0.1) metre, calculated to 40 digits by differentiating its closed form numerically, and rounded to doubles,
        // written with as many digits as it takes to read each back as the same double; in the order psi, d_psi_d_r, d_psi_d_z,
        // d2_psi_d_r2, d2_psi_d_r_d_z, d2_psi_d_z2, d3_psi_d_r2_d_z, d3_psi_d_r_d_z2, d3_psi_d_z3
        let n_multipole: usize = 0;
        let centre_r: f64 = 0.6;
        let centre_z: f64 = 0.1;
        let moment: Array2<f64> = parallelogram_moments(0.01, 0.0, 0.0, 0.02, n_multipole);
        let mut work_arrays: FarFieldDerivativesWorkArrays = FarFieldDerivativesWorkArrays::new(n_multipole);

        let n_point: usize = 5;
        let r: Array1<f64> = array![0.3, 1.0, 0.62, 0.2, 0.61];
        let z: Array1<f64> = array![0.5, -0.4, 0.15, 0.1, 0.1];
        #[rustfmt::skip]
        let references: [[f64; 9]; 5] = [
            [1.6297441225072212e-07, 1.0250312172541893e-06, -4.205063424202711e-07, 2.453018402571386e-06, -2.9093969976069093e-06,
                9.637523216092447e-07, -9.978006081767494e-06, 9.226527885862826e-06, 2.8001608974446265e-07],
            [4.965771153311033e-07, -5.522718302581204e-08, 7.782454889522146e-07, -9.971159559454939e-07, -1.3634908472975107e-06,
                9.41888772919682e-07, 5.057402363148e-07, -6.137796425960764e-06, -1.8692310836123106e-06],
            [1.925794165120443e-06, -3.0646011433094684e-06, -1.307283009777211e-05, -0.00019849099045753503, 0.00017126200254359686,
                0.00019354808538768104, 0.004378824099234785, -0.008767884048277679, -0.004102595062874146],
            [1.3749243300780325e-07, 1.4384183161956136e-06, 0.0, 8.5604506230732e-06, 0.0, -1.368359042095132e-06, 0.0,
                -1.6906069319571707e-05, 0.0],
            [3.179587785498572e-06, -7.275325940650193e-05, 0.0, 0.0074789608129866144, 0.0, -0.007598228451357929, 0.0,
                1.5141667473631273, 0.0],
        ];
        let greens: PointToPoint = PointToPoint::new(
            r.clone(),
            z.clone(),
            array![centre_r],
            array![centre_z],
            array![f64::NAN],
            array![f64::NAN],
            false,
        );
        let greens_psi: Array2<f64> = greens.psi();
        let greens_d_psi_d_r: Array2<f64> = greens.d_psi_d_r();
        let greens_d_psi_d_z: Array2<f64> = greens.d_psi_d_z();
        let greens_d2_psi_d_r2: Array2<f64> = greens.d2_psi_d_r2();
        let greens_d2_psi_d_r_d_z: Array2<f64> = greens.d2_psi_d_r_d_z();
        let greens_d2_psi_d_z2: Array2<f64> = greens.d2_psi_d_z2();
        let greens_d3_psi_d_r2_d_z: Array2<f64> = greens.d3_psi_d_r2_d_z();
        let greens_d3_psi_d_r_d_z2: Array2<f64> = greens.d3_psi_d_r_d_z2();
        let greens_d3_psi_d_z3: Array2<f64> = greens.d3_psi_d_z3();

        for i_point in 0..n_point {
            let reference: PsiDerivatives = PsiDerivatives {
                psi: references[i_point][0],
                d_psi_d_r: references[i_point][1],
                d_psi_d_z: references[i_point][2],
                d2_psi_d_r2: references[i_point][3],
                d2_psi_d_r_d_z: references[i_point][4],
                d2_psi_d_z2: references[i_point][5],
                d3_psi_d_r2_d_z: references[i_point][6],
                d3_psi_d_r_d_z2: references[i_point][7],
                d3_psi_d_z3: references[i_point][8],
            };
            let derivatives: PsiDerivatives =
                greens_derivatives_parallelogram_far_field(r[i_point], z[i_point], centre_r, centre_z, &moment, n_multipole, &mut work_arrays);
            let relative_differences: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&derivatives, &reference);

            // `Greens` writes the derivatives in closed form, which loses some digits of the third derivatives close to the filament
            let filament: PsiDerivatives = PsiDerivatives {
                psi: greens_psi[(i_point, 0)],
                d_psi_d_r: greens_d_psi_d_r[(i_point, 0)],
                d_psi_d_z: greens_d_psi_d_z[(i_point, 0)],
                d2_psi_d_r2: greens_d2_psi_d_r2[(i_point, 0)],
                d2_psi_d_r_d_z: greens_d2_psi_d_r_d_z[(i_point, 0)],
                d2_psi_d_z2: greens_d2_psi_d_z2[(i_point, 0)],
                d3_psi_d_r2_d_z: greens_d3_psi_d_r2_d_z[(i_point, 0)],
                d3_psi_d_r_d_z2: greens_d3_psi_d_r_d_z2[(i_point, 0)],
                d3_psi_d_z3: greens_d3_psi_d_z3[(i_point, 0)],
            };
            let relative_differences_greens: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&filament, &reference);
            for i_order in 0..N_DERIVATIVE_ORDER {
                assert!(
                    relative_differences[i_order] < 1.0e-13,
                    "point {i_point}, order {i_order}: the point filament differs from the reference by {:e}",
                    relative_differences[i_order]
                );
                assert!(
                    relative_differences_greens[i_order] < 1.0e-10,
                    "point {i_point}, order {i_order}: `Greens` differs from the reference by {:e}",
                    relative_differences_greens[i_order]
                );
            }
        }

        // Far away, a conductor of finite size differs from the filament at its centre by about the square of its size over the distance
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[1];
        let conductor_centre: PointRZ = point_in_conductor(conductor, 0.5, 0.5);
        let derivatives: PsiDerivatives = derivatives_at_one_point(1.01, -1.0, conductor, &ST40_SETTINGS).expect("a series converges");
        let greens_far: PointToPoint = PointToPoint::new(
            array![1.01],
            array![-1.0],
            array![conductor_centre.r],
            array![conductor_centre.z],
            array![f64::NAN],
            array![f64::NAN],
            false,
        );
        let filament_far: PsiDerivatives = PsiDerivatives {
            psi: greens_far.psi()[(0, 0)],
            d_psi_d_r: greens_far.d_psi_d_r()[(0, 0)],
            d_psi_d_z: greens_far.d_psi_d_z()[(0, 0)],
            d2_psi_d_r2: greens_far.d2_psi_d_r2()[(0, 0)],
            d2_psi_d_r_d_z: greens_far.d2_psi_d_r_d_z()[(0, 0)],
            d2_psi_d_z2: greens_far.d2_psi_d_z2()[(0, 0)],
            d3_psi_d_r2_d_z: greens_far.d3_psi_d_r2_d_z()[(0, 0)],
            d3_psi_d_r_d_z2: greens_far.d3_psi_d_r_d_z2()[(0, 0)],
            d3_psi_d_z3: greens_far.d3_psi_d_z3()[(0, 0)],
        };
        let relative_differences: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&derivatives, &filament_far);
        for i_order in 0..N_DERIVATIVE_ORDER {
            assert!(relative_differences[i_order] < 1.0e-3, "order {i_order}: {:e}", relative_differences[i_order]);
            assert!(relative_differences[i_order] > 1.0e-8, "order {i_order}: {:e}", relative_differences[i_order]);
        }
    }

    #[test]
    fn the_same_parallelogram_described_from_another_corner_gives_the_same_derivatives() {
        let conductor: &ReferenceConductor = &REFERENCE_CONDUCTORS[2];
        let side_alpha_r: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.cos();
        let side_alpha_z: f64 = conductor.oblique_length_alpha * conductor.oblique_alpha.sin();
        let side_beta_r: f64 = -conductor.oblique_length_beta * conductor.oblique_beta.sin();
        let side_beta_z: f64 = conductor.oblique_length_beta * conductor.oblique_beta.cos();

        // From the opposite corner, with both sides reversed, and from the same corner, with the two sides exchanged
        let conductor_reversed: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r + side_alpha_r + side_beta_r,
            oblique_z: conductor.oblique_z + side_alpha_z + side_beta_z,
            oblique_length_alpha: conductor.oblique_length_alpha,
            oblique_length_beta: conductor.oblique_length_beta,
            oblique_alpha: conductor.oblique_alpha + PI,
            oblique_beta: conductor.oblique_beta + PI,
        };
        let conductor_exchanged: ReferenceConductor = ReferenceConductor {
            oblique_r: conductor.oblique_r,
            oblique_z: conductor.oblique_z,
            oblique_length_alpha: conductor.oblique_length_beta,
            oblique_length_beta: conductor.oblique_length_alpha,
            oblique_alpha: conductor.oblique_beta + 0.5 * PI,
            oblique_beta: conductor.oblique_alpha - 0.5 * PI,
        };

        let points: Vec<PointRZ> = points_around_conductor(conductor);
        let n_point: usize = points.len();
        for i_point in 0..n_point {
            let derivatives: PsiDerivatives = derivatives_at_one_point(points[i_point].r, points[i_point].z, conductor, &ST40_SETTINGS).expect("converges");
            let derivatives_reversed: PsiDerivatives =
                derivatives_at_one_point(points[i_point].r, points[i_point].z, &conductor_reversed, &ST40_SETTINGS).expect("converges");
            let derivatives_exchanged: PsiDerivatives =
                derivatives_at_one_point(points[i_point].r, points[i_point].z, &conductor_exchanged, &ST40_SETTINGS).expect("converges");
            let relative_differences_reversed: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&derivatives_reversed, &derivatives);
            let relative_differences_exchanged: [f64; N_DERIVATIVE_ORDER] = relative_differences_by_order(&derivatives_exchanged, &derivatives);
            for i_order in 0..N_DERIVATIVE_ORDER {
                assert!(
                    relative_differences_reversed[i_order] < 1.0e-9,
                    "point {i_point}, order {i_order}: both sides reversed"
                );
                assert!(
                    relative_differences_exchanged[i_order] < 1.0e-9,
                    "point {i_point}, order {i_order}: the sides exchanged"
                );
            }
        }
    }

    /// The grid points of GSFit's grid of ST40: `n_r * n_z` points on `0.01 <= r <= 1.01` metre and `-1 <= z <= 1` metre, row by row
    fn st40_grid(n_r: usize, n_z: usize) -> Vec<Array1<f64>> {
        let grid_r: Array1<f64> = Array1::linspace(0.01, 1.01, n_r);
        let grid_z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);
        let n_point: usize = n_r * n_z;
        let mut r: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut z: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                r[i_z * n_r + i_r] = grid_r[i_r];
                z[i_z * n_r + i_r] = grid_z[i_z];
            }
        }
        vec![r, z]
    }

    /// How long the flux and its derivatives from ST40's passive conductors take on GSFit's 81 x 161 grid, with ST40's settings, against
    /// the flux alone. Nothing is checked: it prints what it measures, so it is run on its own, optimised, with `--ignored --nocapture`
    #[test]
    #[ignore = "a timing, which prints what it measures"]
    fn timing_of_the_derivatives_from_st40s_passive_conductors_on_the_grid() {
        let path: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../documentation/flux_from_parallelogram_conductor/code/st40_pf_passive_oblique.json"
        );
        let json: String = std::fs::read_to_string(path).expect("the passive conductors of ST40 can be read");
        let n_key: usize = 6;
        let keys: [&str; 6] = [
            "oblique_r",
            "oblique_z",
            "oblique_length_alpha",
            "oblique_length_beta",
            "oblique_alpha",
            "oblique_beta",
        ];
        let mut keys_values: Vec<Vec<f64>> = vec![Vec::new(); n_key];
        for i_key in 0..n_key {
            let list_opening: String = format!("\"{}\": [", keys[i_key]);
            for (i_list_opening, _) in json.match_indices(&list_opening) {
                let i_list_start: usize = i_list_opening + list_opening.len();
                let i_list_end: usize = i_list_start + json[i_list_start..].find(']').expect("the list is closed");
                for number in json[i_list_start..i_list_end].split(',') {
                    keys_values[i_key].push(number.trim().parse::<f64>().expect("the list holds numbers"));
                }
            }
        }
        let oblique_r: Array1<f64> = Array1::from_vec(keys_values[0].clone());
        let oblique_z: Array1<f64> = Array1::from_vec(keys_values[1].clone());
        let oblique_length_alpha: Array1<f64> = Array1::from_vec(keys_values[2].clone());
        let oblique_length_beta: Array1<f64> = Array1::from_vec(keys_values[3].clone());
        let oblique_alpha: Array1<f64> = Array1::from_vec(keys_values[4].clone());
        let oblique_beta: Array1<f64> = Array1::from_vec(keys_values[5].clone());
        let n_conductor: usize = oblique_r.len();

        let grid: Vec<Array1<f64>> = st40_grid(81, 161);
        let n_pair: usize = grid[0].len() * n_conductor;
        println!(
            "{n_conductor} conductors, {} points, {n_pair} pairs, {} thread(s)",
            grid[0].len(),
            rayon::current_num_threads()
        );

        let timing_start: Instant = Instant::now();
        let g_psi: Array2<f64> = mutual_inductance_finite_size_to_point_source(
            &grid[0],
            &grid[1],
            &oblique_r,
            &oblique_z,
            &oblique_length_alpha,
            &oblique_length_beta,
            &oblique_alpha,
            &oblique_beta,
            &ST40_SETTINGS,
        )
        .expect("a series converges for every piece");
        let duration_psi: f64 = timing_start.elapsed().as_secs_f64();

        let timing_start: Instant = Instant::now();
        let tables: FiniteSizeToPointTables = mutual_inductance_finite_size_to_point_source_with_derivatives(
            &grid[0],
            &grid[1],
            &oblique_r,
            &oblique_z,
            &oblique_length_alpha,
            &oblique_length_beta,
            &oblique_alpha,
            &oblique_beta,
            &ST40_SETTINGS,
        )
        .expect("a series converges for every piece");
        let duration_derivatives: f64 = timing_start.elapsed().as_secs_f64();

        let relative_difference_max: f64 = ((&tables.psi - &g_psi) / &g_psi)
            .mapv(f64::abs)
            .fold(0.0, |maximum: f64, value: &f64| maximum.max(*value));
        println!(
            "flux alone: {duration_psi:.3} second; flux and derivatives: {duration_derivatives:.3} second, {:.1} nanosecond for each pair; the two \
             fluxes differ by at most {relative_difference_max:.2e}",
            1.0e9 * duration_derivatives / (n_pair as f64)
        );
    }

    /// How long the flux and its derivatives take on GSFit's 241 x 481 grid of ST40, from the grid's own cells, with ST40's settings. GSFit
    /// calculates the table of the grid with every point and the cells of the lowest row only, as a cell's table depends only on the
    /// height of the point above it. Nothing is checked: it prints what it measures, so it is run on its own, optimised, with
    /// `--ignored --nocapture`
    #[test]
    #[ignore = "a timing, which prints what it measures"]
    fn timing_of_the_derivatives_on_gsfits_plasma_grid() {
        let n_r: usize = 241;
        let n_z: usize = 481;
        let grid: Vec<Array1<f64>> = st40_grid(n_r, n_z);
        let grid_r: Array1<f64> = Array1::linspace(0.01, 1.01, n_r);
        let d_r: f64 = 1.0 / ((n_r - 1) as f64);
        let d_z: f64 = 2.0 / ((n_z - 1) as f64);

        // The cells of the lowest row, centred on its points
        let oblique_r: Array1<f64> = &grid_r - 0.5 * d_r;
        let oblique_z: Array1<f64> = Array1::from_elem(n_r, -1.0 - 0.5 * d_z);
        let oblique_length_alpha: Array1<f64> = Array1::from_elem(n_r, d_r);
        let oblique_length_beta: Array1<f64> = Array1::from_elem(n_r, d_z);
        let oblique_alpha: Array1<f64> = Array1::zeros(n_r);
        let oblique_beta: Array1<f64> = Array1::zeros(n_r);
        let n_pair: usize = grid[0].len() * n_r;
        println!(
            "{n_r} cells, {} points, {n_pair} pairs, {} thread(s)",
            grid[0].len(),
            rayon::current_num_threads()
        );

        let timing_start: Instant = Instant::now();
        let tables: FiniteSizeToPointTables = mutual_inductance_finite_size_to_point_source_with_derivatives(
            &grid[0],
            &grid[1],
            &oblique_r,
            &oblique_z,
            &oblique_length_alpha,
            &oblique_length_beta,
            &oblique_alpha,
            &oblique_beta,
            &ST40_SETTINGS,
        )
        .expect("a series converges for every piece");
        let duration: f64 = timing_start.elapsed().as_secs_f64();
        println!(
            "flux and derivatives: {duration:.3} second, {:.1} nanosecond for each pair; sum of d2_psi_d_z2 = {:.6e} henry / metre ** 2",
            1.0e9 * duration / (n_pair as f64),
            tables.d2_psi_d_z2.sum()
        );
    }
}
