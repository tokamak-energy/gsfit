//! `time_slice(itime)/boundary/minor_radius`, `.../geometric_axis/r`, `.../geometric_axis/z`,
//! `.../elongation`, `.../triangularity`, `.../triangularity_lower`, `.../triangularity_upper`,
//! `.../squareness_lower_inner`, `.../squareness_lower_outer`, `.../squareness_upper_inner` and
//! `.../squareness_upper_outer`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use ndarray_stats::QuantileExt;
#[cfg(test)]
use std::f64::consts::PI;

/// Calculate the scalar geometry of the plasma boundary, and store it in the time-slice.
///
/// Every quantity here is a property of the boundary contour alone, which is why they are filled
/// together rather than one file per data dictionary path.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the `boundary` scalars listed above are written into it
///
/// **Must run after `epp_equilibrium_time_slice_boundary_outline`**, which supplies the contour.
pub fn epp_equilibrium_time_slice_boundary_geometry(time_slice: &mut EquilibriumTimeSlice) {
    // A slice which did not converge has no boundary to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.boundary.minor_radius = Some(f64::NAN);
        time_slice.boundary.geometric_axis.r = Some(f64::NAN);
        time_slice.boundary.geometric_axis.z = Some(f64::NAN);
        time_slice.boundary.elongation = Some(f64::NAN);
        time_slice.boundary.triangularity = Some(f64::NAN);
        time_slice.boundary.triangularity_lower = Some(f64::NAN);
        time_slice.boundary.triangularity_upper = Some(f64::NAN);
        time_slice.boundary.squareness_lower_inner = Some(f64::NAN);
        time_slice.boundary.squareness_lower_outer = Some(f64::NAN);
        time_slice.boundary.squareness_upper_inner = Some(f64::NAN);
        time_slice.boundary.squareness_upper_outer = Some(f64::NAN);
        return;
    }

    let boundary_r: &Array1<f64> = time_slice.boundary.outline.r.as_ref().unwrap();
    let boundary_z: &Array1<f64> = time_slice.boundary.outline.z.as_ref().unwrap();

    // Minor radius
    let r_minor: f64 = (boundary_r.max().unwrap().to_owned() - boundary_r.min().unwrap().to_owned()) / 2.0;
    // Geometric radius
    let r_geo: f64 = (boundary_r.max().unwrap().to_owned() + boundary_r.min().unwrap().to_owned()) / 2.0;
    // Geometric radius
    let z_geo: f64 = (boundary_z.max().unwrap().to_owned() + boundary_z.min().unwrap().to_owned()) / 2.0;

    // Boundary shape: elongation, triangularity and squareness
    let (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o): (f64, f64, f64, f64, f64, f64, f64, f64) =
        epp_boundary_geometry(boundary_r, boundary_z);

    time_slice.boundary.minor_radius = Some(r_minor);
    time_slice.boundary.geometric_axis.r = Some(r_geo);
    time_slice.boundary.geometric_axis.z = Some(z_geo);
    time_slice.boundary.elongation = Some(elongation);
    time_slice.boundary.triangularity = Some(triang);
    time_slice.boundary.triangularity_lower = Some(triang_l);
    time_slice.boundary.triangularity_upper = Some(triang_u);
    time_slice.boundary.squareness_lower_inner = Some(square_l_i);
    time_slice.boundary.squareness_lower_outer = Some(square_l_o);
    time_slice.boundary.squareness_upper_inner = Some(square_u_i);
    time_slice.boundary.squareness_upper_outer = Some(square_u_o);
}

/// Calculate the shape of the plasma boundary, following the IMAS definitions:
/// https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids/equilibrium.html
///
/// The squareness follows the definition from: T.C. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009
///
/// # Arguments
/// * `boundary_r` - radial coordinates of the plasma boundary contour [metre]
/// * `boundary_z` - vertical coordinates of the plasma boundary contour [metre]
///
/// # Returns
/// * `elongation` - elongation of the plasma boundary [dimensionless]
/// * `triang` - average triangularity [dimensionless]
/// * `triang_l` - lower triangularity [dimensionless]
/// * `triang_u` - upper triangularity [dimensionless]
/// * `square_l_i` - lower inner squareness [dimensionless]
/// * `square_l_o` - lower outer squareness [dimensionless]
/// * `square_u_i` - upper inner squareness [dimensionless]
/// * `square_u_o` - upper outer squareness [dimensionless]
fn epp_boundary_geometry(boundary_r: &Array1<f64>, boundary_z: &Array1<f64>) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let nan_result: (f64, f64, f64, f64, f64, f64, f64, f64) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);

    // Defensive programming: when a time-slice has failed the boundary contour can be
    // empty, or contain NAN's or junk; return NAN's instead of panicking
    let n_boundary: usize = boundary_r.len();
    if n_boundary < 4 || boundary_z.len() != n_boundary {
        return nan_result;
    }
    let all_finite: bool = boundary_r.iter().chain(boundary_z.iter()).all(|value| value.is_finite());
    if !all_finite {
        return nan_result;
    }

    // Extremal points of the boundary contour
    // (`argmax` and `argmin` cannot fail, since the contour is non-empty and finite)
    let i_r_max: usize = boundary_r.argmax().unwrap(); // outboard point
    let i_r_min: usize = boundary_r.argmin().unwrap(); // inboard point
    let i_z_max: usize = boundary_z.argmax().unwrap(); // top point
    let i_z_min: usize = boundary_z.argmin().unwrap(); // bottom point

    let r_max: f64 = boundary_r[i_r_max];
    let z_at_r_max: f64 = boundary_z[i_r_max];
    let r_min: f64 = boundary_r[i_r_min];
    let z_at_r_min: f64 = boundary_z[i_r_min];
    let z_max: f64 = boundary_z[i_z_max];
    let r_at_z_max: f64 = boundary_r[i_z_max];
    let z_min: f64 = boundary_z[i_z_min];
    let r_at_z_min: f64 = boundary_r[i_z_min];

    // Degenerate contour, with zero width or height
    if (r_max - r_min) < 10.0 * f64::EPSILON || (z_max - z_min) < 10.0 * f64::EPSILON {
        return nan_result;
    }

    // Minor radius
    let r_minor: f64 = (r_max - r_min) / 2.0;
    // Geometric radius
    let r_geo: f64 = (r_max + r_min) / 2.0;

    // Elongation
    let elongation: f64 = (z_max - z_min) / (2.0 * r_minor);

    // Triangularity
    let triang_u: f64 = (r_geo - r_at_z_max) / r_minor;
    let triang_l: f64 = (r_geo - r_at_z_min) / r_minor;
    let triang: f64 = (triang_u + triang_l) / 2.0;

    // Squareness for each quadrant.
    // Each quadrant has a bounding box spanned by two extremal points of the boundary,
    // e.g. the upper outer quadrant is spanned by the top point and the outboard point.
    // The quadrant "centre" is `(r, z) = (r_of_the_top_or_bottom_point, z_of_the_inboard_or_outboard_point)`
    // and the quadrant "corner" is the opposite corner of the bounding box.
    let square_u_o: f64 = epp_squareness(boundary_r, boundary_z, r_at_z_max, z_at_r_max, r_max, z_max);
    let square_u_i: f64 = epp_squareness(boundary_r, boundary_z, r_at_z_max, z_at_r_min, r_min, z_max);
    let square_l_o: f64 = epp_squareness(boundary_r, boundary_z, r_at_z_min, z_at_r_max, r_max, z_min);
    let square_l_i: f64 = epp_squareness(boundary_r, boundary_z, r_at_z_min, z_at_r_min, r_min, z_min);

    return (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o);
}

/// Calculate the squareness of one quadrant of the plasma boundary, using the
/// definition from: T.C. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009
///
/// The squareness measures where the boundary crosses the diagonal from the quadrant
/// "centre" `D` to the bounding box "corner" `C`, relative to where an ellipse through the two
/// extremal points would cross it (an ellipse crosses the diagonal at `1 / sqrt(2)` of its length):
/// `squareness = 0` for an ellipse, `1` for a rectangle, and `< 0` for a more pointed shape
///
/// # Arguments
/// * `boundary_r` - radial coordinates of the plasma boundary contour [metre]
/// * `boundary_z` - vertical coordinates of the plasma boundary contour [metre]
/// * `centre_r`, `centre_z` - quadrant centre `D` [metre]
/// * `corner_r`, `corner_z` - quadrant bounding box corner `C` [metre]
///
/// # Returns
/// * `squareness` - squareness of the quadrant [dimensionless]
fn epp_squareness(boundary_r: &Array1<f64>, boundary_z: &Array1<f64>, centre_r: f64, centre_z: f64, corner_r: f64, corner_z: f64) -> f64 {
    let n_boundary: usize = boundary_r.len();

    // Defensive programming: when a time-slice has failed the boundary contour can be
    // empty, or contain NAN's or junk; return NAN instead of panicking
    if n_boundary == 0 || boundary_z.len() != n_boundary {
        return f64::NAN;
    }
    if !centre_r.is_finite() || !centre_z.is_finite() || !corner_r.is_finite() || !corner_z.is_finite() {
        return f64::NAN;
    }

    // Diagonal from the quadrant centre `D` to the bounding box corner `C`
    let diag_r: f64 = corner_r - centre_r;
    let diag_z: f64 = corner_z - centre_z;

    // Degenerate quadrant, e.g. when the top point coincides with the outboard point
    if diag_r.abs() < 10.0 * f64::EPSILON || diag_z.abs() < 10.0 * f64::EPSILON {
        return f64::NAN;
    }

    // Find where the boundary crosses the diagonal, as a fraction `t_boundary` of the diagonal length.
    // Solved as a segment-segment intersection:
    // `D + t * (C - D) = P1 + u * (P2 - P1)` with `t` and `u` both in `[0, 1]`
    let mut t_boundary: f64 = f64::NAN;
    for i_point in 0..n_boundary {
        // Include the wrap-around segment, in case the contour is not closed
        let i_next: usize = (i_point + 1) % n_boundary;
        let segment_r: f64 = boundary_r[i_next] - boundary_r[i_point];
        let segment_z: f64 = boundary_z[i_next] - boundary_z[i_point];

        let denominator: f64 = diag_r * segment_z - diag_z * segment_r;
        // Skip segments which are parallel to the diagonal
        if denominator.abs() < 10.0 * f64::EPSILON {
            continue;
        }

        let t: f64 = ((boundary_r[i_point] - centre_r) * segment_z - (boundary_z[i_point] - centre_z) * segment_r) / denominator;
        let u: f64 = ((boundary_r[i_point] - centre_r) * diag_z - (boundary_z[i_point] - centre_z) * diag_r) / denominator;

        // Keep the crossing which is furthest from the quadrant centre
        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 && (t_boundary.is_nan() || t > t_boundary) {
            t_boundary = t;
        }
    }

    if t_boundary.is_nan() {
        return f64::NAN;
    }

    // An ellipse through the two extremal points crosses the diagonal at `1 / sqrt(2)` of its length
    let t_ellipse: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let squareness: f64 = (t_boundary - t_ellipse) / (1.0 - t_ellipse);
    return squareness;
}

#[test]
fn test_epp_boundary_geometry_ellipse() {
    use approx::assert_abs_diff_eq;

    // An ellipse has zero triangularity and zero squareness
    let r_geo: f64 = 2.5;
    let z_geo: f64 = 0.1;
    let r_minor: f64 = 1.0;
    let kappa: f64 = 1.8;

    // `n_theta` divisible by 4, so that the extremal points are exactly on the contour
    let n_theta: usize = 1000;
    let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);
    let boundary_r: Array1<f64> = r_geo + r_minor * theta.mapv(f64::cos);
    let boundary_z: Array1<f64> = z_geo + kappa * r_minor * theta.mapv(f64::sin);

    let (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o) = epp_boundary_geometry(&boundary_r, &boundary_z);

    assert_abs_diff_eq!(elongation, kappa, epsilon = 1e-6);
    assert_abs_diff_eq!(triang, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_l, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_u, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(square_l_i, 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(square_l_o, 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(square_u_i, 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(square_u_o, 0.0, epsilon = 1e-3);
}

#[test]
fn test_epp_boundary_geometry_miller() {
    use approx::assert_abs_diff_eq;

    // Miller parameterisation: `r = r_geo + r_minor * cos(theta + arcsin(delta) * sin(theta))`
    // The top point is at `theta = pi / 2`, where `r = r_geo - r_minor * delta`, so `triang = delta` exactly
    let r_geo: f64 = 0.9;
    let z_geo: f64 = 0.0;
    let r_minor: f64 = 0.6;
    let kappa: f64 = 2.2;
    let delta: f64 = 0.4;

    let n_theta: usize = 1000;
    let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);
    let boundary_r: Array1<f64> = r_geo + r_minor * theta.mapv(|theta_local| (theta_local + delta.asin() * theta_local.sin()).cos());
    let boundary_z: Array1<f64> = z_geo + kappa * r_minor * theta.mapv(f64::sin);

    let (elongation, triang, triang_l, triang_u, _square_l_i, _square_l_o, _square_u_i, _square_u_o) = epp_boundary_geometry(&boundary_r, &boundary_z);

    assert_abs_diff_eq!(elongation, kappa, epsilon = 1e-6);
    assert_abs_diff_eq!(triang, delta, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_l, delta, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_u, delta, epsilon = 1e-6);
}

#[test]
fn test_epp_boundary_geometry_superellipse() {
    use approx::assert_abs_diff_eq;

    // Superellipse: `|(r - r_geo) / r_minor| ** n_exponent + |(z - z_geo) / (kappa * r_minor)| ** n_exponent = 1`
    // The boundary crosses the quadrant diagonal at `t = (1 / 2) ** (1 / n_exponent)` of its length,
    // giving an analytic squareness: `(t - 1 / sqrt(2)) / (1 - 1 / sqrt(2))`
    // Note: `n_exponent = 2` is an ellipse (squareness = 0); `n_exponent = infinity` is a rectangle (squareness = 1)
    let r_geo: f64 = 2.0;
    let z_geo: f64 = -0.2;
    let r_minor: f64 = 0.8;
    let kappa: f64 = 1.5;
    let n_exponent: f64 = 10.0;

    let n_theta: usize = 1000;
    let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);
    let boundary_r: Array1<f64> = r_geo + r_minor * theta.mapv(|theta_local| theta_local.cos().signum() * theta_local.cos().abs().powf(2.0 / n_exponent));
    let boundary_z: Array1<f64> =
        z_geo + kappa * r_minor * theta.mapv(|theta_local| theta_local.sin().signum() * theta_local.sin().abs().powf(2.0 / n_exponent));

    let (elongation, triang, _triang_l, _triang_u, square_l_i, square_l_o, square_u_i, square_u_o) = epp_boundary_geometry(&boundary_r, &boundary_z);

    let t_expected: f64 = (0.5f64).powf(1.0 / n_exponent);
    let squareness_expected: f64 = (t_expected - std::f64::consts::FRAC_1_SQRT_2) / (1.0 - std::f64::consts::FRAC_1_SQRT_2);

    // Note: the tolerances are looser than the other tests because the `powf(2.0 / n_exponent)`
    // amplifies the floating point error in `cos(pi / 2)` near the extremal points
    assert_abs_diff_eq!(elongation, kappa, epsilon = 1e-3);
    assert_abs_diff_eq!(triang, 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(square_l_i, squareness_expected, epsilon = 1e-3);
    assert_abs_diff_eq!(square_l_o, squareness_expected, epsilon = 1e-3);
    assert_abs_diff_eq!(square_u_i, squareness_expected, epsilon = 1e-3);
    assert_abs_diff_eq!(square_u_o, squareness_expected, epsilon = 1e-3);
}
