//! `time_slice(itime)/boundary/minor_radius`, `.../geometric_axis/r`, `.../geometric_axis/z`,
//! `.../elongation`, `.../triangularity`, `.../triangularity_lower`, `.../triangularity_upper`,
//! `.../squareness_lower_inner`, `.../squareness_lower_outer`, `.../squareness_upper_inner` and
//! `.../squareness_upper_outer`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::{PsiDerivatives, PsiInterpolator};
use imas_rs::EquilibriumTimeSlice;
use imas_rs::ids::equilibrium::EquilibriumContourTreeNode;
use ndarray::Array1;
use ndarray_stats::QuantileExt;
#[cfg(test)]
use std::f64::consts::PI;

/// The `contour_tree/node/critical_type` of a saddle point, an X-point
const CRITICAL_TYPE_SADDLE: i32 = 1;

/// An extremal point of a contour closer than this to an X-point, as a fraction of a grid cell, is
/// that X-point [dimensionless]
const X_POINT_TOLERANCE: f64 = 1.0e-6;

/// The Newton iteration which refines the top or the bottom of a contour has converged once its
/// step is smaller than this fraction of a grid cell [dimensionless]
const NEWTON_TOLERANCE: f64 = 1.0e-10;

/// The most Newton iterations taken to refine the top or the bottom of a contour
const N_NEWTON_ITER_MAX: usize = 50;

/// The Newton iteration's Jacobian is singular once the sine of the angle between its two rows is
/// smaller than this [dimensionless]
const JACOBIAN_SINGULAR_TOLERANCE: f64 = 1.0e-12;

/// Calculate the scalar geometry of the plasma boundary, and store it in the time-slice.
///
/// Every quantity here describes the shape of the boundary contour, which is why they are filled
/// together rather than one file per data dictionary path.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the `boundary` scalars listed above are written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let boundary_r: &Array1<f64> = &time_slice.boundary.outline.r;
    let boundary_z: &Array1<f64> = &time_slice.boundary.outline.z;
    let psi_b: f64 = time_slice.boundary.psi;

    // Minor radius
    let r_minor: f64 = (boundary_r.max().unwrap().to_owned() - boundary_r.min().unwrap().to_owned()) / 2.0;
    // Geometric radius
    let r_geo: f64 = (boundary_r.max().unwrap().to_owned() + boundary_r.min().unwrap().to_owned()) / 2.0;
    // Geometric radius
    let z_geo: f64 = (boundary_z.max().unwrap().to_owned() + boundary_z.min().unwrap().to_owned()) / 2.0;

    // Boundary shape: elongation, triangularity and squareness.
    // The top and the bottom of the boundary are refined onto `psi_b`; see `boundary_geometry`
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
    let (xpts_r, xpts_z): (Array1<f64>, Array1<f64>) = x_points(time_slice);
    let (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o): (f64, f64, f64, f64, f64, f64, f64, f64) =
        boundary_geometry(boundary_r, boundary_z, psi_b, &psi_interpolator, &xpts_r, &xpts_z);

    time_slice.boundary.minor_radius = r_minor;
    time_slice.boundary.geometric_axis.r = r_geo;
    time_slice.boundary.geometric_axis.z = z_geo;
    time_slice.boundary.elongation = elongation;
    time_slice.boundary.triangularity = triang;
    time_slice.boundary.triangularity_lower = triang_l;
    time_slice.boundary.triangularity_upper = triang_u;
    time_slice.boundary.squareness_lower_inner = square_l_i;
    time_slice.boundary.squareness_lower_outer = square_l_o;
    time_slice.boundary.squareness_upper_inner = square_u_i;
    time_slice.boundary.squareness_upper_outer = square_u_o;
}

/// The X-points of a time-slice: the saddle points of `psi` which the solver found, from
/// `contour_tree`.
///
/// `marching_squares` puts the X-point a diverted boundary is bounded by onto the contour with
/// exactly these coordinates, which is how `boundary_geometry` recognises it.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// * `xpts_r` - radial coordinate of each X-point [metre]
/// * `xpts_z` - vertical coordinate of each X-point [metre]
pub(in crate::equilibrium_post_processor) fn x_points(time_slice: &EquilibriumTimeSlice) -> (Array1<f64>, Array1<f64>) {
    let n_node: usize = time_slice.contour_tree.node.len();

    let mut xpts_r: Vec<f64> = Vec::with_capacity(n_node);
    let mut xpts_z: Vec<f64> = Vec::with_capacity(n_node);
    for i_node in 0..n_node {
        let node: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_node];
        if node.critical_type == CRITICAL_TYPE_SADDLE {
            xpts_r.push(node.r);
            xpts_z.push(node.z);
        }
    }

    (Array1::from_vec(xpts_r), Array1::from_vec(xpts_z))
}

/// Calculate the shape of a closed poloidal contour, following the IMAS definitions:
/// https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids/equilibrium.html
///
/// The squareness follows the definition from: T.C. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009
///
/// The contour is the plasma boundary here, and one flux surface in
/// `profiles_1d::elongation__squareness__triangularity::calculate`. The data dictionary
/// defines the `boundary` scalars and the `profiles_1d` profiles the same way, so both go through
/// this one function and the last point of each profile is the corresponding boundary scalar.
///
/// # The top and the bottom of the contour
///
/// The contour's vertices are where it crosses the grid lines. Around its top the contour runs
/// almost horizontally, so the vertices there are on the vertical grid lines, and the highest of them
/// can be up to half a cell from the top of the flux surface. Taken from the vertices, `R_Zmax` is
/// quantised by the grid, and the upper triangularity with it, in steps of `d_r / a`: steps of 0.04
/// on ST40's 12.5 mm grid. So the radius of the top, and likewise of the bottom, is refined onto
/// the flux surface itself, using the bicubic interpolation of `psi`; see `refine_top_or_bottom`.
///
/// An X-point is not refined. The contour has a corner there, not a horizontal top, and the X-point
/// is already found to sub-grid accuracy, by `find_stationary_points_using_winding_number`, and put
/// onto the contour by `marching_squares`.
///
/// Everything else is taken from the vertices. `R_min`, `R_max`, `Z_min` and `Z_max` are each
/// resolved by the grid lines the contour crosses there, so their error is only second order: a
/// vertex `delta` from the extremum of a surface with curvature `kappa` is `kappa * delta ** 2 / 2`
/// short of it, at most 0.2 mm on ST40's grid. Keeping them is what keeps the `minor_radius`, the
/// `geometric_axis` and the `elongation` the extent of the `outline`. The heights of the inboard and
/// the outboard points are quantised by the horizontal grid lines in the same way `R_Zmax` was by
/// the vertical ones, but they only enter the squareness, and are taken from the vertices too.
///
/// # Arguments
/// * `boundary_r` - radial coordinates of the closed contour [metre]
/// * `boundary_z` - vertical coordinates of the closed contour [metre]
/// * `psi_contour` - the poloidal flux on the contour, which its top and bottom are refined onto [weber]
/// * `psi_interpolator` - the bicubic interpolation of `psi`
/// * `xpts_r`, `xpts_z` - the X-points, which are not refined; see `x_points` [metre]
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
pub(in crate::equilibrium_post_processor) fn boundary_geometry(
    boundary_r: &Array1<f64>,
    boundary_z: &Array1<f64>,
    psi_contour: f64,
    psi_interpolator: &PsiInterpolator,
    xpts_r: &Array1<f64>,
    xpts_z: &Array1<f64>,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let nan_result: (f64, f64, f64, f64, f64, f64, f64, f64) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);

    let Some(mut extremal_points) = extremal_points(boundary_r, boundary_z) else {
        return nan_result;
    };

    let d_r: f64 = psi_interpolator.d_r;
    let d_z: f64 = psi_interpolator.d_z;

    // The top and the bottom, refined onto the flux surface, unless either is an X-point.
    // When a refinement fails the vertex is kept, which is the best estimate there is
    if !is_x_point(extremal_points.r_at_z_max, extremal_points.z_max, xpts_r, xpts_z, d_r, d_z)
        && let Some((top_r, _top_z)) = refine_top_or_bottom(extremal_points.r_at_z_max, extremal_points.z_max, psi_contour, psi_interpolator)
    {
        extremal_points.r_at_z_max = top_r;
    }
    if !is_x_point(extremal_points.r_at_z_min, extremal_points.z_min, xpts_r, xpts_z, d_r, d_z)
        && let Some((bottom_r, _bottom_z)) = refine_top_or_bottom(extremal_points.r_at_z_min, extremal_points.z_min, psi_contour, psi_interpolator)
    {
        extremal_points.r_at_z_min = bottom_r;
    }

    shape_from_extremal_points(boundary_r, boundary_z, &extremal_points)
}

/// Where a closed contour touches each side of its bounding box.
struct ExtremalPoints {
    /// Radial coordinate of the top point, `R_Zmax` [metre]
    r_at_z_max: f64,
    /// Radial coordinate of the bottom point, `R_Zmin` [metre]
    r_at_z_min: f64,
    /// Radial coordinate of the outboard point [metre]
    r_max: f64,
    /// Radial coordinate of the inboard point [metre]
    r_min: f64,
    /// Vertical coordinate of the outboard point [metre]
    z_at_r_max: f64,
    /// Vertical coordinate of the inboard point [metre]
    z_at_r_min: f64,
    /// Vertical coordinate of the top point [metre]
    z_max: f64,
    /// Vertical coordinate of the bottom point [metre]
    z_min: f64,
}

/// The extremal points of a closed contour's vertices.
///
/// # Arguments
/// * `boundary_r` - radial coordinates of the closed contour [metre]
/// * `boundary_z` - vertical coordinates of the closed contour [metre]
///
/// # Returns
/// * `extremal_points` - or `None` for a contour which cannot be measured
fn extremal_points(boundary_r: &Array1<f64>, boundary_z: &Array1<f64>) -> Option<ExtremalPoints> {
    // Defensive programming: when a time-slice has failed the boundary contour can be
    // empty, or contain NAN's or junk; return `None` instead of panicking
    let n_boundary: usize = boundary_r.len();
    if n_boundary < 4 || boundary_z.len() != n_boundary {
        return None;
    }
    let all_finite: bool = boundary_r.iter().chain(boundary_z.iter()).all(|value| value.is_finite());
    if !all_finite {
        return None;
    }

    // Extremal points of the boundary contour
    // (`argmax` and `argmin` cannot fail, since the contour is non-empty and finite)
    let i_r_max: usize = boundary_r.argmax().unwrap(); // outboard point
    let i_r_min: usize = boundary_r.argmin().unwrap(); // inboard point
    let i_z_max: usize = boundary_z.argmax().unwrap(); // top point
    let i_z_min: usize = boundary_z.argmin().unwrap(); // bottom point

    let extremal_points: ExtremalPoints = ExtremalPoints {
        r_at_z_max: boundary_r[i_z_max],
        r_at_z_min: boundary_r[i_z_min],
        r_max: boundary_r[i_r_max],
        r_min: boundary_r[i_r_min],
        z_at_r_max: boundary_z[i_r_max],
        z_at_r_min: boundary_z[i_r_min],
        z_max: boundary_z[i_z_max],
        z_min: boundary_z[i_z_min],
    };

    // Degenerate contour, with zero width or height
    if (extremal_points.r_max - extremal_points.r_min) < 10.0 * f64::EPSILON || (extremal_points.z_max - extremal_points.z_min) < 10.0 * f64::EPSILON {
        return None;
    }

    Some(extremal_points)
}

/// Whether a point is one of the X-points.
///
/// # Arguments
/// * `point_r`, `point_z` - the point [metre]
/// * `xpts_r`, `xpts_z` - the X-points [metre]
/// * `d_r`, `d_z` - the grid spacing, which `X_POINT_TOLERANCE` is a fraction of [metre]
fn is_x_point(point_r: f64, point_z: f64, xpts_r: &Array1<f64>, xpts_z: &Array1<f64>, d_r: f64, d_z: f64) -> bool {
    let n_xpt: usize = xpts_r.len();
    for i_xpt in 0..n_xpt {
        let distance_in_cells: f64 = ((point_r - xpts_r[i_xpt]) / d_r).hypot((point_z - xpts_z[i_xpt]) / d_z);
        if distance_in_cells < X_POINT_TOLERANCE {
            return true;
        }
    }

    false
}

/// Refine the top, or the bottom, of a closed flux surface onto the flux surface itself.
///
/// The top of the surface `psi = psi_contour` is where the surface is horizontal, `d(psi)/d(r) = 0`.
/// Newton's method solves the two conditions together, starting from the highest vertex:
///
/// ```text
/// [ d(psi)/d(r)      d(psi)/d(z)      ] [ delta_r ]       [ psi - psi_contour ]
/// [ d2(psi)/d(r)2    d2(psi)/d(r)d(z) ] [ delta_z ]  =  - [ d(psi)/d(r)       ]
/// ```
///
/// `psi` is the bicubic of `psi`, along whose grid lines `marching_squares` finds the vertices, so
/// the refined point is on the same surface as the vertices. `d(psi)/d(r)`, and the derivatives of it
/// in the second row, are the bicubic of `d(psi)/d(r)` itself rather than the derivative of the
/// bicubic of `psi`: the X-points are found the same way; see `PsiInterpolator`.
///
/// The bottom is the surface's other horizontal point, and is found the same way, from the lowest
/// vertex.
///
/// # Arguments
/// * `start_r`, `start_z` - the highest, or the lowest, vertex of the contour [metre]
/// * `psi_contour` - the poloidal flux on the contour [weber]
/// * `psi_interpolator` - the bicubic interpolation of `psi`
///
/// # Returns
/// * `(refined_r, refined_z)` - the top, or the bottom, of the flux surface [metre]
///
/// `None` when Newton's method does not converge onto a point within a grid cell of the vertex,
/// where the top has to be, as the vertex is the highest of the vertices.
fn refine_top_or_bottom(start_r: f64, start_z: f64, psi_contour: f64, psi_interpolator: &PsiInterpolator) -> Option<(f64, f64)> {
    if !start_r.is_finite() || !start_z.is_finite() || !psi_contour.is_finite() {
        return None;
    }

    let d_r: f64 = psi_interpolator.d_r;
    let d_z: f64 = psi_interpolator.d_z;

    let mut refined_r: f64 = start_r;
    let mut refined_z: f64 = start_z;
    for _i_iter in 0..N_NEWTON_ITER_MAX {
        // `None` once the iteration has left the grid
        let psi: f64 = psi_interpolator.psi(refined_r, refined_z)?;
        let psi_derivatives: PsiDerivatives = psi_interpolator.derivatives(refined_r, refined_z)?;
        let d_psi_d_r: f64 = psi_derivatives.d_psi_d_r;
        let d_psi_d_z: f64 = psi_derivatives.d_psi_d_z;
        let d2_psi_d_r2: f64 = psi_derivatives.d2_psi_d_r2;
        let d2_psi_d_r_d_z: f64 = psi_derivatives.d2_psi_d_r_d_z;

        // The two rows of the Jacobian carry different units, so whether they are parallel is judged
        // scale-free, by the sine of the angle between them. They are when the top of the surface is
        // flat, `d2(psi)/d(r)2 = 0`, or at a null of `psi`, `grad(psi) = 0`
        let jacobian_determinant: f64 = d_psi_d_r * d2_psi_d_r_d_z - d_psi_d_z * d2_psi_d_r2;
        let jacobian_row_norms: f64 = d_psi_d_r.hypot(d_psi_d_z) * d2_psi_d_r2.hypot(d2_psi_d_r_d_z);
        if !jacobian_determinant.is_finite() || jacobian_determinant.abs() <= JACOBIAN_SINGULAR_TOLERANCE * jacobian_row_norms {
            return None;
        }

        // Solve for the Newton step by Cramer's rule
        let psi_residual: f64 = psi - psi_contour;
        let delta_r: f64 = (-psi_residual * d2_psi_d_r_d_z + d_psi_d_r * d_psi_d_z) / jacobian_determinant;
        let delta_z: f64 = (-d_psi_d_r * d_psi_d_r + psi_residual * d2_psi_d_r2) / jacobian_determinant;
        refined_r += delta_r;
        refined_z += delta_z;

        if delta_r.abs() <= NEWTON_TOLERANCE * d_r && delta_z.abs() <= NEWTON_TOLERANCE * d_z {
            let is_near_vertex: bool = (refined_r - start_r).abs() <= d_r && (refined_z - start_z).abs() <= d_z;
            if is_near_vertex {
                return Some((refined_r, refined_z));
            }
            return None;
        }
    }

    // Newton's method did not converge
    None
}

/// The shape of a closed contour, from its extremal points; see `boundary_geometry`.
///
/// # Arguments
/// * `boundary_r` - radial coordinates of the closed contour [metre]
/// * `boundary_z` - vertical coordinates of the closed contour [metre]
/// * `extremal_points` - where the contour touches each side of its bounding box
///
/// # Returns
/// The same eight quantities as `boundary_geometry` [dimensionless]
fn shape_from_extremal_points(
    boundary_r: &Array1<f64>,
    boundary_z: &Array1<f64>,
    extremal_points: &ExtremalPoints,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let r_at_z_max: f64 = extremal_points.r_at_z_max;
    let r_at_z_min: f64 = extremal_points.r_at_z_min;
    let r_max: f64 = extremal_points.r_max;
    let r_min: f64 = extremal_points.r_min;
    let z_at_r_max: f64 = extremal_points.z_at_r_max;
    let z_at_r_min: f64 = extremal_points.z_at_r_min;
    let z_max: f64 = extremal_points.z_max;
    let z_min: f64 = extremal_points.z_min;

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
    let square_u_o: f64 = squareness(boundary_r, boundary_z, r_at_z_max, z_at_r_max, r_max, z_max);
    let square_u_i: f64 = squareness(boundary_r, boundary_z, r_at_z_max, z_at_r_min, r_min, z_max);
    let square_l_o: f64 = squareness(boundary_r, boundary_z, r_at_z_min, z_at_r_max, r_max, z_min);
    let square_l_i: f64 = squareness(boundary_r, boundary_z, r_at_z_min, z_at_r_min, r_min, z_min);

    (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o)
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
fn squareness(boundary_r: &Array1<f64>, boundary_z: &Array1<f64>, centre_r: f64, centre_z: f64, corner_r: f64, corner_z: f64) -> f64 {
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
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) && (t_boundary.is_nan() || t > t_boundary) {
            t_boundary = t;
        }
    }

    if t_boundary.is_nan() {
        return f64::NAN;
    }

    // An ellipse through the two extremal points crosses the diagonal at `1 / sqrt(2)` of its length
    let t_ellipse: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let squareness: f64 = (t_boundary - t_ellipse) / (1.0 - t_ellipse);
    squareness
}

/// The shape of a contour from its vertices alone, without refining its top and bottom, for the
/// analytic shapes whose extremal points are among their vertices
#[cfg(test)]
fn shape_of_vertices(boundary_r: &Array1<f64>, boundary_z: &Array1<f64>) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let extremal_points: ExtremalPoints = extremal_points(boundary_r, boundary_z).expect("the contour can be measured");
    shape_from_extremal_points(boundary_r, boundary_z, &extremal_points)
}

/// A time-slice on ST40's 12.5 mm grid whose `psi` is
/// `psi = 1 - (a_rr * x ** 2 + a_rz * x * y + a_zz * y ** 2)`, with `x = r - centre_r` and
/// `y = z - centre_z`.
///
/// Its flux surfaces are ellipses, tilted when `a_rz` is not zero, nested around the magnetic axis at
/// `(centre_r, centre_z)` where `psi_magnetic_axis = 1`; the `psi = 0` surface is taken as the
/// boundary. `psi` is quadratic, so its bicubic interpolation is exact.
#[cfg(test)]
pub(in crate::equilibrium_post_processor) fn time_slice_with_elliptical_psi(
    centre_r: f64,
    centre_z: f64,
    a_rr: f64,
    a_rz: f64,
    a_zz: f64,
) -> EquilibriumTimeSlice {
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::Array2;

    let n_r: usize = 49;
    let n_z: usize = 97;
    let r: Array1<f64> = Array1::linspace(0.2, 0.8, n_r);
    let z: Array1<f64> = Array1::linspace(-0.6, 0.6, n_z);

    let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let x: f64 = r[i_r] - centre_r;
            let y: f64 = z[i_z] - centre_z;
            psi_2d[(i_z, i_r)] = 1.0 - (a_rr * x.powi(2) + a_rz * x * y + a_zz * y.powi(2));
            d_psi_d_r_2d[(i_z, i_r)] = -(2.0 * a_rr * x + a_rz * y);
            d_psi_d_z_2d[(i_z, i_r)] = -(a_rz * x + 2.0 * a_zz * y);
        }
    }

    let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
    profiles_2d.grid.dim1 = r;
    profiles_2d.grid.dim2 = z;
    profiles_2d.psi = psi_2d;
    profiles_2d.d_psi_d_r = d_psi_d_r_2d;
    profiles_2d.d_psi_d_z = d_psi_d_z_2d;
    profiles_2d.d2_psi_d_r2 = Array2::from_elem((n_z, n_r), -2.0 * a_rr);
    profiles_2d.d2_psi_d_r_d_z = Array2::from_elem((n_z, n_r), -a_rz);
    profiles_2d.d2_psi_d_z2 = Array2::from_elem((n_z, n_r), -2.0 * a_zz);

    let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
    time_slice.global_quantities.magnetic_axis.r = centre_r;
    time_slice.global_quantities.magnetic_axis.z = centre_z;
    time_slice.global_quantities.psi_magnetic_axis = 1.0;
    time_slice.boundary.psi = 0.0;
    time_slice.profiles_2d = vec![profiles_2d];
    time_slice
}

/// The `psi = psi_contour` flux surface around the magnetic axis, traced by marching squares, as
/// `flux_surfaces` traces one
#[cfg(test)]
pub(in crate::equilibrium_post_processor) fn traced_contour(time_slice: &EquilibriumTimeSlice, psi_contour: f64) -> (Array1<f64>, Array1<f64>) {
    use crate::plasma_geometry::MarchingContour;
    use crate::plasma_geometry::flood_fill_mask_at_psi;
    use crate::plasma_geometry::marching_squares::marching_squares;
    use ndarray::Array2;

    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;

    let mask_2d: Array2<f64> = flood_fill_mask_at_psi(r, z, psi_2d, psi_contour, mag_r, mag_z);
    let contour: MarchingContour = marching_squares(
        r,
        z,
        psi_2d,
        &time_slice.profiles_2d[0].d_psi_d_r,
        &time_slice.profiles_2d[0].d_psi_d_z,
        psi_contour,
        &mask_2d,
        None,
        None,
        None,
        None,
        mag_r,
        mag_z,
    );

    (contour.r, contour.z)
}

#[test]
fn test_boundary_geometry_ellipse() {
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

    let (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o) = shape_of_vertices(&boundary_r, &boundary_z);

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
fn test_boundary_geometry_miller() {
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

    let (elongation, triang, triang_l, triang_u, _square_l_i, _square_l_o, _square_u_i, _square_u_o) = shape_of_vertices(&boundary_r, &boundary_z);

    assert_abs_diff_eq!(elongation, kappa, epsilon = 1e-6);
    assert_abs_diff_eq!(triang, delta, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_l, delta, epsilon = 1e-6);
    assert_abs_diff_eq!(triang_u, delta, epsilon = 1e-6);
}

#[test]
fn test_boundary_geometry_superellipse() {
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

    let (elongation, triang, _triang_l, _triang_u, square_l_i, square_l_o, square_u_i, square_u_o) = shape_of_vertices(&boundary_r, &boundary_z);

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

/// A tilted ellipse has its top and bottom away from its centre, at analytic positions which the
/// refinement has to find from vertices on the grid lines either side of them
#[test]
fn test_refine_top_or_bottom_of_a_tilted_ellipse() {
    use approx::assert_abs_diff_eq;

    // An ellipse 0.5 m wide, tilted by `a_rz`
    let centre_r: f64 = 0.5;
    let centre_z: f64 = 0.0;
    let a_rr: f64 = 16.0;
    let a_rz: f64 = 4.0;
    let a_zz: f64 = 6.25;
    let time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(centre_r, centre_z, a_rr, a_rz, a_zz);
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);

    // On `a_rr * x ** 2 + a_rz * x * y + a_zz * y ** 2 = 1` the top is where `2 * a_rr * x + a_rz * y = 0`.
    // The bottom is the top reflected through the centre
    let top_y: f64 = 1.0 / (a_zz - a_rz.powi(2) / (4.0 * a_rr)).sqrt();
    let top_x: f64 = -a_rz * top_y / (2.0 * a_rr);

    // The ellipse's height at `x`, on its upper branch when `sign = 1` and its lower when `sign = -1`
    let ellipse_y =
        |x: f64, sign: f64| -> f64 { (-a_rz * x + sign * (a_rz.powi(2) * x.powi(2) - 4.0 * a_zz * (a_rr * x.powi(2) - 1.0)).sqrt()) / (2.0 * a_zz) };

    // Start from the vertices on the vertical grid lines either side of the top, where marching
    // squares puts them: the grid starts at `r = 0.2`. The highest vertex is one of these two
    let d_r: f64 = psi_interpolator.d_r;
    let start_x_on_grid_line: f64 = (centre_r + top_x - 0.2).div_euclid(d_r) * d_r + 0.2 - centre_r;
    let starts_x: Array1<f64> = ndarray::array![start_x_on_grid_line, start_x_on_grid_line + d_r];
    let n_start: usize = starts_x.len();
    for i_start in 0..n_start {
        let (top_r, top_z): (f64, f64) = refine_top_or_bottom(
            centre_r + starts_x[i_start],
            centre_z + ellipse_y(starts_x[i_start], 1.0),
            0.0,
            &psi_interpolator,
        )
        .expect("the top is found");
        assert_abs_diff_eq!(top_r, centre_r + top_x, epsilon = 1e-10);
        assert_abs_diff_eq!(top_z, centre_z + top_y, epsilon = 1e-10);

        let (bottom_r, bottom_z): (f64, f64) = refine_top_or_bottom(
            centre_r - starts_x[i_start],
            centre_z + ellipse_y(-starts_x[i_start], -1.0),
            0.0,
            &psi_interpolator,
        )
        .expect("the bottom is found");
        assert_abs_diff_eq!(bottom_r, centre_r - top_x, epsilon = 1e-10);
        assert_abs_diff_eq!(bottom_z, centre_z - top_y, epsilon = 1e-10);
    }
}

/// An ellipse centred between two vertical grid lines has its top and bottom half a cell from the
/// nearest vertex, which is the worst case for the triangularity of the vertices
#[test]
fn test_boundary_geometry_refines_the_triangularity_off_the_grid() {
    use approx::assert_abs_diff_eq;

    // Centred half a cell off a vertical grid line, and on a horizontal one, so that the inboard
    // and the outboard points are vertices, and the minor radius is exact
    let d_grid: f64 = 0.0125;
    let centre_r: f64 = 0.5 + d_grid / 2.0;
    let centre_z: f64 = 0.0;
    let r_minor: f64 = 0.25;
    let kappa: f64 = 1.6;
    let time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(centre_r, centre_z, 1.0 / r_minor.powi(2), 0.0, 1.0 / (kappa * r_minor).powi(2));
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);
    let (boundary_r, boundary_z): (Array1<f64>, Array1<f64>) = traced_contour(&time_slice, 0.0);
    let (xpts_r, xpts_z): (Array1<f64>, Array1<f64>) = (Array1::zeros(0), Array1::zeros(0));

    // The vertices put the top and the bottom half a cell off the centre: a triangularity of `d_r / (2 * a)`
    let (elongation_vertices, _triang, triang_l_vertices, triang_u_vertices, _square_l_i, _square_l_o, _square_u_i, _square_u_o) =
        shape_of_vertices(&boundary_r, &boundary_z);
    assert_abs_diff_eq!(triang_u_vertices.abs(), d_grid / (2.0 * r_minor), epsilon = 1e-9);
    assert_abs_diff_eq!(triang_l_vertices.abs(), d_grid / (2.0 * r_minor), epsilon = 1e-9);

    // Refined, an ellipse's triangularity is zero
    let (elongation, triang, triang_l, triang_u, _square_l_i, _square_l_o, _square_u_i, _square_u_o) =
        boundary_geometry(&boundary_r, &boundary_z, 0.0, &psi_interpolator, &xpts_r, &xpts_z);
    assert_abs_diff_eq!(triang_u, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(triang_l, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(triang, 0.0, epsilon = 1e-9);

    // Only `R_Zmax` and `R_Zmin` are refined: the extent of the contour, and so the elongation, is
    // that of its vertices
    assert_eq!(elongation, elongation_vertices);
}

/// An X-point is on the contour already, so it is left where it is
#[test]
fn test_boundary_geometry_does_not_refine_an_x_point() {
    use approx::assert_abs_diff_eq;

    let d_grid: f64 = 0.0125;
    let centre_r: f64 = 0.5 + d_grid / 2.0;
    let centre_z: f64 = 0.0;
    let r_minor: f64 = 0.25;
    let kappa: f64 = 1.6;
    let time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(centre_r, centre_z, 1.0 / r_minor.powi(2), 0.0, 1.0 / (kappa * r_minor).powi(2));
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);
    let (boundary_r, boundary_z): (Array1<f64>, Array1<f64>) = traced_contour(&time_slice, 0.0);

    // Pretend the bottom vertex is an X-point
    let bottom_vertex: ExtremalPoints = extremal_points(&boundary_r, &boundary_z).expect("the contour can be measured");
    let xpts_r: Array1<f64> = ndarray::array![bottom_vertex.r_at_z_min];
    let xpts_z: Array1<f64> = ndarray::array![bottom_vertex.z_min];

    let (_elongation_vertices, _triang, triang_l_vertices, _triang_u, _square_l_i, _square_l_o, _square_u_i, _square_u_o) =
        shape_of_vertices(&boundary_r, &boundary_z);
    let (_elongation, _triang, triang_l, triang_u, _square_l_i, _square_l_o, _square_u_i, _square_u_o) =
        boundary_geometry(&boundary_r, &boundary_z, 0.0, &psi_interpolator, &xpts_r, &xpts_z);

    // The bottom stays at the vertex; the top is still refined
    assert_eq!(triang_l, triang_l_vertices);
    assert!(triang_l.abs() > 0.01);
    assert_abs_diff_eq!(triang_u, 0.0, epsilon = 1e-9);
}
