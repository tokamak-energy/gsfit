use super::MarchingContour;
use super::cubic_interpolation::cubic_interpolation;
use approx::abs_diff_eq;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::collections::BTreeMap;

/// Trace the `psi = psi_b` contour along the edge of the mask.
///
/// # Arguments
/// * `r`, `z` - grid points [metre]
/// * `psi_2d`, `d_psi_d_r_2d`, `d_psi_d_z_2d` - poloidal flux and its derivatives, shape = (n_z, n_r)
/// * `psi_b` - the flux of the contour [weber]
/// * `mask_2d` - 1.0 inside the contour, 0.0 outside; the contour is only looked for on the grid edges where it changes
/// * `xpt_r_or_none`, `xpt_z_or_none` - the X-point bounding a diverted plasma, or `None` for a limited one [metre]
/// * `xpt_secondary_r_or_none`, `xpt_secondary_z_or_none` - a second X-point on the contour, for a perfect double null,
///   or `None`. Only used when the plasma is diverted; see `marching_squares_double_null`
/// * `mag_r`, `mag_z` - the magnetic axis [metre]
#[allow(clippy::too_many_arguments)]
pub fn marching_squares(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    d_psi_d_r_2d: &Array2<f64>,
    d_psi_d_z_2d: &Array2<f64>,
    psi_b: f64,
    mask_2d: &Array2<f64>,
    xpt_r_or_none: Option<f64>,
    xpt_z_or_none: Option<f64>,
    xpt_secondary_r_or_none: Option<f64>,
    xpt_secondary_z_or_none: Option<f64>,
    mag_r: f64,
    mag_z: f64,
) -> MarchingContour {
    let n_z: usize = z.len();
    let n_r: usize = r.len();

    // Key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // Value: (r_cross, z_cross)
    //
    // A `BTreeMap` rather than a `HashMap`, because the contour walk below starts from whichever
    // point the iteration yields first. `HashMap` iteration order is unspecified, and Rust seeds
    // `RandomState` per instance, so two maps holding identical crossings iterate differently even
    // within one process: the traced contour came back rotated by a different amount on every call.
    // The curve and its point set were always the same, but everything downstream which depends on
    // the ordering - the polygon area and volume, the flux surface line integrals behind `q`, and
    // the squareness - moved with it. Ordering by grid index makes the walk reproducible
    let mut unsorted_boundary_points: BTreeMap<(usize, usize, usize, usize), (f64, f64)> = BTreeMap::new();

    // March from left to right
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            // Look for where `mask_2d` changes from 0 to 1 or 1 to 0
            if abs_diff_eq!(mask_2d[(i_z, i_r)] + mask_2d[(i_z, i_r + 1)], 1.0) {
                let left_r: f64 = r[i_r];
                let left_psi: f64 = psi_2d[(i_z, i_r)];
                let left_d_psi_d_r: f64 = d_psi_d_r_2d[(i_z, i_r)];
                let right_r: f64 = r[i_r + 1];
                let right_psi: f64 = psi_2d[(i_z, i_r + 1)];
                let right_d_psi_d_r: f64 = d_psi_d_r_2d[(i_z, i_r + 1)];

                let cubic_interpolation_or_error: Result<Array1<f64>, String> =
                    cubic_interpolation(left_r, left_psi, left_d_psi_d_r, right_r, right_psi, right_d_psi_d_r, psi_b);
                if cubic_interpolation_or_error.is_ok() {
                    let cubic_interpolation_r: Array1<f64> = cubic_interpolation_or_error.unwrap();
                    if cubic_interpolation_r.len() != 1 {
                        println!("Warning: Found {} crossing points, in left to right march", cubic_interpolation_r.len());
                    }
                    let r_cross: f64 = cubic_interpolation_r[0];
                    let z_cross: f64 = z[i_z];
                    // note, this key and value structure assumes there is only one crossing per cell edge
                    unsorted_boundary_points.insert((i_r, i_z, i_r + 1, i_z), (r_cross, z_cross));
                }
            }
        }
    }

    // March from bottom to top
    for i_r in 0..n_r {
        for i_z in 0..n_z - 1 {
            // Look for where `mask_2d` changes from 0 to 1 or 1 to 0
            if abs_diff_eq!(mask_2d[(i_z, i_r)] + mask_2d[(i_z + 1, i_r)], 1.0) {
                let bottom_z: f64 = z[i_z];
                let bottom_psi: f64 = psi_2d[(i_z, i_r)];
                let bottom_d_psi_d_z: f64 = d_psi_d_z_2d[(i_z, i_r)];
                let top_z: f64 = z[i_z + 1];
                let top_psi: f64 = psi_2d[(i_z + 1, i_r)];
                let top_d_psi_d_z: f64 = d_psi_d_z_2d[(i_z + 1, i_r)];

                let cubic_interpolation_or_error: Result<Array1<f64>, String> =
                    cubic_interpolation(bottom_z, bottom_psi, bottom_d_psi_d_z, top_z, top_psi, top_d_psi_d_z, psi_b);
                if cubic_interpolation_or_error.is_ok() {
                    let cubic_interpolation_z: Array1<f64> = cubic_interpolation_or_error.unwrap();

                    // TODO: after some testing, it would appear that the difference between cubic_interpolation_z[0] and cubic_interpolation_z[1]
                    // doesn't affect anything?

                    // let z_cross: f64;
                    // if cubic_interpolation_z.len() != 1 {
                    //     println!("Warning: Found {} crossing points, in bottom to top march", cubic_interpolation_z.len());
                    //     println!("cubic_interpolation_z={:?}", cubic_interpolation_z);
                    //     z_cross = cubic_interpolation_z[1];
                    // }
                    // else {
                    //     z_cross = cubic_interpolation_z[0];
                    // }
                    let r_cross: f64 = r[i_r];
                    let z_cross: f64 = cubic_interpolation_z[0];
                    // note, this key and value structure assumes there is only one crossing per cell edge
                    unsorted_boundary_points.insert((i_r, i_z, i_r, i_z + 1), (r_cross, z_cross));
                }
            }
        }
    }

    // If empty
    if unsorted_boundary_points.is_empty() {
        return MarchingContour {
            r: Array1::zeros(0),
            z: Array1::zeros(0),
            n: 0,
        };
    }

    // If x-point is not provided, we are limited
    if xpt_r_or_none.is_none() || xpt_z_or_none.is_none() {
        // Collect boundary points, in grid-index order
        let mut unsorted_boundary_r: Vec<f64> = unsorted_boundary_points.values().map(|&(r_val, _)| r_val).collect();
        let mut unsorted_boundary_z: Vec<f64> = unsorted_boundary_points.values().map(|&(_, z_val)| z_val).collect();

        // Sort the boundary points using nearest-neighbor algorithm
        let mut sorted_r: Vec<f64> = Vec::new();
        let mut sorted_z: Vec<f64> = Vec::new();
        sorted_r.push(unsorted_boundary_r.last().copied().unwrap());
        sorted_z.push(unsorted_boundary_z.last().copied().unwrap());
        unsorted_boundary_r.pop();
        unsorted_boundary_z.pop();
        let (mut boundary_r_sorted, mut boundary_z_sorted): (Vec<f64>, Vec<f64>) =
            sort_boundary_points(sorted_r, sorted_z, &unsorted_boundary_r, &unsorted_boundary_z);

        // Add the first point to close the contour
        boundary_r_sorted.push(boundary_r_sorted[0]);
        boundary_z_sorted.push(boundary_z_sorted[0]);

        let n: usize = boundary_r_sorted.len();
        let boundary_contour: MarchingContour = MarchingContour {
            r: Array1::from_vec(boundary_r_sorted),
            z: Array1::from_vec(boundary_z_sorted),
            n,
        };

        return boundary_contour;
    }

    // From now on we are handling an x-point diverted plasma

    // Unwrap the x-point coordinates
    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    // Special condition: a perfect double null, whose boundary passes through a second X-point
    if let (Some(xpt_secondary_r), Some(xpt_secondary_z)) = (xpt_secondary_r_or_none, xpt_secondary_z_or_none) {
        return marching_squares_double_null(
            r,
            z,
            psi_2d,
            d_psi_d_r_2d,
            d_psi_d_z_2d,
            psi_b,
            unsorted_boundary_points,
            (xpt_r, xpt_z),
            (xpt_secondary_r, xpt_secondary_z),
            mag_r,
            mag_z,
        );
    }

    // Find the closest grid point
    let i_r_nearest_xpt: usize = (r - xpt_r).abs().argmin().unwrap();
    let i_z_nearest_xpt: usize = (z - xpt_z).abs().argmin().unwrap();
    // let mut i_r_nearest_xpt: usize = 0;
    // let mut min_r_dist: f64 = f64::INFINITY;
    // for (i, &r_val) in r.iter().enumerate() {
    //     let dist: f64 = (xpt_r - r_val).abs();
    //     if dist < min_r_dist {
    //         min_r_dist = dist;
    //         i_r_nearest_xpt = i;
    //     }
    // }
    // let mut i_z_nearest_xpt: usize = 0;
    // let mut min_z_dist: f64 = f64::INFINITY;
    // for (i, &z_val) in z.iter().enumerate() {
    //     let dist: f64 = (xpt_z - z_val).abs();
    //     if dist < min_z_dist {
    //         min_z_dist = dist;
    //         i_z_nearest_xpt = i;
    //     }
    // }

    // Find the four corner grid points surrounding the x-point
    let i_r_nearest_xpt_left: usize;
    let i_r_nearest_xpt_right: usize;
    let i_z_nearest_xpt_lower: usize;
    let i_z_nearest_xpt_upper: usize;
    if xpt_r > r[i_r_nearest_xpt] {
        i_r_nearest_xpt_left = i_r_nearest_xpt;
        i_r_nearest_xpt_right = i_r_nearest_xpt + 1;
    } else {
        i_r_nearest_xpt_left = i_r_nearest_xpt - 1;
        i_r_nearest_xpt_right = i_r_nearest_xpt;
    }
    if xpt_z > z[i_z_nearest_xpt] {
        i_z_nearest_xpt_lower = i_z_nearest_xpt;
        i_z_nearest_xpt_upper = i_z_nearest_xpt + 1;
    } else {
        i_z_nearest_xpt_lower = i_z_nearest_xpt - 1;
        i_z_nearest_xpt_upper = i_z_nearest_xpt;
    }

    // Remove the connections between the four grid points from the boundary map, which enclose the x-point
    let cells_to_remove: Vec<(usize, usize, usize, usize)> = vec![
        // Marching left to right
        // lower edge
        (i_r_nearest_xpt_left, i_z_nearest_xpt_lower, i_r_nearest_xpt_right, i_z_nearest_xpt_lower),
        // upper edge
        (i_r_nearest_xpt_left, i_z_nearest_xpt_upper, i_r_nearest_xpt_right, i_z_nearest_xpt_upper),
        // Marching bottom to top
        // left edge
        (i_r_nearest_xpt_left, i_z_nearest_xpt_lower, i_r_nearest_xpt_left, i_z_nearest_xpt_upper),
        // right edge
        (i_r_nearest_xpt_right, i_z_nearest_xpt_lower, i_r_nearest_xpt_right, i_z_nearest_xpt_upper),
    ];
    // Remove
    for cell in &cells_to_remove {
        unsorted_boundary_points.remove(cell);
    }

    // The maximum number of points is 12 (cubic polynomial, and 4 edges)
    // But from the saddle point geometry, we expect 4 points
    // Unless there was an unusual snowflake configuration
    let mut boundary_points_near_xpt_r: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_z: Vec<f64> = Vec::new();

    // Use cubic interpolation to find the boundary points on the four edges around the x-point
    for cell in cells_to_remove {
        let (i_r_from, i_z_from, i_r_to, i_z_to) = cell;
        if i_z_from == i_z_to {
            // marching left to right
            let left_r: f64 = r[i_r_from];
            let left_psi: f64 = psi_2d[(i_z_from, i_r_from)];
            let left_d_psi_d_r: f64 = d_psi_d_r_2d[(i_z_from, i_r_from)];
            let right_r: f64 = r[i_r_to];
            let right_psi: f64 = psi_2d[(i_z_from, i_r_to)];
            let right_d_psi_d_r: f64 = d_psi_d_r_2d[(i_z_from, i_r_to)];
            let cubic_interpolation_or_error: Result<Array1<f64>, String> =
                cubic_interpolation(left_r, left_psi, left_d_psi_d_r, right_r, right_psi, right_d_psi_d_r, psi_b);
            if cubic_interpolation_or_error.is_ok() {
                let cubic_interpolation_r: Array1<f64> = cubic_interpolation_or_error.unwrap();
                for i_r in 0..cubic_interpolation_r.len() {
                    boundary_points_near_xpt_r.push(cubic_interpolation_r[i_r]);
                    boundary_points_near_xpt_z.push(z[i_z_from]);
                }
            }
        } else if i_r_from == i_r_to {
            // marching bottom to top
            let bottom_z: f64 = z[i_z_from];
            let bottom_psi: f64 = psi_2d[(i_z_from, i_r_from)];
            let bottom_d_psi_d_z: f64 = d_psi_d_z_2d[(i_z_from, i_r_from)];
            let top_z: f64 = z[i_z_to];
            let top_psi: f64 = psi_2d[(i_z_to, i_r_from)];
            let top_d_psi_d_z: f64 = d_psi_d_z_2d[(i_z_to, i_r_from)];
            let cubic_interpolation_or_error: Result<Array1<f64>, String> =
                cubic_interpolation(bottom_z, bottom_psi, bottom_d_psi_d_z, top_z, top_psi, top_d_psi_d_z, psi_b);
            if cubic_interpolation_or_error.is_ok() {
                let cubic_interpolation_z: Array1<f64> = cubic_interpolation_or_error.unwrap();
                for i_z in 0..cubic_interpolation_z.len() {
                    boundary_points_near_xpt_r.push(r[i_r_from]);
                    boundary_points_near_xpt_z.push(cubic_interpolation_z[i_z]);
                }
            }
        }
    }

    // direction vector from x-point to magnetic axis
    let delta_r: f64 = mag_r - xpt_r;
    let delta_z: f64 = mag_z - xpt_z;
    let d_mag: f64 = (delta_r.powi(2) + delta_z.powi(2)).sqrt();
    let delta_r_unit: f64 = delta_r / d_mag;
    let delta_z_unit: f64 = delta_z / d_mag;

    // Sort the boundary points around the x-point by how parallel they are to the direction to the magnetic axis
    let mut dot_products: Vec<f64> = Vec::new();
    for i in 0..boundary_points_near_xpt_r.len() {
        let d_r: f64 = boundary_points_near_xpt_r[i] - xpt_r;
        let d_z: f64 = boundary_points_near_xpt_z[i] - xpt_z;
        let d: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt();
        let d_r_unit: f64 = d_r / d;
        let d_z_unit: f64 = d_z / d;
        let dot_product: f64 = d_r_unit * delta_r_unit + d_z_unit * delta_z_unit;
        dot_products.push(dot_product);
    }

    let i_sorted: Vec<usize> = {
        let mut indices: Vec<usize> = (0..dot_products.len()).collect();
        // Sort by descending dot product (most parallel first)
        indices.sort_by(|&i, &j| dot_products[j].partial_cmp(&dot_products[i]).unwrap());
        indices
    };
    let mut boundary_points_near_xpt_r_sorted: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_z_sorted: Vec<f64> = Vec::new();
    for &i in &i_sorted {
        boundary_points_near_xpt_r_sorted.push(boundary_points_near_xpt_r[i]);
        boundary_points_near_xpt_z_sorted.push(boundary_points_near_xpt_z[i]);
    }

    let mut first_and_last_boundary_points: Vec<(f64, f64)> = Vec::new();
    first_and_last_boundary_points.push((boundary_points_near_xpt_r_sorted[0], boundary_points_near_xpt_z_sorted[0]));
    first_and_last_boundary_points.push((boundary_points_near_xpt_r_sorted[1], boundary_points_near_xpt_z_sorted[1]));

    let mut boundary_sorted_r: Vec<f64> = Vec::new();
    let mut boundary_sorted_z: Vec<f64> = Vec::new();

    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    boundary_sorted_r.push(xpt_r);
    boundary_sorted_z.push(xpt_z);
    boundary_sorted_r.push(first_and_last_boundary_points[0].0);
    boundary_sorted_z.push(first_and_last_boundary_points[0].1);

    // // Sort the boundary points using nearest-neighbor algorithm
    // let (mut boundary_sorted_r, mut boundary_sorted_z): (Vec<f64>, Vec<f64>) =
    //     sort_boundary_points(boundary_sorted_r, boundary_sorted_z, &unsorted_boundary_r, &unsorted_boundary_z);

    // // Add the last point to close the contour
    // boundary_sorted_r.push(first_and_last_boundary_points[1].0);
    // boundary_sorted_z.push(first_and_last_boundary_points[1].1);

    // // Add the x-point
    // boundary_sorted_r.push(xpt_r);
    // boundary_sorted_z.push(xpt_z);

    // re-add "sort_boundary_points_version_2" here..
    let (boundary_sorted_r, boundary_sorted_z): (Vec<f64>, Vec<f64>) =
        sort_boundary_points_version_2(unsorted_boundary_points, xpt_r, xpt_z, mag_r, mag_z, first_and_last_boundary_points);

    let n: usize = boundary_sorted_r.len();
    let boundary_contour: MarchingContour = MarchingContour {
        r: Array1::from_vec(boundary_sorted_r),
        z: Array1::from_vec(boundary_sorted_z),
        n,
    };

    boundary_contour
}

/// Check if two line segments intersect
/// Segment 1: (r1, z1) to (r2, z2)
/// Segment 2: (r3, z3) to (r4, z4)
fn segments_intersect(r1: f64, z1: f64, r2: f64, z2: f64, r3: f64, z3: f64, r4: f64, z4: f64) -> bool {
    // Vector from p1 to p2
    let d1r = r2 - r1;
    let d1z = z2 - z1;

    // Vector from p3 to p4
    let d2r = r4 - r3;
    let d2z = z4 - z3;

    // Cross product of d1 and d2
    let cross = d1r * d2z - d1z * d2r;

    // Parallel or coincident lines (no intersection)
    if cross.abs() < 1e-10 {
        return false;
    }

    // Vector from p1 to p3
    let d3r = r3 - r1;
    let d3z = z3 - z1;

    // Calculate parameters t and u for the intersection point
    let t = (d3r * d2z - d3z * d2r) / cross;
    let u = (d3r * d1z - d3z * d1r) / cross;

    // Check if intersection point is within both segments
    // Use a small epsilon to avoid false positives at endpoints
    const EPSILON: f64 = 1e-9;
    t > EPSILON && t < 1.0 - EPSILON && u > EPSILON && u < 1.0 - EPSILON
}

/// sort points
fn sort_boundary_points(sorted_r: Vec<f64>, sorted_z: Vec<f64>, unsorted_boundary_r: &[f64], unsorted_boundary_z: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // let n_points: usize = unsorted_boundary_r.len();

    // Create mutable copies
    let mut sorted_r: Vec<f64> = sorted_r;
    let mut sorted_z: Vec<f64> = sorted_z;
    let mut unsorted_boundary_r: Vec<f64> = unsorted_boundary_r.to_vec();
    let mut unsorted_boundary_z: Vec<f64> = unsorted_boundary_z.to_vec();

    // Start from the last point in sorted lists
    let mut current_r: f64 = sorted_r.last().copied().unwrap();
    let mut current_z: f64 = sorted_z.last().copied().unwrap();

    // Each iteration removes exactly one point, so the loop runs for the initial number of points.
    let n_point: usize = unsorted_boundary_r.len();

    // Iteratively find nearest unvisited point
    for _i_point in 0..n_point {
        let mut min_dist: f64 = f64::INFINITY;
        let mut nearest_idx: usize = 0;

        // Find nearest point
        for i in 0..unsorted_boundary_r.len() {
            let d_r: f64 = unsorted_boundary_r[i] - current_r;
            let d_z: f64 = unsorted_boundary_z[i] - current_z;
            let dist: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt();

            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        // Add nearest point to sorted lists
        current_r = unsorted_boundary_r[nearest_idx];
        current_z = unsorted_boundary_z[nearest_idx];
        sorted_r.push(current_r);
        sorted_z.push(current_z);

        // Remove from unsorted lists
        unsorted_boundary_r.remove(nearest_idx);
        unsorted_boundary_z.remove(nearest_idx);
    }

    (sorted_r, sorted_z)
}

/// Order points between the two boundary points adjacent to the X-point,
/// with a left/right decider near the X-point to avoid wrong starts.
/// Uses the left/right distinguisher for the first 3 interior points after `start`,
/// then switches to nearest-neighbour with a no-cross guard.
/// Returns [xpt, start, ..., end, xpt].
pub fn sort_boundary_points_version_2(
    unsorted_boundary_points: BTreeMap<(usize, usize, usize, usize), (f64, f64)>,
    xpt_r: f64,
    xpt_z: f64,
    mag_r: f64,
    mag_z: f64,
    first_and_last_boundary_points: Vec<(f64, f64)>,
) -> (Vec<f64>, Vec<f64>) {
    // Collect interior points (everything except the two end points near the X-point)
    let mut interior: Vec<(f64, f64)> = unsorted_boundary_points.values().copied().collect();

    if first_and_last_boundary_points.len() < 2 {
        return (vec![xpt_r, xpt_r], vec![xpt_z, xpt_z]);
    }
    let mut start = first_and_last_boundary_points[0];
    let mut end = first_and_last_boundary_points[1];

    // Remove any accidental duplicates of start/end from interior (with a small tolerance)
    const EPS: f64 = 1e-12;
    interior.retain(|p| ((p.0 - start.0).abs() > EPS || (p.1 - start.1).abs() > EPS) && ((p.0 - end.0).abs() > EPS || (p.1 - end.1).abs() > EPS));

    if interior.is_empty() {
        return (vec![xpt_r, start.0, end.0, xpt_r], vec![xpt_z, start.1, end.1, xpt_z]);
    }

    // Classify "side" relative to the line from X-point to magnetic axis
    #[inline]
    fn side_of_axis(xr: f64, xz: f64, vr: f64, vz: f64, p: (f64, f64)) -> i8 {
        let cross = vr * (p.1 - xz) - vz * (p.0 - xr);
        if cross > 0.0 {
            1
        } else if cross < 0.0 {
            -1
        } else {
            0
        }
    }

    // Direction X -> magnetic axis (unit)
    let mut vr = mag_r - xpt_r;
    let mut vz = mag_z - xpt_z;
    let v_norm = (vr * vr + vz * vz).sqrt();
    if v_norm > 0.0 {
        vr /= v_norm;
        vz /= v_norm;
    } else {
        vr = 1.0;
        vz = 0.0;
    }

    // Prefer start and end on opposite sides; swap if both on same side
    let s_start = side_of_axis(xpt_r, xpt_z, vr, vz, start);
    let s_end = side_of_axis(xpt_r, xpt_z, vr, vz, end);
    if s_start != 0 && s_end != 0 && s_start == s_end {
        std::mem::swap(&mut start, &mut end);
    }

    // Helper: would adding segment cross the existing polyline?
    fn would_cross(r_path: &[f64], z_path: &[f64], r_last: f64, z_last: f64, r_new: f64, z_new: f64) -> bool {
        if r_path.len() < 2 {
            return false;
        }
        for i in 0..(r_path.len() - 2) {
            if segments_intersect(r_path[i], z_path[i], r_path[i + 1], z_path[i + 1], r_last, z_last, r_new, z_new) {
                return true;
            }
        }
        false
    }

    // Start chain at X-point -> start
    let mut r_sorted: Vec<f64> = vec![xpt_r, start.0];
    let mut z_sorted: Vec<f64> = vec![xpt_z, start.1];

    let mut used = vec![false; interior.len()];
    let mut cur = start;

    // Precompute sides of interior points
    let interior_side: Vec<i8> = interior.iter().map(|&p| side_of_axis(xpt_r, xpt_z, vr, vz, p)).collect();
    let preferred_side = side_of_axis(xpt_r, xpt_z, vr, vz, start);

    // Lock to left/right decider for the first K interior points
    const SIDE_LOCK_STEPS: usize = 3;
    let mut locked_steps_done: usize = 0;

    // Each iteration marks one interior point used (or breaks), so the loop runs at most `n_point_max` times.
    let n_point_max: usize = interior.len();
    for _i_point in 0..n_point_max {
        // Candidates by distance
        let mut cand: Vec<(usize, f64)> = interior
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(i, p)| {
                let d2 = (p.0 - cur.0) * (p.0 - cur.0) + (p.1 - cur.1) * (p.1 - cur.1);
                (i, d2)
            })
            .collect();
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply left/right restriction for the first few steps
        if locked_steps_done < SIDE_LOCK_STEPS && preferred_side != 0 {
            cand.retain(|(i, _)| interior_side[*i] == preferred_side || interior_side[*i] == 0);
            if cand.is_empty() {
                // If nothing on that side, fall back to all candidates
                cand = interior
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !used[*i])
                    .map(|(i, p)| {
                        let d2 = (p.0 - cur.0) * (p.0 - cur.0) + (p.1 - cur.1) * (p.1 - cur.1);
                        (i, d2)
                    })
                    .collect();
                cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // Pick nearest that doesn't cause a crossing
        let mut chosen: Option<usize> = None;
        for (i, _d2) in cand.into_iter() {
            let p = interior[i];
            if !would_cross(&r_sorted, &z_sorted, cur.0, cur.1, p.0, p.1) {
                chosen = Some(i);
                break;
            }
        }
        // If all would cross, pick nearest unused to make progress
        if chosen.is_none() {
            let mut best_i = None;
            let mut best_d2 = f64::INFINITY;
            for (i, p) in interior.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let d2 = (p.0 - cur.0) * (p.0 - cur.0) + (p.1 - cur.1) * (p.1 - cur.1);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_i = Some(i);
                }
            }
            chosen = best_i;
        }

        if let Some(i) = chosen {
            let p = interior[i];
            r_sorted.push(p.0);
            z_sorted.push(p.1);
            used[i] = true;
            cur = p;
            if locked_steps_done < SIDE_LOCK_STEPS {
                locked_steps_done += 1;
            }
        } else {
            break;
        }
    }

    // Append/insert 'end' avoiding crossings if possible
    if !would_cross(&r_sorted, &z_sorted, cur.0, cur.1, end.0, end.1) {
        r_sorted.push(end.0);
        z_sorted.push(end.1);
    } else {
        let m = r_sorted.len();
        let mut best_pos: Option<(usize, f64)> = None;
        for k in 0..(m - 1) {
            let mut crosses = false;
            for i in 0..(m - 1) {
                if i == k {
                    continue;
                }
                if segments_intersect(
                    r_sorted[i],
                    z_sorted[i],
                    r_sorted[i + 1],
                    z_sorted[i + 1],
                    r_sorted[k],
                    z_sorted[k],
                    end.0,
                    end.1,
                ) || segments_intersect(
                    r_sorted[i],
                    z_sorted[i],
                    r_sorted[i + 1],
                    z_sorted[i + 1],
                    end.0,
                    end.1,
                    r_sorted[k + 1],
                    z_sorted[k + 1],
                ) {
                    crosses = true;
                    break;
                }
            }
            if !crosses {
                let a = ((r_sorted[k] - end.0).powi(2) + (z_sorted[k] - end.1).powi(2)).sqrt();
                let b = ((end.0 - r_sorted[k + 1]).powi(2) + (end.1 - z_sorted[k + 1]).powi(2)).sqrt();
                let old = ((r_sorted[k] - r_sorted[k + 1]).powi(2) + (z_sorted[k] - z_sorted[k + 1]).powi(2)).sqrt();
                let added = a + b - old;
                if best_pos.map(|(_, best)| added < best).unwrap_or(true) {
                    best_pos = Some((k + 1, added));
                }
            }
        }
        if let Some((pos, _)) = best_pos {
            r_sorted.insert(pos, end.0);
            z_sorted.insert(pos, end.1);
        } else {
            r_sorted.push(end.0);
            z_sorted.push(end.1);
        }
    }

    // Close at the X-point
    r_sorted.push(xpt_r);
    z_sorted.push(xpt_z);

    (r_sorted, z_sorted)
}

/// The boundary of a perfect double null: the `psi = psi_b` contour through both X-points.
///
/// The line joining the two X-points splits the boundary into two arcs, one on each side of it.
/// Each arc is ordered by nearest neighbour, from one X-point to the other, and the two are joined
/// at the X-points: primary X-point, first arc, secondary X-point, second arc, primary X-point.
///
/// The grid cell holding each X-point is handled as `marching_squares` handles a single X-point's:
/// its edges' crossings are taken out of the mask's, and the contour is looked for on all four of
/// its edges, whatever the mask. Of those crossings on each side of the line, the one in the
/// direction most towards the magnetic axis is where the arc leaves the X-point; the others are on
/// the divertor legs, and are dropped. Choosing one on each side, rather than the two most towards
/// the axis, keeps both arcs when `psi` at an X-point is not exactly `psi_b`, and the contour
/// passes to one side of it. A side with no crossing around an X-point joins its arc straight to
/// the X-point.
///
/// # Arguments
/// * `r`, `z` - grid points [metre]
/// * `psi_2d`, `d_psi_d_r_2d`, `d_psi_d_z_2d` - poloidal flux and its derivatives, shape = (n_z, n_r)
/// * `psi_b` - the flux of the contour [weber]
/// * `unsorted_boundary_points` - the contour's crossings of the grid edges where the mask changes, keyed by edge
/// * `xpt_primary` - `(r, z)` of the X-point bounding the plasma [metre]
/// * `xpt_secondary` - `(r, z)` of the other X-point [metre]
/// * `mag_r`, `mag_z` - the magnetic axis [metre]
///
/// # Returns
/// * the closed contour, from the primary X-point round to it again
#[allow(clippy::too_many_arguments)]
fn marching_squares_double_null(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    d_psi_d_r_2d: &Array2<f64>,
    d_psi_d_z_2d: &Array2<f64>,
    psi_b: f64,
    mut unsorted_boundary_points: BTreeMap<(usize, usize, usize, usize), (f64, f64)>,
    xpt_primary: (f64, f64),
    xpt_secondary: (f64, f64),
    mag_r: f64,
    mag_z: f64,
) -> MarchingContour {
    let crossings_primary: Vec<(f64, f64)> =
        crossings_around_x_point(r, z, psi_2d, d_psi_d_r_2d, d_psi_d_z_2d, psi_b, xpt_primary, &mut unsorted_boundary_points);
    let crossings_secondary: Vec<(f64, f64)> =
        crossings_around_x_point(r, z, psi_2d, d_psi_d_r_2d, d_psi_d_z_2d, psi_b, xpt_secondary, &mut unsorted_boundary_points);

    // Which side of the line from the primary to the secondary X-point a point is on
    let line_r: f64 = xpt_secondary.0 - xpt_primary.0;
    let line_z: f64 = xpt_secondary.1 - xpt_primary.1;
    let is_left = |point: (f64, f64)| -> bool { line_r * (point.1 - xpt_primary.1) - line_z * (point.0 - xpt_primary.0) >= 0.0 };

    let mut contour_r: Vec<f64> = Vec::new();
    let mut contour_z: Vec<f64> = Vec::new();
    // The left arc runs from the primary X-point to the secondary, and the right arc back again
    for (left, xpt_from, crossings_from, xpt_to, crossings_to) in [
        (true, xpt_primary, &crossings_primary, xpt_secondary, &crossings_secondary),
        (false, xpt_secondary, &crossings_secondary, xpt_primary, &crossings_primary),
    ] {
        let crossings_from_side: Vec<(f64, f64)> = crossings_from.iter().copied().filter(|&point| is_left(point) == left).collect();
        let crossings_to_side: Vec<(f64, f64)> = crossings_to.iter().copied().filter(|&point| is_left(point) == left).collect();
        let interior: Vec<(f64, f64)> = unsorted_boundary_points.values().copied().filter(|&point| is_left(point) == left).collect();
        let interior_r: Vec<f64> = interior.iter().map(|point| point.0).collect();
        let interior_z: Vec<f64> = interior.iter().map(|point| point.1).collect();

        let mut arc_r: Vec<f64> = vec![xpt_from.0];
        let mut arc_z: Vec<f64> = vec![xpt_from.1];
        if let Some(first) = crossing_towards_magnetic_axis(&crossings_from_side, xpt_from, mag_r, mag_z) {
            arc_r.push(first.0);
            arc_z.push(first.1);
        }
        let (mut arc_r, mut arc_z): (Vec<f64>, Vec<f64>) = sort_boundary_points(arc_r, arc_z, &interior_r, &interior_z);
        if let Some(last) = crossing_towards_magnetic_axis(&crossings_to_side, xpt_to, mag_r, mag_z) {
            arc_r.push(last.0);
            arc_z.push(last.1);
        }

        // Each arc ends where the next begins, so it is added without its end X-point
        contour_r.extend(arc_r);
        contour_z.extend(arc_z);
    }

    // Close at the primary X-point
    contour_r.push(xpt_primary.0);
    contour_z.push(xpt_primary.1);

    let n: usize = contour_r.len();
    MarchingContour {
        r: Array1::from_vec(contour_r),
        z: Array1::from_vec(contour_z),
        n,
    }
}

/// The contour's crossings of the four edges of the grid cell holding an X-point, whatever the
/// mask. Those edges are removed from `unsorted_boundary_points`, as `marching_squares` does for a
/// single X-point.
///
/// # Arguments
/// * `r`, `z` - grid points [metre]
/// * `psi_2d`, `d_psi_d_r_2d`, `d_psi_d_z_2d` - poloidal flux and its derivatives, shape = (n_z, n_r)
/// * `psi_b` - the flux of the contour [weber]
/// * `xpt` - `(r, z)` of the X-point [metre]
/// * `unsorted_boundary_points` - the contour's crossings of the grid edges where the mask changes, keyed by edge
///
/// # Returns
/// * the crossings, `(r, z)` [metre]
#[allow(clippy::too_many_arguments)]
fn crossings_around_x_point(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    d_psi_d_r_2d: &Array2<f64>,
    d_psi_d_z_2d: &Array2<f64>,
    psi_b: f64,
    xpt: (f64, f64),
    unsorted_boundary_points: &mut BTreeMap<(usize, usize, usize, usize), (f64, f64)>,
) -> Vec<(f64, f64)> {
    // The grid cell holding the X-point
    let i_r_nearest_xpt: usize = (r - xpt.0).abs().argmin().unwrap();
    let i_z_nearest_xpt: usize = (z - xpt.1).abs().argmin().unwrap();
    let (i_r_left, i_r_right): (usize, usize) = if xpt.0 > r[i_r_nearest_xpt] {
        (i_r_nearest_xpt, i_r_nearest_xpt + 1)
    } else {
        (i_r_nearest_xpt - 1, i_r_nearest_xpt)
    };
    let (i_z_lower, i_z_upper): (usize, usize) = if xpt.1 > z[i_z_nearest_xpt] {
        (i_z_nearest_xpt, i_z_nearest_xpt + 1)
    } else {
        (i_z_nearest_xpt - 1, i_z_nearest_xpt)
    };

    let mut crossings: Vec<(f64, f64)> = Vec::new();
    // Left to right, along the lower and upper edges
    for i_z in [i_z_lower, i_z_upper] {
        unsorted_boundary_points.remove(&(i_r_left, i_z, i_r_right, i_z));
        let cubic_interpolation_or_error: Result<Array1<f64>, String> = cubic_interpolation(
            r[i_r_left],
            psi_2d[(i_z, i_r_left)],
            d_psi_d_r_2d[(i_z, i_r_left)],
            r[i_r_right],
            psi_2d[(i_z, i_r_right)],
            d_psi_d_r_2d[(i_z, i_r_right)],
            psi_b,
        );
        if let Ok(r_crossings) = cubic_interpolation_or_error {
            crossings.extend(r_crossings.iter().map(|&r_cross| (r_cross, z[i_z])));
        }
    }
    // Bottom to top, along the left and right edges
    for i_r in [i_r_left, i_r_right] {
        unsorted_boundary_points.remove(&(i_r, i_z_lower, i_r, i_z_upper));
        let cubic_interpolation_or_error: Result<Array1<f64>, String> = cubic_interpolation(
            z[i_z_lower],
            psi_2d[(i_z_lower, i_r)],
            d_psi_d_z_2d[(i_z_lower, i_r)],
            z[i_z_upper],
            psi_2d[(i_z_upper, i_r)],
            d_psi_d_z_2d[(i_z_upper, i_r)],
            psi_b,
        );
        if let Ok(z_crossings) = cubic_interpolation_or_error {
            crossings.extend(z_crossings.iter().map(|&z_cross| (r[i_r], z_cross)));
        }
    }

    crossings
}

/// Of the crossings around an X-point, the one in the direction most towards the magnetic axis,
/// which is where the boundary leaves the X-point; or `None` when there are none.
///
/// # Arguments
/// * `crossings` - `(r, z)` of the crossings [metre]
/// * `xpt` - `(r, z)` of the X-point [metre]
/// * `mag_r`, `mag_z` - the magnetic axis [metre]
fn crossing_towards_magnetic_axis(crossings: &[(f64, f64)], xpt: (f64, f64), mag_r: f64, mag_z: f64) -> Option<(f64, f64)> {
    let axis_distance: f64 = (mag_r - xpt.0).hypot(mag_z - xpt.1);
    let mut best_crossing: Option<(f64, f64)> = None;
    let mut best_cosine: f64 = f64::NEG_INFINITY;
    for &crossing in crossings {
        let crossing_distance: f64 = (crossing.0 - xpt.0).hypot(crossing.1 - xpt.1);
        // A crossing on the X-point itself has no direction
        if crossing_distance == 0.0 {
            continue;
        }
        let cosine: f64 = ((crossing.0 - xpt.0) * (mag_r - xpt.0) + (crossing.1 - xpt.1) * (mag_z - xpt.1)) / (crossing_distance * axis_distance);
        if cosine > best_cosine {
            best_cosine = cosine;
            best_crossing = Some(crossing);
        }
    }
    best_crossing
}

#[test]
fn test_marching_squares_is_deterministic() {
    use ndarray::Array1;

    // A circular flux function, `psi = (r - r_axis) ** 2 + (z - z_axis) ** 2`, contoured at a level
    // which encloses a good number of grid cells, so the traced contour has many points and any
    // change in the walk's starting point shows up
    let n_r: usize = 41;
    let n_z: usize = 41;
    let r: Array1<f64> = Array1::linspace(0.1, 1.1, n_r);
    let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_z);

    let r_axis: f64 = 0.6;
    let z_axis: f64 = 0.0;
    let psi_b: f64 = 0.09; // a circle of radius 0.3 about the axis

    let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut mask_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let d_r: f64 = r[i_r] - r_axis;
            let d_z: f64 = z[i_z] - z_axis;
            psi_2d[(i_z, i_r)] = d_r.powi(2) + d_z.powi(2);
            d_psi_d_r_2d[(i_z, i_r)] = 2.0 * d_r;
            d_psi_d_z_2d[(i_z, i_r)] = 2.0 * d_z;
            // `psi` increases outwards here, so the plasma is the low-`psi` region
            mask_2d[(i_z, i_r)] = if psi_2d[(i_z, i_r)] < psi_b { 1.0 } else { 0.0 };
        }
    }

    // Limited, so no x-point is supplied. This is the branch which orders the crossings by
    // collecting them out of the map, and so the branch which was nondeterministic when that map
    // was a `HashMap`
    let first: MarchingContour = marching_squares(
        &r,
        &z,
        &psi_2d,
        &d_psi_d_r_2d,
        &d_psi_d_z_2d,
        psi_b,
        &mask_2d,
        None,
        None,
        None,
        None,
        r_axis,
        z_axis,
    );
    let second: MarchingContour = marching_squares(
        &r,
        &z,
        &psi_2d,
        &d_psi_d_r_2d,
        &d_psi_d_z_2d,
        psi_b,
        &mask_2d,
        None,
        None,
        None,
        None,
        r_axis,
        z_axis,
    );

    assert!(first.n > 20, "expected a well-resolved contour, got {} points", first.n);
    assert_eq!(first.n, second.n, "contour length is not reproducible");

    // Bitwise equality, not a tolerance: the same input must give the same contour, in the same
    // order, every time
    assert_eq!(first.r, second.r, "contour `r` is not reproducible");
    assert_eq!(first.z, second.z, "contour `z` is not reproducible");
}

#[test]
fn test_marching_squares_perfect_double_null() {
    use ndarray::Array1;

    // `psi = -a * (r - r_axis) ** 2 + b * (z ** 2 - z_xpt ** 2) ** 2` has its maximum, the magnetic
    // axis, at (r_axis, 0), and saddle points at (r_axis, +/- z_xpt), where `psi = 0`: a perfect double
    // null with `psi_b = 0`. Its boundary is `z ** 2 = z_xpt ** 2 - k * |r - r_axis|`, with `k = sqrt(a / b)`,
    // which encloses an area of `8 * z_xpt ** 3 / (3 * k)`. Beyond each X-point, between the divertor legs,
    // is a private flux region, where `psi > psi_b` too
    let a: f64 = 1.44;
    let b: f64 = 1.0;
    let r_axis: f64 = 0.6;
    let z_xpt: f64 = 0.6;
    let psi_b: f64 = 0.0;
    let psi_a: f64 = b * z_xpt.powi(4);
    let psi = |r: f64, z: f64| -> f64 { -a * (r - r_axis).powi(2) + b * (z * z - z_xpt * z_xpt).powi(2) };
    let area_expected: f64 = 8.0 * z_xpt.powi(3) / (3.0 * (a / b).sqrt());

    // Neither X-point, nor the axis, is on a grid point
    let n_r: usize = 50;
    let n_z: usize = 100;
    let r: Array1<f64> = Array1::linspace(0.1, 1.1, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut mask_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            psi_2d[(i_z, i_r)] = psi(r[i_r], z[i_z]);
            d_psi_d_r_2d[(i_z, i_r)] = -2.0 * a * (r[i_r] - r_axis);
            d_psi_d_z_2d[(i_z, i_r)] = 4.0 * b * z[i_z] * (z[i_z] * z[i_z] - z_xpt * z_xpt);
            // The mask stops at the X-points, as the flood fill's does, so the private flux regions are not in it
            mask_2d[(i_z, i_r)] = if psi_2d[(i_z, i_r)] > psi_b && z[i_z].abs() < z_xpt { 1.0 } else { 0.0 };
        }
    }

    // The lower X-point bounds the plasma, and the upper one is the secondary
    let contour: MarchingContour = marching_squares(
        &r,
        &z,
        &psi_2d,
        &d_psi_d_r_2d,
        &d_psi_d_z_2d,
        psi_b,
        &mask_2d,
        Some(r_axis),
        Some(-z_xpt),
        Some(r_axis),
        Some(z_xpt),
        r_axis,
        0.0,
    );
    let n: usize = contour.n;
    assert!(n > 50, "expected a well-resolved contour, got {n} points");

    // Closed at the primary X-point, and through the secondary one
    assert_eq!(
        (contour.r[0], contour.z[0]),
        (r_axis, -z_xpt),
        "the contour does not start at the primary X-point"
    );
    assert_eq!(
        (contour.r[n - 1], contour.z[n - 1]),
        (r_axis, -z_xpt),
        "the contour does not end at the primary X-point"
    );
    let n_secondary: usize = (0..n).filter(|&i| contour.r[i] == r_axis && contour.z[i] == z_xpt).count();
    assert_eq!(n_secondary, 1, "the contour passes through the secondary X-point {n_secondary} times");

    // Every point is on the boundary, and none is in a private flux region
    for i in 0..n {
        assert!(
            (psi(contour.r[i], contour.z[i]) - psi_b).abs() < 1.0e-5 * psi_a,
            "point {i}, ({}, {}), is not on psi = psi_b",
            contour.r[i],
            contour.z[i]
        );
        assert!(
            contour.z[i].abs() <= z_xpt,
            "point {i}, ({}, {}), is beyond the X-points",
            contour.r[i],
            contour.z[i]
        );
    }

    // No two segments cross, so the contour does not loop back on itself
    for i in 0..n - 1 {
        for j in (i + 2)..n - 1 {
            assert!(
                !segments_intersect(
                    contour.r[i],
                    contour.z[i],
                    contour.r[i + 1],
                    contour.z[i + 1],
                    contour.r[j],
                    contour.z[j],
                    contour.r[j + 1],
                    contour.z[j + 1]
                ),
                "segments {i} and {j} cross"
            );
        }
    }

    // It encloses the whole plasma, both tips included
    let mut area: f64 = 0.0;
    for i in 0..n - 1 {
        area += contour.r[i] * contour.z[i + 1] - contour.r[i + 1] * contour.z[i];
    }
    area = 0.5 * area.abs();
    assert!(
        (area / area_expected - 1.0).abs() < 1.0e-2,
        "enclosed area {area} differs from the analytic {area_expected}"
    );
}
