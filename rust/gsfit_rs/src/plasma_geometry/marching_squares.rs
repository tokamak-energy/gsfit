use std::collections::HashMap;

use super::cubic_interpolation::cubic_interpolation;
use approx::abs_diff_eq;
use ndarray::{Array1, Array2, ArrayBase};

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct BoundaryContourNew {
    pub r: Array1<f64>,
    pub z: Array1<f64>,
    pub n: usize,
}

pub fn marching_squares(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    psi_b: f64,
    mask_2d: &Array2<f64>,
    xpt_r_or_none: Option<f64>,
    xpt_z_or_none: Option<f64>,
    mag_r: f64,
    mag_z: f64,
) -> BoundaryContourNew {
    let n_z: usize = z.len();
    let n_r: usize = r.len();

    // HashMap key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // HashMap value: (r_cross, z_cross)
    let mut unsorted_boundary_points: HashMap<(usize, usize, usize, usize), (f64, f64)> = HashMap::new();

    // March from left to right
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            // Look for where `mask_2d` changes from 0 to 1 or 1 to 0
            if abs_diff_eq!(mask_2d[(i_z, i_r)] + mask_2d[(i_z, i_r + 1)], 1.0) {
                let left_r: f64 = r[i_r];
                let left_psi: f64 = psi_2d[(i_z, i_r)];
                let left_d_psi_d_r: f64 = bz_2d[(i_z, i_r)] * (2.0 * PI * r[i_r]);
                let right_r: f64 = r[i_r + 1];
                let right_psi: f64 = psi_2d[(i_z, i_r + 1)];
                let right_d_psi_d_r: f64 = bz_2d[(i_z, i_r + 1)] * (2.0 * PI * r[i_r + 1]);

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
                let bottom_d_psi_d_z: f64 = -br_2d[(i_z, i_r)] * (2.0 * PI * r[i_r]);
                let top_z: f64 = z[i_z + 1];
                let top_psi: f64 = psi_2d[(i_z + 1, i_r)];
                let top_d_psi_d_z: f64 = -br_2d[(i_z + 1, i_r)] * (2.0 * PI * r[i_r]);

                let cubic_interpolation_or_error: Result<Array1<f64>, String> =
                    cubic_interpolation(bottom_z, bottom_psi, bottom_d_psi_d_z, top_z, top_psi, top_d_psi_d_z, psi_b);
                if cubic_interpolation_or_error.is_ok() {
                    let cubic_interpolation_z: Array1<f64> = cubic_interpolation_or_error.unwrap();
                    if cubic_interpolation_z.len() != 1 {
                        println!("Warning: Found {} crossing points, in bottom to top march", cubic_interpolation_z.len());
                    }
                    let r_cross: f64 = r[i_r];
                    let z_cross: f64 = cubic_interpolation_z[0];
                    // note, this key and value structure assumes there is only one crossing per cell edge
                    unsorted_boundary_points.insert((i_r, i_z, i_r, i_z + 1), (r_cross, z_cross));
                }
            }
        }
    }

    // If x-point is not provided, we are limited
    if xpt_r_or_none.is_none() || xpt_z_or_none.is_none() {
        // Collect boundary points from the HashMap
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
        let boundary_contour: BoundaryContourNew = BoundaryContourNew {
            r: Array1::from_vec(boundary_r_sorted),
            z: Array1::from_vec(boundary_z_sorted),
            n: n,
        };

        return boundary_contour;
    }

    // From now on we are handling an x-point diverted plasma

    // Unwrap the x-point coordinates
    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    // Find the closest grid point
    let mut i_r_nearest_xpt: usize = 0;
    let mut min_r_dist: f64 = f64::INFINITY;
    for (i, &r_val) in r.iter().enumerate() {
        let dist: f64 = (xpt_r - r_val).abs();
        if dist < min_r_dist {
            min_r_dist = dist;
            i_r_nearest_xpt = i;
        }
    }
    let mut i_z_nearest_xpt: usize = 0;
    let mut min_z_dist: f64 = f64::INFINITY;
    for (i, &z_val) in z.iter().enumerate() {
        let dist: f64 = (xpt_z - z_val).abs();
        if dist < min_z_dist {
            min_z_dist = dist;
            i_z_nearest_xpt = i;
        }
    }

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

    // Remove the connections between the four grid points from the boundary HashMap, which enclose the x-point
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

    // println!("left_r={}, right_r={}", r[i_r_nearest_xpt_left], r[i_r_nearest_xpt_right]);
    // println!("lower_z={}, upper_z={}", z[i_z_nearest_xpt_lower], z[i_z_nearest_xpt_upper]);

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
            let left_d_psi_d_r: f64 = bz_2d[(i_z_from, i_r_from)] * (2.0 * PI * r[i_r_from]);
            let right_r: f64 = r[i_r_to];
            let right_psi: f64 = psi_2d[(i_z_from, i_r_to)];
            let right_d_psi_d_r: f64 = bz_2d[(i_z_from, i_r_to)] * (2.0 * PI * r[i_r_to]);
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
            let bottom_d_psi_d_z: f64 = -br_2d[(i_z_from, i_r_from)] * (2.0 * PI * r[i_r_from]);
            let top_z: f64 = z[i_z_to];
            let top_psi: f64 = psi_2d[(i_z_to, i_r_from)];
            let top_d_psi_d_z: f64 = -br_2d[(i_z_to, i_r_from)] * (2.0 * PI * r[i_r_from]);
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

    // println!("boundary_points_near_xpt_r={:?}", boundary_points_near_xpt_r);
    // println!("boundary_points_near_xpt_z={:?}", boundary_points_near_xpt_z);

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

    // Collect remaining boundary points from the HashMap
    let unsorted_boundary_r: Vec<f64> = unsorted_boundary_points.values().map(|&(r_val, _)| r_val).collect();
    let unsorted_boundary_z: Vec<f64> = unsorted_boundary_points.values().map(|&(_, z_val)| z_val).collect();

    let mut boundary_sorted_r: Vec<f64> = Vec::new();
    let mut boundary_sorted_z: Vec<f64> = Vec::new();

    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    boundary_sorted_r.push(xpt_r);
    boundary_sorted_z.push(xpt_z);
    boundary_sorted_r.push(first_and_last_boundary_points[0].0);
    boundary_sorted_z.push(first_and_last_boundary_points[0].1);

    // Sort the boundary points using nearest-neighbor algorithm
    let (mut boundary_sorted_r, mut boundary_sorted_z): (Vec<f64>, Vec<f64>) =
        sort_boundary_points(boundary_sorted_r, boundary_sorted_z, &unsorted_boundary_r, &unsorted_boundary_z);

    // Add the last point to close the contour
    boundary_sorted_r.push(first_and_last_boundary_points[1].0);
    boundary_sorted_z.push(first_and_last_boundary_points[1].1);

    // Add the x-point
    boundary_sorted_r.push(xpt_r);
    boundary_sorted_z.push(xpt_z);

    let n: usize = boundary_sorted_r.len();
    let boundary_contour: BoundaryContourNew = BoundaryContourNew {
        r: Array1::from_vec(boundary_sorted_r),
        z: Array1::from_vec(boundary_sorted_z),
        n: n,
    };

    // Repeat, but starting from the other side of the x-point
    // Collect remaining boundary points from the HashMap
    let unsorted_boundary_r: Vec<f64> = unsorted_boundary_points.values().map(|&(r_val, _)| r_val).collect();
    let unsorted_boundary_z: Vec<f64> = unsorted_boundary_points.values().map(|&(_, z_val)| z_val).collect();

    let mut boundary_sorted_r: Vec<f64> = Vec::new();
    let mut boundary_sorted_z: Vec<f64> = Vec::new();

    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    boundary_sorted_r.push(xpt_r);
    boundary_sorted_z.push(xpt_z);
    boundary_sorted_r.push(first_and_last_boundary_points[1].0);
    boundary_sorted_z.push(first_and_last_boundary_points[1].1);

    // Sort the boundary points using nearest-neighbor algorithm
    let (mut boundary_sorted_r, mut boundary_sorted_z): (Vec<f64>, Vec<f64>) =
        sort_boundary_points(boundary_sorted_r, boundary_sorted_z, &unsorted_boundary_r, &unsorted_boundary_z);

    // Add the last point to close the contour
    boundary_sorted_r.push(first_and_last_boundary_points[0].0);
    boundary_sorted_z.push(first_and_last_boundary_points[0].1);

    // Add the x-point
    boundary_sorted_r.push(xpt_r);
    boundary_sorted_z.push(xpt_z);

    let n: usize = boundary_sorted_r.len();
    let boundary_contour_other_direction: BoundaryContourNew = BoundaryContourNew {
        r: Array1::from_vec(boundary_sorted_r),
        z: Array1::from_vec(boundary_sorted_z),
        n: n,
    };

    // Check if the boundary contour crosses itself
    let boundary_contour_crosses_itself: bool = contour_crosses_itself(&boundary_contour.r.to_vec(), &boundary_contour.z.to_vec());
    let boundary_contour_other_direction_crosses_itself: bool =
        contour_crosses_itself(&boundary_contour_other_direction.r.to_vec(), &boundary_contour_other_direction.z.to_vec());

    if !boundary_contour_crosses_itself {
        return boundary_contour;
    } else if !boundary_contour_other_direction_crosses_itself {
        return boundary_contour_other_direction;
    } else {
        println!("Warning: Both boundary contours cross themselves, returning the first one");
        return boundary_contour_other_direction;
    }
}

/// Check if a contour crosses itself by testing for line segment intersections
/// Returns true if any two non-adjacent segments intersect
fn contour_crosses_itself(r: &Vec<f64>, z: &Vec<f64>) -> bool {
    let n = r.len();
    if n < 4 {
        // Can't have self-intersection with fewer than 4 points
        return false;
    }

    // Check each pair of non-adjacent segments
    for i in 0..n - 1 {
        let (r1, z1) = (r[i], z[i]);
        let (r2, z2) = (r[i + 1], z[i + 1]);

        // Check against all non-adjacent segments
        // Start at i+2 to skip the adjacent segment
        let start_j = if i == 0 { i + 2 } else { i + 2 };
        let end_j = if i == n - 2 { n - 2 } else { n - 1 };

        for j in start_j..end_j {
            // Skip if segments are adjacent or wrap-around adjacent
            if j == i || j == i + 1 || (i == 0 && j == n - 2) {
                continue;
            }

            let (r3, z3) = (r[j], z[j]);
            let (r4, z4) = (r[j + 1], z[j + 1]);

            if segments_intersect(r1, z1, r2, z2, r3, z3, r4, z4) {
                return true;
            }
        }
    }

    false
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

/// Sort boundary points in counter-clockwise order using nearest-neighbor algorithm
///
/// Starting point:
/// - If x-point is provided, start from the x-point
/// - Otherwise, start from the lowest point (minimum z)
///
/// Algorithm:
/// - At each step, find the nearest unvisited point
/// - Continue until all points are visited
/// - Ensures counter-clockwise ordering by always taking the shortest step
fn sort_boundary_points(sorted_r: Vec<f64>, sorted_z: Vec<f64>, unsorted_boundary_r: &Vec<f64>, unsorted_boundary_z: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    // let n_points: usize = unsorted_boundary_r.len();

    // Create mutable copies
    let mut sorted_r: Vec<f64> = sorted_r;
    let mut sorted_z: Vec<f64> = sorted_z;
    let mut unsorted_boundary_r: Vec<f64> = unsorted_boundary_r.clone();
    let mut unsorted_boundary_z: Vec<f64> = unsorted_boundary_z.clone();

    // Start from the last point in sorted lists
    let mut current_r: f64 = sorted_r.last().copied().unwrap();
    let mut current_z: f64 = sorted_z.last().copied().unwrap();

    // Iteratively find nearest unvisited point
    while !unsorted_boundary_r.is_empty() {
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
