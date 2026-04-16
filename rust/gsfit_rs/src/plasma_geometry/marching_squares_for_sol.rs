use super::MarchingContour;
use super::cubic_interpolation::cubic_interpolation;
use approx::abs_diff_eq;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Contains, Coord, Line, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use core::f64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Given a cell with lower-left corner at (cell_i_r, cell_i_z), return its 4 edge keys.
/// Edges are: bottom, top, left, right.
fn edges_of_cell(cell_i_r: usize, cell_i_z: usize) -> [(usize, usize, usize, usize); 4] {
    [
        (cell_i_r, cell_i_z, cell_i_r + 1, cell_i_z),         // bottom (horizontal)
        (cell_i_r, cell_i_z + 1, cell_i_r + 1, cell_i_z + 1), // top (horizontal)
        (cell_i_r, cell_i_z, cell_i_r, cell_i_z + 1),         // left (vertical)
        (cell_i_r + 1, cell_i_z, cell_i_r + 1, cell_i_z + 1), // right (vertical)
    ]
}

/// Given an edge key, return the (at most 2) cells that share this edge.
/// Each cell is identified by its lower-left corner (i_r, i_z).
fn cells_sharing_edge(edge: (usize, usize, usize, usize), n_r: usize, n_z: usize) -> Vec<(usize, usize)> {
    let (i_r_from, i_z_from, i_r_to, i_z_to) = edge;
    let mut cells: Vec<(usize, usize)> = Vec::new();

    if i_z_from == i_z_to {
        // Horizontal edge: (i_r_from, i_z) -- (i_r_from+1, i_z)
        let i_r_min: usize = i_r_from.min(i_r_to);
        let i_z: usize = i_z_from;
        // Cell above: lower-left at (i_r_min, i_z)
        if i_z < n_z - 1 {
            cells.push((i_r_min, i_z));
        }
        // Cell below: lower-left at (i_r_min, i_z - 1)
        if i_z > 0 {
            cells.push((i_r_min, i_z - 1));
        }
    } else {
        // Vertical edge: (i_r, i_z_from) -- (i_r, i_z_from+1)
        let i_r: usize = i_r_from;
        let i_z_min: usize = i_z_from.min(i_z_to);
        // Cell to the right: lower-left at (i_r, i_z_min)
        if i_r < n_r - 1 {
            cells.push((i_r, i_z_min));
        }
        // Cell to the left: lower-left at (i_r - 1, i_z_min)
        if i_r > 0 {
            cells.push((i_r - 1, i_z_min));
        }
    }

    cells
}

pub fn marching_squares_for_sol(
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
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> (Result<MarchingContour, String>, Result<MarchingContour, String>) {
    let n_z: usize = z.len();
    let n_r: usize = r.len();

    // HashMap key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // HashMap value: (r_cross, z_cross)
    let mut unsorted_boundary_points: HashMap<(usize, usize, usize, usize), (f64, f64)> = HashMap::new();

    // March from left to right
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            // Look for where `mask_2d` changes from 0 to 1 or 1 to 0
            if mask_2d[(i_z, i_r)] + mask_2d[(i_z, i_r + 1)] > 0.9 && mask_2d[(i_z, i_r)] + mask_2d[(i_z, i_r + 1)] < 1.1 {
                // if abs_diff_eq!(mask_2d[(i_z, i_r)] + mask_2d[(i_z, i_r + 1)], 1.0) {
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
                } else {
                    println!(
                        "Cubic interpolation error in left to right march at i_r = {}, i_z = {}: {}",
                        i_r,
                        i_z,
                        cubic_interpolation_or_error.err().unwrap()
                    );
                    println!(
                        "Cubic interpolation error in left to right march, left_r = {}, left_psi = {}, left_d_psi_d_r = {}, right_r = {}, right_psi = {}, right_d_psi_d_r = {}, psi_b = {}",
                        left_r, left_psi, left_d_psi_d_r, right_r, right_psi, right_d_psi_d_r, psi_b
                    );
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

                    // TODO: after some testing, it would appear that the difference between cubic_interpolation_z[0] and cubic_interpolation_z[1]
                    // doesn't affect anything? But this still needs investigating.
                    let r_cross: f64 = r[i_r];
                    let z_cross: f64 = cubic_interpolation_z[0];
                    // note, this key and value structure assumes there is only one crossing per cell edge
                    unsorted_boundary_points.insert((i_r, i_z, i_r, i_z + 1), (r_cross, z_cross));
                }
            }
        }
    }

    let empty_marching_contour: MarchingContour = MarchingContour {
        r: Array1::from_elem(0, f64::NAN),
        z: Array1::from_elem(0, f64::NAN),
        n: 0,
    };

    // If empty
    if unsorted_boundary_points.is_empty() {
        return (Err("No boundary points found".to_string()), Err("No boundary points found".to_string()));
    }

    // If x-point is not provided, we are limited
    if xpt_r_or_none.is_none() || xpt_z_or_none.is_none() {
        return (Err("X-point not provided".to_string()), Err("X-point not provided".to_string()));
    }

    // From now on we are handling an x-point diverted plasma

    // Unwrap the x-point coordinates
    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    // Find the closest grid point
    let i_r_nearest_xpt: usize = (r - xpt_r).abs().argmin().unwrap();
    let i_z_nearest_xpt: usize = (z - xpt_z).abs().argmin().unwrap();
    // Guard against underflow, when x-point is on grid boundary
    if i_r_nearest_xpt==0 || i_r_nearest_xpt==n_r-1 || i_z_nearest_xpt==0 || i_z_nearest_xpt==n_z-1 {
        return (Err("X-point is on the grid boundary, which is not supported".to_string()), Err("X-point is on the grid boundary, which is not supported".to_string()));
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
    // Remove points
    for cell in &cells_to_remove {
        unsorted_boundary_points.remove(cell);
    }

    // The maximum number of points is 12 (cubic polynomial, and 4 edges)
    // A regular saddle point geometry has 4 "legs", i.e. 4 boundary points near the x-point
    // Unless there was an unusual snowflake configuration
    // TODO: implement snowflake geometry
    let mut boundary_points_near_xpt_r: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_z: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_edge: Vec<(usize, usize, usize, usize)> = Vec::new();

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
                    boundary_points_near_xpt_edge.push(cell);
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
                    boundary_points_near_xpt_edge.push(cell);
                }
            }
        }
    }

    // direction vector from x-point to private flux point
    let delta_r: f64 = mag_r - xpt_r;
    let delta_z: f64 = mag_z - xpt_z;
    let d_mag: f64 = (delta_r.powi(2) + delta_z.powi(2)).sqrt();
    // No need to add guard against zero division, the x-point cannot be exactly on the magnetic axis
    // * Magnetic axis is local maximum/minimum in psi (positive/negative plasma current)
    // * X-point is saddle point in psi
    let delta_r_mag_unit: f64 = delta_r / d_mag;
    let delta_z_mag_unit: f64 = delta_z / d_mag;

    // `dot_product = dot(direction_vector_from_x-point_to_candidate_point,  direction_vector_from_x-point_to_magnetic_axis)`
    // * `dot_product = 1.0` means candidate point is pointing directly towards the magnetic axis
    // * `dot_product = -1.0` means candidate point is pointing directly towards the private flux region
    let mut dot_products: Vec<f64> = Vec::new();
    'loop_over_boundary_points: for i_point in 0..boundary_points_near_xpt_r.len() {
        let delta_r: f64 = boundary_points_near_xpt_r[i_point] - xpt_r;
        let delta_z: f64 = boundary_points_near_xpt_z[i_point] - xpt_z;
        let d_xpt: f64 = (delta_r.powi(2) + delta_z.powi(2)).sqrt();
        // It is possible the x-point could be exactly on a grid point
        // The expected values for dot_product are between -1.0 and 1.0.
        // So let's set the dot_product value to 1.0, which means it will be discarded in the sorting step
        // Note: in practice this is extremely unlikely to ever happen
        if d_xpt.abs() < 1e-12 {
            dot_products.push(1.0);

            // Go to next candidate point
            continue 'loop_over_boundary_points;
        }
        let delta_r_xpt_unit: f64 = delta_r / d_xpt;
        let delta_z_xpt_unit: f64 = delta_z / d_xpt;
        let dot_product: f64 = delta_r_xpt_unit * delta_r_mag_unit + delta_z_xpt_unit * delta_z_mag_unit;
        dot_products.push(dot_product);
    }

    let i_sorted: Vec<usize> = {
        let mut indices: Vec<usize> = (0..dot_products.len()).collect();
        // Sort by descending dot product (most parallel first)
        indices.sort_by(|&i, &j| dot_products[j].partial_cmp(&dot_products[i]).unwrap());

        indices
    };
    // Reverse the sort order because we want points which are directed away from the magnetic axis, towards the private flux region
    let i_sorted: Vec<usize> = i_sorted.into_iter().rev().collect();
    let mut boundary_points_near_xpt_r_sorted: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_z_sorted: Vec<f64> = Vec::new();
    let mut boundary_points_near_xpt_edge_sorted: Vec<(usize, usize, usize, usize)> = Vec::new();
    for &i in &i_sorted {
        boundary_points_near_xpt_r_sorted.push(boundary_points_near_xpt_r[i]);
        boundary_points_near_xpt_z_sorted.push(boundary_points_near_xpt_z[i]);
        boundary_points_near_xpt_edge_sorted.push(boundary_points_near_xpt_edge[i]);
    }

    if boundary_points_near_xpt_r_sorted.len() < 2 {
        return (Err("Less than 2 boundary points found near x-point, cannot determine leg directions".to_string()), Err("Less than 2 boundary points found near x-point, cannot determine leg directions".to_string()));
    }

    // The x-point cell, used as the "previous cell" for the first step of contour tracing
    let xpt_cell: (usize, usize) = (i_r_nearest_xpt_left, i_z_nearest_xpt_lower);

    let xpt_r: f64 = xpt_r_or_none.unwrap();
    let xpt_z: f64 = xpt_z_or_none.unwrap();

    let hfs_start_r: f64;
    let hfs_start_z: f64;
    let hfs_start_edge: (usize, usize, usize, usize);
    let lfs_start_r: f64;
    let lfs_start_z: f64;
    let lfs_start_edge: (usize, usize, usize, usize);
    if boundary_points_near_xpt_r_sorted[0] > boundary_points_near_xpt_r_sorted[1] {
        // first point is on the right, so it is the LFS leg
        hfs_start_r = boundary_points_near_xpt_r_sorted[1];
        hfs_start_z = boundary_points_near_xpt_z_sorted[1];
        hfs_start_edge = boundary_points_near_xpt_edge_sorted[1];
        lfs_start_r = boundary_points_near_xpt_r_sorted[0];
        lfs_start_z = boundary_points_near_xpt_z_sorted[0];
        lfs_start_edge = boundary_points_near_xpt_edge_sorted[0];
    } else {
        // first point is on the left, so it is the HFS leg
        hfs_start_r = boundary_points_near_xpt_r_sorted[0];
        hfs_start_z = boundary_points_near_xpt_z_sorted[0];
        hfs_start_edge = boundary_points_near_xpt_edge_sorted[0];
        lfs_start_r = boundary_points_near_xpt_r_sorted[1];
        lfs_start_z = boundary_points_near_xpt_z_sorted[1];
        lfs_start_edge = boundary_points_near_xpt_edge_sorted[1];
    }

    let (hfs_leg, _left_vessel): (MarchingContour, bool) = sort_boundary_points_version_4(
        &unsorted_boundary_points,
        hfs_start_r,
        hfs_start_z,
        hfs_start_edge,
        xpt_r,
        xpt_z,
        xpt_cell,
        vessel_r,
        vessel_z,
        n_r,
        n_z,
    );
    let (lfs_leg, _left_vessel): (MarchingContour, bool) = sort_boundary_points_version_4(
        &unsorted_boundary_points,
        lfs_start_r,
        lfs_start_z,
        lfs_start_edge,
        xpt_r,
        xpt_z,
        xpt_cell,
        vessel_r,
        vessel_z,
        n_r,
        n_z,
    );

    (Ok(hfs_leg), Ok(lfs_leg))
}

/// Trace a divertor leg contour using cell-adjacency from the marching squares grid.
/// Instead of nearest-neighbour (which can jump between legs), this follows the
/// cell-edge connectivity: entering a cell on one edge and exiting on another.
/// Stops when the contour leaves the vessel polygon or runs out of connected edges.
/// Returns (sorted_r, sorted_z, left_vessel).
pub fn sort_boundary_points_version_4(
    unsorted_boundary_points: &HashMap<(usize, usize, usize, usize), (f64, f64)>,
    start_r: f64,
    start_z: f64,
    start_edge: (usize, usize, usize, usize),
    xpt_r: f64,
    xpt_z: f64,
    xpt_cell: (usize, usize),
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    n_r: usize,
    n_z: usize,
) -> (MarchingContour, bool) {
    // Construct the vessel polygon using the geo crate
    let n_vessel: usize = vessel_r.len();
    let mut vessel_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_vessel);
    for i_vessel in 0..n_vessel {
        vessel_coordinates.push(Coord {
            x: vessel_r[i_vessel],
            y: vessel_z[i_vessel],
        });
    }
    let vessel_polygon: Polygon = Polygon::new(
        LineString::new(vessel_coordinates),
        vec![], // No holes
    );

    // Initialize the contour with x-point and the first boundary point
    let mut sorted_r: Vec<f64> = vec![xpt_r, start_r];
    let mut sorted_z: Vec<f64> = vec![xpt_z, start_z];

    let mut current_edge: (usize, usize, usize, usize) = start_edge;
    let mut previous_cell: (usize, usize) = xpt_cell;
    let mut current_r: f64 = start_r;
    let mut current_z: f64 = start_z;

    // Trace the contour by following cell-edge adjacency
    'cell_adjacency_loop: loop {
        // Find the cells that share the current edge
        let adjacent_cells: Vec<(usize, usize)> = cells_sharing_edge(current_edge, n_r, n_z);

        // Pick the cell that is NOT the previous cell (i.e. step into the next cell)
        let mut next_cell: Option<(usize, usize)> = None;
        for cell in &adjacent_cells {
            if *cell != previous_cell {
                next_cell = Some(*cell);
            }
        }

        // No next cell (at grid boundary)
        if next_cell.is_none() {
            break 'cell_adjacency_loop;
        }
        let next_cell: (usize, usize) = next_cell.unwrap();

        // Get the 4 edges of the next cell and collect all candidate exit edges
        let cell_edges: [(usize, usize, usize, usize); 4] = edges_of_cell(next_cell.0, next_cell.1);
        let mut candidates: Vec<((usize, usize, usize, usize), f64, f64)> = Vec::new();
        for edge in &cell_edges {
            // Skip the edge we entered through
            if *edge == current_edge {
                continue;
            }
            // Check if this edge has a boundary crossing
            if let Some(&(candidate_r, candidate_z)) = unsorted_boundary_points.get(edge) {
                candidates.push((*edge, candidate_r, candidate_z));
            }
        }

        if candidates.is_empty() {
            break 'cell_adjacency_loop;
        }

        // If multiple candidate exits, pick the one most aligned with the current direction of travel
        // TODO: consider changing to "closest in psi gradient direction" — pick the candidate whose
        // (current→candidate) direction is most perpendicular to the local (Br, Bz) field, which
        // uses the physics directly and is more robust than geometric smoothness assumptions.
        let (best_edge, best_r, best_z): ((usize, usize, usize, usize), f64, f64) = if candidates.len() == 1 {
            candidates[0]
        } else {
            let n_sorted: usize = sorted_r.len();
            let dir_r: f64 = current_r - sorted_r[n_sorted - 2];
            let dir_z: f64 = current_z - sorted_z[n_sorted - 2];
            let dir_mag: f64 = (dir_r * dir_r + dir_z * dir_z).sqrt();

            let mut best_idx: usize = 0;
            let mut best_dot: f64 = f64::NEG_INFINITY;
            let n_candidates: usize = candidates.len();
            for i_candidate in 0..n_candidates {
                let (_, candidate_r, candidate_z) = candidates[i_candidate];
                let d_r: f64 = candidate_r - current_r;
                let d_z: f64 = candidate_z - current_z;
                let d_mag: f64 = (d_r * d_r + d_z * d_z).sqrt();
                if d_mag > 0.0 && dir_mag > 0.0 {
                    let dot: f64 = (d_r * dir_r + d_z * dir_z) / (d_mag * dir_mag);
                    if dot > best_dot {
                        best_dot = dot;
                        best_idx = i_candidate;
                    }
                }
            }
            candidates[best_idx]
        };

        let next_r: f64 = best_r;
        let next_z: f64 = best_z;

        // Check if this point is outside the vessel
        let test_point: Point = Point::new(next_r, next_z);
        if !vessel_polygon.contains(&test_point) {
            // Find the intersection with the vessel wall
            let sol_segment: Line<f64> = Line::new(Coord { x: current_r, y: current_z }, Coord { x: next_r, y: next_z });
            let mut min_dist_sq: f64 = f64::INFINITY;
            let mut nearest_intersection: Option<Coord<f64>> = None;
            for vessel_segment in vessel_polygon.exterior().lines() {
                if let Some(LineIntersection::SinglePoint { intersection, .. }) = line_intersection(sol_segment, vessel_segment) {
                    let delta_r: f64 = intersection.x - current_r;
                    let delta_z: f64 = intersection.y - current_z;
                    let dist_sq: f64 = delta_r * delta_r + delta_z * delta_z;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest_intersection = Some(intersection);
                    }
                }
            }

            // Add the vessel intersection as the final point
            if let Some(intersection) = nearest_intersection {
                sorted_r.push(intersection.x);
                sorted_z.push(intersection.y);
            } else {
                sorted_r.push(next_r);
                sorted_z.push(next_z);
            }

            let n: usize = sorted_r.len();
            let leg_contour: MarchingContour = MarchingContour {
                r: Array1::from_vec(sorted_r),
                z: Array1::from_vec(sorted_z),
                n,
            };
            return (leg_contour, true);
        }

        // Point is inside the vessel, append and continue
        sorted_r.push(next_r);
        sorted_z.push(next_z);
        current_r = next_r;
        current_z = next_z;
        previous_cell = next_cell;
        current_edge = best_edge;
    }

    let n: usize = sorted_r.len();
    let leg_contour: MarchingContour = MarchingContour {
        r: Array1::from_vec(sorted_r),
        z: Array1::from_vec(sorted_z),
        n,
    };

    let left_vessel: bool = false;
    (leg_contour, left_vessel)
}
