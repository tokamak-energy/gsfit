use super::StationaryPoint;
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Flood fill algorithm to create a mask of points inside the plasma boundary
/// starting from the magnetic axis location
/// * mask=0.0 --> outside the plasma
/// * mask=1.0 --> inside the plasma
///
/// # Arguments
/// * `r` - R grid points, [metre]
/// * `z` - Z grid points, [metre]
/// * `psi_2d` - poloidal flux, shape = (n_z, n_r), [weber]
/// * `psi_b` - poloidal flux at the boundary, [weber]
/// * `stationary_points` - Vector of `StationaryPoint` objects (including maxima/minima, and can include saddle points)
/// * `mag_r_previous` - R coordinate of magnetic axis from previous iteration, [metre]
/// * `mag_z_previous` - Z coordinate of magnetic axis from previous iteration, [metre]
/// * `vessel_r` - R coordinates of vessel points, [metre]
/// * `vessel_z` - Z coordinates of vessel points, [metre]
///
/// # Returns
/// * `mask_2d` - 1.0 = inside plasma boundary, 0.0 for points outside plasma boundary; f64 to make multiplication easier, shape = (n_z, n_r), dimensionless
///
/// # Algorithm
/// This is a Breadth-First Search (BFS) flood fill using 4 neighbour connectivity
pub fn flood_fill_mask(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>, // Might need `br_2d`, `bz_2d`, and others for contouring?
    psi_b: f64,
    stationary_points: &[StationaryPoint],
    mag_r_previous: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
    mag_z_previous: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Array2<f64> {
    // TODO 1: Is there a problem left/right of the x-point?
    // TODO 2: Perhaps start the fill between the x-point and the previous magnetic axis location,
    // TODO 2: this way if the magnetic axis moves vertically a lot we won't lose the plasma?
    // TODO 3: The test for crossing the saddle point should only apply if the saddle point is outside

    // Make a mutable copy of the `stationary_points` vector, so that we can filter out points
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.to_vec();

    // Find the maximum delta_psi between two grid points
    let n_z_grid: usize = psi_2d.nrows();
    let n_r_grid: usize = psi_2d.ncols();
    let mut max_psi_separation: f64 = 0.0_f64;
    for i_z in 0..n_z_grid {
        for i_r in 0..n_r_grid - 1 {
            max_psi_separation = max_psi_separation.max((psi_2d[(i_z, i_r + 1)] - psi_2d[(i_z, i_r)]).abs());
        }
    }
    for i_z in 0..n_z_grid - 1 {
        for i_r in 0..n_r_grid {
            max_psi_separation = max_psi_separation.max((psi_2d[(i_z + 1, i_r)] - psi_2d[(i_z, i_r)]).abs());
        }
    }

    // Filter out non-saddle points, e.g. magnetic axis
    stationary_points.retain(|stationary_point| {
        let saddle_point_test: bool = stationary_point.hessian_determinant < 0.0;

        saddle_point_test
    });

    // Filter out saddle points which are not near the boundary, e.g. inside PF coils
    stationary_points.retain(|stationary_point| {
        let near_boundary_test: bool = (stationary_point.psi - psi_b).abs() < 2.0 * max_psi_separation;

        near_boundary_test
    });

    // Label points we should not cross
    let mut indices_do_not_cross: HashMap<(usize, usize), ()> = HashMap::new();
    for stationary_point in stationary_points.iter() {
        // Find the nearest grid point to the stationary point
        let i_r_nearest: usize = stationary_point.i_r_nearest;
        let i_z_nearest: usize = stationary_point.i_z_nearest;
        let i_z_nearest_lower: usize = stationary_point.i_z_lower;
        let i_z_nearest_upper: usize = stationary_point.i_z_upper;

        // Apply a **VERY** simple z.max() masking
        // TODO: improve the masking logic near the x-point
        // x-point could conceivably escape left/right too
        if z[i_z_nearest] > 0.0 {
            if i_r_nearest > 1 {
                indices_do_not_cross.insert((i_z_nearest_upper, i_r_nearest - 2), ());
            }
            if i_r_nearest > 0 {
                indices_do_not_cross.insert((i_z_nearest_upper, i_r_nearest - 1), ());
            }
            indices_do_not_cross.insert((i_z_nearest_upper, i_r_nearest), ());
            if i_r_nearest < r.len() - 1 {
                indices_do_not_cross.insert((i_z_nearest_upper, i_r_nearest + 1), ());
            }
            if i_r_nearest < r.len() - 2 {
                indices_do_not_cross.insert((i_z_nearest_upper, i_r_nearest + 2), ());
            }
        }
        if z[i_z_nearest] < 0.0 {
            if i_r_nearest > 1 {
                indices_do_not_cross.insert((i_z_nearest_lower, i_r_nearest - 2), ());
            }
            if i_r_nearest > 0 {
                indices_do_not_cross.insert((i_z_nearest_lower, i_r_nearest - 1), ());
            }
            indices_do_not_cross.insert((i_z_nearest_lower, i_r_nearest), ());
            if i_r_nearest < r.len() - 1 {
                indices_do_not_cross.insert((i_z_nearest_lower, i_r_nearest + 1), ());
            }
            if i_r_nearest < r.len() - 2 {
                indices_do_not_cross.insert((i_z_nearest_lower, i_r_nearest + 2), ());
            }
        }
    }

    // Grid sizes
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let mut mask_2d: Array2<f64> = Array2::from_elem((n_z, n_r), 0.0);

    // Pre-compute a mask for points that lie inside the vessel so we can stop flood fill steps at the wall
    let n_vessel_pts: usize = vessel_r.len();
    let mut vessel_coords: Vec<Coord<f64>> = Vec::with_capacity(n_vessel_pts);
    for i_vessel in 0..n_vessel_pts {
        vessel_coords.push(Coord {
            x: vessel_r[i_vessel],
            y: vessel_z[i_vessel],
        });
    }
    let vessel_polygon: Polygon = Polygon::new(
        LineString::from(vessel_coords),
        vec![], // No holes
    );
    let mut mask_vessel_2d: Array2<bool> = Array2::from_elem((n_z, n_r), false);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let test_point: Point = Point::new(r[i_r], z[i_z]);
            mask_vessel_2d[(i_z, i_r)] = vessel_polygon.contains(&test_point);
        }
    }

    // Find the index of the grid point closest to the magnetic axis
    let i_r_nearest_mag: usize = (r - mag_r_previous).abs().argmin().unwrap();
    let i_z_nearest_mag: usize = (z - mag_z_previous).abs().argmin().unwrap();

    // Add a test that the magnetic axis is higher than psi_b
    if psi_2d[(i_z_nearest_mag, i_r_nearest_mag)] < psi_b {
        return mask_2d;
    }

    // Magnetic axis must be inside the vessel, otherwise we cannot start the fill
    if !mask_vessel_2d[(i_z_nearest_mag, i_r_nearest_mag)] {
        return mask_2d;
    }

    // Start the flood fill algorithm, going anticlockwise from the magnetic axis
    // Directions: right, down, left, up (anticlockwise)
    let directions: [(isize, isize); 4] = [
        (0, 1),  // right (i_z, i_r+1)
        (-1, 0), // down  (i_z-1, i_r)
        (0, -1), // left  (i_z, i_r-1)
        (1, 0),  // up    (i_z+1, i_r)
    ];

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((i_z_nearest_mag, i_r_nearest_mag));
    mask_2d[(i_z_nearest_mag, i_r_nearest_mag)] = 1.0;

    while let Some((i_z, i_r)) = queue.pop_front() {
        'loop_over_directions: for &(dz, dr) in &directions {
            let new_i_z: isize = i_z as isize + dz;
            let new_i_r: isize = i_r as isize + dr;

            // Check bounds
            if new_i_z < 0 || new_i_z >= n_z as isize || new_i_r < 0 || new_i_r >= n_r as isize {
                continue 'loop_over_directions;
            }

            let new_i_z: usize = new_i_z as usize;
            let new_i_r: usize = new_i_r as usize;

            // Respect the vessel wall during the flood fill
            // This prevents the fill from painting the centre post and wrapping all the way round to the opposite Z
            if !mask_vessel_2d[(new_i_z, new_i_r)] {
                continue 'loop_over_directions;
            }

            // Check if we are going past a saddle point
            if indices_do_not_cross.contains_key(&(new_i_z, new_i_r)) {
                // Don't add this point to the `mask`, and don't add it to the `queue`
                continue 'loop_over_directions;
            }

            if mask_2d[(new_i_z, new_i_r)] == 0.0 && psi_2d[(new_i_z, new_i_r)] > psi_b {
                // `mask` is not allowed to pass a saddle point, regardless of if the plasma is diverted or limited
                if !indices_do_not_cross.contains_key(&(new_i_z, new_i_r)) {
                    mask_2d[(new_i_z, new_i_r)] = 1.0;
                    queue.push_back((new_i_z, new_i_r));
                }
            }
        }
    }

    // Note: this is handled in the flood fill while loop.
    // // Ensure points outside the vessel are masked out (defensive, should already be zero)
    // for (mask_val, inside_vessel) in mask_2d.iter_mut().zip(mask_vessel_2d.iter()) {
    //     if !inside_vessel {
    //         *mask_val = 0.0;
    //     }
    // }

    mask_2d
}

#[test]
fn test_flood_fill_mask() {
    use npy_reader_and_writer;

    let n_r: usize = 80;
    let n_z: usize = 161;
    let r: Array1<f64> = Array1::linspace(0.01, 1.0, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    // Load `psi_2d` from `test_data` file
    let test_data_path: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/plasma_geometry/flood_fill_mask/psi_2d.npy");
    let psi_2d: Array2<f64> = npy_reader_and_writer::read_npy_2d(std::path::Path::new(test_data_path));

    // Load `vessel_r` from `test_data` file
    let test_data_path: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/plasma_geometry/flood_fill_mask/vessel_r.npy");
    let vessel_r: Array1<f64> = npy_reader_and_writer::read_npy_1d(std::path::Path::new(test_data_path));

    // Load `vessel_z` from `test_data` file
    let test_data_path: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/plasma_geometry/flood_fill_mask/vessel_z.npy");
    let vessel_z: Array1<f64> = npy_reader_and_writer::read_npy_1d(std::path::Path::new(test_data_path));

    let psi_b: f64 = -0.05901706777528778;
    let _xpt_r: f64 = 0.1;
    let _xpt_z: f64 = -0.5713182906388565;
    let stationary_points: Vec<StationaryPoint> = vec![];
    let mag_r_previous: f64 = 0.5115792196972574;
    let mag_z_previous: f64 = -0.007343976139471093;

    let mask_2d: Array2<f64> = flood_fill_mask(&r, &z, &psi_2d, psi_b, &stationary_points, mag_r_previous, mag_z_previous, &vessel_r, &vessel_z);

    let total_number_of_painted_cells: f64 = mask_2d.sum();
    assert_eq!(
        total_number_of_painted_cells, 2776.0,
        "test_flood_fill_mask: mask should only paint the plasma interior (2776 cells)",
    );

    let mut leaking_cells_above_0p6: f64 = 0.0;
    for (i_z, z_val) in z.iter().enumerate() {
        if *z_val > 0.6 {
            for i_r in 0..n_r {
                leaking_cells_above_0p6 += mask_2d[(i_z, i_r)];
            }
        }
    }
    assert_eq!(leaking_cells_above_0p6, 0.0, "test_flood_fill_mask: flood fill leaked above z=0.6 m",);
}
