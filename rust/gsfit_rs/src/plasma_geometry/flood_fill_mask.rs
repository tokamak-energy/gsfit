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
/// * `r` - R grid points, metre
/// * `z` - Z grid points, metre
/// * `psi_2d` - poloidal flux, shape = (n_z, n_r), weber
/// * `psi_b` - poloidal flux at the boundary, weber
/// * `stationary_points` - Vector of `StationaryPoint` objects (including maxima/minima, and can include saddle points)
/// * `mag_r_previous` - R coordinate of magnetic axis from previous iteration, metre
/// * `mag_z_previous` - Z coordinate of magnetic axis from previous iteration, metre
/// * `vessel_r` - R coordinates of vessel points, metre
/// * `vessel_z` - Z coordinates of vessel points, metre
///
/// # Returns
/// * `mask_2d` - 1.0 = inside plasma boundary, 0.0 for points outside plasma boundary; f64 to make multiplication easier, shape = (n_z, n_r), dimensionless
///
pub fn flood_fill_mask(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>, // Might need `br_2d`, `bz_2d`, and others for contouring?
    psi_b: f64,
    stationary_points: &Vec<StationaryPoint>,
    mag_r_previous: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
    mag_z_previous: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Array2<f64> {
    // TODO 1: Is there a problem left/right of the x-point?
    // TODO 2: Perhaps start the fill between the x-point and the previous magnetic axis location,
    // TODO 2: this way if the magnetic axis moves vertically a lot we won't lose the plasma?

    // TODO 3: The test for crossing the saddle point should only apply if the saddle point is outside

    // Filter stationary points to only include saddle points
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.clone();
    stationary_points.retain(|stationary_point| {
        let saddle_point_test: bool = stationary_point.hessian_determinant < 0.0;
        return saddle_point_test;
    });

    // Label points we should not cross
    let mut indicies_do_not_cross: HashMap<(usize, usize), ()> = HashMap::new();
    for stationary_point in stationary_points.iter() {
        // Find the nearest grid point to the stationary point
        let i_r_nearest: usize = stationary_point.i_r_nearest;
        let i_z_nearest: usize = stationary_point.i_z_nearest;
        let i_z_nearest_lower: usize = stationary_point.i_z_nearest_lower;
        let i_z_nearest_upper: usize = stationary_point.i_z_nearest_upper;

        // Apply a **VERY** simple z.max() masking
        // TODO: improve the masking logic near the x-point
        // x-point could conceivably escape left/right too
        if z[i_z_nearest] > 0.0 {
            if i_r_nearest > 1 {
                indicies_do_not_cross.insert((i_z_nearest_upper, i_r_nearest - 2), ());
            }
            if i_r_nearest > 0 {
                indicies_do_not_cross.insert((i_z_nearest_upper, i_r_nearest - 1), ());
            }
            indicies_do_not_cross.insert((i_z_nearest_upper, i_r_nearest), ());
            if i_r_nearest < r.len() - 1 {
                indicies_do_not_cross.insert((i_z_nearest_upper, i_r_nearest + 1), ());
            }
            if i_r_nearest < r.len() - 2 {
                indicies_do_not_cross.insert((i_z_nearest_upper, i_r_nearest + 2), ());
            }
        }
        if z[i_z_nearest] < 0.0 {
            if i_r_nearest > 1 {
                indicies_do_not_cross.insert((i_z_nearest_lower, i_r_nearest - 2), ());
            }
            if i_r_nearest > 0 {
                indicies_do_not_cross.insert((i_z_nearest_lower, i_r_nearest - 1), ());
            }
            indicies_do_not_cross.insert((i_z_nearest_lower, i_r_nearest), ());
            if i_r_nearest < r.len() - 1 {
                indicies_do_not_cross.insert((i_z_nearest_lower, i_r_nearest + 1), ());
            }
            if i_r_nearest < r.len() - 2 {
                indicies_do_not_cross.insert((i_z_nearest_lower, i_r_nearest + 2), ());
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
    if mask_vessel_2d[(i_z_nearest_mag, i_r_nearest_mag)] == false {
        return mask_2d;
    }

    // Start the flood fill algorithm, going anticlockwise from the magnetic axis
    // Directions: right, up, left, down (anticlockwise)
    let directions: [(isize, isize); 4] = [
        (0, 1),  // right (i_z, i_r+1)
        (-1, 0), // up    (i_z-1, i_r)
        (0, -1), // left  (i_z, i_r-1)
        (1, 0),  // down  (i_z+1, i_r)
    ];

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((i_z_nearest_mag, i_r_nearest_mag));
    mask_2d[(i_z_nearest_mag, i_r_nearest_mag)] = 1.0;

    while let Some((i_z, i_r)) = queue.pop_front() {
        for &(dz, dr) in &directions {
            let new_i_z: isize = i_z as isize + dz;
            let new_i_r: isize = i_r as isize + dr;

            // Check bounds
            if new_i_z < 0 || new_i_z >= n_z as isize || new_i_r < 0 || new_i_r >= n_r as isize {
                continue;
            }

            let new_i_z: usize = new_i_z as usize;
            let new_i_r: usize = new_i_r as usize;

            // Respect the vessel wall during the flood fill
            // This prevents the fill from painting the centre post and wrapping all the way round to the opposite Z
            if mask_vessel_2d[(new_i_z, new_i_r)] == false {
                continue;
            }

            // Check if we are going past a saddle point
            if indicies_do_not_cross.contains_key(&(new_i_z, new_i_r)) {
                // Don't add this point to the `mask`, and don't add it to the `queue`
                continue;
            }

            if mask_2d[(new_i_z, new_i_r)] == 0.0 && psi_2d[(new_i_z, new_i_r)] > psi_b {
                // `mask` is not allowed to pass a saddle point, regardless of if the plasma is diverted or limited
                if !indicies_do_not_cross.contains_key(&(new_i_z, new_i_r)) {
                    mask_2d[(new_i_z, new_i_r)] = 1.0;
                    queue.push_back((new_i_z, new_i_r));
                }
            }
        }
    }

    // Ensure points outside the vessel are masked out (defensive, should already be zero)
    for (mask_val, inside_vessel) in mask_2d.iter_mut().zip(mask_vessel_2d.iter()) {
        if inside_vessel == &false {
            *mask_val = 0.0;
        }
    }

    return mask_2d;
}

#[test]
fn test_flood_fill_mask() {
    use std::fs;

    let n_r: usize = 80;
    let n_z: usize = 161;
    let r: Array1<f64> = Array1::linspace(0.01, 1.0, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    // Load `psi_2d` from `test_data` file
    let test_data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/flood_fill_mask/psi2d.txt");
    let data_str = fs::read_to_string(test_data_path).expect("Failed to read test data file");
    let v: Vec<f64> = data_str
        .lines()
        .map(|line| line.trim().parse::<f64>().expect("test_flood_fill_mask: Failed to read `psi_2d` from file"))
        .collect();
    assert_eq!(v.len(), 12880, "test_flood_fill_mask: `psi_2d` test data should have 12880 values");
    let psi_2d: Array2<f64> = Array2::from_shape_vec((n_z, n_r), v).expect("Failed to create array from test data");

    // Load `vessel_r` from `test_data` file
    let test_data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/flood_fill_mask/vessel_r.txt");
    let data_str = fs::read_to_string(test_data_path).expect("Failed to read test data file");
    let v: Vec<f64> = data_str
        .lines()
        .map(|line| line.trim().parse::<f64>().expect("test_flood_fill_mask: Failed to read `vessel_r` from file"))
        .collect();
    assert_eq!(v.len(), 150, "test_flood_fill_mask: `vessel_r` test data should have 150 values");
    let vessel_r: Array1<f64> = Array1::from_shape_vec(150, v).expect("Failed to create array from test data");

    // Load `vessel_z` from `test_data` file
    let test_data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/flood_fill_mask/vessel_z.txt");
    let data_str = fs::read_to_string(test_data_path).expect("Failed to read test data file");
    let v: Vec<f64> = data_str
        .lines()
        .map(|line| line.trim().parse::<f64>().expect("test_flood_fill_mask: Failed to read `vessel_z` from file"))
        .collect();
    assert_eq!(v.len(), 150, "test_flood_fill_mask: `vessel_z` test data should have 150 values");
    let vessel_z: Array1<f64> = Array1::from_shape_vec(150, v).expect("Failed to create array from test data");

    let psi_b: f64 = -0.05901706777528778;
    let _xpt_r: f64 = 0.1;
    let _xpt_z: f64 = -0.5713182906388565;
    let stationary_points: Vec<StationaryPoint> = vec![];
    let mag_r_previous: f64 = 0.5115792196972574;
    let mag_z_previous: f64 = -0.007343976139471093;

    let mask_2d: Array2<f64> = flood_fill_mask(&r, &z, &psi_2d, psi_b, &stationary_points, mag_r_previous, mag_z_previous, &vessel_r, &vessel_z);

    let painted_cells: f64 = mask_2d.sum();
    assert_eq!(
        painted_cells, 2776.0,
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
