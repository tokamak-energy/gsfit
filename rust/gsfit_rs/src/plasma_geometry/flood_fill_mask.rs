use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::collections::VecDeque;
use super::StationaryPoint;

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
/// * `xpt_r` - R coordinate of x-point, metre
/// * `xpt_z` - Z coordinate of x-point, metre
/// * `xpt_diverted` - true = diverted, false = limited
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
    psi_2d: &Array2<f64>,
    psi_b: f64,
    _xpt_r: f64,
    xpt_z: f64,
    stationary_points: &Vec<StationaryPoint>,
    xpt_diverted: bool,
    mag_r_previous: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
    mag_z_previous: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Array2<f64> {
    // TODO 1: Is there a problem left/right of the x-point?
    // TODO 2: Perhaps start the fill between the x-point and the previous magnetic axis location,
    // TODO 2: this way if the magnetic axis moves vertically a lot we won't lose the plasma?
    // TODO 3: Need to stop the flood fill when it's left the plasma, i.e. crossed the LCFS - somehow it still leaks out???

    // Filter stationary points to only include the saddle points
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.clone();
    stationary_points.retain(|stationary_point| {
        let saddle_point_test: bool = stationary_point.hessian_determinant < 0.0;
        return saddle_point_test;
    });
    let mut indicies_closest_to_saddle_points: Vec<(usize, usize)> = Vec::new();
    for stationary_point in stationary_points.iter() {
        let i_r_nearest: usize = (r - stationary_point.r).abs().argmin().unwrap();
        let i_z_nearest: usize = (z - stationary_point.z).abs().argmin().unwrap();
        indicies_closest_to_saddle_points.push((i_r_nearest, i_z_nearest));
    }
    // for stationary_point in stationary_points.iter() {
    //         let saddle_points_to_mask: Vec<(usize, usize)> = saddle_point_masking(
    //         r,
    //         z,
    //         psi_2d,
    //         0.0,
    //         0.0,
    //     );
    // }
    // TODO: we could create a new function to figure out if the four grid points around the saddle should be masked or not

    // Grid sizes
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let mut mask_2d: Array2<f64> = Array2::from_elem((n_z, n_r), 0.0);

    // Find the index of the grid point closest to the magnetic axis
    let i_r_nearest: usize = (r - mag_r_previous).abs().argmin().unwrap();
    let i_z_nearest: usize = (z - mag_z_previous).abs().argmin().unwrap();

    // Start the flood fill algorithm, going anticlockwise from the magnetic axis
    // Directions: right, up, left, down (anticlockwise)
    let directions: [(isize, isize); 4] = [
        (0, 1),  // right (i_z, i_r+1)
        (-1, 0), // up    (i_z-1, i_r)
        (0, -1), // left  (i_z, i_r-1)
        (1, 0),  // down  (i_z+1, i_r)
    ];

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((i_z_nearest, i_r_nearest));
    mask_2d[(i_z_nearest, i_r_nearest)] = 1.0;

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

            if mask_2d[(new_i_z, new_i_r)] == 0.0 && psi_2d[(new_i_z, new_i_r)] > psi_b {
                // TODO: need to be much more careful around the x-point
                if xpt_diverted {
                    // Diverted plasma, only fill in points below the x-point
                    if z[new_i_z].abs() < xpt_z.abs() {
                        mask_2d[(new_i_z, new_i_r)] = 1.0;
                        queue.push_back((new_i_z, new_i_r));
                    }
                } else {
                    // Limited plasma
                    mask_2d[(new_i_z, new_i_r)] = 1.0;
                    queue.push_back((new_i_z, new_i_r));
                }
            }
        }
    }

    // Vessel polygon
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

    // Ensure points outside the vessel are masked out
    // TODO: This could be pre-calculated as mask_vessel_2d, and we do `mask_2d *= mask_vessel_2d`
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            // Check if the point is inside the polygon
            let test_point: Point = Point::new(r[i_r], z[i_z]);
            let inside_vessel: bool = vessel_polygon.contains(&test_point);
            if inside_vessel == false {
                // Ensure points outside the vessel are masked out
                mask_2d[(i_z, i_r)] = 0.0;
            }
        }
    }

    println!("psi_2d = {:#?}", psi_2d);
    // println!()

    return mask_2d;
}


fn saddle_point_masking(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    saddle_r: f64,
    saddle_z: f64,
) -> Vec<(usize, usize)> {


    // Find the nearest grid point to the saddle point
    let i_r_nearest: usize = (r - saddle_r).abs().argmin().unwrap();
    let i_z_nearest: usize = (z - saddle_z).abs().argmin().unwrap();
    // Find the four corner grid points surrounding the magnetic axis
    let i_r_nearest_left: usize;
    let i_r_nearest_right: usize;
    let i_z_nearest_lower: usize;
    let i_z_nearest_upper: usize;
    if saddle_r > r[i_r_nearest] {
        i_r_nearest_left = i_r_nearest;
        i_r_nearest_right = i_r_nearest + 1;
    } else {
        i_r_nearest_left = i_r_nearest - 1;
        i_r_nearest_right = i_r_nearest;
    }
    if saddle_z > z[i_z_nearest] {
        i_z_nearest_lower = i_z_nearest;
        i_z_nearest_upper = i_z_nearest + 1;
    } else {
        i_z_nearest_lower = i_z_nearest - 1;
        i_z_nearest_upper = i_z_nearest;
    }

    unimplemented!();
}


// #[test]
// fn test_flood_fill_mask() {

//     let r: Array1<f64> = Array1::linspace(0.01, 1.0, 80);
//     let z: Array1<f64> = Array1::linspace(-1.0, 1.0, 161);

//     let mask_2d: Array2<f64> = flood_fill_mask(
//         &r,
//         &z,
//         &psi_2d,
//         psi_b: f64,
//         _xpt_r: f64,
//         xpt_z: f64,
//         stationary_points: &Vec<StationaryPoint>,
//         xpt_diverted: bool,
//         mag_r_previous: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
//         mag_z_previous: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
//         vessel_r: &Array1<f64>,
//         vessel_z: &Array1<f64>,
//     );

// }