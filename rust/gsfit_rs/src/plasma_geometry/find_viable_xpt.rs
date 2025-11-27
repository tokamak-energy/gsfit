use super::BoundaryContour;
use super::StationaryPoint;
use core::f64;
use geo::Contains;
use geo::{Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

/// Finds the most likely x-point (i.e. one which passes all the x-point tests, and the x-point with the largest poloidal flux `psi`).
///
/// The x-point is a saddle point in the poloidal flux, this function searches for intersections between the `br=0` and `bz=0` contours.
/// For each candidate X-point, it constructs the corresponding plasma boundary and checks if:
/// - The X-point must be within the vacuum vessel
/// - The magnetic axis must be inside the candidate plasma boundary.
/// - The boundary contour is sorted by the largest to smallest `psi` value.
///
/// # Arguments
/// * `r` - 1D array of R (major radius) grid points.
/// * `z` - 1D array of Z (vertical) grid points.
/// * `psi_2d` - 2D array of poloidal flux values.
/// * `vessel_r` - 1D array of R coordinates defining the vessel boundary polygon.
/// * `vessel_z` - 1D array of Z coordinates defining the vessel boundary polygon.
/// * `mag_r` - R coordinate of the magnetic axis. Note, this is from the previous time-step.
/// * `mag_z` - Z coordinate of the magnetic axis. Note, this is from the previous time-step.
///
/// # Returns
/// * `Ok(BoundaryContour)` - The boundary contour and X-point information for the most viable candidate.
/// * `Err(String)` - An error message if no suitable X-point is found.
///
/// # Algorithm
/// 1. Find contours where `br=0` and `bz=0`
/// 2. Find stationary points, where `br=0` and `bz=0`
/// 3. Down select stationary points to only saddle points
/// 4. Find the contours associated with each saddle point
/// 5. Draw a vector from (mag_r, mag_z) to (r.max(), mag_z)
/// 6. Find intersection with boundary ==> we now know the x-point flux and x-point location (if there are multiple intersections, use the one )
/// 7. Collect all contours for x-point flux
///
/// # Example
/// ```ignore
/// let result = find_viable_xpt(&r, &z, &br_2d, &bz_2d, &psi_2d, &vessel_r, &vessel_z, mag_r, mag_z, d2_psi_d_r2_calculator);
/// ```
pub fn find_viable_xpt(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    stationary_points: &Vec<StationaryPoint>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    mag_r_previous: f64,
    mag_z_previous: f64,
) -> Result<BoundaryContour, String> {
    // TODO: add logic for negative plasma current

    // Exit if we haven't found any stationary points
    // TODO: perhaps I should never call `find_viable_xpt` if there are no stationary points?
    if stationary_points.len() == 0 {
        return Err("find_viable_xpt: no stationary points found".to_string());
    }

    // Grid variables
    let n_r: usize = r.len();

    // Number of stationary points
    let n_stationary_points: usize = stationary_points.len();

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

    // Construct potential x-points
    let mut potential_xpts: Vec<BoundaryContour> = Vec::with_capacity(n_stationary_points);
    for i_xpt in 0..n_stationary_points {
        let stationary_point: StationaryPoint = stationary_points[i_xpt];

        // Only add saddle points
        let test_for_saddle_point: bool = stationary_point.hessian_determinant < 0.0;
        if test_for_saddle_point == true {
            potential_xpts.push(
                BoundaryContour {
                    boundary_r: Array1::zeros(0),
                    boundary_z: Array1::zeros(0),
                    n_points: 0,
                    bounding_psi: stationary_point.psi,
                    bounding_r: stationary_point.r,
                    bounding_z: stationary_point.z,
                    xpt_diverted: false,
                    mask: None, // mask calculated later using method
                }
            )
        }
    }
    if potential_xpts.len() == 0 {
        return Err("find_viable_xpt: no saddle points found".to_string());
    }

    // Sort from largest to smallest `psi`
    potential_xpts.sort_by(|a, b| {
        b.bounding_psi // using `b` first gives descending order
            .partial_cmp(&a.bounding_psi)
            .expect("find_viable_limit_point: cannot sort potential_limit_points by bounding_psi")
    });

    // Find the closest grid point to the magnetic axis
    let index_mag_r: usize = (r - mag_r_previous).abs().argmin().expect("find_viable_xpt: unwrapping index_mag_r");
    let index_mag_z: usize = (z - mag_z_previous).abs().argmin().expect("find_viable_xpt: unwrapping index_mag_z");

    'loop_over_potential_xpts: for potential_xpt in &mut potential_xpts {
        // Test if saddle point is within vessel
        let xpt_r_local: f64 = potential_xpt.bounding_r;
        let xpt_z_local: f64 = potential_xpt.bounding_z;
        let xpt_point: Point = Point::new(xpt_r_local, xpt_z_local);
        let test_inside_vessel: bool = vessel_polygon.contains(&xpt_point);
        if test_inside_vessel == false {
            continue 'loop_over_potential_xpts;
        }

        // Test if there is a LFS boundary at the same height as the magnetic axis
        // March from the magnetic axis to the LFS
        let mut test_intersects_lfs_boundary: bool = false;
        for i_r in index_mag_r..n_r - 1 {
            if psi_2d[(index_mag_z, i_r)] < potential_xpt.bounding_psi {
                test_intersects_lfs_boundary = true;
            }
        }
        // No LFS boundary encountered
        if !test_intersects_lfs_boundary {
            continue 'loop_over_potential_xpts;
        }

        // If we make it to the end, then return this, ending `find_viable_xpt` function
        return Ok(potential_xpt.to_owned());

    }

    return Err("find_viable_xpt: no viable x-point found".to_string());

}

// #[test]
// /// Test the `find_viable_xpt` function with:
// /// 1. Double null diverted (DND) configuration
// /// 2. Slightly displaced DND configuration
// /// 3. Deliberately moving `br` and `bz` contours a fractionally away from the x-point
// ///
// /// I believe the fundamental problem with finding the x-point / plasma boundary when diverted is that the current carrying
// /// grid is extremely close to the x-point. This causes the function to be fractionally more complicated than analytic equations.
// /// This complexity is apparent in the derivatives of `psi`, i.e. `br` and `bz`.
// /// TODO: need to prove and quantify this theory for function "complexity".
// fn test_find_viable_xpt() {
//     // use approx::assert_abs_diff_eq;

//     const PI: f64 = std::f64::consts::PI;

//     let n_r: usize = 100;
//     let n_z: usize = 201;
//     let r: Array1<f64> = Array1::linspace(0.1, 1.0, n_r);
//     let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

//     /// This analytic equation has two x-points, one at the top and one at the bottom, and looks quite similar to
//     /// a tokamak plasma.
//     /// Note: this analytic solution is not itself a solution to the Grad-Shafranov equation.
//     fn analytic_equation(r: Array1<f64>, z: Array1<f64>, vertical_offset: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
//         let n_r: usize = r.len();
//         let n_z: usize = z.len();

//         // Calculate psi
//         let psi_2d: Array2<f64> = Array2::from_shape_fn((n_z, n_r), |(i_z, i_r)| {
//             let r_local = r[i_r];
//             let z_local = z[i_z];
//             let r_shifted: f64 = 5.0 * (r_local - 0.6); // Shift the R coordinate to center the plasma
//             let z_shfited: f64 = 1.8 * (z_local + 1.0 / 1.8); // Shift the Z coordinate to center the plasma
//             let z_shfited_opposite: f64 = 1.8 * (-z_local + 1.0 / 1.8); // Shift the opposite Z coordinate to center the plasma

//             // plain text equation:
//             let psi_local: f64 = (-z_shfited.powi(4) + 2.0 * z_shfited.powi(2) - r_shifted.powi(2)) * (-2.0 * z_local).exp()
//                 + vertical_offset * (-z_shfited_opposite.powi(4) + 2.0 * z_shfited_opposite.powi(2) - r_shifted.powi(2)) * (2.0 * z_local).exp();

//             return psi_local;
//         });

//         // Calculate br
//         let br_2d: Array2<f64> = Array2::from_shape_fn((n_z, n_r), |(i_z, i_r)| {
//             let r_local: f64 = r[i_r];
//             let z_local: f64 = z[i_z];
//             let a: f64 = 1.8;
//             let b: f64 = 0.6;
//             let d_psi_dz: f64 = 2.0
//                 * (-2.0 * z_local).exp()
//                 * (a.powi(4) * z_local.powi(4) - 2.0 * a.powi(4) * z_local.powi(3) + 4.0 * a.powi(3) * z_local.powi(3) - 6.0 * a.powi(3) * z_local.powi(2)
//                     + 4.0 * a.powi(2) * z_local.powi(2)
//                     - 4.0 * a.powi(2) * z_local
//                     + 25.0 * b.powi(2)
//                     - 50.0 * b * r_local
//                     + 25.0 * r_local.powi(2)
//                     - 1.0)
//                 - 2.0
//                     * vertical_offset
//                     * (2.0 * z_local).exp()
//                     * (a.powi(4) * z_local.powi(4) + 2.0 * a.powi(4) * z_local.powi(3) - 4.0 * a.powi(3) * z_local.powi(3) - 6.0 * a.powi(3) * z_local.powi(2)
//                         + 4.0 * a.powi(2) * z_local.powi(2)
//                         + 4.0 * a.powi(2) * z_local
//                         + 25.0 * b.powi(2)
//                         - 50.0 * b * r_local
//                         + 25.0 * r_local.powi(2)
//                         - 1.0);
//             let br_local: f64 = -d_psi_dz / (2.0 * PI * r_local);
//             return br_local;
//         });

//         let bz_2d: Array2<f64> = Array2::from_shape_fn((n_z, n_r), |(_i_z, i_r)| {
//             let r_local: f64 = r[i_r];
//             let z_local: f64 = z[_i_z];
//             let d_psi_dr: f64 = -50.0 * (r_local - 0.6) * (-2.0 * z_local).exp() * (vertical_offset * (4.0 * z_local).exp() + 1.0);
//             let bz_local: f64 = d_psi_dr / (2.0 * PI * r_local);
//             return bz_local;
//         });

//         return (psi_2d, br_2d, bz_2d);
//     }

//     // Vessel polygon
//     let vessel_r: Array1<f64> = Array1::from(vec![0.17, 0.90, 0.90, 0.17, 0.17]);
//     let vessel_z: Array1<f64> = Array1::from(vec![-0.85, -0.85, 0.85, 0.85, -0.85]);

//     // Magnetic axis
//     let mag_r: f64 = 0.6;
//     let mag_z: f64 = 0.0;

//     // Calculate a perfect DND
//     let vertical_offset: f64 = 1.0; // 1.0 means no offset
//     let (psi_2d, br_2d, bz_2d): (Array2<f64>, Array2<f64>, Array2<f64>) = analytic_equation(r.clone(), z.clone(), vertical_offset);

//     // Central differencing for d_br_d_z_2d and d_bz_d_z_2d
//     // TODO: this can be improved with an analytic equation
//     let mut d_br_d_z_2d: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
//     let mut d_bz_d_z_2d: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
//     let d_z: f64 = z[1] - z[0];

//     for i_z in 1..n_z - 1 {
//         for i_r in 0..n_r {
//             d_br_d_z_2d[(i_z, i_r)] = (br_2d[(i_z + 1, i_r)] - br_2d[(i_z - 1, i_r)]) / (2.0 * d_z);
//             d_bz_d_z_2d[(i_z, i_r)] = (bz_2d[(i_z + 1, i_r)] - bz_2d[(i_z - 1, i_r)]) / (2.0 * d_z);
//         }
//     }
//     // For boundaries, use forward/backward difference
//     for i_r in 0..n_r {
//         d_br_d_z_2d[(0, i_r)] = (br_2d[(1, i_r)] - br_2d[(0, i_r)]) / d_z;
//         d_br_d_z_2d[(n_z - 1, i_r)] = (br_2d[(n_z - 1, i_r)] - br_2d[(n_z - 2, i_r)]) / d_z;
//         d_bz_d_z_2d[(0, i_r)] = (bz_2d[(1, i_r)] - bz_2d[(0, i_r)]) / d_z;
//         d_bz_d_z_2d[(n_z - 1, i_r)] = (bz_2d[(n_z - 1, i_r)] - bz_2d[(n_z - 2, i_r)]) / d_z;
//     }

//     // Find the boundary contour, raise exception if boundary not found
//     let mut xpt_boundary: BoundaryContour = find_viable_xpt(&r, &z, &br_2d, &bz_2d, &psi_2d, &d_br_d_z_2d, &d_bz_d_z_2d, &vessel_r, &vessel_z, mag_r, mag_z)
//         .expect("find_viable_xpt: error, we should have found a viable x-point");
//     // Calculate the plasma volume
//     xpt_boundary.calculate_plasma_volume();

//     // // Calculate a LSN
//     // let vertical_offset: f64 = 1.0 - 1e-2; // small offset
//     // Calculate a USN
//     let vertical_offset: f64 = 1.0 + 1e-2; // small offset
//     let (psi_2d, br_2d, _bz_2d): (Array2<f64>, Array2<f64>, Array2<f64>) = analytic_equation(r.clone(), z.clone(), vertical_offset);

//     // Find the boundary contour
//     // Note: br=0 contour is curved, but bz=0 is nearly a vertical line (with this test function)
//     // br is a bit complicated, so I will shift bz radially a little
//     // Test 3: Deliberately add a small offset to bz
//     let bz_2d_perturbed: Array2<f64> = Array2::from_shape_fn((n_z, n_r), |(_i_z, i_r)| {
//         let r_local: f64 = r[i_r] + 1e-3; // small offset
//         let z_local: f64 = z[_i_z];
//         let d_psi_dr: f64 = -50.0 * (r_local - 0.6) * (-2.0 * z_local).exp() * (vertical_offset * (4.0 * z_local).exp() + 1.0);
//         let bz_local: f64 = d_psi_dr / (2.0 * PI * r_local);
//         return bz_local;
//     });
//     let mut xpt_boundary: BoundaryContour = find_viable_xpt(
//         &r,
//         &z,
//         &br_2d,
//         &bz_2d_perturbed,
//         &psi_2d,
//         &d_br_d_z_2d,
//         &d_bz_d_z_2d,
//         &vessel_r,
//         &vessel_z,
//         mag_r,
//         mag_z,
//     )
//     .expect("find_viable_xpt: error, we should have found a viable x-point");
//     xpt_boundary.refine_xpt_diverted_boundary(&r, &z, &psi_2d, mag_r, mag_z, &br_2d, &bz_2d);
//     xpt_boundary.calculate_plasma_volume();

//     // Imports
//     use std::fs::File;
//     use std::io::{BufWriter, Write};
//     // write to file
//     let file = File::create("boundary_r.csv").expect("can't make file");
//     let mut writer = BufWriter::new(file);
//     for row in xpt_boundary.boundary_r.rows() {
//         let line: String = row.iter().map(|&value| value.to_string()).collect::<Vec<_>>().join(", ");
//         writeln!(writer, "{}", line).expect("can't write line");
//     }
//     writer.flush().expect("can't flush writer");
//     // write to file
//     let file = File::create("boundary_z.csv").expect("can't make file");
//     let mut writer = BufWriter::new(file);
//     for row in xpt_boundary.boundary_z.rows() {
//         let line: String = row.iter().map(|&value| value.to_string()).collect::<Vec<_>>().join(", ");
//         writeln!(writer, "{}", line).expect("can't write line");
//     }
//     writer.flush().expect("can't flush writer");
// }
