use super::BoundaryContour;
use super::StationaryPoint;
use contour::ContourBuilder;
use core::f64;
use geo::Contains;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Coord, Line, LineString, MultiPolygon, Point, Polygon};
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
/// * `br_2d` - 2D array of poloidal magnetic field component B_R.
/// * `bz_2d` - 2D array of poloidal magnetic field component B_Z.
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
/// 8. Pass to `BoundaryContour.refine_xpt_diverted_boundary`
///
/// # Example
/// ```ignore
/// let result = find_viable_xpt(&r, &z, &br_2d, &bz_2d, &psi_2d, &vessel_r, &vessel_z, mag_r, mag_z, d2_psi_d_r2_calculator);
/// ```
pub fn find_viable_xpt(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    stationary_points: &Vec<StationaryPoint>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    mag_r_previous: f64,
    mag_z_previous: f64,
) -> Result<BoundaryContour, String> {
    // Create a mutable copy of `stationary_points`, because we want to filter it
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.clone();
    // Exit if we haven't found any stationary points
    // TODO: should not have called `find_viable_xpt` if there are no stationary points
    if stationary_points.len() == 0 {
        return Err("find_viable_xpt: no stationary points found".to_string());
    }

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];
    let near_distance: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt() * 2.0;

    // Construct an empty `contour_grid` object
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true)
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

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

    // Filter to retain only `stationary_points` which are within the vessel
    stationary_points.retain(|stationary_point| {
        let xpt_r_local: f64 = stationary_point.r;
        let xpt_z_local: f64 = stationary_point.z;
        let xpt_point: Point = Point::new(xpt_r_local, xpt_z_local);
        let inside_vessel: bool = vessel_polygon.contains(&xpt_point);

        return inside_vessel;
    });
    // Exit if we haven't found any `stationary_points` within the vessel
    if stationary_points.len() == 0 {
        return Err("find_viable_xpt: no stationary points found inside vessel".to_string());
    }

    // Filter to retain only `stationary_points` which are saddle points
    stationary_points.retain(|stationary_point| {
        let hessian_det: f64 = stationary_point.hessian_determinant;
        let saddle_point_test: bool = hessian_det < 0.0;

        return saddle_point_test;
    });
    // Exit if we haven't found any `stationary_points` which have saddle curvature
    if stationary_points.len() == 0 {
        return Err("find_viable_xpt: no stationary points with saddle curvature".to_string());
    }

    // Flatten psi for contouring
    let psi_2d_flattened: Vec<f64> = psi_2d.flatten().to_vec();

    // Create a vector from the magnetic axis to the right edge of the grid
    let mag_axis_point: Point = Point::new(mag_r_previous, mag_z_previous);
    let right_edge_point: Point = Point::new(r.max().expect("find_viable_xpt: r.max()").to_owned(), mag_z_previous);
    let mag_axis_to_right_edge: Line<f64> = Line::new(mag_axis_point, right_edge_point);

    // See which contours are on the LFS
    let mut potential_boundary_contours: Vec<geo_types::MultiPolygon> = Vec::new();
    let mut intersection_radius: Vec<f64> = Vec::new();
    let mut potential_psi_b: Vec<f64> = Vec::new();
    let mut potential_xpt_r: Vec<f64> = Vec::new();
    let mut potential_xpt_z: Vec<f64> = Vec::new();
    // let mut potential_boundary: Vec<BoundaryContour> = Vec::new();
    for stationary_point in stationary_points {
        let possible_xpts_psi: f64 = stationary_point.psi;
        let possible_xpts_r: f64 = stationary_point.r;
        let possible_xpts_z: f64 = stationary_point.z;

        let boundary_contours_local: Vec<contour::Contour> = contour_grid
            .contours(&psi_2d_flattened, &[possible_xpts_psi])
            .expect("find_viable_xpt: cannot find `boundary_contours_tmp`");
        let boundary_contours: &geo_types::MultiPolygon = boundary_contours_local[0].geometry(); // The [0] is because I have only supplied one threshold

        // Check if the contour is close to the saddle point
        let n_contours: usize = boundary_contours.iter().count();
        for i_contour in 0..n_contours {
            let boundary_contour: &Polygon = boundary_contours
                .iter()
                .nth(i_contour)
                .expect("find_viable_xpt: cannot find `boundary_contour`");

            // Find the minimum distance from the contour to the potential x-point
            let boundary_r: Array1<f64> = boundary_contour.exterior().coords().map(|coord| coord.x).collect::<Array1<f64>>();
            let boundary_z: Array1<f64> = boundary_contour.exterior().coords().map(|coord| coord.y).collect::<Array1<f64>>();
            let distances: Array1<f64> = ((boundary_r - possible_xpts_r).powi(2) + (boundary_z - possible_xpts_z).powi(2)).sqrt();
            let min_distance: f64 = distances.min().expect("find_viable_xpt: cannot find min distance").to_owned();

            // Check if the contour is "close" to the potential x-point
            if &min_distance < &near_distance {
                // Check if the boundary contour intersects the line from the magnetic axis to the right edge
                let boundary_contour_line: LineString<f64> = boundary_contour.exterior().to_owned();

                // Check intersection between each segment of the `boundary_contour_line` and the `mag_axis_to_right_edge` line
                for segment in boundary_contour_line.lines() {
                    let intersection = line_intersection(segment, mag_axis_to_right_edge);
                    if intersection.is_some() {
                        let intersection2: LineIntersection<f64> = intersection.expect("find_viable_xpt: error, intersection");
                        let intersection_point: Coord = match intersection2 {
                            LineIntersection::SinglePoint { intersection, .. } => intersection,
                            _ => panic!("find_viable_xpt: expected SinglePoint intersection"),
                        };
                        // let boundary_contour = BoundaryContour{
                        //     boundary_polygon: boundary_contour.clone(),
                        // };
                        // potential_boundary.push()
                        intersection_radius.push(intersection_point.x);
                        potential_boundary_contours.push(boundary_contours.clone());
                        potential_psi_b.push(possible_xpts_psi);
                        potential_xpt_r.push(possible_xpts_r);
                        potential_xpt_z.push(possible_xpts_z);
                    }
                }
            }
        }
    }

    let n_intersections: usize = intersection_radius.len();
    if n_intersections == 0 {
        return Err("find_viable_xpt: contours passing the LFS found".to_string());
    }

    // Find the index of the minimum value in intersection_radius
    let mut index_min: usize = usize::MAX; // initialize to an invalid index, will panic if used
    let mut min_radius: f64 = f64::INFINITY;
    for (i_intersection, &radius) in intersection_radius.iter().enumerate() {
        if radius < min_radius {
            min_radius = radius;
            index_min = i_intersection;
        }
    }

    let boundary_contour: MultiPolygon = potential_boundary_contours[index_min].clone();
    // We are going to add all of the contours at the psi_b value to bounary_r and boundary_z.
    // Then refine_xpt_diverted_boundary will figure out what the boundary is.
    let mut boundary_r: Vec<f64> = Vec::new();
    let mut boundary_z: Vec<f64> = Vec::new();
    for contour in &boundary_contour {
        // Add all exterior contour to the boundary
        let tmp_r: Vec<f64> = contour.exterior().coords().map(|coord| coord.x).collect::<Vec<f64>>();
        let tmp_z: Vec<f64> = contour.exterior().coords().map(|coord| coord.y).collect::<Vec<f64>>();
        for i_point in 0..tmp_r.len() {
            boundary_r.push(tmp_r[i_point]);
            boundary_z.push(tmp_z[i_point]);
        }
        // Add all interiors to the boundary
        for interior in contour.interiors() {
            let tmp_r: Vec<f64> = interior.coords().map(|coord| coord.x).collect::<Vec<f64>>();
            let tmp_z: Vec<f64> = interior.coords().map(|coord| coord.y).collect::<Vec<f64>>();
            for i_point in 0..tmp_r.len() {
                boundary_r.push(tmp_r[i_point]);
                boundary_z.push(tmp_z[i_point]);
            }
        }
    }

    let n_points: usize = boundary_r.len();
    let mut final_contour: BoundaryContour = BoundaryContour {
        // boundary_polygon: Polygon::new(LineString::new(vec![]), vec![]), // TODO: this is not correct. But I want to remove this variable from BoundaryContour
        boundary_r: Array1::from_vec(boundary_r),
        boundary_z: Array1::from_vec(boundary_z),
        n_points,
        bounding_psi: potential_psi_b[index_min],
        bounding_r: potential_xpt_r[index_min],
        bounding_z: potential_xpt_z[index_min],
        fraction_inside_vessel: f64::NAN,
        xpt_diverted: true,
        mask: None, // mask calculated using method
        secondary_xpt_r: f64::NAN,
        secondary_xpt_z: f64::NAN,
        secondary_xpt_distance: f64::MAX,
    };

    // final_contour.refine_xpt_diverted_boundary(r, z, psi_2d, mag_r_previous, mag_z_previous, br_2d, bz_2d);

    return Ok(final_contour);
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
//     // println!("test_find_viable_xpt: xpt_boundary.plasma_volume = {:?}", xpt_boundary.plasma_volume);

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
//     // println!("test_find_viable_xpt: xpt_boundary.plasma_volume = {:?}", xpt_boundary.plasma_volume);

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
