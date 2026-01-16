use super::StationaryPoint;
use core::f64;
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::Array1;

pub struct MagneticAxis {
    pub r: f64,
    pub z: f64,
    pub psi: f64,
}

/// Find the magnetic axis using bicubic interpolation
///
/// # Arguments
/// * `stationary_points` - vector of stationary points found from intersecting br=0 and bz=0 contours
/// * `mag_r_previous` - previous magnetic axis radius, (metre)
/// * `mag_z_previous` - previous magnetic axis height, (metre)
/// * `vessel_r` - vessel boundary radial coordinates, (metre)
/// * `vessel_z` - vessel boundary vertical coordinates, (metre)
///
/// # Returns
/// * `mag_r` - the radial magnetic axis, (metre)
/// * `mag_z` - the vertical magnetic axis, (metre)
/// * `mag_psi` - the magnetic flux at the magnetic axis, (weber)
///
/// # Algorithm
/// 1. Find contours where `br=0` and `bz=0`. Return Err if no contours found.
/// 2. Find intersection points between these contours = stationary points.  Return Err if no intersections found.
/// 3. Check the Hessian to determine if the stationary point is maximum/minimum/saddle => magnetic axis found. Return Err if no maximum found.
/// 4. Find the 4 cells nearest to the magnetic axis. Return Err if nearest point is on grid boundary.
/// 5. Fit a bicubic polynomial.
/// 6. Iteratively find a more accurate location for the magnetic axis, and the flux value.
/// 7. Fall back to a brute-force search if the iterative method does not converge.
///
/// # Examples
/// ```rust
/// use gsfit_rs::plasma_geometry::find_magnetic_axis;
/// use ndarray::{Array1, Array2};
/// ```
///
pub fn find_magnetic_axis(
    stationary_points: &Vec<StationaryPoint>,
    mag_r_previous: f64,
    mag_z_previous: f64,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Result<MagneticAxis, String> {
    // TODO: need to add plasma current sign: we are assuming positive current = hill in psi;
    // but negative plasma current will make a valley and break all the logic!

    // Create a mutable copy of `stationary_points`, because we want to filter it
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.to_owned();

    // Filter `stationary_points` to keep only those with "turning curvature" (either maximum or minimum)
    stationary_points.retain(|stationary_point| {
        let turning_point_test: bool = stationary_point.hessian_determinant > 0.0;
        return turning_point_test;
    });
    // Exit if we haven't found any `stationary_points` within the vessel
    if stationary_points.is_empty() {
        return Err("find_magnetic_axis: no `stationary_points` with turning curvature found".to_string());
    }

    // Filter `stationary_points` to keep only those within the vessel
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
    stationary_points.retain(|stationary_point| {
        let test_point: Point = Point::new(stationary_point.r, stationary_point.z);
        let within_vessel_test: bool = vessel_polygon.contains(&test_point);
        return within_vessel_test;
    });
    // Exit if we haven't found any `stationary_points` within the vessel
    if stationary_points.is_empty() {
        return Err("find_magnetic_axis: no `stationary_points` found within the vessel".to_string());
    }

    // If we have multiple stationary points, choose the one nearest to the previous magnetic axis position
    let mag_r: f64;
    let mag_z: f64;
    let mag_psi: f64;
    if stationary_points.len() > 1 {
        let mut distances: Vec<f64> = Vec::with_capacity(stationary_points.len());
        for stationary_point in &stationary_points {
            let delta_r: f64 = stationary_point.r - mag_r_previous;
            let delta_z: f64 = stationary_point.z - mag_z_previous;
            let distance: f64 = (delta_r.powi(2) + delta_z.powi(2)).sqrt();
            distances.push(distance);
        }
        let (i_min, _min_value) = distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("find_magnetic_axis: finding min distance");
        mag_r = stationary_points[i_min].r;
        mag_z = stationary_points[i_min].z;
        mag_psi = stationary_points[i_min].psi;
    } else {
        // We have exactly one stationary point within the vessel
        mag_r = stationary_points[0].r;
        mag_z = stationary_points[0].z;
        mag_psi = stationary_points[0].psi;
    }

    let magnetic_axis: MagneticAxis = MagneticAxis {
        r: mag_r,
        z: mag_z,
        psi: mag_psi,
    };

    return Ok(magnetic_axis);
}

// /// Test finding the magnetic axis
// ///
// /// We add a pair of BV PF coils, and a PF coil to represent the plasma.
// /// The maximum in `psi` will be where the plasma PF coil is.
// /// To make this a useful test I have shifted both the BV PF coils and the
// /// plasma upwards, so the function `find_magnetic_axis` should find the
// /// magnetic axis at a positive z value.
// #[test]
// fn test_find_magnetic_axis() {
//     // Lazy loading of packages which are not used anywhere else in the code
//     use crate::greens::greens_b;
//     use crate::greens::greens_d_b_d_z;
//     use crate::greens::greens_psi;
//     use approx::assert_abs_diff_eq;
//     use ndarray::s;
//     use ndarray::Array2;
//     use crate::plasma_geometry::find_stationary_points;

//     let z_shift: f64 = 0.02345;
//     let n_r: usize = 100;
//     let n_z: usize = 201;
//     let vec_r: Array1<f64> = Array1::linspace(0.01, 1.0, n_r);
//     let vec_z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

//     // Meshgrid
//     let mut mesh_r: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
//     let mut mesh_z: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
//     for i_z in 0..n_z {
//         for i_r in 0..n_r {
//             mesh_r[(i_z, i_r)] = vec_r[i_r];
//             mesh_z[(i_z, i_r)] = vec_z[i_z];
//         }
//     }

//     // Flatten 2d mesh
//     let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
//     let flat_z: Array1<f64> = mesh_z.flatten().to_owned();

//     // Create some BV PF coils
//     let bv_coils_r: Array1<f64> = Array1::from_vec(vec![1.4, 1.4]);
//     let bv_coils_z: Array1<f64> = Array1::from_vec(vec![0.6123 + z_shift, -0.6123 + z_shift]);
//     let bv_g: Array2<f64> = greens_psi(
//         flat_r.clone(),
//         flat_z.clone(),
//         bv_coils_r.clone(),
//         bv_coils_z.clone(),
//         flat_r.clone(),
//         flat_z.clone(),
//     );
//     let (bv_g_br, bv_g_bz): (Array2<f64>, Array2<f64>) = greens_b(flat_r.clone(), flat_z.clone(), bv_coils_r.clone(), bv_coils_z.clone());
//     let (g_d_br_dz, g_d_bz_dz): (Array2<f64>, Array2<f64>) = greens_d_b_d_z(flat_r.clone(), flat_z.clone(), bv_coils_r.clone(), bv_coils_z.clone());

//     // Create a PF coil at mid-plane to represent the plasma
//     let plasma_coil_r: Array1<f64> = Array1::from_vec(vec![0.41111]);
//     let plasma_coil_z: Array1<f64> = Array1::from_vec(vec![0.0 + z_shift]);
//     let plasma_g: Array2<f64> = greens_psi(
//         flat_r.clone(),
//         flat_z.clone(),
//         plasma_coil_r.clone(),
//         plasma_coil_z.clone(),
//         flat_r.clone(),
//         flat_z.clone(),
//     );
//     let (plasma_g_br, plasma_g_bz): (Array2<f64>, Array2<f64>) = greens_b(flat_r.clone(), flat_z.clone(), plasma_coil_r.clone(), plasma_coil_z.clone());
//     let (_plasma_d_g_br_dz, plasma_d_g_bz_dz): (Array2<f64>, Array2<f64>) =
//         greens_d_b_d_z(flat_r.clone(), flat_z.clone(), plasma_coil_r.clone(), plasma_coil_z.clone());

//     // Currents
//     let bv_current: f64 = -1.0e5; // Negative current for BV coils, (amperes)
//     let plasma_current: f64 = 2.0e5; // Positive current for plasma coil, (amperes)

//     let psi: Array1<f64> = bv_current * &bv_g.slice(s![.., 0]) + bv_current * &bv_g.slice(s![.., 1]) + plasma_current * &plasma_g.slice(s![.., 0]);
//     let br: Array1<f64> = bv_current * &bv_g_br.slice(s![.., 0]) + bv_current * &bv_g_br.slice(s![.., 1]) + plasma_current * &plasma_g_br.slice(s![.., 0]);
//     let bz: Array1<f64> = bv_current * &bv_g_bz.slice(s![.., 0]) + bv_current * &bv_g_bz.slice(s![.., 1]) + plasma_current * &plasma_g_bz.slice(s![.., 0]);
//     let d_bz_dz: Array1<f64> =
//         bv_current * &g_d_bz_dz.slice(s![.., 0]) + bv_current * &g_d_bz_dz.slice(s![.., 1]) + plasma_current * &plasma_d_g_bz_dz.slice(s![.., 0]);

//     let psi_2d: Array2<f64> = psi.to_shape((n_z, n_r)).unwrap().to_owned();
//     let br_2d: Array2<f64> = br.to_shape((n_z, n_r)).unwrap().to_owned();
//     let bz_2d: Array2<f64> = bz.to_shape((n_z, n_r)).unwrap().to_owned();
//     let d_br_d_z_2d: Array2<f64> = d_bz_dz.to_shape((n_z, n_r)).unwrap().to_owned();
//     let d_bz_d_z_2d: Array2<f64> = d_bz_dz.to_shape((n_z, n_r)).unwrap().to_owned();

//     let mag_r_previous: f64 = 0.51111; // initial condition
//     let mag_z_previous: f64 = 0.0; // initial condition

//     let d2_psi_d_r2_calculator = crate::greens::greens_d2_psi_d_r2::GreensD2PsiDR2Calculator::new(
//         flat_r.clone(),
//         flat_z.clone(),
//         bv_coils_r.clone(),
//         bv_coils_z.clone(),
//         plasma_coil_r.clone(),
//         plasma_coil_z.clone(),
//     );

//     // Find stationary points in `psi`
//     let stationary_points_or_error: Result<Vec<StationaryPoint>, String> =
//         find_stationary_points(&vec_r, &vec_z, &psi_2d, &br_2d, &bz_2d, &d_br_d_z_2d, &d_bz_d_z_2d, d2_psi_d_r2_calculator.clone());

//                 // let magnetic_axis: MagneticAxis = find_magnetic_axis(&vec_r, &vec_z, &psi_2d, mag_r_previous, mag_z_previous)
//         // .expect("test_find_magnetic_axis: unwrapping magnetic_axis_result");

//     // Fairly large epsilon, since we have only given the bicubic interpolator the function
//     // and it's derivatives at the four corner grid points
//     // assert_abs_diff_eq!(magnetic_axis.z, 0.0 + z_shift, epsilon = 1e-3);
// }
