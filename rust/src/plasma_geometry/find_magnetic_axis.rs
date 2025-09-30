use super::StationaryPoint;
use crate::bicubic_interpolator::BicubicInterpolator;
use crate::bicubic_interpolator::BicubicStationaryPoint;
use core::f64;
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

const PI: f64 = std::f64::consts::PI;

pub struct MagneticAxis {
    pub r: f64,
    pub z: f64,
    pub psi: f64,
}

/// Find the magnetic axis using bicubic interpolation
///
/// # Arguments
/// * `r` - radial grid, length=n_r, (metre)
/// * `z` - vertical grid, length=n_z, (metre)
/// * `br_2d` - radial magnetic field, shape=(n_z, n_r), (tesla)
/// * `bz_2d` - vertical magnetic field, shape=(n_z, n_r), (tesla)
/// * `d_bz_d_z` - vertical magnetic field gradient, shape=(n_z, n_r), (tesla/metre)
/// * `psi_2d` - magnetic flux function, shape=(n_z, n_r), (weber)
/// * `mag_r_previous` - previous magnetic axis radius, (metre)
/// * `mag_z_previous` - previous magnetic axis height, (metre)
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
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    stationary_points: &Vec<StationaryPoint>,
    mag_r_previous: f64,
    mag_z_previous: f64,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Result<MagneticAxis, String> {
    // TODO: need to add plasma current sign: we are assuming positive current = hill in psi;
    // but negative plasma current will make a valley and break all the logic!

    // // Imports
    // use std::fs::File;
    // use std::io::{BufWriter, Write};
    // // write to file
    // let file = File::create("psi_2d.csv").expect("can't make file");
    // let mut writer = BufWriter::new(file);
    // for row in psi_2d.rows() {
    //     let line: String = row.iter()
    //         .map(|&value| value.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ");
    //     writeln!(writer, "{}", line).expect("can't write line");
    // }
    // writer.flush().expect("can't flush writer");
    // panic!("stop for debugging");

    // Create a mutable copy of `stationary_points`, because we want to filter it
    let mut stationary_points: Vec<StationaryPoint> = stationary_points.to_owned();

    // println!("find_magnetic_axis: stationary_points={:?}", stationary_points);

    // Grid variables
    let d_r: f64 = (r.last().expect("find_magnetic_axis: unwrapping last `r`") - r[0]) / (r.len() as f64 - 1.0);
    let d_z: f64 = (z.last().expect("find_magnetic_axis: unwrapping last `z`") - z[0]) / (z.len() as f64 - 1.0);

    // Filter `stationary_points` to keep only those with "turning curvature" (either maximum or minimum)
    stationary_points.retain(|stationary_point| {
        let turning_point_test: bool = stationary_point.hessian_determinant > 0.0;
        return turning_point_test;
    });
    // Exit if we haven't found any `stationary_points` within the vessel
    if stationary_points.len() == 0 {
        return Err("find_magnetic_axis: no stationary points with turning shape found".to_string());
    }
    // println!("find_magnetic_axis: turning stationary_points={:?}", stationary_points);

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
    if stationary_points.len() == 0 {
        return Err("find_magnetic_axis: no stationary points found within the vessel".to_string());
    }
    // println!("find_magnetic_axis: in vessel stationary_points={:?}", stationary_points);

    // If we have multiple stationary points, choose the one nearest to the previous magnetic axis position
    let mut mag_r: f64;
    let mut mag_z: f64;
    if stationary_points.len() > 1 {
        let mut distances: Vec<f64> = Vec::with_capacity(stationary_points.len());
        for stationary_point in &stationary_points {
            let dr: f64 = stationary_point.r - mag_r_previous;
            let dz: f64 = stationary_point.z - mag_z_previous;
            let distance: f64 = (dr * dr + dz * dz).sqrt();
            distances.push(distance);
        }
        let (i_min, _min_value) = distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("find_magnetic_axis: finding min distance");
        mag_r = stationary_points[i_min].r;
        mag_z = stationary_points[i_min].z;
    } else {
        // We have exactly one stationary point within the vessel
        mag_r = stationary_points[0].r;
        mag_z = stationary_points[0].z;
    }

    // println!("mag_r, mag_z = {}, {}", mag_r, mag_z);

    // Find the nearest grid-point to the magnetic axis
    let i_r_nearest: usize = (r - mag_r).mapv(|x| x.abs()).argmin().unwrap();
    let i_z_nearest: usize = (z - mag_z).mapv(|x| x.abs()).argmin().unwrap();

    // Check that the magnetic axis is within the grid bounds
    if i_r_nearest == 0 || i_z_nearest == 0 || i_r_nearest == r.len() - 1 || i_z_nearest == z.len() - 1 {
        return Err("find_magnetic_axis: Magnetic axis is on the grid boundary".to_string());
    }

    // Find the four corner grid points surrounding the magnetic axis
    let i_r_nearest_left: usize;
    let i_r_nearest_right: usize;
    let i_z_nearest_lower: usize;
    let i_z_nearest_upper: usize;
    if mag_r > r[i_r_nearest] {
        i_r_nearest_left = i_r_nearest;
        i_r_nearest_right = i_r_nearest + 1;
    } else {
        i_r_nearest_left = i_r_nearest - 1;
        i_r_nearest_right = i_r_nearest;
    }
    if mag_z > z[i_z_nearest] {
        i_z_nearest_lower = i_z_nearest;
        i_z_nearest_upper = i_z_nearest + 1;
    } else {
        i_z_nearest_lower = i_z_nearest - 1;
        i_z_nearest_upper = i_z_nearest;
    }

    // Now that we have found the magnetic axis we will not be returning any errors

    // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
    let mut f: Array2<f64> = Array2::zeros([2, 2]);
    let mut d_f_d_r: Array2<f64> = Array2::zeros([2, 2]);
    let mut d_f_d_z: Array2<f64> = Array2::zeros([2, 2]);
    let mut d2_f_d_r_d_z: Array2<f64> = Array2::zeros([2, 2]);

    // Function values
    f[(0, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_left)];
    f[(0, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_left)];
    f[(1, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_right)];
    f[(1, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_right)];

    // d(psi)/d(r)
    // bz = 1 / (2.0 * PI * r) * d_psi_d_r
    d_f_d_r[(0, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_r[(0, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_r[(1, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
    d_f_d_r[(1, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

    // d(psi)/d(z)
    // br = - 1 / (2.0 * PI * r) * d_psi_d_z
    d_f_d_z[(0, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_z[(0, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_z[(1, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
    d_f_d_z[(1, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

    // d^2(psi)/(d(r)*d(z))
    // d_bz_d_z = 1 / (2 * PI * r) * d2_psi_dr_dz
    d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
    d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
    d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

    // Create a bicubic interpolator
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

    // Find the location of the stationary point in the bicubit fit function:
    // `f(x, y) = [1, x, x^2, x^3] * a_matrix * [1, y, y^2, y^3].T`
    // `d(f(x,y)/d(x) = 0` and `d(f(x,y)/d(y) = 0)`
    // Initial conditions are at the centre of the four corner grid points.
    // From looking at some solutions there does not appear to be multiple maxima,
    // so starting at the centre seems fine.
    let stationary_point_result: Result<BicubicStationaryPoint, String> = bicubic_interpolator.find_stationary_point(0.5, 0.5, 1e-6, 100);

    // Extract the magnetic axis values
    let mag_psi: f64;
    match stationary_point_result {
        Ok(stationary_point) => {
            // Extract and store results
            mag_psi = stationary_point.f;
            mag_r = r[i_r_nearest_left] + stationary_point.x * d_r;
            mag_z = z[i_z_nearest_lower] + stationary_point.y * d_z;
        }
        Err(_error_string) => {
            // The selection of the four bounding grid points was chosen by using linear interpolation.
            // When the magnetic axis is close to one of the grid-point boundaries we can select the wrong four bounding points
            // This will cause `find_stationary_point` to fail and not find the magnetic axis.
            // If we fail to converge onto a solution, then fall back on a brute force method
            // TODO: Perhaps the first part of the fallback should be to try shifting the four corner grid points?
            // println!("Warning, find_stationary_point failed: {}", _error_string);
            // println!("Falling back to brute-force search");
            // println!("a_matrix={:?}", bicubic_interpolator.a_matrix);
            let n_r_test: usize = 30;
            let n_z_test: usize = 33;

            let r_tests: Array1<f64> = Array1::linspace(0.0, 1.0, n_r_test);
            let z_tests: Array1<f64> = Array1::linspace(0.0, 1.0, n_z_test);
            let mut f_test: Array2<f64> = Array2::zeros([n_z_test, n_r_test]);
            for i_r in 0..n_r_test {
                for i_z in 0..n_z_test {
                    f_test[(i_z, i_r)] = bicubic_interpolator.interpolate(r_tests[i_r], z_tests[i_z]);
                }
            }
            mag_psi = f_test.max().unwrap().to_owned();
            let (index_z_max, index_r_max) = f_test
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|((iz, ir), _)| (iz, ir))
                .unwrap();
            mag_r = r[i_r_nearest_left] + r_tests[index_r_max] * d_r;
            mag_z = z[i_z_nearest_lower] + z_tests[index_z_max] * d_z;
        }
    }

    let magnetic_axis: MagneticAxis = MagneticAxis {
        r: mag_r,
        z: mag_z,
        psi: mag_psi,
    };

    return Ok(magnetic_axis);
}

/// Test finding the magnetic axis
///
/// We add a pair of BV PF coils, and a PF coil to represent the plasma.
/// The maximum in `psi` will be where the plasma PF coil is.
/// To make this a useful test I have shifted both the BV PF coils and the
/// plasma upwards, so the function `find_magnetic_axis` should find the
/// magnetic axis at a positive z value.
#[test]
fn test_find_magnetic_axis() {
    // Lazy loading of packages which are not used anywhere else in the code
    use crate::greens::greens_b;
    use crate::greens::greens_d_b_d_z;
    use crate::greens::greens_psi;
    use approx::assert_abs_diff_eq;
    use ndarray::s;

    let z_shift: f64 = 0.02345;
    let n_r: usize = 100;
    let n_z: usize = 201;
    let vec_r: Array1<f64> = Array1::linspace(0.01, 1.0, n_r);
    let vec_z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    // Meshgrid
    let mut mesh_r: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
    let mut mesh_z: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            mesh_r[(i_z, i_r)] = vec_r[i_r];
            mesh_z[(i_z, i_r)] = vec_z[i_z];
        }
    }

    // Flatten 2d mesh
    let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
    let flat_z: Array1<f64> = mesh_z.flatten().to_owned();

    // Create some BV PF coils
    let bv_coils_r: Array1<f64> = Array1::from_vec(vec![1.4, 1.4]);
    let bv_coils_z: Array1<f64> = Array1::from_vec(vec![0.6123 + z_shift, -0.6123 + z_shift]);
    let bv_g: Array2<f64> = greens_psi(
        flat_r.clone(),
        flat_z.clone(),
        bv_coils_r.clone(),
        bv_coils_z.clone(),
        flat_r.clone(),
        flat_z.clone(),
    );
    let (bv_g_br, bv_g_bz): (Array2<f64>, Array2<f64>) = greens_b(flat_r.clone(), flat_z.clone(), bv_coils_r.clone(), bv_coils_z.clone());
    let (_bv_d_g_br_dz, bv_d_g_bz_dz): (Array2<f64>, Array2<f64>) = greens_d_b_d_z(flat_r.clone(), flat_z.clone(), bv_coils_r.clone(), bv_coils_z.clone());

    // Create a PF coil at mid-plane to represent the plasma
    let plasma_coil_r: Array1<f64> = Array1::from_vec(vec![0.41111]);
    let plasma_coil_z: Array1<f64> = Array1::from_vec(vec![0.0 + z_shift]);
    let plasma_g: Array2<f64> = greens_psi(
        flat_r.clone(),
        flat_z.clone(),
        plasma_coil_r.clone(),
        plasma_coil_z.clone(),
        flat_r.clone(),
        flat_z.clone(),
    );
    let (plasma_g_br, plasma_g_bz): (Array2<f64>, Array2<f64>) = greens_b(flat_r.clone(), flat_z.clone(), plasma_coil_r.clone(), plasma_coil_z.clone());
    let (_plasma_d_g_br_dz, plasma_d_g_bz_dz): (Array2<f64>, Array2<f64>) =
        greens_d_b_d_z(flat_r.clone(), flat_z.clone(), plasma_coil_r.clone(), plasma_coil_z.clone());

    // Currents
    let bv_current: f64 = -1.0e5; // Negative current for BV coils, (amperes)
    let plasma_current: f64 = 2.0e5; // Positive current for plasma coil, (amperes)

    let psi: Array1<f64> = bv_current * &bv_g.slice(s![.., 0]) + bv_current * &bv_g.slice(s![.., 1]) + plasma_current * &plasma_g.slice(s![.., 0]);
    let br: Array1<f64> = bv_current * &bv_g_br.slice(s![.., 0]) + bv_current * &bv_g_br.slice(s![.., 1]) + plasma_current * &plasma_g_br.slice(s![.., 0]);
    let bz: Array1<f64> = bv_current * &bv_g_bz.slice(s![.., 0]) + bv_current * &bv_g_bz.slice(s![.., 1]) + plasma_current * &plasma_g_bz.slice(s![.., 0]);
    let d_bz_dz: Array1<f64> =
        bv_current * &bv_d_g_bz_dz.slice(s![.., 0]) + bv_current * &bv_d_g_bz_dz.slice(s![.., 1]) + plasma_current * &plasma_d_g_bz_dz.slice(s![.., 0]);

    let psi_2d: Array2<f64> = psi.to_shape((n_z, n_r)).unwrap().to_owned();
    let br_2d: Array2<f64> = br.to_shape((n_z, n_r)).unwrap().to_owned();
    let bz_2d: Array2<f64> = bz.to_shape((n_z, n_r)).unwrap().to_owned();
    let d_bz_d_z_2d: Array2<f64> = d_bz_dz.to_shape((n_z, n_r)).unwrap().to_owned();

    let mag_r_previous: f64 = 0.51111; // initial condition
    let mag_z_previous: f64 = 0.0; // initial condition

    // let magnetic_axis: MagneticAxis = find_magnetic_axis(&vec_r, &vec_z, &br_2d, &bz_2d, &d_bz_d_z_2d, &psi_2d, mag_r_previous, mag_z_previous)
    //     .expect("test_find_magnetic_axis: unwrapping magnetic_axis_result");

    // // Fairly large epsilon, since we have only given the bicubic interpolator the function
    // // and it's derivatives at the four corner grid points
    // assert_abs_diff_eq!(magnetic_axis.z, 0.0 + z_shift, epsilon = 1e-3);
}
