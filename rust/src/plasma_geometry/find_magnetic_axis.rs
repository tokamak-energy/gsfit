use crate::bicubic_interpolator::bicubic_interpolation;
use contour::ContourBuilder;
use core::f64;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Line, MultiPolygon};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

const PI: f64 = std::f64::consts::PI;

pub fn find_magnetic_axis(
    r: &Array1<f64>,
    z: &Array1<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    _d_bz_d_z: &Array2<f64>,
    psi_2d: &Array2<f64>,
    r_mag_previous: f64,
    z_mag_previous: f64,
) -> (f64, f64, f64) {
    // TODO: idea regarding the bicubic interpolation: the interpolation essentially fits a bicubic polynomial and then evaluates it at the desired point.
    // Instead we can still fit, but differentiate the polynomial to find the maximum in the function, i.e. magnetic axis.
    let mut mag_axis_r: f64 = r_mag_previous; // initial condition
    let mut mag_axis_z: f64 = z_mag_previous; // initial condition

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    // Construct an empty "contour_grid" object
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true) // x dim., y dim., smoothing
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    // Find the contours for br=0
    let br_flattened: Vec<f64> = br_2d.iter().cloned().collect();
    let br_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&br_flattened, &[0.0f64]).expect("br_contours_tmp: error");
    let br_contours: &MultiPolygon = br_contours_tmp[0].geometry(); // The [0] is because I have only supplied one `thresholds`

    // Find the contours for bz=0
    let bz_flattened: Vec<f64> = bz_2d.iter().cloned().collect();
    let bz_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&bz_flattened, &[0.0f64]).expect("bz_contours_tmp: error");
    let bz_contours: &MultiPolygon = bz_contours_tmp[0].geometry(); // The [0] is because I have only supplied one `thresholds`

    // Search for o-points; intersections between br and bz
    // let mut possible_xpts_r: Vec<f64> = Vec::new();
    // let mut possible_xpts_z: Vec<f64> = Vec::new();
    for br_contour in br_contours {
        let br_lines: Vec<Line<_>> = br_contour.exterior().lines().collect();
        for bz_contour in bz_contours {
            let bz_lines: Vec<Line<_>> = bz_contour.exterior().lines().collect();

            // Loop over the lines (lines have two coordinates: start and end)
            for br_line in &br_lines {
                for bz_line in &bz_lines {
                    if let Some(line_intersection) = line_intersection(br_line.to_owned(), bz_line.to_owned()) {
                        if let LineIntersection::SinglePoint {
                            intersection,
                            is_proper: _is_proper,
                        } = line_intersection
                        {
                            // Note:
                            // `_is_proper = true` means the intersection occurs at a point that is not one of the endpoints of either line
                            // `_is_proper = false` means the intersection occurs at an endpoint of one or both line segments
                            let intersection_r: f64 = intersection.x;
                            let intersection_z: f64 = intersection.y;

                            // Test if the x-point is within the range
                            // TODO: I don't like this!
                            // should use Hessian matrix to determine if the intersection is a maximum/minimum or saddle point
                            if intersection_r > 0.17 && intersection_r < 0.8 {
                                if intersection_z > -0.25 && intersection_z < 0.25 {
                                    // add bottom x-point
                                    mag_axis_r = intersection_r;
                                    mag_axis_z = intersection_z;

                                    // println!("find_magnetic_axis: mag_axis_r={}, mag_axis_z={}", mag_axis_r, mag_axis_z);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Interpolate psi at the magnetic axis
    // remember:
    // br = - 1 / (2.0 * PI * r) * d_psi_d_z
    // bz = 1 / (2.0 * PI * r) * d_psi_d_r

    let mut f: Array2<f64> = Array2::zeros([2, 2]);
    let mut d_f_d_r: Array2<f64> = Array2::zeros([2, 2]);
    let mut d_f_d_z: Array2<f64> = Array2::zeros([2, 2]);
    let mut d2_f_d_r_d_z: Array2<f64> = Array2::zeros([2, 2]);

    let i_r_nearest: usize = (r - mag_axis_r).mapv(|x| x.abs()).argmin().unwrap();
    let i_z_nearest: usize = (z - mag_axis_z).mapv(|x| x.abs()).argmin().unwrap();

    let i_r_nearest_left: usize;
    let i_r_nearest_right: usize;
    let i_z_nearest_lower: usize;
    let i_z_nearest_upper: usize;
    if mag_axis_r > r[i_r_nearest] {
        i_r_nearest_left = i_r_nearest;
        i_r_nearest_right = i_r_nearest + 1;
    } else {
        i_r_nearest_left = i_r_nearest - 1;
        i_r_nearest_right = i_r_nearest;
    }
    if mag_axis_z > z[i_z_nearest] {
        i_z_nearest_lower = i_z_nearest;
        i_z_nearest_upper = i_z_nearest + 1;
    } else {
        i_z_nearest_lower = i_z_nearest - 1;
        i_z_nearest_upper = i_z_nearest;
    }

    // Assign the function values
    f[[0, 0]] = psi_2d[[i_z_nearest_lower, i_r_nearest_left]];
    f[[0, 1]] = psi_2d[[i_z_nearest_upper, i_r_nearest_left]];
    f[[1, 0]] = psi_2d[[i_z_nearest_lower, i_r_nearest_right]];
    f[[1, 1]] = psi_2d[[i_z_nearest_upper, i_r_nearest_right]];

    // Use br and bz, which are analytic!
    // d(psi)/d(r)
    d_f_d_r[[0, 0]] = bz_2d[[i_z_nearest_lower, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_r[[0, 1]] = bz_2d[[i_z_nearest_upper, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_r[[1, 0]] = bz_2d[[i_z_nearest_lower, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);
    d_f_d_r[[1, 1]] = bz_2d[[i_z_nearest_upper, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);

    // d(psi)/d(z)
    d_f_d_z[[0, 0]] = -br_2d[[i_z_nearest_lower, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_z[[0, 1]] = -br_2d[[i_z_nearest_upper, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    d_f_d_z[[1, 0]] = -br_2d[[i_z_nearest_lower, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);
    d_f_d_z[[1, 1]] = -br_2d[[i_z_nearest_upper, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);

    // println!("d_f_d_r[lower, left]={}", d_f_d_r[[0, 0]]);
    // println!("d_f_d_r[upper, left]={}", d_f_d_r[[0, 1]]);
    // println!("d_f_d_r[lower, right]={}", d_f_d_r[[1, 0]]);
    // println!("d_f_d_r[upper, right]={}", d_f_d_r[[1, 1]]);
    // println!("d_f_d_z[lower, left]={}", d_f_d_z[[0, 0]]);
    // println!("d_f_d_z[upper, left]={}", d_f_d_z[[0, 1]]);
    // println!("d_f_d_z[lower, right]={}", d_f_d_z[[1, 0]]);
    // println!("d_f_d_z[upper, right]={}", d_f_d_z[[1, 1]]);
    // println!("f={:#}", f);

    // WARNING: Central differencing is unreliable near the peak of the function
    // // should use forward/backward differencing instead
    // // d[ d(f)/d(r) ]/d(z)
    // // d_f_d_r = bz * (2.0 * pi * r)
    // d2_f_d_r_d_z[[0, 0]] = (bz_2d[[i_z_nearest_lower, i_r_nearest_left]] - bz_2d[[i_z_nearest_lower - 1, i_r_nearest_left]]) * (2.0 * PI * r[i_r_nearest_left]) / d_z;
    // d2_f_d_r_d_z[[0, 1]] = (bz_2d[[i_z_nearest_upper + 1, i_r_nearest_left]] - bz_2d[[i_z_nearest_upper, i_r_nearest_left]]) * (2.0 * PI * r[i_r_nearest_left]) / d_z;
    // d2_f_d_r_d_z[[1, 0]] = (bz_2d[[i_z_nearest_lower, i_r_nearest_right]] - bz_2d[[i_z_nearest_lower - 1, i_r_nearest_right]]) * (2.0 * PI * r[i_r_nearest_right]) / d_z;
    // d2_f_d_r_d_z[[1, 1]] = (bz_2d[[i_z_nearest_upper + 1, i_r_nearest_right]] - bz_2d[[i_z_nearest_upper, i_r_nearest_right]]) * (2.0 * PI * r[i_r_nearest_right]) / d_z;
    // println!("d2_f_d_r_d_z={:#}", d2_f_d_r_d_z);

    // // d[ d(f)/d(z) ]/d(r)
    // // d_f_d_z = -br * (2.0 * pi * r)
    // d2_f_d_r_d_z[[0, 0]] = -(br_2d[[i_z_nearest_lower, i_r_nearest_left]] - br_2d[[i_z_nearest_lower, i_r_nearest_left - 1]]) * (2.0 * PI * r[i_r_nearest_left]) / d_r;
    // d2_f_d_r_d_z[[0, 1]] = -(br_2d[[i_z_nearest_upper, i_r_nearest_left]] - br_2d[[i_z_nearest_upper, i_r_nearest_left - 1]]) * (2.0 * PI * r[i_r_nearest_left]) / d_r;
    // d2_f_d_r_d_z[[1, 0]] = -(br_2d[[i_z_nearest_lower, i_r_nearest_right + 1]] - br_2d[[i_z_nearest_lower, i_r_nearest_right]]) * (2.0 * PI * r[i_r_nearest_right]) / d_r;
    // d2_f_d_r_d_z[[1, 1]] = -(br_2d[[i_z_nearest_upper, i_r_nearest_right + 1]] - br_2d[[i_z_nearest_upper, i_r_nearest_right]]) * (2.0 * PI * r[i_r_nearest_right]) / d_r;
    // println!("d2_f_d_r_d_z={:#}", d2_f_d_r_d_z);

    // this seems to be wrong??
    // d2_f_d_r_d_z[[0, 0]] = d_bz_d_z[[i_z_nearest_lower, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    // d2_f_d_r_d_z[[0, 1]] = d_bz_d_z[[i_z_nearest_upper, i_r_nearest_left]] * (2.0 * PI * r[i_r_nearest_left]);
    // d2_f_d_r_d_z[[1, 0]] = d_bz_d_z[[i_z_nearest_lower, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);
    // d2_f_d_r_d_z[[1, 1]] = d_bz_d_z[[i_z_nearest_upper, i_r_nearest_right]] * (2.0 * PI * r[i_r_nearest_right]);
    // println!("d2_f_d_r_d_z={:#}", d2_f_d_r_d_z);

    d2_f_d_r_d_z[[0, 0]] = 0.0;
    d2_f_d_r_d_z[[0, 1]] = 0.0;
    d2_f_d_r_d_z[[1, 0]] = 0.0;
    d2_f_d_r_d_z[[1, 1]] = 0.0;

    // let x: f64 = (mag_axis_r - r[i_r_nearest_left]) / d_r;
    // let y: f64 = (mag_axis_z - z[i_z_nearest_lower]) / d_z;
    // let mag_axis_psi: f64 = bicubic_interpolation(&f, &(&d_f_d_r / d_r), &(&d_f_d_z / d_z), &(&d2_f_d_r_d_z / (d_z * d_r)), y, x);

    let n_r_test: usize = 30;
    let n_z_test: usize = 33;

    let r_tests: Array1<f64> = Array1::linspace(0.0001, 0.9999, n_r_test);
    let z_tests: Array1<f64> = Array1::linspace(0.0001, 0.9999, n_z_test);
    let mut f_test: Array2<f64> = Array2::zeros([n_z_test, n_r_test]);
    for i_r in 0..n_r_test {
        for i_z in 0..n_z_test {
            f_test[[i_z, i_r]] = bicubic_interpolation(&f, &(&d_f_d_r * d_r), &(&d_f_d_z * d_z), &d2_f_d_r_d_z, r_tests[i_r], z_tests[i_z]);
        }
    }

    // TODO very sketchy fix!!
    let mag_axis_psi: f64 = f_test.max().unwrap().to_owned();

    return (mag_axis_r, mag_axis_z, mag_axis_psi);
}
