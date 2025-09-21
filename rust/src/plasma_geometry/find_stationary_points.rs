use super::BoundaryContour;
use crate::bicubic_interpolator::BicubicInterpolator;
use crate::bicubic_interpolator::BicubicStationaryPoint;
use crate::greens::D2PsiDR2Calculator;
use crate::plasma_geometry::hessian;
use contour::ContourBuilder;
use core::f64;
use geo::Contains;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Coord, Line, LineString, MultiPolygon, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;
use ndarray_stats::QuantileExt;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct StationaryPoint {
    pub r: f64,
    pub z: f64,
    pub psi: f64,
    pub hessian_determinant: f64,
    pub hessian_trace: f64,
}

pub fn find_stationary_points(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_br_d_z_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    d2_psi_d_r2_calculator: D2PsiDR2Calculator,
) -> Result<Vec<StationaryPoint>, String> {
    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    // Construct an empty `contour_grid` object
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true)
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    // Find the contours for br=0
    let br_flattened: Vec<f64> = br_2d.flatten().to_vec();
    let br_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&br_flattened, &[0.0f64]).expect("br_contours_tmp: error");
    let br_contours: &MultiPolygon = br_contours_tmp[0].geometry(); // The [0] is because I have only supplied one `thresholds`
    if br_contours.iter().count() == 0 {
        return Err("find_viable_xpt: no br=0 contours found".to_string());
    }

    // Find the contours for bz=0
    let bz_flattened: Vec<f64> = bz_2d.flatten().to_vec();
    let bz_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&bz_flattened, &[0.0f64]).expect("bz_contours_tmp: error");
    let bz_contours: &MultiPolygon = bz_contours_tmp[0].geometry(); // The [0] is because I have only supplied one `thresholds`
    if bz_contours.iter().count() == 0 {
        return Err("find_viable_xpt: no bz=0 contours found".to_string());
    }

    // Collect br contours
    let mut br_contour_line_segments: Vec<Line<f64>> = Vec::new();
    // Loop over all contours
    for br_contour in br_contours {
        // "exterior" is the outer boundary of the contour
        // There is only one "exterior"
        // `Line` object has two coordinates: "start" and "end", so there will be multiple Lines
        let br_line: Vec<Line<f64>> = br_contour.exterior().lines().collect();
        for line_segment in br_line {
            br_contour_line_segments.push(line_segment);
        }

        // "interiors" are the holes in the contour
        // There can be multiple holes or None
        // `Line` object has two coordinates: "start" and "end", so there will be multiple Lines
        for br_interior in br_contour.interiors() {
            let br_line: Vec<Line<f64>> = br_interior.lines().collect();
            for line_segment in br_line {
                br_contour_line_segments.push(line_segment);
            }
        }
    }

    // Collect bz contours
    let mut bz_contour_line_segments: Vec<Line<f64>> = Vec::new();
    // Loop over all contours
    for bz_contour in bz_contours {
        // "exterior" is the outer boundary of the contour
        // There is only one "exterior"
        // `LineString` object has a list of x and y coordinate pairs
        // `Line` object has two coordinates: "start" and "end". When casting into `Line` object we can expect many entries in the Vec
        let bz_line: Vec<Line<f64>> = bz_contour.exterior().lines().collect();
        for line_segment in bz_line {
            bz_contour_line_segments.push(line_segment);
        }

        // "interiors" are the holes in the contour
        // There can be multiple holes or None
        // `Line` object has two coordinates: "start" and "end", so there will be multiple Lines
        for bz_interior in bz_contour.interiors() {
            let bz_line: Vec<Line<f64>> = bz_interior.lines().collect();
            for line_segment in bz_line {
                bz_contour_line_segments.push(line_segment);
            }
        }
    }

    // Create an interpolator for psi
    let psi_interpolator = Interp2D::builder(psi_2d.clone())
        .x(z.clone())
        .y(r.clone())
        .build()
        .expect("find_stationary_points: Can't make Interp2D");

    // Loop over the `br` and `bz` contours to find the intersections, which are the stationary points
    let mut stationary_points: Vec<StationaryPoint> = Vec::new();
    for br_line in &br_contour_line_segments {
        for bz_line in &bz_contour_line_segments {
            if let Some(line_intersection) = line_intersection(br_line.to_owned(), bz_line.to_owned()) {
                if let LineIntersection::SinglePoint {
                    intersection,
                    is_proper: _is_proper,
                } = line_intersection
                {
                    // Note:
                    // `_is_proper` is a variable which is assigned but not used. It means:
                    // * `is_proper = true` means the intersection occurs at a point that is not one of the endpoints of either line
                    // * `is_proper = false` means the intersection occurs at an endpoint of one or both line segments
                    let mut stationary_r: f64 = intersection.x;
                    let mut stationary_z: f64 = intersection.y;

                    // TODO: Very worringly, the contours can be off by 1/2 a grid point.
                    // This has been reported to GitHub as an issue. But until it's fixed we shall add this extra test
                    // that the intersection is within the grid.
                    if stationary_r < r.max().expect("find_viable_xpt: r.max()").to_owned()
                        && stationary_r > r.min().expect("find_viable_xpt: r.min()").to_owned()
                        && stationary_z < z.max().expect("find_viable_xpt: z.max()").to_owned()
                        && stationary_z > z.min().expect("find_viable_xpt: z.max()").to_owned()
                    {
                        // TODO: need to use bicubic interpolator. So that the boundary is consistent with `psi_boundary`

                        // Find the closest grid point
                        let mut i_r_nearest: usize = 0;
                        let mut min_r_dist: f64 = f64::INFINITY;
                        for (i, &r_val) in r.iter().enumerate() {
                            let dist: f64 = (stationary_r - r_val).abs();
                            if dist < min_r_dist {
                                min_r_dist = dist;
                                i_r_nearest = i;
                            }
                        }
                        let mut i_z_nearest: usize = 0;
                        let mut min_z_dist: f64 = f64::INFINITY;
                        for (i, &z_val) in z.iter().enumerate() {
                            let dist: f64 = (stationary_z - z_val).abs();
                            if dist < min_z_dist {
                                min_z_dist = dist;
                                i_z_nearest = i;
                            }
                        }

                        // Calculate the Hessian at the nearest grid point
                        // d^2(psi)/(d_r^2)
                        let d2_psi_d_r2: f64 = d2_psi_d_r2_calculator.calculate(i_r_nearest, i_z_nearest);

                        // d^2(psi)/(d_z^2)
                        // Check:
                        // d(psi)/d(z) = -2.0 * PI * r * br
                        // d^2(psi)/(d(z)^2) = -2.0 * PI * r * d(br)/d(z)
                        let d2_psi_d_z2: f64 = -2.0 * PI * r[i_r_nearest] * d_br_d_z_2d[(i_z_nearest, i_r_nearest)];

                        // d^2(psi)/(d_r * d_z)
                        // Check:
                        // d(psi)/d(r) = 2.0 * PI * r * bz
                        // d^2(psi)/(d(r) * d(z)) = 2.0 * PI * r * d(bz)/d(z)
                        let d2_psi_d_r_d_z: f64 = 2.0 * PI * r[i_r_nearest] * d_bz_d_z_2d[(i_z_nearest, i_r_nearest)];

                        let (hessian_det, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);

                        let stationary_psi: f64 = psi_interpolator
                            .interp_scalar(stationary_z, stationary_r)
                            .expect("find_stationary_points: can't interpolate psi");

                        // // Calculate `psi` at the stationary point
                        // // Find the four corner grid points surrounding the magnetic axis
                        // let i_r_nearest_left: usize;
                        // let i_r_nearest_right: usize;
                        // let i_z_nearest_lower: usize;
                        // let i_z_nearest_upper: usize;
                        // if stationary_r > r[i_r_nearest] {
                        //     i_r_nearest_left = i_r_nearest;
                        //     i_r_nearest_right = i_r_nearest + 1;
                        // } else {
                        //     i_r_nearest_left = i_r_nearest - 1;
                        //     i_r_nearest_right = i_r_nearest;
                        // }
                        // if stationary_z > z[i_z_nearest] {
                        //     i_z_nearest_lower = i_z_nearest;
                        //     i_z_nearest_upper = i_z_nearest + 1;
                        // } else {
                        //     i_z_nearest_lower = i_z_nearest - 1;
                        //     i_z_nearest_upper = i_z_nearest;
                        // }

                        // // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
                        // let mut f: Array2<f64> = Array2::zeros([2, 2]);
                        // let mut d_f_d_r: Array2<f64> = Array2::zeros([2, 2]);
                        // let mut d_f_d_z: Array2<f64> = Array2::zeros([2, 2]);
                        // let mut d2_f_d_r_d_z: Array2<f64> = Array2::zeros([2, 2]);

                        // // Function values
                        // f[(0, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_left)];
                        // f[(0, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_left)];
                        // f[(1, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_right)];
                        // f[(1, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_right)];

                        // // d(psi)/d(r)
                        // // bz = 1 / (2.0 * PI * r) * d_psi_d_r
                        // d_f_d_r[(0, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d_f_d_r[(0, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d_f_d_r[(1, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                        // d_f_d_r[(1, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                        // // d(psi)/d(z)
                        // // br = - 1 / (2.0 * PI * r) * d_psi_d_z
                        // d_f_d_z[(0, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d_f_d_z[(0, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d_f_d_z[(1, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                        // d_f_d_z[(1, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                        // // d^2(psi)/(d(r)*d(z))
                        // // d_bz_d_z = 1 / (2 * PI * r) * d2_psi_dr_dz
                        // // TODO: d_bz_d_z_2d has a delta_z correction missing!
                        // d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        // d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                        // d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                        // // Create a bicubic interpolator
                        // let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

                        // // Find the location of the stationary point in the bicubit fit function:
                        // // `f(x, y) = [1, x, x^2, x^3] * a_matrix * [1, y, y^2, y^3].T`
                        // // `d(f(x,y)/d(x) = 0` and `d(f(x,y)/d(y) = 0)`
                        // // Initial conditions are at the centre of the four corner grid points.
                        // // From looking at some solutions there does not appear to be multiple maxima,
                        // // so starting at the centre seems fine.
                        // let x_start: f64 = (stationary_r - r[i_r_nearest_left]) / d_r;
                        // let y_start: f64 = (stationary_z - z[i_z_nearest_lower]) / d_z;
                        // // let stationary_point_or_error: Result<BicubicStationaryPoint, String> = bicubic_interpolator.find_stationary_point(0.5, 0.5, 1e-6, 100);
                        // println!("Starting search for stationary point at x_start={x_start}, y_start={y_start}");
                        // let stationary_point_or_error: Result<BicubicStationaryPoint, String> = bicubic_interpolator.find_stationary_point(x_start, y_start, 1e-6, 100);

                        // // Extract the magnetic axis values
                        // let stationary_psi: f64;
                        // match stationary_point_or_error {
                        //     Ok(stationary_point) => {
                        //         // Extract and store results
                        //         println!("Found stationary point at x={}, y={}", stationary_point.x, stationary_point.y);
                        //         stationary_psi = stationary_point.f;
                        //         stationary_r = r[i_r_nearest_left] + stationary_point.x * d_r;
                        //         stationary_z = z[i_z_nearest_lower] + stationary_point.y * d_z;
                        //     }
                        //     Err(_error_string) => {
                        //         println!("Warning, find_stationary_point failed: {}", _error_string);
                        //         println!("Falling back to linear interpolation");
                        //         // For now we will fall back on the linear interpolation.
                        //         // TODO: figure out correct fallback, magnetic axis will be maximum in psi, but saddle point won't be??
                        //         // Interpolate `psi` at the stationary point
                        //         stationary_psi = psi_interpolator
                        //             .interp_scalar(stationary_z, stationary_r)
                        //             .expect("find_stationary_points: can't interpolate psi");

                        //         // // The selection of the four bounding grid points was chosen by using linear interpolation.
                        //         // // When the magnetic axis is close to one of the grid-point boundaries we can select the wrong four bounding points
                        //         // // This will cause `find_stationary_point` to fail and not find the magnetic axis.
                        //         // // If we fail to converge onto a solution, then fall back on a brute force method
                        //         // // TODO: Perhaps the first part of the fallback should be to try shifting the four corner grid points?
                        //         // // println!("Warning, find_stationary_point failed: {}", _error_string);
                        //         // // println!("Falling back to brute-force search");
                        //         // // println!("a_matrix={:?}", bicubic_interpolator.a_matrix);
                        //         // let n_r_test: usize = 30;
                        //         // let n_z_test: usize = 33;

                        //         // let r_tests: Array1<f64> = Array1::linspace(0.0, 1.0, n_r_test);
                        //         // let z_tests: Array1<f64> = Array1::linspace(0.0, 1.0, n_z_test);
                        //         // let mut f_test: Array2<f64> = Array2::zeros([n_z_test, n_r_test]);
                        //         // for i_r in 0..n_r_test {
                        //         //     for i_z in 0..n_z_test {
                        //         //         f_test[(i_z, i_r)] = bicubic_interpolator.interpolate(r_tests[i_r], z_tests[i_z]);
                        //         //     }
                        //         // }
                        //         // // TODO: logic wrong!!
                        //         // stationary_psi = f_test.max().unwrap().to_owned();
                        //         // let (index_z_max, index_r_max) = f_test
                        //         //     .indexed_iter()
                        //         //     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        //         //     .map(|((iz, ir), _)| (iz, ir))
                        //         //     .unwrap();
                        //         // stationary_r = r[i_r_nearest_left] + r_tests[index_r_max] * d_r;
                        //         // stationary_z = z[i_z_nearest_lower] + z_tests[index_z_max] * d_z;
                        //     }
                        // }

                        stationary_points.push(StationaryPoint {
                            r: stationary_r,
                            z: stationary_z,
                            psi: stationary_psi,
                            hessian_determinant: hessian_det,
                            hessian_trace: hessian_trace,
                        });
                    }
                }
            }
        }
    }

    // Exit if we haven't found any stationary points
    if stationary_points.len() == 0 {
        return Err("find_stationary_points: no intersection between `br` and `bz` contours found".to_string());
    }

    // Return
    Ok(stationary_points)
}
