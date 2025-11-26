use super::bicubic_interpolator::{BicubicInterpolator, BicubicStationaryPoint};
use crate::greens::D2PsiDR2Calculator;
use crate::plasma_geometry::hessian;
use contour::ContourBuilder;
use core::f64;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Line, MultiPolygon};
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
    pub i_r_nearest: usize,
    pub i_z_nearest: usize,
    pub i_r_nearest_left: usize,
    pub i_r_nearest_right: usize,
    pub i_z_nearest_lower: usize,
    pub i_z_nearest_upper: usize,
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
) -> Result<Vec<StationaryPoint>, String> {  // I'm thinking of always returning a Vec, even if it's empty?
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
                        let d2_psi_d_z2: f64 = -2.0 * PI * r[i_r_nearest] * d_br_d_z_2d[(i_z_nearest, i_r_nearest)];

                        // d^2(psi)/(d_r * d_z)
                        let d2_psi_d_r_d_z: f64 = 2.0 * PI * r[i_r_nearest] * d_bz_d_z_2d[(i_z_nearest, i_r_nearest)];

                        let (hessian_det, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);

                        // Calculate the stationary point using bicubic interpolation
                        // Find the four corner grid points surrounding the magnetic axis
                        let i_r_nearest_left: usize;
                        let i_r_nearest_right: usize;
                        let i_z_nearest_lower: usize;
                        let i_z_nearest_upper: usize;
                        if stationary_r > r[i_r_nearest] {
                            i_r_nearest_left = i_r_nearest;
                            i_r_nearest_right = i_r_nearest + 1;
                        } else {
                            i_r_nearest_left = i_r_nearest - 1;
                            i_r_nearest_right = i_r_nearest;
                        }
                        if stationary_z > z[i_z_nearest] {
                            i_z_nearest_lower = i_z_nearest;
                            i_z_nearest_upper = i_z_nearest + 1;
                        } else {
                            i_z_nearest_lower = i_z_nearest - 1;
                            i_z_nearest_upper = i_z_nearest;
                        }

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
                        // TODO: d_bz_d_z_2d has a delta_z correction missing!  <-- I think I have fixed this in `gs_solution`
                        d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
                        d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
                        d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

                        // Create a bicubic interpolator
                        let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

                        // Find the location of the stationary point in the bicubic fit function:
                        // `f(x, y) = [1, x, x^2, x^3] * a_matrix * [1, y, y^2, y^3].T`
                        // `d(f(x,y)/d(x) = 0` and `d(f(x,y)/d(y) = 0)`
                        // because this is a cubic function in both `x` and `y` we need to use an iterative method to find the stationary point

                        // Initial conditions from the linear interpolation
                        let x_start: f64 = (stationary_r - r[i_r_nearest_left]) / d_r;
                        let y_start: f64 = (stationary_z - z[i_z_nearest_lower]) / d_z;

                        // Find the stationary point using the bicubic interpolaton
                        // The reason why `stationary_point_or_error` sometimes fails to converge is due to "resolution" issues,
                        // where the features are smaller than the grid spacing. This happens particularly near the PF coils,
                        // such as DIV/SOL. Because we are effectively randomly sampling the function the contouring can contain
                        // intersections between `br=0` and `bz=0` contours which are not real stationary points.
                        let stationary_point_or_error: Result<BicubicStationaryPoint, String> =
                            bicubic_interpolator.find_stationary_point(x_start, y_start, 1e-6, 100);

                        // Extract the magnetic axis values
                        let stationary_psi: f64;
                        match stationary_point_or_error {
                            Ok(stationary_point) => {
                                // Extract and store results
                                // println!("Found stationary point at x={}, y={}", stationary_point.x, stationary_point.y);
                                stationary_psi = stationary_point.f;
                                stationary_r = r[i_r_nearest_left] + stationary_point.x * d_r;
                                stationary_z = z[i_z_nearest_lower] + stationary_point.y * d_z;
                            }
                            Err(_error_string) => {
                                // For now we will fall back on the linear interpolation.
                                stationary_psi = psi_interpolator
                                    .interp_scalar(stationary_z, stationary_r)
                                    .expect("find_stationary_points: can't interpolate psi");

                                // If we fail to converge onto a solution, then fall back on a brute force method
                                // TODO: Perhaps the first part of the fallback should be to try shifting the four corner grid points?
                            }
                        }

                        stationary_points.push(StationaryPoint {
                            r: stationary_r,
                            z: stationary_z,
                            psi: stationary_psi,
                            hessian_determinant: hessian_det,
                            hessian_trace: hessian_trace,
                            i_r_nearest,
                            i_z_nearest,
                            i_r_nearest_left,
                            i_r_nearest_right,
                            i_z_nearest_lower,
                            i_z_nearest_upper,
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
