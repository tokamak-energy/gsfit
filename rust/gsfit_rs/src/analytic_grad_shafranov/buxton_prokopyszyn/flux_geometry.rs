//! Geometry of `(1 - psi_N)(R, Z)`: stationary points (the X-points and the magnetic axis), the LCFS contour, Gauss-Legendre quadrature
//! over the plasma, and which points lie inside a polygon.
//!
//! The flux functions take points `(R, Z)` and return `(1 - psi_N)` (`value_function`) or `(1 - psi_N)` and its derivatives
//! (`flux_function`), which is 1 on the magnetic axis and 0 on the LCFS.

use super::configuration::{Point, Points};
use super::lattice::FluxValues;
use super::numerics::{GaussLegendre, gauss_legendre};
use ndarray::Array1;
use std::f64::consts::PI;

/// Gauss-Legendre nodes and weights over the plasma, `int f dR dZ ~ sum weight f(node_r, node_z)`
pub struct PlasmaQuadrature {
    /// [metre]
    pub node_r: Array1<f64>,
    /// [metre]
    pub node_z: Array1<f64>,
    /// [metre ** 2]
    pub weight: Array1<f64>,
}

/// The separatrix's crossings of horizontal lines
struct HeightCrossings {
    /// The crossing outboard of the segment from the axis to the X-point, NaN where there is none [metre]
    outboard_r: Array1<f64>,
    /// The crossing inboard of it [metre]
    inboard_r: Array1<f64>,
}

/// The solution of the 2 x 2 system `matrix step = rhs` by Gaussian elimination with partial pivoting
pub fn solve_2x2(matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> [f64; 2] {
    let (row_pivot, row_other): (usize, usize) = if matrix[1][0].abs() > matrix[0][0].abs() { (1, 0) } else { (0, 1) };
    let factor: f64 = matrix[row_other][0] / matrix[row_pivot][0];
    let reduced_11: f64 = matrix[row_other][1] - factor * matrix[row_pivot][1];
    let reduced_rhs: f64 = rhs[row_other] - factor * rhs[row_pivot];
    let solution_1: f64 = reduced_rhs / reduced_11;
    let solution_0: f64 = (rhs[row_pivot] - matrix[row_pivot][1] * solution_1) / matrix[row_pivot][0];
    [solution_0, solution_1]
}

/// A stationary point of `(1 - psi_N)` (an X-point, or the magnetic axis) near a guess, by Newton's method on its gradient being zero,
/// with a step of at most `0.05 a_minor`.
///
/// # Arguments
/// * `flux_function` - `(1 - psi_N)` and its derivatives at points
/// * `guess` - starting point [metre]
/// * `a_minor` - the paper's `a_minor`, the scale of the steps [metre]
///
/// # Returns
/// * the stationary point [metre]
pub fn find_stationary_point<F>(flux_function: F, guess: Point, a_minor: f64) -> Point
where
    F: Fn(&[f64], &[f64]) -> FluxValues,
{
    let n_iteration_max: usize = 50;
    let step_max: f64 = 0.05 * a_minor;
    let mut point_r: f64 = guess.r;
    let mut point_z: f64 = guess.z;
    'newton_loop: for _i_iteration in 0..n_iteration_max {
        let flux: FluxValues = flux_function(&[point_r], &[point_z]);
        let gradient: [f64; 2] = [flux.d_d_r[0], flux.d_d_z[0]];
        let hessian: [[f64; 2]; 2] = [[flux.d2_d_r2[0], flux.d2_d_r_d_z[0]], [flux.d2_d_r_d_z[0], flux.d2_d_z2[0]]];
        let mut step: [f64; 2] = solve_2x2(hessian, gradient);
        let step_length: f64 = step[0].hypot(step[1]);
        if step_length > step_max {
            step = [step[0] * step_max / step_length, step[1] * step_max / step_length];
        }
        point_r -= step[0];
        point_z -= step[1];
        if step_length < 1.0e-14 * a_minor {
            break 'newton_loop;
        }
    }
    Point { r: point_r, z: point_z }
}

/// The distance from the magnetic axis to the LCFS, `(1 - psi_N) = 0`, along rays: the first crossing of each ray, refined by bisection
/// to rounding.
///
/// # Arguments
/// * `value_function` - `(1 - psi_N)(R, Z)` at points [dimensionless]
/// * `centre` - the magnetic axis [metre]
/// * `angles` - angles of the rays, anticlockwise from the outboard direction [radian]
/// * `radius_max` - longest ray [metre]
/// * `r_minimum` - rays are not followed below this `R` (the inboard end of where the radial series can be evaluated) [metre]
///
/// # Returns
/// * the length of each ray to its crossing, NaN where it has none [metre]
pub fn lcfs_radii_along_rays<F>(value_function: &F, centre: Point, angles: &Array1<f64>, radius_max: f64, r_minimum: f64) -> Array1<f64>
where
    F: Fn(&[f64], &[f64]) -> Array1<f64>,
{
    let n_point: usize = angles.len();
    let mut lower: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
    let mut upper: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
    // A coarse scan of every ray, then a fine one of the rays it found no crossing on (the series are costly to evaluate)
    for n_step in [150_usize, 1200] {
        let mut rays: Vec<usize> = Vec::new();
        for i_point in 0..n_point {
            if lower[i_point].is_nan() {
                rays.push(i_point);
            }
        }
        let n_ray: usize = rays.len();
        if n_ray == 0 {
            break;
        }
        let steps: Array1<f64> = Array1::linspace(0.0, radius_max, n_step);
        let mut ray_r: Vec<f64> = Vec::with_capacity(n_ray * n_step);
        let mut ray_z: Vec<f64> = Vec::with_capacity(n_ray * n_step);
        for i_ray in 0..n_ray {
            for i_step in 0..n_step {
                ray_r.push((centre.r + angles[rays[i_ray]].cos() * steps[i_step]).max(r_minimum));
                ray_z.push(centre.z + angles[rays[i_ray]].sin() * steps[i_step]);
            }
        }
        let values: Array1<f64> = value_function(&ray_r, &ray_z);
        for i_ray in 0..n_ray {
            'step_loop: for i_step in 0..n_step - 1 {
                if values[i_ray * n_step + i_step] > 0.0 && 0.0 >= values[i_ray * n_step + i_step + 1] {
                    lower[rays[i_ray]] = steps[i_step];
                    upper[rays[i_ray]] = steps[i_step + 1];
                    break 'step_loop;
                }
            }
        }
    }
    let n_bisection: usize = 50;
    for _i_bisection in 0..n_bisection {
        let middle: Array1<f64> = (&lower + &upper) * 0.5;
        let mut middle_r: Vec<f64> = Vec::with_capacity(n_point);
        let mut middle_z: Vec<f64> = Vec::with_capacity(n_point);
        for i_point in 0..n_point {
            middle_r.push(centre.r + angles[i_point].cos() * middle[i_point]);
            middle_z.push(centre.z + angles[i_point].sin() * middle[i_point]);
        }
        let middle_values: Array1<f64> = value_function(&middle_r, &middle_z);
        for i_point in 0..n_point {
            if middle_values[i_point] > 0.0 {
                lower[i_point] = middle[i_point];
            } else {
                upper[i_point] = middle[i_point];
            }
        }
    }
    (&lower + &upper) * 0.5
}

/// The separatrix's outboard and inboard crossings of horizontal lines between the magnetic axis' height and an X-point's: on each line
/// the first `(1 - psi_N) = 0` crossings outboard and inboard of the segment from the axis to the X-point (which lies inside the
/// separatrix), refined by bisection to rounding. Horizontal lines between the axis and the X-point never enter the private-flux region
/// beyond it, where `(1 - psi_N) > 0` again, which rays through the X-point can.
fn lcfs_crossings_at_heights<F>(value_function: &F, centre: Point, x_point: Point, heights: &Array1<f64>, r_minimum: f64, r_maximum: f64) -> HeightCrossings
where
    F: Fn(&[f64], &[f64]) -> Array1<f64>,
{
    let n_height: usize = heights.len();
    let mut start_r: Array1<f64> = Array1::from_elem(n_height, f64::NAN);
    for i_height in 0..n_height {
        start_r[i_height] = centre.r + (x_point.r - centre.r) * (heights[i_height] - centre.z) / (x_point.z - centre.z);
    }
    let mut crossings: Vec<Array1<f64>> = Vec::with_capacity(2);
    for r_end in [r_maximum, r_minimum] {
        let mut lower: Array1<f64> = Array1::from_elem(n_height, f64::NAN);
        let mut upper: Array1<f64> = Array1::from_elem(n_height, f64::NAN);
        // A coarse scan of every line, then a fine one of the lines it found no crossing on (the series are costly to evaluate)
        for n_scan in [100_usize, 800] {
            let mut lines: Vec<usize> = Vec::new();
            for i_height in 0..n_height {
                if lower[i_height].is_nan() {
                    lines.push(i_height);
                }
            }
            let n_line: usize = lines.len();
            if n_line == 0 {
                break;
            }
            let scan: Array1<f64> = Array1::linspace(0.0, 1.0, n_scan);
            let mut line_r: Vec<f64> = Vec::with_capacity(n_line * n_scan);
            let mut line_z: Vec<f64> = Vec::with_capacity(n_line * n_scan);
            for i_line in 0..n_line {
                for i_scan in 0..n_scan {
                    line_r.push(start_r[lines[i_line]] + (r_end - start_r[lines[i_line]]) * scan[i_scan]);
                    line_z.push(heights[lines[i_line]]);
                }
            }
            let values: Array1<f64> = value_function(&line_r, &line_z);
            for i_line in 0..n_line {
                'scan_loop: for i_scan in 0..n_scan - 1 {
                    if values[i_line * n_scan + i_scan] > 0.0 && 0.0 >= values[i_line * n_scan + i_scan + 1] {
                        lower[lines[i_line]] = scan[i_scan];
                        upper[lines[i_line]] = scan[i_scan + 1];
                        break 'scan_loop;
                    }
                }
            }
        }
        let n_bisection: usize = 50;
        for _i_bisection in 0..n_bisection {
            let middle: Array1<f64> = (&lower + &upper) * 0.5;
            let mut middle_r: Vec<f64> = Vec::with_capacity(n_height);
            for i_height in 0..n_height {
                middle_r.push(start_r[i_height] + middle[i_height] * (r_end - start_r[i_height]));
            }
            let middle_values: Array1<f64> = value_function(&middle_r, heights.as_slice().expect("contiguous"));
            for i_height in 0..n_height {
                if middle_values[i_height] > 0.0 {
                    lower[i_height] = middle[i_height];
                } else {
                    upper[i_height] = middle[i_height];
                }
            }
        }
        let mut crossing: Array1<f64> = Array1::from_elem(n_height, f64::NAN);
        for i_height in 0..n_height {
            crossing[i_height] = start_r[i_height] + 0.5 * (lower[i_height] + upper[i_height]) * (r_end - start_r[i_height]);
        }
        crossings.push(crossing);
    }
    let inboard_r: Array1<f64> = crossings.pop().expect("two crossings");
    let outboard_r: Array1<f64> = crossings.pop().expect("two crossings");
    HeightCrossings { outboard_r, inboard_r }
}

/// The X-point on one side of the axis' height (at most one above and one below the axis)
fn x_point_on_side(x_points: &[Point], centre: Point, is_upper: bool) -> Option<Point> {
    let mut found: Option<Point> = None;
    for x_point in x_points {
        if (x_point.z > centre.z) == is_upper {
            found = Some(*x_point);
        }
    }
    found
}

/// Gauss-Legendre nodes and weights over the plasma, half by half as `lcfs_contour` traces the LCFS: above and below the axis' height,
/// in heights and then across between the separatrix' crossings up to an X-point on that side, or in angle and then along each ray to
/// the LCFS where that side is smooth. Each half's boundary is analytic up to its ends (a separatrix' branches are analytic through the
/// X-point), so the rule converges exponentially for an integrand analytic inside, even one that does not vanish on the LCFS (constant
/// terms), unlike a grid's.
///
/// # Arguments
/// * `value_function` - `(1 - psi_N)(R, Z)`, positive inside [dimensionless]
/// * `centre` - the magnetic axis [metre]
/// * `x_points` - the X-points, at most one above and one below the axis [metre]
/// * `n_node` - Gauss-Legendre nodes in each direction of each half [dimensionless]
/// * `r_minimum` - inboard end of the search for the LCFS [metre]
/// * `r_maximum` - outboard end of the search [metre]
/// * `radius_max` - longest ray [metre]
///
/// # Returns
/// * `2 n_node**2` nodes [metre] and their weights [metre ** 2]
#[allow(clippy::too_many_arguments)]
pub fn plasma_quadrature<F>(
    value_function: &F,
    centre: Point,
    x_points: &[Point],
    n_node: usize,
    r_minimum: f64,
    r_maximum: f64,
    radius_max: f64,
) -> PlasmaQuadrature
where
    F: Fn(&[f64], &[f64]) -> Array1<f64>,
{
    let legendre: GaussLegendre = gauss_legendre(n_node);
    // On [0, 1]
    let fractions: Array1<f64> = legendre.nodes.mapv(|node: f64| 0.5 * (node + 1.0));
    let fraction_weights: Array1<f64> = legendre.weights.mapv(|weight: f64| 0.5 * weight);
    let mut node_r: Vec<f64> = Vec::with_capacity(2 * n_node * n_node);
    let mut node_z: Vec<f64> = Vec::with_capacity(2 * n_node * n_node);
    let mut weight: Vec<f64> = Vec::with_capacity(2 * n_node * n_node);
    for is_upper in [true, false] {
        match x_point_on_side(x_points, centre, is_upper) {
            None => {
                // In angle over the half, then along each ray: dR dZ = rho drho dtheta
                let angle_start: f64 = if is_upper { 0.0 } else { PI };
                let angles: Array1<f64> = fractions.mapv(|fraction: f64| angle_start + PI * fraction);
                let radii: Array1<f64> = lcfs_radii_along_rays(value_function, centre, &angles, radius_max, r_minimum);
                for i_angle in 0..n_node {
                    for i_radius in 0..n_node {
                        let rho: f64 = radii[i_angle] * fractions[i_radius];
                        node_r.push(centre.r + angles[i_angle].cos() * rho);
                        node_z.push(centre.z + angles[i_angle].sin() * rho);
                        weight.push(PI * fraction_weights[i_angle] * radii[i_angle] * fraction_weights[i_radius] * rho);
                    }
                }
            }
            Some(x_point) => {
                // In height from the axis' to the X-point's, then across between the separatrix' crossings
                let heights: Array1<f64> = fractions.mapv(|fraction: f64| centre.z + (x_point.z - centre.z) * fraction);
                let crossings: HeightCrossings = lcfs_crossings_at_heights(value_function, centre, x_point, &heights, r_minimum, r_maximum);
                for i_height in 0..n_node {
                    let width: f64 = crossings.outboard_r[i_height] - crossings.inboard_r[i_height];
                    for i_across in 0..n_node {
                        node_r.push(crossings.inboard_r[i_height] + width * fractions[i_across]);
                        node_z.push(heights[i_height]);
                        weight.push((x_point.z - centre.z).abs() * fraction_weights[i_height] * width * fraction_weights[i_across]);
                    }
                }
            }
        }
    }
    PlasmaQuadrature {
        node_r: Array1::from(node_r),
        node_z: Array1::from(node_z),
        weight: Array1::from(weight),
    }
}

/// The closed LCFS, `(1 - psi_N) = 0`, around the magnetic axis, half by half: above and below the axis' height, along horizontal lines up to an
/// X-point on that side, or along rays where that side is smooth.
///
/// # Arguments
/// * `value_function` - `(1 - psi_N)(R, Z)` [dimensionless]
/// * `centre` - the magnetic axis [metre]
/// * `x_points` - the X-points, at most one above and one below the axis [metre]
/// * `n_point` - about the number of points on the contour [dimensionless]
/// * `r_minimum` - inboard end of the search [metre]
/// * `r_maximum` - outboard end of the search [metre]
/// * `radius_max` - longest ray [metre]
///
/// # Returns
/// * anticlockwise from the outboard side of the axis' height, closed (the last point repeats the first); points of a horizontal line
///   that has no crossing (just short of an X-point whose `(1 - psi_N)` is slightly negative) are left out, and the contour is empty
///   if no crossing was found at all [metre]
#[allow(clippy::too_many_arguments)]
pub fn lcfs_contour<F>(value_function: &F, centre: Point, x_points: &[Point], n_point: usize, r_minimum: f64, r_maximum: f64, radius_max: f64) -> Points
where
    F: Fn(&[f64], &[f64]) -> Array1<f64>,
{
    let n_half: usize = (n_point / 2).max(8);
    let mut contour_r: Vec<f64> = Vec::new();
    let mut contour_z: Vec<f64> = Vec::new();
    for is_upper in [true, false] {
        match x_point_on_side(x_points, centre, is_upper) {
            None => {
                let angle_start: f64 = if is_upper { 0.0 } else { PI };
                let angles: Array1<f64> = Array1::linspace(angle_start, angle_start + PI, n_half);
                let radius: Array1<f64> = lcfs_radii_along_rays(value_function, centre, &angles, radius_max, r_minimum);
                for i_angle in 0..n_half {
                    contour_r.push(centre.r + angles[i_angle].cos() * radius[i_angle]);
                    contour_z.push(centre.z + angles[i_angle].sin() * radius[i_angle]);
                }
            }
            Some(x_point) => {
                // Lines from the axis' height to the X-point's, clustered towards the X-point
                let n_height: usize = n_half / 2;
                let angle: Array1<f64> = Array1::linspace(0.0, 0.5 * PI, n_height);
                let heights: Array1<f64> = angle.mapv(|value: f64| centre.z + (x_point.z - centre.z) * value.sin());
                let crossings: HeightCrossings = lcfs_crossings_at_heights(value_function, centre, x_point, &heights, r_minimum, r_maximum);
                let mut outboard_r: Array1<f64> = crossings.outboard_r;
                let mut inboard_r: Array1<f64> = crossings.inboard_r;
                // The X-point itself, where the last line touches (1 - psi_N) = 0
                outboard_r[n_height - 1] = x_point.r;
                inboard_r[n_height - 1] = x_point.r;
                if is_upper {
                    // Up the outboard side to the X-point, down the inboard side
                    for i_height in 0..n_height {
                        contour_r.push(outboard_r[i_height]);
                        contour_z.push(heights[i_height]);
                    }
                    for i_height in (0..n_height - 1).rev() {
                        contour_r.push(inboard_r[i_height]);
                        contour_z.push(heights[i_height]);
                    }
                } else {
                    // Down the inboard side to the X-point, up the outboard side
                    for i_height in 0..n_height {
                        contour_r.push(inboard_r[i_height]);
                        contour_z.push(heights[i_height]);
                    }
                    for i_height in (0..n_height - 1).rev() {
                        contour_r.push(outboard_r[i_height]);
                        contour_z.push(heights[i_height]);
                    }
                }
            }
        }
    }
    let n_contour: usize = contour_r.len();
    let mut found_r: Vec<f64> = Vec::with_capacity(n_contour + 1);
    let mut found_z: Vec<f64> = Vec::with_capacity(n_contour + 1);
    for i_contour in 0..n_contour {
        if contour_r[i_contour].is_finite() {
            found_r.push(contour_r[i_contour]);
            found_z.push(contour_z[i_contour]);
        }
    }
    // Closed, unless no crossing was found anywhere (an empty contour)
    if !found_r.is_empty() {
        found_r.push(found_r[0]);
        found_z.push(found_z[0]);
    }
    Points {
        r: Array1::from(found_r),
        z: Array1::from(found_z),
    }
}

/// Even-odd rule: whether each point lies inside a closed polygon.
///
/// # Arguments
/// * `point_r` - points [any length unit]
/// * `point_z` - points [same unit]
/// * `polygon_r` - closed polygon, last vertex = first [same unit]
/// * `polygon_z` - closed polygon [same unit]
pub fn inside_polygon(point_r: &[f64], point_z: &[f64], polygon_r: &Array1<f64>, polygon_z: &Array1<f64>) -> Vec<bool> {
    let n_point: usize = point_r.len();
    let mut inside: Vec<bool> = vec![false; n_point];
    let n_edge: usize = polygon_r.len().saturating_sub(1);
    for i_edge in 0..n_edge {
        let r_1: f64 = polygon_r[i_edge];
        let z_1: f64 = polygon_z[i_edge];
        let r_2: f64 = polygon_r[i_edge + 1];
        let z_2: f64 = polygon_z[i_edge + 1];
        if z_1 == z_2 {
            continue;
        }
        for i_point in 0..n_point {
            let straddles: bool = (z_1 > point_z[i_point]) != (z_2 > point_z[i_point]);
            let crossing_r: f64 = r_1 + (point_z[i_point] - z_1) * (r_2 - r_1) / (z_2 - z_1);
            if straddles && point_r[i_point] < crossing_r {
                inside[i_point] = !inside[i_point];
            }
        }
    }
    inside
}
