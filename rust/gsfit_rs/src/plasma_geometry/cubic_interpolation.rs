use ndarray::Array1;

/// Cubic interpolation (consistent with bicubic interpolation)
///
/// Arguments:
/// * `cell0_x` - x coordinate of first cell, [metre]
/// * `cell0_f` - function value at first cell, [any]
/// * `cell0_d_f_d_x` - derivative at first cell, [any/metre]
/// * `cell1_x` - x coordinate of second cell, [metre]
/// * `cell1_f` - function value at second cell, [any]
/// * `cell1_d_f_d_x` - derivative at second cell, [any/metre]
/// * `f_target` - target function value, [any]
///
/// Returns:
/// * `x` - the coordinates where `f` crosses `f_target`, in increasing `t` (0 to 3 values), [metre]
///
/// Only **crossings**, where `f - f_target` changes sign, are returned: a point where the cubic
/// only touches `f_target` is not one. The winding number in
/// `find_stationary_points_using_winding_number` flips the sign of a field at every crossing, so a
/// touch counted as a crossing would break its count.
///
/// The crossings are found from signs alone, with no tolerance on `f`, so the result does not depend
/// on the units or the size of `f`. This matters on a plane of up/down symmetry, where
/// `d(psi)/d(z)` is round-off (~1e-11) along a whole grid row: solving the cubic with absolute
/// tolerances on its coefficients reported a double root, a crossing which was not there, and lost
/// the magnetic axis. Instead, `[0, 1]` is split at the turning points of the cubic into pieces on
/// which it is monotonic, and each piece whose ends have different signs holds exactly one crossing,
/// found by bisection.
///
/// The signs use the same tie-break as the winding number: `0.0` counts as positive.
pub fn cubic_interpolation_v2(cell0_x: f64, cell0_f: f64, cell0_d_f_d_x: f64, cell1_x: f64, cell1_f: f64, cell1_d_f_d_x: f64, f_target: f64) -> Vec<f64> {
    let delta_x: f64 = cell1_x - cell0_x;

    // Cubic Hermite basis functions, with `t` in [0.0, 1.0]:
    // h00(t) = 2 * t**3 - 3 * t**2 + 1     d(h00)/dt = 6 * t**2 - 6 * t
    // h10(t) = t**3 - 2 * t**2 + t         d(h10)/dt = 3 * t**2 - 4 * t + 1
    // h01(t) = -2 * t**3 + 3 * t**2        d(h01)/dt = -6 * t**2 + 6 * t
    // h11(t) = t**3 - t**2                 d(h11)/dt = 3 * t**2 - 2 * t
    //
    // With the properties that:
    // h00(0) = 1, h00(1) = 0;  h00'(0) = 0, h00'(1) = 0
    // h10(0) = 0, h10(1) = 0;  h10'(0) = 1, h10'(1) = 0
    // h01(0) = 0, h01(1) = 1;  h01'(0) = 0, h01'(1) = 0
    // h11(0) = 0, h11(1) = 0;  h11'(0) = 0, h11'(1) = 1
    //
    // f(t) = cell0_f * h00(t)
    //        + delta_x * cell0_d_f_d_x * h10(t)
    //        + cell1_f * h01(t)
    //        + delta_x * cell1_d_f_d_x * h11(t)

    // Solve: a * t**3 + b * t**2 + c * t + d = f_target
    let a: f64 = 2.0 * cell0_f + delta_x * cell0_d_f_d_x - 2.0 * cell1_f + delta_x * cell1_d_f_d_x;
    let b: f64 = -3.0 * cell0_f - 2.0 * delta_x * cell0_d_f_d_x + 3.0 * cell1_f - delta_x * cell1_d_f_d_x;
    let c: f64 = delta_x * cell0_d_f_d_x;
    let d: f64 = cell0_f;

    // `g(t) = f(t) - f_target`. At the ends the given values are used, rather than the polynomial,
    // so that the signs match the corner values exactly
    let g = |t: f64| -> f64 { ((a * t + b) * t + c) * t + (d - f_target) };
    let g_at = |t: f64| -> f64 {
        if t == 0.0 {
            cell0_f - f_target
        } else if t == 1.0 {
            cell1_f - f_target
        } else {
            g(t)
        }
    };

    // Split [0, 1] at the turning points of `g`, the roots of `g'(t) = 3 a t**2 + 2 b t + c`
    let mut breakpoints: Vec<f64> = vec![0.0];
    for t_turning in quadratic_real_roots(3.0 * a, 2.0 * b, c) {
        if t_turning > 0.0 && t_turning < 1.0 {
            breakpoints.push(t_turning);
        }
    }
    breakpoints.push(1.0);
    breakpoints.sort_by(|t_1: &f64, t_2: &f64| t_1.total_cmp(t_2));

    // `g` is monotonic between neighbouring breakpoints, so a change of sign there is exactly one
    // crossing
    let is_positive = |value: f64| -> bool { value >= 0.0 };
    let mut x_values: Vec<f64> = Vec::new();
    for i_piece in 0..breakpoints.len() - 1 {
        let mut t_low: f64 = breakpoints[i_piece];
        let mut t_high: f64 = breakpoints[i_piece + 1];
        let low_is_positive: bool = is_positive(g_at(t_low));
        if low_is_positive == is_positive(g_at(t_high)) {
            continue;
        }
        // Bisection; 64 halvings take the bracket below the spacing of `f64` on [0, 1]
        for _i_bisection in 0..64 {
            let t_middle: f64 = 0.5 * (t_low + t_high);
            if t_middle <= t_low || t_middle >= t_high {
                break;
            }
            if is_positive(g(t_middle)) == low_is_positive {
                t_low = t_middle;
            } else {
                t_high = t_middle;
            }
        }
        x_values.push(cell0_x + 0.5 * (t_low + t_high) * delta_x);
    }

    x_values
}

/// The real roots of `a x**2 + b x + c = 0`, with no tolerances: `a == 0.0` is linear, and a
/// negative discriminant has none. The roots are computed in the form which avoids cancellation.
fn quadratic_real_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a == 0.0 {
        if b == 0.0 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let discriminant: f64 = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let q: f64 = -0.5 * (b + b.signum() * discriminant.sqrt());
    if q == 0.0 {
        // `b == 0` and `c == 0`: a double root at zero
        return vec![0.0];
    }
    vec![q / a, c / q]
}

/// Cubic interpolation (consistent with bicubic interpolation)
///
/// Arguments:
/// * `cell0_x` - x coordinate of first cell, [metre]
/// * `cell0_f` - function value at first cell, [any]
/// * `cell0_d_f_d_x` - derivative at first cell, [any/metre]
/// * `cell1_x` - x coordinate of second cell, [metre]
/// * `cell1_f` - function value at second cell, [any]
/// * `cell1_d_f_d_x` - derivative at second cell, [any/metre]
/// * `f_target` - target function value, [any]
///
/// Returns:
/// * `x` - array of coordinate where `f(x) = f_target` (minimum 1 `x` value; maximum 3 `x` values), [metre]
pub fn cubic_interpolation(
    cell0_x: f64,
    cell0_f: f64,
    cell0_d_f_d_x: f64,
    cell1_x: f64,
    cell1_f: f64,
    cell1_d_f_d_x: f64,
    f_target: f64,
) -> Result<Array1<f64>, String> {
    let delta_x: f64 = cell1_x - cell0_x;

    // Cubic Hermite basis functions, with `t` in [0.0, 1.0]:
    // h00(t) = 2 * t**3 - 3 * t**2 + 1     d(h00)/dt = 6 * t**2 - 6 * t
    // h10(t) = t**3 - 2 * t**2 + t         d(h10)/dt = 3 * t**2 - 4 * t + 1
    // h01(t) = -2 * t**3 + 3 * t**2        d(h01)/dt = -6 * t**2 + 6 * t
    // h11(t) = t**3 - t**2                 d(h11)/dt = 3 * t**2 - 2 * t
    //
    // With the properties that:
    // h00(0) = 1, h00(1) = 0;  h00'(0) = 0, h00'(1) = 0
    // h10(0) = 0, h10(1) = 0;  h10'(0) = 1, h10'(1) = 0
    // h01(0) = 0, h01(1) = 1;  h01'(0) = 0, h01'(1) = 0
    // h11(0) = 0, h11(1) = 0;  h11'(0) = 0, h11'(1) = 1
    //
    // f(t) = cell0_f * h00(t)
    //        + delta_x * cell0_d_f_d_x * h10(t)
    //        + cell1_f * h01(t)
    //        + delta_x * cell1_d_f_d_x * h11(t)

    // Solve: a * t**3 + b * t**2 + c * t + d = f_target
    let a: f64 = 2.0 * cell0_f + delta_x * cell0_d_f_d_x - 2.0 * cell1_f + delta_x * cell1_d_f_d_x;
    let b: f64 = -3.0 * cell0_f - 2.0 * delta_x * cell0_d_f_d_x + 3.0 * cell1_f - delta_x * cell1_d_f_d_x;
    let c: f64 = delta_x * cell0_d_f_d_x;
    let d: f64 = cell0_f;

    // Rearrange: a*t**3 + b*t**2 + c*t + (d - f_target) = 0
    let roots: Vec<f64> = solve_cubic(a, b, c, d - f_target);

    // Find the root in [0.0, 1.0] (valid interpolation range)
    let mut x_values: Vec<f64> = Vec::new();
    for &t in &roots {
        if (0.0..=1.0).contains(&t) {
            x_values.push(cell0_x + t * delta_x);
        }
    }

    if !x_values.is_empty() {
        return Ok(Array1::from_vec(x_values));
    }

    Err(format!(
        "No solution found in range [{}, {}] for f = {}. Roots: {:?}",
        cell0_x, cell1_x, f_target, roots
    ))
}

/// Solve cubic equation: a*x**3 + b*x**2 + c*x + d = 0
/// Returns all real roots
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    const EPS: f64 = 1e-12;

    // Handle degenerate cases
    if a.abs() < EPS {
        return solve_quadratic(b, c, d);
    }

    // Normalize to monic: x**3 + A x**2 + B x + C = 0
    let a_norm: f64 = b / a;
    let b_norm: f64 = c / a;
    let c_norm: f64 = d / a;

    // Depressed cubic: y**3 + p y + q = 0 with x = y - A/3
    let a2_norm: f64 = a_norm * a_norm;
    let p: f64 = b_norm - a2_norm / 3.0;
    let q: f64 = 2.0 * a_norm * a2_norm / 27.0 - a_norm * b_norm / 3.0 + c_norm;
    let offset: f64 = a_norm / 3.0;

    // Discriminant for depressed cubic: delta = (q/2)**2 + (p/3)**3
    let delta: f64 = (q * 0.5) * (q * 0.5) + (p / 3.0) * (p / 3.0) * (p / 3.0);

    let mut roots: Vec<f64> = Vec::new();

    if delta > EPS {
        // One real root (Cardano)
        let sqrt_delta: f64 = delta.sqrt();
        let u: f64 = (-q * 0.5 + sqrt_delta).cbrt();
        let v: f64 = (-q * 0.5 - sqrt_delta).cbrt();
        let y: f64 = u + v;
        roots.push(y - offset);
    } else if delta.abs() <= EPS {
        // Multiple real roots
        if q.abs() <= EPS {
            // Triple root at y = 0
            roots.push(-offset);
        } else {
            let u: f64 = (-q * 0.5).cbrt();
            roots.push(2.0 * u - offset);
            roots.push(-u - offset);
        }
    } else {
        // Three distinct real roots (trigonometric solution)
        let rho: f64 = (-p / 3.0).sqrt();
        let theta: f64 = ((-q) / (2.0 * rho * rho * rho)).acos();
        for k in 0..3usize {
            let y: f64 = 2.0 * rho * ((theta + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos();
            roots.push(y - offset);
        }
    }

    roots
}

/// Solve quadratic equation: a*x**2 + b*x + c = 0
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    const EPS: f64 = 1e-12;

    if a.abs() < EPS {
        // Linear equation bx + c = 0
        if b.abs() < EPS {
            return Vec::new();
        }
        return vec![-c / b];
    }

    let discriminant: f64 = b * b - 4.0 * a * c;

    if discriminant < -EPS {
        Vec::new()
    } else if discriminant.abs() < EPS {
        vec![-b / (2.0 * a)]
    } else {
        let sqrt_disc: f64 = discriminant.sqrt();
        // Use numerically stable formula
        let q: f64 = -0.5 * (b + b.signum() * sqrt_disc);
        vec![q / a, c / q]
    }
}

pub fn cubic_interpolation_at_x(cell0_x: f64, cell0_f: f64, cell0_d_f_d_x: f64, cell1_x: f64, cell1_f: f64, cell1_d_f_d_x: f64, x: f64) -> f64 {
    let delta_x: f64 = cell1_x - cell0_x;

    // Cubic Hermite basis functions, with `t` in [0.0, 1.0]:
    // h00(t) = 2 * t**3 - 3 * t**2 + 1     d(h00)/dt = 6 * t**2 - 6 * t
    // h10(t) = t**3 - 2 * t**2 + t         d(h10)/dt = 3 * t**2 - 4 * t + 1
    // h01(t) = -2 * t**3 + 3 * t**2        d(h01)/dt = -6 * t**2 + 6 * t
    // h11(t) = t**3 - t**2                 d(h11)/dt = 3 * t**2 - 2 * t
    //
    // With the properties that:
    // h00(0) = 1, h00(1) = 0;  h00'(0) = 0, h00'(1) = 0
    // h10(0) = 0, h10(1) = 0;  h10'(0) = 1, h10'(1) = 0
    // h01(0) = 0, h01(1) = 1;  h01'(0) = 0, h01'(1) = 0
    // h11(0) = 0, h11(1) = 0;  h11'(0) = 0, h11'(1) = 1
    //
    // f(t) = cell0_f * h00(t)
    //        + delta_x * cell0_d_f_d_x * h10(t)
    //        + cell1_f * h01(t)
    //        + delta_x * cell1_d_f_d_x * h11(t)

    // Evaluate: a * t**3 + b * t**2 + c * t + d
    let a: f64 = 2.0 * cell0_f + delta_x * cell0_d_f_d_x - 2.0 * cell1_f + delta_x * cell1_d_f_d_x;
    let b: f64 = -3.0 * cell0_f - 2.0 * delta_x * cell0_d_f_d_x + 3.0 * cell1_f - delta_x * cell1_d_f_d_x;
    let c: f64 = delta_x * cell0_d_f_d_x;
    let d: f64 = cell0_f;
    let t: f64 = (x - cell0_x) / delta_x;

    assert!(
        (0.0..=1.0).contains(&t),
        "x={x} is out of interpolation range: (cell0_x, cell1_x)=({cell0_x}, {cell1_x})"
    );

    let value: f64 = a * t.powi(3) + b * t.powi(2) + c * t + d;

    value
}

#[test]
fn test_cubic_interpolation() {
    // Lazy loading of packages which are not used anywhere else in the code
    use approx::assert_abs_diff_eq;

    // Let's assume this interval:
    let left_x: f64 = 2.3;
    let right_x: f64 = 4.5;

    // Define some values
    let a: f64 = 0.5;
    let b: f64 = 1.23;
    let c: f64 = 2.1;
    let d: f64 = 5.5;

    // Cubic function
    fn f(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
        // let value: f64 = a + b * x + c * x.powi(2) + d * x.powi(3);
        let value: f64 = a * x.powi(3) + b * x.powi(2) + c * x + d;

        value
    }
    // Derivative of cubic function
    fn d_f_d_x(x: f64, a: f64, b: f64, c: f64, _d: f64) -> f64 {
        // let value: f64 = b + 2.0 * c * x + 3.0 * d * x.powi(2);
        let value: f64 = 3.0 * a * x.powi(2) + 2.0 * b * x + c;

        value
    }

    // Calculate the function values
    // f(x) = a * x**3 + b * x**2 + c * x + d
    let left_f: f64 = f(left_x, a, b, c, d);
    let right_f: f64 = f(right_x, a, b, c, d);

    // Calculate the derivatives
    // f'(x) = 3 * a * x**2 + 2 * b * x + c
    let left_df_dx: f64 = d_f_d_x(left_x, a, b, c, d);
    let right_df_dx: f64 = d_f_d_x(right_x, a, b, c, d);

    // Make up test values
    let x_target: f64 = 3.4567;
    let f_target: f64 = f(x_target, a, b, c, d);

    let x_value_or_error: Result<Array1<f64>, String> = cubic_interpolation(left_x, left_f, left_df_dx, right_x, right_f, right_df_dx, f_target);

    println!("x_target: {}, f_target: {}", x_target, f_target);

    let x_value: Array1<f64> = x_value_or_error.unwrap();

    println!("x_value: {}", x_value);

    assert_abs_diff_eq!(x_value[0], x_target, epsilon = 1e-6);
}

/// The edge from example 17 where a crossing was reported which is not there. Along a grid row on a
/// plane of up/down symmetry, `d(psi)/d(z)` is round-off: both ends are -1e-10 (the value the winding
/// number search clamps round-off to) and the slopes are ~1e-11, so it never reaches zero.
#[test]
fn test_cubic_interpolation_v2_round_off_edge_has_no_crossing() {
    let x_values: Vec<f64> = cubic_interpolation_v2(1.090625, -1.0e-10, -1.5196e-11, 1.1146875, -1.0e-10, -1.4721e-11, 0.0);

    assert!(x_values.is_empty(), "x_values = {x_values:?}");
}

/// The crossings do not depend on the size of `f`
#[test]
fn test_cubic_interpolation_v2_is_independent_of_scale() {
    use approx::assert_abs_diff_eq;

    // f(x) = scale * (x - 0.3), on x in [0, 1]
    for scale in [1.0e-15, 1.0, 1.0e15] {
        let x_values: Vec<f64> = cubic_interpolation_v2(0.0, -0.3 * scale, scale, 1.0, 0.7 * scale, scale, 0.0);

        assert_eq!(x_values.len(), 1, "scale = {scale}");
        assert_abs_diff_eq!(x_values[0], 0.3, epsilon = 1.0e-14);
    }
}

/// A cubic which only touches the target is not a crossing; one which dips just below it crosses twice
#[test]
fn test_cubic_interpolation_v2_touch_is_not_a_crossing() {
    use approx::assert_abs_diff_eq;

    // f(x) = (x - 0.5)**2, on x in [0, 1], touches 0 at x = 0.5
    let x_values: Vec<f64> = cubic_interpolation_v2(0.0, 0.25, -1.0, 1.0, 0.25, 1.0, 0.0);
    assert!(x_values.is_empty(), "x_values = {x_values:?}");

    // f(x) = (x - 0.5)**2 - 1e-6 crosses 0 at x = 0.5 -/+ 1e-3
    let x_values: Vec<f64> = cubic_interpolation_v2(0.0, 0.25 - 1.0e-6, -1.0, 1.0, 0.25 - 1.0e-6, 1.0, 0.0);
    assert_eq!(x_values.len(), 2, "x_values = {x_values:?}");
    assert_abs_diff_eq!(x_values[0], 0.499, epsilon = 1.0e-12);
    assert_abs_diff_eq!(x_values[1], 0.501, epsilon = 1.0e-12);
}

/// Three crossings, returned in order along the edge, and relative to a non-zero target
#[test]
fn test_cubic_interpolation_v2_three_crossings() {
    use approx::assert_abs_diff_eq;

    // f(x) = 2 + (t - 0.2) * (t - 0.5) * (t - 0.8), with t = (x - 1) / 2 on x in [1, 3]; crosses 2 at t = 0.2, 0.5, 0.8.
    // f(t=0) = 2 - 0.08, f(t=1) = 2 + 0.08, and d(f)/d(t) = 0.66 at both ends, so d(f)/d(x) = 0.33
    let x_values: Vec<f64> = cubic_interpolation_v2(1.0, 2.0 - 0.08, 0.33, 3.0, 2.0 + 0.08, 0.33, 2.0);

    assert_eq!(x_values.len(), 3, "x_values = {x_values:?}");
    assert_abs_diff_eq!(x_values[0], 1.4, epsilon = 1.0e-12);
    assert_abs_diff_eq!(x_values[1], 2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(x_values[2], 2.6, epsilon = 1.0e-12);
}
