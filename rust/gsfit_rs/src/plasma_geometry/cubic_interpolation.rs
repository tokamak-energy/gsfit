use core::f64;
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
/// * `x` - array of coordinate where `f(x) = f_target` (minimum 1 `x` value; maximum 3 `x` values), [metre]
pub fn cubic_interpolation_v2(
    cell0_x: f64,
    cell0_f: f64,
    cell0_d_f_d_x: f64,
    cell1_x: f64,
    cell1_f: f64,
    cell1_d_f_d_x: f64,
    f_target: f64,
) -> Vec<f64> {
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

    // Find the root in [0, 1] (valid interpolation range)
    let mut x_values: Vec<f64> = Vec::new();
    for &t in &roots {
        if (0.0..=1.0).contains(&t) {
            x_values.push(cell0_x + t * delta_x);
        }
    }

    x_values
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

    // Find the root in [0, 1] (valid interpolation range)
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
