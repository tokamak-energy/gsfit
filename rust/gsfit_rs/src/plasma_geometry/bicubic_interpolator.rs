use core::f64;
use ndarray::{Array1, Array2, array, s};

pub struct BicubicInterpolator {
    pub a_matrix: Array2<f64>,
}

pub struct BicubicValueAndDerivatives {
    pub f: f64,
    pub d_f_d_x: f64,
    pub d_f_d_y: f64,
    pub d2_f_d_x2: f64,
    pub d2_f_d_x_d_y: f64,
    pub d2_f_d_y2: f64,
}

pub struct BicubicStationaryPoint {
    pub x: f64,
    pub y: f64,
    pub f: f64,
    #[allow(dead_code)]
    pub is_max: bool,
    #[allow(dead_code)]
    pub grad_norm: f64,
    #[allow(dead_code)]
    pub iter: usize,
}

pub enum ErrorType {
    HessianIsSingular,
    BacktrackingFailed,
    MaxIterationsReached,
}

impl BicubicInterpolator {
    /// Bicubic interpolation
    /// https://en.wikipedia.org/wiki/Bicubic_interpolation
    ///
    /// # Arguments
    /// * `delta_x` - grid spacing in x direction, between 0 and 1, [dimensionless]
    /// * `delta_y` - grid spacing in y direction, between 0 and 1, [dimensionless]
    /// * `f` - function values at the four corners of the grid, [any]
    /// * `d_f_d_x` - partial derivative of `f` with respect to `x` at the four corners, [any]
    /// * `d_f_d_y` - partial derivative of `f` with respect to `y` at the four corners, [any]
    /// * `d2_f_d_x_d_y` - second partial derivative of `f` with respect to `x` and `y` at the four corners, [any]
    ///
    /// # Returns
    /// * `BicubicInterpolator` - the bicubic interpolator object
    ///
    /// # Algorithm
    /// `f`, `d_f_d_x`, `d_f_d_y`, and `d2_f_d_x_d_y` should be indexed like:
    /// * `f[(0, 0)] = f[(i_x_left, i_y_lower)]`;
    /// * `f[(0, 1)] = f[(i_x_left, i_y_upper)]`;
    /// * `f[(1, 0)] = f[(i_x_right, i_y_lower)]`;
    /// * `f[(1, 1)] = f[(i_x_right, i_y_upper)]`;
    ///
    /// The bicubic fit is:
    /// `P(x, y) = [1, x, x^2, x^3] * a * [1, y, y^2, y^3].T`
    /// where:
    /// ```text
    /// M = [
    ///   f(x=0, y=0)          f(x=0, y=1)          d(f(x=0, y=0))/d(y)          d(f(x=0, y=1))/d(y)
    ///   f(x=1, y=0)          f(x=1, y=1)          d(f(x=1, y=0))/d(y)          d(f(x=1, y=1))/d(y)
    ///   d(f(x=0, y=0))/d(x)  d(f(x=0, y=1))/d(x)  d2(f(x=0, y=0))/(d(x)*d(y))  d2(f(x=0, y=1))/(d(x)*d(y))
    ///   d(f(x=1, y=0))/d(x)  d(f(x=1, y=1))/d(x)  d2(f(x=1, y=0))/(d(x)*d(y))  d2(f(x=1, y=1))/(d(x)*d(y))
    /// ]
    /// ```
    ///
    /// # Examples
    /// ```rust
    /// use gsfit_rs::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
    /// use ndarray::{Array2};
    /// ```
    pub fn new(delta_x: f64, delta_y: f64, f: &Array2<f64>, d_f_d_x: &Array2<f64>, d_f_d_y: &Array2<f64>, d2_f_d_x_d_y: &Array2<f64>) -> Self {
        #[rustfmt::skip]
        let coeff_matrix_1: Array2<f64> = array![
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [-3.0,  3.0, -2.0, -1.0],
            [ 2.0, -2.0,  1.0,  1.0],
        ];

        #[rustfmt::skip]
        let coeff_matrix_2: Array2<f64> = array![
            [ 1.0,  0.0, -3.0,  2.0],
            [ 0.0,  0.0,  3.0, -2.0],
            [ 0.0,  1.0, -2.0,  1.0],
            [ 0.0,  0.0, -1.0,  1.0],
        ];

        let mut function_matrix: Array2<f64> = Array2::from_elem((4, 4), f64::NAN);
        let d_f_d_x_normalised: Array2<f64> = d_f_d_x.to_owned() * delta_x;
        let d_f_d_y_normalised: Array2<f64> = d_f_d_y.to_owned() * delta_y;
        let d2_f_d_x_d_y_normalised: Array2<f64> = d2_f_d_x_d_y.to_owned() * delta_x * delta_y;
        function_matrix.slice_mut(s![0..2, 0..2]).assign(f);
        function_matrix.slice_mut(s![2..4, 0..2]).assign(&d_f_d_x_normalised);
        function_matrix.slice_mut(s![0..2, 2..4]).assign(&d_f_d_y_normalised);
        function_matrix.slice_mut(s![2..4, 2..4]).assign(&d2_f_d_x_d_y_normalised);

        let a_matrix: Array2<f64> = coeff_matrix_1.dot(&function_matrix).dot(&coeff_matrix_2);

        BicubicInterpolator { a_matrix }
    }

    /// Interpolate the value at (x, y)
    ///
    /// # Arguments
    /// * `x` - x-coordinate, normalised to (0.0, 1.0), [dimensionless]
    /// * `y` - y-coordinate, normalised to (0.0, 1.0), [dimensionless]
    ///
    /// # Returns
    /// * `f` - interpolated value at (x, y), [any]
    ///
    #[allow(dead_code)]
    pub fn interpolate(&self, x: f64, y: f64) -> f64 {
        let x_vec: Array1<f64> = Array1::from_vec(vec![1.0, x, x.powi(2), x.powi(3)]);
        let y_vec: Array1<f64> = Array1::from_vec(vec![1.0, y, y.powi(2), y.powi(3)]);
        let f: f64 = x_vec.dot(&self.a_matrix).dot(&y_vec);

        f
    }

    /// Value, first derivatives, and second derivatives
    ///
    /// # Arguments
    /// * `x` - x-coordinate, normalised to (0.0, 1.0), [dimensionless]
    /// * `y` - y-coordinate, normalised to (0.0, 1.0), [dimensionless]
    ///
    /// # Returns
    /// * `BicubicValueAndDerivatives` - struct containing value and derivatives
    ///
    pub fn value_and_derivatives(&self, x: f64, y: f64) -> BicubicValueAndDerivatives {
        // Extract from self
        let a_matrix: &Array2<f64> = &self.a_matrix;

        // Calculated values and derivatives
        let v: Array1<f64> = Array1::from(vec![1.0, y, y * y, y * y * y]);
        let d_v_d_y: Array1<f64> = Array1::from(vec![0.0, 1.0, 2.0 * y, 3.0 * y * y]);
        let d2_v_d_y2: Array1<f64> = Array1::from(vec![0.0, 0.0, 2.0, 6.0 * y]);

        let u: Array1<f64> = Array1::from(vec![1.0, x, x * x, x * x * x]);
        let d_u_d_x: Array1<f64> = Array1::from(vec![0.0, 1.0, 2.0 * x, 3.0 * x * x]);
        let d2_u_d_x2: Array1<f64> = Array1::from(vec![0.0, 0.0, 2.0, 6.0 * x]);

        // Calculate some intermediate values
        let a_v: Array1<f64> = a_matrix.dot(&v);
        let a_d_v_d_y: Array1<f64> = a_matrix.dot(&d_v_d_y);
        let a_d2_v_d_y2: Array1<f64> = a_matrix.dot(&d2_v_d_y2);

        // Value
        let f: f64 = u.dot(&a_v);
        // First derivatives
        let d_f_d_x: f64 = d_u_d_x.dot(&a_v);
        let d_f_d_y: f64 = u.dot(&a_d_v_d_y);
        // Second derivatives
        let d2_f_d_x2: f64 = d2_u_d_x2.dot(&a_v);
        let d2_f_d_x_d_y: f64 = d_u_d_x.dot(&a_d_v_d_y);
        let d2_f_d_y2: f64 = u.dot(&a_d2_v_d_y2);

        // Return results
        BicubicValueAndDerivatives {
            f,
            d_f_d_x,
            d_f_d_y,
            d2_f_d_x2,
            d2_f_d_x_d_y,
            d2_f_d_y2,
        }
    }

    /// Finds the stationary point of the bicubic fit
    ///
    /// # Arguments
    /// * `x_start` - initial x-coordinate guess, note x is normalised to be in (0.0, 1.0), [dimensionless]
    /// * `y_start` - initial y-coordinate guess, note y is normalised to be in (0.0, 1.0), [dimensionless]
    /// * `tol` - tolerance for convergence, [dimensionless]
    /// * `max_iter` - maximum number of iterations
    ///
    /// # Returns
    /// * `Ok(BicubicStationaryPoint)` if a stationary point is found
    /// * `Err(String)` if no solution found
    ///
    /// # Algorithm
    /// A "stationary point" can be:
    /// * Turning point:
    ///   * Maxima
    ///   * Minima
    /// * Saddle point
    /// * Inflection point
    pub fn find_stationary_point(&self, tol: f64, max_iter: usize) -> Result<BicubicStationaryPoint, String> {
        // Linear interpolation using the corner values, to give an initial guess for the stationary point
        // Approximate `d_f_d_x` and `d_f_d_y` as linear functions of `(x, y)` using corner values:
        //   d_f_d_x(x, y) ≈ g_x_00 + (g_x_10 - g_x_00) * x + (g_x_01 - g_x_00) * y = 0
        //   d_f_d_y(x, y) ≈ g_y_00 + (g_y_10 - g_y_00) * x + (g_y_01 - g_y_00) * y = 0
        // Solve this 2x2 linear system via Cramer's rule
        let corner_00: BicubicValueAndDerivatives = self.value_and_derivatives(0.0, 0.0); // (left, bottom)
        let corner_10: BicubicValueAndDerivatives = self.value_and_derivatives(1.0, 0.0); // (right, bottom)
        let corner_01: BicubicValueAndDerivatives = self.value_and_derivatives(0.0, 1.0); // (left, top)

        let a_1: f64 = corner_10.d_f_d_x - corner_00.d_f_d_x;
        let a_2: f64 = corner_01.d_f_d_x - corner_00.d_f_d_x;
        let b_1: f64 = corner_10.d_f_d_y - corner_00.d_f_d_y;
        let b_2: f64 = corner_01.d_f_d_y - corner_00.d_f_d_y;
        let det_linear: f64 = a_1 * b_2 - a_2 * b_1;

        // Give (x, y) a fallback value at the centre of the cell, in case the linear solver fails
        // TODO: do we need the fallback value? If the linear solver fails perhaps there naturally isn't a stationary point and we should just exit?
        let mut x: f64 = 0.5;
        let mut y: f64 = 0.5;
        if det_linear.abs() > f64::EPSILON {
            let x_linear: f64 = (-corner_00.d_f_d_x * b_2 + corner_00.d_f_d_y * a_2) / det_linear;
            let y_linear: f64 = (-corner_00.d_f_d_y * a_1 + corner_00.d_f_d_x * b_1) / det_linear;
            if (0.0..=1.0).contains(&x_linear) && (0.0..=1.0).contains(&y_linear) {
                x = x_linear;
                y = y_linear;

                // println!("x_linear = {x_linear}, y_linear = {y_linear}");
            }
        }

        let bicubic_value_and_derivatives: BicubicValueAndDerivatives = self.value_and_derivatives(x, y);
        let mut f: f64 = bicubic_value_and_derivatives.f;
        let mut g_x: f64 = bicubic_value_and_derivatives.d_f_d_x;
        let mut g_y: f64 = bicubic_value_and_derivatives.d_f_d_y;
        let mut h_x_x: f64 = bicubic_value_and_derivatives.d2_f_d_x2;
        let mut h_x_y: f64 = bicubic_value_and_derivatives.d2_f_d_x_d_y;
        let mut h_y_y: f64 = bicubic_value_and_derivatives.d2_f_d_y2;

        // "gradient norm" = L2 normalisation
        let mut g_norm: f64 = (g_x * g_x + g_y * g_y).sqrt();

        // Choosing the convergence criteria is actually quite tricky:
        // * Testing against the absolute gradient norm `g_norm <= tol`, is bad because the functions gradients can naturally be small
        // * Testing against the relative gradient norm `g_norm / g_norm_initial <= tol`, is bad because if we have a "good" initial guess this can make it harder to converge
        // * Normalising against the function value `g_norm / f.abs() <= tol`, is bad because the function value can be small/zero at the stationary point

        for iter in 0..max_iter {
            if g_norm <= tol {
                let hessian_det: f64 = h_x_x * h_y_y - h_x_y * h_x_y;
                let hessian_trace: f64 = h_x_x + h_y_y;
                let is_max: bool = hessian_det > 0.0 && hessian_trace < 0.0;
                return Ok(BicubicStationaryPoint {
                    x,
                    y,
                    f,
                    is_max,
                    grad_norm: g_norm,
                    iter,
                });
            }

            // Solve Hessian * [delta_x delta_y]^T = -grad
            let hessian_det: f64 = h_x_x * h_y_y - h_x_y * h_x_y;
            if hessian_det.abs() < f64::EPSILON {
                return Err("Hessian is singular".to_string());
            }
            let delta_x: f64 = (-g_x * h_y_y + g_y * h_x_y) / hessian_det;
            let delta_y: f64 = (g_x * h_x_y - g_y * h_x_x) / hessian_det;

            // Compute maximum step size that stays within [0, 1] bounds
            let mut alpha: f64 = 1.0;
            if delta_x > 0.0 {
                alpha = alpha.min((1.0 - x) / delta_x);
            } else if delta_x < 0.0 {
                alpha = alpha.min(-x / delta_x);
            }
            if delta_y > 0.0 {
                alpha = alpha.min((1.0 - y) / delta_y);
            } else if delta_y < 0.0 {
                alpha = alpha.min(-y / delta_y);
            }

            // // If the step size is effectively zero after boundary clamping,
            // // the stationary point is at or beyond the cell boundary.
            // // Accept the current position as the best solution within this cell.
            // if alpha < 1e-14 {
            //     let hessian_det: f64 = h_x_x * h_y_y - h_x_y * h_x_y;
            //     let hessian_trace: f64 = h_x_x + h_y_y;
            //     let is_max: bool = hessian_det > 0.0 && hessian_trace < 0.0;
            //     return Ok(BicubicStationaryPoint {
            //         x,
            //         y,
            //         f,
            //         is_max,
            //         grad_norm: g_norm,
            //         iter,
            //     });
            // }

            // Backtracking to stay inside and reduce ||grad||
            let mut accepted: bool = false;
            for _adjust_step_size in 0..30usize {
                let new_x: f64 = x + alpha * delta_x;
                let new_y: f64 = y + alpha * delta_y;

                let bicubic_value_and_derivatives: BicubicValueAndDerivatives = self.value_and_derivatives(new_x, new_y);
                let new_f: f64 = bicubic_value_and_derivatives.f;
                let new_g_x: f64 = bicubic_value_and_derivatives.d_f_d_x;
                let new_g_y: f64 = bicubic_value_and_derivatives.d_f_d_y;
                let new_h_x_x: f64 = bicubic_value_and_derivatives.d2_f_d_x2;
                let new_h_x_y: f64 = bicubic_value_and_derivatives.d2_f_d_x_d_y;
                let new_h_y_y: f64 = bicubic_value_and_derivatives.d2_f_d_y2;

                let new_g_norm: f64 = (new_g_x * new_g_x + new_g_y * new_g_y).sqrt();
                if new_g_norm < g_norm {
                    x = new_x;
                    y = new_y;
                    f = new_f;
                    g_x = new_g_x;
                    g_y = new_g_y;
                    h_x_x = new_h_x_x;
                    h_x_y = new_h_x_y;
                    h_y_y = new_h_y_y;
                    g_norm = new_g_norm;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Err("Backtracking failed to find acceptable step".to_string());
            }
        }
        Err("Maximum iterations reached without convergence".to_string())
    }
}

#[test]
fn test_bicubic_interpolation() {
    // Lazy loading for crates which are only used within the tests
    use approx::assert_abs_diff_eq;

    // Setup an analytic polynomial function
    // The bicubic interpolation will be exact for polynomials up to cubic
    // This tests a peaked quadratic in both directions
    fn calculate_f(x: f64, y: f64) -> f64 {
        let result: f64 = -(x - 0.5).powi(2) - (y - 0.5).powi(2);

        result
    }
    fn calculate_d_f_d_x(x: f64, _y: f64) -> f64 {
        let result: f64 = 1.0 - 2.0 * x;

        result
    }
    fn calculate_d_f_d_y(_x: f64, y: f64) -> f64 {
        let result: f64 = 1.0 - 2.0 * y;

        result
    }
    fn calculate_d2_f_d_x_d_y(_x: f64, _y: f64) -> f64 {
        let result: f64 = 0.0;

        result
    }

    // Empty arrays to store the function values and derivatives at the four corners of the grid
    let mut f: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_x: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d2_f_d_x_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

    // Calculate values for a test function at the four corners of the grid
    let n_x: usize = 2;
    let n_y: usize = 2;
    let x_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    let y_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    for i_x in 0..n_x {
        for i_y in 0..n_y {
            f[(i_x, i_y)] = calculate_f(x_grid[i_x], y_grid[i_y]);
            d_f_d_x[(i_x, i_y)] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y]);
            d_f_d_y[(i_x, i_y)] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y]);
            d2_f_d_x_d_y[(i_x, i_y)] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
        }
    }

    // Create a grid to interpolate onto, and to calculate the analytic values
    let n_x_target: usize = 6;
    let n_y_target: usize = 5;
    let x_targets: Array1<f64> = Array1::linspace(0.01, 0.99, n_x_target);
    let y_targets: Array1<f64> = Array1::linspace(0.01, 0.99, n_y_target);
    let mut f_analytic: Array2<f64> = Array2::from_elem([n_x_target, n_y_target], f64::NAN);
    let mut f_interpolated: Array2<f64> = Array2::from_elem([n_x_target, n_y_target], f64::NAN);
    let delta_x: f64 = x_grid[1] - x_grid[0];
    let delta_y: f64 = y_grid[1] - y_grid[0];
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(delta_x, delta_y, &f, &d_f_d_x, &d_f_d_y, &d2_f_d_x_d_y);
    for i_x_target in 0..n_x_target {
        for i_y_target in 0..n_y_target {
            f_analytic[(i_x_target, i_y_target)] = calculate_f(x_targets[i_x_target], y_targets[i_y_target]);
            let f_interp = bicubic_interpolator.interpolate(x_targets[i_x_target], y_targets[i_y_target]);
            f_interpolated[(i_x_target, i_y_target)] = f_interp;
        }
    }

    assert_abs_diff_eq!(&f_analytic, &f_interpolated);
}

#[test]
fn test_bicubic_find_stationary_point_near_boundary() {
    // Lazy loading for crates which are only used within the tests
    use approx::assert_abs_diff_eq;

    // Quadratic peaked at (0.9999, 0.9999), near the cell boundary
    let x_peak: f64 = 0.9999;
    let y_peak: f64 = 0.9999;

    let calculate_f = |x: f64, y: f64| -> f64 { -(x - x_peak).powi(2) - (y - y_peak).powi(2) };
    let calculate_d_f_d_x = |x: f64, _y: f64| -> f64 { -2.0 * (x - x_peak) };
    let calculate_d_f_d_y = |_x: f64, y: f64| -> f64 { -2.0 * (y - y_peak) };
    let calculate_d2_f_d_x_d_y = |_x: f64, _y: f64| -> f64 { 0.0 };

    let mut f: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_x: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d2_f_d_x_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

    let n_x: usize = 2;
    let n_y: usize = 2;
    let x_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    let y_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    for i_x in 0..n_x {
        for i_y in 0..n_y {
            f[(i_x, i_y)] = calculate_f(x_grid[i_x], y_grid[i_y]);
            d_f_d_x[(i_x, i_y)] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y]);
            d_f_d_y[(i_x, i_y)] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y]);
            d2_f_d_x_d_y[(i_x, i_y)] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
        }
    }

    let delta_x: f64 = x_grid[1] - x_grid[0];
    let delta_y: f64 = y_grid[1] - y_grid[0];
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(delta_x, delta_y, &f, &d_f_d_x, &d_f_d_y, &d2_f_d_x_d_y);

    let result: BicubicStationaryPoint = bicubic_interpolator
        .find_stationary_point(1e-12, 100)
        .expect("Should find stationary point near cell boundary");

    assert_abs_diff_eq!(result.x, x_peak, epsilon = 1e-10);
    assert_abs_diff_eq!(result.y, y_peak, epsilon = 1e-10);
    assert!(result.is_max);
}

#[test]
fn test_bicubic_find_stationary_point_on_boundary() {
    // Lazy loading for crates which are only used within the tests
    use approx::assert_abs_diff_eq;

    // Quadratic peaked at (1.0, 0.78), exactly on the cell boundary
    let x_peak: f64 = 1.0;
    let y_peak: f64 = 0.78;

    let calculate_f = |x: f64, y: f64| -> f64 { -(x - x_peak).powi(2) - (y - y_peak).powi(2) };
    let calculate_d_f_d_x = |x: f64, _y: f64| -> f64 { -2.0 * (x - x_peak) };
    let calculate_d_f_d_y = |_x: f64, y: f64| -> f64 { -2.0 * (y - y_peak) };
    let calculate_d2_f_d_x_d_y = |_x: f64, _y: f64| -> f64 { 0.0 };

    let mut f: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_x: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d2_f_d_x_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

    let n_x: usize = 2;
    let n_y: usize = 2;
    let x_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    let y_grid: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
    for i_x in 0..n_x {
        for i_y in 0..n_y {
            f[(i_x, i_y)] = calculate_f(x_grid[i_x], y_grid[i_y]);
            d_f_d_x[(i_x, i_y)] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y]);
            d_f_d_y[(i_x, i_y)] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y]);
            d2_f_d_x_d_y[(i_x, i_y)] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
        }
    }

    let delta_x: f64 = x_grid[1] - x_grid[0];
    let delta_y: f64 = y_grid[1] - y_grid[0];
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(delta_x, delta_y, &f, &d_f_d_x, &d_f_d_y, &d2_f_d_x_d_y);

    let result: BicubicStationaryPoint = bicubic_interpolator
        .find_stationary_point(1e-12, 100)
        .expect("Should find stationary point on cell boundary");

    assert_abs_diff_eq!(result.x, x_peak, epsilon = 1e-10);
    assert_abs_diff_eq!(result.y, y_peak, epsilon = 1e-10);
    assert!(result.is_max);
}
