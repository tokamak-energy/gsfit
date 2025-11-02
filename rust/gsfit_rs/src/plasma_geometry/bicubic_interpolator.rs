use core::f64;
use ndarray::{Array1, Array2, s};

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

impl BicubicInterpolator {
    /// Bicubic interpolation
    /// https://en.wikipedia.org/wiki/Bicubic_interpolation
    ///
    /// # Arguments
    /// * `f` - function values at the four corners of the grid
    /// * `d_f_d_x` - partial derivative of `f` with respect to `x` at the four corners
    /// * `d_f_d_y` - partial derivative of `f` with respect to `y` at the four corners
    /// * `d2_f_d_x_d_y` - second partial derivative of `f` with respect to `x` and `y` at the four corners
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
    /// use gsfit_rs::bicubic_interpolator::BicubicInterpolator;
    /// use ndarray::{Array2};
    /// ```
    pub fn new(delta_x: f64, delta_y: f64, f: &Array2<f64>, d_f_d_x: &Array2<f64>, d_f_d_y: &Array2<f64>, d2_f_d_x_d_y: &Array2<f64>) -> Self {
        let coeff_matrix_1: Array2<f64> =
            Array2::from_shape_vec((4, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 3.0, -2.0, -1.0, 2.0, -2.0, 1.0, 1.0])
                .expect("Failed to create coeff_matrix_1");

        let coeff_matrix_2: Array2<f64> =
            Array2::from_shape_vec((4, 4), vec![1.0, 0.0, -3.0, 2.0, 0.0, 0.0, 3.0, -2.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, -1.0, 1.0])
                .expect("Failed to create coeff_matrix_2");

        let mut function_matrix: Array2<f64> = Array2::zeros((4, 4));
        let d_f_d_x_normalised: Array2<f64> = d_f_d_x.to_owned() * delta_x;
        let d_f_d_y_normalised: Array2<f64> = d_f_d_y.to_owned() * delta_y;
        let d2_f_d_x_d_y_normalised: Array2<f64> = d2_f_d_x_d_y.to_owned() * delta_x * delta_y;
        function_matrix.slice_mut(s![0..2, 0..2]).assign(&f);
        function_matrix.slice_mut(s![2..4, 0..2]).assign(&d_f_d_x_normalised);
        function_matrix.slice_mut(s![0..2, 2..4]).assign(&d_f_d_y_normalised);
        function_matrix.slice_mut(s![2..4, 2..4]).assign(&d2_f_d_x_d_y_normalised);

        let a_matrix: Array2<f64> = coeff_matrix_1.dot(&function_matrix).dot(&coeff_matrix_2);

        return BicubicInterpolator { a_matrix };
    }

    /// Interpolate the value at (x, y)
    ///
    /// # Arguments
    /// * `x` - x-coordinate, normalised to (0.0, 1.0), (dimensionless)
    /// * `y` - y-coordinate, normalised to (0.0, 1.0), (dimensionless)
    ///
    /// # Returns
    /// * `f` - interpolated value at (x, y)
    ///
    #[allow(dead_code)]
    pub fn interpolate(&self, x: f64, y: f64) -> f64 {
        let x_vec: Array1<f64> = Array1::from_vec(vec![1.0, x, x.powi(2), x.powi(3)]);
        let y_vec: Array1<f64> = Array1::from_vec(vec![1.0, y, y.powi(2), y.powi(3)]);
        let f: f64 = x_vec.dot(&self.a_matrix).dot(&y_vec);
        return f;
    }

    /// Value, first derivatives, and second derivatives
    ///
    /// # Arguments
    /// * `x` - x-coordinate, normalised to (0.0, 1.0), (dimensionless)
    /// * `y` - y-coordinate, normalised to (0.0, 1.0), (dimensionless)
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
        return BicubicValueAndDerivatives {
            f,
            d_f_d_x,
            d_f_d_y,
            d2_f_d_x2,
            d2_f_d_x_d_y,
            d2_f_d_y2,
        };
    }

    /// Finds the stationary point of the bicubic fit
    ///
    /// # Arguments
    /// * `x_start` - initial x-coordinate guess, note x is normalised to be in (0.0, 1.0), (dimensionless)
    /// * `y_start` - initial y-coordinate guess, note y is normalised to be in (0.0, 1.0), (dimensionless)
    /// * `tol` - tolerance for convergence, (dimensionless)
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
    pub fn find_stationary_point(&self, x_start: f64, y_start: f64, tol: f64, max_iter: usize) -> Result<BicubicStationaryPoint, String> {
        let mut x: f64 = x_start;
        let mut y: f64 = y_start;

        let in_bounds = |value: f64| value >= 0.0 && value <= 1.0;
        if !in_bounds(x) || !in_bounds(y) {
            x = 0.5;
            y = 0.5;
        }

        let bicubic_value_and_derivatives: BicubicValueAndDerivatives = self.value_and_derivatives(x, y);
        let mut f: f64 = bicubic_value_and_derivatives.f;
        let mut gx: f64 = bicubic_value_and_derivatives.d_f_d_x;
        let mut gy: f64 = bicubic_value_and_derivatives.d_f_d_y;
        let mut hxx: f64 = bicubic_value_and_derivatives.d2_f_d_x2;
        let mut hxy: f64 = bicubic_value_and_derivatives.d2_f_d_x_d_y;
        let mut hyy: f64 = bicubic_value_and_derivatives.d2_f_d_y2;

        let mut gnorm: f64 = (gx * gx + gy * gy).sqrt();

        for iter in 0..max_iter {
            if gnorm <= tol {
                let det: f64 = hxx * hyy - hxy * hxy;
                let tr: f64 = hxx + hyy;
                let is_max: bool = det > 0.0 && tr < 0.0;
                return Ok(BicubicStationaryPoint {
                    x,
                    y,
                    f,
                    is_max,
                    grad_norm: gnorm,
                    iter,
                });
            }

            // Solve Hessian * [dx dy]^T = -grad
            let det: f64 = hxx * hyy - hxy * hxy;
            if det.abs() < 1e-14 {
                return Err("Hessian is singular".to_string());
            }
            let dx: f64 = (-gx * hyy + gy * hxy) / det;
            let dy: f64 = (gx * hxy - gy * hxx) / det;

            // Backtracking to stay inside and reduce ||grad||
            let mut alpha: f64 = 1.0;
            let mut accepted: bool = false;
            for _adjust_step_size in 0..20usize {
                let xn: f64 = x + alpha * dx;
                let yn: f64 = y + alpha * dy;
                if !in_bounds(xn) || !in_bounds(yn) {
                    alpha *= 0.5;
                    continue;
                }

                let bicubic_value_and_derivatives: BicubicValueAndDerivatives = self.value_and_derivatives(xn, yn);
                let f_n: f64 = bicubic_value_and_derivatives.f;
                let gxn: f64 = bicubic_value_and_derivatives.d_f_d_x;
                let gyn: f64 = bicubic_value_and_derivatives.d_f_d_y;
                let hxxn: f64 = bicubic_value_and_derivatives.d2_f_d_x2;
                let hxyn: f64 = bicubic_value_and_derivatives.d2_f_d_x_d_y;
                let hyyn: f64 = bicubic_value_and_derivatives.d2_f_d_y2;

                let gnn: f64 = (gxn * gxn + gyn * gyn).sqrt();
                if gnn < 0.5 * gnorm {
                    x = xn;
                    y = yn;
                    f = f_n;
                    gx = gxn;
                    gy = gyn;
                    hxx = hxxn;
                    hxy = hxyn;
                    hyy = hyyn;
                    gnorm = gnn;
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
        return result;
    }
    fn calculate_d_f_d_x(x: f64, _y: f64) -> f64 {
        let result: f64 = 1.0 - 2.0 * x;
        return result;
    }
    fn calculate_d_f_d_y(_x: f64, y: f64) -> f64 {
        let result: f64 = 1.0 - 2.0 * y;
        return result;
    }
    fn calculate_d2_f_d_x_d_y(_x: f64, _y: f64) -> f64 {
        let result: f64 = 0.0;
        return result;
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
