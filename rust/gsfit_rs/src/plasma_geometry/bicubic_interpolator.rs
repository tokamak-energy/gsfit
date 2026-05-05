use ndarray::{Array1, Array2, ArrayView2, array, s};

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
    /// * `d_x` - grid spacing in x direction, [metre]
    /// * `d_y` - grid spacing in y direction, [metre]
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
    pub fn new(d_x: f64, d_y: f64, f: ArrayView2<f64>, d_f_d_x: ArrayView2<f64>, d_f_d_y: ArrayView2<f64>, d2_f_d_x_d_y: ArrayView2<f64>) -> Self {
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
        let d_f_d_x_normalised: Array2<f64> = d_f_d_x.to_owned() * d_x;
        let d_f_d_y_normalised: Array2<f64> = d_f_d_y.to_owned() * d_y;
        let d2_f_d_x_d_y_normalised: Array2<f64> = d2_f_d_x_d_y.to_owned() * d_x * d_y;
        function_matrix.slice_mut(s![0..2, 0..2]).assign(&f);
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
    #[allow(dead_code)]
    pub fn interpolate(&self, x: f64, y: f64) -> f64 {
        let x_vec: Array1<f64> = array![1.0, x, x.powi(2), x.powi(3)];
        let y_vec: Array1<f64> = array![1.0, y, y.powi(2), y.powi(3)];
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
    /// The bicubic fit is a 3rd degree polynomial in both `x` and `y`:
    /// f = a_33 * x³ y³ + a_32 * x³ y² + ... + a_00
    /// The stationary point is where the gradient is zero in both directions:
    /// ∂f/∂x = 3 a_33 x² y³ + 3 a_32 x² y² + ... + a_10  =  0      (degree: x=2, y=3)
    /// ∂f/∂y = 3 a_33 x³ y² + 2 a_23 x² y + ... + a_01   =  0      (degree: x=3, y=2)
    /// This system of equations does not have an exact solution, and must be solved iteratively.
    ///
    /// # Arguments
    /// * `tol` - convergence tolerance, expressed as a fraction of cell width (since `(x, y)` are normalised to `[0, 1]`)
    /// * `max_iter` - maximum number of Newton iterations
    ///
    /// # Returns
    /// * `Ok(BicubicStationaryPoint)` if a stationary point is found
    /// * `Err(String)` if no solution found
    ///
    /// # Algorithm
    /// A "stationary point" can be:
    /// * Extreme point:
    ///   * Maxima
    ///   * Minima
    /// * Saddle point
    /// * Inflection point
    /// * Higher order stationary points? TODO: improve documentation here
    ///
    /// The initial guess is computed internally by linearising `(d_f_d_x, d_f_d_y)`
    /// from the corner gradients and solving the resulting 2x2 system; if that
    /// solve falls outside the cell or is degenerate, we fall back to the cell
    /// centre `(0.5, 0.5)`.
    ///
    /// # Convergence criterion
    /// Choosing the convergence criteria is actually quite tricky:
    /// * Testing against the absolute gradient norm `g_norm <= tol`, is bad because the function could have naturally small gradients
    /// * Testing against the relative gradient norm `g_norm / g_norm_initial <= tol`, is bad because if we have a "good" initial guess this can make it harder to converge
    /// * Normalising against the function value `g_norm / f.abs() <= tol`, is bad because the function value can be small/zero at the stationary point
    /// The insight which we use is that the grid size - which is between 0.0 and 1.0 in the normalised coordinates.
    /// * Our chosen convergence criteria is to test the Newton step size `delta`: **`abs(delta) <= tol`**
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

        // Threshold for treating an iterate as sitting exactly on a cell boundary.
        // Alpha clamping snaps to a boundary in floating point, so this only needs
        // to absorb roundoff.
        let boundary_tol: f64 = 1e-12;

        for iter in 0..max_iter {
            // Hessian determinant for the 2x2 Newton solve.
            let hessian_det: f64 = h_x_x * h_y_y - h_x_y * h_x_y;

            // Check if the Hessian is singular or ill-conditioned
            let hessian_scale: f64 = h_x_x.abs().max(h_x_y.abs()).max(h_y_y.abs());
            // Whichever is larger: 1e-12 or 16 * machine epsilon * (largest Hessian element)^2
            let hessian_det_tol: f64 = 1.0e-12_f64.max(16.0 * f64::EPSILON * hessian_scale * hessian_scale);
            if hessian_det.abs() <= hessian_det_tol {
                return Err("Hessian is singular or ill-conditioned".to_string());
            }

            // Unconstrained Newton step: predicted displacement to the stationary
            // point under the local quadratic model. Solves H * delta = -g.
            let mut delta_x: f64 = (-g_x * h_y_y + g_y * h_x_y) / hessian_det;
            let mut delta_y: f64 = (g_x * h_x_y - g_y * h_x_x) / hessian_det;

            // If the iterate sits on a cell boundary and the Newton step would push
            // outward, the bicubic's unconstrained stationary point lies outside this
            // cell. Switch to a 1D Newton step along the boundary; only the tangential
            // component then enters the convergence test.
            let on_x_low: bool = x <= boundary_tol;
            let on_x_high: bool = x >= 1.0 - boundary_tol;
            let on_y_low: bool = y <= boundary_tol;
            let on_y_high: bool = y >= 1.0 - boundary_tol;
            let pinned_x: bool = (on_x_low && delta_x < 0.0) || (on_x_high && delta_x > 0.0);
            let pinned_y: bool = (on_y_low && delta_y < 0.0) || (on_y_high && delta_y > 0.0);

            if pinned_x && pinned_y {
                // Pinned at a corner: no direction inside the cell is descent for ||grad||.
                // Accept the corner as the best constrained stationary point we can offer.
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
            } else if pinned_x {
                // Constrained 1D problem along x = boundary; only g_y matters.
                if h_y_y.abs() <= f64::EPSILON {
                    return Err("1D Hessian along x boundary is singular".to_string());
                }
                delta_x = 0.0;
                delta_y = -g_y / h_y_y;
            } else if pinned_y {
                if h_x_x.abs() <= f64::EPSILON {
                    return Err("1D Hessian along y boundary is singular".to_string());
                }
                delta_y = 0.0;
                delta_x = -g_x / h_x_x;
            }

            // Convergence test: Newton-step size in cell-normalised coordinates.
            // ||delta|| is the linear-extrapolation distance from the current iterate
            // to the stationary point under the local quadratic model. When pinned
            // to a boundary, only the constrained component is used.
            let step_norm: f64 = if pinned_x {
                delta_y.abs()
            } else if pinned_y {
                delta_x.abs()
            } else {
                (delta_x * delta_x + delta_y * delta_y).sqrt()
            };
            if step_norm <= tol {
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

            // Backtracking to stay inside and reduce the relevant gradient norm.
            // When constrained to a boundary, only the tangential component of the
            // gradient should drive sufficient-decrease — the normal component is
            // expected to remain non-zero.
            let current_norm: f64 = if pinned_x {
                g_y.abs()
            } else if pinned_y {
                g_x.abs()
            } else {
                g_norm
            };
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
                let new_decrement_norm: f64 = if pinned_x {
                    new_g_y.abs()
                } else if pinned_y {
                    new_g_x.abs()
                } else {
                    new_g_norm
                };
                if new_decrement_norm < current_norm {
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
    let x_grid: Array1<f64> = array![0.0, 1.0];
    let y_grid: Array1<f64> = array![0.0, 1.0];
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
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(delta_x, delta_y, f.view(), d_f_d_x.view(), d_f_d_y.view(), d2_f_d_x_d_y.view());
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
    let x_grid: Array1<f64> = array![0.0, 1.0];
    let y_grid: Array1<f64> = array![0.0, 1.0];
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
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(delta_x, delta_y, f.view(), d_f_d_x.view(), d_f_d_y.view(), d2_f_d_x_d_y.view());

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

    // Setup an analytic polynomial function
    fn calculate_f(x: f64, y: f64, x_peak: f64, y_peak: f64) -> f64 {
        let result: f64 = -(x - x_peak).powi(2) - (y - y_peak).powi(2);

        result
    }
    fn calculate_d_f_d_x(x: f64, _y: f64, x_peak: f64) -> f64 {
        let result: f64 = -2.0 * (x - x_peak);

        result
    }
    fn calculate_d_f_d_y(_x: f64, y: f64, y_peak: f64) -> f64 {
        let result: f64 = -2.0 * (y - y_peak);

        result
    }
    fn calculate_d2_f_d_x_d_y(_x: f64, _y: f64) -> f64 {
        let result: f64 = 0.0;

        result
    }

    let mut f: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_x: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d2_f_d_x_d_y: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

    let n_x: usize = 2;
    let n_y: usize = 2;
    let x_grid: Array1<f64> = array![0.0, 1.0];
    let y_grid: Array1<f64> = array![0.0, 1.0];
    for i_x in 0..n_x {
        for i_y in 0..n_y {
            f[(i_x, i_y)] = calculate_f(x_grid[i_x], y_grid[i_y], x_peak, y_peak);
            d_f_d_x[(i_x, i_y)] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y], x_peak);
            d_f_d_y[(i_x, i_y)] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y], y_peak);
            d2_f_d_x_d_y[(i_x, i_y)] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
        }
    }

    let d_x: f64 = x_grid[1] - x_grid[0];
    let d_y: f64 = y_grid[1] - y_grid[0];
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_x, d_y, f.view(), d_f_d_x.view(), d_f_d_y.view(), d2_f_d_x_d_y.view());

    let result: BicubicStationaryPoint = bicubic_interpolator
        .find_stationary_point(1e-12, 100)
        .expect("Should find stationary point on cell boundary");

    assert_abs_diff_eq!(result.x, x_peak, epsilon = 1e-10);
    assert_abs_diff_eq!(result.y, y_peak, epsilon = 1e-10);
    assert!(result.is_max);
}
