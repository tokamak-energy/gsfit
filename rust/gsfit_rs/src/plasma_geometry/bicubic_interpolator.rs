use ndarray::{Array1, Array2, ArrayView2, array};

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
    /// x-coordinate of the root, normalised to `[0, 1]` within the cell, [dimensionless]
    pub x: f64,
    /// y-coordinate of the root, normalised to `[0, 1]` within the cell, [dimensionless]
    pub y: f64,
    /// `true` if the root is a maximum. Evaluated from the Jacobian in **normalised** cell
    /// coordinates: the determinant's sign is unaffected by that scaling, and at a genuine
    /// extremum both diagonal entries share a sign so the trace's sign is unaffected too.
    /// A caller needing curvature **magnitudes** should recompute the Hessian in physical units.
    pub is_max: bool,
    /// `sqrt(u**2 + v**2)` at the root, i.e. how close the two fields actually got to zero
    pub residual_norm: f64,
    /// Number of Newton iterations used by the starting point which succeeded
    pub iter: usize,
}

#[derive(Debug)]
pub enum ErrorType {
    /// The 2x2 Jacobian is singular or too ill-conditioned to solve
    JacobianIsSingular,
    /// Newton converged, but to a root outside this cell
    NoInteriorRootFound,
    /// Newton did not converge within `max_iter`
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
    /// * `f[(0, 0)] = f[(i_y_lower, i_x_left)]`;
    /// * `f[(0, 1)] = f[(i_y_upper, i_x_left)]`;
    /// * `f[(1, 0)] = f[(i_y_lower, i_x_right)]`;
    /// * `f[(1, 1)] = f[(i_y_upper, i_x_right)]`;
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
        assert!(f.shape() == [2, 2], "f should be a 2x2 array");
        assert!(d_f_d_x.shape() == [2, 2], "d_f_d_x should be a 2x2 array");
        assert!(d_f_d_y.shape() == [2, 2], "d_f_d_y should be a 2x2 array");
        assert!(d2_f_d_x_d_y.shape() == [2, 2], "d2_f_d_x_d_y should be a 2x2 array");

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

        let d_f_d_x_normalised: Array2<f64> = d_f_d_x.to_owned() * d_x;
        let d_f_d_y_normalised: Array2<f64> = d_f_d_y.to_owned() * d_y;
        let d2_f_d_x_d_y_normalised: Array2<f64> = d2_f_d_x_d_y.to_owned() * d_x * d_y;

        // Note: we want the data to come in as (i_y, i_x) because arrays are stored as (i_z, i_r).
        // To achieve this we have effectively transposed the indexing of `f`, `d_f_d_x`, `d_f_d_y`, and `d2_f_d_x_d_y`, i.e.
        // function_matrix = [f.t(),       d_f_d_y.t()
        //                    d_f_d_x.t(), d2_f_d_x_d_y.t()]
        #[rustfmt::skip]
        let function_matrix: Array2<f64> = array![
            [f[(0, 0)],                  f[(1, 0)],                  d_f_d_y_normalised[(0, 0)],      d_f_d_y_normalised[(1, 0)]     ],
            [f[(0, 1)],                  f[(1, 1)],                  d_f_d_y_normalised[(0, 1)],      d_f_d_y_normalised[(1, 1)]     ],
            [d_f_d_x_normalised[(0, 0)], d_f_d_x_normalised[(1, 0)], d2_f_d_x_d_y_normalised[(0, 0)], d2_f_d_x_d_y_normalised[(1, 0)]],
            [d_f_d_x_normalised[(0, 1)], d_f_d_x_normalised[(1, 1)], d2_f_d_x_d_y_normalised[(0, 1)], d2_f_d_x_d_y_normalised[(1, 1)]],
        ];

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

}

/// Find the `(x, y)` at which **two** bicubic fields are simultaneously zero.
///
/// For the equilibrium code the two fields are `d(psi)/d(r)` and `d(psi)/d(z)`, so their common
/// root is a stationary point of `psi`.
///
/// This is a **root-find on a vector field**, not a stationary-point search on a scalar. The
/// system solved by each Newton step is
/// ```text
///     [ d(u)/d(x)  d(u)/d(y) ] [delta_x]      [u]
///     [ d(v)/d(x)  d(v)/d(y) ] [delta_y]  =  -[v]
/// ```
/// Note the Jacobian is **not** symmetric in general, because `d(u)/d(y)` and `d(v)/d(x)` are
/// estimated from two independent interpolants. That is deliberate: it avoids ever fitting a
/// bicubic to `psi` and differentiating it, which is unreliable inside a discretised current
/// region (see the note on `find_stationary_points_using_winding_number`).
///
/// # Arguments
/// * `u_interpolator` - bicubic model of the first field, e.g. `d(psi)/d(r)`
/// * `v_interpolator` - bicubic model of the second field, e.g. `d(psi)/d(z)`
/// * `tol` - convergence tolerance on the Newton step, as a fraction of the cell size
/// * `max_iter` - maximum Newton iterations **per starting point**
///
/// # Returns
/// * `Ok(BicubicStationaryPoint)` if a root was found inside the cell
/// * `Err(ErrorType)` otherwise; the caller should treat the cell as a false positive
///
/// # Starting points
/// The first guess comes from linearising `u` and `v` from their corner values and solving the
/// resulting 2x2 system. If that is degenerate or lands outside the cell it is skipped. Should
/// the Newton from it fail, the mid-point of each cell edge is tried in turn: bottom `(0.5, 0.0)`,
/// right `(1.0, 0.5)`, top `(0.5, 1.0)`, left `(0.0, 0.5)`, and then the four corners. A root
/// sitting near one edge or corner is usually reached easily from that edge's mid-point or that
/// corner.
pub fn find_stationary_point(
    u_interpolator: &BicubicInterpolator,
    v_interpolator: &BicubicInterpolator,
    tol: f64,
    max_iter: usize,
) -> Result<BicubicStationaryPoint, ErrorType> {
    // Corner values of both fields, used for the linearised initial guess
    let u_00: f64 = u_interpolator.value_and_derivatives(0.0, 0.0).f;
    let u_10: f64 = u_interpolator.value_and_derivatives(1.0, 0.0).f;
    let u_01: f64 = u_interpolator.value_and_derivatives(0.0, 1.0).f;
    let v_00: f64 = v_interpolator.value_and_derivatives(0.0, 0.0).f;
    let v_10: f64 = v_interpolator.value_and_derivatives(1.0, 0.0).f;
    let v_01: f64 = v_interpolator.value_and_derivatives(0.0, 1.0).f;

    // Approximate both fields as linear in `(x, y)` and solve `u = 0`, `v = 0` by Cramer's rule:
    //   u(x, y) ~ u_00 + (u_10 - u_00) * x + (u_01 - u_00) * y = 0
    //   v(x, y) ~ v_00 + (v_10 - v_00) * x + (v_01 - v_00) * y = 0
    let a_1: f64 = u_10 - u_00;
    let a_2: f64 = u_01 - u_00;
    let b_1: f64 = v_10 - v_00;
    let b_2: f64 = v_01 - v_00;
    let det_linear: f64 = a_1 * b_2 - a_2 * b_1;

    let mut starting_points: Vec<(f64, f64)> = Vec::with_capacity(9);
    if det_linear.abs() > f64::EPSILON {
        let x_linear: f64 = (-u_00 * b_2 + v_00 * a_2) / det_linear;
        let y_linear: f64 = (-v_00 * a_1 + u_00 * b_1) / det_linear;
        if (0.0..=1.0).contains(&x_linear) && (0.0..=1.0).contains(&y_linear) {
            starting_points.push((x_linear, y_linear));
        }
    }
    
    // We might have cases where the Newton iteration fails, e.g. when there is a stationary point in an adjacent cell, the Newton iteration can
    // follow that root and miss the true root in this cell. We therefore add several fallback starting points to try to find the root in this cell.
    // Fallback to the centre of the cell
    starting_points.push((0.5, 0.5));
    // Fallback to the mid-point of each cell edge: bottom, right, top, left
    starting_points.push((0.5, 0.0));
    starting_points.push((1.0, 0.5));
    starting_points.push((0.5, 1.0));
    starting_points.push((0.0, 0.5));
    // Then the four corners
    starting_points.push((0.0, 0.0));
    starting_points.push((1.0, 0.0));
    starting_points.push((1.0, 1.0));
    starting_points.push((0.0, 1.0));

    let mut last_error: ErrorType = ErrorType::NoInteriorRootFound;
    for &(x_start, y_start) in &starting_points {
        match newton_solve_for_common_root(u_interpolator, v_interpolator, x_start, y_start, tol, max_iter) {
            Ok(stationary_point) => return Ok(stationary_point),
            Err(error) => last_error = error,
        }
    }

    return Err(last_error);
}

/// One Newton solve for `u = 0`, `v = 0`, from a single starting point.
fn newton_solve_for_common_root(
    u_interpolator: &BicubicInterpolator,
    v_interpolator: &BicubicInterpolator,
    x_start: f64,
    y_start: f64,
    tol: f64,
    max_iter: usize,
) -> Result<BicubicStationaryPoint, ErrorType> {
    let mut x: f64 = x_start;
    let mut y: f64 = y_start;

    for i_iteration in 0..max_iter {
        let u: BicubicValueAndDerivatives = u_interpolator.value_and_derivatives(x, y);
        let v: BicubicValueAndDerivatives = v_interpolator.value_and_derivatives(x, y);

        let jacobian_determinant: f64 = u.d_f_d_x * v.d_f_d_y - u.d_f_d_y * v.d_f_d_x;

        // Guard against a singular or ill-conditioned Jacobian. The tolerance is scaled by the
        // magnitude of the Jacobian entries, so it behaves the same whatever units the fields
        // carry; a bare absolute threshold would be meaningless for a quantity with dimensions.
        let jacobian_scale: f64 = u.d_f_d_x.abs().max(u.d_f_d_y.abs()).max(v.d_f_d_x.abs()).max(v.d_f_d_y.abs());
        let jacobian_determinant_tol: f64 = 1.0e-12_f64.max(16.0 * f64::EPSILON * jacobian_scale * jacobian_scale);
        if jacobian_determinant.abs() <= jacobian_determinant_tol {
            return Err(ErrorType::JacobianIsSingular);
        }

        let delta_x: f64 = (-u.f * v.d_f_d_y + v.f * u.d_f_d_y) / jacobian_determinant;
        let delta_y: f64 = (-v.f * u.d_f_d_x + u.f * v.d_f_d_x) / jacobian_determinant;

        // Limit a single step so the iterate cannot fly far outside the cell. It is still free to
        // leave the cell and come back, which a hard clamp onto the boundary would prevent.
        let step_norm: f64 = (delta_x * delta_x + delta_y * delta_y).sqrt();
        let step_limit: f64 = 2.0;
        let scaling: f64 = if step_norm > step_limit { step_limit / step_norm } else { 1.0 };
        x += scaling * delta_x;
        y += scaling * delta_y;

        // Convergence test on the Newton step, in cell-normalised coordinates
        if (scaling * delta_x).abs() <= tol && (scaling * delta_y).abs() <= tol {
            // A small tolerance lets a root sitting exactly on a shared edge be accepted
            let inside_cell: bool = (-1e-9..=1.0 + 1e-9).contains(&x) && (-1e-9..=1.0 + 1e-9).contains(&y);
            if !inside_cell {
                return Err(ErrorType::NoInteriorRootFound);
            }

            let u_at_root: BicubicValueAndDerivatives = u_interpolator.value_and_derivatives(x, y);
            let v_at_root: BicubicValueAndDerivatives = v_interpolator.value_and_derivatives(x, y);
            let determinant: f64 = u_at_root.d_f_d_x * v_at_root.d_f_d_y - u_at_root.d_f_d_y * v_at_root.d_f_d_x;
            let trace: f64 = u_at_root.d_f_d_x + v_at_root.d_f_d_y;

            return Ok(BicubicStationaryPoint {
                x,
                y,
                is_max: determinant > 0.0 && trace < 0.0,
                residual_norm: (u_at_root.f * u_at_root.f + v_at_root.f * v_at_root.f).sqrt(),
                iter: i_iteration,
            });
        }
    }

    return Err(ErrorType::MaxIterationsReached);
}

#[test]
fn test_bicubic_interpolation() {
    // Lazy loading for crates which are only used within the tests
    use approx::assert_abs_diff_eq;

    // Setup an analytic polynomial function
    // The bicubic interpolation will be exact for polynomials up to cubic
    // This tests a peaked quadratic in both directions
    fn calculate_f(x: f64, y: f64) -> f64 {
        -(x - 0.5).powi(2) - (y - 0.5).powi(2)
    }
    fn calculate_d_f_d_x(x: f64, _y: f64) -> f64 {
        1.0 - 2.0 * x
    }
    fn calculate_d_f_d_y(_x: f64, y: f64) -> f64 {
        1.0 - 2.0 * y
    }
    fn calculate_d2_f_d_x_d_y(_x: f64, _y: f64) -> f64 {
        0.0
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
            f[(i_y, i_x)] = calculate_f(x_grid[i_x], y_grid[i_y]);
            d_f_d_x[(i_y, i_x)] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y]);
            d_f_d_y[(i_y, i_x)] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y]);
            d2_f_d_x_d_y[(i_y, i_x)] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
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

/// Build interpolators for `d(f)/d(x)` and `d(f)/d(y)` of the quadratic
/// `f = -(x - x_peak)^2 - (y - y_peak)^2`, whose common root is `(x_peak, y_peak)`.
#[cfg(test)]
fn quadratic_gradient_interpolators(x_peak: f64, y_peak: f64) -> (BicubicInterpolator, BicubicInterpolator) {
    let x_grid: Array1<f64> = array![0.0, 1.0];
    let y_grid: Array1<f64> = array![0.0, 1.0];
    let d_x: f64 = x_grid[1] - x_grid[0];
    let d_y: f64 = y_grid[1] - y_grid[0];

    // u = d(f)/d(x) = -2 * (x - x_peak);  d(u)/d(x) = -2, d(u)/d(y) = 0, d2(u)/d(x)d(y) = 0
    // v = d(f)/d(y) = -2 * (y - y_peak);  d(v)/d(x) = 0, d(v)/d(y) = -2, d2(v)/d(x)d(y) = 0
    let mut u: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut v: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    for i_x in 0..2usize {
        for i_y in 0..2usize {
            u[(i_y, i_x)] = -2.0 * (x_grid[i_x] - x_peak);
            v[(i_y, i_x)] = -2.0 * (y_grid[i_y] - y_peak);
        }
    }
    let minus_two: Array2<f64> = Array2::from_elem([2, 2], -2.0);
    let zeros: Array2<f64> = Array2::zeros((2, 2));

    let u_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_x, d_y, u.view(), minus_two.view(), zeros.view(), zeros.view());
    let v_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_x, d_y, v.view(), zeros.view(), minus_two.view(), zeros.view());

    return (u_interpolator, v_interpolator);
}

/// The root lies just inside the cell, very close to the top-right corner
#[test]
fn test_bicubic_find_stationary_point_near_boundary() {
    use approx::assert_abs_diff_eq;

    let x_peak: f64 = 0.9999;
    let y_peak: f64 = 0.9999;
    let (u_interpolator, v_interpolator): (BicubicInterpolator, BicubicInterpolator) = quadratic_gradient_interpolators(x_peak, y_peak);

    let result: BicubicStationaryPoint =
        find_stationary_point(&u_interpolator, &v_interpolator, 1e-12, 100).expect("Should find a root near the cell boundary");

    assert_abs_diff_eq!(result.x, x_peak, epsilon = 1e-10);
    assert_abs_diff_eq!(result.y, y_peak, epsilon = 1e-10);
    assert!(result.is_max);
    assert!(result.residual_norm < 1e-9);
}

/// The root lies exactly on a cell edge, so it must still be accepted
#[test]
fn test_bicubic_find_stationary_point_on_boundary() {
    use approx::assert_abs_diff_eq;

    let x_peak: f64 = 1.0;
    let y_peak: f64 = 0.78;
    let (u_interpolator, v_interpolator): (BicubicInterpolator, BicubicInterpolator) = quadratic_gradient_interpolators(x_peak, y_peak);

    let result: BicubicStationaryPoint =
        find_stationary_point(&u_interpolator, &v_interpolator, 1e-12, 100).expect("Should find a root on the cell boundary");

    assert_abs_diff_eq!(result.x, x_peak, epsilon = 1e-10);
    assert_abs_diff_eq!(result.y, y_peak, epsilon = 1e-10);
    assert!(result.is_max);
}

/// A root outside the cell must be rejected rather than reported on the boundary, which is the
/// failure mode that discarded a valid magnetic axis before this routine was rewritten.
#[test]
fn test_bicubic_find_stationary_point_rejects_root_outside_the_cell() {
    let (u_interpolator, v_interpolator): (BicubicInterpolator, BicubicInterpolator) = quadratic_gradient_interpolators(2.5, 0.5);

    let result: Result<BicubicStationaryPoint, ErrorType> = find_stationary_point(&u_interpolator, &v_interpolator, 1e-12, 100);

    assert!(result.is_err(), "a root at x = 2.5 is outside the cell and must not be returned");
}
