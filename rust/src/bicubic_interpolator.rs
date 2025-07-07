use core::f64;
use ndarray::{Array1, Array2, s};

/// Bicubic interpolation
/// https://en.wikipedia.org/wiki/Bicubic_interpolation
///
/// f, d_f_d_x, d_f_d_y, and d2_f_d_x_d_y should be indexed like:
/// = [f(x=0, y=0), f(x=0, y=1);
///    f(x=1, y=0), f(x=1, y=1)]
/// equivently,
/// f[[0, 0]] = f[[i_x_left, i_y_lower]];
/// f[[0, 1]] = f[[i_x_left, i_y_upper]];
/// f[[1, 0]] = f[[i_x_right, i_y_lower]];
/// f[[1, 1]] = f[[i_x_right, i_y_upper]];
pub fn bicubic_interpolation(f: &Array2<f64>, d_f_d_x: &Array2<f64>, d_f_d_y: &Array2<f64>, d2_f_d_x_d_y: &Array2<f64>, x: f64, y: f64) -> f64 {
    let x_vec: Array1<f64> = Array1::from_vec(vec![1.0, x, x.powi(2), x.powi(3)]);
    let y_vec: Array1<f64> = Array1::from_vec(vec![1.0, y, y.powi(2), y.powi(3)]);

    let coeff_matrix_1: Array2<f64> = Array2::from_shape_vec((4, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 3.0, -2.0, -1.0, 2.0, -2.0, 1.0, 1.0])
        .expect("bicubic_interpolation.coeff_matrix_1: Failed to create Array2 from shape vec");

    let coeff_matrix_2: Array2<f64> = Array2::from_shape_vec((4, 4), vec![1.0, 0.0, -3.0, 2.0, 0.0, 0.0, 3.0, -2.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, -1.0, 1.0])
        .expect("bicubic_interpolation.coeff_matrix_2: Failed to create Array2 from shape vec");

    let mut function_matrix: Array2<f64> = Array2::zeros((4, 4));
    function_matrix.slice_mut(s![0..2, 0..2]).assign(&f);
    function_matrix.slice_mut(s![2..4, 0..2]).assign(&d_f_d_x);
    function_matrix.slice_mut(s![0..2, 2..4]).assign(&d_f_d_y);
    function_matrix.slice_mut(s![2..4, 2..4]).assign(&d2_f_d_x_d_y);

    let a: Array2<f64> = coeff_matrix_1.dot(&function_matrix).dot(&coeff_matrix_2);

    let result: f64 = x_vec.dot(&a).dot(&y_vec);

    return result;
}

#[test]
fn test_bicubic_interpolation_fn() {
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
            f[[i_x, i_y]] = calculate_f(x_grid[i_x], y_grid[i_y]);
            d_f_d_x[[i_x, i_y]] = calculate_d_f_d_x(x_grid[i_x], y_grid[i_y]);
            d_f_d_y[[i_x, i_y]] = calculate_d_f_d_y(x_grid[i_x], y_grid[i_y]);
            d2_f_d_x_d_y[[i_x, i_y]] = calculate_d2_f_d_x_d_y(x_grid[i_x], y_grid[i_y]);
        }
    }

    // Create a grid to interpolate onto, and to calculate the analytic values
    let n_x_target: usize = 6;
    let n_y_target: usize = 5;
    let x_targets: Array1<f64> = Array1::linspace(0.01, 0.99, n_x_target);
    let y_targets: Array1<f64> = Array1::linspace(0.01, 0.99, n_y_target);
    let mut f_analytic: Array2<f64> = Array2::from_elem([n_x_target, n_y_target], f64::NAN);
    let mut f_interpolated: Array2<f64> = Array2::from_elem([n_x_target, n_y_target], f64::NAN);
    for i_x_target in 0..n_x_target {
        for i_y_target in 0..n_y_target {
            f_analytic[[i_x_target, i_y_target]] = calculate_f(x_targets[i_x_target], y_targets[i_y_target]);
            f_interpolated[[i_x_target, i_y_target]] =
                bicubic_interpolation(&f, &d_f_d_x, &d_f_d_y, &d2_f_d_x_d_y, x_targets[i_x_target], y_targets[i_y_target]);
        }
    }

    assert_abs_diff_eq!(&f_analytic, &f_interpolated);
}
