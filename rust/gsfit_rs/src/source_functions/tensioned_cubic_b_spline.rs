use crate::source_functions::SourceFunctionTraits;
use ndarray::{Array1, Array2, s};
use ndarray_linalg::assert;
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct TensionedCubicBSpline {
    pub n_dof: usize,
    pub regularisations: Array2<f64>,
    pub interior_knots: Array1<f64>,
    pub knots: Array1<f64>,
    pub interval_tensions: Array1<f64>,
    pub hyperbolic_upper_cutoff: f64,
    pub hyperbolic_lower_cutoff: f64,
    pub delta_cutoff: f64,
    pub tensions: Array1<f64>,
    pub delta_knots: Array1<f64>,
}

/// Python accessible methods
#[pymethods]
impl TensionedCubicBSpline {
    /// Create a new TensionedCubicBSpline
    #[new]
    pub fn new(regularisations: PyReadonlyArray2<f64>, interior_knots: PyReadonlyArray1<f64>, interval_tensions: PyReadonlyArray1<f64>) -> Self {
        // Change Python types into Rust types
        let regularisations_ndarray: Array2<f64> = regularisations.to_owned_array();
        let interior_knots_ndarray: Array1<f64> = interior_knots.to_owned_array();
        let interval_tensions_ndarray: Array1<f64> = interval_tensions.to_owned_array();

        let n_dof: usize = regularisations_ndarray.ncols();
        let n_knots: usize = interior_knots_ndarray.len() + 8;
        let mut knots: Array1<f64> = Array1::from_elem(n_knots, f64::NAN);
        knots.slice_mut(s![0..4]).fill(0.0);
        knots.slice_mut(s![4..(n_knots - 4)]).assign(&interior_knots_ndarray);
        knots.slice_mut(s![(n_knots - 4)..n_knots]).fill(1.0);
        let knots: Array1<f64> = knots;

        let mut delta_knots: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        for i_knot in 0..(n_knots - 1) {
            delta_knots[i_knot] = knots[i_knot + 1] - knots[i_knot];
        }
        let delta_knots: Array1<f64> = delta_knots;

        let mut tensions: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        tensions.slice_mut(s![0..3]).fill(0.0);
        tensions.slice_mut(s![3..(n_knots - 4)]).assign(&interval_tensions_ndarray);
        tensions.slice_mut(s![(n_knots - 4)..(n_knots - 1)]).fill(0.0);
        let tensions: Array1<f64> = tensions;

        // The `rho_delta_upper_cutoff` value is used by the gamma3 and gamma2 functions below to avoid overflow issues
        // when taking the cosh and sinh of large numbers.
        // Note that the largest number you can take exp(x) in rust for 64-bit floats should be around
        // ln(1.7976931348623157e308) = 709.782712893384, but we use 500 to be safe.
        let hyperbolic_upper_cutoff: f64 = 100.0;

        // Decides the cutoff for when we take the taylor series for small arguments for sinh and cosh in the
        // gamma3 and gamma2 functions.
        // This prevents taking e.g. sinh(x) - x of a very small number which results in numerical precision loss.
        let hyperbolic_lower_cutoff: f64 = 1e-3;

        //  Used by gamma3 and gamma2 functions below.
        // Decides the cut off for when we consider the distance between knots to be zero,
        // and therefore we must be in an exterioir (clamped) knot region, not the interior knot region.
        let delta_cutoff: f64 = 1e-12;

        // Create the struct
        TensionedCubicBSpline {
            n_dof,
            regularisations: regularisations_ndarray,
            interior_knots: interior_knots_ndarray,
            knots,
            interval_tensions: interval_tensions_ndarray,
            hyperbolic_upper_cutoff,
            hyperbolic_lower_cutoff,
            delta_cutoff,
            tensions,
            delta_knots,
        }
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║ {:<75} ║\n", " <gsfit_rs.Coils>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_dof: usize = self.n_dof;
        string_output += &format!("║ {:<75} ║\n", format!(" n_dof = {}", n_dof.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}

impl TensionedCubicBSpline {
    // This is Equation (2.1) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // Note that we have added extra checks to avoid overflow and numerical precision issues
    fn gamma3(&self, x_val: f64, rho: f64, delta: f64) -> f64 {
        // x_val, rho and delta must be positive
        assert!(x_val >= 0.0, "x_val must be non-negative");
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");

        let rho_delta: f64 = rho * delta;
        let rho_x: f64 = rho * x_val;

        if delta < self.delta_cutoff {
            // We are in the exterior (clamped) knot region where delta is zero
            return 0.0;
        }

        if rho_delta > self.hyperbolic_upper_cutoff {
            // To avoid overflow issues when taking sinh of large numbers we replace sinh() with exp()/2 approximation
            return (rho * (x_val - delta)).exp() * rho.powi(-2);
        }

        if rho_delta < self.hyperbolic_lower_cutoff {
            // Since x_val < delta, then we need to taylor expand both the numerator and denominator for small arguments
            return x_val.powi(3) / 6.0;
        }

        if rho_x < self.hyperbolic_lower_cutoff {
            // Use Taylor series expansion for just the numerator
            return rho * x_val.powi(3) / (6.0 * rho_delta.sinh());
        }

        // Normal case
        (rho_x.sinh() - rho_x) / (rho.powi(2) * rho_delta.sinh())
    }

    // This is Equation (2.2) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // Note that x always equals delta in our implementation so we have removed x as an argument.
    fn gamma2(&self, rho: f64, delta: f64) -> f64 {
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");

        let rho_delta: f64 = rho * delta;

        if delta < self.delta_cutoff {
            // We are in the exterior (clamped) knot region where delta is zero
            return 0.0;
        }

        if rho_delta > self.hyperbolic_upper_cutoff {
            // To avoid overflow issues when taking sinh or cosh of large numbers we replace sinh and cosh with exp()/2 approximation
            return rho.powi(-1);
        }

        if rho_delta < self.hyperbolic_lower_cutoff {
            // Use Taylor series expansion for small arguments to avoid numerical precision loss
            return 0.5 * delta;
        }

        // Normal case
        (rho_delta.cosh() - 1.0) / (rho * rho_delta.sinh())
    }

    // This is Equation (2.7) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // For the case where r = 2.
    // fn b_spline2(&self, j_index: usize, psi_n: f64) -> f64 {
    //     assert!(j_index < self.n_dof, "j_index for b_spline2 out of bounds");
    //     assert!(j_index >= 0, "j_index for b_spline2 must be non-negative");
    //     assert!(psi_n >= 0.0 && psi_n <= 1.0, "psi_n must be in the range [0, 1]");

    //     let sigma_j = gamma2(self.tensions[j_index], self.delta_knots[j_index]) + gamma2(self.tensions[j_index + 1], self.delta_knots[j_index + 1]);
    //     let sigma_j1 = gamma2(self.tensions[j_index + 1], self.delta_knots[j_index + 1]) + gamma2(self.tensions[j_index + 2], self.delta_knots[j_index + 2]);

    //     if psi_n < self.knots[j_index] || psi_n >= self.knots[j_index + 3] {
    //         return 0.0;
    //     }

    //     if psi_n < self.knots[j_index + 1] && psi_n >= self.knots[j_index] {
    //         return gamma2(psi_n - self.knots[j_index]) / sigma_j;
    //     }

    //     if psi_n < self.knots[j_index + 2] && psi_n >= self.knots[j_index + 1] {
    //         return 1 - gamma2(psi_n - self.knots[j_index + 1]) / sigma_j1 - gamma2(self.knots[j_index + 3] - psi_n) / sigma_j1;
    //     }

    //     0.0
    // }

    // This is Equation (2.6) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // For the case where r = 2.
    // fn phi2(spline_index: usize, x_val: f64,)
}

impl SourceFunctionTraits for TensionedCubicBSpline {
    fn source_function_value_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        let value: Array1<f64> = (1.0 - psi_n) * &psi_n.powi(i_dof as i32);
        return value;
    }

    fn source_function_derivative_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("Source function is not implemented yet");
    }

    fn source_function_integral_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        let integral: Array1<f64> = (1.0 - psi_n / 2.0) * psi_n.mapv(|x| x.powi(i_dof as i32 + 1)) / (i_dof as f64 + 1.0);
        return integral;
    }

    fn source_function_value(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        let n_psi_n: usize = psi_n.len();
        let n_dof: usize = polynomial_dof.len();

        let mut value: Array1<f64> = Array1::zeros(n_psi_n);
        for i_dof in 0..n_dof {
            value = value + polynomial_dof[i_dof] * self.source_function_value_single_dof(psi_n, i_dof);
        }

        return value;
    }

    fn source_function_derivative(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("Source function is not implemented yet");
    }

    fn source_function_integral(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        let n_dof: usize = self.n_dof;
        let n_psi_n: usize = psi_n.len();

        let mut constant_of_integration: f64 = 0.0;
        let psi_edge: Array1<f64> = Array1::from_vec(vec![1.0]);
        for i_dof in 0..n_dof {
            constant_of_integration = constant_of_integration - polynomial_dof[i_dof] * self.source_function_integral_single_dof(&psi_edge, i_dof)[0];
        }

        let mut integral: Array1<f64> = Array1::zeros(n_psi_n);
        for i_dof in 0..n_dof {
            integral = integral + polynomial_dof[i_dof] * self.source_function_integral_single_dof(&psi_n, i_dof);
        }
        integral = integral + constant_of_integration;

        return integral;
    }

    fn source_function_regularisation(&self) -> Array2<f64> {
        return self.regularisations.clone();
    }

    fn source_function_n_dof(&self) -> usize {
        return self.n_dof;
    }
}
