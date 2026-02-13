use crate::source_functions::SourceFunctionTraits;
use ndarray::{Array1, Array2, s};
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
    pub gamma3_array: Array1<f64>,
    pub sigma1_array: Array1<f64>,
    pub sigma2_array: Array1<f64>,
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
        let delta_cutoff: f64 = 1e-6;

        let mut gamma3_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        let mut gamma2_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        for i_knot in 0..n_knots-1 {
            gamma3_array[i_knot] = TensionedCubicBSpline::gamma3(
                delta_knots[i_knot],
                tensions[i_knot],
                delta_knots[i_knot],
                hyperbolic_upper_cutoff,
                hyperbolic_lower_cutoff,
                delta_cutoff,
            );
            gamma2_array[i_knot] = TensionedCubicBSpline::gamma2(
                tensions[i_knot],
                delta_knots[i_knot],
                hyperbolic_upper_cutoff,
                hyperbolic_lower_cutoff,
                delta_cutoff,
            );
        }
        let gamma3_array: Array1<f64> = gamma3_array;
        let gamma2_array: Array1<f64> = gamma2_array;
        let mut sigma1_array: Array1<f64> = Array1::from_elem(n_knots - 2, f64::NAN);
        for i in 0..n_knots - 2 {
            if knots[i + 2] - knots[i] < delta_cutoff {
                sigma1_array[i] = 0.0;
            } else {
                sigma1_array[i] = gamma2_array[i] + gamma2_array[i + 1];
            }
        }
        let sigma1_array: Array1<f64> = sigma1_array;
        let mut sigma2_array: Array1<f64> = Array1::from_elem(n_knots - 3, f64::NAN);
        for i in 0..n_knots - 3 {
            if knots[i + 3] - knots[i] < delta_cutoff {
                sigma2_array[i] = 0.0;
            } else {
                let tstar_jp: f64 = if knots[i + 2] - knots[i] > delta_cutoff {
                    knots[i + 1] + (gamma3_array[i + 1] - gamma3_array[i]) / (gamma2_array[i] + gamma2_array[i + 1])
                } else {
                    knots[i + 1]
                };
                let tstar_jpp: f64 = if knots[i + 3] - knots[i + 1] > delta_cutoff {
                    knots[i + 2] + (gamma3_array[i + 2] - gamma3_array[i + 1]) / (gamma2_array[i + 1] + gamma2_array[i + 2])
                } else {
                    knots[i + 2]
                };
                sigma2_array[i] = tstar_jpp - tstar_jp;
            }
        }
        let sigma2_array: Array1<f64> = sigma2_array;

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
            gamma3_array,
            sigma1_array,
            sigma2_array,
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

    pub fn phi2_python(&self, j_index: usize, x_val: f64) -> f64 {
        self.phi2(j_index, x_val)
    }

    pub fn gamma3_python(&self, x_val: f64, rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        TensionedCubicBSpline::gamma3(x_val, rho, delta, hyperbolic_upper_cutoff, hyperbolic_lower_cutoff, delta_cutoff)
    }

    pub fn gamma2_python(&self, rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        TensionedCubicBSpline::gamma2(rho, delta, hyperbolic_upper_cutoff, hyperbolic_lower_cutoff, delta_cutoff)
    }

    pub fn source_function_value_single_dof_python(&self, psi_n: f64, i_dof: usize) -> f64 {
        // convert single value to array of length 1 to reuse existing function
        let psi_n_array: Array1<f64> = Array1::from_elem(1, psi_n);
        self.source_function_value_single_dof(&psi_n_array, i_dof)[0]
    }
}

impl TensionedCubicBSpline {
    // This is Equation (2.1) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // Note that we have added extra checks to avoid overflow and numerical precision issues
    fn gamma3(x_val: f64, rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        // x_val, rho and delta must be positive
        assert!(x_val >= 0.0, "x_val must be non-negative");
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");
        assert!(x_val <= delta, "x_val must be less than or equal to delta");

        let rho_delta: f64 = rho * delta;
        let rho_x: f64 = rho * x_val;

        if delta < delta_cutoff {
            // We are in the exterior (clamped) knot region where delta is zero
            return 0.0;
        }

        if rho_delta > hyperbolic_upper_cutoff {
            // To avoid overflow issues when taking sinh of large numbers we replace sinh() with exp()/2 approximation
            return (rho * (x_val - delta)).exp() * rho.powi(-2);
        }

        if rho_delta < hyperbolic_lower_cutoff {
            // Since x_val < delta, then we need to taylor expand both the numerator and denominator for small arguments
            return x_val.powi(3) / (6.0 * delta);
        }

        if rho_x < hyperbolic_lower_cutoff {
            // Use Taylor series expansion for just the numerator
            return rho * x_val.powi(3) / (6.0 * rho_delta.sinh());
        }

        // Normal case
        (rho_x.sinh() - rho_x) / (rho.powi(2) * rho_delta.sinh())
    }

    // This is Equation (2.2) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // Note that x always equals delta in our implementation so we have removed x as an argument.
    fn gamma2(rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");

        let rho_delta: f64 = rho * delta;

        if delta < delta_cutoff {
            // We are in the exterior (clamped) knot region where delta is zero
            return 0.0;
        }

        if rho_delta > hyperbolic_upper_cutoff {
            // To avoid overflow issues when taking sinh or cosh of large numbers we replace sinh and cosh with exp()/2 approximation
            return rho.powi(-1);
        }

        if rho_delta < hyperbolic_lower_cutoff {
            // Use Taylor series expansion for small arguments to avoid numerical precision loss
            return 0.5 * delta;
        }

        // Normal case
        (rho_delta.cosh() - 1.0) / (rho * rho_delta.sinh())
    }

    // This is Equation (2.6) with Equation (2.7) substiuted in and integrated
    // from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // For the case where r = 2.
    fn phi2(&self, j_index: usize, x_val: f64) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");

        if x_val < self.knots[j_index] {
            return 0.0;
        }

        if (self.knots[j_index + 3] - self.knots[j_index]) < self.delta_cutoff {
            // See note just below Equation (2.6)
            return 1.0;
        }

        // Calculate sigma2_j using Equation (2.9)
        // Note that since self.knots[j_index + 3] - self.knots[j_index] > 0 we know that sigma2_j is not zero.
        let gamma3_j: f64 = self.gamma3_array[j_index]; // gamma3(delta_t_j, rho_j, delta_t_j)
        let gamma3_jp: f64 = self.gamma3_array[j_index + 1]; // gamma3(delta_t_{j+1}, rho_{j+1}, delta_t_{j+1})
        let gamma3_jpp: f64 = self.gamma3_array[j_index + 2]; // gamma3(delta_t_{j+2}, rho_{
        let sigma2_j: f64 = self.sigma2_array[j_index];

        // Evaluate integral of the first row of Equation (2.7)
        let sigma1_j: f64 = self.sigma1_array[j_index];
        if x_val < self.knots[j_index + 1] {
            if self.delta_knots[j_index] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitally set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
                return 0.0;
            } else {
                // Need to calculate int_{t_j}^x B2_j(y) dy
                // int_{t_j}^x B2_j(y) dy =
                // int_{t_j}^x gamma2(y-t_j) / sigma1_j dy =
                // [gamma3(y-t_j) / sigma1_j]_{t_j}^x =
                // gamma3(x-t_j) / sigma1_j - gamma3(0) / sigma1_j
                // Note that gamma3(0) = 0, so we do not need to subtract anything.

                // gamma3(x-t_j, rho_j, delta_t_j):
                let gamma3_x_m_tj: f64 = TensionedCubicBSpline::gamma3(
                    x_val - self.knots[j_index],
                    self.tensions[j_index],
                    self.delta_knots[j_index],
                    self.hyperbolic_upper_cutoff,
                    self.hyperbolic_lower_cutoff,
                    self.delta_cutoff,
                );

                // int_{t_j}^{x} B2_j(y) dy
                let integral_tj_x: f64 = gamma3_x_m_tj / sigma1_j;

                return integral_tj_x / sigma2_j;
            }
        }
        // x >= t_{j+1} so we need to calculate int_{t_j}^{t_{j+1}} B2_j(y) dy
        let integral_tj_tjp: f64 = if self.delta_knots[j_index] < self.delta_cutoff {
            // Since the distance between knots is very small the integral over the range will be approximately zero.
            // But we explicitally set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
            0.0
        } else {
            // Using expression above
            // int_{t_j}^{t_{j+1}} B2_j(y) dy =
            // gamma3(t_{j+1}-t_j) / sigma1_j
            gamma3_j / sigma1_j
        };

        // Evaluate integral of the second row of Equation (2.7)
        let sigma1_jp: f64 = self.sigma1_array[j_index + 1];
        if x_val < self.knots[j_index + 2] {
            if self.delta_knots[j_index + 1] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitally set integral of second row to zero (so only first row remains) to avoid possible
                // numerical issue of dividing by zero when sigma1_j or sigma1_{j+1} is zero
                return integral_tj_tjp / sigma2_j;
            } else {
                // Need to calculate int_{t_{j+1}}^x B2_j(y) dy
                // int_{t_{j+1}}^x B2_j(y) dy =
                // int_{t_{j+1}}^x (1 - gamma2(y-t_{j+1}) / sigma1_jp - gamma2(t_{j+2}-y) / sigma1_j) dy
                // = [y - gamma3(y-t_{j+1}) / sigma1_jp + gamma3(t_{j+2}-y) / sigma1_j]_{t_{j+1}}^x
                // = x - gamma3(x-t_{j+1}) / sigma1_jp + gamma3(t_{j+2}-x) / sigma1_j
                // - (t_{j+1} - gamma3(0) / sigma1_jp + gamma3(t_{j+2}-t_{j+1}) / sigma1_j)
                // Note that gamma3(0) = 0
                let gamma3_x_m_tjp: f64 = TensionedCubicBSpline::gamma3(
                    x_val - self.knots[j_index + 1],
                    self.tensions[j_index + 1],
                    self.delta_knots[j_index + 1],
                    self.hyperbolic_upper_cutoff,
                    self.hyperbolic_lower_cutoff,
                    self.delta_cutoff,
                );
                let gamma3_tjpp_m_x: f64 = TensionedCubicBSpline::gamma3(
                    self.knots[j_index + 2] - x_val,
                    self.tensions[j_index + 1],
                    self.delta_knots[j_index + 1],
                    self.hyperbolic_upper_cutoff,
                    self.hyperbolic_lower_cutoff,
                    self.delta_cutoff,
                );

                let integral_tjp_x: f64 = x_val - gamma3_x_m_tjp / sigma1_jp + gamma3_tjpp_m_x / sigma1_j - self.knots[j_index + 1] - gamma3_jp / sigma1_j;

                return (integral_tj_tjp + integral_tjp_x) / sigma2_j;
            }
        }
        // x >= t_{j+2} so we need to calculate int_{t_{j+1}}^{t_{j+2}} B2_j(y) dy
        let integral_tjp_tjpp: f64 = if self.delta_knots[j_index + 1] < self.delta_cutoff {
            // Since the distance between knots is very small the integral over the range will be approximately zero.
            // But we explicitally set integral of second row to zero to avoid possible
            // numerical issue of dividing by zero when sigma1_j or sigma1_{j+1} is zero
            0.0
        } else {
            // Using expression above
            // int_{t_{j+1}}^{t_{j+2}} B2_j(y) dy =
            // t_{j+2} - gamma3(t_{j+2}-t_{j+1}) / sigma1_jp + gamma3(0) / sigma1_j
            // - (t_{j+1} - gamma3(0) / sigma1_jp + gamma3(t_{j+2}-t_{j+1}) / sigma1_j) =
            // Δt_{j+1} - gamma3(Δt_{j+1}) / sigma1_jp - gamma3(Δt_{j+1}) / sigma1_j
            self.delta_knots[j_index + 1] - gamma3_jp / sigma1_jp - gamma3_jp / sigma1_j
        };

        // Evaluate integral of the third row of Equation (2.7)
        if x_val < self.knots[j_index + 3] {
            if self.delta_knots[j_index + 2] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitally set integral of third row to zero (so only first and second rows remain) to avoid possible
                // numerical issue of dividing by zero when sigma1_{j+1} is zero
                return (integral_tj_tjp + integral_tjp_tjpp) / sigma2_j;
            } else {
                // Need to calculate int_{t_{j+2}}^x B2_j(y) dy
                // int_{t_{j+2}}^x B2_j(y) dy =
                // int_{t_{j+2}}^x gamma2(t_{j+3} - y) / sigma1_jp dy =
                // [-gamma3(t_{j+3} - y) / sigma1_jp]_{t_{j+2}}^x =
                // -gamma3(t_{j+3} - x) / sigma1_jp + gamma3(t_{j+3} - t_{j+2}) / sigma1_jp
                let gamma3_tjppp_m_x: f64 = TensionedCubicBSpline::gamma3(
                    self.knots[j_index + 3] - x_val,
                    self.tensions[j_index + 2],
                    self.delta_knots[j_index + 2],
                    self.hyperbolic_upper_cutoff,
                    self.hyperbolic_lower_cutoff,
                    self.delta_cutoff,
                );
                let integral_tjpp_x: f64 = (gamma3_jpp - gamma3_tjppp_m_x) / sigma1_jp;
                return (integral_tj_tjp + integral_tjp_tjpp + integral_tjpp_x) / sigma2_j;
            }
        }

        // x >= t_{j+3}
        // Return 1.0, see third row of Equation (2.6) P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        1.0
    }
}

impl SourceFunctionTraits for TensionedCubicBSpline {
    fn source_function_value_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // See equation (2.5) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        let mut value: Array1<f64> = Array1::from_elem(psi_n.len(), f64::NAN);
        for i_psi_n in 0..psi_n.len() {
            let x = psi_n[i_psi_n];
            if x <= self.knots[i_dof] {
                value[i_psi_n] = 0.0;
            } else if x <= self.knots[i_dof + 1] {
                value[i_psi_n] = self.phi2(i_dof, x);
            } else if x <= self.knots[i_dof + 3] {
                value[i_psi_n] = self.phi2(i_dof, x) - self.phi2(i_dof + 1, x);
            } else if x <= self.knots[i_dof + 4] {
                value[i_psi_n] = 1.0 - self.phi2(i_dof + 1, x);
            } else {
                value[i_psi_n] = 0.0;
            }
        }
        return value;
    }

    fn source_function_derivative_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("Source function is not implemented yet");
    }

    fn source_function_integral_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        let n = psi_n.len();
        let mut out = Array1::zeros(n);

        // Compute f(psi) at all points
        let f_vals = self.source_function_value_single_dof(psi_n, i_dof);

        // Trapezoidal cumulative integral
        for i in 1..n {
            let dx: f64 = psi_n[i] - psi_n[i - 1];
            let trap: f64 = 0.5 * dx * (f_vals[i] + f_vals[i - 1]);
            out[i] = out[i - 1] + trap;
        }

        out
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

        let mut integral: Array1<f64> = Array1::zeros(n_psi_n);
        for i_dof in 0..n_dof {
            integral = integral + polynomial_dof[i_dof] * self.source_function_integral_single_dof(psi_n, i_dof);
        }

        let last_value = integral[n_psi_n - 1];

        return integral - last_value;
    }

    fn source_function_regularisation(&self) -> Array2<f64> {
        return self.regularisations.clone();
    }

    fn source_function_n_dof(&self) -> usize {
        return self.n_dof;
    }
}
