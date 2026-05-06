use crate::source_functions::SourceFunctionTraits;
use ndarray::{Array1, Array2, s};
use numpy::PyArray1;
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass(skip_from_py_object)]
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
    pub gamma2_array: Array1<f64>,
    pub gamma3_array: Array1<f64>,
    pub gamma4_array: Array1<f64>,
    pub sigma1_array: Array1<f64>,
    pub sigma2_array: Array1<f64>,
    pub tstar_array: Array1<f64>,
}

/// Python accessible methods
#[pymethods]
impl TensionedCubicBSpline {
    /// Create a new TensionedCubicBSpline
    #[new]
    pub fn new_py(regularisations: PyReadonlyArray2<f64>, interior_knots: PyReadonlyArray1<f64>, interval_tensions: PyReadonlyArray1<f64>) -> Self {
        // Change Python types into Rust types
        let regularisations_ndarray: Array2<f64> = regularisations.to_owned_array();
        let interior_knots_ndarray: Array1<f64> = interior_knots.to_owned_array();
        let interval_tensions_ndarray: Array1<f64> = interval_tensions.to_owned_array();

        // Call the Rust constructor
        TensionedCubicBSpline::new(regularisations_ndarray, interior_knots_ndarray, interval_tensions_ndarray)
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║ {:<75} ║\n", " <gsfit_rs.TensionedCubicBSpline>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_dof: usize = self.n_dof;
        string_output += &format!("║ {:<75} ║\n", format!(" n_dof = {}", n_dof.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    pub fn phi2_python(&self, j_index: usize, x_val: f64) -> f64 {
        self.phi2(j_index, x_val)
    }

    pub fn psi2_python(&self, j_index: usize, x_val: f64) -> f64 {
        self.psi2(j_index, x_val)
    }

    pub fn gamma4_python(&self, x_val: f64, rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        TensionedCubicBSpline::gamma4(x_val, rho, delta, hyperbolic_upper_cutoff, hyperbolic_lower_cutoff, delta_cutoff)
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

    pub fn source_function_integral_single_dof_python(&self, psi_n: f64, i_dof: usize) -> f64 {
        // convert single value to array of length 1 to reuse existing function
        let psi_n_array: Array1<f64> = Array1::from_elem(1, psi_n);
        self.source_function_integral_single_dof(&psi_n_array, i_dof)[0]
    }

    pub fn get_array1<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match name {
            "interior_knots" => Ok(PyArray1::from_array(py, &self.interior_knots)),
            "knots" => Ok(PyArray1::from_array(py, &self.knots)),
            "interval_tensions" => Ok(PyArray1::from_array(py, &self.interval_tensions)),
            "tensions" => Ok(PyArray1::from_array(py, &self.tensions)),
            "delta_knots" => Ok(PyArray1::from_array(py, &self.delta_knots)),
            "gamma2_array" => Ok(PyArray1::from_array(py, &self.gamma2_array)),
            "gamma3_array" => Ok(PyArray1::from_array(py, &self.gamma3_array)),
            "gamma4_array" => Ok(PyArray1::from_array(py, &self.gamma4_array)),
            "sigma1_array" => Ok(PyArray1::from_array(py, &self.sigma1_array)),
            "sigma2_array" => Ok(PyArray1::from_array(py, &self.sigma2_array)),
            "tstar_array" => Ok(PyArray1::from_array(py, &self.tstar_array)),
            _ => Err(PyValueError::new_err(format!("Unknown Array1 attribute: {}", name))),
        }
    }
}

impl TensionedCubicBSpline {
    pub fn new(regularisations: Array2<f64>, interior_knots: Array1<f64>, interval_tensions: Array1<f64>) -> Self {
        let n_dof: usize = regularisations.ncols();
        let n_knots: usize = interior_knots.len() + 8;
        let mut knots: Array1<f64> = Array1::from_elem(n_knots, f64::NAN);
        knots.slice_mut(s![0..4]).fill(0.0);
        knots.slice_mut(s![4..(n_knots - 4)]).assign(&interior_knots);
        knots.slice_mut(s![(n_knots - 4)..n_knots]).fill(1.0);
        let knots: Array1<f64> = knots;

        let mut delta_knots: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        for i_knot in 0..(n_knots - 1) {
            delta_knots[i_knot] = knots[i_knot + 1] - knots[i_knot];
        }
        let delta_knots: Array1<f64> = delta_knots;

        let mut tensions: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        tensions.slice_mut(s![0..3]).fill(0.0);
        tensions.slice_mut(s![3..(n_knots - 4)]).assign(&interval_tensions);
        tensions.slice_mut(s![(n_knots - 4)..(n_knots - 1)]).fill(0.0);
        let tensions: Array1<f64> = tensions;

        // The `rho_delta_upper_cutoff` value is used by the gamma functions below to avoid overflow issues
        // when taking the cosh and sinh of large numbers.
        // Note that the largest number you can take exp(x) in rust for 64-bit floats should be around
        // ln(1.7976931348623157e308) = 709.782712893384, but we use 100 to be safe.
        let hyperbolic_upper_cutoff: f64 = 100.0;

        // Decides the cutoff for when we take the taylor series for small arguments for sinh and cosh in the
        // gamma functions.
        // This prevents taking e.g. sinh(x) - x of a very small number which results in numerical precision loss.
        let hyperbolic_lower_cutoff: f64 = 1e-3;

        //  Used by gamma functions below.
        // Decides the cut off for when we consider the distance between knots to be zero,
        // and therefore we must be in an exterioir (clamped) knot region, not the interior knot region.
        let delta_cutoff: f64 = 1e-6;

        let mut gamma3_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        let mut gamma2_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        let mut gamma4_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        for i_knot in 0..n_knots - 1 {
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
            gamma4_array[i_knot] = TensionedCubicBSpline::gamma4(
                delta_knots[i_knot],
                tensions[i_knot],
                delta_knots[i_knot],
                hyperbolic_upper_cutoff,
                hyperbolic_lower_cutoff,
                delta_cutoff,
            );
        }
        let gamma3_array: Array1<f64> = gamma3_array;
        let gamma2_array: Array1<f64> = gamma2_array;
        let gamma4_array: Array1<f64> = gamma4_array;
        // sigma1 calculated using equation just above Equation (2.7) in P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        let mut sigma1_array: Array1<f64> = Array1::from_elem(n_knots - 2, f64::NAN);
        for i in 0..n_knots - 2 {
            if knots[i + 2] - knots[i] < delta_cutoff {
                sigma1_array[i] = 1.0;
            } else {
                sigma1_array[i] = gamma2_array[i] + gamma2_array[i + 1];
            }
        }
        let sigma1_array: Array1<f64> = sigma1_array;
        // tstar calculated using the equation just below Equation (2.7) in P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        // Note that we remap tstar_j there to tstar_{j+2} here as we think it makes more sense.
        let mut tstar_array: Array1<f64> = Array1::from_elem(n_knots - 1, f64::NAN);
        for i in 0..n_knots - 2 {
            if knots[i + 2] - knots[i] < delta_cutoff {
                tstar_array[i + 1] = knots[i + 1];
            } else {
                tstar_array[i + 1] = knots[i + 1] + (gamma3_array[i + 1] - gamma3_array[i]) / (gamma2_array[i] + gamma2_array[i + 1]);
            }
        }
        let tstar_array: Array1<f64> = tstar_array;
        let mut sigma2_array: Array1<f64> = Array1::from_elem(n_knots - 3, f64::NAN);
        // sigma2 calculated using Equation (2.9) in P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        for i in 0..n_knots - 3 {
            if knots[i + 3] - knots[i] < delta_cutoff {
                sigma2_array[i] = 1.0;
            } else {
                // let tstar_jp: f64 = if knots[i + 2] - knots[i] > delta_cutoff {
                //     knots[i + 1] + (gamma3_array[i + 1] - gamma3_array[i]) / (gamma2_array[i] + gamma2_array[i + 1])
                // } else {
                //     knots[i + 1]
                // };
                // let tstar_jpp: f64 = if knots[i + 3] - knots[i + 1] > delta_cutoff {
                //     knots[i + 2] + (gamma3_array[i + 2] - gamma3_array[i + 1]) / (gamma2_array[i + 1] + gamma2_array[i + 2])
                // } else {
                //     knots[i + 2]
                // };
                sigma2_array[i] = tstar_array[i + 2] - tstar_array[i + 1];
            }
        }
        let sigma2_array: Array1<f64> = sigma2_array;

        // Create the struct
        TensionedCubicBSpline {
            n_dof,
            regularisations,
            interior_knots,
            knots,
            interval_tensions,
            hyperbolic_upper_cutoff,
            hyperbolic_lower_cutoff,
            delta_cutoff,
            tensions,
            delta_knots,
            gamma2_array,
            gamma3_array,
            gamma4_array,
            sigma1_array,
            sigma2_array,
            tstar_array,
        }
    }

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
            // We assume we are in the exterior (clamped) knot region
            return 0.0;
        }

        if rho_delta > hyperbolic_upper_cutoff {
            // To avoid overflow for large arguments, we approximate sinh(x) and cosh(x)
            // by exp(x)/2, neglect smaller terms in the numerator, and replace the
            // division by subtraction in the exponent.
            return (rho * (x_val - delta)).exp() * rho.powi(-2);
        }

        if rho_delta < hyperbolic_lower_cutoff {
            // Since rho * delta is very small we need to taylor expand both the numerator and denominator for small arguments to avoid
            // significant numerical precision errors in the numerator and remove possibility of dividing by zero if rho is zero.
            return x_val.powi(3) / (6.0 * delta);
        }

        if rho_x < hyperbolic_lower_cutoff {
            // Use Taylor series expansion for just the numerator
            return rho * x_val.powi(3) / (6.0 * rho_delta.sinh());
        }

        // Normal case
        (rho_x.sinh() - rho_x) / (rho.powi(2) * rho_delta.sinh())
    }

    // This is the integral of gamma3 from 0 to x, which is used in the psi2 function below.
    fn gamma4(x_val: f64, rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        // x_val, rho and delta must be positive
        assert!(x_val >= 0.0, "x_val must be non-negative");
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");
        assert!(x_val <= delta, "x_val must be less than or equal to delta");

        let rho_delta: f64 = rho * delta;
        let rho_x: f64 = rho * x_val;

        if delta < delta_cutoff {
            // We assume we are in the exterior (clamped) knot region
            return 0.0;
        }

        if rho_delta > hyperbolic_upper_cutoff {
            // To avoid overflow for large arguments, we approximate sinh(x) and cosh(x)
            // by exp(x)/2, neglect smaller terms in the numerator, and replace the
            // division by subtraction in the exponent.
            return (rho * (x_val - delta)).exp() * rho.powi(-3);
        }

        if rho_delta < hyperbolic_lower_cutoff {
            // Since rho * delta is very small we need to taylor expand both the numerator and denominator for small arguments to avoid
            // significant numerical precision errors in the numerator and remove possibility of dividing by zero if rho is zero.
            return x_val.powi(4) / (24.0 * delta);
        }

        if rho_x < hyperbolic_lower_cutoff {
            // Use Taylor series expansion for just the numerator
            return rho * x_val.powi(4) / (24.0 * rho_delta.sinh());
        }

        // Normal case
        (rho_x.cosh() - 1.0 - 0.5 * rho_x.powi(2)) / (rho.powi(3) * rho_delta.sinh())
    }

    // This is Equation (2.2) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
    // Note that x always equals delta in our implementation so we have removed x as an argument.
    fn gamma2(rho: f64, delta: f64, hyperbolic_upper_cutoff: f64, hyperbolic_lower_cutoff: f64, delta_cutoff: f64) -> f64 {
        assert!(rho >= 0.0, "rho must be non-negative");
        assert!(delta >= 0.0, "delta must be non-negative");

        let rho_delta: f64 = rho * delta;

        if delta < delta_cutoff {
            // We assume we are in the exterior (clamped) knot region
            return 0.0;
        }

        if rho_delta > hyperbolic_upper_cutoff {
            // To avoid overflow for large arguments, we approximate sinh(x) and cosh(x)
            // by exp(x)/2, neglect smaller terms in the numerator, and replace the
            // division by subtraction in the exponent.
            return rho.powi(-1);
        }

        if rho_delta < hyperbolic_lower_cutoff {
            // Since rho * delta is very small we need to taylor expand both the numerator and denominator for small arguments to avoid
            // significant numerical precision errors in the numerator and remove possibility of dividing by zero if rho is zero.
            return 0.5 * delta;
        }

        // Normal case
        (rho_delta.cosh() - 1.0) / (rho * rho_delta.sinh())
    }

    // This is Equation (2.6) with Equation (2.7) substituted in and integrated
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
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");

        // Evaluate integral of the first row of Equation (2.7)
        let sigma1_j: f64 = self.sigma1_array[j_index];
        if x_val < self.knots[j_index + 1] {
            if self.delta_knots[j_index] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitly set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
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
            // But we explicitly set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
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
                // But we explicitly set integral of second row to zero (so only first row remains) to avoid possible
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
            // But we explicitly set integral of second row to zero to avoid possible
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
                // But we explicitly set integral of third row to zero (so only first and second rows remain) to avoid possible
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

    // This is the antiderivative of the phi2 function above in the segment
    // where t_{j} <= x < t_{j+1}
    fn phi2_antiderivative_seg1(&self, j_index: usize, x_val: f64) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        assert!(x_val >= self.knots[j_index], "x_val must be greater than or equal to t_j");
        assert!(x_val < self.knots[j_index + 1], "x_val must be less than t_{{j+1}}");
        // Note that integral_tj_x / sigma2_j in phi2 above simplifies to gamma3(x-t_j) / (sigma1_j * sigma2_j)
        // Hence the antiderivative of this with respect to x is gamma4_j(x-t_j) / (sigma1_j * sigma2_j).
        let gamma4_x_m_tj: f64 = TensionedCubicBSpline::gamma4(
            x_val - self.knots[j_index],
            self.tensions[j_index],
            self.delta_knots[j_index],
            self.hyperbolic_upper_cutoff,
            self.hyperbolic_lower_cutoff,
            self.delta_cutoff,
        );
        let sigma1_j: f64 = self.sigma1_array[j_index];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_j > 0.0, "sigma1_j must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        gamma4_x_m_tj / (sigma1_j * sigma2_j)
    }

    // Evaluates phi2_antiderivative_seg1 at x = t_{j+1} by using arrays precomputed in the constructor to avoid doing redundant calculations.
    fn phi2_antiderivative_seg1_at_tjp(&self, j_index: usize) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        let sigma1_j: f64 = self.sigma1_array[j_index];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_j > 0.0, "sigma1_j must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        self.gamma4_array[j_index] / (sigma1_j * sigma2_j)
    }

    // This is the antiderivative of the phi2 function above in the segment
    // where t_{j+1} <= x < t_{j+2}
    fn phi2_antiderivative_seg2(&self, j_index: usize, x_val: f64) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        assert!(x_val >= self.knots[j_index + 1], "x_val must be greater than or equal to t_{{j+1}}");
        assert!(x_val < self.knots[j_index + 2], "x_val must be less than t_{{j+2}}");
        // Note that (integral_tj_tjp + integral_tjp_x) / sigma2_j in phi2 above simplifies to
        // [x - gamma3_{j+1}(x - t_{j+1}) / sigma1_{j+1} + gamma3_{j+1}(t_{j+2}-x) / sigma1_j - tstar_{j+1}] / sigma2_j
        // Hence the antiderivative of this with respect to x is
        // [x^2 / 2 - gamma4_{j+1}(x - t_{j+1}) / sigma1_{j+1} - gamma4_{j+1}(t_{j+2}-x) / sigma1_j - tstar_{j+1} * x] / sigma2_j
        let gamma4_x_m_tjp: f64 = TensionedCubicBSpline::gamma4(
            x_val - self.knots[j_index + 1],
            self.tensions[j_index + 1],
            self.delta_knots[j_index + 1],
            self.hyperbolic_upper_cutoff,
            self.hyperbolic_lower_cutoff,
            self.delta_cutoff,
        );
        let gamma4_tjpp_m_x: f64 = TensionedCubicBSpline::gamma4(
            self.knots[j_index + 2] - x_val,
            self.tensions[j_index + 1],
            self.delta_knots[j_index + 1],
            self.hyperbolic_upper_cutoff,
            self.hyperbolic_lower_cutoff,
            self.delta_cutoff,
        );
        let tstar_jp: f64 = self.tstar_array[j_index + 1];
        let sigma1_j: f64 = self.sigma1_array[j_index];
        let sigma1_jp: f64 = self.sigma1_array[j_index + 1];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_j > 0.0, "sigma1_j must be positive");
        assert!(sigma1_jp > 0.0, "sigma1_jp must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        (0.5 * x_val.powi(2) - gamma4_x_m_tjp / sigma1_jp - gamma4_tjpp_m_x / sigma1_j - tstar_jp * x_val) / sigma2_j
    }

    // Evaluates phi2_antiderivative_seg2 at x = t_{j+1}
    fn phi2_antiderivative_seg2_at_tjp(&self, j_index: usize) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        // Note that at x = t_{j+1} the expression for phi2_antiderivative_seg2 simplifies to
        // [t_{j+1}^2 / 2 - gamma4_{j+1}(Δt_{j+1}) / sigma1_j - tstar_{j+1} * t_{j+1}] / sigma2_j
        let t_jp: f64 = self.knots[j_index + 1];
        let gamma4_jp: f64 = self.gamma4_array[j_index + 1];
        let tstar_jp: f64 = self.tstar_array[j_index + 1];
        let sigma1_j: f64 = self.sigma1_array[j_index];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_j > 0.0, "sigma1_j must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        (0.5 * t_jp.powi(2) - gamma4_jp / sigma1_j - tstar_jp * t_jp) / sigma2_j
    }

    // Evaluates phi2_antiderivative_seg2 at x = t_{j+2}
    fn phi2_antiderivative_seg2_at_tjpp(&self, j_index: usize) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        // Note that at x = t_{j+2} the expression for phi2_antiderivative_seg2 simplifies to
        // [t_{j+2}^2 / 2 - gamma4_{j+1}(Δt_{j+1}) / sigma1_{j+1} - tstar_{j+1} * t_{j+2}] / sigma2_j
        let t_jpp: f64 = self.knots[j_index + 2];
        let gamma4_jp: f64 = self.gamma4_array[j_index + 1];
        let tstar_jp: f64 = self.tstar_array[j_index + 1];
        let sigma1_jp: f64 = self.sigma1_array[j_index + 1];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_jp > 0.0, "sigma1_jp must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        (0.5 * t_jpp.powi(2) - gamma4_jp / sigma1_jp - tstar_jp * t_jpp) / sigma2_j
    }

    // This is the antiderivative of the phi2 function above in the segment
    // where t_{j+2} <= x < t_{j+3}
    fn phi2_antiderivative_seg3(&self, j_index: usize, x_val: f64) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        assert!(x_val >= self.knots[j_index + 2], "x_val must be greater than or equal to t_{{j+2}}");
        assert!(x_val < self.knots[j_index + 3], "x_val must be less than t_{{j+3}}");

        // Note that (integral_tj_tjp + integral_tjp_tjpp + integral_tjpp_x) / sigma2_j in phi2 above simplifies to
        // 1 - gamma3_{j+2}(t_{j+3}-x) / (sigma1_jp * sigma2_j)
        // Hence the antiderivative of this with respect to x is
        // x + gamma4_{j+2}(t_{j+3}-x) / (sigma1_jp * sigma2_j)
        let gamma4_tjppp_m_x: f64 = TensionedCubicBSpline::gamma4(
            self.knots[j_index + 3] - x_val,
            self.tensions[j_index + 2],
            self.delta_knots[j_index + 2],
            self.hyperbolic_upper_cutoff,
            self.hyperbolic_lower_cutoff,
            self.delta_cutoff,
        );
        let sigma1_jp: f64 = self.sigma1_array[j_index + 1];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_jp > 0.0, "sigma1_jp must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        x_val + gamma4_tjppp_m_x / (sigma1_jp * sigma2_j)
    }

    // Evaluates phi2_antiderivative_seg3 at x = t_{j+2}
    fn phi2_antiderivative_seg3_at_tjpp(&self, j_index: usize) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        // Note that at x = t_{j+2} the expression for phi2_antiderivative_seg3 simplifies to
        // t_{j+2} + gamma4_{j+2}(Δt_{j+2}) / (sigma1_{j+1} * sigma2_j)
        let t_jpp: f64 = self.knots[j_index + 2];
        let gamma4_jpp: f64 = self.gamma4_array[j_index + 2];
        let sigma1_jp: f64 = self.sigma1_array[j_index + 1];
        let sigma2_j: f64 = self.sigma2_array[j_index];
        assert!(sigma1_jp > 0.0, "sigma1_jp must be positive");
        assert!(sigma2_j > 0.0, "sigma2_j must be positive");
        t_jpp + gamma4_jpp / (sigma1_jp * sigma2_j)
    }

    // Evaluates phi2_antiderivative_seg3 at x = t_{j+3}
    fn phi2_antiderivative_seg3_at_tjppp(&self, j_index: usize) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for phi2 out of bounds");
        // Note that at x = t_{j+3} the expression for phi2_antiderivative_seg3 simplifies to
        // t_{j+3} + gamma4_{j+2}(0) / (sigma1_{j+1} * sigma2_j) = t_{j+3}
        self.knots[j_index + 3]
    }

    // We define psi2 as
    // Ψ²_j(x) = ∫_{−∞}^{x} Φ²_j(y) dy
    // This is used in the source_function_integral_single_dof function below.
    fn psi2(&self, j_index: usize, x_val: f64) -> f64 {
        assert!(j_index <= self.n_dof, "j_index for psi2 out of bounds");

        if x_val < self.knots[j_index] {
            return 0.0;
        }

        if (self.knots[j_index + 3] - self.knots[j_index]) < self.delta_cutoff {
            return x_val - self.knots[j_index];
        }

        // Calculate Ψ² in the segment where t_j <= x < t_{j+1}
        if x_val < self.knots[j_index + 1] {
            if self.delta_knots[j_index] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitly set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
                return 0.0;
            } else {
                return self.phi2_antiderivative_seg1(j_index, x_val);
            }
        }

        // x >= t_{j+1} so we need to calculate ∫_{t_j}^{t_{j+1}} Φ²_j(y) dy
        let integral_tj_tjp: f64 = if self.delta_knots[j_index] < self.delta_cutoff {
            // Since the distance between knots is very small the integral over the range will be approximately zero.
            // But we explicitly set integral to zero to avoid possible numerical issue of dividing by zero when sigma1_j is zero
            0.0
        } else {
            self.phi2_antiderivative_seg1_at_tjp(j_index)
        };

        // Calculate Ψ² in the segment where t_{j+1} <= x < t_{j+2}
        if x_val < self.knots[j_index + 2] {
            if self.delta_knots[j_index + 1] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitly set integral of second segment to zero (so only first segment remains) to avoid possible
                // numerical issue of dividing by zero when sigma1_j or sigma1_{j+1} is zero
                return integral_tj_tjp;
            } else {
                return integral_tj_tjp + self.phi2_antiderivative_seg2(j_index, x_val) - self.phi2_antiderivative_seg2_at_tjp(j_index);
            }
        }

        // x >= t_{j+2} so we need to calculate ∫_{t_{j+1}}^{t_{j+2}} Φ²_j(y) dy
        let integral_tjp_tjpp: f64 = if self.delta_knots[j_index + 1] < self.delta_cutoff {
            // Since the distance between knots is very small the integral over the range will be approximately zero.
            // But we explicitly set integral of second segment to zero (so only first segment remains) to avoid possible
            // numerical issue of dividing by zero when sigma1_j or sigma1_{j+1} is zero
            0.0
        } else {
            self.phi2_antiderivative_seg2_at_tjpp(j_index) - self.phi2_antiderivative_seg2_at_tjp(j_index)
        };

        // Calculate Ψ² in the segment where t_{j+2} <= x < t_{j+3}
        if x_val < self.knots[j_index + 3] {
            if self.delta_knots[j_index + 2] < self.delta_cutoff {
                // Since the distance between knots is very small the integral over the range will be approximately zero.
                // But we explicitly set integral of third segment to zero (so only first and second segments remain) to avoid possible
                // numerical issue of dividing by zero when sigma1_{j+1} is zero
                return integral_tj_tjp + integral_tjp_tjpp;
            } else {
                return integral_tj_tjp + integral_tjp_tjpp + self.phi2_antiderivative_seg3(j_index, x_val) - self.phi2_antiderivative_seg3_at_tjpp(j_index);
            }
        }

        // x >= t_{j+3} so we need to calculate ∫_{t_{j+2}}^{t_{j+3}} Φ²_j(y) dy
        let integral_tjpp_tjppp: f64 = if self.delta_knots[j_index + 2] < self.delta_cutoff {
            // Since the distance between knots is very small the integral over the range will be approximately zero.
            // But we explicitly set integral of third segment to zero (so only first and second segments remain) to avoid possible
            // numerical issue of dividing by zero when sigma1_{j+1} is zero
            0.0
        } else {
            self.phi2_antiderivative_seg3_at_tjppp(j_index) - self.phi2_antiderivative_seg3_at_tjpp(j_index)
        };

        // Calculate Ψ² in the segment where x >= t_{j+3}
        x_val - self.knots[j_index + 3] + integral_tj_tjp + integral_tjp_tjpp + integral_tjpp_tjppp
    }
}

impl SourceFunctionTraits for TensionedCubicBSpline {
    fn source_function_value_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // See equation (2.5) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993)
        let mut value: Array1<f64> = Array1::from_elem(psi_n.len(), f64::NAN);
        for i_psi_n in 0..psi_n.len() {
            let x: f64 = psi_n[i_psi_n];
            if x < self.knots[i_dof] {
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
        value
    }

    fn source_function_derivative_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("Source function is not implemented yet");
    }

    /// Integral of a single degree of freedom
    ///
    /// We take the integral from 1 to `psi_n`, this ensures that the integral is zero at `psi_n = 1`
    /// Note that the psi2 function above is defined as the integral of phi2 from -infinity to `psi_n`,
    /// which is the same as the integral of phi2 from 0 to x since phi2 is zero for x < 0.
    /// Since int_0^x + int_x^1 = int_0^1, we can rearrange this to get int_1^x = int_0^x - int_0^1.
    /// Note that int_0^1 phi2(y) dy = psi2(1) - psi2(0) = psi2(1) since psi2(0) = 0.
    ///
    /// # Arguments
    /// * `psi_n` - The points at which we want to evaluate the integral
    /// * `i_dof` - The index of the degree of freedom to evaluate
    ///
    /// # Returns
    /// An array of the same length as `psi_n` containing the integral of the source function for the specified degree of freedom
    ///
    fn source_function_integral_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        let mut value: Array1<f64> = Array1::from_elem(psi_n.len(), f64::NAN);
        let integration_constant: f64 = self.psi2(i_dof, 1.0) - self.psi2(i_dof + 1, 1.0);
        let n_psi_n: usize = psi_n.len();
        for i_psi_n in 0..n_psi_n {
            let x: f64 = psi_n[i_psi_n];
            // From equation (2.5) from P. E. Koch & T. Lyche "Interpolation with Exponential B-Splines in Tension" (1993) we have
            // int_1^x B3_j(y) dy = int_1^x phi2_j(y) dy + int_1^x phi2_{j+1}(y) dy
            //                    = int_0^x phi2_j(y) dy - int_0^1 phi2_j(y) dy - (int_0^x phi2_{j+1}(y) dy - int_0^1 phi2_{j+1}(y) dy)
            //                    = psi2_j(x) - psi2_{j+1}(x) - (psi2_j(1) - psi2_{j+1}(1))
            //                    = psi2_j(x) - psi2_{j+1}(x) - integration_constant
            value[i_psi_n] = self.psi2(i_dof, x) - self.psi2(i_dof + 1, x) - integration_constant;
        }
        value
    }

    fn source_function_value(&self, psi_n: &Array1<f64>, spline_dof: &Array1<f64>) -> Array1<f64> {
        let n_psi_n: usize = psi_n.len();
        let n_dof: usize = spline_dof.len();

        let mut value: Array1<f64> = Array1::zeros(n_psi_n);
        for i_dof in 0..n_dof {
            value = value + spline_dof[i_dof] * self.source_function_value_single_dof(psi_n, i_dof);
        }

        value
    }

    fn source_function_derivative(&self, psi_n: &Array1<f64>, spline_dof: &Array1<f64>) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("Source function is not implemented yet");
    }

    fn source_function_integral(&self, psi_n: &Array1<f64>, spline_dof: &Array1<f64>) -> Array1<f64> {
        let n_dof: usize = self.n_dof;
        let n_psi_n: usize = psi_n.len();

        let mut integral: Array1<f64> = Array1::zeros(n_psi_n);
        for i_dof in 0..n_dof {
            integral = integral + spline_dof[i_dof] * self.source_function_integral_single_dof(psi_n, i_dof);
        }

        // Alex Prok: Don't need to find the constant of integration as we take integral from 1 to psi_n
        // in source_function_integral_single_dof which ensures that the integral is zero at psi_n = 1.
        // Hence we can just return the integral calculated above without adding any constant of integration.
        // // Find the constant of integration
        // let psi_n_at_boundary: Array1<f64> = Array1::from_vec(vec![1.0]);
        // let mut integral_at_boundary: f64 = 0.0;
        // for i_dof in 0..n_dof {
        //     integral_at_boundary += spline_dof[i_dof] * self.source_function_integral_single_dof(&psi_n_at_boundary, i_dof)[0];
        // }

        integral
    }

    fn source_function_regularisation(&self) -> Array2<f64> {
        let regularisations: Array2<f64> = self.regularisations.clone();

        regularisations
    }

    fn source_function_n_dof(&self) -> usize {
        let n_dof: usize = self.n_dof;

        n_dof
    }
}

/// Test that source_function_integral is consistent with numerical integration of source_function_value
/// using the trapezoidal rule, and that it returns zero at psi_n = 1.0.
#[test]
fn test_source_function_integral() {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};

    // Setup with 2 interior knots
    //
    //   n_knots             = n_interior_knots + 8 = 10
    //   n_interval_tensions = n_interior_knots + 1 = 3
    //   n_dof               = n_interior_knots + 4 = 6
    //
    //   Full knot vector:   [0, 0, 0, 0,   0.4,   0.7,   1, 1, 1, 1]
    //                        ╰─clamped─╯                 ╰─clamped─╯
    //
    //   psi_n values:   [0.0,0.4] [0.4,0.7] [0.7,1]
    //                       ↓         ↓         ↓
    //   Tension:         [ 1.0,      1.0,      1.0 ]
    //
    //   Spline DOFs:     [1.234, 2.345,   3.456, 4.567,   5.678, 6.789]
    //                    ╰─endpoints─╯  ╰─interior DOFs─╯ ╰─endpoints─╯
    //
    // The first two and last two DOFs control behaviour at the boundaries (psi_n = 0, 1).

    // Setup with 2 interior knots
    #[rustfmt::skip]
    //                      full_knot_vector = [  0.0,   0.0,         0.4,         0.7,         1.0,   1.0]
    let interior_knots: Array1<f64>    = array![                      0.4,         0.7                    ];
    #[rustfmt::skip]
    //                     tension_intervals =               (0.0→0.4)    (0.4→0.7)    (0.7→1.0)
    let interval_tensions: Array1<f64> = array![                1.0,         1.0,          1.0            ];
    #[rustfmt::skip]
    //                                 psi_n = [  0.0,   0.0,         0.4,         0.7,         1.0,   1.0]
    let spline_dof: Array1<f64>        = array![1.234, 2.345,       3.456,       4.567,       5.678, 6.789];
    //                                          ╰─endpoints─╯       ╰──interior DOFs──╯       ╰─endpoints─╯
    // The first two and last two DOFs control behaviour at the boundaries (psi_n = 0, 1).

    let n_dof: usize = interior_knots.len() + 4;

    // `regularisations` is used within `gs_solution` to calculate `spline_dof` values.
    // Since we are supplying our own `spline_dof` values, `regularisations` has no effect, so can be set to NaN
    let regularisations: Array2<f64> = Array2::from_elem((n_dof, n_dof), f64::NAN);

    let spline: TensionedCubicBSpline = TensionedCubicBSpline::new(regularisations, interior_knots, interval_tensions);

    // Test 1: `source_function_integral` should be zero at `psi_n = 1.0`
    let psi_n_boundary: Array1<f64> = array![1.0];
    let integral_at_boundary: Array1<f64> = spline.source_function_integral(&psi_n_boundary, &spline_dof);
    assert_abs_diff_eq!(integral_at_boundary[0], 0.0, epsilon = 1e-12);

    // Test 2: Verify that the numerical derivative of `source_function_integral` equals `source_function_value`
    // using finite differences: d/dx integral(x) ≈ (integral(x + h) - integral(x - h)) / (2h)
    let n_test: usize = 50;
    let psi_n_test: Array1<f64> = Array1::linspace(0.05, 0.95, n_test);
    let value: Array1<f64> = spline.source_function_value(&psi_n_test, &spline_dof);

    let delta_psi_n: f64 = 1e-7;
    let mut psi_n_plus: Array1<f64> = Array1::from_elem(1, f64::NAN);
    let mut psi_n_minus: Array1<f64> = Array1::from_elem(1, f64::NAN);
    for i_test in 0..n_test {
        psi_n_plus[0] = psi_n_test[i_test] + delta_psi_n;
        psi_n_minus[0] = psi_n_test[i_test] - delta_psi_n;
        let integral_plus: Array1<f64> = spline.source_function_integral(&psi_n_plus, &spline_dof);
        let integral_minus: Array1<f64> = spline.source_function_integral(&psi_n_minus, &spline_dof);
        let numerical_derivative: f64 = (integral_plus[0] - integral_minus[0]) / (2.0 * delta_psi_n);
        assert_abs_diff_eq!(numerical_derivative, value[i_test], epsilon = 1e-5);
    }

    // Test 3: Verify that `source_function_integral` matches numerical trapezoidal integration of `source_function_value`
    // Integrate from 1.0 to psi_n using the trapezoidal rule with a fine grid
    let n_trap: usize = 100_000;
    let psi_n_eval: f64 = 0.3;
    let trap_grid: Array1<f64> = Array1::linspace(psi_n_eval, 1.0, n_trap);
    let trap_values: Array1<f64> = spline.source_function_value(&trap_grid, &spline_dof);
    let d_psi: f64 = (1.0 - psi_n_eval) / (n_trap as f64 - 1.0);
    let mut trap_integral: f64 = 0.0;
    for i_trap in 0..n_trap - 1 {
        trap_integral += 0.5 * (trap_values[i_trap] + trap_values[i_trap + 1]) * d_psi;
    }
    // `source_function_integral` integrates from 1 to psi_n, so the sign is flipped relative to integrating from psi_n to 1
    let trap_integral: f64 = -trap_integral;

    let psi_n_single: Array1<f64> = array![psi_n_eval];
    let analytic_integral: Array1<f64> = spline.source_function_integral(&psi_n_single, &spline_dof);
    assert_abs_diff_eq!(analytic_integral[0], trap_integral, epsilon = 1e-6);
}
