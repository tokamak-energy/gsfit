use crate::source_functions::SourceFunctionTraits;
use ndarray::{Array1, Array2};
use numpy::PyArray2;
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct EfitPolynomial {
    pub n_dof: usize,
    pub regularisations: Array2<f64>,
}

/// Python accessible methods
#[pymethods]
impl EfitPolynomial {
    #[new]
    pub fn new(n_dof: usize, regularisations: &Bound<'_, PyArray2<f64>>) -> Self {
        // Change Python types into Rust types
        let regularisations_ndarray: Array2<f64> = Array2::from(unsafe { regularisations.as_array() }.to_owned());

        // Create the struct
        EfitPolynomial {
            n_dof,
            regularisations: regularisations_ndarray,
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

impl SourceFunctionTraits for EfitPolynomial {
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
