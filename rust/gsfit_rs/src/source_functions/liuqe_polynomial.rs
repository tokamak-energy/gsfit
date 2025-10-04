use crate::source_functions::SourceFunctionTraits;
use ndarray::{Array1, Array2};
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct LiuqePolynomial {
    pub n_dof: usize,
    pub regularisations: Array2<f64>,
}

/// Python accessible methods
#[pymethods]
impl LiuqePolynomial {
    #[new]
    pub fn new(n_dof: usize, regularisations: &Bound<'_, PyArray2<f64>>) -> Self {
        let regularisations_ndarray: Array2<f64> = Array2::from(unsafe { regularisations.as_array() }.to_owned());

        // Create the struct
        LiuqePolynomial {
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

impl SourceFunctionTraits for LiuqePolynomial {
    fn source_function_value_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_derivative_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_integral_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_value(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_derivative(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_integral(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        // This function is not implemented yet
        unimplemented!("liuqe not implemented yet");
    }

    fn source_function_regularisation(&self) -> Array2<f64> {
        return self.regularisations.clone();
    }

    fn source_function_n_dof(&self) -> usize {
        return self.n_dof;
    }
}
