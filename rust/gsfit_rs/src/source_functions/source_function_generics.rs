use ndarray::{Array1, Array2};

pub trait SourceFunctionTraits {
    fn source_function_value_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_derivative_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_integral_single_dof(&self, psi_n: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_value(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    fn source_function_derivative(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    fn source_function_integral(&self, psi_n: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    fn source_function_regularisation(&self) -> Array2<f64>;
    fn source_function_n_dof(&self) -> usize;
}
