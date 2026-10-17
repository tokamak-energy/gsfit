use ndarray::{Array1, Array2};
use std::sync::Arc;

pub trait SourceFunctionTraits {
    fn source_function_value_single_dof(&self, psi_norm: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_derivative_single_dof(&self, psi_norm: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_integral_single_dof(&self, psi_norm: &Array1<f64>, i_dof: usize) -> Array1<f64>;
    fn source_function_value(&self, psi_norm: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    fn source_function_derivative(&self, psi_norm: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    fn source_function_integral(&self, psi_norm: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64>;
    /// shape = [n_regularisation, n_dof]
    fn source_function_regularisation(&self) -> Array2<f64>;
    /// The number of degrees of freedom: the coefficients which the solver fits. This is 0 for an
    /// "exact" source function, whose coefficients are all known
    fn source_function_n_dof(&self) -> usize;

    /// The fixed coefficients of an "exact" source function, which the solver uses rather than
    /// fitting them to the constraints. This is what allows a forward solve, where `p_prime` and
    /// `ff_prime` are known, and so have `n_dof = 0`. Their shapes are then fixed, but they share
    /// one fitted amplitude if there is a constraint for it, such as the plasma current; see
    /// `EquilibriumSolver::solve`.
    ///
    /// `None`, the default, means the coefficients are fitted.
    fn source_function_exact_dof_values(&self) -> Option<Array1<f64>> {
        None
    }
}

/// Owned, thread-safe handle to any source function implementation
pub type SharedSourceFunction = Arc<dyn SourceFunctionTraits + Send + Sync>;
