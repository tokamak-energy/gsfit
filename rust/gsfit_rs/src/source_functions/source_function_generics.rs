use super::efit_polynomial::EfitPolynomial;
use super::tensioned_cubic_b_spline::TensionedCubicBSpline;
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use std::sync::Arc;

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

/// Owned, thread-safe handle to any source function implementation
pub type SharedSourceFunction = Arc<dyn SourceFunctionTraits + Send + Sync>;

/// Convert a Python source function object (e.g. `EfitPolynomial`, `TensionedCubicBSpline`)
/// into an owned `SharedSourceFunction`.
///
/// Every concrete source function type derives `Clone`, so we simply clone the whole
/// object.
///
/// # Arguments
/// * `obj` - the Python source function object (a GIL-bound reference)
///
/// # Returns
/// * `SharedSourceFunction` - an owned trait object
pub fn extract_source_function(obj: &Bound<'_, PyAny>) -> SharedSourceFunction {
    if let Ok(efit) = obj.extract::<PyRef<EfitPolynomial>>() {
        Arc::new(EfitPolynomial::clone(&efit))
    } else if let Ok(cubic_bspline) = obj.extract::<PyRef<TensionedCubicBSpline>>() {
        Arc::new(TensionedCubicBSpline::clone(&cubic_bspline))
    } else {
        panic!("source function must implement SourceFunctionTraits");
    }
}
