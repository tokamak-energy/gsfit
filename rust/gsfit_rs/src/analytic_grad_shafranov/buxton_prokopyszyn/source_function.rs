//! The source functions of the analytic equilibrium, for the equilibrium post-processor.
//!
//! The analytic `p'` and `ff'` are GSFit's EFIT polynomials plus a constant term, `c_0 + sum_{i >= 1} c_i (1 - psi_n) psi_n**(i - 1)`
//! (`c_0` the constant, Solov'ev's term; `c_1` the coefficient of `1 - psi_n`, `c_2` that of `(1 - psi_n) psi_n`, ...). GSFit's
//! `EfitPolynomial` has no constant term, so it is used only when both constants are zero, with the DOFs `[c_1, ..., c_n]`;
//! otherwise `EfitPolynomialWithConstant` is, with the DOFs `[c_1, ..., c_n, c_0]` (the constant last). Neither is fitted: the
//! post-processor evaluates them with the coefficients the IDS holds.

use crate::source_functions::{EfitPolynomial, SharedSourceFunction, SourceFunctionTraits};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// `c_0 + sum_{i >= 1} c_i (1 - psi_n) psi_n**(i - 1)`: GSFit's EFIT basis, `n_efit` terms (`c_1..c_n`), then the constant `c_0`, its
/// last DOF
pub struct EfitPolynomialWithConstant {
    /// The number of EFIT terms; the constant `c_0` is DOF `n_efit` [dimensionless]
    pub n_efit: usize,
}

impl SourceFunctionTraits for EfitPolynomialWithConstant {
    fn source_function_value_single_dof(&self, psi_norm: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        if i_dof == self.n_efit {
            return Array1::ones(psi_norm.len());
        }
        psi_norm.mapv(|value: f64| (1.0 - value) * value.powi(i_dof as i32))
    }

    fn source_function_derivative_single_dof(&self, _psi_norm: &Array1<f64>, _i_dof: usize) -> Array1<f64> {
        unimplemented!("EfitPolynomialWithConstant: the derivative is not implemented, as for EfitPolynomial");
    }

    /// The integral from 1 to `psi_norm`, zero at `psi_norm = 1`: `psi_n - 1` for the constant, and as `EfitPolynomial`'s for the
    /// EFIT terms
    fn source_function_integral_single_dof(&self, psi_norm: &Array1<f64>, i_dof: usize) -> Array1<f64> {
        if i_dof == self.n_efit {
            return psi_norm.mapv(|value: f64| value - 1.0);
        }
        let i_dof_f64: f64 = i_dof as f64;
        let integration_constant: f64 = -1.0 / (i_dof_f64 + 1.0) + 1.0 / (i_dof_f64 + 2.0);
        psi_norm.mapv(|value: f64| value.powi(i_dof as i32 + 1) / (i_dof_f64 + 1.0) - value.powi(i_dof as i32 + 2) / (i_dof_f64 + 2.0) + integration_constant)
    }

    fn source_function_value(&self, psi_norm: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        let n_dof: usize = polynomial_dof.len();
        let mut value: Array1<f64> = Array1::zeros(psi_norm.len());
        for i_dof in 0..n_dof {
            value = value + polynomial_dof[i_dof] * self.source_function_value_single_dof(psi_norm, i_dof);
        }
        value
    }

    fn source_function_derivative(&self, _psi_norm: &Array1<f64>, _polynomial_dof: &Array1<f64>) -> Array1<f64> {
        unimplemented!("EfitPolynomialWithConstant: the derivative is not implemented, as for EfitPolynomial");
    }

    fn source_function_integral(&self, psi_norm: &Array1<f64>, polynomial_dof: &Array1<f64>) -> Array1<f64> {
        let n_dof: usize = polynomial_dof.len();
        let mut integral: Array1<f64> = Array1::zeros(psi_norm.len());
        for i_dof in 0..n_dof {
            integral = integral + polynomial_dof[i_dof] * self.source_function_integral_single_dof(psi_norm, i_dof);
        }
        integral
    }

    fn source_function_regularisation(&self) -> Array2<f64> {
        Array2::zeros((0, 0))
    }

    /// Nothing is fitted
    fn source_function_n_dof(&self) -> usize {
        0
    }
}

/// The source function for one profile, and the DOFs the IDS's `source_functions/<profile>/coefficients` holds for it: GSFit's layout,
/// `[c_1, ..., c_n]`, with `c_0` appended when either profile has a constant term.
///
/// # Arguments
/// * `coefficients` - `[c_0, c_1, ..., c_n]`, `n >= 1` [pascal / weber or tesla ** 2 * metre ** 2 / weber]
/// * `is_with_constant` - use `EfitPolynomialWithConstant` (when either profile has a constant term, so that both profiles' DOFs have
///   the same layout)
pub fn source_function_and_dofs(coefficients: &[f64], is_with_constant: bool) -> (SharedSourceFunction, Array1<f64>) {
    let efit_coefficients: &[f64] = &coefficients[1..];
    let n_efit: usize = efit_coefficients.len();
    if !is_with_constant {
        let source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
            n_dof: n_efit,
            regularisations: Array2::zeros((0, n_efit)),
            dof_values: Array1::zeros(0),
            exact: false,
        });
        return (source_function, Array1::from(efit_coefficients.to_vec()));
    }
    let mut dofs: Vec<f64> = efit_coefficients.to_vec();
    dofs.push(coefficients[0]);
    let source_function: SharedSourceFunction = Arc::new(EfitPolynomialWithConstant { n_efit });
    (source_function, Array1::from(dofs))
}
