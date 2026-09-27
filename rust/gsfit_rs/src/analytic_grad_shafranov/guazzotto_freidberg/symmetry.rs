use pyo3::prelude::*;

/// Up–down symmetry of the equilibrium configuration.
///
/// Selects the flux-function expansion (GF Eq. (3.2) vs (5.8)):
/// * `Symmetric` — 7-term `cos(h_n y)` series (limiter or double-null),
/// * `Asymmetric` — adds the `sin(h_n y)` family for the 12-term single-null solve.
#[pyclass(module = "gsfit_rs.analytic_grad_shafranov.guazzotto_freidberg", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    Symmetric,
    Asymmetric,
}
