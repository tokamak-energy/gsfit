use super::Symmetry;
use pyo3::prelude::*;

/// Plasma boundary topology and the shape parameters it requires.
///
/// Each variant carries exactly its own shape inputs, so illegal combinations are
/// unrepresentable. The up–down symmetry (7- vs 12-term expansion) is implied by the
/// variant and can be read via [`Configuration::symmetry`].
///
/// References are to L. Guazzotto & J. P. Freidberg, J. Plasma Phys. 87 (2021) 905870303.
#[pyclass(module = "gsfit_rs.analytic_grad_shafranov.guazzotto_freidberg", from_py_object)]
#[derive(Clone, Debug)]
pub enum Configuration {
    /// Up–down symmetric smooth limiter (GF §3): 7-term `cos(h_n y)` series.
    SymmetricLimited { kappa: f64, delta: f64 },
    /// Up–down asymmetric smooth limiter (GF §7, "different upper and lower shapes").
    /// NOT YET IMPLEMENTED.
    AntisymmetricLimited {
        kappa_upper: f64,
        delta_upper: f64,
        kappa_lower: f64,
        delta_lower: f64,
    },
    /// Up–down symmetric double-null divertor (GF §4): 7-term series with an X-point.
    DoubleNull { kappa_x: f64, delta_x: f64 },
    /// Up–down asymmetric lower single-null divertor (GF §5): 12-term series. The upper half is
    /// a smooth Miller profile (`kappa`, `delta`) and the lower half has the X-point (`kappa_x`, `delta_x`).
    SingleNull { kappa: f64, delta: f64, kappa_x: f64, delta_x: f64 },
}

impl Configuration {
    /// Up–down symmetry implied by the topology (selects the 7- vs 12-term expansion).
    pub fn symmetry(&self) -> Symmetry {
        match self {
            Configuration::SymmetricLimited { .. } | Configuration::DoubleNull { .. } => Symmetry::Symmetric,
            Configuration::AntisymmetricLimited { .. } | Configuration::SingleNull { .. } => Symmetry::Asymmetric,
        }
    }
}
