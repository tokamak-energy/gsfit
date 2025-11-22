use pyo3::prelude::*;
mod sensors;
pub use sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils, StationaryPoint};
pub use sensors::{SensorsDynamic, SensorsStatic};
mod grad_shafranov;
pub mod plasma_geometry;
pub use grad_shafranov::GsSolution;
pub mod greens;
mod plasma;
pub use plasma::Plasma;
mod passives;
pub use passives::Passives;
mod coils;
pub use coils::Coils;
mod circuit_equations;
use circuit_equations::solve_circuit_equations;
mod source_functions;
pub use grad_shafranov::solve_grad_shafranov;
use greens::greens_py;
use source_functions::{EfitPolynomial, LiuqePolynomial};

/// A Python module implemented in Rust; bindings added here
#[pymodule]
fn gsfit_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose functions
    m.add_function(wrap_pyfunction!(greens_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_grad_shafranov, m)?)?;
    m.add_function(wrap_pyfunction!(solve_circuit_equations, m)?)?;

    // Expose current source classes
    m.add_class::<Coils>()?;
    m.add_class::<Passives>()?;
    m.add_class::<Plasma>()?;

    // Expose sensor classes
    m.add_class::<BpProbes>()?;
    m.add_class::<Dialoop>()?;
    m.add_class::<FluxLoops>()?;
    m.add_class::<Isoflux>()?;
    m.add_class::<IsofluxBoundary>()?;
    m.add_class::<StationaryPoint>()?;
    m.add_class::<RogowskiCoils>()?;
    m.add_class::<Pressure>()?;

    // Expose source functions
    m.add_class::<EfitPolynomial>()?;
    m.add_class::<LiuqePolynomial>()?;

    Ok(())
}
