// Load for use in this file
use pyo3::prelude::*;

// Load modules
mod circuit_equations;
mod coils;
mod grad_shafranov;
mod passives;
mod plasma;
mod sensors;
mod source_functions;

// Load structs and functions
use circuit_equations::solve_circuit_equations;
use coils::Coils;
use grad_shafranov::solve_grad_shafranov;
use greens::{
    greens_d_psi_d_r, greens_d_psi_d_z, greens_d2_psi_d_r_d_z, greens_d2_psi_d_r2, greens_d2_psi_d_z2, greens_d3_psi_d_r_d_z2, greens_d3_psi_d_r2_d_z,
    greens_d3_psi_d_z3, greens_py,
};
mod material_properties;
use passives::Passives;
use plasma::Plasma;
mod python_pickling_methods;
use sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils, StationaryPoint};
use source_functions::{EfitPolynomial, TensionedCubicBSpline};

// Load public modules
pub mod greens;

// Load public structs and functions for Doctests
// Doctests are compiled as their own tiny standalone crate linked against `gsfit_rs` exactly like an external user would.
// We must therefore expose all structures and functions which have an example.
// But we mark them as "hidden", as they are not the intended public API (note, they are still technically public).
#[doc(hidden)]
pub mod plasma_geometry;

// Future modules
// mod solovev_equilibrium;
// pub use solovev_equilibrium::run_solovev;
// mod analytic_grad_shafranov;
// mod equilibrium_post_processor;

/// A Python module implemented in Rust; bindings added here
#[pymodule]
fn gsfit_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose functions
    m.add_function(wrap_pyfunction!(greens_py, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d_psi_d_r, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d_psi_d_z, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d2_psi_d_r2, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d2_psi_d_r_d_z, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d2_psi_d_z2, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d3_psi_d_r_d_z2, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d3_psi_d_r2_d_z, m)?)?;
    m.add_function(wrap_pyfunction!(greens_d3_psi_d_z3, m)?)?;
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
    m.add_class::<TensionedCubicBSpline>()?;

    // Expose solovev equilibrium function
    // m.add_function(wrap_pyfunction!(run_solovev, m)?)?;

    Ok(())
}
