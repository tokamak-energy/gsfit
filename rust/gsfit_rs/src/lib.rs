// Load for use in this file
use pyo3::prelude::*;

// Load modules
mod analytic_grad_shafranov;
mod circuit_equations;
mod equilibrium;
mod equilibrium_post_processor;
mod grad_shafranov;
mod limiter;
mod magnetic_sensors;
mod magnetics;
mod pf_active;
mod pf_passive;
mod pulse_schedule;
mod sensors;
mod source_functions;
mod tf;
mod wall;

// Load structs and functions
use analytic_grad_shafranov::{Configuration, GuazzottoFreidberg, Symmetry};
use circuit_equations::solve_circuit_equations;
use equilibrium::Equilibrium;
use grad_shafranov::solve_grad_shafranov;
use greens::{
    greens_d_psi_d_r, greens_d_psi_d_z, greens_d2_psi_d_r_d_z, greens_d2_psi_d_r2, greens_d2_psi_d_z2, greens_d3_psi_d_r_d_z2, greens_d3_psi_d_r2_d_z,
    greens_d3_psi_d_z3, greens_py,
};
use magnetics::Magnetics;
mod material_properties;
use pf_active::PfActive;
use pf_passive::PfPassive;
use pulse_schedule::PulseSchedule;
mod python_pickling_methods;
use sensors::{Isoflux, IsofluxBoundary, Pressure, StationaryPoint};
use source_functions::{EfitPolynomial, TensionedCubicBSpline};
use tf::Tf;
use wall::Wall;

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
// mod equilibrium_post_processor;

/// A Python module implemented in Rust; bindings added here
#[pymodule]
fn gsfit_rs(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<Equilibrium>()?;
    m.add_class::<PfActive>()?;
    m.add_class::<PfPassive>()?;
    m.add_class::<Tf>()?;
    m.add_class::<Wall>()?;

    // Expose machine operation classes
    m.add_class::<PulseSchedule>()?;

    // Expose sensor classes
    m.add_class::<Isoflux>()?;
    m.add_class::<IsofluxBoundary>()?;
    m.add_class::<Magnetics>()?;
    m.add_class::<StationaryPoint>()?;
    m.add_class::<Pressure>()?;

    // Expose source functions
    m.add_class::<EfitPolynomial>()?;
    m.add_class::<TensionedCubicBSpline>()?;

    // Expose analytic equilibria
    m.add_class::<GuazzottoFreidberg>()?;
    m.add_class::<Symmetry>()?;
    m.add_class::<Configuration>()?;

    // Expose solovev equilibrium function
    // m.add_function(wrap_pyfunction!(run_solovev, m)?)?;

    // Expose the IMAS data dictionary as the `gsfit_rs.imas` submodule.
    // The classes themselves live in the `imas_rs` crate (behind its `python` feature);
    // `gsfit_rs` only mounts them, because it is the crate that builds the `cdylib`.
    let imas: Bound<'_, PyModule> = PyModule::new(py, "imas")?;
    imas_rs::python::register(&imas)?;
    m.add_submodule(&imas)?;
    // `add_submodule` only sets the attribute; registering in `sys.modules` is what makes
    // `from gsfit_rs.imas import equilibrium_paths` work.
    py.import("sys")?.getattr("modules")?.set_item("gsfit_rs.imas", &imas)?;

    // Expose the data dictionary structures which are not IDSs, e.g. `ParametersInput`, as the
    // `gsfit_rs.dd_support` submodule, in the same way
    let dd_support: Bound<'_, PyModule> = PyModule::new(py, "dd_support")?;
    imas_rs::python::register_dd_support(&dd_support)?;
    m.add_submodule(&dd_support)?;
    py.import("sys")?.getattr("modules")?.set_item("gsfit_rs.dd_support", &dd_support)?;

    Ok(())
}
