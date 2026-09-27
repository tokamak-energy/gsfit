// Analytic solutions to the Grad–Shafranov equation.
//
// These generate self-consistent, free-boundary equilibria (flux + external PF-coil
// currents) that GSFit can be asked to reconstruct.

// Load submodules
mod guazzotto_freidberg;

// Public flattened exports
pub use guazzotto_freidberg::{Configuration, GuazzottoFreidberg, Symmetry};
