//! The nonlinear solvers, which make the flux and the current density consistent with each other.
//!
//! All are built on the two halves of a Picard iteration, which are methods on `EquilibriumSolver`:
//! * `update_psi_and_geometry`: the flux from the current density, and from the flux the magnetic
//!   axis, the plasma boundary and `psi_norm`
//! * `fit_and_update_current`: the fit of the source functions, passive currents and `delta_z` to
//!   the constraints, and from it the new current density
//!
//! Which one is used is chosen by `equilibrium_settings/numerics/nonlinear_solver/method`. `Equilibrium::new` makes it, with its
//! settings, and `EquilibriumSolver` only ever calls it through `NonlinearSolverTraits`.

// Load modules
mod newton_krylov;
mod newton_picard;
mod nonlinear_solver_generics;
mod picard;

// Expose to the crate
pub(crate) use newton_krylov::NewtonKrylov;
pub(crate) use newton_picard::NewtonPicard;
pub(crate) use nonlinear_solver_generics::{NonlinearSolverTraits, SharedNonlinearSolver};
pub(crate) use picard::Picard;
