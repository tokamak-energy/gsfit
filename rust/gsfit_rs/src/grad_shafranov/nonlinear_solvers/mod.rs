//! The nonlinear solvers, which make the flux and the current density consistent with each other.
//!
//! All are built on the two halves of a Picard iteration, which are methods on `EquilibriumSolver`:
//! * `update_psi_and_geometry`: the flux from the current density, and from the flux the magnetic
//!   axis, the plasma boundary and `psi_norm`
//! * `fit_and_update_current`: the fit of the source functions, passive currents and `delta_z` to
//!   the constraints, and from it the new current density
//!
//! Which one is used is chosen by `code/numerics/nonlinear_solver/method`; see
//! `EquilibriumSolver::solve`.

pub(super) mod newton_krylov;
pub(super) mod newton_picard;
pub(super) mod picard;
