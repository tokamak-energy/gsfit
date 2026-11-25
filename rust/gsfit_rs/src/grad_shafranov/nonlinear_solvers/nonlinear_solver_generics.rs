use crate::grad_shafranov::equilibrium_solve::EquilibriumSolver;
use std::sync::Arc;

pub trait NonlinearSolverTraits {
    /// Solve the Grad-Shafranov equation at the time-slice `solver` holds, making its flux and its current density consistent with
    /// each other. The time-slice is left either converged, by `set_to_converged_time_slice`, or failed, by `set_to_failed_time_slice`
    fn solve(&self, solver: &mut EquilibriumSolver);
}

/// Owned, thread-safe handle to any nonlinear solver implementation
pub type SharedNonlinearSolver = Arc<dyn NonlinearSolverTraits + Send + Sync>;
