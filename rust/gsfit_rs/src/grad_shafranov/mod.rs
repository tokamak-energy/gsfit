// Load modules
mod chi_sq_mag;
mod equilibrium_solve;
mod grad_shafranov_solver;
mod initial_current_seed;
mod nonlinear_solvers;

// Expose functions to public
pub(crate) use equilibrium_solve::CONVERGENCE_STATUS_CONVERGED;
pub(crate) use equilibrium_solve::contour_tree_nodes;
pub use equilibrium_solve::output_flag;
pub use grad_shafranov_solver::solve_grad_shafranov;

// Define the possible **external** failures this module can produce
#[derive(Debug)]
pub enum Error {
    InvalidInitialCurrent(String),
    NoBoundaryFound { no_xpt_reason: String, no_limit_point_reason: String },
    NoMagneticAxisFound,
    MaxIterReached,
    NoStationaryPointsFound,
}
