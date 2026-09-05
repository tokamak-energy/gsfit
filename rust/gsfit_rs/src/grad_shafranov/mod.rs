// Load modules
mod epp_chi_sq_mag;
mod equilibrium_solve;
mod grad_shafranov_solver;
mod gs_solution;

// Expose functions to public
pub use equilibrium_solve::{GradShafranovInputs, GradShafranovSolve, output_flag};
pub use grad_shafranov_solver::solve_grad_shafranov;
pub use gs_solution::GsSolution;

// Define the possible **external** failures this module can produce
#[derive(Debug)]
pub enum Error {
    InvalidInitialCurrent(String),
    NoBoundaryFound { no_xpt_reason: String, no_limit_point_reason: String },
    NoMagneticAxisFound,
    MaxIterReached,
    NoStationaryPointsFound,
}
