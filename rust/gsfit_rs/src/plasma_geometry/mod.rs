// External crates
use ndarray::Array1;

// Private modules
mod boundary_contour;
mod calculate_winding_number;
mod find_boundary;
mod find_magnetic_axis;
mod find_stationary_points_using_sign_differences;
mod find_stationary_points_using_winding_number;
mod find_viable_limit_point;
mod find_viable_xpt;
mod flood_fill_mask;
mod hessian;

// Public modules
pub mod bicubic_interpolator;
pub mod cubic_interpolation;
pub mod marching_squares;
pub mod marching_squares_for_sol;

// Public flattened exports
pub use boundary_contour::BoundaryContour;
pub use find_boundary::find_boundary;
pub use find_magnetic_axis::MagneticAxis;
pub use find_magnetic_axis::find_magnetic_axis;
pub use find_stationary_points_using_sign_differences::find_stationary_points_using_sign_differences;
pub use find_stationary_points_using_winding_number::find_stationary_points_using_winding_number;
pub use find_viable_xpt::find_viable_xpt;
pub use flood_fill_mask::flood_fill_mask;
pub use hessian::hessian;

// Define the possible **external** failures this module can produce
#[derive(Debug)]
pub enum Error {
    NoBoundaryFound { no_xpt_reason: String, no_limit_point_reason: String },
}

// Define data structures
#[derive(Debug, Clone)]
pub struct MarchingContour {
    pub r: Array1<f64>,
    pub z: Array1<f64>,
    pub n: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StationaryPoint {
    pub r: f64,
    pub z: f64,
    pub psi: f64,
    pub hessian_determinant: f64,
    pub hessian_trace: f64,
    pub i_r_nearest: usize,
    pub i_z_nearest: usize,
    pub i_r_left: usize,
    pub i_r_right: usize,
    pub i_z_lower: usize,
    pub i_z_upper: usize,
}
