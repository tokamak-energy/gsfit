// Private modules
mod boundary_contour;
mod find_boundary;
mod find_magnetic_axis;
mod find_stationary_points;
mod find_viable_limit_point;
mod find_viable_xpt;
mod flood_fill_mask;
mod hessian;

// Public modules
pub mod bicubic_interpolator;
pub mod cubic_interpolation;
pub mod marching_squares;

// Public flattened exports
pub use boundary_contour::BoundaryContour;
pub use find_boundary::find_boundary;
pub use find_magnetic_axis::MagneticAxis;
pub use find_magnetic_axis::find_magnetic_axis;
pub use find_stationary_points::StationaryPoint;
pub use find_stationary_points::find_stationary_points;
pub use find_viable_xpt::find_viable_xpt;
pub use flood_fill_mask::flood_fill_mask;
pub use hessian::hessian;

// Define the possible **external** failures this module can produce
#[derive(Debug)]
pub enum Error {
    NoBoundaryFound { no_xpt_reason: String, no_limit_point_reason: String },
}
