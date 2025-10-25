use core::f64;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct BoundaryContour {
    pub boundary_r: Array1<f64>,
    pub boundary_z: Array1<f64>,
    pub n_points: usize,
    pub bounding_psi: f64,
    pub bounding_r: f64,
    pub bounding_z: f64,
    pub fraction_inside_vessel: f64,
    pub xpt_diverted: bool,
    pub mask: Option<Array2<f64>>,
    pub secondary_xpt_r: f64,
    pub secondary_xpt_z: f64,
    pub secondary_xpt_distance: f64,
}
