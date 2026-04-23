use super::bicubic_interpolator::BicubicInterpolator;
use core::f64;
use std::f64::consts::PI;

/// Computes the topological winding number
///
/// # Arguments
/// * `bicubic_interpolator` - the bicubic interpolator for the cell
///
/// # Returns
/// * `winding_number` - the winding number, [dimensionless]:
///   * `winding_number = 1` indicates a maximum or minimum
///   * `winding_number = -1` indicates a saddle point
///   * `winding_number = 0` indicates no stationary point in this cell
///
/// # Algorithm
/// Samples the gradient vector at 8 evenly spaced points along the 4 edges.
/// The total signed angle change of the gradient vector is accumulated,
/// and divided by 2π to give the winding number.
/// Note: we use bicubic interpolation (not bilinear) because bilinear interpolation
/// of the gradient along each edge is equivalent to the sign-change method — and won't
/// detect a case where a contour enters and exits through the same edge.
pub fn calculate_winding_number(bicubic_interpolator: &BicubicInterpolator) -> i64 {
    let n_samples_per_edge: usize = 8;
    let mut winding_angle: f64 = 0.0;
    let mut first_g_x: f64 = f64::NAN;
    let mut first_g_y: f64 = f64::NAN;
    let mut prev_g_x: f64 = f64::NAN;
    let mut prev_g_y: f64 = f64::NAN;
    let mut is_first_sample: bool = true;

    // Edges traversed counter-clockwise: (start_x, start_y) -> (end_x, end_y)
    let edges: [(f64, f64, f64, f64); 4] = [
        (0.0, 0.0, 1.0, 0.0), // (bottom, left) to (bottom, right)
        (1.0, 0.0, 1.0, 1.0), // (right, bottom) to (right, top)
        (1.0, 1.0, 0.0, 1.0), // (top, right) to (top, left)
        (0.0, 1.0, 0.0, 0.0), // (left, top) to (left, bottom)
    ];

    for &(start_x, start_y, end_x, end_y) in &edges {
        for i_sample in 0..n_samples_per_edge {
            let t: f64 = i_sample as f64 / n_samples_per_edge as f64;
            let sample_x: f64 = start_x + t * (end_x - start_x);
            let sample_y: f64 = start_y + t * (end_y - start_y);

            let sample = bicubic_interpolator.value_and_derivatives(sample_x, sample_y);
            let cur_g_x: f64 = sample.d_f_d_x;
            let cur_g_y: f64 = sample.d_f_d_y;

            if is_first_sample {
                first_g_x = cur_g_x;
                first_g_y = cur_g_y;
                is_first_sample = false;
            } else {
                // Signed angle change using atan2 of the cross and dot products
                let cross: f64 = prev_g_x * cur_g_y - prev_g_y * cur_g_x;
                let dot: f64 = prev_g_x * cur_g_x + prev_g_y * cur_g_y;
                winding_angle += cross.atan2(dot);
            }

            prev_g_x = cur_g_x;
            prev_g_y = cur_g_y;
        }
    }

    // Close the loop: angle change from last sample back to first sample
    let cross: f64 = prev_g_x * first_g_y - prev_g_y * first_g_x;
    let dot: f64 = prev_g_x * first_g_x + prev_g_y * first_g_y;
    winding_angle += cross.atan2(dot);

    let winding_number: f64 = winding_angle / (2.0 * PI);

    winding_number.round() as i64
}
