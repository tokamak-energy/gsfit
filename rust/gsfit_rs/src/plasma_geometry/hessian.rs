/// Calculates the Hessian determinant for the poloidal flux function `psi`:
/// * det(Hessian(R, Z)) > 0: Both curvatures have the same sign ⇒ turning point (minimum or maximum)
///     * trace > 0: Minimum
///     * trace < 0: Maximum
/// * det(Hessian(R, Z)) < 0: Curvatures have opposite signs ⇒ saddle point
/// * det(Hessian(R, Z)) ≈ 0: Degenerate case ⇒ function could be flat (can be saddle point, local minimum, or local maximum)
///     * trace can be any value
///
/// # Arguments
/// * `d2_psi_d_r2` - Second derivative of poloidal flux `psi` with respect to R, [weber**2 / metre**2]
/// * `d2_psi_d_z2` - Second derivative of poloidal flux `psi` with respect to Z, [weber**2 / metre**2]
/// * `d2_psi_d_r_d_z` - Mixed second derivative of poloidal flux `psi` with respect to R and Z, [weber**2 / metre**2]
///
/// # Returns
/// * `determinant` - the determinant of the Hessian matrix, [weber**4 / metre**4]
/// * `trace` - the sum of the diagonal elements of the Hessian matrix, [weber**2 / metre**2]
///
/// # Example
/// ```
/// use gsfit_rs::plasma_geometry::hessian;
///
/// // Second derivatives
/// let d2_psi_d_r2: f64 = 1.0;
/// let d2_psi_d_z2: f64 = -1.0;
/// let d2_psi_d_r_d_z: f64 = 0.0;
///
/// let (hessian_det, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);
/// if hessian_det < 0.0 {
///     println!("Saddle point");
/// } else if hessian_det > 0.0 && hessian_trace > 0.0 {
///     println!("Local minimum");
/// } else if hessian_det > 0.0 && hessian_trace < 0.0 {
///     println!("Local maximum");
/// } else {
///     println!("This point can be a saddle point, local minimum, or local maximum");
/// }
/// ```
pub fn hessian(d2_psi_d_r2: f64, d2_psi_d_z2: f64, d2_psi_d_r_d_z: f64) -> (f64, f64) {
    // Calculate the Hessian determinant
    let hessian_det: f64 = d2_psi_d_r2 * d2_psi_d_z2 - d2_psi_d_r_d_z.powi(2);

    let hessian_trace: f64 = d2_psi_d_r2 + d2_psi_d_z2;

    (hessian_det, hessian_trace)
}

/// Test the `hessian` function with a known maximum point.
/// 
/// Note: The `hessian` function will be used on the nearest grid point to the stationary point.
/// So in the test we will similarly evaluate the Hessian at a nearby point, not exactly at the stationary point.
///
/// See the Jupyter notebook for a plot detailing the test
/// `rust/gsfit_rs/test_data/plasma_geometry/hessian/test_hessian_for_maximum.ipynb`
#[test]
fn test_hessian_for_maximum() {
    // Define the plasma geometry parameters
    let r_center: f64 = 0.43;
    // let z_center: f64 = 0.12;
    let vertical_curvature: f64 = 0.35;

    // Points close to the center
    let r: f64 = 0.41000000000000003;
    let z: f64 = 0.33333333333333326;

    // // Calculate the Hessian determinant and trace
    // let psi_2d: f64 = -(r - r_center).powi(2) - (z + 25e-3).powi(2);

    // // d_psi_d_r = -2 * (r - r_center)
    // let d_psi_d_r: f64 = -2.0 * (r - r_center);

    // // d_psi_d_z = -2 * (r - r_center) * 2 * vertical_curvature * z - 2 * (z + 0.025)
    // let d_psi_d_z: f64 = -2.0 * (r - r_center) * 2.0 * vertical_curvature * z - 2.0 * (z + 25e-3);

    let d2_psi_d_r2: f64 = -2.0;

    let d2_psi_d_z2: f64 = -4.0 * vertical_curvature * (r - r_center) - 8.0 * vertical_curvature.powi(2) * z.powi(2) - 2.0;

    let d2_psi_d_r_d_z: f64 = -4.0 * vertical_curvature * z;

    let (hessian_det, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);

    assert!(hessian_det > 0.0);
    assert!(hessian_trace < 0.0);
}
