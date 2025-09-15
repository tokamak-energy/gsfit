/// Calculates the Hessian determinant for the poloidal flux function `psi`:
/// * det(Hessian(R, Z)) > 0: Both curvatures have the same sign ⇒ turning point (minimum or maximum)
///     * trace > 0: Minimum
///     * trace < 0: Maximum
/// * det(Hessian(R, Z)) < 0: Curvatures have opposite signs ⇒ saddle point
/// * det(Hessian(R, Z)) ≈ 0: Degenerate case ⇒ function could be flat (can be saddle point, local minimum, or local maximum)
///     * trace can be any value
///
/// # Arguments
/// * `d2_psi_d_r2` - Second derivative of poloidal flux `psi` with respect to R, weber^2 / metre^2
/// * `d2_psi_d_z2` - Second derivative of poloidal flux `psi` with respect to Z, weber^2 / metre^2
/// * `d2_psi_d_r_d_z` - Mixed second derivative of poloidal flux `psi` with respect to R and Z, weber^2 / metre^2
///
/// # Returns
/// * `trace` - the sum of the diagonal elements of the Hessian matrix
/// * `determinant` - the determinant of the Hessian matrix
///
/// # Example
/// ```
/// use gsfit_rs::plasma_geometry::hessian;
/// 
/// // Second derivatives
/// let d2_psi_d_r2: f64 = 1.0; // weber^2 / metre^2
/// let d2_psi_d_z2: f64 = -1.0; // weber^2 / metre^2
/// let d2_psi_d_r_d_z: f64 = 0.0; // weber^2 / metre^2
/// 
/// let (hessian_det, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);
/// if hessian_det < 0.0 {
///     println!("Saddle point = x-point");
/// } else if hessian_det > 0.0 && d2_psi_d_r2 > 0.0 {
///     println!("Local minimum");
/// } else if hessian_det > 0.0 && d2_psi_d_r2 < 0.0 {
///     println!("Local maximum = magnetic axis");
/// } else {
///     println!("This point can be a saddle point, local minimum, or local maximum");
/// }
/// ```
///
pub fn hessian(d2_psi_d_r2: f64, d2_psi_d_z2: f64, d2_psi_d_r_d_z: f64) -> (f64, f64) {
    // Calculate the Hessian determinant
    let hessian_det: f64 = d2_psi_d_r2 * d2_psi_d_z2 - d2_psi_d_r_d_z.powi(2);

    let hessian_trace: f64 = d2_psi_d_r2 + d2_psi_d_z2;
    return (hessian_det, hessian_trace);
}
