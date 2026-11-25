//! The flux-surface-averaged squared poloidal magnetic field.

use super::constant_values::ConstantValues;
use super::flux_surfaces::{FluxSurface, calculate_at_psi_norm};
use super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use std::f64::consts::PI;

// For diverted plasmas `b_p = 0` at the x-point, which lies on the boundary, making
// `∮ d_ell / b_p` log-divergent on the separatrix, so the average is taken slightly inside it
const BP_SQ_FS_AVG_PSI_NORM: f64 = 0.995;

/// Calculate the flux-surface-averaged squared poloidal magnetic field and store it on the
/// shared intermediate values.
///
/// This is the uniform-signature entry point the post-processor dispatches; the average itself is
/// calculated by `calculate_bp_sq_fs_avg`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `intermediate_values` - the shared intermediate values; `bp_sq_fs_avg` is written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let bp_sq_fs_avg: f64 = calculate_bp_sq_fs_avg(time_slice);
    intermediate_values.bp_sq_fs_avg = bp_sq_fs_avg;
}

/// Calculate the flux-surface-averaged squared poloidal magnetic field,
///
/// `<b_p ** 2> = contour_integral(b_p d_ell) / contour_integral(d_ell / b_p)`,
///
/// with `b_p = |grad(psi)| / (2 * pi * R)`, on the flux surface at `psi_norm = BP_SQ_FS_AVG_PSI_NORM`, found along rays from the
/// magnetic axis; see `flux_surfaces`.
///
/// Note: `∮ b_p d_ell` is calculated numerically rather than using Ampere's law
/// (`∮ b_p d_ell = mu_0 * ip`), because Ampere's law with the full plasma current only
/// holds exactly on the boundary, and the average is evaluated slightly inside.
///
/// Like `flux_surfaces::calculate` this fills no data dictionary path; it is an intermediate quantity,
/// used by `beta_pol_1` and `li_1`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// * `bp_sq_fs_avg` - flux-surface-averaged `b_p ** 2`; NaN when the surface cannot be found [tesla ** 2]
fn calculate_bp_sq_fs_avg(time_slice: &EquilibriumTimeSlice) -> f64 {
    let flux_surface: FluxSurface = calculate_at_psi_norm(time_slice, BP_SQ_FS_AVG_PSI_NORM);

    flux_surface.flux_surface_average(|r_here: f64, grad_psi_here: f64| (grad_psi_here / (2.0 * PI * r_here)).powi(2))
}

#[cfg(test)]
mod tests {
    use super::super::boundary::geometry::time_slice_with_elliptical_psi;
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Circular surfaces, `psi = 1 - a * rho ** 2` about `(mag_r, 0)`: on the circle of radius `rho`, `b_p = a * rho / (pi * R)`, and
    /// the average, `integral(x * R d(theta)) / (2 * pi * mag_r)`, of `1 / R ** 2` is `1 / (mag_r * sqrt(mag_r ** 2 - rho ** 2))`
    #[test]
    fn a_circular_surface_matches_the_analytic_average() {
        let a: f64 = 25.0;
        let mag_r: f64 = 0.5;
        let minor_radius: f64 = 0.2;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(mag_r, 0.0, a, 0.0, a);
        time_slice.boundary.psi = 1.0 - a * minor_radius.powi(2);

        let bp_sq_fs_avg: f64 = calculate_bp_sq_fs_avg(&time_slice);

        let rho: f64 = minor_radius * BP_SQ_FS_AVG_PSI_NORM.sqrt();
        let bp_sq_fs_avg_expected: f64 = (a * rho / PI).powi(2) / (mag_r * (mag_r.powi(2) - rho.powi(2)).sqrt());
        assert_abs_diff_eq!(bp_sq_fs_avg, bp_sq_fs_avg_expected, epsilon = 1e-12 * bp_sq_fs_avg_expected);
    }
}
