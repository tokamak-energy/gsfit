//! `time_slice(itime)/profiles_1d/q`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the safety factor profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/q` is written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    let q_profile: Array1<f64> = q_profile(time_slice, flux_surfaces, f_profile);

    time_slice.profiles_1d.q = q_profile;
}

/// Calculate the safety factor profile for an arbitrary `f` profile.
///
/// Kept separate from the writer above because it is evaluated twice: once with the reconstructed
/// `f`, and once with the vacuum `f`, which is what the diamagnetic flux is measured against.
///
/// On each flux surface
///
/// ```text
/// q = f / (2 * pi) * integral(d_l / (R ** 2 * b_p))
///   = f * integral(d_l / (R * |grad(psi)|)),
/// ```
///
/// with the contour integral from the surface's quadrature; see `flux_surfaces`. On the magnetic axis this is its limit,
///
/// ```text
/// q_axis = 2 * pi * f(psi_N = 0) / (mag_r * sqrt(det(H)))
/// ```
///
/// with `H` the Hessian of `psi` there, as the axis's flux surface has one quadrature point, of weight `2 * pi / sqrt(det(H))`.
/// `q` has the sign of `f` everywhere.
///
/// The exact `psi_norm = 1` point is deliberately left as NaN. For a diverted plasma the last
/// closed flux surface passes through an X-point, where `b_p = 0`, and its safety factor has a
/// logarithmic singularity. A finite number there would only be an implicit grid-dependent
/// regularisation.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `flux_surfaces` - the flux surfaces from `flux_surfaces::calculate`, one per `psi_norm`
/// * `f_profile` - the `f = R * B_phi` profile to integrate with [tesla metre]
///
/// # Returns
/// * `q_profile` - the safety factor profile [dimensionless]
pub(in crate::equilibrium_post_processor) fn q_profile(
    time_slice: &EquilibriumTimeSlice,
    flux_surfaces: &[FluxSurface],
    f_profile: &Array1<f64>,
) -> Array1<f64> {
    let n_psi_norm: usize = flux_surfaces.len();
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    assert_eq!(psi_norm.len(), n_psi_norm);
    assert_eq!(f_profile.len(), n_psi_norm);

    let mut q_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        if psi_norm[i_psi_norm] == 1.0 {
            continue;
        }
        q_profile[i_psi_norm] = q_on_flux_surface(&flux_surfaces[i_psi_norm], f_profile[i_psi_norm]);
    }

    q_profile
}

/// Calculate the safety factor on one flux surface, `q = f * integral(d_l / (R * |grad(psi)|))`.
///
/// # Arguments
/// * `flux_surface` - the surface
/// * `f_here` - `f = R * B_phi` on it [tesla metre]
///
/// # Returns
/// * `q` - NaN when the surface could not be found [dimensionless]
pub(in crate::equilibrium_post_processor) fn q_on_flux_surface(flux_surface: &FluxSurface, f_here: f64) -> f64 {
    f_here * flux_surface.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here)
}

#[cfg(test)]
mod tests {
    use super::super::super::boundary::geometry::time_slice_with_elliptical_psi;
    use super::super::super::flux_surfaces::calculate_flux_surfaces;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn circular_surfaces_match_the_analytic_q_and_the_boundary_is_nan() {
        // psi = 1 - a rho ** 2 about (0.5, 0), with the boundary at rho = 0.2, so |grad(psi)| = 2 a rho
        let a: f64 = 25.0;
        let mag_r: f64 = 0.5;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(mag_r, 0.0, a, 0.0, a);
        time_slice.boundary.psi = 1.0 - a * 0.2_f64.powi(2);
        time_slice.profiles_1d.psi_norm = array![0.0, 0.25, 0.5, 1.0];
        let f_here: f64 = 0.8;
        let f_profile: Array1<f64> = Array1::from_elem(4, f_here);

        let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(&time_slice);
        let q_profile: Array1<f64> = q_profile(&time_slice, &flux_surfaces, &f_profile);

        // q = f integral(d(theta) / (2 a (mag_r + rho cos(theta)))) = pi f / (a sqrt(mag_r ** 2 - rho ** 2)), and on the axis rho = 0
        for i_psi_norm in 0..3 {
            let rho: f64 = 0.2 * time_slice.profiles_1d.psi_norm[i_psi_norm].sqrt();
            let q_expected: f64 = PI * f_here / (a * (mag_r.powi(2) - rho.powi(2)).sqrt());
            assert_abs_diff_eq!(q_profile[i_psi_norm], q_expected, epsilon = 1e-12);
        }
        assert!(q_profile[3].is_nan());
    }

    #[test]
    fn q_axis_is_the_limit_of_the_surfaces_and_has_the_sign_of_f() {
        // A tilted ellipse, psi = 1 - (a_rr x ** 2 + a_rz x y + a_zz y ** 2)
        let a_rr: f64 = 30.0;
        let a_rz: f64 = 8.0;
        let a_zz: f64 = 12.0;
        let mag_r: f64 = 0.5;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(mag_r, 0.05, a_rr, a_rz, a_zz);
        time_slice.profiles_1d.psi_norm = array![0.0, 1.0e-10, 1.0e-6, 1.0];
        let f_here: f64 = -0.8;
        let f_profile: Array1<f64> = Array1::from_elem(4, f_here);

        let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(&time_slice);
        let q_profile: Array1<f64> = q_profile(&time_slice, &flux_surfaces, &f_profile);

        let hessian_determinant: f64 = (2.0 * a_rr) * (2.0 * a_zz) - a_rz.powi(2);
        let q_axis_expected: f64 = 2.0 * PI * f_here / (mag_r * hessian_determinant.sqrt());
        assert_abs_diff_eq!(q_profile[0], q_axis_expected, epsilon = 1e-12);
        assert_abs_diff_eq!(q_profile[1] / q_profile[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(q_profile[2] / q_profile[0], 1.0, epsilon = 1e-5);
        assert!(q_profile[0] < 0.0);
    }
}
