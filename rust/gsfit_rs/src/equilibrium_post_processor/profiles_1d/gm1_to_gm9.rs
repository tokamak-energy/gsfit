//! `time_slice(itime)/profiles_1d/gm1`, `.../gm2`, ... `.../gm9`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::{PsiDerivatives, PsiInterpolator};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// The number of `gm` quantities the data dictionary defines, `gm1` to `gm9`
const N_GM: usize = 9;
/// The number of angles with which the average of `|grad(psi)|` around the infinitesimal surfaces about the magnetic axis is taken;
/// the integrand is smooth and periodic, so the trapezoidal rule converges geometrically [count]
const N_AXIS_ANGLE: usize = 256;

/// Calculate the nine flux-surface-averaged `gm` quantities, and store them in the time-slice.
///
/// The flux-surface average is
///
/// ```text
/// <x> = contour_integral(x d_l / b_p) / contour_integral(d_l / b_p),
/// ```
///
/// with `d_l / b_p = 2 * pi * R * d_l / |grad(psi)|`, taken with the surface's quadrature; see `flux_surfaces`. The nine quantities are:
///
/// ```text
/// gm1 = <1 / R ** 2>                       [1 / metre ** 2]
/// gm2 = <|grad(rho_tor)| ** 2 / R ** 2>    [1 / metre ** 2]
/// gm3 = <|grad(rho_tor)| ** 2>             [dimensionless]
/// gm4 = <1 / b ** 2>                       [1 / tesla ** 2]
/// gm5 = <b ** 2>                           [tesla ** 2]
/// gm6 = <|grad(rho_tor)| ** 2 / b ** 2>    [1 / tesla ** 2]
/// gm7 = <|grad(rho_tor)|>                  [dimensionless]
/// gm8 = <R>                                [metre]
/// gm9 = <1 / R>                            [1 / metre]
/// ```
///
/// where `b ** 2 = b_p ** 2 + b_phi ** 2`, with `b_p = |grad(psi)| / (2 * pi * R)` and `b_phi = f / R`.
///
/// # `grad(rho_tor)`
///
/// `rho_tor = sqrt(phi / (pi * b0))` [metre] is a flux label, so `|grad(rho_tor)| = |d(rho_tor)/d(psi)| * |grad(psi)|`. With
/// `q = d(phi)/d(psi)`,
///
/// ```text
/// |d(rho_tor)/d(psi)| = |q| / (2 * pi * |b0| * rho_tor),    |q| = |f| * contour_integral(d_l / (R * |grad(psi)|)),
/// ```
///
/// on the surface itself, rather than differenced between the surfaces.
///
/// # The magnetic axis
///
/// The axis is a surface of one point, on the axis, where `b_p = 0`; see `flux_surfaces`. `gm1`, `gm4`, `gm5`, `gm8` and `gm9` are
/// their values there. The other four are `|grad(psi)| -> 0` times `|d(rho_tor)/d(psi)| -> infinity`, and are their limits instead.
/// About the axis `psi - psi_axis = x^T H x / 2`, with `H` the Hessian of `psi` there, and `phi = q_axis * (psi - psi_axis)`, so
///
/// ```text
/// gm3 = |q_axis| * |trace(H)| / (4 * pi * |b0|),    gm2 = gm3 / mag_r ** 2,    gm6 = gm3 * mag_r ** 2 / f_axis ** 2,
/// gm7 = sqrt(|q_axis| / (2 * pi * |b0|)) * mean(sqrt(|lambda_1| * cos(t) ** 2 + |lambda_2| * sin(t) ** 2)),
/// ```
///
/// with `lambda_1` and `lambda_2` the eigenvalues of `H`, and the mean over `0 <= t < 2 * pi`.
///
/// A surface which could not be found is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the nine `profiles_1d/gm*` nodes are written into it
/// * `constant_values` - the constant values; `b0`, `vacuum_toroidal_field/b0` [tesla], is read
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let n_psi_norm: usize = flux_surfaces.len();
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    assert_eq!(psi_norm.len(), n_psi_norm);
    let b0: f64 = constant_values.b0;
    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;

    let mut gm_profiles: Vec<Array1<f64>> = vec![Array1::from_elem(n_psi_norm, f64::NAN); N_GM];
    for i_psi_norm in 0..n_psi_norm {
        let flux_surface: &FluxSurface = &flux_surfaces[i_psi_norm];
        let f_here: f64 = f_profile[i_psi_norm];
        let b_sq = |r_here: f64, grad_psi_here: f64| -> f64 { (grad_psi_here / (2.0 * PI * r_here)).powi(2) + (f_here / r_here).powi(2) };
        let q_magnitude: f64 = f_here.abs() * flux_surface.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);

        gm_profiles[0][i_psi_norm] = flux_surface.flux_surface_average(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here.powi(2));
        gm_profiles[3][i_psi_norm] = flux_surface.flux_surface_average(|r_here: f64, grad_psi_here: f64| 1.0 / b_sq(r_here, grad_psi_here));
        gm_profiles[4][i_psi_norm] = flux_surface.flux_surface_average(b_sq);
        gm_profiles[7][i_psi_norm] = flux_surface.flux_surface_average(|r_here: f64, _grad_psi_here: f64| r_here);
        gm_profiles[8][i_psi_norm] = flux_surface.flux_surface_average(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);

        if psi_norm[i_psi_norm] == 0.0 {
            // The limits on the magnetic axis; see this function's description
            let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
            let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
            let Some(psi_derivatives) = PsiInterpolator::new(time_slice).derivatives(mag_r, mag_z) else {
                continue;
            };
            let PsiDerivatives {
                d2_psi_d_r2,
                d2_psi_d_r_d_z,
                d2_psi_d_z2,
                ..
            } = psi_derivatives;
            let hessian_half_trace: f64 = 0.5 * (d2_psi_d_r2 + d2_psi_d_z2);
            let hessian_eigenvalue_spread: f64 = (0.5 * (d2_psi_d_r2 - d2_psi_d_z2)).hypot(d2_psi_d_r_d_z);
            let eigenvalue_1_magnitude: f64 = (hessian_half_trace + hessian_eigenvalue_spread).abs();
            let eigenvalue_2_magnitude: f64 = (hessian_half_trace - hessian_eigenvalue_spread).abs();
            let mut grad_psi_mean: f64 = 0.0;
            for i_angle in 0..N_AXIS_ANGLE {
                let angle: f64 = 2.0 * PI * (i_angle as f64) / (N_AXIS_ANGLE as f64);
                grad_psi_mean += (eigenvalue_1_magnitude * angle.cos().powi(2) + eigenvalue_2_magnitude * angle.sin().powi(2)).sqrt() / (N_AXIS_ANGLE as f64);
            }
            let gm3_axis: f64 = q_magnitude * (2.0 * hessian_half_trace).abs() / (4.0 * PI * b0.abs());
            gm_profiles[1][i_psi_norm] = gm3_axis / mag_r.powi(2);
            gm_profiles[2][i_psi_norm] = gm3_axis;
            gm_profiles[5][i_psi_norm] = gm3_axis * mag_r.powi(2) / f_here.powi(2);
            gm_profiles[6][i_psi_norm] = (q_magnitude / (2.0 * PI * b0.abs())).sqrt() * grad_psi_mean;
        } else {
            let d_rho_tor_d_psi_magnitude: f64 = q_magnitude / (2.0 * PI * b0.abs() * rho_tor[i_psi_norm]);
            gm_profiles[1][i_psi_norm] =
                d_rho_tor_d_psi_magnitude.powi(2) * flux_surface.flux_surface_average(|r_here: f64, grad_psi_here: f64| (grad_psi_here / r_here).powi(2));
            gm_profiles[2][i_psi_norm] =
                d_rho_tor_d_psi_magnitude.powi(2) * flux_surface.flux_surface_average(|_r_here: f64, grad_psi_here: f64| grad_psi_here.powi(2));
            gm_profiles[5][i_psi_norm] = d_rho_tor_d_psi_magnitude.powi(2)
                * flux_surface.flux_surface_average(|r_here: f64, grad_psi_here: f64| grad_psi_here.powi(2) / b_sq(r_here, grad_psi_here));
            gm_profiles[6][i_psi_norm] = d_rho_tor_d_psi_magnitude * flux_surface.flux_surface_average(|_r_here: f64, grad_psi_here: f64| grad_psi_here);
        }
    }

    time_slice.profiles_1d.gm1 = gm_profiles[0].to_owned();
    time_slice.profiles_1d.gm2 = gm_profiles[1].to_owned();
    time_slice.profiles_1d.gm3 = gm_profiles[2].to_owned();
    time_slice.profiles_1d.gm4 = gm_profiles[3].to_owned();
    time_slice.profiles_1d.gm5 = gm_profiles[4].to_owned();
    time_slice.profiles_1d.gm6 = gm_profiles[5].to_owned();
    time_slice.profiles_1d.gm7 = gm_profiles[6].to_owned();
    time_slice.profiles_1d.gm8 = gm_profiles[7].to_owned();
    time_slice.profiles_1d.gm9 = gm_profiles[8].to_owned();
}

#[cfg(test)]
mod tests {
    use super::super::super::boundary::geometry::time_slice_with_elliptical_psi;
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::flux_surfaces::calculate_flux_surfaces;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// The nine profiles in `gm1` to `gm9` order
    fn gm_profiles(time_slice: &EquilibriumTimeSlice) -> [&Array1<f64>; N_GM] {
        let profiles_1d = &time_slice.profiles_1d;
        [
            &profiles_1d.gm1,
            &profiles_1d.gm2,
            &profiles_1d.gm3,
            &profiles_1d.gm4,
            &profiles_1d.gm5,
            &profiles_1d.gm6,
            &profiles_1d.gm7,
            &profiles_1d.gm8,
            &profiles_1d.gm9,
        ]
    }

    /// Circular surfaces, `psi = 1 - a * rho ** 2` about `(mag_r, 0)`, with the boundary at `rho = 0.2`. The surface at `psi_norm` is the
    /// circle of radius `rho = 0.2 * sqrt(psi_norm)`, on which `|grad(psi)| = 2 * a * rho` and the average is
    /// `<x> = integral(x * R d(theta)) / (2 * pi * mag_r)`, so every `gm` is known exactly, and on the axis as the limit `rho -> 0`
    #[test]
    fn circular_surfaces_match_the_analytic_averages_and_their_limits_on_the_axis() {
        let a: f64 = 25.0;
        let mag_r: f64 = 0.5;
        let minor_radius: f64 = 0.2;
        let b0: f64 = -0.7;
        let f_here: f64 = -0.35;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(mag_r, 0.0, a, 0.0, a);
        time_slice.boundary.psi = 1.0 - a * minor_radius.powi(2);
        let n_outline: usize = 2001;
        let outline_theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_outline);
        time_slice.boundary.outline.r = outline_theta.mapv(|theta: f64| mag_r + minor_radius * theta.cos());
        time_slice.boundary.outline.z = outline_theta.mapv(|theta: f64| minor_radius * theta.sin());
        let psi_norm: Array1<f64> = array![0.0, 0.01, 0.25, 0.81, 1.0];
        let n_psi_norm: usize = psi_norm.len();
        time_slice.profiles_1d.psi_norm = psi_norm.clone();
        time_slice.profiles_1d.f = Array1::from_elem(n_psi_norm, f_here);
        // q = pi f / (a sqrt(mag_r ** 2 - rho ** 2)), phi = integral(q d(psi)) and rho_tor = sqrt(phi / (pi b0))
        let mut rho_tor: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        for i_psi_norm in 0..n_psi_norm {
            let rho: f64 = minor_radius * psi_norm[i_psi_norm].sqrt();
            let flux_toroidal: f64 = 2.0 * PI * f_here * (mag_r - (mag_r.powi(2) - rho.powi(2)).sqrt());
            rho_tor[i_psi_norm] = (flux_toroidal / (PI * b0)).sqrt();
        }
        time_slice.profiles_1d.rho_tor = rho_tor.clone();

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.b0 = b0;
        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = calculate_flux_surfaces(&time_slice);
        calculate(&mut time_slice, &constant_values, &mut intermediate_values);

        // The surfaces inside the boundary are exact, and the boundary outline, a polygon inscribed in the circle, nearly so
        let gm_profiles: [&Array1<f64>; N_GM] = gm_profiles(&time_slice);
        for i_psi_norm in 0..n_psi_norm {
            let tolerance: f64 = if i_psi_norm == n_psi_norm - 1 { 1.0e-5 } else { 1.0e-12 };
            let rho: f64 = minor_radius * psi_norm[i_psi_norm].sqrt();
            let q_here: f64 = PI * f_here / (a * (mag_r.powi(2) - rho.powi(2)).sqrt());
            // |grad(rho_tor)| is the same all round the circle; on the axis, rho / rho_tor -> sqrt(mag_r * b0 / f)
            let grad_rho_tor: f64 = if i_psi_norm == 0 {
                q_here.abs() * a / (PI * b0.abs()) * (mag_r * b0 / f_here).sqrt()
            } else {
                q_here.abs() / (2.0 * PI * b0.abs() * rho_tor[i_psi_norm]) * 2.0 * a * rho
            };
            // b ** 2 = c / R ** 2
            let c: f64 = (a * rho / PI).powi(2) + f_here.powi(2);
            let gm1_expected: f64 = 1.0 / (mag_r * (mag_r.powi(2) - rho.powi(2)).sqrt());
            let gm_expected: [f64; N_GM] = [
                gm1_expected,
                grad_rho_tor.powi(2) * gm1_expected,
                grad_rho_tor.powi(2),
                (mag_r.powi(2) + 1.5 * rho.powi(2)) / c,
                c * gm1_expected,
                grad_rho_tor.powi(2) * (mag_r.powi(2) + 1.5 * rho.powi(2)) / c,
                grad_rho_tor,
                (mag_r.powi(2) + 0.5 * rho.powi(2)) / mag_r,
                1.0 / mag_r,
            ];
            for i_gm in 0..N_GM {
                assert_abs_diff_eq!(gm_profiles[i_gm][i_psi_norm], gm_expected[i_gm], epsilon = tolerance * gm_expected[i_gm].abs());
            }
        }
    }

    /// For a tilted ellipse, the four quantities with `grad(rho_tor)` on the axis are the limits of those on the surfaces next to it
    #[test]
    fn the_gradient_quantities_on_the_axis_are_the_limits_of_the_surfaces_next_to_it() {
        let b0: f64 = 0.9;
        let f_here: f64 = 0.45;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(0.5, 0.05, 30.0, 8.0, 12.0);
        let psi_norm: Array1<f64> = array![0.0, 1.0e-8, 1.0];
        let n_psi_norm: usize = psi_norm.len();
        time_slice.profiles_1d.psi_norm = psi_norm.clone();
        time_slice.profiles_1d.f = Array1::from_elem(n_psi_norm, f_here);
        let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(&time_slice);
        // Next to the axis, phi = q (psi - psi_axis) to first order; psi_axis = 1 and psi_boundary = 0
        let mut rho_tor: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        for i_psi_norm in 0..n_psi_norm - 1 {
            let q_here: f64 = f_here * flux_surfaces[i_psi_norm].contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);
            rho_tor[i_psi_norm] = (q_here * psi_norm[i_psi_norm] / (PI * b0)).sqrt();
        }
        time_slice.profiles_1d.rho_tor = rho_tor;

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.b0 = b0;
        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values);

        let gm_profiles: [&Array1<f64>; N_GM] = gm_profiles(&time_slice);
        for i_gm in 0..N_GM {
            assert!(gm_profiles[i_gm][0].is_finite() && gm_profiles[i_gm][0] > 0.0);
            assert_abs_diff_eq!(gm_profiles[i_gm][1] / gm_profiles[i_gm][0], 1.0, epsilon = 1e-6);
        }
    }
}
