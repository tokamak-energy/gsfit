//! `time_slice(itime)/profiles_1d/volume`, `.../dvolume_dpsi`, `.../dvolume_dpsi_norm`,
//! `.../dvolume_drho_tor`, `.../area`, `.../darea_dpsi`, `.../darea_drho_tor` and `.../surface`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate the volume and cross-sectional area enclosed by each flux surface, the area of the surface itself, and the derivatives of
/// the enclosed volume and area with respect to both radial coordinates, and of the enclosed volume with respect to `psi_norm`, and
/// store them in the time-slice.
///
/// The enclosed volume and area are those of the surface; see `flux_surfaces`. The rest are contour integrals over it,
///
/// ```text
/// surface        = 2 * pi * contour_integral(R d_l)
/// d(volume)/d(psi) = 2 * pi * contour_integral(R d_l / |grad(psi)|)
/// d(area)/d(psi)   = contour_integral(d_l / |grad(psi)|),
/// ```
///
/// because the surfaces at `psi` and `psi + d(psi)` are `d(psi) / |grad(psi)|` apart; the two derivatives have the sign of
/// `psi_boundary - psi_magnetic_axis`, as the volume and area grow outwards. With `q = d(phi)/d(psi)` and `rho_tor = sqrt(phi / (pi * b0))`,
///
/// ```text
/// |d(psi)/d(rho_tor)| = 2 * pi * |b0| * rho_tor / |q|,    |q| = |f| * contour_integral(d_l / (R * |grad(psi)|)),
/// ```
///
/// which turns the derivatives with respect to `psi` into those with respect to `rho_tor`. They are zero at the magnetic axis, where
/// `rho_tor` is. No derivative is a finite difference between the surfaces.
///
/// A flux surface which could not be found is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the eight `profiles_1d` nodes above are written into it
/// * `constant_values` - the constant values; `b0`, `vacuum_toroidal_field/b0` [tesla], is read
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces`, one per `psi_norm`, is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let n_psi_norm: usize = flux_surfaces.len();
    assert_eq!(time_slice.profiles_1d.psi_norm.len(), n_psi_norm);

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;
    let sign_psi_outwards: f64 = (psi_b - psi_a).signum();
    let b0: f64 = constant_values.b0;
    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;

    let mut volume_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut dvolume_dpsi_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut dvolume_dpsi_norm_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut dvolume_drho_tor_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut area_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut darea_dpsi_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut darea_drho_tor_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut surface_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        let flux_surface: &FluxSurface = &flux_surfaces[i_psi_norm];
        let integral_times_r: f64 = flux_surface.contour_integral(|r_here: f64, _grad_psi_here: f64| r_here);
        let integral_over_r: f64 = flux_surface.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);
        let integral_of_one: f64 = flux_surface.contour_integral(|_r_here: f64, _grad_psi_here: f64| 1.0);
        let d_psi_d_rho_tor_magnitude: f64 = 2.0 * PI * b0.abs() * rho_tor[i_psi_norm] / (f_profile[i_psi_norm].abs() * integral_over_r);

        volume_profile[i_psi_norm] = flux_surface.volume;
        area_profile[i_psi_norm] = flux_surface.area;
        surface_profile[i_psi_norm] = 2.0 * PI * flux_surface.contour_integral(|r_here: f64, grad_psi_here: f64| r_here * grad_psi_here);
        dvolume_dpsi_profile[i_psi_norm] = sign_psi_outwards * 2.0 * PI * integral_times_r;
        dvolume_dpsi_norm_profile[i_psi_norm] = 2.0 * PI * integral_times_r * (psi_b - psi_a).abs();
        darea_dpsi_profile[i_psi_norm] = sign_psi_outwards * integral_of_one;
        dvolume_drho_tor_profile[i_psi_norm] = 2.0 * PI * integral_times_r * d_psi_d_rho_tor_magnitude;
        darea_drho_tor_profile[i_psi_norm] = integral_of_one * d_psi_d_rho_tor_magnitude;
    }

    time_slice.profiles_1d.volume = volume_profile;
    time_slice.profiles_1d.dvolume_dpsi = dvolume_dpsi_profile;
    time_slice.profiles_1d.dvolume_dpsi_norm = dvolume_dpsi_norm_profile;
    time_slice.profiles_1d.dvolume_drho_tor = dvolume_drho_tor_profile;
    time_slice.profiles_1d.area = area_profile;
    time_slice.profiles_1d.darea_dpsi = darea_dpsi_profile;
    time_slice.profiles_1d.darea_drho_tor = darea_drho_tor_profile;
    time_slice.profiles_1d.surface = surface_profile;
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

    /// Circular surfaces, `psi = 1 - a * rho ** 2` about `(mag_r, 0)`, with the boundary at `rho = 0.2`, and their exact profiles: the
    /// surface at `psi_norm` is the circle of radius `rho = 0.2 * sqrt(psi_norm)`, which encloses `area = pi * rho ** 2` and
    /// `volume = 2 * pi ** 2 * mag_r * rho ** 2`, and sweeps out `surface = 4 * pi ** 2 * mag_r * rho`. `psi` decreases outwards, so
    /// `dvolume_dpsi` and `darea_dpsi` are negative while `dvolume_dpsi_norm` is not
    #[test]
    fn circular_surfaces_match_their_analytic_profiles() {
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
        let profiles_1d = &time_slice.profiles_1d;
        let psi_span: f64 = a * minor_radius.powi(2);
        for i_psi_norm in 0..n_psi_norm {
            let tolerance: f64 = if i_psi_norm == n_psi_norm - 1 { 1.0e-5 } else { 1.0e-12 };
            let rho: f64 = minor_radius * psi_norm[i_psi_norm].sqrt();
            let q_here: f64 = PI * f_here / (a * (mag_r.powi(2) - rho.powi(2)).sqrt());
            let d_psi_d_rho_tor_magnitude: f64 = 2.0 * PI * b0.abs() * rho_tor[i_psi_norm] / q_here.abs();
            let expected: [(f64, f64); 8] = [
                (profiles_1d.area[i_psi_norm], PI * rho.powi(2)),
                (profiles_1d.volume[i_psi_norm], 2.0 * PI.powi(2) * mag_r * rho.powi(2)),
                (profiles_1d.surface[i_psi_norm], 4.0 * PI.powi(2) * mag_r * rho),
                // d(volume)/d(psi) = -2 pi ** 2 mag_r / a and d(area)/d(psi) = -pi / a, the same on every circle
                (profiles_1d.dvolume_dpsi[i_psi_norm], -2.0 * PI.powi(2) * mag_r / a),
                (profiles_1d.dvolume_dpsi_norm[i_psi_norm], 2.0 * PI.powi(2) * mag_r / a * psi_span),
                (profiles_1d.darea_dpsi[i_psi_norm], -PI / a),
                (
                    profiles_1d.dvolume_drho_tor[i_psi_norm],
                    2.0 * PI.powi(2) * mag_r / a * d_psi_d_rho_tor_magnitude,
                ),
                (profiles_1d.darea_drho_tor[i_psi_norm], PI / a * d_psi_d_rho_tor_magnitude),
            ];
            for (value, value_expected) in expected {
                assert_abs_diff_eq!(value, value_expected, epsilon = tolerance * value_expected.abs().max(1.0));
            }
        }
        // Both derivatives with respect to rho_tor are zero at the magnetic axis
        assert_eq!(profiles_1d.dvolume_drho_tor[0], 0.0);
        assert_eq!(profiles_1d.darea_drho_tor[0], 0.0);
    }
}
