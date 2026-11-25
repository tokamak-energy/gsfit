//! `time_slice(itime)/profiles_1d/b_field_average`, `.../b_field_max` and `.../b_field_min`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::{FluxSurface, ValueRange};
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate the flux-surface average and extrema of the magnetic-field magnitude, and store them in the time-slice.
///
/// The total field at a point on a flux surface is
///
/// ```text
/// |b| = sqrt(b_p ** 2 + (f / R) ** 2),    b_p = |grad(psi)| / (2 * pi * R),
/// ```
///
/// with `f` constant on the surface. The average, `<|b|>`, and the largest and smallest `|b|` are taken with the surface's quadrature;
/// see `flux_surfaces`. On the magnetic axis `b_p = 0`, so all three are `|f / mag_r|`. A surface which could not be found is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/f` is read, and the three `profiles_1d` fields are written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let n_psi_norm: usize = flux_surfaces.len();
    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    assert_eq!(f_profile.len(), n_psi_norm);

    let mut b_field_average: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut b_field_max: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut b_field_min: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        let flux_surface: &FluxSurface = &flux_surfaces[i_psi_norm];
        let f_here: f64 = f_profile[i_psi_norm];
        let b_field = |r_here: f64, grad_psi_here: f64| -> f64 { (grad_psi_here / (2.0 * PI * r_here)).hypot(f_here / r_here) };
        let b_field_range: ValueRange = flux_surface.value_range(b_field);
        b_field_average[i_psi_norm] = flux_surface.flux_surface_average(b_field);
        b_field_max[i_psi_norm] = b_field_range.maximum;
        b_field_min[i_psi_norm] = b_field_range.minimum;
    }

    time_slice.profiles_1d.b_field_average = b_field_average;
    time_slice.profiles_1d.b_field_max = b_field_max;
    time_slice.profiles_1d.b_field_min = b_field_min;
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

    /// Circular surfaces, `psi = 1 - a * rho ** 2` about `(mag_r, 0)`: on the circle of radius `rho`, `|b| = sqrt(c) / R` with
    /// `c = (a * rho / pi) ** 2 + f ** 2`, whose average, `integral(x * R d(theta)) / (2 * pi * mag_r)`, is `sqrt(c) / mag_r`, and whose
    /// extremes are at `R = mag_r -+ rho`
    #[test]
    fn circular_surfaces_match_the_analytic_average_and_extrema() {
        let a: f64 = 25.0;
        let mag_r: f64 = 0.5;
        let minor_radius: f64 = 0.2;
        let f_here: f64 = -0.35;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(mag_r, 0.0, a, 0.0, a);
        time_slice.boundary.psi = 1.0 - a * minor_radius.powi(2);
        let n_outline: usize = 2001;
        let outline_theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_outline);
        time_slice.boundary.outline.r = outline_theta.mapv(|theta: f64| mag_r + minor_radius * theta.cos());
        time_slice.boundary.outline.z = outline_theta.mapv(|theta: f64| minor_radius * theta.sin());
        let psi_norm: Array1<f64> = array![0.0, 0.25, 0.81, 1.0];
        let n_psi_norm: usize = psi_norm.len();
        time_slice.profiles_1d.psi_norm = psi_norm.clone();
        time_slice.profiles_1d.f = Array1::from_elem(n_psi_norm, f_here);

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = calculate_flux_surfaces(&time_slice);
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        // The surfaces inside the boundary are exact, and the boundary outline, a polygon inscribed in the circle, nearly so
        for i_psi_norm in 0..n_psi_norm {
            let tolerance: f64 = if i_psi_norm == n_psi_norm - 1 { 1.0e-5 } else { 1.0e-12 };
            let rho: f64 = minor_radius * psi_norm[i_psi_norm].sqrt();
            let c_sqrt: f64 = ((a * rho / PI).powi(2) + f_here.powi(2)).sqrt();
            let profiles_1d = &time_slice.profiles_1d;
            assert_abs_diff_eq!(profiles_1d.b_field_average[i_psi_norm], c_sqrt / mag_r, epsilon = tolerance);
            assert_abs_diff_eq!(profiles_1d.b_field_max[i_psi_norm], c_sqrt / (mag_r - rho), epsilon = tolerance);
            assert_abs_diff_eq!(profiles_1d.b_field_min[i_psi_norm], c_sqrt / (mag_r + rho), epsilon = tolerance);
        }
    }
}
