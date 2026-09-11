//! `time_slice(itime)/profiles_1d/b_field_average`, `.../b_field_max` and `.../b_field_min`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;

const GAUSS_LEGENDRE_ABSCISSA: f64 = 0.5773502691896258;

/// Calculate the flux-surface average and extrema of the magnetic-field magnitude.
///
/// The total field at a point on a flux surface is
///
/// ```text
/// |b| = sqrt(b_r ** 2 + b_z ** 2 + (f / r) ** 2).
/// ```
///
/// `b_r` and `b_z` are interpolated independently from the rectangular grid and combined at the
/// requested point. `b_phi = f / r` is evaluated directly because `f` is constant on a flux
/// surface. No Green's table is evaluated away from the rectangular grid.
///
/// The flux-surface average uses the standard volume weighting
///
/// ```text
/// <<|b|>> = integral((|b| / b_p) d_ell) / integral(d_ell / b_p),
/// ```
///
/// evaluated with two-point Gauss-Legendre quadrature on every straight contour segment. The
/// extrema include the contour vertices as well as the two quadrature points on each segment.
///
/// The magnetic axis is a point rather than a contour. There `b_p = 0`, so all three quantities
/// are exactly `|f / r_axis|`. A surface which cannot be traced or sampled is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the three `profiles_1d` fields are written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();
    assert_eq!(flux_surfaces.len(), n_psi_norm);

    let mut b_field_average: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut b_field_max: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut b_field_min: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        store(time_slice, b_field_average, b_field_max, b_field_min);
        return;
    }

    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let b_field_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_r;
    let b_field_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_z;
    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    assert_eq!(f_profile.len(), n_psi_norm);

    let b_field_r_interpolator = Interp2D::builder(b_field_r_2d.to_owned()).x(z.clone()).y(r.clone()).build().unwrap();
    let b_field_z_interpolator = Interp2D::builder(b_field_z_2d.to_owned()).x(z.clone()).y(r.clone()).build().unwrap();

    if n_psi_norm > 0 {
        let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
        let b_field_axis: f64 = (f_profile[0] / mag_r).abs();
        b_field_average[0] = b_field_axis;
        b_field_max[0] = b_field_axis;
        b_field_min[0] = b_field_axis;
    }

    'flux_surface_loop: for i_psi_norm in 1..n_psi_norm {
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;
        let n_fs: usize = fs_r.len();
        if n_fs < 4 || fs_z.len() != n_fs || !f_profile[i_psi_norm].is_finite() {
            continue 'flux_surface_loop;
        }

        let field_at = |r_here: f64, z_here: f64| -> Option<(f64, f64)> {
            if r_here <= 0.0 {
                return None;
            }
            let b_field_r_here: f64 = b_field_r_interpolator.interp_scalar(z_here, r_here).ok()?;
            let b_field_z_here: f64 = b_field_z_interpolator.interp_scalar(z_here, r_here).ok()?;
            let b_field_p_here: f64 = b_field_r_here.hypot(b_field_z_here);
            let b_field_phi_here: f64 = f_profile[i_psi_norm] / r_here;
            let b_field_here: f64 = b_field_p_here.hypot(b_field_phi_here);
            if !b_field_p_here.is_finite() || !b_field_here.is_finite() {
                return None;
            }
            return Some((b_field_here, b_field_p_here));
        };

        let mut average_denominator: f64 = 0.0;
        let mut average_numerator: f64 = 0.0;
        let mut maximum: f64 = f64::NEG_INFINITY;
        let mut minimum: f64 = f64::INFINITY;

        for i_fs in 1..n_fs {
            let (b_field_at_vertex, _b_field_p_at_vertex): (f64, f64) = match field_at(fs_r[i_fs - 1], fs_z[i_fs - 1]) {
                Some(field) => field,
                None => continue 'flux_surface_loop,
            };
            maximum = maximum.max(b_field_at_vertex);
            minimum = minimum.min(b_field_at_vertex);

            let delta_r: f64 = fs_r[i_fs] - fs_r[i_fs - 1];
            let delta_z: f64 = fs_z[i_fs] - fs_z[i_fs - 1];
            let delta_ell: f64 = delta_r.hypot(delta_z);
            if delta_ell == 0.0 {
                continue;
            }

            let midpoint_r: f64 = 0.5 * (fs_r[i_fs] + fs_r[i_fs - 1]);
            let midpoint_z: f64 = 0.5 * (fs_z[i_fs] + fs_z[i_fs - 1]);
            let offset_r: f64 = 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_r;
            let offset_z: f64 = 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_z;

            let (b_field_minus, b_field_p_minus): (f64, f64) = match field_at(midpoint_r - offset_r, midpoint_z - offset_z) {
                Some(field) => field,
                None => continue 'flux_surface_loop,
            };
            let (b_field_plus, b_field_p_plus): (f64, f64) = match field_at(midpoint_r + offset_r, midpoint_z + offset_z) {
                Some(field) => field,
                None => continue 'flux_surface_loop,
            };
            if b_field_p_minus == 0.0 || b_field_p_plus == 0.0 {
                continue 'flux_surface_loop;
            }

            average_numerator += 0.5 * delta_ell * (b_field_minus / b_field_p_minus + b_field_plus / b_field_p_plus);
            average_denominator += 0.5 * delta_ell * (1.0 / b_field_p_minus + 1.0 / b_field_p_plus);
            maximum = maximum.max(b_field_minus).max(b_field_plus);
            minimum = minimum.min(b_field_minus).min(b_field_plus);
        }

        let (b_field_at_last_vertex, _b_field_p_at_last_vertex): (f64, f64) = match field_at(fs_r[n_fs - 1], fs_z[n_fs - 1]) {
            Some(field) => field,
            None => continue 'flux_surface_loop,
        };
        maximum = maximum.max(b_field_at_last_vertex);
        minimum = minimum.min(b_field_at_last_vertex);

        if average_denominator.is_finite() && average_denominator > 0.0 && average_numerator.is_finite() && maximum.is_finite() && minimum.is_finite() {
            b_field_average[i_psi_norm] = average_numerator / average_denominator;
            b_field_max[i_psi_norm] = maximum;
            b_field_min[i_psi_norm] = minimum;
        }
    }

    store(time_slice, b_field_average, b_field_max, b_field_min);
}

fn store(time_slice: &mut EquilibriumTimeSlice, b_field_average: Array1<f64>, b_field_max: Array1<f64>, b_field_min: Array1<f64>) {
    time_slice.profiles_1d.b_field_average = b_field_average;
    time_slice.profiles_1d.b_field_max = b_field_max;
    time_slice.profiles_1d.b_field_min = b_field_min;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use std::f64::consts::PI;

    #[test]
    fn circular_surface_matches_analytic_average_and_extrema() {
        let r_axis: f64 = 1.0;
        let z_axis: f64 = 0.0;
        let minor_radius: f64 = 0.25;
        let psi_curvature: f64 = 2.5;
        let f_here: f64 = 0.8;
        let n_grid: usize = 101;
        let r: Array1<f64> = Array1::linspace(0.5, 1.5, n_grid);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_grid);

        let mut b_field_r_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
        let mut b_field_z_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
        for i_r in 0..n_grid {
            for i_z in 0..n_grid {
                b_field_r_2d[(i_z, i_r)] = -psi_curvature * (z[i_z] - z_axis) / (2.0 * PI * r[i_r]);
                b_field_z_2d[(i_z, i_r)] = psi_curvature * (r[i_r] - r_axis) / (2.0 * PI * r[i_r]);
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.b_field_r = b_field_r_2d;
        profiles_2d.b_field_z = b_field_z_2d;

        let n_theta: usize = 2001;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_theta);
        let flux_surface: FluxSurface = FluxSurface {
            r: r_axis + minor_radius * theta.mapv(f64::cos),
            z: z_axis + minor_radius * theta.mapv(f64::sin),
        };
        let empty_surface: FluxSurface = FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = 0.0;
        time_slice.global_quantities.magnetic_axis.r = r_axis;
        time_slice.profiles_1d.psi_norm = Array1::from_vec(vec![0.0, 0.25, 0.5]);
        time_slice.profiles_1d.f = Array1::from_vec(vec![f_here, f_here, f_here]);
        time_slice.profiles_2d = vec![profiles_2d];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = vec![empty_surface.clone(), empty_surface, flux_surface];
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        let field_numerator: f64 = ((psi_curvature * minor_radius / (2.0 * PI)).powi(2) + f_here.powi(2)).sqrt();
        let average_expected: f64 = field_numerator / r_axis;
        let maximum_expected: f64 = field_numerator / (r_axis - minor_radius);
        let minimum_expected: f64 = field_numerator / (r_axis + minor_radius);
        let b_field_average: &Array1<f64> = &time_slice.profiles_1d.b_field_average;
        let b_field_max: &Array1<f64> = &time_slice.profiles_1d.b_field_max;
        let b_field_min: &Array1<f64> = &time_slice.profiles_1d.b_field_min;

        assert_abs_diff_eq!(b_field_average[0], f_here / r_axis, epsilon = 1e-15);
        assert_abs_diff_eq!(b_field_max[0], f_here / r_axis, epsilon = 1e-15);
        assert_abs_diff_eq!(b_field_min[0], f_here / r_axis, epsilon = 1e-15);
        assert!(b_field_average[1].is_nan());
        assert!(b_field_max[1].is_nan());
        assert!(b_field_min[1].is_nan());
        assert_abs_diff_eq!(b_field_average[2], average_expected, epsilon = 1e-5);
        assert_abs_diff_eq!(b_field_max[2], maximum_expected, epsilon = 1e-5);
        assert_abs_diff_eq!(b_field_min[2], minimum_expected, epsilon = 1e-5);
    }
}
