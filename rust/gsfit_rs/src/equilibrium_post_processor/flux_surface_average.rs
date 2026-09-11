//! Flux-surface averaging of scalar fields stored on the rectangular `(R, Z)` grid.
//!
//! This is a shared intermediate calculation and writes no data-dictionary path.

use super::flux_surfaces::FluxSurface;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;

const GAUSS_LEGENDRE_ABSCISSA: f64 = 0.5773502691896258;

/// Calculate a radially weighted flux-surface average of a 2-D scalar field.
///
/// The standard flux-surface average is recovered with `radial_weight(R) = 1`:
///
/// ```text
/// <<x>> = integral(x d_ell / b_p) / integral(d_ell / b_p).
/// ```
///
/// A different radial weight `w(R)` calculates
///
/// ```text
/// integral(x w(R) d_ell / b_p) / integral(w(R) d_ell / b_p).
/// ```
///
/// The magnetic-axis value is sampled directly because the axis is a point rather than a contour.
/// Each contour segment is integrated with two-point Gauss-Legendre quadrature.
pub(super) fn calculate<F>(time_slice: &EquilibriumTimeSlice, flux_surfaces: &[FluxSurface], values_2d: &Array2<f64>, radial_weight: F) -> Array1<f64>
where
    F: Fn(f64) -> f64,
{
    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();
    assert_eq!(flux_surfaces.len(), n_psi_norm);

    let mut values_average: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        return values_average;
    }

    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let b_field_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_r;
    let b_field_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_z;
    let (n_z, n_r): (usize, usize) = values_2d.dim();
    assert_eq!(b_field_r_2d.dim(), (n_z, n_r));
    assert_eq!(b_field_z_2d.dim(), (n_z, n_r));
    assert_eq!(r.len(), n_r);
    assert_eq!(z.len(), n_z);

    let b_field_p_2d: Array2<f64> = (b_field_r_2d.mapv(|value| value.powi(2)) + b_field_z_2d.mapv(|value| value.powi(2))).mapv(f64::sqrt);
    let b_field_p_interpolator = Interp2D::builder(b_field_p_2d).x(z.clone()).y(r.clone()).build().unwrap();
    let values_interpolator = Interp2D::builder(values_2d.to_owned()).x(z.clone()).y(r.clone()).build().unwrap();

    if n_psi_norm > 0 {
        let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
        let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
        values_average[0] = values_interpolator.interp_scalar(mag_z, mag_r).unwrap_or(f64::NAN);
    }

    'flux_surface_loop: for i_psi_norm in 1..n_psi_norm {
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;
        let n_fs: usize = fs_r.len();
        if n_fs < 4 || fs_z.len() != n_fs {
            continue 'flux_surface_loop;
        }

        let mut denominator: f64 = 0.0;
        let mut numerator: f64 = 0.0;
        for i_fs in 1..n_fs {
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

            let quadrature_points: [(f64, f64); 2] = [(midpoint_r - offset_r, midpoint_z - offset_z), (midpoint_r + offset_r, midpoint_z + offset_z)];
            for (r_here, z_here) in quadrature_points {
                let b_field_p_here: f64 = match b_field_p_interpolator.interp_scalar(z_here, r_here) {
                    Ok(value) => value,
                    Err(_) => continue 'flux_surface_loop,
                };
                let value_here: f64 = match values_interpolator.interp_scalar(z_here, r_here) {
                    Ok(value) => value,
                    Err(_) => continue 'flux_surface_loop,
                };
                let weight_here: f64 = radial_weight(r_here);
                if !b_field_p_here.is_finite() || b_field_p_here <= 0.0 || !value_here.is_finite() || !weight_here.is_finite() {
                    continue 'flux_surface_loop;
                }

                denominator += 0.5 * delta_ell * weight_here / b_field_p_here;
                numerator += 0.5 * delta_ell * value_here * weight_here / b_field_p_here;
            }
        }

        if denominator.is_finite() && denominator != 0.0 && numerator.is_finite() {
            values_average[i_psi_norm] = numerator / denominator;
        }
    }

    values_average
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use std::f64::consts::PI;

    #[test]
    fn standard_and_inverse_radius_weighted_averages_match_a_circle() {
        let mag_r: f64 = 1.0;
        let mag_z: f64 = 0.0;
        let minor_radius: f64 = 0.25;
        let n_grid: usize = 51;
        let r: Array1<f64> = Array1::linspace(0.5, 1.5, n_grid);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_grid);

        let b_field_r_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), 1.0);
        let b_field_z_2d: Array2<f64> = Array2::zeros((n_grid, n_grid));
        let mut values_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
        for i_r in 0..n_grid {
            for i_z in 0..n_grid {
                values_2d[(i_z, i_r)] = r[i_r];
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
            r: mag_r + minor_radius * theta.mapv(f64::cos),
            z: mag_z + minor_radius * theta.mapv(f64::sin),
        };
        let axis: FluxSurface = FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = 0.0;
        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice.profiles_1d.psi_norm = Array1::from_vec(vec![0.0, 1.0]);
        time_slice.profiles_2d = vec![profiles_2d];

        let standard_average: Array1<f64> = calculate(&time_slice, &[axis.clone(), flux_surface.clone()], &values_2d, |_r_here| 1.0);
        let inverse_radius_average: Array1<f64> = calculate(&time_slice, &[axis, flux_surface], &values_2d, |r_here| 1.0 / r_here);

        assert_abs_diff_eq!(standard_average[0], mag_r, epsilon = 1e-14);
        assert_abs_diff_eq!(standard_average[1], mag_r, epsilon = 1e-8);
        assert_abs_diff_eq!(inverse_radius_average[0], mag_r, epsilon = 1e-14);
        assert_abs_diff_eq!(inverse_radius_average[1], (mag_r.powi(2) - minor_radius.powi(2)).sqrt(), epsilon = 1e-7);
    }
}
