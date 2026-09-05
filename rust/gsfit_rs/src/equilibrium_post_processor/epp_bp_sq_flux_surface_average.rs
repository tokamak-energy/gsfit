//! The flux-surface-averaged squared poloidal magnetic field.

use super::epp_flux_surfaces::FluxSurface;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;

/// Calculate the flux-surface-averaged squared poloidal magnetic field:
///
/// `<<b_p ** 2>> = ∮ b_p d_ell / ∮ (d_ell / b_p)`
///
/// which follows from the standard flux surface average, `<<x>> = ∮ (x / b_p) d_ell / ∮ (d_ell / b_p)`,
/// where the `1 / b_p` weighting comes from the volume element between neighbouring flux surfaces.
///
/// Note: `∮ b_p d_ell` is calculated numerically rather than using Ampere's law
/// (`∮ b_p d_ell = mu_0 * ip`), because Ampere's law with the full plasma current only
/// holds exactly on the boundary, and the average is evaluated slightly inside.
///
/// Like `epp_flux_surfaces` this fills no data dictionary path; it is an intermediate quantity,
/// used by `beta_pol_1` and `li_1`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `flux_surfaces` - the flux surfaces from `epp_flux_surfaces`, one per `psi_norm`
/// * `bp_sq_fs_avg_psi_norm` - which surface to average over [dimensionless]. For diverted plasmas
///   `b_p = 0` at the x-point, which lies on the boundary, making `∮ d_ell / b_p` log-divergent on
///   the separatrix; so this should be slightly inside the boundary, e.g. `0.995`
///
/// # Returns
/// * `bp_sq_fs_avg` - flux-surface-averaged `b_p ** 2` [tesla ** 2]
///
/// **Must run after `epp_flux_surfaces`**, which supplies the surfaces, and after
/// `epp_equilibrium_time_slice_profiles_2d_b_field_r_and_z`, which supplies the poloidal field.
///
/// Note: the surfaces are only known on the `psi_norm` grid, so the average is taken on the last
/// grid surface at or below `bp_sq_fs_avg_psi_norm`, rather than on a contour traced at exactly
/// that value. Rounding down rather than to nearest is what keeps it off the separatrix.
pub fn epp_bp_sq_flux_surface_average(time_slice: &EquilibriumTimeSlice, flux_surfaces: &[FluxSurface], bp_sq_fs_avg_psi_norm: f64) -> f64 {
    // A slice which did not converge has no flux surfaces to average over
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        return f64::NAN;
    }

    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let n_psi_norm: usize = psi_norm.len();

    // The last grid surface at or below the requested `psi_norm`
    let mut i_psi_norm_local: usize = 0;
    for i_psi_norm in 0..n_psi_norm {
        if psi_norm[i_psi_norm] <= bp_sq_fs_avg_psi_norm {
            i_psi_norm_local = i_psi_norm;
        }
    }

    // Note: the exterior ring is closed, i.e. the last point repeats the first point
    let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm_local].r;
    let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm_local].z;
    let fs_n: usize = fs_r.len();
    if fs_n < 4 {
        // Degenerate contour
        return f64::NAN;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let b_field_r_2d: &Array2<f64> = time_slice.profiles_2d[0].b_field_r.as_ref().unwrap();
    let b_field_z_2d: &Array2<f64> = time_slice.profiles_2d[0].b_field_z.as_ref().unwrap();

    // Interpolator for b_p on the (R, Z) grid
    let bp_2d: Array2<f64> = (b_field_r_2d.mapv(|x| x.powi(2)) + b_field_z_2d.mapv(|x| x.powi(2))).mapv(f64::sqrt);
    let bp_interpolator = Interp2D::builder(bp_2d).x(z.clone()).y(r.clone()).build().unwrap();

    // b_p along the flux surface
    let mut fs_bp: Array1<f64> = Array1::from_elem(fs_n, f64::NAN);
    for i_fs in 0..fs_n {
        fs_bp[i_fs] = bp_interpolator.interp_scalar(fs_z[i_fs], fs_r[i_fs]).unwrap();
    }

    // Trapezoidal integration around the closed contour
    let mut bp_d_ell_integral: f64 = 0.0; // ∮ b_p d_ell
    let mut d_ell_over_bp_integral: f64 = 0.0; // ∮ (d_ell / b_p)
    for i_fs in 1..fs_n {
        let delta_ell: f64 = (fs_r[i_fs] - fs_r[i_fs - 1]).hypot(fs_z[i_fs] - fs_z[i_fs - 1]);
        bp_d_ell_integral += 0.5 * delta_ell * (fs_bp[i_fs] + fs_bp[i_fs - 1]);
        d_ell_over_bp_integral += 0.5 * delta_ell * (1.0 / fs_bp[i_fs] + 1.0 / fs_bp[i_fs - 1]);
    }

    let bp_sq_fs_avg: f64 = bp_d_ell_integral / d_ell_over_bp_integral;

    return bp_sq_fs_avg;
}
