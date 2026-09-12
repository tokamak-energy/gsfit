//! `time_slice(itime)/profiles_1d/gm1`, `.../gm2`, ... `.../gm9`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;
use std::f64::consts::PI;

/// The number of `gm` quantities the data dictionary defines, `gm1` to `gm9`
const N_GM: usize = 9;

/// Calculate the nine flux-surface-averaged `gm` quantities, and store them in the time-slice.
///
/// The flux surface average is the volume average between neighbouring surfaces:
///
/// ```text
/// <<x>> = ∮ (x / b_p) d_ell / ∮ (d_ell / b_p)
/// ```
///
/// which is the same weighting `bp_sq_flux_surface_average::calculate` uses. It follows from
/// `d(V)/d(psi) = 2 * pi * ∮ r d_ell / |grad(psi)|` together with `|grad(psi)| = 2 * pi * r * b_p`,
/// where the two factors of `r` cancel.
///
/// The nine quantities are:
///
/// ```text
/// gm1 = <<1 / r ** 2>>                       [1 / metre ** 2]
/// gm2 = <<|grad(rho_tor)| ** 2 / r ** 2>>    [1 / metre ** 2]
/// gm3 = <<|grad(rho_tor)| ** 2>>             [dimensionless]
/// gm4 = <<1 / b ** 2>>                       [1 / tesla ** 2]
/// gm5 = <<b ** 2>>                           [tesla ** 2]
/// gm6 = <<|grad(rho_tor)| ** 2 / b ** 2>>    [1 / tesla ** 2]
/// gm7 = <<|grad(rho_tor)|>>                  [dimensionless]
/// gm8 = <<r>>                                [metre]
/// gm9 = <<1 / r>>                            [1 / metre]
/// ```
///
/// where `b ** 2 = b_p ** 2 + b_phi ** 2` is the total field, and `b_phi = f / r` is taken from the
/// `f` profile rather than interpolated from `profiles_2d`, because `f` is constant on a flux
/// surface and so is exact there.
///
/// # `grad(rho_tor)`
///
/// `rho_tor` is a flux label, so it varies only across surfaces:
///
/// ```text
/// |grad(rho_tor)| = |d(rho_tor)/d(psi)| * |grad(psi)| = |d(rho_tor)/d(psi)| * 2 * pi * r * b_p
/// ```
///
/// `d(rho_tor)/d(psi)` is one value per surface, differenced from the profiles, so the only part
/// which varies around a contour is `r * b_p`.
///
/// `rho_tor` is read from `profiles_1d/rho_tor`, which is the data dictionary's
/// `sqrt(phi / (pi * b0))` in metre. That the units are metre is what makes `gm2` come out in
/// `1 / metre ** 2` and `gm3` and `gm7` dimensionless; the normalised `rho_tor_norm` would leave
/// all four of these wrong by a factor of `rho_tor(boundary)`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the nine `profiles_1d/gm*` nodes are written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
///
/// # The magnetic axis and missing surfaces
///
/// A surface which could not be traced is left as NaN, exactly as `q` and the enclosed volume are.
///
/// The magnetic axis is a point rather than a surface, so the average of a smooth quantity there is
/// just its value at the axis: `gm1`, `gm4`, `gm5`, `gm8` and `gm9` are filled exactly, using
/// `b_p = 0` so that `b ** 2 = b_phi ** 2`. The four which depend on `|grad(rho_tor)|` are not
/// exact there - `|grad(psi)|` vanishes at the axis while `d(rho_tor)/d(psi)` diverges, an
/// indeterminate form with a finite limit - so those are linearly extrapolated from the first two
/// traced surfaces.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

    // A slice which did not converge has no flux surfaces to integrate around
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        store(time_slice, &vec![Array1::from_elem(n_psi_norm, f64::NAN); N_GM]);
        return;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let b_field_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_r;
    let b_field_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_z;

    let bp_2d: Array2<f64> = (b_field_r_2d.mapv(|x| x.powi(2)) + b_field_z_2d.mapv(|x| x.powi(2))).mapv(f64::sqrt);
    let bp_interpolator = Interp2D::builder(bp_2d).x(z.clone()).y(r.clone()).build().unwrap();

    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    let psi_profile: &Array1<f64> = &time_slice.profiles_1d.psi;
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;

    let d_rho_tor_d_psi: Array1<f64> = epp_d_rho_tor_d_psi(rho_tor, psi_profile);

    let mut gm_profiles: Vec<Array1<f64>> = vec![Array1::from_elem(n_psi_norm, f64::NAN); N_GM];

    // The magnetic axis is a point, so it is filled after the loop, from the axis values and by
    // extrapolation; see the note in this function's documentation
    'flux_surface_loop: for i_psi_norm in 1..n_psi_norm {
        // Note: the contour is closed, i.e. the last point repeats the first point
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;
        let n_fs: usize = fs_r.len();

        // A ring needs 3 distinct points to enclose an area, plus the repeated first point
        if n_fs < 4 {
            continue 'flux_surface_loop;
        }

        // The nine quantities at each point of the contour, plus the `1 / b_p` weight
        let mut fs_quantities: Vec<[f64; N_GM]> = Vec::with_capacity(n_fs);
        let mut fs_bp: Array1<f64> = Array1::from_elem(n_fs, f64::NAN);
        for i_fs in 0..n_fs {
            let bp_here: f64 = bp_interpolator.interp_scalar(fs_z[i_fs], fs_r[i_fs]).unwrap();
            fs_bp[i_fs] = bp_here;
            fs_quantities.push(gm_quantities_at_point(fs_r[i_fs], bp_here, f_profile[i_psi_norm], d_rho_tor_d_psi[i_psi_norm]));
        }

        // Trapezoidal integration around the closed contour, accumulating the nine numerators
        // `∮ (x / b_p) d_ell` and the shared denominator `∮ (d_ell / b_p)` in one pass
        let mut gm_numerators: [f64; N_GM] = [0.0; N_GM];
        let mut denominator: f64 = 0.0;
        for i_fs in 1..n_fs {
            let delta_ell: f64 = (fs_r[i_fs] - fs_r[i_fs - 1]).hypot(fs_z[i_fs] - fs_z[i_fs - 1]);
            denominator += 0.5 * delta_ell * (1.0 / fs_bp[i_fs] + 1.0 / fs_bp[i_fs - 1]);
            for i_gm in 0..N_GM {
                let this_point: f64 = fs_quantities[i_fs][i_gm] / fs_bp[i_fs];
                let previous_point: f64 = fs_quantities[i_fs - 1][i_gm] / fs_bp[i_fs - 1];
                gm_numerators[i_gm] += 0.5 * delta_ell * (this_point + previous_point);
            }
        }

        for i_gm in 0..N_GM {
            gm_profiles[i_gm][i_psi_norm] = gm_numerators[i_gm] / denominator;
        }
    }

    epp_gm_at_magnetic_axis(time_slice, f_profile, &mut gm_profiles);

    store(time_slice, &gm_profiles);
}

/// The nine `gm` quantities at one point on a flux surface, in `gm1` to `gm9` order.
///
/// These are the `x` of `<<x>>`; the `1 / b_p` weighting of the flux surface average is applied by
/// the caller, which needs `b_p` for the denominator anyway.
///
/// # Arguments
/// * `r_here` - major radius of this point, [metre]
/// * `bp_here` - poloidal field at this point, [tesla]
/// * `f_here` - `f = r * b_phi` on this surface, [tesla * metre]
/// * `d_rho_tor_d_psi_here` - `d(rho_tor)/d(psi)` on this surface, [metre / weber]
fn gm_quantities_at_point(r_here: f64, bp_here: f64, f_here: f64, d_rho_tor_d_psi_here: f64) -> [f64; N_GM] {
    // `b_phi = f / r`, exact on a flux surface because `f` is constant along it
    let b_phi_here: f64 = f_here / r_here;
    let b_sq_here: f64 = bp_here.powi(2) + b_phi_here.powi(2);

    // `rho_tor` is a flux label, so `grad(rho_tor)` is parallel to `grad(psi)`, whose magnitude is
    // `2 * pi * r * b_p`
    let grad_psi_here: f64 = 2.0 * PI * r_here * bp_here;
    let grad_rho_tor_here: f64 = d_rho_tor_d_psi_here.abs() * grad_psi_here;
    let grad_rho_tor_sq_here: f64 = grad_rho_tor_here.powi(2);

    [
        1.0 / r_here.powi(2),                  // gm1
        grad_rho_tor_sq_here / r_here.powi(2), // gm2
        grad_rho_tor_sq_here,                  // gm3
        1.0 / b_sq_here,                       // gm4
        b_sq_here,                             // gm5
        grad_rho_tor_sq_here / b_sq_here,      // gm6
        grad_rho_tor_here,                     // gm7
        r_here,                                // gm8
        1.0 / r_here,                          // gm9
    ]
}

/// Differentiate `rho_tor` with respect to `psi`, one value per flux surface.
///
/// # Arguments
/// * `rho_tor` - the toroidal flux coordinate profile, `profiles_1d/rho_tor` [metre]
/// * `psi_profile` - the poloidal flux profile, `profiles_1d/psi` [weber]
///
/// # Returns
/// * `d_rho_tor_d_psi` - one value per flux surface [metre / weber]
///
/// `psi_profile` runs in increasing index order from the magnetic axis outwards, so the numerators
/// do too: a forward difference at the first point, central differences in the interior, and a
/// backward difference at the last point.
fn epp_d_rho_tor_d_psi(rho_tor: &Array1<f64>, psi_profile: &Array1<f64>) -> Array1<f64> {
    let n_psi_norm: usize = psi_profile.len();

    let mut d_rho_tor_d_psi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    d_rho_tor_d_psi[0] = (rho_tor[1] - rho_tor[0]) / (psi_profile[1] - psi_profile[0]);
    for i_psi_norm in 1..n_psi_norm - 1 {
        d_rho_tor_d_psi[i_psi_norm] = (rho_tor[i_psi_norm + 1] - rho_tor[i_psi_norm - 1]) / (psi_profile[i_psi_norm + 1] - psi_profile[i_psi_norm - 1]);
    }
    d_rho_tor_d_psi[n_psi_norm - 1] = (rho_tor[n_psi_norm - 1] - rho_tor[n_psi_norm - 2]) / (psi_profile[n_psi_norm - 1] - psi_profile[n_psi_norm - 2]);

    d_rho_tor_d_psi
}

/// Fill the magnetic axis end of the nine profiles.
///
/// The axis is a point, so the flux surface average of a smooth quantity there is simply its value
/// at the axis. Five of the nine follow directly, using `b_p = 0` so that `b ** 2 = b_phi ** 2`.
///
/// The four which depend on `|grad(rho_tor)|` do not: `|grad(psi)|` vanishes at the axis while
/// `d(rho_tor)/d(psi)` diverges there, so the product is indeterminate even though its limit is
/// finite. Those are linearly extrapolated from the first two traced surfaces instead.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `f_profile` - the `f = r * b_phi` profile [tesla * metre]
/// * `gm_profiles` - the nine profiles, with index 0 still NaN; written into
fn epp_gm_at_magnetic_axis(time_slice: &EquilibriumTimeSlice, f_profile: &Array1<f64>, gm_profiles: &mut [Array1<f64>]) {
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;

    // `b_p = 0` on the magnetic axis, so the total field is the toroidal field alone
    let b_phi_axis: f64 = f_profile[0] / mag_r;
    let b_sq_axis: f64 = b_phi_axis.powi(2);

    gm_profiles[0][0] = 1.0 / mag_r.powi(2); // gm1
    gm_profiles[3][0] = 1.0 / b_sq_axis; // gm4
    gm_profiles[4][0] = b_sq_axis; // gm5
    gm_profiles[7][0] = mag_r; // gm8
    gm_profiles[8][0] = 1.0 / mag_r; // gm9

    // gm2, gm3, gm6 and gm7, by extrapolation from the first two traced surfaces
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    let mut i_first: Option<usize> = None;
    let mut i_second: Option<usize> = None;
    'traced_surface_loop: for i_psi_norm in 1..n_psi_norm {
        if gm_profiles[2][i_psi_norm].is_finite() {
            if i_first.is_none() {
                i_first = Some(i_psi_norm);
            } else {
                i_second = Some(i_psi_norm);
                break 'traced_surface_loop;
            }
        }
    }

    // Fewer than two traced surfaces: there is nothing to extrapolate from, so they stay NaN
    if i_first.is_none() || i_second.is_none() {
        return;
    }
    let i_first: usize = i_first.unwrap();
    let i_second: usize = i_second.unwrap();

    let psi_norm_span: f64 = psi_norm[i_second] - psi_norm[i_first];
    for i_gm in [1, 2, 5, 6] {
        let gradient: f64 = (gm_profiles[i_gm][i_second] - gm_profiles[i_gm][i_first]) / psi_norm_span;
        gm_profiles[i_gm][0] = gm_profiles[i_gm][i_first] + gradient * (psi_norm[0] - psi_norm[i_first]);
    }
}

/// Write the nine profiles into the time-slice, in `gm1` to `gm9` order.
fn store(time_slice: &mut EquilibriumTimeSlice, gm_profiles: &[Array1<f64>]) {
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
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// A circular flux surface of radius `a` centred at `(r_centre, 0)`, with a uniform `b_p`.
    ///
    /// With `b_p` uniform the `1 / b_p` weighting cancels, so the flux surface average collapses to
    /// a plain contour average and the `r` moments can be checked against their analytic values.
    #[test]
    fn a_circular_surface_with_uniform_bp_gives_the_analytic_r_moments() {
        let r_centre: f64 = 1.0;
        let a: f64 = 0.2;
        let n_theta: usize = 20_001;

        let mut gm8_numerator: f64 = 0.0; // <<r>>
        let mut gm9_numerator: f64 = 0.0; // <<1 / r>>
        let mut gm1_numerator: f64 = 0.0; // <<1 / r ** 2>>
        let mut denominator: f64 = 0.0;
        for i_theta in 1..n_theta {
            let theta_previous: f64 = 2.0 * PI * ((i_theta - 1) as f64) / ((n_theta - 1) as f64);
            let theta_here: f64 = 2.0 * PI * (i_theta as f64) / ((n_theta - 1) as f64);
            let r_previous: f64 = r_centre + a * theta_previous.cos();
            let r_here: f64 = r_centre + a * theta_here.cos();
            let delta_ell: f64 = a * (theta_here - theta_previous);

            denominator += delta_ell;
            gm8_numerator += 0.5 * delta_ell * (r_here + r_previous);
            gm9_numerator += 0.5 * delta_ell * (1.0 / r_here + 1.0 / r_previous);
            gm1_numerator += 0.5 * delta_ell * (1.0 / r_here.powi(2) + 1.0 / r_previous.powi(2));
        }

        // ∮ r d_theta / (2 * pi) = r_centre
        assert_abs_diff_eq!(gm8_numerator / denominator, r_centre, epsilon = 1e-9);
        // ∮ (1 / r) d_theta / (2 * pi) = 1 / sqrt(r_centre ** 2 - a ** 2)
        assert_abs_diff_eq!(gm9_numerator / denominator, 1.0 / (r_centre.powi(2) - a.powi(2)).sqrt(), epsilon = 1e-9);
        // ∮ (1 / r ** 2) d_theta / (2 * pi) = r_centre / (r_centre ** 2 - a ** 2) ** (3 / 2)
        assert_abs_diff_eq!(gm1_numerator / denominator, r_centre / (r_centre.powi(2) - a.powi(2)).powf(1.5), epsilon = 1e-9);
    }

    #[test]
    fn the_quantities_at_a_point_are_consistent_with_each_other() {
        let r_here: f64 = 0.5;
        let bp_here: f64 = 0.3;
        let f_here: f64 = -0.24;
        let d_rho_tor_d_psi_here: f64 = 2.5;

        let quantities: [f64; N_GM] = gm_quantities_at_point(r_here, bp_here, f_here, d_rho_tor_d_psi_here);

        let b_sq: f64 = bp_here.powi(2) + (f_here / r_here).powi(2);
        let grad_rho_tor: f64 = d_rho_tor_d_psi_here.abs() * 2.0 * PI * r_here * bp_here;

        assert_abs_diff_eq!(quantities[0], 1.0 / r_here.powi(2), epsilon = 1e-15); // gm1
        assert_abs_diff_eq!(quantities[1], grad_rho_tor.powi(2) / r_here.powi(2), epsilon = 1e-15); // gm2
        assert_abs_diff_eq!(quantities[2], grad_rho_tor.powi(2), epsilon = 1e-15); // gm3
        assert_abs_diff_eq!(quantities[3], 1.0 / b_sq, epsilon = 1e-15); // gm4
        assert_abs_diff_eq!(quantities[4], b_sq, epsilon = 1e-15); // gm5
        assert_abs_diff_eq!(quantities[5], grad_rho_tor.powi(2) / b_sq, epsilon = 1e-15); // gm6
        assert_abs_diff_eq!(quantities[6], grad_rho_tor, epsilon = 1e-15); // gm7
        assert_abs_diff_eq!(quantities[7], r_here, epsilon = 1e-15); // gm8
        assert_abs_diff_eq!(quantities[8], 1.0 / r_here, epsilon = 1e-15); // gm9

        // A negative `f` must not leak a sign into the squared quantities
        assert!(quantities[4] > 0.0);
        assert!(quantities[6] >= 0.0);
    }

    /// For `rho_tor = k * sqrt(psi)` the derivative is `0.5 * k / sqrt(psi)`, which is the shape
    /// `rho_tor = sqrt(phi / (pi * b0))` takes when `phi` is linear in `psi`
    #[test]
    fn d_rho_tor_d_psi_matches_the_analytic_derivative() {
        let k: f64 = 0.63;
        let psi_profile: Array1<f64> = Array1::linspace(0.0, 1.0, 501);
        let rho_tor: Array1<f64> = psi_profile.mapv(|psi| k * psi.sqrt());

        let d_rho_tor_d_psi: Array1<f64> = epp_d_rho_tor_d_psi(&rho_tor, &psi_profile);

        let i_check: usize = 250;
        let psi_check: f64 = psi_profile[i_check];
        let expected: f64 = 0.5 * k / psi_check.sqrt();

        assert_abs_diff_eq!(d_rho_tor_d_psi[i_check], expected, epsilon = 1e-6);
    }

    #[test]
    fn a_slice_with_two_traced_surfaces_extrapolates_the_gradient_quantities() {
        let psi_norm: Array1<f64> = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = psi_norm;
        time_slice.global_quantities.magnetic_axis.r = 0.5;

        let f_profile: Array1<f64> = Array1::from_elem(5, -0.24);

        // gm3 traced on surfaces 1 and 2 only, rising linearly by 2.0 per 0.25 of psi_norm
        let mut gm_profiles: Vec<Array1<f64>> = vec![Array1::from_elem(5, f64::NAN); N_GM];
        gm_profiles[2][1] = 3.0;
        gm_profiles[2][2] = 5.0;

        epp_gm_at_magnetic_axis(&time_slice, &f_profile, &mut gm_profiles);

        // The exact point values
        assert_abs_diff_eq!(gm_profiles[7][0], 0.5, epsilon = 1e-15); // gm8 = r_axis
        assert_abs_diff_eq!(gm_profiles[8][0], 2.0, epsilon = 1e-15); // gm9 = 1 / r_axis
        assert_abs_diff_eq!(gm_profiles[0][0], 4.0, epsilon = 1e-15); // gm1 = 1 / r_axis ** 2
        // b_phi = f / r = -0.48, so b ** 2 = 0.2304
        assert_abs_diff_eq!(gm_profiles[4][0], 0.2304, epsilon = 1e-12); // gm5
        assert_abs_diff_eq!(gm_profiles[3][0], 1.0 / 0.2304, epsilon = 1e-9); // gm4

        // Extrapolated back one step: 3.0 - 2.0
        assert_abs_diff_eq!(gm_profiles[2][0], 1.0, epsilon = 1e-12); // gm3
    }
}
