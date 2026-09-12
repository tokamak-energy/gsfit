//! `time_slice(itime)/profiles_1d/volume`, `.../dvolume_dpsi`, `.../dvolume_drho_tor`,
//! `.../area`, `.../darea_dpsi`, `.../darea_drho_tor` and `.../surface`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use geo::Area;
use geo::Centroid;
use geo::{Coord, LineString, Point, Polygon};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate the volume and cross-sectional area enclosed by each flux surface, the area of the
/// surface itself, and the derivatives of the enclosed volume and area with respect to both radial
/// coordinates, and store them in the time-slice.
///
/// The enclosed area comes from the flux surface polygon directly, and Pappus's theorem turns it
/// into a volume:
///
/// ```text
/// volume = 2 * pi * r_centroid * area
/// ```
///
/// The area of the surface itself is the area of the surface of revolution the contour sweeps out,
/// which is the same theorem applied to the contour rather than to the region it encloses:
///
/// ```text
/// surface = ∮ 2 * pi * r d_ell
/// ```
///
/// The magnetic axis (`psi_norm = 0`) is special-cased, because it is a point: it encloses nothing
/// and sweeps out nothing. A flux surface which could not be found is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the seven `profiles_1d` nodes above are written into it
/// * `flux_surfaces` - the flux surfaces from `flux_surfaces::calculate`, one per `psi_norm`
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    // A slice which did not converge has no flux surfaces to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        let nan_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        time_slice.profiles_1d.volume = nan_profile.clone();
        time_slice.profiles_1d.dvolume_dpsi = nan_profile.clone();
        time_slice.profiles_1d.dvolume_drho_tor = nan_profile.clone();
        time_slice.profiles_1d.area = nan_profile.clone();
        time_slice.profiles_1d.darea_dpsi = nan_profile.clone();
        time_slice.profiles_1d.darea_drho_tor = nan_profile.clone();
        time_slice.profiles_1d.surface = nan_profile;
        return;
    }

    // The psi grid spacing, taken from the psi profile rather than recomputed, so that it cannot
    // disagree with it
    let psi_profile: &Array1<f64> = &time_slice.profiles_1d.psi;
    let d_psi: f64 = psi_profile[1] - psi_profile[0];

    let mut volume_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut area_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut surface_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // Set the volume at the magnetic axis (psiN=0) to be zero
    volume_profile[0] = 0.0;
    area_profile[0] = 0.0;
    surface_profile[0] = 0.0;

    // Don't do the first point
    for i_psi_norm in 1..n_psi_norm {
        let flux_surface: &FluxSurface = &flux_surfaces[i_psi_norm];

        // A flux surface which could not be found is stored with zero points, and is left as NaN
        if flux_surface.r.is_empty() {
            continue;
        }

        let flux_surface_coordinates: Vec<Coord<f64>> = flux_surface.r.iter().zip(flux_surface.z.iter()).map(|(&x, &y)| Coord { x, y }).collect();
        let flux_surface_polygon: Polygon = Polygon::new(
            LineString::from(flux_surface_coordinates),
            vec![], // No holes
        );

        // Calculate the area of the flux surface
        let area: f64 = flux_surface_polygon.unsigned_area();

        let mass_centroid: Point = flux_surface_polygon.centroid().unwrap();
        let mass_centroid_r: f64 = mass_centroid.x();

        // Calculate the volume
        area_profile[i_psi_norm] = area;
        volume_profile[i_psi_norm] = 2.0 * PI * mass_centroid_r * area;
        surface_profile[i_psi_norm] = epp_surface_of_revolution(&flux_surface.r, &flux_surface.z);
    }

    // Take derivatives.
    // `d_psi = psi_profile[1] - psi_profile[0]`, so the numerators must also run in increasing
    // index order: a forward difference at the first point, central differences in the interior,
    // and a backward difference at the last point.
    let mut volume_prime_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    volume_prime_profile[0] = (volume_profile[1] - volume_profile[0]) / d_psi;
    for i_psi_norm in 1..n_psi_norm - 1 {
        volume_prime_profile[i_psi_norm] = (volume_profile[i_psi_norm + 1] - volume_profile[i_psi_norm - 1]) / (2.0 * d_psi);
    }
    volume_prime_profile[n_psi_norm - 1] = (volume_profile[n_psi_norm - 1] - volume_profile[n_psi_norm - 2]) / d_psi;

    let mut area_prime_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    area_prime_profile[0] = (area_profile[1] - area_profile[0]) / d_psi;
    for i_psi_norm in 1..n_psi_norm - 1 {
        area_prime_profile[i_psi_norm] = (area_profile[i_psi_norm + 1] - area_profile[i_psi_norm - 1]) / (2.0 * d_psi);
    }
    area_prime_profile[n_psi_norm - 1] = (area_profile[n_psi_norm - 1] - area_profile[n_psi_norm - 2]) / d_psi;

    // The same two derivatives, against the other radial coordinate
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;
    let volume_prime_rho_tor_profile: Array1<f64> = epp_d_profile_d_rho_tor(&volume_profile, rho_tor);
    let area_prime_rho_tor_profile: Array1<f64> = epp_d_profile_d_rho_tor(&area_profile, rho_tor);

    time_slice.profiles_1d.volume = volume_profile;
    time_slice.profiles_1d.dvolume_dpsi = volume_prime_profile;
    time_slice.profiles_1d.dvolume_drho_tor = volume_prime_rho_tor_profile;
    time_slice.profiles_1d.area = area_profile;
    time_slice.profiles_1d.darea_dpsi = area_prime_profile;
    time_slice.profiles_1d.darea_drho_tor = area_prime_rho_tor_profile;
    time_slice.profiles_1d.surface = surface_profile;
}

/// Calculate the area of the surface of revolution swept out by one closed contour.
///
/// ```text
/// surface = ∮ 2 * pi * r d_ell
/// ```
///
/// integrated trapezoidally, segment by segment, around the contour.
///
/// # Arguments
/// * `fs_r` - radial coordinates of the closed contour [metre]
/// * `fs_z` - vertical coordinates of the closed contour [metre]
///
/// # Returns
/// * `surface` - the area of the toroidal surface [metre ** 2]
fn epp_surface_of_revolution(fs_r: &Array1<f64>, fs_z: &Array1<f64>) -> f64 {
    let n_fs: usize = fs_r.len();

    // The ring is closed, so the last point repeats the first and every segment is covered
    let mut surface: f64 = 0.0;
    for i_fs in 1..n_fs {
        let delta_ell: f64 = (fs_r[i_fs] - fs_r[i_fs - 1]).hypot(fs_z[i_fs] - fs_z[i_fs - 1]);
        surface += 2.0 * PI * 0.5 * (fs_r[i_fs] + fs_r[i_fs - 1]) * delta_ell;
    }

    surface
}

/// Differentiate a profile with respect to `rho_tor`, one value per flux surface.
///
/// # Arguments
/// * `profile` - the quantity to differentiate, one value per flux surface
/// * `rho_tor` - the toroidal flux coordinate profile, `profiles_1d/rho_tor` [metre]
///
/// # Returns
/// * `d_profile_d_rho_tor` - one value per flux surface, in the units of `profile` per metre
///
/// `rho_tor` increases outwards from the magnetic axis but is **not** evenly spaced - it goes
/// roughly as the square root of `psi_norm` - so each difference is divided by the span of
/// `rho_tor` it was taken over rather than by a single grid spacing: a forward difference at the
/// first point, central differences in the interior, and a backward difference at the last point.
///
/// Written this way the result is identical to `d(profile)/d(psi) / (d(rho_tor)/d(psi))` taken with
/// the same stencils, because the `d(psi)` of the two cancels exactly. Doing it in one step avoids
/// the `0 / 0` that chain rule would hit at the magnetic axis, where `d(rho_tor)/d(psi)` diverges.
fn epp_d_profile_d_rho_tor(profile: &Array1<f64>, rho_tor: &Array1<f64>) -> Array1<f64> {
    let n_psi_norm: usize = rho_tor.len();

    let mut d_profile_d_rho_tor: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    d_profile_d_rho_tor[0] = (profile[1] - profile[0]) / (rho_tor[1] - rho_tor[0]);
    for i_psi_norm in 1..n_psi_norm - 1 {
        d_profile_d_rho_tor[i_psi_norm] = (profile[i_psi_norm + 1] - profile[i_psi_norm - 1]) / (rho_tor[i_psi_norm + 1] - rho_tor[i_psi_norm - 1]);
    }
    d_profile_d_rho_tor[n_psi_norm - 1] = (profile[n_psi_norm - 1] - profile[n_psi_norm - 2]) / (rho_tor[n_psi_norm - 1] - rho_tor[n_psi_norm - 2]);

    d_profile_d_rho_tor
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// A circular contour of minor radius `a` about `(r_geo, 0)` sweeps out a torus, whose surface
    /// area is `4 * pi ** 2 * r_geo * a`
    #[test]
    fn a_circular_contour_sweeps_out_the_area_of_a_torus() {
        let r_geo: f64 = 0.5;
        let a: f64 = 0.2;

        // Closed: the last point repeats the first
        let n_theta: usize = 20_001;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_theta);
        let fs_r: Array1<f64> = r_geo + a * theta.mapv(f64::cos);
        let fs_z: Array1<f64> = a * theta.mapv(f64::sin);

        let surface: f64 = epp_surface_of_revolution(&fs_r, &fs_z);

        assert_abs_diff_eq!(surface, 4.0 * PI.powi(2) * r_geo * a, epsilon = 1e-6);
    }

    /// For `profile = k * rho_tor ** 2`, which is the shape the enclosed area takes near the
    /// magnetic axis, the derivative is `2 * k * rho_tor`
    #[test]
    fn the_rho_tor_derivative_matches_the_analytic_derivative_on_an_uneven_grid() {
        let k: f64 = 1.7;

        // Unevenly spaced, as `rho_tor` really is: `rho_tor` goes as the square root of `psi_norm`
        let psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, 2001);
        let rho_tor: Array1<f64> = psi_norm.mapv(|psi_norm_here| 0.4 * psi_norm_here.sqrt());
        let profile: Array1<f64> = rho_tor.mapv(|rho_tor_here| k * rho_tor_here.powi(2));

        let d_profile_d_rho_tor: Array1<f64> = epp_d_profile_d_rho_tor(&profile, &rho_tor);

        // In the interior, where the central difference is used. It is exact for a quadratic on an
        // even grid, so all that is left is the unevenness of `rho_tor` itself
        assert_abs_diff_eq!(d_profile_d_rho_tor[1000], 2.0 * k * rho_tor[1000], epsilon = 1e-6);

        // At the boundary, where the one-sided backward difference is used. That is first order in
        // the grid spacing, so it is a couple of orders of magnitude less accurate
        assert_abs_diff_eq!(d_profile_d_rho_tor[2000], 2.0 * k * rho_tor[2000], epsilon = 1e-3);

        // The forward difference at the magnetic axis: `(k * rho_tor[1] ** 2 - 0) / rho_tor[1]`
        assert_abs_diff_eq!(d_profile_d_rho_tor[0], k * rho_tor[1], epsilon = 1e-12);
    }

    #[test]
    fn a_slice_which_did_not_converge_is_all_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.5, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = f64::NAN;

        let flux_surface_empty: FluxSurface = FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };
        let flux_surfaces: Vec<FluxSurface> = vec![flux_surface_empty; 3];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        // Including the magnetic axis, which is only filled once there is a converged solution
        assert!(time_slice.profiles_1d.surface.iter().all(|value| value.is_nan()));
        assert!(time_slice.profiles_1d.dvolume_drho_tor.iter().all(|value| value.is_nan()));
        assert!(time_slice.profiles_1d.darea_drho_tor.iter().all(|value| value.is_nan()));
    }
}
