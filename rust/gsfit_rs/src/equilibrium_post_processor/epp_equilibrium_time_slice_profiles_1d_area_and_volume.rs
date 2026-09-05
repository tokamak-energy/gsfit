//! `time_slice(itime)/profiles_1d/volume`, `.../dvolume_dpsi`, `.../area` and `.../darea_dpsi`

use super::epp_flux_surfaces::FluxSurface;
use geo::Area;
use geo::Centroid;
use geo::{Coord, LineString, Point, Polygon};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate the volume and cross-sectional area enclosed by each flux surface, and their
/// derivatives with respect to psi, and store them in the time-slice.
///
/// The enclosed area comes from the flux surface polygon directly, and Pappus's theorem turns it
/// into a volume:
///
/// ```text
/// volume = 2 * pi * r_centroid * area
/// ```
///
/// The magnetic axis (`psi_norm = 0`) is special-cased, because it is a point and so encloses
/// nothing. A flux surface which could not be found is left as NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the four `profiles_1d` nodes are written into it
/// * `flux_surfaces` - the flux surfaces from `epp_flux_surfaces`, one per `psi_norm`
///
/// **Must run after `epp_flux_surfaces`**, which supplies the surfaces, and after
/// `epp_equilibrium_time_slice_profiles_1d_psi`, which supplies `d_psi`.
pub fn epp_equilibrium_time_slice_profiles_1d_area_and_volume(time_slice: &mut EquilibriumTimeSlice, flux_surfaces: &[FluxSurface]) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let n_psi_norm: usize = psi_norm.len();

    // A slice which did not converge has no flux surfaces to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_1d.volume = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        time_slice.profiles_1d.dvolume_dpsi = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        time_slice.profiles_1d.area = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        time_slice.profiles_1d.darea_dpsi = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        return;
    }

    // The psi grid spacing, taken from the psi profile rather than recomputed, so that it cannot
    // disagree with it
    let psi_profile: &Array1<f64> = time_slice.profiles_1d.psi.as_ref().unwrap();
    let d_psi: f64 = psi_profile[1] - psi_profile[0];

    let mut volume_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut area_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // Set the volume at the magnetic axis (psiN=0) to be zero
    volume_profile[0] = 0.0;
    area_profile[0] = 0.0;

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
    }

    // Take derivatives
    let mut volume_prime_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    volume_prime_profile[0] = (volume_profile[0] - volume_profile[1]) / d_psi;
    for i_psi_norm in 1..n_psi_norm - 1 {
        volume_prime_profile[i_psi_norm] = (volume_profile[i_psi_norm - 1] - volume_profile[i_psi_norm + 1]) / (2.0 * d_psi);
    }
    volume_prime_profile[n_psi_norm - 1] = (volume_profile[n_psi_norm - 2] - volume_profile[n_psi_norm - 1]) / d_psi;

    // Note: `equilibrium_post_processor` in `plasma.rs` differences `volume_profile` here rather
    // than `area_profile`, so its `area_prime` is a second copy of `vol_prime`. That is a
    // copy-paste bug, fixed here, so this is the one quantity where the two post-processors
    // deliberately disagree
    let mut area_prime_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    area_prime_profile[0] = (area_profile[0] - area_profile[1]) / d_psi;
    for i_psi_norm in 1..n_psi_norm - 1 {
        area_prime_profile[i_psi_norm] = (area_profile[i_psi_norm - 1] - area_profile[i_psi_norm + 1]) / (2.0 * d_psi);
    }
    area_prime_profile[n_psi_norm - 1] = (area_profile[n_psi_norm - 2] - area_profile[n_psi_norm - 1]) / d_psi;

    time_slice.profiles_1d.volume = Some(volume_profile);
    time_slice.profiles_1d.dvolume_dpsi = Some(volume_prime_profile);
    time_slice.profiles_1d.area = Some(area_profile);
    time_slice.profiles_1d.darea_dpsi = Some(area_prime_profile);
}
