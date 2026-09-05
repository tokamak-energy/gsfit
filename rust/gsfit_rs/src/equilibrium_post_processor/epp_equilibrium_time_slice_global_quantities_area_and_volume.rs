//! `time_slice(itime)/global_quantities/area` and `.../volume`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the plasma cross-sectional area and volume in the time-slice.
///
/// Both are the value enclosed by the last closed flux surface, which is the final point of the
/// corresponding profile.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `global_quantities` nodes are written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_area_and_volume`**, which supplies the
/// profiles.
pub fn epp_equilibrium_time_slice_global_quantities_area_and_volume(time_slice: &mut EquilibriumTimeSlice) {
    let area_profile: &Array1<f64> = time_slice.profiles_1d.area.as_ref().unwrap();
    let volume_profile: &Array1<f64> = time_slice.profiles_1d.volume.as_ref().unwrap();

    let area: f64 = area_profile.last().unwrap().to_owned();
    let volume: f64 = volume_profile.last().unwrap().to_owned();

    time_slice.global_quantities.area = Some(area);
    time_slice.global_quantities.volume = Some(volume);
}
