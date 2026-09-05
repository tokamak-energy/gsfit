//! `time_slice(itime)/profiles_1d/rho_tor`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use ndarray_stats::QuantileExt;

/// Calculate the toroidal flux radius profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_tor` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_phi`**, which supplies the toroidal
/// flux. A slice which did not converge carries NaN in the toroidal flux, so `max` is undefined and
/// the profile comes out NaN without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_1d_rho_tor(time_slice: &mut EquilibriumTimeSlice) {
    let flux_tor_profile: &Array1<f64> = time_slice.profiles_1d.phi.as_ref().unwrap();
    let n_psi_n: usize = flux_tor_profile.len();

    let flux_tor_max: Result<&f64, ndarray_stats::errors::MinMaxError> = flux_tor_profile.max();
    if flux_tor_max.is_err() {
        time_slice.profiles_1d.rho_tor = Some(Array1::from_elem(n_psi_n, f64::NAN));
        return;
    }

    let rho_tor: Array1<f64> = (flux_tor_profile / flux_tor_max.unwrap().to_owned()).mapv(|x| x.sqrt());

    time_slice.profiles_1d.rho_tor = Some(rho_tor);
}
