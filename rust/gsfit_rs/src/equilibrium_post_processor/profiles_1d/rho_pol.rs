//! `time_slice(itime)/profiles_1d/rho_pol`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the normalised poloidal flux radius profile, and store it in the time-slice.
///
/// This is simply `sqrt(psi_norm)`, the poloidal counterpart of `rho_tor_norm`. It is a custom key,
/// because the data dictionary does not define it.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_pol` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;

    let rho_pol_profile: Array1<f64> = psi_norm.mapv(|x| x.sqrt());

    time_slice.profiles_1d.rho_pol = rho_pol_profile;
}
