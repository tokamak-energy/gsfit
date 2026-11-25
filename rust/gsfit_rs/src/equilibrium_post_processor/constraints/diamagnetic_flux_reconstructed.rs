//! `time_slice(itime)/constraints/diamagnetic_flux/reconstructed`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;

/// Store the diamagnetic flux in the time-slice.
///
/// The diamagnetic flux is the toroidal flux which the plasma's poloidal current adds to that of the
/// vacuum toroidal field, which is `boundary/phi_poloidal_current`, so it is copied from there.
///
/// This is the reconstruction's prediction of the diamagnetic loop signal, which is why it is also
/// stored under `constraints`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/phi_poloidal_current` is read, and
///   `constraints/diamagnetic_flux/reconstructed` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    time_slice.constraints.diamagnetic_flux.reconstructed = time_slice.boundary.phi_poloidal_current;
}
