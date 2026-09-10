//! `time_slice(itime)/global_quantities/q_axis`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the safety factor on the magnetic axis in the time-slice.
///
/// This is the first point of the safety factor profile, which
/// `profiles_1d::q::calculate` calculates from the curvature of `psi` at the axis
/// rather than by a flux surface integral.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/q_axis` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let q_profile: &Array1<f64> = time_slice.profiles_1d.q.as_ref().unwrap();

    time_slice.global_quantities.q_axis = Some(q_profile[0]);
}
