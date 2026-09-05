//! `time_slice(itime)/global_quantities/q_axis`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the safety factor on the magnetic axis in the time-slice.
///
/// This is the first point of the safety factor profile, which
/// `epp_equilibrium_time_slice_profiles_1d_q` calculates from the curvature of `psi` at the axis
/// rather than by a flux surface integral.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/q_axis` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_q`**, which supplies `q`.
pub fn epp_equilibrium_time_slice_global_quantities_q_axis(time_slice: &mut EquilibriumTimeSlice) {
    let q_profile: &Array1<f64> = time_slice.profiles_1d.q.as_ref().unwrap();

    time_slice.global_quantities.q_axis = Some(q_profile[0]);
}
