//! `time_slice(itime)/profiles_1d/dpressure_dpsi`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the p′ profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/dpressure_dpsi` is written into it
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
///
/// A time-slice which failed to converge carries `NaN` coefficients, so `dpressure_dpsi` comes out
/// `NaN` without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_1d_dpressure_dpsi(time_slice: &mut EquilibriumTimeSlice, p_prime_source_function: &SharedSourceFunction) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let p_prime_dof_values: &Array1<f64> = time_slice.source_functions.p_prime.coefficients.as_ref().unwrap();

    // p′(ψ_N) is one of the two source functions the Grad-Shafranov equation is parameterised by,
    // and the solver has already fitted its degrees of freedom. So the profile is just the source
    // function evaluated on the ψ_N grid - no integration or change of variable is needed, unlike
    // the pressure itself.
    let p_prime_profile: Array1<f64> = p_prime_source_function.source_function_value(psi_norm, p_prime_dof_values);

    time_slice.profiles_1d.dpressure_dpsi = Some(p_prime_profile);
}
