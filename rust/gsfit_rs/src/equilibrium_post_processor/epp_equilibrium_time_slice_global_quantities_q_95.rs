//! `time_slice(itime)/global_quantities/q_95`

use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, array};

/// Calculate the safety factor at `psi_norm = 0.95`, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/q_95` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_q`**, which supplies `q`.
pub fn epp_equilibrium_time_slice_global_quantities_q_95(time_slice: &mut EquilibriumTimeSlice) {
    // A slice which did not converge has a NaN safety factor profile, which the interpolator
    // cannot be built from
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.global_quantities.q_95 = Some(f64::NAN);
        return;
    }

    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let q_profile: &Array1<f64> = time_slice.profiles_1d.q.as_ref().unwrap();

    let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(psi_norm.clone(), q_profile.clone()).unwrap();

    let psi_95: Array1<f64> = array![0.95];
    let q95: f64 = interpolator.interpolate_array1(&psi_95).unwrap()[0];

    time_slice.global_quantities.q_95 = Some(q95);
}
