//! `time_slice(itime)/global_quantities/pressure_2d_sum`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array2;

/// Sum the 2D pressure over every grid cell, and store it in the time-slice.
///
/// This is a plain sum rather than an integral: the cells are not weighted by their volume, so it
/// is a diagnostic of the 2D pressure rather than a physical quantity. It is kept because GSFit has
/// always reported it, as `global/p`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/pressure_2d_sum` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_2d_pressure`**, which supplies the 2D
/// pressure.
pub fn epp_equilibrium_time_slice_global_quantities_pressure_2d_sum(time_slice: &mut EquilibriumTimeSlice) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let p_2d: &Array2<f64> = time_slice.profiles_2d[0].pressure.as_ref().unwrap();

    let pressure_2d_sum: f64 = p_2d.sum();

    time_slice.global_quantities.pressure_2d_sum = Some(pressure_2d_sum);
}
