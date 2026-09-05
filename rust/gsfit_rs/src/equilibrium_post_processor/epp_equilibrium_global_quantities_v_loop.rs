//! `time_slice(itime)/global_quantities/v_loop`

use imas_rs::Equilibrium;
use ndarray::Array1;

/// Calculate the loop voltage at the plasma boundary, and store it in every time-slice.
///
/// ```text
/// v_loop = -d(boundary/psi)/d(time)
/// ```
///
/// Unlike every other helper here this one needs all of the time-slices at once, because it
/// differentiates across them, so it runs after the per-slice loop rather than inside it.
///
/// # Arguments
/// * `equilibrium_ids` - the solved equilibrium IDS; `global_quantities/v_loop` is written into
///   every time-slice
pub fn epp_equilibrium_global_quantities_v_loop(equilibrium_ids: &mut Equilibrium) {
    // v_loop = - d(psi_b)/d(time)

    // Note: when the time-slice is "user_defined", the time-vecor can have variable time steps
    let n_time: usize = equilibrium_ids.time_slice.len();

    let time: Array1<f64> = equilibrium_ids.time_slice.iter().map(|time_slice| time_slice.time.unwrap()).collect();
    let psi_b: Array1<f64> = equilibrium_ids.time_slice.iter().map(|time_slice| time_slice.boundary.psi.unwrap()).collect();

    let mut v_loop: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

    // Exit if we only have one time-slice
    if n_time == 1 {
        equilibrium_ids.time_slice[0].global_quantities.v_loop = Some(v_loop[0]);
        return;
    }

    // forward/backward differences for the first time point
    v_loop[0] = -(psi_b[1] - psi_b[0]) / (time[1] - time[0]);
    // Central differencing for the rest
    for i_time in 1..n_time - 1 {
        let d_psi_b: f64 = -(psi_b[i_time + 1] - psi_b[i_time - 1]);
        let d_time: f64 = time[i_time + 1] - time[i_time - 1];
        v_loop[i_time] = d_psi_b / d_time;
    }
    // forward/backward difference for the last time point
    v_loop[n_time - 1] = -(psi_b[n_time - 1] - psi_b[n_time - 2]) / (time[n_time - 1] - time[n_time - 2]);

    for i_time in 0..n_time {
        equilibrium_ids.time_slice[i_time].global_quantities.v_loop = Some(v_loop[i_time]);
    }
}
