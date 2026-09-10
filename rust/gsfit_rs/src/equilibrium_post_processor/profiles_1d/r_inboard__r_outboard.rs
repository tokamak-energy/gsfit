//! `time_slice(itime)/profiles_1d/r_inboard` and `.../r_outboard`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the major radius of each flux surface at the height of the magnetic axis, on each side
/// of it, and store them in the time-slice.
///
/// The surface is walked segment by segment and every segment which straddles `z = mag_z` is
/// interpolated for its `r`. A nested surface is crossed exactly twice, so `r_inboard` is the
/// smaller crossing and `r_outboard` the larger; taking the extremes rather than assuming two
/// crossings keeps it right for a surface which doubles back across the height of the axis.
///
/// The magnetic axis (`psi_norm = 0`) is a point, so both radii are `mag_r` there.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `r_inboard` and `r_outboard` are written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
///
/// A surface which was not found, or which does not reach the height of the magnetic axis, is left
/// as NaN. That is most likely on the innermost surfaces, where the contour is only a few grid
/// cells across.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let n_psi_norm: usize = psi_norm.len();

    let mut r_inboard_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut r_outboard_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // A slice which did not converge has no flux surfaces to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_1d.r_inboard = Some(r_inboard_profile);
        time_slice.profiles_1d.r_outboard = Some(r_outboard_profile);
        return;
    }

    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    // The magnetic axis is a point, so both radii collapse onto it
    r_inboard_profile[0] = mag_r;
    r_outboard_profile[0] = mag_r;

    'psi_norm_loop: for i_psi_norm in 1..n_psi_norm {
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;
        let fs_n: usize = fs_r.len();

        // A flux surface which could not be found is stored with zero points, and is left as NaN
        if fs_n < 2 {
            continue 'psi_norm_loop;
        }

        let mut r_crossing_min: f64 = f64::INFINITY;
        let mut r_crossing_max: f64 = f64::NEG_INFINITY;

        // The ring is closed, so the last point repeats the first and every segment is covered
        'segment_loop: for i_fs in 1..fs_n {
            let delta_z_from: f64 = fs_z[i_fs - 1] - mag_z;
            let delta_z_to: f64 = fs_z[i_fs] - mag_z;

            // A segment crosses the height of the axis when its two ends fall on opposite sides.
            // A point sitting exactly on that height counts as being below it, which is what stops
            // a surface which only touches the height from registering as two crossings
            if (delta_z_from > 0.0) == (delta_z_to > 0.0) {
                continue 'segment_loop;
            }

            let fraction: f64 = delta_z_from / (delta_z_from - delta_z_to);
            let r_crossing: f64 = fs_r[i_fs - 1] + fraction * (fs_r[i_fs] - fs_r[i_fs - 1]);

            r_crossing_min = r_crossing_min.min(r_crossing);
            r_crossing_max = r_crossing_max.max(r_crossing);
        }

        // A surface which does not reach the height of the magnetic axis is left as NaN
        if !r_crossing_min.is_finite() {
            continue 'psi_norm_loop;
        }

        r_inboard_profile[i_psi_norm] = r_crossing_min;
        r_outboard_profile[i_psi_norm] = r_crossing_max;
    }

    time_slice.profiles_1d.r_inboard = Some(r_inboard_profile);
    time_slice.profiles_1d.r_outboard = Some(r_outboard_profile);
}
