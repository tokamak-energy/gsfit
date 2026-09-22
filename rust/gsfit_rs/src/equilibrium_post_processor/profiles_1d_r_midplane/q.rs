//! `time_slice(itime)/profiles_1d_r_midplane/q`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::interpolate_profile::interpolate_profile;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Map the safety factor onto the mid-plane, and store it in the time-slice.
///
/// `q` is a flux-surface quantity: unlike the pressure and the source functions it has no closed
/// form to evaluate at a point, because it comes from an integral around a whole flux surface. So
/// the completed `profiles_1d/q` is linearly interpolated in `psi_norm` at each mid-plane point,
/// exactly as `profiles_2d(0)/phi` maps the toroidal flux onto the grid.
///
/// Outside the plasma boundary there is no closed flux surface, so those points are left `NaN`.
/// Inside and outside are decided by `profiles_2d(0)/mask`, not by the value of `psi_norm`: the
/// solver masks `psi_norm` to zero outside the plasma, which would otherwise place every outside
/// point on the magnetic axis. An inside point's `psi_norm` is clamped onto the profile's domain
/// before interpolating, because it can sit a rounding error outside `[0, 1]`; the reasons are the
/// ones set out in `profiles_2d/phi.rs`.
///
/// `profiles_1d/q` is deliberately `NaN` at `psi_norm = 1`, where a diverted plasma's safety factor
/// is singular, so the mid-plane points in the last interval of the profile are `NaN` too.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d_r_midplane/q` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_norm_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi_norm;
    let mask_2d: &Array2<f64> = &time_slice.profiles_2d[0].mask;

    // The mid-plane is the middle row of the grid
    let (n_z, n_r): (usize, usize) = psi_norm_2d.dim();
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
    let midplane_psi_norm: Array1<f64> = psi_norm_2d.row(i_z_centre).to_owned();
    let midplane_mask: Array1<f64> = mask_2d.row(i_z_centre).to_owned();

    let psi_norm_profile: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let q_profile: &Array1<f64> = &time_slice.profiles_1d.q;

    let n_psi_norm: usize = psi_norm_profile.len();
    assert!(
        n_psi_norm >= 2,
        "profiles_1d_r_midplane/q: `profiles_1d/psi_norm` needs at least two points, has {n_psi_norm}"
    );
    assert_eq!(
        q_profile.len(),
        n_psi_norm,
        "profiles_1d_r_midplane/q: `profiles_1d/q` and `profiles_1d/psi_norm` differ in length"
    );
    let psi_norm_profile_min: f64 = psi_norm_profile[0];
    let psi_norm_profile_max: f64 = psi_norm_profile[n_psi_norm - 1];

    let mut midplane_q: Array1<f64> = Array1::from_elem(n_r, f64::NAN);

    for i_r in 0..n_r {
        if midplane_mask[i_r] > 0.99 {
            let psi_norm_here: f64 = midplane_psi_norm[i_r].clamp(psi_norm_profile_min, psi_norm_profile_max);
            midplane_q[i_r] = interpolate_profile(psi_norm_here, psi_norm_profile, q_profile);
        }
    }

    time_slice.profiles_1d_r_midplane.q = midplane_q;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::super::time_slice_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn inside_points_interpolate_the_profile_and_outside_points_are_nan() {
        // `profiles_1d/q = [1, 2, 8]` at `psi_norm = [0, 0.5, 1]`
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The mid-plane row is `psi_norm = [0.0, 0.25, 0.5, 0.0]` with `mask = [0, 1, 1, 0]`, so the
        // two outside points keep the zeroed `psi_norm` the solver leaves there and must not be
        // read as sitting on the magnetic axis
        let midplane_q: &Array1<f64> = &time_slice.profiles_1d_r_midplane.q;
        assert!(midplane_q[0].is_nan());
        assert_abs_diff_eq!(midplane_q[1], 1.5, epsilon = 1e-15);
        assert_abs_diff_eq!(midplane_q[2], 2.0, epsilon = 1e-15);
        assert!(midplane_q[3].is_nan());
    }
}
