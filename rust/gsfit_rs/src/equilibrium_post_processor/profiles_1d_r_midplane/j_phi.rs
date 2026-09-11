//! `time_slice(itime)/profiles_1d_r_midplane/j_phi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Take the toroidal current density along the mid-plane, and store it in the time-slice.
///
/// This is a cut through the solved `profiles_2d(0)/j_phi`, not an average of it, which is what
/// makes it different in kind from `profiles_1d/j_phi`: that one is the flux-surface average the
/// data dictionary defines, this one is the current density at each point along the line.
///
/// The solver builds `profiles_2d(0)/j_phi` as `2 * pi * (r * p' + ff' / (mu_0 * r)) * mask`, so it
/// is already zero outside the plasma boundary and nothing is masked here.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d_r_midplane/j_phi` is written into it
///
/// A time-slice which failed to converge carries a `NaN` current density, so the profile comes out
/// `NaN` without needing a special case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let j_phi_2d: &Array2<f64> = &time_slice.profiles_2d[0].j_phi;

    // The mid-plane is the middle row of the grid
    let n_z: usize = j_phi_2d.dim().0;
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
    let j_phi_profile: Array1<f64> = j_phi_2d.row(i_z_centre).to_owned();

    time_slice.profiles_1d_r_midplane.j_phi = j_phi_profile;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::super::time_slice_for_test;
    use super::*;

    #[test]
    fn the_mid_plane_row_of_the_solved_current_density_is_taken() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The rows either side of the mid-plane hold 99, which no correct answer contains
        let j_phi_profile: &Array1<f64> = &time_slice.profiles_1d_r_midplane.j_phi;
        assert_eq!(j_phi_profile.to_vec(), vec![0.0, 3.0, 4.0, 0.0]);
    }
}
