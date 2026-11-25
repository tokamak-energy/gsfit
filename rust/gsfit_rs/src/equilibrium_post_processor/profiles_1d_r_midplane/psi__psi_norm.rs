//! `time_slice(itime)/profiles_1d_r_midplane/psi` and `.../psi_norm`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Take the poloidal flux along the mid-plane, normalise it, and store both in the time-slice.
///
/// `psi` is the solved grid's own `profiles_2d/psi` along its middle row, `floor(n_z / 2)`. The
/// poloidal flux is defined everywhere on the grid, so it is not masked to the plasma.
///
/// `psi_norm` is normalised here from `psi`, and **not** taken from `profiles_2d/psi_norm`, which the
/// solver masks to the plasma and so leaves at zero everywhere outside it. Outside the plasma
/// boundary this one carries on past 1.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `profiles_1d_r_midplane` nodes are written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

    // The mid-plane is the middle row of the grid
    let (n_z, n_r): (usize, usize) = psi_2d.dim();
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
    let midplane_psi: Array1<f64> = psi_2d.row(i_z_centre).to_owned();

    let d_psi_d_psi_norm: f64 = psi_b - psi_a;
    let mut midplane_psi_norm: Array1<f64> = Array1::from_elem(n_r, f64::NAN);
    for i_r in 0..n_r {
        midplane_psi_norm[i_r] = (midplane_psi[i_r] - psi_a) / d_psi_d_psi_norm;
    }

    time_slice.profiles_1d_r_midplane.psi = midplane_psi;
    time_slice.profiles_1d_r_midplane.psi_norm = midplane_psi_norm;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::super::time_slice_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn psi_is_the_middle_row_and_psi_norm_is_not_masked() {
        // `psi_magnetic_axis = 0` and `boundary/psi = -1`, so `psi_norm = -psi`. The middle row's two
        // inside points match the solver's `profiles_2d/psi_norm = [0.0, 0.25, 0.5, 0.0]`, and its two
        // outside points, where the solver leaves zero, carry on past 1
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_2d[0].psi = array![[9.0, 9.0, 9.0, 9.0], [-1.5, -0.25, -0.5, -1.25], [9.0, 9.0, 9.0, 9.0]];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let midplane_psi: &Array1<f64> = &time_slice.profiles_1d_r_midplane.psi;
        let midplane_psi_norm: &Array1<f64> = &time_slice.profiles_1d_r_midplane.psi_norm;
        let midplane_psi_expected: [f64; 4] = [-1.5, -0.25, -0.5, -1.25];
        let midplane_psi_norm_expected: [f64; 4] = [1.5, 0.25, 0.5, 1.25];
        assert_eq!(midplane_psi.len(), 4);
        assert_eq!(midplane_psi_norm.len(), 4);
        for i_r in 0..4 {
            assert_abs_diff_eq!(midplane_psi[i_r], midplane_psi_expected[i_r], epsilon = 0.0);
            assert_abs_diff_eq!(midplane_psi_norm[i_r], midplane_psi_norm_expected[i_r], epsilon = 1e-15);
        }
    }
}
