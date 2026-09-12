//! `time_slice(itime)/profiles_1d_r_midplane/dpressure_dpsi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the p′ profile along the mid-plane, and store it in the time-slice.
///
/// p′(ψ_N) is one of the two source functions the Grad-Shafranov equation is parameterised by, and
/// the solver has already fitted its degrees of freedom, so this is that source function evaluated
/// at the mid-plane's own `psi_norm` - exactly as `profiles_1d/dpressure_dpsi` evaluates it on the
/// flux-surface grid.
///
/// The mask zeroes the profile outside the plasma boundary, where no plasma current flows. It has
/// to: the solver masks `profiles_2d(0)/psi_norm` to zero outside the plasma, so an outside point
/// would otherwise be evaluated as though it sat on the magnetic axis.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d_r_midplane/dpressure_dpsi` is written into it
/// * `constant_values` - the constant values; `p_prime_source_function` is read
///
/// A time-slice which failed to converge carries `NaN` coefficients, so the profile comes out `NaN`
/// without needing a special case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let p_prime_source_function: &SharedSourceFunction = constant_values.p_prime_source_function;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_norm_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi_norm;
    let mask_2d: &Array2<f64> = &time_slice.profiles_2d[0].mask;

    // The mid-plane is the middle row of the grid
    let n_z: usize = psi_norm_2d.dim().0;
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
    let midplane_psi_norm: Array1<f64> = psi_norm_2d.row(i_z_centre).to_owned();
    let midplane_mask: Array1<f64> = mask_2d.row(i_z_centre).to_owned();

    let p_prime_dof_values: &Array1<f64> = &time_slice.source_functions.p_prime.coefficients;

    let p_prime_profile: Array1<f64> = p_prime_source_function.source_function_value(&midplane_psi_norm, p_prime_dof_values) * &midplane_mask;

    time_slice.profiles_1d_r_midplane.dpressure_dpsi = p_prime_profile;
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
    fn the_source_function_is_evaluated_on_the_mid_plane_row_and_masked_outside() {
        // The placeholder source function is `EfitPolynomial` with one degree of freedom, so
        // `p' = coefficient * (1 - psi_norm)`. With the coefficient below, `p' = 2 * (1 - psi_norm)`
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.source_functions.p_prime.coefficients = array![2.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The mid-plane row is `psi_norm = [0.0, 0.25, 0.5, 0.0]` with `mask = [0, 1, 1, 0]`
        let p_prime_profile: &Array1<f64> = &time_slice.profiles_1d_r_midplane.dpressure_dpsi;
        assert_abs_diff_eq!(p_prime_profile[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(p_prime_profile[1], 1.5, epsilon = 1e-15);
        assert_abs_diff_eq!(p_prime_profile[2], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(p_prime_profile[3], 0.0, epsilon = 1e-15);
    }
}
