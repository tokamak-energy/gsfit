//! `time_slice(itime)/profiles_1d/j_parallel`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the flux-surface-averaged parallel current density, and store it in the time-slice.
///
/// The data dictionary defines this profile as `<j . b> / b0`, with `b0` the vacuum toroidal field at the machine's reference radius. In the
/// equilibrium, with `psi` the total flux rather than the flux per radian,
///
/// ```text
/// j . b = 2 * pi * f' * b_p ** 2 / mu_0 + j_phi * f / R,    j_phi = 2 * pi * R * p' + 2 * pi * ff' / (mu_0 * R);
/// ```
///
/// see `profiles_2d::j_parallel`. `f`, `f' = ff' / f` and `p'` are constant on a flux surface, and `<b_p ** 2> = <b ** 2> - f ** 2 * <1 / R ** 2>`,
/// so
///
/// ```text
/// <j . b> / b0 = 2 * pi * (f * p' + ff' * gm5 / (mu_0 * f)) / b0,
/// ```
///
/// with `gm5 = <b ** 2>`. This is exact for the equilibrium, where an average of `profiles_2d(0)/j_parallel` would also average how the
/// current density is discretised on the grid; it is also how the reconstructed `constraints/j_parallel` is calculated.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/f`, `.../dpressure_dpsi`, `.../f_df_dpsi` and `.../gm5` are read, and
///   `profiles_1d/j_parallel` is written into it
/// * `constant_values` - the constant values; `b0`, `vacuum_toroidal_field/b0` [tesla], is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let b0: f64 = constant_values.b0;
    let profiles_1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = profiles_1d.psi_norm.len();

    let mut j_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        let f_here: f64 = profiles_1d.f[i_psi_norm];
        j_parallel[i_psi_norm] =
            2.0 * PI * (f_here * profiles_1d.dpressure_dpsi[i_psi_norm] + profiles_1d.f_df_dpsi[i_psi_norm] * profiles_1d.gm5[i_psi_norm] / (MU_0 * f_here))
                / b0;
    }

    time_slice.profiles_1d.j_parallel = j_parallel;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// On a surface with `b_p = 0` everywhere, `<b ** 2> = f ** 2 * <1 / R ** 2>`, and `<j . b> = f * <j_phi / R>`, which is
    /// `2 * pi * f * (p' + ff' * <1 / R ** 2> / mu_0)`
    #[test]
    fn without_poloidal_field_it_is_the_toroidal_current_times_the_toroidal_field() {
        let b0: f64 = -0.6;
        let f_here: f64 = -0.3;
        let gm1: f64 = 4.2;
        let p_prime: f64 = -2.0e4;
        let ff_prime: f64 = 0.05;
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0];
        time_slice.profiles_1d.f = array![f_here];
        time_slice.profiles_1d.dpressure_dpsi = array![p_prime];
        time_slice.profiles_1d.f_df_dpsi = array![ff_prime];
        time_slice.profiles_1d.gm5 = array![f_here.powi(2) * gm1];

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.b0 = b0;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        let expected: f64 = 2.0 * PI * f_here * (p_prime + ff_prime * gm1 / MU_0) / b0;
        assert_abs_diff_eq!(time_slice.profiles_1d.j_parallel[0], expected, epsilon = 1e-9 * expected.abs());
    }
}
