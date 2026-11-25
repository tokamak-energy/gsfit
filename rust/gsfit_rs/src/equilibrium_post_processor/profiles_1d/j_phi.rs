//! `time_slice(itime)/profiles_1d/j_phi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the flux-surface-averaged toroidal current density, and store it in the time-slice.
///
/// The data dictionary defines this profile as `<j_phi / R> / <1 / R>`. The toroidal current density the Grad-Shafranov equation is solved
/// with is
///
/// ```text
/// j_phi = 2 * pi * R * p' + 2 * pi * ff' / (mu_0 * R),
/// ```
///
/// where the `2 * pi` is GSFit's convention of `psi` being the total flux rather than the flux per radian, and `p'` and `ff'` are constant
/// on a flux surface, so
///
/// ```text
/// <j_phi / R> / <1 / R> = 2 * pi * (p' + ff' * gm1 / mu_0) / gm9,
/// ```
///
/// with `gm1 = <1 / R ** 2>` and `gm9 = <1 / R>`. This is exact for the equilibrium, where an average of `profiles_2d(0)/j_phi` would
/// also average how the current density is discretised on the grid.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/dpressure_dpsi`, `.../f_df_dpsi`, `.../gm1` and `.../gm9` are read, and
///   `profiles_1d/j_phi` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let profiles_1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = profiles_1d.psi_norm.len();

    let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        j_phi[i_psi_norm] = 2.0 * PI * (profiles_1d.dpressure_dpsi[i_psi_norm] + profiles_1d.f_df_dpsi[i_psi_norm] * profiles_1d.gm1[i_psi_norm] / MU_0)
            / profiles_1d.gm9[i_psi_norm];
    }

    time_slice.profiles_1d.j_phi = j_phi;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// With only `p'`, `j_phi / R = 2 * pi * p'` is the same everywhere on the surface, so the profile is `2 * pi * p' / <1 / R>`; with only
    /// `ff'`, `j_phi * R = 2 * pi * ff' / mu_0` is, and the profile is `2 * pi * ff' * <1 / R ** 2> / (mu_0 * <1 / R>)`
    #[test]
    fn each_source_function_gives_its_own_average() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 1.0];
        time_slice.profiles_1d.dpressure_dpsi = array![-3.0e4, 0.0];
        time_slice.profiles_1d.f_df_dpsi = array![0.0, 0.2];
        time_slice.profiles_1d.gm1 = array![4.0, 4.5];
        time_slice.profiles_1d.gm9 = array![2.0, 2.1];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let j_phi: &Array1<f64> = &time_slice.profiles_1d.j_phi;
        assert_abs_diff_eq!(j_phi[0], 2.0 * PI * -3.0e4 / 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(j_phi[1], 2.0 * PI * 0.2 * 4.5 / (MU_0 * 2.1), epsilon = 1e-6);
    }
}
