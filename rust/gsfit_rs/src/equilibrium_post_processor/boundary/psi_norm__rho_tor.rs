//! `time_slice(itime)/boundary/psi_norm` and `.../rho_tor`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the normalized poloidal flux and toroidal flux coordinate at the plasma boundary.
///
/// The selected boundary is the final surface in `profiles_1d`, so both quantities are the final
/// values of their corresponding profiles. A time-slice which failed to converge carries `NaN` at
/// those endpoints, which is propagated without a special case.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/psi_norm` and `boundary/rho_tor` are written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi_norm_profile: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let rho_tor_profile: &Array1<f64> = &time_slice.profiles_1d.rho_tor;

    time_slice.boundary.psi_norm = psi_norm_profile.last().copied().unwrap_or(f64::NAN);
    time_slice.boundary.rho_tor = rho_tor_profile.last().copied().unwrap_or(f64::NAN);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn boundary_values_are_the_profile_endpoints() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.4, 1.0];
        time_slice.profiles_1d.rho_tor = array![0.0, 0.2, 0.5];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.boundary.psi_norm, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.boundary.rho_tor, 0.5, epsilon = 1e-15);
    }
}
