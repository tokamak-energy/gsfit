//! `time_slice(itime)/profiles_1d/rho_tor`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate the toroidal flux coordinate profile, and store it in the time-slice.
///
/// This is the data dictionary's definition, which has units of metre:
///
/// ```text
/// rho_tor = sqrt(phi / (pi * b0))
/// ```
///
/// where `phi` is `profiles_1d/phi` and `b0` is `vacuum_toroidal_field/b0`.
///
/// The dimensionless `sqrt(phi / phi_boundary)` is a different quantity, `rho_tor_norm`; see
/// `profiles_1d::rho_tor_norm::calculate`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_tor` is written into it
/// * `constant_values` - the constant values; `b0`, the vacuum toroidal field at `r0` for this time-slice,
///   `vacuum_toroidal_field/b0` [tesla], is read
///
/// A slice which did not converge carries NaN in `phi`, so the profile comes out NaN without
/// needing a special case.
///
/// `phi` and `b0` share a sign - both are positive when the toroidal field is counter-clockwise
/// viewed from above - so their ratio is positive. The magnitude is taken anyway, because
/// `rho_tor` is a positive radial coordinate and a sign convention disagreeing between the two
/// should not turn the whole profile into NaN.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let b0: f64 = constant_values.b0;

    let flux_toroidal_profile: &Array1<f64> = time_slice.profiles_1d.phi.as_ref().unwrap();

    let rho_tor: Array1<f64> = flux_toroidal_profile.mapv(|flux_toroidal| (flux_toroidal / (PI * b0)).abs().sqrt());

    time_slice.profiles_1d.rho_tor = Some(rho_tor);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn rho_tor_is_the_data_dictionary_definition_in_metre() {
        let b0: f64 = -0.4;
        let flux_toroidal_profile: Array1<f64> = array![0.0, -0.1, -0.4];

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.phi = Some(flux_toroidal_profile);

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.b0 = b0;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        let rho_tor: Array1<f64> = time_slice.profiles_1d.rho_tor.unwrap();

        // Zero toroidal flux at the magnetic axis, so zero radius
        assert_abs_diff_eq!(rho_tor[0], 0.0, epsilon = 1e-15);
        // A negative `phi` with a negative `b0` gives a positive, real radius
        assert_abs_diff_eq!(rho_tor[1], (0.1 / (PI * 0.4)).sqrt(), epsilon = 1e-15);
        assert_abs_diff_eq!(rho_tor[2], (0.4 / (PI * 0.4)).sqrt(), epsilon = 1e-15);
        // Four times the flux is twice the radius
        assert_abs_diff_eq!(rho_tor[2] / rho_tor[1], 2.0, epsilon = 1e-12);
    }
}
