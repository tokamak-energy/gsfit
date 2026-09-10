//! `time_slice(itime)/global_quantities/magnetic_axis/b_field_phi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the signed toroidal magnetic field at the magnetic axis.
///
/// The diamagnetic function is `F = R B_phi`, so its first profile value gives the axis field when
/// divided by the magnetic-axis major radius.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let f_profile: &Array1<f64> = time_slice.profiles_1d.f.as_ref().unwrap();
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    time_slice.global_quantities.magnetic_axis.b_field_phi = Some(f_profile[0] / mag_r);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn magnetic_axis_field_retains_the_sign_of_f() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = Some(0.8);
        time_slice.profiles_1d.f = Some(array![-0.48, -0.47]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.global_quantities.magnetic_axis.b_field_phi.unwrap(), -0.6, epsilon = 1e-15);
    }
}
