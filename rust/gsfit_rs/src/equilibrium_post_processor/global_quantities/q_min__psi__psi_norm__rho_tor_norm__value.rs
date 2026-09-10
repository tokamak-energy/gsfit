//! `time_slice(itime)/global_quantities/q_min/psi`, `.../psi_norm`,
//! `.../rho_tor_norm` and `.../value`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the minimum finite safety factor and its radial coordinates in the time-slice.
///
/// The minimum is taken over the signed values in `profiles_1d/q`. Non-finite values are ignored,
/// including the deliberately singular `q` at a diverted boundary. All three coordinates are
/// copied from the same profile index as the minimum. If the profile has no finite value, all four
/// outputs are `NaN`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the four `global_quantities/q_min` nodes are written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi_profile: &Array1<f64> = time_slice.profiles_1d.psi.as_ref().unwrap();
    let psi_norm_profile: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let q_profile: &Array1<f64> = time_slice.profiles_1d.q.as_ref().unwrap();
    let rho_tor_norm_profile: &Array1<f64> = time_slice.profiles_1d.rho_tor_norm.as_ref().unwrap();
    let n_q: usize = q_profile.len();

    assert_eq!(psi_profile.len(), n_q);
    assert_eq!(psi_norm_profile.len(), n_q);
    assert_eq!(rho_tor_norm_profile.len(), n_q);

    let mut i_q_min: Option<usize> = None;
    let mut q_min_value: f64 = f64::NAN;
    for i_q in 0..n_q {
        let q_here: f64 = q_profile[i_q];
        if q_here.is_finite() && (i_q_min.is_none() || q_here < q_min_value) {
            i_q_min = Some(i_q);
            q_min_value = q_here;
        }
    }

    let mut q_min_psi: f64 = f64::NAN;
    let mut q_min_psi_norm: f64 = f64::NAN;
    let mut q_min_rho_tor_norm: f64 = f64::NAN;
    if let Some(i_q_min) = i_q_min {
        q_min_psi = psi_profile[i_q_min];
        q_min_psi_norm = psi_norm_profile[i_q_min];
        q_min_rho_tor_norm = rho_tor_norm_profile[i_q_min];
    }

    time_slice.global_quantities.q_min.psi = Some(q_min_psi);
    time_slice.global_quantities.q_min.psi_norm = Some(q_min_psi_norm);
    time_slice.global_quantities.q_min.rho_tor_norm = Some(q_min_rho_tor_norm);
    time_slice.global_quantities.q_min.value = Some(q_min_value);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn minimum_and_coordinates_come_from_the_same_finite_profile_point() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi = Some(array![3.0, 2.5, 2.0, 1.5, 1.0]);
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.25, 0.5, 0.75, 1.0]);
        time_slice.profiles_1d.q = Some(array![2.0, f64::NAN, 1.4, 1.6, f64::NAN]);
        time_slice.profiles_1d.rho_tor_norm = Some(array![0.0, 0.4, 0.65, 0.85, 1.0]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.global_quantities.q_min.psi.unwrap(), 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.q_min.psi_norm.unwrap(), 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.q_min.rho_tor_norm.unwrap(), 0.65, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.q_min.value.unwrap(), 1.4, epsilon = 1e-15);
    }

    #[test]
    fn all_outputs_are_nan_when_q_has_no_finite_value() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi = Some(array![f64::NAN, f64::NAN]);
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 1.0]);
        time_slice.profiles_1d.q = Some(array![f64::NAN, f64::NAN]);
        time_slice.profiles_1d.rho_tor_norm = Some(array![f64::NAN, f64::NAN]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.global_quantities.q_min.psi.unwrap().is_nan());
        assert!(time_slice.global_quantities.q_min.psi_norm.unwrap().is_nan());
        assert!(time_slice.global_quantities.q_min.rho_tor_norm.unwrap().is_nan());
        assert!(time_slice.global_quantities.q_min.value.unwrap().is_nan());
    }
}
