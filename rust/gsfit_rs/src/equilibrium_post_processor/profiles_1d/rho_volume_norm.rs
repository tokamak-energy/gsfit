//! `time_slice(itime)/profiles_1d/rho_volume_norm`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the normalised square root of enclosed volume, and store it in the time-slice.
///
/// The data dictionary defines this radial coordinate as
///
/// ```text
/// rho_volume_norm = sqrt(volume / volume(boundary)).
/// ```
///
/// The enclosed volume is zero at the magnetic axis, so the profile runs from 0 on the axis to 1
/// at the equilibrium boundary. A missing, negative, or non-finite volume remains NaN. If the
/// boundary volume is not finite and strictly positive, the whole profile is NaN because it cannot
/// be normalised.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_volume_norm` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let volume: &Array1<f64> = &time_slice.profiles_1d.volume;
    let n_psi_norm: usize = volume.len();
    let mut rho_volume_norm: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    if n_psi_norm == 0 {
        time_slice.profiles_1d.rho_volume_norm = rho_volume_norm;
        return;
    }

    let volume_boundary: f64 = volume[n_psi_norm - 1];
    if !volume_boundary.is_finite() || volume_boundary <= 0.0 {
        time_slice.profiles_1d.rho_volume_norm = rho_volume_norm;
        return;
    }

    for i_psi_norm in 0..n_psi_norm {
        if volume[i_psi_norm].is_finite() && volume[i_psi_norm] >= 0.0 {
            rho_volume_norm[i_psi_norm] = (volume[i_psi_norm] / volume_boundary).sqrt();
        }
    }

    time_slice.profiles_1d.rho_volume_norm = rho_volume_norm;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn profile_is_the_normalised_square_root_of_volume() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.volume = array![0.0, 0.25, 1.0, 4.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let rho_volume_norm: &Array1<f64> = &time_slice.profiles_1d.rho_volume_norm;
        assert_abs_diff_eq!(rho_volume_norm[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_volume_norm[1], 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_volume_norm[2], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_volume_norm[3], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn invalid_volumes_remain_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.volume = array![0.0, f64::NAN, -1.0, 4.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let rho_volume_norm: &Array1<f64> = &time_slice.profiles_1d.rho_volume_norm;
        assert_abs_diff_eq!(rho_volume_norm[0], 0.0, epsilon = 1e-15);
        assert!(rho_volume_norm[1].is_nan());
        assert!(rho_volume_norm[2].is_nan());
        assert_abs_diff_eq!(rho_volume_norm[3], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn invalid_boundary_volume_makes_the_whole_profile_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.volume = array![0.0, 0.25, f64::NAN];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.rho_volume_norm.iter().all(|value| value.is_nan()));
    }
}
