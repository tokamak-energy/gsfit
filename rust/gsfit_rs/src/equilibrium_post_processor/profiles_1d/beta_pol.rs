//! `time_slice(itime)/profiles_1d/beta_pol`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the poloidal-beta profile, and store it in the time-slice.
///
/// The data dictionary defines poloidal beta as
///
/// ```text
/// beta_pol = 4 * integral(p dV) / (r0 * mu_0 * ip ** 2).
/// ```
///
/// For this radial profile, the pressure integral at each point contains the volume enclosed by
/// that flux surface:
///
/// ```text
/// beta_pol(psi) = 4 / (r0 * mu_0 * ip ** 2) * integral_axis^psi p dV.
/// ```
///
/// `r0` and `ip` are the same machine reference radius and total plasma current used by
/// `global_quantities/beta_pol`; only the enclosed pressure integral varies radially. The integral
/// is accumulated with the trapezoidal rule on the enclosed-volume coordinate. Consequently the
/// profile is zero at the magnetic axis, and its boundary value is the profile-based evaluation
/// of the global definition.
///
/// A point whose pressure or enclosed volume is unavailable remains NaN. The next valid point is
/// integrated from the preceding valid point, so one missing contour does not discard every
/// larger surface.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/beta_pol` is written into it
/// * `constant_values` - the constant values; `r0`, the vacuum toroidal field reference radius
///   `vacuum_toroidal_field/r0` [metre], is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let r0: f64 = constant_values.r0;

    let pressure: &Array1<f64> = time_slice.profiles_1d.pressure.as_ref().unwrap();
    let volume: &Array1<f64> = time_slice.profiles_1d.volume.as_ref().unwrap();
    let n_psi_norm: usize = pressure.len();
    assert_eq!(volume.len(), n_psi_norm);

    let mut beta_pol: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    if n_psi_norm == 0 {
        time_slice.profiles_1d.beta_pol = Some(beta_pol);
        return;
    }

    let ip: f64 = time_slice.global_quantities.ip.unwrap();
    let denominator: f64 = r0 * MU_0 * ip.powi(2);
    if !denominator.is_finite() || denominator <= 0.0 || !pressure[0].is_finite() || !volume[0].is_finite() || volume[0] != 0.0 {
        time_slice.profiles_1d.beta_pol = Some(beta_pol);
        return;
    }

    beta_pol[0] = 0.0;
    let mut pressure_volume_integral: f64 = 0.0;
    let mut i_previous_valid: usize = 0;
    for i_psi_norm in 1..n_psi_norm {
        if !pressure[i_psi_norm].is_finite() || !volume[i_psi_norm].is_finite() || volume[i_psi_norm] < volume[i_previous_valid] {
            continue;
        }

        let delta_volume: f64 = volume[i_psi_norm] - volume[i_previous_valid];
        let pressure_average: f64 = 0.5 * (pressure[i_psi_norm] + pressure[i_previous_valid]);
        pressure_volume_integral += pressure_average * delta_volume;
        beta_pol[i_psi_norm] = 4.0 * pressure_volume_integral / denominator;
        i_previous_valid = i_psi_norm;
    }

    time_slice.profiles_1d.beta_pol = Some(beta_pol);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn constant_pressure_matches_the_analytic_enclosed_integral() {
        let pressure: f64 = 12_000.0;
        let ip: f64 = 800_000.0;
        let r0: f64 = 0.9;
        let volume: Array1<f64> = array![0.0, 0.4, 1.5, 3.0];

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.ip = Some(ip);
        time_slice.profiles_1d.pressure = Some(Array1::from_elem(volume.len(), pressure));
        time_slice.profiles_1d.volume = Some(volume.clone());

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.r0 = r0;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        let beta_pol: &Array1<f64> = time_slice.profiles_1d.beta_pol.as_ref().unwrap();
        for i_psi_norm in 0..volume.len() {
            let beta_pol_expected: f64 = 4.0 * pressure * volume[i_psi_norm] / (r0 * MU_0 * ip.powi(2));
            assert_abs_diff_eq!(beta_pol[i_psi_norm], beta_pol_expected, epsilon = 1e-15);
        }
    }

    #[test]
    fn an_unavailable_surface_is_nan_without_discarding_larger_surfaces() {
        let pressure: f64 = 10_000.0;
        let ip: f64 = 700_000.0;
        let r0: f64 = 1.0;

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.ip = Some(ip);
        time_slice.profiles_1d.pressure = Some(array![pressure, f64::NAN, pressure]);
        time_slice.profiles_1d.volume = Some(array![0.0, f64::NAN, 2.0]);

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.r0 = r0;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        let beta_pol: &Array1<f64> = time_slice.profiles_1d.beta_pol.as_ref().unwrap();
        let beta_pol_boundary_expected: f64 = 4.0 * pressure * 2.0 / (r0 * MU_0 * ip.powi(2));
        assert_abs_diff_eq!(beta_pol[0], 0.0, epsilon = 1e-15);
        assert!(beta_pol[1].is_nan());
        assert_abs_diff_eq!(beta_pol[2], beta_pol_boundary_expected, epsilon = 1e-15);
    }

    #[test]
    fn invalid_normalisation_makes_the_whole_profile_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.ip = Some(0.0);
        time_slice.profiles_1d.pressure = Some(array![1.0, 0.0]);
        time_slice.profiles_1d.volume = Some(array![0.0, 1.0]);

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.r0 = 1.0;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.beta_pol.as_ref().unwrap().iter().all(|value| value.is_nan()));
    }
}
