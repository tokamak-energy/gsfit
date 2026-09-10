//! `time_slice(itime)/profiles_2d(0)/theta`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the geometric poloidal angle on the rectangular `(R, Z)` grid.
///
/// The angle is centred on the magnetic axis and is zero on the outboard midplane:
///
/// ```text
/// theta(R, Z) = -atan2(Z - Z_axis, R - R_axis).
/// ```
///
/// It therefore increases clockwise in the usual plot with `R` to the right and `Z` upwards, in
/// the principal range `[-pi, pi]`. At the outboard midplane, `grad(rho_tor_norm)` points along
/// `+R` and `grad(theta)` along `-Z`; consequently
/// `(grad(rho_tor_norm), grad(theta), grad(phi))` is right-handed, as required by the IMAS COCOS 17
/// convention.
///
/// The angle is undefined at the magnetic axis itself. If a grid node coincides exactly with the
/// axis, `0` is stored there, matching the established GSFit convention. A time-slice which did
/// not converge has no magnetic axis, so its complete `theta` grid is NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/theta` is written into it [radian]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let mut theta: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);

    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();
    if !mag_r.is_finite() || !mag_z.is_finite() {
        time_slice.profiles_2d[0].theta = Some(theta);
        return;
    }

    for i_z in 0..n_z {
        let delta_z: f64 = z[i_z] - mag_z;
        for i_r in 0..n_r {
            let delta_r: f64 = r[i_r] - mag_r;
            if delta_r == 0.0 && delta_z == 0.0 {
                theta[(i_z, i_r)] = 0.0;
            } else {
                theta[(i_z, i_r)] = -delta_z.atan2(delta_r);
            }
        }
    }

    time_slice.profiles_2d[0].theta = Some(theta);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn theta_is_clockwise_from_the_outboard_midplane() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = Some(1.0);
        time_slice.global_quantities.magnetic_axis.z = Some(0.0);
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = Some(array![0.0, 1.0, 2.0]);
        time_slice.profiles_2d[0].grid.dim2 = Some(array![-1.0, 0.0, 1.0]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let theta: &Array2<f64> = time_slice.profiles_2d[0].theta.as_ref().unwrap();
        assert_eq!(theta.dim(), (3, 3));
        assert_abs_diff_eq!(theta[(1, 2)], 0.0, epsilon = 1e-15); // outboard
        assert_abs_diff_eq!(theta[(2, 1)], -PI / 2.0, epsilon = 1e-15); // above
        assert_abs_diff_eq!(theta[(1, 0)], -PI, epsilon = 1e-15); // inboard
        assert_abs_diff_eq!(theta[(0, 1)], PI / 2.0, epsilon = 1e-15); // below
        assert_abs_diff_eq!(theta[(1, 1)], 0.0, epsilon = 1e-15); // magnetic axis
    }

    #[test]
    fn failed_slice_is_all_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = Some(f64::NAN);
        time_slice.global_quantities.magnetic_axis.z = Some(f64::NAN);
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = Some(array![0.5, 1.0]);
        time_slice.profiles_2d[0].grid.dim2 = Some(array![-0.5, 0.5]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_2d[0].theta.as_ref().unwrap().iter().all(|value| value.is_nan()));
    }
}
