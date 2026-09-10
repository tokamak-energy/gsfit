//! `time_slice(itime)/profiles_1d_r_midplane/f`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::profiles_1d;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the poloidal-current function `f` along the mid-plane, and store it in the time-slice.
///
/// Inside the plasma boundary this is the same function of `psi_norm` as `profiles_1d/f`, evaluated
/// at the mid-plane's own `psi_norm`. Outside it is the vacuum value `f_vac = mu_0 * i_rod /
/// (2 * pi)`: no poloidal current flows there, so `f = R * b_field_phi` is constant. That is why
/// this profile is not masked to zero the way the pressure and the source functions are - a zero
/// would say the toroidal field vanishes outside the plasma.
///
/// Which side of the boundary a point is on is decided by `profiles_2d(0)/mask`, not by the value
/// of `psi_norm`: the solver masks `psi_norm` to zero outside the plasma, which would otherwise
/// place every outside point on the magnetic axis.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d_r_midplane/f` is written into it
/// * `constant_values` - the constant values; `ff_prime_source_function` and `i_rod` are read
///
/// A time-slice which failed to converge carries `NaN` in the coefficients, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so the inside points come out `NaN` without needing a
/// special case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let ff_prime_source_function: &SharedSourceFunction = constant_values.ff_prime_source_function;
    let i_rod: f64 = constant_values.i_rod;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_norm_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();
    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();

    // The mid-plane is the middle row of the grid
    let (n_z, n_r): (usize, usize) = psi_norm_2d.dim();
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
    let midplane_psi_norm: Array1<f64> = psi_norm_2d.row(i_z_centre).to_owned();
    let midplane_mask: Array1<f64> = mask_2d.row(i_z_centre).to_owned();

    // Vacuum value, which is what `f` is outside the plasma
    let f_vac: f64 = i_rod * MU_0 / (2.0 * PI);

    let mut f_profile: Array1<f64> = Array1::from_elem(n_r, f64::NAN);

    // The formula itself lives with `profiles_1d/f`, so that the two profiles cannot drift apart
    for i_r in 0..n_r {
        if midplane_mask[i_r] > 0.99 {
            f_profile[i_r] = profiles_1d::f::value_at_psi_norm(time_slice, ff_prime_source_function, i_rod, midplane_psi_norm[i_r]);
        } else {
            f_profile[i_r] = f_vac;
        }
    }

    time_slice.profiles_1d_r_midplane.f = Some(f_profile);
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
    fn inside_points_follow_the_source_function_and_outside_points_are_the_vacuum_value() {
        // The placeholder source function is `EfitPolynomial` with one degree of freedom, so
        // `ff' = coefficient * (1 - psi_norm)` and, integrating from 1 to `psi_norm`,
        // `integral(ff') = -coefficient * (1 - psi_norm) ** 2 / 2`. With `d(psi)/d(psi_norm) = -1`,
        //   f = sqrt(f_vac ** 2 + coefficient * (1 - psi_norm) ** 2)
        let ff_prime_coefficient: f64 = 12.0;
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.source_functions.ff_prime.coefficients = Some(array![ff_prime_coefficient]);

        // `i_rod` chosen so that `f_vac` is exactly 1
        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.i_rod = 2.0 * PI / MU_0;
        let f_vac: f64 = 1.0;

        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        // The mid-plane row is `psi_norm = [0.0, 0.25, 0.5, 0.0]` with `mask = [0, 1, 1, 0]`
        let f_profile: &Array1<f64> = time_slice.profiles_1d_r_midplane.f.as_ref().unwrap();
        assert_abs_diff_eq!(f_profile[0], f_vac, epsilon = 1e-15);
        assert_abs_diff_eq!(f_profile[1], f_expected(0.25, f_vac, ff_prime_coefficient), epsilon = 1e-15);
        assert_abs_diff_eq!(f_profile[2], f_expected(0.5, f_vac, ff_prime_coefficient), epsilon = 1e-15);
        assert_abs_diff_eq!(f_profile[3], f_vac, epsilon = 1e-15);

        // `psi_norm = 0.5` is where the closed form above is exactly 2
        assert_abs_diff_eq!(f_profile[2], 2.0, epsilon = 1e-15);
    }

    /// The closed form `f` [tesla metre] must follow inside the plasma, for this test's
    /// `d(psi)/d(psi_norm) = -1`
    fn f_expected(psi_norm: f64, f_vac: f64, ff_prime_coefficient: f64) -> f64 {
        return (f_vac * f_vac + ff_prime_coefficient * (1.0 - psi_norm).powi(2)).sqrt();
    }
}
