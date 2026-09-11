//! `time_slice(itime)/global_quantities/f_x`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;

/// Calculate the flux expansion from the outboard midplane to the outboard strike point.
///
/// ```text
/// f_x = (r_omp * b_p_omp) / (r_strike * b_p_strike)
/// ```
///
/// The outboard-midplane point is the intersection of the last closed flux surface with
/// `z = magnetic_axis/z`. The outboard strike point is the end of the low-field-side separatrix
/// leg traced from the active X-point. This is the axisymmetric total poloidal flux expansion; it
/// does not include the additional expansion along the target caused by the field-line incidence
/// angle.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/f_x` is written into it
///
/// A limited or failed time-slice, an unavailable point, or a zero poloidal field at either point
/// gives `NaN`.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    time_slice.global_quantities.f_x = f64::NAN;

    // `boundary/type` is 1 only for a diverted plasma. A limited plasma has no strike point, and a
    // failed time-slice carries `EMPTY_INT`, which this test also rejects.
    let xpt_diverted: bool = time_slice.boundary.r#type == 1;
    if !xpt_diverted {
        return;
    }

    let r_outboard: &Array1<f64> = &time_slice.profiles_1d.r_outboard;
    let r_omp: f64 = r_outboard.last().copied().unwrap_or(f64::NAN);
    let z_omp: f64 = time_slice.global_quantities.magnetic_axis.z;
    let strike_r: f64 = time_slice.sol.lfs.strike_point.r;
    let strike_z: f64 = time_slice.sol.lfs.strike_point.z;
    if !r_omp.is_finite() || !z_omp.is_finite() || !strike_r.is_finite() || !strike_z.is_finite() || r_omp <= 0.0 || strike_r <= 0.0 {
        return;
    }

    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let b_field_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_r;
    let b_field_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].b_field_z;

    let b_field_r_interpolator = Interp2D::builder(b_field_r_2d.to_owned()).x(z.clone()).y(r.clone()).build().unwrap();
    let b_field_z_interpolator = Interp2D::builder(b_field_z_2d.to_owned()).x(z.clone()).y(r.clone()).build().unwrap();
    let b_field_p_at = |r_point: f64, z_point: f64| -> Option<f64> {
        let b_field_r: f64 = b_field_r_interpolator.interp_scalar(z_point, r_point).ok()?;
        let b_field_z: f64 = b_field_z_interpolator.interp_scalar(z_point, r_point).ok()?;
        let b_field_p: f64 = b_field_r.hypot(b_field_z);
        if !b_field_p.is_finite() || b_field_p == 0.0 {
            return None;
        }
        Some(b_field_p)
    };

    let Some(b_field_p_omp) = b_field_p_at(r_omp, z_omp) else {
        return;
    };
    let Some(b_field_p_strike) = b_field_p_at(strike_r, strike_z) else {
        return;
    };

    let f_x: f64 = r_omp * b_field_p_omp / (strike_r * b_field_p_strike);
    if f_x.is_finite() {
        time_slice.global_quantities.f_x = f_x;
    }
}
#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;

    #[test]
    fn diverted_plasma_uses_radius_weighted_poloidal_field_ratio() {
        let r: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let z: Array1<f64> = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        let b_field_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), 3.0);
        let mut b_field_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_r in 0..n_r {
            for i_z in 0..n_z {
                b_field_z_2d[(i_z, i_r)] = r[i_r] + 2.0 * z[i_z];
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.b_field_r = b_field_r_2d;
        profiles_2d.b_field_z = b_field_z_2d;

        let r_omp: f64 = 2.5;
        let z_omp: f64 = 0.0;
        let strike_r: f64 = 1.5;
        let strike_z: f64 = -0.5;
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.boundary.r#type = 1;
        time_slice.global_quantities.magnetic_axis.z = z_omp;
        time_slice.profiles_1d.r_outboard = Array1::from_vec(vec![1.5, r_omp]);
        time_slice.profiles_2d = vec![profiles_2d];
        time_slice.sol.lfs.strike_point.r = strike_r;
        time_slice.sol.lfs.strike_point.z = strike_z;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let b_field_p_omp: f64 = 3.0_f64.hypot(r_omp + 2.0 * z_omp);
        let b_field_p_strike: f64 = 3.0_f64.hypot(strike_r + 2.0 * strike_z);
        let f_x_expected: f64 = r_omp * b_field_p_omp / (strike_r * b_field_p_strike);
        assert_abs_diff_eq!(time_slice.global_quantities.f_x, f_x_expected, epsilon = 1e-14);
    }

    #[test]
    fn limited_plasma_is_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.boundary.r#type = 0;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.global_quantities.f_x.is_nan());
    }
}
