//! `time_slice(itime)/global_quantities/bp_omp`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::{BicubicInterpolator, BicubicValueAndDerivatives};
use imas_rs::{EquilibriumProfiles2d, EquilibriumTimeSlice};
use ndarray::{Array1, Array2, s};
use std::f64::consts::PI;

/// Calculate the magnitude of the poloidal magnetic field at the outboard midplane, and store it in
/// the time-slice.
///
/// The outboard midplane is where the plasma boundary crosses the height of the magnetic axis on
/// the outboard side, `(profiles_1d/r_outboard[-1], magnetic_axis/z)`. This is the same point
/// `global_quantities/delta_r_sep` and `global_quantities/f_x` use.
///
/// ```text
/// bp_omp = sqrt(d(psi)/d(r) ** 2 + d(psi)/d(z) ** 2) / (2 * pi * omp_r)
/// ```
///
/// The gradient comes from the solver's bicubic interpolation of `psi` in the cell containing the
/// point. That is the same interpolation `profiles_1d/r_outboard` inverts to find `omp_r`, so the
/// field is evaluated on exactly the boundary that was found. It is built from the solver's
/// analytic derivatives of `psi`, so it is more accurate than interpolating the gridded
/// `profiles_2d/b_field_r` and `.../b_field_z` between grid points.
///
/// `d(psi)/d(z)` is kept, although it is small here: the midplane of an up-down asymmetric plasma
/// is not where the flux surface is vertical.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/bp_omp` is written into it
///
/// NaN when the boundary does not cross the height of the magnetic axis within the grid on the
/// outboard side.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    time_slice.global_quantities.bp_omp = f64::NAN;

    // The last entry of `r_outboard` is on the plasma boundary, `psi_norm = 1`
    let r_outboard: &Array1<f64> = &time_slice.profiles_1d.r_outboard;
    let omp_r: f64 = r_outboard.last().copied().unwrap_or(f64::NAN);
    let omp_z: f64 = time_slice.global_quantities.magnetic_axis.z;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let profiles_2d: &EquilibriumProfiles2d = &time_slice.profiles_2d[0];

    let Some((d_psi_d_r, d_psi_d_z)) = psi_gradient_at(profiles_2d, omp_r, omp_z) else {
        return;
    };

    let bp_omp: f64 = d_psi_d_r.hypot(d_psi_d_z) / (2.0 * PI * omp_r);

    time_slice.global_quantities.bp_omp = bp_omp;
}

/// `d(psi)/d(r)` and `d(psi)/d(z)` at an arbitrary point, from the bicubic model of the cell
/// containing it.
///
/// # Arguments
/// * `profiles_2d` - the solved grid; its axes, `psi`, `d_psi_d_r`, `d_psi_d_z` and
///   `d2_psi_d_r_d_z` are read
/// * `r_point` - where to evaluate, [metre]
/// * `z_point` - where to evaluate, [metre]
///
/// # Returns
/// * `d_psi_d_r` - [weber / metre]
/// * `d_psi_d_z` - [weber / metre]
///
/// `None` when the point is off the grid, or is NaN.
fn psi_gradient_at(profiles_2d: &EquilibriumProfiles2d, r_point: f64, z_point: f64) -> Option<(f64, f64)> {
    let r: &Array1<f64> = &profiles_2d.grid.dim1;
    let z: &Array1<f64> = &profiles_2d.grid.dim2;
    let psi_2d: &Array2<f64> = &profiles_2d.psi;
    let d_psi_d_r_2d: &Array2<f64> = &profiles_2d.d_psi_d_r;
    let d_psi_d_z_2d: &Array2<f64> = &profiles_2d.d_psi_d_z;
    let d2_psi_d_r_d_z_2d: &Array2<f64> = &profiles_2d.d2_psi_d_r_d_z;

    let n_r: usize = r.len();
    let n_z: usize = z.len();
    if n_r < 2 || n_z < 2 {
        return None;
    }

    // Written so that a NaN point fails the test too. It has to, because the cell indices below
    // come from casting to `usize`, which would saturate a NaN or negative index to the first cell
    let on_grid: bool = r_point >= r[0] && r_point <= r[n_r - 1] && z_point >= z[0] && z_point <= z[n_z - 1];
    if !on_grid {
        return None;
    }

    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // The cell containing the point, clamped so that a point on the last grid line uses the last cell
    let i_r_left: usize = (((r_point - r[0]) / d_r).floor() as usize).min(n_r - 2);
    let i_z_lower: usize = (((z_point - z[0]) / d_z).floor() as usize).min(n_z - 2);

    let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(
        d_r,
        d_z,
        psi_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
    );

    let x: f64 = (r_point - r[i_r_left]) / d_r;
    let y: f64 = (z_point - z[i_z_lower]) / d_z;

    // `value_and_derivatives` differentiates with respect to the normalised cell coordinates
    let psi_value_and_derivatives: BicubicValueAndDerivatives = psi_interpolator.value_and_derivatives(x, y);
    let d_psi_d_r: f64 = psi_value_and_derivatives.d_f_d_x / d_r;
    let d_psi_d_z: f64 = psi_value_and_derivatives.d_f_d_y / d_z;

    Some((d_psi_d_r, d_psi_d_z))
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Build the time-slice this calculator is tested against.
    ///
    /// A quadratic well on a 12-by-11 grid, with its derivatives filled in exactly:
    ///
    /// ```text
    /// psi = 1 + 8 * (r - 0.65) ** 2 + 7 * (r - 0.65) * (z - 0.05) + 5 * (z - 0.05) ** 2
    /// ```
    ///
    /// A bicubic reproduces a function which is quadratic in each variable exactly, so the gradient
    /// has a closed form to check against. The cross term makes `d(psi)/d(z)` non-zero on the row
    /// `z = mag_z`, so a calculator which dropped it, or mixed up the `r` and `z` axes, fails.
    ///
    /// The magnetic axis is at `(0.65, 0.05)` and the outboard boundary at `r = 1.15`, both half
    /// way between grid lines, so that a calculator which sampled the nearest grid point instead of
    /// interpolating fails.
    fn time_slice_for_test() -> EquilibriumTimeSlice {
        let r: Array1<f64> = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let z: Array1<f64> = array![-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - 0.65;
                let delta_z: f64 = z[i_z] - 0.05;
                psi_2d[(i_z, i_r)] = 1.0 + 8.0 * delta_r.powi(2) + 7.0 * delta_r * delta_z + 5.0 * delta_z.powi(2);
                d_psi_d_r_2d[(i_z, i_r)] = 16.0 * delta_r + 7.0 * delta_z;
                d_psi_d_z_2d[(i_z, i_r)] = 7.0 * delta_r + 10.0 * delta_z;
                d2_psi_d_r_d_z_2d[(i_z, i_r)] = 7.0;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;
        profiles_2d.d_psi_d_z = d_psi_d_z_2d;
        profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = 0.65;
        time_slice.global_quantities.magnetic_axis.z = 0.05;
        time_slice.profiles_1d.r_outboard = array![0.65, 0.9, 1.15];
        time_slice.profiles_2d = vec![profiles_2d];

        time_slice
    }

    #[test]
    fn poloidal_field_is_evaluated_on_the_boundary_at_the_height_of_the_magnetic_axis() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // At `(1.15, 0.05)` the fixture gives `d(psi)/d(r) = 16 * 0.5` and `d(psi)/d(z) = 7 * 0.5`
        let omp_r: f64 = 1.15;
        let d_psi_d_r: f64 = 16.0 * 0.5;
        let d_psi_d_z: f64 = 7.0 * 0.5;
        let bp_omp_expected: f64 = d_psi_d_r.hypot(d_psi_d_z) / (2.0 * PI * omp_r);
        assert_abs_diff_eq!(time_slice.global_quantities.bp_omp, bp_omp_expected, epsilon = 1e-12);
    }

    #[test]
    fn boundary_which_does_not_reach_the_height_of_the_magnetic_axis_is_nan() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.r_outboard = array![0.65, 0.9, f64::NAN];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.global_quantities.bp_omp.is_nan());
    }

    #[test]
    fn point_off_the_grid_is_nan() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.r_outboard = array![0.65, 0.9, 1.3];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.global_quantities.bp_omp.is_nan());
    }
}
