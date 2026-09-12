//! `time_slice(itime)/profiles_2d(0)/grid/volume_element`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calculate the toroidal volume represented by every cell of the rectangular `(R, Z)` grid.
///
/// A poloidal cell bounded by `R_i`, `R_(i+1)`, `Z_j` and `Z_(j+1)` sweeps out the axisymmetric
/// volume
///
/// ```text
/// volume_element(j, i)
///     = integral_0^(2*pi) integral_Zj^Z(j+1) integral_Ri^R(i+1) R dR dZ dphi
///     = pi * (R_(i+1) ** 2 - R_i ** 2) * (Z_(j+1) - Z_j).
/// ```
///
/// The result has shape `(n_z - 1, n_r - 1)`: one value for each cell formed by four adjacent
/// nodes. This follows the project's array ordering, where the vertical index comes first. The
/// expression integrates the cylindrical Jacobian exactly and also supports a future nonuniform
/// rectangular grid.
///
/// This is purely grid geometry, so it is filled even when the equilibrium solve did not converge.
///
/// # Arguments
/// * `time_slice` - the time-slice whose `profiles_2d(0)/grid/volume_element` is written [metre ** 3]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    assert!(n_r >= 2);
    assert!(n_z >= 2);

    let mut volume_element: Array2<f64> = Array2::from_elem((n_z - 1, n_r - 1), f64::NAN);
    for i_z in 0..n_z - 1 {
        let delta_z: f64 = z[i_z + 1] - z[i_z];
        assert!(delta_z > 0.0);
        for i_r in 0..n_r - 1 {
            let delta_r: f64 = r[i_r + 1] - r[i_r];
            assert!(delta_r > 0.0);
            volume_element[(i_z, i_r)] = PI * (r[i_r + 1].powi(2) - r[i_r].powi(2)) * delta_z;
        }
    }

    time_slice.profiles_2d[0].grid.volume_element = volume_element;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::array;

    #[test]
    fn cells_have_the_exact_axisymmetric_volume() {
        let r: Array1<f64> = array![1.0, 2.0, 4.0];
        let z: Array1<f64> = array![-1.0, 1.0, 4.0];

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = r;
        time_slice.profiles_2d[0].grid.dim2 = z;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let volume_element: &Array2<f64> = &time_slice.profiles_2d[0].grid.volume_element;
        assert_eq!(volume_element.dim(), (2, 2));
        assert_abs_diff_eq!(volume_element[(0, 0)], 6.0 * PI, epsilon = 1e-14);
        assert_abs_diff_eq!(volume_element[(0, 1)], 24.0 * PI, epsilon = 1e-14);
        assert_abs_diff_eq!(volume_element[(1, 0)], 9.0 * PI, epsilon = 1e-14);
        assert_abs_diff_eq!(volume_element[(1, 1)], 36.0 * PI, epsilon = 1e-14);
        assert_abs_diff_eq!(volume_element.sum(), 75.0 * PI, epsilon = 1e-13);
    }
}
