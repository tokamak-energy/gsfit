//! `time_slice(itime)/profiles_2d(0)/grid/volume_element`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calculate the toroidal volume represented by every node of the rectangular `(R, Z)` grid.
///
/// The grid is equally spaced, and each node stands for the `d_r` by `d_z` cell centred on it,
/// which sweeps out the axisymmetric volume
///
/// ```text
/// volume_element(i_z, i_r)
///     = integral_0^(2*pi) integral_(z - d_z/2)^(z + d_z/2) integral_(r - d_r/2)^(r + d_r/2) R dR dZ dphi
///     = 2 * pi * r[i_r] * d_area
/// ```
///
/// which is exact, not a midpoint approximation, because the integrand is linear in `R`. These are
/// the same cells `energy_mhd` and `li` integrate over, so summing `volume_element` over the plasma
/// mask gives the volume those quantities use.
///
/// The result has shape `(n_z, n_r)`, one value per node, in the same order as every other
/// `profiles_2d` map. Note that the cells of the nodes on the edge of the grid extend half a cell
/// beyond it; the plasma never reaches the edge of the grid, so this does not affect any plasma
/// integral.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/grid/volume_element` is written into it
///   [metre ** 3]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area;
    let n_r: usize = r.len();
    let n_z: usize = z.len();

    let mut volume_element: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            volume_element[(i_z, i_r)] = 2.0 * PI * r[i_r] * d_area;
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

    /// The cells tile the rectangle `[r_min - d_r / 2, r_max + d_r / 2] x [z_min - d_z / 2, z_max + d_z / 2]`
    /// exactly, so together they sweep out the hollow cylinder `pi * (r_outer ** 2 - r_inner ** 2) * height`
    #[test]
    fn nodes_have_the_exact_axisymmetric_volume_of_their_cells() {
        let n_r: usize = 5;
        let n_z: usize = 7;
        let r: Array1<f64> = Array1::linspace(0.5, 1.5, n_r);
        let z: Array1<f64> = Array1::linspace(-0.9, 0.9, n_z);
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = r.clone();
        time_slice.profiles_2d[0].grid.dim2 = z.clone();
        time_slice.profiles_2d[0].grid.d_area = d_r * d_z;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let volume_element: &Array2<f64> = &time_slice.profiles_2d[0].grid.volume_element;
        assert_eq!(volume_element.dim(), (n_z, n_r));

        // One cell, integrated directly
        let r_inner: f64 = r[2] - 0.5 * d_r;
        let r_outer: f64 = r[2] + 0.5 * d_r;
        assert_abs_diff_eq!(volume_element[(3, 2)], PI * (r_outer.powi(2) - r_inner.powi(2)) * d_z, epsilon = 1e-14);

        // All the cells together
        let r_inner: f64 = r[0] - 0.5 * d_r;
        let r_outer: f64 = r[n_r - 1] + 0.5 * d_r;
        let height: f64 = z[n_z - 1] - z[0] + d_z;
        assert_abs_diff_eq!(volume_element.sum(), PI * (r_outer.powi(2) - r_inner.powi(2)) * height, epsilon = 1e-13);
    }
}
