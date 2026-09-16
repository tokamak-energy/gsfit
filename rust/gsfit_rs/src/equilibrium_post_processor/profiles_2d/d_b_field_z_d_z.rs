//! `time_slice(itime)/profiles_2d(i1)/d_b_field_z_d_z`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calculate the vertical derivative of the vertical magnetic field, and store it in the
/// time-slice.
///
/// `b_z = (1 / (2 * pi * r)) * d(psi)/d(r)`, so differentiating vertically at fixed `r`:
///
/// ```text
/// d(b_z)/d(z) = (1 / (2 * pi * r)) * d2(psi)/d(r)d(z)
/// ```
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/d_b_field_z_d_z` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
    let d2_psi_d_r_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d2_psi_d_r_d_z;

    let mesh_r_local: Array2<f64> = Array2::from_shape_fn(psi_2d.dim(), |(_i_z, i_r)| r[i_r]);
    let d_b_field_z_d_z_2d: Array2<f64> = d2_psi_d_r_d_z_2d / (2.0 * PI * &mesh_r_local);

    time_slice.profiles_2d[0].d_b_field_z_d_z = d_b_field_z_d_z_2d;
}
