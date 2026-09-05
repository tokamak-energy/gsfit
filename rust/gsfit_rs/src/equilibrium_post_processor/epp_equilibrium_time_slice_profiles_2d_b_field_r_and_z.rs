//! `time_slice(itime)/profiles_2d(i1)/b_field_r` and `.../b_field_z`

use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calculate the poloidal magnetic field on the grid, and store it in the time-slice.
///
/// The poloidal field follows from the poloidal flux:
///
/// ```text
/// b_r = -(1 / (2 * pi * r)) * d(psi)/d(z)
/// b_z = +(1 / (2 * pi * r)) * d(psi)/d(r)
/// ```
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/b_field_r` and `.../b_field_z` are
///   written into it
///
/// A time-slice which failed to converge carries NaN in the flux derivatives, so the fields come
/// out NaN without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_2d_b_field_r_and_z(time_slice: &mut EquilibriumTimeSlice) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();

    let mesh_r_local: Array2<f64> = Array2::from_shape_fn(psi_2d.dim(), |(_i_z, i_r)| r[i_r]);
    let b_field_r_2d: Array2<f64> = -d_psi_d_z_2d / (2.0 * PI * &mesh_r_local);
    let b_field_z_2d: Array2<f64> = d_psi_d_r_2d / (2.0 * PI * &mesh_r_local);

    time_slice.profiles_2d[0].b_field_r = Some(b_field_r_2d);
    time_slice.profiles_2d[0].b_field_z = Some(b_field_z_2d);
}
