//! `time_slice(itime)/global_quantities/li_1`, `.../li_2` and `.../li_3`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the normalised internal inductance, under three different normalisations, and store
/// them in the time-slice.
///
/// All three share the volume integral of the squared poloidal field, `int(b_p ** 2 dV)`, taken
/// over the grid cells inside the plasma. They differ only in what it is normalised to:
///
/// ```text
/// li_1 = <b_p ** 2> / <<b_p ** 2>>
/// li_2 = 2 * int(b_p ** 2 dV) / (mu_0 ** 2 * ip ** 2 * magnetic_axis/r)
/// li_3 = 2 * int(b_p ** 2 dV) / (mu_0 ** 2 * ip ** 2 * boundary/geometric_axis/r)
/// ```
///
/// where `<x>` is the volume average and `<<x>>` the flux surface average. `li_3` is the data
/// dictionary's own definition; the other two are custom keys, kept because GSFit has reported all
/// three historically.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the three `global_quantities` nodes are written into it
/// * `bp_sq_fs_avg` - flux-surface-averaged `b_p ** 2` from
///   `bp_sq_flux_surface_average::calculate`
///   [tesla ** 2]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let bp_sq_fs_avg: f64 = intermediate_values.bp_sq_fs_avg;

    let ip: f64 = time_slice.global_quantities.ip;
    let r_mag: f64 = time_slice.global_quantities.magnetic_axis.r;
    let r_geo: f64 = time_slice.boundary.geometric_axis.r;

    // The plasma volume is the volume enclosed by the last closed flux surface
    let volume_profile: &Array1<f64> = &time_slice.profiles_1d.volume;
    let plasma_volume: f64 = volume_profile.last().unwrap().to_owned();

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area;
    let b_r: &Array2<f64> = &time_slice.profiles_2d[0].b_field_r;
    let b_z: &Array2<f64> = &time_slice.profiles_2d[0].b_field_z;
    let mask: &Array2<f64> = &time_slice.profiles_2d[0].mask;

    let dims: &[usize] = b_r.shape();
    let n_z: usize = dims[0];
    let n_r: usize = dims[1];

    let mut bp_sq_vol_int: f64 = 0.0;
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            let bp_sq: f64 = b_r[(i_z, i_r)].powi(2) + b_z[(i_z, i_r)].powi(2);
            bp_sq_vol_int += bp_sq * mask[(i_z, i_r)] * 2.0 * PI * r[i_r] * d_area;
        }
    }

    // li_1 = <b_p ** 2> / <<b_p ** 2>>, where `<x>` is the volume average and `<<x>>` is the flux surface average
    let bp_sq_vol_avg: f64 = bp_sq_vol_int / plasma_volume;
    let li_1: f64 = bp_sq_vol_avg / bp_sq_fs_avg;
    let li_2: f64 = 2.0 * bp_sq_vol_int / (MU_0.powi(2) * ip.powi(2) * r_mag);
    let li_3: f64 = 2.0 * bp_sq_vol_int / (MU_0.powi(2) * ip.powi(2) * r_geo);

    time_slice.global_quantities.li_1 = li_1;
    time_slice.global_quantities.li_2 = li_2;
    time_slice.global_quantities.li_3 = li_3;
}
