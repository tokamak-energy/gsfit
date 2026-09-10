//! `time_slice(itime)/global_quantities/beta_pol`, `.../beta_pol_1`, `.../beta_pol_2` and
//! `.../beta_pol_3`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the poloidal beta, under four different normalisations, and store them in the
/// time-slice.
///
/// All four share the numerator `int(p dV)`, which comes from the stored MHD energy,
/// `w_mhd = (3 / 2) * int(p dV)`. They differ only in what the poloidal field is represented by:
///
/// ```text
/// beta_pol   = 4 * int(p dV) / (mu_0 * ip ** 2 * r0)                     <- data dictionary
/// beta_pol_1 = 2 * mu_0 * <p> / <<b_p ** 2>>
/// beta_pol_2 = 4 * int(p dV) / (mu_0 * ip ** 2 * magnetic_axis/r)
/// beta_pol_3 = 4 * int(p dV) / (mu_0 * ip ** 2 * boundary/geometric_axis/r)
/// ```
///
/// where `<x>` is the volume average and `<<x>>` the flux surface average. `beta_pol` is the data
/// dictionary's own definition, `4 int(p dV) / [R_0 * mu_0 * Ip^2]`; the other three are custom
/// keys, kept because GSFit has reported all three historically.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the four `global_quantities` nodes are written into it
/// * `bp_sq_fs_avg` - flux-surface-averaged `b_p ** 2` from
///   `bp_sq_flux_surface_average::calculate`
///   [tesla ** 2]
/// * `r0` - the vacuum toroidal field reference radius `vacuum_toroidal_field/r0` [metre], which
///   `solve_grad_shafranov` copies from `tf/r0`
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let bp_sq_fs_avg: f64 = intermediate_values.bp_sq_fs_avg;
    let r0: f64 = constant_values.r0;

    let w_mhd: f64 = time_slice.global_quantities.energy_mhd.unwrap();
    let ip: f64 = time_slice.global_quantities.ip.unwrap();
    let r_mag: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let r_geo: f64 = time_slice.boundary.geometric_axis.r.unwrap();

    // The plasma volume is the volume enclosed by the last closed flux surface
    let volume_profile: &Array1<f64> = time_slice.profiles_1d.volume.as_ref().unwrap();
    let plasma_volume: f64 = volume_profile.last().unwrap().to_owned();

    // w_mhd = (3 / 2) * int(p dV)
    let p_vol_int: f64 = w_mhd * 2.0 / 3.0;

    // beta_p_1 = 2 * mu_0 * <p> / <<b_p ** 2>>, where `<x>` is the volume average and `<<x>>` is the flux surface average
    let p_vol_avg: f64 = p_vol_int / plasma_volume;
    let beta_p_1: f64 = 2.0 * MU_0 * p_vol_avg / bp_sq_fs_avg;

    let beta_p_2: f64 = 4.0 * p_vol_int / (MU_0 * ip * ip * r_mag);

    let beta_p_3: f64 = 4.0 * p_vol_int / (MU_0 * ip * ip * r_geo);

    let beta_pol: f64 = 4.0 * p_vol_int / (MU_0 * ip * ip * r0);

    time_slice.global_quantities.beta_pol = Some(beta_pol);
    time_slice.global_quantities.beta_pol_1 = Some(beta_p_1);
    time_slice.global_quantities.beta_pol_2 = Some(beta_p_2);
    time_slice.global_quantities.beta_pol_3 = Some(beta_p_3);
}
