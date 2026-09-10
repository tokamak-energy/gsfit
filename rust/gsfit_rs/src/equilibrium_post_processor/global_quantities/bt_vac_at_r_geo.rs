//! `time_slice(itime)/global_quantities/bt_vac_at_r_geo`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the vacuum toroidal magnetic field at the plasma geometric axis, and store it in the
/// time-slice.
///
/// ```text
/// bt_vac_at_r_geo = mu_0 * i_rod / (2 * pi * r_geo)
/// ```
///
/// This follows the plasma, so it moves from time-slice to time-slice, unlike
/// `vacuum_toroidal_field/b0`, which is evaluated at the fixed machine reference radius.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/bt_vac_at_r_geo` is written into it
/// * `i_rod` - current in the toroidal field coil's central rod [ampere]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let i_rod: f64 = constant_values.i_rod;

    let r_geo: f64 = time_slice.boundary.geometric_axis.r.unwrap();

    let bt_vac_at_r_geo: f64 = MU_0 * i_rod / (2.0 * PI * r_geo);

    time_slice.global_quantities.bt_vac_at_r_geo = Some(bt_vac_at_r_geo);
}
