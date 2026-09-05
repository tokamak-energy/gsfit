//! `time_slice(itime)/global_quantities/beta_tor` and `.../beta_tor_norm`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the toroidal beta and the normalised toroidal beta, and store them in the time-slice.
///
/// ```text
/// beta_tor      = 2 * mu_0 * <p> / bt_vac_at_r_geo ** 2
/// beta_tor_norm = 100 * beta_tor * minor_radius * bt_vac_at_r_geo / (ip / 1e6)
/// ```
///
/// where `<p>` is the volume-averaged pressure, which comes from the stored MHD energy,
/// `w_mhd = (3 / 2) * int(p dV)`.
///
/// Note: the old post-processor reported `beta_tor` as a **percentage**, whereas the data
/// dictionary defines it as a fraction ("beta_toroidal = 2 mu0 int(p dV) / V / B0^2"). This writes
/// the fraction, so `global_quantities/beta_tor` is 100 times smaller than the old
/// `global/beta_t`. `beta_tor_norm` is unaffected: its data dictionary definition carries the
/// factor of 100 explicitly, so it already matched.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `global_quantities` nodes are written into it
///
/// **Must run after `epp_equilibrium_time_slice_global_quantities_bt_vac_at_r_geo`**, which
/// supplies the toroidal field, after
/// `epp_equilibrium_time_slice_global_quantities_energy_mhd`, which supplies `int(p dV)`, after
/// `epp_equilibrium_time_slice_profiles_1d_area_and_volume`, which supplies the plasma volume, and
/// after `epp_equilibrium_time_slice_boundary_geometry`, which supplies the minor radius.
pub fn epp_equilibrium_time_slice_global_quantities_beta_tor(time_slice: &mut EquilibriumTimeSlice) {
    let w_mhd: f64 = time_slice.global_quantities.energy_mhd.unwrap();
    let ip: f64 = time_slice.global_quantities.ip.unwrap();
    let bt_vac_at_r_geo: f64 = time_slice.global_quantities.bt_vac_at_r_geo.unwrap();
    let r_minor: f64 = time_slice.boundary.minor_radius.unwrap();

    // The plasma volume is the volume enclosed by the last closed flux surface
    let volume_profile: &Array1<f64> = time_slice.profiles_1d.volume.as_ref().unwrap();
    let plasma_volume: f64 = volume_profile.last().unwrap().to_owned();

    // w_mhd = (3 / 2) * int(p dV)
    let p_vol_int: f64 = (2.0 / 3.0) * w_mhd;
    let p_vol_avg: f64 = p_vol_int / plasma_volume;

    let beta_tor: f64 = 2.0 * MU_0 * p_vol_avg / bt_vac_at_r_geo.powi(2);

    // The percentage form is kept as an intermediate, so that `beta_tor_norm` is evaluated in
    // exactly the same order as the old post-processor evaluated it
    let beta_tor_percent: f64 = 2.0 * MU_0 * p_vol_avg * 100.0 / bt_vac_at_r_geo.powi(2);
    let beta_tor_norm: f64 = beta_tor_percent * r_minor * bt_vac_at_r_geo / (ip / 1e6);

    time_slice.global_quantities.beta_tor = Some(beta_tor);
    time_slice.global_quantities.beta_tor_norm = Some(beta_tor_norm);
}
