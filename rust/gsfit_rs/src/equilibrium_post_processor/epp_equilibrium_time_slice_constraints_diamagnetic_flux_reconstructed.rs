//! `time_slice(itime)/constraints/diamagnetic_flux/reconstructed`

use super::epp_equilibrium_time_slice_profiles_1d_phi::epp_flux_toroidal_profile;
use super::epp_equilibrium_time_slice_profiles_1d_q::epp_q_profile;
use super::epp_flux_surfaces::FluxSurface;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the diamagnetic flux, and store it in the time-slice.
///
/// The diamagnetic flux is the difference between the toroidal flux the plasma actually encloses
/// and the toroidal flux the vacuum toroidal field alone would give through the same flux surfaces,
/// so it measures how far the plasma has expelled or compressed the toroidal field. It is
/// calculated by running the safety factor and toroidal flux a second time with `f` set to the
/// vacuum value `f_vac = mu_0 * i_rod / (2 * pi)` everywhere.
///
/// This is the reconstruction's prediction of the diamagnetic loop signal, which is why it lives
/// under `constraints` rather than `global_quantities`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/diamagnetic_flux/reconstructed` is written
///   into it
/// * `flux_surfaces` - the flux surfaces from `epp_flux_surfaces`, one per `psi_norm`
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_phi`**, which supplies the toroidal
/// flux the vacuum one is subtracted from.
pub fn epp_equilibrium_time_slice_constraints_diamagnetic_flux_reconstructed(time_slice: &mut EquilibriumTimeSlice, flux_surfaces: &[FluxSurface]) {
    // A slice which did not converge has no flux surfaces to integrate around
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.constraints.diamagnetic_flux.reconstructed = Some(f64::NAN);
        return;
    }

    let f_profile: &Array1<f64> = time_slice.profiles_1d.f.as_ref().unwrap();
    let psi_profile: &Array1<f64> = time_slice.profiles_1d.psi.as_ref().unwrap();
    let flux_tor_profile: &Array1<f64> = time_slice.profiles_1d.phi.as_ref().unwrap();
    let i_rod: f64 = time_slice.global_quantities.i_rod.unwrap();

    // TODO: this is **VERY** hacky, and **SHOULD** be improved!!
    // set f_profile to the vacuum profile, then calculate the vacuum q-profile, then the vacuum toroidal flux
    let f_profile_vacuum: Array1<f64> = 0.0 * f_profile + MU_0 * i_rod / (2.0 * PI);
    let q_profile_vacuum: Array1<f64> = epp_q_profile(time_slice, flux_surfaces, &f_profile_vacuum);
    let flux_tor_profile_vacuum: Array1<f64> = epp_flux_toroidal_profile(&q_profile_vacuum, psi_profile);

    let flux_dia: f64 = flux_tor_profile.last().unwrap().to_owned() - flux_tor_profile_vacuum.last().unwrap().to_owned();

    time_slice.constraints.diamagnetic_flux.reconstructed = Some(flux_dia);
}
