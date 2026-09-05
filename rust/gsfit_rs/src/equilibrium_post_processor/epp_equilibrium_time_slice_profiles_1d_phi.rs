//! `time_slice(itime)/profiles_1d/phi`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the toroidal flux profile, and store it in the time-slice.
///
/// The toroidal flux follows from the definition of the safety factor, `q = d(phi) / d(psi)`,
/// integrated outwards from the magnetic axis where the enclosed toroidal flux is zero.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/phi` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_1d_q`**, which supplies `q`, and after
/// `epp_equilibrium_time_slice_profiles_1d_psi`, which supplies `psi`.
pub fn epp_equilibrium_time_slice_profiles_1d_phi(time_slice: &mut EquilibriumTimeSlice) {
    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.as_ref().unwrap().len();

    // A slice which did not converge has no safety factor to integrate. Without this the flux at
    // the magnetic axis would come out as the hard-coded 0.0 rather than NaN
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_1d.phi = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        return;
    }

    let q_profile: &Array1<f64> = time_slice.profiles_1d.q.as_ref().unwrap();
    let psi_profile: &Array1<f64> = time_slice.profiles_1d.psi.as_ref().unwrap();

    let flux_toroidal_profile: Array1<f64> = epp_flux_toroidal_profile(q_profile, psi_profile);

    time_slice.profiles_1d.phi = Some(flux_toroidal_profile);
}

/// Integrate `q` to give the toroidal flux profile.
///
/// Kept separate from the writer above because it is evaluated twice: once with the reconstructed
/// `q`, and once with the vacuum `q`, which is what the diamagnetic flux is measured against.
///
/// # Arguments
/// * `q_profile` - the safety factor profile [dimensionless]
/// * `psi_profile` - the poloidal flux profile [weber]
///
/// # Returns
/// * `flux_toroidal_profile` - the enclosed toroidal flux profile [weber]
pub(super) fn epp_flux_toroidal_profile(q_profile: &Array1<f64>, psi_profile: &Array1<f64>) -> Array1<f64> {
    let n_psi_n: usize = psi_profile.len();

    let mut flux_toroidal_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    flux_toroidal_profile[0] = 0.0; // no toroidal flux at the magnetic axis
    for i_psi_n in 1..n_psi_n {
        let avg_y: f64 = (q_profile[i_psi_n] + q_profile[i_psi_n - 1]) / 2.0;
        let dx: f64 = psi_profile[i_psi_n] - psi_profile[i_psi_n - 1];
        flux_toroidal_profile[i_psi_n] = flux_toroidal_profile[i_psi_n - 1] - avg_y * dx;
    }

    return flux_toroidal_profile;
}
