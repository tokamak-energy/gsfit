//! `time_slice(itime)/profiles_1d/rho_pol`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the normalised poloidal flux radius profile, and store it in the time-slice.
///
/// This is simply `sqrt(psi_norm)`, the poloidal counterpart of `rho_tor_norm`. It is a custom key,
/// because the data dictionary does not define it.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_pol` is written into it
pub fn epp_equilibrium_time_slice_profiles_1d_rho_pol(time_slice: &mut EquilibriumTimeSlice) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let n_psi_norm: usize = psi_norm.len();

    // `psi_norm` is set up before the solve, so unlike the other profiles it is still valid for a
    // slice which did not converge. The old post-processor left this NaN, so match that
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_1d.rho_pol = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        return;
    }

    let rho_pol_profile: Array1<f64> = psi_norm.mapv(|x| x.sqrt());

    time_slice.profiles_1d.rho_pol = Some(rho_pol_profile);
}
