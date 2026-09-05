//! `time_slice(itime)/profiles_1d/psi`

use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the poloidal flux profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/psi` is written into it
///
/// A time-slice which failed to converge carries `NaN` in `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so `psi` comes out `NaN` without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_1d_psi(time_slice: &mut EquilibriumTimeSlice) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    // ψ_N = (ψ_A − ψ) / (ψ_A − ψ_B), so inverting for ψ:
    //   ψ = ψ_A + (ψ_B − ψ_A)·ψ_N
    // which runs from ψ_A on the magnetic axis (ψ_N = 0) to ψ_B on the boundary (ψ_N = 1).
    let psi_profile: Array1<f64> = psi_norm * (psi_b - psi_a) + psi_a;

    time_slice.profiles_1d.psi = Some(psi_profile);
}
