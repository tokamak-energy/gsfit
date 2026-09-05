//! `time_slice(itime)/profiles_1d/pressure`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the pressure profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/pressure` is written into it
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
///
/// A time-slice which failed to converge carries `NaN` in the coefficients, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so the pressure comes out `NaN` without needing a special
/// case.
pub fn epp_equilibrium_time_slice_profiles_1d_pressure(time_slice: &mut EquilibriumTimeSlice, p_prime_source_function: &SharedSourceFunction) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let p_prime_dof_values: &Array1<f64> = time_slice.source_functions.p_prime.coefficients.as_ref().unwrap();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    // ψ_N = (ψ_A − ψ) / (ψ_A − ψ_B), so:
    //   ψ = ψ_A − (ψ_A − ψ_B)·ψ_N
    //   dψ/dψ_N = ψ_B − ψ_A
    //
    // Pressure is zero at the boundary (ψ_N = 1) and satisfies:
    //   p(ψ) = ∫_{ψ_B}^{ψ} p′(ψ′) dψ′
    //        = ∫_1^{ψ_N} p′(ψ_N′) · (dψ/dψ_N) dψ_N′
    //        = (ψ_B − ψ_A) · ∫_1^{ψ_N} p′(ψ_N′) dψ_N′
    // Note: `source_function_integral` integrates from 1 to ψ_N and is zero at ψ_N = 1.

    // dψ/dψ_N = ψ_B − ψ_A
    let d_psi_d_psi_norm: f64 = psi_b - psi_a;

    // p = (dψ/dψ_N) · ∫_1^{ψ_N} p′(ψ_N′) dψ_N′
    let p_profile: Array1<f64> = p_prime_source_function.source_function_integral(psi_norm, p_prime_dof_values) * d_psi_d_psi_norm;

    time_slice.profiles_1d.pressure = Some(p_profile);
}
