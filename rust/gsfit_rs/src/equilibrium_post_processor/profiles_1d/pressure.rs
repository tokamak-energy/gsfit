//! `time_slice(itime)/profiles_1d/pressure`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the pressure profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/pressure` is written into it
/// * `constant_values` - the constant values; `p_prime_source_function` is read
///
/// A time-slice which failed to converge carries `NaN` in the coefficients, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so the pressure comes out `NaN` without needing a special
/// case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let p_prime_source_function: &SharedSourceFunction = constant_values.p_prime_source_function;

    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let p_prime_dof_values: &Array1<f64> = &time_slice.source_functions.p_prime.coefficients;

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

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

    time_slice.profiles_1d.pressure = p_profile;
}
