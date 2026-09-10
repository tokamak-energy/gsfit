//! `time_slice(itime)/profiles_1d/f`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, array};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the poloidal-current function `f`, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/f` is written into it
/// * `constant_values` - the constant values; `ff_prime_source_function` and `i_rod` are read
///
/// A time-slice which failed to converge carries `NaN` in the coefficients, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so `f` comes out `NaN` without needing a special case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let ff_prime_source_function: &SharedSourceFunction = constant_values.ff_prime_source_function;
    let i_rod: f64 = constant_values.i_rod;

    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let ff_prime_dof_values: &Array1<f64> = time_slice.source_functions.ff_prime.coefficients.as_ref().unwrap();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    let n_psi_norm: usize = psi_norm.len();

    let mut f_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // f(ψ) = R·B_T(R) is the poloidal-current function.
    // It satisfies:
    //   f²/2 = ∫_{ψ_B}^{ψ} ff′(ψ′) dψ′ + f_vac²/2
    // where f_vac = μ₀·I_rod / (2π) ensures f = f_vac at the boundary (ψ_N = 1).
    //
    // ψ_N = (ψ_A − ψ) / (ψ_A − ψ_B), so dψ/dψ_N = ψ_B − ψ_A.
    // Changing variable to ψ_N:
    //   f² = f_vac² + 2·(ψ_B − ψ_A) · ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′
    //   f  = √( f_vac² + 2·(dψ/dψ_N) · ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′ )
    let f_vac: f64 = i_rod * MU_0 / (2.0 * PI);

    // dψ/dψ_N = ψ_B − ψ_A
    let d_psi_d_psi_norm: f64 = psi_b - psi_a;

    for i_psi_norm in 0..n_psi_norm {
        f_profile[i_psi_norm] = value_from_source_function(psi_norm[i_psi_norm], ff_prime_source_function, ff_prime_dof_values, f_vac, d_psi_d_psi_norm);
    }

    time_slice.profiles_1d.f = Some(f_profile);
}

/// Evaluate the fitted poloidal-current function at one normalised poloidal flux.
pub(in crate::equilibrium_post_processor) fn value_at_psi_norm(
    time_slice: &EquilibriumTimeSlice,
    ff_prime_source_function: &SharedSourceFunction,
    i_rod: f64,
    psi_norm: f64,
) -> f64 {
    let ff_prime_dof_values: &Array1<f64> = time_slice.source_functions.ff_prime.coefficients.as_ref().unwrap();
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let f_vac: f64 = i_rod * MU_0 / (2.0 * PI);
    let d_psi_d_psi_norm: f64 = psi_b - psi_a;

    return value_from_source_function(psi_norm, ff_prime_source_function, ff_prime_dof_values, f_vac, d_psi_d_psi_norm);
}

fn value_from_source_function(
    psi_norm: f64,
    ff_prime_source_function: &SharedSourceFunction,
    ff_prime_dof_values: &Array1<f64>,
    f_vac: f64,
    d_psi_d_psi_norm: f64,
) -> f64 {
    // ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′
    let ff_prime_integral: f64 = ff_prime_source_function.source_function_integral(&array![psi_norm], ff_prime_dof_values)[0];

    // Preserve the sign of the vacuum boundary condition when taking the square root.
    let f_sign: f64 = if f_vac >= 0.0 { 1.0 } else { -1.0 };
    return f_sign * (f_vac * f_vac + 2.0 * d_psi_d_psi_norm * ff_prime_integral).sqrt();
}
