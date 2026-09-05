//! `time_slice(itime)/profiles_1d/f`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, array};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the poloidal-current function `f`, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/f` is written into it
/// * `ff_prime_source_function` - the FF' source function the reconstruction was run with
///
/// A time-slice which failed to converge carries `NaN` in the coefficients, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so `f` comes out `NaN` without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_1d_f(time_slice: &mut EquilibriumTimeSlice, ff_prime_source_function: &SharedSourceFunction) {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let ff_prime_dof_values: &Array1<f64> = time_slice.source_functions.ff_prime.coefficients.as_ref().unwrap();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let i_rod: f64 = time_slice.global_quantities.i_rod.unwrap();

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
        // ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′
        let ff_prime_integral: f64 = ff_prime_source_function.source_function_integral(&array![psi_norm[i_psi_norm]], ff_prime_dof_values)[0];

        // f = sign(f_vac)·√( f_vac² + 2·(dψ/dψ_N)·∫_1^{ψ_N} ff′ dψ_N′ )
        // The sign of f_vac must be preserved so that a negative TF rod current
        // (f_vac < 0) yields a negative f, matching the vacuum boundary condition
        // f(ψ_N = 1) = f_vac.
        let f_sign: f64 = if f_vac >= 0.0 { 1.0 } else { -1.0 };
        f_profile[i_psi_norm] = f_sign * (f_vac * f_vac + 2.0 * d_psi_d_psi_norm * ff_prime_integral).sqrt();
    }

    time_slice.profiles_1d.f = Some(f_profile);
}
