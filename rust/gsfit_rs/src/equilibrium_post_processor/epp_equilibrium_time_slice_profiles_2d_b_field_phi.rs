//! `time_slice(itime)/profiles_2d(0)/b_field_phi`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, array, s};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the toroidal magnetic field on the grid in the poloidal plane, and store it in the
/// time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/b_field_phi` is written into it
/// * `ff_prime_source_function` - the FF' source function the reconstruction was run with
///
/// A time-slice which failed to converge gets `NaN` everywhere. Without the guard the vacuum field
/// is written unconditionally before the mask is applied, so the array would come back as the
/// vacuum field rather than `NaN` - which is what the old post-processor produced, because it
/// skipped failed slices entirely and left the array at its `NaN` initialisation.
pub fn epp_equilibrium_time_slice_profiles_2d_b_field_phi(time_slice: &mut EquilibriumTimeSlice, ff_prime_source_function: &SharedSourceFunction) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let mask: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();
    let psi_norm_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();

    let ff_prime_dof_values: &Array1<f64> = time_slice.source_functions.ff_prime.coefficients.as_ref().unwrap();
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let i_rod: f64 = time_slice.global_quantities.i_rod.unwrap();

    let (n_z, n_r): (usize, usize) = mask.dim();

    let mut bt_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);

    // A slice which did not converge has no plasma, so no toroidal field is reported for it
    if psi_a.is_nan() {
        time_slice.profiles_2d[0].b_field_phi = Some(bt_2d);
        return;
    }

    // BT vacuum
    let bt_vac_vs_r: Array1<f64> = MU_0 * i_rod / (2.0 * PI * r);
    for i_z in 0..n_z {
        bt_2d.slice_mut(s![i_z, ..]).assign(&bt_vac_vs_r);
    }

    // ψ_N = (ψ_A − ψ) / (ψ_A − ψ_B), so:
    //   ψ = ψ_A − (ψ_A − ψ_B)·ψ_N
    //   dψ/dψ_N = ψ_B − ψ_A
    //
    // Outside the plasma:
    //   B_T(R) = μ₀·I_rod / (2π·R)
    //
    // Inside the plasma:
    //   B_T(R) = f(ψ) / R
    // where f is defined by the identity:
    //   f²/2 = ∫_{ψ_B}^{ψ} ff′(ψ′) dψ′ + f_vac²/2
    // with f_vac = μ₀·I_rod / (2π), which ensures f = f_vac at the boundary (ψ_N = 1).
    //
    // Changing integration variable to ψ_N:
    //   f²/2 = (ψ_B − ψ_A) · ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′ + f_vac²/2
    //
    // Rearranging:
    //   f² = f_vac² + 2·(ψ_B − ψ_A) · ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′
    //   f  = √( f_vac² + 2·(ψ_B − ψ_A) · ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′ )
    //
    // Note: `source_function_integral` integrates from 1 to ψ_N, so it is zero at ψ_N = 1
    // and we recover f = f_vac at the boundary as required.
    let f_vac: f64 = i_rod * MU_0 / (2.0 * PI);

    // dψ/dψ_N = ψ_B − ψ_A
    let d_psi_d_psi_norm: f64 = psi_b - psi_a;

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            if mask[(i_z, i_r)] > 0.99 {
                // ∫_1^{ψ_N} ff′(ψ_N′) dψ_N′
                let ff_prime_integral: f64 = ff_prime_source_function.source_function_integral(&array![psi_norm_2d[(i_z, i_r)]], ff_prime_dof_values)[0];

                // f = sign(f_vac)·√( f_vac² + 2·(dψ/dψ_N)·∫_1^{ψ_N} ff′ dψ_N′ )
                // The sign of f_vac must be preserved so that a negative TF rod current
                // (f_vac < 0) yields a negative f inside the plasma, matching the vacuum
                // boundary condition f(ψ_N = 1) = f_vac.
                let f_sign: f64 = if f_vac >= 0.0 { 1.0 } else { -1.0 };
                let f_at_this_rz: f64 = f_sign * (f_vac * f_vac + 2.0 * d_psi_d_psi_norm * ff_prime_integral).sqrt();

                // Toroidal field
                bt_2d[(i_z, i_r)] = f_at_this_rz / r[i_r];
            }
        }
    }

    time_slice.profiles_2d[0].b_field_phi = Some(bt_2d);
}
