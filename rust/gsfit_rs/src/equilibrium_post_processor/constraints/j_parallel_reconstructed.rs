//! `time_slice(itime)/constraints/j_parallel(i1)/reconstructed`, with `.../position/psi`, `.../position/rho_tor_norm` and `.../chi_squared`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::interpolate_profile::interpolate_profile;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the flux-surface averaged parallel current at each constraint's flux surface in the reconstructed equilibrium, and store it
/// as the constraint's `reconstructed`, with the poloidal flux and normalised toroidal flux there as its `position/psi` and
/// `position/rho_tor_norm`, and, where both its `measured` and `reconstructed` are finite, its `chi_squared`.
///
/// The constraint is on the flux surface `position/psi_norm`, where the reconstructed equilibrium's parallel current is
///
/// ```text
/// <J . B> / B0 = 2 pi (F p' + <B ** 2> F F' / (mu_0 F)) / B0
/// ```
///
/// with `p'` and `F F'` the source functions, `source_functions/p_prime/coefficients` and `.../ff_prime/coefficients`, there, and `F` and
/// `<B ** 2>` the reconstructed equilibrium's `profiles_1d/f` and `profiles_1d/gm5`, interpolated linearly in `psi_norm`. These are the
/// equilibrium's own, not the constraint's `f` and `gm5`, which the solver held them at; the two agree once the equilibrium the
/// constraints were calculated on is this one. `chi_squared` is the data dictionary's, `weight ** 2 * (reconstructed - measured) ** 2 /
/// sigma ** 2`, with the constraint's `expected_value` as `sigma`, as for the pressure constraints.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/j_parallel(:)/reconstructed`, `.../position/psi`, `.../position/rho_tor_norm` and
///   `.../chi_squared` are written into it
/// * `constant_values` - the constant values; `b0`, `p_prime_source_function` and `ff_prime_source_function` are read
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_j_parallel: usize = time_slice.constraints.j_parallel.len();
    if n_j_parallel == 0 {
        return;
    }

    let psi_axis: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_boundary: f64 = time_slice.boundary.psi;
    let b0: f64 = constant_values.b0;
    let p_prime_dof_values: Array1<f64> = time_slice.source_functions.p_prime.coefficients.to_owned();
    let ff_prime_dof_values: Array1<f64> = time_slice.source_functions.ff_prime.coefficients.to_owned();
    let profiles_1d_psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;

    let mut psi_at_position: Array1<f64> = Array1::from_elem(n_j_parallel, f64::NAN);
    let mut rho_tor_norm_at_position: Array1<f64> = Array1::from_elem(n_j_parallel, f64::NAN);
    let mut reconstructed: Array1<f64> = Array1::from_elem(n_j_parallel, f64::NAN);
    for i_j_parallel in 0..n_j_parallel {
        let psi_norm: f64 = time_slice.constraints.j_parallel[i_j_parallel].position.psi_norm;
        if !(0.0..=1.0).contains(&psi_norm) {
            continue;
        }
        psi_at_position[i_j_parallel] = psi_axis + psi_norm * (psi_boundary - psi_axis);
        if time_slice.profiles_1d.rho_tor_norm.len() == profiles_1d_psi_norm.len() {
            rho_tor_norm_at_position[i_j_parallel] = interpolate_profile(psi_norm, profiles_1d_psi_norm, &time_slice.profiles_1d.rho_tor_norm);
        }
        if time_slice.profiles_1d.f.len() != profiles_1d_psi_norm.len() || time_slice.profiles_1d.gm5.len() != profiles_1d_psi_norm.len() {
            continue;
        }
        let f: f64 = interpolate_profile(psi_norm, profiles_1d_psi_norm, &time_slice.profiles_1d.f);
        let b_field_squared_average: f64 = interpolate_profile(psi_norm, profiles_1d_psi_norm, &time_slice.profiles_1d.gm5);
        let psi_norm_here: Array1<f64> = Array1::from_vec(vec![psi_norm]);
        let p_prime: f64 = constant_values
            .p_prime_source_function
            .source_function_value(&psi_norm_here, &p_prime_dof_values)[0];
        let ff_prime: f64 = constant_values
            .ff_prime_source_function
            .source_function_value(&psi_norm_here, &ff_prime_dof_values)[0];
        reconstructed[i_j_parallel] = 2.0 * PI * (f * p_prime + b_field_squared_average * ff_prime / (MU_0 * f)) / b0;
    }

    for i_j_parallel in 0..n_j_parallel {
        let constraint = &mut time_slice.constraints.j_parallel[i_j_parallel];
        constraint.position.psi = psi_at_position[i_j_parallel];
        constraint.position.rho_tor_norm = rho_tor_norm_at_position[i_j_parallel];
        constraint.reconstructed = reconstructed[i_j_parallel];
        if constraint.measured.is_finite() && constraint.reconstructed.is_finite() {
            constraint.chi_squared = constraint.weight.powi(2) * (constraint.reconstructed - constraint.measured).powi(2) / constraint.expected_value.powi(2);
        } else {
            constraint.chi_squared = f64::NAN;
        }
    }
}
