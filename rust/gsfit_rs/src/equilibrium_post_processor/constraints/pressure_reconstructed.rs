//! `time_slice(itime)/constraints/pressure(i1)/reconstructed`, with `.../position/psi` and `.../chi_squared`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::PsiInterpolator;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the pressure at each pressure constraint's position in the reconstructed equilibrium, and store it as the constraint's
/// `reconstructed`, with the poloidal flux there as its `position/psi`, and, where both its `measured` and `reconstructed` are finite,
/// its `chi_squared`.
///
/// Every pressure constraint in `constraints/pressure` gets one. The poloidal flux is interpolated at the constraint's `position/r`,
/// `position/z`. Inside the plasma the pressure is the p' source function, `source_functions/p_prime/coefficients`, integrated from the
/// boundary, where the pressure is zero; outside it, or off the grid, `reconstructed` is NaN. `chi_squared` is the data dictionary's,
/// `weight ** 2 * (reconstructed - measured) ** 2 / sigma ** 2`, with the constraint's `expected_value` as `sigma`, which is the residual
/// the solver fitted, as its least-squares weight is `weight / expected_value`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/pressure(:)/reconstructed`, `.../position/psi` and `.../chi_squared` are written
///   into it
/// * `constant_values` - the constant values; `p_prime_source_function` is read
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_pressure: usize = time_slice.constraints.pressure.len();
    if n_pressure == 0 {
        return;
    }

    let psi_axis: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_boundary: f64 = time_slice.boundary.psi;
    let p_prime_dof_values: Array1<f64> = time_slice.source_functions.p_prime.coefficients.to_owned();

    // The poloidal flux and the pressure at each constraint's position
    let mut psi_at_position: Array1<f64> = Array1::from_elem(n_pressure, f64::NAN);
    let mut reconstructed: Array1<f64> = Array1::from_elem(n_pressure, f64::NAN);
    {
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
        for i_pressure in 0..n_pressure {
            let position_r: f64 = time_slice.constraints.pressure[i_pressure].position.r;
            let position_z: f64 = time_slice.constraints.pressure[i_pressure].position.z;
            let Some(psi) = psi_interpolator.psi(position_r, position_z) else {
                continue;
            };
            psi_at_position[i_pressure] = psi;

            // Outside the plasma the pressure is not evaluated
            let psi_norm: f64 = (psi - psi_axis) / (psi_boundary - psi_axis);
            if !(0.0..=1.0).contains(&psi_norm) {
                continue;
            }

            // `source_function_integral` integrates p'(psi_norm) from psi_norm = 1, where the pressure is zero
            reconstructed[i_pressure] = constant_values
                .p_prime_source_function
                .source_function_integral(&Array1::from_vec(vec![psi_norm]), &p_prime_dof_values)[0]
                * (psi_boundary - psi_axis);
        }
    }

    for i_pressure in 0..n_pressure {
        let constraint = &mut time_slice.constraints.pressure[i_pressure];
        constraint.position.psi = psi_at_position[i_pressure];
        constraint.reconstructed = reconstructed[i_pressure];
        if constraint.measured.is_finite() && constraint.reconstructed.is_finite() {
            constraint.chi_squared = constraint.weight.powi(2) * (constraint.reconstructed - constraint.measured).powi(2) / constraint.expected_value.powi(2);
        } else {
            constraint.chi_squared = f64::NAN;
        }
    }
}
