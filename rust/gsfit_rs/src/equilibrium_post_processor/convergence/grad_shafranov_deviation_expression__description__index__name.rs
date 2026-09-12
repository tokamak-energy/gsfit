//! `time_slice(itime)/convergence/grad_shafranov_deviation_expression/description`,
//! `.../index` and `.../name`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;

/// Identify the Grad-Shafranov deviation value as the normalized inter-iteration flux residual.
///
/// GSFit evaluates the EFIT-style convergence error from the change in magnetic-axis flux,
/// normalized by the axis-to-boundary flux difference. This is the
/// `max_absolute_psi_residual_norm` entry in the data dictionary's
/// `equilibrium_gs_deviation` identifier table:
///
/// ```text
/// name  = "max_absolute_psi_residual_norm"
/// index = 6
/// ```
///
/// This is metadata, so it is filled even when the equilibrium solve did not converge.
///
/// # Arguments
/// * `time_slice` - the time-slice whose convergence-expression identifier is written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    time_slice.convergence.grad_shafranov_deviation_expression.description =
        "Maximum absolute difference over the plasma poloidal cross-section of the normalised poloidal flux (with normalization being the poloidal flux difference between the axis and boundary) between the current and preceding iteration, on fixed grid points".to_string();
    time_slice.convergence.grad_shafranov_deviation_expression.index = 6;
    time_slice.convergence.grad_shafranov_deviation_expression.name = "max_absolute_psi_residual_norm".to_string();
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;

    #[test]
    fn identifier_records_normalized_flux_residual() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_eq!(time_slice.convergence.grad_shafranov_deviation_expression.index, 6);
        assert_eq!(
            time_slice.convergence.grad_shafranov_deviation_expression.name,
            "max_absolute_psi_residual_norm"
        );
        assert_eq!(
            time_slice.convergence.grad_shafranov_deviation_expression.description,
            "Maximum absolute difference over the plasma poloidal cross-section of the normalised poloidal flux (with normalization being the poloidal flux difference between the axis and boundary) between the current and preceding iteration, on fixed grid points"
        );
    }
}
