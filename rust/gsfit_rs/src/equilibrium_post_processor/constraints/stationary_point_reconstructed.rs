//! `time_slice(itime)/constraints/stationary_point(i1)/d_psi_d_r/reconstructed` and `.../d_psi_d_z/reconstructed`, with
//! `.../position/psi` and each derivative's `chi_squared`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::{PsiDerivatives, PsiInterpolator};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate both derivatives of the poloidal flux at each stationary point constraint's point in the reconstructed equilibrium, and store
/// them as the `reconstructed` of the constraint's `d_psi_d_r` and `d_psi_d_z`, with the poloidal flux at the point as its `position/psi`,
/// and, where a derivative's `measured` and `reconstructed` are both finite, its `chi_squared`.
///
/// Every stationary point constraint in `constraints/stationary_point` gets one, whether or not it was fitted. The derivatives are
/// interpolated at the point; a point which is not known, or is off the grid, is NaN. Each `chi_squared` is the data dictionary's,
/// `weight ** 2 * (reconstructed - measured) ** 2 / expected_value ** 2`, with the stationary point's `weight` and `expected_value`, which
/// is the residual the solver fitted.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/stationary_point(:)/d_psi_d_r/reconstructed`, `.../d_psi_d_z/reconstructed`,
///   `.../position/psi` and each derivative's `chi_squared` are written into it
/// * `_constant_values` - not used
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_stationary_point: usize = time_slice.constraints.stationary_point.len();
    if n_stationary_point == 0 {
        return;
    }

    // The poloidal flux, and both of its derivatives, at each constraint's point
    let mut psi_at_position: Array1<f64> = Array1::from_elem(n_stationary_point, f64::NAN);
    let mut d_psi_d_r_at_position: Array1<f64> = Array1::from_elem(n_stationary_point, f64::NAN);
    let mut d_psi_d_z_at_position: Array1<f64> = Array1::from_elem(n_stationary_point, f64::NAN);
    {
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
        for i_stationary_point in 0..n_stationary_point {
            let position_r: f64 = time_slice.constraints.stationary_point[i_stationary_point].position.r;
            let position_z: f64 = time_slice.constraints.stationary_point[i_stationary_point].position.z;
            if !(position_r.is_finite() && position_z.is_finite()) {
                continue;
            }
            psi_at_position[i_stationary_point] = psi_interpolator.psi(position_r, position_z).unwrap_or(f64::NAN);
            if let Some(derivatives) = psi_interpolator.derivatives(position_r, position_z) {
                let derivatives: PsiDerivatives = derivatives;
                d_psi_d_r_at_position[i_stationary_point] = derivatives.d_psi_d_r;
                d_psi_d_z_at_position[i_stationary_point] = derivatives.d_psi_d_z;
            }
        }
    }

    for i_stationary_point in 0..n_stationary_point {
        let constraint = &mut time_slice.constraints.stationary_point[i_stationary_point];
        constraint.position.psi = psi_at_position[i_stationary_point];
        let (weight, expected_value): (f64, f64) = (constraint.weight, constraint.expected_value);
        for (derivative, reconstructed) in [
            (&mut constraint.d_psi_d_r, d_psi_d_r_at_position[i_stationary_point]),
            (&mut constraint.d_psi_d_z, d_psi_d_z_at_position[i_stationary_point]),
        ] {
            derivative.reconstructed = reconstructed;
            if derivative.measured.is_finite() && derivative.reconstructed.is_finite() {
                derivative.chi_squared = weight.powi(2) * (derivative.reconstructed - derivative.measured).powi(2) / expected_value.powi(2);
            } else {
                derivative.chi_squared = f64::NAN;
            }
        }
    }
}
