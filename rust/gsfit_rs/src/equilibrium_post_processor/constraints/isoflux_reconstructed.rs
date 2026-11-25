//! `time_slice(itime)/constraints/isoflux(i1)/reconstructed`, with `.../position_1/psi`, `.../position_2/psi` and `.../chi_squared`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::PsiInterpolator;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the poloidal flux at each isoflux constraint's two points in the reconstructed equilibrium, and store their difference as
/// the constraint's `reconstructed`, with the poloidal flux at each point as its `position_1/psi` and `position_2/psi`, and, where both its
/// `measured` and `reconstructed` are finite, its `chi_squared`.
///
/// Every isoflux constraint in `constraints/isoflux` gets one, whether or not it was fitted. The poloidal flux is interpolated at each
/// point; a point which is not known, or is off the grid, is NaN, and so is `reconstructed`. `chi_squared` is the data dictionary's,
/// `weight ** 2 * (reconstructed - measured) ** 2 / expected_value ** 2`, which is the residual the solver fitted.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/isoflux(:)/reconstructed`, `.../position_1/psi`, `.../position_2/psi` and
///   `.../chi_squared` are written into it
/// * `_constant_values` - not used
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_isoflux: usize = time_slice.constraints.isoflux.len();
    if n_isoflux == 0 {
        return;
    }

    // The poloidal flux at each constraint's two points
    let mut psi_at_position_1: Array1<f64> = Array1::from_elem(n_isoflux, f64::NAN);
    let mut psi_at_position_2: Array1<f64> = Array1::from_elem(n_isoflux, f64::NAN);
    {
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
        for i_isoflux in 0..n_isoflux {
            let (position_1_r, position_1_z): (f64, f64) = (
                time_slice.constraints.isoflux[i_isoflux].position_1.r,
                time_slice.constraints.isoflux[i_isoflux].position_1.z,
            );
            let (position_2_r, position_2_z): (f64, f64) = (
                time_slice.constraints.isoflux[i_isoflux].position_2.r,
                time_slice.constraints.isoflux[i_isoflux].position_2.z,
            );
            if position_1_r.is_finite() && position_1_z.is_finite() {
                psi_at_position_1[i_isoflux] = psi_interpolator.psi(position_1_r, position_1_z).unwrap_or(f64::NAN);
            }
            if position_2_r.is_finite() && position_2_z.is_finite() {
                psi_at_position_2[i_isoflux] = psi_interpolator.psi(position_2_r, position_2_z).unwrap_or(f64::NAN);
            }
        }
    }

    for i_isoflux in 0..n_isoflux {
        let constraint = &mut time_slice.constraints.isoflux[i_isoflux];
        constraint.position_1.psi = psi_at_position_1[i_isoflux];
        constraint.position_2.psi = psi_at_position_2[i_isoflux];
        constraint.reconstructed = psi_at_position_1[i_isoflux] - psi_at_position_2[i_isoflux];
        if constraint.measured.is_finite() && constraint.reconstructed.is_finite() {
            constraint.chi_squared = constraint.weight.powi(2) * (constraint.reconstructed - constraint.measured).powi(2) / constraint.expected_value.powi(2);
        } else {
            constraint.chi_squared = f64::NAN;
        }
    }
}
