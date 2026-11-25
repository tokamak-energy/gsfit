//! `time_slice(itime)/constraints/isoflux_boundary(i1)/reconstructed`, with `.../position/psi` and `.../chi_squared`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::PsiInterpolator;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the poloidal flux at each isoflux boundary constraint's point in the reconstructed equilibrium, and store it, less the
/// poloidal flux at the boundary, `boundary/psi`, as the constraint's `reconstructed`, with the poloidal flux at the point as its
/// `position/psi`, and, where both its `measured` and `reconstructed` are finite, its `chi_squared`.
///
/// Every isoflux boundary constraint in `constraints/isoflux_boundary` gets one, whether or not it was fitted. The poloidal flux is
/// interpolated at the point; a point which is not known, or is off the grid, is NaN, and so is `reconstructed`. `chi_squared` is the data
/// dictionary's, `weight ** 2 * (reconstructed - measured) ** 2 / expected_value ** 2`, which is the residual the solver fitted.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/isoflux_boundary(:)/reconstructed`, `.../position/psi` and `.../chi_squared` are
///   written into it
/// * `_constant_values` - not used
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_isoflux_boundary: usize = time_slice.constraints.isoflux_boundary.len();
    if n_isoflux_boundary == 0 {
        return;
    }

    // The poloidal flux at each constraint's point
    let mut psi_at_position: Array1<f64> = Array1::from_elem(n_isoflux_boundary, f64::NAN);
    {
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            let position_r: f64 = time_slice.constraints.isoflux_boundary[i_isoflux_boundary].position.r;
            let position_z: f64 = time_slice.constraints.isoflux_boundary[i_isoflux_boundary].position.z;
            if position_r.is_finite() && position_z.is_finite() {
                psi_at_position[i_isoflux_boundary] = psi_interpolator.psi(position_r, position_z).unwrap_or(f64::NAN);
            }
        }
    }

    let psi_boundary: f64 = time_slice.boundary.psi;
    for i_isoflux_boundary in 0..n_isoflux_boundary {
        let constraint = &mut time_slice.constraints.isoflux_boundary[i_isoflux_boundary];
        constraint.position.psi = psi_at_position[i_isoflux_boundary];
        constraint.reconstructed = psi_at_position[i_isoflux_boundary] - psi_boundary;
        if constraint.measured.is_finite() && constraint.reconstructed.is_finite() {
            constraint.chi_squared = constraint.weight.powi(2) * (constraint.reconstructed - constraint.measured).powi(2) / constraint.expected_value.powi(2);
        } else {
            constraint.chi_squared = f64::NAN;
        }
    }
}
