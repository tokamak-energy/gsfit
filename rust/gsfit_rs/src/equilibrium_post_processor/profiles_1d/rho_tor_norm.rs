//! `time_slice(itime)/profiles_1d/rho_tor_norm`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the normalised toroidal flux coordinate profile, and store it in the time-slice.
///
/// The data dictionary defines it as
///
/// ```text
/// rho_tor_norm = (rho_tor - rho_tor(magnetic axis)) / (rho_tor(boundary) - rho_tor(magnetic axis))
/// ```
///
/// and the enclosed toroidal flux is zero on the magnetic axis, so `rho_tor(magnetic axis)` is zero
/// and this reduces to `rho_tor / rho_tor(boundary)`. It runs from 0 on the axis to 1 on the
/// boundary, and is the dimensionless counterpart of `rho_pol = sqrt(psi_norm)`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/rho_tor_norm` is written into it
///
/// A slice which did not converge carries NaN in `rho_tor`, so the profile comes out NaN without
/// needing a special case.
///
/// Note: the boundary value is taken from the last point of the profile rather than from the
/// largest, because `rho_tor` increases outwards by construction and the last point is the boundary
/// by definition. Taking the largest would divide by zero for a profile which runs the other way.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let rho_tor: &Array1<f64> = time_slice.profiles_1d.rho_tor.as_ref().unwrap();
    let n_psi_norm: usize = rho_tor.len();

    let rho_tor_boundary: f64 = rho_tor[n_psi_norm - 1];

    let rho_tor_norm: Array1<f64> = rho_tor.mapv(|rho_tor_here| rho_tor_here / rho_tor_boundary);

    time_slice.profiles_1d.rho_tor_norm = Some(rho_tor_norm);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn rho_tor_norm_runs_from_zero_on_the_axis_to_one_on_the_boundary() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.rho_tor = Some(array![0.0, 0.1, 0.2, 0.4]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let rho_tor_norm: Array1<f64> = time_slice.profiles_1d.rho_tor_norm.unwrap();

        assert_abs_diff_eq!(rho_tor_norm[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_tor_norm[1], 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_tor_norm[2], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(rho_tor_norm[3], 1.0, epsilon = 1e-15);
    }
}
