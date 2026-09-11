//! `time_slice(itime)/profiles_1d/magnetic_shear`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the magnetic shear profile, and store it in the time-slice.
///
/// This is the data dictionary's definition, which is dimensionless because the metre of `rho_tor`
/// cancels against the metre of `d(rho_tor)`:
///
/// ```text
/// magnetic_shear = (rho_tor / q) * d(q)/d(rho_tor)
/// ```
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/magnetic_shear` is written into it
///
/// A slice which did not converge carries NaN in both `q` and `rho_tor`, so the profile comes out
/// NaN without needing a special case.
///
/// # The magnetic axis
///
/// `rho_tor` is zero on the magnetic axis, and `q` is finite there, so the shear is exactly zero at
/// the first point. That is the right limit: `q` is even in `rho_tor` about the axis, so
/// `d(q)/d(rho_tor)` vanishes there too and the product goes to zero from both sides.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let q_profile: &Array1<f64> = &time_slice.profiles_1d.q;
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;
    let n_psi_norm: usize = rho_tor.len();

    let d_q_d_rho_tor: Array1<f64> = epp_d_q_d_rho_tor(q_profile, rho_tor);

    let mut magnetic_shear: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        magnetic_shear[i_psi_norm] = rho_tor[i_psi_norm] * d_q_d_rho_tor[i_psi_norm] / q_profile[i_psi_norm];
    }

    time_slice.profiles_1d.magnetic_shear = magnetic_shear;
}

/// Differentiate `q` with respect to `rho_tor`, one value per flux surface.
///
/// # Arguments
/// * `q_profile` - the safety factor profile, `profiles_1d/q` [dimensionless]
/// * `rho_tor` - the toroidal flux coordinate profile, `profiles_1d/rho_tor` [metre]
///
/// # Returns
/// * `d_q_d_rho_tor` - one value per flux surface [1 / metre]
///
/// `rho_tor` increases outwards from the magnetic axis but is **not** evenly spaced - it goes
/// roughly as the square root of `psi_norm` - so each difference is divided by the span of
/// `rho_tor` it was taken over rather than by a single grid spacing: a forward difference at the
/// first point, central differences in the interior, and a backward difference at the last point.
fn epp_d_q_d_rho_tor(q_profile: &Array1<f64>, rho_tor: &Array1<f64>) -> Array1<f64> {
    let n_psi_norm: usize = rho_tor.len();

    let mut d_q_d_rho_tor: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    d_q_d_rho_tor[0] = (q_profile[1] - q_profile[0]) / (rho_tor[1] - rho_tor[0]);
    for i_psi_norm in 1..n_psi_norm - 1 {
        d_q_d_rho_tor[i_psi_norm] = (q_profile[i_psi_norm + 1] - q_profile[i_psi_norm - 1]) / (rho_tor[i_psi_norm + 1] - rho_tor[i_psi_norm - 1]);
    }
    d_q_d_rho_tor[n_psi_norm - 1] = (q_profile[n_psi_norm - 1] - q_profile[n_psi_norm - 2]) / (rho_tor[n_psi_norm - 1] - rho_tor[n_psi_norm - 2]);

    d_q_d_rho_tor
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;

    /// For `q = q_axis * (1 + k * rho_tor ** 2)` the shear is analytic:
    ///
    /// ```text
    /// magnetic_shear = (rho_tor / q) * 2 * k * q_axis * rho_tor
    ///                = 2 * k * rho_tor ** 2 / (1 + k * rho_tor ** 2)
    /// ```
    ///
    /// which is the shape `q` takes near the magnetic axis, where it is even in `rho_tor`
    #[test]
    fn magnetic_shear_matches_the_analytic_profile_for_a_parabolic_q() {
        let q_axis: f64 = 1.2;
        let k: f64 = 3.0;

        // Unevenly spaced, as `rho_tor` really is: `rho_tor` goes as the square root of `psi_norm`
        let psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, 2001);
        let rho_tor: Array1<f64> = psi_norm.mapv(|psi_norm_here| 0.4 * psi_norm_here.sqrt());
        let q_profile: Array1<f64> = rho_tor.mapv(|rho_tor_here| q_axis * (1.0 + k * rho_tor_here.powi(2)));

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.q = q_profile;
        time_slice.profiles_1d.rho_tor = rho_tor.clone();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let magnetic_shear: Array1<f64> = time_slice.profiles_1d.magnetic_shear;

        // Zero on the magnetic axis, exactly
        assert_abs_diff_eq!(magnetic_shear[0], 0.0, epsilon = 1e-15);

        // In the interior, where the central difference is used. It is exact for a quadratic on an
        // even grid, so all that is left is the unevenness of `rho_tor` itself
        let rho_tor_interior: f64 = rho_tor[1000];
        let expected_interior: f64 = 2.0 * k * rho_tor_interior.powi(2) / (1.0 + k * rho_tor_interior.powi(2));
        assert_abs_diff_eq!(magnetic_shear[1000], expected_interior, epsilon = 1e-7);

        // At the boundary, where the one-sided backward difference is used. That is first order in
        // the grid spacing, so it is a couple of orders of magnitude less accurate
        let rho_tor_boundary: f64 = rho_tor[2000];
        let expected_boundary: f64 = 2.0 * k * rho_tor_boundary.powi(2) / (1.0 + k * rho_tor_boundary.powi(2));
        assert_abs_diff_eq!(magnetic_shear[2000], expected_boundary, epsilon = 1e-4);
    }
}
