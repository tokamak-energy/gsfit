//! `time_slice(itime)/profiles_1d/phi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the toroidal flux profile, and store it in the time-slice.
///
/// The toroidal flux follows from the definition of the safety factor, `q = d(phi) / d(psi)`,
/// integrated outwards from the magnetic axis where the enclosed toroidal flux is zero.
/// `profiles_1d/q` is NaN at exactly `psi_norm = 1`, but its singularity at a diverted separatrix
/// is integrable. The final interval is therefore integrated from the last two finite `q` values
/// without manufacturing a finite boundary value in the stored `q` profile.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/phi` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

    // A slice which did not converge has no safety factor to integrate. Without this the flux at
    // the magnetic axis would come out as the hard-coded 0.0 rather than NaN
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        time_slice.profiles_1d.phi = Array1::from_elem(n_psi_norm, f64::NAN);
        return;
    }

    let q_profile: &Array1<f64> = &time_slice.profiles_1d.q;
    let psi_profile: &Array1<f64> = &time_slice.profiles_1d.psi;

    let boundary_diverted: bool = time_slice.boundary.r#type == 1;
    let flux_toroidal_profile: Array1<f64> = epp_flux_toroidal_profile(q_profile, psi_profile, boundary_diverted);

    time_slice.profiles_1d.phi = flux_toroidal_profile;
}

/// Integrate `q` to give the toroidal flux profile.
///
/// Kept separate from the writer above because it is evaluated twice: once with the reconstructed
/// `q`, and once with the vacuum `q`, which is what the diamagnetic flux is measured against.
///
/// Finite intervals use the trapezoidal rule. When the final `q` is NaN, as required at
/// `psi_norm = 1`, the last interval is handled according to the boundary topology:
///
/// * A diverted boundary uses `q(x) = a + b * ln(1 - x)`, fitted to the last two finite points.
///   This is the logarithmic separatrix asymptote, whose integral remains finite at `x = 1`.
/// * A limited boundary linearly extrapolates `q` from the last two finite points and integrates
///   that linear model. The safety factor itself is finite there; only its stored endpoint is NaN.
///
/// Here `x = psi_norm`, recovered from the linear `psi` profile. If either fit cannot be formed,
/// the final toroidal-flux value remains NaN rather than silently using an arbitrary endpoint.
///
/// # Arguments
/// * `q_profile` - the safety factor profile [dimensionless]
/// * `psi_profile` - the poloidal flux profile [weber]
/// * `boundary_diverted` - whether the plasma boundary passes through an X-point
///
/// # Returns
/// * `flux_toroidal_profile` - the enclosed toroidal flux profile [weber]
pub(in crate::equilibrium_post_processor) fn epp_flux_toroidal_profile(
    q_profile: &Array1<f64>,
    psi_profile: &Array1<f64>,
    boundary_diverted: bool,
) -> Array1<f64> {
    let n_psi_n: usize = psi_profile.len();
    assert_eq!(q_profile.len(), n_psi_n);

    let mut flux_toroidal_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    if n_psi_n == 0 {
        return flux_toroidal_profile;
    }

    flux_toroidal_profile[0] = 0.0; // no toroidal flux at the magnetic axis
    for i_psi_n in 1..n_psi_n.saturating_sub(1) {
        let avg_y: f64 = (q_profile[i_psi_n] + q_profile[i_psi_n - 1]) / 2.0;
        let dx: f64 = psi_profile[i_psi_n] - psi_profile[i_psi_n - 1];
        flux_toroidal_profile[i_psi_n] = flux_toroidal_profile[i_psi_n - 1] - avg_y * dx;
    }

    if n_psi_n >= 2 {
        let i_boundary: usize = n_psi_n - 1;
        if q_profile[i_boundary].is_finite() {
            let avg_q: f64 = 0.5 * (q_profile[i_boundary - 1] + q_profile[i_boundary]);
            let delta_psi: f64 = psi_profile[i_boundary] - psi_profile[i_boundary - 1];
            flux_toroidal_profile[i_boundary] = flux_toroidal_profile[i_boundary - 1] - avg_q * delta_psi;
        } else if n_psi_n >= 3 && flux_toroidal_profile[i_boundary - 1].is_finite() {
            let integral_q_d_psi: Option<f64> = boundary_interval_q_integral(q_profile, psi_profile, boundary_diverted);
            if let Some(integral_q_d_psi) = integral_q_d_psi {
                flux_toroidal_profile[i_boundary] = flux_toroidal_profile[i_boundary - 1] - integral_q_d_psi;
            }
        }
    }

    return flux_toroidal_profile;
}

/// Integrate `q d(psi)` over the final profile interval when `q(psi_norm = 1)` is NaN.
fn boundary_interval_q_integral(q_profile: &Array1<f64>, psi_profile: &Array1<f64>, boundary_diverted: bool) -> Option<f64> {
    let n_psi_n: usize = q_profile.len();
    let i_before: usize = n_psi_n - 2;
    let i_before_previous: usize = n_psi_n - 3;

    let psi_span: f64 = psi_profile[n_psi_n - 1] - psi_profile[0];
    if psi_span == 0.0 || !psi_span.is_finite() || !q_profile[i_before].is_finite() || !q_profile[i_before_previous].is_finite() {
        return None;
    }

    let psi_norm_before: f64 = (psi_profile[i_before] - psi_profile[0]) / psi_span;
    let psi_norm_before_previous: f64 = (psi_profile[i_before_previous] - psi_profile[0]) / psi_span;
    let delta_psi_norm: f64 = 1.0 - psi_norm_before;
    if !(0.0..1.0).contains(&psi_norm_before_previous) || !(psi_norm_before_previous..1.0).contains(&psi_norm_before) || delta_psi_norm == 0.0 {
        return None;
    }

    let integral_q_d_psi_norm: f64;
    if boundary_diverted {
        let log_distance_before: f64 = delta_psi_norm.ln();
        let log_distance_before_previous: f64 = (1.0 - psi_norm_before_previous).ln();
        let log_coefficient: f64 = (q_profile[i_before] - q_profile[i_before_previous]) / (log_distance_before - log_distance_before_previous);
        integral_q_d_psi_norm = delta_psi_norm * (q_profile[i_before] - log_coefficient);
    } else {
        let q_slope: f64 = (q_profile[i_before] - q_profile[i_before_previous]) / (psi_norm_before - psi_norm_before_previous);
        let q_boundary_extrapolated: f64 = q_profile[i_before] + q_slope * delta_psi_norm;
        integral_q_d_psi_norm = 0.5 * delta_psi_norm * (q_profile[i_before] + q_boundary_extrapolated);
    }

    let integral_q_d_psi: f64 = psi_span * integral_q_d_psi_norm;
    return integral_q_d_psi.is_finite().then_some(integral_q_d_psi);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn diverted_boundary_integrates_a_logarithmic_q_singularity() {
        let psi_profile: Array1<f64> = array![2.0, 1.5, 1.25, 1.0];
        let psi_norm: Array1<f64> = (&psi_profile - psi_profile[0]) / (psi_profile[psi_profile.len() - 1] - psi_profile[0]);
        let intercept: f64 = 1.2;
        let log_coefficient: f64 = -0.4;
        let mut q_profile: Array1<f64> = psi_norm.mapv(|psi_norm_here| intercept + log_coefficient * (1.0 - psi_norm_here).ln());
        let i_boundary: usize = q_profile.len() - 1;
        q_profile[i_boundary] = f64::NAN;

        let flux_toroidal_profile: Array1<f64> = epp_flux_toroidal_profile(&q_profile, &psi_profile, true);
        let i_before: usize = q_profile.len() - 2;
        let delta_psi_norm: f64 = 1.0 - psi_norm[i_before];
        let final_interval_expected: f64 = (psi_profile[psi_profile.len() - 1] - psi_profile[0]) * delta_psi_norm * (q_profile[i_before] - log_coefficient);

        assert_abs_diff_eq!(
            flux_toroidal_profile[flux_toroidal_profile.len() - 1] - flux_toroidal_profile[i_before],
            -final_interval_expected,
            epsilon = 1e-14
        );
        assert!(flux_toroidal_profile[flux_toroidal_profile.len() - 1].is_finite());
    }

    #[test]
    fn limited_boundary_integrates_a_linear_q_extrapolation() {
        let psi_profile: Array1<f64> = array![0.0, 0.4, 0.7, 1.0];
        let mut q_profile: Array1<f64> = psi_profile.mapv(|psi| 2.0 + 3.0 * psi);
        let i_boundary: usize = q_profile.len() - 1;
        q_profile[i_boundary] = f64::NAN;

        let flux_toroidal_profile: Array1<f64> = epp_flux_toroidal_profile(&q_profile, &psi_profile, false);
        let i_before: usize = q_profile.len() - 2;
        let final_interval_expected: f64 = 2.0 * (1.0 - psi_profile[i_before]) + 1.5 * (1.0 - psi_profile[i_before].powi(2));

        assert_abs_diff_eq!(
            flux_toroidal_profile[flux_toroidal_profile.len() - 1] - flux_toroidal_profile[i_before],
            -final_interval_expected,
            epsilon = 1e-14
        );
        assert!(flux_toroidal_profile[flux_toroidal_profile.len() - 1].is_finite());
    }
}
