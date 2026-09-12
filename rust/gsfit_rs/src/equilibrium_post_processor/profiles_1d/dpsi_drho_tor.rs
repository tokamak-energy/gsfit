//! `time_slice(itime)/profiles_1d/dpsi_drho_tor`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the derivative of poloidal flux with respect to toroidal-flux radius.
///
/// `rho_tor` is generally not uniformly spaced, so the derivative is evaluated by differentiating
/// the quadratic through each point and its two neighbours. The final point uses the corresponding
/// three-point backward stencil.
///
/// At the magnetic axis, `psi - psi_axis` is proportional to `rho_tor ** 2`, making
/// `dpsi_drho_tor` exactly zero. It is assigned from that limit rather than from a forward
/// difference. The derivative retains the sign of the poloidal-flux convention.
///
/// Points without three finite, distinct `rho_tor` coordinates remain NaN. With only two profile
/// points, the boundary uses their first-order secant while the axis remains zero.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/dpsi_drho_tor` is written into it
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi: &Array1<f64> = &time_slice.profiles_1d.psi;
    let rho_tor: &Array1<f64> = &time_slice.profiles_1d.rho_tor;
    let n_psi_norm: usize = psi.len();
    assert_eq!(rho_tor.len(), n_psi_norm);

    let mut dpsi_drho_tor: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    if n_psi_norm == 0 || time_slice.global_quantities.psi_magnetic_axis.is_nan() {
        time_slice.profiles_1d.dpsi_drho_tor = dpsi_drho_tor;
        return;
    }

    if psi[0].is_finite() && rho_tor[0].is_finite() {
        dpsi_drho_tor[0] = 0.0;
    }

    if n_psi_norm == 1 {
        time_slice.profiles_1d.dpsi_drho_tor = dpsi_drho_tor;
        return;
    }

    if n_psi_norm == 2 {
        dpsi_drho_tor[1] = first_derivative_from_two_points(rho_tor[0], rho_tor[1], psi[0], psi[1]).unwrap_or(f64::NAN);
        time_slice.profiles_1d.dpsi_drho_tor = dpsi_drho_tor;
        return;
    }

    for i_psi_norm in 1..n_psi_norm - 1 {
        dpsi_drho_tor[i_psi_norm] = first_derivative_from_three_points(
            rho_tor[i_psi_norm - 1],
            rho_tor[i_psi_norm],
            rho_tor[i_psi_norm + 1],
            psi[i_psi_norm - 1],
            psi[i_psi_norm],
            psi[i_psi_norm + 1],
            rho_tor[i_psi_norm],
        )
        .unwrap_or(f64::NAN);
    }

    dpsi_drho_tor[n_psi_norm - 1] = first_derivative_from_three_points(
        rho_tor[n_psi_norm - 3],
        rho_tor[n_psi_norm - 2],
        rho_tor[n_psi_norm - 1],
        psi[n_psi_norm - 3],
        psi[n_psi_norm - 2],
        psi[n_psi_norm - 1],
        rho_tor[n_psi_norm - 1],
    )
    .unwrap_or(f64::NAN);

    time_slice.profiles_1d.dpsi_drho_tor = dpsi_drho_tor;
}

fn first_derivative_from_two_points(x_0: f64, x_1: f64, y_0: f64, y_1: f64) -> Option<f64> {
    let delta_x: f64 = x_1 - x_0;
    if !x_0.is_finite() || !x_1.is_finite() || !y_0.is_finite() || !y_1.is_finite() || delta_x == 0.0 {
        return None;
    }
    Some((y_1 - y_0) / delta_x)
}

/// Differentiate the quadratic through three points at `x_evaluate`.
#[allow(clippy::too_many_arguments)]
fn first_derivative_from_three_points(x_0: f64, x_1: f64, x_2: f64, y_0: f64, y_1: f64, y_2: f64, x_evaluate: f64) -> Option<f64> {
    if !x_0.is_finite()
        || !x_1.is_finite()
        || !x_2.is_finite()
        || !y_0.is_finite()
        || !y_1.is_finite()
        || !y_2.is_finite()
        || !x_evaluate.is_finite()
        || x_0 == x_1
        || x_0 == x_2
        || x_1 == x_2
    {
        return None;
    }

    let coefficient_0: f64 = (2.0 * x_evaluate - x_1 - x_2) / ((x_0 - x_1) * (x_0 - x_2));
    let coefficient_1: f64 = (2.0 * x_evaluate - x_0 - x_2) / ((x_1 - x_0) * (x_1 - x_2));
    let coefficient_2: f64 = (2.0 * x_evaluate - x_0 - x_1) / ((x_2 - x_0) * (x_2 - x_1));

    Some(coefficient_0 * y_0 + coefficient_1 * y_1 + coefficient_2 * y_2)
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn derivative_is_exact_for_quadratic_psi_on_a_nonuniform_grid() {
        let coefficient: f64 = -2.5;
        let psi_axis: f64 = 0.7;
        let rho_tor: Array1<f64> = array![0.0, 0.1, 0.35, 0.8];
        let psi: Array1<f64> = rho_tor.mapv(|rho_tor_here| psi_axis + coefficient * rho_tor_here.powi(2));

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = psi_axis;
        time_slice.profiles_1d.psi = psi;
        time_slice.profiles_1d.rho_tor = rho_tor.clone();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let dpsi_drho_tor: &Array1<f64> = &time_slice.profiles_1d.dpsi_drho_tor;
        for i_psi_norm in 0..rho_tor.len() {
            assert_abs_diff_eq!(dpsi_drho_tor[i_psi_norm], 2.0 * coefficient * rho_tor[i_psi_norm], epsilon = 1e-13);
        }
    }

    #[test]
    fn failed_slice_is_all_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = f64::NAN;
        time_slice.profiles_1d.psi = array![f64::NAN, f64::NAN, f64::NAN];
        time_slice.profiles_1d.rho_tor = array![f64::NAN, f64::NAN, f64::NAN];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.dpsi_drho_tor.iter().all(|value| value.is_nan()));
    }
}
