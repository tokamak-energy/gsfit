//! `time_slice(itime)/profiles_2d(0)/phi`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the toroidal flux on the grid in the poloidal plane, and store it in the time-slice.
///
/// Toroidal flux is a flux function, so inside the plasma the completed `profiles_1d/phi` is
/// linearly interpolated in `psi_norm` at every grid point. Outside the plasma there is no closed
/// flux surface to enclose a toroidal flux, so those points are left `NaN`.
///
/// Inside and outside are decided by `profiles_2d(0)/mask`, not by the value of
/// `profiles_2d(0)/psi_norm`: the solver masks `psi_norm` to zero outside the plasma, which makes an
/// outside point indistinguishable from the magnetic axis by value (and, as `0.0 * negative`, leaves
/// it `-0.0`).
///
/// An inside point's `psi_norm` is clamped onto the profile's domain before interpolating. It can
/// sit a rounding error outside `[0, 1]` for two reasons, neither of which should punch a `NaN` hole
/// in the plasma: `psi_magnetic_axis` comes from a sub-grid stationary-point fit, so the grid point
/// nearest the axis can land just beyond it; and the mask is flood-filled inside the boundary
/// *polygon*, so a grid point just inside the polygon can land just beyond `psi_norm = 1`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/phi` is written into it
///
/// A time-slice which failed to converge gets `NaN` everywhere.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_norm_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();
    let (n_z, n_r): (usize, usize) = psi_norm_2d.dim();

    let mut phi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);

    // A slice which did not converge has no plasma, so no toroidal flux is reported for it
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_2d[0].phi = Some(phi_2d);
        return;
    }

    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();
    let psi_norm_profile: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let phi_profile: &Array1<f64> = time_slice.profiles_1d.phi.as_ref().unwrap();
    assert_eq!(mask_2d.dim(), (n_z, n_r), "profiles_2d/phi: `mask` and `psi_norm` grids differ in shape");

    // The profile has to be a usable interpolation table: at least two points, finite, and strictly
    // increasing in `psi_norm`. `>` is false against `NaN`, so the last assertion also rejects one
    let n_psi_norm: usize = psi_norm_profile.len();
    assert!(
        n_psi_norm >= 2,
        "profiles_2d/phi: `profiles_1d/psi_norm` needs at least two points, has {n_psi_norm}"
    );
    assert_eq!(
        phi_profile.len(),
        n_psi_norm,
        "profiles_2d/phi: `profiles_1d/phi` and `profiles_1d/psi_norm` differ in length"
    );
    assert!(
        psi_norm_profile[0].is_finite() && psi_norm_profile[n_psi_norm - 1].is_finite(),
        "profiles_2d/phi: `profiles_1d/psi_norm` is not finite"
    );
    for i_psi_norm in 1..n_psi_norm {
        assert!(
            psi_norm_profile[i_psi_norm] > psi_norm_profile[i_psi_norm - 1],
            "profiles_2d/phi: `profiles_1d/psi_norm` is not strictly increasing at index {i_psi_norm}"
        );
    }
    let psi_norm_profile_min: f64 = psi_norm_profile[0];
    let psi_norm_profile_max: f64 = psi_norm_profile[n_psi_norm - 1];

    for i_r in 0..n_r {
        for i_z in 0..n_z {
            if mask_2d[(i_z, i_r)] > 0.99 {
                let psi_norm_local: f64 = psi_norm_2d[(i_z, i_r)].clamp(psi_norm_profile_min, psi_norm_profile_max);
                phi_2d[(i_z, i_r)] = interpolate_profile(psi_norm_local, psi_norm_profile, phi_profile);
            }
        }
    }

    time_slice.profiles_2d[0].phi = Some(phi_2d);
}

/// Linearly interpolate one value on a finite, strictly increasing grid `x`.
///
/// `x_new` outside `[x[0], x[n_x - 1]]`, or not finite, gives `NaN`. Every comparison is an IEEE
/// one, so `-0.0` and `0.0` are the same point. (`f64::total_cmp` orders `-0.0` before `0.0`, which
/// is how an earlier version of this search reached index `0 - 1`.)
fn interpolate_profile(x_new: f64, x: &Array1<f64>, values: &Array1<f64>) -> f64 {
    let n_x: usize = x.len();
    if !x_new.is_finite() || x_new < x[0] || x_new > x[n_x - 1] {
        return f64::NAN;
    }

    // The number of grid points at or below `x_new`: at least 1, because `x[0] <= x_new` here, so the
    // subtraction cannot underflow
    let x_slice: &[f64] = x.as_slice().unwrap();
    let n_x_at_or_below: usize = x_slice.partition_point(|x_here| *x_here <= x_new);
    let i_x_lower: usize = n_x_at_or_below - 1;

    // On a grid point, which includes the last one, above which there is no interval
    if x[i_x_lower] == x_new {
        return values[i_x_lower];
    }

    let i_x_upper: usize = i_x_lower + 1;
    let interval_fraction: f64 = (x_new - x[i_x_lower]) / (x[i_x_upper] - x[i_x_lower]);
    return values[i_x_lower] * (1.0 - interval_fraction) + values[i_x_upper] * interval_fraction;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::array;

    /// A converged time-slice with the profile `phi = 8 * psi_norm ** 3 / 1`-ish sampled at three points
    /// (`phi = 0, 2, 8` at `psi_norm = 0, 0.5, 1`), the given 2-D `psi_norm` and the given mask
    fn time_slice_with(psi_norm_2d: Array2<f64>, mask_2d: Array2<f64>) -> EquilibriumTimeSlice {
        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.psi_norm = Some(psi_norm_2d);
        profiles_2d.mask = Some(mask_2d);

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = Some(-0.1);
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.5, 1.0]);
        time_slice.profiles_1d.phi = Some(array![0.0, 2.0, 8.0]);
        time_slice.profiles_2d = vec![profiles_2d];
        return time_slice;
    }

    #[test]
    fn inside_points_interpolate_the_profile_and_the_mask_alone_decides_outside() {
        // The bottom-right point carries an inside-looking `psi_norm` but is outside by the mask
        let psi_norm_2d: Array2<f64> = array![[0.0, 0.25, 0.5], [0.75, 1.0, 0.3]];
        let mask_2d: Array2<f64> = array![[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let mut time_slice: EquilibriumTimeSlice = time_slice_with(psi_norm_2d, mask_2d);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let phi_2d: &Array2<f64> = time_slice.profiles_2d[0].phi.as_ref().unwrap();
        assert_abs_diff_eq!(phi_2d[(0, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(0, 1)], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(0, 2)], 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(1, 0)], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(1, 1)], 8.0, epsilon = 1e-15);
        assert!(phi_2d[(1, 2)].is_nan());
    }

    #[test]
    fn outside_points_are_nan_whatever_masked_psi_norm_they_carry() {
        // The solver's `mask * (psi - psi_a) / (psi_b - psi_a)` leaves `-0.0` and `0.0` outside
        let psi_norm_2d: Array2<f64> = array![[-0.0, 0.0], [0.0, -0.0]];
        let mask_2d: Array2<f64> = Array2::zeros((2, 2));
        let mut time_slice: EquilibriumTimeSlice = time_slice_with(psi_norm_2d, mask_2d);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let phi_2d: &Array2<f64> = time_slice.profiles_2d[0].phi.as_ref().unwrap();
        assert!(phi_2d.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn inside_points_a_rounding_error_beyond_the_profile_domain_are_clamped_not_nan() {
        // Just below the axis, just beyond the boundary, a negative zero, and a normal point
        let psi_norm_2d: Array2<f64> = array![[-1.0e-12, 1.0 + 1.0e-12], [-0.0, 0.5]];
        let mask_2d: Array2<f64> = Array2::ones((2, 2));
        let mut time_slice: EquilibriumTimeSlice = time_slice_with(psi_norm_2d, mask_2d);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let phi_2d: &Array2<f64> = time_slice.profiles_2d[0].phi.as_ref().unwrap();
        assert_abs_diff_eq!(phi_2d[(0, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(0, 1)], 8.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(1, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(phi_2d[(1, 1)], 2.0, epsilon = 1e-15);
    }

    #[test]
    fn failed_slice_is_nan_everywhere() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_with(Array2::from_elem((2, 3), f64::NAN), Array2::zeros((2, 3)));
        time_slice.global_quantities.psi_magnetic_axis = Some(f64::NAN);
        time_slice.profiles_1d.phi = Some(array![f64::NAN, f64::NAN, f64::NAN]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let phi_2d: &Array2<f64> = time_slice.profiles_2d[0].phi.as_ref().unwrap();
        assert!(phi_2d.iter().all(|value| value.is_nan()));
    }

    #[test]
    #[should_panic(expected = "not strictly increasing")]
    fn a_non_increasing_profile_panics() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_with(Array2::zeros((1, 1)), Array2::ones((1, 1)));
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.5, 0.5]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());
    }

    #[test]
    fn interpolate_profile_hits_grid_points_exactly_and_interpolates_between_them() {
        let x: Array1<f64> = array![0.0, 0.5, 1.0];
        let values: Array1<f64> = array![0.0, 2.0, 8.0];

        // Exact hits at both ends and in the middle, including a negative zero at the bottom
        assert_eq!(interpolate_profile(0.0, &x, &values), 0.0);
        assert_eq!(interpolate_profile(-0.0, &x, &values), 0.0);
        assert_eq!(interpolate_profile(0.5, &x, &values), 2.0);
        assert_eq!(interpolate_profile(1.0, &x, &values), 8.0);
        // Linear between grid points
        assert_abs_diff_eq!(interpolate_profile(0.25, &x, &values), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(interpolate_profile(0.75, &x, &values), 5.0, epsilon = 1e-15);
        // Outside the grid, or not finite
        assert!(interpolate_profile(-1.0e-12, &x, &values).is_nan());
        assert!(interpolate_profile(1.0 + 1.0e-12, &x, &values).is_nan());
        assert!(interpolate_profile(f64::NAN, &x, &values).is_nan());
        assert!(interpolate_profile(f64::INFINITY, &x, &values).is_nan());
    }
}
