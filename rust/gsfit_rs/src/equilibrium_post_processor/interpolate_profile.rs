//! Linear interpolation of a `profiles_1d` profile, for the calculators which map one onto a
//! different abscissa.

use ndarray::Array1;

/// Linearly interpolate one value on a finite, strictly increasing grid `x`.
///
/// `x_new` outside `[x[0], x[n_x - 1]]`, or not finite, gives `NaN`. Every comparison is an IEEE
/// one, so `-0.0` and `0.0` are the same point. (`f64::total_cmp` orders `-0.0` before `0.0`, which
/// is how an earlier version of this search reached index `0 - 1`.)
///
/// # Arguments
/// * `x_new` - the abscissa to interpolate at
/// * `x` - the profile's abscissa, finite and strictly increasing
/// * `values` - the profile itself, the same length as `x`
pub(super) fn interpolate_profile(x_new: f64, x: &Array1<f64>, values: &Array1<f64>) -> f64 {
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
    values[i_x_lower] * (1.0 - interval_fraction) + values[i_x_upper] * interval_fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
