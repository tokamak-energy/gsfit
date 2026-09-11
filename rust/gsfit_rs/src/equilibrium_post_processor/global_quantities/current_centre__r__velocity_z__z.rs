//! `time_slice(itime)/global_quantities/current_centre/r`, `.../velocity_z` and `.../z`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::Equilibrium;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the current centre's position, and store it in the time-slice.
///
/// The current centre is the toroidal-current-weighted mean position of the plasma, which the data
/// dictionary defines as
///
/// ```text
/// current_centre/r = sum_grid( j_phi(R, Z) * R * d_area ) / ip
/// current_centre/z = sum_grid( j_phi(R, Z) * Z * d_area ) / ip
/// ```
///
/// `ip` is evaluated here from the same `j_phi`, as `sum_grid( j_phi * d_area )`, which is also how
/// the solver defines it. Taking both from one field makes the centre exactly the weighted mean of
/// the grid points, independent of whatever rounding `global_quantities/ip` carries.
///
/// `j_phi` is already masked to zero outside the plasma boundary, so summing over the whole grid is
/// an integral over the plasma alone.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/current_centre/r` and `.../z` are
///   written into it. `.../velocity_z` differentiates across time-slices, so it is filled by
///   [`calculate_velocity_z`] after the per-slice loop instead
///
/// A time-slice which failed to converge carries `NaN` in `j_phi`, so the centre comes out `NaN`
/// without needing a special case; so does a time-slice carrying no plasma current.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let j_phi_2d: &Array2<f64> = &time_slice.profiles_2d[0].j_phi;
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area;

    let (n_z, n_r): (usize, usize) = j_phi_2d.dim();

    // Total current and its first moments about the origin
    let mut ip: f64 = 0.0;
    let mut current_moment_r: f64 = 0.0;
    let mut current_moment_z: f64 = 0.0;
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            let current_cell: f64 = j_phi_2d[(i_z, i_r)] * d_area;
            ip += current_cell;
            current_moment_r += current_cell * r[i_r];
            current_moment_z += current_cell * z[i_z];
        }
    }

    time_slice.global_quantities.current_centre.r = current_moment_r / ip;
    time_slice.global_quantities.current_centre.z = current_moment_z / ip;
}

/// Calculate the current centre's vertical velocity, and store it in every time-slice.
///
/// ```text
/// current_centre/velocity_z = d(current_centre/z)/d(time)
/// ```
///
/// Like `v_loop`, this differentiates across time-slices, so it needs all of them at once and runs
/// after the per-slice loop rather than inside it. It has to run after [`calculate`] has filled
/// `current_centre/z` in every time-slice.
///
/// # Arguments
/// * `equilibrium_ids` - the solved equilibrium IDS; `global_quantities/current_centre/velocity_z`
///   is written into every time-slice
///
/// A single time-slice has no derivative and is given `NaN`. A `NaN` in `current_centre/z` poisons
/// the velocity at the time-slices either side of it, which difference across it, but not at the
/// time-slice itself: the central difference there uses only its neighbours. An unconverged
/// time-slice is skipped by the per-slice loop, so its `current_centre/z` is `None`; that is
/// treated as `NaN` here, so it behaves the same way.
pub fn calculate_velocity_z(equilibrium_ids: &mut Equilibrium) {
    let n_time: usize = equilibrium_ids.time_slice.len();

    let mut time: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
    let mut current_centre_z: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
    for i_time in 0..n_time {
        time[i_time] = equilibrium_ids.time_slice[i_time].time;
        // An unconverged time-slice is skipped by the per-slice loop, so `calculate` never filled
        // it and it is still `NaN`, which the derivative below already handles
        current_centre_z[i_time] = equilibrium_ids.time_slice[i_time].global_quantities.current_centre.z;
    }

    let velocity_z: Array1<f64> = d_dt(&current_centre_z, &time);

    for i_time in 0..n_time {
        equilibrium_ids.time_slice[i_time].global_quantities.current_centre.velocity_z = velocity_z[i_time];
    }
}

/// Differentiate `values` with respect to `time`, on the same stencil as `v_loop`: one-sided
/// differences at the two ends and central differences between them, so that a variable time step
/// is handled. Fewer than two points have no derivative and give `NaN`.
fn d_dt(values: &Array1<f64>, time: &Array1<f64>) -> Array1<f64> {
    let n_time: usize = values.len();
    let mut derivative: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

    if n_time < 2 {
        return derivative;
    }

    derivative[0] = (values[1] - values[0]) / (time[1] - time[0]);
    for i_time in 1..n_time - 1 {
        derivative[i_time] = (values[i_time + 1] - values[i_time - 1]) / (time[i_time + 1] - time[i_time - 1]);
    }
    derivative[n_time - 1] = (values[n_time - 1] - values[n_time - 2]) / (time[n_time - 1] - time[n_time - 2]);

    return derivative;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::array;

    /// A time-slice carrying `j_phi_2d` on the uniform grid whose points are `r` and `z`
    fn time_slice_with_current(r: Array1<f64>, z: Array1<f64>, j_phi_2d: Array2<f64>) -> EquilibriumTimeSlice {
        let d_area: f64 = (r[1] - r[0]) * (z[1] - z[0]);
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = r;
        time_slice.profiles_2d[0].grid.dim2 = z;
        time_slice.profiles_2d[0].grid.d_area = d_area;
        time_slice.profiles_2d[0].j_phi = j_phi_2d;
        return time_slice;
    }

    /// An equilibrium whose current centre sits at `current_centre_z[i_time]` at `time[i_time]`
    fn equilibrium_with_centre_z(time: Array1<f64>, current_centre_z: Array1<f64>) -> Equilibrium {
        let n_time: usize = time.len();
        let mut equilibrium_ids: Equilibrium = Equilibrium::default();
        equilibrium_ids.time_slice = vec![EquilibriumTimeSlice::default(); n_time];
        for i_time in 0..n_time {
            equilibrium_ids.time_slice[i_time].time = time[i_time];
            equilibrium_ids.time_slice[i_time].global_quantities.current_centre.z = current_centre_z[i_time];
        }
        return equilibrium_ids;
    }

    #[test]
    fn a_uniform_current_has_its_centre_at_the_grid_centroid() {
        let r: Array1<f64> = array![1.0, 1.5, 2.0, 2.5];
        let z: Array1<f64> = array![-0.3, 0.0, 0.3];
        let j_phi_2d: Array2<f64> = Array2::from_elem((3, 4), 2.0e6);
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_current(r, z, j_phi_2d);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.global_quantities.current_centre.r, 1.75, epsilon = 1e-14);
        assert_abs_diff_eq!(time_slice.global_quantities.current_centre.z, 0.0, epsilon = 1e-14);
        assert!(time_slice.global_quantities.current_centre.velocity_z.is_nan());
    }

    #[test]
    fn a_single_current_carrying_cell_puts_the_centre_on_that_cell() {
        let r: Array1<f64> = array![1.0, 1.5, 2.0, 2.5];
        let z: Array1<f64> = array![-0.3, 0.0, 0.3];
        let mut j_phi_2d: Array2<f64> = Array2::zeros((3, 4));
        j_phi_2d[(2, 1)] = 5.0e5;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_current(r, z, j_phi_2d);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.global_quantities.current_centre.r, 1.5, epsilon = 1e-14);
        assert_abs_diff_eq!(time_slice.global_quantities.current_centre.z, 0.3, epsilon = 1e-14);
    }

    #[test]
    fn the_centre_is_the_current_weighted_mean_and_independent_of_the_current_scale() {
        let r: Array1<f64> = array![1.0, 2.0];
        let z: Array1<f64> = array![0.0, 1.0];
        // Cells at (R, Z) = (1, 0) and (2, 1), carrying current in the ratio 1 : 3
        let mut j_phi_2d: Array2<f64> = Array2::zeros((2, 2));
        j_phi_2d[(0, 0)] = 1.0;
        j_phi_2d[(1, 1)] = 3.0;

        for current_scale in [1.0, 1.0e6] {
            let mut time_slice: EquilibriumTimeSlice = time_slice_with_current(r.clone(), z.clone(), &j_phi_2d * current_scale);

            calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

            assert_abs_diff_eq!(time_slice.global_quantities.current_centre.r, (1.0 * 1.0 + 3.0 * 2.0) / 4.0, epsilon = 1e-14);
            assert_abs_diff_eq!(time_slice.global_quantities.current_centre.z, (1.0 * 0.0 + 3.0 * 1.0) / 4.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn a_failed_time_slice_and_a_currentless_one_both_give_nan() {
        let r: Array1<f64> = array![1.0, 2.0];
        let z: Array1<f64> = array![0.0, 1.0];

        for j_phi_2d in [Array2::from_elem((2, 2), f64::NAN), Array2::zeros((2, 2))] {
            let mut time_slice: EquilibriumTimeSlice = time_slice_with_current(r.clone(), z.clone(), j_phi_2d);

            calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

            assert!(time_slice.global_quantities.current_centre.r.is_nan());
            assert!(time_slice.global_quantities.current_centre.z.is_nan());
        }
    }

    #[test]
    fn velocity_z_of_a_linearly_moving_centre_is_exact_on_a_variable_time_step() {
        let time: Array1<f64> = array![0.0, 0.01, 0.015, 0.03, 0.032];
        let velocity_z_expected: f64 = -2.0;
        let current_centre_z: Array1<f64> = 0.1 + velocity_z_expected * &time;
        let mut equilibrium_ids: Equilibrium = equilibrium_with_centre_z(time, current_centre_z);

        calculate_velocity_z(&mut equilibrium_ids);

        for time_slice in &equilibrium_ids.time_slice {
            assert_abs_diff_eq!(time_slice.global_quantities.current_centre.velocity_z, velocity_z_expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn velocity_z_of_a_single_time_slice_is_nan() {
        let mut equilibrium_ids: Equilibrium = equilibrium_with_centre_z(array![0.05], array![0.1]);

        calculate_velocity_z(&mut equilibrium_ids);

        assert!(equilibrium_ids.time_slice[0].global_quantities.current_centre.velocity_z.is_nan());
    }

    #[test]
    fn velocity_z_at_a_nan_centre_is_lost_at_its_neighbours_but_not_at_itself() {
        let time: Array1<f64> = array![0.0, 0.01, 0.02, 0.03, 0.04];
        let current_centre_z: Array1<f64> = array![0.0, 0.1, f64::NAN, 0.3, 0.4];
        let mut equilibrium_ids: Equilibrium = equilibrium_with_centre_z(time, current_centre_z);

        calculate_velocity_z(&mut equilibrium_ids);

        let velocity_z: Vec<f64> = equilibrium_ids
            .time_slice
            .iter()
            .map(|time_slice| time_slice.global_quantities.current_centre.velocity_z)
            .collect();
        assert_abs_diff_eq!(velocity_z[0], 10.0, epsilon = 1e-12);
        assert!(velocity_z[1].is_nan());
        // The central difference at the NaN slice differences its neighbours, z[3] - z[1], so it is finite
        assert_abs_diff_eq!(velocity_z[2], 10.0, epsilon = 1e-12);
        assert!(velocity_z[3].is_nan());
        assert_abs_diff_eq!(velocity_z[4], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_z_treats_an_unconverged_time_slice_as_a_nan_centre() {
        let time: Array1<f64> = array![0.0, 0.01, 0.02, 0.03, 0.04];
        let current_centre_z: Array1<f64> = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let mut equilibrium_ids: Equilibrium = equilibrium_with_centre_z(time, current_centre_z);
        // The per-slice loop skips an unconverged time-slice, so `calculate` never fills its centre
        equilibrium_ids.time_slice[2].global_quantities.current_centre.z = f64::NAN;

        calculate_velocity_z(&mut equilibrium_ids);

        let velocity_z: Vec<f64> = equilibrium_ids
            .time_slice
            .iter()
            .map(|time_slice| time_slice.global_quantities.current_centre.velocity_z)
            .collect();
        assert_abs_diff_eq!(velocity_z[0], 10.0, epsilon = 1e-12);
        assert!(velocity_z[1].is_nan());
        assert_abs_diff_eq!(velocity_z[2], 10.0, epsilon = 1e-12);
        assert!(velocity_z[3].is_nan());
        assert_abs_diff_eq!(velocity_z[4], 10.0, epsilon = 1e-12);
    }
}
