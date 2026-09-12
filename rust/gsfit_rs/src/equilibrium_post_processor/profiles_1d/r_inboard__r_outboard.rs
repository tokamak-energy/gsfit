//! `time_slice(itime)/profiles_1d/r_inboard` and `.../r_outboard`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::{BicubicInterpolator, BicubicValueAndDerivatives};
use crate::plasma_geometry::cubic_interpolation::cubic_interpolation;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, s};

/// Which side of the magnetic axis a flux surface is wanted on.
///
/// `psi` rises away from the magnetic axis in *both* directions along the row, so it has a minimum
/// there and is not invertible across the whole row. The side says which of the two branches to
/// walk
#[derive(Clone, Copy)]
enum MidplaneSide {
    Inboard,
    Outboard,
}

/// Calculate the major radius of each flux surface at the height of the magnetic axis, on each side
/// of it, and store them in the time-slice.
///
/// Each radius is found by inverting the solver's own interpolation of `psi` along that row for the
/// `R` at which it reaches the surface's `psi`, rather than by intersecting the traced flux surface
/// with the row.
///
/// Inverting the interpolant is the more accurate of the two by a wide margin, and is why it is
/// done that way round. A traced surface is a polyline whose vertices come from *linear*
/// interpolation along the edges of the grid cells, so intersecting it inherits that linear error
/// over a whole cell - millimetres on ST40's 12.5 mm grid. The interpolant is built from the
/// solver's analytic derivatives of `psi`, and is inverted by solving the cubic exactly. It also
/// does not depend on `marching_squares`, whose contour ordering varies between calls.
///
/// The row is cut the same way `profiles_1d_r_midplane_h` cuts it - the same bicubic at the same
/// height - so inverting that profile for a major radius lands on the radii here.
///
/// The magnetic axis (`psi_norm = 0`) is a point rather than a surface, so both radii are `mag_r`
/// there.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `r_inboard` and `r_outboard` are written into it
///
/// A surface which does not reach the height of the magnetic axis within the grid is left as NaN,
/// as is every surface on a time-slice which did not converge.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    let mut r_inboard_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut r_outboard_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

    // A time-slice which did not converge has a NaN magnetic axis, and so no row to cut along.
    // This has to be tested rather than left to propagate, because the axis is used to pick cell
    // indices below and a NaN cast to `usize` would silently saturate to the first cell
    if mag_r.is_finite() && mag_z.is_finite() {
        // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is
        // only ever one entry in this array of structures
        let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
        let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
        let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
        let d_psi_d_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_r;
        let d_psi_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_z;
        let d2_psi_d_r_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d2_psi_d_r_d_z;

        let n_r: usize = r.len();
        let n_z: usize = z.len();
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];

        // The row of cells containing the magnetic axis, clamped so that an axis on the last grid
        // line uses the last cell
        let i_z_lower: usize = (((mag_z - z[0]) / d_z).floor() as usize).min(n_z - 2);
        let y: f64 = (mag_z - z[i_z_lower]) / d_z;

        // `psi` and its radial derivative where the row crosses each of the grid's radial lines.
        // Restricting the bicubic to a fixed height leaves a cubic in `R`, and a cubic is fixed by
        // its value and gradient at the two ends of a cell, so these two arrays *are* the row: what
        // follows needs no further reference to the two-dimensional fields
        let mut psi_at_grid_lines: Array1<f64> = Array1::from_elem(n_r, f64::NAN);
        let mut d_psi_d_r_at_grid_lines: Array1<f64> = Array1::from_elem(n_r, f64::NAN);
        for i_r_left in 0..n_r - 1 {
            let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(
                d_r,
                d_z,
                psi_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            );

            // Each cell fills its own left-hand grid line; the last cell also fills the right-hand
            // one, as there is no next cell to fill it. `value_and_derivatives` differentiates with
            // respect to the normalised cell coordinate, hence the division by `d_r`
            let at_left: BicubicValueAndDerivatives = psi_interpolator.value_and_derivatives(0.0, y);
            psi_at_grid_lines[i_r_left] = at_left.f;
            d_psi_d_r_at_grid_lines[i_r_left] = at_left.d_f_d_x / d_r;

            if i_r_left == n_r - 2 {
                let at_right: BicubicValueAndDerivatives = psi_interpolator.value_and_derivatives(1.0, y);
                psi_at_grid_lines[i_r_left + 1] = at_right.f;
                d_psi_d_r_at_grid_lines[i_r_left + 1] = at_right.d_f_d_x / d_r;
            }
        }

        for i_psi_norm in 0..n_psi_norm {
            // The magnetic axis is a point rather than a surface, so both radii collapse onto it
            if psi_norm[i_psi_norm] <= 0.0 {
                r_inboard_profile[i_psi_norm] = mag_r;
                r_outboard_profile[i_psi_norm] = mag_r;
                continue;
            }

            let psi_target: f64 = psi_a + psi_norm[i_psi_norm] * (psi_b - psi_a);
            r_inboard_profile[i_psi_norm] = r_at_psi(r, &psi_at_grid_lines, &d_psi_d_r_at_grid_lines, mag_r, psi_a, psi_target, MidplaneSide::Inboard);
            r_outboard_profile[i_psi_norm] = r_at_psi(r, &psi_at_grid_lines, &d_psi_d_r_at_grid_lines, mag_r, psi_a, psi_target, MidplaneSide::Outboard);
        }
    }

    time_slice.profiles_1d.r_inboard = r_inboard_profile;
    time_slice.profiles_1d.r_outboard = r_outboard_profile;
}

/// The major radius at which a flux surface crosses the row, on one side of the magnetic axis.
///
/// The row is walked outwards from the magnetic axis and the **first** crossing is taken, so the
/// surface found is the one nearest the axis. That is what makes the answer the nested flux surface
/// rather than some re-entrant part of the same flux value further out.
///
/// # Arguments
/// * `r` - the grid's radial lines, [metre]
/// * `psi_at_grid_lines` - `psi` where the row crosses each of them, [weber]
/// * `d_psi_d_r_at_grid_lines` - and its radial derivative there, [weber / metre]
/// * `mag_r` - major radius of the magnetic axis, [metre]
/// * `psi_a` - `psi` on the magnetic axis, which is where the walk starts, [weber]
/// * `psi_target` - the flux to find, [weber]
/// * `side` - which side of the magnetic axis to walk
///
/// # Returns
/// * `r_crossing` - [metre]
///
/// NaN when the surface does not cross the row anywhere on this side of the grid.
fn r_at_psi(
    r: &Array1<f64>,
    psi_at_grid_lines: &Array1<f64>,
    d_psi_d_r_at_grid_lines: &Array1<f64>,
    mag_r: f64,
    psi_a: f64,
    psi_target: f64,
    side: MidplaneSide,
) -> f64 {
    let n_r: usize = r.len();
    let d_r: f64 = r[1] - r[0];

    // The cell holding the magnetic axis, clamped so that an axis on the last grid line uses the
    // last cell. The walk starts here, in the cell the axis is inside, rather than at a grid line
    let i_r_axis: usize = (((mag_r - r[0]) / d_r).floor() as usize).min(n_r - 2);


    match side {
        MidplaneSide::Outboard => {
            'outboard_cell_loop: for i_r_left in i_r_axis..n_r - 1 {
                // The axis's own cell is bracketed from the axis outwards, and every cell beyond it
                // from its own left-hand grid line
                // The axis itself is the near end of its own cell, and `psi` there is `psi_a`
                // exactly, because that is what a magnetic axis is
                let psi_near: f64 = if i_r_left == i_r_axis { psi_a } else { psi_at_grid_lines[i_r_left] };
                if !brackets(psi_near, psi_at_grid_lines[i_r_left + 1], psi_target) {
                    continue 'outboard_cell_loop;
                }

                return crossing_in_cell(r, psi_at_grid_lines, d_psi_d_r_at_grid_lines, i_r_left, psi_target, mag_r, side);
            }
        }
        MidplaneSide::Inboard => {
            'inboard_cell_loop: for i_r_left in (0..=i_r_axis).rev() {
                let psi_near: f64 = if i_r_left == i_r_axis { psi_a } else { psi_at_grid_lines[i_r_left + 1] };
                if !brackets(psi_near, psi_at_grid_lines[i_r_left], psi_target) {
                    continue 'inboard_cell_loop;
                }

                return crossing_in_cell(r, psi_at_grid_lines, d_psi_d_r_at_grid_lines, i_r_left, psi_target, mag_r, side);
            }
        }
    }

    // The surface does not reach the height of the magnetic axis on this side of the grid
    return f64::NAN;
}

/// Solve one radial cell's cubic for the crossing on one side of the magnetic axis.
///
/// # Arguments
/// * `r` - the grid's radial lines, [metre]
/// * `psi_at_grid_lines` - `psi` where the row crosses each of them, [weber]
/// * `d_psi_d_r_at_grid_lines` - and its radial derivative there, [weber / metre]
/// * `i_r_left` - the cell, named by the grid line on its left
/// * `psi_target` - the flux to find, [weber]
/// * `mag_r` - major radius of the magnetic axis, [metre]
/// * `side` - which side of the magnetic axis the crossing has to be on
///
/// # Returns
/// * `r_crossing` - [metre]
///
/// A cubic can cross the same value up to three times inside one cell, and the cell holding the
/// magnetic axis crosses it on both sides of the axis, so the root nearest the axis on the wanted
/// side is the one taken. NaN if the cell turns out to hold no root on that side.
fn crossing_in_cell(
    r: &Array1<f64>,
    psi_at_grid_lines: &Array1<f64>,
    d_psi_d_r_at_grid_lines: &Array1<f64>,
    i_r_left: usize,
    psi_target: f64,
    mag_r: f64,
    side: MidplaneSide,
) -> f64 {
    let roots_or_error: Result<Array1<f64>, String> = cubic_interpolation(
        r[i_r_left],
        psi_at_grid_lines[i_r_left],
        d_psi_d_r_at_grid_lines[i_r_left],
        r[i_r_left + 1],
        psi_at_grid_lines[i_r_left + 1],
        d_psi_d_r_at_grid_lines[i_r_left + 1],
        psi_target,
    );

    let mut r_crossing: f64 = f64::NAN;
    if let Ok(roots) = roots_or_error {
        let n_root: usize = roots.len();
        'root_loop: for i_root in 0..n_root {
            let wanted_side: bool = match side {
                MidplaneSide::Outboard => roots[i_root] > mag_r,
                MidplaneSide::Inboard => roots[i_root] < mag_r,
            };
            if !wanted_side {
                continue 'root_loop;
            }

            // Nearest the axis, which on the outboard side is the smallest root and on the inboard
            // side the largest
            let nearer_the_axis: bool = if r_crossing.is_nan() {
                true
            } else {
                match side {
                    MidplaneSide::Outboard => roots[i_root] < r_crossing,
                    MidplaneSide::Inboard => roots[i_root] > r_crossing,
                }
            };
            if nearer_the_axis {
                r_crossing = roots[i_root];
            }
        }
    }

    return r_crossing;
}

/// Does `psi_target` lie between the two ends of an interval being walked?
///
/// Half-open: the value at the far end counts as inside and the value at the near end does not, so
/// that a target sitting exactly on a grid line is found once - by the interval which reaches it -
/// rather than twice or not at all.
///
/// Written as a change of sign rather than an ordered comparison because `psi` rises away from the
/// magnetic axis for a positive plasma current and falls for a negative one, and because it may
/// turn over again further out in the scrape-off layer.
///
/// # Arguments
/// * `psi_near` - `psi` at the end nearer the magnetic axis, [weber]
/// * `psi_far` - `psi` at the end further from it, [weber]
/// * `psi_target` - the flux being looked for, [weber]
fn brackets(psi_near: f64, psi_far: f64, psi_target: f64) -> bool {
    // A NaN end is no bracket at all. Without this it would compare below every target and so read
    // as a crossing, and the cell solve would then be handed values it cannot answer from
    if !psi_near.is_finite() || !psi_far.is_finite() {
        return false;
    }

    return (psi_near < psi_target) != (psi_far < psi_target);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Build the time-slice this calculator is tested against.
    ///
    /// A quadratic well on a 12-by-11 grid, with its derivatives filled in exactly:
    ///
    /// ```text
    /// psi = 1 + 8 * (r - 0.65) ** 2 + 7 * (r - 0.65) * (z - 0.05) + 5 * (z - 0.05) ** 2
    /// ```
    ///
    /// A bicubic reproduces a function which is quadratic in each variable exactly, so every
    /// interpolated value has a closed form to check against. The cross term makes a calculator
    /// which mixed up the `r` and `z` axes, or dropped the cross derivative, come out wrong, but
    /// vanishes on the row `z = mag_z`, which leaves `psi_norm = 4 * (r - 0.65) ** 2` along the row
    /// itself. A surface is therefore at `r = 0.65 +/- sqrt(psi_norm) / 2`.
    ///
    /// The magnetic axis is at `(0.65, 0.05)`, deliberately half way between grid lines on both
    /// axes, so that a calculator which sampled the nearest row or column instead of interpolating
    /// fails. `psi_magnetic_axis = 1` and `boundary/psi = 3`, so `psi_norm = (psi - 1) / 2`.
    fn time_slice_for_test() -> EquilibriumTimeSlice {
        use imas_rs::EquilibriumProfiles2d;

        let r: Array1<f64> = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let z: Array1<f64> = array![-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - 0.65;
                let delta_z: f64 = z[i_z] - 0.05;
                psi_2d[(i_z, i_r)] = 1.0 + 8.0 * delta_r.powi(2) + 7.0 * delta_r * delta_z + 5.0 * delta_z.powi(2);
                d_psi_d_r_2d[(i_z, i_r)] = 16.0 * delta_r + 7.0 * delta_z;
                d_psi_d_z_2d[(i_z, i_r)] = 7.0 * delta_r + 10.0 * delta_z;
                d2_psi_d_r_d_z_2d[(i_z, i_r)] = 7.0;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;
        profiles_2d.d_psi_d_z = d_psi_d_z_2d;
        profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = 0.65;
        time_slice.global_quantities.magnetic_axis.z = 0.05;
        time_slice.global_quantities.psi_magnetic_axis = 1.0;
        time_slice.boundary.psi = 3.0;
        time_slice.profiles_2d = vec![profiles_2d];

        return time_slice;
    }

    #[test]
    fn each_surface_is_found_either_side_of_the_magnetic_axis() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.0016, 0.25, 1.0, 1.2];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The fixture puts a surface at `r = 0.65 +/- sqrt(psi_norm) / 2`, and the magnetic axis is
        // a point so both radii collapse onto it there. Inverting the interpolant is exact, so
        // these hold to machine precision rather than to the width of a grid cell. 0.0016 lands
        // inside the axis's own cell, which is the one bracketed from the axis rather than from a
        // grid line
        let psi_norm: Array1<f64> = time_slice.profiles_1d.psi_norm.to_owned();
        let n_psi_norm: usize = psi_norm.len();
        for i_psi_norm in 0..n_psi_norm {
            let delta_r: f64 = psi_norm[i_psi_norm].sqrt() / 2.0;
            assert_abs_diff_eq!(time_slice.profiles_1d.r_inboard[i_psi_norm], 0.65 - delta_r, epsilon = 1e-12);
            assert_abs_diff_eq!(time_slice.profiles_1d.r_outboard[i_psi_norm], 0.65 + delta_r, epsilon = 1e-12);
        }
    }

    #[test]
    fn a_surface_which_leaves_the_grid_is_nan_without_disturbing_the_others() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        // `psi_norm` only reaches 1.21 by the edge of the fixture's grid, so the second surface is
        // off it on both sides
        time_slice.profiles_1d.psi_norm = array![0.25, 2.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.profiles_1d.r_inboard[0], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(time_slice.profiles_1d.r_outboard[0], 0.9, epsilon = 1e-12);
        assert!(time_slice.profiles_1d.r_inboard[1].is_nan());
        assert!(time_slice.profiles_1d.r_outboard[1].is_nan());
    }

    #[test]
    fn a_surface_landing_exactly_on_a_grid_line_is_found_there() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        // `psi_norm` at the grid line `r = 1.0` is `4 * 0.35 ** 2`. A target sitting exactly on a
        // grid line is the case the half-open bracket exists for: it must be found once, by the
        // cell which reaches it, and not fall between the two cells which share that line
        time_slice.profiles_1d.psi_norm = array![4.0 * 0.35_f64.powi(2)];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.profiles_1d.r_inboard[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(time_slice.profiles_1d.r_outboard[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn the_row_through_the_magnetic_axis_is_cut_and_not_a_grid_row() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.psi_norm = array![1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The axis is at `z = 0.05`, half way between the grid rows at 0.0 and 0.1. On `z = 0` the
        // boundary would be at 1.1708 on the outboard side rather than 1.15, so a calculator which
        // cut the nearest grid row instead of interpolating fails here
        assert_abs_diff_eq!(time_slice.profiles_1d.r_outboard[0], 1.15, epsilon = 1e-12);
    }

    #[test]
    fn a_slice_which_did_not_converge_gives_nan_radii() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.global_quantities.magnetic_axis.r = f64::NAN;
        time_slice.global_quantities.magnetic_axis.z = f64::NAN;
        time_slice.profiles_1d.psi_norm = array![0.0, 0.25, 1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // Including at `psi_norm = 0`, which is the magnetic axis and so has no radius to report
        // when there is no magnetic axis
        assert!(time_slice.profiles_1d.r_inboard.iter().all(|r_here: &f64| r_here.is_nan()));
        assert!(time_slice.profiles_1d.r_outboard.iter().all(|r_here: &f64| r_here.is_nan()));
    }
}
