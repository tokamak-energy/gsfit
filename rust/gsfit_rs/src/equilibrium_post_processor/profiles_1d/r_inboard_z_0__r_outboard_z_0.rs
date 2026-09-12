//! `time_slice(itime)/profiles_1d/r_inboard_z_0` and `.../r_outboard_z_0`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::cubic_interpolation::cubic_interpolation;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// How far off `z = 0` a grid row may be and still count as being on the machine's mid-plane.
///
/// One nanometre, which no grid lands within by accident and every symmetric grid lands on exactly
const Z_0_TOLERANCE: f64 = 1e-9;

/// Which side of the magnetic axis a flux surface is wanted on.
///
/// `psi` rises away from the magnetic axis in *both* directions along the row, so it has a minimum
/// near there and is not invertible across the whole row. The side says which of the two branches
/// to walk
#[derive(Clone, Copy)]
enum MidplaneSide {
    Inboard,
    Outboard,
}

/// Calculate the major radius of each flux surface where it crosses `z = 0`, on each side of the
/// magnetic axis, and store them in the time-slice.
///
/// The same measurement as `profiles_1d/r_inboard` and `profiles_1d/r_outboard`, and found the same
/// way - by inverting the solver's own interpolation of `psi` along the row - but taken on the
/// machine's mid-plane rather than at the height of the magnetic axis. That height moves between
/// time-slices and this one does not, so a series of these is a series of radii measured at one
/// fixed height.
///
/// Because `z = 0` is a row *of the grid*, there is no interpolation in `z` to do and the row is
/// the plain cubic Hermite in `R`, straight off the solver's `psi` and `d(psi)/d(r)`. That is the
/// same function the bicubic would give at that height, for a fraction of the work. It needs the
/// grid to have a row on `z = 0`, which a symmetric `z` range with an odd `n_z` always does; a grid
/// without one leaves these profiles NaN rather than quietly answering from a nearby row.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `r_inboard_z_0` and `r_outboard_z_0` are written into it
///
/// The two sides are split at the major radius of the magnetic axis, which is the point every flux
/// surface crossing this row encloses. The axis is above or below `z = 0` rather than on it, so
/// unlike `r_inboard` these do not reach `psi_norm = 0`: the innermost surfaces do not come down to
/// this height at all and are left as NaN, as is every surface on a time-slice which did not
/// converge.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    let mut r_inboard_z_0_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut r_outboard_z_0_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
    let d_psi_d_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_r;

    let n_r: usize = r.len();
    let n_z: usize = z.len();

    // The grid row on the machine's mid-plane, if the grid has one
    let mut i_z_0: usize = n_z;
    'z_loop: for i_z in 0..n_z {
        if z[i_z].abs() <= Z_0_TOLERANCE {
            i_z_0 = i_z;
            break 'z_loop;
        }
    }

    // A time-slice which did not converge has a NaN magnetic axis, and every surface here is
    // measured against it. This has to be tested rather than left to propagate, because the axis is
    // used to pick a cell index below and a NaN cast to `usize` would silently saturate to the
    // first cell
    if mag_r.is_finite() && i_z_0 < n_z {
        // The row is the grid's own values along `z = 0`; a cubic is fixed by its value and
        // gradient at the two ends of a cell, so these two are the whole row
        let mut psi_at_grid_lines: Array1<f64> = Array1::from_elem(n_r, f64::NAN);
        let mut d_psi_d_r_at_grid_lines: Array1<f64> = Array1::from_elem(n_r, f64::NAN);
        for i_r in 0..n_r {
            psi_at_grid_lines[i_r] = psi_2d[(i_z_0, i_r)];
            d_psi_d_r_at_grid_lines[i_r] = d_psi_d_r_2d[(i_z_0, i_r)];
        }

        // `psi` on this row directly below or above the magnetic axis, which is where the walk
        // outwards starts. It is *not* `psi_a`, because the axis is not on this row
        let d_r: f64 = r[1] - r[0];
        let i_r_axis: usize = (((mag_r - r[0]) / d_r).floor() as usize).min(n_r - 2);
        let psi_at_mag_r: f64 = psi_hermite_at(r, &psi_at_grid_lines, &d_psi_d_r_at_grid_lines, i_r_axis, mag_r);
        let psi_norm_at_mag_r: f64 = (psi_at_mag_r - psi_a) / (psi_b - psi_a);

        for i_psi_norm in 0..n_psi_norm {
            // A surface inside the flux this row reaches at the magnetic axis's major radius does
            // not enclose the point the walk starts from, so it has no inboard and outboard
            // crossing to report. The axis sits a few centimetres off `z = 0` at most, so this
            // excludes only the innermost fraction of a per cent in `psi_norm`
            if psi_norm[i_psi_norm] < psi_norm_at_mag_r {
                continue;
            }

            let psi_target: f64 = psi_a + psi_norm[i_psi_norm] * (psi_b - psi_a);
            r_inboard_z_0_profile[i_psi_norm] = r_at_psi(
                r,
                &psi_at_grid_lines,
                &d_psi_d_r_at_grid_lines,
                mag_r,
                psi_at_mag_r,
                psi_target,
                MidplaneSide::Inboard,
            );
            r_outboard_z_0_profile[i_psi_norm] = r_at_psi(
                r,
                &psi_at_grid_lines,
                &d_psi_d_r_at_grid_lines,
                mag_r,
                psi_at_mag_r,
                psi_target,
                MidplaneSide::Outboard,
            );
        }
    }

    time_slice.profiles_1d.r_inboard_z_0 = r_inboard_z_0_profile;
    time_slice.profiles_1d.r_outboard_z_0 = r_outboard_z_0_profile;
}

/// The major radius at which a flux surface crosses the row, on one side of the magnetic axis.
///
/// The row is walked outwards from the magnetic axis's major radius and the **first** crossing is
/// taken, so the surface found is the one nearest the axis. That is what makes the answer the
/// nested flux surface rather than some re-entrant part of the same flux value further out.
///
/// # Arguments
/// * `r` - the grid's radial lines, [metre]
/// * `psi_at_grid_lines` - `psi` along the row at each of them, [weber]
/// * `d_psi_d_r_at_grid_lines` - and its radial derivative there, [weber / metre]
/// * `mag_r` - major radius of the magnetic axis, [metre]
/// * `psi_at_mag_r` - `psi` on this row at `mag_r`, [weber]
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
    psi_at_mag_r: f64,
    psi_target: f64,
    side: MidplaneSide,
) -> f64 {
    let n_r: usize = r.len();
    let d_r: f64 = r[1] - r[0];

    // The cell holding the magnetic axis's major radius, clamped so that an axis on the last grid
    // line uses the last cell. The walk starts here, inside that cell, rather than at a grid line
    let i_r_axis: usize = (((mag_r - r[0]) / d_r).floor() as usize).min(n_r - 2);

    match side {
        MidplaneSide::Outboard => {
            'outboard_cell_loop: for i_r_left in i_r_axis..n_r - 1 {
                // The axis's own cell is bracketed from the axis outwards, and every cell beyond it
                // from its own left-hand grid line
                let psi_near: f64 = if i_r_left == i_r_axis { psi_at_mag_r } else { psi_at_grid_lines[i_r_left] };
                if !brackets(psi_near, psi_at_grid_lines[i_r_left + 1], psi_target) {
                    continue 'outboard_cell_loop;
                }

                return crossing_in_cell(r, psi_at_grid_lines, d_psi_d_r_at_grid_lines, i_r_left, psi_target, mag_r, side);
            }
        }
        MidplaneSide::Inboard => {
            'inboard_cell_loop: for i_r_left in (0..=i_r_axis).rev() {
                let psi_near: f64 = if i_r_left == i_r_axis { psi_at_mag_r } else { psi_at_grid_lines[i_r_left + 1] };
                if !brackets(psi_near, psi_at_grid_lines[i_r_left], psi_target) {
                    continue 'inboard_cell_loop;
                }

                return crossing_in_cell(r, psi_at_grid_lines, d_psi_d_r_at_grid_lines, i_r_left, psi_target, mag_r, side);
            }
        }
    }

    // The surface does not reach `z = 0` on this side of the grid
    return f64::NAN;
}

/// Solve one radial cell's cubic for the crossing on one side of the magnetic axis.
///
/// # Arguments
/// * `r` - the grid's radial lines, [metre]
/// * `psi_at_grid_lines` - `psi` along the row at each of them, [weber]
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

/// `psi` at a major radius inside one radial cell, from the cubic Hermite along the row.
///
/// The same cubic `cubic_interpolation` inverts, evaluated forwards.
///
/// # Arguments
/// * `r` - the grid's radial lines, [metre]
/// * `psi_at_grid_lines` - `psi` along the row at each of them, [weber]
/// * `d_psi_d_r_at_grid_lines` - and its radial derivative there, [weber / metre]
/// * `i_r_left` - the cell, named by the grid line on its left
/// * `r_here` - where in that cell to evaluate, [metre]
///
/// # Returns
/// * `psi` - [weber]
fn psi_hermite_at(r: &Array1<f64>, psi_at_grid_lines: &Array1<f64>, d_psi_d_r_at_grid_lines: &Array1<f64>, i_r_left: usize, r_here: f64) -> f64 {
    let delta_r: f64 = r[i_r_left + 1] - r[i_r_left];
    let t: f64 = (r_here - r[i_r_left]) / delta_r;

    let psi_left: f64 = psi_at_grid_lines[i_r_left];
    let psi_right: f64 = psi_at_grid_lines[i_r_left + 1];
    let d_psi_d_r_left: f64 = d_psi_d_r_at_grid_lines[i_r_left];
    let d_psi_d_r_right: f64 = d_psi_d_r_at_grid_lines[i_r_left + 1];

    // `a * t ** 3 + b * t ** 2 + c * t + d`, with the same Hermite basis `cubic_interpolation` uses
    let a: f64 = 2.0 * psi_left + delta_r * d_psi_d_r_left - 2.0 * psi_right + delta_r * d_psi_d_r_right;
    let b: f64 = -3.0 * psi_left - 2.0 * delta_r * d_psi_d_r_left + 3.0 * psi_right - delta_r * d_psi_d_r_right;
    let c: f64 = delta_r * d_psi_d_r_left;
    let d: f64 = psi_left;

    return ((a * t + b) * t + c) * t + d;
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
    /// The same quadratic well `r_inboard__r_outboard` is tested against, on the same 12-by-11
    /// grid, whose `z` axis carries a `z = 0` row at `z[5]`:
    ///
    /// ```text
    /// psi = 1 + 8 * (r - 0.65) ** 2 + 7 * (r - 0.65) * (z - 0.05) + 5 * (z - 0.05) ** 2
    /// ```
    ///
    /// The magnetic axis is at `(0.65, 0.05)`, so it sits 0.05 metre *above* `z = 0` and the cross
    /// term does not vanish on this row. That leaves
    /// `psi_norm = 4 * delta_r ** 2 - 0.175 * delta_r + 0.00625` along it, against
    /// `psi_norm = 4 * delta_r ** 2` along the row through the axis, so the two rows have visibly
    /// different answers and a calculator which cut the wrong one fails.
    fn time_slice_for_test() -> EquilibriumTimeSlice {
        use imas_rs::EquilibriumProfiles2d;

        let r: Array1<f64> = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let z: Array1<f64> = array![-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - 0.65;
                let delta_z: f64 = z[i_z] - 0.05;
                psi_2d[(i_z, i_r)] = 1.0 + 8.0 * delta_r.powi(2) + 7.0 * delta_r * delta_z + 5.0 * delta_z.powi(2);
                d_psi_d_r_2d[(i_z, i_r)] = 16.0 * delta_r + 7.0 * delta_z;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = 0.65;
        time_slice.global_quantities.magnetic_axis.z = 0.05;
        time_slice.global_quantities.psi_magnetic_axis = 1.0;
        time_slice.boundary.psi = 3.0;
        time_slice.profiles_2d = vec![profiles_2d];

        return time_slice;
    }

    /// The major radius of a surface on the fixture's `z = 0` row, from the closed form rather than
    /// from the interpolator.
    ///
    /// `psi_norm = 4 * delta_r ** 2 - 0.175 * delta_r + 0.00625`, so the two roots are
    /// `delta_r = (0.175 +/- sqrt(16 * psi_norm - 0.069375)) / 8`, [metre]
    fn r_at_z_0_for_test(psi_norm: f64, side: MidplaneSide) -> f64 {
        let discriminant: f64 = (16.0 * psi_norm - 0.069375).sqrt();

        return match side {
            MidplaneSide::Outboard => 0.65 + (0.175 + discriminant) / 8.0,
            MidplaneSide::Inboard => 0.65 + (0.175 - discriminant) / 8.0,
        };
    }

    #[test]
    fn each_surface_is_found_either_side_of_the_magnetic_axis() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.psi_norm = array![0.25, 0.5, 1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // Inverting the cubic is exact, so these hold to machine precision rather than to the width
        // of a grid cell
        let psi_norm: Array1<f64> = time_slice.profiles_1d.psi_norm.to_owned();
        let n_psi_norm: usize = psi_norm.len();
        for i_psi_norm in 0..n_psi_norm {
            assert_abs_diff_eq!(
                time_slice.profiles_1d.r_inboard_z_0[i_psi_norm],
                r_at_z_0_for_test(psi_norm[i_psi_norm], MidplaneSide::Inboard),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                time_slice.profiles_1d.r_outboard_z_0[i_psi_norm],
                r_at_z_0_for_test(psi_norm[i_psi_norm], MidplaneSide::Outboard),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn the_mid_plane_is_cut_and_not_the_height_of_the_magnetic_axis() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.profiles_1d.psi_norm = array![1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The fixture's magnetic axis is at `z = 0.05`, where the boundary is at `r = 1.15` on the
        // outboard side. On `z = 0` it is at 1.1708 instead, so a calculator which cut the axis's
        // own height rather than the mid-plane fails here
        assert_abs_diff_eq!(time_slice.profiles_1d.r_outboard_z_0[0], 1.170790, epsilon = 1e-6);
        assert!((time_slice.profiles_1d.r_outboard_z_0[0] - 1.15).abs() > 1e-3);
    }

    #[test]
    fn a_surface_which_does_not_reach_the_mid_plane_is_nan() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        // The magnetic axis is 0.05 metre above `z = 0`, so the innermost surfaces never come down
        // to this row. `psi_norm = 2` is off the other end of the grid
        time_slice.profiles_1d.psi_norm = array![0.0, 0.25, 2.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.r_inboard_z_0[0].is_nan());
        assert!(time_slice.profiles_1d.r_outboard_z_0[0].is_nan());
        assert!(time_slice.profiles_1d.r_inboard_z_0[1].is_finite());
        assert!(time_slice.profiles_1d.r_outboard_z_0[1].is_finite());
        assert!(time_slice.profiles_1d.r_inboard_z_0[2].is_nan());
        assert!(time_slice.profiles_1d.r_outboard_z_0[2].is_nan());
    }

    #[test]
    fn a_grid_with_no_row_on_the_mid_plane_gives_nan_radii() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        // Shifted so that the grid straddles `z = 0` without landing on it. Answering from the
        // nearest row would be wrong by half a cell in `z`, so the profiles must be NaN instead
        time_slice.profiles_2d[0].grid.dim2 = array![-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55];
        time_slice.profiles_1d.psi_norm = array![0.25, 1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.r_inboard_z_0.iter().all(|r_here: &f64| r_here.is_nan()));
        assert!(time_slice.profiles_1d.r_outboard_z_0.iter().all(|r_here: &f64| r_here.is_nan()));
    }

    #[test]
    fn a_slice_which_did_not_converge_gives_nan_radii() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.global_quantities.magnetic_axis.r = f64::NAN;
        time_slice.global_quantities.magnetic_axis.z = f64::NAN;
        time_slice.profiles_1d.psi_norm = array![0.25, 1.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.profiles_1d.r_inboard_z_0.iter().all(|r_here: &f64| r_here.is_nan()));
        assert!(time_slice.profiles_1d.r_outboard_z_0.iter().all(|r_here: &f64| r_here.is_nan()));
    }
}
