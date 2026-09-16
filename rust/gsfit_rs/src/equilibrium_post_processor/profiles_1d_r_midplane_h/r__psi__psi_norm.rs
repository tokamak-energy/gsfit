//! `time_slice(itime)/profiles_1d_r_midplane_h/r`, `.../psi` and `.../psi_norm`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, s};

/// How many pieces each radial cell of the solved grid is cut into.
///
/// Subdividing, rather than laying down an independent axis, keeps every solved grid point as one
/// of the refined points. The count is **even** so that one refined point also lands exactly on
/// each cell centre; an odd count would straddle it, leaving the nearest point `d_r / (2 * n)`
/// short.
///
/// 20 takes ST40's 12.5 mm cells to 0.625 mm. That is set by what the answer is read back for:
/// inverting `psi_norm` for `R` off the solved grid is worth about 6 mm, and off this one about
/// 0.3 mm even when the caller simply takes the nearest tabulated point.
const N_R_SUBDIVISIONS_PER_CELL: usize = 20;

/// Calculate the poloidal flux along the mid-plane on a refined radial grid, and store it in the
/// time-slice.
///
/// The three nodes are filled together because they share an abscissa and one sweep of the
/// interpolator. The cut is taken at the height of the magnetic axis, which is where
/// `global_quantities/delta_r_sep` and `global_quantities/f_x` put the outboard mid-plane too.
///
/// This exists to map a normalised flux back to a major radius in the scrape-off layer, to within
/// a few millimetres. The solved grid cannot do that on its own: ST40's `d_r` is 12.5 mm, so
/// inverting `psi_norm` for `R` from the solved points is worth about 6 mm. Refining it costs
/// nothing in accuracy because the interpolation is not the limit - in the scrape-off layer `psi`
/// is generated entirely by sources some distance away, so it is smooth there, and the bicubic is
/// built from **analytic** derivatives rather than finite differences of `psi`. Its error over a
/// cell is around 0.1 micron once converted into an error in `R`.
///
/// What the refinement cannot do is undo the discretisation of the plasma current onto the same
/// 12.5 mm cells. Within the first cell outside the boundary that, not the interpolation, is what
/// sets the accuracy.
///
/// `profiles_1d/r_inboard` and `profiles_1d/r_outboard` invert this same bicubic, at this same
/// height, for a major radius directly. Inverting the profile written here lands on those radii.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the three `profiles_1d_r_midplane_h` nodes are written
///   into it
///
/// `r` is filled whatever happens, since it depends only on the grid and callers index against it.
/// A time-slice which did not converge has no magnetic axis to cut through, so `psi` and
/// `psi_norm` come out NaN.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
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

    // The refined radial axis, which depends on the grid alone and so is the same on every
    // time-slice
    let n_r_high_resolution: usize = (n_r - 1) * N_R_SUBDIVISIONS_PER_CELL + 1;
    let d_r_high_resolution: f64 = d_r / N_R_SUBDIVISIONS_PER_CELL as f64;
    let mut r_high_resolution: Array1<f64> = Array1::from_elem(n_r_high_resolution, f64::NAN);
    for i_r_high_resolution in 0..n_r_high_resolution {
        r_high_resolution[i_r_high_resolution] = r[0] + i_r_high_resolution as f64 * d_r_high_resolution;
    }

    let mut psi_high_resolution: Array1<f64> = Array1::from_elem(n_r_high_resolution, f64::NAN);
    let mut psi_norm_high_resolution: Array1<f64> = Array1::from_elem(n_r_high_resolution, f64::NAN);

    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

    // A time-slice which did not converge has a NaN magnetic axis, and so no row to cut along.
    // This has to be tested rather than left to propagate, because `mag_z` is used to pick a cell
    // index and a NaN cast to `usize` would silently saturate to the first one
    if mag_z.is_finite() {
        // The row of cells containing the magnetic axis, clamped so that an axis on the last grid
        // line uses the last cell
        let i_z_lower: usize = (((mag_z - z[0]) / d_z).floor() as usize).min(n_z - 2);
        let y: f64 = (mag_z - z[i_z_lower]) / d_z;

        // One interpolator per radial cell, each evaluated at every refined point inside it,
        // rather than one rebuilt per refined point
        for i_r_left in 0..n_r - 1 {
            let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(
                d_r,
                d_z,
                psi_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
                d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            );

            // Each cell owns the refined points from its left-hand grid line up to, but not
            // including, its right-hand one, which belongs to the next cell. The last cell also
            // owns the final point, as there is no next cell to take it
            let n_subdivision: usize = if i_r_left == n_r - 2 {
                N_R_SUBDIVISIONS_PER_CELL + 1
            } else {
                N_R_SUBDIVISIONS_PER_CELL
            };

            for i_subdivision in 0..n_subdivision {
                let x: f64 = i_subdivision as f64 / N_R_SUBDIVISIONS_PER_CELL as f64;
                let i_r_high_resolution: usize = i_r_left * N_R_SUBDIVISIONS_PER_CELL + i_subdivision;
                psi_high_resolution[i_r_high_resolution] = psi_interpolator.interpolate(x, y);
            }
        }

        // Normalised here from `psi`, and **not** taken from `profiles_2d/psi_norm`, which the
        // solver masks to the plasma and so leaves at zero across the whole scrape-off layer. The
        // result carries on past 1 going outwards, which is the point of this profile
        let d_psi_d_psi_norm: f64 = psi_b - psi_a;
        for i_r_high_resolution in 0..n_r_high_resolution {
            psi_norm_high_resolution[i_r_high_resolution] = (psi_high_resolution[i_r_high_resolution] - psi_a) / d_psi_d_psi_norm;
        }
    }

    time_slice.profiles_1d_r_midplane_h.r = r_high_resolution;
    time_slice.profiles_1d_r_midplane_h.psi = psi_high_resolution;
    time_slice.profiles_1d_r_midplane_h.psi_norm = psi_norm_high_resolution;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::super::{psi_for_test, time_slice_for_test};
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn the_refined_axis_subdivides_the_solved_grid() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        let r: Array1<f64> = time_slice.profiles_2d[0].grid.dim1.to_owned();
        let r_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.r;

        // 4 solved points means 3 cells, each cut into 20
        assert_eq!(r_high_resolution.len(), 61);
        assert_abs_diff_eq!(r_high_resolution[0], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(r_high_resolution[60], 0.8, epsilon = 1e-12);

        // Subdividing means every solved grid point is still one of the refined points
        let n_r: usize = r.len();
        for i_r in 0..n_r {
            assert_abs_diff_eq!(r_high_resolution[i_r * N_R_SUBDIVISIONS_PER_CELL], r[i_r], epsilon = 1e-12);
        }

        // An even subdivision count also puts one refined point exactly on each cell centre
        for i_r in 0..n_r - 1 {
            let i_r_high_resolution: usize = i_r * N_R_SUBDIVISIONS_PER_CELL + N_R_SUBDIVISIONS_PER_CELL / 2;
            assert_abs_diff_eq!(r_high_resolution[i_r_high_resolution], (r[i_r] + r[i_r + 1]) / 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn psi_is_interpolated_along_the_row_through_the_magnetic_axis() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The fixture's `psi` is bilinear, which a bicubic reproduces exactly, so every refined
        // point has a closed form. The cut is at the magnetic axis height, which lies between two
        // grid rows, so getting this right needs the `z` interpolation as well as the `r` one
        let r_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.r;
        let psi_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.psi;
        let n_r_high_resolution: usize = r_high_resolution.len();
        for i_r_high_resolution in 0..n_r_high_resolution {
            assert_abs_diff_eq!(
                psi_high_resolution[i_r_high_resolution],
                psi_for_test(r_high_resolution[i_r_high_resolution], mag_z),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn psi_norm_is_normalised_from_psi_and_not_masked_to_the_plasma() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // `profiles_2d/psi_norm` and `profiles_2d/mask` are zero throughout the fixture, so a
        // calculator reading either of them instead of normalising `psi` itself fails here
        let psi_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.psi;
        let psi_norm_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.psi_norm;
        let n_r_high_resolution: usize = psi_norm_high_resolution.len();
        for i_r_high_resolution in 0..n_r_high_resolution {
            assert_abs_diff_eq!(
                psi_norm_high_resolution[i_r_high_resolution],
                (psi_high_resolution[i_r_high_resolution] - 1.0) / 2.0,
                epsilon = 1e-12
            );
        }

        // And it must carry on past 1 going outwards, which is what the scrape-off layer needs
        assert_abs_diff_eq!(psi_norm_high_resolution[60], 1.175, epsilon = 1e-12);
        assert!(psi_norm_high_resolution[60] > 1.0);
    }

    #[test]
    fn a_slice_with_no_magnetic_axis_gives_nan_profiles_but_still_gives_r() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test();
        time_slice.global_quantities.magnetic_axis.z = f64::NAN;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // `r` depends on the grid alone, and callers index against it, so it is filled regardless
        let r_high_resolution: &Array1<f64> = &time_slice.profiles_1d_r_midplane_h.r;
        assert_eq!(r_high_resolution.len(), 61);
        assert!(r_high_resolution.iter().all(|r_point: &f64| r_point.is_finite()));

        // A NaN height cast to a cell index would saturate to the first row, so this must not
        // quietly return the profile along the bottom of the grid
        assert!(time_slice.profiles_1d_r_midplane_h.psi.iter().all(|psi: &f64| psi.is_nan()));
        assert!(time_slice.profiles_1d_r_midplane_h.psi_norm.iter().all(|psi_norm: &f64| psi_norm.is_nan()));
    }
}
