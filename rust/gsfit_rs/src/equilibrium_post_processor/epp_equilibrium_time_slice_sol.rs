//! `time_slice(itime)/sol/hfs/contour/r`, `.../hfs/contour/z`, `.../hfs/strike_point/r`,
//! `.../hfs/strike_point/z`, and the same four for `lfs`

use crate::plasma_geometry::marching_squares_for_sol::marching_squares_for_sol;
use crate::wall::vacuum_vessel_outline;
use imas_rs::EquilibriumTimeSlice;
use imas_rs::ids::wall::Wall as WallIds;
use ndarray::{Array1, Array2};

/// Trace the two scrape-off layer legs, and store them in the time-slice.
///
/// Each leg is traced from the active X-point along the separatrix until it meets the vacuum
/// vessel; the strike point is the last point of the leg. Only a diverted plasma has an X-point, so
/// a limited plasma gets empty legs.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the `sol` contours and strike points are written into it
/// * `wall_ids` - the wall IDS, which supplies the vacuum vessel the legs are traced up to
pub fn epp_equilibrium_time_slice_sol(time_slice: &mut EquilibriumTimeSlice, wall_ids: &WallIds) {
    // A slice which did not converge has no boundary at all, and so no `boundary/type`: the solver
    // leaves that one unset rather than NaN, because a boundary which does not exist is neither
    // limited nor diverted. So this has to be tested before `boundary/type` is read
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();

    // `boundary/type` is 0 for a limited plasma and 1 for a diverted one. Only a diverted plasma
    // has an X-point, and so only a diverted plasma has scrape-off layer legs
    let xpt_diverted: bool = !psi_a.is_nan() && time_slice.boundary.r#type.unwrap() == 1;
    if !xpt_diverted {
        store_legs(
            time_slice,
            &Array1::from_elem(0, f64::NAN),
            &Array1::from_elem(0, f64::NAN),
            &Array1::from_elem(0, f64::NAN),
            &Array1::from_elem(0, f64::NAN),
        );
        return;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();

    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    // When the plasma is diverted the bounding point *is* the active X-point, so it does not have
    // to be searched for among the stationary points
    let xpt_r: f64 = time_slice.boundary.bounding.r.unwrap();
    let xpt_z: f64 = time_slice.boundary.bounding.z.unwrap();

    let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(wall_ids).unwrap();

    let n_r: usize = r.len();
    let n_z: usize = z.len();

    // For `marching_squares_for_sol` we don't segment the mask into core and private flux regions, this is done in `marching_squares_for_sol`
    let mut mask: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            if psi_2d[(i_z, i_r)] > psi_b {
                mask[(i_z, i_r)] = 1.0;
            } else {
                mask[(i_z, i_r)] = 0.0;
            }
        }
    }

    let (hfs_leg, lfs_leg): (
        Result<crate::plasma_geometry::MarchingContour, String>,
        Result<crate::plasma_geometry::MarchingContour, String>,
    ) = marching_squares_for_sol(
        r,
        z,
        psi_2d,
        d_psi_d_r_2d,
        d_psi_d_z_2d,
        psi_b,
        &mask,
        Some(xpt_r),
        Some(xpt_z),
        mag_r,
        mag_z,
        &vessel_r,
        &vessel_z,
    );

    match (hfs_leg, lfs_leg) {
        (Ok(hfs_leg), Ok(lfs_leg)) => {
            store_legs(time_slice, &hfs_leg.r, &hfs_leg.z, &lfs_leg.r, &lfs_leg.z);
        }
        _ => {
            // One or both legs could not be traced, so neither is stored
            store_legs(
                time_slice,
                &Array1::from_elem(0, f64::NAN),
                &Array1::from_elem(0, f64::NAN),
                &Array1::from_elem(0, f64::NAN),
                &Array1::from_elem(0, f64::NAN),
            );
        }
    }
}

/// Store both legs and their strike points in the time-slice.
///
/// The strike point is where the leg meets the wall, which is the last point of the contour. A leg
/// which could not be traced is empty, and its strike point is NaN.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the `sol` contours and strike points are written into it
/// * `hfs_leg_r`, `hfs_leg_z` - the high field side leg [metre]
/// * `lfs_leg_r`, `lfs_leg_z` - the low field side leg [metre]
fn store_legs(time_slice: &mut EquilibriumTimeSlice, hfs_leg_r: &Array1<f64>, hfs_leg_z: &Array1<f64>, lfs_leg_r: &Array1<f64>, lfs_leg_z: &Array1<f64>) {
    time_slice.sol.hfs.contour.r = Some(hfs_leg_r.to_owned());
    time_slice.sol.hfs.contour.z = Some(hfs_leg_z.to_owned());
    time_slice.sol.hfs.strike_point.r = Some(hfs_leg_r.last().copied().unwrap_or(f64::NAN));
    time_slice.sol.hfs.strike_point.z = Some(hfs_leg_z.last().copied().unwrap_or(f64::NAN));

    time_slice.sol.lfs.contour.r = Some(lfs_leg_r.to_owned());
    time_slice.sol.lfs.contour.z = Some(lfs_leg_z.to_owned());
    time_slice.sol.lfs.strike_point.r = Some(lfs_leg_r.last().copied().unwrap_or(f64::NAN));
    time_slice.sol.lfs.strike_point.z = Some(lfs_leg_z.last().copied().unwrap_or(f64::NAN));
}
