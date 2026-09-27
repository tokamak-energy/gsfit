//! `time_slice(itime)/boundary/outline/r` and `.../z`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::MarchingContour;
use crate::plasma_geometry::marching_squares::marching_squares;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

/// A second X-point whose flux is within this fraction of `psi_b - psi_a` of the boundary's is on
/// the boundary too: the plasma is a perfect double null
const DOUBLE_NULL_PSI_NORM_TOLERANCE: f64 = 1.0e-4;

/// The `contour_tree/node/critical_type` of a saddle point, an X-point
const CRITICAL_TYPE_SADDLE: i32 = 1;

/// Trace the last closed flux surface, and store it in the time-slice.
///
/// The contour is the `psi = psi_b` isoline, traced by marching squares over the plasma mask. For a
/// diverted plasma the X-point is passed in so that the trace can be cut there, rather than
/// following the separatrix out along the divertor legs. For a perfect double null the other
/// X-point is passed in too, so that the trace passes through both; see `secondary_x_point`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/outline/r` and `.../z` are written into it
///
/// Note: the solver already traces this contour while testing whether a candidate boundary point is
/// viable, and `find_boundary` traces it again, but neither keeps the result - only the mask and
/// the bounding point are stored. So it is traced here a third time. The solver only needs the
/// mask, so the cheaper fix is to stop tracing during the iterations at all.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
    let d_psi_d_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_r;
    let d_psi_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_z;
    let mask_2d: &Array2<f64> = &time_slice.profiles_2d[0].mask;

    let psi_b: f64 = time_slice.boundary.psi;
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;

    // `boundary/type` is 0 for a limited plasma and 1 for a diverted one. Only a diverted plasma
    // has an X-point, and when it does the bounding point *is* the X-point
    let xpt_diverted: bool = time_slice.boundary.r#type == 1;
    let xpt_r_or_none: Option<f64>;
    let xpt_z_or_none: Option<f64>;
    if xpt_diverted {
        xpt_r_or_none = Some(time_slice.boundary.bounding.r);
        xpt_z_or_none = Some(time_slice.boundary.bounding.z);
    } else {
        xpt_r_or_none = None;
        xpt_z_or_none = None;
    }

    // Special condition: a perfect double null
    let (xpt_secondary_r_or_none, xpt_secondary_z_or_none): (Option<f64>, Option<f64>) = match secondary_x_point(time_slice) {
        Some((xpt_secondary_r, xpt_secondary_z)) if xpt_diverted => (Some(xpt_secondary_r), Some(xpt_secondary_z)),
        _ => (None, None),
    };

    let boundary_contour: MarchingContour = marching_squares(
        r,
        z,
        psi_2d,
        d_psi_d_r_2d,
        d_psi_d_z_2d,
        psi_b,
        mask_2d,
        xpt_r_or_none,
        xpt_z_or_none,
        xpt_secondary_r_or_none,
        xpt_secondary_z_or_none,
        mag_r,
        mag_z,
    );

    time_slice.boundary.outline.r = boundary_contour.r;
    time_slice.boundary.outline.z = boundary_contour.z;
}

/// The second X-point of a perfect double null, or `None` when the plasma is not one.
///
/// The X-points are the saddle points in `contour_tree`. A second one, besides the bounding point,
/// is on the boundary when its flux is within `DOUBLE_NULL_PSI_NORM_TOLERANCE` of the boundary's,
/// and the plasma mask reaches it: within two grid points of it. If several are, the one with the
/// flux closest to the boundary's is taken.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, with its `contour_tree`, boundary and mask
///
/// # Returns
/// * `(r, z)` of the second X-point [metre]
fn secondary_x_point(time_slice: &EquilibriumTimeSlice) -> Option<(f64, f64)> {
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let mask_2d: &Array2<f64> = &time_slice.profiles_2d[0].mask;
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_grid: f64 = (r[1] - r[0]).max(z[1] - z[0]);

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;
    let bounding_r: f64 = time_slice.boundary.bounding.r;
    let bounding_z: f64 = time_slice.boundary.bounding.z;

    let mut secondary_x_point: Option<(f64, f64)> = None;
    let mut psi_norm_difference_min: f64 = DOUBLE_NULL_PSI_NORM_TOLERANCE;
    for node in &time_slice.contour_tree.node {
        // Only X-points, other than the bounding point
        if node.critical_type != CRITICAL_TYPE_SADDLE || (node.r - bounding_r).hypot(node.z - bounding_z) < d_grid {
            continue;
        }
        let psi_norm_difference: f64 = ((node.psi - psi_b) / (psi_b - psi_a)).abs();
        if psi_norm_difference > psi_norm_difference_min {
            continue;
        }

        // The plasma must reach the X-point
        let i_r_nearest: usize = (r - node.r).abs().argmin().unwrap();
        let i_z_nearest: usize = (z - node.z).abs().argmin().unwrap();
        let mut is_mask_near: bool = false;
        for i_z in i_z_nearest.saturating_sub(2)..(i_z_nearest + 3).min(n_z) {
            for i_r in i_r_nearest.saturating_sub(2)..(i_r_nearest + 3).min(n_r) {
                is_mask_near |= mask_2d[(i_z, i_r)] > 0.0;
            }
        }
        if is_mask_near {
            psi_norm_difference_min = psi_norm_difference;
            secondary_x_point = Some((node.r, node.z));
        }
    }

    secondary_x_point
}
