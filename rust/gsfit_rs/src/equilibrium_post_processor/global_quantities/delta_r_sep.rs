//! `time_slice(itime)/global_quantities/delta_r_sep`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
use crate::wall::vacuum_vessel_outline;
use geo::{Contains, Coord, LineString, Point, Polygon};
use imas_rs::EquilibriumTimeSlice;
use imas_rs::ids::wall::Wall as WallIds;
use ndarray::{Array1, Array2, s};

// Convention taken from: https://doi.org/10.1088/1741-4326/aac006

/// Calculate the radial separation of the two separatrices at the outboard midplane, and store it
/// in the time-slice.
///
/// `delta_r_sep` is the distance, at the height of the magnetic axis and on the outboard side, between
/// the flux surface through the lower X-point and the one through the upper X-point:
///
/// ```text
/// delta_r_sep = r_outboard(psi_x_lower) - r_outboard(psi_x_upper)
/// ```
///
/// The active X-point is whichever of the two bounds the plasma, so its surface is the innermost of
/// the pair. A lower single null therefore gives a negative `delta_r_sep` and an upper single null a
/// positive one, with a connected double null at zero, and no test of which X-point is active is
/// needed to get that sign.
///
/// It is evaluated in flux rather than by tracing the second separatrix: the difference in `psi`
/// between the two X-points is divided by `d(psi)/d(r)` at the outboard midplane. `psi` is
/// stationary at an X-point, so an error in an X-point's *position* moves its `psi` only at second
/// order, which is what makes a millimetre-scale answer meaningful on a centimetre grid. The
/// linearisation is good because the two separatrices are millimetres apart, over which
/// `d(psi)/d(r)` barely changes.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/delta_r_sep` is written into it
/// * `wall_ids` - the wall IDS, which supplies the vacuum vessel the X-points must lie inside
///
/// NaN when the plasma is limited, when no second X-point was found, or when the slice did not
/// converge.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let wall_ids: &WallIds = constant_values.wall_ids;

    time_slice.global_quantities.delta_r_sep = Some(f64::NAN);

    // `boundary/type` is 0 for a limited plasma and 1 for a diverted one. A limited plasma has no
    // active X-point, so there is no pair to separate. A slice which did not converge has this set
    // to `EMPTY_INT`, which is neither, and so is rejected here too
    let xpt_diverted: bool = time_slice.boundary.r#type.unwrap() == 1;
    if !xpt_diverted {
        return;
    }

    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(wall_ids).unwrap();
    let vessel_coordinates: Vec<Coord<f64>> = vessel_r.iter().zip(vessel_z.iter()).map(|(&x, &y)| Coord { x, y }).collect();
    let vessel_polygon: Polygon = Polygon::new(
        LineString::from(vessel_coordinates),
        vec![], // No holes
    );

    // The innermost X-point above and below the magnetic axis.
    //
    // `psi` falls outwards from the magnetic axis, so of the X-points bounding the plasma the
    // innermost is the one with the largest `psi`, and the active one sits exactly on `psi_b`.
    // The contour tree holds every stationary point the solver found, most of which are nulls in
    // and around the poloidal field coils rather than plasma X-points, so they are filtered: a
    // candidate must be a saddle, must lie inside the vacuum vessel, and must not be inside the
    // plasma, which is what `psi <= psi_b` tests
    let mut psi_x_upper: f64 = f64::NEG_INFINITY;
    let mut psi_x_lower: f64 = f64::NEG_INFINITY;
    let n_node: usize = time_slice.contour_tree.node.len();
    for i_node in 0..n_node {
        let node: &imas_rs::ids::equilibrium::EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_node];

        // `critical_type` is 1 for a saddle point
        if node.critical_type.unwrap() != 1 {
            continue;
        }

        let node_r: f64 = node.r.unwrap();
        let node_z: f64 = node.z.unwrap();
        let node_psi: f64 = node.psi.unwrap();

        if node_psi > psi_b {
            continue;
        }
        if !vessel_polygon.contains(&Point::new(node_r, node_z)) {
            continue;
        }

        if node_z > mag_z {
            psi_x_upper = psi_x_upper.max(node_psi);
        } else {
            psi_x_lower = psi_x_lower.max(node_psi);
        }
    }

    // A single null with no second X-point inside the vessel has no separation to report
    if !psi_x_upper.is_finite() || !psi_x_lower.is_finite() {
        return;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();
    let d2_psi_d_r_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d2_psi_d_r_d_z.as_ref().unwrap();

    // Where the separatrix crosses the height of the magnetic axis, on the outboard side
    let r_outboard: &Array1<f64> = time_slice.profiles_1d.r_outboard.as_ref().unwrap();
    let r_omp: f64 = r_outboard[r_outboard.len() - 1];
    if !r_omp.is_finite() {
        return;
    }

    let d_psi_d_r_omp: f64 = d_psi_d_r_at(r, z, psi_2d, d_psi_d_r_2d, d_psi_d_z_2d, d2_psi_d_r_d_z_2d, r_omp, mag_z);
    if d_psi_d_r_omp == 0.0 {
        return;
    }

    let delta_r_sep: f64 = (psi_x_lower - psi_x_upper) / d_psi_d_r_omp;

    time_slice.global_quantities.delta_r_sep = Some(delta_r_sep);
}

/// `d(psi)/d(r)` at an arbitrary point, from the bicubic model of the cell containing it.
///
/// # Arguments
/// * `r` - R grid points, [metre]
/// * `z` - Z grid points, [metre]
/// * `psi_2d`, `d_psi_d_r_2d`, `d_psi_d_z_2d`, `d2_psi_d_r_d_z_2d` - shape = (n_z, n_r)
/// * `r_point`, `z_point` - where to evaluate, [metre]
///
/// # Returns
/// * `d_psi_d_r` - [weber / metre]
fn d_psi_d_r_at(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    d_psi_d_r_2d: &Array2<f64>,
    d_psi_d_z_2d: &Array2<f64>,
    d2_psi_d_r_d_z_2d: &Array2<f64>,
    r_point: f64,
    z_point: f64,
) -> f64 {
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // The cell containing the point, clamped so that a point on the last grid line uses the last cell
    let i_r_left: usize = (((r_point - r[0]) / d_r).floor() as usize).min(n_r - 2);
    let i_z_lower: usize = (((z_point - z[0]) / d_z).floor() as usize).min(n_z - 2);

    let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(
        d_r,
        d_z,
        psi_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
    );

    let x: f64 = (r_point - r[i_r_left]) / d_r;
    let y: f64 = (z_point - z[i_z_lower]) / d_z;

    // `value_and_derivatives` differentiates with respect to the normalised cell coordinate
    return psi_interpolator.value_and_derivatives(x, y).d_f_d_x / d_r;
}
