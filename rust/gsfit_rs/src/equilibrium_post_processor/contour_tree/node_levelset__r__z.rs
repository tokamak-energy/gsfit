//! `time_slice(itime)/contour_tree/node(i_node)/levelset(i_level)/r` and `.../z`, for the node which
//! is the secondary X-point
//!
//! The four contours which leave the secondary X-point along its separatrix, the flux surface
//! `psi = psi(secondary X-point)`:
//!
//! | `i_level` | leaves the secondary X-point | ends on the limiter on the |
//! |-----------|------------------------------|----------------------------|
//! | 0         | away from the plasma core    | low field side             |
//! | 1         | away from the plasma core    | high field side            |
//! | 2         | towards the plasma core      | low field side             |
//! | 3         | towards the plasma core      | high field side            |
//!
//! Each contour starts at the secondary X-point, and normally ends at its interpolated intersection
//! with the limiter. A pair which instead closes on itself inside the limiter, as the two leaving away
//! from the core do around a private flux region with an O-point of its own, is split at the
//! stagnation point. The closed contour encloses an O-point, where the poloidal field stagnates, and
//! is cut where the straight line from the secondary X-point through that O-point crosses it on the
//! far side; each contour of the pair ends there.
//!
//! This order is GSFit's own: the data dictionary orders an X-point's `levelset` differently.

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::limiter::vacuum_vessel_outline;
use crate::plasma_geometry::MarchingContour;
use crate::plasma_geometry::cubic_interpolation::cubic_interpolation;
use crate::plasma_geometry::marching_squares_for_sol::sort_boundary_points_version_4;
use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Contains, Coord, Line, LineString, Point, Polygon};
use imas_rs::EquilibriumProfiles2d;
use imas_rs::EquilibriumTimeSlice;
use imas_rs::ids::equilibrium::{EquilibriumContourTreeNode, Rz1dDynamicAos};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// The data dictionary's `boundary/type` for a diverted plasma
const BOUNDARY_TYPE_DIVERTED: i32 = 1;

/// The data dictionary's `contour_tree/node/critical_type`
const CRITICAL_TYPE_MINIMUM: i32 = 0;
const CRITICAL_TYPE_SADDLE: i32 = 1;
const CRITICAL_TYPE_MAXIMUM: i32 = 2;

/// The number of contours stored for the secondary X-point, one per `i_level`
const N_LEVEL: usize = 4;

/// The number of contours which leave an X-point
const N_START_POINT: usize = 4;

/// A secondary X-point whose `psi_norm` is within this of 1 lies on the primary separatrix: a perfect
/// double null [dimensionless]
const PSI_NORM_DOUBLE_NULL_TOLERANCE: f64 = 1.0e-6;

/// Points closer together than this are the same point [metre]
const POINT_TOLERANCE: f64 = 1.0e-9;

/// A contour which closes inside the limiter stops when it gets back to the cells around the secondary
/// X-point, so it ends within this many cells of it; one which ends further away stopped short
/// [dimensionless]
const CLOSED_CONTOUR_GAP_MAX: f64 = 4.0;

/// Trace the four contours which leave the secondary X-point, and store them as its `levelset`.
///
/// The secondary X-point is the saddle point in `contour_tree/node` which is inside the limiter, on
/// the other side of the magnetic axis from the primary X-point (the bounding point), and outside the
/// plasma; of several, the one nearest the plasma in flux. The contours are traced over the `(R, Z)`
/// grid, cell edge to cell edge, as the scrape-off layer legs are.
///
/// There is no data, and `levelset` is left empty, when the plasma is limited or did not converge,
/// when there is no secondary X-point inside the limiter, when four contours cannot be found leaving
/// it, and for a perfect double null, where the secondary X-point lies on the primary separatrix and
/// the contours towards the plasma core run into the primary X-point. A contour which neither reaches
/// the limiter nor closes is left empty on its own.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the secondary X-point's `contour_tree/node(:)/levelset` is
///   written into it
/// * `constant_values` - the constant values; `limiter`, whose vacuum vessel is the limiter the
///   contours end on, is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // 1. The plasma and its primary X-point, which only a diverted plasma has: its bounding point. A
    //    time-slice which did not converge has `boundary/type = EMPTY_INT`, so is rejected here too
    if time_slice.boundary.r#type != BOUNDARY_TYPE_DIVERTED {
        return;
    }
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let primary_xpt_z: f64 = time_slice.boundary.bounding.z;
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;
    if !mag_r.is_finite() || !mag_z.is_finite() || !primary_xpt_z.is_finite() || !psi_a.is_finite() || !psi_b.is_finite() || psi_b == psi_a {
        return;
    }

    // 2. The limiter: `equilibrium/limiter/unit(0)/outline`, the closed contour which bounds the
    //    plasma
    let (limiter_r, limiter_z): (Array1<f64>, Array1<f64>) = match vacuum_vessel_outline(constant_values.limiter) {
        Ok(limiter) => limiter,
        Err(_) => return,
    };
    let n_limiter: usize = limiter_r.len();
    if n_limiter < 3 {
        return;
    }
    let mut limiter_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_limiter);
    for i_limiter in 0..n_limiter {
        if !limiter_r[i_limiter].is_finite() || !limiter_z[i_limiter].is_finite() {
            return;
        }
        limiter_coordinates.push(Coord {
            x: limiter_r[i_limiter],
            y: limiter_z[i_limiter],
        });
    }
    let limiter_polygon: Polygon<f64> = Polygon::new(LineString::from(limiter_coordinates), vec![]);

    // 3. The secondary X-point: a saddle point inside the limiter, on the other side of the magnetic
    //    axis from the primary X-point, and outside the plasma; of several, the one nearest the plasma
    //    in flux. `psi_norm` is compared rather than `psi`, because whether `psi` rises or falls away
    //    from the magnetic axis depends on the sign of the plasma current.
    //
    //    The O-points are gathered on the way, for splitting a closed contour (step 10)
    let n_node: usize = time_slice.contour_tree.node.len();
    let mut i_node_secondary_xpt: Option<usize> = None;
    let mut secondary_xpt_psi_norm: f64 = f64::INFINITY;
    let mut o_points_r: Vec<f64> = Vec::new();
    let mut o_points_z: Vec<f64> = Vec::new();
    let mut o_points_psi: Vec<f64> = Vec::new();
    for i_node in 0..n_node {
        let node: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_node];
        if !node.r.is_finite() || !node.z.is_finite() || !node.psi.is_finite() {
            continue;
        }
        if node.critical_type == CRITICAL_TYPE_MINIMUM || node.critical_type == CRITICAL_TYPE_MAXIMUM {
            o_points_r.push(node.r);
            o_points_z.push(node.z);
            o_points_psi.push(node.psi);
            continue;
        }
        if node.critical_type != CRITICAL_TYPE_SADDLE {
            continue;
        }
        if (node.z - mag_z) * (primary_xpt_z - mag_z) >= 0.0 {
            continue;
        }
        if !limiter_polygon.contains(&Point::new(node.r, node.z)) {
            continue;
        }
        // A saddle point with `psi_norm < 1` is in a region of its own, which the plasma's flux
        // surfaces do not reach: if they did, it would bound the plasma
        let node_psi_norm: f64 = (node.psi - psi_a) / (psi_b - psi_a);
        if node_psi_norm < 1.0 - PSI_NORM_DOUBLE_NULL_TOLERANCE {
            continue;
        }
        if node_psi_norm < secondary_xpt_psi_norm {
            i_node_secondary_xpt = Some(i_node);
            secondary_xpt_psi_norm = node_psi_norm;
        }
    }
    let i_node_secondary_xpt: usize = match i_node_secondary_xpt {
        Some(i_node_secondary_xpt) => i_node_secondary_xpt,
        None => return,
    };

    // 4. A perfect double null, where the secondary X-point lies on the primary separatrix, has no data
    if (secondary_xpt_psi_norm - 1.0).abs() < PSI_NORM_DOUBLE_NULL_TOLERANCE {
        return;
    }
    let secondary_xpt_r: f64 = time_slice.contour_tree.node[i_node_secondary_xpt].r;
    let secondary_xpt_z: f64 = time_slice.contour_tree.node[i_node_secondary_xpt].z;
    let secondary_xpt_psi: f64 = time_slice.contour_tree.node[i_node_secondary_xpt].psi;

    // 5. The flux on the grid. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    let profiles_2d: &EquilibriumProfiles2d = match time_slice.profiles_2d.first() {
        Some(profiles_2d) => profiles_2d,
        None => return,
    };
    let r: &Array1<f64> = &profiles_2d.grid.dim1;
    let z: &Array1<f64> = &profiles_2d.grid.dim2;
    let psi_2d: &Array2<f64> = &profiles_2d.psi;
    let d_psi_d_r_2d: &Array2<f64> = &profiles_2d.d_psi_d_r;
    let d_psi_d_z_2d: &Array2<f64> = &profiles_2d.d_psi_d_z;
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    // The 2x2 cells around the secondary X-point (step 7) need at least 3x3 grid points
    if n_r < 3 || n_z < 3 {
        return;
    }
    if psi_2d.dim() != (n_z, n_r) || d_psi_d_r_2d.dim() != (n_z, n_r) || d_psi_d_z_2d.dim() != (n_z, n_r) {
        return;
    }

    // 6. Where the separatrix crosses each cell edge: the root of the cubic through `psi` and its
    //    derivative at the two ends of the edge; when the cubic has more than one root, the one nearest
    //    the straight line between them. Keyed by edge, `(i_r_from, i_z_from, i_r_to, i_z_to)`, as
    //    `sort_boundary_points_version_4` expects
    let mut unsorted_contour_points: HashMap<(usize, usize, usize, usize), (f64, f64)> = HashMap::new();
    // Edges along R
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            let psi_from: f64 = psi_2d[(i_z, i_r)];
            let psi_to: f64 = psi_2d[(i_z, i_r + 1)];
            // Only an edge whose ends are on opposite sides of the separatrix is crossed
            if (psi_from > secondary_xpt_psi) == (psi_to > secondary_xpt_psi) {
                continue;
            }
            let roots_r: Array1<f64> = match cubic_interpolation(
                r[i_r],
                psi_from,
                d_psi_d_r_2d[(i_z, i_r)],
                r[i_r + 1],
                psi_to,
                d_psi_d_r_2d[(i_z, i_r + 1)],
                secondary_xpt_psi,
            ) {
                Ok(roots_r) => roots_r,
                Err(_) => continue,
            };
            let linear_r: f64 = r[i_r] + (r[i_r + 1] - r[i_r]) * (secondary_xpt_psi - psi_from) / (psi_to - psi_from);
            let mut crossing_r: f64 = f64::NAN;
            let n_root: usize = roots_r.len();
            for i_root in 0..n_root {
                if crossing_r.is_nan() || (roots_r[i_root] - linear_r).abs() < (crossing_r - linear_r).abs() {
                    crossing_r = roots_r[i_root];
                }
            }
            if crossing_r.is_finite() {
                unsorted_contour_points.insert((i_r, i_z, i_r + 1, i_z), (crossing_r, z[i_z]));
            }
        }
    }
    // Edges along Z
    for i_r in 0..n_r {
        for i_z in 0..n_z - 1 {
            let psi_from: f64 = psi_2d[(i_z, i_r)];
            let psi_to: f64 = psi_2d[(i_z + 1, i_r)];
            if (psi_from > secondary_xpt_psi) == (psi_to > secondary_xpt_psi) {
                continue;
            }
            let roots_z: Array1<f64> = match cubic_interpolation(
                z[i_z],
                psi_from,
                d_psi_d_z_2d[(i_z, i_r)],
                z[i_z + 1],
                psi_to,
                d_psi_d_z_2d[(i_z + 1, i_r)],
                secondary_xpt_psi,
            ) {
                Ok(roots_z) => roots_z,
                Err(_) => continue,
            };
            let linear_z: f64 = z[i_z] + (z[i_z + 1] - z[i_z]) * (secondary_xpt_psi - psi_from) / (psi_to - psi_from);
            let mut crossing_z: f64 = f64::NAN;
            let n_root: usize = roots_z.len();
            for i_root in 0..n_root {
                if crossing_z.is_nan() || (roots_z[i_root] - linear_z).abs() < (crossing_z - linear_z).abs() {
                    crossing_z = roots_z[i_root];
                }
            }
            if crossing_z.is_finite() {
                unsorted_contour_points.insert((i_r, i_z, i_r, i_z + 1), (r[i_r], crossing_z));
            }
        }
    }

    // 7. Where the four contours leave the block of 2x2 cells around the grid point nearest the
    //    secondary X-point. A block rather than the one cell containing the secondary X-point keeps it
    //    at least half a cell from where the contours leave, so that they leave as four distinct points
    //    even when it is on a grid line, as it is in a symmetric equilibrium
    let mut i_r_nearest: usize = 0;
    for i_r in 1..n_r {
        if (r[i_r] - secondary_xpt_r).abs() < (r[i_r_nearest] - secondary_xpt_r).abs() {
            i_r_nearest = i_r;
        }
    }
    let mut i_z_nearest: usize = 0;
    for i_z in 1..n_z {
        if (z[i_z] - secondary_xpt_z).abs() < (z[i_z_nearest] - secondary_xpt_z).abs() {
            i_z_nearest = i_z;
        }
    }
    if i_r_nearest == 0 || i_r_nearest == n_r - 1 || i_z_nearest == 0 || i_z_nearest == n_z - 1 {
        return;
    }
    let i_r_left: usize = i_r_nearest - 1;
    let i_r_right: usize = i_r_nearest + 1;
    let i_z_lower: usize = i_z_nearest - 1;
    let i_z_upper: usize = i_z_nearest + 1;
    // The eight edges around the outside of the block: bottom, top, left and right
    let block_outer_edges: [(usize, usize, usize, usize); 8] = [
        (i_r_left, i_z_lower, i_r_nearest, i_z_lower),
        (i_r_nearest, i_z_lower, i_r_right, i_z_lower),
        (i_r_left, i_z_upper, i_r_nearest, i_z_upper),
        (i_r_nearest, i_z_upper, i_r_right, i_z_upper),
        (i_r_left, i_z_lower, i_r_left, i_z_nearest),
        (i_r_left, i_z_nearest, i_r_left, i_z_upper),
        (i_r_right, i_z_lower, i_r_right, i_z_nearest),
        (i_r_right, i_z_nearest, i_r_right, i_z_upper),
    ];
    // The cell of the block each outer edge belongs to, which a contour leaving through it is traced
    // away from. A cell is named by its lower left grid point
    let block_outer_edges_cell: [(usize, usize); 8] = [
        (i_r_left, i_z_lower),
        (i_r_nearest, i_z_lower),
        (i_r_left, i_z_nearest),
        (i_r_nearest, i_z_nearest),
        (i_r_left, i_z_lower),
        (i_r_left, i_z_nearest),
        (i_r_nearest, i_z_lower),
        (i_r_nearest, i_z_nearest),
    ];
    // The four edges inside the block, which meet at its centre
    let block_inner_edges: [(usize, usize, usize, usize); 4] = [
        (i_r_left, i_z_nearest, i_r_nearest, i_z_nearest),
        (i_r_nearest, i_z_nearest, i_r_right, i_z_nearest),
        (i_r_nearest, i_z_lower, i_r_nearest, i_z_nearest),
        (i_r_nearest, i_z_nearest, i_r_nearest, i_z_upper),
    ];
    // No contour is traced into the block, where the four meet, so a contour which closes inside the
    // limiter stops when it gets back to it
    let n_block_outer_edge: usize = block_outer_edges.len();
    for i_block_outer_edge in 0..n_block_outer_edge {
        unsorted_contour_points.remove(&block_outer_edges[i_block_outer_edge]);
    }
    let n_block_inner_edge: usize = block_inner_edges.len();
    for i_block_inner_edge in 0..n_block_inner_edge {
        unsorted_contour_points.remove(&block_inner_edges[i_block_inner_edge]);
    }
    // Every root on the outer edges, not only on edges whose ends are on opposite sides of the
    // separatrix: two contours can leave through the same edge
    let mut start_points_r: Vec<f64> = Vec::new();
    let mut start_points_z: Vec<f64> = Vec::new();
    let mut start_points_edge: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut start_points_cell: Vec<(usize, usize)> = Vec::new();
    for i_block_outer_edge in 0..n_block_outer_edge {
        let edge: (usize, usize, usize, usize) = block_outer_edges[i_block_outer_edge];
        let cell: (usize, usize) = block_outer_edges_cell[i_block_outer_edge];
        let (i_r_from, i_z_from, i_r_to, i_z_to): (usize, usize, usize, usize) = edge;
        let mut edge_points_r: Vec<f64> = Vec::new();
        let mut edge_points_z: Vec<f64> = Vec::new();
        if i_z_from == i_z_to {
            let roots_r: Array1<f64> = match cubic_interpolation(
                r[i_r_from],
                psi_2d[(i_z_from, i_r_from)],
                d_psi_d_r_2d[(i_z_from, i_r_from)],
                r[i_r_to],
                psi_2d[(i_z_from, i_r_to)],
                d_psi_d_r_2d[(i_z_from, i_r_to)],
                secondary_xpt_psi,
            ) {
                Ok(roots_r) => roots_r,
                Err(_) => Array1::zeros(0),
            };
            let n_root: usize = roots_r.len();
            for i_root in 0..n_root {
                edge_points_r.push(roots_r[i_root]);
                edge_points_z.push(z[i_z_from]);
            }
        } else {
            let roots_z: Array1<f64> = match cubic_interpolation(
                z[i_z_from],
                psi_2d[(i_z_from, i_r_from)],
                d_psi_d_z_2d[(i_z_from, i_r_from)],
                z[i_z_to],
                psi_2d[(i_z_to, i_r_from)],
                d_psi_d_z_2d[(i_z_to, i_r_from)],
                secondary_xpt_psi,
            ) {
                Ok(roots_z) => roots_z,
                Err(_) => Array1::zeros(0),
            };
            let n_root: usize = roots_z.len();
            for i_root in 0..n_root {
                edge_points_r.push(r[i_r_from]);
                edge_points_z.push(roots_z[i_root]);
            }
        }
        // A contour which leaves through a corner of the block is found on both edges at that corner
        let n_edge_point: usize = edge_points_r.len();
        'edge_point_loop: for i_edge_point in 0..n_edge_point {
            if !edge_points_r[i_edge_point].is_finite() || !edge_points_z[i_edge_point].is_finite() {
                continue 'edge_point_loop;
            }
            let n_start_point: usize = start_points_r.len();
            for i_start_point in 0..n_start_point {
                let distance: f64 =
                    (edge_points_r[i_edge_point] - start_points_r[i_start_point]).hypot(edge_points_z[i_edge_point] - start_points_z[i_start_point]);
                if distance < POINT_TOLERANCE {
                    continue 'edge_point_loop;
                }
            }
            start_points_r.push(edge_points_r[i_edge_point]);
            start_points_z.push(edge_points_z[i_edge_point]);
            start_points_edge.push(edge);
            start_points_cell.push(cell);
        }
    }
    if start_points_r.len() != N_START_POINT {
        return;
    }

    // 8. Which contour is which. The two which leave most nearly towards the magnetic axis go towards
    //    the plasma core, and the other two away from it. Of each pair, the one which leaves at the
    //    larger R is on the low field side
    let core_delta_r: f64 = mag_r - secondary_xpt_r;
    let core_delta_z: f64 = mag_z - secondary_xpt_z;
    let core_distance: f64 = core_delta_r.hypot(core_delta_z);
    if core_distance < POINT_TOLERANCE {
        return;
    }
    // The cosine of the angle between where each contour leaves and the direction of the magnetic axis
    let mut start_points_cos_angle_to_core: [f64; N_START_POINT] = [f64::NAN; N_START_POINT];
    for i_start_point in 0..N_START_POINT {
        let delta_r: f64 = start_points_r[i_start_point] - secondary_xpt_r;
        let delta_z: f64 = start_points_z[i_start_point] - secondary_xpt_z;
        let distance: f64 = delta_r.hypot(delta_z);
        if distance < POINT_TOLERANCE {
            return;
        }
        start_points_cos_angle_to_core[i_start_point] = (delta_r * core_delta_r + delta_z * core_delta_z) / (distance * core_distance);
    }
    // Ranked by how nearly each leaves towards the core, 0 the most nearly; a tie goes to the lower index
    let mut towards_core: Vec<usize> = Vec::with_capacity(2);
    let mut away_from_core: Vec<usize> = Vec::with_capacity(2);
    for i_start_point in 0..N_START_POINT {
        let mut rank: usize = 0;
        for j_start_point in 0..N_START_POINT {
            let cos_angle_i: f64 = start_points_cos_angle_to_core[i_start_point];
            let cos_angle_j: f64 = start_points_cos_angle_to_core[j_start_point];
            if cos_angle_j > cos_angle_i || (cos_angle_j == cos_angle_i && j_start_point < i_start_point) {
                rank += 1;
            }
        }
        if rank < 2 {
            towards_core.push(i_start_point);
        } else {
            away_from_core.push(i_start_point);
        }
    }
    if towards_core.len() != 2 || away_from_core.len() != 2 {
        return;
    }
    // `level_i_start_point[i_level]` is the start point of contour `i_level`
    let mut level_i_start_point: [usize; N_LEVEL] = [0; N_LEVEL];
    if start_points_r[away_from_core[0]] > start_points_r[away_from_core[1]] {
        level_i_start_point[0] = away_from_core[0];
        level_i_start_point[1] = away_from_core[1];
    } else {
        level_i_start_point[0] = away_from_core[1];
        level_i_start_point[1] = away_from_core[0];
    }
    if start_points_r[towards_core[0]] > start_points_r[towards_core[1]] {
        level_i_start_point[2] = towards_core[0];
        level_i_start_point[3] = towards_core[1];
    } else {
        level_i_start_point[2] = towards_core[1];
        level_i_start_point[3] = towards_core[0];
    }

    // 9. Trace each contour from the secondary X-point, out of the block and on, cell edge to cell edge,
    //    until it crosses the limiter, where it ends at the interpolated intersection
    let mut level_contours: Vec<MarchingContour> = Vec::with_capacity(N_LEVEL);
    let mut level_crosses_limiter: [bool; N_LEVEL] = [false; N_LEVEL];
    for i_level in 0..N_LEVEL {
        let i_start_point: usize = level_i_start_point[i_level];
        let start_r: f64 = start_points_r[i_start_point];
        let start_z: f64 = start_points_z[i_start_point];

        // A contour which has already crossed the limiter by the time it leaves the block crosses it
        // between the secondary X-point and there
        if !limiter_polygon.contains(&Point::new(start_r, start_z)) {
            let first_segment: Line<f64> = Line::new(
                Coord {
                    x: secondary_xpt_r,
                    y: secondary_xpt_z,
                },
                Coord { x: start_r, y: start_z },
            );
            let mut intersection_distance: f64 = f64::INFINITY;
            let mut intersection_r: f64 = f64::NAN;
            let mut intersection_z: f64 = f64::NAN;
            for limiter_segment in limiter_polygon.exterior().lines() {
                if let Some(LineIntersection::SinglePoint { intersection, .. }) = line_intersection(first_segment, limiter_segment) {
                    let distance: f64 = (intersection.x - secondary_xpt_r).hypot(intersection.y - secondary_xpt_z);
                    if distance < intersection_distance {
                        intersection_distance = distance;
                        intersection_r = intersection.x;
                        intersection_z = intersection.y;
                    }
                }
            }
            if intersection_r.is_finite() && intersection_z.is_finite() {
                level_contours.push(MarchingContour {
                    r: Array1::from_vec(vec![secondary_xpt_r, intersection_r]),
                    z: Array1::from_vec(vec![secondary_xpt_z, intersection_z]),
                    n: 2,
                });
                level_crosses_limiter[i_level] = true;
            } else {
                level_contours.push(MarchingContour {
                    r: Array1::zeros(0),
                    z: Array1::zeros(0),
                    n: 0,
                });
            }
            continue;
        }

        let (contour, crosses_limiter): (MarchingContour, bool) = sort_boundary_points_version_4(
            &unsorted_contour_points,
            start_r,
            start_z,
            start_points_edge[i_start_point],
            secondary_xpt_r,
            secondary_xpt_z,
            start_points_cell[i_start_point],
            &limiter_r,
            &limiter_z,
            n_r,
            n_z,
        );
        level_contours.push(contour);
        level_crosses_limiter[i_level] = crosses_limiter;
    }

    // The contours which reach the limiter are stored as they are
    let mut levelset: Vec<Rz1dDynamicAos> = vec![Rz1dDynamicAos::default(); N_LEVEL];
    for i_level in 0..N_LEVEL {
        if level_crosses_limiter[i_level] {
            levelset[i_level].r = level_contours[i_level].r.clone();
            levelset[i_level].z = level_contours[i_level].z.clone();
        }
    }

    // 10. A pair neither of which reaches the limiter has closed on itself inside it: each contour
    //     went all the way round, until it got back to the block around the secondary X-point. The
    //     closed contour is split at the stagnation point, the O-point it encloses; of several, the one
    //     furthest in flux from the separatrix, at the centre of the contours nested inside. Each
    //     contour of the pair is cut where the straight line from the secondary X-point through the
    //     stagnation point first crosses it beyond the stagnation point, so that between them the pair
    //     goes round the closed contour once
    let cell_size: f64 = (r[1] - r[0]).abs().max((z[1] - z[0]).abs());
    // Longer than any contour on the grid is across [metre]
    let ray_length: f64 = (r[n_r - 1] - r[0]).hypot(z[n_z - 1] - z[0]);
    let n_o_point: usize = o_points_r.len();
    'pair_loop: for i_pair in 0..N_LEVEL / 2 {
        let i_level_lfs: usize = 2 * i_pair;
        let i_level_hfs: usize = 2 * i_pair + 1;
        if level_crosses_limiter[i_level_lfs] || level_crosses_limiter[i_level_hfs] {
            continue 'pair_loop;
        }

        // Both must have got back to the secondary X-point, rather than stopped short
        for i_level in [i_level_lfs, i_level_hfs] {
            let n_point: usize = level_contours[i_level].r.len();
            if n_point < 3 || level_contours[i_level].z.len() != n_point {
                continue 'pair_loop;
            }
            let gap: f64 = (level_contours[i_level].r[n_point - 1] - secondary_xpt_r).hypot(level_contours[i_level].z[n_point - 1] - secondary_xpt_z);
            if gap.is_nan() || gap > CLOSED_CONTOUR_GAP_MAX * cell_size {
                continue 'pair_loop;
            }
        }

        // The stagnation point
        let n_point_lfs: usize = level_contours[i_level_lfs].r.len();
        let mut closed_contour_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_point_lfs);
        for i_point in 0..n_point_lfs {
            closed_contour_coordinates.push(Coord {
                x: level_contours[i_level_lfs].r[i_point],
                y: level_contours[i_level_lfs].z[i_point],
            });
        }
        let closed_contour_polygon: Polygon<f64> = Polygon::new(LineString::from(closed_contour_coordinates), vec![]);
        let mut i_o_point_stagnation: Option<usize> = None;
        let mut stagnation_delta_psi: f64 = -1.0;
        for i_o_point in 0..n_o_point {
            if !closed_contour_polygon.contains(&Point::new(o_points_r[i_o_point], o_points_z[i_o_point])) {
                continue;
            }
            let delta_psi: f64 = (o_points_psi[i_o_point] - secondary_xpt_psi).abs();
            if delta_psi > stagnation_delta_psi {
                i_o_point_stagnation = Some(i_o_point);
                stagnation_delta_psi = delta_psi;
            }
        }
        let i_o_point_stagnation: usize = match i_o_point_stagnation {
            Some(i_o_point_stagnation) => i_o_point_stagnation,
            None => continue 'pair_loop,
        };
        let stagnation_r: f64 = o_points_r[i_o_point_stagnation];
        let stagnation_z: f64 = o_points_z[i_o_point_stagnation];

        // The straight line on from the stagnation point, directly away from the secondary X-point
        let ray_delta_r: f64 = stagnation_r - secondary_xpt_r;
        let ray_delta_z: f64 = stagnation_z - secondary_xpt_z;
        let ray_distance: f64 = ray_delta_r.hypot(ray_delta_z);
        if ray_distance < POINT_TOLERANCE {
            continue 'pair_loop;
        }
        let ray: Line<f64> = Line::new(
            Coord {
                x: stagnation_r,
                y: stagnation_z,
            },
            Coord {
                x: stagnation_r + ray_length * ray_delta_r / ray_distance,
                y: stagnation_z + ray_length * ray_delta_z / ray_distance,
            },
        );

        // Cut each contour of the pair where it crosses the line nearest the stagnation point
        let mut pair_r: Vec<Array1<f64>> = Vec::with_capacity(2);
        let mut pair_z: Vec<Array1<f64>> = Vec::with_capacity(2);
        for i_level in [i_level_lfs, i_level_hfs] {
            let contour: &MarchingContour = &level_contours[i_level];
            let n_point: usize = contour.r.len();
            let mut split_distance: f64 = f64::INFINITY;
            let mut split_i_point: usize = 0;
            let mut split_r: f64 = f64::NAN;
            let mut split_z: f64 = f64::NAN;
            for i_point in 0..n_point - 1 {
                let segment: Line<f64> = Line::new(
                    Coord {
                        x: contour.r[i_point],
                        y: contour.z[i_point],
                    },
                    Coord {
                        x: contour.r[i_point + 1],
                        y: contour.z[i_point + 1],
                    },
                );
                if let Some(LineIntersection::SinglePoint { intersection, .. }) = line_intersection(ray, segment) {
                    let distance: f64 = (intersection.x - stagnation_r).hypot(intersection.y - stagnation_z);
                    if distance < split_distance {
                        split_distance = distance;
                        split_i_point = i_point;
                        split_r = intersection.x;
                        split_z = intersection.y;
                    }
                }
            }
            if !split_r.is_finite() || !split_z.is_finite() {
                continue 'pair_loop;
            }
            // The contour up to the segment the line crosses, then the crossing
            let mut cut_r: Array1<f64> = Array1::from_elem(split_i_point + 2, f64::NAN);
            let mut cut_z: Array1<f64> = Array1::from_elem(split_i_point + 2, f64::NAN);
            for i_point in 0..=split_i_point {
                cut_r[i_point] = contour.r[i_point];
                cut_z[i_point] = contour.z[i_point];
            }
            cut_r[split_i_point + 1] = split_r;
            cut_z[split_i_point + 1] = split_z;
            pair_r.push(cut_r);
            pair_z.push(cut_z);
        }
        levelset[i_level_lfs].r = pair_r[0].clone();
        levelset[i_level_lfs].z = pair_z[0].clone();
        levelset[i_level_hfs].r = pair_r[1].clone();
        levelset[i_level_hfs].z = pair_z[1].clone();
    }

    // 11. Store, unless there is no data at all
    let mut n_level_with_data: usize = 0;
    for i_level in 0..N_LEVEL {
        if !levelset[i_level].r.is_empty() {
            n_level_with_data += 1;
        }
    }
    if n_level_with_data == 0 {
        return;
    }
    if let Some(node) = time_slice.contour_tree.node.get_mut(i_node_secondary_xpt) {
        node.levelset = levelset;
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use crate::grad_shafranov::contour_tree_nodes;
    use crate::plasma_geometry::{StationaryPoint, find_stationary_points_using_winding_number};
    use imas_rs::{EquilibriumLimiter, EquilibriumLimiterUnit, EquilibriumLimiterUnitOutline};
    use ndarray::array;

    // The flux of the test equilibria is a bowl with a Gaussian dip above and below its centre:
    //
    //     psi = psi_sign * PSI_SCALE * g
    //     g = ((r - BOWL_R) / BOWL_LENGTH) ** 2 + (z / BOWL_LENGTH) ** 2
    //         - sum over the dips of depth * exp(-((r - dip_r) ** 2 + (z - dip_z) ** 2) / DIP_WIDTH ** 2)
    //
    // The bottom of the bowl is the magnetic axis. Each dip is an O-point, with an X-point between it and
    // the magnetic axis; the X-point of the deeper dip bounds the plasma. The separatrix of either
    // X-point closes around its dip at about `|z| = 0.5`, so a limiter at `|z| = 0.38` cuts the contours
    // (the left of the figure), and one at `|z| = 0.75` leaves them closed (the right of the figure)

    /// Centre of the bowl [metre]
    const BOWL_R: f64 = 0.5;
    /// Distance over which `g` rises by 1 from the centre of the bowl [metre]
    const BOWL_LENGTH: f64 = 0.3;
    /// Width of each dip [metre]
    const DIP_WIDTH: f64 = 0.12;
    /// Height of the dip above the centre of the bowl [metre]
    const DIP_UPPER_Z: f64 = 0.45;
    /// Position of the dip below the centre of the bowl [metre]
    const DIP_LOWER_R: f64 = 0.5;
    const DIP_LOWER_Z: f64 = -0.45;
    /// Flux per unit of `g` [weber]
    const PSI_SCALE: f64 = 0.1;
    /// Extent in R of the rectangular limiter [metre]
    const LIMITER_R_MIN: f64 = 0.15;
    const LIMITER_R_MAX: f64 = 0.85;

    /// `psi` and its first and second derivatives, `[psi, d_psi_d_r, d_psi_d_z, d2_psi_d_r2, d2_psi_d_r_d_z, d2_psi_d_z2]`.
    ///
    /// # Arguments
    /// * `r`, `z` - where to evaluate [metre]
    /// * `dip_upper_r` - R of the dip above the centre of the bowl [metre]
    /// * `dip_upper_depth`, `dip_lower_depth` - the depths of the two dips [dimensionless]
    /// * `psi_sign` - +1 for `psi` to rise away from the magnetic axis, -1 for it to fall
    fn psi_and_derivatives(r: f64, z: f64, dip_upper_r: f64, dip_upper_depth: f64, dip_lower_depth: f64, psi_sign: f64) -> [f64; 6] {
        let mut g: f64 = ((r - BOWL_R) / BOWL_LENGTH).powi(2) + (z / BOWL_LENGTH).powi(2);
        let mut d_g_d_r: f64 = 2.0 * (r - BOWL_R) / BOWL_LENGTH.powi(2);
        let mut d_g_d_z: f64 = 2.0 * z / BOWL_LENGTH.powi(2);
        let mut d2_g_d_r2: f64 = 2.0 / BOWL_LENGTH.powi(2);
        let mut d2_g_d_r_d_z: f64 = 0.0;
        let mut d2_g_d_z2: f64 = 2.0 / BOWL_LENGTH.powi(2);

        let dips_r: [f64; 2] = [dip_upper_r, DIP_LOWER_R];
        let dips_z: [f64; 2] = [DIP_UPPER_Z, DIP_LOWER_Z];
        let dips_depth: [f64; 2] = [dip_upper_depth, dip_lower_depth];
        let width_sq: f64 = DIP_WIDTH.powi(2);
        for i_dip in 0..2 {
            let delta_r: f64 = r - dips_r[i_dip];
            let delta_z: f64 = z - dips_z[i_dip];
            let dip: f64 = dips_depth[i_dip] * (-(delta_r.powi(2) + delta_z.powi(2)) / width_sq).exp();
            g -= dip;
            d_g_d_r += 2.0 * delta_r / width_sq * dip;
            d_g_d_z += 2.0 * delta_z / width_sq * dip;
            d2_g_d_r2 += (2.0 / width_sq - 4.0 * delta_r.powi(2) / width_sq.powi(2)) * dip;
            d2_g_d_r_d_z += -4.0 * delta_r * delta_z / width_sq.powi(2) * dip;
            d2_g_d_z2 += (2.0 / width_sq - 4.0 * delta_z.powi(2) / width_sq.powi(2)) * dip;
        }

        let psi_per_g: f64 = psi_sign * PSI_SCALE;
        [
            psi_per_g * g,
            psi_per_g * d_g_d_r,
            psi_per_g * d_g_d_z,
            psi_per_g * d2_g_d_r2,
            psi_per_g * d2_g_d_r_d_z,
            psi_per_g * d2_g_d_z2,
        ]
    }

    /// A converged, diverted time-slice with the flux of `psi_and_derivatives`, on a 1 cm grid. The
    /// stationary points are found, and classified, by the solver's own functions; the X-point nearest
    /// the magnetic axis in flux is the bounding point.
    fn time_slice_with_dips(dip_upper_r: f64, dip_upper_depth: f64, dip_lower_depth: f64, psi_sign: f64) -> EquilibriumTimeSlice {
        let n_r: usize = 81;
        let n_z: usize = 161;
        let r: Array1<f64> = Array1::linspace(0.1, 0.9, n_r);
        let z: Array1<f64> = Array1::linspace(-0.8, 0.8, n_z);
        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_z2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let psi: [f64; 6] = psi_and_derivatives(r[i_r], z[i_z], dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign);
                psi_2d[(i_z, i_r)] = psi[0];
                d_psi_d_r_2d[(i_z, i_r)] = psi[1];
                d_psi_d_z_2d[(i_z, i_r)] = psi[2];
                d2_psi_d_r2_2d[(i_z, i_r)] = psi[3];
                d2_psi_d_r_d_z_2d[(i_z, i_r)] = psi[4];
                d2_psi_d_z2_2d[(i_z, i_r)] = psi[5];
            }
        }

        let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
            r.view(),
            z.view(),
            psi_2d.view(),
            d_psi_d_r_2d.view(),
            d_psi_d_z_2d.view(),
            d2_psi_d_r2_2d.view(),
            d2_psi_d_r_d_z_2d.view(),
            d2_psi_d_z2_2d.view(),
        );
        let n_stationary_point: usize = stationary_points.len();
        let mut i_magnetic_axis: usize = 0;
        for i_stationary_point in 1..n_stationary_point {
            let stationary_point: &StationaryPoint = &stationary_points[i_stationary_point];
            let magnetic_axis: &StationaryPoint = &stationary_points[i_magnetic_axis];
            if (stationary_point.r - BOWL_R).hypot(stationary_point.z) < (magnetic_axis.r - BOWL_R).hypot(magnetic_axis.z) {
                i_magnetic_axis = i_stationary_point;
            }
        }
        let magnetic_axis: StationaryPoint = stationary_points[i_magnetic_axis];
        let mut primary_xpt: Option<StationaryPoint> = None;
        for i_stationary_point in 0..n_stationary_point {
            let stationary_point: StationaryPoint = stationary_points[i_stationary_point];
            if stationary_point.hessian_determinant >= 0.0 {
                continue;
            }
            let is_nearer: bool = match primary_xpt {
                None => true,
                Some(primary_xpt) => (stationary_point.psi - magnetic_axis.psi).abs() < (primary_xpt.psi - magnetic_axis.psi).abs(),
            };
            if is_nearer {
                primary_xpt = Some(stationary_point);
            }
        }
        let primary_xpt: StationaryPoint = primary_xpt.expect("the test equilibrium has X-points");

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;
        profiles_2d.d_psi_d_z = d_psi_d_z_2d;
        profiles_2d.d2_psi_d_r2 = d2_psi_d_r2_2d;
        profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;
        profiles_2d.d2_psi_d_z2 = d2_psi_d_z2_2d;
        time_slice.profiles_2d = vec![profiles_2d];
        time_slice.contour_tree.node = contour_tree_nodes(&stationary_points);
        time_slice.global_quantities.magnetic_axis.r = magnetic_axis.r;
        time_slice.global_quantities.magnetic_axis.z = magnetic_axis.z;
        time_slice.global_quantities.psi_magnetic_axis = magnetic_axis.psi;
        time_slice.boundary.r#type = BOUNDARY_TYPE_DIVERTED;
        time_slice.boundary.psi = primary_xpt.psi;
        time_slice.boundary.bounding.r = primary_xpt.r;
        time_slice.boundary.bounding.z = primary_xpt.z;

        time_slice
    }

    /// A limiter whose vacuum vessel is a rectangle, `LIMITER_R_MIN <= r <= LIMITER_R_MAX` and
    /// `-limiter_half_height <= z <= limiter_half_height` [metre]
    fn rectangular_limiter(limiter_half_height: f64) -> EquilibriumLimiter {
        let vacuum_vessel: EquilibriumLimiterUnit = EquilibriumLimiterUnit {
            outline: EquilibriumLimiterUnitOutline {
                r: array![LIMITER_R_MIN, LIMITER_R_MAX, LIMITER_R_MAX, LIMITER_R_MIN, LIMITER_R_MIN],
                z: array![
                    -limiter_half_height,
                    -limiter_half_height,
                    limiter_half_height,
                    limiter_half_height,
                    -limiter_half_height
                ],
            },
            ..Default::default()
        };

        EquilibriumLimiter { unit: vec![vacuum_vessel] }
    }

    /// Run the calculator on `time_slice`, inside `limiter`
    fn calculate_with_limiter(time_slice: &mut EquilibriumTimeSlice, limiter: &EquilibriumLimiter) {
        let constant_values: ConstantValues = ConstantValues {
            limiter,
            ..constant_values_for_test()
        };
        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        calculate(time_slice, &constant_values, &mut intermediate_values);
    }

    /// The nodes with a `levelset`
    fn nodes_with_levelset(time_slice: &EquilibriumTimeSlice) -> Vec<usize> {
        let mut i_nodes: Vec<usize> = Vec::new();
        let n_node: usize = time_slice.contour_tree.node.len();
        for i_node in 0..n_node {
            if !time_slice.contour_tree.node[i_node].levelset.is_empty() {
                i_nodes.push(i_node);
            }
        }

        i_nodes
    }

    /// The mean of `values`
    fn mean(values: &Array1<f64>) -> f64 {
        values.sum() / values.len() as f64
    }

    /// The left of the figure: a lower single null whose secondary X-point's four contours all reach
    /// the limiter. The upper dip is off-centre, so nothing is symmetric
    #[test]
    fn contours_end_on_the_limiter() {
        let limiter_half_height: f64 = 0.38;
        let (dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign): (f64, f64, f64, f64) = (0.53, 2.5, 3.0, -1.0);
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign);
        calculate_with_limiter(&mut time_slice, &rectangular_limiter(limiter_half_height));

        let i_nodes: Vec<usize> = nodes_with_levelset(&time_slice);
        assert_eq!(i_nodes.len(), 1, "only the secondary X-point has a levelset");
        let node: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_nodes[0]];
        assert_eq!(node.critical_type, CRITICAL_TYPE_SADDLE);
        assert!(node.z > time_slice.global_quantities.magnetic_axis.z, "the secondary X-point is the upper one");
        assert_eq!(node.levelset.len(), N_LEVEL);

        for i_level in 0..N_LEVEL {
            let contour_r: &Array1<f64> = &node.levelset[i_level].r;
            let contour_z: &Array1<f64> = &node.levelset[i_level].z;
            let n_point: usize = contour_r.len();
            assert!(n_point > 3, "contour {i_level} has {n_point} points");
            assert_eq!(contour_z.len(), n_point);
            assert_eq!(
                (contour_r[0], contour_z[0]),
                (node.r, node.z),
                "contour {i_level} starts at the secondary X-point"
            );

            // Away from the core ends on the top of the limiter, towards it on the bottom
            let end_z_expected: f64 = if i_level < 2 { limiter_half_height } else { -limiter_half_height };
            assert!(
                (contour_z[n_point - 1] - end_z_expected).abs() < 1.0e-12,
                "contour {i_level} ends at z = {}",
                contour_z[n_point - 1]
            );

            // Every point between is on the separatrix, to within the cubic interpolation along each cell edge
            for i_point in 1..n_point - 1 {
                let psi: f64 = psi_and_derivatives(contour_r[i_point], contour_z[i_point], dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign)[0];
                assert!(
                    (psi - node.psi).abs() < 1.0e-5 * PSI_SCALE,
                    "contour {i_level}, point {i_point}: psi = {psi}, separatrix psi = {}",
                    node.psi
                );
            }
        }

        // The low field side contour of each pair ends at the larger R
        let mut level_end_r: [f64; N_LEVEL] = [f64::NAN; N_LEVEL];
        for i_level in 0..N_LEVEL {
            let n_point: usize = node.levelset[i_level].r.len();
            level_end_r[i_level] = node.levelset[i_level].r[n_point - 1];
        }
        assert!(level_end_r[0] > node.r && node.r > level_end_r[1], "away from the core: {level_end_r:?}");
        assert!(level_end_r[2] > node.r && node.r > level_end_r[3], "towards the core: {level_end_r:?}");
        // Towards the core, the contours go round the outside of the plasma
        let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
        assert!(node.levelset[2].r.iter().any(|&r| r > mag_r + 0.15), "towards the core, low field side");
        assert!(node.levelset[3].r.iter().any(|&r| r < mag_r - 0.15), "towards the core, high field side");
    }

    /// The same, upside down: an upper single null, whose secondary X-point is below the magnetic axis
    #[test]
    fn secondary_x_point_below_the_magnetic_axis() {
        let limiter_half_height: f64 = 0.38;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 3.0, 2.5, -1.0);
        calculate_with_limiter(&mut time_slice, &rectangular_limiter(limiter_half_height));

        let i_nodes: Vec<usize> = nodes_with_levelset(&time_slice);
        assert_eq!(i_nodes.len(), 1);
        let node: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_nodes[0]];
        assert!(node.z < time_slice.global_quantities.magnetic_axis.z, "the secondary X-point is the lower one");
        assert_eq!(node.levelset.len(), N_LEVEL);
        for i_level in 0..N_LEVEL {
            let n_point: usize = node.levelset[i_level].z.len();
            let end_z_expected: f64 = if i_level < 2 { -limiter_half_height } else { limiter_half_height };
            assert!((node.levelset[i_level].z[n_point - 1] - end_z_expected).abs() < 1.0e-12, "contour {i_level}");
        }
        let n_point_0: usize = node.levelset[0].r.len();
        let n_point_1: usize = node.levelset[1].r.len();
        assert!(
            node.levelset[0].r[n_point_0 - 1] > node.levelset[1].r[n_point_1 - 1],
            "level 0 is the low field side"
        );
    }

    /// The right of the figure: inside a larger limiter both pairs close on themselves, so each is
    /// split at its stagnation point
    #[test]
    fn closed_contours_are_split_at_the_stagnation_point() {
        let limiter_half_height: f64 = 0.75;
        let (dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign): (f64, f64, f64, f64) = (0.53, 2.5, 3.0, -1.0);
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign);
        calculate_with_limiter(&mut time_slice, &rectangular_limiter(limiter_half_height));

        let i_nodes: Vec<usize> = nodes_with_levelset(&time_slice);
        assert_eq!(i_nodes.len(), 1);
        let node: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_nodes[0]];
        assert_eq!(node.levelset.len(), N_LEVEL);

        // The stagnation points: the O-point of the upper dip for the pair away from the core, and of
        // the lower dip, the deepest of the O-points the pair towards the core encloses, for that pair
        let n_node: usize = time_slice.contour_tree.node.len();
        let mut stagnation_points: [(f64, f64); 2] = [(f64::NAN, f64::NAN); 2];
        let dips_r: [f64; 2] = [dip_upper_r, DIP_LOWER_R];
        let dips_z: [f64; 2] = [DIP_UPPER_Z, DIP_LOWER_Z];
        for i_pair in 0..2 {
            let mut distance_min: f64 = f64::INFINITY;
            for i_node in 0..n_node {
                let o_point: &EquilibriumContourTreeNode = &time_slice.contour_tree.node[i_node];
                if o_point.critical_type == CRITICAL_TYPE_SADDLE {
                    continue;
                }
                let distance: f64 = (o_point.r - dips_r[i_pair]).hypot(o_point.z - dips_z[i_pair]);
                if distance < distance_min {
                    distance_min = distance;
                    stagnation_points[i_pair] = (o_point.r, o_point.z);
                }
            }
        }

        for i_pair in 0..2 {
            let level_lfs: &Rz1dDynamicAos = &node.levelset[2 * i_pair];
            let level_hfs: &Rz1dDynamicAos = &node.levelset[2 * i_pair + 1];
            let n_point_lfs: usize = level_lfs.r.len();
            let n_point_hfs: usize = level_hfs.r.len();
            assert!(n_point_lfs > 3 && n_point_hfs > 3, "pair {i_pair}");
            assert_eq!((level_lfs.r[0], level_lfs.z[0]), (node.r, node.z));
            assert_eq!((level_hfs.r[0], level_hfs.z[0]), (node.r, node.z));

            // The two end at the same point
            let split_r: f64 = level_lfs.r[n_point_lfs - 1];
            let split_z: f64 = level_lfs.z[n_point_lfs - 1];
            assert!(
                (level_hfs.r[n_point_hfs - 1] - split_r).hypot(level_hfs.z[n_point_hfs - 1] - split_z) < 1.0e-12,
                "pair {i_pair}"
            );

            // That point is on the straight line from the secondary X-point through the stagnation
            // point, beyond the stagnation point
            let (stagnation_r, stagnation_z): (f64, f64) = stagnation_points[i_pair];
            let line_r: f64 = stagnation_r - node.r;
            let line_z: f64 = stagnation_z - node.z;
            let split_delta_r: f64 = split_r - node.r;
            let split_delta_z: f64 = split_z - node.z;
            let cross: f64 = (line_r * split_delta_z - line_z * split_delta_r) / (line_r.hypot(line_z) * split_delta_r.hypot(split_delta_z));
            assert!(cross.abs() < 1.0e-9, "pair {i_pair}: the split point is off the line by sin(angle) = {cross}");
            assert!(
                split_delta_r.hypot(split_delta_z) > line_r.hypot(line_z),
                "pair {i_pair}: the split point is beyond the stagnation point"
            );

            // And on the separatrix, to within the straight line between the two points on it either side,
            // which is a tenth of a cell
            let psi_split: [f64; 6] = psi_and_derivatives(split_r, split_z, dip_upper_r, dip_upper_depth, dip_lower_depth, psi_sign);
            let split_distance_from_separatrix: f64 = (psi_split[0] - node.psi).abs() / psi_split[1].hypot(psi_split[2]);
            assert!(
                split_distance_from_separatrix < 1.0e-3,
                "pair {i_pair}: the split point is {split_distance_from_separatrix} metre from the separatrix"
            );

            // Nowhere near the limiter
            for level in [level_lfs, level_hfs] {
                assert!(level.z.iter().all(|&z| z.abs() < limiter_half_height - 0.1));
                assert!(level.r.iter().all(|&r| r > LIMITER_R_MIN + 0.05 && r < LIMITER_R_MAX - 0.05));
            }

            // The low field side half is the one at the larger R
            assert!(mean(&level_lfs.r) > mean(&level_hfs.r), "pair {i_pair}");
        }

        // Away from the core goes round the upper dip; towards it, round the plasma and the lower dip
        assert!(node.levelset[0].z.iter().all(|&z| z >= node.z - 1.0e-12));
        assert!(node.levelset[2].z.iter().any(|&z| z < DIP_LOWER_Z));
    }

    /// The same contours whichever way `psi` runs, which depends on the sign of the plasma current
    #[test]
    fn the_sign_of_psi_does_not_matter() {
        let mut time_slice_falling: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        let mut time_slice_rising: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, 1.0);
        for limiter_half_height in [0.38, 0.75] {
            let limiter: EquilibriumLimiter = rectangular_limiter(limiter_half_height);
            calculate_with_limiter(&mut time_slice_falling, &limiter);
            calculate_with_limiter(&mut time_slice_rising, &limiter);

            let i_nodes_falling: Vec<usize> = nodes_with_levelset(&time_slice_falling);
            let i_nodes_rising: Vec<usize> = nodes_with_levelset(&time_slice_rising);
            assert_eq!(i_nodes_falling.len(), 1);
            assert_eq!(i_nodes_rising.len(), 1);
            let levelset_falling: &Vec<Rz1dDynamicAos> = &time_slice_falling.contour_tree.node[i_nodes_falling[0]].levelset;
            let levelset_rising: &Vec<Rz1dDynamicAos> = &time_slice_rising.contour_tree.node[i_nodes_rising[0]].levelset;
            for i_level in 0..N_LEVEL {
                assert_eq!(levelset_falling[i_level].r.len(), levelset_rising[i_level].r.len(), "contour {i_level}");
                let n_point: usize = levelset_falling[i_level].r.len();
                for i_point in 0..n_point {
                    assert!((levelset_falling[i_level].r[i_point] - levelset_rising[i_level].r[i_point]).abs() < 1.0e-12);
                    assert!((levelset_falling[i_level].z[i_point] - levelset_rising[i_level].z[i_point]).abs() < 1.0e-12);
                }
            }
        }
    }

    /// A perfect double null has no data: the two dips are the same, so the two X-points are mirror
    /// images, on the same flux surface
    #[test]
    fn perfect_double_null_has_no_data() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(DIP_LOWER_R, 2.5, 2.5, -1.0);

        // It is a perfect double null
        let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
        let psi_b: f64 = time_slice.boundary.psi;
        let mut n_xpt_on_separatrix: usize = 0;
        for node in &time_slice.contour_tree.node {
            if node.critical_type == CRITICAL_TYPE_SADDLE && ((node.psi - psi_a) / (psi_b - psi_a) - 1.0).abs() < 1.0e-12 {
                n_xpt_on_separatrix += 1;
            }
        }
        assert_eq!(n_xpt_on_separatrix, 2);

        for limiter_half_height in [0.38, 0.5, 0.75] {
            calculate_with_limiter(&mut time_slice, &rectangular_limiter(limiter_half_height));
            assert!(nodes_with_levelset(&time_slice).is_empty(), "limiter at |z| = {limiter_half_height}");
        }
    }

    /// No data, and no panic, when there is nothing to find or the inputs are not usable
    #[test]
    fn no_data_without_usable_inputs() {
        let limiter: EquilibriumLimiter = rectangular_limiter(0.38);

        // A limited plasma
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        time_slice.boundary.r#type = 0;
        calculate_with_limiter(&mut time_slice, &limiter);
        assert!(nodes_with_levelset(&time_slice).is_empty(), "limited");

        // No limiter
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        let constant_values: ConstantValues = constant_values_for_test();
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());
        assert!(nodes_with_levelset(&time_slice).is_empty(), "no limiter");

        // The secondary X-point, at z = 0.26, is outside the limiter
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        calculate_with_limiter(&mut time_slice, &rectangular_limiter(0.2));
        assert!(nodes_with_levelset(&time_slice).is_empty(), "outside the limiter");

        // The flux is NaN
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        time_slice.profiles_2d[0].psi.fill(f64::NAN);
        calculate_with_limiter(&mut time_slice, &limiter);
        assert!(nodes_with_levelset(&time_slice).is_empty(), "NaN flux");

        // The flux is the wrong shape
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        time_slice.profiles_2d[0].psi = Array2::zeros((3, 3));
        calculate_with_limiter(&mut time_slice, &limiter);
        assert!(nodes_with_levelset(&time_slice).is_empty(), "wrong shape");

        // No grid at all
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_dips(0.53, 2.5, 3.0, -1.0);
        time_slice.profiles_2d.clear();
        calculate_with_limiter(&mut time_slice, &limiter);
        assert!(nodes_with_levelset(&time_slice).is_empty(), "no grid");

        // A time-slice which did not converge
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        calculate_with_limiter(&mut time_slice, &limiter);
        assert!(nodes_with_levelset(&time_slice).is_empty(), "not converged");
    }
}
