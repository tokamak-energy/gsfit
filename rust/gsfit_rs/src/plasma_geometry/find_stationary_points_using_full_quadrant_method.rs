use super::bicubic_interpolator::{BicubicInterpolator, BicubicStationaryPoint};
use crate::plasma_geometry::hessian;
use core::f64;
use ndarray::{Array1, Array2, s};
use ndarray::{SliceInfo, SliceInfoElem, Dim};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use super::cubic_interpolation::cubic_interpolation_v2;
use super::StationaryPoint;

// #[derive(Debug, Clone, Copy)]
// pub struct StationaryPoint {
//     pub r: f64,
//     pub z: f64,
//     pub psi: f64,
//     pub hessian_determinant: f64,
//     pub hessian_trace: f64,
//     pub i_r_nearest: usize,
//     pub i_z_nearest: usize,
//     pub i_r_left: usize,
//     pub i_r_right: usize,
//     pub i_z_lower: usize,
//     pub i_z_upper: usize,
// }

#[derive(Clone)]
struct Coordinate {
    r: f64,
    z: f64,
}

#[derive(Copy, Clone)]
enum CrossingKind {
    BrZero,
    BzZero,
}

/// Quadrant of (Br, Bz). Q1: (+,+), Q2: (-,+), Q3: (-,-), Q4: (+,-).
/// CCW traversal of the (Br, Bz) plane visits Q1 → Q2 → Q3 → Q4 → Q1.
fn classify_quadrant(sign_br: i8, sign_bz: i8) -> i8 {
    match (sign_br > 0, sign_bz > 0) {
        (true, true) => 1,
        (false, true) => 2,
        (false, false) => 3,
        (true, false) => 4,
    }
}

/// Maps `0.0` to `+1` so corners that lie exactly on an axis still classify into a quadrant.
fn sign_with_tiebreak(value: f64) -> i8 {
    if value >= 0.0 {
        return 1;
    } else {
        return -1;
    }
}

/// Merge the Br=0 and Bz=0 crossings on a single edge into one ordered event sequence
/// along the traversal direction. Endpoint crossings are dropped — corners are handled
/// by direct grid-point sampling, not by this list.
fn combine_and_order_edge_events(
    br_crossing_points_this_edge: &[Coordinate],
    bz_crossing_points_this_edge: &[Coordinate],
    edge_start: f64,
    edge_end: f64,
    use_r_axis: bool,
) -> Vec<CrossingKind> {
    let endpoint_tol: f64 = 1e-12 * (edge_end - edge_start).abs();
    let mut events: Vec<(f64, CrossingKind)> = Vec::with_capacity(br_crossing_points_this_edge.len() + bz_crossing_points_this_edge.len());
    for c in br_crossing_points_this_edge {
        let pos: f64 = if use_r_axis { c.r } else { c.z };
        if (pos - edge_start).abs() > endpoint_tol && (edge_end - pos).abs() > endpoint_tol {
            events.push((pos, CrossingKind::BrZero));
        }
    }
    for c in bz_crossing_points_this_edge {
        let pos: f64 = if use_r_axis { c.r } else { c.z };
        if (pos - edge_start).abs() > endpoint_tol && (edge_end - pos).abs() > endpoint_tol {
            events.push((pos, CrossingKind::BzZero));
        }
    }
    if edge_end > edge_start {
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    } else {
        events.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }
    events.into_iter().map(|(_, k)| k).collect()
}


/// Finds stationary points: extrema's (minima or maxima) and saddle points
/// TODO: improve documentation - can this method also find higher-order stationary points?
/// 
/// # Arguments
/// * `r` - the radial coordinate, [meters]
/// * `z` - the vertical coordinate, [meters]
/// * `psi_2d` - the 2D array of poloidal flux values, [weber]
/// * `d_psi_d_r_2d` - d(psi)/d(r), [tesla]
/// * `d_psi_d_z_2d` - d(psi)/d(z), [tesla]
/// * `d2_psi_d_r2_2d` - d^2(psi)/d(r)^2, [weber/meter^2]
/// * `d2_psi_d_rz_2d` - d^2(psi)/d(r)d(z), [weber/meter^2]
/// * `d2_psi_d_z2_2d` - d^2(psi)/d(z)^2, [weber/meter^2]
/// 
/// # Returns
/// * `Vec<StationaryPoint>` - a vector of stationary points, which may be empty if no stationary points are found
/// 
/// # Notes
/// The simplest way to calculate the winding number involves sampling the function along the cell edges and calculating the angle using `atan2`.
/// However, `atan2` is an expensive operation, so we instead count the number of rotations through the d(ψ)/d(r) d(ψ)/d(z) quadrants.
/// 
/// The Q1, Q2, Q3, and Q4 quadrant labels are standard definitions.
/// Typically, the axes are not part of the quadrants labels.
/// But for our purposes, we need to assign a quadrant label to the axes:
///      d(ψ)/d(z) ~ -Br                                   d(ψ)/d(z) ~ -Br                                  d(ψ)/d(z) ~ -Br
///          ▲                                                 ▲                                                ▲        Special case:
///     Q2   │   Q1                                            Q2                                               │        (0.0, 0.0) is in Q1
///   (-,+)  │  (+,+)                                          │                                                │ Q1
/// ─────────┼─────────► d(ψ)/d(r) ~ Bz               ────Q3───┼───Q1────► d(ψ)/d(r) ∝ Bz               ────────•────────► d(ψ)/d(r) ∝ Bz
///     Q3   │   Q4                                            │                                                │
///   (-,-)  │  (+,-)                                          Q4                                               │
///          │                                                 |                                                │
/// 
/// To calculate the winding number, we trace the path around the cell edges (in real space) and plot the quadrants.
/// The winding number is then the number of full CCW rotations, going through all quadrants.
/// * `winding_number=-2`
///     ==> indicates two saddle points (or one saddle point with multiplicity 2).
/// * `winding_number=-1`, e.g. "Q1 → Q4 → Q3 → Q2 → Q1" (CW loop)
///    ==> indicates saddle point
/// * `winding_number=0`, e.g. "Q1 → Q2 → Q1" or "Q1 → Q2 → Q3 → Q2 → Q1" (no complete loop)
///    ==> indicates no stationary points
/// * `winding_number=+1`, e.g. "Q1 → Q2 → Q3 → Q4 → Q1" (CCW loop)
///    ==> indicates a local extremum (minimum or maximum; need to check Hessian trace to determine minimum or maximum)
/// * `winding_number=+2`
///   ==> indicates two local extrema (minimum or maximum) (or one local extremum with multiplicity 2)
///
/// Example: `winding_number = +2`; CCW inward spiral
/// Joining points: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
///      d(ψ)/d(z) ∝ -Br
///          ▲
///     2 •  │  • 1
///       6• │ •5
/// ─────────┼─────────► d(ψ)/d(r) ∝ Bz
///       7• │ •8
///     3 •  │  • 4
///          │
/// 
/// A subtle failure mode is when there is both an o-point and an x-point in the same cell,
/// `winding_number = (+1) + (-1) = 0`, which would incorrectly indicate no stationary points.
pub fn find_stationary_points_using_full_quadrant_method(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    d_psi_d_r_2d: &Array2<f64>,
    d_psi_d_z_2d: &Array2<f64>,
    d2_psi_d_r2_2d: &Array2<f64>,
    d2_psi_d_rz_2d: &Array2<f64>,
    d2_psi_d_z2_2d: &Array2<f64>,
) -> Vec<StationaryPoint> {
    // Empty stationary_points, ready to return if no stationary points found
    let mut stationary_points: Vec<StationaryPoint> = Vec::new();

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // value: number of crossings
    let mut br_crossing_points: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();
    let mut bz_crossing_points: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();

    // Horizontal march: (i_r, i_z) → (i_r+1, i_z)
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            // `br = - 1 / (2.0 * pi) d(psi)/d(z)`
            let br_crossings: Vec<f64> = cubic_interpolation_v2(
                r[i_r],
                d_psi_d_z_2d[(i_z, i_r)],
                d2_psi_d_rz_2d[(i_z, i_r)],
                r[i_r + 1],
                d_psi_d_z_2d[(i_z, i_r + 1)],
                d2_psi_d_rz_2d[(i_z, i_r + 1)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(br_crossings.len());
            for &r_cross in &br_crossings {
                crossing_coordinates_this_edge.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            br_crossing_points.insert((i_r, i_z, i_r + 1, i_z), crossing_coordinates_this_edge);

            // `bz = 1 / (2.0 * pi) * d(psi)/d(r)`
            let bz_crossings: Vec<f64> = cubic_interpolation_v2(
                r[i_r],
                d_psi_d_r_2d[(i_z, i_r)],
                d2_psi_d_r2_2d[(i_z, i_r)],
                r[i_r + 1],
                d_psi_d_r_2d[(i_z, i_r + 1)],
                d2_psi_d_r2_2d[(i_z, i_r + 1)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(bz_crossings.len());
            for &r_cross in &bz_crossings {
                crossing_coordinates_this_edge.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            bz_crossing_points.insert((i_r, i_z, i_r + 1, i_z), crossing_coordinates_this_edge);
        }
    }

    // Vertical march: (i_r, i_z) → (i_r, i_z+1)
    //
    // To prevent double counting, we discard crossings which are exactly at the grid points,
    // since they will have already been counted in the horizontal march.
    // In practice, this is fairly unlikely.
    // But it could happen in a perfect double null from synthetic data.
    for i_r in 0..n_r {
        for i_z in 0..n_z - 1 {
            let z_start: f64 = z[i_z];
            let z_end: f64 = z[i_z + 1];
            let endpoint_tol: f64 = 1e-12 * (z_end - z_start);

            // `br = - 1 / (2.0 * pi) d(psi)/d(z)`
            let br_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                z_start,
                d_psi_d_z_2d[(i_z, i_r)],
                d2_psi_d_z2_2d[(i_z, i_r)],
                z_end,
                d_psi_d_z_2d[(i_z + 1, i_r)],
                d2_psi_d_z2_2d[(i_z + 1, i_r)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(br_crossings_this_edge.len());
            for &z_cross in &br_crossings_this_edge {
                if z_cross - z_start > endpoint_tol && z_end - z_cross > endpoint_tol {
                    crossing_coordinates_this_edge.push(Coordinate { r: r[i_r], z: z_cross });
                }
            }
            br_crossing_points.insert((i_r, i_z, i_r, i_z + 1), crossing_coordinates_this_edge);

            // `bz = 1 / (2.0 * pi) * d(psi)/d(r)`
            let bz_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                z_start,
                d_psi_d_r_2d[(i_z, i_r)],
                d2_psi_d_rz_2d[(i_z, i_r)],
                z_end,
                d_psi_d_r_2d[(i_z + 1, i_r)],
                d2_psi_d_rz_2d[(i_z + 1, i_r)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(bz_crossings_this_edge.len());
            for &z_cross in &bz_crossings_this_edge {
                if z_cross - z_start > endpoint_tol && z_end - z_cross > endpoint_tol {
                    crossing_coordinates_this_edge.push(Coordinate { r: r[i_r], z: z_cross });
                }
            }
            bz_crossing_points.insert((i_r, i_z, i_r, i_z + 1), crossing_coordinates_this_edge);
        }
    }

    // Given that we are using bicubic interpolation, which can produce up to 3 crossings for both `br=0` and `bz=0` per edge.
    // The maximum possible `winding_number = 6`.
    for i_r in 0..n_r - 1 {
        let i_r_left: usize = i_r;
        let i_r_right: usize = i_r + 1;
        for i_z in 0..n_z - 1 {
            let i_z_lower: usize = i_z;
            let i_z_upper: usize = i_z + 1;

            // Walk the perimeter of the cell CCW: BL → BR → TR → TL → BL.
            // Track every time `br` or `bz` cross zero.
            // Each zero-crossing "event" changes `total_quarter_turns` by +/-1.

            // Build the ordered list of zero-crossing events.
            let mut perimeter_events: Vec<CrossingKind> = Vec::new();

            // Bottom edge: BL → BR (r increasing at z = z[i_z]).
            // Bottom edge: (i_r_left, i_z_bottom) → (i_r_right, i_z_bottom)
            let bottom_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &br_crossing_points[&(i_r_left, i_z_lower, i_r_right, i_z_lower)],
                &bz_crossing_points[&(i_r_left, i_z_lower, i_r_right, i_z_lower)],
                r[i_r],
                r[i_r + 1],
                true,
            );
            for event in bottom_events {
                perimeter_events.push(event);
            }

            // Right edge: BR → TR (z increasing at r = r[i_r+1]).
            let right_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &br_crossing_points[&(i_r_right, i_z_lower, i_r_right, i_z_upper)],
                &bz_crossing_points[&(i_r_right, i_z_lower, i_r_right, i_z_upper)],
                z[i_z],
                z[i_z + 1],
                false,
            );
            for event in right_events {
                perimeter_events.push(event);
            }

            // Top edge: TR → TL (r decreasing at z = z[i_z+1]).
            let top_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &br_crossing_points[&(i_r_left, i_z_upper, i_r_right, i_z_upper)],
                &bz_crossing_points[&(i_r_left, i_z_upper, i_r_right, i_z_upper)],
                r[i_r + 1],
                r[i_r],
                true,
            );
            for event in top_events {
                perimeter_events.push(event);
            }

            // Left edge: TL → BL (z decreasing at r = r[i_r]).
            let left_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &br_crossing_points[&(i_r_left, i_z_lower, i_r_left, i_z_upper)],
                &bz_crossing_points[&(i_r_left, i_z_lower, i_r_left, i_z_upper)],
                z[i_z + 1],
                z[i_z],
                false,
            );
            for event in left_events {
                perimeter_events.push(event);
            }

            // Walk the perimeter event-by-event, tracking which quadrant of (Br, Bz) we are in.
            // Each event changes `total_quarter_turns` by +/-1.
            //
            // Seed the walk at the bottom-left grid point.
            let mut sign_br: i8 = sign_with_tiebreak(d_psi_d_z_2d[(i_z_lower, i_r_left)]); // if `d_psi_d_z_2d[(i_z, i_r)] == 0.0`, then `sign_br = 1`
            let mut sign_bz: i8 = sign_with_tiebreak(d_psi_d_r_2d[(i_z_lower, i_r_left)]);
            let mut prev_q: i8 = classify_quadrant(sign_br, sign_bz);
            let mut total_quarter_turns: i8 = 0;

            for event in perimeter_events {
                // Flip the sign of whichever component just crossed zero.
                match event {
                    CrossingKind::BrZero => sign_br = -sign_br,
                    CrossingKind::BzZero => sign_bz = -sign_bz,
                }

                // Step the quadrant tracker: +1 for an adjacent CCW step, -1 for an adjacent CW step.
                let new_q: i8 = classify_quadrant(sign_br, sign_bz);
                let quadrant_step: i8 = (new_q - prev_q).rem_euclid(4) as i8;
                match quadrant_step {
                    0 => {} // same quadrant; no change in `total_quarter_turns`
                    1 => total_quarter_turns += 1,
                    2 => println!("Warning: diagonal jump in (Br, Bz) quadrants. This should not happen with proper sampling."),
                    3 => total_quarter_turns -= 1,
                    _ => {println!("Error: invalid `quadrant_step={}`. This should never happen?", quadrant_step);}
                }
                prev_q = new_q;
            }

            let winding_number: i8 = total_quarter_turns / 4;

            if winding_number != 0 {
                // Create a slice that covers the 4 corner points
                let slice_cell_perimeter: SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> = s![i_z_lower..=i_z_upper, i_r_left..=i_r_right];

                // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis.
                // `psi_2d` is stored as (i_z, i_r), but `BicubicInterpolator` expects axis 0 to be x (= r)
                // and axis 1 to be y (= z), so we transpose the slice.
                // TODO: It is better to use ArrayView2<f64> data types, rather than doing a copy
                let f: Array2<f64> = psi_2d.slice(slice_cell_perimeter).t().to_owned();
                let d_f_d_r: Array2<f64> = d_psi_d_r_2d.slice(slice_cell_perimeter).t().to_owned();
                let d_f_d_z: Array2<f64> = d_psi_d_z_2d.slice(slice_cell_perimeter).t().to_owned();
                let d2_f_d_r_d_z: Array2<f64> = d2_psi_d_rz_2d.slice(slice_cell_perimeter).t().to_owned();

                // Create a bicubic interpolator
                let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

                // Find the stationary point using the bicubic interpolation
                let stationary_point_or_error: Result<BicubicStationaryPoint, String> = bicubic_interpolator.find_stationary_point(1e-6, 100);

                // Extract the stationary point values
                // If the bicubic solver failed to converge, this cell is a false positive from
                // the sign-change detection (near-parallel nullclines passing through the cell
                // without actually crossing), so skip it.
                match stationary_point_or_error {
                    Ok(stationary_point) => {
                        // Extract and store results
                        let stationary_r: f64 = r[i_r_left] + stationary_point.x * d_r;
                        let stationary_z: f64 = z[i_z_lower] + stationary_point.y * d_z;
                        let stationary_psi: f64 = stationary_point.f;

                        // Compute nearest grid indices from the refined stationary point position
                        // let i_r_nearest: usize = ((stationary_r - r[0]) / d_r).round() as usize;
                        let i_r_nearest: usize = (r - stationary_r).abs().argmin().unwrap();
                        let i_z_nearest: usize = (z - stationary_z).abs().argmin().unwrap();

                        // Calculate the Hessian at the nearest grid point of the cell
                        let d2_psi_d_r2: f64 = d2_psi_d_r2_2d[(i_z_nearest, i_r_nearest)];
                        let d2_psi_d_z2: f64 = d2_psi_d_z2_2d[(i_z_nearest, i_r_nearest)];
                        let d2_psi_d_r_d_z: f64 = d2_psi_d_rz_2d[(i_z_nearest, i_r_nearest)];
                        let (hessian_determinant, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);

                        stationary_points.push(StationaryPoint {
                            r: stationary_r,
                            z: stationary_z,
                            psi: stationary_psi,
                            hessian_determinant,
                            hessian_trace,
                            i_r_nearest,
                            i_z_nearest,
                            i_r_left,
                            i_r_right,
                            i_z_lower,
                            i_z_upper,
                        });
                        // println!("ax.plot(r[{i_r_left}], z[{i_z_lower}], 'gx')")
                    }
                    Err(_error_string) => {
                        // println!("Warning: bicubic solver failed to converge for cell with corners at (i_r, i_z) = ({}, {}), ({}, {}), ({}, {}), ({}, {}). This cell is a false positive from the sign-change detection, likely due to near-parallel nullclines passing through the cell without actually crossing.", i_r_left, i_z_lower, i_r_right, i_z_lower, i_r_left, i_z_upper, i_r_right, i_z_upper);
                        // println!("ax.plot(r[{i_r_left}], z[{i_z_lower}], 'ro')")
                    }
                }
            }
        }
    }

    // use std::path::Path;
    // npy_reader_and_writer::write_npy_2d(Path::new("/home/peter.buxton/github/gsfit_github/examples/psi_2d.npy"), psi_2d);
    // panic!("stopping for debugging");

    stationary_points
}