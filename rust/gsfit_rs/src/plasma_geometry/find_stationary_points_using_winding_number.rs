use super::StationaryPoint;
use super::bicubic_interpolator::{BicubicInterpolator, BicubicStationaryPoint};
use super::cubic_interpolation::cubic_interpolation_v2;
use crate::plasma_geometry::hessian;
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use ndarray::{Dim, SliceInfo, SliceInfoElem};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

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
/// `br = - 1 / (2.0 * pi) d(psi)/d(z)` and `bz = 1 / (2.0 * pi) * d(psi)/d(r)`, so `d(psi)/d(r) = 0` and `d(psi)/d(z) = 0`
/// are equivalent to `bz = 0` and `br = 0`. But to avoid the confusion with the minus sign in `br`, we only deal with
/// gradients of `psi`.
///
/// The simplest way to calculate the winding number involves sampling the function along the cell edges and calculating the angle using `atan2`.
/// However, `atan2` is an expensive operation, so we instead count the number of rotations through the d(ψ)/d(r) d(ψ)/d(z) quadrants.
///
/// The Q1, Q2, Q3, and Q4 quadrant labels are standard definitions.
/// Typically, the axes are not part of the quadrants labels.
/// But for our purposes, we need to assign a quadrant label to the axes:
///      d(ψ)/d(z)                              d(ψ)/d(z)                           d(ψ)/d(z)
///          ▲                                     ▲                                    ▲        Special case:
///     Q2   │   Q1                                Q2                                   │        (0.0, 0.0) is in Q1
///   (-,+)  │  (+,+)                              │                                    │ Q1
/// ─────────┼─────────► d(ψ)/d(r)        ────Q3───┼───Q1────► d(ψ)/d(r)        ────────•────────► d(ψ)/d(r)
///     Q3   │   Q4                                │                                    │
///   (-,-)  │  (+,-)                              Q4                                   │
///          │                                     |                                    │
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
///      d(ψ)/d(z)
///          ▲
///     2 •  │  • 1
///       6• │ •5
/// ─────────┼─────────► d(ψ)/d(r)
///       7• │ •8
///     3 •  │  • 4
///          │
///
/// A subtle failure mode is when there is both an o-point and an x-point in the same cell,
/// `winding_number = (+1) + (-1) = 0`, which would incorrectly indicate no stationary points.
pub fn find_stationary_points_using_winding_number(
    r: ArrayView1<f64>,
    z: ArrayView1<f64>,
    psi_2d: ArrayView2<f64>,
    d_psi_d_r_2d: ArrayView2<f64>,
    d_psi_d_z_2d: ArrayView2<f64>,
    d2_psi_d_r2_2d: ArrayView2<f64>,
    d2_psi_d_rz_2d: ArrayView2<f64>,
    d2_psi_d_z2_2d: ArrayView2<f64>,
) -> Vec<StationaryPoint> {
    // Empty stationary_points, ready to return if no stationary points found
    let mut stationary_points: Vec<StationaryPoint> = Vec::new();

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // value: vector of crossing coordinates
    // `(r,z)` location where `d(psi)/d(z)=0`, along all edges
    let mut d_psi_d_z_zero_coordinates: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();
    // `(r,z)` location where `d(psi)/d(r)=0`, along all edges
    let mut d_psi_d_r_zero_coordinates: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();

    // Horizontal march: (i_r_left, i_z) → (i_r_right, i_z)
    // We don't need to do (i_r_left, i_z+1) → (i_r_right, i_z+1) as this will be covered by the next horizontal march
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            let i_r_left: usize = i_r;
            let i_r_right: usize = i_r + 1;

            let d_psi_d_z_zero_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                r[i_r_left],
                d_psi_d_z_2d[(i_z, i_r_left)],
                d2_psi_d_rz_2d[(i_z, i_r_left)],
                r[i_r_right],
                d_psi_d_z_2d[(i_z, i_r_right)],
                d2_psi_d_rz_2d[(i_z, i_r_right)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(d_psi_d_z_zero_crossings_this_edge.len());
            for &r_cross in &d_psi_d_z_zero_crossings_this_edge {
                crossing_coordinates_this_edge.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            d_psi_d_z_zero_coordinates.insert((i_r_left, i_z, i_r_right, i_z), crossing_coordinates_this_edge);

            let d_psi_d_r_zero_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                r[i_r_left],
                d_psi_d_r_2d[(i_z, i_r_left)],
                d2_psi_d_r2_2d[(i_z, i_r_left)],
                r[i_r_right],
                d_psi_d_r_2d[(i_z, i_r_right)],
                d2_psi_d_r2_2d[(i_z, i_r_right)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(d_psi_d_r_zero_crossings_this_edge.len());
            for &r_cross in &d_psi_d_r_zero_crossings_this_edge {
                crossing_coordinates_this_edge.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            d_psi_d_r_zero_coordinates.insert((i_r_left, i_z, i_r_right, i_z), crossing_coordinates_this_edge);
        }
    }

    // Vertical march: (i_r, i_z_lower) → (i_r, i_z_upper)
    // We don't need to do (i_r+1, i_z_lower) → (i_r+1, i_z_upper) as this will be covered by the next vertical march
    //
    // To prevent double counting, we discard crossings which are exactly at the grid points,
    // since they will have already been counted in the horizontal march.
    // In practice, this is fairly unlikely.
    // But it could happen in a perfect double null from synthetic data.
    //
    // Note: having `i_r` as the inner loop is optimal order to prevent cache thrashing
    for i_z in 0..n_z - 1 {
        let i_z_lower: usize = i_z;
        let i_z_upper: usize = i_z + 1;
        let z_start: f64 = z[i_z_lower];
        let z_end: f64 = z[i_z_upper];
        let endpoint_tol: f64 = 1e-12 * (z_end - z_start);
        for i_r in 0..n_r {
            let br_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                z_start,
                d_psi_d_z_2d[(i_z_lower, i_r)],
                d2_psi_d_z2_2d[(i_z_lower, i_r)],
                z_end,
                d_psi_d_z_2d[(i_z_upper, i_r)],
                d2_psi_d_z2_2d[(i_z_upper, i_r)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(br_crossings_this_edge.len());
            for &z_cross in &br_crossings_this_edge {
                if z_cross - z_start > endpoint_tol && z_end - z_cross > endpoint_tol {
                    crossing_coordinates_this_edge.push(Coordinate { r: r[i_r], z: z_cross });
                }
            }
            d_psi_d_z_zero_coordinates.insert((i_r, i_z_lower, i_r, i_z_upper), crossing_coordinates_this_edge);

            let bz_crossings_this_edge: Vec<f64> = cubic_interpolation_v2(
                z_start,
                d_psi_d_r_2d[(i_z_lower, i_r)],
                d2_psi_d_rz_2d[(i_z_lower, i_r)],
                z_end,
                d_psi_d_r_2d[(i_z_upper, i_r)],
                d2_psi_d_rz_2d[(i_z_upper, i_r)],
                0.0,
            );
            let mut crossing_coordinates_this_edge: Vec<Coordinate> = Vec::with_capacity(bz_crossings_this_edge.len());
            for &z_cross in &bz_crossings_this_edge {
                if z_cross - z[i_z_lower] > endpoint_tol && z[i_z_upper] - z_cross > endpoint_tol {
                    crossing_coordinates_this_edge.push(Coordinate { r: r[i_r], z: z_cross });
                }
            }
            d_psi_d_r_zero_coordinates.insert((i_r, i_z_lower, i_r, i_z_upper), crossing_coordinates_this_edge);
        }
    }

    if d_psi_d_z_zero_coordinates.len() == 0 || d_psi_d_r_zero_coordinates.len() == 0 {
        // No zero-crossings found, so no stationary points possible
        return stationary_points;
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
                &d_psi_d_z_zero_coordinates[&(i_r_left, i_z_lower, i_r_right, i_z_lower)],
                &d_psi_d_r_zero_coordinates[&(i_r_left, i_z_lower, i_r_right, i_z_lower)],
                r[i_r],
                r[i_r + 1],
                true,
            );
            for event in bottom_events {
                perimeter_events.push(event);
            }

            // Right edge: BR → TR (z increasing at r = r[i_r+1]).
            let right_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &d_psi_d_z_zero_coordinates[&(i_r_right, i_z_lower, i_r_right, i_z_upper)],
                &d_psi_d_r_zero_coordinates[&(i_r_right, i_z_lower, i_r_right, i_z_upper)],
                z[i_z],
                z[i_z + 1],
                false,
            );
            for event in right_events {
                perimeter_events.push(event);
            }

            // Top edge: TR → TL (r decreasing at z = z[i_z+1]).
            let top_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &d_psi_d_z_zero_coordinates[&(i_r_left, i_z_upper, i_r_right, i_z_upper)],
                &d_psi_d_r_zero_coordinates[&(i_r_left, i_z_upper, i_r_right, i_z_upper)],
                r[i_r + 1],
                r[i_r],
                true,
            );
            for event in top_events {
                perimeter_events.push(event);
            }

            // Left edge: TL → BL (z decreasing at r = r[i_r]).
            let left_events: Vec<CrossingKind> = combine_and_order_edge_events(
                &d_psi_d_z_zero_coordinates[&(i_r_left, i_z_lower, i_r_left, i_z_upper)],
                &d_psi_d_r_zero_coordinates[&(i_r_left, i_z_lower, i_r_left, i_z_upper)],
                z[i_z_upper],
                z[i_z_lower],
                false,
            );
            for event in left_events {
                perimeter_events.push(event);
            }

            // Walk the perimeter event-by-event, tracking which quadrant of we are in.
            // Each event changes `total_quarter_turns` by +/-1.
            //
            // Seed the walk at the bottom-left grid point.
            let mut sign_d_psi_d_z: i8 = sign_with_tiebreak(d_psi_d_z_2d[(i_z_lower, i_r_left)]); // if `d_psi_d_z_2d[(i_z, i_r)] == 0.0`, then `sign_br = 1`
            let mut sign_d_psi_d_r: i8 = sign_with_tiebreak(d_psi_d_r_2d[(i_z_lower, i_r_left)]);
            let mut prev_q: i8 = classify_quadrant(sign_d_psi_d_z, sign_d_psi_d_r);
            let mut total_quarter_turns: i8 = 0;

            for event in perimeter_events.clone() {
                // Flip the sign of whichever component just crossed zero.
                match event {
                    CrossingKind::BrZero => sign_d_psi_d_z = -sign_d_psi_d_z,
                    CrossingKind::BzZero => sign_d_psi_d_r = -sign_d_psi_d_r,
                }

                // Step the quadrant tracker: +1 for an adjacent CCW step, -1 for an adjacent CW step.
                let new_q: i8 = classify_quadrant(sign_d_psi_d_z, sign_d_psi_d_r);
                let quadrant_step: i8 = (new_q - prev_q).rem_euclid(4) as i8;
                match quadrant_step {
                    0 => {} // same quadrant; no change in `total_quarter_turns`
                    1 => total_quarter_turns += 1,
                    2 => {
                        // println!("Warning: diagonal jump in (Br, Bz) quadrants. This should not happen with proper sampling."),
                    }
                    3 => total_quarter_turns -= 1,
                    _ => {
                        // println!("Error: invalid `quadrant_step={}`. This should never happen?", quadrant_step);
                    }
                }
                prev_q = new_q;
            }

            let winding_number: i8 = total_quarter_turns / 4;

            if winding_number != 0 {
                // Create a slice that covers the 4 corner points
                let slice_cell_perimeter: SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> = s![i_z_lower..=i_z_upper, i_r_left..=i_r_right];

                // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis.
                // TODO:`psi_2d` is stored as (i_z, i_r), but `BicubicInterpolator` expects `f[(i_r, i_z)]`.
                // TODO:So we need to transpose the arrays!
                // TODO: It is better to use ArrayView2<f64> data types in `BicubicInterpolator::new`, rather than doing a copy
                let f: Array2<f64> = psi_2d.slice(slice_cell_perimeter).t().to_owned();
                let d_f_d_r: Array2<f64> = d_psi_d_r_2d.slice(slice_cell_perimeter).t().to_owned();
                let d_f_d_z: Array2<f64> = d_psi_d_z_2d.slice(slice_cell_perimeter).t().to_owned();
                let d2_f_d_r_d_z: Array2<f64> = d2_psi_d_rz_2d.slice(slice_cell_perimeter).t().to_owned();

                // Create a bicubic interpolator
                let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, f.view(), d_f_d_r.view(), d_f_d_z.view(), d2_f_d_r_d_z.view());

                // Find the stationary point using the bicubic interpolation
                let stationary_point_or_error: Result<BicubicStationaryPoint, String> = bicubic_interpolator.find_stationary_point(1e-6, 100);

                // Extract the stationary point values
                // If the bicubic solver failed to converge, this cell is a false positive, so skip it.
                match stationary_point_or_error {
                    Ok(stationary_point) => {
                        // Extract and store results
                        let stationary_r: f64 = r[i_r_left] + stationary_point.x * d_r;
                        let stationary_z: f64 = z[i_z_lower] + stationary_point.y * d_z;
                        let stationary_psi: f64 = stationary_point.f;

                        // Compute nearest grid indices from the refined stationary point position
                        let i_r_nearest: usize = (r.to_owned() - stationary_r).abs().argmin().unwrap();
                        let i_z_nearest: usize = (z.to_owned() - stationary_z).abs().argmin().unwrap();

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
                        // Do nothing

                        // println!("Warning: bicubic solver failed to converge for cell with corners at (i_r, i_z) = ({}, {}), ({}, {}), ({}, {}), ({}, {}). This cell is a false positive from the sign-change detection, likely due to near-parallel nullclines passing through the cell without actually crossing.", i_r_left, i_z_lower, i_r_right, i_z_lower, i_r_left, i_z_upper, i_r_right, i_z_upper);
                        // println!("winding_number = {winding_number}");
                        // println!("ax.plot(r[{i_r_left}], z[{i_z_lower}], 'ro')");

                        // println!("perimeter_events = {:?}", perimeter_events);

                        // // Sample the bicubic interpolation at various points in 2d
                        // let n_r_sample: usize = 70;
                        // let n_z_sample: usize = 65;
                        // use ndarray::Array1;
                        // let r_sample: Array1<f64> = Array1::linspace(r[i_r_left], r[i_r_right], n_r_sample);
                        // let z_sample: Array1<f64> = Array1::linspace(z[i_z_lower], z[i_z_upper], n_z_sample);
                        // let mut psi_2d: Array2<f64> = Array2::from_elem([n_z_sample, n_r_sample], f64::NAN);
                        // for i_z_sample in 0..n_z_sample {
                        //     for i_r_sample in 0..n_r_sample {
                        //         let r_s: f64 = (r_sample[i_r_sample] - r[i_r_left]) / d_r; // to be between 0.0 and 1.0
                        //         let z_s: f64 = (z_sample[i_z_sample] - z[i_z_lower]) / d_z; // to be between 0.0 and 1.0
                        //         psi_2d[(i_z_sample, i_r_sample)] = bicubic_interpolator.interpolate(r_s, z_s);
                        //     }
                        // }
                        // println!("psi_2d = {:#?}", psi_2d);

                        // use std::path::Path;
                        // npy_reader_and_writer::write_npy_2d(Path::new("/home/peter.buxton/github/gsfit_github/examples/psi_2d.npy"), &psi_2d.to_owned());
                        // panic!("stopping for debugging");
                    }
                }
            }
        }
    }

    stationary_points
}

#[derive(Clone)]
struct Coordinate {
    r: f64,
    z: f64,
}

#[derive(Copy, Clone, Debug)]
enum CrossingKind {
    BrZero,
    BzZero,
}

/// Quadrant of (-br, bz). Q1: (+,+), Q2: (-,+), Q3: (-,-), Q4: (+,-).
/// CCW traversal of the (-br, bz) plane visits Q1 → Q2 → Q3 → Q4 → Q1.
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
    d_psi_d_z_zero_crossings_this_edge: &[Coordinate],
    d_psi_d_r_zero_crossings_this_edge: &[Coordinate],
    edge_start: f64,
    edge_end: f64,
    use_r_axis: bool,
) -> Vec<CrossingKind> {
    let endpoint_tol: f64 = 1e-12 * (edge_end - edge_start).abs();
    let mut events: Vec<(f64, CrossingKind)> = Vec::with_capacity(d_psi_d_z_zero_crossings_this_edge.len() + d_psi_d_r_zero_crossings_this_edge.len());
    for c in d_psi_d_z_zero_crossings_this_edge {
        let pos: f64 = if use_r_axis { c.r } else { c.z };
        if (pos - edge_start).abs() > endpoint_tol && (edge_end - pos).abs() > endpoint_tol {
            events.push((pos, CrossingKind::BrZero));
        }
    }
    for c in d_psi_d_r_zero_crossings_this_edge {
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

/// In this test the `d(psi)/d(r)=0` contour enters and exits through the same cell edge
///
/// See the Jupyter notebook for a plot detailing the test
/// `rust/gsfit_rs/test_assets/plasma_geometry/find_stationary_points/test_1_find_stationary_points_using_winding_number_with_contour_entering_and_exiting_same_cell_edge.ipynb`
#[test]
fn test_1_find_stationary_points_using_winding_number_with_contour_entering_and_exiting_same_cell_edge() {
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    let n_r: usize = 6;
    let n_z: usize = 4;

    let r: Array1<f64> = Array1::linspace(0.01, 1.01, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    let mut psi_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_r2_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_rz_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_z2_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);

    let vertical_curvature: f64 = 0.35;

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let r_center: f64 = 0.43 - vertical_curvature * z[i_z].powi(2);
            let delta_r: f64 = r[i_r] - r_center;
            let delta_z: f64 = z[i_z] + 25e-3;

            psi_2d[(i_z, i_r)] = -delta_r.powi(2) - delta_z.powi(2);
            d_psi_d_r_2d[(i_z, i_r)] = -2.0 * delta_r;
            d_psi_d_z_2d[(i_z, i_r)] = -4.0 * vertical_curvature * z[i_z] * delta_r - 2.0 * delta_z;
            d2_psi_d_r2_2d[(i_z, i_r)] = -2.0;
            d2_psi_d_rz_2d[(i_z, i_r)] = -4.0 * vertical_curvature * z[i_z];
            d2_psi_d_z2_2d[(i_z, i_r)] = -4.0 * vertical_curvature * delta_r - 8.0 * vertical_curvature.powi(2) * z[i_z].powi(2) - 2.0;
        }
    }

    let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
        r.view(),
        z.view(),
        psi_2d.view(),
        d_psi_d_r_2d.view(),
        d_psi_d_z_2d.view(),
        d2_psi_d_r2_2d.view(),
        d2_psi_d_rz_2d.view(),
        d2_psi_d_z2_2d.view(),
    );

    // There should be only one stationary point
    assert_eq!(stationary_points.len(), 1);
    let stationary_point: StationaryPoint = stationary_points[0];

    // Expected stationary point value and location
    let expected_stationary_point_psi: f64 = 0.0;
    let expected_stationary_point_z: f64 = -25e-3;
    let expected_stationary_point_r: f64 = 0.43 - vertical_curvature * expected_stationary_point_z.powi(2);

    // Find the scale which we need to use for the epsilon values
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];
    let mut max_delta_psi: f64 = 0.0;
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            if i_r + 1 < n_r {
                let delta_psi: f64 = (psi_2d[(i_z, i_r + 1)] - psi_2d[(i_z, i_r)]).abs();
                if delta_psi > max_delta_psi {
                    max_delta_psi = delta_psi;
                }
            }
            if i_z + 1 < n_z {
                let delta_psi: f64 = (psi_2d[(i_z + 1, i_r)] - psi_2d[(i_z, i_r)]).abs();
                if delta_psi > max_delta_psi {
                    max_delta_psi = delta_psi;
                }
            }
        }
    }

    // Check the stationary point values against the expected values
    assert_abs_diff_eq!(expected_stationary_point_r, stationary_point.r, epsilon = d_r / 10.0);
    assert_abs_diff_eq!(expected_stationary_point_z, stationary_point.z, epsilon = d_z / 10.0);
    assert_abs_diff_eq!(expected_stationary_point_psi, stationary_point.psi, epsilon = max_delta_psi / 10.0);

    // Expected stationary point type:
    // * extremum: `hessian_determinant > 0`
    // * maximum: `hessian_trace < 0`
    assert!(stationary_point.hessian_determinant > 0.0);
    assert!(stationary_point.hessian_trace < 0.0);
}

/// In this test the stationary point lies exactly on the cell edge
///
/// See the Jupyter notebook for a plot detailing the test
/// `rust/gsfit_rs/test_assets/plasma_geometry/find_stationary_points/test_2_find_stationary_points_using_winding_number_with_stationary_point_at_cell_edge.ipynb`
#[test]
fn test_2_find_stationary_points_using_winding_number_with_stationary_point_at_cell_edge() {
    // use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    let n_r: usize = 7;
    let n_z: usize = 4;

    let vertical_curvature: f64 = 0.35;
    let expected_stationary_point_z: f64 = -25e-3;
    let expected_stationary_point_r: f64 = 0.43 - vertical_curvature * expected_stationary_point_z.powi(2);

    let r: Array1<f64> = Array1::linspace(expected_stationary_point_r - 0.35, expected_stationary_point_r + 0.35, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    let mut psi_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_r2_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_rz_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d2_psi_d_z2_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let r_center: f64 = 0.43 - vertical_curvature * z[i_z].powi(2);
            let delta_r: f64 = r[i_r] - r_center;
            let delta_z: f64 = z[i_z] + 25e-3;

            psi_2d[(i_z, i_r)] = -delta_r.powi(2) - delta_z.powi(2);
            d_psi_d_r_2d[(i_z, i_r)] = -2.0 * delta_r;
            d_psi_d_z_2d[(i_z, i_r)] = -4.0 * vertical_curvature * z[i_z] * delta_r - 2.0 * delta_z;
            d2_psi_d_r2_2d[(i_z, i_r)] = -2.0;
            d2_psi_d_rz_2d[(i_z, i_r)] = -4.0 * vertical_curvature * z[i_z];
            d2_psi_d_z2_2d[(i_z, i_r)] = -4.0 * vertical_curvature * delta_r - 8.0 * vertical_curvature.powi(2) * z[i_z].powi(2) - 2.0;
        }
    }

    let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
        r.view(),
        z.view(),
        psi_2d.view(),
        d_psi_d_r_2d.view(),
        d_psi_d_z_2d.view(),
        d2_psi_d_r2_2d.view(),
        d2_psi_d_rz_2d.view(),
        d2_psi_d_z2_2d.view(),
    );

    println!("stationary_points = {:#?}", stationary_points);
    assert!(stationary_points.len() == 1);
}
