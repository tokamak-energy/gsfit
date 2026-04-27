use super::bicubic_interpolator::{BicubicInterpolator, BicubicStationaryPoint};
use super::calculate_winding_number::calculate_winding_number;
use crate::greens::D2PsiDR2Calculator;
use crate::plasma_geometry::hessian;
use core::f64;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use std::f64::consts::PI;
use super::cubic_interpolation::cubic_interpolation_v2;

#[derive(Debug, Clone, Copy)]
pub struct StationaryPoint {
    pub r: f64,
    pub z: f64,
    pub psi: f64,
    pub hessian_determinant: f64,
    pub hessian_trace: f64,
    pub i_r_nearest: usize,
    pub i_z_nearest: usize,
    pub i_r_left: usize,
    pub i_r_right: usize,
    pub i_z_lower: usize,
    pub i_z_upper: usize,
}

#[derive(Clone)]
struct Coordinate {
    r: f64,
    z: f64,
}

pub fn find_stationary_points_using_full_quadrant_method(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_br_d_z_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    d2_psi_d_r2_calculator: D2PsiDR2Calculator,
) -> Result<Vec<StationaryPoint>, String> {

    let n_r: usize = r.len();
    let n_z: usize = z.len();
    
    // key: (i_r_from, i_z_from, i_r_to, i_z_to)
    // value: number of crossings
    let mut all_br_crossings: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();
    let mut all_bz_crossings: HashMap<(usize, usize, usize, usize), Vec<Coordinate>> = HashMap::new();

    // Horizontal march: (i_r, i_z) → (i_r+1, i_z)
    for i_z in 0..n_z {
        for i_r in 0..n_r - 1 {
            let br_crossings: Vec<f64> = cubic_interpolation_v2(
                r[i_r],
                br_2d[(i_z, i_r)],
                d_br_d_z_2d[(i_z, i_r)],
                r[i_r + 1],
                br_2d[(i_z, i_r + 1)],
                d_br_d_z_2d[(i_z, i_r + 1)],
                0.0,
            );
            let mut crossing_coordinates: Vec<Coordinate> = Vec::with_capacity(br_crossings.len());
            for &r_cross in &br_crossings {
                crossing_coordinates.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            all_br_crossings.insert((i_r, i_z, i_r + 1, i_z), crossing_coordinates);

            let bz_crossings: Vec<f64> = cubic_interpolation_v2(
                r[i_r],
                bz_2d[(i_z, i_r)],
                d_bz_d_z_2d[(i_z, i_r)],
                r[i_r + 1],
                bz_2d[(i_z, i_r + 1)],
                d_bz_d_z_2d[(i_z, i_r + 1)],
                0.0,
            );
            let mut crossing_coordinates: Vec<Coordinate> = Vec::with_capacity(bz_crossings.len());
            for &r_cross in &bz_crossings {
                crossing_coordinates.push(Coordinate { r: r_cross, z: z[i_z] });
            }
            all_bz_crossings.insert((i_r, i_z, i_r + 1, i_z), crossing_coordinates);
        }
    }

    // Vertical march: (i_r, i_z) → (i_r, i_z+1)
    for i_r in 0..n_r {
        for i_z in 0..n_z - 1 {
            let br_crossings: Vec<f64> = cubic_interpolation_v2(
                z[i_z],
                br_2d[(i_z, i_r)],
                d_br_d_z_2d[(i_z, i_r)],
                z[i_z + 1],
                br_2d[(i_z + 1, i_r)],
                d_br_d_z_2d[(i_z + 1, i_r)],
                0.0,
            );
            let mut crossing_coordinates: Vec<Coordinate> = Vec::with_capacity(br_crossings.len());
            for &z_cross in &br_crossings {
                crossing_coordinates.push(Coordinate { r: r[i_r], z: z_cross });
            }
            all_br_crossings.insert((i_r, i_z, i_r, i_z + 1), crossing_coordinates);

            let bz_crossings: Vec<f64> = cubic_interpolation_v2(
                z[i_z],
                bz_2d[(i_z, i_r)],
                d_bz_d_z_2d[(i_z, i_r)],
                z[i_z + 1],
                bz_2d[(i_z + 1, i_r)],
                d_bz_d_z_2d[(i_z + 1, i_r)],
                0.0,
            );
            let mut crossing_coordinates: Vec<Coordinate> = Vec::with_capacity(bz_crossings.len());
            for &z_cross in &bz_crossings {
                crossing_coordinates.push(Coordinate { r: r[i_r], z: z_cross });
            }
            all_bz_crossings.insert((i_r, i_z, i_r, i_z + 1), crossing_coordinates);
        }
    }

    // Loop over each cell
    let mut stationary_points: Vec<StationaryPoint> = Vec::new();
    for i_r in 0..n_r - 1 {
        for i_z in 0..n_z - 1 {
            // Consider the cell defined by corners:
            // * (i_r, i_z) --> (left, bottom)
            // * (i_r+1, i_z) --> (right, bottom)
            // * (i_r, i_z+1) --> (left, top)
            // * (i_r+1, i_z+1) --> (right, top)

            // Collect boundary points (corners and all br=0, bz=0 crossings) in order
            let mut boundary_points: Vec<Coordinate> = Vec::new();
            // Bottom edge: (i_r, i_z) -> (i_r+1, i_z)
            boundary_points.push(Coordinate { r: r[i_r], z: z[i_z] });
            if let Some(crossings) = all_br_crossings.get(&(i_r, i_z, i_r + 1, i_z)) {
                boundary_points.extend_from_slice(crossings);
            }
            if let Some(crossings) = all_bz_crossings.get(&(i_r, i_z, i_r + 1, i_z)) {
                boundary_points.extend_from_slice(crossings);
            }
            boundary_points.push(Coordinate { r: r[i_r + 1], z: z[i_z] });
            // Right edge: (i_r+1, i_z) -> (i_r+1, i_z+1)
            if let Some(crossings) = all_br_crossings.get(&(i_r + 1, i_z, i_r + 1, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            if let Some(crossings) = all_bz_crossings.get(&(i_r + 1, i_z, i_r + 1, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            boundary_points.push(Coordinate { r: r[i_r + 1], z: z[i_z + 1] });
            // Top edge: (i_r+1, i_z+1) -> (i_r, i_z+1)
            if let Some(crossings) = all_br_crossings.get(&(i_r, i_z + 1, i_r + 1, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            if let Some(crossings) = all_bz_crossings.get(&(i_r, i_z + 1, i_r + 1, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            boundary_points.push(Coordinate { r: r[i_r], z: z[i_z + 1] });
            // Left edge: (i_r, i_z+1) -> (i_r, i_z)
            if let Some(crossings) = all_br_crossings.get(&(i_r, i_z, i_r, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            if let Some(crossings) = all_bz_crossings.get(&(i_r, i_z, i_r, i_z + 1)) {
                boundary_points.extend_from_slice(crossings);
            }
            // Remove duplicates (corners may be repeated)
            boundary_points.sort_by(|a, b| {
                a.r.partial_cmp(&b.r).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal))
            });
            boundary_points.dedup_by(|a, b| (a.r - b.r).abs() < 1e-12 && (a.z - b.z).abs() < 1e-12);

            // For each boundary point, compute gradient vector (bz, -br)
            let mut quadrants: Vec<u8> = Vec::new();
            let mut grads: Vec<(f64, f64)> = Vec::new();
            for pt in &boundary_points {
                // Find nearest grid indices
                let i_r_nearest = (r - pt.r).abs().argmin().unwrap();
                let i_z_nearest = (z - pt.z).abs().argmin().unwrap();
                let bz_val = bz_2d[(i_z_nearest, i_r_nearest)];
                let br_val = br_2d[(i_z_nearest, i_r_nearest)];
                let gx = bz_val;
                let gy = -br_val;
                grads.push((gx, gy));
                // Quadrant assignment
                let quadrant = match (gx > 0.0, gy > 0.0) {
                    (true, true) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                    (true, false) => 4,
                };
                quadrants.push(quadrant);
            }
            // Compute winding number using quadrant difference method
            let mut winding = 0i32;
            let n = quadrants.len();
            for i in 0..n {
                let q1 = quadrants[i] as i32;
                let q2 = quadrants[(i + 1) % n] as i32;
                let mut dq = q2 - q1;
                if dq == 3 { dq = -1; }
                if dq == -3 { dq = 1; }
                if dq.abs() == 2 {
                    // Diagonal: use cross product to determine sign
                    let (gx1, gy1) = grads[i];
                    let (gx2, gy2) = grads[(i + 1) % n];
                    let cross = gx1 * gy2 - gy1 * gx2;
                    dq = if cross > 0.0 { 2 } else { -2 };
                }
                winding += dq;
            }
            let winding_number = winding / 4;
            if winding_number.abs() == 1 {
                // Candidate cell: use bicubic interpolation to refine
                let bicubic_interpolator = BicubicInterpolator::new(
                    r[1] - r[0],
                    z[1] - z[0],
                    &{
                        let mut f = Array2::from_elem([2, 2], f64::NAN);
                        f[(0, 0)] = psi_2d[(i_z, i_r)];
                        f[(0, 1)] = psi_2d[(i_z + 1, i_r)];
                        f[(1, 0)] = psi_2d[(i_z, i_r + 1)];
                        f[(1, 1)] = psi_2d[(i_z + 1, i_r + 1)];
                        f
                    },
                    &{
                        let mut d_f_d_r = Array2::from_elem([2, 2], f64::NAN);
                        d_f_d_r[(0, 0)] = bz_2d[(i_z, i_r)] * (2.0 * PI * r[i_r]);
                        d_f_d_r[(0, 1)] = bz_2d[(i_z + 1, i_r)] * (2.0 * PI * r[i_r]);
                        d_f_d_r[(1, 0)] = bz_2d[(i_z, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d_f_d_r[(1, 1)] = bz_2d[(i_z + 1, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d_f_d_r
                    },
                    &{
                        let mut d_f_d_z = Array2::from_elem([2, 2], f64::NAN);
                        d_f_d_z[(0, 0)] = -br_2d[(i_z, i_r)] * (2.0 * PI * r[i_r]);
                        d_f_d_z[(0, 1)] = -br_2d[(i_z + 1, i_r)] * (2.0 * PI * r[i_r]);
                        d_f_d_z[(1, 0)] = -br_2d[(i_z, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d_f_d_z[(1, 1)] = -br_2d[(i_z + 1, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d_f_d_z
                    },
                    &{
                        let mut d2_f_d_r_d_z = Array2::from_elem([2, 2], f64::NAN);
                        d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z, i_r)] * (2.0 * PI * r[i_r]);
                        d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z + 1, i_r)] * (2.0 * PI * r[i_r]);
                        d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z + 1, i_r + 1)] * (2.0 * PI * r[i_r + 1]);
                        d2_f_d_r_d_z
                    },
                );
                let stationary_point_or_error = bicubic_interpolator.find_stationary_point(1e-6, 100);
                if let Ok(stationary_point) = stationary_point_or_error {
                    let i_r_left = i_r;
                    let i_r_right = i_r + 1;
                    let i_z_lower = i_z;
                    let i_z_upper = i_z + 1;
                    let stationary_r = r[i_r_left] + stationary_point.x * (r[i_r_right] - r[i_r_left]);
                    let stationary_z = z[i_z_lower] + stationary_point.y * (z[i_z_upper] - z[i_z_lower]);
                    let stationary_psi = stationary_point.f;
                    let i_r_nearest = (r - stationary_r).abs().argmin().unwrap();
                    let i_z_nearest = (z - stationary_z).abs().argmin().unwrap();
                    let d2_psi_d_r2 = d2_psi_d_r2_calculator.calculate(i_r_nearest, i_z_nearest);
                    let d2_psi_d_z2 = -2.0 * PI * r[i_r_nearest] * d_br_d_z_2d[(i_z_nearest, i_r_nearest)];
                    let d2_psi_d_r_d_z = 2.0 * PI * r[i_r_nearest] * d_bz_d_z_2d[(i_z_nearest, i_r_nearest)];
                    let (hessian_determinant, hessian_trace) = hessian(d2_psi_d_r2, d2_psi_d_z2, d2_psi_d_r_d_z);
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
                }
            }
        }
    }
    if stationary_points.is_empty() {
        return Err("find_stationary_points_using_full_quadrant_method: no intersection between br and bz contours found".to_string());
    }
    Ok(stationary_points)
}