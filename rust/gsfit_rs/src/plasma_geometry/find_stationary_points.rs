use super::bicubic_interpolator::{BicubicInterpolator, BicubicStationaryPoint};
use super::calculate_winding_number::calculate_winding_number;
use crate::greens::D2PsiDR2Calculator;
use crate::plasma_geometry::hessian;
use core::f64;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

struct PotentialStationaryPoint {
    i_r: usize,
    i_z: usize,
    bicubic_interpolator: BicubicInterpolator,
}

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

pub fn find_stationary_points(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_br_d_z_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    d2_psi_d_r2_calculator: D2PsiDR2Calculator,
) -> Result<Vec<StationaryPoint>, String> {
    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // Assuming the grid resolution is large enough to see features.
    // The sign-change method is vulnerable to near-tangency contours (false positives).
    // But the sign-change method is not vulnerable to false negatives (i.e. missing a crossing).
    // But the winding number is robust against near-tangency contours.

    // Find candidate cells using sign-change detection
    // For each 2x2 cell, check if both `br` and `bz` change sign across the four corners.
    // A sign change in both fields means the nullclines `br=0` and `bz=0` both pass through the cell,
    // indicating a candidate stationary point.
    // Note: this method can produce false positives (near-parallel nullclines passing through the same cell
    // without crossing), but these are filtered out by the bicubic refinement step below.
    let mut possible_stationary_points: Vec<PotentialStationaryPoint> = Vec::new();
    for i_r in 0..n_r - 1 {
        'cell_loop: for i_z in 0..n_z - 1 {
            let mut n_br_sign_changes: usize = 0;
            let mut n_bz_sign_changes: usize = 0;

            // Lower-left to lower-right
            if br_2d[(i_z, i_r)] * br_2d[(i_z, i_r + 1)] < 0.0 {
                n_br_sign_changes += 1;
            }
            if bz_2d[(i_z, i_r)] * bz_2d[(i_z, i_r + 1)] < 0.0 {
                n_bz_sign_changes += 1;
            }

            // Upper-left to upper-right
            if br_2d[(i_z + 1, i_r)] * br_2d[(i_z + 1, i_r + 1)] < 0.0 {
                n_br_sign_changes += 1;
            }
            if bz_2d[(i_z + 1, i_r)] * bz_2d[(i_z + 1, i_r + 1)] < 0.0 {
                n_bz_sign_changes += 1;
            }

            // Lower-left to upper-left
            if br_2d[(i_z, i_r)] * br_2d[(i_z + 1, i_r)] < 0.0 {
                n_br_sign_changes += 1;
            }
            if bz_2d[(i_z, i_r)] * bz_2d[(i_z + 1, i_r)] < 0.0 {
                n_bz_sign_changes += 1;
            }

            // Lower-right to upper-right
            if br_2d[(i_z, i_r + 1)] * br_2d[(i_z + 1, i_r + 1)] < 0.0 {
                n_br_sign_changes += 1;
            }
            if bz_2d[(i_z, i_r + 1)] * bz_2d[(i_z + 1, i_r + 1)] < 0.0 {
                n_bz_sign_changes += 1;
            }

            // both `br` and `bz` must change sign to have a candidate stationary point
            if !(n_br_sign_changes == 2 && n_bz_sign_changes == 2) {
                // Go to next cell
                continue 'cell_loop;
            }

            let bicubic_interpolator: BicubicInterpolator = setup_bicubic_interpolator(r, z, psi_2d, br_2d, bz_2d, d_bz_d_z_2d, i_r, i_z);

            possible_stationary_points.push(PotentialStationaryPoint {
                i_r,
                i_z,
                bicubic_interpolator,
            });
        }
    }

    // Loop over all possible stationary points and see if we should be considering the neighbours
    let mut additional_stationary_points: Vec<PotentialStationaryPoint> = Vec::new();
    let mut indices_to_remove: Vec<usize> = Vec::new();

    for (idx, possible_stationary_point) in possible_stationary_points.iter().enumerate() {
        // Calculate the winding number to filter out false positives from the sign-change method
        let winding_number: i64 = calculate_winding_number(&possible_stationary_point.bicubic_interpolator);

        // There are occasionally edge cases where the neighbouring cell has the stationary point.
        // This happens when the contour enters and exits through the same edge (which is missed by the sign-change method).
        // So we will check the four neighbouring cells for stationary points. No need to check the diagonal neighbours as entering and exiting through the diagonal point is unlikely.
        if winding_number == 0 {
            // Remember that we will be removing this point
            indices_to_remove.push(idx);

            let i_r: usize = possible_stationary_point.i_r;
            let i_z: usize = possible_stationary_point.i_z;

            // Left neighbour (and check array bounds)
            if i_r > 0 {
                let bicubic_interpolator_left: BicubicInterpolator = setup_bicubic_interpolator(r, z, psi_2d, br_2d, bz_2d, d_bz_d_z_2d, i_r - 1, i_z);
                let winding_number_left: i64 = calculate_winding_number(&bicubic_interpolator_left);
                if winding_number_left != 0 {
                    additional_stationary_points.push(PotentialStationaryPoint {
                        i_r: i_r - 1,
                        i_z,
                        bicubic_interpolator: bicubic_interpolator_left,
                    });
                }
            }

            // Right neighbour (and check array bounds)
            if i_r + 1 < n_r - 1 {
                let bicubic_interpolator_right: BicubicInterpolator = setup_bicubic_interpolator(r, z, psi_2d, br_2d, bz_2d, d_bz_d_z_2d, i_r + 1, i_z);
                let winding_number_right: i64 = calculate_winding_number(&bicubic_interpolator_right);
                if winding_number_right != 0 {
                    additional_stationary_points.push(PotentialStationaryPoint {
                        i_r: i_r + 1,
                        i_z,
                        bicubic_interpolator: bicubic_interpolator_right,
                    });
                }
            }

            // Lower neighbour (and check array bounds)
            if i_z > 0 {
                let bicubic_interpolator_lower: BicubicInterpolator = setup_bicubic_interpolator(r, z, psi_2d, br_2d, bz_2d, d_bz_d_z_2d, i_r, i_z - 1);
                let winding_number_lower: i64 = calculate_winding_number(&bicubic_interpolator_lower);
                if winding_number_lower != 0 {
                    additional_stationary_points.push(PotentialStationaryPoint {
                        i_r,
                        i_z: i_z - 1,
                        bicubic_interpolator: bicubic_interpolator_lower,
                    });
                }
            }

            // Upper neighbour (and check array bounds)
            if i_z + 1 < n_z - 1 {
                let bicubic_interpolator_upper: BicubicInterpolator = setup_bicubic_interpolator(r, z, psi_2d, br_2d, bz_2d, d_bz_d_z_2d, i_r, i_z + 1);
                let winding_number_upper: i64 = calculate_winding_number(&bicubic_interpolator_upper);
                if winding_number_upper != 0 {
                    additional_stationary_points.push(PotentialStationaryPoint {
                        i_r,
                        i_z: i_z + 1,
                        bicubic_interpolator: bicubic_interpolator_upper,
                    });
                }
            }
        }
    }

    // Remove false positives (in reverse, so that we don't mess up the indices before removing)
    for &idx in indices_to_remove.iter().rev() {
        possible_stationary_points.remove(idx);
    }
    possible_stationary_points.extend(additional_stationary_points);

    let mut stationary_points: Vec<StationaryPoint> = Vec::new();
    for possible_stationary_point in &possible_stationary_points {
        // Find the stationary point using the bicubic interpolation
        let stationary_point_or_error: Result<BicubicStationaryPoint, String> = possible_stationary_point.bicubic_interpolator.find_stationary_point(1e-6, 100);

        // Extract the stationary point values
        // If the bicubic solver failed to converge, this cell is a false positive from
        // the sign-change detection (near-parallel nullclines passing through the cell
        // without actually crossing), so skip it.
        match stationary_point_or_error {
            Ok(stationary_point) => {
                // Geometry of the cell corners
                let i_r_left: usize = possible_stationary_point.i_r;
                let i_r_right: usize = possible_stationary_point.i_r + 1;
                let i_z_lower: usize = possible_stationary_point.i_z;
                let i_z_upper: usize = possible_stationary_point.i_z + 1;

                // Extract and store results
                let stationary_r: f64 = r[i_r_left] + stationary_point.x * d_r;
                let stationary_z: f64 = z[i_z_lower] + stationary_point.y * d_z;
                let stationary_psi: f64 = stationary_point.f;

                // Compute nearest grid indices from the refined stationary point position
                let i_r_nearest: usize = ((stationary_r - r[0]) / d_r).round() as usize;
                let i_z_nearest: usize = ((stationary_z - z[0]) / d_z).round() as usize;

                // Calculate the Hessian at the nearest grid point of the cell
                // d^2(psi)/(d_r^2)
                let d2_psi_d_r2: f64 = d2_psi_d_r2_calculator.calculate(i_r_nearest, i_z_nearest);

                // d^2(psi)/(d_z^2)
                let d2_psi_d_z2: f64 = -2.0 * PI * r[i_r_nearest] * d_br_d_z_2d[(i_z_nearest, i_r_nearest)];

                // d^2(psi)/(d_r * d_z)
                let d2_psi_d_r_d_z: f64 = 2.0 * PI * r[i_r_nearest] * d_bz_d_z_2d[(i_z_nearest, i_r_nearest)];

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
            }
            Err(_error_string) => {
                // Do nothing
            }
        }
    }

    // Exit if we haven't found any stationary points
    if stationary_points.is_empty() {
        return Err("find_stationary_points: no intersection between `br` and `bz` contours found".to_string());
    }

    // Return
    Ok(stationary_points)
}

/// Helper function to reduce code duplication
fn setup_bicubic_interpolator(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    i_r: usize,
    i_z: usize,
) -> BicubicInterpolator {
    // Grid variables
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // The cell corners are (i_r, i_z), (i_r+1, i_z), (i_r, i_z+1), (i_r+1, i_z+1)
    let i_r_left: usize = i_r;
    let i_r_right: usize = i_r + 1;
    let i_z_lower: usize = i_z;
    let i_z_upper: usize = i_z + 1;

    // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
    let mut f: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_r: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d_f_d_z: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);
    let mut d2_f_d_r_d_z: Array2<f64> = Array2::from_elem([2, 2], f64::NAN);

    // Function values
    f[(0, 0)] = psi_2d[(i_z_lower, i_r_left)];
    f[(0, 1)] = psi_2d[(i_z_upper, i_r_left)];
    f[(1, 0)] = psi_2d[(i_z_lower, i_r_right)];
    f[(1, 1)] = psi_2d[(i_z_upper, i_r_right)];

    // d(psi)/d(r)
    // bz = 1 / (2.0 * PI * r) * d_psi_d_r
    d_f_d_r[(0, 0)] = bz_2d[(i_z_lower, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d_f_d_r[(0, 1)] = bz_2d[(i_z_upper, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d_f_d_r[(1, 0)] = bz_2d[(i_z_lower, i_r_right)] * (2.0 * PI * r[i_r_right]);
    d_f_d_r[(1, 1)] = bz_2d[(i_z_upper, i_r_right)] * (2.0 * PI * r[i_r_right]);

    // d(psi)/d(z)
    // br = - 1 / (2.0 * PI * r) * d_psi_d_z
    d_f_d_z[(0, 0)] = -br_2d[(i_z_lower, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d_f_d_z[(0, 1)] = -br_2d[(i_z_upper, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d_f_d_z[(1, 0)] = -br_2d[(i_z_lower, i_r_right)] * (2.0 * PI * r[i_r_right]);
    d_f_d_z[(1, 1)] = -br_2d[(i_z_upper, i_r_right)] * (2.0 * PI * r[i_r_right]);

    // d^2(psi)/(d(r)*d(z))
    // d_bz_d_z = 1 / (2 * PI * r) * d2_psi_dr_dz
    // TODO: d_bz_d_z_2d has a delta_z correction missing!  <-- I think I have fixed this in `gs_solution`
    d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_lower, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_upper, i_r_left)] * (2.0 * PI * r[i_r_left]);
    d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_lower, i_r_right)] * (2.0 * PI * r[i_r_right]);
    d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_upper, i_r_right)] * (2.0 * PI * r[i_r_right]);

    // Create a bicubic interpolator
    let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

    bicubic_interpolator
}

// TODO: Add a test for this function. shot=12050, time=131ms failed with the sign-change method, but succeeded with flood fill.

/// In this test the `bz=0` contour enters and exits through the same cell edge
///
/// See the Jupyter notebook for a plot detailing the test
/// `rust/gsfit_rs/test_data/plasma_geometry/find_stationary_points/test_find_stationary_points.ipynb`
#[test]
fn test_find_stationary_points() {
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    let n_r: usize = 6;
    let n_z: usize = 4;

    let r: Array1<f64> = Array1::linspace(0.01, 1.01, n_r);
    let z: Array1<f64> = Array1::linspace(-1.0, 1.0, n_z);

    let mut psi_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut br_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut bz_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_br_d_z_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let mut d_bz_d_z_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);

    let vertical_curvature: f64 = 0.35;

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let r_center: f64 = 0.43 - vertical_curvature * z[i_z].powi(2);

            // psi = -(r - r_center)^2 - (z + 0.025)^2
            psi_2d[(i_z, i_r)] = -(r[i_r] - r_center).powi(2) - (z[i_z] + 25e-3).powi(2);

            // d_psi_d_r = -2 * (r - r_center)
            let d_psi_d_r: f64 = -2.0 * (r[i_r] - r_center);

            // d_psi_d_z = -2 * (r - r_center) * 2 * vertical_curvature * z - 2 * (z + 0.025)
            let d_psi_d_z: f64 = -2.0 * (r[i_r] - r_center) * 2.0 * vertical_curvature * z[i_z] - 2.0 * (z[i_z] + 25e-3);

            // bz = d_psi_d_r / (2 * PI * r)
            bz_2d[(i_z, i_r)] = d_psi_d_r / (2.0 * PI * r[i_r]);

            // br = -d_psi_d_z / (2 * PI * r)
            br_2d[(i_z, i_r)] = -d_psi_d_z / (2.0 * PI * r[i_r]);

            // d_bz_d_z = d/dz [d_psi_d_r / (2 * PI * r)]
            d_bz_d_z_2d[(i_z, i_r)] = -4.0 * vertical_curvature * z[i_z] / (2.0 * PI * r[i_r]);

            // d_br_d_z = d/dz [-d_psi_d_z / (2 * PI * r)]
            let d2_psi_d_z2: f64 = -4.0 * vertical_curvature * (r[i_r] - r_center) - 8.0 * vertical_curvature.powi(2) * z[i_z].powi(2) - 2.0;
            d_br_d_z_2d[(i_z, i_r)] = -d2_psi_d_z2 / (2.0 * PI * r[i_r]);
        }
    }

    // The d2_psi_d_r2_calculator is initialised with NaN-filled arrays because the
    // Hessian values are not being tested here. No coils or passives are included.
    let n_coils: usize = 0;
    let n_passives: usize = 0;
    let g_d2_psi_d_r2_coils: Array3<f64> = Array3::from_elem([n_z, n_r, n_coils], f64::NAN);
    let pf_coil_currents: Array1<f64> = Array1::from_elem(n_coils, f64::NAN);
    let g_d2_psi_d_r2_passives: Array2<f64> = Array2::from_elem([n_z * n_r, n_passives], f64::NAN);
    let passive_dof_values: Array1<f64> = Array1::from_elem(n_passives, f64::NAN);
    let g_d2_psi_d_r2_plasma: Array2<f64> = Array2::from_elem([n_z * n_r, n_r], f64::NAN);
    let j_2d: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let d_area: f64 = f64::NAN;
    let g_bz_plasma: Array2<f64> = Array2::from_elem([n_z * n_r, n_r], f64::NAN);
    let d_bz_d_z: Array2<f64> = Array2::from_elem([n_z, n_r], f64::NAN);
    let delta_z: f64 = f64::NAN;
    let d2_psi_d_r2_calculator: D2PsiDR2Calculator = D2PsiDR2Calculator::new(
        &g_d2_psi_d_r2_coils,
        &pf_coil_currents,
        &g_d2_psi_d_r2_passives,
        &passive_dof_values,
        &g_d2_psi_d_r2_plasma,
        &j_2d,
        d_area,
        &r,
        &g_bz_plasma,
        &d_bz_d_z,
        delta_z,
    );

    let stationary_points_or_error: Result<Vec<StationaryPoint>, String> =
        find_stationary_points(&r, &z, &psi_2d, &br_2d, &bz_2d, &d_br_d_z_2d, &d_bz_d_z_2d, d2_psi_d_r2_calculator);
    let stationary_points: Vec<StationaryPoint> = stationary_points_or_error.expect("test_find_stationary_points: failed to find any stationary points");

    // There should be only one stationary point
    assert_eq!(stationary_points.len(), 1);
    let stationary_point: StationaryPoint = stationary_points[0];

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
    assert_abs_diff_eq!(expected_stationary_point_z, stationary_point.z, epsilon = d_r / 10.0);
    assert_abs_diff_eq!(expected_stationary_point_r, stationary_point.r, epsilon = d_z / 10.0);
    assert_abs_diff_eq!(expected_stationary_point_psi, stationary_point.psi, epsilon = max_delta_psi / 10.0);
}
