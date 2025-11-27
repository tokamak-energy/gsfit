use super::BoundaryContour;
use super::bicubic_interpolator::BicubicInterpolator;
use super::flood_fill_mask::flood_fill_mask;
use core::f64;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use super::StationaryPoint;
use super::marching_squares::MarchingContour;
use super::marching_squares::marching_squares;

const PI: f64 = std::f64::consts::PI;

/// Find a viable limit point which can be used to define the plasma boundary
///
///  # Arguments
/// * `r` - R grid points, metre
/// * `z` - Z grid points, metre
/// * `psi_2d` - poloidal flux, shape = (n_z, n_r), weber
/// * `limit_pts_r` - R coordinates of limiter points, metre
/// * `limit_pts_z` - Z coordinates of limiter points, metre
/// * `mag_r_previous` - R coordinate of magnetic axis from previous iteration, metre
/// * `mag_z_previous` - Z coordinate of magnetic axis from previous iteration, metre
/// * `vessel_r` - R coordinates of vessel points, metre
/// * `vessel_z` - Z coordinates of vessel points, metre
/// * `stationary_points` - Vector of `StationaryPoint` objects representing stationary points in psi
///
/// # Returns
/// * `BoundaryContour` - A `BoundaryContour` object representing the plasma boundary
///
pub fn find_viable_limit_point(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    limit_pts_r: &Array1<f64>,  // TODO: might be better to have limit_pts_r and limit_pts_z as a "struct"
    limit_pts_z: &Array1<f64>,
    mag_r_previous: f64,
    mag_z_previous: f64,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    stationary_points: &Vec<StationaryPoint>,
) -> Result<BoundaryContour, String> {
    // TODO: add logic for negative plasma current

    // Create a mutable copy of `stationary_points`, because we want to filter it
    let mut saddle_points: Vec<StationaryPoint> = stationary_points.clone();
    // Filter to retain only `stationary_points` which are saddle points
    saddle_points.retain(|stationary_point| {
        let saddle_point_test: bool = stationary_point.hessian_determinant < 0.0;
        return saddle_point_test;
    });

    // Grid variables
    let n_r: usize = r.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let d_l: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt();

    // Find the closest grid point to the magnetic axis
    let index_mag_r: usize = (r - mag_r_previous).abs().argmin().expect("find_viable_limit_point: unwrapping index_mag_r");
    let index_mag_z: usize = (z - mag_z_previous).abs().argmin().expect("find_viable_limit_point: unwrapping index_mag_z");

    // Number of limit points
    let n_limit_pts: usize = limit_pts_r.len();

    let mut potential_limit_points: Vec<BoundaryContour> = Vec::with_capacity(n_limit_pts);

    for i_limit in 0..n_limit_pts {

        let i_r_nearest: usize = (r - limit_pts_r[i_limit]).abs().argmin().expect("find_viable_limit_point: unwrapping i_r_nearest");
        let i_z_nearest: usize = (z - limit_pts_z[i_limit]).abs().argmin().expect("find_viable_limit_point: unwrapping i_z_nearest");

        // Find the four corner grid points surrounding the limit point
        let i_r_nearest_left: usize;
        let i_r_nearest_right: usize;
        let i_z_nearest_lower: usize;
        let i_z_nearest_upper: usize;
        if limit_pts_r[i_limit] > r[i_r_nearest] {
            i_r_nearest_left = i_r_nearest;
            i_r_nearest_right = i_r_nearest + 1;
        } else {
            i_r_nearest_left = i_r_nearest - 1;
            i_r_nearest_right = i_r_nearest;
        }
        if limit_pts_z[i_limit] > z[i_z_nearest] {
            i_z_nearest_lower = i_z_nearest;
            i_z_nearest_upper = i_z_nearest + 1;
        } else {
            i_z_nearest_lower = i_z_nearest - 1;
            i_z_nearest_upper = i_z_nearest;
        }

        // Find psi at the limit point
        // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
        let mut f: Array2<f64> = Array2::zeros([2, 2]);
        let mut d_f_d_r: Array2<f64> = Array2::zeros([2, 2]);
        let mut d_f_d_z: Array2<f64> = Array2::zeros([2, 2]);
        let mut d2_f_d_r_d_z: Array2<f64> = Array2::zeros([2, 2]);

        // Function values
        f[(0, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_left)];
        f[(0, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_left)];
        f[(1, 0)] = psi_2d[(i_z_nearest_lower, i_r_nearest_right)];
        f[(1, 1)] = psi_2d[(i_z_nearest_upper, i_r_nearest_right)];

        // d(psi)/d(r)
        // bz = 1 / (2.0 * PI * r) * d_psi_d_r
        d_f_d_r[(0, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d_f_d_r[(0, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d_f_d_r[(1, 0)] = bz_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
        d_f_d_r[(1, 1)] = bz_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

        // d(psi)/d(z)
        // br = - 1 / (2.0 * PI * r) * d_psi_d_z
        d_f_d_z[(0, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d_f_d_z[(0, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d_f_d_z[(1, 0)] = -br_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
        d_f_d_z[(1, 1)] = -br_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

        // d^2(psi)/(d(r)*d(z))
        // d_bz_d_z = 1 / (2 * PI * r) * d2_psi_dr_dz
        d2_f_d_r_d_z[(0, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d2_f_d_r_d_z[(0, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_left)] * (2.0 * PI * r[i_r_nearest_left]);
        d2_f_d_r_d_z[(1, 0)] = d_bz_d_z_2d[(i_z_nearest_lower, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);
        d2_f_d_r_d_z[(1, 1)] = d_bz_d_z_2d[(i_z_nearest_upper, i_r_nearest_right)] * (2.0 * PI * r[i_r_nearest_right]);

        // Create a bicubic interpolator
        let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &f, &d_f_d_r, &d_f_d_z, &d2_f_d_r_d_z);

        let x: f64 = (limit_pts_r[i_limit] - r[i_r_nearest_left]) / d_r;
        let y: f64 = (limit_pts_z[i_limit] - z[i_z_nearest_lower]) / d_z;
        let psi_at_limit_pt: f64 = bicubic_interpolator.interpolate(x, y);

        potential_limit_points.push(
            BoundaryContour {
                boundary_r: Array1::zeros(0),
                boundary_z: Array1::zeros(0),
                n_points: 0,
                bounding_psi: psi_at_limit_pt,
                bounding_r: limit_pts_r[i_limit],
                bounding_z: limit_pts_z[i_limit],
                xpt_diverted: false,
                mask: None, // mask calculated later using method
            }
        )
    }

    // Sort from largest to smallest `psi`
    potential_limit_points.sort_by(|a, b| {
        b.bounding_psi // using `b` first gives descending order
            .partial_cmp(&a.bounding_psi)
            .expect("find_viable_limit_point: cannot sort potential_limit_points by bounding_psi")
    });

    // Loop over potential limit points; by doing the loop like this we don't do extra calculations, and exit as soon as we find a viable limit point
    'loop_over_potential_limit_points: for potential_limit_point in &mut potential_limit_points {
        // Test if there is a LFS boundary at the same height as the magnetic axis
        // March from the magnetic axis to the LFS
        let mut test_intersects_lfs_boundary: bool = false;
        for i_r in index_mag_r..n_r - 1 {
            if psi_2d[(index_mag_z, i_r)] < potential_limit_point.bounding_psi {
                test_intersects_lfs_boundary = true;
            }
        }
        // No LFS boundary encountered
        if !test_intersects_lfs_boundary {
            continue 'loop_over_potential_limit_points;
        }

        // Find the mask
        let mask_2d: Array2<f64> = flood_fill_mask(
            &r,
            &z,
            &psi_2d,
            potential_limit_point.bounding_psi,
            &stationary_points,
            mag_r_previous,
            mag_z_previous,
            &vessel_r,
            &vessel_z,
        );
        potential_limit_point.mask = Some(mask_2d.clone());

        // Test if the magnetic axis is inside the boundary (`mask`)
        let test_mask: bool = potential_limit_point.mask.as_ref().expect("find_viable_limit_point: unwrapping mask")[(index_mag_z, index_mag_r)] > 0.0;
        // Skip if magnetic axis is outside boundary
        if !test_mask {
            continue 'loop_over_potential_limit_points;
        }

        // Calculate the plasma boundary
        // TODO: update `marching_squares` to only do points near limit_point
        let psi_b: f64 = potential_limit_point.bounding_psi;
        let plasma_boundary: MarchingContour = marching_squares(
            &r,
            &z,
            &psi_2d,
            &br_2d,
            &bz_2d,
            psi_b,
            &mask_2d,
            None,
            None,
            mag_r_previous,
            mag_z_previous,
        );
        if plasma_boundary.r.len() == 0 {
            continue 'loop_over_potential_limit_points;
        }

        let distance_limit_point_to_boundary: Array1<f64> = ((potential_limit_point.bounding_r - plasma_boundary.r).powi(2)
            + (potential_limit_point.bounding_z - plasma_boundary.z).powi(2)).sqrt();

        let distance_min: f64 = distance_limit_point_to_boundary.min().expect("find_viable_limit_point: no minimum distance found").to_owned();

        // Keep if distance to plasma boundary is less than 1 grid spacing
        if distance_min > 1.0 * d_l {
            continue 'loop_over_potential_limit_points;
        }

        // If we make it to the end, then return this, ending `find_viable_limit_point` function
        return Ok(potential_limit_point.to_owned());
    }

    return Err("find_viable_limit_point: no viable limit point found".to_string());

}
