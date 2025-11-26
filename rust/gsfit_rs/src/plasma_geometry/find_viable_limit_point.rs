use super::BoundaryContour;
use super::bicubic_interpolator::BicubicInterpolator;
use super::flood_fill_mask::flood_fill_mask;
use core::f64;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;
use ndarray_stats::QuantileExt;
use rand::rand_core::le;
use super::StationaryPoint;
use super::marching_squares::BoundaryContourNew;
use super::marching_squares::marching_squares;

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

    // Create an interpolator for psi
    // TODO: replace with `BicubicInterpolator` to be consistent
    // let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &psi_2d, &d_psi_d_r, &d_psi_d_z, &d2_psi_d_r_d_z);
    let psi_interpolator = Interp2D::builder(psi_2d.clone())
        .x(z.clone())
        .y(r.clone())
        .build()
        .expect("find_boundary: Can't make Interp2D");

    // Number of limit points
    let n_limit_pts: usize = limit_pts_r.len();

    let mut potential_limit_points: Vec<BoundaryContour> = Vec::with_capacity(n_limit_pts);

    for i_limit in 0..n_limit_pts {
        let psi_at_limit_pt: f64 = psi_interpolator
            .interp_scalar(limit_pts_z[i_limit], limit_pts_r[i_limit])
            .expect("possible_bounding_psi: error, limiter");

        potential_limit_points.push(
            BoundaryContour {
                boundary_r: Array1::zeros(0),
                boundary_z: Array1::zeros(0),
                n_points: 0,
                bounding_psi: psi_at_limit_pt,
                bounding_r: limit_pts_r[i_limit],
                bounding_z: limit_pts_z[i_limit],
                fraction_inside_vessel: f64::NAN,
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

    // Find the closest grid point to the magnetic axis
    let index_mag_r: usize = (r - mag_r_previous).abs().argmin().expect("find_viable_limit_point: unwrapping index_mag_r");
    let index_mag_z: usize = (z - mag_z_previous).abs().argmin().expect("find_viable_limit_point: unwrapping index_mag_z");

    // March from the magnetic axis to the LFS and check if we intersect any contours
    potential_limit_points.retain(|potential_limit_point| {
        for i_r in index_mag_r..n_r - 1 {
            if psi_2d[(index_mag_z, i_r)] < potential_limit_point.bounding_psi {
                return true;
            }
        }
        // No LFS boundary encountered
        return false;
    });
    // Exit if we haven't found any `stationary_points` which have saddle curvature
    if potential_limit_points.len() == 0 {
        return Err("find_viable_limit_point: no stationary points with LFS boundary".to_string());
    }

    // TODO: Check distance from limit_point to plasma boundary ==> need accurate boundary calculation, not escaping saddle point
    // BUT: I think `marching_squares` is quite slow, so I don't want to call it during GS Picard iteration.
    // TODO: create a reduced version of `marching_squares` which is only applied near the limit_point

    // As a stop-gap measure I can use the distance from the limit_point to `mask_2d`
    // If I decide to improve, I will still need `mask_2d`

    // Add the mast to all potential limit points
    // TODO: we will only be keeping the first `potential_limit_point`, so perhpaps we don't need to calculate all masks
    for potential_limit_point in &mut potential_limit_points {
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

        potential_limit_point.mask = Some(mask_2d);
    }

    // Retain only `potential_limit_points` where the magnetic axis is inside the boundary (`mask`)
    potential_limit_points.retain(|potential_limit_point| {
        let mask_test: bool = potential_limit_point.mask.as_ref().expect("find_viable_limit_point: unwrapping mask")[(index_mag_z, index_mag_r)] > 0.0;
        return mask_test;
    });
    // Exit if we haven't found any `potential_limit_points` which have saddle curvature
    if potential_limit_points.len() == 0 {
        return Err("find_viable_limit_point: no potential_limit_points with LFS boundary".to_string());
    }

    // Check the distance from the limit_point to the plasma boundary
    potential_limit_points.retain(|potential_limit_point| {
        let mask_2d: &Array2<f64> = potential_limit_point.mask.as_ref().expect("find_viable_limit_point: unwrapping mask 2");
        let psi_b: f64 = potential_limit_point.bounding_psi;

        // TODO: update `marching_squares` to only do points near limit_point
        let plasma_boundary: BoundaryContourNew = marching_squares(
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

        // TODO: how did this happen?
        if plasma_boundary.r.len() == 0 {
            println!("find_viable_limit_point: plasma_boundary has zero length");
            return false;
        }

        let distance_limit_point_to_boundary: Array1<f64> = ((potential_limit_point.bounding_r - plasma_boundary.r).powi(2)
            + (potential_limit_point.bounding_z - plasma_boundary.z).powi(2)).sqrt();

        let distance_min: f64 = distance_limit_point_to_boundary.min().expect("find_viable_limit_point: no minimum distance found").to_owned();

        // Keep if distance to plasma boundary is less than 1 grid spacing
        if distance_min < 1.0 * d_l {
            return true;
        } else {
            return false;
        }
    });
    // Exit if we haven't found any `potential_limit_points`
    if potential_limit_points.len() == 0 {
        return Err("find_viable_limit_point: no stationary points with LFS boundary".to_string());
    }

    return Ok(potential_limit_points[0].to_owned());

}
