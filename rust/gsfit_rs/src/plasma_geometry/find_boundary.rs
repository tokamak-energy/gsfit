use super::BoundaryContour;
use super::Error;
use super::StationaryPoint;
use super::find_viable_limit_point::find_viable_limit_point;
use super::find_viable_xpt::find_viable_xpt;
use super::flood_fill_mask::flood_fill_mask;
use core::f64;
use ndarray::{Array1, Array2};

/// Find the plasma boundary
///
/// # Arguments
/// * `r` - R grid points, metre
/// * `z` - Z grid points, metre
/// * `psi_2d` - poloidal flux, shape = (n_z, n_r), weber
/// * `limit_pts_r` - R coordinates of limiter points, metre
/// * `limit_pts_z` - Z coordinates of limiter points, metre
/// * `vessel_r` - R coordinates of vessel points, metre
/// * `vessel_z` - Z coordinates of vessel points, metre
/// * `mag_r_previous` - R coordinate of magnetic axis (from the previous iteration), metre
/// * `mag_z_previous` - Z coordinate of magnetic axis (from the previous iteration), metre
///
/// # Returns
/// * `boundary_contour` - A `BoundaryContour` object representing the plasma boundary
///
pub fn find_boundary(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    stationary_points: &Vec<StationaryPoint>,
    limit_pts_r: &Array1<f64>,
    limit_pts_z: &Array1<f64>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    mag_r_previous: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
    mag_z_previous: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
) -> Result<BoundaryContour, Error> {
    // Find x-points inside the vacuum vessel which could be the plasma boundary
    let xpt_boundary_or_error: Result<BoundaryContour, String> =
        find_viable_xpt(&r, &z, &psi_2d, &stationary_points, &vessel_r, &vessel_z, mag_r_previous, mag_z_previous);
    // println!("find_boundary: xpt_boundary_or_error = {:?}", xpt_boundary_or_error);

    // Extract results from `xpt_boundary` object
    let xpt_r: f64;
    let xpt_z: f64;
    let xpt_psi_b: f64;
    let xpt_boundary_r: Array1<f64>;
    let xpt_boundary_z: Array1<f64>;
    if let Ok(xpt_boundary) = &xpt_boundary_or_error {
        xpt_r = xpt_boundary.bounding_r;
        xpt_z = xpt_boundary.bounding_z;
        xpt_psi_b = xpt_boundary.bounding_psi;
        xpt_boundary_r = xpt_boundary.to_owned().boundary_r;
        xpt_boundary_z = xpt_boundary.to_owned().boundary_z;
    } else {
        xpt_r = f64::NAN;
        xpt_z = f64::NAN;
        xpt_psi_b = f64::NAN;
        xpt_boundary_r = Array1::zeros(0);
        xpt_boundary_z = Array1::zeros(0);
    }

    // Find a viable limiter point
    let limit_boundary_or_error: Result<BoundaryContour, String> = find_viable_limit_point(
        &r,
        &z,
        &psi_2d,
        &br_2d,
        &bz_2d,
        &d_bz_d_z_2d,
        &limit_pts_r,
        &limit_pts_z,
        mag_r_previous,
        mag_z_previous,
        &vessel_r,
        &vessel_z,
        &stationary_points,
    );
    // println!("find_boundary: limit_boundary_or_error = {:?}", limit_boundary_or_error);

    // Extract results from `limit_boundary` object
    let limit_pt_r: f64;
    let limit_pt_z: f64;
    let limit_pt_psi_b: f64;
    let limit_pt_boundary_r: Array1<f64>;
    let limit_pt_boundary_z: Array1<f64>;
    if let Ok(limit_boundary) = &limit_boundary_or_error {
        limit_pt_r = limit_boundary.bounding_r;
        limit_pt_z = limit_boundary.bounding_z;
        limit_pt_psi_b = limit_boundary.bounding_psi;
        limit_pt_boundary_r = limit_boundary.to_owned().boundary_r;
        limit_pt_boundary_z = limit_boundary.to_owned().boundary_z;
    } else {
        limit_pt_r = f64::NAN;
        limit_pt_z = f64::NAN;
        limit_pt_psi_b = f64::NAN;
        limit_pt_boundary_r = Array1::zeros(0);
        limit_pt_boundary_z = Array1::zeros(0);
    }

    // Figure out which one is the boundary
    let psi_b: f64;
    let bounding_r: f64;
    let bounding_z: f64;
    let xpt_diverted: bool;
    let boundary_r: Array1<f64>;
    let boundary_z: Array1<f64>;
    if limit_boundary_or_error.is_err() && xpt_boundary_or_error.is_err() {
        xpt_diverted = false;
        bounding_r = f64::NAN;
        bounding_z = f64::NAN;
        psi_b = f64::NAN;
        boundary_r = Array1::zeros(0);
        boundary_z = Array1::zeros(0);
    } else if limit_boundary_or_error.is_err() {
        psi_b = xpt_psi_b;
        bounding_r = xpt_r;
        bounding_z = xpt_z;
        xpt_diverted = true;
        boundary_r = xpt_boundary_r;
        boundary_z = xpt_boundary_z;
    } else if xpt_boundary_or_error.is_err() {
        psi_b = limit_pt_psi_b;
        bounding_r = limit_pt_r;
        bounding_z = limit_pt_z;
        xpt_diverted = false;
        boundary_r = limit_pt_boundary_r;
        boundary_z = limit_pt_boundary_z;
    } else if limit_pt_psi_b > xpt_psi_b {
        psi_b = limit_pt_psi_b;
        bounding_r = limit_pt_r;
        bounding_z = limit_pt_z;
        xpt_diverted = false;
        boundary_r = limit_pt_boundary_r;
        boundary_z = limit_pt_boundary_z;
    } else if xpt_psi_b > limit_pt_psi_b {
        psi_b = xpt_psi_b;
        bounding_r = xpt_r;
        bounding_z = xpt_z;
        xpt_diverted = true;
        boundary_r = xpt_boundary_r;
        boundary_z = xpt_boundary_z;
    } else {
        xpt_diverted = false;
        bounding_r = f64::NAN;
        bounding_z = f64::NAN;
        psi_b = f64::NAN;
        boundary_r = Array1::zeros(0);
        boundary_z = Array1::zeros(0);
    }

    if psi_b.is_nan() {
        return Err(Error::NoBoundaryFound {
            no_xpt_reason: "".to_string(),
            no_limit_point_reason: "".to_string(),
        });
    }

    // println!("xpt_diverted = {}", xpt_diverted);

    // Calculate the mask
    let mask: Array2<f64> = flood_fill_mask(&r, &z, &psi_2d, psi_b, &stationary_points, mag_r_previous, mag_z_previous, &vessel_r, &vessel_z);

    let boundary_contour: BoundaryContour = BoundaryContour {
        boundary_r: boundary_r.clone(),
        boundary_z,
        n_points: boundary_r.len(),
        bounding_psi: psi_b,
        bounding_r,
        bounding_z,
        xpt_diverted,
        mask: Some(mask),
    };

    return Ok(boundary_contour);
}
