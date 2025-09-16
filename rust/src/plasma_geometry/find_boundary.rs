use super::BoundaryContour;
use super::StationaryPoint;
use super::find_viable_limit_point::find_viable_limit_point;
use super::find_viable_xpt::find_viable_xpt;
use crate::greens::D2PsiDR2Calculator;
use core::f64;
use geo::Contains;
use geo::{Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2, s};

/// Find the plasma boundary
///
/// # Arguments
/// * `r` - R grid points, metre
/// * `z` - Z grid points, metre
/// * `psi_2d` - poloidal flux, shape = (n_z, n_r), weber
/// * `br_2d` - radial magnetic field, shape = (n_z, n_r), tesla
/// * `bz_2d` - vertical magnetic field, shape = (n_z, n_r), tesla
/// * `d_br_d_z_2d` - derivative of radial magnetic field with respect to Z, shape = (n_z, n_r), tesla / metre
/// * `d_bz_d_z_2d` - derivative of vertical magnetic field with respect to Z, shape = (n_z, n_r), tesla / metre
/// * `limit_pts_r` - R coordinates of limiter points, metre
/// * `limit_pts_z` - Z coordinates of limiter points, metre
/// * `vessel_r` - R coordinates of vessel points, metre
/// * `vessel_z` - Z coordinates of vessel points, metre
/// * `mag_r` - R coordinate of magnetic axis, metre
/// * `mag_z` - Z coordinate of magnetic axis, metre
/// * `d2_psi_d_r2_calculator` - object to calculate d^2(psi)/d(r^2) at (r, z) location, will return: weber^2 / metre^2
///
/// # Returns
/// * `boundary_contour` - A `BoundaryContour` object representing the plasma boundary
///
pub fn find_boundary(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    stationary_points: &Vec<StationaryPoint>,
    br_2d: &Array2<f64>,
    bz_2d: &Array2<f64>,
    d_br_d_z_2d: &Array2<f64>,
    d_bz_d_z_2d: &Array2<f64>,
    limit_pts_r: &Array1<f64>,
    limit_pts_z: &Array1<f64>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    mag_r: f64, // Note: mag_r and mag_z are from previous iteration; this can be a problem if the magnetic axis moves significantly
    mag_z: f64, // which can happen when the plasma is significantly displaced vertically from the initial guess location, e.g. during a VDE
    d2_psi_d_r2_calculator: D2PsiDR2Calculator,
) -> Result<BoundaryContour, String> {
    // Find x-points inside the vacuum vessel which could be the plasma boundary
    let xpt_boundary: Result<BoundaryContour, String> =
        find_viable_xpt(&r, &z, &psi_2d, &br_2d, &bz_2d, &stationary_points, &vessel_r, &vessel_z, mag_r, mag_z);

    // Extract results from `xpt_boundary` object
    let xpt_r: f64;
    let xpt_z: f64;
    let xpt_psi_b: f64;
    let xpt_boundary_r: Array1<f64>;
    let xpt_boundary_z: Array1<f64>;
    if let Ok(xpt_boundary_ok) = &xpt_boundary {
        xpt_r = xpt_boundary_ok.bounding_r;
        xpt_z = xpt_boundary_ok.bounding_z;
        xpt_psi_b = xpt_boundary_ok.bounding_psi;
        xpt_boundary_r = xpt_boundary_ok.to_owned().boundary_r;
        xpt_boundary_z = xpt_boundary_ok.to_owned().boundary_z;
    } else {
        xpt_r = f64::NAN;
        xpt_z = f64::NAN;
        xpt_psi_b = f64::NAN;
        xpt_boundary_r = Array1::zeros(0);
        xpt_boundary_z = Array1::zeros(0);
    }

    // Find a viable limiter point
    let limit_boundary: Result<BoundaryContour, String> =
        find_viable_limit_point(&r, &z, &psi_2d, &limit_pts_r, &limit_pts_z, mag_r, mag_z, &vessel_r, &vessel_z);
    // println!("find_boundary: limit_boundary={:?}", limit_boundary);

    // Extract results from `limit_boundary` object
    let limit_pt_r: f64;
    let limit_pt_z: f64;
    let limit_pt_psi_b: f64;
    let limit_pt_boundary_r: Array1<f64>;
    let limit_pt_boundary_z: Array1<f64>;
    if let Ok(limit_boundary_ok) = &limit_boundary {
        limit_pt_r = limit_boundary_ok.bounding_r;
        limit_pt_z = limit_boundary_ok.bounding_z;
        limit_pt_psi_b = limit_boundary_ok.bounding_psi;
        limit_pt_boundary_r = limit_boundary_ok.to_owned().boundary_r;
        limit_pt_boundary_z = limit_boundary_ok.to_owned().boundary_z;
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
    if limit_boundary.is_err() && xpt_boundary.is_err() {
        // println!("find_boundary: No boundary found");
        xpt_diverted = false;
        bounding_r = f64::NAN;
        bounding_z = f64::NAN;
        psi_b = f64::NAN;
        boundary_r = Array1::zeros(0);
        boundary_z = Array1::zeros(0);
    } else if limit_boundary.is_err() {
        // println!("find_boundary: Found xpt");
        psi_b = xpt_psi_b;
        bounding_r = xpt_r;
        bounding_z = xpt_z;
        xpt_diverted = true;
        boundary_r = xpt_boundary_r;
        boundary_z = xpt_boundary_z;
    } else if xpt_boundary.is_err() {
        // println!("find_boundary: Found limiter");
        psi_b = limit_pt_psi_b;
        bounding_r = limit_pt_r;
        bounding_z = limit_pt_z;
        xpt_diverted = false;
        boundary_r = limit_pt_boundary_r;
        boundary_z = limit_pt_boundary_z;
    } else if limit_pt_psi_b > xpt_psi_b {
        // println!("find_boundary: Found limiter");
        psi_b = limit_pt_psi_b;
        bounding_r = limit_pt_r;
        bounding_z = limit_pt_z;
        xpt_diverted = false;
        boundary_r = limit_pt_boundary_r;
        boundary_z = limit_pt_boundary_z;
    } else if xpt_psi_b > limit_pt_psi_b {
        // println!("find_boundary: Found xpt");
        psi_b = xpt_psi_b;
        bounding_r = xpt_r;
        bounding_z = xpt_z;
        xpt_diverted = true;
        boundary_r = xpt_boundary_r;
        boundary_z = xpt_boundary_z;
    } else {
        // println!("find_boundary: Found None");
        xpt_diverted = false;
        bounding_r = f64::NAN;
        bounding_z = f64::NAN;
        psi_b = f64::NAN;
        boundary_r = Array1::zeros(0);
        boundary_z = Array1::zeros(0);
    }

    if psi_b.is_nan() {
        return Err("find_boundary: no boundary found".to_string());
    }

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();

    // Calculate the "mask"; check if grid points are inside or outside the boundary
    let mut mask: Array2<f64> = Array2::zeros((n_z, n_r));

    // boundary polygon
    let polygon_coordinates: Vec<Coord<f64>> = boundary_r.iter().zip(boundary_z.iter()).map(|(&x, &y)| Coord { x, y }).collect();

    let boundary_polygon: Polygon = Polygon::new(
        LineString::from(polygon_coordinates),
        vec![], // No holes
    );

    // Vessel polygon
    let n_vessel_pts: usize = vessel_r.len();
    let mut vessel_coords: Vec<Coord<f64>> = Vec::with_capacity(n_vessel_pts);
    for i_vessel in 0..n_vessel_pts {
        vessel_coords.push(Coord {
            x: vessel_r[i_vessel],
            y: vessel_z[i_vessel],
        });
    }
    let vessel_polygon: Polygon = Polygon::new(
        LineString::from(vessel_coords),
        vec![], // No holes
    );

    // Construct the mask
    // mask=0.0 outside the plasma
    // mask=1.0 inside the plasma
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            // Check if the point is inside the polygon
            let test_point: Point = Point::new(r[i_r], z[i_z]);
            let inside_bounding_contour: bool = boundary_polygon.contains(&test_point);
            let inside_vessel: bool = vessel_polygon.contains(&test_point);
            if inside_bounding_contour && inside_vessel {
                mask[(i_z, i_r)] = 1.0;
            }
        }
    }

    // Remove the centre post (do we actually need this????)
    let mut i_vessel_r: usize = 0;
    let target_r: f64 = 0.1704; // the inboard limiter; actually 0.1705
    for i_r in 0..n_r {
        if r[i_r] < target_r {
            // ensure less than the limiter
            i_vessel_r = i_r;
        }
    }
    mask.slice_mut(s![.., 0..i_vessel_r]).fill(0.0); // shape = (n_z, n_r)

    // If there are any x-points, then mask above and below the x-point
    // SHOULDN'T NEED TO DO THIS!!!!
    if xpt_diverted {
        for i_z in 0..n_z {
            if z[i_z] > bounding_z.abs() {
                mask.slice_mut(s![i_z..n_z, ..]).fill(0.0); // shape = (n_z, n_r)
            }
            if z[i_z] < -bounding_z.abs() {
                mask.slice_mut(s![0..i_z, ..]).fill(0.0); // shape = (n_z, n_r)
            }
        }
    }

    // // Calculate psi_n_2d
    // let psi_n_2d: Array2<f64> = &mask * (&psi_2d - psi_a) / (psi_b - psi_a);

    // println!("find_boundary: psi_b={psi_b}, bounding_r={bounding_r}, bounding_z={bounding_z}");

    let result: BoundaryContour = BoundaryContour {
        boundary_polygon,
        boundary_r: boundary_r.clone(),
        boundary_z,
        n_points: boundary_r.len(),
        bounding_psi: psi_b,
        bounding_r,
        bounding_z,
        fraction_inside_vessel: f64::NAN, // fraction inside vessel not calculated here
        xpt_diverted,
        plasma_volume: None, // volume calculated using method
        mask: Some(mask),
        secondary_xpt_r: f64::NAN,
        secondary_xpt_z: f64::NAN,
        secondary_xpt_distance: f64::NAN,
    };

    return Ok(result);
}
