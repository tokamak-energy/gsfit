use super::BoundaryContour;
use super::bicubic_interpolator::BicubicInterpolator;
use super::flood_fill_mask::flood_fill_mask;
use approx::abs_diff_eq;
use contour::ContourBuilder;
use core::f64;
use geo::Contains;
use geo::{Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;
use ndarray_stats::QuantileExt;
use super::StationaryPoint;

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
///
/// # Returns
/// * `BoundaryContour` - A `BoundaryContour` object representing the plasma boundary
///
pub fn find_viable_limit_point(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    limit_pts_r: &Array1<f64>,
    limit_pts_z: &Array1<f64>,
    mag_r_previous: f64,
    mag_z_previous: f64,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    stationary_points: &Vec<StationaryPoint>,
) -> Result<BoundaryContour, String> {
    // TODO: add logic for negative plasma current

    // Magnetic axis point
    let magnetic_axis_point: Point = Point::new(mag_r_previous, mag_z_previous);

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

    // Create an interpolator for psi
    // TODO: replace with `BicubicInterpolator` to be consistent
    // let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, &psi_2d, &d_psi_d_r, &d_psi_d_z, &d2_psi_d_r_d_z);
    let psi_interpolator = Interp2D::builder(psi_2d.clone())
        .x(z.clone())
        .y(r.clone())
        .build()
        .expect("find_boundary: Can't make Interp2D");

    // Find psi at all possible limit points
    let n_limit_pts: usize = limit_pts_r.len();
    let mut possible_bounding_psi: Array1<f64> = Array1::from_elem(n_limit_pts, f64::NAN);
    for i_limiter in 0..n_limit_pts {
        possible_bounding_psi[i_limiter] = psi_interpolator
            .interp_scalar(limit_pts_z[i_limiter], limit_pts_r[i_limiter])
            .expect("possible_bounding_psi: error, limiter");
    }

    // Sort from largest to smallest psi
    let mut index_lim: Vec<usize> = (0..n_limit_pts).collect();
    index_lim.sort_by(|&i_left, &i_right| {
        possible_bounding_psi[i_right]
            .partial_cmp(&possible_bounding_psi[i_left])
            .expect("find_viable_limit_point: cannot find `index_lim`")
    });

    // Grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = &r[1] - &r[0];
    let d_z: f64 = &z[1] - &z[0];
    let d_l: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt();
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    // Construct an empty "contour_grid" object
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true) // x dim., y dim., smoothing
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    // Flatten psi_2d
    let psi_2d_flattened: Vec<f64> = psi_2d.iter().cloned().collect();

    // Loop over all limit points and contours associated with each limit point.
    // Essentially I am flattening the double for loop
    let mut boundary_contours_all: Vec<BoundaryContour> = Vec::new();
    for i_limiter in index_lim {
        let psi_b: f64 = possible_bounding_psi[i_limiter];
        let boundary_contours_tmp: Vec<contour::Contour> = contour_grid
            .contours(&psi_2d_flattened, &[psi_b])
            .expect("find_viable_limit_point: cannot find `boundary_contours_tmp`");
        let boundary_contours: &geo_types::MultiPolygon = boundary_contours_tmp[0].geometry(); // The [0] is because I have only supplied one threshold

        // Loop over all contours and find if there is a contour which contains the magnetic axis
        let n_contours: usize = boundary_contours.iter().count();
        for i_contour in 0..n_contours {
            // This limit point
            let limit_pt_r: f64 = limit_pts_r[i_limiter];
            let limit_pt_z: f64 = limit_pts_z[i_limiter];

            // Get the boundary_contour for this path
            let boundary_contour: &Polygon = boundary_contours
                .iter()
                .nth(i_contour)
                .expect("find_viable_limit_point: cannot find `boundary_contour`");

            let boundary_r: Array1<f64> = Array1::zeros(0);
            let boundary_z: Array1<f64> = Array1::zeros(0);
            let n_points: usize = 0;

            boundary_contours_all.push(BoundaryContour {
                boundary_polygon: boundary_contour.clone(),
                boundary_r,
                boundary_z,
                n_points,
                bounding_psi: psi_b,
                bounding_r: limit_pt_r,
                bounding_z: limit_pt_z,
                fraction_inside_vessel: f64::NAN,
                xpt_diverted: false,
                mask: None, // mask calculated later using method
            });
        }
    }
    // Exit if we haven't found any boundary contours
    if boundary_contours_all.len() == 0 {
        return Err("no boundary found 01".to_string());
    }

    // Create a contour using `stationary_point.psi`; and check if it exists on the LFS at the magnetic axis height
    // Find the closest grid point to the magnetic axis
    let index_mag_r: usize = (r - mag_r_previous).abs().argmin().expect("find_limit_point: unwrapping index_mag_r");
    let index_mag_z: usize = (z - mag_z_previous).abs().argmin().expect("find_limit_point: unwrapping index_mag_z");
    // March from the magnetic axis to the LFS and check if we intersect any contours
    let n_r: usize = r.len();
    let mut index_distance: Vec<usize> = Vec::new();
    boundary_contours_all.retain(|stationary_point| {
        for i_r in index_mag_r..n_r - 1 {
            if psi_2d[(index_mag_z, i_r)] < stationary_point.bounding_psi {
                index_distance.push(i_r);
                return true;
            }
        }
        // No LFS boundary encountered
        return false;
    });
    // Exit if we haven't found any `stationary_points` which have saddle curvature
    if boundary_contours_all.len() == 0 {
        return Err("find_viable_xpt: no stationary points with LFS boundary".to_string());
    }

    // Find the shortest distance from any boundary point to the `limit_point`
    // Find the minimum distance from any of the points which describe the boundary to the limit_pt
    // Retain contours which get "close" to the limit point
    boundary_contours_all.retain(|boundary_contour| {
        let mut boundary_crosses_limit_point: bool = false;

        let limit_point_r: f64 = boundary_contour.bounding_r;
        let limit_point_z: f64 = boundary_contour.bounding_z;

        let polygon: Polygon = boundary_contour.boundary_polygon.clone();
        for coord in polygon.exterior().coords() {
            let boundary_r: f64 = coord.x;
            let boundary_z: f64 = coord.y;

            let distance: f64 = ((limit_point_r - boundary_r).powi(2) + (limit_point_z - boundary_z).powi(2)).sqrt();
            if distance < 1.0 * d_l {
                boundary_crosses_limit_point = true;
            }
        }

        return boundary_crosses_limit_point;
    });
    // Exit if we haven't found any boundary contours
    if boundary_contours_all.len() == 0 {
        return Err("no boundary found 03".to_string());
    }

    // TODO: Check distance from limit_point to plasma boundary ==> need accurate boundary calculation, not escaping saddle point
    // BUT: I think `marching_squares` is quite slow, so I don't want to call it during GS Picard iteration.
    // TODO: create a reduced version of `marching_squares` which is only applied near the limit_point

    // As a stop-gap measure I can use the distance from the limit_point to `mask_2d`
    // If I decide to improve, I will still need `mask_2d`

    // mask_2d = flood_fill_mask(&r, &z, &psi_2d, f64::INFINITY, &Vec::new(), mag_r_previous, mag_z_previous, &vessel_r, &vessel_z);

    // Check if the magnetic axis (point) is inside the boundary (polygon)
    // The contours are already sorted by `psi_b`
    for boundary_contour in &boundary_contours_all {
        let boundary_polygon: Polygon = boundary_contour.boundary_polygon.clone();
        let inside: bool = boundary_polygon.contains(&magnetic_axis_point);
        if inside {
            return Ok(boundary_contour.to_owned());
        }
    }

    // Return the first element, i.e. the largest `fraction_inside_vessel`
    // return Ok(boundary_contours_all[0].to_owned());
    return Err("no boundary found 05".to_string());
}
