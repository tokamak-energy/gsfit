use super::BoundaryContour;
use approx::abs_diff_eq;
use contour::ContourBuilder;
use core::f64;
use geo::Contains;
use geo::{Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;

pub fn find_viable_limit_point(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    limit_pts_r: &Array1<f64>,
    limit_pts_z: &Array1<f64>,
    r_mag: f64,
    z_mag: f64,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
) -> Result<BoundaryContour, String> {
    // Magnetic axis point
    let magnetic_axis_point: Point = Point::new(r_mag, z_mag);

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

            let boundary_r: Array1<f64> = boundary_contour.exterior().coords().map(|coord| coord.x).collect::<Array1<f64>>();
            let boundary_z: Array1<f64> = boundary_contour.exterior().coords().map(|coord| coord.y).collect::<Array1<f64>>();
            let n_points: usize = boundary_r.len();

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
                plasma_volume: None, // volume calculated using method
                mask: None,          // mask calculated using method
                secondary_xpt_r: f64::NAN,
                secondary_xpt_z: f64::NAN,
                secondary_xpt_distance: f64::NAN,
            });
        }
    }

    // Check if the boundary has reached the grid edge
    boundary_contours_all.retain(|boundary_contour| {
        let polygon: Polygon = boundary_contour.boundary_polygon.clone();

        let mut has_reached_edge: bool = false;
        // Check if the boundary has reached the grid edge
        for coord in polygon.exterior().coords() {
            let r_coord: f64 = coord.x;
            let z_coord: f64 = coord.y;

            // TODO: should I set a tolerance here, or is machine precision ok??
            if abs_diff_eq!(r_coord, r[0]) || abs_diff_eq!(r_coord, r[n_r - 1]) || abs_diff_eq!(z_coord, z[0]) || abs_diff_eq!(z_coord, z[n_z - 1]) {
                // println!("find_viable_limit_point: boundary is not closed!!!");
                has_reached_edge = true;
            }
        }

        // "retain" the contours which have not reached the edge
        let not_reached_edge: bool = !has_reached_edge;
        // println!("not_reached_edge={not_reached_edge}");
        return not_reached_edge;
    });

    // Find the shortest distance from any boundary point to the limit_point
    // Find the minimum distance from any of the points which describe the boundary to the limit_pt
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

        // "retain" the contours which get close to the limit point
        // println!("boundary_crosses_limit_point={boundary_crosses_limit_point}");
        return boundary_crosses_limit_point;
    });
    // Exit if we haven't found any boundary contours
    if boundary_contours_all.len() == 0 {
        // println!("find_viable_limit_point: no boundary found");
        return Err("no boundary found".to_string());
    }

    // Check if the magnetic axis (point) is inside the boundary (polygon)
    // The contours are already sorted by `psi_b`
    for boundary_contour in &boundary_contours_all {
        let boundary_polygon: Polygon = boundary_contour.boundary_polygon.clone();
        let inside: bool = boundary_polygon.contains(&magnetic_axis_point);
        if inside {
            return Ok(boundary_contour.to_owned());
        }
    }

    // If we don't find any contours which contain the magnetic axis we will fall
    // back and use the largest `fraction_inside_vessel`
    for boundary_contour in &mut boundary_contours_all {
        let mut n_inside_vessel: usize = 0;

        let contour_polygon: Polygon = boundary_contour.boundary_polygon.clone();

        for coord in contour_polygon.exterior().coords() {
            let boundary_r: f64 = coord.x;
            let boundary_z: f64 = coord.y;
            let inside_vessel: bool = vessel_polygon.contains(&Point::new(boundary_r, boundary_z));

            if inside_vessel {
                n_inside_vessel += 1;
            }
        }
        let fraction_inside_vessel: f64 = n_inside_vessel as f64 / (boundary_contour.n_points as f64);

        // Store
        boundary_contour.fraction_inside_vessel = fraction_inside_vessel;
    }

    // Only retain if fraction_inside_vessel is greater than 0.8
    boundary_contours_all.retain(|boundary_contour| {
        let fraction_inside_vessel_above_threshold: bool = boundary_contour.fraction_inside_vessel > 0.8;
        return fraction_inside_vessel_above_threshold;
    });
    // Exit if we haven't found any boundary contours
    if boundary_contours_all.len() == 0 {
        // println!("find_viable_limit_point: no boundary found 02");
        return Err("no boundary found 02".to_string());
    }

    // Sort by `fraction_inside_vessel`
    boundary_contours_all.sort_by(|a, b| {
        b.fraction_inside_vessel
            .partial_cmp(&a.fraction_inside_vessel)
            .expect("find_viable_limit_point: cannot find `boundary_contours_all`")
    });

    // for boundary_contour in &boundary_contours_all {
    //     println!("fraction_inside_vessel={}", boundary_contour.fraction_inside_vessel);
    // }

    // Return the first element, i.e. the largest `fraction_inside_vessel`
    return Ok(boundary_contours_all[0].to_owned());
}
