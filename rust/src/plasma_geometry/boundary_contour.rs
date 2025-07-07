use core::f64;

use geo::Area;
use geo::Centroid;
use geo::Contains;
use geo::{Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_interp::interp2d::Interp2D;
use ndarray_linalg::Solve;
use ndarray_stats::QuantileExt;
use rand::rand_core::le;
use rayon::vec;
use std::ops::Range;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct BoundaryContour {
    pub boundary_polygon: Polygon,
    pub boundary_r: Array1<f64>,
    pub boundary_z: Array1<f64>,
    pub n_points: usize,
    pub bounding_psi: f64,
    pub bounding_r: f64,
    pub bounding_z: f64,
    pub fraction_inside_vessel: f64,
    pub xpt_diverted: bool,
    pub plasma_volume: Option<f64>,
    pub mask: Option<Array2<f64>>,
    pub secondary_xpt_r: f64,
    pub secondary_xpt_z: f64,
    pub secondary_xpt_distance: f64,
}

impl BoundaryContour {
    /// Calculates the volume of revolution (around the Z axis) using the boundary_r and boundary_z arrays.
    /// Uses the method of disks (Pappus's centroid theorem).
    pub fn calculate_plasma_volume(&mut self) {
        let n_boundary_points: usize = self.n_points;
        let boundary_r: Array1<f64> = self.boundary_r.to_owned();
        let boundary_z: Array1<f64> = self.boundary_z.to_owned();

        // Collect the coordinates
        let mut boundary_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_boundary_points);
        for i_boundary_point in 0..n_boundary_points {
            boundary_coordinates.push(Coord {
                x: boundary_r[i_boundary_point],
                y: boundary_z[i_boundary_point],
            });
        }

        // Construct the contour
        let boundary_contour: Polygon = Polygon::new(
            LineString::new(boundary_coordinates),
            vec![], // No holes
        );

        let cross_sectional_area: f64 = boundary_contour.unsigned_area();

        let mass_centroid_r: f64 = boundary_contour
            .centroid()
            .expect("calculate_plasma_volume: Error with boundary_contour.centroid()")
            .x();

        let plasma_volume: f64 = 2.0 * PI * mass_centroid_r * cross_sectional_area;

        self.plasma_volume = Some(plasma_volume);
    }

    /// Refine the boundary contoury by:
    /// * The contour starts at the x-point, then goes to the high field side, around the plasma and back to the x-point.
    /// * Ensuring that the contour does not excape into the private flux region.
    /// * The x-point is repeated as the first and last pointss in the contour.
    /// We will do all the logic for a LSN (Lower Single Null) configuration. For USN we will flip the array.
    /// TODO: There is a bug where the boundary is made up of two contours, one for the LFS and one for the HFS. This case is missed!!
    pub fn refine_xpt_diverted_boundary(
        &mut self,
        r: &Array1<f64>,
        z: &Array1<f64>,
        psi_2d_in: &Array2<f64>,
        mag_r: f64,
        mag_z: f64,
        br_2d: &Array2<f64>,
        bz_2d: &Array2<f64>,
    ) {
        let boundary_r: Array1<f64> = self.boundary_r.to_owned();
        let xpt_r: f64 = self.bounding_r;

        // All of the logic is done for a LSN (Lower Single Null) configuration.
        // If we have an USN (Upper Single Null), we will flip the psi_2d_in array and treat it as a LSN.
        let psi_2d: Array2<f64>;
        let xpt_z: f64;
        let boundary_z: Array1<f64>;
        if self.bounding_z > 0.0 {
            // Treat USN as LSN
            psi_2d = psi_2d_in.slice(s![..;-1, ..]).to_owned();
            xpt_z = -self.bounding_z.to_owned();
            boundary_z = -self.boundary_z.to_owned();
            // boundary_z = -boundary_z;
        } else {
            // LSN
            psi_2d = psi_2d_in.to_owned();
            xpt_z = self.bounding_z.to_owned();
            boundary_z = self.boundary_z.to_owned();
        }

        // TODO: Very worringly, the `Contour` crate can produce points whihc are outside the (R, Z) grid by 1/2 a grid spacing.
        // This has been reported to GitHub as an issue.
        // But until it's fixed we shall add this extra test.
        let n_boundary_points: usize = boundary_r.len();
        let mut indices_inside_grid: Vec<usize> = Vec::with_capacity(n_boundary_points);
        for i_point in 0..n_boundary_points {
            if boundary_r[i_point] < r.max().expect("find_viable_xpt: r.max()").to_owned()
                && boundary_r[i_point] > r.min().expect("find_viable_xpt: r.min()").to_owned()
                && boundary_z[i_point] < z.max().expect("find_viable_xpt: z.max()").to_owned()
                && boundary_z[i_point] > z.min().expect("find_viable_xpt: z.min()").to_owned()
            {
                indices_inside_grid.push(i_point);
            }
        }
        let boundary_r: Array1<f64> = boundary_r.select(Axis(0), &indices_inside_grid).to_owned();
        let boundary_z: Array1<f64> = boundary_z.select(Axis(0), &indices_inside_grid).to_owned();

        // Store number of boundary points
        let n_boundary_points: usize = boundary_r.len();

        // Grid spacing
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        // Calculate the gradient of psi at the boundary points#
        let mut d_psi_d_r: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_r in 0..n_r {
            for i_z in 0..n_z {
                d_psi_d_r[[i_z, i_r]] = 2.0 * PI * r[i_r] * bz_2d[[i_z, i_r]];
                d_psi_d_z[[i_z, i_r]] = -2.0 * PI * r[i_r] * br_2d[[i_z, i_r]];
            }
        }

        let d_psi_d_r_interpolator = Interp2D::builder(d_psi_d_r.clone())
            .x(z.clone())
            .y(r.clone())
            .build()
            .expect("refine_xpt_diverted_boundary: Can't make Interp2D for d_psi_d_r");
        let d_psi_d_z_interpolator = Interp2D::builder(d_psi_d_z.clone())
            .x(z.clone())
            .y(r.clone())
            .build()
            .expect("refine_xpt_diverted_boundary: Can't make Interp2D for d_psi_d_z");

        // Interpolate the gradient of psi at the boundary points
        let boundary_d_psi_d_r: Array1<f64> = d_psi_d_r_interpolator
            .interp_array(&boundary_z, &boundary_r)
            .expect("refine_xpt_diverted_boundary: Can't interpolate d_psi_d_r for boundary points");
        let boundary_d_psi_d_z: Array1<f64> = d_psi_d_z_interpolator
            .interp_array(&boundary_z, &boundary_r)
            .expect("refine_xpt_diverted_boundary: Can't interpolate d_psi_d_z for boundary points");

        // Direction vector from boundary point to magnetic axis
        let normalisation: Array1<f64> = ((mag_r - &boundary_r).powi(2) + (mag_z - &boundary_z).powi(2)).sqrt();
        let boundary_to_mag_direction_vec_r: Array1<f64> = (mag_r - &boundary_r) / &normalisation;
        let boundary_to_mag_direction_vec_z: Array1<f64> = (mag_z - &boundary_z) / &normalisation;

        // Vector dot product to indicate if the boundary point is on the plasma surface or the private flux region
        let v_dot_grad_psi: Array1<f64> = boundary_d_psi_d_r * boundary_to_mag_direction_vec_r + boundary_d_psi_d_z * boundary_to_mag_direction_vec_z;

        // Radius of circle around x-point
        let circle_radius: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt() / 1.7; // ~1cm with n_r=80, n_z=161

        // Exit if the circle goes outside the grid
        if xpt_r - circle_radius < r.min().expect("refine_xpt_diverted_boundary: r.min()").to_owned()
            || xpt_r + circle_radius > r.max().expect("refine_xpt_diverted_boundary: r.max()").to_owned()
            || xpt_z - circle_radius < z.min().expect("refine_xpt_diverted_boundary: z.min()").to_owned()
            || xpt_z + circle_radius > z.max().expect("refine_xpt_diverted_boundary: z.max()").to_owned()
        {
            return;
        }

        // Find the turning points near the x-point - this gives us direction vectors to know where the plasma is and isn't
        // println!("boundary_r= {:#?}", boundary_r);
        // println!("boundary_z= {:#?}", boundary_z);
        let turning_points_near_xpt: Result<(f64, f64, f64, f64, f64, f64), String> = find_minima_maxima_about_xpt(&r, &z, &psi_2d, xpt_r, xpt_z, true);
        if turning_points_near_xpt.is_err() {
            println!(
                "refine_xpt_diverted_boundary: error with find_minima_maxima_about_xpt: {}",
                turning_points_near_xpt.err().unwrap()
            );
            return;
        }
        // Extract the turning point results
        let (
            xpt_turning_lfs_minima_r,
            xpt_turning_lfs_minima_z,
            xpt_turning_hfs_minima_r,
            xpt_turning_hfs_minima_z,
            xpt_turning_plasma_maxima_r,
            xpt_turning_plasma_maxima_z,
        ): (f64, f64, f64, f64, f64, f64) = turning_points_near_xpt.expect("refine_xpt_diverted_boundary: error unwrapping turning_points_near_xpt");

        // Create a polygon where the plasma can be, with 6 points
        let mut plasma_region_coords: Vec<Coord<f64>> = Vec::with_capacity(6);
        // Point 1: x-point
        plasma_region_coords.push(Coord { x: xpt_r, y: xpt_z });
        // Direction vector from x-point to the lfs minima
        let direction_dr: f64 = xpt_turning_lfs_minima_r - xpt_r;
        let direction_dz: f64 = xpt_turning_lfs_minima_z - xpt_z;
        // Normalize the direction vector
        let direction_norm: f64 = (direction_dr.powi(2) + direction_dz.powi(2)).sqrt();
        let direction_dr: f64 = direction_dr / direction_norm;
        let direction_dz: f64 = direction_dz / direction_norm;
        // Point 2: lower right
        plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0, // 100m should always be enough!
            y: xpt_z + direction_dz * 100.0,
        });
        // Point 3: top right
        plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0, // 100m should always be enough!
            y: xpt_z + direction_dz * 100.0 + 100.0,
        });
        // Direction vector from x-point to the hfs minima
        let direction_dr: f64 = xpt_turning_hfs_minima_r - xpt_r;
        let direction_dz: f64 = xpt_turning_hfs_minima_z - xpt_z;
        // Normalize the direction vector
        let direction_norm: f64 = (direction_dr.powi(2) + direction_dz.powi(2)).sqrt();
        let direction_dr: f64 = direction_dr / direction_norm;
        let direction_dz: f64 = direction_dz / direction_norm;
        // Point 4: top left
        plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0, // 100m should always be enough!
            y: xpt_z + direction_dz * 100.0 + 100.0,
        });
        // Point 5: lower left
        plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0,
            y: xpt_z + direction_dz * 100.0,
        });
        // Point 6: x-point (again)
        plasma_region_coords.push(Coord { x: xpt_r, y: xpt_z });
        // Create the polygon
        let plasma_region: Polygon = Polygon::new(
            LineString::from(plasma_region_coords),
            vec![], // No holes
        );
        // Loop over all boundary points and check if they are inside the plasma region
        let mut inside_plasma_region: Vec<usize> = Vec::with_capacity(n_boundary_points);
        for i_point in 0..n_boundary_points {
            let boundary_r_local: f64 = boundary_r[i_point];
            let boundary_z_local: f64 = boundary_z[i_point];
            let test_point: Point = Point::new(boundary_r_local, boundary_z_local);
            let inside: bool = plasma_region.contains(&test_point);
            if inside {
                inside_plasma_region.push(i_point);
            }
        }

        // Create a polygon for the LFS plasma region, 4 points
        let mut lfs_plasma_region_coords: Vec<Coord<f64>> = Vec::with_capacity(4);
        // Point 1: x-point
        lfs_plasma_region_coords.push(Coord { x: xpt_r, y: xpt_z });
        // Direction vector from x-point to the lfs minima
        let direction_dr: f64 = xpt_turning_lfs_minima_r - xpt_r;
        let direction_dz: f64 = xpt_turning_lfs_minima_z - xpt_z;
        // Normalize the direction vector
        let direction_norm: f64 = (direction_dr.powi(2) + direction_dz.powi(2)).sqrt();
        let direction_dr: f64 = direction_dr / direction_norm;
        let direction_dz: f64 = direction_dz / direction_norm;
        // Point 2: lower right
        lfs_plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0,
            y: xpt_z + direction_dz * 100.0, // 100m should alwasy be enough!
        });
        // Direction vector from x-point to the lfs minima
        let direction_dr: f64 = xpt_turning_plasma_maxima_r - xpt_r;
        let direction_dz: f64 = xpt_turning_plasma_maxima_z - xpt_z;
        // Normalize the direction vector
        let direction_norm: f64 = (direction_dr.powi(2) + direction_dz.powi(2)).sqrt();
        let direction_dr: f64 = direction_dr / direction_norm;
        let direction_dz: f64 = direction_dz / direction_norm;
        // Point 3: upper right
        lfs_plasma_region_coords.push(Coord {
            x: xpt_r + direction_dr * 100.0,
            y: xpt_z + direction_dz * 100.0,
        });
        // Point 4: x-point (again)
        lfs_plasma_region_coords.push(Coord { x: xpt_r, y: xpt_z });
        // Create the polygon
        let lfs_plasma_region: Polygon = Polygon::new(
            LineString::from(lfs_plasma_region_coords),
            vec![], // No holes
        );

        // Loop over all boundary points and check if they are inside the lfs plasma region
        let mut inside_lfs_plasma_region: Vec<usize> = Vec::with_capacity(n_boundary_points);
        for i_point in 0..n_boundary_points {
            let boundary_r_local: f64 = boundary_r[i_point];
            let boundary_z_local: f64 = boundary_z[i_point];
            let test_point: Point = Point::new(boundary_r_local, boundary_z_local);
            let inside: bool = lfs_plasma_region.contains(&test_point);
            if inside {
                inside_lfs_plasma_region.push(i_point);
            }
        }
        // Store the LFS plasma boundary points
        let mut lfs_plasma_boundary_points_r: Vec<f64> = boundary_r.select(Axis(0), &inside_lfs_plasma_region).to_vec();
        let mut lfs_plasma_boundary_points_z: Vec<f64> = boundary_z.select(Axis(0), &inside_lfs_plasma_region).to_vec();
        let mut lfs_plasma_boundary_v_dot_grad_psi: Vec<f64> = v_dot_grad_psi.select(Axis(0), &inside_lfs_plasma_region).to_vec();
        let n_lfs_points: usize = lfs_plasma_boundary_points_r.len();

        // Create the new plasma boundary
        let mut new_boundary_r: Vec<f64> = Vec::with_capacity(n_boundary_points + 2); // +2 for the x-points (start and end)
        let mut new_boundary_z: Vec<f64> = Vec::with_capacity(n_boundary_points + 2);
        new_boundary_r.push(xpt_r);
        new_boundary_z.push(xpt_z);
        // The contour is found using "marching squares", so for a continious contour the maximum distance between two points is
        // either d_r or d_z. The maximum distance is from one point to the next along the grid diagonal.
        let maximum_distance: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt() * 2.0001;

        // Traverse from the x-point arround the LFS
        let mut i_point_new: usize = 0;
        'lfs_contour_loop: for i_point in 0..n_lfs_points {
            let n_lfs_points_now: usize = lfs_plasma_boundary_points_r.len();
            let mut distance: Array1<f64> = Array1::from_elem(n_lfs_points_now, f64::NAN);
            for i_point_prime in 0..n_lfs_points_now {
                distance[i_point_prime] = ((lfs_plasma_boundary_points_r[i_point_prime] - new_boundary_r[i_point_new]).powi(2)
                    + (lfs_plasma_boundary_points_z[i_point_prime] - new_boundary_z[i_point_new]).powi(2))
                .sqrt();
            }
            let index_minimum: usize = distance.argmin().expect("check_boundary_points: error with argmin");
            // Check the distance to the next_point, but allow the first point (from the x-point) to be longer!
            if (i_point != 0) && (distance[index_minimum] > maximum_distance) {
                break 'lfs_contour_loop;
            }
            // if lfs_plasma_boundary_v_dot_grad_psi[index_minimum] < 0.0 {
            //     // Skip and remove points which are in the private flux region
            //     lfs_plasma_boundary_points_r.remove(index_minimum);
            //     lfs_plasma_boundary_points_z.remove(index_minimum);
            //     lfs_plasma_boundary_v_dot_grad_psi.remove(index_minimum);
            //     continue 'lfs_contour_loop;
            // }
            // Add the nearest point to the new_boundary
            new_boundary_r.push(lfs_plasma_boundary_points_r[index_minimum]);
            new_boundary_z.push(lfs_plasma_boundary_points_z[index_minimum]);
            // Add to new point counter
            i_point_new += 1;
            // Remove the point from the LFS plasma boundary points
            lfs_plasma_boundary_points_r.remove(index_minimum);
            lfs_plasma_boundary_points_z.remove(index_minimum);
            lfs_plasma_boundary_v_dot_grad_psi.remove(index_minimum);
        }
        // We don't use all of the LFS points (if they are too far away)
        let n_lfs_points: usize = new_boundary_r.len();

        // Remove the LFS points from the 'plasma region' points, leaving only the HFS points
        let inside_lfs_set: std::collections::HashSet<_> = inside_lfs_plasma_region.iter().cloned().collect();
        let indices_hfs_points: Vec<usize> = inside_plasma_region.iter().cloned().filter(|i| !inside_lfs_set.contains(i)).collect();

        // Get the HFS boundary points
        let mut hfs_boundary_points_r: Vec<f64> = boundary_r.select(Axis(0), &indices_hfs_points).to_vec();
        let mut hfs_boundary_points_z: Vec<f64> = boundary_z.select(Axis(0), &indices_hfs_points).to_vec();
        let mut hfs_boundary_v_dot_grad_psi: Vec<f64> = v_dot_grad_psi.select(Axis(0), &indices_hfs_points).to_vec();
        let n_hfs_points: usize = hfs_boundary_points_r.len();

        // Traverse from the LFS to the HFS
        let mut i_point_new: usize = 0;
        'hfs_contour_loop: for _i_point in 0..n_hfs_points {
            let n_hfs_points_now: usize = hfs_boundary_points_r.len();
            let mut distance: Array1<f64> = Array1::from_elem(n_hfs_points_now, f64::NAN);
            for i_point_prime in 0..n_hfs_points_now {
                distance[i_point_prime] = ((hfs_boundary_points_r[i_point_prime] - new_boundary_r[i_point_new + n_lfs_points - 1]).powi(2)
                    + (hfs_boundary_points_z[i_point_prime] - new_boundary_z[i_point_new + n_lfs_points - 1]).powi(2))
                .sqrt();
            }
            let index_minimum: usize = distance.argmin().expect("check_boundary_points: error with argmin");
            // Check the distance to the next_point
            if distance[index_minimum] > maximum_distance {
                break 'hfs_contour_loop;
            }
            if hfs_boundary_v_dot_grad_psi[index_minimum] < 0.0 {
                // Skip and remove points which are in the private flux region
                hfs_boundary_points_r.remove(index_minimum);
                hfs_boundary_points_z.remove(index_minimum);
                hfs_boundary_v_dot_grad_psi.remove(index_minimum);
                continue 'hfs_contour_loop;
            }
            // Add the nearest point to the new_boundary
            new_boundary_r.push(hfs_boundary_points_r[index_minimum]);
            new_boundary_z.push(hfs_boundary_points_z[index_minimum]);
            // Add to new point counter
            i_point_new += 1;
            // Remove the point from the HFS plasma boundary points
            hfs_boundary_points_r.remove(index_minimum);
            hfs_boundary_points_z.remove(index_minimum);
            hfs_boundary_v_dot_grad_psi.remove(index_minimum);
        }

        // TODO: Could add a test here to see if the contour is closed, i.e. if the last point is close to the x-point.
        // Add the x-point again to close the contour
        new_boundary_r.push(xpt_r);
        new_boundary_z.push(xpt_z);

        // Convert to Array1<f64>
        let new_boundary_r: Array1<f64> = Array1::from(new_boundary_r);
        let mut new_boundary_z: Array1<f64> = Array1::from(new_boundary_z);

        // Change back to USN
        if self.bounding_z > 0.0 {
            new_boundary_z = -new_boundary_z;
        }

        // Now set the new boundary contour
        self.n_points = new_boundary_r.len();
        self.boundary_r = new_boundary_r;
        self.boundary_z = new_boundary_z;
    }
}

fn find_minima_maxima_about_xpt(
    r: &Array1<f64>,
    z: &Array1<f64>,
    psi_2d: &Array2<f64>,
    xpt_r: f64,
    xpt_z: f64,
    lsn: bool,
) -> Result<(f64, f64, f64, f64, f64, f64), String> {
    // Grid spacing
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    // Radius of circle around x-point
    let circle_radius: f64 = (d_r.powi(2) + d_z.powi(2)).sqrt() / 1.7; // ~1cm with n_r=80, n_z=161
    // Number of points to sample on circle
    let n_theta: usize = 100;
    // Avoid double counting theta=0.0 and theta=2.0*PI
    let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_theta + 1).slice(s![..n_theta]).to_owned();

    // Create an interpolator for psi
    let psi_interpolator = Interp2D::builder(psi_2d.clone())
        .x(z.clone())
        .y(r.clone())
        .build()
        .expect("find_minima_maxima_about_xpt: Can't make Interp2D");

    // Sample points on the circle and collect (r, z, psi) values
    let mut circle_points: Vec<(f64, f64, f64)> = Vec::with_capacity(n_theta);
    let mut circle_psi: Array1<f64> = Array1::from_elem(n_theta, f64::NAN);
    for i_point in 0..n_theta {
        let theta_local: f64 = theta[i_point];
        let circle_r: f64 = xpt_r + circle_radius * theta_local.cos();
        let circle_z: f64 = xpt_z + circle_radius * theta_local.sin();
        let psi_local: f64 = psi_interpolator
            .interp_scalar(circle_z, circle_r)
            .expect("find_minima_maxima_about_xpt: Can't interpolate psi");
        circle_points.push((circle_r, circle_z, psi_local));
        circle_psi[i_point] = psi_local;
    }

    // Find local minima in `circle_psi` (`%` makes periodic array)
    let mut indices_minima: Vec<usize> = Vec::new();
    let mut indices_maxima: Vec<usize> = Vec::new();
    for i_point in 0..n_theta {
        let psi_previous: f64 = circle_psi[(i_point + n_theta - 1) % n_theta];
        let psi_current: f64 = circle_psi[i_point];
        let psi_next: f64 = circle_psi[(i_point + 1) % n_theta];
        // Look for turning points
        if psi_current < psi_previous && psi_current < psi_next {
            indices_minima.push(i_point);
        }
        if psi_current > psi_previous && psi_current > psi_next {
            indices_maxima.push(i_point);
        }
    }

    // If we have found less than two minima or two maxima, we cannot proceed
    // TODO: check if these exceptions were really a saddle point?
    if indices_minima.len() < 2 {
        // TODO: do we need to improve this - do we often not have two minima, when we should have?
        println!("check_boundary_points: indices_minima.len()={} but should be 2", indices_minima.len());
        return Err("check_boundary_points: indices_minima.len() != 2".to_string());
    }
    if indices_maxima.len() < 2 {
        // TODO: do we need to improve this - do we often not have two maxima, when we should have?
        println!("check_boundary_points: indices_maxima.len()={} but should be 2", indices_maxima.len());
        return Err("check_boundary_points: indices_maxima.len() != 2".to_string());
    }

    // println!("xpt_r= {}, xpt_z= {}", xpt_r, xpt_z);
    // If we have found more than two minima or two maxima, we will down select the minima and maxima, using a fit as a guide.
    if indices_minima.len() > 2 || indices_maxima.len() > 2 {
        // TODO: It might be necessary to use a FFT fit?
        // Fit: `circle_psi_fit = a * sin(2.0 * theta) + b * cos(2.0 * theta) + c`
        // Calculate sums needed for least squares fit with doubled frequency
        let n: f64 = n_theta as f64;
        let sum_sin: f64 = 0.0;
        let sum_cos: f64 = 0.0;
        let sum_sin2: f64 = n / 2.0;
        let sum_cos2: f64 = n / 2.0;
        let sum_sin_cos: f64 = 0.0;
        let sum_y: f64 = circle_psi.sum();
        let mut sum_y_sin: f64 = 0.0;
        let mut sum_y_cos: f64 = 0.0;
        for i_theta in 0..n_theta {
            let theta_local: f64 = theta[i_theta];
            let circle_psi_local: f64 = circle_psi[i_theta];
            sum_y_sin += circle_psi_local * (2.0 * theta_local).sin();
            sum_y_cos += circle_psi_local * (2.0 * theta_local).cos();
        }

        // Solve for coefficients, using this matrix equation:
        // [sum_sin2,    sum_sin_cos, sum_sin]   [a]   [sum_y_sin]
        // [sum_sin_cos, sum_cos2,    sum_cos] * [b] = [sum_y_cos]
        // [sum_sin,     sum_cos,     n      ]   [c]   [sum_y    ]
        let mat: Array2<f64> = Array2::from_shape_vec(
            (3, 3),
            vec![sum_sin2, sum_sin_cos, sum_sin, sum_sin_cos, sum_cos2, sum_cos, sum_sin, sum_cos, n],
        )
        .expect("find_minima_maxima_about_xpt: Failed to create matrix for least squares fit");
        let rhs: Array1<f64> = Array1::from_vec(vec![sum_y_sin, sum_y_cos, sum_y]);

        let solution: Array1<f64> = mat.solve_into(rhs).expect("find_minima_maxima_about_xpt: Failed to solve least squares");
        let a: f64 = solution[0];
        let b: f64 = solution[1];
        let _c: f64 = solution[2];

        // We will now transform the fit into: `circle_psi_fit = amplitude * sin(2 * theta + phase) + c`
        let _amplitude: f64 = (a.powi(2) + b.powi(2)).sqrt();
        // let phase: f64 = a.atan2(b);
        let phase: f64 = (a / b).atan();

        // The zeros of the derivative of the fit (i.e., turning points) are at:
        // `theta = (n * PI - phase) / 2`
        let mut theta_minimum: Vec<f64> = Vec::with_capacity(2);
        let mut theta_maximum: Vec<f64> = Vec::with_capacity(2);
        for i_turning_point in 0..4 {
            // Calculate the turning point
            let theta_turn: f64 = (phase + PI * (i_turning_point as f64)) / 2.0;
            // The % is used to map `theta_turn` into [0, 2*PI)
            let theta_turn_0_2pi: f64 = ((theta_turn % (2.0 * PI)) + (2.0 * PI)) % (2.0 * PI);
            // Even i_turning_point: minima, Odd i_turning_point: maxima (since sin(2θ + φ) alternates)
            if i_turning_point % 2 == 0 {
                // even
                theta_minimum.push(theta_turn_0_2pi);
            } else {
                // odd
                theta_maximum.push(theta_turn_0_2pi);
            }
        }
        // // Fit
        // let fit: Array1<f64> = a * (2.0 * &theta).sin() + b * (2.0 * &theta).cos() + _c;
        // println!("fit= {:?}", fit);

        // Sort the minima and maxima
        theta_minimum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        theta_maximum.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // println!("theta_minimum= {:?}", theta_minimum);
        // println!("theta_maximum= {:?}", theta_maximum);

        let mut index_fit_minimum: Array1<usize> = Array1::from_elem(2, usize::MAX);
        index_fit_minimum[0] = (&theta - theta_minimum[0])
            .abs()
            .argmin()
            .expect("find_minima_maxima_about_xpt: error with argmin for minima");
        index_fit_minimum[1] = (&theta - theta_minimum[1])
            .abs()
            .argmin()
            .expect("find_minima_maxima_about_xpt: error with argmin for minima");
        let mut index_fit_maximum: Array1<usize> = Array1::from_elem(2, usize::MAX);
        index_fit_maximum[0] = (&theta - theta_maximum[0])
            .abs()
            .argmin()
            .expect("find_minima_maxima_about_xpt: error with argmin for maxima");
        index_fit_maximum[1] = (&theta - theta_maximum[1])
            .abs()
            .argmin()
            .expect("find_minima_maxima_about_xpt: error with argmin for maxima");

        // Down select `indices_minima` and `indices_maxima` to the closest two points to the fitted minima and maxima
        // println!("(prior to comparing with fit) indices_minima= {:?}", indices_minima);
        // println!("index_fit_minimum= {:?}", index_fit_minimum);
        let n_minma: usize = indices_minima.len();
        let mut indices_minima_down_selected: Vec<usize> = Vec::with_capacity(2);
        if n_minma > 2 {
            for i_fit in 0..2 {
                // Down select `indices_minima` to the two closest to the fitted minima (accounting for periodicity)
                let n_minma: usize = indices_minima.len(); // need to recalculate, since we are removing elements
                let mut distance_from_fit_min_to_measured_min: Vec<usize> = Vec::with_capacity(n_minma);
                for i_measured in 0..n_minma {
                    let distance: usize = (index_fit_minimum[i_fit] as isize - indices_minima[i_measured] as isize).abs() as usize;
                    let distance_wrapped: usize = n_theta - distance;
                    if distance < distance_wrapped {
                        distance_from_fit_min_to_measured_min.push(distance);
                    } else {
                        distance_from_fit_min_to_measured_min.push(distance_wrapped);
                    }
                }
                // Sort the distances and take the two smallest
                let i_closest: usize = distance_from_fit_min_to_measured_min
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.cmp(b.1))
                    .map(|(idx, _)| idx)
                    .expect("No minimum found in distance_from_fit_min_to_measured_min");
                indices_minima_down_selected.push(indices_minima[i_closest]);
                indices_minima.remove(i_closest);
            }
        }
        // Store the down selected minima
        indices_minima = indices_minima_down_selected;
        // println!("(after comparing with fit) indices_minima= {:?}", indices_minima);

        // println!("index_fit_maximum= {:?}", index_fit_maximum);
        // println!("(prior to comparing with fit) indices_maxima= {:?}", indices_maxima);
        let n_maxma: usize = indices_maxima.len();
        let mut indices_maxima_down_selected: Vec<usize> = Vec::with_capacity(2);
        if n_maxma > 2 {
            for i_fit in 0..2 {
                // Down select `indices_maxima` to the two closest to the fitted maxima (accounting for periodicity)
                let n_maxma: usize = indices_maxima.len(); // need to recalculate, since we are removing elements
                let mut distance_from_fit_max_to_measured_max: Vec<usize> = Vec::with_capacity(n_maxma);
                for i_measured in 0..n_maxma {
                    let distance: usize = (index_fit_maximum[i_fit] as isize - indices_maxima[i_measured] as isize).abs() as usize;
                    let distance_wrapped: usize = n_theta - distance;
                    if distance < distance_wrapped {
                        distance_from_fit_max_to_measured_max.push(distance);
                    } else {
                        distance_from_fit_max_to_measured_max.push(distance_wrapped);
                    }
                }
                // Sort the distances and take the two smallest
                let i_closest: usize = distance_from_fit_max_to_measured_max
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.cmp(b.1))
                    .map(|(idx, _)| idx)
                    .expect("No minimum found in distance_from_fit_max_to_measured_max");
                indices_maxima_down_selected.push(indices_maxima[i_closest]);
                indices_maxima.remove(i_closest);
            }
        }
        // Store the down selected maxima
        indices_maxima = indices_maxima_down_selected;
        // println!("(after comparing with fit) indices_maxima= {:?}", indices_maxima);
    }

    // // Search for the maxima in the `circle_psi`.
    // let mut indices_maxima: Array1<usize> = Array1::from_vec(vec![usize::MAX, usize::MAX]);  // if these are used without being set, they will cause a panic
    // // maxima_1 will be between fitted_minimum_1 and fitted_minimum_2.
    // println!("indices_minima_to_minima_1");
    // let indices_minima_to_minima_1: Range<usize> = index_fit_minimum[0]..index_fit_minimum[1];
    // let mut maximum_psi: f64 = -f64::INFINITY;
    // for i_theta in indices_minima_to_minima_1 {
    //     println!("{i_theta}");
    //     if circle_psi[i_theta] > maximum_psi {
    //         maximum_psi = circle_psi[i_theta];
    //         indices_maxima[0] = i_theta;
    //     }
    // }
    // // maxima_2 will be between fitted_maximum_2 and fitted_maximum_1 (note we are wrapping around the circle).
    // let mut indices_minima_to_minima_2: Vec<usize> = vec![];
    // for i_theta in 0..n_theta {
    //     if !(i_theta > index_fit_minimum[0] && i_theta < index_fit_minimum[1]) {
    //         // skip over between index_fit_minimum[0] and index_fit_maximum[1]
    //         indices_minima_to_minima_2.push(i_theta);
    //     }
    // }
    // println!("indices_minima_to_minima_2");
    // let mut maximum_psi: f64 = -f64::INFINITY;
    // for i_theta in indices_minima_to_minima_2 {
    //     println!("{i_theta}");
    //     if circle_psi[i_theta] > maximum_psi {
    //         maximum_psi = circle_psi[i_theta];
    //         indices_maxima[1] = i_theta;
    //     }
    // }

    // // Search for the minima in the `circle_psi`
    // let mut indices_minima: Array1<usize> = Array1::from_vec(vec![usize::MAX, usize::MAX]);  // if these are used without being set, they will cause a panic
    // // minima_1 will be between fitted_maxima_1 and fitted_minima_2.
    // let indices_maxima_to_minima_1: Range<usize> = index_fit_maximum[0]..index_fit_maximum[1];
    // let mut minimum_psi: f64 = f64::INFINITY;
    // for i_theta in indices_maxima_to_minima_1 {
    //     if circle_psi[i_theta] < minimum_psi {
    //         minimum_psi = circle_psi[i_theta];
    //         indices_minima[0] = i_theta;
    //     }
    // }
    // // minima_2 will be between fitted_minima_2 and fitted_maxima_1 (note we are wrapping around the circle).
    // let mut indices_maxima_to_maxima_2: Vec<usize> = vec![];
    // for i_theta in 0..n_theta {
    //     if !(i_theta > index_fit_maximum[0] && i_theta < index_fit_maximum[1]) {
    //         // skip over between index_fit_maximum[0] and index_fit_minimum[1]
    //         indices_maxima_to_maxima_2.push(i_theta);
    //     }
    // }
    // minimum_psi = f64::INFINITY;
    // for i_theta in indices_maxima_to_maxima_2 {
    //     if circle_psi[i_theta] < minimum_psi {
    //         minimum_psi = circle_psi[i_theta];
    //         indices_minima[1] = i_theta;
    //     }
    // }

    // println!("indices_minima= {:?}", indices_minima);
    // println!("indices_maxima= {:?}", indices_maxima);
    // println!("circle_psi= {:#?}", circle_psi);
    // println!("  ");

    // Find LFS and HFS minima
    let index_minima_hfs: usize;
    let index_minima_lfs: usize;
    if circle_points[indices_minima[0]].0 < circle_points[indices_minima[1]].0 {
        index_minima_hfs = indices_minima[0];
        index_minima_lfs = indices_minima[1];
    } else {
        index_minima_hfs = indices_minima[1];
        index_minima_lfs = indices_minima[0];
    }

    // Find plasma and private flux maxima
    let index_maxima_plasma: usize;
    // let index_maxima_private_flux: usize;
    // Note, logic is for LSN
    if lsn {
        if circle_points[indices_maxima[0]].1 > circle_points[indices_maxima[1]].1 {
            //
            index_maxima_plasma = indices_maxima[0];
            // index_maxima_private_flux = indices_maxima[1];
        } else {
            index_maxima_plasma = indices_maxima[1];
            // index_maxima_private_flux = indices_maxima[0];
        }
    } else {
        // USN
        if circle_points[indices_maxima[0]].1 < circle_points[indices_maxima[1]].1 {
            //
            index_maxima_plasma = indices_maxima[0];
            // index_maxima_private_flux = indices_maxima[1];
        } else {
            index_maxima_plasma = indices_maxima[1];
            // index_maxima_private_flux = indices_maxima[0];
        }
    }

    // Collet the reuslts
    let lfs_minima_r: f64 = circle_points[index_minima_lfs].0;
    let lfs_minima_z: f64 = circle_points[index_minima_lfs].1;
    let hfs_minima_r: f64 = circle_points[index_minima_hfs].0;
    let hfs_minima_z: f64 = circle_points[index_minima_hfs].1;
    let plasma_maxima_r: f64 = circle_points[index_maxima_plasma].0;
    let plasma_maxima_z: f64 = circle_points[index_maxima_plasma].1;

    // panic!("stopping for debugging");

    return Ok((lfs_minima_r, lfs_minima_z, hfs_minima_r, hfs_minima_z, plasma_maxima_r, plasma_maxima_z));
}

#[test]
fn test_calculate_plasma_volume() {
    // In this test we create two rectangular boxes and check that the volume is calculated correctly.

    use approx::assert_abs_diff_eq;

    let boundary_r: Array1<f64> = Array1::from(vec![0.30, 0.77, 1.22, 1.22, 0.77, 0.77, 0.30]);
    let boundary_z: Array1<f64> = Array1::from(vec![-0.15, -0.15, -0.15, -0.10, -0.10, 0.35, 0.35]);
    let n_points: usize = boundary_r.len();

    // box 1
    let box_1_d_r: f64 = 0.77 - 0.3;
    let box_1_d_z: f64 = 0.35 - (-0.15);
    let box_1_r_center: f64 = 0.5 * (0.3 + 0.77);

    // box 2
    let box_2_d_r: f64 = 1.22 - 0.77;
    let box_2_d_z: f64 = -0.10 - (-0.15);
    let box_2_r_center: f64 = 0.5 * (1.22 + 0.77);

    // Calculate the analytic volume of the two boxes
    let plasma_volume_analytic: f64 = 2.0 * PI * (box_1_d_r * box_1_d_z * box_1_r_center + box_2_d_r * box_2_d_z * box_2_r_center);

    let mut bounary_coords: Vec<Coord> = vec![];

    for i_boundary in 0..n_points {
        bounary_coords.push(Coord {
            x: boundary_r[i_boundary],
            y: boundary_z[i_boundary],
        });
    }

    let boundary_polygon: Polygon = Polygon::new(LineString::from(bounary_coords), vec![]);

    let mut contour: BoundaryContour = BoundaryContour {
        boundary_polygon, // technically this value doesn't matter for this test
        boundary_r,
        boundary_z,
        n_points,
        bounding_psi: 1.2345,             // value doesn't matter for this test
        bounding_r: 0.3,                  // value doesn't matter for this test
        bounding_z: -0.15,                // value doesn't matter for this test
        fraction_inside_vessel: 0.0,      // value doesn't matter for this test
        xpt_diverted: false,              // value doesn't matter for this test
        plasma_volume: None,              // volume calculated using method
        mask: None,                       // mask calculated using method
        secondary_xpt_r: f64::NAN,        // value doesn't matter for this test
        secondary_xpt_z: f64::NAN,        // value doesn't matter for this test
        secondary_xpt_distance: f64::NAN, // value doesn't matter for this test
    };

    contour.calculate_plasma_volume();

    let plasma_volume: f64 = contour.plasma_volume.expect("test_calculate_plasma_volume: plasma_volume should not be None");
    assert_abs_diff_eq!(plasma_volume, plasma_volume_analytic, epsilon = 1e-10);
}
