//! The closed flux surfaces, as (R, Z) contours.
//!
//! Unlike the other `epp_*` helpers this one fills no data-dictionary path. The flux surfaces are
//! an intermediate quantity: several profiles are line-integrals around them, so they are worth
//! calculating once and passing to the helpers which need them, rather than re-contouring per
//! quantity.

use contour::ContourBuilder;
use geo::{Contains, Coord, LineString, Point, Polygon};
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// A single closed flux surface, as an (R, Z) contour.
///
/// A surface which could not be found is stored with zero points, rather than being absent, so that
/// `flux_surfaces[i_psi_norm]` always corresponds to `psi_norm[i_psi_norm]`.
#[derive(Clone)]
pub struct FluxSurface {
    pub r: Array1<f64>,
    pub z: Array1<f64>,
}

/// Find the closed flux surface at each `psi_norm`, and return them.
///
/// Each surface is found by contouring `psi_2d` at that surface's `psi`. A contour is accepted only
/// if every one of its points lies inside the plasma boundary, which is what stops a
/// private-flux-region contour at the same `psi` from being mistaken for the flux surface.
///
/// The two ends are special cases. The magnetic axis (`psi_norm = 0`) is a point rather than a
/// surface, so it is left empty. The last closed flux surface is taken from the boundary outline,
/// because a contour drawn on the boundary is rejected by the containment test above.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// One `FluxSurface` per `psi_norm`, in the same order.
///
/// **Must run after `epp_equilibrium_time_slice_boundary_outline`**, which supplies the boundary
/// polygon.
pub fn epp_flux_surfaces(time_slice: &EquilibriumTimeSlice) -> Vec<FluxSurface> {
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    let n_psi_norm: usize = psi_norm.len();

    let flux_surface_empty: FluxSurface = FluxSurface {
        r: Array1::from_elem(0, f64::NAN),
        z: Array1::from_elem(0, f64::NAN),
    };
    let mut flux_surfaces: Vec<FluxSurface> = vec![flux_surface_empty; n_psi_norm];

    // A slice which did not converge has no flux surfaces to find
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        return flux_surfaces;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();

    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let boundary_r: &Array1<f64> = time_slice.boundary.outline.r.as_ref().unwrap();
    let boundary_z: &Array1<f64> = time_slice.boundary.outline.z.as_ref().unwrap();

    // Sizes and grid variables
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];
    let r_origin: f64 = r[0];
    let z_origin: f64 = z[0];

    // Create an empty contour grid
    let contour_grid: ContourBuilder = ContourBuilder::new(n_r, n_z, true) // x dim., y dim., smoothing
        .x_step(d_r)
        .y_step(d_z)
        .x_origin(r_origin - d_r / 2.0)
        .y_origin(z_origin - d_z / 2.0);

    let psi_2d_flattened: Vec<f64> = psi_2d.iter().cloned().collect();

    // Create the plasma boundary polygon
    let boundary_polygon_coordinates: Vec<Coord<f64>> = boundary_r.iter().zip(boundary_z.iter()).map(|(&x, &y)| Coord { x, y }).collect();
    let boundary_polygon: Polygon = Polygon::new(
        LineString::from(boundary_polygon_coordinates),
        vec![], // No holes
    );

    // Add on the last closed flux surface
    let flux_surface_last_closed: FluxSurface = FluxSurface {
        r: boundary_r.to_owned(),
        z: boundary_z.to_owned(),
    };
    flux_surfaces[n_psi_norm - 1] = flux_surface_last_closed;

    // Loop over psi_n
    'psi_n_loop: for i_psi_norm in 1..n_psi_norm {
        let psi_local: f64 = psi_norm[i_psi_norm] * (psi_b - psi_a) + psi_a;

        let flux_surface_contours_tmp: Vec<contour::Contour> = contour_grid.contours(&psi_2d_flattened, &[psi_local]).unwrap();

        let flux_surface_contours: &geo_types::MultiPolygon = flux_surface_contours_tmp[0].geometry(); // The [0] is because I have only supplied one threshold

        // Loop over all contours and find the one which is inside (r_cur, z_cur)
        let n_contour: usize = flux_surface_contours.iter().count();

        'contour_loop: for i_contour in 0..n_contour {
            let fs_contour: &Polygon = flux_surface_contours.iter().nth(i_contour).unwrap();

            // Test if all the points are inside the plasma boundary
            for coord in fs_contour.exterior() {
                let fs_r: f64 = coord.x;
                let fs_z: f64 = coord.y;
                let point: Point = Point::new(fs_r, fs_z);

                let inside_boundary: bool = boundary_polygon.contains(&point);
                if !inside_boundary {
                    // Not a valid contour, so try the next contour
                    continue 'contour_loop;
                }
            }

            // Store the flux surface
            let fs_r: Array1<f64> = fs_contour.exterior().coords().map(|coord| coord.x).collect::<Array1<f64>>();
            let fs_z: Array1<f64> = fs_contour.exterior().coords().map(|coord| coord.y).collect::<Array1<f64>>();
            let flux_surface: FluxSurface = FluxSurface { r: fs_r, z: fs_z };
            flux_surfaces[i_psi_norm] = flux_surface;

            // Go to the next psi_n
            continue 'psi_n_loop;
        }
    }

    return flux_surfaces;
}
