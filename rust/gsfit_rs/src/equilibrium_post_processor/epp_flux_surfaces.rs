//! The closed flux surfaces, as (R, Z) contours.
//!
//! Unlike the other `epp_*` helpers this one fills no data-dictionary path. The flux surfaces are
//! an intermediate quantity: several profiles are line-integrals around them, so they are worth
//! calculating once and passing to the helpers which need them, rather than re-contouring per
//! quantity.

use crate::plasma_geometry::MarchingContour;
use crate::plasma_geometry::flood_fill_mask_at_psi;
use crate::plasma_geometry::marching_squares::marching_squares;
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
/// Each surface is the `psi = psi_norm * (psi_b - psi_a) + psi_a` contour **containing the magnetic
/// axis**, and it is found the same way the boundary is: flood fill the region enclosed by that
/// flux value outwards from the magnetic axis, then march its edge. Seeding the fill at the axis is
/// what picks out the one surface wanted - a private-flux region at the same `psi` is never
/// reached, so it can never be mistaken for the flux surface.
///
/// The two ends are special cases. The magnetic axis (`psi_norm = 0`) is a point rather than a
/// surface, so it is left empty. The last closed flux surface is taken from the boundary outline,
/// which is traced with the x-point in it rather than from the grid alone.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// One `FluxSurface` per `psi_norm`, in the same order.
///
/// **Must run after `epp_equilibrium_time_slice_boundary_outline`**, which supplies the last closed
/// flux surface.
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
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();

    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let boundary_r: &Array1<f64> = time_slice.boundary.outline.r.as_ref().unwrap();
    let boundary_z: &Array1<f64> = time_slice.boundary.outline.z.as_ref().unwrap();
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    // Add on the last closed flux surface
    let flux_surface_last_closed: FluxSurface = FluxSurface {
        r: boundary_r.to_owned(),
        z: boundary_z.to_owned(),
    };
    flux_surfaces[n_psi_norm - 1] = flux_surface_last_closed;

    // Loop over psi_n. The last surface is skipped, because it has already been taken from the
    // boundary outline, which is the better answer there: it carries the x-point
    'psi_n_loop: for i_psi_norm in 1..n_psi_norm - 1 {
        let psi_local: f64 = psi_norm[i_psi_norm] * (psi_b - psi_a) + psi_a;

        let mask_2d: Array2<f64> = flood_fill_mask_at_psi(r, z, psi_2d, psi_local, mag_r, mag_z);

        // No x-point is passed: an inner flux surface is a smooth closed curve, and the x-point
        // only sits on the boundary, which this loop does not reach
        let flux_surface_contour: MarchingContour = marching_squares(r, z, psi_2d, d_psi_d_r_2d, d_psi_d_z_2d, psi_local, &mask_2d, None, None, mag_r, mag_z);

        // A ring needs 3 distinct points to enclose an area, plus the repeated first point
        if flux_surface_contour.n < 4 {
            continue 'psi_n_loop;
        }

        flux_surfaces[i_psi_norm] = FluxSurface {
            r: flux_surface_contour.r,
            z: flux_surface_contour.z,
        };
    }

    return flux_surfaces;
}
