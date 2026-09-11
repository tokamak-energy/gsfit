//! The closed flux surfaces, as (R, Z) contours.
//!
//! Unlike the other `epp_*` helpers this one fills no data-dictionary path. The flux surfaces are
//! an intermediate quantity: several profiles are line-integrals around them, so they are worth
//! calculating once and passing to the helpers which need them, rather than re-contouring per
//! quantity.

use super::constant_values::ConstantValues;
use super::intermediate_values::IntermediateValues;
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

/// Find the closed flux surfaces and store them on the shared intermediate values.
///
/// This is the uniform-signature entry point the post-processor dispatches; the surfaces
/// themselves are found by `calculate_flux_surfaces`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(time_slice);
    intermediate_values.flux_surfaces = flux_surfaces;
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
fn calculate_flux_surfaces(time_slice: &EquilibriumTimeSlice) -> Vec<FluxSurface> {
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    let flux_surface_empty: FluxSurface = empty_flux_surface();
    let mut flux_surfaces: Vec<FluxSurface> = vec![flux_surface_empty; n_psi_norm];

    // A slice which did not converge has no flux surfaces to find
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        return flux_surfaces;
    }

    let boundary_r: &Array1<f64> = &time_slice.boundary.outline.r;
    let boundary_z: &Array1<f64> = &time_slice.boundary.outline.z;

    // Add on the last closed flux surface
    let flux_surface_last_closed: FluxSurface = FluxSurface {
        r: boundary_r.to_owned(),
        z: boundary_z.to_owned(),
    };
    flux_surfaces[n_psi_norm - 1] = flux_surface_last_closed;

    // Loop over psi_norm. The last surface is skipped, because it has already been taken from the
    // boundary outline, which is the better answer there: it carries the x-point
    for i_psi_norm in 1..n_psi_norm - 1 {
        flux_surfaces[i_psi_norm] = calculate_at_psi_norm(time_slice, psi_norm[i_psi_norm]);
    }

    flux_surfaces
}

/// Find the closed flux surface at one normalised poloidal flux strictly inside the boundary.
///
/// The contour is selected by flood-filling outwards from the magnetic axis before marching its
/// edge, so a disconnected private-flux contour at the same flux cannot be returned accidentally.
/// A value outside `0 < psi_norm < 1`, a failed time-slice, or a contour with fewer than three
/// distinct points returns an empty surface.
pub(in crate::equilibrium_post_processor) fn calculate_at_psi_norm(time_slice: &EquilibriumTimeSlice, psi_norm: f64) -> FluxSurface {
    if !psi_norm.is_finite() || psi_norm <= 0.0 || psi_norm >= 1.0 {
        return empty_flux_surface();
    }

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        return empty_flux_surface();
    }

    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
    let d_psi_d_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_r;
    let d_psi_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_z;

    let psi_b: f64 = time_slice.boundary.psi;
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let psi_local: f64 = psi_norm * (psi_b - psi_a) + psi_a;
    let mask_2d: Array2<f64> = flood_fill_mask_at_psi(r, z, psi_2d, psi_local, mag_r, mag_z);

    // An interior flux surface is smooth, so there is no X-point to supply.
    let flux_surface_contour: MarchingContour = marching_squares(r, z, psi_2d, d_psi_d_r_2d, d_psi_d_z_2d, psi_local, &mask_2d, None, None, mag_r, mag_z);
    if flux_surface_contour.n < 4 {
        return empty_flux_surface();
    }

    FluxSurface {
        r: flux_surface_contour.r,
        z: flux_surface_contour.z,
    }
}

fn empty_flux_surface() -> FluxSurface {
    FluxSurface {
        r: Array1::from_elem(0, f64::NAN),
        z: Array1::from_elem(0, f64::NAN),
    }
}
