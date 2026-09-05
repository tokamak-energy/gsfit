//! `time_slice(itime)/boundary/outline/r` and `.../z`

use crate::plasma_geometry::MarchingContour;
use crate::plasma_geometry::marching_squares::marching_squares;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Trace the last closed flux surface, and store it in the time-slice.
///
/// The contour is the `psi = psi_b` isoline, traced by marching squares over the plasma mask. For a
/// diverted plasma the X-point is passed in so that the trace can be cut there, rather than
/// following the separatrix out along the divertor legs.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/outline/r` and `.../z` are written into it
///
/// A time-slice which failed to converge gets an empty outline, which is how the old
/// post-processor represented "no boundary" too.
///
/// Note: the solver already traces this contour while testing whether a candidate boundary point is
/// viable, and `find_boundary` traces it again, but neither keeps the result - only the mask and
/// the bounding point are stored. So it is traced here a third time. The solver only needs the
/// mask, so the cheaper fix is to stop tracing during the iterations at all.
pub fn epp_equilibrium_time_slice_boundary_outline(time_slice: &mut EquilibriumTimeSlice) {
    // A slice which did not converge has no boundary to trace
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.boundary.outline.r = Some(Array1::from_elem(0, f64::NAN));
        time_slice.boundary.outline.z = Some(Array1::from_elem(0, f64::NAN));
        return;
    }

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();
    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();

    let psi_b: f64 = time_slice.boundary.psi.unwrap();
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    // `boundary/type` is 0 for a limited plasma and 1 for a diverted one. Only a diverted plasma
    // has an X-point, and when it does the bounding point *is* the X-point
    let xpt_diverted: bool = time_slice.boundary.r#type.unwrap() == 1;
    let xpt_r_or_none: Option<f64>;
    let xpt_z_or_none: Option<f64>;
    if xpt_diverted {
        xpt_r_or_none = Some(time_slice.boundary.bounding.r.unwrap());
        xpt_z_or_none = Some(time_slice.boundary.bounding.z.unwrap());
    } else {
        xpt_r_or_none = None;
        xpt_z_or_none = None;
    }

    let boundary_contour: MarchingContour = marching_squares(
        r,
        z,
        psi_2d,
        d_psi_d_r_2d,
        d_psi_d_z_2d,
        psi_b,
        mask_2d,
        xpt_r_or_none,
        xpt_z_or_none,
        mag_r,
        mag_z,
    );

    time_slice.boundary.outline.r = Some(boundary_contour.r);
    time_slice.boundary.outline.z = Some(boundary_contour.z);
}
