//! `time_slice(itime)/profiles_1d/geometric_axis/r` and `.../geometric_axis/z`

use super::super::boundary::geometry::{ExtremalPoints, contour_extremal_points, x_points};
use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use super::super::psi_interpolator::PsiInterpolator;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the geometric axis of every flux surface, and store it in the time-slice.
///
/// The data dictionary defines it as the centre of the bounding box of the surface:
///
/// ```text
/// geometric_axis/r = (r_min + r_max) / 2
/// geometric_axis/z = (z_min + z_max) / 2
/// ```
///
/// which is how `boundary/geometric_axis` is measured too. The bounding box is that of
/// `boundary::geometry::contour_extremal_points`, as for the elongation: refined onto the flux surface for the
/// surfaces inside the boundary, and the vertices' for the last surface, the boundary outline, so the last point
/// of each profile is the corresponding `boundary` scalar.
///
/// Note that this is the centre of the bounding box, not the centroid: a `D`-shaped surface has
/// more of its area on the inboard side, so the two do not agree. The bounding box is what the data
/// dictionary asks for, and it is what `boundary/minor_radius` and the triangularity are measured
/// against, so it keeps `r_geo + r_minor = r_max` true.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `profiles_1d/geometric_axis` nodes are written
///   into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
///
/// # The magnetic axis and missing surfaces
///
/// A surface which could not be traced is left as NaN, exactly as the enclosed volume is.
///
/// The magnetic axis is a point rather than a contour, and the surfaces shrink onto it, so the
/// geometric axis there is the magnetic axis itself. Unlike the elongation, this limit needs no
/// curvature argument: it holds whatever shape the surfaces take.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
    let (xpts_r, xpts_z): (Array1<f64>, Array1<f64>) = x_points(time_slice);

    let mut geometric_axis_r_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut geometric_axis_z_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // The surfaces shrink onto the magnetic axis; see the note in this function's documentation
    geometric_axis_r_profile[0] = time_slice.global_quantities.magnetic_axis.r;
    geometric_axis_z_profile[0] = time_slice.global_quantities.magnetic_axis.z;

    // Don't do the first point
    'psi_norm_loop: for i_psi_norm in 1..n_psi_norm {
        // The last surface is the boundary outline, whose extent is its vertices'
        let is_extent_refined: bool = i_psi_norm != n_psi_norm - 1;
        let psi_contour: f64 = if is_extent_refined {
            psi_a + psi_norm[i_psi_norm] * (psi_b - psi_a)
        } else {
            psi_b
        };

        // A flux surface which could not be found, or carries a NaN, is left as NaN
        let Some(extremal_points) = contour_extremal_points(
            &flux_surfaces[i_psi_norm].r,
            &flux_surfaces[i_psi_norm].z,
            psi_contour,
            &psi_interpolator,
            &xpts_r,
            &xpts_z,
            is_extent_refined,
        ) else {
            continue 'psi_norm_loop;
        };
        let ExtremalPoints {
            r_max, r_min, z_max, z_min, ..
        } = extremal_points;

        geometric_axis_r_profile[i_psi_norm] = (r_min + r_max) / 2.0;
        geometric_axis_z_profile[i_psi_norm] = (z_min + z_max) / 2.0;
    }

    time_slice.profiles_1d.geometric_axis.r = geometric_axis_r_profile;
    time_slice.profiles_1d.geometric_axis.z = geometric_axis_z_profile;
}

#[cfg(test)]
mod tests {
    use super::super::super::boundary::geometry::{time_slice_with_elliptical_psi, traced_contour};
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::flux_surfaces::{calculate_flux_surfaces, empty_flux_surface, shape_only_flux_surface};
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::f64::consts::PI;

    /// A time-slice with a grid, whose surfaces the tests replace with their own; `boundary/psi` is NaN, so there is no flux for the
    /// points of those surfaces, which are not contours of its `psi`, to be refined onto
    fn time_slice_for_given_surfaces(psi_norm: Array1<f64>, mag_r: f64, mag_z: f64) -> EquilibriumTimeSlice {
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(0.5, 0.0, 25.0, 0.0, 25.0);
        time_slice.boundary.psi = f64::NAN;
        time_slice.profiles_1d.psi_norm = psi_norm;
        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice
    }

    /// A closed elliptical contour
    fn elliptical_flux_surface(r_geo: f64, z_geo: f64, r_minor: f64, kappa: f64) -> FluxSurface {
        // `n_theta` divisible by 4, so that the extremal points are exactly on the contour
        let n_theta: usize = 1000;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);

        shape_only_flux_surface(r_geo + r_minor * theta.mapv(f64::cos), z_geo + kappa * r_minor * theta.mapv(f64::sin))
    }

    #[test]
    fn each_surface_gives_the_centre_of_its_bounding_box() {
        let mag_r: f64 = 0.47;
        let mag_z: f64 = 0.03;
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_given_surfaces(array![0.0, 0.5, 1.0], mag_r, mag_z);

        // The magnetic axis, one surface which could not be traced, and one ellipse which is
        // deliberately not centred on the magnetic axis, as a Shafranov-shifted surface is not
        let flux_surfaces: Vec<FluxSurface> = vec![empty_flux_surface(), empty_flux_surface(), elliptical_flux_surface(0.52, -0.01, 0.2, 1.7)];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        let geometric_axis_r_profile: Array1<f64> = time_slice.profiles_1d.geometric_axis.r;
        let geometric_axis_z_profile: Array1<f64> = time_slice.profiles_1d.geometric_axis.z;

        // The surfaces shrink onto the magnetic axis
        assert_abs_diff_eq!(geometric_axis_r_profile[0], mag_r, epsilon = 1e-15);
        assert_abs_diff_eq!(geometric_axis_z_profile[0], mag_z, epsilon = 1e-15);

        // A surface which could not be traced stays NaN
        assert!(geometric_axis_r_profile[1].is_nan());
        assert!(geometric_axis_z_profile[1].is_nan());

        // The centre of the ellipse, not the magnetic axis
        assert_abs_diff_eq!(geometric_axis_r_profile[2], 0.52, epsilon = 1e-6);
        assert_abs_diff_eq!(geometric_axis_z_profile[2], -0.01, epsilon = 1e-6);
    }

    /// The bounding box is not the centroid: a triangle leans one way but its bounding box does not
    #[test]
    fn the_bounding_box_is_not_the_centroid() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_given_surfaces(array![0.0, 1.0], 0.5, 0.0);

        // A closed triangle spanning r in [0.2, 0.8] and z in [-0.1, 0.3]. Its centroid is at
        // r = (0.2 + 0.8 + 0.2) / 3 = 0.4, but the centre of its bounding box is 0.5
        let flux_surface: FluxSurface = shape_only_flux_surface(array![0.2, 0.8, 0.2, 0.2], array![-0.1, -0.1, 0.3, -0.1]);
        let flux_surfaces: Vec<FluxSurface> = vec![empty_flux_surface(), flux_surface];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.r[1], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.z[1], 0.1, epsilon = 1e-15);
    }

    #[test]
    fn a_contour_carrying_a_nan_is_not_measured() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_given_surfaces(array![0.0, 1.0], 0.5, 0.0);

        let flux_surface: FluxSurface = shape_only_flux_surface(array![0.2, 0.8, f64::NAN, 0.2], array![-0.1, -0.1, 0.3, -0.1]);
        let flux_surfaces: Vec<FluxSurface> = vec![empty_flux_surface(), flux_surface];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        assert!(time_slice.profiles_1d.geometric_axis.r[1].is_nan());
    }

    /// The surfaces of a tilted, off-centre elliptical `psi` are all centred on its centre; inside the boundary, found along rays and
    /// with their extent refined, they give it to rounding
    #[test]
    fn ray_traced_surfaces_give_the_centre_of_their_refined_bounding_box() {
        let centre_r: f64 = 0.5;
        let centre_z: f64 = 0.05;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(centre_r, centre_z, 30.0, 8.0, 12.0);
        time_slice.profiles_1d.psi_norm = array![0.0, 0.3, 0.6, 1.0];
        let (boundary_r, boundary_z): (Array1<f64>, Array1<f64>) = traced_contour(&time_slice, 0.0);
        time_slice.boundary.outline.r = boundary_r;
        time_slice.boundary.outline.z = boundary_z;

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = calculate_flux_surfaces(&time_slice);
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        for i_psi_norm in 0..3 {
            assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.r[i_psi_norm], centre_r, epsilon = 1e-10);
            assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.z[i_psi_norm], centre_z, epsilon = 1e-10);
        }
        // The boundary outline's extent is its vertices', on the grid lines
        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.r[3], centre_r, epsilon = 1e-3);
        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.z[3], centre_z, epsilon = 1e-3);
    }
}
