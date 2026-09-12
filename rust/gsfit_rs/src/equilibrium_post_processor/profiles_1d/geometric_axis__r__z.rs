//! `time_slice(itime)/profiles_1d/geometric_axis/r` and `.../geometric_axis/z`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
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
/// which is how `boundary/geometric_axis` is measured too, so the last point of each profile is the
/// corresponding `boundary` scalar.
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

    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

    let mut geometric_axis_r_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut geometric_axis_z_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // A slice which did not converge has no flux surfaces to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        time_slice.profiles_1d.geometric_axis.r = geometric_axis_r_profile;
        time_slice.profiles_1d.geometric_axis.z = geometric_axis_z_profile;
        return;
    }

    // The surfaces shrink onto the magnetic axis; see the note in this function's documentation
    geometric_axis_r_profile[0] = time_slice.global_quantities.magnetic_axis.r;
    geometric_axis_z_profile[0] = time_slice.global_quantities.magnetic_axis.z;

    // Don't do the first point
    'psi_norm_loop: for i_psi_norm in 1..n_psi_norm {
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;

        // A flux surface which could not be found is stored with zero points, and is left as NaN
        if fs_r.is_empty() {
            continue 'psi_norm_loop;
        }

        let (r_min, r_max): (f64, f64) = epp_bounding_range(fs_r);
        let (z_min, z_max): (f64, f64) = epp_bounding_range(fs_z);

        geometric_axis_r_profile[i_psi_norm] = (r_min + r_max) / 2.0;
        geometric_axis_z_profile[i_psi_norm] = (z_min + z_max) / 2.0;
    }

    time_slice.profiles_1d.geometric_axis.r = geometric_axis_r_profile;
    time_slice.profiles_1d.geometric_axis.z = geometric_axis_z_profile;
}

/// The smallest and largest value along one coordinate of a contour.
///
/// # Arguments
/// * `values` - one coordinate of the contour, `r` or `z` [metre]
///
/// # Returns
/// * `(value_min, value_max)` - the ends of the bounding box along that coordinate [metre]
///
/// Defensive programming: when a time-slice has failed a contour can carry NaN's or junk. A single
/// bad point makes the whole bounding box meaningless rather than only shifting one edge of it, so
/// it turns the surface into NaN instead of being skipped over.
fn epp_bounding_range(values: &Array1<f64>) -> (f64, f64) {
    let n_values: usize = values.len();

    let mut value_min: f64 = f64::INFINITY;
    let mut value_max: f64 = f64::NEG_INFINITY;
    for i_value in 0..n_values {
        if !values[i_value].is_finite() {
            return (f64::NAN, f64::NAN);
        }
        value_min = value_min.min(values[i_value]);
        value_max = value_max.max(values[i_value]);
    }

    (value_min, value_max)
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::f64::consts::PI;

    /// A closed elliptical contour, as `flux_surfaces::calculate` would return it
    fn elliptical_flux_surface(r_geo: f64, z_geo: f64, r_minor: f64, kappa: f64) -> FluxSurface {
        // `n_theta` divisible by 4, so that the extremal points are exactly on the contour
        let n_theta: usize = 1000;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);

        return FluxSurface {
            r: r_geo + r_minor * theta.mapv(f64::cos),
            z: z_geo + kappa * r_minor * theta.mapv(f64::sin),
        };
    }

    /// An empty flux surface, as `flux_surfaces::calculate` stores one it could not trace
    fn untraced_flux_surface() -> FluxSurface {
        return FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };
    }

    #[test]
    fn each_surface_gives_the_centre_of_its_bounding_box() {
        let mag_r: f64 = 0.47;
        let mag_z: f64 = 0.03;

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.5, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = 0.1;
        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;

        // The magnetic axis, one surface which could not be traced, and one ellipse which is
        // deliberately not centred on the magnetic axis, as a Shafranov-shifted surface is not
        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), untraced_flux_surface(), elliptical_flux_surface(0.52, -0.01, 0.3, 1.7)];

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
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = 0.1;
        time_slice.global_quantities.magnetic_axis.r = 0.5;
        time_slice.global_quantities.magnetic_axis.z = 0.0;

        // A closed triangle spanning r in [0.2, 0.8] and z in [-0.1, 0.3]. Its centroid is at
        // r = (0.2 + 0.8 + 0.2) / 3 = 0.4, but the centre of its bounding box is 0.5
        let flux_surface: FluxSurface = FluxSurface {
            r: array![0.2, 0.8, 0.2, 0.2],
            z: array![-0.1, -0.1, 0.3, -0.1],
        };
        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), flux_surface];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.r[1], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.profiles_1d.geometric_axis.z[1], 0.1, epsilon = 1e-15);
    }

    #[test]
    fn a_contour_carrying_a_nan_is_not_measured() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = 0.1;
        time_slice.global_quantities.magnetic_axis.r = 0.5;
        time_slice.global_quantities.magnetic_axis.z = 0.0;

        let flux_surface: FluxSurface = FluxSurface {
            r: array![0.2, 0.8, f64::NAN, 0.2],
            z: array![-0.1, -0.1, 0.3, -0.1],
        };
        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), flux_surface];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        assert!(time_slice.profiles_1d.geometric_axis.r[1].is_nan());
    }

    #[test]
    fn a_slice_which_did_not_converge_is_all_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.5, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = f64::NAN;

        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(); 3];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        // Including the magnetic axis, which is only filled once there is a converged solution
        assert!(time_slice.profiles_1d.geometric_axis.r.iter().all(|value| value.is_nan()));
        assert!(time_slice.profiles_1d.geometric_axis.z.iter().all(|value| value.is_nan()));
    }
}
