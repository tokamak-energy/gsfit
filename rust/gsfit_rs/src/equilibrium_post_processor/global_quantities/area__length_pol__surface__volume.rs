//! `time_slice(itime)/global_quantities/area`, `.../length_pol`, `.../surface` and
//! `.../volume`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Store the area, poloidal length, surface area and volume of the plasma boundary.
///
/// Area, toroidal surface area and volume are the final values of the corresponding 1-D profiles,
/// whose final surface is the plasma boundary. Poloidal length is the length of the boundary
/// outline in the poloidal plane. The outline is closed, with its final point repeating its first.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the four `global_quantities` nodes are written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let area_profile: &Array1<f64> = &time_slice.profiles_1d.area;
    let boundary_r: &Array1<f64> = &time_slice.boundary.outline.r;
    let boundary_z: &Array1<f64> = &time_slice.boundary.outline.z;
    let surface_profile: &Array1<f64> = &time_slice.profiles_1d.surface;
    let volume_profile: &Array1<f64> = &time_slice.profiles_1d.volume;

    let n_boundary: usize = boundary_r.len();
    assert_eq!(boundary_z.len(), n_boundary);

    let mut length_pol: f64 = 0.0;
    for i_boundary in 1..n_boundary {
        let delta_r: f64 = boundary_r[i_boundary] - boundary_r[i_boundary - 1];
        let delta_z: f64 = boundary_z[i_boundary] - boundary_z[i_boundary - 1];
        length_pol += delta_r.hypot(delta_z);
    }

    time_slice.global_quantities.area = area_profile.last().copied().unwrap_or(f64::NAN);
    time_slice.global_quantities.length_pol = length_pol;
    time_slice.global_quantities.surface = surface_profile.last().copied().unwrap_or(f64::NAN);
    time_slice.global_quantities.volume = volume_profile.last().copied().unwrap_or(f64::NAN);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn global_geometry_comes_from_the_boundary_and_profile_endpoints() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.boundary.outline.r = array![1.0, 2.0, 2.0, 1.0, 1.0];
        time_slice.boundary.outline.z = array![0.0, 0.0, 1.0, 1.0, 0.0];
        time_slice.profiles_1d.area = array![0.0, 1.0];
        time_slice.profiles_1d.surface = array![0.0, 12.0];
        time_slice.profiles_1d.volume = array![0.0, 8.0];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.global_quantities.area, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.length_pol, 4.0, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.surface, 12.0, epsilon = 1e-15);
        assert_abs_diff_eq!(time_slice.global_quantities.volume, 8.0, epsilon = 1e-15);
    }
}
