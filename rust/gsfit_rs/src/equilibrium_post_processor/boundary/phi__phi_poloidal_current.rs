//! `time_slice(itime)/boundary/phi` and `.../phi_poloidal_current`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use super::super::profiles_1d::phi::flux_toroidal_profile;
use super::super::profiles_1d::q::q_profile;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Store the toroidal flux through the plasma boundary, and the part of it which the plasma's
/// poloidal current generates.
///
/// The selected boundary is the final surface in `profiles_1d`, so `phi` is the final value of
/// `profiles_1d/phi`.
///
/// `phi_poloidal_current` is the difference between that and the toroidal flux which the vacuum
/// toroidal field alone would give through the same boundary, so it measures how far the plasma has
/// expelled or compressed the toroidal field. The vacuum toroidal flux is calculated by running the
/// safety factor and toroidal flux a second time with `f` set to the vacuum value
/// `f_vac = mu_0 * i_rod / (2 * pi)` everywhere.
///
/// Outside the boundary no poloidal current flows, so `f` is the vacuum value there, and this is
/// also the flux a diamagnetic loop around the plasma measures: `constraints/diamagnetic_flux/reconstructed`
/// is a copy of it.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/phi` and `boundary/phi_poloidal_current` are
///   written into it
/// * `constant_values` - the constant values; `i_rod`, the current in the toroidal field coil's
///   central rod [ampere], is read
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces`, one per `psi_norm`, is
///   read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let i_rod: f64 = constant_values.i_rod;

    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    let psi_profile: &Array1<f64> = &time_slice.profiles_1d.psi;
    let flux_tor_profile: &Array1<f64> = &time_slice.profiles_1d.phi;

    // TODO: this is **VERY** hacky, and **SHOULD** be improved!!
    // set f_profile to the vacuum profile, then calculate the vacuum q-profile, then the vacuum toroidal flux
    let f_profile_vacuum: Array1<f64> = 0.0 * f_profile + MU_0 * i_rod / (2.0 * PI);
    let q_profile_vacuum: Array1<f64> = q_profile(time_slice, flux_surfaces, &f_profile_vacuum);
    let boundary_diverted: bool = time_slice.boundary.r#type == 1;
    let flux_tor_profile_vacuum: Array1<f64> = flux_toroidal_profile(&q_profile_vacuum, psi_profile, boundary_diverted);

    let boundary_phi: f64 = flux_tor_profile.last().copied().unwrap_or(f64::NAN);
    let boundary_phi_vacuum: f64 = flux_tor_profile_vacuum.last().copied().unwrap_or(f64::NAN);

    time_slice.boundary.phi = boundary_phi;
    time_slice.boundary.phi_poloidal_current = boundary_phi - boundary_phi_vacuum;
}

#[cfg(test)]
mod tests {
    use super::super::super::boundary::geometry::time_slice_with_elliptical_psi;
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::flux_surfaces::calculate_flux_surfaces;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// A uniform `f = c * f_vac` makes `q = c * q_vac` on every surface, and the toroidal flux is
    /// linear in `q`, so `phi = c * phi_vac` exactly and `phi_poloidal_current = (1 - 1 / c) * phi`,
    /// whatever the discretisation
    #[test]
    fn uniform_f_scales_the_vacuum_toroidal_flux() {
        // psi = 1 - a rho ** 2 about (0.5, 0), with the boundary at psi = 0
        let a: f64 = 25.0;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(0.5, 0.0, a, 0.0, a);
        time_slice.boundary.psi = 1.0 - a * 0.2_f64.powi(2);
        let psi_norm_profile: Array1<f64> = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let n_psi_norm: usize = psi_norm_profile.len();
        let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
        let psi_b: f64 = time_slice.boundary.psi;
        let mut psi_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        for i_psi_norm in 0..n_psi_norm {
            psi_profile[i_psi_norm] = psi_a + psi_norm_profile[i_psi_norm] * (psi_b - psi_a);
        }
        time_slice.profiles_1d.psi_norm = psi_norm_profile;
        time_slice.profiles_1d.psi = psi_profile.clone();

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.i_rod = 1.0e6;
        let f_vacuum: f64 = MU_0 * constant_values.i_rod / (2.0 * PI);
        let c: f64 = 1.05;
        let f_profile: Array1<f64> = Array1::from_elem(n_psi_norm, c * f_vacuum);

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = calculate_flux_surfaces(&time_slice);
        let q_profile_plasma: Array1<f64> = q_profile(&time_slice, &intermediate_values.flux_surfaces, &f_profile);
        let flux_tor_profile: Array1<f64> = flux_toroidal_profile(&q_profile_plasma, &psi_profile, false);
        time_slice.profiles_1d.f = f_profile;
        time_slice.profiles_1d.phi = flux_tor_profile.clone();

        calculate(&mut time_slice, &constant_values, &mut intermediate_values);

        let boundary_phi_expected: f64 = flux_tor_profile[n_psi_norm - 1];
        assert!(boundary_phi_expected.is_finite() && boundary_phi_expected != 0.0);
        assert_abs_diff_eq!(time_slice.boundary.phi, boundary_phi_expected, epsilon = 0.0);
        assert_abs_diff_eq!(
            time_slice.boundary.phi_poloidal_current,
            (1.0 - 1.0 / c) * boundary_phi_expected,
            epsilon = 1e-14 * boundary_phi_expected.abs()
        );
    }
}
