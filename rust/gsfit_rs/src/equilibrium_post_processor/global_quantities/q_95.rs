//! `time_slice(itime)/global_quantities/q_95`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::{self, FluxSurface};
use super::super::intermediate_values::IntermediateValues;
use super::super::profiles_1d;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;

const Q_95_PSI_NORM: f64 = 0.95;

/// Calculate the safety factor at `psi_norm = 0.95`, and store it in the time-slice.
///
/// The `psi_norm = 0.95` contour is traced directly and integrated with the same quadrature as the
/// safety-factor profile. The poloidal-current function is evaluated at exactly 0.95 from the
/// fitted FF' source function, so the result does not depend on the `profiles_1d` sampling grid.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/q_95` is written into it
/// * `ff_prime_source_function` - the FF' source function the reconstruction was run with
/// * `i_rod` - current in the toroidal field coil's central rod [ampere]
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let ff_prime_source_function: &SharedSourceFunction = constant_values.ff_prime_source_function;
    let i_rod: f64 = constant_values.i_rod;

    // A slice which did not converge has no flux surface to integrate around.
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.global_quantities.q_95 = Some(f64::NAN);
        return;
    }

    let flux_surface: FluxSurface = flux_surfaces::calculate_at_psi_norm(time_slice, Q_95_PSI_NORM);
    let f_95: f64 = profiles_1d::f::value_at_psi_norm(time_slice, ff_prime_source_function, i_rod, Q_95_PSI_NORM);
    let q95: f64 = profiles_1d::q::calculate_on_flux_surface(time_slice, &flux_surface, f_95);

    time_slice.global_quantities.q_95 = Some(q95);
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use crate::source_functions::EfitPolynomial;
    use approx::assert_abs_diff_eq;
    use imas_rs::ids::equilibrium::EquilibriumProfiles2d;
    use ndarray::{Array1, Array2, array};
    use std::f64::consts::PI;
    use std::sync::Arc;

    #[test]
    fn q_95_uses_a_direct_contour_and_not_the_stored_profiles() {
        let r_axis: f64 = 1.0;
        let z_axis: f64 = 0.0;
        let boundary_minor_radius: f64 = 0.4;
        let psi_curvature: f64 = 2.5;
        let f_95: f64 = 0.8;
        let n_r: usize = 161;
        let n_z: usize = 161;
        let r: Array1<f64> = Array1::linspace(0.5, 1.5, n_r);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_z);

        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_r in 0..n_r {
            for i_z in 0..n_z {
                let delta_r: f64 = r[i_r] - r_axis;
                let delta_z: f64 = z[i_z] - z_axis;
                psi_2d[(i_z, i_r)] = -0.5 * psi_curvature * (delta_r.powi(2) + delta_z.powi(2));
                d_psi_d_r_2d[(i_z, i_r)] = -psi_curvature * delta_r;
                d_psi_d_z_2d[(i_z, i_r)] = -psi_curvature * delta_z;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = Some(r);
        profiles_2d.grid.dim2 = Some(z);
        profiles_2d.psi = Some(psi_2d);
        profiles_2d.d_psi_d_r = Some(d_psi_d_r_2d);
        profiles_2d.d_psi_d_z = Some(d_psi_d_z_2d);
        profiles_2d.d2_psi_d_r2 = Some(Array2::from_elem((n_z, n_r), -psi_curvature));
        profiles_2d.d2_psi_d_r_d_z = Some(Array2::zeros((n_z, n_r)));
        profiles_2d.d2_psi_d_z2 = Some(Array2::from_elem((n_z, n_r), -psi_curvature));

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = Some(0.0);
        time_slice.global_quantities.magnetic_axis.r = Some(r_axis);
        time_slice.global_quantities.magnetic_axis.z = Some(z_axis);
        time_slice.boundary.psi = Some(-0.5 * psi_curvature * boundary_minor_radius.powi(2));
        time_slice.profiles_2d = vec![profiles_2d];
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.5, 1.0]);
        time_slice.profiles_1d.f = Some(Array1::from_elem(3, f64::NAN));
        time_slice.profiles_1d.q = Some(Array1::from_elem(3, f64::NAN));
        time_slice.source_functions.ff_prime.coefficients = Some(array![0.0]);

        let ff_prime_source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
            n_dof: 1,
            regularisations: Array2::zeros((1, 1)),
            dof_values: Array1::zeros(0),
        });
        let i_rod: f64 = 2.0 * PI * f_95 / physical_constants::VACUUM_MAG_PERMEABILITY;

        let mut constant_values: ConstantValues = constant_values_for_test();
        constant_values.ff_prime_source_function = &ff_prime_source_function;
        constant_values.i_rod = i_rod;
        calculate(&mut time_slice, &constant_values, &mut intermediate_values_for_test());

        let minor_radius_95: f64 = boundary_minor_radius * Q_95_PSI_NORM.sqrt();
        let q_95_expected: f64 = 2.0 * PI * f_95 / (psi_curvature * (r_axis.powi(2) - minor_radius_95.powi(2)).sqrt());
        assert_abs_diff_eq!(time_slice.global_quantities.q_95.unwrap(), q_95_expected, epsilon = 2e-4);
    }
}
