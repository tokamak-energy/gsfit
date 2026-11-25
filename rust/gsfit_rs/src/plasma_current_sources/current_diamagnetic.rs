//! The diamagnetic current: the current across the field lines which balances the pressure gradient.

use super::flux_surface_samples::FluxSurfaceSamples;
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, pressure_driven_components};
use imas_rs::Equilibrium as EquilibriumIds;
use std::f64::consts::PI;

/// Calculate the diamagnetic current on every time-slice of the equilibrium, and write it to the plasma current's components, as
/// `diamagnetic`.
///
/// Force balance, `J x B = grad(p)`, fixes the current across the field lines, `J_dia = B x grad(p) / B ** 2 = p' B x grad(psi) / B ** 2`.
/// In GSFit's convention, `B = F grad(phi) - grad(phi) x grad(psi) / (2 pi)` with `b_p = |grad(psi)| / (2 pi R)`, its toroidal part is
/// `J_dia . grad(phi) = 2 pi p' b_p ** 2 / B ** 2`, so its toroidal current density is
///
/// ```text
/// <j_phi / R> / <1 / R> = 2 pi p' <b_p ** 2 / B ** 2> / <1 / R>
/// ```
///
/// For a pressure which falls outwards it flows with the plasma current, and it vanishes on the magnetic axis, where `b_p = 0`. Being
/// perpendicular to `B`, it carries no `<J . B>`, and so has no place in `core_sources`. With the Pfirsch-Schlüter current it is the current
/// the pressure drives, which the Grad-Shafranov equation contains once `p'` is fixed (see the notes in
/// `paper/gsfit_consistent_with_neoclassical_current`, "Which currents the Grad-Shafranov equation already contains").
///
/// `p'` is the equilibrium's, and the averages are of the field sampled on the flux surfaces; see `pressure_driven_components`.
///
/// # Arguments
/// * `equilibrium` - the Equilibrium IDS, post-processed: on each time-slice `profiles_1d` with `psi_norm`, `f`, `dpressure_dpsi` and
///   `area`, and `profiles_2d(0)`, the magnetic axis and the boundary, for the flux surfaces
/// * `plasma_current_components` - the plasma current's components, made from `equilibrium` by `PlasmaCurrentComponents::new`
///
/// # Errors
/// The equilibrium's `profiles_1d` is not filled, or `plasma_current_components` was not made from `equilibrium`, which is not changed then
pub fn current_diamagnetic(equilibrium: &EquilibriumIds, plasma_current_components: &mut PlasmaCurrentComponents) -> Result<(), String> {
    plasma_current_components.check_made_from(equilibrium)?;

    let mut components: Vec<PlasmaCurrentComponent> =
        pressure_driven_components(equilibrium, "diamagnetic", |p_prime: f64, _f, samples: &FluxSurfaceSamples| {
            let b_field_p_squared_over_b_field_squared_average: f64 = samples.average(|_r, b_field: f64, b_field_p: f64| b_field_p.powi(2) / b_field.powi(2));
            let r_inverse_average: f64 = samples.average(|r_here: f64, _b_field, _b_field_p| 1.0 / r_here);
            2.0 * PI * p_prime * b_field_p_squared_over_b_field_squared_average / r_inverse_average
        })?;

    let n_time: usize = components.len();
    for i_time in 0..n_time {
        plasma_current_components.time_slice[i_time].diamagnetic = std::mem::take(&mut components[i_time]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_equilibrium::{MINOR_RADIUS, PSI_AXIS, R_AXIS, circular_equilibrium};
    use super::*;
    use approx::assert_relative_eq;
    use imas_rs::EquilibriumProfiles1d;

    /// In the circular equilibrium `R b_p = |grad(psi)| / (2 pi) = c` is the same everywhere on each surface, with
    /// `c = 2 PSI_AXIS rho / (2 pi MINOR_RADIUS ** 2)` at minor radius `rho`, and the flux-surface average is `<x> = integral(x R d_theta) /
    /// integral(R d_theta)`. So `b_p ** 2 / B ** 2 = c ** 2 / (c ** 2 + F ** 2)` and `<1 / R> = 1 / R_AXIS`, and the diamagnetic current
    /// density is exactly `2 pi p' R_AXIS c ** 2 / (c ** 2 + F ** 2)`. The flux surfaces are traced on the `profiles_2d(0)` grid, and the
    /// field interpolated from it, so the error is second order in the grid spacing over the minor radius, about `0.14 (d_r / rho) ** 2`.
    #[test]
    fn the_diamagnetic_current_agrees_with_the_circular_equilibrium_s_and_flows_with_the_plasma_current() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_diamagnetic(&equilibrium, &mut plasma_current_components).unwrap();

        let profiles_1d: &EquilibriumProfiles1d = &equilibrium.time_slice[1].profiles_1d;
        let grid_r: &ndarray::Array1<f64> = &equilibrium.time_slice[1].profiles_2d[0].grid.dim1;
        let delta_r: f64 = grid_r[1] - grid_r[0];
        let j_phi: &ndarray::Array1<f64> = &plasma_current_components.time_slice[1].diamagnetic.j_phi;
        let n_psi_norm: usize = profiles_1d.psi_norm.len();
        assert_eq!(j_phi[0], 0.0);
        for i_psi_norm in 1..n_psi_norm {
            let minor_radius: f64 = MINOR_RADIUS * profiles_1d.psi_norm[i_psi_norm].sqrt();
            let r_b_field_p: f64 = 2.0 * PSI_AXIS * minor_radius / (2.0 * PI * MINOR_RADIUS.powi(2));
            let j_phi_exact: f64 =
                2.0 * PI * profiles_1d.dpressure_dpsi[i_psi_norm] * R_AXIS * r_b_field_p.powi(2) / (r_b_field_p.powi(2) + profiles_1d.f[i_psi_norm].powi(2));
            assert_relative_eq!(
                j_phi[i_psi_norm],
                j_phi_exact,
                epsilon = 1.0e-12,
                max_relative = 0.25 * (delta_r / minor_radius).powi(2)
            );
            if i_psi_norm < n_psi_norm - 1 {
                assert!(j_phi[i_psi_norm] > 0.0);
            }
        }
    }

    #[test]
    fn a_time_slice_the_solver_could_not_use_is_nan() {
        let mut equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        // As the solver leaves a time-slice it did not converge on: not post-processed, with only `profiles_1d/psi_norm`
        equilibrium.code.output_flag[0] = -1;
        equilibrium.time_slice[0].profiles_1d = EquilibriumProfiles1d {
            psi_norm: equilibrium.time_slice[0].profiles_1d.psi_norm.clone(),
            ..Default::default()
        };
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_diamagnetic(&equilibrium, &mut plasma_current_components).unwrap();

        let n_psi_norm: usize = equilibrium.time_slice[0].profiles_1d.psi_norm.len();
        for i_psi_norm in 0..n_psi_norm {
            assert!(plasma_current_components.time_slice[0].diamagnetic.j_phi[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].diamagnetic.current_phi_inside[i_psi_norm].is_nan());
        }
        assert!(plasma_current_components.time_slice[1].diamagnetic.current_phi_inside[n_psi_norm - 1] > 0.0);
    }

    #[test]
    fn an_equilibrium_without_the_pressure_gradient_is_an_error() {
        let mut equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        equilibrium.time_slice[0].profiles_1d.dpressure_dpsi = ndarray::Array1::zeros(0);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        let error: String = current_diamagnetic(&equilibrium, &mut plasma_current_components).unwrap_err();

        assert!(error.contains("`profiles_1d/dpressure_dpsi` has 0 values"), "{error}");
        assert!(plasma_current_components.time_slice[0].diamagnetic.j_phi.is_empty());
    }
}
