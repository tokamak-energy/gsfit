//! The Pfirsch-Schlüter current: the current along the field lines which closes the diamagnetic current, so that `div(J) = 0`.

use super::flux_surface_samples::FluxSurfaceSamples;
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, pressure_driven_components};
use imas_rs::Equilibrium as EquilibriumIds;
use std::f64::consts::PI;

/// Calculate the Pfirsch-Schlüter current on every time-slice of the equilibrium, and write it to the plasma current's components, as
/// `pfirsch_schluter`.
///
/// The diamagnetic current is not divergence-free by itself. The current along the field lines which closes it is, in GSFit's convention,
/// `B = F grad(phi) - grad(phi) x grad(psi) / (2 pi)`,
///
/// ```text
/// J_PS = 2 pi F p' (1 / B ** 2 - 1 / <B ** 2>) B
/// ```
///
/// whose toroidal part is `J_PS . grad(phi) = 2 pi F ** 2 p' (1 / B ** 2 - 1 / <B ** 2>) / R ** 2`, so its toroidal current density is
///
/// ```text
/// <j_phi / R> / <1 / R> = 2 pi F ** 2 p' (<1 / (R ** 2 B ** 2)> - <R ** -2> / <B ** 2>) / <1 / R>
/// ```
///
/// It flows one way on the outboard side and the other way on the inboard side, so its toroidal current is the small difference of two
/// averages; both are taken from the same samples of the field, so that the difference is free of discretisation error. It vanishes on the
/// magnetic axis. Its `<J . B>` is zero by construction, so it has no place in `core_sources`. With the diamagnetic current it is the
/// current the pressure drives, which the Grad-Shafranov equation contains once `p'` is fixed (see the notes in
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
pub fn current_pfirsch_schluter(equilibrium: &EquilibriumIds, plasma_current_components: &mut PlasmaCurrentComponents) -> Result<(), String> {
    plasma_current_components.check_made_from(equilibrium)?;

    let mut components: Vec<PlasmaCurrentComponent> =
        pressure_driven_components(equilibrium, "Pfirsch-Schlüter", |p_prime: f64, f: f64, samples: &FluxSurfaceSamples| {
            let r_b_field_inverse_squared_average: f64 = samples.average(|r_here: f64, b_field: f64, _b_field_p| 1.0 / (r_here * b_field).powi(2));
            let r_inverse_squared_average: f64 = samples.average(|r_here: f64, _b_field, _b_field_p| r_here.powi(-2));
            let b_field_squared_average: f64 = samples.average(|_r, b_field: f64, _b_field_p| b_field.powi(2));
            let r_inverse_average: f64 = samples.average(|r_here: f64, _b_field, _b_field_p| 1.0 / r_here);
            2.0 * PI * f.powi(2) * p_prime * (r_b_field_inverse_squared_average - r_inverse_squared_average / b_field_squared_average) / r_inverse_average
        })?;

    let n_time: usize = components.len();
    for i_time in 0..n_time {
        plasma_current_components.time_slice[i_time].pfirsch_schluter = std::mem::take(&mut components[i_time]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::current_diamagnetic::current_diamagnetic;
    use super::super::test_equilibrium::circular_equilibrium;
    use super::*;

    /// Where `R b_p` is the same everywhere on each surface, as in the circular equilibrium, `R ** 2 B ** 2 = (R b_p) ** 2 + F ** 2` is too,
    /// so `<1 / (R ** 2 B ** 2)> = <R ** -2> / <B ** 2>` and the Pfirsch-Schlüter current has no toroidal current. Its two averages are each
    /// about `1 / F ** 2`, and their difference is to vanish against the diamagnetic current, which is `(R b_p / F) ** 2` smaller.
    /// `the_components_add_up_to_the_grad_shafranov_current_density` tests it where it does not vanish.
    #[test]
    fn the_pfirsch_schluter_current_vanishes_where_r_b_p_is_the_same_on_each_surface() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_diamagnetic(&equilibrium, &mut plasma_current_components).unwrap();
        current_pfirsch_schluter(&equilibrium, &mut plasma_current_components).unwrap();

        let diamagnetic_j_phi: &ndarray::Array1<f64> = &plasma_current_components.time_slice[0].diamagnetic.j_phi;
        let pfirsch_schluter_j_phi: &ndarray::Array1<f64> = &plasma_current_components.time_slice[0].pfirsch_schluter.j_phi;
        let n_psi_norm: usize = pfirsch_schluter_j_phi.len();
        assert_eq!(pfirsch_schluter_j_phi[0], 0.0);
        for i_psi_norm in 1..n_psi_norm - 1 {
            assert!(
                pfirsch_schluter_j_phi[i_psi_norm].abs() < 1.0e-2 * diamagnetic_j_phi[i_psi_norm],
                "j_phi_pfirsch_schluter({i_psi_norm}) = {}, j_phi_diamagnetic = {}",
                pfirsch_schluter_j_phi[i_psi_norm],
                diamagnetic_j_phi[i_psi_norm]
            );
        }
    }
}
