//! The bootstrap current, from the formulae of Redl, Angioni, Belli and Sauter, "A new set of analytical formulae for the computation of
//! the bootstrap current and the neoclassical conductivity in tokamaks", Physics of Plasmas 28, 022502 (2021).

use super::collisionality::{electrons_collisionality, ions_collisionality};
use super::core_sources_source::{core_sources_grid, write_core_sources_source};
use super::flux_surface_samples::{FluxSurfaceSamples, flux_surface_samples};
use super::kinetic_profiles::{KineticProfiles, core_profiles_time_index, kinetic_profiles};
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, integral_inside, j_phi_of_parallel_current};
use super::settings::SETTINGS;
use super::trapped_fraction::trapped_fraction;
use imas_rs::ids::core_profiles::{CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d};
use imas_rs::ids::core_sources::{CoreSources as CoreSourcesIds, CoreSourcesSource, CoreSourcesSourceGlobal, CoreSourcesSourceProfiles1d, Identifier};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumProfiles1d, EquilibriumTimeSlice};
use ndarray::Array1;
use std::f64::consts::PI;

const ELEMENTARY_CHARGE: f64 = physical_constants::ELEMENTARY_CHARGE;

/// The bootstrap current's `core_sources/source(i1)/identifier`, from the data dictionary's `core_source_identifier.xml`
const IDENTIFIER_NAME: &str = "bootstrap_current";
const IDENTIFIER_INDEX: i32 = 13;
const IDENTIFIER_DESCRIPTION: &str = "Bootstrap current";

/// Calculate the bootstrap current on every time-slice of the equilibrium, and write it to `core_sources`, as the source
/// `bootstrap_current`, and to the plasma current's components, as `bootstrap`.
///
/// On each flux surface the flux-surface averaged bootstrap current is Redl's equation 2, with their equations 7 and 8 for several ion
/// species, every ion species at the main ion's temperature:
///
/// ```text
/// <j_bs . B> = -F (L31 dp/dpsi_s + L32 p_e d(ln T_e)/dpsi_s + L34 alpha p_i d(ln T_i)/dpsi_s)
/// ```
///
/// where `p = p_e + p_i` is the thermal pressure, `p_i` that of every ion species, `F = R B_phi`, and `psi_s` is the poloidal flux per
/// radian in Sauter and Redl's convention, `B = F grad(phi) + grad(phi) x grad(psi_s)`. GSFit's `psi` is the total poloidal flux, with
/// `B = F grad(phi) - grad(phi) x grad(psi) / (2 pi)` (COCOS 13, and 17), so `psi_s = -psi / (2 pi)`, and with `L34 = L31` (Redl's
/// equation 19)
///
/// ```text
/// <j_bs . B> = 2 pi F (L31 dp/dpsi + L32 e n_e dT_e/dpsi + L31 alpha e n_i dT_i/dpsi)
/// ```
///
/// writing `p_e d(ln T_e) = e n_e dT_e` and `p_i d(ln T_i) = e n_i dT_i`, which stay finite where a temperature falls to zero. For a
/// pressure which falls outwards it flows with the plasma current. The coefficients `L31`, `L32` and `alpha` are Redl's equations 10 to
/// 16, 20 and 21, of the trapped fraction, `f_t`, the collisionalities, `nu_e*` and `nu_i*`, and `Z_eff`:
/// * `f_t` is Lin-Liu and Miller's, from the flux surfaces (`trapped_fraction.rs`)
/// * `nu_e*` and `nu_i*` are Sauter's (`collisionality.rs`), with the inverse aspect ratio and major radius of each surface from its radii
///   at the height of the magnetic axis, `(r_outboard - r_inboard) / (r_outboard + r_inboard)` and `(r_outboard + r_inboard) / 2`
/// * the kinetic profiles and their derivatives are mapped onto the flux surfaces by `kinetic_profiles.rs`
///
/// On the magnetic axis `f_t = 0`, so the bootstrap current is zero. Where a collisionality is infinite, so are the coefficients'
/// denominators, and the bootstrap current is zero, the limit of Redl's formulae. That includes the boundary, `psi_norm = 1`, where the
/// equilibrium's `q` is NaN: the post-processor leaves it so because `q` is infinite on a separatrix, so it is taken as infinite there.
/// On a separatrix that is exact; on a limiter, where `q` is finite, it is an approximation on the last surface only.
///
/// The source holds, on each time-slice:
/// * `profiles_1d(itime)/j_parallel`, `<j_bs . B> / B0`, with `B0` the equilibrium's `vacuum_toroidal_field/b0`, on the flux surfaces of
///   the equilibrium's `profiles_1d`, whose coordinates are copied into `profiles_1d(itime)/grid`
/// * `profiles_1d(itime)/current_parallel_inside`, its integral over the cross-section inside each surface, by the trapezium rule in
///   `profiles_1d/area`
/// * `global_quantities(itime)/current_parallel`, the same integral inside the boundary
///
/// and the plasma current's `bootstrap` component holds its toroidal current density, `<j_phi / R> / <1 / R>`, the toroidal current
/// inside each surface (see `j_phi_of_parallel_current`), and the source's `j_parallel`. A time-slice the solver could not use, with a negative `code/output_flag`, or
/// with no kinetic profiles at its time, has NaN in each.
///
/// # Arguments
/// * `equilibrium` - the Equilibrium IDS, post-processed: on each time-slice `profiles_1d` with `psi`, `psi_norm`, `rho_tor_norm`, `f`,
///   `q`, `r_inboard`, `r_outboard` and `area`, and `profiles_2d(0)`, the magnetic axis and the boundary, for the flux surfaces
/// * `core_profiles` - the kinetic profiles, on `profiles_1d(:)`, at the time of each equilibrium time-slice which has them: see
///   `kinetic_profiles.rs` for the nodes read
/// * `core_sources` - the plasma's sources, which the bootstrap current is written into (see `write_core_sources_source`)
/// * `plasma_current_components` - the plasma current's components, made from `equilibrium` by `PlasmaCurrentComponents::new`
///
/// # Errors
/// The kinetic profiles at an equilibrium time-slice's time are not usable (see `kinetic_profiles.rs`); the equilibrium's
/// `profiles_1d` is not filled, or its `vacuum_toroidal_field/b0` does not have one value per time-slice; `core_sources` holds other
/// sources at other times, or with another vacuum toroidal field; or `plasma_current_components` was not made from `equilibrium`.
/// Neither `core_sources` nor `plasma_current_components` is changed then.
pub fn current_bootstrap(
    equilibrium: &EquilibriumIds,
    core_profiles: &CoreProfilesIds,
    core_sources: &mut CoreSourcesIds,
    plasma_current_components: &mut PlasmaCurrentComponents,
) -> Result<(), String> {
    plasma_current_components.check_made_from(equilibrium)?;
    let n_time: usize = equilibrium.time_slice.len();
    let b0: &Array1<f64> = &equilibrium.vacuum_toroidal_field.b0;
    if b0.len() != n_time {
        return Err(format!(
            "`equilibrium/vacuum_toroidal_field/b0` has {} values, but `equilibrium/time_slice` has {n_time} time-slices",
            b0.len()
        ));
    }

    let mut profiles_1d: Vec<CoreSourcesSourceProfiles1d> = Vec::with_capacity(n_time);
    let mut global_quantities: Vec<CoreSourcesSourceGlobal> = Vec::with_capacity(n_time);
    let mut components: Vec<PlasmaCurrentComponent> = Vec::with_capacity(n_time);
    for i_time in 0..n_time {
        let time_slice: &EquilibriumTimeSlice = &equilibrium.time_slice[i_time];
        let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

        // A negative `code/output_flag` means the time-slice shall not be used. The solver did not converge on it, so its `profiles_1d` was
        // not post-processed, and only `psi_norm` is filled: its `area` is checked and used only when it is usable
        let usable: bool = equilibrium.code.output_flag.len() != n_time || equilibrium.code.output_flag[i_time] >= 0;
        let area_not_usable: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let area: &Array1<f64> = if usable { &time_slice.profiles_1d.area } else { &area_not_usable };
        if area.len() != n_psi_norm {
            return Err(format!(
                "`equilibrium/time_slice({i_time})/profiles_1d/area` has {} values, but `profiles_1d/psi_norm` has {n_psi_norm}",
                area.len()
            ));
        }

        let mut j_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        // Without kinetic profiles at the time-slice's time, the bootstrap current is NaN on it
        if usable && let Some(i_time_core_profiles) = core_profiles_time_index(core_profiles, time_slice.time) {
            let samples: Vec<Option<FluxSurfaceSamples>> =
                flux_surface_samples(time_slice).map_err(|reason: String| format!("bootstrap current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            let j_dot_b: Array1<f64> = j_dot_b_on_time_slice(time_slice, &samples, &core_profiles.profiles_1d[i_time_core_profiles])
                .map_err(|reason: String| format!("bootstrap current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            j_parallel = &j_dot_b / b0[i_time];
            j_phi = j_phi_of_parallel_current(&j_dot_b, &time_slice.profiles_1d.f, &samples, time_slice.global_quantities.magnetic_axis.r);
        }

        components.push(PlasmaCurrentComponent::new(j_phi, j_parallel.clone(), area));
        let current_parallel_inside: Array1<f64> = integral_inside(&j_parallel, area);
        let current_parallel: f64 = if n_psi_norm > 0 { current_parallel_inside[n_psi_norm - 1] } else { f64::NAN };
        profiles_1d.push(CoreSourcesSourceProfiles1d {
            grid: core_sources_grid(time_slice),
            j_parallel,
            current_parallel_inside,
            time: time_slice.time,
            ..Default::default()
        });
        global_quantities.push(CoreSourcesSourceGlobal {
            current_parallel,
            time: time_slice.time,
            ..Default::default()
        });
    }

    let source: CoreSourcesSource = CoreSourcesSource {
        identifier: Identifier {
            name: IDENTIFIER_NAME.to_string(),
            index: IDENTIFIER_INDEX,
            description: IDENTIFIER_DESCRIPTION.to_string(),
        },
        global_quantities,
        profiles_1d,
        ..Default::default()
    };
    write_core_sources_source(core_sources, equilibrium, source)?;
    for i_time in 0..n_time {
        plasma_current_components.time_slice[i_time].bootstrap = std::mem::take(&mut components[i_time]);
    }

    Ok(())
}

/// The flux-surface averaged bootstrap current, `<j_bs . B>`, on the flux surfaces of one equilibrium time-slice; see `current_bootstrap`.
///
/// # Arguments
/// * `time_slice` - the post-processed equilibrium time-slice
/// * `flux_surface_samples` - the field sampled on its flux surfaces, `flux_surface_samples`
/// * `core_profiles_profiles_1d` - the kinetic profiles at the time-slice's time
///
/// # Returns
/// `<j_bs . B>` on each surface of `profiles_1d/psi_norm` [ampere * tesla / metre ** 2]
///
/// # Errors
/// The equilibrium's `profiles_1d` is not filled, or the kinetic profiles are not usable
fn j_dot_b_on_time_slice(
    time_slice: &EquilibriumTimeSlice,
    flux_surface_samples: &[Option<FluxSurfaceSamples>],
    core_profiles_profiles_1d: &CoreProfilesProfiles1d,
) -> Result<Array1<f64>, String> {
    let equilibrium_profiles_1d: &EquilibriumProfiles1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = equilibrium_profiles_1d.psi_norm.len();
    let named_profiles: [(&str, &Array1<f64>); 5] = [
        ("f", &equilibrium_profiles_1d.f),
        ("q", &equilibrium_profiles_1d.q),
        ("r_inboard", &equilibrium_profiles_1d.r_inboard),
        ("r_outboard", &equilibrium_profiles_1d.r_outboard),
        ("psi", &equilibrium_profiles_1d.psi),
    ];
    for (name, values) in named_profiles {
        if values.len() != n_psi_norm {
            return Err(format!(
                "`profiles_1d/{name}` has {} values, but `profiles_1d/psi_norm` has {n_psi_norm}",
                values.len()
            ));
        }
    }

    let trapped_fraction: Array1<f64> = trapped_fraction(flux_surface_samples);
    let kinetic_profiles: KineticProfiles = kinetic_profiles(core_profiles_profiles_1d, equilibrium_profiles_1d, SETTINGS.ion_main_index)?;

    let mut j_dot_b: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        // On the magnetic axis the trapped fraction is zero
        if i_psi_norm == 0 {
            j_dot_b[i_psi_norm] = 0.0;
            continue;
        }

        let r_inboard: f64 = equilibrium_profiles_1d.r_inboard[i_psi_norm];
        let r_outboard: f64 = equilibrium_profiles_1d.r_outboard[i_psi_norm];
        let epsilon: f64 = (r_outboard - r_inboard) / (r_outboard + r_inboard);
        let r_major: f64 = 0.5 * (r_outboard + r_inboard);
        // The post-processor leaves `q` NaN on the boundary, as it is infinite on a separatrix
        let q: f64 = if i_psi_norm == n_psi_norm - 1 && equilibrium_profiles_1d.q[i_psi_norm].is_nan() {
            f64::INFINITY
        } else {
            equilibrium_profiles_1d.q[i_psi_norm]
        };
        let zeff: f64 = kinetic_profiles.zeff[i_psi_norm];
        let trapped_fraction_here: f64 = trapped_fraction[i_psi_norm];

        let electrons_collisionality_here: f64 = electrons_collisionality(
            q,
            r_major,
            epsilon,
            kinetic_profiles.electrons_density_thermal[i_psi_norm],
            kinetic_profiles.electrons_temperature[i_psi_norm],
            zeff,
        );
        let ions_collisionality_here: f64 = ions_collisionality(
            q,
            r_major,
            epsilon,
            kinetic_profiles.ions_density_thermal[i_psi_norm],
            kinetic_profiles.ion_main_density_thermal[i_psi_norm],
            kinetic_profiles.ion_main_temperature[i_psi_norm],
            kinetic_profiles.ion_main_charge,
        );

        // Redl's formulae tend to zero as a collisionality tends to infinity, but evaluate to `infinity / infinity` there
        if electrons_collisionality_here.is_infinite() || ions_collisionality_here.is_infinite() {
            j_dot_b[i_psi_norm] = 0.0;
            continue;
        }

        let l31: f64 = redl_l31(trapped_fraction_here, electrons_collisionality_here, zeff);
        let l32: f64 = redl_l32(trapped_fraction_here, electrons_collisionality_here, zeff);
        let alpha: f64 = redl_alpha(trapped_fraction_here, ions_collisionality_here, zeff);

        j_dot_b[i_psi_norm] = 2.0
            * PI
            * equilibrium_profiles_1d.f[i_psi_norm]
            * (l31 * kinetic_profiles.d_pressure_thermal_d_psi[i_psi_norm]
                + l32
                    * ELEMENTARY_CHARGE
                    * kinetic_profiles.electrons_density_thermal[i_psi_norm]
                    * kinetic_profiles.d_electrons_temperature_d_psi[i_psi_norm]
                + l31
                    * alpha
                    * ELEMENTARY_CHARGE
                    * kinetic_profiles.ions_density_thermal[i_psi_norm]
                    * kinetic_profiles.d_ion_main_temperature_d_psi[i_psi_norm]);
    }

    Ok(j_dot_b)
}

/// Redl's `L31`, the coefficient of the pressure gradient, their equations 10 and 11.
///
/// # Arguments
/// * `trapped_fraction` - `f_t` [dimensionless]
/// * `electrons_collisionality` - `nu_e*` [dimensionless]
/// * `zeff` - `Z_eff`, at least 1 [dimensionless]
pub(super) fn redl_l31(trapped_fraction: f64, electrons_collisionality: f64, zeff: f64) -> f64 {
    let collisionality_sqrt: f64 = electrons_collisionality.sqrt();

    // Equation 11
    let x31: f64 = trapped_fraction
        / (1.0
            + 0.67 * (1.0 - 0.7 * trapped_fraction) * collisionality_sqrt / (0.56 + 0.44 * zeff)
            + (0.52 + 0.086 * collisionality_sqrt) * (1.0 + 0.87 * trapped_fraction) * electrons_collisionality / (1.0 + 1.13 * (zeff - 1.0).sqrt()));

    // Equation 10
    let zeff_denominator: f64 = zeff.powf(1.2) - 0.71;
    (1.0 + 0.15 / zeff_denominator) * x31 - 0.22 / zeff_denominator * x31.powi(2)
        + 0.01 / zeff_denominator * x31.powi(3)
        + 0.06 / zeff_denominator * x31.powi(4)
}

/// Redl's `L32`, the coefficient of the electron temperature gradient, their equations 12 to 16: the sum of an electron-electron and an
/// electron-ion part.
///
/// # Arguments
/// * `trapped_fraction` - `f_t` [dimensionless]
/// * `electrons_collisionality` - `nu_e*` [dimensionless]
/// * `zeff` - `Z_eff`, at least 1 [dimensionless]
fn redl_l32(trapped_fraction: f64, electrons_collisionality: f64, zeff: f64) -> f64 {
    let collisionality_sqrt: f64 = electrons_collisionality.sqrt();

    // Equation 14
    let x32_ee: f64 = trapped_fraction
        / (1.0
            + 0.23 * (1.0 - 0.96 * trapped_fraction) * collisionality_sqrt / zeff.sqrt()
            + 0.13 * (1.0 - 0.38 * trapped_fraction) * electrons_collisionality / zeff.powi(2)
                * ((1.0 + 2.0 * (zeff - 1.0).sqrt()).sqrt()
                    + trapped_fraction.powi(2) * ((0.075 + 0.25 * (zeff - 1.0).powi(2)) * electrons_collisionality).sqrt()));

    // Equation 13
    let f32_ee: f64 = (0.1 + 0.6 * zeff) / (zeff * (0.77 + 0.63 * (1.0 + (zeff - 1.0).powf(1.1)))) * (x32_ee - x32_ee.powi(4))
        + 0.7 / (1.0 + 0.2 * zeff) * (x32_ee.powi(2) - x32_ee.powi(4) - 1.2 * (x32_ee.powi(3) - x32_ee.powi(4)))
        + 1.3 / (1.0 + 0.5 * zeff) * x32_ee.powi(4);

    // Equation 16
    let x32_ei: f64 = trapped_fraction
        / (1.0
            + 0.87 * (1.0 + 0.39 * trapped_fraction) * collisionality_sqrt / (1.0 + 2.95 * (zeff - 1.0).powi(2))
            + 1.53 * (1.0 - 0.37 * trapped_fraction) * electrons_collisionality * (2.0 + 0.375 * (zeff - 1.0)));

    // Equation 15
    let f32_ei: f64 = -(0.4 + 1.93 * zeff) / (zeff * (0.8 + 0.6 * zeff)) * (x32_ei - x32_ei.powi(4))
        + 5.5 / (1.5 + 2.0 * zeff) * (x32_ei.powi(2) - x32_ei.powi(4) - 0.8 * (x32_ei.powi(3) - x32_ei.powi(4)))
        - 1.3 / (1.0 + 0.5 * zeff) * x32_ei.powi(4);

    // Equation 12
    f32_ee + f32_ei
}

/// Redl's `alpha`, the ion flow coefficient, their equations 20 and 21.
///
/// # Arguments
/// * `trapped_fraction` - `f_t` [dimensionless]
/// * `ions_collisionality` - `nu_i*` [dimensionless]
/// * `zeff` - `Z_eff`, at least 1 [dimensionless]
fn redl_alpha(trapped_fraction: f64, ions_collisionality: f64, zeff: f64) -> f64 {
    // Equation 20
    let alpha_0: f64 = -(0.62 + 0.055 * (zeff - 1.0)) / (0.53 + 0.17 * (zeff - 1.0)) * (1.0 - trapped_fraction)
        / (1.0 - (0.31 - 0.065 * (zeff - 1.0)) * trapped_fraction - 0.25 * trapped_fraction.powi(2));

    // Equation 21
    let collisionality_sqrt: f64 = ions_collisionality.sqrt();
    ((alpha_0 + 0.7 * zeff * trapped_fraction.sqrt() * collisionality_sqrt) / (1.0 + 0.18 * collisionality_sqrt)
        - 0.002 * ions_collisionality.powi(2) * trapped_fraction.powi(6))
        / (1.0 + 0.004 * ions_collisionality.powi(2) * trapped_fraction.powi(6))
}

#[cfg(test)]
mod tests {
    use super::super::test_equilibrium::{B0, R_AXIS, circular_equilibrium, kinetic_profiles_at};
    use super::*;
    use approx::assert_relative_eq;

    /// Redl's coefficients from TORAX's own `calculate_L31`, `calculate_L32` and `calculate_alpha` (commit b53645a), evaluated in float64,
    /// over the range of trapped fraction, collisionality and effective charge
    #[test]
    fn redl_coefficients_match_torax() {
        // f_t, nu* (nu_e* for L31 and L32, nu_i* for alpha), Z_eff, then TORAX's L31, L32 and alpha
        let cases: [[f64; 6]; 8] = [
            [0.05, 10000.0, 1.5, 1.0987342804336879e-06, 1.870620999966034e-05, 1.1718558576119167],
            [0.24, 0.0001, 1.0, 0.3200285291359702, -0.1768893414815997, -0.9705224173074375],
            [0.45, 0.1, 1.8, 0.4134860909175043, -0.0692514675393886, -0.36973223706350455],
            [0.63, 1.0, 2.5, 0.4008795080501807, 0.1265430019139548, 0.8109112753824838],
            [0.82, 0.01, 1.5, 0.8072111915288439, -0.06783077506317947, -0.21463525878866174],
            [0.82, 10.0, 3.5, 0.13435876146436285, 0.17831809464709433, 3.8178322079295897],
            [0.95, 0.01, 1.8, 0.9317248516871821, 7.053707654824937e-05, 0.02811816503255594],
            [0.95, 100.0, 1.0, 0.005635290127933798, 0.00892476643729444, -0.40484152247692334],
        ];
        let n_case: usize = cases.len();
        for i_case in 0..n_case {
            let [trapped_fraction, collisionality, zeff, l31_torax, l32_torax, alpha_torax]: [f64; 6] = cases[i_case];
            assert_relative_eq!(
                redl_l31(trapped_fraction, collisionality, zeff),
                l31_torax,
                epsilon = 1.0e-15,
                max_relative = 1.0e-12
            );
            assert_relative_eq!(
                redl_l32(trapped_fraction, collisionality, zeff),
                l32_torax,
                epsilon = 1.0e-15,
                max_relative = 1.0e-12
            );
            assert_relative_eq!(
                redl_alpha(trapped_fraction, collisionality, zeff),
                alpha_torax,
                epsilon = 1.0e-15,
                max_relative = 1.0e-12
            );
        }
    }

    #[test]
    fn the_bootstrap_current_is_written_as_its_source_and_flows_with_the_plasma_current() {
        let times: [f64; 2] = [0.1, 0.2];
        let equilibrium: EquilibriumIds = circular_equilibrium(&times);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_bootstrap(&equilibrium, &kinetic_profiles_at(&times), &mut core_sources, &mut plasma_current_components).unwrap();

        assert_eq!(core_sources.source.len(), 1);
        let source: &CoreSourcesSource = &core_sources.source[0];
        assert_eq!(source.identifier.name, "bootstrap_current");
        assert_eq!(source.identifier.index, 13);
        assert_eq!(core_sources.vacuum_toroidal_field.b0, Array1::from_elem(2, B0));
        assert_eq!(source.profiles_1d.len(), 2);
        assert_eq!(source.global_quantities.len(), 2);
        let profiles_1d: &CoreSourcesSourceProfiles1d = &source.profiles_1d[1];
        assert_eq!(profiles_1d.time, 0.2);
        assert_eq!(profiles_1d.grid.psi, equilibrium.time_slice[1].profiles_1d.psi);

        // Zero on the magnetic axis and on the boundary, where `q` is NaN; positive, as the plasma current is, in between
        let j_parallel: &Array1<f64> = &profiles_1d.j_parallel;
        let j_phi: &Array1<f64> = &plasma_current_components.time_slice[1].bootstrap.j_phi;
        let n_psi_norm: usize = j_parallel.len();
        assert_eq!(j_phi.len(), n_psi_norm);
        assert_eq!(j_parallel[0], 0.0);
        assert_eq!(j_parallel[n_psi_norm - 1], 0.0);
        assert_eq!(j_phi[0], 0.0);
        assert_eq!(j_phi[n_psi_norm - 1], 0.0);
        for i_psi_norm in 1..n_psi_norm - 1 {
            assert!(j_parallel[i_psi_norm] > 0.0, "j_parallel({i_psi_norm}) = {}", j_parallel[i_psi_norm]);
            assert!(j_phi[i_psi_norm] > 0.0, "j_phi({i_psi_norm}) = {}", j_phi[i_psi_norm]);
        }

        // Its integrals, which grow outwards to the totals
        let current_parallel_inside: &Array1<f64> = &profiles_1d.current_parallel_inside;
        let current_phi_inside: &Array1<f64> = &plasma_current_components.time_slice[1].bootstrap.current_phi_inside;
        assert_eq!(current_parallel_inside[0], 0.0);
        assert_eq!(current_phi_inside[0], 0.0);
        for i_psi_norm in 1..n_psi_norm {
            assert!(current_parallel_inside[i_psi_norm] > current_parallel_inside[i_psi_norm - 1]);
            assert!(current_phi_inside[i_psi_norm] > current_phi_inside[i_psi_norm - 1]);
        }
        assert_eq!(source.global_quantities[1].current_parallel, current_parallel_inside[n_psi_norm - 1]);

        // The other components are not calculated
        assert!(plasma_current_components.time_slice[1].ohmic.j_phi.is_empty());
        assert!(plasma_current_components.time_slice[1].diamagnetic.j_phi.is_empty());
    }

    #[test]
    fn the_bootstrap_current_replaces_its_own_source_and_keeps_the_others() {
        let times: [f64; 1] = [0.1];
        let equilibrium: EquilibriumIds = circular_equilibrium(&times);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut ohmic: CoreSourcesSource = CoreSourcesSource::default();
        ohmic.identifier.name = "ohmic".to_string();
        ohmic.identifier.index = 7;
        ohmic.profiles_1d = vec![CoreSourcesSourceProfiles1d {
            time: 0.1,
            ..Default::default()
        }];
        core_sources.source.push(ohmic);
        core_sources.vacuum_toroidal_field.r0 = R_AXIS;
        core_sources.vacuum_toroidal_field.b0 = Array1::from_elem(1, B0);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_bootstrap(&equilibrium, &kinetic_profiles_at(&times), &mut core_sources, &mut plasma_current_components).unwrap();
        current_bootstrap(&equilibrium, &kinetic_profiles_at(&times), &mut core_sources, &mut plasma_current_components).unwrap();

        assert_eq!(core_sources.source.len(), 2);
        assert_eq!(core_sources.source[0].identifier.name, "ohmic");
        assert_eq!(core_sources.source[1].identifier.name, "bootstrap_current");
    }

    #[test]
    fn a_time_slice_without_kinetic_profiles_is_nan() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        // Kinetic profiles at the second time-slice only, the first's 1 millisecond out
        current_bootstrap(
            &equilibrium,
            &kinetic_profiles_at(&[0.101, 0.2]),
            &mut core_sources,
            &mut plasma_current_components,
        )
        .unwrap();

        let source: &CoreSourcesSource = &core_sources.source[0];
        let n_psi_norm: usize = source.profiles_1d[0].j_parallel.len();
        for i_psi_norm in 0..n_psi_norm {
            assert!(source.profiles_1d[0].j_parallel[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].bootstrap.j_phi[i_psi_norm].is_nan());
        }
        assert!(source.global_quantities[0].current_parallel.is_nan());
        assert!(source.global_quantities[1].current_parallel > 0.0);
    }

    #[test]
    fn other_sources_at_other_times_are_an_error() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut ohmic: CoreSourcesSource = CoreSourcesSource::default();
        ohmic.identifier.name = "ohmic".to_string();
        ohmic.identifier.index = 7;
        ohmic.profiles_1d = vec![CoreSourcesSourceProfiles1d {
            time: 0.3,
            ..Default::default()
        }];
        core_sources.source.push(ohmic);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        let error: String = current_bootstrap(&equilibrium, &kinetic_profiles_at(&[0.1]), &mut core_sources, &mut plasma_current_components).unwrap_err();

        assert!(error.contains("`ohmic`, is not at the equilibrium's times"), "{error}");
        assert!(plasma_current_components.time_slice[0].bootstrap.j_phi.is_empty());
    }

    #[test]
    fn components_made_from_another_equilibrium_are_an_error() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&circular_equilibrium(&[0.1]));

        let error: String = current_bootstrap(
            &equilibrium,
            &kinetic_profiles_at(&[0.1, 0.2]),
            &mut core_sources,
            &mut plasma_current_components,
        )
        .unwrap_err();

        assert!(error.contains("make them with `PlasmaCurrentComponents::new` from this equilibrium"), "{error}");
        assert!(core_sources.source.is_empty());
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
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        // Kinetic profiles are not needed at an unusable time-slice
        current_bootstrap(&equilibrium, &kinetic_profiles_at(&[0.2]), &mut core_sources, &mut plasma_current_components).unwrap();

        let source: &CoreSourcesSource = &core_sources.source[0];
        let n_psi_norm: usize = source.profiles_1d[0].j_parallel.len();
        for i_psi_norm in 0..n_psi_norm {
            assert!(source.profiles_1d[0].j_parallel[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].bootstrap.j_phi[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].bootstrap.current_phi_inside[i_psi_norm].is_nan());
        }
        assert!(source.global_quantities[0].current_parallel.is_nan());
        assert!(source.global_quantities[1].current_parallel > 0.0);
        assert!(plasma_current_components.time_slice[1].bootstrap.current_phi_inside[n_psi_norm - 1] > 0.0);
    }
}
