//! The ohmic current, from the neoclassical conductivity of Redl, Angioni, Belli and Sauter, Physics of Plasmas 28, 022502 (2021), their
//! equations 17 and 18, and the Spitzer conductivity of Sauter, Angioni and Lin-Liu, Physics of Plasmas 6, 2834 (1999), their equation 18a.

use super::collisionality::{electrons_collisionality, electrons_coulomb_logarithm};
use super::core_sources_source::{core_sources_grid, write_core_sources_source};
use super::flux_surface_samples::{FluxSurfaceSamples, flux_surface_samples};
use super::kinetic_profiles::{ElectronProfiles, core_profiles_time_index, electron_profiles};
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, integral_inside, j_phi_of_parallel_current};
use super::trapped_fraction::trapped_fraction;
use imas_rs::ids::core_profiles::CoreProfiles as CoreProfilesIds;
use imas_rs::ids::core_sources::{CoreSources as CoreSourcesIds, CoreSourcesSource, CoreSourcesSourceGlobal, CoreSourcesSourceProfiles1d, Identifier};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumProfiles1d, EquilibriumTimeSlice};
use ndarray::Array1;

/// The ohmic current's `core_sources/source(i1)/identifier`, from the data dictionary's `core_source_identifier.xml`
const IDENTIFIER_NAME: &str = "ohmic";
const IDENTIFIER_INDEX: i32 = 7;
const IDENTIFIER_DESCRIPTION: &str = "Source from ohmic heating";

/// Calculate the neoclassical conductivity and the ohmic current on every time-slice of the equilibrium, and write them to `core_sources`,
/// as the source `ohmic`, and the current to the plasma current's components, as `ohmic`.
///
/// The ohmic current is the parallel electric field times the neoclassical conductivity, `<J . B> = sigma_neo <E . B>`. The conductivity is
/// the Spitzer conductivity, Sauter's equation 18a,
///
/// ```text
/// sigma_spitzer = 1.9012e4 T_e ** 1.5 / (Z_eff N(Z_eff) ln_lambda_e),  N(Z) = 0.58 + 0.74 / (0.76 + Z)
/// ```
///
/// [siemens / metre, with `T_e` in electron_volt], reduced by the trapped electrons, which carry no current, with Redl's equations 17 and 18,
/// `sigma_neo = sigma_spitzer F33(f_t, nu_e*, Z_eff)`. `f_t` is Lin-Liu and Miller's trapped fraction (`trapped_fraction.rs`) and `nu_e*`
/// the electron collisionality (`collisionality.rs`), with the inverse aspect ratio and major radius of each surface from its radii at the
/// height of the magnetic axis, as in the bootstrap current. On the magnetic axis, and on the boundary, where the equilibrium's `q` is NaN
/// as it is infinite on a separatrix, `nu_e*` is infinite, no electron is trapped, and `sigma_neo = sigma_spitzer`.
///
/// `<E . B>` is the kinetic profiles' `core_profiles/profiles_1d(itime)/e_field/parallel` times their `core_profiles/vacuum_toroidal_field/b0`.
/// Without it, only the conductivity is written, and the current is NaN.
///
/// The source holds, on each time-slice:
/// * `profiles_1d(itime)/conductivity_parallel`, `sigma_neo`, on the flux surfaces of the equilibrium's `profiles_1d`, whose coordinates
///   are copied into `profiles_1d(itime)/grid`
/// * `profiles_1d(itime)/j_parallel`, `<J . B> / B0`, with `B0` the equilibrium's `vacuum_toroidal_field/b0`
/// * `profiles_1d(itime)/current_parallel_inside`, its integral over the cross-section inside each surface, by the trapezium rule in
///   `profiles_1d/area`
/// * `global_quantities(itime)/current_parallel`, the same integral inside the boundary
///
/// and the plasma current's `ohmic` component holds its toroidal current density, `<j_phi / R> / <1 / R>`, the toroidal current inside
/// each surface (see `j_phi_of_parallel_current`), and the source's `j_parallel`. A time-slice the solver could not use, with a negative `code/output_flag`, or with no
/// kinetic profiles at its time, has NaN in each.
///
/// # Arguments
/// * `equilibrium` - the Equilibrium IDS, post-processed: on each time-slice `profiles_1d` with `psi_norm`, `rho_tor_norm`, `f`, `q`,
///   `r_inboard`, `r_outboard` and `area`, and `profiles_2d(0)`, the magnetic axis and the boundary, for the flux surfaces
/// * `core_profiles` - the kinetic profiles, on `profiles_1d(:)`, at the time of each equilibrium time-slice which has them: see
///   `electron_profiles` for the nodes read, and `vacuum_toroidal_field/b0`, one per `profiles_1d(:)`, with `e_field/parallel`
/// * `core_sources` - the plasma's sources, which the conductivity and the ohmic current are written into (see `write_core_sources_source`)
/// * `plasma_current_components` - the plasma current's components, made from `equilibrium` by `PlasmaCurrentComponents::new`
///
/// # Errors
/// The kinetic profiles at an equilibrium time-slice's time are not usable (see `electron_profiles`), or have `e_field/parallel`
/// without `vacuum_toroidal_field/b0`; the equilibrium's `profiles_1d` is not filled, or its `vacuum_toroidal_field/b0` does not have one
/// value per time-slice; `core_sources` holds other sources at other times, or with another vacuum toroidal field; or
/// `plasma_current_components` was not made from `equilibrium`. Neither `core_sources` nor `plasma_current_components` is changed then.
pub fn current_ohmic(
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

        let mut conductivity_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let mut j_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        // Without kinetic profiles at the time-slice's time, the conductivity and the ohmic current are NaN on it
        if usable && let Some(i_time_core_profiles) = core_profiles_time_index(core_profiles, time_slice.time) {
            let samples: Vec<Option<FluxSurfaceSamples>> =
                flux_surface_samples(time_slice).map_err(|reason: String| format!("ohmic current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            let electron_profiles_here: ElectronProfiles = electron_profiles(&core_profiles.profiles_1d[i_time_core_profiles], &time_slice.profiles_1d)
                .map_err(|reason: String| format!("ohmic current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            conductivity_parallel = conductivity_on_time_slice(time_slice, &samples, &electron_profiles_here)
                .map_err(|reason: String| format!("ohmic current at `equilibrium/time_slice({i_time})`: {reason}"))?;

            // `<E . B>`, from `e_field/parallel = <E . B> / B0` with the kinetic profiles' own `B0`
            if let Some(e_field_parallel) = &electron_profiles_here.e_field_parallel {
                let b0_core_profiles: f64 = match core_profiles.vacuum_toroidal_field.b0.get(i_time_core_profiles) {
                    Some(b0_here) => *b0_here,
                    None => {
                        return Err(format!(
                            "ohmic current at `equilibrium/time_slice({i_time})`: `core_profiles/profiles_1d({i_time_core_profiles})/e_field/parallel` is \
                             normalised by `core_profiles/vacuum_toroidal_field/b0({i_time_core_profiles})`, which is not there"
                        ));
                    }
                };
                let j_dot_b: Array1<f64> = &conductivity_parallel * e_field_parallel * b0_core_profiles;
                j_parallel = &j_dot_b / b0[i_time];
                j_phi = j_phi_of_parallel_current(&j_dot_b, &time_slice.profiles_1d.f, &samples, time_slice.global_quantities.magnetic_axis.r);
            }
        }

        components.push(PlasmaCurrentComponent::new(j_phi, j_parallel.clone(), area));
        let current_parallel_inside: Array1<f64> = integral_inside(&j_parallel, area);
        let current_parallel: f64 = if n_psi_norm > 0 { current_parallel_inside[n_psi_norm - 1] } else { f64::NAN };
        profiles_1d.push(CoreSourcesSourceProfiles1d {
            grid: core_sources_grid(time_slice),
            j_parallel,
            current_parallel_inside,
            conductivity_parallel,
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
        plasma_current_components.time_slice[i_time].ohmic = std::mem::take(&mut components[i_time]);
    }

    Ok(())
}

/// The neoclassical conductivity, `sigma_neo`, on the flux surfaces of one equilibrium time-slice; see `current_ohmic`.
///
/// # Arguments
/// * `time_slice` - the post-processed equilibrium time-slice
/// * `flux_surface_samples` - the field sampled on its flux surfaces, `flux_surface_samples`
/// * `electron_profiles` - the electron profiles and the effective charge on its flux surfaces
///
/// # Returns
/// `sigma_neo` on each surface of `profiles_1d/psi_norm` [siemens / metre]
///
/// # Errors
/// The equilibrium's `profiles_1d` is not filled
fn conductivity_on_time_slice(
    time_slice: &EquilibriumTimeSlice,
    flux_surface_samples: &[Option<FluxSurfaceSamples>],
    electron_profiles: &ElectronProfiles,
) -> Result<Array1<f64>, String> {
    let equilibrium_profiles_1d: &EquilibriumProfiles1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = equilibrium_profiles_1d.psi_norm.len();
    let named_profiles: [(&str, &Array1<f64>); 3] = [
        ("q", &equilibrium_profiles_1d.q),
        ("r_inboard", &equilibrium_profiles_1d.r_inboard),
        ("r_outboard", &equilibrium_profiles_1d.r_outboard),
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
    let mut conductivity: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
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
        let electrons_density: f64 = electron_profiles.electrons_density_thermal[i_psi_norm];
        let electrons_temperature: f64 = electron_profiles.electrons_temperature[i_psi_norm];
        let zeff: f64 = electron_profiles.zeff[i_psi_norm];

        let electrons_collisionality_here: f64 = electrons_collisionality(q, r_major, epsilon, electrons_density, electrons_temperature, zeff);
        let conductivity_spitzer: f64 = spitzer_conductivity(electrons_density, electrons_temperature, zeff);
        conductivity[i_psi_norm] = conductivity_spitzer * redl_f33(trapped_fraction[i_psi_norm], electrons_collisionality_here, zeff);
    }

    Ok(conductivity)
}

/// The Spitzer conductivity, Sauter's equation 18a.
///
/// # Arguments
/// * `electrons_density` - the thermal electron density [1 / metre ** 3]
/// * `electrons_temperature` - the electron temperature [electron_volt]
/// * `zeff` - the effective charge [dimensionless]
///
/// # Returns
/// `sigma_spitzer` [siemens / metre]
fn spitzer_conductivity(electrons_density: f64, electrons_temperature: f64, zeff: f64) -> f64 {
    let n_of_zeff: f64 = 0.58 + 0.74 / (0.76 + zeff);
    1.9012e4 * electrons_temperature.powf(1.5) / (zeff * n_of_zeff * electrons_coulomb_logarithm(electrons_density, electrons_temperature))
}

/// Redl's `F33 = sigma_neo / sigma_spitzer`, their equations 17 and 18.
///
/// # Arguments
/// * `trapped_fraction` - `f_t` [dimensionless]
/// * `electrons_collisionality` - `nu_e*` [dimensionless]
/// * `zeff` - `Z_eff`, at least 1 [dimensionless]
fn redl_f33(trapped_fraction: f64, electrons_collisionality: f64, zeff: f64) -> f64 {
    // Equation 18
    let x33: f64 = trapped_fraction
        / (1.0
            + 0.25 * (1.0 - 0.7 * trapped_fraction) * electrons_collisionality.sqrt() * (1.0 + 0.45 * (zeff - 1.0).sqrt())
            + 0.61 * (1.0 - 0.41 * trapped_fraction) * electrons_collisionality / zeff.sqrt());

    // Equation 17
    1.0 - (1.0 + 0.21 / zeff) * x33 + 0.54 / zeff * x33.powi(2) - 0.33 / zeff * x33.powi(3)
}

#[cfg(test)]
mod tests {
    use super::super::test_equilibrium::{B0, R_AXIS, circular_equilibrium, kinetic_profiles_at};
    use super::*;
    use approx::assert_relative_eq;

    /// Redl's `F33` from TORAX's own `calculate_L33` (commit b53645a), evaluated in float64
    #[test]
    fn redl_f33_matches_torax() {
        // f_t, nu_e*, Z_eff, then TORAX's F33
        let cases: [[f64; 4]; 7] = [
            [0.05, 0.0001, 1.5, 0.9440504377153149],
            [0.24, 0.0001, 1.0, 0.736657467357458],
            [0.45, 0.1, 1.8, 0.585451227964041],
            [0.63, 1.0, 2.5, 0.5738356204811285],
            [0.82, 0.01, 1.5, 0.19981610220222548],
            [0.82, 10.0, 3.5, 0.7740628369512923],
            [0.95, 100.0, 1.0, 0.9708988672661197],
        ];
        for [trapped_fraction, collisionality, zeff, f33_torax] in cases {
            assert_relative_eq!(redl_f33(trapped_fraction, collisionality, zeff), f33_torax, max_relative = 1.0e-12);
        }

        // Nothing is trapped when the collisionality is infinite
        assert_eq!(redl_f33(0.5, f64::INFINITY, 2.0), 1.0);
    }

    /// The Spitzer conductivity from TORAX's own `calculate_spitzer_conductivity` and `calculate_log_lambda_ei` (commit b53645a), evaluated in
    /// float64
    #[test]
    fn spitzer_conductivity_matches_torax() {
        // n_e, T_e, Z_eff, then TORAX's sigma_spitzer
        let cases: [[f64; 4]; 4] = [
            [1e+19, 1000.0, 1.0, 36792495.58244928],
            [3e+19, 300.0, 1.5, 4977926.491803459],
            [5e+18, 50.0, 3.0, 210781.89474649535],
            [2e+20, 5000.0, 2.0, 240973797.14837876],
        ];
        for [electrons_density, electrons_temperature, zeff, conductivity_torax] in cases {
            assert_relative_eq!(
                spitzer_conductivity(electrons_density, electrons_temperature, zeff),
                conductivity_torax,
                max_relative = 1.0e-12
            );
        }
    }

    #[test]
    fn without_a_parallel_electric_field_only_the_conductivity_is_written() {
        let times: [f64; 2] = [0.1, 0.2];
        let equilibrium: EquilibriumIds = circular_equilibrium(&times);
        let core_profiles: CoreProfilesIds = kinetic_profiles_at(&times);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_ohmic(&equilibrium, &core_profiles, &mut core_sources, &mut plasma_current_components).unwrap();

        assert_eq!(core_sources.source.len(), 1);
        let source: &CoreSourcesSource = &core_sources.source[0];
        assert_eq!(source.identifier.name, "ohmic");
        assert_eq!(source.identifier.index, 7);
        let profiles_1d: &CoreSourcesSourceProfiles1d = &source.profiles_1d[1];
        assert_eq!(profiles_1d.time, 0.2);

        // The Spitzer conductivity where nothing is trapped: on the magnetic axis, and on the boundary, where `q` is NaN; below it in
        // between
        let electron_profiles_here: ElectronProfiles = electron_profiles(&core_profiles.profiles_1d[1], &equilibrium.time_slice[1].profiles_1d).unwrap();
        let conductivity: &Array1<f64> = &profiles_1d.conductivity_parallel;
        let n_psi_norm: usize = conductivity.len();
        for i_psi_norm in 0..n_psi_norm {
            let conductivity_spitzer: f64 = spitzer_conductivity(
                electron_profiles_here.electrons_density_thermal[i_psi_norm],
                electron_profiles_here.electrons_temperature[i_psi_norm],
                electron_profiles_here.zeff[i_psi_norm],
            );
            if i_psi_norm == 0 || i_psi_norm == n_psi_norm - 1 {
                assert_relative_eq!(conductivity[i_psi_norm], conductivity_spitzer, max_relative = 1.0e-14);
            } else {
                assert!(conductivity[i_psi_norm] > 0.0 && conductivity[i_psi_norm] < conductivity_spitzer);
            }
        }

        // No current
        for i_psi_norm in 0..n_psi_norm {
            assert!(profiles_1d.j_parallel[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[1].ohmic.j_phi[i_psi_norm].is_nan());
        }
        assert!(source.global_quantities[1].current_parallel.is_nan());
    }

    #[test]
    fn a_time_slice_the_solver_could_not_use_is_nan() {
        let times: [f64; 2] = [0.1, 0.2];
        let mut equilibrium: EquilibriumIds = circular_equilibrium(&times);
        // As the solver leaves a time-slice it did not converge on: not post-processed, with only `profiles_1d/psi_norm`
        equilibrium.code.output_flag[0] = -1;
        equilibrium.time_slice[0].profiles_1d = EquilibriumProfiles1d {
            psi_norm: equilibrium.time_slice[0].profiles_1d.psi_norm.clone(),
            ..Default::default()
        };
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_ohmic(&equilibrium, &kinetic_profiles_at(&times), &mut core_sources, &mut plasma_current_components).unwrap();

        let source: &CoreSourcesSource = &core_sources.source[0];
        let n_psi_norm: usize = equilibrium.time_slice[0].profiles_1d.psi_norm.len();
        for i_psi_norm in 0..n_psi_norm {
            assert!(source.profiles_1d[0].conductivity_parallel[i_psi_norm].is_nan());
            assert!(source.profiles_1d[0].current_parallel_inside[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].ohmic.j_phi[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].ohmic.current_phi_inside[i_psi_norm].is_nan());
        }
        assert!(source.profiles_1d[1].conductivity_parallel[0] > 0.0);
    }

    #[test]
    fn the_ohmic_current_is_the_conductivity_times_the_parallel_electric_field() {
        let times: [f64; 1] = [0.1];
        let equilibrium: EquilibriumIds = circular_equilibrium(&times);
        let mut core_profiles: CoreProfilesIds = kinetic_profiles_at(&times);
        // `e_field/parallel = <E . B> / B0`, normalised by the kinetic profiles' own `B0`, which is not the equilibrium's
        let e_field_parallel: f64 = 0.1;
        let b0_core_profiles: f64 = 2.0;
        let n_rho_pol_norm: usize = core_profiles.profiles_1d[0].grid.rho_pol_norm.len();
        core_profiles.profiles_1d[0].e_field.parallel = Array1::from_elem(n_rho_pol_norm, e_field_parallel);
        core_profiles.vacuum_toroidal_field.b0 = Array1::from_elem(1, b0_core_profiles);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_ohmic(&equilibrium, &core_profiles, &mut core_sources, &mut plasma_current_components).unwrap();

        let profiles_1d: &CoreSourcesSourceProfiles1d = &core_sources.source[0].profiles_1d[0];
        let component: &PlasmaCurrentComponent = &plasma_current_components.time_slice[0].ohmic;
        let n_psi_norm: usize = profiles_1d.j_parallel.len();
        for i_psi_norm in 0..n_psi_norm {
            let j_dot_b: f64 = profiles_1d.conductivity_parallel[i_psi_norm] * e_field_parallel * b0_core_profiles;
            assert_relative_eq!(profiles_1d.j_parallel[i_psi_norm], j_dot_b / B0, max_relative = 1.0e-14);
            assert!(component.j_phi[i_psi_norm] > 0.0);
        }
        // On the magnetic axis, `j_phi = R <J . B> / F`
        let f_axis: f64 = equilibrium.time_slice[0].profiles_1d.f[0];
        assert_relative_eq!(
            component.j_phi[0],
            R_AXIS * profiles_1d.conductivity_parallel[0] * e_field_parallel * b0_core_profiles / f_axis,
            max_relative = 1.0e-14
        );
        assert_eq!(
            core_sources.source[0].global_quantities[0].current_parallel,
            profiles_1d.current_parallel_inside[n_psi_norm - 1]
        );
        assert!(component.current_phi_inside[n_psi_norm - 1] > 0.0);
    }

    #[test]
    fn a_parallel_electric_field_without_its_vacuum_field_is_an_error() {
        let times: [f64; 1] = [0.1];
        let equilibrium: EquilibriumIds = circular_equilibrium(&times);
        let mut core_profiles: CoreProfilesIds = kinetic_profiles_at(&times);
        let n_rho_pol_norm: usize = core_profiles.profiles_1d[0].grid.rho_pol_norm.len();
        core_profiles.profiles_1d[0].e_field.parallel = Array1::from_elem(n_rho_pol_norm, 0.1);
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        let error: String = current_ohmic(&equilibrium, &core_profiles, &mut core_sources, &mut plasma_current_components).unwrap_err();

        assert!(error.contains("`core_profiles/vacuum_toroidal_field/b0(0)`, which is not there"), "{error}");
        assert!(core_sources.source.is_empty());
        assert!(plasma_current_components.time_slice[0].ohmic.j_phi.is_empty());
    }
}
