//! The beam-driven current: the current of the neutral beams' fast ions, which the fast-ion code gives, shielded by the electrons, with the
//! trapped-electron correction of Lin-Liu and Hinton, Physics of Plasmas 4, 4179 (1997), as Redl's `L31`.

use super::collisionality::electrons_collisionality;
use super::current_bootstrap::redl_l31;
use super::flux_surface_samples::{FluxSurfaceSamples, flux_surface_samples};
use super::kinetic_profiles::{ElectronProfiles, core_profiles_time_index, electron_profiles, nearest_time_index, on_surfaces, psi_norm_of_rho_pol_norm};
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, j_phi_of_parallel_current};
use super::trapped_fraction::trapped_fraction;
use imas_rs::ids::core_profiles::{CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d};
use imas_rs::ids::core_sources::{CoreSources as CoreSourcesIds, CoreSourcesSource, CoreSourcesSourceProfiles1d};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumProfiles1d, EquilibriumTimeSlice};
use ndarray::Array1;

/// The neutral beams' `core_sources/source(i1)/identifier/index`, from the data dictionary's `core_source_identifier.xml`
const NBI_IDENTIFIER_INDEX: i32 = 2;

/// Calculate the beam-driven current on every time-slice of the equilibrium, and write it to the plasma current's components, as
/// `beam_driven`.
///
/// The neutral beams' fast ions carry a current along the field lines, `<J_fast . B>`, which the fast-ion code, e.g. ASCOT5, gives, and
/// which `core_sources` holds as the neutral beams' source ("NBI", `identifier/index` 2): `profiles_1d(itime)/j_parallel = <J_fast . B> /
/// B0`, with `B0` its `vacuum_toroidal_field/b0`, on `profiles_1d(itime)/grid/rho_pol_norm`. The electrons partly cancel it, and the net,
/// beam-driven, current is, on each flux surface,
///
/// ```text
/// <J_NBCD . B> = (1 - (Z_b / Z_eff) (1 - L31)) <J_fast . B>
/// ```
///
/// with `Z_b` the beam ions' charge, the source's `species/ion/z_ion`, and `L31` Redl's coefficient of the pressure gradient, with the same
/// trapped fraction, `f_t`, electron collisionality, `nu_e*`, and `Z_eff` as the bootstrap current (`current_bootstrap`). On the magnetic
/// axis `f_t = 0`, and where `nu_e*` is infinite, as on the boundary, where the equilibrium's `q` is NaN, `L31` tends to zero, so the factor
/// is `1 - Z_b / Z_eff` there.
///
/// `<J_fast . B>` is interpolated onto the flux surfaces linearly in `psi_norm = rho_pol_norm ** 2`, and is NaN outside the source's grid.
///
/// The `beam_driven` component holds the toroidal current density, `<j_phi / R> / <1 / R>`, and the toroidal current inside each surface
/// (see `j_phi_of_parallel_current`), and `<J_NBCD . B> / B0`, with `B0` the equilibrium's `vacuum_toroidal_field/b0`. A time-slice the
/// solver could not use, with a negative `code/output_flag`, or with no kinetic profiles, or no neutral beams' `profiles_1d`, at its time,
/// has NaN in each. Nothing is written to `core_sources`: the neutral beams' source keeps the fast ions' own current.
///
/// # Arguments
/// * `equilibrium` - the Equilibrium IDS, post-processed: on each time-slice `profiles_1d` with `psi_norm`, `f`, `q`, `r_inboard`,
///   `r_outboard` and `area`, and `profiles_2d(0)`, the magnetic axis and the boundary, for the flux surfaces
/// * `core_profiles` - the kinetic profiles, on `profiles_1d(:)`, at the time of each equilibrium time-slice which has them: the electrons
///   and `zeff`, see `electron_profiles`
/// * `core_sources` - the plasma's sources, holding the neutral beams' source, with `species/ion/z_ion`, `profiles_1d(:)` at the time of
///   each equilibrium time-slice which has them, and `core_sources/vacuum_toroidal_field/b0`, one per `profiles_1d(:)`
/// * `plasma_current_components` - the plasma current's components, made from `equilibrium` by `PlasmaCurrentComponents::new`
///
/// # Errors
/// `core_sources` has no neutral beams' source, its `species/ion/z_ion` is not a positive number, its `vacuum_toroidal_field/b0` does not
/// have one value per `profiles_1d(:)`, or a `profiles_1d(itime)` used is not on a usable grid; the kinetic profiles at an equilibrium
/// time-slice's time are not usable (see `electron_profiles`); the equilibrium's `profiles_1d` is not filled; or `plasma_current_components`
/// was not made from `equilibrium`. `plasma_current_components` is not changed then.
pub fn current_beam_driven(
    equilibrium: &EquilibriumIds,
    core_profiles: &CoreProfilesIds,
    core_sources: &CoreSourcesIds,
    plasma_current_components: &mut PlasmaCurrentComponents,
) -> Result<(), String> {
    plasma_current_components.check_made_from(equilibrium)?;

    // The neutral beams' source, and the beam ions' charge
    let n_source: usize = core_sources.source.len();
    let mut i_source_nbi: Option<usize> = None;
    for i_source in 0..n_source {
        if core_sources.source[i_source].identifier.index == NBI_IDENTIFIER_INDEX {
            i_source_nbi = Some(i_source);
        }
    }
    let source_nbi: &CoreSourcesSource = match i_source_nbi {
        Some(i_source) => &core_sources.source[i_source],
        None => {
            return Err(format!(
                "`core_sources` has no neutral beams' source, with `identifier/index` {NBI_IDENTIFIER_INDEX}, to make the beam-driven current from"
            ));
        }
    };
    let beam_charge: f64 = source_nbi.species.ion.z_ion;
    if !(beam_charge.is_finite() && beam_charge > 0.0) {
        return Err(format!(
            "`core_sources/source(NBI)/species/ion/z_ion` = {beam_charge}, the beam ions' charge, is not a positive number"
        ));
    }
    let n_time_source: usize = source_nbi.profiles_1d.len();
    let b0_source: &Array1<f64> = &core_sources.vacuum_toroidal_field.b0;
    if b0_source.len() != n_time_source {
        return Err(format!(
            "`core_sources/vacuum_toroidal_field/b0` has {} values, but the neutral beams' source has {n_time_source} `profiles_1d`",
            b0_source.len()
        ));
    }
    let mut times_source: Array1<f64> = Array1::from_elem(n_time_source, f64::NAN);
    for i_time_source in 0..n_time_source {
        times_source[i_time_source] = source_nbi.profiles_1d[i_time_source].time;
    }

    let n_time: usize = equilibrium.time_slice.len();
    // The component's `<J . B> / B0` is normalised by the equilibrium's `B0`, as every component's is
    let b0: &Array1<f64> = &equilibrium.vacuum_toroidal_field.b0;
    if b0.len() != n_time {
        return Err(format!(
            "`equilibrium/vacuum_toroidal_field/b0` has {} values, but `equilibrium/time_slice` has {n_time} time-slices",
            b0.len()
        ));
    }
    let mut components: Vec<PlasmaCurrentComponent> = Vec::with_capacity(n_time);
    for i_time in 0..n_time {
        let time_slice: &EquilibriumTimeSlice = &equilibrium.time_slice[i_time];
        let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

        // A negative `code/output_flag` means the time-slice shall not be used. The solver did not converge on it, so its `profiles_1d` was
        // not post-processed, and only `psi_norm` is filled: its `area` is checked and used only when it is usable. Without kinetic profiles,
        // or the neutral beams' source, at the time-slice's time, the beam-driven current is NaN on it
        let usable: bool = equilibrium.code.output_flag.len() != n_time || equilibrium.code.output_flag[i_time] >= 0;
        let area_not_usable: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let area: &Array1<f64> = if usable { &time_slice.profiles_1d.area } else { &area_not_usable };
        if area.len() != n_psi_norm {
            return Err(format!(
                "`equilibrium/time_slice({i_time})/profiles_1d/area` has {} values, but `profiles_1d/psi_norm` has {n_psi_norm}",
                area.len()
            ));
        }

        let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let mut j_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        if usable
            && let Some(i_time_core_profiles) = core_profiles_time_index(core_profiles, time_slice.time)
            && let Some(i_time_source) = nearest_time_index(&times_source, time_slice.time)
        {
            let samples: Vec<Option<FluxSurfaceSamples>> =
                flux_surface_samples(time_slice).map_err(|reason: String| format!("beam-driven current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            let j_dot_b: Array1<f64> = j_dot_b_on_time_slice(
                time_slice,
                &samples,
                &core_profiles.profiles_1d[i_time_core_profiles],
                &source_nbi.profiles_1d[i_time_source],
                b0_source[i_time_source],
                beam_charge,
            )
            .map_err(|reason: String| format!("beam-driven current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            j_parallel = &j_dot_b / b0[i_time];
            j_phi = j_phi_of_parallel_current(&j_dot_b, &time_slice.profiles_1d.f, &samples, time_slice.global_quantities.magnetic_axis.r);
        }
        components.push(PlasmaCurrentComponent::new(j_phi, j_parallel, area));
    }

    for i_time in 0..n_time {
        plasma_current_components.time_slice[i_time].beam_driven = std::mem::take(&mut components[i_time]);
    }

    Ok(())
}

/// The flux-surface averaged beam-driven current, `<J_NBCD . B>`, on the flux surfaces of one equilibrium time-slice; see
/// `current_beam_driven`.
///
/// # Arguments
/// * `time_slice` - the post-processed equilibrium time-slice
/// * `flux_surface_samples` - the field sampled on its flux surfaces, `flux_surface_samples`
/// * `core_profiles_profiles_1d` - the kinetic profiles at the time-slice's time
/// * `source_profiles_1d` - the neutral beams' source at the time-slice's time: `grid/rho_pol_norm` and `j_parallel`
/// * `b0_source` - the source's `core_sources/vacuum_toroidal_field/b0` at that time, which `j_parallel` is normalised by [tesla]
/// * `beam_charge` - the beam ions' charge, `Z_b` [dimensionless]
///
/// # Returns
/// `<J_NBCD . B>` on each surface of `profiles_1d/psi_norm` [ampere * tesla / metre ** 2]
///
/// # Errors
/// The equilibrium's `profiles_1d` is not filled, the kinetic profiles are not usable, or the source's grid is not usable
fn j_dot_b_on_time_slice(
    time_slice: &EquilibriumTimeSlice,
    flux_surface_samples: &[Option<FluxSurfaceSamples>],
    core_profiles_profiles_1d: &CoreProfilesProfiles1d,
    source_profiles_1d: &CoreSourcesSourceProfiles1d,
    b0_source: f64,
    beam_charge: f64,
) -> Result<Array1<f64>, String> {
    let equilibrium_profiles_1d: &EquilibriumProfiles1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = equilibrium_profiles_1d.psi_norm.len();
    let named_profiles: [(&str, &Array1<f64>); 4] = [
        ("f", &equilibrium_profiles_1d.f),
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

    // The fast ions' current, `<J_fast . B>`, on the flux surfaces
    let source_psi_norm: Array1<f64> = psi_norm_of_rho_pol_norm(
        "`core_sources/source(NBI)/profiles_1d(itime)/grid/rho_pol_norm`",
        &source_profiles_1d.grid.rho_pol_norm,
    )?;
    let j_parallel_source: &Array1<f64> = &source_profiles_1d.j_parallel;
    if j_parallel_source.len() != source_psi_norm.len() {
        return Err(format!(
            "`core_sources/source(NBI)/profiles_1d(itime)/j_parallel` has {} values, but `grid/rho_pol_norm` has {}",
            j_parallel_source.len(),
            source_psi_norm.len()
        ));
    }
    let j_fast_dot_b: Array1<f64> = on_surfaces(&equilibrium_profiles_1d.psi_norm, &source_psi_norm, j_parallel_source) * b0_source;

    let trapped_fraction: Array1<f64> = trapped_fraction(flux_surface_samples);
    let electron_profiles_here: ElectronProfiles = electron_profiles(core_profiles_profiles_1d, equilibrium_profiles_1d)?;

    let mut j_dot_b: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        let zeff: f64 = electron_profiles_here.zeff[i_psi_norm];
        // On the magnetic axis the trapped fraction is zero, so is `L31`
        let l31: f64 = if i_psi_norm == 0 {
            0.0
        } else {
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
            let electrons_collisionality_here: f64 = electrons_collisionality(
                q,
                r_major,
                epsilon,
                electron_profiles_here.electrons_density_thermal[i_psi_norm],
                electron_profiles_here.electrons_temperature[i_psi_norm],
                zeff,
            );
            // Redl's `L31` tends to zero as the collisionality tends to infinity, but evaluates to `infinity / infinity` there
            if electrons_collisionality_here.is_infinite() {
                0.0
            } else {
                redl_l31(trapped_fraction[i_psi_norm], electrons_collisionality_here, zeff)
            }
        };
        let shielding: f64 = 1.0 - (beam_charge / zeff) * (1.0 - l31);
        j_dot_b[i_psi_norm] = shielding * j_fast_dot_b[i_psi_norm];
    }

    Ok(j_dot_b)
}

#[cfg(test)]
mod tests {
    use super::super::test_equilibrium::{B0, F_PARAMAGNETIC, R_AXIS, circular_equilibrium, kinetic_profiles_at};
    use super::*;
    use approx::assert_relative_eq;
    use imas_rs::ids::core_sources::Identifier;

    /// The fast ions' `j_parallel`, the same on every surface [ampere / metre ** 2]
    const J_PARALLEL_FAST: f64 = 1.0e5;

    /// `core_sources` with the neutral beams' source alone, at each of `times`, of ions of charge `beam_charge`, with `j_parallel` =
    /// `J_PARALLEL_FAST` on 21 points of `rho_pol_norm` from 0 to 1, normalised by the test equilibrium's vacuum field
    fn core_sources_nbi_at(times: &[f64], beam_charge: f64) -> CoreSourcesIds {
        let n_rho_pol_norm: usize = 21;
        let mut core_sources: CoreSourcesIds = CoreSourcesIds::default();
        let mut source: CoreSourcesSource = CoreSourcesSource {
            identifier: Identifier {
                name: "NBI".to_string(),
                index: NBI_IDENTIFIER_INDEX,
                description: String::new(),
            },
            ..Default::default()
        };
        source.species.ion.z_ion = beam_charge;
        let n_time: usize = times.len();
        for i_time in 0..n_time {
            let mut profiles_1d: CoreSourcesSourceProfiles1d = CoreSourcesSourceProfiles1d {
                j_parallel: Array1::from_elem(n_rho_pol_norm, J_PARALLEL_FAST),
                time: times[i_time],
                ..Default::default()
            };
            profiles_1d.grid.rho_pol_norm = Array1::linspace(0.0, 1.0, n_rho_pol_norm);
            source.profiles_1d.push(profiles_1d);
        }
        core_sources.source.push(source);
        core_sources.vacuum_toroidal_field.r0 = R_AXIS;
        core_sources.vacuum_toroidal_field.b0 = Array1::from_elem(n_time, B0);

        core_sources
    }

    /// `<J_NBCD . B>` on the test equilibrium's surfaces, whose kinetic profiles have `Z_eff = 2`
    fn j_dot_b_on_test_equilibrium(beam_charge: f64) -> Array1<f64> {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let core_profiles: CoreProfilesIds = kinetic_profiles_at(&[0.1]);
        let core_sources: CoreSourcesIds = core_sources_nbi_at(&[0.1], beam_charge);
        let time_slice: &EquilibriumTimeSlice = &equilibrium.time_slice[0];
        let samples: Vec<Option<FluxSurfaceSamples>> = flux_surface_samples(time_slice).unwrap();

        j_dot_b_on_time_slice(
            time_slice,
            &samples,
            &core_profiles.profiles_1d[0],
            &core_sources.source[0].profiles_1d[0],
            B0,
            beam_charge,
        )
        .unwrap()
    }

    #[test]
    fn the_shielding_is_one_less_z_b_over_z_eff_where_no_electron_is_trapped_and_less_elsewhere() {
        // `Z_b = 1` and `Z_eff = 2`
        let j_dot_b: Array1<f64> = j_dot_b_on_test_equilibrium(1.0);
        let j_fast_dot_b: f64 = J_PARALLEL_FAST * B0;
        let n_psi_norm: usize = j_dot_b.len();

        // On the magnetic axis, and on the boundary, where `q` is NaN so the collisionality is infinite, `L31 = 0`
        assert_relative_eq!(j_dot_b[0], 0.5 * j_fast_dot_b, max_relative = 1.0e-14);
        assert_relative_eq!(j_dot_b[n_psi_norm - 1], 0.5 * j_fast_dot_b, max_relative = 1.0e-14);
        // Elsewhere the trapped electrons carry no current, so they shield less: `0 < L31 < 1`
        let mut n_shielded_less: usize = 0;
        for i_psi_norm in 1..n_psi_norm - 1 {
            let shielding: f64 = j_dot_b[i_psi_norm] / j_fast_dot_b;
            assert!((0.5..1.0).contains(&shielding), "shielding = {shielding} at {i_psi_norm}");
            if shielding > 0.5 + 1.0e-3 {
                n_shielded_less += 1;
            }
        }
        assert!(n_shielded_less > n_psi_norm / 2);
    }

    #[test]
    fn beam_ions_of_the_plasma_s_effective_charge_are_shielded_to_l31() {
        // `Z_b = Z_eff = 2`, so `<J_NBCD . B> = L31 <J_fast . B>`, which vanishes on the magnetic axis
        let j_dot_b: Array1<f64> = j_dot_b_on_test_equilibrium(2.0);
        let j_dot_b_shielded_partly: Array1<f64> = j_dot_b_on_test_equilibrium(1.0);
        let j_fast_dot_b: f64 = J_PARALLEL_FAST * B0;
        let n_psi_norm: usize = j_dot_b.len();

        assert_eq!(j_dot_b[0], 0.0);
        for i_psi_norm in 1..n_psi_norm - 1 {
            // `1 - (1 / 2) (1 - L31) = (1 + L31) / 2`, so `L31` from `Z_b = 1` and from `Z_b = 2` agree
            let l31_from_z_b_2: f64 = j_dot_b[i_psi_norm] / j_fast_dot_b;
            let l31_from_z_b_1: f64 = 2.0 * j_dot_b_shielded_partly[i_psi_norm] / j_fast_dot_b - 1.0;
            assert_relative_eq!(l31_from_z_b_2, l31_from_z_b_1, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn the_beam_driven_current_on_the_magnetic_axis_is_r_j_dot_b_over_f() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let core_sources: CoreSourcesIds = core_sources_nbi_at(&[0.1], 1.0);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_beam_driven(&equilibrium, &kinetic_profiles_at(&[0.1]), &core_sources, &mut plasma_current_components).unwrap();

        let beam_driven: &PlasmaCurrentComponent = &plasma_current_components.time_slice[0].beam_driven;
        let f_axis: f64 = R_AXIS * B0 * (1.0 + F_PARAMAGNETIC);
        assert_relative_eq!(beam_driven.j_phi[0], R_AXIS * 0.5 * J_PARALLEL_FAST * B0 / f_axis, max_relative = 1.0e-14);
        // It flows with the fast ions' current
        let n_psi_norm: usize = beam_driven.current_phi_inside.len();
        assert!(beam_driven.current_phi_inside[n_psi_norm - 1] > 0.0);
    }

    #[test]
    fn a_time_slice_without_the_source_or_kinetic_profiles_is_nan() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2, 0.3]);
        // The source at the first two time-slices, the kinetic profiles at the first and last
        let core_sources: CoreSourcesIds = core_sources_nbi_at(&[0.1, 0.2], 1.0);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_beam_driven(&equilibrium, &kinetic_profiles_at(&[0.1, 0.3]), &core_sources, &mut plasma_current_components).unwrap();

        let n_psi_norm: usize = plasma_current_components.time_slice[0].beam_driven.j_phi.len();
        assert!(plasma_current_components.time_slice[0].beam_driven.j_phi[0].is_finite());
        for i_time in 1..3 {
            for i_psi_norm in 0..n_psi_norm {
                assert!(plasma_current_components.time_slice[i_time].beam_driven.j_phi[i_psi_norm].is_nan());
            }
        }
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
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_beam_driven(
            &equilibrium,
            &kinetic_profiles_at(&times),
            &core_sources_nbi_at(&times, 1.0),
            &mut plasma_current_components,
        )
        .unwrap();

        let n_psi_norm: usize = equilibrium.time_slice[0].profiles_1d.psi_norm.len();
        for i_psi_norm in 0..n_psi_norm {
            assert!(plasma_current_components.time_slice[0].beam_driven.j_phi[i_psi_norm].is_nan());
            assert!(plasma_current_components.time_slice[0].beam_driven.current_phi_inside[i_psi_norm].is_nan());
        }
        assert!(plasma_current_components.time_slice[1].beam_driven.current_phi_inside[n_psi_norm - 1] > 0.0);
    }

    #[test]
    fn sources_without_the_neutral_beams_or_their_charge_are_an_error() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        let error_no_source: String = current_beam_driven(
            &equilibrium,
            &kinetic_profiles_at(&[0.1]),
            &CoreSourcesIds::default(),
            &mut plasma_current_components,
        )
        .unwrap_err();
        assert!(error_no_source.contains("has no neutral beams' source"), "{error_no_source}");

        let error_no_charge: String = current_beam_driven(
            &equilibrium,
            &kinetic_profiles_at(&[0.1]),
            &core_sources_nbi_at(&[0.1], f64::NAN),
            &mut plasma_current_components,
        )
        .unwrap_err();
        assert!(error_no_charge.contains("the beam ions' charge"), "{error_no_charge}");
        assert!(plasma_current_components.time_slice[0].beam_driven.j_phi.is_empty());
    }
}
