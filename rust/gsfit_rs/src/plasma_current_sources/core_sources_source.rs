//! Writing a current source into the `core_sources` IDS, for the components of the plasma current which have a place there: those along
//! the field lines, whose `core_sources/source(i1)/profiles_1d/j_parallel = <J . B> / B0`.

use super::settings::SETTINGS;
use imas_rs::ids::core_sources::{CoreRadialGrid, CoreSources as CoreSourcesIds, CoreSourcesSource};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumProfiles1d, EquilibriumTimeSlice};
use ndarray::Array1;

/// The radial grid of a source, `core_sources/source(i1)/profiles_1d(itime)/grid`: the flux surfaces of the equilibrium time-slice's
/// `profiles_1d`.
///
/// # Arguments
/// * `time_slice` - the post-processed equilibrium time-slice
pub(super) fn core_sources_grid(time_slice: &EquilibriumTimeSlice) -> CoreRadialGrid {
    let equilibrium_profiles_1d: &EquilibriumProfiles1d = &time_slice.profiles_1d;
    let n_psi_norm: usize = equilibrium_profiles_1d.psi_norm.len();

    // `rho_pol_norm = sqrt(psi_norm)`, the data dictionary's definition
    let mut rho_pol_norm: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        rho_pol_norm[i_psi_norm] = equilibrium_profiles_1d.psi_norm[i_psi_norm].sqrt();
    }

    CoreRadialGrid {
        rho_tor_norm: equilibrium_profiles_1d.rho_tor_norm.clone(),
        rho_tor: equilibrium_profiles_1d.rho_tor.clone(),
        rho_pol_norm,
        psi: equilibrium_profiles_1d.psi.clone(),
        volume: equilibrium_profiles_1d.volume.clone(),
        area: equilibrium_profiles_1d.area.clone(),
        surface: equilibrium_profiles_1d.surface.clone(),
        psi_magnetic_axis: time_slice.global_quantities.psi_magnetic_axis,
        psi_boundary: time_slice.boundary.psi,
    }
}

/// Write `source` into `core_sources`, replacing the source with its `identifier/index`, or after the other sources, and set
/// `core_sources/vacuum_toroidal_field` to the equilibrium's, which `source` is normalised by.
///
/// Every source of `core_sources` is on its one time base and normalised by its one `vacuum_toroidal_field`. So the other sources already
/// in it must be at the equilibrium's times, which `source` is at, with the equilibrium's vacuum toroidal field.
///
/// # Arguments
/// * `core_sources` - the plasma's sources, which `source` is written into
/// * `equilibrium` - the Equilibrium IDS `source` was calculated from
/// * `source` - the source, one `profiles_1d` and `global_quantities` per equilibrium time-slice
///
/// # Errors
/// Another source is at other times, or `core_sources/vacuum_toroidal_field` is not the equilibrium's while there are other sources
pub(super) fn write_core_sources_source(core_sources: &mut CoreSourcesIds, equilibrium: &EquilibriumIds, source: CoreSourcesSource) -> Result<(), String> {
    let n_time: usize = equilibrium.time_slice.len();
    let n_source: usize = core_sources.source.len();
    let mut n_source_other: usize = 0;
    let mut i_source_same: Option<usize> = None;
    for i_source in 0..n_source {
        let source_other: &CoreSourcesSource = &core_sources.source[i_source];
        if source_other.identifier.index == source.identifier.index {
            i_source_same = Some(i_source);
            continue;
        }
        n_source_other += 1;
        let mut same_times: bool = source_other.profiles_1d.len() == n_time;
        if same_times {
            for i_time in 0..n_time {
                if (source_other.profiles_1d[i_time].time - equilibrium.time_slice[i_time].time).abs() > SETTINGS.time_tolerance {
                    same_times = false;
                }
            }
        }
        if !same_times {
            return Err(format!(
                "`core_sources/source({i_source})`, `{}`, is not at the equilibrium's times, which `{}` is written at",
                source_other.identifier.name, source.identifier.name
            ));
        }
    }
    if n_source_other > 0
        && (core_sources.vacuum_toroidal_field.r0 != equilibrium.vacuum_toroidal_field.r0
            || core_sources.vacuum_toroidal_field.b0 != equilibrium.vacuum_toroidal_field.b0)
    {
        return Err(format!(
            "`core_sources/vacuum_toroidal_field` is not the equilibrium's, which `{}` is normalised by, so the sources would be normalised differently",
            source.identifier.name
        ));
    }

    match i_source_same {
        Some(i_source) => core_sources.source[i_source] = source,
        None => core_sources.source.push(source),
    }
    core_sources.vacuum_toroidal_field.r0 = equilibrium.vacuum_toroidal_field.r0;
    core_sources.vacuum_toroidal_field.b0 = equilibrium.vacuum_toroidal_field.b0.clone();

    Ok(())
}
