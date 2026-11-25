//! The kinetic profiles of one `core_profiles/profiles_1d(itime)`, mapped onto the flux surfaces of an equilibrium time-slice, with their
//! derivatives with the poloidal flux.
//!
//! The kinetic profiles are on `grid/rho_pol_norm`, and the flux surfaces are at `psi_norm`. Both are mapped through the normalised
//! poloidal flux, `psi_norm = rho_pol_norm ** 2`, which, unlike `rho_pol_norm`, is linear in the poloidal flux, so needs no toroidal flux.
//! It is also the grid of the fast ions, e.g. from ASCOT5, which share `profiles_1d(itime)`'s grid. Values are interpolated linearly in
//! `psi_norm` onto the flux surfaces, and a surface outside the kinetic profiles' grid is NaN.

use super::settings::SETTINGS;
use crate::equilibrium_post_processor::interpolate_profile::interpolate_profile;
use imas_rs::EquilibriumProfiles1d;
use imas_rs::ids::core_profiles::{CoreProfileIons, CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d};
use ndarray::Array1;

const ELEMENTARY_CHARGE: f64 = physical_constants::ELEMENTARY_CHARGE;

/// Which of `core_profiles/profiles_1d(:)` holds the kinetic profiles at an equilibrium time-slice's time: the nearest within
/// `SETTINGS.time_tolerance`, as the kinetic profiles are not interpolated in time.
///
/// # Arguments
/// * `core_profiles` - the kinetic profiles
/// * `time` - the equilibrium time-slice's time [second]
///
/// # Returns
/// The index of the `core_profiles/profiles_1d(:)` at `time`, or `None` when there are no kinetic profiles at it; the currents which need
/// them are then NaN on that time-slice
pub(super) fn core_profiles_time_index(core_profiles: &CoreProfilesIds, time: f64) -> Option<usize> {
    let n_time_core_profiles: usize = core_profiles.profiles_1d.len();
    let mut times: Array1<f64> = Array1::from_elem(n_time_core_profiles, f64::NAN);
    for i_time_core_profiles in 0..n_time_core_profiles {
        times[i_time_core_profiles] = core_profiles.profiles_1d[i_time_core_profiles].time;
    }

    nearest_time_index(&times, time)
}

/// Which of `times` is at an equilibrium time-slice's time: the nearest within `SETTINGS.time_tolerance`, as profiles are not
/// interpolated in time.
///
/// # Arguments
/// * `times` - the times of the profiles [second]
/// * `time` - the equilibrium time-slice's time [second]
///
/// # Returns
/// The index of the time within `SETTINGS.time_tolerance` of `time`, the nearest if there are several, or `None` when there is none
pub(super) fn nearest_time_index(times: &Array1<f64>, time: f64) -> Option<usize> {
    let n_time: usize = times.len();
    let mut i_time_nearest: Option<usize> = None;
    let mut time_difference_nearest: f64 = f64::INFINITY;
    for i_time in 0..n_time {
        let time_difference: f64 = (times[i_time] - time).abs();
        if time_difference <= SETTINGS.time_tolerance && time_difference < time_difference_nearest {
            i_time_nearest = Some(i_time);
            time_difference_nearest = time_difference;
        }
    }

    i_time_nearest
}

/// The electron profiles, the effective charge and the parallel electric field, which the conductivity and the ohmic current need, on
/// the flux surfaces of an equilibrium time-slice, one value per surface.
pub(super) struct ElectronProfiles {
    /// `electrons/density_thermal` [1 / metre ** 3]
    pub electrons_density_thermal: Array1<f64>,
    /// `electrons/temperature` [electron_volt]
    pub electrons_temperature: Array1<f64>,
    /// `zeff` [dimensionless]
    pub zeff: Array1<f64>,
    /// `e_field/parallel`, `<E . B> / B0`, with `B0` the kinetic profiles' `core_profiles/vacuum_toroidal_field/b0` [volt / metre];
    /// `None` when the kinetic profiles do not have it
    pub e_field_parallel: Option<Array1<f64>>,
}

/// Map the electron profiles, the effective charge and the parallel electric field of `core_profiles_profiles_1d` onto the flux surfaces
/// of `equilibrium_profiles_1d`.
///
/// # Arguments
/// * `core_profiles_profiles_1d` - the kinetic profiles: `grid/rho_pol_norm`, `electrons/density_thermal` and `/temperature`, `zeff`, and
///   `e_field/parallel` if it is there
/// * `equilibrium_profiles_1d` - the flux surfaces: `psi_norm`
///
/// # Errors
/// The kinetic profiles are not on a grid of at least three increasing `rho_pol_norm`, a profile does not have a value at each of them, or
/// `zeff` is below 1
pub(super) fn electron_profiles(
    core_profiles_profiles_1d: &CoreProfilesProfiles1d,
    equilibrium_profiles_1d: &EquilibriumProfiles1d,
) -> Result<ElectronProfiles, String> {
    let psi_norm: Array1<f64> = grid_psi_norm(core_profiles_profiles_1d)?;
    let n_rho_pol_norm: usize = psi_norm.len();
    let electrons_density_thermal: &Array1<f64> = &core_profiles_profiles_1d.electrons.density_thermal;
    let electrons_temperature: &Array1<f64> = &core_profiles_profiles_1d.electrons.temperature;
    let zeff: &Array1<f64> = &core_profiles_profiles_1d.zeff;
    let e_field_parallel: &Array1<f64> = &core_profiles_profiles_1d.e_field.parallel;
    check_on_grid("electrons/density_thermal", electrons_density_thermal, n_rho_pol_norm)?;
    check_on_grid("electrons/temperature", electrons_temperature, n_rho_pol_norm)?;
    check_on_grid("zeff", zeff, n_rho_pol_norm)?;
    check_zeff(zeff)?;
    let with_e_field_parallel: bool = !e_field_parallel.is_empty();
    if with_e_field_parallel {
        check_on_grid("e_field/parallel", e_field_parallel, n_rho_pol_norm)?;
    }

    let surfaces_psi_norm: &Array1<f64> = &equilibrium_profiles_1d.psi_norm;
    Ok(ElectronProfiles {
        electrons_density_thermal: on_surfaces(surfaces_psi_norm, &psi_norm, electrons_density_thermal),
        electrons_temperature: on_surfaces(surfaces_psi_norm, &psi_norm, electrons_temperature),
        zeff: on_surfaces(surfaces_psi_norm, &psi_norm, zeff),
        e_field_parallel: if with_e_field_parallel {
            Some(on_surfaces(surfaces_psi_norm, &psi_norm, e_field_parallel))
        } else {
            None
        },
    })
}

/// The kinetic profiles the bootstrap current needs on the flux surfaces of an equilibrium time-slice,
/// `equilibrium/time_slice(itime)/profiles_1d`, one value per surface.
pub(super) struct KineticProfiles {
    /// `electrons/density_thermal` [1 / metre ** 3]
    pub electrons_density_thermal: Array1<f64>,
    /// `electrons/temperature` [electron_volt]
    pub electrons_temperature: Array1<f64>,
    /// `ion(:)/density_thermal`, summed over every ion species, impurities included [1 / metre ** 3]
    pub ions_density_thermal: Array1<f64>,
    /// The main ion's `ion(i_main)/density_thermal` [1 / metre ** 3]
    pub ion_main_density_thermal: Array1<f64>,
    /// The main ion's `ion(i_main)/temperature`, which is taken as every ion's [electron_volt]
    pub ion_main_temperature: Array1<f64>,
    /// The main ion's charge, `ion(i_main)/z_ion` [dimensionless]
    pub ion_main_charge: f64,
    /// `zeff` [dimensionless]
    pub zeff: Array1<f64>,
    /// The derivative with the poloidal flux of the thermal pressure, `e (n_e T_e + n_i T_i)`, every ion species at the main ion's
    /// temperature [pascal / weber]
    pub d_pressure_thermal_d_psi: Array1<f64>,
    /// The derivative with the poloidal flux of the electron temperature [electron_volt / weber]
    pub d_electrons_temperature_d_psi: Array1<f64>,
    /// The derivative with the poloidal flux of the main ion's temperature [electron_volt / weber]
    pub d_ion_main_temperature_d_psi: Array1<f64>,
}

/// Map the kinetic profiles of `core_profiles_profiles_1d` onto the flux surfaces of `equilibrium_profiles_1d`, with their derivatives
/// with the poloidal flux.
///
/// * the derivatives with `psi_norm` are taken on the kinetic profiles' own grid, so a profile is differentiated before it is
///   interpolated
/// * values and derivatives are interpolated linearly in `psi_norm` onto the flux surfaces
/// * `d(psi_norm) / d(psi)`, a constant, `1 / (psi_boundary - psi_axis)`, is taken on the flux surfaces, and turns the derivatives with
///   `psi_norm` into derivatives with `psi`
///
/// A derivative with `rho_pol_norm`, divided by `d(psi) / d(rho_pol_norm)`, would be `0 / 0` on the axis instead.
///
/// The derivatives are those of the quadratic through each point and its two neighbours, or through the first or last three points at
/// the two ends.
///
/// # Arguments
/// * `core_profiles_profiles_1d` - the kinetic profiles: `grid/rho_pol_norm`, `electrons/density_thermal` and `/temperature`, every
///   `ion(:)/density_thermal`, the main ion's `temperature` and `z_ion`, and `zeff`
/// * `equilibrium_profiles_1d` - the flux surfaces: `psi` and `psi_norm`
/// * `ion_main_index` - which of `ion(:)` is the main ion
///
/// # Errors
/// The kinetic profiles are not on a grid of at least three increasing `rho_pol_norm`, a profile does not have a value at each of them,
/// there is no main ion, its charge is not positive, `zeff` is below 1, or there are fewer than three flux surfaces
pub(super) fn kinetic_profiles(
    core_profiles_profiles_1d: &CoreProfilesProfiles1d,
    equilibrium_profiles_1d: &EquilibriumProfiles1d,
    ion_main_index: usize,
) -> Result<KineticProfiles, String> {
    let psi_norm: Array1<f64> = grid_psi_norm(core_profiles_profiles_1d)?;
    let n_rho_pol_norm: usize = psi_norm.len();

    // The profiles, each with one value at each point of the grid
    let electrons_density_thermal: &Array1<f64> = &core_profiles_profiles_1d.electrons.density_thermal;
    let electrons_temperature: &Array1<f64> = &core_profiles_profiles_1d.electrons.temperature;
    let zeff: &Array1<f64> = &core_profiles_profiles_1d.zeff;
    let ions: &Vec<CoreProfileIons> = &core_profiles_profiles_1d.ion;
    let n_ion: usize = ions.len();
    if ion_main_index >= n_ion {
        return Err(format!(
            "`core_profiles/profiles_1d(itime)/ion` has {n_ion} ion(s), but the main ion is `ion({ion_main_index})`"
        ));
    }
    let ion_main_temperature: &Array1<f64> = &ions[ion_main_index].temperature;
    let ion_main_charge: f64 = ions[ion_main_index].z_ion;
    if !ion_main_charge.is_finite() || ion_main_charge <= 0.0 {
        return Err(format!(
            "`core_profiles/profiles_1d(itime)/ion({ion_main_index})/z_ion` = {ion_main_charge}, the main ion's charge, is not positive"
        ));
    }
    check_on_grid("electrons/density_thermal", electrons_density_thermal, n_rho_pol_norm)?;
    check_on_grid("electrons/temperature", electrons_temperature, n_rho_pol_norm)?;
    check_on_grid("zeff", zeff, n_rho_pol_norm)?;
    check_on_grid(&format!("ion({ion_main_index})/temperature"), ion_main_temperature, n_rho_pol_norm)?;
    for i_ion in 0..n_ion {
        check_on_grid(&format!("ion({i_ion})/density_thermal"), &ions[i_ion].density_thermal, n_rho_pol_norm)?;
    }
    check_zeff(zeff)?;

    // On the kinetic profiles' grid: the total thermal ion density, the thermal pressure, and the derivatives with `psi_norm`
    let mut ions_density_thermal: Array1<f64> = Array1::zeros(n_rho_pol_norm);
    let mut pressure_thermal: Array1<f64> = Array1::from_elem(n_rho_pol_norm, f64::NAN);
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        for i_ion in 0..n_ion {
            ions_density_thermal[i_rho_pol_norm] += ions[i_ion].density_thermal[i_rho_pol_norm];
        }
        pressure_thermal[i_rho_pol_norm] = ELEMENTARY_CHARGE
            * (electrons_density_thermal[i_rho_pol_norm] * electrons_temperature[i_rho_pol_norm]
                + ions_density_thermal[i_rho_pol_norm] * ion_main_temperature[i_rho_pol_norm]);
    }
    let d_pressure_thermal_d_psi_norm: Array1<f64> = derivative(&psi_norm, &pressure_thermal);
    let d_electrons_temperature_d_psi_norm: Array1<f64> = derivative(&psi_norm, electrons_temperature);
    let d_ion_main_temperature_d_psi_norm: Array1<f64> = derivative(&psi_norm, ion_main_temperature);

    // The flux surfaces, and `d(psi_norm) / d(psi)` on them, which is constant, as `psi_norm` is linear in `psi`
    let surfaces_psi: &Array1<f64> = &equilibrium_profiles_1d.psi;
    let surfaces_psi_norm: &Array1<f64> = &equilibrium_profiles_1d.psi_norm;
    let n_surface: usize = surfaces_psi_norm.len();
    if n_surface < 3 {
        return Err(format!(
            "`profiles_1d/psi_norm` has {n_surface} flux surfaces, but the derivatives need at least 3"
        ));
    }
    if surfaces_psi.len() != n_surface {
        return Err(format!(
            "`profiles_1d/psi` has {} values, but `profiles_1d/psi_norm` has {n_surface}",
            surfaces_psi.len()
        ));
    }
    let d_psi_norm_d_psi: Array1<f64> = derivative(surfaces_psi, surfaces_psi_norm);

    // The profiles on the flux surfaces
    Ok(KineticProfiles {
        electrons_density_thermal: on_surfaces(surfaces_psi_norm, &psi_norm, electrons_density_thermal),
        electrons_temperature: on_surfaces(surfaces_psi_norm, &psi_norm, electrons_temperature),
        ions_density_thermal: on_surfaces(surfaces_psi_norm, &psi_norm, &ions_density_thermal),
        ion_main_density_thermal: on_surfaces(surfaces_psi_norm, &psi_norm, &ions[ion_main_index].density_thermal),
        ion_main_temperature: on_surfaces(surfaces_psi_norm, &psi_norm, ion_main_temperature),
        ion_main_charge,
        zeff: on_surfaces(surfaces_psi_norm, &psi_norm, zeff),
        d_pressure_thermal_d_psi: on_surfaces(surfaces_psi_norm, &psi_norm, &d_pressure_thermal_d_psi_norm) * &d_psi_norm_d_psi,
        d_electrons_temperature_d_psi: on_surfaces(surfaces_psi_norm, &psi_norm, &d_electrons_temperature_d_psi_norm) * &d_psi_norm_d_psi,
        d_ion_main_temperature_d_psi: on_surfaces(surfaces_psi_norm, &psi_norm, &d_ion_main_temperature_d_psi_norm) * &d_psi_norm_d_psi,
    })
}

/// The normalised poloidal flux, `psi_norm = rho_pol_norm ** 2`, of the kinetic profiles' grid, `grid/rho_pol_norm`.
///
/// # Errors
/// The grid has fewer than three points, or they are not finite, non-negative and increasing
pub(super) fn grid_psi_norm(core_profiles_profiles_1d: &CoreProfilesProfiles1d) -> Result<Array1<f64>, String> {
    psi_norm_of_rho_pol_norm(
        "`core_profiles/profiles_1d(itime)/grid/rho_pol_norm`",
        &core_profiles_profiles_1d.grid.rho_pol_norm,
    )
}

/// The normalised poloidal flux, `psi_norm = rho_pol_norm ** 2`, of a radial grid, `rho_pol_norm`, onto which profiles are mapped by
/// `on_surfaces`.
///
/// # Arguments
/// * `path` - the grid's path, for the errors, e.g. "`core_profiles/profiles_1d(itime)/grid/rho_pol_norm`"
/// * `rho_pol_norm` - the radial grid [dimensionless]
///
/// # Errors
/// The grid has fewer than three points, as the derivatives need, or they are not finite, non-negative and increasing
pub(super) fn psi_norm_of_rho_pol_norm(path: &str, rho_pol_norm: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n_rho_pol_norm: usize = rho_pol_norm.len();
    if n_rho_pol_norm < 3 {
        return Err(format!("{path} has {n_rho_pol_norm} points, but the derivatives need at least 3"));
    }
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        if !rho_pol_norm[i_rho_pol_norm].is_finite() || rho_pol_norm[i_rho_pol_norm] < 0.0 {
            return Err(format!(
                "{path}({i_rho_pol_norm}) = {} is not a finite, non-negative number",
                rho_pol_norm[i_rho_pol_norm]
            ));
        }
    }
    for i_rho_pol_norm in 1..n_rho_pol_norm {
        if rho_pol_norm[i_rho_pol_norm] <= rho_pol_norm[i_rho_pol_norm - 1] {
            return Err(format!("{path} does not increase"));
        }
    }

    let mut psi_norm: Array1<f64> = Array1::from_elem(n_rho_pol_norm, f64::NAN);
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        psi_norm[i_rho_pol_norm] = rho_pol_norm[i_rho_pol_norm].powi(2);
    }

    Ok(psi_norm)
}

/// Check that a kinetic profile has one value at each point of the grid.
///
/// # Errors
/// It does not
pub(super) fn check_on_grid(name: &str, values: &Array1<f64>, n_rho_pol_norm: usize) -> Result<(), String> {
    if values.len() != n_rho_pol_norm {
        return Err(format!(
            "`core_profiles/profiles_1d(itime)/{name}` has {} values, but `grid/rho_pol_norm` has {n_rho_pol_norm}",
            values.len()
        ));
    }

    Ok(())
}

/// Check that the effective charge is at least 1, as the neoclassical formulae need.
///
/// # Errors
/// It is below 1 somewhere
fn check_zeff(zeff: &Array1<f64>) -> Result<(), String> {
    let n_rho_pol_norm: usize = zeff.len();
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        if zeff[i_rho_pol_norm] < 1.0 {
            return Err(format!(
                "`core_profiles/profiles_1d(itime)/zeff({i_rho_pol_norm})` = {} is below 1",
                zeff[i_rho_pol_norm]
            ));
        }
    }

    Ok(())
}

/// A profile on the kinetic profiles' grid, `values` at `psi_norm`, interpolated linearly onto the flux surfaces at `surfaces_psi_norm`;
/// NaN outside the grid
pub(super) fn on_surfaces(surfaces_psi_norm: &Array1<f64>, psi_norm: &Array1<f64>, values: &Array1<f64>) -> Array1<f64> {
    let n_surface: usize = surfaces_psi_norm.len();
    let mut values_on_surfaces: Array1<f64> = Array1::from_elem(n_surface, f64::NAN);
    for i_surface in 0..n_surface {
        values_on_surfaces[i_surface] = interpolate_profile(surfaces_psi_norm[i_surface], psi_norm, values);
    }

    values_on_surfaces
}

/// The derivative `d(y) / d(x)` at each point of a grid `x` of at least three distinct points, from the quadratic through the point and
/// its two neighbours, or through the first or the last three points at the two ends. It is exact for a quadratic.
///
/// # Arguments
/// * `x` - the grid, increasing or decreasing
/// * `y` - the values, one at each point of `x`
///
/// # Returns
/// `d(y) / d(x)` at each point of `x`
fn derivative(x: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let n_x: usize = x.len();
    let mut d_y_d_x: Array1<f64> = Array1::from_elem(n_x, f64::NAN);
    for i_x in 0..n_x {
        // The first of the three points the quadratic goes through
        let i_x_first: usize = if i_x == 0 {
            0
        } else if i_x == n_x - 1 {
            n_x - 3
        } else {
            i_x - 1
        };
        let x_0: f64 = x[i_x_first];
        let x_1: f64 = x[i_x_first + 1];
        let x_2: f64 = x[i_x_first + 2];
        let x_here: f64 = x[i_x];
        d_y_d_x[i_x] = y[i_x_first] * (2.0 * x_here - x_1 - x_2) / ((x_0 - x_1) * (x_0 - x_2))
            + y[i_x_first + 1] * (2.0 * x_here - x_0 - x_2) / ((x_1 - x_0) * (x_1 - x_2))
            + y[i_x_first + 2] * (2.0 * x_here - x_0 - x_1) / ((x_2 - x_0) * (x_2 - x_1));
    }

    d_y_d_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn the_derivative_is_exact_for_a_quadratic_on_an_unequally_spaced_grid() {
        let x_increasing: Array1<f64> = array![0.0, 0.1, 0.35, 0.4, 0.8, 1.0];
        let x_decreasing: Array1<f64> = array![0.3, 0.25, 0.1, -0.2, -0.4];
        for x in [x_increasing, x_decreasing] {
            let y: Array1<f64> = x.mapv(|x_here: f64| 2.0 - 3.0 * x_here + 5.0 * x_here.powi(2));
            let d_y_d_x: Array1<f64> = derivative(&x, &y);
            let n_x: usize = x.len();
            for i_x in 0..n_x {
                assert_relative_eq!(d_y_d_x[i_x], -3.0 + 10.0 * x[i_x], epsilon = 1.0e-12);
            }
        }
    }

    /// Kinetic profiles quadratic in `psi_norm`, on flux surfaces on the kinetic profiles' grid, are mapped exactly
    #[test]
    fn profiles_quadratic_in_the_poloidal_flux_are_mapped_exactly() {
        let psi_axis: f64 = 0.3;
        let psi_boundary: f64 = 0.0;
        let psi_norm: Array1<f64> = array![0.0, 0.1, 0.3, 0.6, 1.0];
        let n_psi_norm: usize = psi_norm.len();

        let equilibrium_profiles_1d: EquilibriumProfiles1d = EquilibriumProfiles1d {
            psi: psi_norm.mapv(|psi_norm_here: f64| psi_axis + (psi_boundary - psi_axis) * psi_norm_here),
            psi_norm: psi_norm.clone(),
            ..Default::default()
        };

        let electrons_density: f64 = 4.0e19;
        let ion_densities: [f64; 2] = [3.0e19, 1.0e18];
        let mut core_profiles_profiles_1d: CoreProfilesProfiles1d = CoreProfilesProfiles1d::default();
        core_profiles_profiles_1d.grid.rho_pol_norm = psi_norm.mapv(f64::sqrt);
        core_profiles_profiles_1d.electrons.density_thermal = Array1::from_elem(n_psi_norm, electrons_density);
        core_profiles_profiles_1d.electrons.temperature = psi_norm.mapv(|psi_norm_here: f64| 1000.0 - 600.0 * psi_norm_here.powi(2));
        core_profiles_profiles_1d.zeff = Array1::from_elem(n_psi_norm, 1.5);
        let ion_charges: [f64; 2] = [1.0, 6.0];
        for i_ion in 0..2 {
            core_profiles_profiles_1d.ion.push(CoreProfileIons {
                z_ion: ion_charges[i_ion],
                density_thermal: Array1::from_elem(n_psi_norm, ion_densities[i_ion]),
                temperature: psi_norm.mapv(|psi_norm_here: f64| 2000.0 - 1500.0 * psi_norm_here.powi(2)),
                ..Default::default()
            });
        }

        let kinetic_profiles: KineticProfiles = kinetic_profiles(&core_profiles_profiles_1d, &equilibrium_profiles_1d, 0).unwrap();

        let ions_density: f64 = ion_densities[0] + ion_densities[1];
        let d_psi_norm_d_psi: f64 = 1.0 / (psi_boundary - psi_axis);
        assert_eq!(kinetic_profiles.ion_main_charge, 1.0);
        for i_psi_norm in 0..n_psi_norm {
            let psi_norm_here: f64 = psi_norm[i_psi_norm];
            let d_electrons_temperature_d_psi: f64 = -1200.0 * psi_norm_here * d_psi_norm_d_psi;
            let d_ion_main_temperature_d_psi: f64 = -3000.0 * psi_norm_here * d_psi_norm_d_psi;
            let d_pressure_thermal_d_psi: f64 =
                ELEMENTARY_CHARGE * (electrons_density * d_electrons_temperature_d_psi + ions_density * d_ion_main_temperature_d_psi);
            assert_relative_eq!(kinetic_profiles.ions_density_thermal[i_psi_norm], ions_density, max_relative = 1.0e-15);
            assert_relative_eq!(kinetic_profiles.ion_main_density_thermal[i_psi_norm], ion_densities[0], max_relative = 1.0e-15);
            assert_relative_eq!(
                kinetic_profiles.electrons_temperature[i_psi_norm],
                1000.0 - 600.0 * psi_norm_here.powi(2),
                max_relative = 1.0e-14
            );
            assert_relative_eq!(
                kinetic_profiles.d_electrons_temperature_d_psi[i_psi_norm],
                d_electrons_temperature_d_psi,
                epsilon = 1.0e-9,
                max_relative = 1.0e-12
            );
            assert_relative_eq!(
                kinetic_profiles.d_ion_main_temperature_d_psi[i_psi_norm],
                d_ion_main_temperature_d_psi,
                epsilon = 1.0e-9,
                max_relative = 1.0e-12
            );
            assert_relative_eq!(
                kinetic_profiles.d_pressure_thermal_d_psi[i_psi_norm],
                d_pressure_thermal_d_psi,
                epsilon = 1.0e-9,
                max_relative = 1.0e-12
            );
        }
    }

    #[test]
    fn kinetic_profiles_without_the_main_ion_are_an_error() {
        let mut core_profiles_profiles_1d: CoreProfilesProfiles1d = CoreProfilesProfiles1d::default();
        core_profiles_profiles_1d.grid.rho_pol_norm = array![0.0, 0.5, 1.0];
        let equilibrium_profiles_1d: EquilibriumProfiles1d = EquilibriumProfiles1d {
            psi: array![0.3, 0.15, 0.0],
            psi_norm: array![0.0, 0.5, 1.0],
            ..Default::default()
        };

        let error: String = kinetic_profiles(&core_profiles_profiles_1d, &equilibrium_profiles_1d, 0).err().unwrap();

        assert!(error.contains("the main ion is `ion(0)`"), "{error}");
    }
}
