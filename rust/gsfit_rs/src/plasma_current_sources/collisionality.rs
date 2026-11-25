//! The electron and ion collisionalities which the neoclassical formulae take as input, from Sauter, Angioni and Lin-Liu, Physics of
//! Plasmas 6, 2834 (1999), equations 18b to 18e. Redl et al., Physics of Plasmas 28, 022502 (2021), say that these definitions must be
//! used with their formulae.
//!
//! The numerical constants assume densities in [1 / metre ** 3], temperatures in [electron_volt] and lengths in [metre].
//!
//! The electron collisionality is also written to the kinetic profiles, as `core_profiles/profiles_1d(itime)/electrons/collisionality_norm`
//! (`electrons_collisionality_norm`); the data dictionary has no ion collisionality in `core_profiles`.

use super::kinetic_profiles::{check_on_grid, grid_psi_norm, nearest_time_index};
use crate::equilibrium_post_processor::interpolate_profile::interpolate_profile;
use imas_rs::ids::core_profiles::{CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumProfiles1d};
use ndarray::Array1;

/// The electron collisionality, `nu_e*`, the electron collision frequency over the bounce frequency: Sauter's equation 18b, with the
/// electron Coulomb logarithm of his equation 18d,
///
/// ```text
/// ln_lambda_e = 31.3 - ln(sqrt(n_e) / T_e)
/// nu_e* = 6.921e-18 |q| R n_e Z_eff ln_lambda_e / (T_e ** 2 epsilon ** 1.5)
/// ```
///
/// # Arguments
/// * `q` - the safety factor of the surface, whose sign is ignored [dimensionless]
/// * `r_major` - the major radius of the surface [metre]
/// * `epsilon` - the inverse aspect ratio of the surface [dimensionless]
/// * `electrons_density` - the thermal electron density [1 / metre ** 3]
/// * `electrons_temperature` - the electron temperature [electron_volt]
/// * `zeff` - the effective charge [dimensionless]
///
/// # Returns
/// `nu_e*` [dimensionless]. It is infinite on the magnetic axis, where `epsilon = 0`, and where the electron temperature is zero
pub(super) fn electrons_collisionality(q: f64, r_major: f64, epsilon: f64, electrons_density: f64, electrons_temperature: f64, zeff: f64) -> f64 {
    let coulomb_logarithm: f64 = electrons_coulomb_logarithm(electrons_density, electrons_temperature);

    6.921e-18 * q.abs() * r_major * electrons_density * zeff * coulomb_logarithm / (electrons_temperature.powi(2) * epsilon.powf(1.5))
}

/// The electron Coulomb logarithm, Sauter's equation 18d, `ln_lambda_e = 31.3 - ln(sqrt(n_e) / T_e)`, which the electron collisionality and
/// the Spitzer conductivity share.
///
/// # Arguments
/// * `electrons_density` - the thermal electron density [1 / metre ** 3]
/// * `electrons_temperature` - the electron temperature [electron_volt]
///
/// # Returns
/// `ln_lambda_e` [dimensionless]
pub(super) fn electrons_coulomb_logarithm(electrons_density: f64, electrons_temperature: f64) -> f64 {
    31.3 - (electrons_density.sqrt() / electrons_temperature).ln()
}

/// The ion collisionality, `nu_i*`: Sauter's equation 18c, with the main ion's charge, `Z_i`, rather than `Z_eff`, and the total thermal
/// ion density, which is how NEO, the code Redl's formulae were fitted to, evaluates it. The Coulomb logarithm is the main ion's, Sauter's
/// equation 18e,
///
/// ```text
/// ln_lambda_ii = 30 - ln(Z_i ** 3 sqrt(n_i_main) / T_i ** 1.5)
/// nu_i* = 4.90e-18 |q| R n_i Z_i ** 4 ln_lambda_ii / (T_i ** 2 epsilon ** 1.5)
/// ```
///
/// # Arguments
/// * `q` - the safety factor of the surface, whose sign is ignored [dimensionless]
/// * `r_major` - the major radius of the surface [metre]
/// * `epsilon` - the inverse aspect ratio of the surface [dimensionless]
/// * `ions_density` - the thermal density of every ion species together, `n_i` [1 / metre ** 3]
/// * `ion_main_density` - the thermal density of the main ion, `n_i_main` [1 / metre ** 3]
/// * `ion_main_temperature` - the main ion's temperature, `T_i` [electron_volt]
/// * `ion_main_charge` - the main ion's charge, `Z_i` [dimensionless]
///
/// # Returns
/// `nu_i*` [dimensionless]. It is infinite on the magnetic axis, where `epsilon = 0`, and where the ion temperature is zero
pub(super) fn ions_collisionality(
    q: f64,
    r_major: f64,
    epsilon: f64,
    ions_density: f64,
    ion_main_density: f64,
    ion_main_temperature: f64,
    ion_main_charge: f64,
) -> f64 {
    let coulomb_logarithm: f64 = 30.0 - (ion_main_charge.powi(3) * ion_main_density.sqrt() / ion_main_temperature.powf(1.5)).ln();

    4.90e-18 * q.abs() * r_major * ions_density * ion_main_charge.powi(4) * coulomb_logarithm / (ion_main_temperature.powi(2) * epsilon.powf(1.5))
}

/// Write the electron collisionality, `nu_e*` (`electrons_collisionality`), into the kinetic profiles, as
/// `core_profiles/profiles_1d(itime)/electrons/collisionality_norm`, at each point of their grid, `grid/rho_pol_norm`, for every
/// `profiles_1d(itime)` at the time of an equilibrium time-slice which the solver could use.
///
/// It is the electron collisionality of the bootstrap current and of the neoclassical conductivity, evaluated at the kinetic profiles'
/// points rather than on the equilibrium's flux surfaces: the electron density and temperature and `zeff` are the kinetic profiles' own,
/// and the flux surfaces' safety factor, `q`, major radius, `(r_outboard + r_inboard) / 2`, and inverse aspect ratio,
/// `(r_outboard - r_inboard) / (r_outboard + r_inboard)`, come from the equilibrium's `profiles_1d`. `q` is interpolated linearly in
/// `psi_norm = rho_pol_norm ** 2`, in which it is smooth on the magnetic axis, and `r_inboard` and `r_outboard` linearly in
/// `rho_pol_norm`, in which they are close to linear there: the minor radius grows as `sqrt(psi_norm)`, so interpolated in `psi_norm` it
/// would be too small inside the innermost surface, and `nu_e*`, which goes as `epsilon ** -1.5`, too large. `nu_e*` is infinite on the
/// magnetic axis, where the inverse aspect ratio is zero, and NaN where `q` is NaN, as on the boundary, where the post-processor leaves it
/// NaN.
///
/// A `profiles_1d(itime)` without a usable equilibrium time-slice at its time, or with a negative `code/output_flag`, is not changed.
///
/// # Arguments
/// * `equilibrium` - the post-processed equilibrium: on each time-slice, `profiles_1d` with `psi_norm`, `q`, `r_inboard` and `r_outboard`
/// * `core_profiles` - the kinetic profiles: on each `profiles_1d(itime)`, `grid/rho_pol_norm`, `electrons/density_thermal` and
///   `/temperature`, and `zeff`
///
/// # Errors
/// A `profiles_1d(itime)` at a usable equilibrium time-slice's time is not on a grid of at least three increasing `rho_pol_norm`, or one of
/// its profiles does not have a value at each point; or that time-slice's `profiles_1d` is not filled. `core_profiles` is not changed then.
pub(super) fn electrons_collisionality_norm(equilibrium: &EquilibriumIds, core_profiles: &mut CoreProfilesIds) -> Result<(), String> {
    let n_time: usize = equilibrium.time_slice.len();
    let mut times: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
    for i_time in 0..n_time {
        times[i_time] = equilibrium.time_slice[i_time].time;
    }

    // Every `collisionality_norm` is calculated before any is written, so that none is written when one cannot be calculated
    let n_time_core_profiles: usize = core_profiles.profiles_1d.len();
    let mut collisionalities: Vec<Option<Array1<f64>>> = Vec::with_capacity(n_time_core_profiles);
    for i_time_core_profiles in 0..n_time_core_profiles {
        let core_profiles_profiles_1d: &CoreProfilesProfiles1d = &core_profiles.profiles_1d[i_time_core_profiles];
        let mut collisionality: Option<Array1<f64>> = None;
        // A negative `code/output_flag` means the time-slice shall not be used
        if let Some(i_time) = nearest_time_index(&times, core_profiles_profiles_1d.time)
            && (equilibrium.code.output_flag.len() != n_time || equilibrium.code.output_flag[i_time] >= 0)
        {
            let collisionality_here: Array1<f64> = electrons_collisionality_on_grid(&equilibrium.time_slice[i_time].profiles_1d, core_profiles_profiles_1d)
                .map_err(|reason: String| {
                    format!("electron collisionality of `core_profiles/profiles_1d({i_time_core_profiles})`, at `equilibrium/time_slice({i_time})`: {reason}")
                })?;
            collisionality = Some(collisionality_here);
        }
        collisionalities.push(collisionality);
    }

    for i_time_core_profiles in 0..n_time_core_profiles {
        if let Some(collisionality) = collisionalities[i_time_core_profiles].take() {
            core_profiles.profiles_1d[i_time_core_profiles].electrons.collisionality_norm = collisionality;
        }
    }

    Ok(())
}

/// The electron collisionality, `nu_e*`, at each point of the kinetic profiles' grid; see `electrons_collisionality_norm`.
///
/// # Arguments
/// * `equilibrium_profiles_1d` - the flux surfaces: `psi_norm`, `q`, `r_inboard` and `r_outboard`
/// * `core_profiles_profiles_1d` - the kinetic profiles: `grid/rho_pol_norm`, `electrons/density_thermal` and `/temperature`, and `zeff`
///
/// # Returns
/// `nu_e*` at each point of `grid/rho_pol_norm` [dimensionless]
///
/// # Errors
/// The flux surfaces are fewer than two, or a profile of theirs does not have a value at each; the kinetic profiles' grid is not usable, or
/// one of their profiles does not have a value at each point
fn electrons_collisionality_on_grid(
    equilibrium_profiles_1d: &EquilibriumProfiles1d,
    core_profiles_profiles_1d: &CoreProfilesProfiles1d,
) -> Result<Array1<f64>, String> {
    let surfaces_psi_norm: &Array1<f64> = &equilibrium_profiles_1d.psi_norm;
    let n_psi_norm: usize = surfaces_psi_norm.len();
    if n_psi_norm < 2 {
        return Err(format!(
            "`profiles_1d/psi_norm` has {n_psi_norm} values, but the interpolation needs at least 2"
        ));
    }
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

    let surfaces_rho_pol_norm: Array1<f64> = surfaces_psi_norm.mapv(f64::sqrt);

    // `grid_psi_norm` checks the kinetic profiles' grid too
    let psi_norm: Array1<f64> = grid_psi_norm(core_profiles_profiles_1d)?;
    let rho_pol_norm: &Array1<f64> = &core_profiles_profiles_1d.grid.rho_pol_norm;
    let n_rho_pol_norm: usize = psi_norm.len();
    let electrons_density: &Array1<f64> = &core_profiles_profiles_1d.electrons.density_thermal;
    let electrons_temperature: &Array1<f64> = &core_profiles_profiles_1d.electrons.temperature;
    let zeff: &Array1<f64> = &core_profiles_profiles_1d.zeff;
    check_on_grid("electrons/density_thermal", electrons_density, n_rho_pol_norm)?;
    check_on_grid("electrons/temperature", electrons_temperature, n_rho_pol_norm)?;
    check_on_grid("zeff", zeff, n_rho_pol_norm)?;

    let mut collisionality: Array1<f64> = Array1::from_elem(n_rho_pol_norm, f64::NAN);
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        let q: f64 = interpolate_profile(psi_norm[i_rho_pol_norm], surfaces_psi_norm, &equilibrium_profiles_1d.q);
        let r_inboard: f64 = interpolate_profile(rho_pol_norm[i_rho_pol_norm], &surfaces_rho_pol_norm, &equilibrium_profiles_1d.r_inboard);
        let r_outboard: f64 = interpolate_profile(rho_pol_norm[i_rho_pol_norm], &surfaces_rho_pol_norm, &equilibrium_profiles_1d.r_outboard);
        let epsilon: f64 = (r_outboard - r_inboard) / (r_outboard + r_inboard);
        let r_major: f64 = 0.5 * (r_outboard + r_inboard);
        collisionality[i_rho_pol_norm] = electrons_collisionality(
            q,
            r_major,
            epsilon,
            electrons_density[i_rho_pol_norm],
            electrons_temperature[i_rho_pol_norm],
            zeff[i_rho_pol_norm],
        );
    }

    Ok(collisionality)
}

#[cfg(test)]
mod tests {
    use super::super::test_equilibrium::{MINOR_RADIUS, R_AXIS, circular_equilibrium, kinetic_profiles_at};
    use super::*;
    use approx::assert_relative_eq;

    /// The collisionalities of TORAX's `_calculate_nu_e_star` and `calculate_nu_i_star` (commit b53645a), evaluated in float64. TORAX
    /// adds 1e-7 to `epsilon` in both, so it is added here too.
    #[test]
    fn collisionalities_match_torax() {
        // n_e, T_e, n_i, n_i_main, T_i, Z_i, Z_eff, q, R, epsilon, then TORAX's nu_e* and nu_i*
        let cases: [[f64; 12]; 6] = [
            [
                1e+19,
                1500.0,
                8e+18,
                7e+18,
                3000.0,
                1.0,
                2.0,
                1.2,
                0.45,
                0.1,
                0.01758450611685286,
                0.0015108376513338461,
            ],
            [
                3e+19,
                300.0,
                2.5e+19,
                2.2e+19,
                600.0,
                1.0,
                1.5,
                2.0,
                0.5,
                0.3,
                0.3070519319007927,
                0.03588104013803106,
            ],
            [
                5e+18,
                50.0,
                4e+18,
                3.5e+18,
                40.0,
                1.0,
                3.0,
                5.0,
                0.55,
                0.5,
                4.419894823477759,
                1.3514572767566642,
            ],
            [
                2e+20,
                5000.0,
                1.7e+20,
                1.5e+20,
                5000.0,
                2.0,
                1.2,
                1.0,
                6.2,
                0.2,
                0.07573806776888928,
                0.6455184807208533,
            ],
            [1e+19, 20.0, 9e+18, 8e+18, 15.0, 1.0, 1.0, 10.0, 0.4, 0.05, 768.909420091854, 862.4475378775318],
            [
                4e+19,
                800.0,
                3e+19,
                2.6e+19,
                1600.0,
                1.0,
                2.5,
                3.0,
                0.48,
                0.4,
                0.09489830305014287,
                0.006116794099673297,
            ],
        ];
        let torax_epsilon_offset: f64 = 1.0e-7;

        let n_case: usize = cases.len();
        for i_case in 0..n_case {
            let [
                electrons_density,
                electrons_temperature,
                ions_density,
                ion_main_density,
                ion_main_temperature,
                ion_main_charge,
                zeff,
                q,
                r_major,
                epsilon,
                electrons_collisionality_torax,
                ions_collisionality_torax,
            ]: [f64; 12] = cases[i_case];

            let electrons_collisionality_here: f64 =
                electrons_collisionality(q, r_major, epsilon + torax_epsilon_offset, electrons_density, electrons_temperature, zeff);
            let ions_collisionality_here: f64 = ions_collisionality(
                q,
                r_major,
                epsilon + torax_epsilon_offset,
                ions_density,
                ion_main_density,
                ion_main_temperature,
                ion_main_charge,
            );

            assert_relative_eq!(electrons_collisionality_here, electrons_collisionality_torax, max_relative = 1.0e-12);
            assert_relative_eq!(ions_collisionality_here, ions_collisionality_torax, max_relative = 1.0e-12);
        }
    }

    #[test]
    fn collisionalities_are_infinite_on_the_magnetic_axis() {
        assert!(electrons_collisionality(1.2, 0.45, 0.0, 1.0e19, 1500.0, 2.0).is_infinite());
        assert!(ions_collisionality(1.2, 0.45, 0.0, 8.0e18, 7.0e18, 3000.0, 1.0).is_infinite());
    }

    #[test]
    fn the_electron_collisionality_is_written_to_the_kinetic_profiles_at_the_equilibrium_s_times() {
        // The kinetic profiles at 0.1 s, at the equilibrium's time-slice, and at 0.3 s, where the equilibrium has none
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let mut core_profiles: CoreProfilesIds = kinetic_profiles_at(&[0.1, 0.3]);

        electrons_collisionality_norm(&equilibrium, &mut core_profiles).unwrap();

        let collisionality: &Array1<f64> = &core_profiles.profiles_1d[0].electrons.collisionality_norm;
        let n_rho_pol_norm: usize = core_profiles.profiles_1d[0].grid.rho_pol_norm.len();
        assert_eq!(collisionality.len(), n_rho_pol_norm);
        // Infinite on the magnetic axis, and NaN on the boundary, where the equilibrium's `q` is NaN
        assert!(collisionality[0].is_infinite());
        assert!(collisionality[n_rho_pol_norm - 1].is_nan());
        // At rho_pol_norm = 0.5, psi_norm = 0.25 is one of the test equilibrium's surfaces, where q = 1 + 2 psi_norm = 1.5 and the
        // surface's inboard and outboard radii are R_AXIS -/+ MINOR_RADIUS sqrt(psi_norm)
        let i_rho_pol_norm_half: usize = (n_rho_pol_norm - 1) / 2;
        let profiles_1d: &CoreProfilesProfiles1d = &core_profiles.profiles_1d[0];
        let epsilon: f64 = 0.5 * MINOR_RADIUS / R_AXIS;
        let collisionality_expected: f64 = electrons_collisionality(
            1.5,
            R_AXIS,
            epsilon,
            profiles_1d.electrons.density_thermal[i_rho_pol_norm_half],
            profiles_1d.electrons.temperature[i_rho_pol_norm_half],
            profiles_1d.zeff[i_rho_pol_norm_half],
        );
        assert_relative_eq!(collisionality[i_rho_pol_norm_half], collisionality_expected, max_relative = 1.0e-12);
        // At rho_pol_norm = 0.05, psi_norm = 0.0025 is inside the innermost interval of surfaces, [0, 0.05], where `q`, linear in
        // `psi_norm`, and the radii, linear in `rho_pol_norm`, are interpolated exactly
        let rho_pol_norm_inner: f64 = profiles_1d.grid.rho_pol_norm[1];
        let collisionality_inner_expected: f64 = electrons_collisionality(
            1.0 + 2.0 * rho_pol_norm_inner.powi(2),
            R_AXIS,
            MINOR_RADIUS * rho_pol_norm_inner / R_AXIS,
            profiles_1d.electrons.density_thermal[1],
            profiles_1d.electrons.temperature[1],
            profiles_1d.zeff[1],
        );
        assert_relative_eq!(collisionality[1], collisionality_inner_expected, max_relative = 1.0e-12);
        // Not written without an equilibrium time-slice at the kinetic profiles' time
        assert!(core_profiles.profiles_1d[1].electrons.collisionality_norm.is_empty());
    }

    #[test]
    fn nothing_is_written_when_a_kinetic_profile_is_not_on_its_grid() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        let mut core_profiles: CoreProfilesIds = kinetic_profiles_at(&[0.1, 0.2]);
        core_profiles.profiles_1d[1].zeff = Array1::from_elem(3, 2.0);

        let error: String = electrons_collisionality_norm(&equilibrium, &mut core_profiles).unwrap_err();

        assert!(error.contains("zeff` has 3 values"), "{error}");
        assert!(core_profiles.profiles_1d[0].electrons.collisionality_norm.is_empty());
    }
}
