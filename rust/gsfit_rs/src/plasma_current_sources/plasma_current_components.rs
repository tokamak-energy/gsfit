//! The toroidal plasma current, split into its components: beam-driven, bootstrap, diamagnetic, ohmic and Pfirsch-Schlüter.
//!
//! The data dictionary has nowhere for this split. `core_sources/source(i1)/profiles_1d` holds a source's flux-surface averaged parallel
//! current, `j_parallel = <J . B> / B0`, which is all there is to the bootstrap, ohmic and beam-driven currents. But the diamagnetic and
//! Pfirsch-Schlüter currents carry no `<J . B>`: the one is perpendicular to `B`, and the other averages to zero over each surface. And
//! nothing holds the toroidal current of each component, which is what adds up to the plasma current.

use super::flux_surface_samples::{FluxSurfaceSamples, flux_surface_samples};
use super::settings::SETTINGS;
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumTimeSlice};
use ndarray::Array1;

/// The toroidal plasma current, split into its components, on the flux surfaces of each equilibrium time-slice.
///
/// Made with `new` from the equilibrium, holding its time-slices' flux surfaces; each component is then filled by its own function,
/// e.g. `current_bootstrap`. A component not yet calculated is empty.
#[derive(Clone, Debug, Default)]
pub struct PlasmaCurrentComponents {
    /// One per equilibrium time-slice
    pub time_slice: Vec<PlasmaCurrentComponentsTimeSlice>,
}

/// The components of the toroidal plasma current on one equilibrium time-slice.
#[derive(Clone, Debug, Default)]
pub struct PlasmaCurrentComponentsTimeSlice {
    /// The equilibrium time-slice's time [second]
    pub time: f64,
    /// The flux surfaces, the equilibrium time-slice's `profiles_1d`
    pub grid: PlasmaCurrentComponentsGrid,
    /// The beam-driven current, the fast ions' current shielded by the electrons, `current_beam_driven`
    pub beam_driven: PlasmaCurrentComponent,
    /// The bootstrap current, `current_bootstrap`
    pub bootstrap: PlasmaCurrentComponent,
    /// The diamagnetic current, `current_diamagnetic`
    pub diamagnetic: PlasmaCurrentComponent,
    /// The ohmic current, `current_ohmic`
    pub ohmic: PlasmaCurrentComponent,
    /// The Pfirsch-Schlüter current, `current_pfirsch_schluter`
    pub pfirsch_schluter: PlasmaCurrentComponent,
}

/// The flux surfaces the components are on, copied from the equilibrium time-slice's `profiles_1d`.
#[derive(Clone, Debug, Default)]
pub struct PlasmaCurrentComponentsGrid {
    /// `psi` [weber]
    pub psi: Array1<f64>,
    /// `psi_norm` [dimensionless]
    pub psi_norm: Array1<f64>,
    /// `rho_tor_norm` [dimensionless]
    pub rho_tor_norm: Array1<f64>,
    /// `area`, the poloidal cross-section inside each surface [metre ** 2]
    pub area: Array1<f64>,
}

/// One component of the toroidal plasma current.
#[derive(Clone, Debug, Default)]
pub struct PlasmaCurrentComponent {
    /// The toroidal current density on each surface, `<j_phi / R> / <1 / R>`, as the data dictionary defines `core_profiles/profiles_1d/j_phi`,
    /// so that its integral over the cross-section is the toroidal current [ampere / metre ** 2]
    pub j_phi: Array1<f64>,
    /// The toroidal current inside each surface, the integral of `j_phi` over the cross-section, by the trapezium rule in `grid/area`
    /// [ampere]
    pub current_phi_inside: Array1<f64>,
    /// The flux-surface averaged parallel current on each surface, `<J . B> / B0`, with `B0` the equilibrium's `vacuum_toroidal_field/b0`,
    /// as the data dictionary defines `core_profiles/profiles_1d/j_total`: zero for the diamagnetic and Pfirsch-Schlüter currents, whose
    /// `<J . B>` is zero [ampere / metre ** 2]
    pub j_parallel: Array1<f64>,
}

impl PlasmaCurrentComponents {
    /// The flux surfaces of each time-slice of `equilibrium`, with every component empty
    ///
    /// # Arguments
    /// * `equilibrium` - the post-processed Equilibrium IDS
    pub fn new(equilibrium: &EquilibriumIds) -> Self {
        let n_time: usize = equilibrium.time_slice.len();
        let mut time_slice: Vec<PlasmaCurrentComponentsTimeSlice> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            let profiles_1d = &equilibrium.time_slice[i_time].profiles_1d;
            time_slice.push(PlasmaCurrentComponentsTimeSlice {
                time: equilibrium.time_slice[i_time].time,
                grid: PlasmaCurrentComponentsGrid {
                    psi: profiles_1d.psi.clone(),
                    psi_norm: profiles_1d.psi_norm.clone(),
                    rho_tor_norm: profiles_1d.rho_tor_norm.clone(),
                    area: profiles_1d.area.clone(),
                },
                ..Default::default()
            });
        }

        PlasmaCurrentComponents { time_slice }
    }

    /// Check that these components were made from `equilibrium`, by `new`: the same time-slices, with the same flux surfaces.
    ///
    /// # Errors
    /// A different number of time-slices, a time-slice at another time, or another number of flux surfaces
    pub(super) fn check_made_from(&self, equilibrium: &EquilibriumIds) -> Result<(), String> {
        let n_time: usize = equilibrium.time_slice.len();
        if self.time_slice.len() != n_time {
            return Err(format!(
                "the plasma current components have {} time-slices, but `equilibrium/time_slice` has {n_time}: make them with `PlasmaCurrentComponents::new` from this equilibrium",
                self.time_slice.len()
            ));
        }
        for i_time in 0..n_time {
            let time_slice: &PlasmaCurrentComponentsTimeSlice = &self.time_slice[i_time];
            if (time_slice.time - equilibrium.time_slice[i_time].time).abs() > SETTINGS.time_tolerance
                || time_slice.grid.psi_norm.len() != equilibrium.time_slice[i_time].profiles_1d.psi_norm.len()
            {
                return Err(format!(
                    "the plasma current components' time-slice {i_time} is not `equilibrium/time_slice({i_time})`: make them with `PlasmaCurrentComponents::new` from this equilibrium"
                ));
            }
        }

        Ok(())
    }
}

impl PlasmaCurrentComponent {
    /// A component from its toroidal current density and its flux-surface averaged parallel current, with the toroidal current inside each
    /// surface integrated over `area`
    ///
    /// # Arguments
    /// * `j_phi` - the toroidal current density on each surface [ampere / metre ** 2]
    /// * `j_parallel` - `<J . B> / B0` on each surface, with `B0` the equilibrium's `vacuum_toroidal_field/b0` [ampere / metre ** 2]
    /// * `area` - the poloidal cross-section inside each surface, from zero on the magnetic axis [metre ** 2]
    pub(super) fn new(j_phi: Array1<f64>, j_parallel: Array1<f64>, area: &Array1<f64>) -> Self {
        let current_phi_inside: Array1<f64> = integral_inside(&j_phi, area);

        PlasmaCurrentComponent {
            j_phi,
            current_phi_inside,
            j_parallel,
        }
    }
}

/// The integral of a current density over the cross-section inside each flux surface, by the trapezium rule in `area`.
///
/// # Arguments
/// * `current_density` - the current density on each surface [ampere / metre ** 2]
/// * `area` - the poloidal cross-section inside each surface, from zero on the magnetic axis [metre ** 2]
///
/// # Returns
/// The current inside each surface [ampere]: zero on the magnetic axis, unless the current density is NaN there, and NaN from the
/// first surface with a NaN current density outwards
pub(super) fn integral_inside(current_density: &Array1<f64>, area: &Array1<f64>) -> Array1<f64> {
    let n_psi_norm: usize = current_density.len();
    let mut current_inside: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    if n_psi_norm > 0 && !current_density[0].is_nan() {
        current_inside[0] = 0.0;
    }
    for i_psi_norm in 1..n_psi_norm {
        current_inside[i_psi_norm] =
            current_inside[i_psi_norm - 1] + 0.5 * (current_density[i_psi_norm] + current_density[i_psi_norm - 1]) * (area[i_psi_norm] - area[i_psi_norm - 1]);
    }

    current_inside
}

/// The toroidal current density, `<j_phi / R> / <1 / R>`, of a current along the field lines, from its flux-surface averaged parallel
/// current, for the bootstrap, ohmic and beam-driven currents.
///
/// A flux function times `B` has `<j_phi / R> = F <R ** -2> <J . B> / <B ** 2>`. On the magnetic axis, where `<R ** -2> = 1 / R ** 2`,
/// `<B ** 2> = F ** 2 / R ** 2` and `<1 / R> = 1 / R`, that is `R <J . B> / F`.
///
/// # Arguments
/// * `j_dot_b` - `<J . B>` on each surface [ampere * tesla / metre ** 2]
/// * `f` - `F = R B_phi` on each surface [tesla * metre]
/// * `flux_surface_samples` - the field sampled on each surface, `flux_surface_samples`, with the magnetic axis first
/// * `magnetic_axis_r` - the major radius of the magnetic axis [metre]
///
/// # Returns
/// `<j_phi / R> / <1 / R>` on each surface [ampere / metre ** 2]; NaN on a surface which could not be sampled
pub(super) fn j_phi_of_parallel_current(
    j_dot_b: &Array1<f64>,
    f: &Array1<f64>,
    flux_surface_samples: &[Option<FluxSurfaceSamples>],
    magnetic_axis_r: f64,
) -> Array1<f64> {
    let n_psi_norm: usize = j_dot_b.len();
    let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        if i_psi_norm == 0 {
            j_phi[i_psi_norm] = magnetic_axis_r * j_dot_b[i_psi_norm] / f[i_psi_norm];
        } else if let Some(samples) = &flux_surface_samples[i_psi_norm] {
            let r_inverse_squared_average: f64 = samples.average(|r_here: f64, _b_field, _b_field_p| r_here.powi(-2));
            let b_field_squared_average: f64 = samples.average(|_r, b_field: f64, _b_field_p| b_field.powi(2));
            let r_inverse_average: f64 = samples.average(|r_here: f64, _b_field, _b_field_p| 1.0 / r_here);
            j_phi[i_psi_norm] = f[i_psi_norm] * r_inverse_squared_average * j_dot_b[i_psi_norm] / (b_field_squared_average * r_inverse_average);
        }
    }

    j_phi
}

/// The toroidal current density of a current the pressure drives, on every time-slice of the equilibrium, for the diamagnetic and
/// Pfirsch-Schlüter currents.
///
/// Both are `p'` times a coefficient which depends only on `F` and the shape of the flux surfaces, so on each surface the current density
/// is `j_phi_of(p', F, samples)`, with `p'` the equilibrium's `profiles_1d/dpressure_dpsi`, the total pressure the Grad-Shafranov equation
/// was solved with, fast ions included, and `samples` the field sampled on the surface (`flux_surface_samples.rs`). Both vanish on the
/// magnetic axis. A time-slice the solver could not use, with a negative `code/output_flag`, is NaN.
///
/// # Arguments
/// * `equilibrium` - the Equilibrium IDS, post-processed: on each time-slice `profiles_1d` with `psi_norm`, `f`, `dpressure_dpsi` and
///   `area`, and `profiles_2d(0)`, the magnetic axis and the boundary, for the flux surfaces
/// * `name` - the current's name, for the errors
/// * `j_phi_of` - the toroidal current density on a surface, `<j_phi / R> / <1 / R>`, of `p'` [pascal / weber], `F` [tesla * metre] and the
///   field sampled on the surface [ampere / metre ** 2]
///
/// # Returns
/// The current on each time-slice
///
/// # Errors
/// The equilibrium's `profiles_1d` is not filled
pub(super) fn pressure_driven_components<F>(equilibrium: &EquilibriumIds, name: &str, j_phi_of: F) -> Result<Vec<PlasmaCurrentComponent>, String>
where
    F: Fn(f64, f64, &FluxSurfaceSamples) -> f64,
{
    let n_time: usize = equilibrium.time_slice.len();
    let mut components: Vec<PlasmaCurrentComponent> = Vec::with_capacity(n_time);
    for i_time in 0..n_time {
        let time_slice: &EquilibriumTimeSlice = &equilibrium.time_slice[i_time];
        let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();

        // A negative `code/output_flag` means the time-slice shall not be used. The solver did not converge on it, so its `profiles_1d` was
        // not post-processed, and only `psi_norm` is filled: its other profiles are checked, and its `area` used, only when it is usable
        let usable: bool = equilibrium.code.output_flag.len() != n_time || equilibrium.code.output_flag[i_time] >= 0;
        if usable {
            let named_profiles: [(&str, &Array1<f64>); 3] = [
                ("f", &time_slice.profiles_1d.f),
                ("dpressure_dpsi", &time_slice.profiles_1d.dpressure_dpsi),
                ("area", &time_slice.profiles_1d.area),
            ];
            for (profile_name, values) in named_profiles {
                if values.len() != n_psi_norm {
                    return Err(format!(
                        "{name} current at `equilibrium/time_slice({i_time})`: `profiles_1d/{profile_name}` has {} values, but `profiles_1d/psi_norm` has {n_psi_norm}",
                        values.len()
                    ));
                }
            }
        }
        let area_not_usable: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let area: &Array1<f64> = if usable { &time_slice.profiles_1d.area } else { &area_not_usable };
        let mut j_phi: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        // Their `<J . B>` is zero by construction: the diamagnetic current is perpendicular to `B`, and the Pfirsch-Schlüter current averages
        // to zero over each surface
        let mut j_parallel: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        if usable {
            j_parallel.fill(0.0);
            let samples: Vec<Option<FluxSurfaceSamples>> =
                flux_surface_samples(time_slice).map_err(|reason: String| format!("{name} current at `equilibrium/time_slice({i_time})`: {reason}"))?;
            for i_psi_norm in 0..n_psi_norm {
                if i_psi_norm == 0 {
                    j_phi[i_psi_norm] = 0.0;
                } else if let Some(samples_here) = &samples[i_psi_norm] {
                    j_phi[i_psi_norm] = j_phi_of(
                        time_slice.profiles_1d.dpressure_dpsi[i_psi_norm],
                        time_slice.profiles_1d.f[i_psi_norm],
                        samples_here,
                    );
                }
            }
        }
        components.push(PlasmaCurrentComponent::new(j_phi, j_parallel, area));
    }

    Ok(components)
}

#[cfg(test)]
mod tests {
    use super::super::current_diamagnetic::current_diamagnetic;
    use super::super::current_pfirsch_schluter::current_pfirsch_schluter;
    use super::super::test_equilibrium::{MINOR_RADIUS, PSI_AXIS, R_AXIS, circular_equilibrium};
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

    #[test]
    fn the_integral_inside_is_exact_for_a_current_density_linear_in_the_area() {
        let area: Array1<f64> = Array1::from_vec(vec![0.0, 0.1, 0.3, 0.6, 1.0, 1.5]);
        let mut current_density: Array1<f64> = area.mapv(|area_here: f64| 1.0 + 2.0 * area_here);

        let current_inside: Array1<f64> = integral_inside(&current_density, &area);

        let n_psi_norm: usize = area.len();
        for i_psi_norm in 0..n_psi_norm {
            assert_relative_eq!(current_inside[i_psi_norm], area[i_psi_norm] + area[i_psi_norm].powi(2), max_relative = 1.0e-14);
        }

        // NaN from the first NaN current density outwards, and everywhere if that is on the magnetic axis
        current_density[3] = f64::NAN;
        let current_inside: Array1<f64> = integral_inside(&current_density, &area);
        for i_psi_norm in 0..n_psi_norm {
            assert_eq!(current_inside[i_psi_norm].is_nan(), i_psi_norm >= 3);
        }
        current_density[0] = f64::NAN;
        let current_inside: Array1<f64> = integral_inside(&current_density, &area);
        for i_psi_norm in 0..n_psi_norm {
            assert!(current_inside[i_psi_norm].is_nan());
        }
    }

    /// For any field `B = F grad(phi) - grad(phi) x grad(psi) / (2 pi)`, the Grad-Shafranov current, `J = 2 pi (F' B / mu_0 + p' R ** 2 grad(phi))`,
    /// is the diamagnetic current, plus the Pfirsch-Schlüter current, plus the current along the field lines with
    /// `<J . B> = 2 pi (F p' + F' <B ** 2> / mu_0)`, and its toroidal current density is `2 pi (p' + F F' <R ** -2> / mu_0) / <1 / R>`.
    /// That is algebra, true of any poloidal field, so the circular equilibrium's `psi` is replaced by an elliptical one: the
    /// Pfirsch-Schlüter current vanishes where `R b_p = |grad(psi)| / (2 pi)` is the same everywhere on each surface, as it is on the
    /// circular equilibrium's circles, but not on these ellipses.
    #[test]
    fn the_components_add_up_to_the_grad_shafranov_current_density() {
        // `psi = PSI_AXIS (1 - ((r - R_AXIS) ** 2 + (z / kappa) ** 2) / MINOR_RADIUS ** 2)`, and its boundary outline
        let kappa: f64 = 0.5;
        let mut equilibrium: EquilibriumIds = circular_equilibrium(&[0.1]);
        let time_slice_elliptical: &mut EquilibriumTimeSlice = &mut equilibrium.time_slice[0];
        let n_outline: usize = time_slice_elliptical.boundary.outline.z.len();
        for i_outline in 0..n_outline {
            time_slice_elliptical.boundary.outline.z[i_outline] *= kappa;
        }
        let profiles_2d = &mut time_slice_elliptical.profiles_2d[0];
        let r: Array1<f64> = profiles_2d.grid.dim1.clone();
        let z: Array1<f64> = profiles_2d.grid.dim2.clone();
        let n_z: usize = z.len();
        let n_r: usize = r.len();
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - R_AXIS;
                profiles_2d.psi[(i_z, i_r)] = PSI_AXIS * (1.0 - (delta_r.powi(2) + (z[i_z] / kappa).powi(2)) / MINOR_RADIUS.powi(2));
                profiles_2d.d_psi_d_r[(i_z, i_r)] = -2.0 * PSI_AXIS * delta_r / MINOR_RADIUS.powi(2);
                profiles_2d.d_psi_d_z[(i_z, i_r)] = -2.0 * PSI_AXIS * z[i_z] / (kappa * MINOR_RADIUS).powi(2);
                profiles_2d.d2_psi_d_z2[(i_z, i_r)] = -2.0 * PSI_AXIS / (kappa * MINOR_RADIUS).powi(2);
                profiles_2d.b_field_r[(i_z, i_r)] = -profiles_2d.d_psi_d_z[(i_z, i_r)] / (2.0 * PI * r[i_r]);
                profiles_2d.b_field_z[(i_z, i_r)] = profiles_2d.d_psi_d_r[(i_z, i_r)] / (2.0 * PI * r[i_r]);
            }
        }
        let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(&equilibrium);

        current_diamagnetic(&equilibrium, &mut plasma_current_components).unwrap();
        current_pfirsch_schluter(&equilibrium, &mut plasma_current_components).unwrap();

        let time_slice: &EquilibriumTimeSlice = &equilibrium.time_slice[0];
        let samples: Vec<Option<FluxSurfaceSamples>> = flux_surface_samples(time_slice).unwrap();
        let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();
        let mut j_dot_b: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        let mut j_phi_grad_shafranov: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
        for i_psi_norm in 0..n_psi_norm {
            let f: f64 = time_slice.profiles_1d.f[i_psi_norm];
            let df_dpsi: f64 = time_slice.profiles_1d.f_df_dpsi[i_psi_norm] / f;
            let p_prime: f64 = time_slice.profiles_1d.dpressure_dpsi[i_psi_norm];
            let (b_field_squared_average, r_inverse_squared_average, r_inverse_average): (f64, f64, f64) = match &samples[i_psi_norm] {
                // The magnetic axis
                None => ((f / R_AXIS).powi(2), R_AXIS.powi(-2), 1.0 / R_AXIS),
                Some(samples_here) => (
                    samples_here.average(|_r, b_field: f64, _b_field_p| b_field.powi(2)),
                    samples_here.average(|r_here: f64, _b_field, _b_field_p| r_here.powi(-2)),
                    samples_here.average(|r_here: f64, _b_field, _b_field_p| 1.0 / r_here),
                ),
            };
            j_dot_b[i_psi_norm] = 2.0 * PI * (f * p_prime + df_dpsi * b_field_squared_average / MU_0);
            j_phi_grad_shafranov[i_psi_norm] = 2.0 * PI * (p_prime + f * df_dpsi * r_inverse_squared_average / MU_0) / r_inverse_average;
        }
        let j_phi_parallel: Array1<f64> = j_phi_of_parallel_current(&j_dot_b, &time_slice.profiles_1d.f, &samples, R_AXIS);

        let diamagnetic: &PlasmaCurrentComponent = &plasma_current_components.time_slice[0].diamagnetic;
        let pfirsch_schluter: &PlasmaCurrentComponent = &plasma_current_components.time_slice[0].pfirsch_schluter;
        for i_psi_norm in 0..n_psi_norm {
            assert!(samples[i_psi_norm].is_some() || i_psi_norm == 0);
            assert_relative_eq!(
                diamagnetic.j_phi[i_psi_norm] + pfirsch_schluter.j_phi[i_psi_norm] + j_phi_parallel[i_psi_norm],
                j_phi_grad_shafranov[i_psi_norm],
                max_relative = 1.0e-12
            );
        }

        // Where the pressure gradient is, the Pfirsch-Schlüter current is not negligible, so the test tests it
        let i_psi_norm_middle: usize = n_psi_norm / 2;
        // (2 % of the diamagnetic current, far above the 1e-12 the sum is checked to)
        assert!(pfirsch_schluter.j_phi[i_psi_norm_middle].abs() > 0.01 * diamagnetic.j_phi[i_psi_norm_middle].abs());
    }

    #[test]
    fn components_made_from_another_equilibrium_are_an_error() {
        let equilibrium: EquilibriumIds = circular_equilibrium(&[0.1, 0.2]);
        PlasmaCurrentComponents::new(&equilibrium).check_made_from(&equilibrium).unwrap();

        let mut equilibrium_other_surfaces: EquilibriumIds = equilibrium.clone();
        let psi_norm_other: Array1<f64> = Array1::linspace(0.0, 1.0, 11);
        equilibrium_other_surfaces.time_slice[1].profiles_1d.psi_norm = psi_norm_other;
        let equilibria_other: [EquilibriumIds; 3] = [circular_equilibrium(&[0.1]), circular_equilibrium(&[0.1, 0.3]), equilibrium_other_surfaces];
        for equilibrium_other in equilibria_other {
            let error: String = PlasmaCurrentComponents::new(&equilibrium_other).check_made_from(&equilibrium).unwrap_err();
            assert!(error.contains("make them with `PlasmaCurrentComponents::new` from this equilibrium"), "{error}");
        }
    }
}
