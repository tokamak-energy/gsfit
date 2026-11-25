//! A synthetic equilibrium and kinetic profiles, shared by the tests of the plasma current's components.

use imas_rs::ids::core_profiles::{CoreProfileIons, CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d};
use imas_rs::{Equilibrium as EquilibriumIds, EquilibriumTimeSlice};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// The major radius of the magnetic axis, and of the vacuum toroidal field [metre]
pub(super) const R_AXIS: f64 = 0.5;
/// The minor radius of the boundary [metre]
pub(super) const MINOR_RADIUS: f64 = 0.25;
/// `psi` on the magnetic axis; it is zero on the boundary [weber]
pub(super) const PSI_AXIS: f64 = 0.05;
/// The vacuum toroidal field at `R_AXIS` [tesla]
pub(super) const B0: f64 = 1.5;
/// The pressure on the magnetic axis [pascal]
pub(super) const PRESSURE_AXIS: f64 = 2.0e4;
/// How much `F` on the magnetic axis exceeds its vacuum value, `R_AXIS B0` [dimensionless]
pub(super) const F_PARAMAGNETIC: f64 = 0.02;

/// A circular equilibrium at each of `times`, with
/// * `psi = PSI_AXIS (1 - ((r - R_AXIS) ** 2 + z ** 2) / MINOR_RADIUS ** 2)`, falling outwards as GSFit's does for a positive plasma
///   current, so `psi_norm = ((r - R_AXIS) ** 2 + z ** 2) / MINOR_RADIUS ** 2`
/// * the pressure `p = PRESSURE_AXIS (1 - psi_norm) ** 2`, in `profiles_1d/dpressure_dpsi`
/// * `F = R_AXIS B0 (1 + F_PARAMAGNETIC (1 - psi_norm))`, the vacuum field on the boundary, with `profiles_1d/f_df_dpsi`
/// * `q` NaN on the boundary, as the post-processor leaves it
///
/// Not a solution of the Grad-Shafranov equation, but enough to trace and average over.
pub(super) fn circular_equilibrium(times: &[f64]) -> EquilibriumIds {
    let n_grid: usize = 81;
    let r: Array1<f64> = Array1::linspace(R_AXIS - 1.3 * MINOR_RADIUS, R_AXIS + 1.3 * MINOR_RADIUS, n_grid);
    let z: Array1<f64> = Array1::linspace(-1.3 * MINOR_RADIUS, 1.3 * MINOR_RADIUS, n_grid);
    let mut psi_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    let mut b_field_r_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    let mut b_field_z_2d: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    for i_z in 0..n_grid {
        for i_r in 0..n_grid {
            let delta_r: f64 = r[i_r] - R_AXIS;
            psi_2d[(i_z, i_r)] = PSI_AXIS * (1.0 - (delta_r.powi(2) + z[i_z].powi(2)) / MINOR_RADIUS.powi(2));
            d_psi_d_r_2d[(i_z, i_r)] = -2.0 * PSI_AXIS * delta_r / MINOR_RADIUS.powi(2);
            d_psi_d_z_2d[(i_z, i_r)] = -2.0 * PSI_AXIS * z[i_z] / MINOR_RADIUS.powi(2);
            b_field_r_2d[(i_z, i_r)] = -d_psi_d_z_2d[(i_z, i_r)] / (2.0 * PI * r[i_r]);
            b_field_z_2d[(i_z, i_r)] = d_psi_d_r_2d[(i_z, i_r)] / (2.0 * PI * r[i_r]);
        }
    }

    let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
    time_slice.global_quantities.psi_magnetic_axis = PSI_AXIS;
    time_slice.global_quantities.magnetic_axis.r = R_AXIS;
    time_slice.global_quantities.magnetic_axis.z = 0.0;
    time_slice.boundary.psi = 0.0;
    let n_outline: usize = 401;
    let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_outline);
    time_slice.boundary.outline.r = theta.mapv(|theta_here: f64| R_AXIS + MINOR_RADIUS * theta_here.cos());
    time_slice.boundary.outline.z = theta.mapv(|theta_here: f64| MINOR_RADIUS * theta_here.sin());
    time_slice.profiles_2d = vec![Default::default()];
    time_slice.profiles_2d[0].grid.dim1 = r;
    time_slice.profiles_2d[0].grid.dim2 = z;
    time_slice.profiles_2d[0].psi = psi_2d;
    time_slice.profiles_2d[0].d_psi_d_r = d_psi_d_r_2d;
    time_slice.profiles_2d[0].d_psi_d_z = d_psi_d_z_2d;
    time_slice.profiles_2d[0].d2_psi_d_r2 = Array2::from_elem((n_grid, n_grid), -2.0 * PSI_AXIS / MINOR_RADIUS.powi(2));
    time_slice.profiles_2d[0].d2_psi_d_r_d_z = Array2::zeros((n_grid, n_grid));
    time_slice.profiles_2d[0].d2_psi_d_z2 = Array2::from_elem((n_grid, n_grid), -2.0 * PSI_AXIS / MINOR_RADIUS.powi(2));
    time_slice.profiles_2d[0].b_field_r = b_field_r_2d;
    time_slice.profiles_2d[0].b_field_z = b_field_z_2d;

    let n_psi_norm: usize = 21;
    let psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, n_psi_norm);
    let mut q: Array1<f64> = psi_norm.mapv(|psi_norm_here: f64| 1.0 + 2.0 * psi_norm_here);
    q[n_psi_norm - 1] = f64::NAN;
    // `d(psi_norm) / d(psi) = -1 / PSI_AXIS`
    let f: Array1<f64> = psi_norm.mapv(|psi_norm_here: f64| R_AXIS * B0 * (1.0 + F_PARAMAGNETIC * (1.0 - psi_norm_here)));
    let df_dpsi: f64 = R_AXIS * B0 * F_PARAMAGNETIC / PSI_AXIS;
    time_slice.profiles_1d.psi = psi_norm.mapv(|psi_norm_here: f64| PSI_AXIS * (1.0 - psi_norm_here));
    time_slice.profiles_1d.rho_tor_norm = psi_norm.mapv(f64::sqrt);
    time_slice.profiles_1d.f_df_dpsi = &f * df_dpsi;
    time_slice.profiles_1d.f = f;
    time_slice.profiles_1d.dpressure_dpsi = psi_norm.mapv(|psi_norm_here: f64| 2.0 * PRESSURE_AXIS * (1.0 - psi_norm_here) / PSI_AXIS);
    time_slice.profiles_1d.q = q;
    time_slice.profiles_1d.r_inboard = psi_norm.mapv(|psi_norm_here: f64| R_AXIS - MINOR_RADIUS * psi_norm_here.sqrt());
    time_slice.profiles_1d.r_outboard = psi_norm.mapv(|psi_norm_here: f64| R_AXIS + MINOR_RADIUS * psi_norm_here.sqrt());
    time_slice.profiles_1d.area = psi_norm.mapv(|psi_norm_here: f64| PI * MINOR_RADIUS.powi(2) * psi_norm_here);
    time_slice.profiles_1d.psi_norm = psi_norm;

    let mut equilibrium: EquilibriumIds = EquilibriumIds::default();
    let n_time: usize = times.len();
    for i_time in 0..n_time {
        let mut time_slice_here: EquilibriumTimeSlice = time_slice.clone();
        time_slice_here.time = times[i_time];
        equilibrium.time_slice.push(time_slice_here);
    }
    equilibrium.vacuum_toroidal_field.r0 = R_AXIS;
    equilibrium.vacuum_toroidal_field.b0 = Array1::from_elem(n_time, B0);
    equilibrium.code.output_flag = Array1::zeros(n_time);

    equilibrium
}

/// Kinetic profiles at each of `times`: electrons, deuterium and carbon, with `Z_eff = 2`, and no parallel electric field
pub(super) fn kinetic_profiles_at(times: &[f64]) -> CoreProfilesIds {
    let n_rho_pol_norm: usize = 21;
    let rho_pol_norm: Array1<f64> = Array1::linspace(0.0, 1.0, n_rho_pol_norm);
    let electrons_density: Array1<f64> = rho_pol_norm.mapv(|rho_here: f64| 0.5e19 + 3.5e19 * (1.0 - rho_here.powi(2)).powf(1.5));
    let electrons_temperature: Array1<f64> = rho_pol_norm.mapv(|rho_here: f64| 30.0 + 1470.0 * (1.0 - rho_here.powi(2)).powf(1.5));

    let mut core_profiles: CoreProfilesIds = CoreProfilesIds::default();
    let n_time: usize = times.len();
    for i_time in 0..n_time {
        let mut profiles_1d: CoreProfilesProfiles1d = CoreProfilesProfiles1d {
            time: times[i_time],
            zeff: Array1::from_elem(n_rho_pol_norm, 2.0),
            ..Default::default()
        };
        profiles_1d.grid.rho_pol_norm = rho_pol_norm.clone();
        profiles_1d.electrons.density_thermal = electrons_density.clone();
        profiles_1d.electrons.temperature = electrons_temperature.clone();
        let ion_charges: [f64; 2] = [1.0, 6.0];
        let ion_density_fractions: [f64; 2] = [0.8, 1.0 / 30.0];
        for i_ion in 0..2 {
            profiles_1d.ion.push(CoreProfileIons {
                z_ion: ion_charges[i_ion],
                density_thermal: &electrons_density * ion_density_fractions[i_ion],
                temperature: &electrons_temperature * 2.0,
                ..Default::default()
            });
        }
        core_profiles.profiles_1d.push(profiles_1d);
    }

    core_profiles
}
