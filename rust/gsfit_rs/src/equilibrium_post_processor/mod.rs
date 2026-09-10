//! Post-processing of the Grad-Shafranov reconstruction, reading the `equilibrium` IDS.
//!
//! Calculators are grouped by the data dictionary path they fill. Files at this module's root
//! either orchestrate the calculation or produce shared intermediate quantities without writing
//! a data dictionary path.
//!
//! Execution order and hidden inter-calculation dependencies are documented in
//! `equilibrium_post_processor.rs`.

mod bp_sq_flux_surface_average;
mod constant_values;
mod dependency_sorter;
mod equilibrium_post_processor;
mod flux_surface_average;
mod flux_surfaces;
mod intermediate_values;
mod boundary;
mod constraints;
mod convergence;
mod global_quantities;
mod profiles_1d;
mod profiles_1d_r_midplane;
mod profiles_2d;
mod sol;

pub use equilibrium_post_processor::equilibrium_post_processor;

/// Names a calculator, so that a dependency can refer to it.
///
/// One variant per calculator, named after what it fills: the module path with `time_slice__`
/// dropped and `::` written `__`. The calculators which fill an `IntermediateValues` field rather
/// than a data dictionary node are `intermediate_values__<field>`
#[expect(non_camel_case_types, reason = "variants are named after the IDS nodes and intermediate-value fields they fill")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalculatorIdentifier {
    boundary__geometry,
    boundary__outline__r__z,
    boundary__psi_norm__rho_tor,
    constraints__diamagnetic_flux_reconstructed,
    convergence__grad_shafranov_deviation_expression__description__index__name,
    global_quantities__area__length_pol__surface__volume,
    global_quantities__beta_pol,
    global_quantities__beta_tor,
    global_quantities__bt_vac_at_r_geo,
    global_quantities__current_centre__r__velocity_z__z,
    global_quantities__delta_r_sep,
    global_quantities__energy_mhd,
    global_quantities__f_x,
    global_quantities__li,
    global_quantities__magnetic_axis_b_field_phi,
    global_quantities__q_95,
    global_quantities__q_axis,
    global_quantities__q_min__psi__psi_norm__rho_tor_norm__value,
    intermediate_values__bp_sq_fs_avg,
    intermediate_values__flux_surfaces,
    profiles_1d__area__volume__derivatives,
    profiles_1d__b_field_average__b_field_max__b_field_min,
    profiles_1d__beta_pol,
    profiles_1d__dpressure_dpsi,
    profiles_1d__dpsi_drho_tor,
    profiles_1d__elongation__squareness__triangularity,
    profiles_1d__f,
    profiles_1d__f_df_dpsi,
    profiles_1d__geometric_axis__r__z,
    profiles_1d__gm1_to_gm9,
    profiles_1d__j_parallel,
    profiles_1d__j_phi,
    profiles_1d__magnetic_shear,
    profiles_1d__phi,
    profiles_1d__pressure,
    profiles_1d__psi,
    profiles_1d__q,
    profiles_1d__r_inboard__r_outboard,
    profiles_1d__rho_pol,
    profiles_1d__rho_tor,
    profiles_1d__rho_tor_norm,
    profiles_1d__rho_volume_norm,
    profiles_1d_r_midplane__pressure,
    profiles_2d__b_field_phi,
    profiles_2d__b_field_r__b_field_z,
    profiles_2d__d_b_field_z_d_z,
    profiles_2d__grid_volume_element,
    profiles_2d__j_parallel,
    profiles_2d__phi,
    profiles_2d__pressure,
    profiles_2d__theta,
    profiles_2d__type__description__index__name,
    sol__legs_and_strike_points,
}