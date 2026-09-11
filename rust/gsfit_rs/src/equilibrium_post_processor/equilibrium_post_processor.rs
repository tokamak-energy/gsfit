//! The IMAS-based equilibrium post-processor.
//!
//! The solver produces the poloidal flux, its derivatives and the fitted degrees of freedom. Every
//! *derived* quantity - the profiles, the boundary geometry, the betas, `q`, the flux surfaces and
//! the scrape-off layer - is calculated afterwards, from that solution.

use super::bp_sq_flux_surface_average;
use super::constant_values::ConstantValues;
use super::dependency_sorter;
use super::flux_surfaces;
use super::intermediate_values::IntermediateValues;
use super::intermediate_values::intermediate_values_placeholders;
use super::boundary;
use super::constraints;
use super::convergence;
use super::global_quantities;
use super::profiles_1d;
use super::profiles_1d_r_midplane;
use super::profiles_2d;
use super::sol;
use crate::source_functions::SharedSourceFunction;
use imas_rs::Equilibrium;
use imas_rs::EquilibriumTimeSlice;
use imas_rs::ids::wall::Wall;
use ndarray::Array1;
use rayon::prelude::*;
use std::f64::consts::PI;
use super::CalculatorIdentifier as CI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// The signature every equilibrium post-processor needs to have:
type CalculatorFunction = fn(&mut EquilibriumTimeSlice, &ConstantValues<'_>, &mut IntermediateValues);

/// One row of the calculator table.
///
/// `Copy` because every field is plain data (an enum, a function pointer and a `&'static` slice),
/// which lets `dependency_sorter::sort` copy the rows into their execution order
#[derive(Clone, Copy)]
pub(super) struct CalculatorEntry {
    pub(super) identifier: CI,
    pub(super) calculator_function: CalculatorFunction,
    /// The calculators which fill a node or an `IntermediateValues` field this one reads
    pub(super) dependencies: &'static [CI],
}

/// Build one row of the calculator table, so that the table reads one calculator per line
fn entry(identifier: CI, calculator_function: CalculatorFunction, dependencies: &'static [CI]) -> CalculatorEntry {
    return CalculatorEntry {
        identifier,
        calculator_function,
        dependencies,
    };
}

/// Post-process the reconstruction, reading the solved `equilibrium` IDS.
///
/// # Arguments
/// * `equilibrium_ids` - the solved equilibrium IDS, one time-slice per reconstruction time.
///   Taken mutably: the derived quantities are written back into it
/// * `wall_ids` - the wall IDS, which supplies the vacuum vessel the scrape-off layer legs are
///   traced up to
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
/// * `ff_prime_source_function` - the FF' source function the reconstruction was run with
pub fn equilibrium_post_processor(
    equilibrium_ids: &mut Equilibrium,
    wall_ids: &Wall,
    p_prime_source_function: &SharedSourceFunction,
    ff_prime_source_function: &SharedSourceFunction,
) {
    println!("equilibrium_post_processor: starting");

    equilibrium_ids.code.name = "gsfit".to_string();

    let n_time: usize = equilibrium_ids.time_slice.len();
    if n_time == 0 {
        println!("equilibrium_post_processor: no time slices to process, returning");
        return;
    }

    // Vacuum toroidal field reference radius
    let r0: f64 = equilibrium_ids.vacuum_toroidal_field.r0;
    // Vacuum toroidal field at reference radius
    let b0: &Array1<f64> = &equilibrium_ids.vacuum_toroidal_field.b0;

    // The calculator table: one row per calculator, in whatever order reads best. The order they
    // run in comes from the `dependencies` column, via `dependency_sorter::sort` below
    #[rustfmt::skip]
    let calculators_unsorted: Vec<CalculatorEntry> = vec![
        // identifier,                                                                         calculator_function,                                                                    dependencies
        entry(CI::boundary__geometry,                                                          boundary::geometry::calculate,                                                          &[CI::boundary__outline__r__z]                                                                                                                      ),
        entry(CI::boundary__outline__r__z,                                                     boundary::outline__r__z::calculate,                                                     &[]                                                                                                                                                 ),
        entry(CI::boundary__psi_norm__rho_tor,                                                 boundary::psi_norm__rho_tor::calculate,                                                 &[CI::profiles_1d__rho_tor]                                                                                                                         ),
        entry(CI::constraints__diamagnetic_flux_reconstructed,                                 constraints::diamagnetic_flux_reconstructed::calculate,                                 &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f, CI::profiles_1d__phi, CI::profiles_1d__psi]                                           ),
        entry(CI::convergence__grad_shafranov_deviation_expression__description__index__name,  convergence::grad_shafranov_deviation_expression__description__index__name::calculate,  &[]                                                                                                                                                 ),
        entry(CI::global_quantities__area__length_pol__surface__volume,                        global_quantities::area__length_pol__surface__volume::calculate,                        &[CI::boundary__outline__r__z, CI::profiles_1d__area__volume__derivatives]                                                                          ),
        entry(CI::global_quantities__beta_pol,                                                 global_quantities::beta_pol::calculate,                                                 &[CI::boundary__geometry, CI::global_quantities__energy_mhd, CI::intermediate_values__bp_sq_fs_avg, CI::profiles_1d__area__volume__derivatives]     ),
        entry(CI::global_quantities__beta_tor,                                                 global_quantities::beta_tor::calculate,                                                 &[CI::boundary__geometry, CI::global_quantities__bt_vac_at_r_geo, CI::global_quantities__energy_mhd, CI::profiles_1d__area__volume__derivatives]    ),
        entry(CI::global_quantities__bt_vac_at_r_geo,                                          global_quantities::bt_vac_at_r_geo::calculate,                                          &[CI::boundary__geometry]                                                                                                                           ),
        entry(CI::global_quantities__current_centre__r__velocity_z__z,                         global_quantities::current_centre__r__velocity_z__z::calculate,                         &[]                                                                                                                                                 ),
        entry(CI::global_quantities__delta_r_sep,                                              global_quantities::delta_r_sep::calculate,                                              &[CI::profiles_1d__r_inboard__r_outboard]                                                                                                           ),
        entry(CI::global_quantities__energy_mhd,                                               global_quantities::energy_mhd::calculate,                                               &[CI::profiles_2d__pressure]                                                                                                                        ),
        entry(CI::global_quantities__f_x,                                                      global_quantities::f_x::calculate,                                                      &[CI::profiles_1d__r_inboard__r_outboard, CI::profiles_2d__b_field_r__b_field_z, CI::sol__legs_and_strike_points]                                   ),
        entry(CI::global_quantities__li,                                                       global_quantities::li::calculate,                                                       &[CI::boundary__geometry, CI::intermediate_values__bp_sq_fs_avg, CI::profiles_1d__area__volume__derivatives, CI::profiles_2d__b_field_r__b_field_z] ),
        entry(CI::global_quantities__magnetic_axis_b_field_phi,                                global_quantities::magnetic_axis_b_field_phi::calculate,                                &[CI::profiles_1d__f]                                                                                                                               ),
        entry(CI::global_quantities__q_95,                                                     global_quantities::q_95::calculate,                                                     &[]                                                                                                                                                 ),
        entry(CI::global_quantities__q_axis,                                                   global_quantities::q_axis::calculate,                                                   &[CI::profiles_1d__q]                                                                                                                               ),
        entry(CI::global_quantities__q_min__psi__psi_norm__rho_tor_norm__value,                global_quantities::q_min__psi__psi_norm__rho_tor_norm__value::calculate,                &[CI::profiles_1d__psi, CI::profiles_1d__q, CI::profiles_1d__rho_tor_norm]                                                                          ),
        entry(CI::intermediate_values__bp_sq_fs_avg,                                           bp_sq_flux_surface_average::calculate,                                                  &[CI::intermediate_values__flux_surfaces, CI::profiles_2d__b_field_r__b_field_z]                                                                    ),
        entry(CI::intermediate_values__flux_surfaces,                                          flux_surfaces::calculate,                                                               &[CI::boundary__outline__r__z]                                                                                                                      ),
        entry(CI::profiles_1d__area__volume__derivatives,                                      profiles_1d::area__volume__derivatives::calculate,                                      &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__psi, CI::profiles_1d__rho_tor]                                                           ),
        entry(CI::profiles_1d__b_field_average__b_field_max__b_field_min,                      profiles_1d::b_field_average__b_field_max__b_field_min::calculate,                      &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f, CI::profiles_2d__b_field_r__b_field_z]                                                ),
        entry(CI::profiles_1d__beta_pol,                                                       profiles_1d::beta_pol::calculate,                                                       &[CI::profiles_1d__area__volume__derivatives, CI::profiles_1d__pressure]                                                                            ),
        entry(CI::profiles_1d__dpressure_dpsi,                                                 profiles_1d::dpressure_dpsi::calculate,                                                 &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__dpsi_drho_tor,                                                  profiles_1d::dpsi_drho_tor::calculate,                                                  &[CI::profiles_1d__psi, CI::profiles_1d__rho_tor]                                                                                                   ),
        entry(CI::profiles_1d__elongation__squareness__triangularity,                          profiles_1d::elongation__squareness__triangularity::calculate,                          &[CI::intermediate_values__flux_surfaces]                                                                                                           ),
        entry(CI::profiles_1d__f,                                                              profiles_1d::f::calculate,                                                              &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__f_df_dpsi,                                                      profiles_1d::f_df_dpsi::calculate,                                                      &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__geometric_axis__r__z,                                           profiles_1d::geometric_axis__r__z::calculate,                                           &[CI::intermediate_values__flux_surfaces]                                                                                                           ),
        entry(CI::profiles_1d__gm1_to_gm9,                                                     profiles_1d::gm1_to_gm9::calculate,                                                     &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f, CI::profiles_1d__psi, CI::profiles_1d__rho_tor, CI::profiles_2d__b_field_r__b_field_z]),
        entry(CI::profiles_1d__j_parallel,                                                     profiles_1d::j_parallel::calculate,                                                     &[CI::intermediate_values__flux_surfaces, CI::profiles_2d__b_field_r__b_field_z, CI::profiles_2d__j_parallel]                                       ),
        entry(CI::profiles_1d__j_phi,                                                          profiles_1d::j_phi::calculate,                                                          &[CI::intermediate_values__flux_surfaces, CI::profiles_2d__b_field_r__b_field_z]                                                                    ),
        entry(CI::profiles_1d__magnetic_shear,                                                 profiles_1d::magnetic_shear::calculate,                                                 &[CI::profiles_1d__q, CI::profiles_1d__rho_tor]                                                                                                     ),
        entry(CI::profiles_1d__phi,                                                            profiles_1d::phi::calculate,                                                            &[CI::profiles_1d__psi, CI::profiles_1d__q]                                                                                                         ),
        entry(CI::profiles_1d__pressure,                                                       profiles_1d::pressure::calculate,                                                       &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__psi,                                                            profiles_1d::psi::calculate,                                                            &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__q,                                                              profiles_1d::q::calculate,                                                              &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f]                                                                                       ),
        entry(CI::profiles_1d__r_inboard__r_outboard,                                          profiles_1d::r_inboard__r_outboard::calculate,                                          &[CI::intermediate_values__flux_surfaces]                                                                                                           ),
        entry(CI::profiles_1d__rho_pol,                                                        profiles_1d::rho_pol::calculate,                                                        &[]                                                                                                                                                 ),
        entry(CI::profiles_1d__rho_tor,                                                        profiles_1d::rho_tor::calculate,                                                        &[CI::profiles_1d__phi]                                                                                                                             ),
        entry(CI::profiles_1d__rho_tor_norm,                                                   profiles_1d::rho_tor_norm::calculate,                                                   &[CI::profiles_1d__rho_tor]                                                                                                                         ),
        entry(CI::profiles_1d__rho_volume_norm,                                                profiles_1d::rho_volume_norm::calculate,                                                &[CI::profiles_1d__area__volume__derivatives]                                                                                                       ),
        entry(CI::profiles_1d_r_midplane__dpressure_dpsi,                                      profiles_1d_r_midplane::dpressure_dpsi::calculate,                                      &[]                                                                                                                                                 ),
        entry(CI::profiles_1d_r_midplane__f,                                                   profiles_1d_r_midplane::f::calculate,                                                   &[]                                                                                                                                                 ),
        entry(CI::profiles_1d_r_midplane__f_df_dpsi,                                           profiles_1d_r_midplane::f_df_dpsi::calculate,                                           &[]                                                                                                                                                 ),
        entry(CI::profiles_1d_r_midplane__j_phi,                                               profiles_1d_r_midplane::j_phi::calculate,                                               &[]                                                                                                                                                 ),
        entry(CI::profiles_1d_r_midplane__pressure,                                            profiles_1d_r_midplane::pressure::calculate,                                            &[]                                                                                                                                                 ),
        entry(CI::profiles_1d_r_midplane__q,                                                   profiles_1d_r_midplane::q::calculate,                                                   &[CI::profiles_1d__q]                                                                                                                               ),
        entry(CI::profiles_2d__b_field_phi,                                                    profiles_2d::b_field_phi::calculate,                                                    &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__b_field_r__b_field_z,                                           profiles_2d::b_field_r__b_field_z::calculate,                                           &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__d_b_field_z_d_z,                                                profiles_2d::d_b_field_z_d_z::calculate,                                                &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__grid_volume_element,                                            profiles_2d::grid_volume_element::calculate,                                            &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__j_parallel,                                                     profiles_2d::j_parallel::calculate,                                                     &[CI::profiles_2d__b_field_phi, CI::profiles_2d__b_field_r__b_field_z]                                                                              ),
        entry(CI::profiles_2d__phi,                                                            profiles_2d::phi::calculate,                                                            &[CI::profiles_1d__phi]                                                                                                                             ),
        entry(CI::profiles_2d__pressure,                                                       profiles_2d::pressure::calculate,                                                       &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__theta,                                                          profiles_2d::theta::calculate,                                                          &[]                                                                                                                                                 ),
        entry(CI::profiles_2d__type__description__index__name,                                 profiles_2d::type__description__index__name::calculate,                                 &[]                                                                                                                                                 ),
        entry(CI::sol__legs_and_strike_points,                                                 sol::legs_and_strike_points::calculate,                                                 &[]                                                                                                                                                 ),
    ];

    // Sort the calculator execution order based on dependencies
    let calculators_sorted: Vec<CalculatorEntry> = dependency_sorter::sort(calculators_unsorted);

    // Every quantity here is per-time-slice, so each one takes a single `time_slice` and knows
    // nothing about which slice it is, exactly as the solver does. That independence is what
    // lets the slices run in parallel
    equilibrium_ids.time_slice.par_iter_mut().enumerate().for_each(|(i_time, time_slice)| {
    // equilibrium_ids.time_slice.iter_mut().enumerate().for_each(|(i_time, time_slice)| {
        // Every calculator takes the same `(time_slice, &constant_values, &mut intermediate_values)`
        // triple, which is what lets them be held in one table and called from one loop
        //
        // The two are split by whether a calculator can create an ordering dependency through
        // them. `constant_values` is fixed before the first calculator runs and passed by shared
        // reference, so the compiler proves it cannot; `intermediate_values` is passed by mutable
        // reference precisely because calculators do fill it for each other, so every field on it
        // is an ordering dependency which has to be declared in the `dependencies` column of
        // `calculators_unsorted`
        let constant_values: ConstantValues = ConstantValues {
            b0: b0[i_time],
            ff_prime_source_function,
            i_rod: 2.0 * PI * r0 * b0[i_time] / MU_0,
            p_prime_source_function,
            r0,
            wall_ids,
        };
        let mut intermediate_values: IntermediateValues = intermediate_values_placeholders();

        // Dispatching is all this loop does: `calculators_sorted` is already in dependency order,
        // so running them in that order is enough
        // println!("i_time = {:#?}", i_time);
        // println!("time = {:#?}", time_slice.time);
        for calculator in &calculators_sorted {
            let convergence_flag: i32 = time_slice.convergence.result.index;
            if convergence_flag == 1 {
                // Correctly converged
                let calculator_function: CalculatorFunction = calculator.calculator_function;
                calculator_function(time_slice, &constant_values, &mut intermediate_values);
            } else {
                // Unconverged, skip equilibrium post-processing
                
            }
            // println!("{:?}", calculator.identifier);  // added for debugging
            
            
        }
    });

    // The loop voltage differentiates the boundary flux across time, so unlike everything above
    // it needs all of the time-slices at once and cannot run inside the loop
    global_quantities::v_loop::calculate(equilibrium_ids);
    // The current centre's vertical velocity likewise differentiates across time; it needs every
    // time-slice's `current_centre/z`, which the loop above has just filled
    global_quantities::current_centre__r__velocity_z__z::calculate_velocity_z(equilibrium_ids);
}
