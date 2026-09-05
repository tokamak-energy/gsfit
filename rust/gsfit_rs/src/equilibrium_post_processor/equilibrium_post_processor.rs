//! The IMAS-based equilibrium post-processor.
//!
//! The solver produces the poloidal flux, its derivatives and the fitted degrees of freedom. Every
//! *derived* quantity - the profiles, the boundary geometry, the betas, `q`, the flux surfaces and
//! the scrape-off layer - is calculated afterwards, from that solution.
//!
//! This is the replacement for `Plasma::equilibrium_post_processor` in `plasma.rs`, which reads
//! `GsSolution`. The two run side by side, and quantities move across one at a time and are checked
//! against the old path as they go. Once this covers everything the old one writes, the old one and
//! `gs_solution.rs` can both be deleted.
//!
//! It is an inherent method on `Plasma` rather than a free function, so the call site reads the
//! same as the old one; Rust allows an inherent `impl` to live in any module of the crate which
//! defines the type.

use super::epp_bp_sq_flux_surface_average::epp_bp_sq_flux_surface_average;
use super::epp_equilibrium_global_quantities_v_loop::epp_equilibrium_global_quantities_v_loop;
use super::epp_equilibrium_time_slice_boundary_geometry::epp_equilibrium_time_slice_boundary_geometry;
use super::epp_equilibrium_time_slice_boundary_outline::epp_equilibrium_time_slice_boundary_outline;
use super::epp_equilibrium_time_slice_constraints_diamagnetic_flux_reconstructed::epp_equilibrium_time_slice_constraints_diamagnetic_flux_reconstructed;
use super::epp_equilibrium_time_slice_global_quantities_area_and_volume::epp_equilibrium_time_slice_global_quantities_area_and_volume;
use super::epp_equilibrium_time_slice_global_quantities_beta_pol::epp_equilibrium_time_slice_global_quantities_beta_pol;
use super::epp_equilibrium_time_slice_global_quantities_beta_tor::epp_equilibrium_time_slice_global_quantities_beta_tor;
use super::epp_equilibrium_time_slice_global_quantities_bt_vac_at_r_geo::epp_equilibrium_time_slice_global_quantities_bt_vac_at_r_geo;
use super::epp_equilibrium_time_slice_global_quantities_energy_mhd::epp_equilibrium_time_slice_global_quantities_energy_mhd;
use super::epp_equilibrium_time_slice_global_quantities_li::epp_equilibrium_time_slice_global_quantities_li;
use super::epp_equilibrium_time_slice_global_quantities_pressure_2d_sum::epp_equilibrium_time_slice_global_quantities_pressure_2d_sum;
use super::epp_equilibrium_time_slice_global_quantities_q_95::epp_equilibrium_time_slice_global_quantities_q_95;
use super::epp_equilibrium_time_slice_global_quantities_q_axis::epp_equilibrium_time_slice_global_quantities_q_axis;
use super::epp_equilibrium_time_slice_profiles_1d_area_and_volume::epp_equilibrium_time_slice_profiles_1d_area_and_volume;
use super::epp_equilibrium_time_slice_profiles_1d_dpressure_dpsi::epp_equilibrium_time_slice_profiles_1d_dpressure_dpsi;
use super::epp_equilibrium_time_slice_profiles_1d_f::epp_equilibrium_time_slice_profiles_1d_f;
use super::epp_equilibrium_time_slice_profiles_1d_f_df_dpsi::epp_equilibrium_time_slice_profiles_1d_f_df_dpsi;
use super::epp_equilibrium_time_slice_profiles_1d_phi::epp_equilibrium_time_slice_profiles_1d_phi;
use super::epp_equilibrium_time_slice_profiles_1d_pressure::epp_equilibrium_time_slice_profiles_1d_pressure;
use super::epp_equilibrium_time_slice_profiles_1d_psi::epp_equilibrium_time_slice_profiles_1d_psi;
use super::epp_equilibrium_time_slice_profiles_1d_q::epp_equilibrium_time_slice_profiles_1d_q;
use super::epp_equilibrium_time_slice_profiles_1d_rho_pol::epp_equilibrium_time_slice_profiles_1d_rho_pol;
use super::epp_equilibrium_time_slice_profiles_1d_rho_tor::epp_equilibrium_time_slice_profiles_1d_rho_tor;
use super::epp_equilibrium_time_slice_profiles_2d_b_field_phi::epp_equilibrium_time_slice_profiles_2d_b_field_phi;
use super::epp_equilibrium_time_slice_profiles_2d_b_field_r_and_z::epp_equilibrium_time_slice_profiles_2d_b_field_r_and_z;
use super::epp_equilibrium_time_slice_profiles_2d_d_b_field_z_d_z::epp_equilibrium_time_slice_profiles_2d_d_b_field_z_d_z;
use super::epp_equilibrium_time_slice_profiles_2d_pressure::epp_equilibrium_time_slice_profiles_2d_pressure;
use super::epp_equilibrium_time_slice_profiles_r_midplane::epp_equilibrium_time_slice_profiles_r_midplane;
use super::epp_equilibrium_time_slice_sol::epp_equilibrium_time_slice_sol;
use super::epp_flux_surfaces::FluxSurface;
use super::epp_flux_surfaces::epp_flux_surfaces;
use crate::coils::Coils;
use crate::plasma::Plasma;
use crate::source_functions::SharedSourceFunction;
use imas_rs::Equilibrium;
use imas_rs::ids::wall::Wall as WallIds;
use rayon::prelude::*;

impl Plasma {
    /// Post-process the reconstruction, reading the solved `equilibrium` IDS.
    ///
    /// This is the successor to `equilibrium_post_processor`, which reads `GsSolution` instead.
    /// The two run side by side while the post-processing is moved across quantity by quantity, in
    /// the same way the solver itself was moved: `gs_solution.rs` cannot be deleted until this
    /// method covers everything the old one writes.
    ///
    /// Quantities are moved into it one at a time and checked against the old path as they go, so
    /// for now it fills only part of what the old one does.
    ///
    /// # Arguments
    /// * `equilibrium_ids` - the solved equilibrium IDS, one time-slice per reconstruction time.
    ///   Taken mutably: the derived quantities are written back into it
    /// * `coils` - the coils, for quantities which need the measured currents
    /// * `wall_ids` - the wall IDS, which supplies the vacuum vessel the scrape-off layer legs are
    ///   traced up to
    /// * `p_prime_source_function` - the p' source function the reconstruction was run with
    /// * `ff_prime_source_function` - the FF' source function the reconstruction was run with
    ///
    /// The source functions are behaviour rather than data, so they cannot live on the IDS and are
    /// passed in alongside it. The grid comes from `time_slice/profiles_2d(0)/grid`, so no snapshot
    /// of `Plasma` is needed.
    #[allow(unused_variables)]
    pub fn equilibrium_post_processor_new(
        &mut self,
        equilibrium_ids: &mut Equilibrium,
        coils: &Coils,
        wall_ids: &WallIds,
        p_prime_source_function: &SharedSourceFunction,
        ff_prime_source_function: &SharedSourceFunction,
    ) {
        println!("equilibrium_post_processor_new: starting");

        let n_time: usize = equilibrium_ids.time_slice.len();
        if n_time == 0 {
            println!("Plasma.equilibrium_post_processor_new: no time slices to process, returning");
            return;
        }

        // The vacuum toroidal field reference radius is a property of the machine rather than of a
        // time-slice, so it is read once, here. `Plasma::new` sets it from the
        // `vacuum_toroidal_field_reference_radius` setting
        let r0: f64 = equilibrium_ids.vacuum_toroidal_field.r0.unwrap();

        // Flux-surface-averaged b_p ** 2 is evaluated slightly inside the boundary, because
        // b_p = 0 at the x-point, which lies on the boundary for diverted plasmas, making
        // `∮ d_ell / b_p` log-divergent on the separatrix
        let bp_sq_fs_avg_psi_norm: f64 = 0.995;

        // Every quantity here is per-time-slice, so each one takes a single `time_slice` and knows
        // nothing about which slice it is, exactly as the solver does. That independence is what
        // lets the slices run in parallel
        equilibrium_ids.time_slice.par_iter_mut().for_each(|time_slice| {
            // Order matters: each of these may read what an earlier one wrote. `energy_mhd`
            // integrates the pressure, so the pressure has to be there first
            epp_equilibrium_time_slice_boundary_outline(time_slice);

            // The flux surfaces fill no data dictionary path; they are an intermediate quantity,
            // calculated once here and handed to the helpers which integrate around them, rather
            // than each of those re-contouring for itself
            let flux_surfaces: Vec<FluxSurface> = epp_flux_surfaces(time_slice);

            epp_equilibrium_time_slice_profiles_2d_pressure(time_slice, p_prime_source_function);
            epp_equilibrium_time_slice_global_quantities_energy_mhd(time_slice);
            epp_equilibrium_time_slice_profiles_1d_f(time_slice, ff_prime_source_function);
            epp_equilibrium_time_slice_profiles_1d_f_df_dpsi(time_slice, ff_prime_source_function);
            epp_equilibrium_time_slice_profiles_1d_psi(time_slice);
            epp_equilibrium_time_slice_profiles_1d_pressure(time_slice, p_prime_source_function);
            epp_equilibrium_time_slice_profiles_1d_dpressure_dpsi(time_slice, p_prime_source_function);
            epp_equilibrium_time_slice_profiles_2d_b_field_phi(time_slice, ff_prime_source_function);
            epp_equilibrium_time_slice_profiles_2d_b_field_r_and_z(time_slice);
            epp_equilibrium_time_slice_profiles_2d_d_b_field_z_d_z(time_slice);
            epp_equilibrium_time_slice_profiles_r_midplane(time_slice, p_prime_source_function);
            epp_equilibrium_time_slice_global_quantities_pressure_2d_sum(time_slice);
            epp_equilibrium_time_slice_profiles_1d_area_and_volume(time_slice, &flux_surfaces);
            epp_equilibrium_time_slice_global_quantities_area_and_volume(time_slice);
            epp_equilibrium_time_slice_profiles_1d_q(time_slice, &flux_surfaces);
            epp_equilibrium_time_slice_profiles_1d_phi(time_slice);
            epp_equilibrium_time_slice_profiles_1d_rho_tor(time_slice);
            epp_equilibrium_time_slice_profiles_1d_rho_pol(time_slice);
            epp_equilibrium_time_slice_global_quantities_q_axis(time_slice);
            epp_equilibrium_time_slice_global_quantities_q_95(time_slice);
            epp_equilibrium_time_slice_constraints_diamagnetic_flux_reconstructed(time_slice, &flux_surfaces);
            epp_equilibrium_time_slice_boundary_geometry(time_slice);

            // Flux-surface-averaged b_p ** 2 fills no data dictionary path either; it is shared by
            // `beta_pol_1` and `li_1`
            let bp_sq_fs_avg: f64 = epp_bp_sq_flux_surface_average(time_slice, &flux_surfaces, bp_sq_fs_avg_psi_norm);

            epp_equilibrium_time_slice_global_quantities_beta_pol(time_slice, bp_sq_fs_avg, r0);
            epp_equilibrium_time_slice_global_quantities_li(time_slice, bp_sq_fs_avg);
            epp_equilibrium_time_slice_global_quantities_bt_vac_at_r_geo(time_slice);
            epp_equilibrium_time_slice_global_quantities_beta_tor(time_slice);
            epp_equilibrium_time_slice_sol(time_slice, wall_ids);
        });

        // The loop voltage differentiates the boundary flux across time, so unlike everything above
        // it needs all of the time-slices at once and cannot run inside the loop
        epp_equilibrium_global_quantities_v_loop(equilibrium_ids);

        // TODO: move the post-processing across from `equilibrium_post_processor`, one quantity at
        // a time, checking each against the old path as it moves
    }
}
