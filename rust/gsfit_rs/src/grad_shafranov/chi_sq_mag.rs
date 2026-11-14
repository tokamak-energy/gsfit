use crate::settings::settings_equilibrium::{SettingsEquilibrium, SettingsEquilibriumEquilibriumConstraints};
use imas_rs::{EquilibriumConstraints, EquilibriumConstraints0d, EquilibriumConstraints0dIpLike, EquilibriumConstraints0dOneLike, EquilibriumTimeSlice};

/// Calculate the fitted magnetic sensors' chi squared on every time-slice, and store it as `constraints/chi_squared_reduced`.
///
/// The fitted sensors are the ones the solver fits: those whose fit settings, `equilibrium_settings/constraints/<sensor>/<sensor>(i1)`,
/// include them, found by the name of their constraint, `time_slice(itime)/constraints/<sensor>(i1)`, and the diamagnetic flux, a single
/// constraint with no name, when its own fit settings, `equilibrium_settings/constraints/diamagnetic_flux/diamagnetic_flux`, include it.
/// Each adds `((measured - reconstructed) / sigma) ** 2`, where `sigma = expected_value / weight`, all its constraint's.
///
/// # Arguments
/// * `time_slices` - the solved time-slices, whose `reconstructed` the equilibrium post-processor has filled;
///   `constraints/chi_squared_reduced` is written into each
/// * `settings_equilibrium` - GSFit's settings, which say which sensors are fitted
pub fn epp_chi_sq_mag(time_slices: &mut [EquilibriumTimeSlice], settings_equilibrium: &SettingsEquilibrium) {
    let settings_constraints: &SettingsEquilibriumEquilibriumConstraints = &settings_equilibrium.constraints;
    let n_time: usize = time_slices.len();
    for i_time in 0..n_time {
        let constraints: &EquilibriumConstraints = &time_slices[i_time].constraints;
        let mut chi_sq_mag: f64 = 0.0;

        // bp_probes
        let n_bp_probe: usize = constraints.b_field_pol_probe.len();
        let n_bp_probe_settings: usize = settings_constraints.b_field_pol_probe.len();
        for i_bp_probe in 0..n_bp_probe {
            let constraint: &EquilibriumConstraints0dOneLike = &constraints.b_field_pol_probe[i_bp_probe];
            for i_settings in 0..n_bp_probe_settings {
                if settings_constraints.b_field_pol_probe[i_settings].name == constraint.name && settings_constraints.b_field_pol_probe[i_settings].include {
                    let sigma: f64 = constraint.expected_value / constraint.weight;
                    chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
                }
            }
        }

        // flux_loops
        let n_flux_loop: usize = constraints.flux_loop.len();
        let n_flux_loop_settings: usize = settings_constraints.flux_loop.len();
        for i_flux_loop in 0..n_flux_loop {
            let constraint: &EquilibriumConstraints0d = &constraints.flux_loop[i_flux_loop];
            for i_settings in 0..n_flux_loop_settings {
                if settings_constraints.flux_loop[i_settings].name == constraint.name && settings_constraints.flux_loop[i_settings].include {
                    let sigma: f64 = constraint.expected_value / constraint.weight;
                    chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
                }
            }
        }

        // rogowski_coils
        let n_rogowski_coil: usize = constraints.rogowski_coil.len();
        let n_rogowski_coil_settings: usize = settings_constraints.rogowski_coil.len();
        for i_rogowski_coil in 0..n_rogowski_coil {
            let constraint: &EquilibriumConstraints0dIpLike = &constraints.rogowski_coil[i_rogowski_coil];
            for i_settings in 0..n_rogowski_coil_settings {
                if settings_constraints.rogowski_coil[i_settings].name == constraint.name && settings_constraints.rogowski_coil[i_settings].include {
                    let sigma: f64 = constraint.expected_value / constraint.weight;
                    chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
                }
            }
        }

        // dialoop: the data dictionary has a single diamagnetic flux constraint, so at most one diamagnetic loop is fitted
        if let Some(settings_diamagnetic_flux) = &settings_constraints.diamagnetic_flux
            && settings_diamagnetic_flux.include
        {
            let sigma: f64 = constraints.diamagnetic_flux.expected_value / constraints.diamagnetic_flux.weight;
            chi_sq_mag += (constraints.diamagnetic_flux.measured - constraints.diamagnetic_flux.reconstructed).powi(2) / sigma.powi(2);
        }

        time_slices[i_time].constraints.chi_squared_reduced = chi_sq_mag;
    }
}
