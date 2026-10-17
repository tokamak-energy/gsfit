use super::fitted_magnetic_sensors::FittedMagneticSensors;
use imas_rs::{EquilibriumConstraints, EquilibriumConstraints0d, EquilibriumConstraints0dIpLike, EquilibriumConstraints0dOneLike, EquilibriumTimeSlice};

/// Calculate the fitted magnetic sensors' chi squared on every time-slice, and store it as `constraints/chi_squared_reduced`.
///
/// Each fitted sensor adds `((measured - reconstructed) / sigma) ** 2`, where `measured` and `reconstructed` are its constraint's,
/// found by name in `time_slice(itime)/constraints/<sensor>`, and `sigma = expected_value / weight`. A fitted diamagnetic loop uses
/// `constraints/diamagnetic_flux`.
///
/// # Arguments
/// * `time_slices` - the solved time-slices, whose `reconstructed` the equilibrium post-processor has filled;
///   `constraints/chi_squared_reduced` is written into each
/// * `fitted_magnetic_sensors` - the magnetic sensors fitted on each time-slice, with their weights and expected values, one entry per
///   time-slice: the ones the solver fitted with
pub fn epp_chi_sq_mag(time_slices: &mut [EquilibriumTimeSlice], fitted_magnetic_sensors: &[FittedMagneticSensors]) {
    let n_time: usize = time_slices.len();
    for i_time in 0..n_time {
        let constraints: &EquilibriumConstraints = &time_slices[i_time].constraints;
        let fitted: &FittedMagneticSensors = &fitted_magnetic_sensors[i_time];
        let mut chi_sq_mag: f64 = 0.0;

        // bp_probes
        let n_bp_probe: usize = fitted.b_field_pol_probe.names.len();
        for i_bp_probe in 0..n_bp_probe {
            let constraint: &EquilibriumConstraints0dOneLike = constraints.b_field_pol_probe(&fitted.b_field_pol_probe.names[i_bp_probe]);
            let sigma: f64 = fitted.b_field_pol_probe.expected_values[i_bp_probe] / fitted.b_field_pol_probe.weights[i_bp_probe];
            chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
        }

        // flux_loops
        let n_flux_loop: usize = fitted.flux_loop.names.len();
        for i_flux_loop in 0..n_flux_loop {
            let constraint: &EquilibriumConstraints0d = constraints.flux_loop(&fitted.flux_loop.names[i_flux_loop]);
            let sigma: f64 = fitted.flux_loop.expected_values[i_flux_loop] / fitted.flux_loop.weights[i_flux_loop];
            chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
        }

        // rogowski_coils
        let n_rogowski_coil: usize = fitted.rogowski_coil.names.len();
        for i_rogowski_coil in 0..n_rogowski_coil {
            let constraint: &EquilibriumConstraints0dIpLike = constraints.rogowski_coil(&fitted.rogowski_coil.names[i_rogowski_coil]);
            let sigma: f64 = fitted.rogowski_coil.expected_values[i_rogowski_coil] / fitted.rogowski_coil.weights[i_rogowski_coil];
            chi_sq_mag += (constraint.measured - constraint.reconstructed).powi(2) / sigma.powi(2);
        }

        // dialoop: the data dictionary has a single diamagnetic flux constraint, so at most one diamagnetic loop is fitted
        let n_dialoop: usize = fitted.diamagnetic_flux.names.len();
        for i_dialoop in 0..n_dialoop {
            let sigma: f64 = fitted.diamagnetic_flux.expected_values[i_dialoop] / fitted.diamagnetic_flux.weights[i_dialoop];
            chi_sq_mag += (constraints.diamagnetic_flux.measured - constraints.diamagnetic_flux.reconstructed).powi(2) / sigma.powi(2);
        }

        time_slices[i_time].constraints.chi_squared_reduced = chi_sq_mag;
    }
}
