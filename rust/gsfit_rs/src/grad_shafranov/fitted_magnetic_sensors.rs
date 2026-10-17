use imas_rs::{EMPTY_INT, EquilibriumCodeParametersParsed, EquilibriumTimeSlice};
use ndarray::Array1;

/// The sensors of one kind which are fitted on a time-slice, read from the equilibrium IDS.
///
/// A sensor is fitted when its fit settings, `code/parameters_parsed/constraints/<sensor>/<sensor>(i1)`, have `include` true. The
/// sensors are in the order of their constraints, `time_slice(itime)/constraints/<sensor>`, which is the order of the magnetics IDS.
#[derive(Clone, Debug)]
pub struct FittedSensors {
    /// The name of each fitted sensor, its constraint's `name`. The diamagnetic flux's constraint has no name, as there is only one, so
    /// it is called `diamagnetic_flux`
    pub names: Vec<String>,
    /// Each fitted sensor's measurement, its constraint's `measured`, which `Equilibrium.copy_from_magnetics_ids` copied in
    pub measured: Array1<f64>,
    /// Each fitted sensor's weight, its constraint's `weight`, which `Equilibrium::new` set from its fit settings
    pub weights: Array1<f64>,
    /// Each fitted sensor's expected value, the magnitude its measurement is normalised by, from its fit settings
    pub expected_values: Array1<f64>,
    /// Each fitted sensor's constraint's `source`, which is empty until `Equilibrium.copy_from_magnetics_ids` copies its
    /// measurement in
    pub sources: Vec<String>,
}

impl FittedSensors {
    /// Collect the fitted sensors from `(name, include, measured, weight, expected_value, source)`, one for each constraint
    fn from_constraints(constraints: Vec<(String, bool, f64, f64, f64, String)>) -> Self {
        let mut names: Vec<String> = Vec::new();
        let mut measured: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut expected_values: Vec<f64> = Vec::new();
        let mut sources: Vec<String> = Vec::new();
        for (name, include, constraint_measured, weight, expected_value, source) in constraints {
            if include {
                names.push(name);
                measured.push(constraint_measured);
                weights.push(weight);
                expected_values.push(expected_value);
                sources.push(source);
            }
        }

        Self {
            names,
            measured: Array1::from_vec(measured),
            weights: Array1::from_vec(weights),
            expected_values: Array1::from_vec(expected_values),
            sources,
        }
    }
}

/// The magnetic sensors fitted on one time-slice, read from the equilibrium IDS: which, and with what measurement, weight and
/// expected value.
///
/// This is what the solver fits to, and what the chi squared is calculated from, so both see the same sensors.
#[derive(Clone, Debug)]
pub struct FittedMagneticSensors {
    pub b_field_pol_probe: FittedSensors,
    pub flux_loop: FittedSensors,
    pub rogowski_coil: FittedSensors,
    /// At most one, as the data dictionary has a single diamagnetic flux constraint
    pub diamagnetic_flux: FittedSensors,
}

impl FittedMagneticSensors {
    /// Read the magnetic sensors fitted on `time_slice`.
    ///
    /// Each constraint, in `time_slice(itime)/constraints/<sensor>`, is fitted if the fit settings of the sensor of the same name, in
    /// `code/parameters_parsed/constraints/<sensor>/<sensor>(i1)`, include it. A sensor without fit settings is not fitted, nor is one
    /// whose `include` is unset. The diamagnetic flux, a single constraint with no name, is fitted if its own fit settings,
    /// `code/parameters_parsed/constraints/diamagnetic_flux/diamagnetic_flux`, include it. The weight is the constraint's, and the
    /// expected value the fit settings'.
    ///
    /// # Arguments
    /// * `time_slice` - the time-slice, whose constraints hold the sensors' measurements and weights
    /// * `parameters_parsed` - GSFit's settings, `code/parameters_parsed`, which hold each sensor's `include` and `expected_value`
    pub fn new(time_slice: &EquilibriumTimeSlice, parameters_parsed: &EquilibriumCodeParametersParsed) -> Self {
        // Poloidal field probes
        let settings_b_field_pol_probes = &parameters_parsed.constraints.b_field_pol_probe.b_field_pol_probe;
        let n_b_field_pol_probe: usize = time_slice.constraints.b_field_pol_probe.len();
        let mut b_field_pol_probe_constraints: Vec<(String, bool, f64, f64, f64, String)> = Vec::with_capacity(n_b_field_pol_probe);
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            let constraint = &time_slice.constraints.b_field_pol_probe[i_b_field_pol_probe];
            let mut include: bool = false;
            let mut expected_value: f64 = f64::NAN;
            let n_settings: usize = settings_b_field_pol_probes.len();
            for i_settings in 0..n_settings {
                if settings_b_field_pol_probes[i_settings].name == constraint.name {
                    include = settings_b_field_pol_probes[i_settings].include == 1;
                    expected_value = settings_b_field_pol_probes[i_settings].expected_value;
                }
            }
            b_field_pol_probe_constraints.push((
                constraint.name.clone(),
                include,
                constraint.measured,
                constraint.weight,
                expected_value,
                constraint.source.clone(),
            ));
        }

        // Flux loops
        let settings_flux_loops = &parameters_parsed.constraints.flux_loop.flux_loop;
        let n_flux_loop: usize = time_slice.constraints.flux_loop.len();
        let mut flux_loop_constraints: Vec<(String, bool, f64, f64, f64, String)> = Vec::with_capacity(n_flux_loop);
        for i_flux_loop in 0..n_flux_loop {
            let constraint = &time_slice.constraints.flux_loop[i_flux_loop];
            let mut include: bool = false;
            let mut expected_value: f64 = f64::NAN;
            let n_settings: usize = settings_flux_loops.len();
            for i_settings in 0..n_settings {
                if settings_flux_loops[i_settings].name == constraint.name {
                    include = settings_flux_loops[i_settings].include == 1;
                    expected_value = settings_flux_loops[i_settings].expected_value;
                }
            }
            flux_loop_constraints.push((
                constraint.name.clone(),
                include,
                constraint.measured,
                constraint.weight,
                expected_value,
                constraint.source.clone(),
            ));
        }

        // Rogowski coils
        let settings_rogowski_coils = &parameters_parsed.constraints.rogowski_coil.rogowski_coil;
        let n_rogowski_coil: usize = time_slice.constraints.rogowski_coil.len();
        let mut rogowski_coil_constraints: Vec<(String, bool, f64, f64, f64, String)> = Vec::with_capacity(n_rogowski_coil);
        for i_rogowski_coil in 0..n_rogowski_coil {
            let constraint = &time_slice.constraints.rogowski_coil[i_rogowski_coil];
            let mut include: bool = false;
            let mut expected_value: f64 = f64::NAN;
            let n_settings: usize = settings_rogowski_coils.len();
            for i_settings in 0..n_settings {
                if settings_rogowski_coils[i_settings].name == constraint.name {
                    include = settings_rogowski_coils[i_settings].include == 1;
                    expected_value = settings_rogowski_coils[i_settings].expected_value;
                }
            }
            rogowski_coil_constraints.push((
                constraint.name.clone(),
                include,
                constraint.measured,
                constraint.weight,
                expected_value,
                constraint.source.clone(),
            ));
        }

        // The diamagnetic flux: a single constraint, as the data dictionary has one, with no name. Its fit settings are optional, and their
        // `include` is required, so it is unset only when they are left out
        let settings_diamagnetic_flux = &parameters_parsed.constraints.diamagnetic_flux.diamagnetic_flux;
        let constraint = &time_slice.constraints.diamagnetic_flux;
        let mut diamagnetic_flux_constraints: Vec<(String, bool, f64, f64, f64, String)> = Vec::new();
        if settings_diamagnetic_flux.include != EMPTY_INT {
            diamagnetic_flux_constraints.push((
                "diamagnetic_flux".to_string(),
                settings_diamagnetic_flux.include == 1,
                constraint.measured,
                constraint.weight,
                settings_diamagnetic_flux.expected_value,
                constraint.source.clone(),
            ));
        }

        Self {
            b_field_pol_probe: FittedSensors::from_constraints(b_field_pol_probe_constraints),
            flux_loop: FittedSensors::from_constraints(flux_loop_constraints),
            rogowski_coil: FittedSensors::from_constraints(rogowski_coil_constraints),
            diamagnetic_flux: FittedSensors::from_constraints(diamagnetic_flux_constraints),
        }
    }
}
