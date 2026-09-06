use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
/// Sensor data which does not change from one solver iteration to the next.
///
/// The solver is handed one of these per time-slice, as a `Vec<Arc<SensorsStatic>>`. Some sensor
/// types genuinely need that: the isoflux sensors move, so their Green's tables really are
/// different at every time-slice. For most sensor types, though, the tables are fixed geometry and
/// every time-slice wants the same numbers.
///
/// Rather than special-casing those, the vector is a vector of `Arc` handles. Sensor types whose
/// data does vary build a new one per time-slice; those whose data does not point every entry at a
/// single shared copy. The solver indexes `[i_time]` either way and cannot tell the difference,
/// but the fixed ones are stored once instead of 480 times — for ST40's magnetics that is about
/// 29 MB in place of 13.7 GB.
pub struct SensorsStatic {
    pub greens_with_grid: Array2<f64>,            // shape = [n_z * n_r, n_sensors]
    pub greens_with_pf: Array2<f64>,              // shape = [n_pf, n_sensors]
    pub greens_with_passives: Array2<f64>,        // shape = [n_dof_total, n_sensors]
    pub greens_d_sensor_dz: Array2<f64>,          // shape = [n_z * n_r, n_sensors]
    pub fit_settings_weight: Array1<f64>,         // shape = [n_sensors]
    pub fit_settings_expected_value: Array1<f64>, // shape = [n_sensors]
    pub geometry_r: Array1<f64>,                  // shape = [n_sensors]; only needed for certain sensor types
    pub geometry_z: Array1<f64>,                  // shape = [n_sensors]; only needed for certain sensor types
}

#[derive(Clone, Debug)]
pub struct SensorsDynamic {
    pub measured: Array1<f64>, // shape = [n_sensors]
}

// Define empty data arrays
pub fn create_empty_sensor_data() -> (SensorsStatic, SensorsDynamic) {
    let results_static_empty: SensorsStatic = SensorsStatic {
        greens_with_grid: Array2::zeros((0, 0)),       // TODO: should be: shape = [n_z * n_r, 0]
        greens_with_pf: Array2::zeros((0, 0)),         // TODO: should be: shape = [n_pf, 0]
        greens_with_passives: Array2::zeros((0, 0)),   // TODO: should be: shape = [n_dof_total, 0]
        greens_d_sensor_dz: Array2::zeros((0, 0)),     // TODO: should be: shape = [n_z * n_r, 0]
        fit_settings_weight: Array1::zeros(0),         // TODO: there could still be a weight even if no sensors?
        fit_settings_expected_value: Array1::zeros(0), // TODO: there could still be an expected value even if no sensors?
        geometry_r: Array1::zeros(0),
        geometry_z: Array1::zeros(0),
    };
    let results_dynamic_empty: SensorsDynamic = SensorsDynamic { measured: Array1::zeros(0) };

    // Return the empty data structures
    (results_static_empty, results_dynamic_empty)
}
