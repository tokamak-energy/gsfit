use ndarray::{Array1, Array2};

#[derive(Clone)]
pub struct SensorsStatic {
    pub greens_with_grid: Array2<f64>,
    pub greens_with_pf: Array2<f64>,
    pub greens_with_passives: Array2<f64>,
    pub greens_d_sensor_dz: Array2<f64>,
    pub fit_settings_weight: Array1<f64>,
    pub fit_settings_expected_value: Array1<f64>,
}

#[derive(Clone)]
pub struct SensorsDynamic {
    pub measured: Array1<f64>, // shape = [n_sensors]
}


// Define empty data arrays
pub fn create_empty_sensor_data() -> (SensorsStatic, SensorsDynamic) {
    let results_static_empty: SensorsStatic = SensorsStatic {
        greens_with_grid: Array2::zeros((0, 0)), // should be: shape = [n_z * n_r, 0]
        greens_with_pf: Array2::zeros((0, 0)), // should be: shape = [n_pf, 0]
        greens_with_passives: Array2::zeros((0, 0)), // should be: shape = [n_dof_total, 0]
        greens_d_sensor_dz: Array2::zeros((0, 0)), // should be: shape = [n_z * n_r, 0]
        fit_settings_weight: Array1::zeros(0), // there could still be a weight even if no sensors?
        fit_settings_expected_value: Array1::zeros(0), // there could still be an expected value even if no sensors?
    };
    let results_dynamic_empty: SensorsDynamic = SensorsDynamic { measured: Array1::zeros(0) }; // Correct. Shape should be [0].
    return (results_static_empty, results_dynamic_empty);
}