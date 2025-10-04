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
