use interpolation;
use ndarray::Array1;

pub fn copper_resistivity(temperature_in_kelvin: f64) -> f64 {
    let temperatures: Array1<f64> = Array1::from_vec(vec![
        20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 273.150, 293.0, 300.0, 350.0,
        400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1357.60,
    ]);
    let copper_resistivities: Array1<f64> = Array1::from_vec(vec![
        0.000798000000000000,
        0.00249000000000000,
        0.00628000000000000,
        0.0127000000000000,
        0.0219000000000000,
        0.0338000000000000,
        0.0498000000000000,
        0.0707000000000000,
        0.0951000000000000,
        0.152000000000000,
        0.213000000000000,
        0.279000000000000,
        0.346000000000000,
        0.520000000000000,
        0.697000000000000,
        0.872000000000000,
        1.04400000000000,
        1.21500000000000,
        1.38500000000000,
        1.54100000000000,
        1.67600000000000,
        1.72300000000000,
        2.06100000000000,
        2.40000000000000,
        3.08800000000000,
        3.79000000000000,
        4.51200000000000,
        5.26000000000000,
        6.03900000000000,
        6.85600000000000,
        7.71500000000000,
        8.62400000000000,
        9.59000000000000,
        10.1690000000000,
    ]) * 1.0e-8;

    let interpolator = interpolation::Dim1Linear::new(temperatures, copper_resistivities).expect("copper resistivity interpolation failed");
    let result = interpolator
        .interpolate_array1(&Array1::from_vec(vec![temperature_in_kelvin]))
        .expect("copper resistivity interpolation failed");
    return result[0];
}
