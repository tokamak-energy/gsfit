use ndarray::{Array1, Array2, s};
use physical_constants;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind

// Global constants
const PI: f64 = std::f64::consts::PI;
const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

///
/// d_g_br_dz = d(g_br)/d(z) = the gradient of br with respect to z
/// d_g_bz_dz = d(g_bz)/d(z) = the gradient of bz with respect to z
pub fn d_greens_magnetic_field_dz(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let n_rz: usize = r.len();
    let n_rz_prime: usize = r_prime.len();

    let mut d_g_br_dz: Array2<f64> = Array2::zeros((n_rz, n_rz_prime));
    let mut d_g_bz_dz: Array2<f64> = Array2::zeros((n_rz, n_rz_prime));

    for i_rz in 0..n_rz {
        // Define some variables
        let h: Array1<f64> = z[i_rz] - &z_prime;
        let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
        let u_sq: Array1<f64> = (r[i_rz] + &r_prime).mapv(|x: f64| x.powi(2)) + &h_sq;
        let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
        let rr: Array1<f64> = r[i_rz] * &r_prime;
        let d_sq: Array1<f64> = (&r_prime - r[i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
        let k_sq: Array1<f64> = 4.0 * &rr / &u_sq;
        let v_sq: Array1<f64> = r_prime.mapv(|x: f64| x.powi(2)) + r[i_rz].powi(2) + &h_sq;
        let w_sq: Array1<f64> = r_prime.mapv(|x: f64| x.powi(2)) - r[i_rz].powi(2) - &h_sq;

        // elliptic integral
        let elliptic_integral_e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
        let elliptic_integral_k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined

        let d_g_br_dz_this_slice: Array1<f64> = MU_0 / (2.0 * PI * r[i_rz] * &d_sq * &u * &u_sq)
            * ((&v_sq * &u_sq - &h_sq * (&d_sq + &k_sq * &u_sq * &u_sq / &d_sq)) * &elliptic_integral_e
                + (h_sq * &v_sq - &u_sq * &d_sq) * &elliptic_integral_k);

        let d_g_bz_dz_this_slice: Array1<f64> =
            MU_0 * &h / (2.0 * PI * &d_sq * u * &u_sq) * (-(3.0 * u_sq + 4.0 * v_sq * &w_sq / d_sq) * elliptic_integral_e + w_sq * elliptic_integral_k);

        d_g_br_dz.slice_mut(s![i_rz, ..]).assign(&d_g_br_dz_this_slice);
        d_g_bz_dz.slice_mut(s![i_rz, ..]).assign(&d_g_bz_dz_this_slice);
    }
    return (d_g_br_dz, d_g_bz_dz);
}

#[test]
fn test_d_greens_magnetic_field_dz() {
    // Test d(br)/d(z) and d(bz)/d(z) using a single PF coil.
    // While there might be an analytic solution, what I am doing is numerically
    // differentiating br and bz from `greens_magnetic_field` function.

    // Lazy loading of packages which are not used anywhere else in the code
    use crate::greens::greens_magnetic_field;
    use approx::assert_abs_diff_eq;

    // Create two sensors which have the same radius and are close in z
    // so that the numerical derivative is accurate.
    let sensor_z_minus_epsilon: f64 = 0.245;
    let sensor_z_plus_epsilon: f64 = 0.2451;
    let sensor_r: f64 = 0.1234;
    let z: Array1<f64> = Array1::from(vec![sensor_z_minus_epsilon, sensor_z_plus_epsilon]); // close in z
    let r: Array1<f64> = Array1::from(vec![0.1234, 0.1234]); // same radius

    // A single current source
    let d: f64 = 1.23456789;
    let r_prime: Array1<f64> = Array1::from(vec![d]);
    let z_prime: Array1<f64> = Array1::from(vec![-d / 2.0]);

    // Calculate br and bz using the `greens_magnetic_field` function
    // Rember: g_br[n_rz, n_rz_prime] and g_bz[n_rz, n_rz_prime]
    // If we were to assume 1A of current is flowing in the coil, then `br = g_br * 1` and `bz = g_bz * 1`
    let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens_magnetic_field(r.clone(), z.clone(), r_prime.clone(), z_prime.clone());

    // Numerically differentiate br and bz, i.e. d(g_br)/d(z) and d(g_bz)/d(z)
    let d_g_br_dz_numerical: f64 = (g_br[[1, 0]] - g_br[[0, 0]]) / (sensor_z_plus_epsilon - sensor_z_minus_epsilon);
    let d_g_bz_dz_numerical: f64 = (g_bz[[1, 0]] - g_bz[[0, 0]]) / (sensor_z_plus_epsilon - sensor_z_minus_epsilon);

    // Call the function we are testing
    // We now have a single sensor at the centre of the two sensors we used for numerical differentiation
    let z: Array1<f64> = Array1::from(vec![(sensor_z_minus_epsilon + sensor_z_plus_epsilon) / 2.0]);
    let r: Array1<f64> = Array1::from(vec![sensor_r]);
    let (d_g_br_dz, d_g_bz_dz): (Array2<f64>, Array2<f64>) = d_greens_magnetic_field_dz(r.clone(), z.clone(), r_prime.clone(), z_prime.clone());

    // Check the results
    let precision: f64 = 1e-11;
    assert_abs_diff_eq!(d_g_br_dz[[0, 0]], d_g_br_dz_numerical, epsilon = precision);
    assert_abs_diff_eq!(d_g_bz_dz[[0, 0]], d_g_bz_dz_numerical, epsilon = precision);
}
