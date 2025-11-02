use ndarray::{Array1, Array2, s};
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// d^2(psi)/d(r^2)
///
/// # Arguments
/// * `r` - by convention used for "sensors", metre
/// * `z` - by convention used for "sensors", same length as `r`, metre
/// * `r_prime` - by convention used for "current sources", metre
/// * `z_prime` - by convention used for "current sources", same length as `r_prime`, metre
///
/// # Returns
/// * `g_d2_psi_d_r2[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"
///
pub fn greens_d2_psi_d_r2(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>) -> Array2<f64> {
    let n_rz: usize = r.len();
    let n_rz_prime: usize = r_prime.len();

    let mut g_d2_psi_d_r2: Array2<f64> = Array2::zeros((n_rz, n_rz_prime));

    for i_rz in 0..n_rz {
        // Define some variables
        let h: Array1<f64> = z[i_rz] - &z_prime;
        let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
        let u_sq: Array1<f64> = (r[i_rz] + &r_prime).mapv(|x: f64| x.powi(2)) + &h_sq;
        let rr: Array1<f64> = r[i_rz] * &r_prime;
        let d_sq: Array1<f64> = (&r_prime - r[i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
        let k_sq: Array1<f64> = 4.0 * &rr / &u_sq;
        let w_sq: Array1<f64> = r_prime.mapv(|x: f64| x.powi(2)) - r[i_rz].powi(2) - &h_sq;
        let y_sq: Array1<f64> = 4.0 * &r[i_rz] * &r_prime - &u_sq; // new variable defined, not in J.-M. Moret's LIUQE paper

        // elliptic integral
        let elliptic_integral_e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
        let elliptic_integral_k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined

        // Equation found from Python's "SymPy" package
        let top_1: Array1<f64> = 2.0 * r[i_rz] * &d_sq.mapv(|x| x.powi(2)) * &u_sq * (r[i_rz] + &r_prime) * &elliptic_integral_e;

        let top_2: Array1<f64> = 2.0
            * r[i_rz]
            * &d_sq
            * &w_sq
            * &y_sq
            * (-2.0 * r[i_rz] * &elliptic_integral_e + r[i_rz] * &elliptic_integral_k - 2.0 * &r_prime * &elliptic_integral_e
                + &r_prime * &elliptic_integral_k);

        let top_3: Array1<f64> = 4.0 * r[i_rz] * &u_sq * &w_sq * &y_sq * (-r[i_rz] + &r_prime) * &elliptic_integral_e;

        let top_4: Array1<f64> = -1.0 * &d_sq.mapv(|x| x.powi(2)) * &u_sq.mapv(|x| x.powi(2)) * &elliptic_integral_e;

        let top_5: Array1<f64> = &d_sq.mapv(|x| x.powi(2)) * &u_sq * &y_sq * &elliptic_integral_k;

        let top_6: Array1<f64> =
            &d_sq * &u_sq * &y_sq * (-4.0 * r[i_rz].powi(2) * &elliptic_integral_e + 3.0 * &w_sq * &elliptic_integral_e - &w_sq * &elliptic_integral_k);

        let bottom: Array1<f64> = 2.0 * &d_sq.mapv(|x| x.powi(2)) * &u_sq.mapv(|x| x.powf(1.5)) * &y_sq;

        let g_d2_psi_d_r2_local: Array1<f64> = MU_0 * (top_1 + top_2 + top_3 + top_4 + top_5 + top_6) / bottom;

        g_d2_psi_d_r2.slice_mut(s![i_rz, ..]).assign(&g_d2_psi_d_r2_local);
    }

    return g_d2_psi_d_r2;
}

#[test]
fn test_d2_psi_dr2() {
    use crate::greens::greens_psi;
    use approx::assert_abs_diff_eq;

    let delta_r: f64 = 1e-3;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r: Array1<f64> = Array1::from_vec(vec![r_value]);
    let z: Array1<f64> = Array1::from_vec(vec![z_value]);
    let r_prime: Array1<f64> = Array1::from_vec(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from_vec(vec![0.8234]);

    let current: f64 = 1.0; // ampere
    let d2_psi_dr2_analytic: f64 = current * greens_d2_psi_d_r2(r.clone(), z.clone(), r_prime.clone(), z_prime.clone())[(0, 0)];

    let r_vec: Array1<f64> = Array1::from_vec(vec![r_value - delta_r, r_value, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from_vec(vec![z_value, z_value, z_value]);

    let psi: Array2<f64> = greens_psi(
        r_vec.clone(),
        z_vec.clone(),
        r_prime.clone(),
        z_prime.clone(),
        r_vec.clone() * 0.0,
        z_vec.clone() * 0.0,
    );
    let psi_left: f64 = psi[(0, 0)];
    let psi_center: f64 = psi[(1, 0)];
    let psi_right: f64 = psi[(2, 0)];

    // Numerical derivative
    let d2_psi_d_r2_numerical: f64 = (psi_left - 2.0 * psi_center + psi_right) / (delta_r.powi(2));

    // Assert that analytic and numerical derivatives are the same
    assert_abs_diff_eq!(d2_psi_dr2_analytic, d2_psi_d_r2_numerical, epsilon = 1e-10);
}
