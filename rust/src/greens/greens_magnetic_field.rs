use approx::abs_diff_eq;
use ndarray::{Array1, Array2, s};
use physical_constants;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind

// Global constants
const PI: f64 = std::f64::consts::PI;
const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculates the Green's table for Br and Bz fields
///
/// Equations taken from:
/// J.-M. Moret et. al., "Tokamak equilibrium reconstruction code LIUQE and its real time implementation" Fusion Eng. Des., 2015
///
///
/// # Arguments
///
/// * `r` - "sensor" locations
/// * `z` - "sensor" locations
/// * `r_prime` - "current source" locations
/// * `z_prime` - "current source" locations
/// Calculates the magnetic fields BR and BZ at (R, Z)
/// br = g_br * current;        br = - 1 / (2.0 * PI * r) * d_psi_d_z
/// bz = g_bz * current;        bz = 1 / (2.0 * PI * r) * d_psi_d_r
///
/// IMPORTANT: this is NOT symmetric in (r, z) and (r_prime, z_prime)
///
/// # Arguments
///
/// * `r` - the location where we want to calcualte the field
/// * `z` - the location where we want to calcualte the field
/// * `r_prime` - the current sources
/// * `z_prime` - the current sources
///
/// # Returns
///
/// * `g_br[n_rz, n_rz_prime]` where br = g_br * current
/// * `g_bz[n_rz, n_rz_prime]` where bz = g_bz * current
///
/// # Examples
///
/// ```
/// use gsfit_rs::greens::greens_magnetic_field;
/// use ndarray::{Array1, Array2};
///
/// let r: Array1<f64> = Array1::from(vec![1.0, 2.0]);
/// let z: Array1<f64> = Array1::from(vec![3.0, 4.0]);
/// let r_prime: Array1<f64> = Array1::from(vec![5.0, 6.0]);
/// let z_prime: Array1<f64> = Array1::from(vec![7.0, 8.0]);
///
/// let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens_magnetic_field(r, z, r_prime, z_prime);
///
/// println!("g_br: {:?}", g_br);
/// println!("g_bz: {:?}", g_bz);
/// ```
pub fn greens_magnetic_field(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let n_rz: usize = r.len();
    let n_rz_prime: usize = r_prime.len();

    let mut g_br: Array2<f64> = Array2::zeros((n_rz, n_rz_prime));
    let mut g_bz: Array2<f64> = Array2::zeros((n_rz, n_rz_prime));

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
        let w_sq: Array1<f64> = r_prime.mapv(|x: f64| x.powi(2)) - r[i_rz].powi(2) - h_sq;

        // elliptic integral
        let elliptic_integral_e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
        let elliptic_integral_k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined

        let mut g_br_here: Array1<f64> = MU_0 * h / (2.0 * PI * r[i_rz] * &u) * (v_sq * &elliptic_integral_e / &d_sq - &elliptic_integral_k);
        let mut g_bz_here: Array1<f64> = MU_0 / (2.0 * PI * u) * (w_sq * &elliptic_integral_e / d_sq + &elliptic_integral_k);

        // Check if source and sensor are at same location
        // this is for grid-grid calculation
        // If we do this earlier we can skip calculating elliptic integrals
        // TODO: There is probably a better equation for the self-field!!!
        let n_rz_prime: usize = r_prime.len();
        for i_rz_prime in 0..n_rz_prime {
            if abs_diff_eq!(r[i_rz], r_prime[i_rz_prime]) && abs_diff_eq!(z[i_rz], z_prime[i_rz_prime]) {
                g_br_here[i_rz_prime] = 0.0;
                g_bz_here[i_rz_prime] = 0.0;
            }
        }

        g_br.slice_mut(s![i_rz, ..]).assign(&g_br_here);
        g_bz.slice_mut(s![i_rz, ..]).assign(&g_bz_here);
    }
    return (g_br, g_bz);
}

#[test]
fn test_greens_magnetic_field() {
    // Test the magnetic field using a Helmholtz coil, which has an analytic solution

    // Lazy loading of packages which are not used anywhere else in the code
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    // Current sources
    // The radius of PF coil is "d", so that I'm consistent with Helmholtz notation / equations
    let d: f64 = 1.23456789;
    let r_prime: Array1<f64> = Array1::from(vec![d, d]);
    let z_prime: Array1<f64> = Array1::from(vec![-d / 2.0, d / 2.0]);

    // The first test is for the centre of the Helmholtz coil Bz(R=0, Z=0)
    // Sensors
    let r: Array1<f64> = Array1::from(vec![1.0e-8]); // make it small to avoid the singularity which prevents br from being calculated
    let z: Array1<f64> = Array1::from(vec![0.00]);

    // Calculate br and bz
    let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens_magnetic_field(r.clone(), z.clone(), r_prime.clone(), z_prime.clone());
    let current: f64 = 2.3456789;
    let br_numerical: Array1<f64> = g_br.sum_axis(Axis(1)) * current; // summing over the PF coils
    let bz_numerical: Array1<f64> = g_bz.sum_axis(Axis(1)) * current; // summing over the PF coils

    // Analytic values (at the centre of the coil)
    let br_analytic: f64 = 0.0;
    let bz_analytic: f64 = (4.0_f64 / 5.0_f64).powi(3).sqrt() * MU_0 * &current / &d;

    // Assert equal, to within some precision
    let precision: f64 = 1e-6;
    assert_abs_diff_eq!(br_numerical[0], br_analytic, epsilon = precision); // the [0] is because we have only one sensor
    assert_abs_diff_eq!(bz_numerical[0], bz_analytic, epsilon = precision);

    // Second test is for the mid-plane Bz(R=r_sensor, Z=0); with r_sensor != 0
    // Analytic value for the field along the mid-plane (Z=0)

    // Define a sensor position
    let r_sensor: f64 = 0.5678;

    // Sensors
    let r: Array1<f64> = Array1::from(vec![r_sensor]);
    let z: Array1<f64> = Array1::from(vec![0.00]);

    let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens_magnetic_field(r.clone(), z.clone(), r_prime.clone(), z_prime.clone());
    let br_numerical_test2: Array1<f64> = g_br.sum_axis(Axis(1)) * current;
    let bz_numerical_test2: Array1<f64> = g_bz.sum_axis(Axis(1)) * current;

    let bz_analytic_test2: f64 = MU_0
        * (&d.powi(2) / 2.0)
        * current
        * (((r_sensor + &d / 2.0).powi(2) + &d.powi(2)).powf(-3.0 / 2.0) + ((r_sensor - &d / 2.0).powi(2) + &d.powi(2)).powf(-3.0 / 2.0));

    let precision: f64 = 1e-7;
    assert_abs_diff_eq!(br_numerical_test2[0], 0.0, epsilon = precision); // the [0] is because we have only one sensor
    assert_abs_diff_eq!(bz_numerical_test2[0], bz_analytic_test2, epsilon = precision); // the [0] is because we have only one sensor
}
