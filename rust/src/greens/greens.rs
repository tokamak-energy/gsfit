use core::f64;
use ndarray::{Array1, Array2, s};
use physical_constants;
use rayon::prelude::*;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind

// Global constants
const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculates the Green's table between locations
///
/// Equations taken from:
/// J.-M. Moret et. al., "Tokamak equilibrium reconstruction code LIUQE and its real time implementation" Fusion Eng. Des., 2015
///
/// The equations are invariant under the exchange of (r, z) with (r_prime, z_prime)
///
/// # Arguments
///
/// * `r` - by convention used for "sensors"
/// * `z` - by convention used for "sensors"
/// * `r_prime` - by convention used for "current sources"
/// * `z_prime` - by convention used for "current sources"
/// * `d_r` - horizontal width of filament, must be same length as `r`
/// * `d_z` - vertical height of filament, must be same length as `z`
///
/// # Returns
///
/// * `Array2<f64>` - The Greens table between "sensors" and "current sources"
///
/// # Examples
///
/// ```
/// use gsfit_rs::greens::greens;
/// use ndarray::{Array1, Array2};
///
/// // By convention sensors are first
/// let r: Array1<f64> = Array1::from(vec![1.0e-8]); // make it small to avoid the singularity which prevents br from being calculated
/// let z: Array1<f64> = Array1::from(vec![0.00]);
///
/// // By convention current sources second
/// let d: f64 = 1.23456789;
/// let r_prime: Array1<f64> = Array1::from(vec![d, d]);
/// let z_prime: Array1<f64> = Array1::from(vec![-d / 2.0, d / 2.0]);
///
/// let d_r: Array1<f64> = Array1::from(vec![0.0, 0.0]);
/// let d_z: Array1<f64> = Array1::from(vec![0.0, 0.0]);
///
/// // Calculate br and bz
/// let greens_table: Array2<f64> =
///     greens(r, z, r_prime, z_prime, d_r, d_z);
/// ```
///
pub fn greens(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>, d_r: Array1<f64>, d_z: Array1<f64>) -> Array2<f64> {
    let n_grid: usize = r.len();
    let n_filament: usize = r_prime.len();

    // in Fiesta `r_filament` and `z_filament` were repeated to make them the same
    // size as `r_grid` and `z_grid`. Parallelisation is then done on matrix
    // multiplication
    let results: Vec<Array1<f64>> = (0..n_filament)
        .into_par_iter() // Use Rayon to create a parallel iterator
        .map(|i_filament: usize| {
            let r_sq: Array1<f64> = (&r + r_prime[i_filament]).mapv(|x: f64| x.powi(2));
            let z_sq: Array1<f64> = (&z - z_prime[i_filament]).mapv(|x: f64| x.powi(2));

            let rr: Array1<f64> = &r * r_prime[i_filament];
            let k_sq: Array1<f64> = 4.0 * &rr / (r_sq + z_sq);
            let k: Array1<f64> = k_sq.mapv(|x: f64| x.sqrt());

            let elliptic_integral_e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
            let elliptic_integral_k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined

            let mut green_this_filament: Array1<f64> =
                MU_0 * rr.mapv(|x: f64| x.sqrt()) * ((2.0 - &k_sq) * elliptic_integral_k - 2.0 * elliptic_integral_e) / k;

            // Test for checking if source and sensor are at same location
            // this is for grid-grid calculation
            // If we do this earlier we can skip calculating elliptic integrals
            let epsilon: f64 = 1e-6;
            for i in 0..n_grid {
                if (r[i] - r_prime[i_filament]).abs() < epsilon && (z[i] - z_prime[i_filament]).abs() < epsilon {
                    green_this_filament[i] = MU_0
                        * r[i]
                        * ((1.0 + 2.0 * (d_z[i] / (8.0 * r[i])).powi(2) + 2.0 / 3.0 * (d_r[i] / (8.0 * r[i])).powi(2)) * (8.0 * r[i] / (d_r[i] + d_z[i])).ln()
                            - 0.5
                            + 0.5 * (d_z[i] / (8.0 * r[i])).powi(2))
                }
            }

            return green_this_filament;
        })
        .collect();

    let mut greens_array: Array2<f64> = Array2::from_elem((n_grid, n_filament), f64::NAN);
    for i_filament in 0..n_filament {
        greens_array.slice_mut(s![.., i_filament]).assign(&results[i_filament]);
    }

    return greens_array;
}

#[test]
fn test_greens_mutual_inductance() {
    // Test the poloidal flux using a Helmholtz coil, which has an analytic solution

    // Lazy loading of packages which are not used anywhere else in the code
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;
    const PI: f64 = std::f64::consts::PI;

    // Current sources
    // The radius of PF coil is "d", so that I'm consistent with Helmholtz notation / equations
    let current: f64 = 2.3456789;
    let d: f64 = 1.23456789;
    let r_prime: Array1<f64> = Array1::from(vec![d, d]);
    let z_prime: Array1<f64> = Array1::from(vec![-d / 2.0, d / 2.0]);

    // Sensors
    // Define a sensor position
    let r_sensor: f64 = 0.12345;
    let r: Array1<f64> = Array1::from(vec![r_sensor]);
    let z: Array1<f64> = Array1::from(vec![0.00]);

    // Calculate flux
    let d_r: Array1<f64> = Array1::zeros(r.len());
    let d_z: Array1<f64> = Array1::zeros(z.len());
    let psi: Array2<f64> = greens(r.clone(), z.clone(), r_prime, z_prime, d_r, d_z);
    let psi_numerical: Array1<f64> = psi.sum_axis(Axis(1)) * current;
    let psi_numerical: f64 = psi_numerical[0]; // since we have only one sensor

    fn psi_analytic_integrand(d: f64, r: f64) -> f64 {
        let integrand_value: f64 = (-2.0 * r - 5.0 * d) / (5.0 * d.powi(2) + 4.0 * d * r + 4.0 * r.powi(2)).sqrt()
            + (2.0 * r - 5.0 * d) / (5.0 * d.powi(2) - 4.0 * d * r + 4.0 * r.powi(2)).sqrt();

        return integrand_value;
    }

    // Calculate the analytic solution
    let psi_analytic: f64 = 2.0 * PI * MU_0 * (d / 4.0) * current * (psi_analytic_integrand(d, r_sensor) - psi_analytic_integrand(d, 0.0));

    // Assert equal, to within some precision
    let precision: f64 = 1e-10;
    assert_abs_diff_eq!(psi_numerical, psi_analytic, epsilon = precision);
}
