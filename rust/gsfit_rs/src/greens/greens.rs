use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind

/// Greens-function table between "sensors" `(r, z)` and "current sources" `(r_prime, z_prime)`.
///
/// Several methods (`greens_psi`, `greens_d_psi_d_r`, ...) share the same elliptic integrals.
/// We cache **only** the elliptic integrals (`E(k^2)`, `K(1 - k^2)`) and recompute the algebraic
/// intermediates (`r_sq`, `z_sq`, `rr`, `k_sq`, `k`) in each method.
///
/// Why? On modern CPUs, memory is often slower than compute. Caching only pays off when
/// recomputing is more expensive than reading from RAM. Per-element rough costs:
///
/// | Quantity                            | Cost                | Notes                                          |
/// |-------------------------------------|---------------------|------------------------------------------------|
/// | `r_sq`, `z_sq`, `rr`, `k_sq`        | ~1–4 cycles each    | add/mul, vectorizable                          |
/// | `k = k_sq.sqrt()`                   | ~10–20 cycles       | one SIMD instruction                           |
/// | `ellpe(k_sq)`, `ellpk(1 - k_sq)`    | ~50–200+ cycles     | polynomial approximations, harder to vectorize |
/// | Read `f64` from L2 / L3 / RAM       | ~10 / ~40 / ~200+   | depends on cache state                         |
struct Greens {
    r: Array1<f64>,
    z: Array1<f64>,
    r_prime: Array1<f64>,
    z_prime: Array1<f64>,
    elliptic_integral_e: Array2<f64>, // shape (n_rz, n_rz_prime)
    elliptic_integral_k: Array2<f64>, // shape (n_rz, n_rz_prime)
}

impl Greens {
    /// Create a new Greens object, pre-computing the elliptic integrals.
    ///
    /// # Arguments
    /// * `r` - radial coordinates, by convention used for "sensors", [metre]
    /// * `z` - vertical coordinates, by convention used for "sensors", same length as `r`, [metre]
    /// * `r_prime` - radial coordinates, by convention used for "current sources", [metre]
    /// * `z_prime` - vertical coordinates, by convention used for "current sources", same length as `r_prime`, [metre]
    ///
    /// # Returns
    /// * `greens_calculator` - a Greens object from which we can calculate `g_psi`, `d_g_psi_d_r`, ...
    fn new(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>) -> Self {
        let n_rz: usize = r.len();
        assert!(n_rz == z.len(), "`r` and `z` must have the same length");
        let n_rz_prime: usize = r_prime.len();
        assert!(n_rz_prime == z_prime.len(), "`r_prime` and `z_prime` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let r_sq: Array1<f64> = (&r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &r * r_prime[i_rz_prime];
                let k_sq: Array1<f64> = 4.0 * &rr / (r_sq + z_sq);

                let e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
                let k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // ellpk takes 1 - k^2 by convention
                (e, k)
            })
            .collect();

        // Convert to Array2<f64>, with shape = (n_rz, n_rz_prime)
        let mut elliptic_integral_e: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        let mut elliptic_integral_k: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            elliptic_integral_e.slice_mut(s![.., i_rz_prime]).assign(&elliptic_integrals[i_rz_prime].0);
            elliptic_integral_k.slice_mut(s![.., i_rz_prime]).assign(&elliptic_integrals[i_rz_prime].1);
        }

        Greens {
            r,
            z,
            r_prime,
            z_prime,
            elliptic_integral_e,
            elliptic_integral_k,
        }
    }

    fn psi(&self) -> Array2<f64> {
        unimplemented!()
    }

    fn d_psi_d_r(&self) -> Array2<f64> {
        unimplemented!()
    }

    fn d_psi_d_z(&self) -> Array2<f64> {
        unimplemented!()
    }

    fn d2_psi_d_r2(&self) -> Array2<f64> {
        unimplemented!()
    }

    fn d2_psi_d_r_d_z(&self) -> Array2<f64> {
        unimplemented!()
    }

    fn d2_psi_d_z2(&self) -> Array2<f64> {
        unimplemented!()
    }
}
