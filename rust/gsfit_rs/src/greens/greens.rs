use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

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
pub struct Greens {
    r: Array1<f64>,
    z: Array1<f64>,
    n_rz: usize,
    r_prime: Array1<f64>,
    z_prime: Array1<f64>,
    d_r_prime: Array1<f64>,
    d_z_prime: Array1<f64>,
    n_rz_prime: usize,
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
    /// * `greens_calculator` - a Greens object from which we can calculate `g_psi`, `g_d_psi_d_r`, ...
    ///
    /// # Example
    /// ```
    /// use ndarray::{Array1, Array2, array};
    /// use gsfit_rs::greens::Greens;
    ///
    /// // Sensors:
    /// let r: Array1<f64> = array![0.12345];
    /// let z: Array1<f64> = array![0.0];
    ///
    /// // Conductors:
    /// let r_prime: Array1<f64> = array![1.23456789, 1.23456789];
    /// let z_prime: Array1<f64> = array![-1.23456789 / 2.0, 1.23456789 / 2.0];
    /// let d_r_prime: Array1<f64> = array![0.0, 0.0];
    /// let d_z_prime: Array1<f64> = array![0.0, 0.0];
    ///
    /// let greens_calculator: Greens = Greens::new(
    ///     r,
    ///     z,
    ///     r_prime,
    ///     z_prime,
    ///     d_r_prime,
    ///     d_z_prime,
    /// );
    ///
    /// // Calculate the Greens between the Helmoltz coils and the flux loops.
    /// let g_psi: Array2<f64> = greens_calculator.psi();  // shape=(n_rz, n_rz_prime)
    ///
    /// println!("g_psi = {:#?}", g_psi);
    /// ```
    pub fn new(r: Array1<f64>, z: Array1<f64>, r_prime: Array1<f64>, z_prime: Array1<f64>, d_r_prime: Array1<f64>, d_z_prime: Array1<f64>) -> Self {
        // Sensors
        let n_rz: usize = r.len();
        assert!(z.len() == n_rz, "`r` and `z` must have the same length");

        // Conductors
        let n_rz_prime: usize = r_prime.len();
        assert!(z_prime.len() == n_rz_prime, "`r_prime` and `z_prime` must have the same length");
        assert!(d_r_prime.len() == n_rz_prime, "`d_r_prime` and `r_prime` must have the same length");
        assert!(d_z_prime.len() == n_rz_prime, "`d_z_prime` and `z_prime` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let r_sq: Array1<f64> = (&r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &r * r_prime[i_rz_prime];
                let k_sq: Array1<f64> = 4.0 * &rr / (r_sq + z_sq);

                let e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
                let k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined differently to E
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
            n_rz,
            r_prime,
            z_prime,
            d_r_prime,
            d_z_prime,
            n_rz_prime,
            elliptic_integral_e,
            elliptic_integral_k,
        }
    }

    pub fn psi(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let d_r_prime: &Array1<f64> = &self.d_r_prime;
        let d_z_prime: &Array1<f64> = &self.d_z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let results: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let r_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = r * r_prime[i_rz_prime];
                let k_sq: Array1<f64> = 4.0 * &rr / (&r_sq + &z_sq);

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);
                let mut green_this_filament = Array1::<f64>::zeros(n_rz);
                for i_rz in 0..n_rz {
                    green_this_filament[i_rz] =
                        MU_0 * rr[i_rz].sqrt() * ((2.0 - k_sq[i_rz]) * elliptic_integral_k_local[i_rz] - 2.0 * elliptic_integral_e_local[i_rz])
                            / k_sq[i_rz].sqrt();
                }

                // Test for checking if source and sensor are at same location
                // this is for grid-grid calculation
                // If we do this earlier we can skip calculating elliptic integrals.
                // But this would be quite complicated as we do the elliptic integrals in initialisation.
                // TODO: should this be smaller?
                // TODO: we should consider how close the filaments are when we are doing mutual inductance between discretised passive filaments
                let epsilon: f64 = 1e-6;
                for i_grid in 0..n_rz {
                    if (r[i_grid] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_grid] - z_prime[i_rz_prime]).abs() < epsilon {
                        green_this_filament[i_grid] = MU_0
                            * r[i_grid]
                            * ((1.0 + 2.0 * (d_z_prime[i_grid] / (8.0 * r[i_grid])).powi(2) + 2.0 / 3.0 * (d_r_prime[i_grid] / (8.0 * r[i_grid])).powi(2))
                                * (8.0 * r[i_grid] / (d_r_prime[i_grid] + d_z_prime[i_grid])).ln()
                                - 0.5
                                + 0.5 * (d_z_prime[i_grid] / (8.0 * r[i_grid])).powi(2))
                    }
                }

                return green_this_filament;
            })
            .collect();

        let mut g_psi: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_psi.slice_mut(s![.., i_rz_prime]).assign(&results[i_rz_prime]);
        }

        g_psi
    }

    /// Calculates `g_d_psi_d_r`, where:
    /// `d(psi)/d(r) = g_d_psi_d_r * current`
    ///
    /// Note: `b_z = d_psi_d_r / (2.0 * PI * r)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d_psi_d_r[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"
    pub fn d_psi_d_r(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d_psi_d_r_vec: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let h_sq: Array1<f64> = (z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let w_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) - x.powi(2)) - &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);

                // g_d_psi_d_r = 2 * PI * r * g_bz
                //             = 2 * PI * r * MU_0 / (2 * PI * u) * (w_sq * E / d_sq + K)
                //             = MU_0 * r / u * (w_sq * E / d_sq + K)
                let mut g_d_psi_d_r_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    g_d_psi_d_r_local[i_rz] =
                        MU_0 * r[i_rz] / u[i_rz] * (w_sq[i_rz] * elliptic_integral_e_local[i_rz] / d_sq[i_rz] + elliptic_integral_k_local[i_rz]);
                }

                // Set to zero when source and sensor are at the same location
                let epsilon: f64 = 1e-6;
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        g_d_psi_d_r_local[i_rz] = 0.0;
                    }
                }

                g_d_psi_d_r_local
            })
            .collect();

        let mut g_d_psi_d_r: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_d_psi_d_r.slice_mut(s![.., i_rz_prime]).assign(&g_d_psi_d_r_vec[i_rz_prime]);
        }

        g_d_psi_d_r
    }

    /// Calculates `g_d_psi_d_z`, where:
    /// `d(psi)/d(z) = g_d_psi_d_z * current`
    ///
    /// Note: `b_r = -d_psi_d_z / (2.0 * PI * r)`
    ///
    /// Arguments
    /// * None
    ///
    /// Returns
    /// * `g_d_psi_d_z[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"
    pub fn d_psi_d_z(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d_psi_d_z_vec: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let h: Array1<f64> = z - z_prime[i_rz_prime];
                let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let v_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) + x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);

                // g_d_psi_d_z = -2 * PI * r * g_br
                //             = -2 * PI * r * MU_0 * h / (2 * PI * r * u) * (v_sq * E / d_sq - K)
                //             = -MU_0 * h / u * (v_sq * E / d_sq - K)
                let mut g_d_psi_d_z_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    g_d_psi_d_z_local[i_rz] =
                        -MU_0 * h[i_rz] / u[i_rz] * (v_sq[i_rz] * elliptic_integral_e_local[i_rz] / d_sq[i_rz] - elliptic_integral_k_local[i_rz]);
                }

                // Set to zero when source and sensor are at the same location
                let epsilon: f64 = 1e-6;
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        g_d_psi_d_z_local[i_rz] = 0.0;
                    }
                }

                g_d_psi_d_z_local
            })
            .collect();

        let mut g_d_psi_d_z: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_d_psi_d_z.slice_mut(s![.., i_rz_prime]).assign(&g_d_psi_d_z_vec[i_rz_prime]);
        }

        g_d_psi_d_z
    }

    /// Calculates `g_d2_psi_d_r2`, where:
    /// `d2(psi)/d(r2) = g_d2_psi_d_r2 * current`
    ///
    /// Equation found using Python's SymPy package (see `greens_d2_psi_d_r2.rs`).
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d2_psi_d_r2[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d2_psi_d_r2(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_r2_vec: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let h_sq: Array1<f64> = (z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let rr: Array1<f64> = r * r_prime[i_rz_prime];
                let d_sq: Array1<f64> = (r - r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let w_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) - x.powi(2)) - &h_sq;
                let y_sq: Array1<f64> = 4.0 * &rr - &u_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);

                let mut g_d2_psi_d_r2_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let ri: f64 = r[i_rz];
                    let rp: f64 = r_prime[i_rz_prime];
                    let e: f64 = elliptic_integral_e_local[i_rz];
                    let k: f64 = elliptic_integral_k_local[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ws: f64 = w_sq[i_rz];
                    let ys: f64 = y_sq[i_rz];

                    let top_1: f64 = 2.0 * ri * ds.powi(2) * us * (ri + rp) * e;
                    let top_2: f64 = 2.0 * ri * ds * ws * ys * ((-2.0 * ri - 2.0 * rp) * e + (ri + rp) * k);
                    let top_3: f64 = 4.0 * ri * us * ws * ys * (-ri + rp) * e;
                    let top_4: f64 = -1.0 * ds.powi(2) * us.powi(2) * e;
                    let top_5: f64 = ds.powi(2) * us * ys * k;
                    let top_6: f64 = ds * us * ys * (-4.0 * ri.powi(2) * e + 3.0 * ws * e - ws * k);

                    let bottom: f64 = 2.0 * ds.powi(2) * us.powf(1.5) * ys;

                    g_d2_psi_d_r2_local[i_rz] = MU_0 * (top_1 + top_2 + top_3 + top_4 + top_5 + top_6) / bottom;
                }

                // Set to zero when source and sensor are at the same location
                let epsilon: f64 = 1e-6;
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        g_d2_psi_d_r2_local[i_rz] = 0.0;
                    }
                }

                g_d2_psi_d_r2_local
            })
            .collect();

        let mut g_d2_psi_d_r2: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_d2_psi_d_r2.slice_mut(s![.., i_rz_prime]).assign(&g_d2_psi_d_r2_vec[i_rz_prime]);
        }

        g_d2_psi_d_r2
    }

    /// Calculates `g_d2_psi_d_r_d_z`, where:
    /// `d2(psi)/d(r)d(z) = g_d2_psi_d_r_d_z * current`
    ///
    /// Derived from: `d_psi_d_r = 2*PI*r * b_z`, so `d2_psi_d_r_d_z = 2*PI*r * d(b_z)/d(z)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d2_psi_d_r_d_z[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d2_psi_d_r_d_z(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_r_d_z_vec: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let h: Array1<f64> = z - z_prime[i_rz_prime];
                let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let v_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) + x.powi(2)) + &h_sq;
                let w_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) - x.powi(2)) - &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);

                // g_d2_psi_d_r_d_z = 2 * PI * r * g_d_bz_dz
                //                  = MU_0 * r * h / (d_sq * u * u_sq) * (-(3 * u_sq + 4 * v_sq * w_sq / d_sq) * E + w_sq * K)
                let mut g_d2_psi_d_r_d_z_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    g_d2_psi_d_r_d_z_local[i_rz] = MU_0 * r[i_rz] * h[i_rz] / (d_sq[i_rz] * u[i_rz] * u_sq[i_rz])
                        * (-(3.0 * u_sq[i_rz] + 4.0 * v_sq[i_rz] * w_sq[i_rz] / d_sq[i_rz]) * elliptic_integral_e_local[i_rz]
                            + w_sq[i_rz] * elliptic_integral_k_local[i_rz]);
                }

                // Set to zero when source and sensor are at the same location
                let epsilon: f64 = 1e-6;
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        g_d2_psi_d_r_d_z_local[i_rz] = 0.0;
                    }
                }

                g_d2_psi_d_r_d_z_local
            })
            .collect();

        let mut g_d2_psi_d_r_d_z: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_d2_psi_d_r_d_z.slice_mut(s![.., i_rz_prime]).assign(&g_d2_psi_d_r_d_z_vec[i_rz_prime]);
        }

        g_d2_psi_d_r_d_z
    }

    /// Calculates `g_d2_psi_d_z2`, where:
    /// `d2(psi)/d(z2) = g_d2_psi_d_z2 * current`
    ///
    /// Derived from: `d_psi_d_z = -2*PI*r * b_r`, so `d2_psi_d_z2 = -2*PI*r * d(b_r)/d(z)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d2_psi_d_z2[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d2_psi_d_z2(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let n_rz_prime: usize = self.n_rz_prime;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let r_prime: &Array1<f64> = &self.r_prime;
        let z_prime: &Array1<f64> = &self.z_prime;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_z2_vec: Vec<Array1<f64>> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let h_sq: Array1<f64> = (z - z_prime[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let rr: Array1<f64> = r * r_prime[i_rz_prime];
                let d_sq: Array1<f64> = (r - r_prime[i_rz_prime]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let k_sq: Array1<f64> = 4.0 * &rr / &u_sq;
                let v_sq: Array1<f64> = r.mapv(|x: f64| r_prime[i_rz_prime].powi(2) + x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., i_rz_prime]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., i_rz_prime]);

                // g_d2_psi_d_z2 = -2 * PI * r * g_d_br_dz
                //               = -MU_0 / (d_sq * u * u_sq) * ((v_sq * u_sq - h_sq * (d_sq + k_sq * u_sq^2 / d_sq)) * E + (h_sq * v_sq - u_sq * d_sq) * K)
                let mut g_d2_psi_d_z2_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let e_term: f64 =
                        (v_sq[i_rz] * u_sq[i_rz] - h_sq[i_rz] * (d_sq[i_rz] + k_sq[i_rz] * u_sq[i_rz].powi(2) / d_sq[i_rz])) * elliptic_integral_e_local[i_rz];
                    let k_term: f64 = (h_sq[i_rz] * v_sq[i_rz] - u_sq[i_rz] * d_sq[i_rz]) * elliptic_integral_k_local[i_rz];

                    g_d2_psi_d_z2_local[i_rz] = -MU_0 / (d_sq[i_rz] * u[i_rz] * u_sq[i_rz]) * (e_term + k_term);
                }

                // Set to zero when source and sensor are at the same location
                let epsilon: f64 = 1e-6;
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        g_d2_psi_d_z2_local[i_rz] = 0.0;
                    }
                }

                g_d2_psi_d_z2_local
            })
            .collect();

        let mut g_d2_psi_d_z2: Array2<f64> = Array2::from_elem((n_rz, n_rz_prime), f64::NAN);
        for i_rz_prime in 0..n_rz_prime {
            g_d2_psi_d_z2.slice_mut(s![.., i_rz_prime]).assign(&g_d2_psi_d_z2_vec[i_rz_prime]);
        }

        g_d2_psi_d_z2
    }

    /// Calculates `b_r`, where:
    /// `b_r = -d(psi)/d(z) / (2.0 * PI * r)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_b_r[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn b_r(&self) -> Array2<f64> {
        let d_psi_d_z: Array2<f64> = self.d_psi_d_z();
        let r: &Array1<f64> = &self.r;

        let mut g_b_r: Array2<f64> = Array2::from_elem((self.n_rz, self.n_rz_prime), f64::NAN);
        for i_rz in 0..self.n_rz {
            for i_rz_prime in 0..self.n_rz_prime {
                g_b_r[(i_rz, i_rz_prime)] = -d_psi_d_z[(i_rz, i_rz_prime)] / (2.0 * PI * r[i_rz]);
            }
        }

        g_b_r
    }

    /// Calculates `b_z`, where:
    /// `b_z = d(psi)/d(r) / (2.0 * PI * r)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_b_z[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn b_z(&self) -> Array2<f64> {
        let d_psi_d_r: Array2<f64> = self.d_psi_d_r();
        let r: &Array1<f64> = &self.r;

        let mut g_b_z: Array2<f64> = Array2::from_elem((self.n_rz, self.n_rz_prime), f64::NAN);
        for i_rz in 0..self.n_rz {
            for i_rz_prime in 0..self.n_rz_prime {
                g_b_z[(i_rz, i_rz_prime)] = d_psi_d_r[(i_rz, i_rz_prime)] / (2.0 * PI * r[i_rz]);
            }
        }

        g_b_z
    }

    /// Calculates d_b_r_d_z
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d_b_r_d_z[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d_b_r_d_z(&self) -> Array2<f64> {
        let d2_psi_d_z2: Array2<f64> = self.d2_psi_d_z2();
        let r: &Array1<f64> = &self.r;

        let mut g_d_b_r_d_z: Array2<f64> = Array2::from_elem((self.n_rz, self.n_rz_prime), f64::NAN);
        for i_rz in 0..self.n_rz {
            for i_rz_prime in 0..self.n_rz_prime {
                g_d_b_r_d_z[(i_rz, i_rz_prime)] = -d2_psi_d_z2[(i_rz, i_rz_prime)] / (2.0 * PI * r[i_rz]);
            }
        }

        g_d_b_r_d_z
    }

    /// Calculates d_b_z_d_z
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d_b_z_d_z[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d_b_z_d_z(&self) -> Array2<f64> {
        let d2_psi_d_r_d_z: Array2<f64> = self.d2_psi_d_r_d_z();
        let r: &Array1<f64> = &self.r;

        let mut g_d_b_z_d_z: Array2<f64> = Array2::from_elem((self.n_rz, self.n_rz_prime), f64::NAN);
        for i_rz in 0..self.n_rz {
            for i_rz_prime in 0..self.n_rz_prime {
                g_d_b_z_d_z[(i_rz, i_rz_prime)] = d2_psi_d_r_d_z[(i_rz, i_rz_prime)] / (2.0 * PI * r[i_rz]);
            }
        }

        g_d_b_z_d_z
    }
}

/// Test the poloidal flux using a Helmholtz coil, which has an analytic solution
#[test]
fn test_greens_psi() {
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;
    use std::f64::consts::PI;

    // Current sources
    // The radius of PF coil is "d", so that I'm consistent with Helmholtz notation / equations
    let current: f64 = 2.3456789;
    let d: f64 = 1.23456789;
    let r_prime: Array1<f64> = Array1::from(vec![d, d]);
    let z_prime: Array1<f64> = Array1::from(vec![-d / 2.0, d / 2.0]);
    let n_rz_prime: usize = r_prime.len();
    let d_r_prime: Array1<f64> = Array1::zeros(n_rz_prime);
    let d_z_prime: Array1<f64> = Array1::zeros(n_rz_prime);

    // Sensors
    // Define a sensor position
    let r_sensor: f64 = 0.12345;
    let r: Array1<f64> = Array1::from(vec![r_sensor]);
    let z: Array1<f64> = Array1::from(vec![0.00]);

    // Calculate flux
    let greens_calculator: Greens = Greens::new(r.clone(), z.clone(), r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let psi_numerical: Array1<f64> = psi.sum_axis(Axis(1)) * current;
    let psi_numerical: f64 = psi_numerical[0]; // since we have only one sensor

    fn psi_analytic_integrand(d: f64, r: f64) -> f64 {
        let integrand_value: f64 = (-2.0 * r - 5.0 * d) / (5.0 * d.powi(2) + 4.0 * d * r + 4.0 * r.powi(2)).sqrt()
            + (2.0 * r - 5.0 * d) / (5.0 * d.powi(2) - 4.0 * d * r + 4.0 * r.powi(2)).sqrt();

        integrand_value
    }

    // Calculate the analytic solution
    let psi_analytic: f64 = 2.0 * PI * MU_0 * (d / 4.0) * current * (psi_analytic_integrand(d, r_sensor) - psi_analytic_integrand(d, 0.0));

    // Assert equal, to within some precision
    // TODO: why this precision?
    let precision: f64 = 1e-10;
    assert_abs_diff_eq!(psi_numerical, psi_analytic, epsilon = precision);
}

/// Test d(psi)/d(r) by numerically differentiating psi
#[test]
fn test_d_psi_d_r() {
    use approx::assert_abs_diff_eq;

    let delta_r: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r_prime: Array1<f64> = Array1::from(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from(vec![0.8234]);
    let d_r_prime: Array1<f64> = Array1::zeros(1);
    let d_z_prime: Array1<f64> = Array1::zeros(1);

    // Compute d_psi_d_r analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::new(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d_psi_d_r_analytic: f64 = greens_calculator.d_psi_d_r()[(0, 0)];

    // Compute d_psi_d_r numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value]);
    let greens_calculator: Greens = Greens::new(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let d_psi_d_r_numerical: f64 = (psi[(1, 0)] - psi[(0, 0)]) / (2.0 * delta_r);

    assert_abs_diff_eq!(d_psi_d_r_analytic, d_psi_d_r_numerical, epsilon = 1e-10);
}

/// Test d(psi)/d(z) by numerically differentiating psi
#[test]
fn test_d_psi_d_z() {
    use approx::assert_abs_diff_eq;

    let delta_z: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r_prime: Array1<f64> = Array1::from(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from(vec![0.8234]);
    let d_r_prime: Array1<f64> = Array1::zeros(1);
    let d_z_prime: Array1<f64> = Array1::zeros(1);

    // Compute d_psi_d_z analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::new(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d_psi_d_z_analytic: f64 = greens_calculator.d_psi_d_z()[(0, 0)];

    // Compute d_psi_d_z numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::new(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let d_psi_d_z_numerical: f64 = (psi[(1, 0)] - psi[(0, 0)]) / (2.0 * delta_z);

    assert_abs_diff_eq!(d_psi_d_z_analytic, d_psi_d_z_numerical, epsilon = 1e-10);
}

/// Test d2(psi)/d(r2) by numerically differentiating psi
#[test]
fn test_d2_psi_d_r2() {
    use approx::assert_abs_diff_eq;

    let delta_r: f64 = 1e-3;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r_prime: Array1<f64> = Array1::from(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from(vec![0.8234]);
    let d_r_prime: Array1<f64> = Array1::zeros(1);
    let d_z_prime: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_r2 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::new(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_r2_analytic: f64 = greens_calculator.d2_psi_d_r2()[(0, 0)];

    // Compute d2_psi_d_r2 numerically from psi: (psi_left - 2*psi_center + psi_right) / delta_r^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value, z_value]);
    let greens_calculator: Greens = Greens::new(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let d2_psi_d_r2_numerical: f64 = (psi[(0, 0)] - 2.0 * psi[(1, 0)] + psi[(2, 0)]) / delta_r.powi(2);

    assert_abs_diff_eq!(d2_psi_d_r2_analytic, d2_psi_d_r2_numerical, epsilon = 1e-10);
}

/// Test d2(psi)/d(r)d(z) by numerically differentiating d_psi_d_r w.r.t. z
#[test]
fn test_d2_psi_d_r_d_z() {
    use approx::assert_abs_diff_eq;

    let delta_z: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r_prime: Array1<f64> = Array1::from(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from(vec![0.8234]);
    let d_r_prime: Array1<f64> = Array1::zeros(1);
    let d_z_prime: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_r_d_z analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::new(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_r_d_z_analytic: f64 = greens_calculator.d2_psi_d_r_d_z()[(0, 0)];

    // Compute d2_psi_d_r_d_z numerically: d(d_psi_d_r)/d(z)
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::new(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r();
    let d2_psi_d_r_d_z_numerical: f64 = (d_psi_d_r[(1, 0)] - d_psi_d_r[(0, 0)]) / (2.0 * delta_z);

    assert_abs_diff_eq!(d2_psi_d_r_d_z_analytic, d2_psi_d_r_d_z_numerical, epsilon = 1e-10);
}

/// Test d2(psi)/d(z2) by numerically differentiating psi
#[test]
fn test_d2_psi_d_z2() {
    use approx::assert_abs_diff_eq;

    let delta_z: f64 = 1e-3;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let r_prime: Array1<f64> = Array1::from(vec![1.52345]);
    let z_prime: Array1<f64> = Array1::from(vec![0.8234]);
    let d_r_prime: Array1<f64> = Array1::zeros(1);
    let d_z_prime: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_z2 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::new(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_z2_analytic: f64 = greens_calculator.d2_psi_d_z2()[(0, 0)];

    // Compute d2_psi_d_z2 numerically from psi: (psi_below - 2*psi_center + psi_above) / delta_z^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value, z_value + delta_z]);
    let greens_calculator: Greens = Greens::new(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let d2_psi_d_z2_numerical: f64 = (psi[(0, 0)] - 2.0 * psi[(1, 0)] + psi[(2, 0)]) / delta_z.powi(2);

    assert_abs_diff_eq!(d2_psi_d_z2_analytic, d2_psi_d_z2_numerical, epsilon = 1e-10);
}

// TODO: add tests for b_r and b_z

// TODO: add tests for sensors and current sources at the same location
