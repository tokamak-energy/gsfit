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
    mode: Mode,
}

/// A flag to indicate how the Greens object was initialised
enum Mode {
    SensorToConductor,
    SelfField,
}

/// `F(lambda)`: the fraction of the Grad-Shafranov delta-function source,
/// `delta_star(psi) = -2 * PI * MU_0 * r * j`, which is picked up by `d2(psi)/d(z2)`
/// for a rectangular current-carrying cell of aspect ratio `lambda = d_r / d_z`
/// which is embedded in a regular lattice of filaments (i.e. a discretised conductor
/// or the computational grid).
///
/// The self-cell Greens entry (per unit current) is:
/// `g_d2_psi_d_z2_self = -2 * PI * MU_0 * r * F(lambda) / (d_r * d_z)`
/// and, by the splitting identity `F_r + F_z = 1`:
/// `g_d2_psi_d_r2_self = -2 * PI * MU_0 * r * (1 - F(lambda)) / (d_r * d_z)`
///
/// Closed form (derived from row-wise summation of the 2d dipole lattice):
/// `F(lambda) = 1 - PI / (6 * lambda) + (PI / lambda) * sum_{n>=1} csch^2(n * PI / lambda)`
/// with the complement identity `F(lambda) = 1 - F(1 / lambda)` used for `lambda > 1`.
///
/// Note: `F` can lie outside `[0, 1]` for elongated cells (e.g. `F(2) = 1.047`); this is
/// correct - the self entry also compensates the quadrature error which the point-filament
/// approximation makes in the *neighbouring* lattice cells.
///
/// See `documentation/jump_condition_dbr_dz.md` for the full derivation.
fn d2_psi_d_z2_self_fraction(lambda: f64) -> f64 {
    if lambda > 1.0 {
        return 1.0 - d2_psi_d_z2_self_fraction(1.0 / lambda);
    }
    // For lambda <= 1 the series argument is >= PI, so terms decay like exp(-2 * PI * n / lambda)
    // and 3-4 terms suffice
    let mut sum: f64 = 0.0;
    for n in 1..=20 {
        let sinh_term: f64 = (PI * (n as f64) / lambda).sinh();
        let term: f64 = 1.0 / (sinh_term * sinh_term);
        sum += term;
        if term < 1e-17 {
            break;
        }
    }
    1.0 - PI / (6.0 * lambda) + PI / lambda * sum
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
    /// let greens_calculator: Greens = Greens::sensor_to_conductor(
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
    pub fn sensor_to_conductor(
        r_sensor: Array1<f64>,
        z_sensor: Array1<f64>,
        r_conductor: Array1<f64>,
        z_conductor: Array1<f64>,
        d_r_conductor: Array1<f64>,
        d_z_conductor: Array1<f64>,
    ) -> Self {
        // Sensors
        let n_rz: usize = r_sensor.len();
        assert!(z_sensor.len() == n_rz, "`r_sensor` and `z_sensor` must have the same length");

        // Conductors
        let n_rz_prime: usize = r_conductor.len();
        assert!(z_conductor.len() == n_rz_prime, "`r_conductor` and `z_conductor` must have the same length");
        assert!(d_r_conductor.len() == n_rz_prime, "`d_r_conductor` and `r_conductor` must have the same length");
        assert!(d_z_conductor.len() == n_rz_prime, "`d_z_conductor` and `z_conductor` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let r_sq: Array1<f64> = (&r_sensor + r_conductor[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&z_sensor - z_conductor[i_rz_prime]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &r_sensor * r_conductor[i_rz_prime];
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
            r: r_sensor,
            z: z_sensor,
            n_rz,
            r_prime: r_conductor,
            z_prime: z_conductor,
            d_r_prime: d_r_conductor,
            d_z_prime: d_z_conductor,
            n_rz_prime,
            elliptic_integral_e,
            elliptic_integral_k,
            mode: Mode::SensorToConductor,
        }
    }

    /// A constructor for the grid-to-grid Greens table, where the "sensors" and "current sources" are at the same locations
    pub fn self_field(r: Array1<f64>, z: Array1<f64>, d_r: Array1<f64>, d_z: Array1<f64>) -> Self {
        let r_sensor: Array1<f64> = r.clone();
        let z_sensor: Array1<f64> = z.clone();

        let r_conductor: Array1<f64> = r.clone();
        let z_conductor: Array1<f64> = z.clone();
        let d_r_conductor: Array1<f64> = d_r.clone();
        let d_z_conductor: Array1<f64> = d_z.clone();

        // Sensors
        let n_rz: usize = r_sensor.len();
        assert!(z_sensor.len() == n_rz, "`r_sensor` and `z_sensor` must have the same length");

        // Conductors
        let n_rz_prime: usize = r_conductor.len();
        assert!(z_conductor.len() == n_rz_prime, "`r_conductor` and `z_conductor` must have the same length");
        assert!(d_r_conductor.len() == n_rz_prime, "`d_r_conductor` and `r_conductor` must have the same length");
        assert!(d_z_conductor.len() == n_rz_prime, "`d_z_conductor` and `z_conductor` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..n_rz_prime)
            .into_par_iter()
            .map(|i_rz_prime: usize| {
                let r_sq: Array1<f64> = (&r_sensor + r_conductor[i_rz_prime]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&z_sensor - z_conductor[i_rz_prime]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &r_sensor * r_conductor[i_rz_prime];
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
            r: r_sensor,
            z: z_sensor,
            n_rz,
            r_prime: r_conductor,
            z_prime: z_conductor,
            d_r_prime: d_r_conductor,
            d_z_prime: d_z_conductor,
            n_rz_prime,
            elliptic_integral_e,
            elliptic_integral_k,
            mode: Mode::SelfField,
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
                            * ((1.0
                                + 2.0 * (d_z_prime[i_rz_prime] / (8.0 * r[i_grid])).powi(2)
                                + 2.0 / 3.0 * (d_r_prime[i_rz_prime] / (8.0 * r[i_grid])).powi(2))
                                * (8.0 * r[i_grid] / (d_r_prime[i_rz_prime] + d_z_prime[i_rz_prime])).ln()
                                - 0.5
                                + 0.5 * (d_z_prime[i_rz_prime] / (8.0 * r[i_grid])).powi(2))
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
    /// When the source and sensor coincide the self-term is computed from the source's
    /// finite dimensions; it takes the complement `1 - F(lambda)` of the `d2(psi)/d(z2)`
    /// fraction, so that the two self-terms together integrate the delta-function source:
    /// `g_d2_psi_d_z2_self + g_d2_psi_d_r2_self = -2 * PI * MU_0 * r / (d_r * d_z)`.
    /// See `d2_psi_d_z2_self_fraction` and `documentation/jump_condition_dbr_dz.md`.
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
        let d_r_prime: &Array1<f64> = &self.d_r_prime;
        let d_z_prime: &Array1<f64> = &self.d_z_prime;
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

                // Self-term when source and sensor are at the same location:
                // the complement of the `d2(psi)/d(z2)` fraction, so that together they
                // integrate the delta-function source of `delta_star(psi)`
                let epsilon: f64 = 1e-6;
                let d_r: f64 = d_r_prime[i_rz_prime];
                let d_z: f64 = d_z_prime[i_rz_prime];
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        if d_r > 0.0 && d_z > 0.0 && d_r.is_finite() && d_z.is_finite() {
                            let f_r: f64 = 1.0 - d2_psi_d_z2_self_fraction(d_r / d_z);
                            g_d2_psi_d_r2_local[i_rz] = -2.0 * PI * MU_0 * r[i_rz] * f_r / (d_r * d_z);
                        } else {
                            // A zero-size filament has a genuinely infinite self value; keep zero
                            g_d2_psi_d_r2_local[i_rz] = 0.0;
                        }
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

                // Set to zero when source and sensor are at the same location.
                // Unlike `d2_psi_d_z2` and `d2_psi_d_r2`, zero is exact here: the mixed
                // derivative's singular kernel is odd in both (r - r_prime) and (z - z_prime),
                // so the self-cell contribution vanishes by symmetry
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
    /// When the source and sensor coincide the filament expression is singular (the
    /// delta-function source in `delta_star(psi) = -2 * PI * MU_0 * r * j` cannot be
    /// obtained by differentiating the smooth Greens function). The self-term is instead
    /// computed from the source's finite dimensions, see `d2_psi_d_z2_self_fraction`
    /// and `documentation/jump_condition_dbr_dz.md`.
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
        let d_r_prime: &Array1<f64> = &self.d_r_prime;
        let d_z_prime: &Array1<f64> = &self.d_z_prime;
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

                // Self-term when source and sensor are at the same location:
                // the delta-function part of `delta_star(psi)` split by the cell aspect ratio,
                // plus the lattice quadrature correction (both inside `d2_psi_d_z2_self_fraction`)
                let epsilon: f64 = 1e-6;
                let d_r: f64 = d_r_prime[i_rz_prime];
                let d_z: f64 = d_z_prime[i_rz_prime];
                for i_rz in 0..n_rz {
                    if (r[i_rz] - r_prime[i_rz_prime]).abs() < epsilon && (z[i_rz] - z_prime[i_rz_prime]).abs() < epsilon {
                        if d_r > 0.0 && d_z > 0.0 && d_r.is_finite() && d_z.is_finite() {
                            let f_z: f64 = d2_psi_d_z2_self_fraction(d_r / d_z);
                            g_d2_psi_d_z2_local[i_rz] = -2.0 * PI * MU_0 * r[i_rz] * f_z / (d_r * d_z);
                        } else {
                            // A zero-size filament has a genuinely infinite self value; keep zero
                            g_d2_psi_d_z2_local[i_rz] = 0.0;
                        }
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

    /// Calculates `g_d3_psi_d_r_d_z2`, where:
    /// `d3(psi)/d(r)d(z2) = g_d3_psi_d_r_d_z2 * current`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d3_psi_d_r_d_z2[(i_rz, i_rz_prime)]` - The Greens table between "sensors" and "current sources"`
    pub fn d3_psi_d_r_d_z2(&self) -> Array2<f64> {
        unimplemented!("need to derive equation for d3_psi_d_r_d_z2");
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), r_prime, z_prime, d_r_prime, d_z_prime);
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d_psi_d_r_analytic: f64 = greens_calculator.d_psi_d_r()[(0, 0)];

    // Compute d_psi_d_r numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d_psi_d_z_analytic: f64 = greens_calculator.d_psi_d_z()[(0, 0)];

    // Compute d_psi_d_z numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_r2_analytic: f64 = greens_calculator.d2_psi_d_r2()[(0, 0)];

    // Compute d2_psi_d_r2 numerically from psi: (psi_left - 2*psi_center + psi_right) / delta_r^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value, z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_r_d_z_analytic: f64 = greens_calculator.d2_psi_d_r_d_z()[(0, 0)];

    // Compute d2_psi_d_r_d_z numerically: d(d_psi_d_r)/d(z)
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
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
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime.clone(), z_prime.clone(), d_r_prime.clone(), d_z_prime.clone());
    let d2_psi_d_z2_analytic: f64 = greens_calculator.d2_psi_d_z2()[(0, 0)];

    // Compute d2_psi_d_z2 numerically from psi: (psi_below - 2*psi_center + psi_above) / delta_z^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, r_prime, z_prime, d_r_prime, d_z_prime);
    let psi: Array2<f64> = greens_calculator.psi();
    let d2_psi_d_z2_numerical: f64 = (psi[(0, 0)] - 2.0 * psi[(1, 0)] + psi[(2, 0)]) / delta_z.powi(2);

    assert_abs_diff_eq!(d2_psi_d_z2_analytic, d2_psi_d_z2_numerical, epsilon = 1e-10);
}

/// Test the lattice self-fraction `F(lambda)` used for the `d2_psi_d_z2` / `d2_psi_d_r2` self-terms
#[test]
fn test_d2_psi_d_z2_self_fraction() {
    use approx::assert_abs_diff_eq;

    // Square cells: the delta-function source splits equally, F(1) = 1/2 exactly
    // (numerically this checks the identity sum_{n>=1} csch^2(n * PI) = 1/6 - 1/(2*PI))
    assert_abs_diff_eq!(d2_psi_d_z2_self_fraction(1.0), 0.5, epsilon = 1e-12);

    // Values validated against direct summation of the 2d dipole lattice
    // (midpoint quadrature error of the kernel K = (x^2 - y^2) / (x^2 + y^2)^2)
    let lambda: f64 = 0.4 / 0.26913578; // cell aspect ratio used in examples/d2_psi_d_z2_investigation.ipynb
    assert_abs_diff_eq!(d2_psi_d_z2_self_fraction(lambda), 0.77654899, epsilon = 1e-6);

    // Note: F is not restricted to [0, 1] for elongated cells; the self entry also
    // compensates the filament-approximation error of the neighbouring lattice cells
    assert_abs_diff_eq!(d2_psi_d_z2_self_fraction(2.0), 1.04710990, epsilon = 1e-6);
}

/// The two second-derivative self-terms together must integrate the delta-function
/// source of the Grad-Shafranov equation: `delta_star(psi) = -2 * PI * MU_0 * r * j`
#[test]
fn test_d2_psi_self_terms_integrate_delta_function() {
    use approx::assert_abs_diff_eq;

    let r0: f64 = 1.2345;
    let z0: f64 = 0.54321;
    let d_r: f64 = 0.01;
    let d_z: f64 = 0.0037;

    let r: Array1<f64> = Array1::from(vec![r0]);
    let z: Array1<f64> = Array1::from(vec![z0]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(
        r.clone(),
        z.clone(),
        r.clone(),
        z.clone(),
        Array1::from(vec![d_r]),
        Array1::from(vec![d_z]),
    );

    let g_z2_self: f64 = greens_calculator.d2_psi_d_z2()[(0, 0)];
    let g_r2_self: f64 = greens_calculator.d2_psi_d_r2()[(0, 0)];

    let delta_integral: f64 = -2.0 * PI * MU_0 * r0 / (d_r * d_z);
    assert_abs_diff_eq!(g_z2_self + g_r2_self, delta_integral, epsilon = delta_integral.abs() * 1e-12);
}

/// Construct the filament grid and sensor used by the self-term tests.
/// The conductor geometry matches `examples/d2_psi_d_z2_investigation.ipynb`.
/// `n` must be odd so that the sensor lands exactly on the central filament.
#[cfg(test)]
fn self_term_test_grid() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64, f64, f64, f64) {
    let r_left: f64 = 0.8345;
    let r_right: f64 = 1.2345;
    let z_bottom: f64 = -0.13456789;
    let z_top: f64 = 0.13456789;

    let n: usize = 101;
    let d_r: f64 = (r_right - r_left) / (n as f64);
    let d_z: f64 = (z_top - z_bottom) / (n as f64);

    let mut r_prime: Array1<f64> = Array1::zeros(n * n);
    let mut z_prime: Array1<f64> = Array1::zeros(n * n);
    for i_z in 0..n {
        for i_r in 0..n {
            r_prime[i_z * n + i_r] = r_left + ((i_r as f64) + 0.5) * d_r;
            z_prime[i_z * n + i_r] = z_bottom + ((i_z as f64) + 0.5) * d_z;
        }
    }
    let d_r_prime: Array1<f64> = Array1::from_elem(n * n, d_r);
    let d_z_prime: Array1<f64> = Array1::from_elem(n * n, d_z);

    // Sensor at the conductor centre = the central filament of the (odd) grid
    let r_sensor: f64 = r_prime[(n * n - 1) / 2];
    let z_sensor: f64 = z_prime[(n * n - 1) / 2];

    (r_prime, z_prime, d_r_prime, d_z_prime, d_r, d_z, r_sensor, z_sensor)
}

/// Sum `psi` over the discretised conductor (current density `j_phi = 1`) at given sensor locations
#[cfg(test)]
fn self_term_test_psi(r_sensors: &Array1<f64>, z_sensors: &Array1<f64>) -> Array1<f64> {
    use ndarray::Axis;

    let (r_prime, z_prime, d_r_prime, d_z_prime, d_r, d_z, _, _) = self_term_test_grid();
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_sensors.clone(), z_sensors.clone(), r_prime, z_prime, d_r_prime, d_z_prime);
    greens_calculator.psi().sum_axis(Axis(1)) * d_r * d_z // each filament carries `j_phi * d_r * d_z` ampere
}

/// Test the `d2_psi_d_z2` self-term: sum the Greens table over a discretised rectangular
/// conductor with the sensor exactly on a filament, and compare against a Richardson-
/// extrapolated second difference of `psi`
#[test]
fn test_d2_psi_d_z2_self_term() {
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    let (r_prime, z_prime, d_r_prime, d_z_prime, d_r, d_z, r_sensor, z_sensor) = self_term_test_grid();

    // Greens-table value, including the new self-term
    let r: Array1<f64> = Array1::from(vec![r_sensor]);
    let z: Array1<f64> = Array1::from(vec![z_sensor]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime, z_prime, d_r_prime, d_z_prime);
    let d2_psi_d_z2_greens: f64 = (greens_calculator.d2_psi_d_z2().sum_axis(Axis(1)) * d_r * d_z)[0];

    // Reference: second difference of psi in z.
    // The offsets are exact multiples of `d_z`, so each evaluation point sits on a
    // filament node where `psi` uses its (finite) self-inductance expression.
    let delta: f64 = 8.0 * d_z;
    let r_sensors: Array1<f64> = Array1::from_elem(5, r_sensor);
    let z_sensors: Array1<f64> = Array1::from(vec![
        z_sensor - 2.0 * delta,
        z_sensor - delta,
        z_sensor,
        z_sensor + delta,
        z_sensor + 2.0 * delta,
    ]);
    let psi: Array1<f64> = self_term_test_psi(&r_sensors, &z_sensors);
    let fd_1: f64 = (psi[1] - 2.0 * psi[2] + psi[3]) / delta.powi(2);
    let fd_2: f64 = (psi[0] - 2.0 * psi[2] + psi[4]) / (2.0 * delta).powi(2);
    let d2_psi_d_z2_numerical: f64 = (4.0 * fd_1 - fd_2) / 3.0; // Richardson extrapolation, O(delta^4)

    assert_abs_diff_eq!(d2_psi_d_z2_greens, d2_psi_d_z2_numerical, epsilon = d2_psi_d_z2_numerical.abs() * 0.01);
}

/// Test the `d2_psi_d_r2` self-term, analogously to `test_d2_psi_d_z2_self_term`
#[test]
fn test_d2_psi_d_r2_self_term() {
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    let (r_prime, z_prime, d_r_prime, d_z_prime, d_r, d_z, r_sensor, z_sensor) = self_term_test_grid();

    // Greens-table value, including the new self-term
    let r: Array1<f64> = Array1::from(vec![r_sensor]);
    let z: Array1<f64> = Array1::from(vec![z_sensor]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, r_prime, z_prime, d_r_prime, d_z_prime);
    let d2_psi_d_r2_greens: f64 = (greens_calculator.d2_psi_d_r2().sum_axis(Axis(1)) * d_r * d_z)[0];

    // Reference: second difference of psi in r, offsets exact multiples of `d_r`
    let delta: f64 = 8.0 * d_r;
    let r_sensors: Array1<f64> = Array1::from(vec![
        r_sensor - 2.0 * delta,
        r_sensor - delta,
        r_sensor,
        r_sensor + delta,
        r_sensor + 2.0 * delta,
    ]);
    let z_sensors: Array1<f64> = Array1::from_elem(5, z_sensor);
    let psi: Array1<f64> = self_term_test_psi(&r_sensors, &z_sensors);
    let fd_1: f64 = (psi[1] - 2.0 * psi[2] + psi[3]) / delta.powi(2);
    let fd_2: f64 = (psi[0] - 2.0 * psi[2] + psi[4]) / (2.0 * delta).powi(2);
    let d2_psi_d_r2_numerical: f64 = (4.0 * fd_1 - fd_2) / 3.0; // Richardson extrapolation, O(delta^4)

    assert_abs_diff_eq!(d2_psi_d_r2_greens, d2_psi_d_r2_numerical, epsilon = d2_psi_d_r2_numerical.abs() * 0.01);
}

// TODO: add tests for b_r and b_z
