use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;
use spec_math::cephes64::ellpe; // complete elliptic integral of the second kind
use spec_math::cephes64::ellpk; // complete elliptic integral of the first kind
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Ad hoc parameter which splits the self-point hoop-field correction `d_psi_d_r / r`
/// between `d2_psi_d_r2` (weight `1/2 - XI`) and `d2_psi_d_z2` (weight `1/2 + XI`), so that
/// Ampere's law, `Delta* psi = -2 * PI * MU_0 * r * j`, is satisfied at the self-point.
/// Numerically matched at `(r, z) = (0.41, 0.0)` with `d_r = d_z = 0.0125`.
const XI: f64 = 0.157;

/// Ad hoc parameter which defines when a sensor which is "close" to a conductor should be considered as at the same location (i.e. self-point).
/// Defined as a module constant to ensure consistency across all Greens-function calculations.
/// Units are metres.
/// We could make the distance exactly 0.0, which would work when calculating the grid-to-grid for Plasma which will be "bit exact".
/// But there are other cases with the passives and coils to the grid which might lie on top of each other, but might contain floating point rounding errors, so we need a small tolerance.
/// We could replace this tolerance with one on `k_sq`, as the problem we are avoiding is that `K(k_sq) --> \infty` as `k_sq --> 1.0`, but that loses the physical meaning of "near distance".
const SELF_POINT_DISTANCE_TOLERANCE: f64 = 1e-7; // = 0.1 μm

/// Greens-function table between "sensors" `(r, z)` and "conductors" `(conductor_r, conductor_z)`.
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
///
/// All tables are written in the "far-field basis":
/// `g = coeff_a * (K - E) + coeff_s * E`
/// where `coeff_s` carries the far-field decay factor `4 * r * conductor_r` (as the sensor moves
/// far from the conductor, `K - E -> 0` and the whole expression tends to `(PI / 2) * coeff_s`).
/// Equations derived and verified with SymPy.
pub struct Greens {
    r: Array1<f64>,
    z: Array1<f64>,
    n_rz: usize,
    conductor_r: Array1<f64>,
    conductor_z: Array1<f64>,
    conductor_d_r: Array1<f64>,
    conductor_d_z: Array1<f64>,
    conductor_n_rz: usize,
    elliptic_integral_e: Array2<f64>, // shape (n_rz, conductor_n_rz)
    elliptic_integral_k: Array2<f64>, // shape (n_rz, conductor_n_rz)
    mode: Mode,
}

/// A flag to indicate how the Greens object was initialised
enum Mode {
    SensorToConductor,
    SelfField,
}

impl Greens {
    /// Create a new Greens object, pre-computing the elliptic integrals.
    ///
    /// # Arguments
    /// * `sensor_r` - radial coordinates of the "sensors", [metre]
    /// * `sensor_z` - vertical coordinates of the "sensors", same length as `sensor_r`, [metre]
    /// * `conductor_r` - radial coordinates of the "conductors", [metre]
    /// * `conductor_z` - vertical coordinates of the "conductors", same length as `conductor_r`, [metre]
    /// * `conductor_d_r` - radial widths of the conductor cross-sections, same length as `conductor_r`, [metre]
    /// * `conductor_d_z` - vertical heights of the conductor cross-sections, same length as `conductor_r`, [metre]
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
    /// let sensor_r: Array1<f64> = array![0.12345];
    /// let sensor_z: Array1<f64> = array![0.0];
    ///
    /// // Conductors:
    /// let conductor_r: Array1<f64> = array![1.23456789, 1.23456789];
    /// let conductor_z: Array1<f64> = array![-1.23456789 / 2.0, 1.23456789 / 2.0];
    /// let conductor_d_r: Array1<f64> = array![0.0, 0.0];
    /// let conductor_d_z: Array1<f64> = array![0.0, 0.0];
    ///
    /// let greens_calculator: Greens = Greens::sensor_to_conductor(
    ///     sensor_r,
    ///     sensor_z,
    ///     conductor_r,
    ///     conductor_z,
    ///     conductor_d_r,
    ///     conductor_d_z,
    /// );
    ///
    /// // Calculate the Greens between the Helmholtz coils and the flux loops.
    /// let g_psi: Array2<f64> = greens_calculator.psi();  // shape=(n_rz, conductor_n_rz)
    ///
    /// println!("g_psi = {:#?}", g_psi);
    /// ```
    pub fn sensor_to_conductor(
        sensor_r: Array1<f64>,
        sensor_z: Array1<f64>,
        conductor_r: Array1<f64>,
        conductor_z: Array1<f64>,
        conductor_d_r: Array1<f64>,
        conductor_d_z: Array1<f64>,
    ) -> Self {
        // Check that `sensor_r` and `conductor_r` are > 0.0
        assert!(sensor_r.iter().all(|&x| x > 0.0), "`sensor_r > 0` is required; this is a physically valid case, but the form we have the equations in is not valid");
        assert!(conductor_r.iter().all(|&x| x > 0.0), "`conductor_r > 0` is required; this is not physically valid if `conductor_d_r` is finite");

        // Sensors
        let n_rz: usize = sensor_r.len();
        assert!(sensor_z.len() == n_rz, "`sensor_r` and `sensor_z` must have the same length");

        // Conductors
        let conductor_n_rz: usize = conductor_r.len();
        assert!(conductor_z.len() == conductor_n_rz, "`conductor_r` and `conductor_z` must have the same length");
        assert!(conductor_d_r.len() == conductor_n_rz, "`conductor_d_r` and `conductor_r` must have the same length");
        assert!(conductor_d_z.len() == conductor_n_rz, "`conductor_d_z` and `conductor_z` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let r_sq: Array1<f64> = (&sensor_r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&sensor_z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &sensor_r * conductor_r[conductor_i_rz];
                let k_sq: Array1<f64> = 4.0 * &rr / (r_sq + z_sq);

                let e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
                let k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined differently to E
                (e, k)
            })
            .collect();

        // Convert to Array2<f64>, with shape = (n_rz, conductor_n_rz)
        let mut elliptic_integral_e: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        let mut elliptic_integral_k: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            elliptic_integral_e.slice_mut(s![.., conductor_i_rz]).assign(&elliptic_integrals[conductor_i_rz].0);
            elliptic_integral_k.slice_mut(s![.., conductor_i_rz]).assign(&elliptic_integrals[conductor_i_rz].1);
        }

        Greens {
            r: sensor_r,
            z: sensor_z,
            n_rz,
            conductor_r,
            conductor_z,
            conductor_d_r,
            conductor_d_z,
            conductor_n_rz,
            elliptic_integral_e,
            elliptic_integral_k,
            mode: Mode::SensorToConductor,
        }
    }

    /// A constructor for the grid-to-grid Greens table, where the "sensors" and "conductors" are at the same locations
    /// NOTE: this is not yet implemented anywhere in the code!!
    pub fn grid_to_grid(r: Array1<f64>, z: Array1<f64>, d_r: Array1<f64>, d_z: Array1<f64>) -> Self {
        let sensor_r: Array1<f64> = r.clone();
        let sensor_z: Array1<f64> = z.clone();

        let conductor_r: Array1<f64> = r.clone();
        let conductor_z: Array1<f64> = z.clone();
        let conductor_d_r: Array1<f64> = d_r.clone();
        let conductor_d_z: Array1<f64> = d_z.clone();

        // Sensors
        let n_rz: usize = sensor_r.len();
        assert!(sensor_z.len() == n_rz, "`sensor_r` and `sensor_z` must have the same length");

        // Conductors
        let conductor_n_rz: usize = conductor_r.len();
        assert!(conductor_z.len() == conductor_n_rz, "`conductor_r` and `conductor_z` must have the same length");
        assert!(conductor_d_r.len() == conductor_n_rz, "`conductor_d_r` and `conductor_r` must have the same length");
        assert!(conductor_d_z.len() == conductor_n_rz, "`conductor_d_z` and `conductor_z` must have the same length");

        // Pre-compute the elliptic integrals
        let elliptic_integrals: Vec<(Array1<f64>, Array1<f64>)> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let r_sq: Array1<f64> = (&sensor_r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (&sensor_z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = &sensor_r * conductor_r[conductor_i_rz];
                let k_sq: Array1<f64> = 4.0 * &rr / (r_sq + z_sq);

                let e: Array1<f64> = k_sq.mapv(|x: f64| ellpe(x));
                let k: Array1<f64> = k_sq.mapv(|x: f64| ellpk(1.0 - x)); // very annoying how this is defined differently to E
                (e, k)
            })
            .collect();

        // Convert to Array2<f64>, with shape = (n_rz, conductor_n_rz)
        let mut elliptic_integral_e: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        let mut elliptic_integral_k: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            elliptic_integral_e.slice_mut(s![.., conductor_i_rz]).assign(&elliptic_integrals[conductor_i_rz].0);
            elliptic_integral_k.slice_mut(s![.., conductor_i_rz]).assign(&elliptic_integrals[conductor_i_rz].1);
        }

        Greens {
            r: sensor_r,
            z: sensor_z,
            n_rz,
            conductor_r,
            conductor_z,
            conductor_d_r,
            conductor_d_z,
            conductor_n_rz,
            elliptic_integral_e,
            elliptic_integral_k,
            mode: Mode::SelfField,
        }
    }

    /// Calculates `g_psi`, where:
    /// `psi = g_psi * current`
    ///
    /// At the self-point the filament expression diverges; the flux at the centre of a
    /// conductor with a rectangular cross-section `conductor_d_r * conductor_d_z` is used instead,
    /// with relative error `O((delta/r)^2 * ln(r/delta))`.
    ///
    /// Note: the flux at the cell centre is NOT the same as the self-inductance: the
    /// self-inductance is the flux linkage, averaged over the cross-section, whereas the
    /// flux peaks inside the conductor, so the centre value is larger (~5% for typical cells).
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_psi[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"
    pub fn psi(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let conductor_d_r: &Array1<f64> = &self.conductor_d_r;
        let conductor_d_z: &Array1<f64> = &self.conductor_d_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let results: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let r_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let z_sq: Array1<f64> = (z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));

                let rr: Array1<f64> = r * conductor_r[conductor_i_rz];
                let k_sq: Array1<f64> = 4.0 * &rr / (&r_sq + &z_sq);
                let u: Array1<f64> = (&r_sq + &z_sq).mapv(|x: f64| x.sqrt());

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_psi = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * sqrt(r * conductor_r) * (2 - k_sq) / k
                // coeff_s = -2 * MU_0 * r * conductor_r / u
                let mut green_this_filament = Array1::<f64>::zeros(n_rz);
                for i_rz in 0..n_rz {
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * rr[i_rz].sqrt() * (2.0 - k_sq[i_rz]) / k_sq[i_rz].sqrt();
                    let coeff_s: f64 = -2.0 * MU_0 * rr[i_rz] / u[i_rz];

                    green_this_filament[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Test for checking if conductor and sensor are at same location
                // this is for grid-grid calculation
                // If we do this earlier we can skip calculating elliptic integrals.
                // But this would be quite complicated as we do the elliptic integrals in initialisation.
                //
                // Self-point: the flux at the centre of the cell from its own uniform current
                // psi = MU_0 * r * (ln(8 * r) - 2 - p_c)
                // where `p_c` is the mean log distance from the cell centre to the cross-section:
                // p_c = 0.5 * ln((d_r^2 + d_z^2) / 4) - 3/2 + (d_r / (2 * d_z)) * atan(d_z / d_r) + (d_z / (2 * d_r)) * atan(d_r / d_z)
                for i_grid in 0..n_rz {
                    if (r[i_grid] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_grid] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        let d_r: f64 = conductor_d_r[conductor_i_rz];
                        let d_z: f64 = conductor_d_z[conductor_i_rz];
                        let p_c: f64 = 0.5 * ((d_r.powi(2) + d_z.powi(2)) / 4.0).ln() - 1.5
                            + d_r / (2.0 * d_z) * (d_z / d_r).atan()
                            + d_z / (2.0 * d_r) * (d_r / d_z).atan();
                        green_this_filament[i_grid] = MU_0 * r[i_grid] * ((8.0 * r[i_grid]).ln() - 2.0 - p_c);
                    }
                }

                return green_this_filament;
            })
            .collect();

        let mut g_psi: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_psi.slice_mut(s![.., conductor_i_rz]).assign(&results[conductor_i_rz]);
        }

        g_psi
    }

    /// Calculates `g_d_psi_d_r`, where:
    /// `d(psi)/d(r) = g_d_psi_d_r * current`
    ///
    /// Note: `b_z = d_psi_d_r / (2.0 * PI * r)`
    ///
    /// Geometric variables:
    /// * `h = z - conductor_z` - height above the plane of the conductor ring
    /// * `d_sq = (r - conductor_r)^2 + h^2` - (distance to the nearest point of the ring)^2
    /// * `u_sq = (r + conductor_r)^2 + h^2` - (distance to the farthest point of the ring)^2
    /// * `w_sq = conductor_r^2 - r^2 - h^2` - minus the "power of the point" w.r.t. the ring circle;
    ///   despite the name it is not a literal square: positive inside the ring circle, negative outside
    ///
    /// `g_d_psi_d_r = coeff_a * (K - E) + coeff_s * E` with:
    /// * `coeff_a = MU_0 * r / u`
    /// * `coeff_s = -2 * MU_0 * r * conductor_r * (r - conductor_r) / (u * d_sq)`
    ///
    /// At the self-point the filament expression diverges; the value for a conductor with a
    /// rectangular cross-section `conductor_d_r * conductor_d_z` (the "hoop field") is used instead,
    /// with relative error `O((delta/r)^2 * ln(r/delta))`.
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d_psi_d_r[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"
    pub fn d_psi_d_r(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let conductor_d_r: &Array1<f64> = &self.conductor_d_r;
        let conductor_d_z: &Array1<f64> = &self.conductor_d_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d_psi_d_r_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h_sq: Array1<f64> = (z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d_psi_d_r = coeff_a * (K - E) + coeff_s * E
                let mut g_d_psi_d_r_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * r[i_rz] / u[i_rz];
                    let coeff_s: f64 = -2.0 * MU_0 * r[i_rz] * rp * (r[i_rz] - rp) / (u[i_rz] * d_sq[i_rz]);

                    g_d_psi_d_r_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: the hoop field of the finite rectangular cross-section
                // <d_psi_d_r> = (MU_0 / 2) * (ln(16 * r / sqrt(d_r^2 + d_z^2)) + 1 - (d_z / d_r) * atan(d_r / d_z))
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        let d_r: f64 = conductor_d_r[conductor_i_rz];
                        let d_z: f64 = conductor_d_z[conductor_i_rz];
                        g_d_psi_d_r_local[i_rz] =
                            MU_0 / 2.0 * ((16.0 * r[i_rz] / (d_r.powi(2) + d_z.powi(2)).sqrt()).ln() + 1.0 - (d_z / d_r) * (d_r / d_z).atan());
                    }
                }

                g_d_psi_d_r_local
            })
            .collect();

        let mut g_d_psi_d_r: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d_psi_d_r.slice_mut(s![.., conductor_i_rz]).assign(&g_d_psi_d_r_vec[conductor_i_rz]);
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
    /// * `g_d_psi_d_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"
    pub fn d_psi_d_z(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d_psi_d_z_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h: Array1<f64> = z - conductor_z[conductor_i_rz];
                let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d_psi_d_z = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * h / u
                // coeff_s = -2 * MU_0 * r * conductor_r * h / (u * d_sq)
                let mut g_d_psi_d_z_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * h[i_rz] / u[i_rz];
                    let coeff_s: f64 = -2.0 * MU_0 * r[i_rz] * rp * h[i_rz] / (u[i_rz] * d_sq[i_rz]);

                    g_d_psi_d_z_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: the kernel is odd in h, so the value is EXACTLY zero for any
                // z-symmetric cross-section
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        g_d_psi_d_z_local[i_rz] = 0.0;
                    }
                }

                g_d_psi_d_z_local
            })
            .collect();

        let mut g_d_psi_d_z: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d_psi_d_z.slice_mut(s![.., conductor_i_rz]).assign(&g_d_psi_d_z_vec[conductor_i_rz]);
        }

        g_d_psi_d_z
    }

    /// Calculates `g_d2_psi_d_r2`, where:
    /// `d2(psi)/d(r2) = g_d2_psi_d_r2 * current`
    ///
    /// Equation derived and verified with SymPy.
    ///
    /// Away from the conductor, psi satisfies the homogeneous Grad-Shafranov equation, so this
    /// closed form equals `d_psi_d_r / r - d2_psi_d_z2` (checked in `test_gs_identity`).
    /// That identity does NOT hold at the self-point, where the sourced equation
    /// `Delta* psi = -2 * PI * MU_0 * r * j` applies instead; the self-point uses its own
    /// closed form, including a share (`1/2 - XI`) of the hoop-field correction
    /// `d_psi_d_r / r` so that the sourced equation is satisfied (checked in
    /// `test_self_point_ampere`).
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d2_psi_d_r2[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d2_psi_d_r2(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let conductor_d_r: &Array1<f64> = &self.conductor_d_r;
        let conductor_d_z: &Array1<f64> = &self.conductor_d_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_r2_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h_sq: Array1<f64> = (z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d2_psi_d_r2 = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * (u_sq + d_sq) * h_sq / (2 * u^3 * d_sq)
                // coeff_s = 2 * MU_0 * conductor_r * (conductor_r * u_sq * d_sq - r * (2 * u_sq - d_sq) * h_sq) / (u^3 * d_sq^2)
                let mut g_d2_psi_d_r2_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let hs: f64 = h_sq[i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * (us + ds) * hs / (2.0 * u[i_rz] * us * ds);
                    let coeff_s: f64 = 2.0 * MU_0 * rp * (rp * us * ds - r[i_rz] * (2.0 * us - ds) * hs) / (u[i_rz] * us * ds * ds);

                    g_d2_psi_d_r2_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: dominated by the cell's own interior field, plus a share of the
                // hoop-field correction `d_psi_d_r / r` (split by XI) so that Ampere's law is satisfied
                // <d2_psi_d_r2> = -4 * MU_0 * r * atan(d_z / d_r) / (d_r * d_z) + (1/2 - XI) * <d_psi_d_r> / r
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        let d_r: f64 = conductor_d_r[conductor_i_rz];
                        let d_z: f64 = conductor_d_z[conductor_i_rz];
                        let d_psi_d_r_self: f64 =
                            MU_0 / 2.0 * ((16.0 * r[i_rz] / (d_r.powi(2) + d_z.powi(2)).sqrt()).ln() + 1.0 - (d_z / d_r) * (d_r / d_z).atan());
                        g_d2_psi_d_r2_local[i_rz] = -4.0 * MU_0 * r[i_rz] * (d_z / d_r).atan() / (d_r * d_z) + (0.5 - XI) * d_psi_d_r_self / r[i_rz];
                    }
                }

                g_d2_psi_d_r2_local
            })
            .collect();

        let mut g_d2_psi_d_r2: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d2_psi_d_r2.slice_mut(s![.., conductor_i_rz]).assign(&g_d2_psi_d_r2_vec[conductor_i_rz]);
        }

        g_d2_psi_d_r2
    }

    /// Calculates `g_d2_psi_d_r_d_z`, where:
    /// `d2(psi)/d(r)d(z) = g_d2_psi_d_r_d_z * current`
    ///
    /// Equation derived and verified with SymPy.
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d2_psi_d_r_d_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d2_psi_d_r_d_z(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_r_d_z_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h: Array1<f64> = z - conductor_z[conductor_i_rz];
                let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let w_sq: Array1<f64> = r.mapv(|x: f64| conductor_r[conductor_i_rz].powi(2) - x.powi(2)) - &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d2_psi_d_r_d_z = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * r * h * w_sq / (u^3 * d_sq)
                // coeff_s = 2 * MU_0 * r * conductor_r * h * (2 * u_sq * (r - conductor_r) - (r + conductor_r) * d_sq) / (u^3 * d_sq^2)
                let mut g_d2_psi_d_r_d_z_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * r[i_rz] * h[i_rz] * w_sq[i_rz] / (u[i_rz] * us * ds);
                    let coeff_s: f64 = 2.0 * MU_0 * r[i_rz] * rp * h[i_rz] * (2.0 * us * (r[i_rz] - rp) - (r[i_rz] + rp) * ds) / (u[i_rz] * us * ds * ds);

                    g_d2_psi_d_r_d_z_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: the kernel is odd in h, so the value is EXACTLY zero for any
                // z-symmetric cross-section
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        g_d2_psi_d_r_d_z_local[i_rz] = 0.0;
                    }
                }

                g_d2_psi_d_r_d_z_local
            })
            .collect();

        let mut g_d2_psi_d_r_d_z: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d2_psi_d_r_d_z.slice_mut(s![.., conductor_i_rz]).assign(&g_d2_psi_d_r_d_z_vec[conductor_i_rz]);
        }

        g_d2_psi_d_r_d_z
    }

    /// Calculates `g_d2_psi_d_z2`, where:
    /// `d2(psi)/d(z2) = g_d2_psi_d_z2 * current`
    ///
    /// Equation derived and verified with SymPy.
    ///
    /// # Arguments
    /// * None
    ///
    /// At the self-point the value is dominated by the cell's own interior field; the closed
    /// form for a rectangular cross-section is used, including a share (`1/2 + XI`) of the
    /// hoop-field correction `d_psi_d_r / r` so that the sourced Grad-Shafranov equation
    /// `Delta* psi = -2 * PI * MU_0 * r * j` is satisfied (checked in `test_self_point_ampere`).
    ///
    /// # Returns
    /// * `g_d2_psi_d_z2[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d2_psi_d_z2(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let conductor_d_r: &Array1<f64> = &self.conductor_d_r;
        let conductor_d_z: &Array1<f64> = &self.conductor_d_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d2_psi_d_z2_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h_sq: Array1<f64> = (z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d2_psi_d_z2 = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * (2 * u_sq * d_sq - (u_sq + d_sq) * h_sq) / (2 * u^3 * d_sq)
                // coeff_s = -2 * MU_0 * r * conductor_r * (u_sq * d_sq - (2 * u_sq - d_sq) * h_sq) / (u^3 * d_sq^2)
                let mut g_d2_psi_d_z2_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let hs: f64 = h_sq[i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * (2.0 * us * ds - (us + ds) * hs) / (2.0 * u[i_rz] * us * ds);
                    let coeff_s: f64 = -2.0 * MU_0 * r[i_rz] * rp * (us * ds - (2.0 * us - ds) * hs) / (u[i_rz] * us * ds * ds);

                    g_d2_psi_d_z2_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: dominated by the cell's own interior field, plus a share of the
                // hoop-field correction `d_psi_d_r / r` (split by XI) so that Ampere's law is satisfied
                // <d2_psi_d_z2> = -4 * MU_0 * r * atan(d_r / d_z) / (d_r * d_z) + (1/2 + XI) * <d_psi_d_r> / r
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        let d_r: f64 = conductor_d_r[conductor_i_rz];
                        let d_z: f64 = conductor_d_z[conductor_i_rz];
                        let d_psi_d_r_self: f64 =
                            MU_0 / 2.0 * ((16.0 * r[i_rz] / (d_r.powi(2) + d_z.powi(2)).sqrt()).ln() + 1.0 - (d_z / d_r) * (d_r / d_z).atan());
                        g_d2_psi_d_z2_local[i_rz] = -4.0 * MU_0 * r[i_rz] * (d_r / d_z).atan() / (d_r * d_z) + (0.5 + XI) * d_psi_d_r_self / r[i_rz];
                    }
                }

                g_d2_psi_d_z2_local
            })
            .collect();

        let mut g_d2_psi_d_z2: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d2_psi_d_z2.slice_mut(s![.., conductor_i_rz]).assign(&g_d2_psi_d_z2_vec[conductor_i_rz]);
        }

        g_d2_psi_d_z2
    }

    /// Calculates `g_d3_psi_d_z3`, where:
    /// `d3(psi)/d(z3) = g_d3_psi_d_z3 * current`
    ///
    /// Equation derived and verified with SymPy.
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d3_psi_d_z3[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d3_psi_d_z3(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d3_psi_d_z3_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h: Array1<f64> = z - conductor_z[conductor_i_rz];
                let h_sq: Array1<f64> = h.mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d3_psi_d_z3 = coeff_a * (K - E) + coeff_s * E
                // coeff_a = -MU_0 * h * (u_sq * d_sq * (3 * (u_sq + d_sq) + 10 * h_sq) - 4 * (u_sq + d_sq)^2 * h_sq) / (2 * u^5 * d_sq^2)
                // coeff_s = 2 * MU_0 * r * conductor_r * h * (3 * u_sq * d_sq * (2 * u_sq - d_sq) - (8 * u_sq^2 - u_sq * d_sq - 4 * d_sq^2) * h_sq) / (u^5 * d_sq^3)
                let mut g_d3_psi_d_z3_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let hs: f64 = h_sq[i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 =
                        -MU_0 * h[i_rz] * (us * ds * (3.0 * (us + ds) + 10.0 * hs) - 4.0 * (us + ds) * (us + ds) * hs) / (2.0 * u[i_rz] * us * us * ds * ds);
                    let coeff_s: f64 = 2.0 * MU_0 * r[i_rz] * rp * h[i_rz] * (3.0 * us * ds * (2.0 * us - ds) - (8.0 * us * us - us * ds - 4.0 * ds * ds) * hs)
                        / (u[i_rz] * us * us * ds * ds * ds);

                    g_d3_psi_d_z3_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Set to zero when conductor and sensor are at the same location
                // (d3_psi_d_z3 is odd in h, so zero is the symmetric value at the self-point)
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        g_d3_psi_d_z3_local[i_rz] = 0.0;
                    }
                }

                g_d3_psi_d_z3_local
            })
            .collect();

        let mut g_d3_psi_d_z3: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d3_psi_d_z3.slice_mut(s![.., conductor_i_rz]).assign(&g_d3_psi_d_z3_vec[conductor_i_rz]);
        }

        g_d3_psi_d_z3
    }

    /// Calculates `g_d3_psi_d_r2_d_z`, where:
    /// `d3(psi)/d(r2)d(z) = g_d3_psi_d_r2_d_z * current`
    ///
    /// Away from the conductor, psi satisfies the homogeneous Grad-Shafranov equation; its
    /// z-derivative gives `d3_psi_d_r2_d_z = d2_psi_d_r_d_z / r - d3_psi_d_z3`
    /// (proven with SymPy).
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d3_psi_d_r2_d_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d3_psi_d_r2_d_z(&self) -> Array2<f64> {
        let d2_psi_d_r_d_z: Array2<f64> = self.d2_psi_d_r_d_z();
        let d3_psi_d_z3: Array2<f64> = self.d3_psi_d_z3();
        let r: &Array1<f64> = &self.r;

        // At the self-point both inputs are exactly zero (kernels odd in h), and zero is also
        // the correct value for this table (its kernel is odd in h too). The
        // identity itself remains valid on the diagonal: it is the z-derivative of the sourced
        // Grad-Shafranov equation, and d(j)/d(z) = 0 for a uniform cell current density.
        let mut g_d3_psi_d_r2_d_z: Array2<f64> = Array2::from_elem((self.n_rz, self.conductor_n_rz), f64::NAN);
        for i_rz in 0..self.n_rz {
            for conductor_i_rz in 0..self.conductor_n_rz {
                g_d3_psi_d_r2_d_z[(i_rz, conductor_i_rz)] = d2_psi_d_r_d_z[(i_rz, conductor_i_rz)] / r[i_rz] - d3_psi_d_z3[(i_rz, conductor_i_rz)];
            }
        }

        g_d3_psi_d_r2_d_z
    }

    /// Calculates `g_d3_psi_d_r_d_z2`, where:
    /// `d3(psi)/d(r)d(z2) = g_d3_psi_d_r_d_z2 * current`
    ///
    /// Equation derived and verified with SymPy.
    ///
    /// This is the only third derivative whose self-point value is not zero by symmetry;
    /// the closed form for a rectangular cross-section is a pure toroidal-curvature (hoop)
    /// effect.
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_d3_psi_d_r_d_z2[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d3_psi_d_r_d_z2(&self) -> Array2<f64> {
        let n_rz: usize = self.n_rz;
        let conductor_n_rz: usize = self.conductor_n_rz;

        let r: &Array1<f64> = &self.r;
        let z: &Array1<f64> = &self.z;
        let conductor_r: &Array1<f64> = &self.conductor_r;
        let conductor_z: &Array1<f64> = &self.conductor_z;
        let conductor_d_r: &Array1<f64> = &self.conductor_d_r;
        let conductor_d_z: &Array1<f64> = &self.conductor_d_z;
        let elliptic_integral_k: ArrayView2<f64> = self.elliptic_integral_k.view();
        let elliptic_integral_e: ArrayView2<f64> = self.elliptic_integral_e.view();

        let g_d3_psi_d_r_d_z2_vec: Vec<Array1<f64>> = (0..conductor_n_rz)
            .into_par_iter()
            .map(|conductor_i_rz: usize| {
                let h_sq: Array1<f64> = (z - conductor_z[conductor_i_rz]).mapv(|x: f64| x.powi(2));
                let u_sq: Array1<f64> = (r + conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let u: Array1<f64> = u_sq.mapv(|x: f64| x.sqrt());
                let d_sq: Array1<f64> = (r - conductor_r[conductor_i_rz]).mapv(|x: f64| x.powi(2)) + &h_sq;
                let w_sq: Array1<f64> = r.mapv(|x: f64| conductor_r[conductor_i_rz].powi(2) - x.powi(2)) - &h_sq;

                let elliptic_integral_k_local: ArrayView1<f64> = elliptic_integral_k.slice(s![.., conductor_i_rz]);
                let elliptic_integral_e_local: ArrayView1<f64> = elliptic_integral_e.slice(s![.., conductor_i_rz]);

                // g_d3_psi_d_r_d_z2 = coeff_a * (K - E) + coeff_s * E
                // coeff_a = MU_0 * r * (u_sq * d_sq * w_sq - (5 * u_sq * d_sq + 4 * (u_sq + d_sq) * w_sq) * h_sq) / (u^5 * d_sq^2)
                // coeff_s = MU_0 * r * conductor_r * (conductor_r * (8 * u_sq^3 - 3 * u_sq^2 * d_sq + 4 * d_sq^3)
                //     - r * (8 * u_sq^3 - 3 * u_sq^2 * d_sq - 4 * d_sq^3)
                //     - r * (8 * u_sq^2 - u_sq * d_sq - 4 * d_sq^2) * (w_sq + 2 * h_sq)
                //     - conductor_r * (8 * u_sq^2 + 3 * u_sq * d_sq + 4 * d_sq^2) * w_sq) / (u^5 * d_sq^3)
                let mut g_d3_psi_d_r_d_z2_local: Array1<f64> = Array1::from_elem(n_rz, f64::NAN);
                for i_rz in 0..n_rz {
                    let rp: f64 = conductor_r[conductor_i_rz];
                    let us: f64 = u_sq[i_rz];
                    let ds: f64 = d_sq[i_rz];
                    let hs: f64 = h_sq[i_rz];
                    let ws: f64 = w_sq[i_rz];
                    let k_minus_e: f64 = elliptic_integral_k_local[i_rz] - elliptic_integral_e_local[i_rz];

                    let coeff_a: f64 = MU_0 * r[i_rz] * (us * ds * ws - (5.0 * us * ds + 4.0 * (us + ds) * ws) * hs) / (u[i_rz] * us * us * ds * ds);
                    let coeff_s: f64 = MU_0
                        * r[i_rz]
                        * rp
                        * (rp * (8.0 * us * us * us - 3.0 * us * us * ds + 4.0 * ds * ds * ds)
                            - r[i_rz] * (8.0 * us * us * us - 3.0 * us * us * ds - 4.0 * ds * ds * ds)
                            - r[i_rz] * (8.0 * us * us - us * ds - 4.0 * ds * ds) * (ws + 2.0 * hs)
                            - rp * (8.0 * us * us + 3.0 * us * ds + 4.0 * ds * ds) * ws)
                        / (u[i_rz] * us * us * ds * ds * ds);

                    g_d3_psi_d_r_d_z2_local[i_rz] = coeff_a * k_minus_e + coeff_s * elliptic_integral_e_local[i_rz];
                }

                // Self-point: pure toroidal-curvature (hoop) effect
                // <d3_psi_d_r_d_z2> = -MU_0 * (4 * atan(d_r / d_z) - 2 * d_r * d_z / (d_r^2 + d_z^2)) / (d_r * d_z)
                for i_rz in 0..n_rz {
                    if (r[i_rz] - conductor_r[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                        && (z[i_rz] - conductor_z[conductor_i_rz]).abs() < SELF_POINT_DISTANCE_TOLERANCE
                    {
                        let d_r: f64 = conductor_d_r[conductor_i_rz];
                        let d_z: f64 = conductor_d_z[conductor_i_rz];
                        g_d3_psi_d_r_d_z2_local[i_rz] = -MU_0 * (4.0 * (d_r / d_z).atan() - 2.0 * d_r * d_z / (d_r.powi(2) + d_z.powi(2))) / (d_r * d_z);
                    }
                }

                g_d3_psi_d_r_d_z2_local
            })
            .collect();

        let mut g_d3_psi_d_r_d_z2: Array2<f64> = Array2::from_elem((n_rz, conductor_n_rz), f64::NAN);
        for conductor_i_rz in 0..conductor_n_rz {
            g_d3_psi_d_r_d_z2.slice_mut(s![.., conductor_i_rz]).assign(&g_d3_psi_d_r_d_z2_vec[conductor_i_rz]);
        }

        g_d3_psi_d_r_d_z2
    }

    /// Calculates `b_r`, where:
    /// `b_r = -d(psi)/d(z) / (2.0 * PI * r)`
    ///
    /// # Arguments
    /// * None
    ///
    /// # Returns
    /// * `g_b_r[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn b_r(&self) -> Array2<f64> {
        let d_psi_d_z: Array2<f64> = self.d_psi_d_z();
        let r: &Array1<f64> = &self.r;

        let mut g_b_r: Array2<f64> = Array2::from_elem((self.n_rz, self.conductor_n_rz), f64::NAN);
        for i_rz in 0..self.n_rz {
            for conductor_i_rz in 0..self.conductor_n_rz {
                g_b_r[(i_rz, conductor_i_rz)] = -d_psi_d_z[(i_rz, conductor_i_rz)] / (2.0 * PI * r[i_rz]);
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
    /// * `g_b_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn b_z(&self) -> Array2<f64> {
        let d_psi_d_r: Array2<f64> = self.d_psi_d_r();
        let r: &Array1<f64> = &self.r;

        let mut g_b_z: Array2<f64> = Array2::from_elem((self.n_rz, self.conductor_n_rz), f64::NAN);
        for i_rz in 0..self.n_rz {
            for conductor_i_rz in 0..self.conductor_n_rz {
                g_b_z[(i_rz, conductor_i_rz)] = d_psi_d_r[(i_rz, conductor_i_rz)] / (2.0 * PI * r[i_rz]);
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
    /// * `g_d_b_r_d_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d_b_r_d_z(&self) -> Array2<f64> {
        let d2_psi_d_z2: Array2<f64> = self.d2_psi_d_z2();
        let r: &Array1<f64> = &self.r;

        let mut g_d_b_r_d_z: Array2<f64> = Array2::from_elem((self.n_rz, self.conductor_n_rz), f64::NAN);
        for i_rz in 0..self.n_rz {
            for conductor_i_rz in 0..self.conductor_n_rz {
                g_d_b_r_d_z[(i_rz, conductor_i_rz)] = -d2_psi_d_z2[(i_rz, conductor_i_rz)] / (2.0 * PI * r[i_rz]);
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
    /// * `g_d_b_z_d_z[(i_rz, conductor_i_rz)]` - The Greens table between "sensors" and "conductors"`
    pub fn d_b_z_d_z(&self) -> Array2<f64> {
        let d2_psi_d_r_d_z: Array2<f64> = self.d2_psi_d_r_d_z();
        let r: &Array1<f64> = &self.r;

        let mut g_d_b_z_d_z: Array2<f64> = Array2::from_elem((self.n_rz, self.conductor_n_rz), f64::NAN);
        for i_rz in 0..self.n_rz {
            for conductor_i_rz in 0..self.conductor_n_rz {
                g_d_b_z_d_z[(i_rz, conductor_i_rz)] = d2_psi_d_r_d_z[(i_rz, conductor_i_rz)] / (2.0 * PI * r[i_rz]);
            }
        }

        g_d_b_z_d_z
    }
}

/// Test the poloidal flux using a Helmholtz coil, which has an analytic solution
#[test]
fn test_psi() {
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;
    use std::f64::consts::PI;

    // Conductors
    // The radius of PF coil is "d", so that I'm consistent with Helmholtz notation / equations
    let current: f64 = 2.3456789;
    let d: f64 = 1.23456789;
    let conductor_r: Array1<f64> = Array1::from(vec![d, d]);
    let conductor_z: Array1<f64> = Array1::from(vec![-d / 2.0, d / 2.0]);
    let conductor_n_rz: usize = conductor_r.len();
    let conductor_d_r: Array1<f64> = Array1::zeros(conductor_n_rz);
    let conductor_d_z: Array1<f64> = Array1::zeros(conductor_n_rz);

    // Sensors
    // Define a sensor position
    let sensor_r: f64 = 0.12345;
    let r: Array1<f64> = Array1::from(vec![sensor_r]);
    let z: Array1<f64> = Array1::from(vec![0.00]);

    // Calculate flux
    let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let psi: Array2<f64> = greens_calculator.psi();
    let psi_numerical: Array1<f64> = psi.sum_axis(Axis(1)) * current;
    let psi_numerical: f64 = psi_numerical[0]; // since we have only one sensor

    fn psi_analytic_integrand(d: f64, r: f64) -> f64 {
        let integrand_value: f64 = (-2.0 * r - 5.0 * d) / (5.0 * d.powi(2) + 4.0 * d * r + 4.0 * r.powi(2)).sqrt()
            + (2.0 * r - 5.0 * d) / (5.0 * d.powi(2) - 4.0 * d * r + 4.0 * r.powi(2)).sqrt();

        integrand_value
    }

    // Calculate the analytic solution
    let psi_analytic: f64 = 2.0 * PI * MU_0 * (d / 4.0) * current * (psi_analytic_integrand(d, sensor_r) - psi_analytic_integrand(d, 0.0));

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
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d_psi_d_r analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d_psi_d_r_analytic: f64 = greens_calculator.d_psi_d_r()[(0, 0)];

    // Compute d_psi_d_r numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
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
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d_psi_d_z analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d_psi_d_z_analytic: f64 = greens_calculator.d_psi_d_z()[(0, 0)];

    // Compute d_psi_d_z numerically from psi
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
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
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_r2 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d2_psi_d_r2_analytic: f64 = greens_calculator.d2_psi_d_r2()[(0, 0)];

    // Compute d2_psi_d_r2 numerically from psi: (psi_left - 2*psi_center + psi_right) / delta_r^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value, z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
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
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_r_d_z analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d2_psi_d_r_d_z_analytic: f64 = greens_calculator.d2_psi_d_r_d_z()[(0, 0)];

    // Compute d2_psi_d_r_d_z numerically: d(d_psi_d_r)/d(z)
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
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
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d2_psi_d_z2 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d2_psi_d_z2_analytic: f64 = greens_calculator.d2_psi_d_z2()[(0, 0)];

    // Compute d2_psi_d_z2 numerically from psi: (psi_below - 2*psi_center + psi_above) / delta_z^2
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let psi: Array2<f64> = greens_calculator.psi();
    let d2_psi_d_z2_numerical: f64 = (psi[(0, 0)] - 2.0 * psi[(1, 0)] + psi[(2, 0)]) / delta_z.powi(2);

    assert_abs_diff_eq!(d2_psi_d_z2_analytic, d2_psi_d_z2_numerical, epsilon = 1e-10);
}

/// Test d3(psi)/d(r2)d(z) by numerically differentiating d2_psi_d_r2 w.r.t. z
#[test]
fn test_d3_psi_d_r2_d_z() {
    use approx::assert_abs_diff_eq;

    let delta_z: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d3_psi_d_r2_d_z analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d3_psi_d_r2_d_z_analytic: f64 = greens_calculator.d3_psi_d_r2_d_z()[(0, 0)];

    // Compute d3_psi_d_r2_d_z numerically: d(d2_psi_d_r2)/d(z)
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2();
    let d3_psi_d_r2_d_z_numerical: f64 = (d2_psi_d_r2[(1, 0)] - d2_psi_d_r2[(0, 0)]) / (2.0 * delta_z);

    assert_abs_diff_eq!(d3_psi_d_r2_d_z_analytic, d3_psi_d_r2_d_z_numerical, epsilon = 1e-10);
}

/// Test d3(psi)/d(r)d(z2) by numerically differentiating d2_psi_d_z2 w.r.t. r
#[test]
fn test_d3_psi_d_r_d_z2() {
    use approx::assert_abs_diff_eq;

    let delta_r: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d3_psi_d_r_d_z2 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d3_psi_d_r_d_z2_analytic: f64 = greens_calculator.d3_psi_d_r_d_z2()[(0, 0)];

    // Compute d3_psi_d_r_d_z2 numerically: d(d2_psi_d_z2)/d(r)
    let r_vec: Array1<f64> = Array1::from(vec![r_value - delta_r, r_value + delta_r]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value, z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2();
    let d3_psi_d_r_d_z2_numerical: f64 = (d2_psi_d_z2[(1, 0)] - d2_psi_d_z2[(0, 0)]) / (2.0 * delta_r);

    assert_abs_diff_eq!(d3_psi_d_r_d_z2_analytic, d3_psi_d_r_d_z2_numerical, epsilon = 1e-10);
}

/// Test d3(psi)/d(z3) by numerically differentiating d2_psi_d_z2 w.r.t. z
#[test]
fn test_d3_psi_d_z3() {
    use approx::assert_abs_diff_eq;

    let delta_z: f64 = 1e-4;
    let r_value: f64 = 1.852;
    let z_value: f64 = 0.12345;
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    // Compute d3_psi_d_z3 analytically
    let r: Array1<f64> = Array1::from(vec![r_value]);
    let z: Array1<f64> = Array1::from(vec![z_value]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r, z, conductor_r.clone(), conductor_z.clone(), conductor_d_r.clone(), conductor_d_z.clone());
    let d3_psi_d_z3_analytic: f64 = greens_calculator.d3_psi_d_z3()[(0, 0)];

    // Compute d3_psi_d_z3 numerically: d(d2_psi_d_z2)/d(z)
    let r_vec: Array1<f64> = Array1::from(vec![r_value, r_value]);
    let z_vec: Array1<f64> = Array1::from(vec![z_value - delta_z, z_value + delta_z]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r_vec, z_vec, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2();
    let d3_psi_d_z3_numerical: f64 = (d2_psi_d_z2[(1, 0)] - d2_psi_d_z2[(0, 0)]) / (2.0 * delta_z);

    assert_abs_diff_eq!(d3_psi_d_z3_analytic, d3_psi_d_z3_numerical, epsilon = 1e-10);
}

/// Source-free Ampere's law for the axisymmetric flux: away from the conductor the toroidal
/// current density is zero, so the Grad-Shafranov operator annihilates psi (`Delta* psi = 0`):
/// d2_psi_d_r2 - d_psi_d_r / r + d2_psi_d_z2 = 0, i.e. d2_psi_d_r2 = d_psi_d_r / r - d2_psi_d_z2.
/// (This becomes the Grad-Shafranov equation only when the current is a plasma equilibrium; here
/// the conductor is a bare filament, so it is just magnetostatics.)
#[test]
fn test_ampere_law_in_vacuum() {
    use approx::assert_abs_diff_eq;

    let r: Array1<f64> = Array1::from(vec![1.852, 0.31, 2.5, 0.9]);
    let z: Array1<f64> = Array1::from(vec![0.12345, 1.7, -0.9, 0.03]);
    let n_rz: usize = r.len();
    let conductor_r: Array1<f64> = Array1::from(vec![1.52345]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.8234]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z, conductor_r, conductor_z, conductor_d_r, conductor_d_z);
    let d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r();
    let d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2();
    let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2();

    for i_rz in 0..n_rz {
        let d2_psi_d_r2_from_ampere: f64 = d_psi_d_r[(i_rz, 0)] / r[i_rz] - d2_psi_d_z2[(i_rz, 0)];
        assert_abs_diff_eq!(d2_psi_d_r2[(i_rz, 0)], d2_psi_d_r2_from_ampere, epsilon = 1e-12);
    }
}

/// z-reflection parity of every Greens table.
///
/// To speed up the grid-to-grid Greens calculation and reduce memory usage, we exploit the symmetry
/// of the field produced by the conductor.
///
/// A single filament conductor sits on `z = 0`, with one sensor directly above it and one directly
/// below at the same radius. Reflecting the sensor `z -> -z` while leaving the conductor unchanged,
///
/// The parity is simply `(-1)^(number of z-derivatives)`: each `d/dz` flips the sign.
#[test]
fn test_z_reflection_parity() {
    use approx::assert_relative_eq;

    // Conductor on `z = 0` so the two sensors are exact mirror images (`h = +z0` and `h = -z0` exactly).
    let conductor_r: f64 = 0.9;
    let conductor_z: f64 = 0.0;
    let sensor_r: f64 = 0.65; // != conductor_r, so we are safely off the self-point
    let z0: f64 = 0.37; // generic and non-zero, so the odd tables are genuinely non-zero

    // sensor 0 = above (`h = +z0`), sensor 1 = below (`h = -z0`)
    let r: Array1<f64> = Array1::from(vec![sensor_r, sensor_r]);
    let z: Array1<f64> = Array1::from(vec![conductor_z + z0, conductor_z - z0]);
    let conductor_r: Array1<f64> = Array1::from(vec![conductor_r]);
    let conductor_z: Array1<f64> = Array1::from(vec![conductor_z]);
    let conductor_d_r: Array1<f64> = Array1::zeros(1);
    let conductor_d_z: Array1<f64> = Array1::zeros(1);

    let g: Greens = Greens::sensor_to_conductor(r, z, conductor_r, conductor_z, conductor_d_r, conductor_d_z);

    // `sign = +1` -> even (symmetric) in z; `sign = -1` -> odd (antisymmetric) in z.
    let check = |name: &str, table: Array2<f64>, sign: f64| {
        let above: f64 = table[(0, 0)];
        let below: f64 = table[(1, 0)];
        assert!(above != 0.0, "{name}: value is zero, this geometry does not exercise the table");
        assert_relative_eq!(above, sign * below, max_relative = 1e-12);
    };

    // Flux
    check("psi", g.psi(), 1.0); // Even in z

    // First derivatives
    check("d_psi_d_r", g.d_psi_d_r(), 1.0); // Even in z
    check("d_psi_d_z", g.d_psi_d_z(), -1.0); // Odd in z

    // Second derivatives
    check("d2_psi_d_r2", g.d2_psi_d_r2(), 1.0); // Even in z
    check("d2_psi_d_r_d_z", g.d2_psi_d_r_d_z(), -1.0); // Odd in z
    check("d2_psi_d_z2", g.d2_psi_d_z2(), 1.0); // Even in z

    // Third derivatives
    check("d3_psi_d_r2_d_z", g.d3_psi_d_r2_d_z(), -1.0); // Odd in z
    check("d3_psi_d_r_d_z2", g.d3_psi_d_r_d_z2(), 1.0); // Even in z
    check("d3_psi_d_z3", g.d3_psi_d_z3(), -1.0); // Odd in z

    // Fields
    check("b_r", g.b_r(), -1.0); // Odd in z
    check("b_z", g.b_z(), 1.0); // Even in z

    // Field derivatives
    check("d_b_r_d_z", g.d_b_r_d_z(), 1.0); // Even in z
    check("d_b_z_d_z", g.d_b_z_d_z(), -1.0); // Odd in z
}

/// Test the self-point values against high-precision quadrature of the exact kernels.
/// Ground-truth values computed with tanh-sinh quadrature of the exact filament kernels
/// over the cell cross-section (with mu_0 = 1).
/// The closed forms carry the thin-ring truncation error `O((delta/r)^2 * ln(r/delta))`,
/// hence the loose relative tolerance for `d_psi_d_r` and `d3_psi_d_r_d_z2`.
/// The second derivatives use a tighter tolerance: the hoop-field correction (split by `XI`)
/// reduces their largest observed residual from 2.1e-3 to 6.1e-5, and the tighter tolerance
/// would reject the uncorrected values.
#[test]
fn test_self_point_against_quadrature() {
    use approx::assert_relative_eq;

    // (r, d_r, d_z, then the quadrature values per unit current with mu_0 = 1 for:
    //  d_psi_d_r, d2_psi_d_r2, d2_psi_d_z2, d3_psi_d_r_d_z2)
    let test_cases: Vec<(f64, f64, f64, f64, f64, f64, f64)> = vec![
        (1.8, 0.04, 0.04, 3.22362302, -3533.68531, -3533.10729, -1338.77596),
        (1.8, 0.004, 0.004, 4.37493205, -353428.409, -353427.511, -133850.202),
        (0.9, 0.03, 0.06, 2.72073873, -2213.2044, -925.365218, -586.731493),
    ];

    for (r_value, d_r_value, d_z_value, d_psi_d_r_quadrature, d2_psi_d_r2_quadrature, d2_psi_d_z2_quadrature, d3_psi_d_r_d_z2_quadrature) in test_cases {
        let r: Array1<f64> = Array1::from(vec![r_value]);
        let z: Array1<f64> = Array1::from(vec![0.3]); // the self-point values are independent of z
        let d_r: Array1<f64> = Array1::from(vec![d_r_value]);
        let d_z: Array1<f64> = Array1::from(vec![d_z_value]);
        let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), r, z, d_r, d_z);

        assert_relative_eq!(greens_calculator.d_psi_d_r()[(0, 0)], MU_0 * d_psi_d_r_quadrature, max_relative = 1e-2);
        assert_relative_eq!(greens_calculator.d2_psi_d_r2()[(0, 0)], MU_0 * d2_psi_d_r2_quadrature, max_relative = 1e-3);
        assert_relative_eq!(greens_calculator.d2_psi_d_z2()[(0, 0)], MU_0 * d2_psi_d_z2_quadrature, max_relative = 1e-3);
        assert_relative_eq!(
            greens_calculator.d3_psi_d_r_d_z2()[(0, 0)],
            MU_0 * d3_psi_d_r_d_z2_quadrature,
            max_relative = 1e-2
        );
    }
}

/// Test the self-point flux against high-precision quadrature of the exact kernel.
/// Ground-truth values computed with tanh-sinh quadrature of the exact filament kernel
/// over the cell cross-section (with mu_0 = 1). The closed form carries the thin-ring
/// truncation error `O((delta/r)^2 * ln(r/delta))` (largest observed residual is 1.1e-4);
/// the tolerance is still tight enough to reject the self-INDUCTANCE (flux linkage)
/// value, which is ~5% lower than the flux at the cell centre.
#[test]
fn test_self_point_psi_against_quadrature() {
    use approx::assert_relative_eq;

    // (r, d_r, d_z, quadrature value of psi per unit current with mu_0 = 1)
    let test_cases: Vec<(f64, f64, f64, f64)> = vec![(1.8, 0.04, 0.04, 8.90523586), (0.9, 0.03, 0.06, 3.71615304), (0.41, 0.0125, 0.0125, 1.8987831)];

    for (r_value, d_r_value, d_z_value, psi_quadrature) in test_cases {
        let r: Array1<f64> = Array1::from(vec![r_value]);
        let z: Array1<f64> = Array1::from(vec![0.3]); // the self-point values are independent of z
        let d_r: Array1<f64> = Array1::from(vec![d_r_value]);
        let d_z: Array1<f64> = Array1::from(vec![d_z_value]);
        let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), r, z, d_r, d_z);

        assert_relative_eq!(greens_calculator.psi()[(0, 0)], MU_0 * psi_quadrature, max_relative = 1e-3);
    }
}

/// The self-point values satisfy the sourced Grad-Shafranov (Ampere's law) equation
/// identically:
/// `d2_psi_d_r2 - d_psi_d_r / r + d2_psi_d_z2 = -2 * PI * MU_0 * r / (d_r * d_z)`
/// because `atan(x) + atan(1/x) = PI / 2` and the hoop-field correction shares
/// (`1/2 - XI` and `1/2 + XI`) sum to exactly cancel the `d_psi_d_r / r` term;
/// this holds to machine precision, unlike the thin-ring approximation of the
/// individual entries.
#[test]
fn test_ampere_law_at_self_point() {
    use approx::assert_relative_eq;

    let r: Array1<f64> = Array1::from(vec![0.41]);
    let z: Array1<f64> = Array1::from(vec![0.0]);
    let d_r: Array1<f64> = Array1::from(vec![0.0125]);
    let d_z: Array1<f64> = Array1::from(vec![0.025]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), r.clone(), z, d_r.clone(), d_z.clone());

    let delta_star_psi: f64 = greens_calculator.d2_psi_d_r2()[(0, 0)] - greens_calculator.d_psi_d_r()[(0, 0)] / r[0] + greens_calculator.d2_psi_d_z2()[(0, 0)];
    let delta_star_psi_expected: f64 = -2.0 * PI * MU_0 * r[0] / (d_r[0] * d_z[0]);

    assert_relative_eq!(delta_star_psi, delta_star_psi_expected, max_relative = 1e-13);
}

/// Tables whose kernels are odd in `h = z - conductor_z` are exactly zero at the self-point
#[test]
fn test_self_point_odd_tables_are_zero() {
    let r: Array1<f64> = Array1::from(vec![0.41]);
    let z: Array1<f64> = Array1::from(vec![0.1]);
    let d_r: Array1<f64> = Array1::from(vec![0.0125]);
    let d_z: Array1<f64> = Array1::from(vec![0.0125]);
    let greens_calculator: Greens = Greens::sensor_to_conductor(r.clone(), z.clone(), r, z, d_r, d_z);

    assert_eq!(greens_calculator.d_psi_d_z()[(0, 0)], 0.0);
    assert_eq!(greens_calculator.d2_psi_d_r_d_z()[(0, 0)], 0.0);
    assert_eq!(greens_calculator.d3_psi_d_r2_d_z()[(0, 0)], 0.0);
    assert_eq!(greens_calculator.d3_psi_d_z3()[(0, 0)], 0.0);
}

/// Only the diagonal is special: for a table built over a grid onto itself, the off-diagonal
/// entries (sensor and conductor at different cells) must equal the plain filament values.
#[test]
fn test_self_field_off_diagonal_matches_sensor_to_conductor() {
    use approx::assert_abs_diff_eq;

    let grid_r: Array1<f64> = Array1::from(vec![0.4, 0.4125]);
    let grid_z: Array1<f64> = Array1::from(vec![0.0, 0.0]);
    let grid_d_r: Array1<f64> = Array1::from(vec![0.0125, 0.0125]);
    let grid_d_z: Array1<f64> = Array1::from(vec![0.0125, 0.0125]);
    let greens_self: Greens = Greens::sensor_to_conductor(grid_r.clone(), grid_z.clone(), grid_r, grid_z, grid_d_r, grid_d_z);

    let sensor_r: Array1<f64> = Array1::from(vec![0.4]);
    let sensor_z: Array1<f64> = Array1::from(vec![0.0]);
    let conductor_r: Array1<f64> = Array1::from(vec![0.4125]);
    let conductor_z: Array1<f64> = Array1::from(vec![0.0]);
    let conductor_d_r: Array1<f64> = Array1::from(vec![0.0125]);
    let conductor_d_z: Array1<f64> = Array1::from(vec![0.0125]);
    let greens_filament: Greens = Greens::sensor_to_conductor(sensor_r, sensor_z, conductor_r, conductor_z, conductor_d_r, conductor_d_z);

    assert_abs_diff_eq!(greens_self.d_psi_d_r()[(0, 1)], greens_filament.d_psi_d_r()[(0, 0)], epsilon = 1e-15);
    assert_abs_diff_eq!(greens_self.d_psi_d_z()[(0, 1)], greens_filament.d_psi_d_z()[(0, 0)], epsilon = 1e-15);
    assert_abs_diff_eq!(greens_self.d2_psi_d_r2()[(0, 1)], greens_filament.d2_psi_d_r2()[(0, 0)], epsilon = 1e-15);
    assert_abs_diff_eq!(greens_self.d2_psi_d_r_d_z()[(0, 1)], greens_filament.d2_psi_d_r_d_z()[(0, 0)], epsilon = 1e-15);
    assert_abs_diff_eq!(greens_self.d2_psi_d_z2()[(0, 1)], greens_filament.d2_psi_d_z2()[(0, 0)], epsilon = 1e-15);
    assert_abs_diff_eq!(
        greens_self.d3_psi_d_r_d_z2()[(0, 1)],
        greens_filament.d3_psi_d_r_d_z2()[(0, 0)],
        epsilon = 1e-15
    );
}

/// The self-point branch in `psi` fires only when the sensor coincides with the conductor in
/// *both* `r` and `z`, and it uses a strict `<` on `SELF_POINT_DISTANCE_TOLERANCE` (a point
/// exactly at the tolerance is NOT a self-point). Each case below builds a `sensor_to_conductor`
/// geometry that is just outside the self-region, and obtains the value the branch *would*
/// produce from a coincident 1x1 table (sensor placed at the conductor, whose diagonal is the self-term).
/// The assertion fails iff the self-term is wrongly applied, which pins down the `&&` and the
/// two `<` comparisons against being relaxed to `||` / `<=`.
#[test]
fn test_psi_self_point_detection() {
    // (1) `&&`, not `||`: same z but different r -> only one coordinate coincides -> filament.
    // Under `||` the shared z alone would (wrongly) trigger the self-term at the sensor radius.
    {
        let sensor_r: f64 = 0.4;
        let conductor_r: f64 = 0.4125;
        let z: f64 = 0.0;
        let d_r: Array1<f64> = Array1::from(vec![0.0125]);
        let d_z: Array1<f64> = Array1::from(vec![0.0125]);
        let g: Greens = Greens::sensor_to_conductor(
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            Array1::from(vec![conductor_r]),
            Array1::from(vec![z]),
            d_r.clone(),
            d_z.clone(),
        );
        let self_term: f64 = Greens::sensor_to_conductor(
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            d_r,
            d_z,
        )
        .psi()[(0, 0)];
        assert_ne!(g.psi()[(0, 0)], self_term, "psi used the self-term when only z coincided");
    }

    // (2) strict `<` on the z comparison: same r, z exactly `SELF_POINT_DISTANCE_TOLERANCE` apart -> filament.
    // The self-term is z-independent, so a coincident table at the same r reproduces what `<=` would give.
    {
        let r: f64 = 0.5;
        let conductor_z: f64 = 0.0;
        let sensor_z: f64 = SELF_POINT_DISTANCE_TOLERANCE; // exactly the tolerance from the conductor (conductor_z == 0.0)
        assert_eq!(
            (sensor_z - conductor_z).abs(),
            SELF_POINT_DISTANCE_TOLERANCE,
            "z offset must equal the tolerance exactly"
        );
        let d_r: Array1<f64> = Array1::from(vec![0.0125]);
        let d_z: Array1<f64> = Array1::from(vec![0.0125]);
        let g: Greens = Greens::sensor_to_conductor(
            Array1::from(vec![r]),
            Array1::from(vec![sensor_z]),
            Array1::from(vec![r]),
            Array1::from(vec![conductor_z]),
            d_r.clone(),
            d_z.clone(),
        );
        let self_term: f64 = Greens::sensor_to_conductor(
            Array1::from(vec![r]),
            Array1::from(vec![conductor_z]),
            Array1::from(vec![r]),
            Array1::from(vec![conductor_z]),
            d_r,
            d_z,
        )
        .psi()[(0, 0)];
        assert_ne!(g.psi()[(0, 0)], self_term, "psi used the self-term at z-distance == tolerance");
    }

    // (3) strict `<` on the r comparison: same z, r exactly `SELF_POINT_DISTANCE_TOLERANCE` apart -> filament.
    // The two radii are the tolerance and twice the tolerance, so their difference is *exactly* the
    // tolerance in f64 (`2x - x == x` is exact) and the test follows the constant if its value changes.
    // A coincident table at the sensor radius reproduces what `<=` would give.
    {
        let conductor_r: f64 = SELF_POINT_DISTANCE_TOLERANCE;
        let sensor_r: f64 = 2.0 * SELF_POINT_DISTANCE_TOLERANCE;
        let z: f64 = 0.0;
        assert_eq!(
            (sensor_r - conductor_r).abs(),
            SELF_POINT_DISTANCE_TOLERANCE,
            "r offset must equal the tolerance exactly"
        );
        let d_r: Array1<f64> = Array1::from(vec![SELF_POINT_DISTANCE_TOLERANCE / 10.0]);
        let d_z: Array1<f64> = Array1::from(vec![SELF_POINT_DISTANCE_TOLERANCE / 10.0]);
        let g: Greens = Greens::sensor_to_conductor(
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            Array1::from(vec![conductor_r]),
            Array1::from(vec![z]),
            d_r.clone(),
            d_z.clone(),
        );
        let self_term: f64 = Greens::sensor_to_conductor(
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            Array1::from(vec![sensor_r]),
            Array1::from(vec![z]),
            d_r,
            d_z,
        )
        .psi()[(0, 0)];
        assert_ne!(g.psi()[(0, 0)], self_term, "psi used the self-term at r-distance == tolerance");
    }
}
