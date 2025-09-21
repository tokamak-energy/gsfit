use ndarray::{Array1, Array2, Array3, s};

// Constants
const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct D2PsiDR2Calculator {
    g_d2_psi_d_r2_coils: Array3<f64>,
    pf_coil_currents: Array1<f64>,
    g_d2_psi_d_r2_passives: Array2<f64>,
    passive_dof_values: Array1<f64>,
    g_d2_psi_d_r2_plasma_3d: Array3<f64>,
    j_2d: Array2<f64>,
    d_area: f64,
    n_r: usize,
    n_z: usize,
    r: Array1<f64>,
    d_bz_d_z: Array2<f64>,
    delta_z: f64,
    g_bz_plasma_3d: Array3<f64>,
}

impl D2PsiDR2Calculator {
    /// Create a new D2PsiDR2Calculator
    ///
    /// # Arguments
    /// * `g_d2_psi_d_r2_coils` - Green's function table for d^2(psi)/d(r^2) from PF coil currents, shape (n_z, n_r, n_pf), weber^2 / (metre^2 * ampere)
    /// * `pf_coil_currents` - PF coil currents, shape=(n_pf), ampere
    /// * `g_d2_psi_d_r2_passives` - Green's function table for d^2(psi)/d(r^2) from passive currents, shape (n_z * n_r, n_passive_dof), weber^2 / (metre^2 * ampere)
    /// * `passive_dof_values` - Passive current degrees of freedom values, shape (n_passive_dof), ampere
    /// * `g_d2_psi_d_r2_plasma` - Green's function table for d^2(psi)/d(r^2) from plasma current, shape (n_z * n_r, n_r), weber^2 / (metre^2 * ampere)
    /// * `j_2d` - Plasma current density on grid, shape (n_z, n_r), ampere / metre^2
    /// * `d_area` - Area of each grid cell, metre^2
    /// * `n_r` - Number of R grid points, dimensionless
    /// * `n_z` - Number of Z grid points, dimensionless
    /// * `r` - R grid points, shape (n_r), metre
    /// * `g_bz_plasma` - Green's function table for Bz from plasma current, shape (n_z, n_r), tesla / (ampere / metre^2)
    /// * `delta_z` - Small perturbation to the Z-grid to stabilise the VDE, metre
    ///
    /// # Returns
    /// * `D2PsiDR2Calculator` - New D2PsiDR2Calculator
    ///
    pub fn new(
        g_d2_psi_d_r2_coils: &Array3<f64>,
        pf_coil_currents: &Array1<f64>,
        g_d2_psi_d_r2_passives: &Array2<f64>,
        passive_dof_values: &Array1<f64>,
        g_d2_psi_d_r2_plasma: &Array2<f64>,
        j_2d: &Array2<f64>,
        d_area: f64,
        n_r: usize,
        n_z: usize,
        r: &Array1<f64>,
        g_bz_plasma: &Array2<f64>,
        d_bz_d_z: &Array2<f64>,
        delta_z: f64,
    ) -> Self {
        // Store reshaped version for performance
        let g_d2_psi_d_r2_plasma_3d: Array3<f64> = g_d2_psi_d_r2_plasma
            .to_shape((n_z, n_r, n_r))
            .expect("d2_psi_d_r2_calculator: Failed to reshape into Array3")
            .to_owned();
        let (g_bz_plasma_flat, _): (Vec<f64>, Option<usize>) = g_bz_plasma.to_owned().into_raw_vec_and_offset();
        let g_bz_plasma_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_bz_plasma_flat).expect("Failed to reshape into Array3");
        Self {
            g_d2_psi_d_r2_coils: g_d2_psi_d_r2_coils.to_owned(),
            pf_coil_currents: pf_coil_currents.to_owned(),
            g_d2_psi_d_r2_passives: g_d2_psi_d_r2_passives.to_owned(),
            passive_dof_values: passive_dof_values.to_owned(),
            g_d2_psi_d_r2_plasma_3d,
            j_2d: j_2d.to_owned(),
            d_area,
            n_r,
            n_z,
            r: r.to_owned(),
            d_bz_d_z: d_bz_d_z.to_owned(),
            delta_z,
            g_bz_plasma_3d,
        }
    }

    /// Calculate d^2(psi)/d(r^2) at a given (r,z) location
    ///
    /// # Arguments
    /// * `i_r` - R index where to evaluate d^2(psi)/d(r^2), dimensionless
    /// * `i_z` - Z index where to evaluate d^2(psi)/d(r^2), dimensionless
    ///
    /// # Returns
    /// * `g_d2_psi_d_r2` - d^2(psi)/d(r^2) at (r,z), weber^2 / metre^2
    ///
    pub fn calculate(&self, i_r: usize, i_z: usize) -> f64 {
        // TODO: I am missing the delta_z term throughout this function entirely!

        // Get variables out of self
        let g_d2_psi_d_r2_coils: &Array3<f64> = &self.g_d2_psi_d_r2_coils;
        let pf_coil_currents: &Array1<f64> = &self.pf_coil_currents;
        let g_d2_psi_d_r2_passives: &Array2<f64> = &self.g_d2_psi_d_r2_passives;
        let passive_dof_values: &Array1<f64> = &self.passive_dof_values;
        let j_2d: &Array2<f64> = &self.j_2d;
        let d_area: f64 = self.d_area;
        let n_r: usize = self.n_r;
        let n_z: usize = self.n_z;
        let r: &Array1<f64> = &self.r;
        // Plasma
        let g_d2_psi_d_r2_plasma_3d: &Array3<f64> = &self.g_d2_psi_d_r2_plasma_3d;
        let g_bz_plasma_3d: &Array3<f64> = &self.g_bz_plasma_3d;

        // PF coils
        let d2_psi_d_r2_coils: f64 = g_d2_psi_d_r2_coils.slice(s![i_z, i_r, ..]).dot(pf_coil_currents);
        // let mut d2_psi_d_r2_coils: f64 = 0.0;
        // let n_pf: usize = pf_coil_currents.len();
        // for i_pf in 0..n_pf {
        //     d2_psi_d_r2_coils += g_d2_psi_d_r2_coils[(i_z, i_r, i_pf)] * pf_coil_currents[i_pf];
        // }

        // Passives
        let i_rz: usize = i_z * n_r + i_r;
        let d2_psi_d_r2_passives: f64 = g_d2_psi_d_r2_passives.slice(s![i_rz, ..]).dot(passive_dof_values);
        // let passives_shape: &[usize] = g_d2_psi_d_r2_passives.shape(); // shape = (n_z * n_r, n_passive_dof)
        // let n_passive_dof: usize = passives_shape[1];
        // let i_rz: usize = i_z * n_r + i_r;
        // let mut d2_psi_d_r2_passives: f64 = 0.0;
        // for i_passive_dof in 0..n_passive_dof {
        //     d2_psi_d_r2_passives += g_d2_psi_d_r2_passives[(i_rz, i_passive_dof)] * passive_dof_values[i_passive_dof];
        // }

        // Conceptually, we are looping over the current's and modifying the Green's table for the current
        let mut d2_psi_d_r2_plasma: f64 = 0.0;
        let mut bz_plasma_left: f64 = 0.0;
        let mut bz_plasma_right: f64 = 0.0;
        for i_cur_z in 0..n_z {
            // Cyclic indexing for the z-axis (a current filament "looks" the same in z, but not r)
            let z_indexer: Vec<usize> = (0..n_z).map(|i_z| i_cur_z.abs_diff(i_z)).collect();

            for i_cur_r in 0..n_r {
                // Performance improvement: a lot of the grid doesn't have plasma current
                if j_2d[(i_cur_z, i_cur_r)].abs() > 0.0 {
                    // Jump condition for self effect
                    // Note: this will be zero unless the grid point we are calculating at is carrying plasma current
                    if i_cur_r == i_r && i_cur_z == i_z {
                        let g_bz_plasma_left: f64 = g_bz_plasma_3d[(z_indexer[i_z], i_r - 1, i_cur_r)];
                        let g_bz_plasma_right: f64 = g_bz_plasma_3d[(z_indexer[i_z], i_r + 1, i_cur_r)];
                        bz_plasma_left += g_bz_plasma_left * j_2d[(i_cur_z, i_cur_r)] * d_area;
                        bz_plasma_right += g_bz_plasma_right * j_2d[(i_cur_z, i_cur_r)] * d_area;
                    } else {
                        // Select the Green's table for the radial current source location
                        // selecting the r-axis and re-ordering in one operation, might be fastest
                        let g_d2_psi_d_r2_plasma: f64 = g_d2_psi_d_r2_plasma_3d[(z_indexer[i_z], i_r, i_cur_r)];
                        // Calculate the contribution to psi from this current source
                        d2_psi_d_r2_plasma += g_d2_psi_d_r2_plasma * j_2d[(i_cur_z, i_cur_r)] * d_area;
                    }
                }
            }
        }

        // Add the "self" field derivative
        let d_r: f64 = r[1] - r[0];
        // Used chain rule:
        // psi = 2.0 * PI * r * bz
        // d2_psi_d_r2 = d(2.0 * PI * r * bz)/d(r)
        // d2_psi_d_r2 = 2.0 * PI * bz + 2.0 * PI * r * d(bz)/d(r)
        // bz at the self-point is zero. So:
        // d2_psi_d_r2 = 2.0 * PI * r * d(bz)/d(r)
        // bz = bz_original + delta_z * d_bz_d_z;  // Looks like I might be missing a term, but it should be small, because `delta_z` is small
        let d2_psi_d_r2_plasma_self_grid: f64 = PI * r[i_r] * (bz_plasma_right - bz_plasma_left) / d_r;
        d2_psi_d_r2_plasma += d2_psi_d_r2_plasma_self_grid;

        // Add up all the components
        let d2_psi_d_r2_unshifted: f64 = d2_psi_d_r2_coils + d2_psi_d_r2_passives + d2_psi_d_r2_plasma;

        // Add on the `delta_z` term
        let delta_z: f64 = self.delta_z;
        let d_bz_d_z: Array2<f64> = self.d_bz_d_z.to_owned();
        // TODO: protect against i_r at the edges !!!!!
        let d2_psi_d_r2_delta_z_term: f64;
        if i_r == 0 {
            // Forward difference at the left boundary
            d2_psi_d_r2_delta_z_term =
                2.0 * PI * delta_z * d_bz_d_z[(i_z, i_r)] + 2.0 * PI * r[i_r] * delta_z * (d_bz_d_z[(i_z, i_r + 1)] - d_bz_d_z[(i_z, i_r)]) / d_r
        } else if i_r == n_r - 1 {
            // Backward difference at the right boundary
            d2_psi_d_r2_delta_z_term =
                2.0 * PI * delta_z * d_bz_d_z[(i_z, i_r)] + 2.0 * PI * r[i_r] * delta_z * (d_bz_d_z[(i_z, i_r)] - d_bz_d_z[(i_z, i_r - 1)]) / d_r
        } else {
            // Central difference for interior points
            d2_psi_d_r2_delta_z_term =
                2.0 * PI * delta_z * d_bz_d_z[(i_z, i_r)] + PI * r[i_r] * delta_z * (d_bz_d_z[(i_z, i_r + 1)] - d_bz_d_z[(i_z, i_r - 1)]) / d_r
        };

        let d2_psi_d_r2: f64;
        if delta_z.is_finite() {
            d2_psi_d_r2 = d2_psi_d_r2_unshifted + d2_psi_d_r2_delta_z_term;
        } else {
            d2_psi_d_r2 = d2_psi_d_r2_unshifted;
        }

        return d2_psi_d_r2;
    }
}

#[test]
fn test_d2_psi_d_r2_calculator() {
    use crate::greens::greens_b::greens_b;
    use crate::greens::greens_d2_psi_dr2;
    use crate::greens::greens_psi::greens_psi;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    let n_r_scaling: usize = 2;
    let n_r: usize = 300 * n_r_scaling;
    let n_z: usize = 201;
    let n_pf: usize = 0;
    let n_passive_dof: usize = 0;
    let r_min: f64 = 0.1;
    let r_max: f64 = 1.0;
    let z_min: f64 = -2.0;
    let z_max: f64 = 1.0;

    // Create (r, z) grids
    let r: Array1<f64> = Array1::linspace(r_min, r_max, n_r);
    let z: Array1<f64> = Array1::linspace(z_min, z_max, n_z);

    // Grid spacing
    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];
    let d_area: f64 = d_r * d_z;

    // 2d (r, z) mesh
    let mut mesh_r: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
    let mut mesh_z: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            mesh_r[(i_z, i_r)] = r[i_r];
            mesh_z[(i_z, i_r)] = z[i_z];
        }
    }

    // Flatten 2d mesh
    let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
    let flat_z: Array1<f64> = mesh_z.flatten().to_owned();

    // PF coil
    let g_d2_psi_d_r2_coils: Array3<f64> = Array3::<f64>::zeros((n_z, n_r, n_pf));
    let pf_coil_currents: Array1<f64> = Array1::<f64>::zeros(n_pf);

    // Passive
    let g_d2_psi_d_r2_passives: Array2<f64> = Array2::zeros((n_z * n_r, n_passive_dof));
    let passive_dof_values: Array1<f64> = Array1::<f64>::zeros(n_passive_dof);

    // Plasma
    // d2_g_d_r2
    let mut g_d2_psi_d_r2_plasma: Array2<f64> = greens_d2_psi_dr2(flat_r.clone(), flat_z.clone(), r.clone(), 0.0 * r.clone() + z[0]);
    for i_r in 0..n_r {
        for i_rz in 0..n_r * n_z {
            if g_d2_psi_d_r2_plasma[(i_rz, i_r)].is_nan() {
                g_d2_psi_d_r2_plasma[(i_rz, i_r)] = 0.0; // TODO: this can be improved; avoiding "if" statement
            }
        }
    }

    let mut j_2d: Array2<f64> = Array2::<f64>::zeros((n_z, n_r));
    let d_bz_d_z: Array2<f64> = Array2::<f64>::zeros((n_z, n_r)); // Not used in this test

    // Create some test data
    let i_r_current_location: usize = 60;
    let i_z_current_location: usize = 90;
    j_2d[(i_z_current_location, i_r_current_location)] = 1.24716e9;
    j_2d[(i_z_current_location + 1, i_r_current_location)] = 1.2345e9;
    j_2d[(i_z_current_location + 1, i_r_current_location + 1)] = 0.2345e9;
    j_2d[(i_z_current_location + 1, i_r_current_location + 1)] = 0.2345e9;
    j_2d[(i_z_current_location + 1, i_r_current_location + 2)] = 0.2345e9;
    j_2d[(i_z_current_location - 1, i_r_current_location - 1)] = 0.36547;
    j_2d[(i_z_current_location - 1, i_r_current_location)] = 1.36547;
    j_2d[(i_z_current_location - 2, i_r_current_location - 1)] = 0.6547;
    let delta_z: f64 = 0.0;

    // Calculate Green's for `br` and `bz`
    let (_g_br, g_bz_plasma): (Array2<f64>, Array2<f64>) = greens_b(
        flat_r.clone(), // sensors
        flat_z.clone(),
        r.clone(), // current sources
        0.0 * r.clone() + z[0],
    );

    // Create new D2PsiDR2Calculator
    let d2_psi_d_r2_calculator: D2PsiDR2Calculator = D2PsiDR2Calculator::new(
        &g_d2_psi_d_r2_coils,
        &pf_coil_currents,
        &g_d2_psi_d_r2_passives,
        &passive_dof_values,
        &g_d2_psi_d_r2_plasma,
        &j_2d,
        d_area,
        n_r,
        n_z,
        &r,
        &g_bz_plasma,
        &d_bz_d_z,
        delta_z,
    );

    // Calculate `psi`
    // Calculate the grid-grid Greens
    let d_r_flat: Array1<f64> = &r * 0.0 + d_r;
    let d_z_flat: Array1<f64> = &r * 0.0 + d_z;
    let g_psi: Array2<f64> = greens_psi(
        flat_r.clone(),
        flat_z.clone(),
        r.clone(),
        0.0 * r.clone() + z[0],
        d_r_flat, // TODO: I don't like thsee variables
        d_z_flat,
    );
    let (g_grid_grid_flat, _): (Vec<f64>, Option<usize>) = g_psi.into_raw_vec_and_offset();
    let g_grid_grid_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_grid_grid_flat).expect("Failed to reshape into Array3");

    let mut psi_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));
    for i_cur_z in 0..n_z {
        // Cyclic indexing for the z-axis
        let z_indexer: Vec<usize> = (0..n_z).map(|i_z| i_cur_z.abs_diff(i_z)).collect();
        for i_cur_r in 0..n_r {
            // Performance improvement: a lot of the grid doesn't have plasma current
            if j_2d[(i_cur_z, i_cur_r)].abs() > 0.0 {
                // Select the Green's table for the radial current source location
                // selecting the r-axis and re-ordering in one operation, might be fastest
                let g_grid_grid_2d_reordered: Array2<f64> = g_grid_grid_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer);

                // Calculate the contribution to psi from this current source
                let psi_2d_plasma_this_j: Array2<f64> = g_grid_grid_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;

                // Add contribution from current
                psi_2d_plasma += &psi_2d_plasma_this_j;
            }
        }
    }
    let psi_2d: Array2<f64> = psi_2d_plasma;

    println!("current at i_r={}, i_z={}", i_r_current_location, i_z_current_location);

    let i_r_measure: usize = 50 * n_r_scaling;
    let i_z_measure: usize = 100;
    println!("Calculating at i_r_measure={}, i_z_measure={}", i_r_measure, i_z_measure);
    let d2_psi_d_r2: f64 = d2_psi_d_r2_calculator.calculate(i_r_measure, i_z_measure);
    println!("d2_psi_d_r2_analytic={:?}", d2_psi_d_r2);
    let d2_psi_d_r2_numerical_from_psi: f64 =
        (psi_2d[(i_z_measure, i_r_measure + 1)] - 2.0 * psi_2d[(i_z_measure, i_r_measure)] + psi_2d[(i_z_measure, i_r_measure - 1)]) / (d_r.powi(2));
    println!("d2_psi_d_r2_numerical_from_psi={:?}", d2_psi_d_r2_numerical_from_psi);

    assert_abs_diff_eq!(d2_psi_d_r2, d2_psi_d_r2_numerical_from_psi, epsilon = 1e-3);

    let i_r_measure: usize = 60 * n_r_scaling;
    let i_z_measure: usize = 90;
    println!("Calculating at i_r_measure={}, i_z_measure={}", i_r_measure, i_z_measure);
    let d2_psi_d_r2: f64 = d2_psi_d_r2_calculator.calculate(i_r_measure, i_z_measure);
    println!("d2_psi_d_r2_analytic={:?}", d2_psi_d_r2);
    let d2_psi_d_r2_numerical_from_psi: f64 =
        (psi_2d[(i_z_measure, i_r_measure + 1)] - 2.0 * psi_2d[(i_z_measure, i_r_measure)] + psi_2d[(i_z_measure, i_r_measure - 1)]) / (d_r.powi(2));
    println!("d2_psi_d_r2_numerical_from_psi={:?}", d2_psi_d_r2_numerical_from_psi);

    assert_abs_diff_eq!(d2_psi_d_r2, d2_psi_d_r2_numerical_from_psi, epsilon = 1e-3);
}
