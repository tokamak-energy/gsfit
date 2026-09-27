use crate::greens::Greens;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// The poloidal flux and its derivatives on the `(n_z, n_r)` grid, due to a set of current filaments.
///
/// The flux follows the IMAS convention: the total poloidal flux `2 * pi * R * A_phi` [weber], not
/// the flux per radian.
pub struct FluxOnGrid {
    /// [weber]
    pub psi: Array2<f64>,
    /// [weber / metre]
    pub d_psi_d_r: Array2<f64>,
    /// [weber / metre]
    pub d_psi_d_z: Array2<f64>,
    /// [weber / metre ** 2]
    pub d2_psi_d_r2: Array2<f64>,
    /// [weber / metre ** 2]
    pub d2_psi_d_r_d_z: Array2<f64>,
    /// [weber / metre ** 2]
    pub d2_psi_d_z2: Array2<f64>,
}

/// Result of fitting boundary PF-coil currents to reproduce the analytic equilibrium.
pub struct BoundaryCoilFit {
    pub coil_r: Array1<f64>,
    pub coil_z: Array1<f64>,
    pub coil_d_r: Array1<f64>,
    pub coil_d_z: Array1<f64>,
    pub coil_current: Array1<f64>,
    /// Total flux, plasma plus coils, which is valid in the vacuum as well as inside the plasma.
    /// The derivatives come from the Green's kernels, so they are accurate near the X-point and the
    /// magnetic axis, where a finite-difference gradient would be poor
    pub flux_total: FluxOnGrid,
    /// Flux from the fitted coils alone [weber]
    pub psi_coils: Array2<f64>,
}

/// Solve the symmetric-positive-definite system `a x = b` by Cholesky factorisation.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n_unknown: usize = b.len();
    let mut lower: Array2<f64> = Array2::zeros((n_unknown, n_unknown));
    for i_column in 0..n_unknown {
        let mut diagonal: f64 = a[(i_column, i_column)];
        for i_sum in 0..i_column {
            diagonal -= lower[(i_column, i_sum)] * lower[(i_column, i_sum)];
        }
        let l_diagonal: f64 = diagonal.max(0.0).sqrt();
        lower[(i_column, i_column)] = l_diagonal;
        for i_row in (i_column + 1)..n_unknown {
            let mut sum: f64 = a[(i_row, i_column)];
            for i_sum in 0..i_column {
                sum -= lower[(i_row, i_sum)] * lower[(i_column, i_sum)];
            }
            lower[(i_row, i_column)] = if l_diagonal > 0.0 { sum / l_diagonal } else { 0.0 };
        }
    }
    // Forward substitution: L y = b
    let mut y: Array1<f64> = Array1::zeros(n_unknown);
    for i_row in 0..n_unknown {
        let mut sum: f64 = b[i_row];
        for i_sum in 0..i_row {
            sum -= lower[(i_row, i_sum)] * y[i_sum];
        }
        y[i_row] = if lower[(i_row, i_row)] > 0.0 { sum / lower[(i_row, i_row)] } else { 0.0 };
    }
    // Back substitution: L^T x = y
    let mut x: Array1<f64> = Array1::zeros(n_unknown);
    for i_row in (0..n_unknown).rev() {
        let mut sum: f64 = y[i_row];
        for i_sum in (i_row + 1)..n_unknown {
            sum -= lower[(i_sum, i_row)] * x[i_sum];
        }
        x[i_row] = if lower[(i_row, i_row)] > 0.0 { sum / lower[(i_row, i_row)] } else { 0.0 };
    }
    x
}

/// Toroidal current density of the analytic plasma on the grid, zero outside the plasma.
///
/// `j_phi = 2 * pi * (R * p' + ff' / (mu_0 * R))`, with the derivatives taken with respect to the
/// total flux [weber]. For the GF profiles `p = p_axis * psi_hat ** 2` and
/// `f ** 2 = r_geo ** 2 * b_0 ** 2 * (1 + 2 * (d_b / b_0) * psi_hat ** 2)` this is
/// `j_phi = (4 * pi * psi_hat / psi_0) * (p_axis * R + r_geo ** 2 * b_0 * d_b / (mu_0 * R))`.
///
/// # Arguments
/// * `r` - grid radial axis [metre]
/// * `psi_hat_2d` - normalised flux, shape `(n_z, n_r)` [dimensionless]
/// * `mask_2d` - 1 inside the plasma and 0 outside, shape `(n_z, n_r)` [dimensionless]
/// * `psi_0` - the flux normalisation, `psi = psi_0 * psi_hat` [weber]
/// * `p_axis` - pressure normalisation [pascal]
/// * `r_geo` - geometric major radius [metre]
/// * `bt_vac_at_r_geo` - vacuum toroidal field at `r_geo` [tesla]
/// * `bt_diamagnetic_shift` - diamagnetic change in the toroidal field, `d_b` [tesla]
///
/// # Returns
/// * `j_phi_2d` - shape `(n_z, n_r)` [ampere / metre ** 2]
#[allow(clippy::too_many_arguments)]
pub fn plasma_current_density(
    r: &Array1<f64>,
    psi_hat_2d: &Array2<f64>,
    mask_2d: &Array2<f64>,
    psi_0: f64,
    p_axis: f64,
    r_geo: f64,
    bt_vac_at_r_geo: f64,
    bt_diamagnetic_shift: f64,
) -> Array2<f64> {
    let (n_z, n_r): (usize, usize) = psi_hat_2d.dim();
    let mut j_phi_2d: Array2<f64> = Array2::zeros((n_z, n_r));
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            if mask_2d[(i_z, i_r)] > 0.0 {
                let r_cell: f64 = r[i_r];
                j_phi_2d[(i_z, i_r)] = (4.0 * PI * psi_hat_2d[(i_z, i_r)] / psi_0)
                    * (p_axis * r_cell + r_geo * r_geo * bt_vac_at_r_geo * bt_diamagnetic_shift / (MU_0 * r_cell));
            }
        }
    }
    j_phi_2d
}

/// Plasma current filaments: one filament per grid cell inside the plasma, carrying `j_phi * d_r * d_z`.
/// Shared by the coil fit and the synthetic sensor evaluation, so both use the identical distribution.
///
/// # Returns
/// * `(cell_r, cell_z, cell_current)` - [metre], [metre], [ampere]
pub fn plasma_current_filaments(r: &Array1<f64>, z: &Array1<f64>, j_phi_2d: &Array2<f64>, mask_2d: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_area: f64 = (r[1] - r[0]) * (z[1] - z[0]);
    let mut cell_r_vec: Vec<f64> = Vec::new();
    let mut cell_z_vec: Vec<f64> = Vec::new();
    let mut cell_current_vec: Vec<f64> = Vec::new();
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            if mask_2d[(i_z, i_r)] > 0.0 {
                cell_r_vec.push(r[i_r]);
                cell_z_vec.push(z[i_z]);
                cell_current_vec.push(j_phi_2d[(i_z, i_r)] * d_area);
            }
        }
    }
    (Array1::from(cell_r_vec), Array1::from(cell_z_vec), Array1::from(cell_current_vec))
}

/// The flux and its derivatives on the grid due to `current` flowing in the conductors of `greens`.
///
/// # Arguments
/// * `greens` - Green's functions between the flattened grid (the "sensors") and the conductors
/// * `current` - current in each conductor [ampere]
/// * `n_z`, `n_r` - grid shape, which the flattened grid is reshaped back into
fn flux_on_grid(greens: &Greens, current: &Array1<f64>, n_z: usize, n_r: usize) -> FluxOnGrid {
    // The grid is flattened row-major, `i_z * n_r + i_r`, so it reshapes straight back to `(n_z, n_r)`
    let unflatten = |flat: Array1<f64>| -> Array2<f64> {
        flat.into_shape_with_order((n_z, n_r))
            .expect("guazzotto_freidberg.flux_on_grid: cannot reshape the flattened grid into (n_z, n_r)")
    };
    // One `let` per table: a temporary lives until the end of its statement, so building all six
    // inside one struct literal would hold every `(n_grid, n_conductor)` Green's table at once
    let psi: Array2<f64> = unflatten(greens.psi().dot(current));
    let d_psi_d_r: Array2<f64> = unflatten(greens.d_psi_d_r().dot(current));
    let d_psi_d_z: Array2<f64> = unflatten(greens.d_psi_d_z().dot(current));
    let d2_psi_d_r2: Array2<f64> = unflatten(greens.d2_psi_d_r2().dot(current));
    let d2_psi_d_r_d_z: Array2<f64> = unflatten(greens.d2_psi_d_r_d_z().dot(current));
    let d2_psi_d_z2: Array2<f64> = unflatten(greens.d2_psi_d_z2().dot(current));
    FluxOnGrid {
        psi,
        d_psi_d_r,
        d_psi_d_z,
        d2_psi_d_r2,
        d2_psi_d_r_d_z,
        d2_psi_d_z2,
    }
}

/// Fit filament PF-coil currents on the rectangular grid boundary so that the coils' vacuum field,
/// added to the plasma's own field, reproduces the analytic flux at the control points
/// (see the Guazzotto–Freidberg PF-coil section). Returns the coil geometry, the fitted currents,
/// and the resulting total flux over the whole grid.
///
/// # Arguments
/// * `r`, `z` - grid axes [metre]
/// * `j_phi_2d` - plasma toroidal current density, shape `(n_z, n_r)` [ampere / metre ** 2]
/// * `mask_2d` - 1 inside the plasma and 0 outside, shape `(n_z, n_r)` [dimensionless]
/// * `control_r`, `control_z` - where the flux is matched [metre]
/// * `psi_control` - target flux at the control points, 0 on the LCFS [weber]
/// * `regularisation_weight` - Tikhonov weight `lambda ** 2` on the coil-current magnitude [dimensionless]
#[allow(clippy::too_many_arguments)]
pub fn fit_boundary_coils(
    r: &Array1<f64>,
    z: &Array1<f64>,
    j_phi_2d: &Array2<f64>,
    mask_2d: &Array2<f64>,
    control_r: &Array1<f64>,
    control_z: &Array1<f64>,
    psi_control: &Array1<f64>,
    regularisation_weight: f64,
) -> BoundaryCoilFit {
    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let d_r_cell: f64 = r[1] - r[0];
    let d_z_cell: f64 = z[1] - z[0];

    // 1. Plasma current source: a filament at each grid cell inside the plasma (shared helper, so the
    //    synthetic sensor evaluation in get_sensor_values uses the identical current distribution).
    let (cell_r, cell_z, cell_current): (Array1<f64>, Array1<f64>, Array1<f64>) = plasma_current_filaments(r, z, j_phi_2d, mask_2d);
    let n_cell: usize = cell_r.len();
    let cell_d_r: Array1<f64> = Array1::from_elem(n_cell, d_r_cell);
    let cell_d_z: Array1<f64> = Array1::from_elem(n_cell, d_z_cell);

    // 2. Coil filaments on the grid-boundary perimeter (one per boundary node).
    let mut coil_r_vec: Vec<f64> = Vec::new();
    let mut coil_z_vec: Vec<f64> = Vec::new();
    for i_r in 0..n_r {
        coil_r_vec.push(r[i_r]);
        coil_z_vec.push(z[0]);
    }
    for i_r in 0..n_r {
        coil_r_vec.push(r[i_r]);
        coil_z_vec.push(z[n_z - 1]);
    }
    for i_z in 1..(n_z - 1) {
        coil_r_vec.push(r[0]);
        coil_z_vec.push(z[i_z]);
    }
    for i_z in 1..(n_z - 1) {
        coil_r_vec.push(r[n_r - 1]);
        coil_z_vec.push(z[i_z]);
    }
    let n_coil: usize = coil_r_vec.len();
    let coil_r: Array1<f64> = Array1::from(coil_r_vec);
    let coil_z: Array1<f64> = Array1::from(coil_z_vec);
    let coil_d_r: Array1<f64> = Array1::from_elem(n_coil, d_r_cell);
    let coil_d_z: Array1<f64> = Array1::from_elem(n_coil, d_z_cell);

    // 3. Green's tables and the plasma self-flux at the control points.
    let m_control_coil: Array2<f64> = Greens::sensor_to_conductor(
        control_r.clone(),
        control_z.clone(),
        coil_r.clone(),
        coil_z.clone(),
        coil_d_r.clone(),
        coil_d_z.clone(),
    )
    .psi(); // (n_control, n_coil)
    let psi_plasma_control: Array1<f64> = Greens::sensor_to_conductor(
        control_r.clone(),
        control_z.clone(),
        cell_r.clone(),
        cell_z.clone(),
        cell_d_r.clone(),
        cell_d_z.clone(),
    )
    .psi()
    .dot(&cell_current);

    // 4. Regularised least squares: (M^T M + lambda^2 I) I = M^T (psi_control - psi_plasma).
    //    The weight is scaled by the mean diagonal of M^T M so the caller's
    //    `regularisation_weight` is a dimensionless knob relative to the data scale.
    let b_control: Array1<f64> = psi_control - &psi_plasma_control;
    let mut a_matrix: Array2<f64> = m_control_coil.t().dot(&m_control_coil);
    let mut reg_scale: f64 = 0.0;
    for i_coil in 0..n_coil {
        reg_scale += a_matrix[(i_coil, i_coil)];
    }
    reg_scale /= n_coil as f64;
    let lambda_sq: f64 = regularisation_weight * reg_scale;
    for i_coil in 0..n_coil {
        a_matrix[(i_coil, i_coil)] += lambda_sq;
    }
    let rhs: Array1<f64> = m_control_coil.t().dot(&b_control);
    let coil_current: Array1<f64> = cholesky_solve(&a_matrix, &rhs);

    // 5. Total flux and its derivatives on the whole grid = plasma self-field + fitted coil field
    //    (fills the vacuum).
    let n_grid: usize = n_z * n_r;
    let mut grid_r_flat: Array1<f64> = Array1::zeros(n_grid);
    let mut grid_z_flat: Array1<f64> = Array1::zeros(n_grid);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            grid_r_flat[i_z * n_r + i_r] = r[i_r];
            grid_z_flat[i_z * n_r + i_r] = z[i_z];
        }
    }
    // Each `Greens` holds its elliptic integrals, `(n_grid, n_conductor)` twice over, so the plasma's is
    // dropped before the coils' is built
    let flux_plasma: FluxOnGrid = {
        let greens_grid_cell: Greens = Greens::sensor_to_conductor(grid_r_flat.clone(), grid_z_flat.clone(), cell_r, cell_z, cell_d_r, cell_d_z);
        flux_on_grid(&greens_grid_cell, &cell_current, n_z, n_r)
    };
    let flux_coils: FluxOnGrid = {
        let greens_grid_coil: Greens = Greens::sensor_to_conductor(grid_r_flat, grid_z_flat, coil_r.clone(), coil_z.clone(), coil_d_r.clone(), coil_d_z.clone());
        flux_on_grid(&greens_grid_coil, &coil_current, n_z, n_r)
    };
    let flux_total: FluxOnGrid = FluxOnGrid {
        psi: &flux_plasma.psi + &flux_coils.psi,
        d_psi_d_r: flux_plasma.d_psi_d_r + &flux_coils.d_psi_d_r,
        d_psi_d_z: flux_plasma.d_psi_d_z + &flux_coils.d_psi_d_z,
        d2_psi_d_r2: flux_plasma.d2_psi_d_r2 + &flux_coils.d2_psi_d_r2,
        d2_psi_d_r_d_z: flux_plasma.d2_psi_d_r_d_z + &flux_coils.d2_psi_d_r_d_z,
        d2_psi_d_z2: flux_plasma.d2_psi_d_z2 + &flux_coils.d2_psi_d_z2,
    };

    BoundaryCoilFit {
        coil_r,
        coil_z,
        coil_d_r,
        coil_d_z,
        coil_current,
        flux_total,
        psi_coils: flux_coils.psi,
    }
}
