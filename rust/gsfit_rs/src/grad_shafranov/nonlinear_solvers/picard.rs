//! Picard iteration.
//!
//! Each iteration calculates the flux from the current density, finds the magnetic axis and the
//! plasma boundary, fits the constraints, and calculates a new current density from the fitted
//! source functions. Writing the current density, passive currents and `delta_z` as `x`, this is
//! the fixed-point iteration `x -> G(x)`.
//!
//! It converges when `G` contracts about its fixed point, which is usual for a reconstruction
//! but not guaranteed: a forward solve, for example, is radially unstable. See `newton_krylov`,
//! which solves `G(x) - x = 0` instead.
//!
//! # Anderson acceleration
//! With `apply_anderson_mixing` on, each new state is not `G(x)` itself but a combination of the last few iterates,
//! chosen to cancel their residuals `F(x) = G(x) - x` as far as it can:
//! ```text
//!     x_{k+1} = x_k + beta * F_k - (dX + beta * dF) gamma,    gamma = argmin |F_k - dF gamma|
//! ```
//! where the columns of `dX` and `dF` are the differences between consecutive iterates and between
//! consecutive residuals, over the last `anderson_n_history` iterations, and `beta` is
//! `anderson_mixing`. On a linear problem this is equivalent to GMRES. It costs no extra flux
//! calculations, only a least-squares problem with `anderson_n_history` columns. With no history it
//! is `x_{k+1} = x_k + beta * F_k`, which is under-relaxed Picard, and Picard itself with `beta = 1`.
//!
//! The variables in `x` have very different units, so each block is divided by its typical size
//! (see `scale_for_state`) before the least-squares problem. The history is cleared when the
//! vertical feedback switches on, because that changes `G`. If the flux cannot be calculated from
//! an accelerated state, for example because it has no magnetic axis, the plain Picard update is
//! used instead and the history is cleared.
//!
//! H. F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations", SIAM J. Numer. Anal.,
//! 2011, https://doi.org/10.1137/10078356X

use crate::grad_shafranov::Error;
use crate::grad_shafranov::equilibrium_solve::EquilibriumSolver;
use ndarray::{Array1, Array2, s};
use std::collections::VecDeque;

/// Singular values of the residual differences below this fraction of the largest are left out of
/// the Anderson least-squares problem, so that nearly parallel differences do not produce a huge step
const ANDERSON_SINGULAR_VALUE_CUTOFF: f64 = 1.0e-10;

/// Solve with Picard iterations.
///
/// Converged when the Grad-Shafranov deviation, the change in the flux on the magnetic axis
/// between iterations, is below `code/numerics/grad_shafranov_deviation_tolerance`.
pub(crate) fn solve(solver: &mut EquilibriumSolver) {
    // Solver settings, supplied through `equilibrium.code`
    let n_iter_max: usize = solver.equilibrium_code.numerics.iterations.n_max as usize;
    let n_iter_min: usize = solver.equilibrium_code.numerics.iterations.n_min as usize;
    let grad_shafranov_deviation_tolerance: f64 = solver.equilibrium_code.numerics.grad_shafranov_deviation_tolerance;
    let n_iter_no_vertical_feedback: usize = solver.equilibrium_code.numerics.nonlinear_solver.picard.n_iter_no_vertical_feedback as usize;
    // Without Anderson mixing, the state is left exactly as the fit leaves it
    let use_anderson: bool = solver.equilibrium_code.numerics.nonlinear_solver.picard.apply_anderson_mixing != 0;
    let anderson_n_history: usize = solver.equilibrium_code.numerics.nonlinear_solver.picard.anderson_n_history as usize;
    let anderson_mixing: f64 = solver.equilibrium_code.numerics.nonlinear_solver.picard.anderson_mixing;

    // The passive currents and the initial current density
    if let Err(error) = solver.initialise() {
        solver.set_to_failed_time_slice(error);
        return;
    }

    let mut psi_a_previous: f64 = 0.0; // needed to calculate the Grad-Shafranov deviation
    let mut anderson: Option<Anderson> = None; // set up on the first iteration, from the initial state
    let mut picard_update: Option<Array1<f64>> = None; // `G(x)`, to fall back on if the accelerated state fails
    let mut vertical_feedback_previous: bool = false; // needed to clear the Anderson history when `G` changes

    // Iteration loop
    for i_iter in 0..n_iter_max {
        // The flux from the current density, and from it the magnetic axis and the boundary. If
        // that fails from an accelerated state, fall back on the plain Picard update
        if let Err(error) = solver.update_psi_and_geometry() {
            let Some(picard_update) = picard_update.take() else {
                solver.set_to_failed_time_slice(error);
                return;
            };
            set_state(solver, &picard_update);
            if let Some(anderson) = anderson.as_mut() {
                anderson.clear();
            }
            if let Err(error) = solver.update_psi_and_geometry() {
                solver.set_to_failed_time_slice(error);
                return;
            }
        }
        let psi_a: f64 = solver.time_slice.global_quantities.psi_magnetic_axis;
        let psi_b: f64 = solver.time_slice.boundary.psi;

        // Calculate the Grad-Shafranov deviation
        let grad_shafranov_deviation_value: f64 = EquilibriumSolver::calculate_grad_shafranov_deviation(psi_a, psi_b, psi_a_previous);
        solver.time_slice.convergence.grad_shafranov_deviation_value = grad_shafranov_deviation_value;
        psi_a_previous = psi_a; // needed to calculate the Grad-Shafranov deviation in the next iteration

        // Check for convergence
        if grad_shafranov_deviation_value < grad_shafranov_deviation_tolerance && i_iter > n_iter_min {
            solver.set_to_converged_time_slice(i_iter);
            return;
        }

        // Check if we have reached the maximum number of iterations
        if i_iter == n_iter_max - 1 {
            solver.set_to_failed_time_slice(Error::MaxIterReached);
            return;
        }

        // Fit the constraints, and calculate the new current density from the fitted source functions
        let vertical_feedback: bool = i_iter > n_iter_no_vertical_feedback;
        if !use_anderson {
            solver.fit_and_update_current(vertical_feedback);
            continue;
        }
        let x: Array1<f64> = get_state(solver);
        solver.fit_and_update_current(vertical_feedback);
        let g_x: Array1<f64> = get_state(solver);

        // Anderson acceleration: replace `G(x)` by the combination of the recent iterates
        let anderson: &mut Anderson = anderson.get_or_insert_with(|| Anderson::new(anderson_n_history, anderson_mixing, scale_for_state(solver, &x)));
        if vertical_feedback != vertical_feedback_previous {
            anderson.clear();
        }
        vertical_feedback_previous = vertical_feedback;
        let x_new: Array1<f64> = anderson.next_state(&x, &g_x);
        set_state(solver, &x_new);
        picard_update = Some(g_x);
    }
}

/// The history of Anderson acceleration, and the step it takes; see the module documentation
struct Anderson {
    /// The maximum number of previous iterations combined
    n_history: usize,
    /// `beta`, the fraction of the residual taken
    mixing: f64,
    /// What each variable in the state is divided by
    scale: Array1<f64>,
    /// The scaled states, oldest first, at most `n_history + 1` of them
    states: VecDeque<Array1<f64>>,
    /// The scaled residuals `F = G(x) - x` of `states`
    residuals: VecDeque<Array1<f64>>,
}

impl Anderson {
    fn new(n_history: usize, mixing: f64, scale: Array1<f64>) -> Self {
        Anderson {
            n_history,
            mixing,
            scale,
            states: VecDeque::with_capacity(n_history + 1),
            residuals: VecDeque::with_capacity(n_history + 1),
        }
    }

    /// Forget the history, for when `G` has changed
    fn clear(&mut self) {
        self.states.clear();
        self.residuals.clear();
    }

    /// The next state, from the state `x` and its Picard update `g_x`, both unscaled
    fn next_state(&mut self, x: &Array1<f64>, g_x: &Array1<f64>) -> Array1<f64> {
        let x_scaled: Array1<f64> = x / &self.scale;
        let f_scaled: Array1<f64> = (g_x - x) / &self.scale;
        self.states.push_back(x_scaled.clone());
        self.residuals.push_back(f_scaled.clone());
        if self.states.len() > self.n_history + 1 {
            self.states.pop_front();
            self.residuals.pop_front();
        }

        // `x + beta * F`, less the part of it the history can account for
        let mut x_new_scaled: Array1<f64> = &x_scaled + &(self.mixing * &f_scaled);
        let n_differences: usize = self.states.len() - 1;
        if n_differences > 0 {
            let d_f: Vec<Array1<f64>> = (0..n_differences)
                .map(|i_difference: usize| &self.residuals[i_difference + 1] - &self.residuals[i_difference])
                .collect();
            let gamma: Array1<f64> = least_squares(&d_f, &f_scaled);
            for i_difference in 0..n_differences {
                let d_x: Array1<f64> = &self.states[i_difference + 1] - &self.states[i_difference];
                x_new_scaled.scaled_add(-gamma[i_difference], &(&d_x + &(self.mixing * &d_f[i_difference])));
            }
        }
        x_new_scaled * &self.scale
    }
}

/// `gamma` minimising `|f - sum_i gamma_i * columns_i|`, from the singular value decomposition of
/// the columns, leaving out singular values below `ANDERSON_SINGULAR_VALUE_CUTOFF` times the largest
fn least_squares(columns: &[Array1<f64>], f: &Array1<f64>) -> Array1<f64> {
    let n_rows: usize = f.len();
    let n_columns: usize = columns.len();
    let mut gamma: Array1<f64> = Array1::zeros(n_columns);

    let matrix: faer::Mat<f64> = faer::Mat::from_fn(n_rows, n_columns, |i_row, i_column| columns[i_column][i_row]);
    let Ok(svd) = matrix.thin_svd() else {
        return gamma;
    };
    let u: faer::MatRef<'_, f64> = svd.U();
    let v: faer::MatRef<'_, f64> = svd.V();
    let singular_values: faer::ColRef<'_, f64> = svd.S().column_vector();
    let singular_value_max: f64 = (0..singular_values.nrows()).map(|i_singular| singular_values[i_singular]).fold(0.0, f64::max);
    for i_singular in 0..singular_values.nrows() {
        let singular_value: f64 = singular_values[i_singular];
        if singular_value.is_nan() || singular_value <= ANDERSON_SINGULAR_VALUE_CUTOFF * singular_value_max {
            continue;
        }
        let u_dot_f: f64 = (0..n_rows).map(|i_row| u[(i_row, i_singular)] * f[i_row]).sum();
        for i_column in 0..n_columns {
            gamma[i_column] += v[(i_column, i_singular)] * u_dot_f / singular_value;
        }
    }
    gamma
}

/// The state the flux is calculated from, unscaled: `j_phi` flattened row by row, the passive
/// degrees of freedom, then `delta_z`
fn get_state(solver: &EquilibriumSolver) -> Array1<f64> {
    let j_phi: &Array2<f64> = &solver.time_slice.profiles_2d[0].j_phi;
    let passive_dof_values: &Array1<f64> = &solver.passive_dof_values;
    let n_grid: usize = j_phi.len();
    let n_passive_dof: usize = passive_dof_values.len();

    let mut x: Array1<f64> = Array1::from_elem(n_grid + n_passive_dof + 1, f64::NAN);
    x.slice_mut(s![0..n_grid]).assign(&Array1::from_iter(j_phi.iter().copied()));
    x.slice_mut(s![n_grid..n_grid + n_passive_dof]).assign(passive_dof_values);
    // `delta_z` is NaN until the first fit sets it, which the flux calculation takes as no shift
    let delta_z: f64 = solver.time_slice.convergence.delta_z;
    x[n_grid + n_passive_dof] = if delta_z.is_nan() { 0.0 } else { delta_z };
    x
}

/// Put the unscaled state `x` back into the time-slice; the reverse of `get_state`
fn set_state(solver: &mut EquilibriumSolver, x: &Array1<f64>) {
    let n_r: usize = solver.equilibrium_code.grid.n_r as usize;
    let n_z: usize = solver.equilibrium_code.grid.n_z as usize;
    let n_grid: usize = n_z * n_r;
    let n_passive_dof: usize = solver.passive_dof_values.len();

    solver.time_slice.profiles_2d[0].j_phi = x.slice(s![0..n_grid]).to_shape((n_z, n_r)).unwrap().to_owned();
    solver.passive_dof_values = x.slice(s![n_grid..n_grid + n_passive_dof]).to_owned();
    solver.time_slice.convergence.delta_z = x[n_grid + n_passive_dof];
}

/// What each variable in the state is divided by, so that the blocks are comparable:
/// * `j_phi`: by the norm of `j_phi`, so a relative change of 1 across the whole plasma has norm 1
/// * the passive degrees of freedom: likewise, by their norm (or by 1 A if they are all zero)
/// * `delta_z`: by the height of a grid cell
///
/// Fixed at the start, from the initial guess.
fn scale_for_state(solver: &EquilibriumSolver, x: &Array1<f64>) -> Array1<f64> {
    let n_grid: usize = solver.time_slice.profiles_2d[0].j_phi.len();
    let n_passive_dof: usize = solver.passive_dof_values.len();
    let z: &Array1<f64> = &solver.time_slice.profiles_2d[0].grid.dim2;

    let j_phi_norm: f64 = norm(&x.slice(s![0..n_grid]).to_owned());
    let passive_norm: f64 = norm(&x.slice(s![n_grid..n_grid + n_passive_dof]).to_owned());

    let mut scale: Array1<f64> = Array1::from_elem(x.len(), f64::NAN);
    scale.slice_mut(s![0..n_grid]).fill(j_phi_norm);
    scale
        .slice_mut(s![n_grid..n_grid + n_passive_dof])
        .fill(if passive_norm > 0.0 { passive_norm } else { 1.0 });
    scale[n_grid + n_passive_dof] = z[1] - z[0];
    scale
}

/// The Euclidean norm
fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{Anderson, least_squares, norm};
    use ndarray::{Array1, Array2, array};

    /// A linear fixed point, `G(x) = M x + b`, whose solution is `x* = (I - M)^-1 b`
    fn linear_map(m: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
        m.dot(x) + b
    }

    /// Iterate `x -> G(x)`, accelerated, and return the residual `|G(x) - x|` after `n_iter` iterations
    fn anderson_residual(m: &Array2<f64>, b: &Array1<f64>, n_history: usize, mixing: f64, n_iter: usize) -> f64 {
        let mut anderson: Anderson = Anderson::new(n_history, mixing, Array1::ones(b.len()));
        let mut x: Array1<f64> = Array1::zeros(b.len());
        for _i_iter in 0..n_iter {
            let g_x: Array1<f64> = linear_map(m, b, &x);
            x = anderson.next_state(&x, &g_x);
        }
        norm(&(linear_map(m, b, &x) - &x))
    }

    fn test_problem(slowest_eigenvalue: f64) -> (Array2<f64>, Array1<f64>) {
        // Upper triangular, so the eigenvalues are the diagonal
        let m: Array2<f64> = array![[slowest_eigenvalue, 0.3, 0.1], [0.0, 0.5, -0.2], [0.0, 0.0, -0.4]];
        let b: Array1<f64> = array![1.0, -2.0, 0.5];
        (m, b)
    }

    /// Picard with an eigenvalue of 0.98 barely converges; with a full history Anderson solves the
    /// 3-variable linear problem within a few iterations, like GMRES
    #[test]
    fn anderson_accelerates_a_slow_linear_fixed_point() {
        let (m, b): (Array2<f64>, Array1<f64>) = test_problem(0.98);

        let picard_residual: f64 = anderson_residual(&m, &b, 0, 1.0, 10);
        let anderson_residual_value: f64 = anderson_residual(&m, &b, 3, 1.0, 10);

        assert!(picard_residual > 1.0e-2, "Picard residual {picard_residual}");
        assert!(anderson_residual_value < 1.0e-10, "Anderson residual {anderson_residual_value}");
    }

    /// Picard diverges when an eigenvalue is outside the unit circle; Anderson still converges on a
    /// linear problem
    #[test]
    fn anderson_converges_where_picard_diverges() {
        let (m, b): (Array2<f64>, Array1<f64>) = test_problem(1.5);

        let picard_residual: f64 = anderson_residual(&m, &b, 0, 1.0, 10);
        let anderson_residual_value: f64 = anderson_residual(&m, &b, 3, 1.0, 10);

        assert!(picard_residual > 1.0, "Picard residual {picard_residual}");
        assert!(anderson_residual_value < 1.0e-10, "Anderson residual {anderson_residual_value}");
    }

    /// With no history, the step is `x + beta * F`
    #[test]
    fn anderson_without_history_is_under_relaxed_picard() {
        let mut anderson: Anderson = Anderson::new(0, 0.25, array![2.0, 3.0]);
        let x: Array1<f64> = array![1.0, 1.0];
        let g_x: Array1<f64> = array![5.0, -3.0];
        let x_new: Array1<f64> = anderson.next_state(&x, &g_x);

        assert!(norm(&(&x_new - &array![2.0, 0.0])) < 1.0e-14, "x_new = {x_new}");
    }

    /// Nearly parallel columns do not produce a huge coefficient
    #[test]
    fn least_squares_ignores_a_negligible_singular_value() {
        let columns: Vec<Array1<f64>> = vec![array![1.0, 0.0, 0.0], array![1.0, 1.0e-14, 0.0]];
        let f: Array1<f64> = array![2.0, 1.0, 0.0];
        let gamma: Array1<f64> = least_squares(&columns, &f);

        assert!(gamma.iter().all(|value: &f64| value.abs() < 10.0), "gamma = {gamma}");
        // The first component is fitted: gamma_1 + gamma_2 = 2
        assert!((gamma[0] + gamma[1] - 2.0).abs() < 1.0e-10, "gamma = {gamma}");
    }
}
