//! Jacobian-free Newton-Krylov iteration.
//!
//! A Picard iteration is the fixed-point iteration `x -> G(x)`, where `x` is everything the flux
//! is calculated from: the current density and the passive degrees of freedom. Picard converges
//! only when `G` contracts about its fixed point, and slowly when it only just does. This instead
//! solves
//! ```text
//!     F(x) = G(x) - x = 0
//! ```
//! with Newton's method. Each Newton step solves `J dx = -F(x)` for the step `dx`, where `J` is the
//! Jacobian of `F`, using GMRES. GMRES only needs `J` multiplied onto a vector, which is a finite
//! difference of `G`, so `J` is never formed:
//! ```text
//!     J v = (F(x + h v) - F(x)) / h
//! ```
//! Each of these costs one evaluation of `G`, which is one Picard iteration's worth of work.
//! Because `G` is the Picard update, only the few directions in which it does not contract need
//! resolving, and GMRES finds them in a handful of directions. Those are also the directions in
//! which Picard is slow or unstable, which is where this is faster, or converges where Picard
//! does not; when Picard converges quickly, as it usually does for a reconstruction, Picard is
//! cheaper.
//!
//! The iteration:
//! 1. Picard iterations, with the Picard settings, until the Grad-Shafranov deviation is below
//!    `picard_handover`. Newton's method needs a starting point close to the solution
//! 2. Newton iterations. The finite-difference step `h` is `finite_difference_step` times the
//!    size of the residual `F`, as FreeGSNKE does, so that it shrinks as the solution is
//!    approached: while far away, the step is large enough that `G` is not dominated by the
//!    discrete parts of the boundary search. A Newton step is halved until it reduces `F`, and if
//!    it still does not, a Picard step is taken instead
//!
//! Convergence is tested the same way as for Picard: the deviation is the change in the flux on
//! the magnetic axis that one more Picard iteration would make, relative to `psi_b - psi_a`. On
//! convergence the time-slice is left exactly as Picard leaves it.
//!
//! The Newton iterations do not use `delta_z`. Picard needs it because the plasma is vertically
//! unstable, which makes its fixed point unstable too: `delta_z` shifts the flux vertically, and
//! fitting it (normally to a magnetic axis constraint) holds the plasma in place. Newton converges
//! to an unstable fixed point without help, so `delta_z` is 0 and the flux is the true flux from
//! the coils, passives and plasma. That matters because the shift moves the whole flux, the coils'
//! included, so a converged Picard solution with `delta_z` other than 0 is not quite a
//! Grad-Shafranov equilibrium. It also means that a magnetic axis constraint (`StationaryPoint`)
//! is fitted by the other degrees of freedom instead, so a forward solve should leave it out: the
//! coil currents alone then set the vertical position.
//!
//! The variables in `x` have very different units, so each block is divided by its typical size
//! (see `scale_for_state`) before any norm is taken.
//!
//! D. A. Knoll and D. E. Keyes, "Jacobian-free Newton-Krylov methods: a survey of approaches and
//! applications", J. Comput. Phys., 2004, https://doi.org/10.1016/j.jcp.2003.08.010

use crate::grad_shafranov::Error;
use crate::grad_shafranov::equilibrium_solve::EquilibriumSolver;
use ndarray::{Array1, Array2, s};

/// The number of times a Newton step is halved before it is given up on, and a Picard step taken
/// instead
const N_STEP_HALVINGS_MAX: usize = 5;

/// The number of times the finite-difference step is halved when `G` cannot be evaluated at the
/// perturbed point, for example because the perturbation has removed the magnetic axis
const N_FINITE_DIFFERENCE_HALVINGS_MAX: usize = 3;

/// The smallest finite-difference step, in the scaled units of `scale_for_state`. Below this the
/// Jacobian-vector product would be dominated by rounding in `G`
const FINITE_DIFFERENCE_STEP_MIN: f64 = 1.0e-7;

/// One evaluation of the Picard update, `G`
struct PicardUpdate {
    /// `G(x)`, scaled
    x_new: Array1<f64>,
    /// The flux on the magnetic axis, calculated from `x`
    psi_a: f64,
    /// The magnetic axis, calculated from `x`, from which the next search starts
    magnetic_axis: (f64, f64),
}

/// The settings of the Newton iterations, from `code/numerics/nonlinear_solver/newton_krylov`
struct NewtonKrylovSettings {
    picard_handover: f64,
    n_krylov_max: usize,
    krylov_tolerance: f64,
    finite_difference_step: f64,
    verbose: bool,
}

/// Solve with the Jacobian-free Newton-Krylov method; see the module documentation.
pub(crate) fn solve(solver: &mut EquilibriumSolver) {
    // Solver settings, supplied through `equilibrium.code`
    let n_iter_max: usize = solver.equilibrium_code.numerics.iterations.n_max as usize;
    let n_iter_min: usize = solver.equilibrium_code.numerics.iterations.n_min as usize;
    let grad_shafranov_deviation_tolerance: f64 = solver.equilibrium_code.numerics.grad_shafranov_deviation_tolerance;
    let n_iter_no_vertical_feedback: usize = solver.equilibrium_code.numerics.nonlinear_solver.picard.n_iter_no_vertical_feedback as usize;
    let settings: NewtonKrylovSettings = NewtonKrylovSettings {
        picard_handover: solver.equilibrium_code.numerics.nonlinear_solver.newton_krylov.picard_handover,
        n_krylov_max: solver.equilibrium_code.numerics.nonlinear_solver.newton_krylov.n_krylov_max as usize,
        krylov_tolerance: solver.equilibrium_code.numerics.nonlinear_solver.newton_krylov.krylov_tolerance,
        finite_difference_step: solver.equilibrium_code.numerics.nonlinear_solver.newton_krylov.finite_difference_step,
        verbose: solver.equilibrium_code.numerics.nonlinear_solver.newton_krylov.verbose != 0,
    };

    // The passive currents and the initial current density
    if let Err(error) = solver.initialise() {
        solver.set_to_failed_time_slice(error);
        return;
    }

    // The number of times the flux has been calculated, which is where nearly all the time goes.
    // A Picard iteration calculates it once
    let mut n_flux_evaluations: usize = 0;

    // ==================================================================================
    // 1. Picard iterations, until the deviation is small enough to hand over to Newton
    // ==================================================================================
    let mut psi_a_previous: f64 = 0.0; // needed to calculate the Grad-Shafranov deviation
    let mut i_iter_handover: usize = n_iter_max;
    for i_iter in 0..n_iter_max {
        // The flux from the current density, and from it the magnetic axis and the boundary
        n_flux_evaluations += 1;
        if let Err(error) = solver.update_psi_and_geometry() {
            solver.set_to_failed_time_slice(error);
            return;
        }
        let psi_a: f64 = solver.time_slice.global_quantities.psi_magnetic_axis;
        let psi_b: f64 = solver.time_slice.boundary.psi;

        // Calculate the Grad-Shafranov deviation
        let grad_shafranov_deviation_value: f64 = EquilibriumSolver::calculate_grad_shafranov_deviation(psi_a, psi_b, psi_a_previous);
        solver.time_slice.convergence.grad_shafranov_deviation_value = grad_shafranov_deviation_value;
        psi_a_previous = psi_a;

        // Check for convergence
        if grad_shafranov_deviation_value < grad_shafranov_deviation_tolerance && i_iter > n_iter_min {
            report(&settings, "converged during the Picard iterations", i_iter, n_flux_evaluations);
            solver.set_to_converged_time_slice(i_iter);
            return;
        }

        // Check if we have reached the maximum number of iterations
        if i_iter == n_iter_max - 1 {
            report(&settings, "reached the maximum number of iterations", i_iter, n_flux_evaluations);
            solver.set_to_failed_time_slice(Error::MaxIterReached);
            return;
        }

        // Hand over to Newton
        if grad_shafranov_deviation_value < settings.picard_handover {
            i_iter_handover = i_iter;
            if settings.verbose {
                println!(
                    "newton_krylov: handing over from Picard to Newton at iteration {i_iter}; grad_shafranov_deviation = {grad_shafranov_deviation_value:.3e}"
                );
            }
            break;
        }

        // Fit the constraints, and calculate the new current density from the fitted source functions
        solver.fit_and_update_current(i_iter > n_iter_no_vertical_feedback);
    }

    // ==================================================================================
    // 2. Newton iterations
    // ==================================================================================
    // `G` at the starting point. The Picard iterations calculated the flux there with their
    // `delta_z` shift, so it is calculated again, without
    let x_unscaled: Array1<f64> = get_state(solver);
    let scale: Array1<f64> = scale_for_state(solver, &x_unscaled);
    let mut x: Array1<f64> = &x_unscaled / &scale;
    let magnetic_axis_picard: (f64, f64) = (
        solver.time_slice.global_quantities.magnetic_axis.r,
        solver.time_slice.global_quantities.magnetic_axis.z,
    );
    n_flux_evaluations += 1;
    let update: PicardUpdate = match picard_update(solver, &x, &scale, magnetic_axis_picard) {
        Ok(update) => update,
        Err(error) => {
            solver.set_to_failed_time_slice(error);
            return;
        }
    };
    let mut g_x: Array1<f64> = update.x_new;
    let mut f_x: Array1<f64> = &g_x - &x;
    let mut psi_a_x: f64 = update.psi_a;
    let mut magnetic_axis_x: (f64, f64) = update.magnetic_axis;

    for i_iter in i_iter_handover..n_iter_max {
        // Convergence: the deviation that one more Picard iteration would see, which needs the flux
        // calculated from `G(x)`. This leaves the time-slice exactly as Picard leaves it
        set_state(solver, &(&g_x * &scale), magnetic_axis_x);
        n_flux_evaluations += 1;
        if let Err(error) = solver.update_psi_and_geometry() {
            solver.set_to_failed_time_slice(error);
            return;
        }
        let psi_a: f64 = solver.time_slice.global_quantities.psi_magnetic_axis;
        let psi_b: f64 = solver.time_slice.boundary.psi;
        let grad_shafranov_deviation_value: f64 = EquilibriumSolver::calculate_grad_shafranov_deviation(psi_a, psi_b, psi_a_x);
        solver.time_slice.convergence.grad_shafranov_deviation_value = grad_shafranov_deviation_value;

        // Check for convergence
        if grad_shafranov_deviation_value < grad_shafranov_deviation_tolerance && i_iter > n_iter_min {
            report(&settings, "converged", i_iter, n_flux_evaluations);
            solver.set_to_converged_time_slice(i_iter);
            return;
        }

        // Check if we have reached the maximum number of iterations
        if i_iter == n_iter_max - 1 {
            report(&settings, "reached the maximum number of iterations", i_iter, n_flux_evaluations);
            solver.set_to_failed_time_slice(Error::MaxIterReached);
            return;
        }

        // The Newton step, from GMRES. `J v` is a finite difference of `G`:
        //     J v = (G(x + h v) - G(x)) / h - v
        // with the step `h` halved if `G` cannot be evaluated at the perturbed point, for example
        // because the perturbation has removed the magnetic axis
        let f_x_norm: f64 = norm(&f_x);
        let finite_difference_step: f64 = (settings.finite_difference_step * f_x_norm).max(FINITE_DIFFERENCE_STEP_MIN);
        let jacobian_vector_product = |v: &Array1<f64>| -> Option<Array1<f64>> {
            let mut step: f64 = finite_difference_step;
            for _i_halving in 0..=N_FINITE_DIFFERENCE_HALVINGS_MAX {
                n_flux_evaluations += 1;
                if let Ok(update) = picard_update(solver, &(&x + &(step * v)), &scale, magnetic_axis_x) {
                    return Some((&update.x_new - &g_x) / step - v);
                }
                step *= 0.5;
            }
            None
        };
        let (dx, n_krylov, krylov_relative_residual): (Array1<f64>, usize, f64) =
            gmres(jacobian_vector_product, &f_x, settings.n_krylov_max, settings.krylov_tolerance);

        // Halve the step until it reduces the residual
        let mut step_length: f64 = 1.0;
        let mut accepted: Option<(Array1<f64>, PicardUpdate)> = None;
        if n_krylov > 0 {
            for _i_halving in 0..=N_STEP_HALVINGS_MAX {
                let x_trial: Array1<f64> = &x + &(step_length * &dx);
                n_flux_evaluations += 1;
                if let Ok(update) = picard_update(solver, &x_trial, &scale, magnetic_axis_x) {
                    let f_trial_norm: f64 = norm(&(&update.x_new - &x_trial));
                    if f_trial_norm < f_x_norm {
                        accepted = Some((x_trial, update));
                        break;
                    }
                }
                step_length *= 0.5;
            }
        }

        // Otherwise a Picard step, `x -> G(x)`
        let step_description: String;
        let (x_new, update): (Array1<f64>, PicardUpdate) = match accepted {
            Some(accepted) => {
                step_description = format!("Newton step x {step_length}");
                accepted
            }
            None => {
                step_description = "Picard step".to_string();
                n_flux_evaluations += 1;
                match picard_update(solver, &g_x, &scale, magnetic_axis_x) {
                    Ok(update) => (g_x.clone(), update),
                    Err(error) => {
                        solver.set_to_failed_time_slice(error);
                        return;
                    }
                }
            }
        };

        if settings.verbose {
            println!(
                "newton_krylov: iteration {i_iter}; grad_shafranov_deviation = {grad_shafranov_deviation_value:.3e}; |F| = {f_x_norm:.3e}; \
                 {n_krylov} Krylov directions leave {krylov_relative_residual:.2e} of |F|; {step_description}; {n_flux_evaluations} flux evaluations"
            );
        }

        // Move to the new point
        x = x_new;
        g_x = update.x_new;
        f_x = &g_x - &x;
        psi_a_x = update.psi_a;
        magnetic_axis_x = update.magnetic_axis;
    }
}

/// Approximately solve `J dx = -f` for the Newton step `dx`, with GMRES.
///
/// `J` is only ever multiplied onto a unit vector, by `jacobian_vector_product`, which returns
/// `None` when it cannot be evaluated.
///
/// Stops when the linearised residual `|f + J dx|` is below `krylov_tolerance * |f|`, or after
/// `n_krylov_max` directions, or when `jacobian_vector_product` fails.
///
/// Returns the step, the number of directions it is built from, and `|f + J dx| / |f|`.
///
/// Y. Saad and M. H. Schultz, "GMRES: A generalized minimal residual algorithm for solving
/// nonsymmetric linear systems", SIAM J. Sci. Stat. Comput., 1986, https://doi.org/10.1137/0907058
fn gmres(
    mut jacobian_vector_product: impl FnMut(&Array1<f64>) -> Option<Array1<f64>>,
    f: &Array1<f64>,
    n_krylov_max: usize,
    krylov_tolerance: f64,
) -> (Array1<f64>, usize, f64) {
    let beta: f64 = norm(f);
    if beta == 0.0 || n_krylov_max == 0 {
        return (Array1::zeros(f.len()), 0, 1.0);
    }

    // Orthonormal basis of the Krylov space, starting from the right hand side `-f`
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(n_krylov_max + 1);
    basis.push(-f / beta);
    // The Hessenberg matrix, reduced to upper triangular by Givens rotations as it is built
    let mut hessenberg: Array2<f64> = Array2::zeros((n_krylov_max + 1, n_krylov_max));
    let mut rotation_cos: Vec<f64> = Vec::with_capacity(n_krylov_max);
    let mut rotation_sin: Vec<f64> = Vec::with_capacity(n_krylov_max);
    // The rotated right hand side, `beta * e_1`; its last element is the linearised residual
    let mut rhs: Array1<f64> = Array1::zeros(n_krylov_max + 1);
    rhs[0] = beta;

    let mut n_krylov: usize = 0;
    for i_krylov in 0..n_krylov_max {
        // The next direction, `J v`
        let Some(mut w) = jacobian_vector_product(&basis[i_krylov]) else {
            break;
        };

        // Orthogonalise against the basis (modified Gram-Schmidt)
        for i_basis in 0..=i_krylov {
            let projection: f64 = w.dot(&basis[i_basis]);
            hessenberg[(i_basis, i_krylov)] = projection;
            w.scaled_add(-projection, &basis[i_basis]);
        }
        let w_norm: f64 = norm(&w);
        hessenberg[(i_krylov + 1, i_krylov)] = w_norm;

        // Apply the previous rotations to the new column, then a new rotation to zero its
        // sub-diagonal element
        for i_rotation in 0..i_krylov {
            let upper: f64 = hessenberg[(i_rotation, i_krylov)];
            let lower: f64 = hessenberg[(i_rotation + 1, i_krylov)];
            hessenberg[(i_rotation, i_krylov)] = rotation_cos[i_rotation] * upper + rotation_sin[i_rotation] * lower;
            hessenberg[(i_rotation + 1, i_krylov)] = -rotation_sin[i_rotation] * upper + rotation_cos[i_rotation] * lower;
        }
        let diagonal: f64 = hessenberg[(i_krylov, i_krylov)];
        let hypotenuse: f64 = diagonal.hypot(w_norm);
        rotation_cos.push(diagonal / hypotenuse);
        rotation_sin.push(w_norm / hypotenuse);
        hessenberg[(i_krylov, i_krylov)] = hypotenuse;
        hessenberg[(i_krylov + 1, i_krylov)] = 0.0;
        rhs[i_krylov + 1] = -rotation_sin[i_krylov] * rhs[i_krylov];
        rhs[i_krylov] *= rotation_cos[i_krylov];
        n_krylov = i_krylov + 1;

        // Stop once the linearised residual is small enough, or when the Krylov space has stopped
        // growing, in which case the step is already exact
        if rhs[i_krylov + 1].abs() < krylov_tolerance * beta || w_norm <= f64::EPSILON * beta {
            break;
        }
        basis.push(w / w_norm);
    }

    // The coefficients of the step in the basis, from the triangular system
    let mut coefficients: Array1<f64> = Array1::zeros(n_krylov);
    for i_row in (0..n_krylov).rev() {
        let mut sum: f64 = rhs[i_row];
        for i_column in (i_row + 1)..n_krylov {
            sum -= hessenberg[(i_row, i_column)] * coefficients[i_column];
        }
        coefficients[i_row] = sum / hessenberg[(i_row, i_row)];
    }
    let mut dx: Array1<f64> = Array1::zeros(f.len());
    for i_basis in 0..n_krylov {
        dx.scaled_add(coefficients[i_basis], &basis[i_basis]);
    }

    let relative_residual: f64 = rhs[n_krylov].abs() / beta;
    (dx, n_krylov, relative_residual)
}

/// Evaluate the Picard update `G` at the scaled state `x`: the flux, magnetic axis and boundary
/// calculated from `x`, the fit, and the new current density. There is no vertical feedback:
/// `delta_z` is 0.
///
/// The magnetic axis search starts from `magnetic_axis_start`, rather than from wherever the last
/// evaluation left it, so that `G` depends on `x` alone.
fn picard_update(solver: &mut EquilibriumSolver, x: &Array1<f64>, scale: &Array1<f64>, magnetic_axis_start: (f64, f64)) -> Result<PicardUpdate, Error> {
    set_state(solver, &(x * scale), magnetic_axis_start);
    solver.update_psi_and_geometry()?;
    let psi_a: f64 = solver.time_slice.global_quantities.psi_magnetic_axis;
    let magnetic_axis: (f64, f64) = (
        solver.time_slice.global_quantities.magnetic_axis.r,
        solver.time_slice.global_quantities.magnetic_axis.z,
    );
    solver.fit_and_update_current(false);
    let x_new: Array1<f64> = get_state(solver) / scale;
    Ok(PicardUpdate { x_new, psi_a, magnetic_axis })
}

/// The state the flux is calculated from, unscaled: `j_phi` flattened row by row, then the
/// passive degrees of freedom
fn get_state(solver: &EquilibriumSolver) -> Array1<f64> {
    let j_phi: &Array2<f64> = &solver.time_slice.profiles_2d[0].j_phi;
    let passive_dof_values: &Array1<f64> = &solver.passive_dof_values;
    let n_grid: usize = j_phi.len();
    let n_passive_dof: usize = passive_dof_values.len();

    let mut x: Array1<f64> = Array1::from_elem(n_grid + n_passive_dof, f64::NAN);
    x.slice_mut(s![0..n_grid]).assign(&Array1::from_iter(j_phi.iter().copied()));
    x.slice_mut(s![n_grid..n_grid + n_passive_dof]).assign(passive_dof_values);
    x
}

/// Put the unscaled state `x` back into the time-slice, with no vertical shift, and with the
/// magnetic axis search to start from `magnetic_axis_start`; the reverse of `get_state`
fn set_state(solver: &mut EquilibriumSolver, x: &Array1<f64>, magnetic_axis_start: (f64, f64)) {
    let n_r: usize = solver.equilibrium_code.grid.n_r as usize;
    let n_z: usize = solver.equilibrium_code.grid.n_z as usize;
    let n_grid: usize = n_z * n_r;
    let n_passive_dof: usize = solver.passive_dof_values.len();

    solver.time_slice.profiles_2d[0].j_phi = x.slice(s![0..n_grid]).to_shape((n_z, n_r)).unwrap().to_owned();
    solver.passive_dof_values = x.slice(s![n_grid..n_grid + n_passive_dof]).to_owned();
    solver.time_slice.convergence.delta_z = 0.0;
    solver.time_slice.global_quantities.magnetic_axis.r = magnetic_axis_start.0;
    solver.time_slice.global_quantities.magnetic_axis.z = magnetic_axis_start.1;
}

/// What each variable in the state is divided by, so that the blocks are comparable and a norm of
/// the whole state means something:
/// * `j_phi`: by the norm of `j_phi`, so a relative change of 1 across the whole plasma has norm 1
/// * the passive degrees of freedom: likewise, by their norm (or by 1 A if they are all zero)
///
/// Fixed at the start of the Newton iterations.
fn scale_for_state(solver: &EquilibriumSolver, x: &Array1<f64>) -> Array1<f64> {
    let n_grid: usize = solver.time_slice.profiles_2d[0].j_phi.len();
    let n_passive_dof: usize = solver.passive_dof_values.len();

    let j_phi_norm: f64 = norm(&x.slice(s![0..n_grid]).to_owned());
    let passive_norm: f64 = norm(&x.slice(s![n_grid..n_grid + n_passive_dof]).to_owned());

    let mut scale: Array1<f64> = Array1::from_elem(x.len(), f64::NAN);
    scale.slice_mut(s![0..n_grid]).fill(j_phi_norm);
    scale
        .slice_mut(s![n_grid..n_grid + n_passive_dof])
        .fill(if passive_norm > 0.0 { passive_norm } else { 1.0 });
    scale
}

/// The Euclidean norm
fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// Print how the solve finished, when `verbose` is on
fn report(settings: &NewtonKrylovSettings, outcome: &str, i_iter: usize, n_flux_evaluations: usize) {
    if settings.verbose {
        println!("newton_krylov: {outcome} at iteration {i_iter}, after {n_flux_evaluations} flux evaluations");
    }
}

#[cfg(test)]
mod tests {
    use super::{gmres, norm};
    use ndarray::{Array1, Array2, array};

    /// A small, well conditioned, nonsymmetric matrix, and a right hand side
    fn test_problem() -> (Array2<f64>, Array1<f64>) {
        let a: Array2<f64> = array![
            [4.0, 1.0, 0.0, 0.5, 0.0],
            [-1.0, 3.0, 0.7, 0.0, 0.2],
            [0.3, 0.0, 5.0, -1.2, 0.0],
            [0.0, 0.9, 0.0, 2.5, 0.4],
            [0.1, 0.0, -0.6, 0.0, 3.5],
        ];
        let f: Array1<f64> = array![1.0, -2.0, 0.5, 3.0, -1.0];
        (a, f)
    }

    /// With as many directions as unknowns, GMRES is a direct solve of `A dx = -f`
    #[test]
    fn solves_a_linear_system_exactly_with_enough_directions() {
        let (a, f): (Array2<f64>, Array1<f64>) = test_problem();
        let (dx, n_krylov, relative_residual): (Array1<f64>, usize, f64) = gmres(|v: &Array1<f64>| Some(a.dot(v)), &f, 5, 1.0e-12);

        let residual: Array1<f64> = &f + &a.dot(&dx);
        assert!(n_krylov <= 5);
        assert!(norm(&residual) < 1.0e-10 * norm(&f), "|f + A dx| = {}", norm(&residual));
        assert!(relative_residual < 1.0e-10);
    }

    /// Stopping early, the residual GMRES reports is the true linearised residual
    #[test]
    fn reports_the_residual_it_leaves() {
        let (a, f): (Array2<f64>, Array1<f64>) = test_problem();
        let (dx, n_krylov, relative_residual): (Array1<f64>, usize, f64) = gmres(|v: &Array1<f64>| Some(a.dot(v)), &f, 2, 1.0e-12);

        let residual: Array1<f64> = &f + &a.dot(&dx);
        assert_eq!(n_krylov, 2);
        assert!((norm(&residual) / norm(&f) - relative_residual).abs() < 1.0e-12);
        assert!(relative_residual < 1.0, "two directions should reduce the residual");
    }

    /// When the Jacobian-vector product cannot be evaluated there is no step
    #[test]
    fn returns_no_step_when_the_first_product_fails() {
        let (_a, f): (Array2<f64>, Array1<f64>) = test_problem();
        let (dx, n_krylov, relative_residual): (Array1<f64>, usize, f64) = gmres(|_v: &Array1<f64>| None, &f, 5, 1.0e-12);

        assert_eq!(n_krylov, 0);
        assert_eq!(relative_residual, 1.0);
        assert!(dx.iter().all(|&value| value == 0.0));
    }
}
