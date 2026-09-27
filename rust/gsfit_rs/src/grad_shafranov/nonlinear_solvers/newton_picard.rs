//! Newton-Picard iteration: the recursive projection method (RPM).
//!
//! A Picard iteration is the fixed-point iteration `x -> G(x)`, where `x` is everything the flux
//! is calculated from: the current density and the passive degrees of freedom. Close to the fixed
//! point, each iteration multiplies the error by `M`, the Jacobian of `G`, so Picard converges only
//! when every eigenvalue of `M` is inside the unit circle, and slowly when one is close to it.
//! Usually only a few are, for example the plasma's rigid vertical and radial displacements. This
//! splits the state space in two:
//! * `P`, spanned by an orthonormal basis `Z` of the few slow or unstable directions, where
//!   Newton's method is used
//! * `Q = I - Z Z^T`, everything else, where the Picard iteration converges by itself
//!
//! With `F(x) = G(x) - x`, and `H = Z^T M Z`, which is `M` restricted to `P` and only `p x p`, each
//! iteration is
//! ```text
//!     x -> x + Q F(x) + Z (I - H)^-1 Z^T F(x)
//! ```
//! a Picard step in `Q` and a Newton step in `P`. With no basis it is exactly Picard.
//!
//! The basis is found from the iteration itself, which is what makes the projection "recursive".
//! When the part of `F` in `Q` has shrunk by less than a factor `contraction_threshold` since the
//! previous iteration, Picard is converging too slowly in `Q`, or diverging, and that part of `F` is
//! dominated by the slowest direction left in `Q`, so it joins the basis. `M z`, for each basis
//! vector `z`, is a finite difference of `G`:
//! ```text
//!     M z = (G(x + h z) - G(x)) / h
//! ```
//! Each costs one evaluation of `G`, which is one Picard iteration's worth of work. It is calculated
//! when `z` joins the basis, and for the whole basis again only when the part of `F` in `P` stops
//! shrinking, the sign that `H` is out of date. So once the basis has been found, an iteration costs
//! about one evaluation of `G`, as a Picard iteration does. The finite-difference step `h` is
//! `finite_difference_step` times the size of `F`, as for `newton_krylov`. A step is halved until
//! `G` can be evaluated there.
//!
//! Convergence is tested the same way as for Picard: the deviation is the change in the flux on the
//! magnetic axis between iterations, relative to `psi_b - psi_a`. On convergence the time-slice is
//! left as Picard leaves it: the current density from the last fit, and the flux calculated from it.
//!
//! Like `newton_krylov`, this does not use `delta_z`. Picard needs it because the plasma is
//! vertically unstable, which makes its fixed point unstable too. Here the vertical displacement is
//! one of the directions in `P`, where Newton's method converges to an unstable fixed point without
//! help. So `delta_z` is 0, the flux is the true flux from the coils, passives and plasma, and a
//! forward solve should leave out the magnetic axis constraint (`StationaryPoint`).
//!
//! The variables in `x` have very different units, so each block is divided by its typical size
//! (see `scale_for_state`) before any norm is taken.
//!
//! G. M. Shroff and H. B. Keller, "Stabilization of unstable procedures: the recursive projection
//! method", SIAM J. Numer. Anal., 1993, https://doi.org/10.1137/0730057
//!
//! K. Lust, D. Roose, A. Spence and A. R. Champneys, "An adaptive Newton-Picard algorithm with
//! subspace iteration for computing periodic solutions", SIAM J. Sci. Comput., 1998,
//! https://doi.org/10.1137/S1064827594277673

use crate::grad_shafranov::Error;
use crate::grad_shafranov::equilibrium_solve::EquilibriumSolver;
use faer::linalg::solvers::Solve;
use ndarray::{Array1, Array2, s};

/// The number of times a step is halved when `G` cannot be evaluated at the new point, for example
/// because the step has removed the magnetic axis
const N_STEP_HALVINGS_MAX: usize = 5;

/// The number of times the finite-difference step is halved when `G` cannot be evaluated at the
/// perturbed point
const N_FINITE_DIFFERENCE_HALVINGS_MAX: usize = 3;

/// The smallest finite-difference step, in the scaled units of `scale_for_state`. Below this the
/// Jacobian-vector product would be dominated by rounding in `G`
const FINITE_DIFFERENCE_STEP_MIN: f64 = 1.0e-7;

/// A new basis vector is only accepted if at least this fraction of it is left after it is made
/// orthogonal to the basis; otherwise it is already in the basis, up to rounding
const ORTHOGONALISATION_TOLERANCE: f64 = 1.0e-8;

/// One evaluation of the Picard update, `G`
struct PicardUpdate {
    /// `G(x)`, scaled
    x_new: Array1<f64>,
    /// The flux on the magnetic axis, calculated from `x`
    psi_a: f64,
    /// The flux on the plasma boundary, calculated from `x`
    psi_b: f64,
    /// The magnetic axis, calculated from `x`, from which the next search starts
    magnetic_axis: (f64, f64),
}

/// The settings of the Newton-Picard iterations, from `code/numerics/nonlinear_solver/newton_picard`
struct NewtonPicardSettings {
    n_basis_max: usize,
    contraction_threshold: f64,
    finite_difference_step: f64,
    verbose: bool,
}

/// The subspace `P` of slow or unstable directions, where Newton's method is used
struct Subspace {
    /// The orthonormal basis, `Z`
    basis: Vec<Array1<f64>>,
    /// `M z` for each basis vector `z`, where `M` is the Jacobian of `G`
    basis_images: Vec<Array1<f64>>,
}

impl Subspace {
    fn new() -> Self {
        Subspace {
            basis: Vec::new(),
            basis_images: Vec::new(),
        }
    }

    /// The number of basis vectors, `p`
    fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Split `v` into its coordinates in the basis, `Z^T v`, and its part in `Q`, `v - Z Z^T v`
    fn project(&self, v: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let coordinates: Array1<f64> = Array1::from_iter(self.basis.iter().map(|z: &Array1<f64>| z.dot(v)));
        let mut complement: Array1<f64> = v.clone();
        for i_basis in 0..self.dimension() {
            complement.scaled_add(-coordinates[i_basis], &self.basis[i_basis]);
        }
        (coordinates, complement)
    }

    /// `v` made orthogonal to the basis and normalised, ready to join it. It is made orthogonal twice,
    /// so that rounding leaves none of the basis in it. `None` when `v` is already in the basis
    fn orthonormalise(&self, v: &Array1<f64>) -> Option<Array1<f64>> {
        let (_coordinates, complement_once): (Array1<f64>, Array1<f64>) = self.project(v);
        let (_coordinates, complement): (Array1<f64>, Array1<f64>) = self.project(&complement_once);
        let complement_norm: f64 = norm(&complement);
        if complement_norm.is_nan() || complement_norm <= ORTHOGONALISATION_TOLERANCE * norm(v) {
            return None;
        }
        Some(complement / complement_norm)
    }

    /// `H = Z^T M Z`, the Jacobian of `G` restricted to the subspace
    fn reduced_jacobian(&self) -> Array2<f64> {
        let n_basis: usize = self.dimension();
        Array2::from_shape_fn((n_basis, n_basis), |(i_row, i_column)| self.basis[i_row].dot(&self.basis_images[i_column]))
    }
}

/// Solve with Newton-Picard iterations; see the module documentation.
pub(crate) fn solve(solver: &mut EquilibriumSolver) {
    // Solver settings, supplied through `equilibrium.code`
    let n_iter_max: usize = solver.equilibrium_code.numerics.iterations.n_max as usize;
    let n_iter_min: usize = solver.equilibrium_code.numerics.iterations.n_min as usize;
    let grad_shafranov_deviation_tolerance: f64 = solver.equilibrium_code.numerics.grad_shafranov_deviation_tolerance;
    let settings: NewtonPicardSettings = NewtonPicardSettings {
        n_basis_max: solver.equilibrium_code.numerics.nonlinear_solver.newton_picard.n_basis_max as usize,
        contraction_threshold: solver.equilibrium_code.numerics.nonlinear_solver.newton_picard.contraction_threshold,
        finite_difference_step: solver.equilibrium_code.numerics.nonlinear_solver.newton_picard.finite_difference_step,
        verbose: solver.equilibrium_code.numerics.nonlinear_solver.newton_picard.verbose != 0,
    };

    // The passive currents and the initial current density
    if let Err(error) = solver.initialise() {
        solver.set_to_failed_time_slice(error);
        return;
    }

    // The number of times the flux has been calculated, which is where nearly all the time goes.
    // A Picard iteration calculates it once
    let mut n_flux_evaluations: usize = 0;

    // The state, scaled, and `G` at it. The first magnetic axis search starts from the centre of
    // the initial guess, which `initialise` has put in the IDS
    let x_unscaled: Array1<f64> = get_state(solver);
    let scale: Array1<f64> = scale_for_state(solver, &x_unscaled);
    let mut x: Array1<f64> = &x_unscaled / &scale;
    let magnetic_axis_initial_guess: (f64, f64) = (
        solver.time_slice.global_quantities.magnetic_axis.r,
        solver.time_slice.global_quantities.magnetic_axis.z,
    );
    n_flux_evaluations += 1;
    let update: PicardUpdate = match picard_update(solver, &x, &scale, magnetic_axis_initial_guess) {
        Ok(update) => update,
        Err(error) => {
            solver.set_to_failed_time_slice(error);
            return;
        }
    };
    let mut g_x: Array1<f64> = update.x_new;
    let mut psi_a_x: f64 = update.psi_a;
    let mut psi_b_x: f64 = update.psi_b;
    let mut magnetic_axis_x: (f64, f64) = update.magnetic_axis;

    let mut subspace: Subspace = Subspace::new();
    let mut f_complement_norm_previous: Option<f64> = None; // needed to test how fast the Picard iteration in `Q` converges
    let mut f_subspace_norm_previous: Option<f64> = None; // needed to test whether `H` is out of date
    let mut psi_a_previous: f64 = 0.0; // needed to calculate the Grad-Shafranov deviation

    for i_iter in 0..n_iter_max {
        let f_x: Array1<f64> = &g_x - &x;

        // Calculate the Grad-Shafranov deviation
        let grad_shafranov_deviation_value: f64 = EquilibriumSolver::calculate_grad_shafranov_deviation(psi_a_x, psi_b_x, psi_a_previous);
        solver.time_slice.convergence.grad_shafranov_deviation_value = grad_shafranov_deviation_value;
        psi_a_previous = psi_a_x;

        // Check for convergence. The time-slice is left as Picard leaves it: the current density
        // from the last fit, `G(x)`, and the flux calculated from it
        if grad_shafranov_deviation_value < grad_shafranov_deviation_tolerance && i_iter > n_iter_min {
            set_state(solver, &(&g_x * &scale), magnetic_axis_x);
            n_flux_evaluations += 1;
            if let Err(error) = solver.update_psi_and_geometry() {
                solver.set_to_failed_time_slice(error);
                return;
            }
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

        // The finite-difference step for `M z`, which shrinks as the solution is approached
        let finite_difference_step: f64 = (settings.finite_difference_step * norm(&f_x)).max(FINITE_DIFFERENCE_STEP_MIN);

        // Split `F` into its coordinates in the basis, and its part in `Q`
        let (mut f_subspace, mut f_complement): (Array1<f64>, Array1<f64>) = subspace.project(&f_x);
        let f_complement_norm: f64 = norm(&f_complement);

        // Grow the basis, when Picard is converging too slowly in `Q`: the part of `F` in `Q` is
        // compared with the previous iteration's, before that iteration grew the basis. A direction
        // added then has left `Q`, so if it was the slow one, the part in `Q` now shrinks quickly
        let mut complement_contraction: f64 = f64::NAN;
        let mut basis_grown: bool = false;
        if let Some(f_complement_norm_previous) = f_complement_norm_previous {
            complement_contraction = f_complement_norm / f_complement_norm_previous;
            if complement_contraction > settings.contraction_threshold
                && subspace.dimension() < settings.n_basis_max
                && let Some(direction) = subspace.orthonormalise(&f_complement)
                && let Some(image) = jacobian_vector_product(
                    solver,
                    &x,
                    &g_x,
                    &direction,
                    &scale,
                    magnetic_axis_x,
                    finite_difference_step,
                    &mut n_flux_evaluations,
                )
            {
                subspace.basis.push(direction);
                subspace.basis_images.push(image);
                basis_grown = true;
                (f_subspace, f_complement) = subspace.project(&f_x);
            }
        }

        // Recalculate `M z` for the whole basis when the Newton iteration in `P` has stopped
        // converging, the sign that `H` is out of date. A basis vector whose `M z` cannot be
        // evaluated is dropped
        let mut basis_images_recalculated: bool = false;
        if !basis_grown
            && subspace.dimension() > 0
            && let Some(f_subspace_norm_previous) = f_subspace_norm_previous
            && norm(&f_subspace) > f_subspace_norm_previous
        {
            let mut subspace_recalculated: Subspace = Subspace::new();
            for direction in subspace.basis {
                if let Some(image) = jacobian_vector_product(
                    solver,
                    &x,
                    &g_x,
                    &direction,
                    &scale,
                    magnetic_axis_x,
                    finite_difference_step,
                    &mut n_flux_evaluations,
                ) {
                    subspace_recalculated.basis.push(direction);
                    subspace_recalculated.basis_images.push(image);
                }
            }
            subspace = subspace_recalculated;
            basis_images_recalculated = true;
            (f_subspace, f_complement) = subspace.project(&f_x);
        }
        let f_subspace_norm: f64 = norm(&f_subspace);

        // The step: Newton in `P`, Picard in `Q`
        let reduced_jacobian: Array2<f64> = subspace.reduced_jacobian();
        let d_z: Array1<f64> = newton_step_in_subspace(&reduced_jacobian, &f_subspace);
        let mut d_x: Array1<f64> = f_complement;
        for i_basis in 0..subspace.dimension() {
            d_x.scaled_add(d_z[i_basis], &subspace.basis[i_basis]);
        }

        // Take the step, halving it until `G` can be evaluated at the new point
        let mut step_length: f64 = 1.0;
        let mut accepted: Option<(Array1<f64>, PicardUpdate)> = None;
        let mut step_error: Option<Error> = None;
        for _i_halving in 0..=N_STEP_HALVINGS_MAX {
            let x_trial: Array1<f64> = &x + &(step_length * &d_x);
            n_flux_evaluations += 1;
            match picard_update(solver, &x_trial, &scale, magnetic_axis_x) {
                Ok(update) => {
                    accepted = Some((x_trial, update));
                    break;
                }
                Err(error) => step_error = Some(error),
            }
            step_length *= 0.5;
        }
        let Some((x_new, update)) = accepted else {
            report(&settings, "could not take a step", i_iter, n_flux_evaluations);
            solver.set_to_failed_time_slice(step_error.unwrap()); // the loop runs at least once, so there is an error
            return;
        };

        if settings.verbose {
            let basis_description: &str = if basis_grown {
                " (grown)"
            } else if basis_images_recalculated {
                " (M Z recalculated)"
            } else {
                ""
            };
            let eigenvalue_moduli: Vec<String> = eigenvalue_moduli(&reduced_jacobian)
                .iter()
                .map(|modulus: &f64| format!("{modulus:.3}"))
                .collect();
            println!(
                "newton_picard: iteration {i_iter}; grad_shafranov_deviation = {grad_shafranov_deviation_value:.3e}; |F| = {:.3e}; \
                 Q contraction = {complement_contraction:.3}; basis of {}{basis_description}, |eigenvalues of H| = [{}]; \
                 step x {step_length}; {n_flux_evaluations} flux evaluations",
                norm(&f_x),
                subspace.dimension(),
                eigenvalue_moduli.join(", "),
            );
        }

        // Move to the new point
        x = x_new;
        g_x = update.x_new;
        psi_a_x = update.psi_a;
        psi_b_x = update.psi_b;
        magnetic_axis_x = update.magnetic_axis;
        f_complement_norm_previous = Some(f_complement_norm);
        f_subspace_norm_previous = Some(f_subspace_norm);
    }
}

/// `M v`, where `M` is the Jacobian of `G` at `x`, as a finite difference of `G`:
/// ```text
///     M v = (G(x + h v) - G(x)) / h
/// ```
/// with the step `h` halved if `G` cannot be evaluated at the perturbed point, for example because
/// the perturbation has removed the magnetic axis. `None` if it still cannot be.
#[allow(clippy::too_many_arguments)]
fn jacobian_vector_product(
    solver: &mut EquilibriumSolver,
    x: &Array1<f64>,
    g_x: &Array1<f64>,
    v: &Array1<f64>,
    scale: &Array1<f64>,
    magnetic_axis_start: (f64, f64),
    finite_difference_step: f64,
    n_flux_evaluations: &mut usize,
) -> Option<Array1<f64>> {
    let mut step: f64 = finite_difference_step;
    for _i_halving in 0..=N_FINITE_DIFFERENCE_HALVINGS_MAX {
        *n_flux_evaluations += 1;
        if let Ok(update) = picard_update(solver, &(x + &(step * v)), scale, magnetic_axis_start) {
            return Some((&update.x_new - g_x) / step);
        }
        step *= 0.5;
    }
    None
}

/// The Newton step in the subspace: the solution `d_z` of
/// ```text
///     (I - H) d_z = f_subspace
/// ```
/// where `f_subspace = Z^T F(x)`. When `I - H` is singular, which is when `G` has an eigenvalue of
/// exactly 1 in the subspace, the Picard step `d_z = f_subspace` is taken instead.
fn newton_step_in_subspace(reduced_jacobian: &Array2<f64>, f_subspace: &Array1<f64>) -> Array1<f64> {
    let n_basis: usize = f_subspace.len();
    if n_basis == 0 {
        return Array1::zeros(0);
    }
    let newton_matrix: faer::Mat<f64> = faer::Mat::from_fn(n_basis, n_basis, |i_row, i_column| {
        let identity: f64 = if i_row == i_column { 1.0 } else { 0.0 };
        identity - reduced_jacobian[(i_row, i_column)]
    });
    let rhs: faer::Mat<f64> = faer::Mat::from_fn(n_basis, 1, |i_row, _| f_subspace[i_row]);
    let solution: faer::Mat<f64> = newton_matrix.partial_piv_lu().solve(&rhs);
    let d_z: Array1<f64> = Array1::from_shape_fn(n_basis, |i_row| solution[(i_row, 0)]);
    if d_z.iter().all(|value: &f64| value.is_finite()) {
        d_z
    } else {
        f_subspace.clone()
    }
}

/// The moduli of the eigenvalues of `H`, largest first, which are the rates at which Picard would
/// converge (below 1) or diverge (above 1) in each direction of the subspace
fn eigenvalue_moduli(reduced_jacobian: &Array2<f64>) -> Vec<f64> {
    let n_basis: usize = reduced_jacobian.nrows();
    if n_basis == 0 {
        return Vec::new();
    }
    let reduced_jacobian_faer: faer::Mat<f64> = faer::Mat::from_fn(n_basis, n_basis, |i_row, i_column| reduced_jacobian[(i_row, i_column)]);
    let Ok(eigenvalues) = reduced_jacobian_faer.eigenvalues() else {
        return Vec::new();
    };
    let mut moduli: Vec<f64> = eigenvalues.iter().map(|eigenvalue: &faer::c64| eigenvalue.re.hypot(eigenvalue.im)).collect();
    moduli.sort_by(|a: &f64, b: &f64| b.total_cmp(a));
    moduli
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
    let psi_b: f64 = solver.time_slice.boundary.psi;
    let magnetic_axis: (f64, f64) = (
        solver.time_slice.global_quantities.magnetic_axis.r,
        solver.time_slice.global_quantities.magnetic_axis.z,
    );
    solver.fit_and_update_current(false);
    let x_new: Array1<f64> = get_state(solver) / scale;
    Ok(PicardUpdate {
        x_new,
        psi_a,
        psi_b,
        magnetic_axis,
    })
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
/// Fixed at the start, from the initial guess.
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
fn report(settings: &NewtonPicardSettings, outcome: &str, i_iter: usize, n_flux_evaluations: usize) {
    if settings.verbose {
        println!("newton_picard: {outcome} at iteration {i_iter}, after {n_flux_evaluations} flux evaluations");
    }
}

#[cfg(test)]
mod tests {
    use super::{Subspace, eigenvalue_moduli, newton_step_in_subspace, norm};
    use ndarray::{Array1, Array2, array};

    /// The Newton step solves `(I - H) d_z = f_subspace`
    #[test]
    fn newton_step_solves_the_reduced_system() {
        let reduced_jacobian: Array2<f64> = array![[3.0, 0.5], [-0.2, 0.4]];
        let f_subspace: Array1<f64> = array![1.0, -2.0];
        let d_z: Array1<f64> = newton_step_in_subspace(&reduced_jacobian, &f_subspace);

        let identity: Array2<f64> = Array2::eye(2);
        let residual: Array1<f64> = (&identity - &reduced_jacobian).dot(&d_z) - &f_subspace;
        assert!(norm(&residual) < 1.0e-12, "|(I - H) d_z - f| = {}", norm(&residual));
    }

    /// When `I - H` is singular the step is a Picard step
    #[test]
    fn newton_step_falls_back_to_picard_when_singular() {
        let reduced_jacobian: Array2<f64> = array![[1.0, 0.0], [0.0, 0.5]];
        let f_subspace: Array1<f64> = array![1.0, -2.0];
        let d_z: Array1<f64> = newton_step_in_subspace(&reduced_jacobian, &f_subspace);

        assert_eq!(d_z, f_subspace);
    }

    /// With no basis there is no Newton step
    #[test]
    fn newton_step_is_empty_without_a_basis() {
        let d_z: Array1<f64> = newton_step_in_subspace(&Array2::zeros((0, 0)), &Array1::zeros(0));

        assert_eq!(d_z.len(), 0);
    }

    /// A vector splits into its part in the basis and its part in `Q`, which is orthogonal to the basis
    #[test]
    fn project_splits_a_vector_into_the_basis_and_its_complement() {
        let mut subspace: Subspace = Subspace::new();
        subspace.basis.push(array![1.0, 0.0, 0.0]);
        subspace.basis_images.push(array![0.0, 0.0, 0.0]);
        let v: Array1<f64> = array![2.0, 3.0, -1.0];
        let (coordinates, complement): (Array1<f64>, Array1<f64>) = subspace.project(&v);

        assert_eq!(coordinates, array![2.0]);
        assert_eq!(complement, array![0.0, 3.0, -1.0]);
    }

    /// A new basis vector is orthonormal to the basis, and a vector already in the basis is rejected
    #[test]
    fn orthonormalise_gives_a_new_orthonormal_basis_vector() {
        let mut subspace: Subspace = Subspace::new();
        subspace.basis.push(array![1.0, 0.0, 0.0]);
        subspace.basis_images.push(array![0.0, 0.0, 0.0]);

        let direction: Array1<f64> = subspace.orthonormalise(&array![1.0, 2.0, 2.0]).unwrap();
        assert!((norm(&direction) - 1.0).abs() < 1.0e-14);
        assert!(direction.dot(&subspace.basis[0]).abs() < 1.0e-14);

        assert!(subspace.orthonormalise(&array![3.0, 0.0, 0.0]).is_none());
    }

    /// `H = Z^T M Z`, from the basis and its images
    #[test]
    fn reduced_jacobian_is_the_jacobian_restricted_to_the_basis() {
        let m: Array2<f64> = array![[2.0, 1.0, 0.0], [0.5, 0.3, 0.0], [0.0, 0.0, 0.1]];
        let mut subspace: Subspace = Subspace::new();
        for z in [array![1.0, 0.0, 0.0], array![0.0, 1.0, 0.0]] {
            subspace.basis_images.push(m.dot(&z));
            subspace.basis.push(z);
        }

        assert_eq!(subspace.reduced_jacobian(), array![[2.0, 1.0], [0.5, 0.3]]);
    }

    /// The eigenvalue moduli, largest first, including a complex pair
    #[test]
    fn eigenvalue_moduli_are_sorted_largest_first() {
        // Eigenvalues 0.5 and +/- 2i
        let reduced_jacobian: Array2<f64> = array![[0.5, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]];
        let moduli: Vec<f64> = eigenvalue_moduli(&reduced_jacobian);

        assert_eq!(moduli.len(), 3);
        assert!((moduli[0] - 2.0).abs() < 1.0e-12 && (moduli[1] - 2.0).abs() < 1.0e-12 && (moduli[2] - 0.5).abs() < 1.0e-12);
    }
}
