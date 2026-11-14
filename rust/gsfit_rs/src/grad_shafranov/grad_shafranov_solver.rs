use super::chi_sq_mag::epp_chi_sq_mag;
use super::equilibrium_solve::{EquilibriumSolver, GradShafranovInputs, PsiAndDerivativesGreens};
use super::initial_current_seed::quadratic_current_density_seed;
use super::nonlinear_solvers::SharedNonlinearSolver;
use super::output_flag;
use super::point_constraints::PointConstraints;
use crate::equilibrium::Equilibrium;
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::greens_storage::GreensTables;
use crate::settings::settings_equilibrium::SettingsEquilibrium;
use crate::source_functions::SourceFunctionTraits;
use imas_rs::{
    Equilibrium as EquilibriumIds, EquilibriumConstraints, EquilibriumConstraints0d, EquilibriumLimiter, EquilibriumLimiterUnitOutline, EquilibriumTimeSlice,
};
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[pyfunction]
#[pyo3(signature = (equilibrium))]
pub fn solve_grad_shafranov(mut equilibrium: PyRefMut<Equilibrium>) {
    println!("solve_grad_shafranov starting");
    let timing_start_serial_setup: Instant = Instant::now();

    // The reconstruction times, from the IDS `Equilibrium::new` allocated one time-slice per. Reading
    // them here rather than taking them as an argument makes it impossible to solve a different
    // set of times than the IDS was built for
    let times_to_reconstruct: Array1<f64> = equilibrium.equilibrium_ids.time_slice(..).time.to_array();
    let n_time: usize = times_to_reconstruct.len();

    if n_time == 0 {
        println!("solve_grad_shafranov: no times to reconstruct, returning");
        return;
    }

    // The vacuum toroidal field, from the equilibrium IDS, which `Equilibrium.copy_from_tf_ids` copied it into
    // from the `tf` IDS, with `b0` already interpolated onto the reconstruction times. Checked here,
    // before anything is solved, so that a missing toroidal field stops the solve with one message
    let vacuum_toroidal_field_r0: f64 = equilibrium.equilibrium_ids.vacuum_toroidal_field.r0;
    if vacuum_toroidal_field_r0.is_nan() || equilibrium.equilibrium_ids.vacuum_toroidal_field.b0.len() != n_time {
        panic!(
            "solve_grad_shafranov: `equilibrium/vacuum_toroidal_field` is unset: copy the toroidal field in from the `tf` IDS with `Equilibrium.copy_from_tf_ids`"
        );
    }

    // The limiter, from the equilibrium IDS, which `Equilibrium.copy_from_wall_ids` copied it into from the `wall` IDS, where
    // `Wall.add_limiter_unit` checked the outline of each unit. Checked here, before anything is solved, so that a missing limiter
    // stops the solve with one message rather than failing every time-slice
    if equilibrium.equilibrium_ids.limiter.unit.is_empty() {
        panic!("solve_grad_shafranov: `equilibrium/limiter/unit` is empty: copy the limiter in from the `wall` IDS with `Equilibrium.copy_from_wall_ids`");
    }

    // The PF coil currents, from the equilibrium IDS, which `Equilibrium.copy_from_pf_active_ids` copied onto every
    // time-slice from the `pf_active` IDS. The solver multiplies them by the Greens tables, `greens_tables/grid/pf_active`, so both
    // must list the same coils in the same order, which `pf_active/coil` sets for both. Checked here, before anything
    // is solved. The sensors find their Greens with each coil by name, in this order
    let greens_pf_active_names: Vec<String> = equilibrium.greens_tables.grid.pf_active(..).name.to_vec();
    for i_time in 0..n_time {
        let pf_current_names: Vec<String> = equilibrium.equilibrium_ids.time_slice[i_time].constraints.pf_current(..).name.to_vec();
        if pf_current_names.is_empty() {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_current` is unset: copy the PF coil currents in from the `pf_active` IDS with `Equilibrium.copy_from_pf_active_ids`"
            );
        }
        if pf_current_names != greens_pf_active_names {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_current(:)/name` is {pf_current_names:?}, but the Greens tables, `greens_tables/grid/pf_active(:)/name`, are for {greens_pf_active_names:?}; the two must list the same PF coils, in the same order"
            );
        }
    }

    // The passive loops, from the equilibrium IDS, which `Equilibrium.copy_from_pf_passive_ids` made from the fit settings and the
    // `pf_passive` IDS, and copied onto every time-slice: the degrees of freedom each loop's current is fitted with, and their
    // regularisations. The solver fits an amplitude for each degree of freedom with the Greens tables, `greens_tables/grid/pf_passive`, so both must
    // list the same loops and degrees of freedom in the same order, which is the order of the fit settings, `equilibrium_settings/pf_passive_dof`,
    // and then of each loop's degrees of freedom. Checked here, before anything is solved. This is the order the reconstruction fits them in:
    // the sensors gather their Greens with each degree of freedom by name in it, and every time-slice's `passive_dof_values` follows it
    let greens_pf_passive_names: Vec<String> = equilibrium.greens_tables.grid.pf_passive(..).name.to_vec();
    let mut greens_pf_passive_dof_names: Vec<(String, String)> = Vec::new();
    let n_greens_pf_passive: usize = equilibrium.greens_tables.grid.pf_passive.len();
    for i_greens_pf_passive in 0..n_greens_pf_passive {
        let n_dof: usize = equilibrium.greens_tables.grid.pf_passive[i_greens_pf_passive].dof.len();
        for i_dof in 0..n_dof {
            greens_pf_passive_dof_names.push((
                equilibrium.greens_tables.grid.pf_passive[i_greens_pf_passive].name.clone(),
                equilibrium.greens_tables.grid.pf_passive[i_greens_pf_passive].dof[i_dof].name.clone(),
            ));
        }
    }
    for i_time in 0..n_time {
        let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &equilibrium.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current;
        let n_loop: usize = pf_passive_currents.len();
        // Without any passive loops there is nothing to copy, so an empty `pf_passive_current` is only unset when there are Greens tables
        if n_loop == 0 && n_greens_pf_passive > 0 {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current` is unset: copy the passive loops in from the `pf_passive` IDS with `Equilibrium.copy_from_pf_passive_ids`"
            );
        }
        let pf_passive_current_names: Vec<String> = equilibrium.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current(..).name.to_vec();
        if pf_passive_current_names != greens_pf_passive_names {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current(:)/name` is {pf_passive_current_names:?}, but the Greens tables, `greens_tables/grid/pf_passive(:)/name`, are for {greens_pf_passive_names:?}; the two must list the same passive loops, in the same order"
            );
        }
        let mut pf_passive_current_dof_names: Vec<(String, String)> = Vec::new();
        for i_loop in 0..n_loop {
            let n_dof: usize = pf_passive_currents[i_loop].dof.len();
            for i_dof in 0..n_dof {
                pf_passive_current_dof_names.push((pf_passive_currents[i_loop].name.clone(), pf_passive_currents[i_loop].dof[i_dof].name.clone()));
            }
        }
        if pf_passive_current_dof_names != greens_pf_passive_dof_names {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current(:)/dof(:)/name` is {pf_passive_current_dof_names:?}, but the Greens tables, `greens_tables/grid/pf_passive(:)/dof(:)/name`, are for {greens_pf_passive_dof_names:?}; the two must list the same degrees of freedom, in the same order"
            );
        }
    }

    // The pressure constraints, which `Equilibrium.copy_from_core_profiles_ids` makes when the fit settings, `equilibrium_settings/constraints/pressure`,
    // include pressure. Without any, while the fit settings include pressure, they were not copied in, which is checked here, before anything
    // is solved
    for i_time in 0..n_time {
        if equilibrium.settings_equilibrium.constraints.pressure.include && equilibrium.equilibrium_ids.time_slice[i_time].constraints.pressure.is_empty() {
            panic!(
                "solve_grad_shafranov: the fit settings, `equilibrium_settings/constraints/pressure`, include pressure, but `time_slice({i_time})/constraints/pressure` is empty: make the pressure constraints from the `core_profiles` IDS with `Equilibrium.copy_from_core_profiles_ids`"
            );
        }
    }

    // Every magnetic constraint, fitted or not, needs its measurement, which `Equilibrium.copy_from_magnetics_ids` copies in, recording where
    // from in `source`, and its Greens tables, which `Equilibrium.greens_with_magnetics` calculates, as the solver fits it with them, and the
    // post-processor reconstructs it with them. The diamagnetic flux, which is there when it has fit settings, needs no Greens tables. Checked
    // here, before anything is solved
    let greens_b_field_pol_probe_names: Vec<String> = equilibrium.greens_tables.b_field_pol_probe(..).name.to_vec();
    let greens_flux_loop_names: Vec<String> = equilibrium.greens_tables.flux_loop(..).name.to_vec();
    let greens_rogowski_coil_names: Vec<String> = equilibrium.greens_tables.rogowski_coil(..).name.to_vec();
    for i_time in 0..n_time {
        let constraints: &EquilibriumConstraints = &equilibrium.equilibrium_ids.time_slice[i_time].constraints;
        for (sensor, names, sources, greens_names) in [
            (
                "b_field_pol_probe",
                constraints.b_field_pol_probe(..).name.to_vec(),
                constraints.b_field_pol_probe(..).source.to_vec(),
                &greens_b_field_pol_probe_names,
            ),
            (
                "flux_loop",
                constraints.flux_loop(..).name.to_vec(),
                constraints.flux_loop(..).source.to_vec(),
                &greens_flux_loop_names,
            ),
            (
                "rogowski_coil",
                constraints.rogowski_coil(..).name.to_vec(),
                constraints.rogowski_coil(..).source.to_vec(),
                &greens_rogowski_coil_names,
            ),
        ] {
            let n_constraint: usize = names.len();
            for i_constraint in 0..n_constraint {
                if sources[i_constraint].is_empty() {
                    panic!(
                        "solve_grad_shafranov: `{}`, in `time_slice({i_time})/constraints/{sensor}`, has no measurement: copy the magnetic sensors in from the `magnetics` IDS with `Equilibrium.copy_from_magnetics_ids`",
                        names[i_constraint]
                    );
                }
                if !greens_names.contains(&names[i_constraint]) {
                    panic!(
                        "solve_grad_shafranov: `{}`, in `time_slice({i_time})/constraints/{sensor}`, has no Greens tables in `greens_tables/{sensor}`: calculate them with `Equilibrium.greens_with_magnetics`",
                        names[i_constraint]
                    );
                }
            }
        }
        if equilibrium.settings_equilibrium.constraints.diamagnetic_flux.is_some() && constraints.diamagnetic_flux.source.is_empty() {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/diamagnetic_flux` has no measurement: copy the magnetic sensors in from the `magnetics` IDS with `Equilibrium.copy_from_magnetics_ids`"
            );
        }
    }

    // The constraints at points which can move, the isoflux, isoflux boundary and stationary point constraints, need their points, which
    // `Equilibrium.copy_from_magnetics_ids` copies in, recording where from in `source`, and their Greens tables, which
    // `Equilibrium.greens_with_magnetics` calculates, with a table for each time-slice, as their points can move. Checked here, before
    // anything is solved
    let greens_isoflux_names: Vec<String> = equilibrium.greens_tables.isoflux(..).name.to_vec();
    let greens_isoflux_boundary_names: Vec<String> = equilibrium.greens_tables.isoflux_boundary(..).name.to_vec();
    let greens_stationary_point_names: Vec<String> = equilibrium.greens_tables.stationary_point(..).name.to_vec();
    let mut n_time_greens_tables: Vec<(&str, String, usize)> = Vec::new();
    for greens in &equilibrium.greens_tables.isoflux {
        n_time_greens_tables.push(("isoflux", greens.name.clone(), greens.plasma.dim().0));
    }
    for greens in &equilibrium.greens_tables.isoflux_boundary {
        n_time_greens_tables.push(("isoflux_boundary", greens.name.clone(), greens.plasma.dim().0));
    }
    for greens in &equilibrium.greens_tables.stationary_point {
        n_time_greens_tables.push(("stationary_point", greens.name.clone(), greens.d_psi_d_r.plasma.dim().0));
    }
    for (constraint_kind, name, n_time_greens) in &n_time_greens_tables {
        if *n_time_greens != n_time {
            panic!(
                "solve_grad_shafranov: `greens_tables/{constraint_kind}` has tables for {n_time_greens} time-slice(s) for `{name}`, but there are {n_time}: calculate them with `Equilibrium.greens_with_magnetics` after the time-slices are allocated"
            );
        }
    }
    for i_time in 0..n_time {
        let constraints: &EquilibriumConstraints = &equilibrium.equilibrium_ids.time_slice[i_time].constraints;
        for (constraint_kind, names, sources, greens_names) in [
            (
                "isoflux",
                constraints.isoflux(..).name.to_vec(),
                constraints.isoflux(..).source.to_vec(),
                &greens_isoflux_names,
            ),
            (
                "isoflux_boundary",
                constraints.isoflux_boundary(..).name.to_vec(),
                constraints.isoflux_boundary(..).source.to_vec(),
                &greens_isoflux_boundary_names,
            ),
            (
                "stationary_point",
                constraints.stationary_point(..).name.to_vec(),
                constraints.stationary_point(..).source.to_vec(),
                &greens_stationary_point_names,
            ),
        ] {
            let n_constraint: usize = names.len();
            for i_constraint in 0..n_constraint {
                if sources[i_constraint].is_empty() {
                    panic!(
                        "solve_grad_shafranov: `{}`, in `time_slice({i_time})/constraints/{constraint_kind}`, has no points: copy them in from the `magnetics` IDS with `Equilibrium.copy_from_magnetics_ids`",
                        names[i_constraint]
                    );
                }
                if !greens_names.contains(&names[i_constraint]) {
                    panic!(
                        "solve_grad_shafranov: `{}`, in `time_slice({i_time})/constraints/{constraint_kind}`, has no Greens tables in `greens_tables/{constraint_kind}`: calculate them with `Equilibrium.greens_with_magnetics`",
                        names[i_constraint]
                    );
                }
            }
        }
    }

    // Gathered for each time-slice, those fitted there with their Greens tables at it, so that each time-slice's solver, which does not know
    // which time-slice it is solving, is handed its own
    let mut isoflux_constraints: Vec<PointConstraints> = Vec::with_capacity(n_time);
    let mut isoflux_boundary_constraints: Vec<PointConstraints> = Vec::with_capacity(n_time);
    let mut stationary_point_constraints: Vec<PointConstraints> = Vec::with_capacity(n_time);
    {
        let equilibrium_ref: &Equilibrium = &equilibrium;
        for i_time in 0..n_time {
            let time_slice: &EquilibriumTimeSlice = &equilibrium_ref.equilibrium_ids.time_slice[i_time];
            isoflux_constraints.push(PointConstraints::isoflux(
                time_slice,
                i_time,
                &equilibrium_ref.greens_tables,
                &equilibrium_ref.settings_equilibrium.constraints,
            ));
            isoflux_boundary_constraints.push(PointConstraints::isoflux_boundary(
                time_slice,
                i_time,
                &equilibrium_ref.greens_tables,
                &equilibrium_ref.settings_equilibrium.constraints,
            ));
            stationary_point_constraints.push(PointConstraints::stationary_point(
                time_slice,
                i_time,
                &equilibrium_ref.greens_tables,
                &equilibrium_ref.settings_equilibrium.constraints,
            ));
        }
    }

    // Deref the `PyRefMut` once. Every field access on it calls `deref_mut`, which borrows the
    // whole of it, so `&settings_equilibrium` and `&greens` could not be held while `time_slice` is mutated;
    // off one `&mut Equilibrium` they are disjoint fields and borrow independently
    let equilibrium: &mut Equilibrium = &mut equilibrium;
    let equilibrium_ids: &mut EquilibriumIds = &mut equilibrium.equilibrium_ids;

    let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = equilibrium.p_prime_source_function.clone();
    let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = equilibrium.ff_prime_source_function.clone();
    let nonlinear_solver: SharedNonlinearSolver = equilibrium.nonlinear_solver.clone();

    // Loop over time in parallel and store in "results"

    let settings_equilibrium: &SettingsEquilibrium = &equilibrium.settings_equilibrium;
    // `vacuum_toroidal_field/b0` is indexed by time, so each time-slice's solve is handed its own value
    let vacuum_toroidal_field_b0: &Array1<f64> = &equilibrium_ids.vacuum_toroidal_field.b0;
    // Geometry only, so the same tables serve every time-slice. Borrowed from a different field
    // of the equilibrium than the IDS, whose `time_slice` the parallel solve mutates, so it can hold both at once
    let equilibrium_greens_tables: &GreensTables = &equilibrium.greens_tables;
    // Geometry only too, and borrowed in the same way. Its `unit(0)` is the vacuum vessel
    let equilibrium_limiter: &EquilibriumLimiter = &equilibrium_ids.limiter;
    let vessel_r: &Array1<f64> = &equilibrium_limiter.unit[0].outline.r;
    let vessel_z: &Array1<f64> = &equilibrium_limiter.unit[0].outline.z;
    // The candidate limit points: every point of every limiter unit, in unit order, as a plasma can be limited on any plasma facing
    // component. Gathered once here, for the initial current density and for every time-slice's boundary search
    let n_unit: usize = equilibrium_limiter.unit.len();
    let mut n_limit_pt: usize = 0;
    for i_unit in 0..n_unit {
        n_limit_pt += equilibrium_limiter.unit[i_unit].outline.r.len();
    }
    let mut limit_pts_r: Array1<f64> = Array1::from_elem(n_limit_pt, f64::NAN);
    let mut limit_pts_z: Array1<f64> = Array1::from_elem(n_limit_pt, f64::NAN);
    let mut i_limit_pt: usize = 0;
    for i_unit in 0..n_unit {
        let unit_outline: &EquilibriumLimiterUnitOutline = &equilibrium_limiter.unit[i_unit].outline;
        let n_point_this_unit: usize = unit_outline.r.len();
        for i_point in 0..n_point_this_unit {
            limit_pts_r[i_limit_pt] = unit_outline.r[i_point];
            limit_pts_z[i_limit_pt] = unit_outline.z[i_point];
            i_limit_pt += 1;
        }
    }
    // The same tables, reorganised into the matrix shapes the per-iteration GEMMs want. This
    // depends only on the geometry, so it is built once here and shared by every time-slice;
    // it used to be rebuilt inside each of the 480 parallel solves
    let psi_and_derivatives_greens: PsiAndDerivativesGreens = PsiAndDerivativesGreens::new(equilibrium_greens_tables);

    // The initial current-density guess. Every input to it is shared between time-slices - the
    // grid, the limiter, and `equilibrium_settings/initial_guess` - so it is built once here. It used to be rebuilt
    // inside each of the parallel solves, which put 7% of the whole run inside `geo`, testing a
    // 4097-point ellipse against the vessel polygon once per slice.
    // The `Result` is carried into the solve rather than unwrapped here, so that a bad initial
    // guess still fails every time-slice with the reason it always did
    let initial_j_2d: Result<Array2<f64>, String> = {
        let grid_r: Array1<f64> = equilibrium_ids.time_slice[0].profiles_2d[0].grid.dim1.clone();
        let grid_z: Array1<f64> = equilibrium_ids.time_slice[0].profiles_2d[0].grid.dim2.clone();
        let d_area: f64 = equilibrium_ids.time_slice[0].profiles_2d[0].grid.d_area;
        if grid_r.is_empty() || grid_z.is_empty() || d_area.is_nan() {
            panic!("solve_grad_shafranov: `profiles_2d(0)/grid` is unset");
        }
        let initial_guess_ip: f64 = settings_equilibrium.initial_guess.ip;
        let initial_guess_cur_r: f64 = settings_equilibrium.initial_guess.cur_r;
        let initial_guess_cur_z: f64 = settings_equilibrium.initial_guess.cur_z;
        let initial_guess_minor_radius: f64 = settings_equilibrium.initial_guess.minor_radius;
        let initial_guess_elongation: f64 = settings_equilibrium.initial_guess.elongation;

        quadratic_current_density_seed(
            &grid_r,
            &grid_z,
            &limit_pts_r,
            &limit_pts_z,
            vessel_r,
            vessel_z,
            d_area,
            initial_guess_ip,
            initial_guess_cur_r,
            initial_guess_cur_z,
            initial_guess_minor_radius,
            initial_guess_elongation,
        )
    };

    // Everything above this point is serial: checking the inputs, gathering the constraints at
    // points which can move for each time-slice, the shared Greens tables and the initial current seed. Timed separately
    // because it is a single-core stretch at the head of the solve stage, and without a number
    // for it the only way to size it is to read it off a CPU-utilisation plot
    println!(
        "solve_grad_shafranov: serial setup done; {:.2}ms",
        timing_start_serial_setup.elapsed().as_secs_f64() * 1e3
    );

    // Solve the GS equation for all time-slices, in parallel
    let gsfit_solve_all_time_slices_timing_start: Instant = Instant::now();
    equilibrium_ids
        .time_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(i_time, time_slice): (usize, &mut EquilibriumTimeSlice)| {
            // Select the data for this time-slice
            // Note: the GS solver is designed to consider a single time-slice
            // and deliberately does not know what time-slice it is solving
            let grad_shafranov_inputs: GradShafranovInputs = GradShafranovInputs {
                psi_and_derivatives_greens: &psi_and_derivatives_greens,
                initial_j_2d: &initial_j_2d,
                limit_pts_r: &limit_pts_r,
                limit_pts_z: &limit_pts_z,
                isoflux: &isoflux_constraints[i_time],
                isoflux_boundary: &isoflux_boundary_constraints[i_time],
                stationary_point: &stationary_point_constraints[i_time],
                p_prime_source_function: &p_prime_source_function,
                ff_prime_source_function: &ff_prime_source_function,
                nonlinear_solver: &nonlinear_solver,
            };

            // Solve
            let mut solver: EquilibriumSolver = EquilibriumSolver::new(
                time_slice,
                settings_equilibrium,
                equilibrium_greens_tables,
                equilibrium_limiter,
                vacuum_toroidal_field_r0,
                vacuum_toroidal_field_b0[i_time],
                &grad_shafranov_inputs,
            );
            solver.solve();
            solver.write_to_time_slice();
        });
    println!(
        "solve_grad_shafranov: parallel time-slice solve done; {:.2}ms",
        gsfit_solve_all_time_slices_timing_start.elapsed().as_secs_f64() * 1e3
    );
    let timing_start_serial_finish: Instant = Instant::now();

    // `code/output_flag` is indexed by time, so it is assembled here rather than by the per-slice
    // solver: 0 for a usable slice, negative for one which failed
    let output_flags: Array1<i32> = Array1::from_iter(equilibrium_ids.time_slice.iter().map(output_flag));
    equilibrium_ids.code.output_flag = output_flags;

    for (i_time, time_slice) in equilibrium_ids.time_slice.iter().enumerate() {
        let solution_found: bool = time_slice.global_quantities.ip.is_finite();
        println!(
            "time={:6.1}ms;  solution_found={};  grad_shafranov_deviation_value={:.18};  n_iter={}",
            times_to_reconstruct[i_time] * 1e3,
            solution_found,
            time_slice.convergence.grad_shafranov_deviation_value,
            time_slice.convergence.iterations_n,
        );
    }

    let gsfit_solve_all_time_slices_duration: Duration = gsfit_solve_all_time_slices_timing_start.elapsed();
    info!("GSFit time elapsed: {:?}", gsfit_solve_all_time_slices_duration);

    // Post-process. This also measures the gaps which `Equilibrium.copy_from_pulse_schedule_ids` copied onto
    // the time-slices, where the pulse schedule defines any
    equilibrium_post_processor(equilibrium_ids, equilibrium_greens_tables, &p_prime_source_function, &ff_prime_source_function);

    // Get error codes for failed time-slices

    // The fitted magnetic sensors' chi squared, from their constraints' `measured`, `reconstructed`, `weight` and `expected_value`, for the
    // sensors the solver fitted
    epp_chi_sq_mag(&mut equilibrium.equilibrium_ids.time_slice, settings_equilibrium);

    // Everything after the parallel loop: the per-slice report lines, the post-processors, and
    // recalculating the sensor values. Mostly serial, and it is the second single-core stretch on
    // a CPU-utilisation plot
    println!(
        "solve_grad_shafranov: serial finish done; {:.2}ms",
        timing_start_serial_finish.elapsed().as_secs_f64() * 1e3
    );
}
