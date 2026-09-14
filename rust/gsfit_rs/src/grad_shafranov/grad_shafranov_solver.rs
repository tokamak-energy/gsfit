use super::chi_sq_mag::epp_chi_sq_mag;
use super::equilibrium_solve::{EquilibriumSolver, GradShafranovInputs, PsiAndDerivativesGreens};
use super::initial_current_seed::quadratic_current_density_seed;
use super::output_flag;
use crate::coils::Coils;
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::passives::Passives;
use crate::plasma::Plasma;
use crate::sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils, SensorsDynamic, SensorsStatic, StationaryPoint};
use crate::source_functions::SourceFunctionTraits;
use crate::tf::Tf;
use crate::wall::{Wall, limiter_points, vacuum_vessel_outline};
use imas_rs::ids::wall::Wall as WallIds;
use imas_rs::{Code, Equilibrium, EquilibriumGreens, EquilibriumTimeSlice};
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2, s};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[pyfunction]
pub fn solve_grad_shafranov(
    mut plasma: PyRefMut<Plasma>,
    wall: PyRef<Wall>,
    tf: PyRef<Tf>,
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    mut bp_probes: PyRefMut<BpProbes>,
    mut flux_loops: PyRefMut<FluxLoops>,
    mut rogowski_coils: PyRefMut<RogowskiCoils>,
    mut isoflux: PyRefMut<Isoflux>,
    mut isoflux_boundary: PyRefMut<IsofluxBoundary>,
    mut pressure_sensors: PyRefMut<Pressure>,
    mut stationary_point: PyRefMut<StationaryPoint>,
    mut dialoop: PyRefMut<Dialoop>,
) {
    println!("solve_grad_shafranov starting");
    let timing_start_serial_setup: Instant = Instant::now();

    // The reconstruction times, from the IDS `Plasma::new` allocated one time-slice per. Reading
    // them here rather than taking them as an argument makes it impossible to solve a different
    // set of times than the IDS was built for
    let times_to_reconstruct: Array1<f64> = plasma.equilibrium_ids.time_slice(..).time.to_array();
    let n_time: usize = times_to_reconstruct.len();

    if n_time == 0 {
        println!("solve_grad_shafranov: no times to reconstruct, returning");
        return;
    }

    // Get static and dynamic data
    let coils_dynamic: Vec<SensorsDynamic> = coils.split_into_static_and_dynamic(&times_to_reconstruct);
    // The toroidal field, from the `tf` IDS. `b_field_phi_vacuum_r` is stored on the experimental
    // timebase, so it is interpolated onto `times_to_reconstruct` here
    let b_field_phi_vacuum_r_time: &Array1<f64> = &tf.tf_ids.b_field_phi_vacuum_r.time;
    let b_field_phi_vacuum_r_data: &Array1<f64> = &tf.tf_ids.b_field_phi_vacuum_r.data;
    if b_field_phi_vacuum_r_time.is_empty() || b_field_phi_vacuum_r_data.is_empty() {
        panic!("solve_grad_shafranov: `tf/b_field_phi_vacuum_r` is unset");
    }
    let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(b_field_phi_vacuum_r_time.to_owned(), b_field_phi_vacuum_r_data.to_owned())
        .expect("solve_grad_shafranov: cannot build the `tf/b_field_phi_vacuum_r` interpolator");
    let f_vac_vs_time: Array1<f64> = interpolator
        .interpolate_array1(&times_to_reconstruct)
        .expect("solve_grad_shafranov: cannot interpolate `tf/b_field_phi_vacuum_r` onto the reconstruction times");

    // The reference major radius the vacuum toroidal field is quoted at, and the same signal
    // expressed the way the equilibrium IDS holds it: `f_vac = r0 * b0`. The sign carries through
    let vacuum_toroidal_field_r0: f64 = tf.tf_ids.r0;
    if vacuum_toroidal_field_r0.is_nan() {
        panic!("solve_grad_shafranov: `tf/r0` is unset");
    }
    let b0_vs_time: Array1<f64> = f_vac_vs_time / vacuum_toroidal_field_r0;

    let (bp_probes_static, bp_probes_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) = bp_probes.split_into_static_and_dynamic(&times_to_reconstruct);
    let (flux_loops_static, flux_loops_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        flux_loops.split_into_static_and_dynamic(&times_to_reconstruct);
    let (rogowski_coils_static, rogowski_coils_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        rogowski_coils.split_into_static_and_dynamic(&times_to_reconstruct);
    let (isoflux_statics, isoflux_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) = isoflux.split_into_static_and_dynamic(&times_to_reconstruct);
    let (isoflux_boundary_statics, isoflux_boundary_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        isoflux_boundary.split_into_static_and_dynamic(&times_to_reconstruct);
    let (pressure_statics, pressure_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        pressure_sensors.split_into_static_and_dynamic(&times_to_reconstruct);
    let (stationary_point_statics, stationary_point_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        stationary_point.split_into_static_and_dynamic(&times_to_reconstruct);
    let (dialoop_statics, dialoop_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) = dialoop.split_into_static_and_dynamic(&times_to_reconstruct);

    // TODO: might be better to combine all sensors here, before passing to the solver

    // Deref the `PyRefMut` once. Every field access on it calls `deref_mut`, which borrows the
    // whole of it, so `&code` and `&greens` could not be held while `time_slice` is mutated;
    // off one `&mut Equilibrium` they are disjoint fields and borrow independently
    let plasma: &mut Plasma = &mut plasma;
    let equilibrium_ids: &mut Equilibrium = &mut plasma.equilibrium_ids;

    // Copied out of the `PyRef`, because the per-time-slice solves run on Rayon's threads and
    // a `PyRef` is neither `Send` nor `Sync`
    let wall_ids: WallIds = wall.wall_ids.clone();

    let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = plasma.p_prime_source_function.clone();
    let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = plasma.ff_prime_source_function.clone();

    // Count the number of passive degrees of freedom
    let mut n_passive_dof: usize = 0;
    let mut n_passive_regularisation: usize = 0;
    for passive_name in passives.results.keys() {
        n_passive_dof += passives.results.get(&passive_name).get("dof").keys().len();
        n_passive_regularisation += passives.results.get(&passive_name).get("regularisations").unwrap_array2().shape()[0];
    }

    let mut passive_regularisations: Array2<f64> = Array2::zeros((n_passive_regularisation, n_passive_dof));
    let mut i_reg: usize = 0;
    let mut i_dof: usize = 0;
    for passive_name in passives.results.keys() {
        let n_passive_dof_this_passive: usize = passives.results.get(&passive_name).get("dof").keys().len();
        let n_passive_regularisation_this_passive: usize = passives.results.get(&passive_name).get("regularisations").unwrap_array2().shape()[0];

        let regularisations_this_passive: Array2<f64> = passives.results.get(&passive_name).get("regularisations").unwrap_array2();

        if n_passive_regularisation_this_passive > 0 {
            passive_regularisations
                .slice_mut(s![
                    i_reg..=i_reg + n_passive_regularisation_this_passive - 1,
                    i_dof..=i_dof + n_passive_dof_this_passive - 1
                ])
                .assign(&regularisations_this_passive);
        }

        // Update counter for next passive
        i_dof += n_passive_dof_this_passive;
        i_reg += n_passive_regularisation_this_passive;
    }

    let mut passive_regularisations_weight: Array1<f64> = Array1::zeros(n_passive_regularisation);
    let mut i_reg: usize = 0;
    for passive_name in passives.results.keys() {
        let n_passive_regularisation_this_passive: usize = passives.results.get(&passive_name).get("regularisations").unwrap_array2().shape()[0];
        let regularisations_weight_this_passive: Array1<f64> = passives.results.get(&passive_name).get("regularisations_weight").unwrap_array1();

        if n_passive_regularisation_this_passive > 0 {
            passive_regularisations_weight
                .slice_mut(s![i_reg..=i_reg + n_passive_regularisation_this_passive - 1])
                .assign(&regularisations_weight_this_passive);
        }

        // Update counter for next passive
        i_reg += n_passive_regularisation_this_passive;
    }

    // Loop over time in parallel and store in "results"

    // The data dictionary requires `vacuum_toroidal_field/r0 * b0` to equal the `tf` IDS's
    // `b_field_phi_vacuum_r`, so both are filled from that one signal here. The rod current is not
    // a data dictionary node, so it is recovered from these two wherever it is needed
    equilibrium_ids.vacuum_toroidal_field.r0 = vacuum_toroidal_field_r0;
    equilibrium_ids.vacuum_toroidal_field.b0 = b0_vs_time.clone();

    let equilibrium_code: &Code = &equilibrium_ids.code;
    // Geometry only, so the same tables serve every time-slice. Borrowed from a different field
    // of the IDS than `time_slice`, so the parallel solve can hold both at once
    let equilibrium_greens_tables: &EquilibriumGreens = &equilibrium_ids.greens;
    // The same tables, reorganised into the matrix shapes the per-iteration GEMMs want. This
    // depends only on the geometry, so it is built once here and shared by every time-slice;
    // it used to be rebuilt inside each of the 480 parallel solves
    let psi_and_derivatives_greens: PsiAndDerivativesGreens = PsiAndDerivativesGreens::new(equilibrium_greens_tables);

    // The initial current-density guess. Every input to it is shared between time-slices - the
    // grid, the wall, and `code/initial_guess` - so it is built once here. It used to be rebuilt
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
        let initial_guess_ip: f64 = equilibrium_ids.code.initial_guess.ip;
        let initial_guess_cur_r: f64 = equilibrium_ids.code.initial_guess.cur_r;
        let initial_guess_cur_z: f64 = equilibrium_ids.code.initial_guess.cur_z;
        let initial_guess_minor_radius: f64 = equilibrium_ids.code.initial_guess.minor_radius;
        let initial_guess_elongation: f64 = equilibrium_ids.code.initial_guess.elongation;

        // Limiter, from the `wall` IDS. `limiter_points` gathers every limiter unit,
        // `vacuum_vessel_outline` is `unit(0)` alone
        match (limiter_points(&wall_ids), vacuum_vessel_outline(&wall_ids)) {
            (Ok((limiter_r, limiter_z)), Ok((vessel_r, vessel_z))) => quadratic_current_density_seed(
                &grid_r,
                &grid_z,
                &limiter_r,
                &limiter_z,
                &vessel_r,
                &vessel_z,
                d_area,
                initial_guess_ip,
                initial_guess_cur_r,
                initial_guess_cur_z,
                initial_guess_minor_radius,
                initial_guess_elongation,
            ),
            (Err(reason), _) | (_, Err(reason)) => Err(reason),
        }
    };

    // Everything above this point is serial: splitting the sensors into static and dynamic,
    // building the IDS, the shared Greens tables and the initial current seed. Timed separately
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
                coils_dynamic: &coils_dynamic[i_time],
                bp_probes_static: &bp_probes_static[i_time],
                bp_probes_dynamic: &bp_probes_dynamic[i_time],
                flux_loops_static: &flux_loops_static[i_time],
                flux_loops_dynamic: &flux_loops_dynamic[i_time],
                dialoop_static: &dialoop_statics[i_time],
                dialoop_dynamic: &dialoop_dynamic[i_time],
                rogowski_coils_static: &rogowski_coils_static[i_time],
                rogowski_coils_dynamic: &rogowski_coils_dynamic[i_time],
                isoflux_static: &isoflux_statics[i_time],
                isoflux_dynamic: &isoflux_dynamic[i_time],
                isoflux_boundary_static: &isoflux_boundary_statics[i_time],
                isoflux_boundary_dynamic: &isoflux_boundary_dynamic[i_time],
                pressure_sensors_static: &pressure_statics[i_time],
                pressure_sensors_dynamic: &pressure_dynamic[i_time],
                magnetic_axis_static: &stationary_point_statics[i_time],
                magnetic_axis_dynamic: &stationary_point_dynamic[i_time],
                p_prime_source_function: &p_prime_source_function,
                ff_prime_source_function: &ff_prime_source_function,
                passive_regularisations: &passive_regularisations,
                passive_regularisations_weight: &passive_regularisations_weight,
            };

            // Solve
            let mut solver: EquilibriumSolver = EquilibriumSolver::new(
                time_slice,
                equilibrium_code,
                equilibrium_greens_tables,
                &wall_ids,
                vacuum_toroidal_field_r0,
                b0_vs_time[i_time],
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
            "time={:6.1}ms;  solution_found={};  gs_error={:.18};  n_iter={}",
            times_to_reconstruct[i_time] * 1e3,
            solution_found,
            time_slice.convergence.grad_shafranov_deviation_value,
            time_slice.convergence.iterations_n,
        );
    }

    let gsfit_solve_all_time_slices_duration: Duration = gsfit_solve_all_time_slices_timing_start.elapsed();
    info!("GSFit time elapsed: {:?}", gsfit_solve_all_time_slices_duration);

    // Post-process
    equilibrium_post_processor(equilibrium_ids, &wall_ids, &p_prime_source_function, &ff_prime_source_function);
    passives.equilibrium_post_processor(equilibrium_ids);

    // Get error codes for failed time-slices

    // Calculate sensor values. Borrowed rather than cloned: each of these is a separate
    // `PyRefMut`, so borrowing them here does not clash with the `&mut` on the sensor being written
    bp_probes.calculate_sensor_values_rs(&coils, &passives, &plasma);
    flux_loops.calculate_sensor_values_rs(&coils, &passives, &plasma);
    rogowski_coils.calculate_sensor_values_rs(&coils, &passives, &plasma);
    if pressure_sensors.results.data.len() > 0 {
        pressure_sensors.calculate_sensor_values_rust(&plasma);
    }
    // The diamagnetic loop depends only on the toroidal flux function `f` (no Green's functions)
    dialoop.calculate_sensor_values_rs(&plasma);

    // Calculate chi_sq_mag for each time slice
    let chi_mag: Array1<f64> = epp_chi_sq_mag(&bp_probes, &flux_loops, &rogowski_coils, &dialoop, n_time);
    // The same quantity on the IDS. It is calculated here rather than in the post-processor
    // because it needs the sensors, which the post-processor does not see
    for (i_time, time_slice) in plasma.equilibrium_ids.time_slice.iter_mut().enumerate() {
        time_slice.constraints.chi_squared_reduced = chi_mag[i_time];
    }

    // Everything after the parallel loop: the per-slice report lines, the post-processors, and
    // recalculating the sensor values. Mostly serial, and it is the second single-core stretch on
    // a CPU-utilisation plot
    println!(
        "solve_grad_shafranov: serial finish done; {:.2}ms",
        timing_start_serial_finish.elapsed().as_secs_f64() * 1e3
    );
}
