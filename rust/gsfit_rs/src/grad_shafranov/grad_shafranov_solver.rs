use super::chi_sq_mag::epp_chi_sq_mag;
use super::equilibrium_solve::{EquilibriumSolver, GradShafranovInputs, PsiAndDerivativesGreens};
use super::initial_current_seed::quadratic_current_density_seed;
use super::output_flag;
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::limiter::{limiter_points, vacuum_vessel_outline};
use crate::plasma::Plasma;
use crate::sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils, SensorsDynamic, SensorsStatic, StationaryPoint};
use crate::source_functions::SourceFunctionTraits;
use imas_rs::{Code, Equilibrium, EquilibriumConstraints, EquilibriumConstraints0d, EquilibriumGreens, EquilibriumLimiter, EquilibriumTimeSlice};
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[pyfunction]
#[pyo3(signature = (plasma, bp_probes, flux_loops, rogowski_coils, isoflux, isoflux_boundary, pressure_sensors, stationary_point, dialoop))]
pub fn solve_grad_shafranov(
    mut plasma: PyRefMut<Plasma>,
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

    // The vacuum toroidal field, from the equilibrium IDS, which `Plasma.copy_from_tf_ids` copied it into
    // from the `tf` IDS, with `b0` already interpolated onto the reconstruction times. Checked here,
    // before anything is solved, so that a missing toroidal field stops the solve with one message
    let vacuum_toroidal_field_r0: f64 = plasma.equilibrium_ids.vacuum_toroidal_field.r0;
    if vacuum_toroidal_field_r0.is_nan() || plasma.equilibrium_ids.vacuum_toroidal_field.b0.len() != n_time {
        panic!(
            "solve_grad_shafranov: `equilibrium/vacuum_toroidal_field` is unset: copy the toroidal field in from the `tf` IDS with `Plasma.copy_from_tf_ids`"
        );
    }

    // The limiter, from the equilibrium IDS, which `Plasma.copy_from_wall_ids` copied it into from the
    // `wall` IDS. `limiter_points` gathers every limiter unit, `vacuum_vessel_outline` is `unit(0)`
    // alone. Read here, before anything is solved, so that a missing limiter stops the solve with one
    // message rather than failing every time-slice
    let (limiter_r, limiter_z): (Array1<f64>, Array1<f64>) =
        limiter_points(&plasma.equilibrium_ids.limiter).unwrap_or_else(|reason: String| panic!("solve_grad_shafranov: {reason}"));
    let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) =
        vacuum_vessel_outline(&plasma.equilibrium_ids.limiter).unwrap_or_else(|reason: String| panic!("solve_grad_shafranov: {reason}"));

    // The PF coil currents, from the equilibrium IDS, which `Plasma.copy_from_pf_active_ids` copied onto every
    // time-slice from the `pf_active` IDS. The solver multiplies them by the Greens tables, `greens/grid/pf_active`, so both
    // must list the same coils in the same order, which `pf_active/coil` sets for both. Checked here, before anything
    // is solved. The sensors find their Greens with each coil by name, in this order
    let greens_pf_active_names: Vec<String> = plasma.equilibrium_ids.greens.grid.pf_active(..).name.to_vec();
    for i_time in 0..n_time {
        let pf_current_names: Vec<String> = plasma.equilibrium_ids.time_slice[i_time].constraints.pf_current(..).name.to_vec();
        if pf_current_names.is_empty() {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_current` is unset: copy the PF coil currents in from the `pf_active` IDS with `Plasma.copy_from_pf_active_ids`"
            );
        }
        if pf_current_names != greens_pf_active_names {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_current(:)/name` is {pf_current_names:?}, but the Greens tables, `greens/grid/pf_active(:)/name`, are for {greens_pf_active_names:?}; the two must list the same PF coils, in the same order"
            );
        }
    }

    // The passive loops, from the equilibrium IDS, which `Plasma.copy_from_pf_passive_ids` copied onto every time-slice from the
    // `pf_passive` IDS: the degrees of freedom each loop's current is fitted with, and their regularisations. The solver fits an
    // amplitude for each degree of freedom with the Greens tables, `greens/grid/pf_passive`, so both must list the same loops and degrees of
    // freedom in the same order, which `pf_passive/loop`, and then `loop(i1)/dof`, sets for both. Checked here, before anything is
    // solved. This is the order the reconstruction fits them in: the sensors gather their Greens with each degree of freedom by name in
    // it, and every time-slice's `passive_dof_values` follows it
    let greens_pf_passive_names: Vec<String> = plasma.equilibrium_ids.greens.grid.pf_passive(..).name.to_vec();
    let mut greens_pf_passive_dof_names: Vec<(String, String)> = Vec::new();
    let n_greens_pf_passive: usize = plasma.equilibrium_ids.greens.grid.pf_passive.len();
    for i_greens_pf_passive in 0..n_greens_pf_passive {
        let n_dof: usize = plasma.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].dof.len();
        for i_dof in 0..n_dof {
            greens_pf_passive_dof_names.push((
                plasma.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].name.clone(),
                plasma.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].dof[i_dof].name.clone(),
            ));
        }
    }
    for i_time in 0..n_time {
        let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &plasma.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current;
        let n_loop: usize = pf_passive_currents.len();
        // Without any passive loops there is nothing to copy, so an empty `pf_passive_current` is only unset when there are Greens tables
        if n_loop == 0 && n_greens_pf_passive > 0 {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current` is unset: copy the passive loops in from the `pf_passive` IDS with `Plasma.copy_from_pf_passive_ids`"
            );
        }
        let pf_passive_current_names: Vec<String> = plasma.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current(..).name.to_vec();
        if pf_passive_current_names != greens_pf_passive_names {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current(:)/name` is {pf_passive_current_names:?}, but the Greens tables, `greens/grid/pf_passive(:)/name`, are for {greens_pf_passive_names:?}; the two must list the same passive loops, in the same order"
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
                "solve_grad_shafranov: `time_slice({i_time})/constraints/pf_passive_current(:)/dof(:)/name` is {pf_passive_current_dof_names:?}, but the Greens tables, `greens/grid/pf_passive(:)/dof(:)/name`, are for {greens_pf_passive_dof_names:?}; the two must list the same degrees of freedom, in the same order"
            );
        }
    }

    // The magnetic sensors' measurements are read from the equilibrium IDS, `time_slice(itime)/constraints`, rather than from their
    // dynamic data, which is not used
    let (bp_probes_static, _): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        bp_probes.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (flux_loops_static, _): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        flux_loops.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (rogowski_coils_static, _): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        rogowski_coils.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (isoflux_statics, isoflux_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        isoflux.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (isoflux_boundary_statics, isoflux_boundary_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        isoflux_boundary.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (pressure_statics, pressure_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        pressure_sensors.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (stationary_point_statics, stationary_point_dynamic): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) =
        stationary_point.split_into_static_and_dynamic(&times_to_reconstruct, &greens_pf_active_names, &greens_pf_passive_dof_names);
    let (dialoop_statics, _): (Vec<Arc<SensorsStatic>>, Vec<SensorsDynamic>) = dialoop.split_into_static_and_dynamic(&times_to_reconstruct);

    // The magnetic sensors' measurements are read from the equilibrium IDS, which `Plasma.copy_from_magnetics_ids` copied onto every
    // time-slice from the `magnetics` IDS. The solver finds each fitted sensor's measurement by its name, so every fitted sensor must be
    // there, once. Checked here, before anything is solved
    for i_time in 0..n_time {
        let constraints: &EquilibriumConstraints = &plasma.equilibrium_ids.time_slice[i_time].constraints;

        let b_field_pol_probe_names: Vec<String> = constraints.b_field_pol_probe(..).name.to_vec();
        let n_b_field_pol_probe_fitted: usize = bp_probes_static[i_time].names.len();
        for i_b_field_pol_probe_fitted in 0..n_b_field_pol_probe_fitted {
            let fitted_name: &str = &bp_probes_static[i_time].names[i_b_field_pol_probe_fitted];
            let mut n_found: usize = 0;
            for b_field_pol_probe_name in &b_field_pol_probe_names {
                if b_field_pol_probe_name == fitted_name {
                    n_found += 1;
                }
            }
            if n_found != 1 {
                panic!(
                    "solve_grad_shafranov: `time_slice({i_time})/constraints/b_field_pol_probe(:)/name` has {n_found} entries named `{fitted_name}`, which is fitted, but must have one: copy the magnetic sensors in from the `magnetics` IDS with `Plasma.copy_from_magnetics_ids`"
                );
            }
        }

        let flux_loop_names: Vec<String> = constraints.flux_loop(..).name.to_vec();
        let n_flux_loop_fitted: usize = flux_loops_static[i_time].names.len();
        for i_flux_loop_fitted in 0..n_flux_loop_fitted {
            let fitted_name: &str = &flux_loops_static[i_time].names[i_flux_loop_fitted];
            let mut n_found: usize = 0;
            for flux_loop_name in &flux_loop_names {
                if flux_loop_name == fitted_name {
                    n_found += 1;
                }
            }
            if n_found != 1 {
                panic!(
                    "solve_grad_shafranov: `time_slice({i_time})/constraints/flux_loop(:)/name` has {n_found} entries named `{fitted_name}`, which is fitted, but must have one: copy the magnetic sensors in from the `magnetics` IDS with `Plasma.copy_from_magnetics_ids`"
                );
            }
        }

        let rogowski_coil_names: Vec<String> = constraints.rogowski_coil(..).name.to_vec();
        let n_rogowski_coil_fitted: usize = rogowski_coils_static[i_time].names.len();
        for i_rogowski_coil_fitted in 0..n_rogowski_coil_fitted {
            let fitted_name: &str = &rogowski_coils_static[i_time].names[i_rogowski_coil_fitted];
            let mut n_found: usize = 0;
            for rogowski_coil_name in &rogowski_coil_names {
                if rogowski_coil_name == fitted_name {
                    n_found += 1;
                }
            }
            if n_found != 1 {
                panic!(
                    "solve_grad_shafranov: `time_slice({i_time})/constraints/rogowski_coil(:)/name` has {n_found} entries named `{fitted_name}`, which is fitted, but must have one: copy the magnetic sensors in from the `magnetics` IDS with `Plasma.copy_from_magnetics_ids`"
                );
            }
        }

        // The data dictionary has a single diamagnetic flux constraint
        let n_dialoop_fitted: usize = dialoop_statics[i_time].names.len();
        if n_dialoop_fitted > 1 {
            panic!("solve_grad_shafranov: {n_dialoop_fitted} diamagnetic loops are fitted, but `time_slice({i_time})/constraints/diamagnetic_flux` holds one");
        }
        if n_dialoop_fitted == 1 && constraints.diamagnetic_flux.name != dialoop_statics[i_time].names[0] {
            panic!(
                "solve_grad_shafranov: `time_slice({i_time})/constraints/diamagnetic_flux/name` is `{}`, but `{}` is fitted: copy the magnetic sensors in from the `magnetics` IDS with `Plasma.copy_from_magnetics_ids`",
                constraints.diamagnetic_flux.name, dialoop_statics[i_time].names[0]
            );
        }
    }

    // TODO: might be better to combine all sensors here, before passing to the solver

    // Deref the `PyRefMut` once. Every field access on it calls `deref_mut`, which borrows the
    // whole of it, so `&code` and `&greens` could not be held while `time_slice` is mutated;
    // off one `&mut Equilibrium` they are disjoint fields and borrow independently
    let plasma: &mut Plasma = &mut plasma;
    let equilibrium_ids: &mut Equilibrium = &mut plasma.equilibrium_ids;

    let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = plasma.p_prime_source_function.clone();
    let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = plasma.ff_prime_source_function.clone();

    // Loop over time in parallel and store in "results"

    let equilibrium_code: &Code = &equilibrium_ids.code;
    // `vacuum_toroidal_field/b0` is indexed by time, so each time-slice's solve is handed its own value
    let vacuum_toroidal_field_b0: &Array1<f64> = &equilibrium_ids.vacuum_toroidal_field.b0;
    // Geometry only, so the same tables serve every time-slice. Borrowed from a different field
    // of the IDS than `time_slice`, so the parallel solve can hold both at once
    let equilibrium_greens_tables: &EquilibriumGreens = &equilibrium_ids.greens;
    // Geometry only too, and borrowed in the same way
    let equilibrium_limiter: &EquilibriumLimiter = &equilibrium_ids.limiter;
    // The same tables, reorganised into the matrix shapes the per-iteration GEMMs want. This
    // depends only on the geometry, so it is built once here and shared by every time-slice;
    // it used to be rebuilt inside each of the 480 parallel solves
    let psi_and_derivatives_greens: PsiAndDerivativesGreens = PsiAndDerivativesGreens::new(equilibrium_greens_tables);

    // The initial current-density guess. Every input to it is shared between time-slices - the
    // grid, the limiter, and `code/initial_guess` - so it is built once here. It used to be rebuilt
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

        quadratic_current_density_seed(
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
        )
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
                bp_probes_static: &bp_probes_static[i_time],
                flux_loops_static: &flux_loops_static[i_time],
                dialoop_static: &dialoop_statics[i_time],
                rogowski_coils_static: &rogowski_coils_static[i_time],
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
            };

            // Solve
            let mut solver: EquilibriumSolver = EquilibriumSolver::new(
                time_slice,
                equilibrium_code,
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

    // Post-process. This also measures the gaps which `Plasma.copy_from_pulse_schedule_ids` copied onto
    // the time-slices, where the pulse schedule defines any
    equilibrium_post_processor(equilibrium_ids, &p_prime_source_function, &ff_prime_source_function);

    // Get error codes for failed time-slices

    // Calculate sensor values, with the PF coil currents from `constraints/pf_current` and the passive degrees of freedom from
    // `passive_dof_values`. Borrowed rather than cloned: each of these is a separate `PyRefMut`, so borrowing them here does not
    // clash with the `&mut` on the sensor being written
    bp_probes.calculate_sensor_values_rs(plasma);
    flux_loops.calculate_sensor_values_rs(plasma);
    rogowski_coils.calculate_sensor_values_rs(plasma);
    if pressure_sensors.results.data.len() > 0 {
        pressure_sensors.calculate_sensor_values_rust(plasma);
    }
    // The diamagnetic loop depends only on the toroidal flux function `f` (no Green's functions)
    dialoop.calculate_sensor_values_rs(plasma);

    // Calculate chi_sq_mag for each time slice
    let chi_mag: Array1<f64> = epp_chi_sq_mag(&bp_probes, &flux_loops, &rogowski_coils, &dialoop, n_time);
    // The same quantity on the IDS. It is calculated here rather than in the post-processor
    // because it needs the sensors, which the post-processor does not see
    for (i_time, time_slice) in plasma.equilibrium_ids.time_slice.iter_mut().enumerate() {
        time_slice.constraints.chi_squared_reduced = chi_mag[i_time];
    }

    // Record each magnetic sensor's calculated value as the `reconstructed` of its constraint, on every time-slice of the equilibrium IDS,
    // which `Plasma.copy_from_magnetics_ids` filled with every sensor, fitted or not. The diamagnetic flux's is calculated by the equilibrium
    // post-processor instead. The constraints are in the order of the magnetics IDS, so the calculated values are gathered by name first
    let mut b_field_pol_probes_calculated: HashMap<String, Array1<f64>> = HashMap::new();
    for sensor_name in bp_probes.results.keys() {
        let calculated: Array1<f64> = bp_probes.results.get(&sensor_name).get("b").get("calculated").get("value").unwrap_array1();
        b_field_pol_probes_calculated.insert(sensor_name, calculated);
    }
    let mut flux_loops_calculated: HashMap<String, Array1<f64>> = HashMap::new();
    for sensor_name in flux_loops.results.keys() {
        let calculated: Array1<f64> = flux_loops.results.get(&sensor_name).get("psi").get("calculated").get("value").unwrap_array1();
        flux_loops_calculated.insert(sensor_name, calculated);
    }
    let mut rogowski_coils_calculated: HashMap<String, Array1<f64>> = HashMap::new();
    for sensor_name in rogowski_coils.results.keys() {
        let calculated: Array1<f64> = rogowski_coils.results.get(&sensor_name).get("i").get("calculated").get("value").unwrap_array1();
        rogowski_coils_calculated.insert(sensor_name, calculated);
    }
    // A constraint whose sensor has no calculated value is left unset
    for i_time in 0..n_time {
        let constraints: &mut EquilibriumConstraints = &mut plasma.equilibrium_ids.time_slice[i_time].constraints;
        let n_b_field_pol_probe: usize = constraints.b_field_pol_probe.len();
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            let calculated: Option<&Array1<f64>> = b_field_pol_probes_calculated.get(&constraints.b_field_pol_probe[i_b_field_pol_probe].name);
            if let Some(calculated) = calculated {
                constraints.b_field_pol_probe[i_b_field_pol_probe].reconstructed = calculated[i_time];
            }
        }
        let n_flux_loop: usize = constraints.flux_loop.len();
        for i_flux_loop in 0..n_flux_loop {
            let calculated: Option<&Array1<f64>> = flux_loops_calculated.get(&constraints.flux_loop[i_flux_loop].name);
            if let Some(calculated) = calculated {
                constraints.flux_loop[i_flux_loop].reconstructed = calculated[i_time];
            }
        }
        let n_rogowski_coil: usize = constraints.rogowski_coil.len();
        for i_rogowski_coil in 0..n_rogowski_coil {
            let calculated: Option<&Array1<f64>> = rogowski_coils_calculated.get(&constraints.rogowski_coil[i_rogowski_coil].name);
            if let Some(calculated) = calculated {
                constraints.rogowski_coil[i_rogowski_coil].reconstructed = calculated[i_time];
            }
        }
    }

    // Everything after the parallel loop: the per-slice report lines, the post-processors, and
    // recalculating the sensor values. Mostly serial, and it is the second single-core stretch on
    // a CPU-utilisation plot
    println!(
        "solve_grad_shafranov: serial finish done; {:.2}ms",
        timing_start_serial_finish.elapsed().as_secs_f64() * 1e3
    );
}
