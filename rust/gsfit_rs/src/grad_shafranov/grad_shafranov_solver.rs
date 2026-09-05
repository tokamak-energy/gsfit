use super::epp_chi_sq_mag::epp_chi_sq_mag;
use super::equilibrium_solve::GradShafranovInputs;
use super::gs_solution::GsSolution;
use super::{GradShafranovSolve, output_flag};
use crate::coils::Coils;
use crate::passives::Passives;
use crate::plasma::Plasma;
use crate::sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils, SensorsDynamic, SensorsStatic, StationaryPoint};
use crate::source_functions::SourceFunctionTraits;
use crate::wall::Wall;
use imas_rs::ids::wall::Wall as WallIds;
use imas_rs::{Code, Equilibrium, EquilibriumGreens, EquilibriumTimeSlice};
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2, s};
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[pyfunction]
pub fn solve_grad_shafranov(
    mut plasma: PyRefMut<Plasma>,
    wall: PyRef<Wall>,
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
    times_to_reconstruct: PyReadonlyArray1<f64>,
    n_iter_max: usize,
    n_iter_min: usize,
    n_iter_no_vertical_feedback: usize,
    gs_error: f64,
    use_anderson_mixing: bool,
    anderson_mixing_from_previous_iter: f64,
) {
    println!("solve_grad_shafranov starting");

    // Convert to rust data type
    let times_to_reconstruct_ndarray: Array1<f64> = times_to_reconstruct.to_owned_array();
    let n_time: usize = times_to_reconstruct_ndarray.len();

    if n_time == 0 {
        // Store empty `p_prime` and `ff_prime` profiles
        // TODO: all keys should be like this with zero size arrays
        let n_p_prime_dof: usize = plasma.p_prime_source_function.source_function_n_dof();
        let n_ff_prime_dof: usize = plasma.ff_prime_source_function.source_function_n_dof();
        let p_prime_coefs: Array2<f64> = Array2::zeros((n_time, n_p_prime_dof));
        let ff_prime_coefs: Array2<f64> = Array2::zeros((n_time, n_ff_prime_dof));
        plasma
            .results
            .get_or_insert("source_functions")
            .get_or_insert("p_prime")
            .insert("coefficients", p_prime_coefs);
        plasma
            .results
            .get_or_insert("source_functions")
            .get_or_insert("ff_prime")
            .insert("coefficients", ff_prime_coefs);

        println!("solve_grad_shafranov: no times to reconstruct, returning");
        return;
    }

    plasma.results.insert("time", times_to_reconstruct_ndarray.clone());

    // Import rust implementation
    // let plasma_owned: Plasma = plasma.clone(); // .clone() is a bit expensive

    // Get static and dynamic data
    let coils_dynamic: Vec<SensorsDynamic> = coils.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    // TF rod current interpolated to `times_to_reconstruct`; used as f_vac = MU_0 * i_rod / (2 * PI)
    // in the diamagnetic-loop constraint
    let i_rod_vs_time: Array1<f64> = coils.results.get("tf").get("rod_i").get("measured").get("value").unwrap_array1();
    let (bp_probes_static, bp_probes_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        bp_probes.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (flux_loops_static, flux_loops_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        flux_loops.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (rogowski_coils_static, rogowski_coils_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        rogowski_coils.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_statics, isoflux_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) = isoflux.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_boundary_statics, isoflux_boundary_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        isoflux_boundary.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (pressure_statics, pressure_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        pressure_sensors.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (stationary_point_statics, stationary_point_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        stationary_point.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (dialoop_statics, dialoop_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) = dialoop.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);

    // TODO: might be better to combine all sensors here, before passing to the solver

    // Create the Equilibrium IDS with the pre-allocated time slices; data is Null initially.
    // `plasma` owns it, but it is taken out for the duration of the solve: the solver needs
    // `&Plasma` for the grid at the same time as `&mut EquilibriumTimeSlice`, and `&Plasma`
    // borrows the whole struct, so the two would overlap. It is put back below.
    //
    // Taken *before* `plasma` is cloned below, so that the clone does not duplicate the Greens
    // tables, which are by far the largest thing on the IDS
    plasma.initialise_equilibrium_ids(&times_to_reconstruct_ndarray);
    let mut equilibrium_ids: Equilibrium = std::mem::take(&mut plasma.equilibrium_ids);

    // Create a local copy
    let coils_owned: Coils = coils.to_owned();
    let plasma_owned: Plasma = plasma.to_owned();
    // Copied out of the `PyRef`, because the per-time-slice solves run on Rayon's threads and
    // a `PyRef` is neither `Send` nor `Sync`
    let wall_owned: WallIds = wall.wall_ids.clone();

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

    // Settings the solver is run with. These apply to every time-slice, so they live on the IDS
    // itself rather than inside `time_slice`
    equilibrium_ids.code.iterations_n_max = Some(n_iter_max as i32);
    equilibrium_ids.code.iterations_n_min = Some(n_iter_min as i32);
    equilibrium_ids.code.iterations_n_no_vertical_feedback = Some(n_iter_no_vertical_feedback as i32);
    equilibrium_ids.code.grad_shafranov_deviation_value_tolerance = Some(gs_error);

    // Per-time-slice solver inputs, written into the IDS before the solve begins. The TF rod
    // current is a measurement, so it varies from slice to slice
    for i_time in 0..n_time {
        equilibrium_ids.time_slice[i_time].global_quantities.i_rod = Some(i_rod_vs_time[i_time]);
    }

    let equilibrium_code: &Code = &equilibrium_ids.code;
    // Geometry only, so the same tables serve every time-slice. Borrowed from a different field
    // of the IDS than `time_slice`, so the parallel solve can hold both at once
    let greens_tables: &EquilibriumGreens = &equilibrium_ids.greens;

    // Solve the GS equation for all time-slices, in parallel
    let timing_start_ids: Instant = Instant::now();
    equilibrium_ids
        .time_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(i_time, time_slice): (usize, &mut EquilibriumTimeSlice)| {
            // Select the data for this time-slice
            // Note: the GS solver is designed to consider a single time-slice
            // and deliberately does not know what time-slice it is solving
            let grad_shafranov_inputs: GradShafranovInputs = GradShafranovInputs {
                wall: &wall_owned,
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
            time_slice.solve(&grad_shafranov_inputs, equilibrium_code, greens_tables);
        });
    // `code/output_flag` is indexed by time, so it is assembled here rather than by the per-slice
    // solver: 0 for a usable slice, negative for one which failed
    let output_flags: Array1<i32> = Array1::from_iter(equilibrium_ids.time_slice.iter().map(output_flag));
    equilibrium_ids.code.output_flag = Some(output_flags);

    let duration_ids: Duration = timing_start_ids.elapsed();
    info!("GSFit time elapsed (equilibrium IDS solver): {:?}", duration_ids);

    let timing_start: Instant = Instant::now();
    let mut gs_solutions: Vec<GsSolution> = (0..n_time)
        .into_par_iter() // Use Rayon to create a parallel iterator
        .map(|i_time: usize| {
            // Construct GS-Solution object
            // Note: the GS solver is designed to consider a single time-slice
            // and deliberately does not know what time-slice it is solving
            let mut gs_object: GsSolution = GsSolution::new(
                &plasma_owned,
                &coils_dynamic[i_time],
                &bp_probes_static[i_time],
                &bp_probes_dynamic[i_time],
                &flux_loops_static[i_time],
                &flux_loops_dynamic[i_time],
                &dialoop_statics[i_time],
                &dialoop_dynamic[i_time],
                &rogowski_coils_static[i_time],
                &rogowski_coils_dynamic[i_time],
                &isoflux_statics[i_time],
                &isoflux_dynamic[i_time],
                &isoflux_boundary_statics[i_time],
                &isoflux_boundary_dynamic[i_time],
                &pressure_statics[i_time],
                &pressure_dynamic[i_time],
                &stationary_point_statics[i_time],
                &stationary_point_dynamic[i_time],
                n_iter_max,
                n_iter_min,
                n_iter_no_vertical_feedback,
                gs_error,
                i_rod_vs_time[i_time],
                p_prime_source_function.clone(),
                ff_prime_source_function.clone(),
                passive_regularisations.clone(),
                passive_regularisations_weight.clone(),
            );

            // Solve
            gs_object.solve();
            let solution_found: bool = gs_object.ip.is_finite();
            println!(
                "time={:6.1}ms;  solution_found={};  gs_error={:.18};  n_iter={}",
                times_to_reconstruct_ndarray[i_time] * 1e3,
                solution_found,
                gs_object.gs_error_calculated,
                gs_object.n_iter,
            );

            // Return into the Vec
            return gs_object;
        })
        .collect();
    let duration: Duration = timing_start.elapsed();
    info!("GSFit time elapsed: {:?}", duration);

    // // TEMPORARY: compare the new IDS solver against `GsSolution`, time-slice by time-slice
    // {
    //     let max_abs_diff = |a: &Option<Array2<f64>>, b: &Array2<f64>| -> f64 {
    //         let Some(a) = a else { return f64::INFINITY };
    //         if a.shape() != b.shape() {
    //             return f64::INFINITY;
    //         }
    //         let mut worst: f64 = 0.0;
    //         for (value_a, value_b) in a.iter().zip(b.iter()) {
    //             if value_a.is_nan() && value_b.is_nan() {
    //                 continue;
    //             }
    //             let difference: f64 = (value_a - value_b).abs();
    //             if difference > worst {
    //                 worst = difference;
    //             }
    //         }
    //         return worst;
    //     };
    //     let diff_0d = |a: Option<f64>, b: f64| -> f64 {
    //         let Some(a) = a else { return f64::INFINITY };
    //         if a.is_nan() && b.is_nan() { 0.0 } else { (a - b).abs() }
    //     };
    //     for i_time in 0..n_time {
    //         let time_slice = &equilibrium_ids.time_slice[i_time];
    //         let gs = &gs_solutions[i_time];
    //         let profiles_2d = &time_slice.profiles_2d[0];
    //         println!(
    //             "CHECK i_time={i_time}  psi={:.3e} d_psi_d_r={:.3e} d_psi_d_z={:.3e} d2r2={:.3e} d2rz={:.3e} d2z2={:.3e} psi_n={:.3e} j={:.3e} mask={:.3e} psi_coils={:.3e} | ip={:.3e} psi_a={:.3e} psi_b={:.3e} r_mag={:.3e} z_mag={:.3e} n_iter={}/{} gs_err={:.3e} n_nodes={} | delta_z={:.3e} bounding_r={:.3e} bounding_z={:.3e} boundary_type={}/{} | result={} output_flag={}",
    //             max_abs_diff(&profiles_2d.psi, &gs.psi_2d),
    //             max_abs_diff(&profiles_2d.d_psi_d_r, &gs.d_psi_d_r_2d),
    //             max_abs_diff(&profiles_2d.d_psi_d_z, &gs.d_psi_d_z_2d),
    //             max_abs_diff(&profiles_2d.d2_psi_d_r2, &gs.d2_psi_d_r2_2d),
    //             max_abs_diff(&profiles_2d.d2_psi_d_r_d_z, &gs.d2_psi_d_r_d_z_2d),
    //             max_abs_diff(&profiles_2d.d2_psi_d_z2, &gs.d2_psi_d_z2_2d),
    //             max_abs_diff(&profiles_2d.psi_norm, &gs.psi_n_2d),
    //             max_abs_diff(&profiles_2d.j_phi, &gs.j_2d),
    //             max_abs_diff(&profiles_2d.mask, &gs.mask),
    //             max_abs_diff(&profiles_2d.psi_coils, &gs.psi_2d_coils),
    //             diff_0d(time_slice.global_quantities.ip, gs.ip),
    //             diff_0d(time_slice.global_quantities.psi_magnetic_axis, gs.psi_a),
    //             diff_0d(time_slice.boundary.psi, gs.psi_b),
    //             diff_0d(time_slice.global_quantities.magnetic_axis.r, gs.r_mag),
    //             diff_0d(time_slice.global_quantities.magnetic_axis.z, gs.z_mag),
    //             time_slice.convergence.iterations_n.map(|n| n.to_string()).unwrap_or("unset".to_string()),
    //             gs.n_iter,
    //             diff_0d(time_slice.convergence.grad_shafranov_deviation_value, gs.gs_error_calculated),
    //             time_slice.contour_tree.node.len(),
    //             diff_0d(time_slice.convergence.delta_z, gs.delta_z),
    //             diff_0d(time_slice.boundary.bounding.r, gs.bounding_r),
    //             diff_0d(time_slice.boundary.bounding.z, gs.bounding_z),
    //             time_slice.boundary.r#type.map(|t| t.to_string()).unwrap_or("unset".to_string()),
    //             gs.xpt_diverted,
    //             time_slice.convergence.result.name.clone().unwrap_or("unset".to_string()),
    //             equilibrium_ids
    //                 .code
    //                 .output_flag
    //                 .as_ref()
    //                 .map(|f| f[i_time].to_string())
    //                 .unwrap_or("unset".to_string()),
    //         );
    //     }
    // }

    // Post-process.
    //
    // Two post-processors run side by side while the post-processing is moved off `GsSolution` and
    // onto the IDS, exactly as the solver itself was moved. `equilibrium_post_processor_new` is
    // the destination and is currently empty; `equilibrium_post_processor` still writes everything
    plasma.equilibrium_post_processor(&mut gs_solutions, &coils_owned, &plasma_owned);
    plasma.equilibrium_post_processor_new(
        &mut equilibrium_ids,
        &coils_owned,
        &wall_owned,
        &p_prime_source_function,
        &ff_prime_source_function,
    );
    passives.equilibrium_post_processor(&gs_solutions);

    // Hand the solved IDS back to `plasma`, which owns it. Done after the post-processing, so that
    // `equilibrium_post_processor_new` can borrow the IDS while `plasma` is borrowed mutably
    plasma.equilibrium_ids = equilibrium_ids;

    // Get error codes for failed time-slices

    // Get owned versions for calculating sensor values
    let coils_owned: Coils = coils.to_owned();
    let passives_owned: Passives = passives.to_owned();
    let plasma_owned: Plasma = plasma.to_owned();

    // Calculate sensor values
    bp_probes.calculate_sensor_values_rs(&coils_owned, &passives_owned, &plasma_owned);
    flux_loops.calculate_sensor_values_rs(&coils_owned, &passives_owned, &plasma_owned);
    rogowski_coils.calculate_sensor_values_rs(&coils_owned, &passives_owned, &plasma_owned);
    if pressure_sensors.results.data.len() > 0 {
        pressure_sensors.calculate_sensor_values_rust(&plasma_owned);
    }
    // The diamagnetic loop depends only on the toroidal flux function `f` (no Green's functions)
    dialoop.calculate_sensor_values_rs(&plasma_owned);

    // Calculate chi_sq_mag for each time slice
    let chi_mag: Array1<f64> = epp_chi_sq_mag(&bp_probes, &flux_loops, &rogowski_coils, &dialoop, n_time);
    plasma.results.get_or_insert("global").insert("chi_mag", chi_mag.to_owned());

    // The same quantity on the IDS. It is calculated here rather than in the post-processor
    // because it needs the sensors, which the post-processor does not see
    for (i_time, time_slice) in plasma.equilibrium_ids.time_slice.iter_mut().enumerate() {
        time_slice.constraints.chi_squared_reduced = Some(chi_mag[i_time]);
    }
}
