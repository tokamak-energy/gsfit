use super::epp_chi_sq_mag::epp_chi_sq_mag;
use super::gs_solution::GsSolution;
use crate::coils::Coils;
use crate::passives::Passives;
use crate::plasma::Plasma;
use crate::sensors::{BpProbes, FluxLoops, Isoflux, IsofluxBoundary, MagneticAxis, RogowskiCoils, SensorsDynamic, SensorsStatic};
use crate::source_functions::SourceFunctionTraits;
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2, s};
use numpy::PyArray1;
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[pyfunction]
pub fn solve_grad_shafranov(
    mut plasma: PyRefMut<Plasma>,
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    mut bp_probes: PyRefMut<BpProbes>,
    mut flux_loops: PyRefMut<FluxLoops>,
    mut rogowski_coils: PyRefMut<RogowskiCoils>,
    mut isoflux: PyRefMut<Isoflux>,
    mut isoflux_boundary: PyRefMut<IsofluxBoundary>,
    mut magnetic_axis: PyRefMut<MagneticAxis>,
    times_to_reconstruct: &Bound<'_, PyArray1<f64>>,
    n_iter_max: usize,
    n_iter_min: usize,
    n_iter_no_vertical_feedback: usize,
    gs_error: f64,
    use_anderson_mixing: bool,
    anderson_mixing_from_previous_iter: f64,
) {
    println!("solve_grad_shafranov starting");

    // Convert to rust data type
    let times_to_reconstruct_ndarray: Array1<f64> = Array1::from(unsafe { times_to_reconstruct.as_array() }.to_vec());
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
    let (bp_probes_static, bp_probes_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        bp_probes.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (flux_loops_static, flux_loops_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        flux_loops.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (rogowski_coils_static, rogowski_coils_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        rogowski_coils.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_statics, isoflux_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) = isoflux.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_boundary_statics, isoflux_boundary_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        isoflux_boundary.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);

    let (magnetic_axis_statics, magnetic_axis_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        magnetic_axis.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);

    // TOD: might be better to combine all sensors here, before passing to the solver

    // Create a local copy
    let coils_owned: Coils = coils.to_owned();
    let plasma_owned: Plasma = plasma.to_owned();

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

    // loop over time in parallel and store in "results"
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
                &rogowski_coils_static[i_time],
                &rogowski_coils_dynamic[i_time],
                &isoflux_statics[i_time],
                &isoflux_dynamic[i_time],
                &isoflux_boundary_statics[i_time],
                &isoflux_boundary_dynamic[i_time],
                &magnetic_axis_statics[i_time],
                &magnetic_axis_dynamic[i_time],
                n_iter_max,
                n_iter_min,
                n_iter_no_vertical_feedback,
                gs_error,
                p_prime_source_function.clone(),
                ff_prime_source_function.clone(),
                passive_regularisations.clone(),
                passive_regularisations_weight.clone(),
            );

            // Solve
            gs_object.solve();
            let solution_found: bool = gs_object.ip.is_finite();
            println!(
                "time={:6.1}ms;  solution_found={};  gs_error={};  n_iter={}",
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

    // Post-process
    plasma.equilibrium_post_processor(&mut gs_solutions, &coils_owned, &plasma_owned);
    passives.equilibrium_post_processor(&gs_solutions);

    // Get owned versions for calculating sensor values
    let coils_owned: Coils = coils.to_owned();
    let passives_owned: Passives = passives.to_owned();
    let plasma_owned: Plasma = plasma.to_owned();

    // Calculate sensor values
    bp_probes.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);
    flux_loops.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);
    rogowski_coils.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);

    // Calculate chi_sq_mag for each time slice
    let chi_mag: Array1<f64> = epp_chi_sq_mag(&bp_probes, &flux_loops, &rogowski_coils, n_time);
    plasma.results.get_or_insert("global").insert("chi_mag", chi_mag);
}
