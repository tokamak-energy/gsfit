use core::f64;
use log::info; // use log::{debug, error, info};
use ndarray::{Array1, Array2, s};
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use source_functions::SourceFunctionTraits;
use std::sync::Arc;
use std::time::{Duration, Instant};
mod sensors;
pub use sensors::{BpProbes, Dialoop, FluxLoops, Isoflux, IsofluxBoundary, Pressure, RogowskiCoils};
pub use sensors::{SensorsDynamic, SensorsStatic};
mod grad_shafranov;
mod plasma_geometry;
pub use grad_shafranov::GsSolution;
pub mod greens;
mod nested_dict;
pub use nested_dict::NestedDict;
mod plasma;
pub use plasma::Plasma;
mod passives;
pub use passives::Passives;
mod coils;
pub use coils::Coils;
mod circuit_equations;
use circuit_equations::solve_circuit_equations;
mod source_functions;
use source_functions::{EfitPolynomial, LiuqePolynomial};

mod bicubic_interpolator;

// Global constants
const PI: f64 = std::f64::consts::PI;

/// A Python module implemented in Rust; bindings added here
#[pymodule]
fn gsfit_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (r, z, r_prime, z_prime, d_r=None, d_z=None))]
    fn greens_py(
        py: Python,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        r_prime: &Bound<'_, PyArray1<f64>>,
        z_prime: &Bound<'_, PyArray1<f64>>,
        d_r: Option<&Bound<'_, PyArray1<f64>>>,
        d_z: Option<&Bound<'_, PyArray1<f64>>>,
    ) -> Py<PyArray2<f64>> {
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let r_prime_ndarray: Array1<f64> = Array1::from(unsafe { r_prime.as_array() }.to_vec());
        let z_prime_ndarray: Array1<f64> = Array1::from(unsafe { z_prime.as_array() }.to_vec());

        // Some horible variable type change and fallback when option not supplied
        let n_prime: usize = r_prime_ndarray.len();
        let d_r_fallback: Array1<f64> = Array1::from_elem(n_prime, f64::NAN);
        let d_r_fallback_py: &Bound<'_, PyArray1<f64>> = &d_r_fallback.into_pyarray(py).into();
        let d_r_unwrapped: &Bound<'_, PyArray1<f64>> = d_r.unwrap_or(d_r_fallback_py);
        let d_r: Array1<f64> = Array1::from(unsafe { d_r_unwrapped.as_array() }.to_vec());
        // d_z
        let n_prime: usize = r_prime_ndarray.len();
        let d_z_fallback: Array1<f64> = Array1::from_elem(n_prime, f64::NAN);
        let d_z_fallback_py: &Bound<'_, PyArray1<f64>> = &d_z_fallback.into_pyarray(py).into();
        let d_z_unwrapped: &Bound<'_, PyArray1<f64>> = d_z.unwrap_or(d_z_fallback_py);
        let d_z: Array1<f64> = Array1::from(unsafe { d_z_unwrapped.as_array() }.to_vec());

        let g: Array2<f64> = greens::greens(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r, d_z);

        return g.into_pyarray(py).into();
    }

    #[pyfn(m)]
    fn d_greens_dz_py(
        py: Python,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        r_prime: &Bound<'_, PyArray1<f64>>,
        z_prime: &Bound<'_, PyArray1<f64>>,
    ) -> Py<PyArray2<f64>> {
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let r_prime_ndarray: Array1<f64> = Array1::from(unsafe { r_prime.as_array() }.to_vec());
        let z_prime_ndarray: Array1<f64> = Array1::from(unsafe { z_prime.as_array() }.to_vec());

        let (_g_br, g_bz): (Array2<f64>, Array2<f64>) = greens::greens_magnetic_field(r_ndarray, z_ndarray, r_prime_ndarray.clone(), z_prime_ndarray);

        let result: Array2<f64> = -2.0 * PI * r_prime_ndarray * g_bz;

        return result.into_pyarray(py).into();
    }

    #[pyfn(m)]
    fn d_greens_magnetic_field_dz_py(
        py: Python,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        r_prime: &Bound<'_, PyArray1<f64>>,
        z_prime: &Bound<'_, PyArray1<f64>>,
    ) -> (Py<PyArray2<f64>>, Py<PyArray2<f64>>) {
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let r_prime_ndarray: Array1<f64> = Array1::from(unsafe { r_prime.as_array() }.to_vec());
        let z_prime_ndarray: Array1<f64> = Array1::from(unsafe { z_prime.as_array() }.to_vec());

        let (d_g_br_dz, d_g_bz_dz): (Array2<f64>, Array2<f64>) = greens::d_greens_magnetic_field_dz(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray);

        return (d_g_br_dz.into_pyarray(py).into(), d_g_bz_dz.into_pyarray(py).into());
    }

    #[pyfn(m)]
    fn greens_magnetic_field_py(
        py: Python,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        r_prime: &Bound<'_, PyArray1<f64>>,
        z_prime: &Bound<'_, PyArray1<f64>>,
    ) -> (Py<PyArray2<f64>>, Py<PyArray2<f64>>) {
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let r_prime_ndarray: Array1<f64> = Array1::from(unsafe { r_prime.as_array() }.to_vec());
        let z_prime_ndarray: Array1<f64> = Array1::from(unsafe { z_prime.as_array() }.to_vec());

        let (g_br, g_bz): (Array2<f64>, Array2<f64>) = greens::greens_magnetic_field(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray);

        return (g_br.into_pyarray(py).into(), g_bz.into_pyarray(py).into());
    }

    m.add_function(wrap_pyfunction!(solve_inverse_problem, m)?)?;
    m.add_function(wrap_pyfunction!(solve_circuit_equations, m)?)?;

    #[pyfn(m)]
    fn mutual_inductance_finite_size_to_finite_size_py(
        py: Python,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        d_r: &Bound<'_, PyArray1<f64>>,
        d_z: &Bound<'_, PyArray1<f64>>,
        angle1: &Bound<'_, PyArray1<f64>>,
        angle2: &Bound<'_, PyArray1<f64>>,
        r_prime: &Bound<'_, PyArray1<f64>>,
        z_prime: &Bound<'_, PyArray1<f64>>,
        d_r_prime: &Bound<'_, PyArray1<f64>>,
        d_z_prime: &Bound<'_, PyArray1<f64>>,
        angle1_prime: &Bound<'_, PyArray1<f64>>,
        angle2_prime: &Bound<'_, PyArray1<f64>>,
    ) -> Py<PyArray2<f64>> {
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let d_r_ndarray: Array1<f64> = Array1::from(unsafe { d_r.as_array() }.to_vec());
        let d_z_ndarray: Array1<f64> = Array1::from(unsafe { d_z.as_array() }.to_vec());
        let angle1_ndarray: Array1<f64> = Array1::from(unsafe { angle1.as_array() }.to_vec());
        let angle2_ndarray: Array1<f64> = Array1::from(unsafe { angle2.as_array() }.to_vec());
        let r_prime_ndarray: Array1<f64> = Array1::from(unsafe { r_prime.as_array() }.to_vec());
        let z_prime_ndarray: Array1<f64> = Array1::from(unsafe { z_prime.as_array() }.to_vec());
        let d_r_prime_ndarray: Array1<f64> = Array1::from(unsafe { d_r_prime.as_array() }.to_vec());
        let d_z_prime_ndarray: Array1<f64> = Array1::from(unsafe { d_z_prime.as_array() }.to_vec());
        let angle1_prime_ndarray: Array1<f64> = Array1::from(unsafe { angle1_prime.as_array() }.to_vec());
        let angle2_prime_ndarray: Array1<f64> = Array1::from(unsafe { angle2_prime.as_array() }.to_vec());

        let g: Array2<f64> = greens::mutual_inductance_finite_size_to_finite_size(
            &r_ndarray,
            &z_ndarray,
            &d_r_ndarray,
            &d_z_ndarray,
            &angle1_ndarray,
            &angle2_ndarray,
            &r_prime_ndarray,
            &z_prime_ndarray,
            &d_r_prime_ndarray,
            &d_z_prime_ndarray,
            &angle1_prime_ndarray,
            &angle2_prime_ndarray,
        );

        return g.into_pyarray(py).into();
    }

    m.add_class::<BpProbes>()?;
    m.add_class::<Coils>()?;
    m.add_class::<Dialoop>()?;
    m.add_class::<FluxLoops>()?;
    m.add_class::<Isoflux>()?;
    m.add_class::<IsofluxBoundary>()?;
    m.add_class::<Passives>()?;
    m.add_class::<Plasma>()?;
    m.add_class::<RogowskiCoils>()?;
    m.add_class::<Pressure>()?;

    m.add_class::<EfitPolynomial>()?;
    m.add_class::<LiuqePolynomial>()?;

    Ok(())
}

#[pyfunction]
fn solve_inverse_problem(
    mut plasma: PyRefMut<Plasma>,
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    mut bp_probes: PyRefMut<BpProbes>,
    mut flux_loops: PyRefMut<FluxLoops>,
    mut rogowski_coils: PyRefMut<RogowskiCoils>,
    mut isoflux: PyRefMut<Isoflux>,
    mut isoflux_boundary: PyRefMut<IsofluxBoundary>,
    times_to_reconstruct: &Bound<'_, PyArray1<f64>>,
    n_iter_max: usize,
    n_iter_min: usize,
    n_iter_no_vertical_feedback: usize,
    gs_error: f64,
    use_anderson_mixing: bool,
    anderson_mixing_from_previous_iter: f64,
) {
    // Start logger
    // env_logger::init();
    println!("solve_inverse_problem starting");

    // Convert to rust data type
    let times_to_reconstruct_ndarray: Array1<f64> = Array1::from(unsafe { times_to_reconstruct.as_array() }.to_vec());
    let n_time: usize = times_to_reconstruct_ndarray.len();

    plasma.results.insert("time", times_to_reconstruct_ndarray.clone());

    // Import rust implementation
    // let plasma_owned: Plasma = plasma.clone(); // .clone() is a bit expensive

    // Get static and dynamic data
    let coils_dynamic: Vec<SensorsDynamic> = coils.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (bp_probes_static, bp_probes_dynamic): (SensorsStatic, Vec<SensorsDynamic>) = bp_probes.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (flux_loops_static, flux_loops_dynamic): (SensorsStatic, Vec<SensorsDynamic>) = flux_loops.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (rogowski_coils_static, rogowski_coils_dynamic): (SensorsStatic, Vec<SensorsDynamic>) =
        rogowski_coils.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_statics, isoflux_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) = isoflux.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);
    let (isoflux_boundary_statics, isoflux_boundary_dynamic): (Vec<SensorsStatic>, Vec<SensorsDynamic>) =
        isoflux_boundary.split_into_static_and_dynamic(&times_to_reconstruct_ndarray);

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
                &bp_probes_static,
                &bp_probes_dynamic[i_time],
                &flux_loops_static,
                &flux_loops_dynamic[i_time],
                &rogowski_coils_static,
                &rogowski_coils_dynamic[i_time],
                &isoflux_statics[i_time],
                &isoflux_dynamic[i_time],
                &isoflux_boundary_statics[i_time],
                &isoflux_boundary_dynamic[i_time],
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
                "time={:6.1}ms;  solution_found={};  gs_error={}",
                times_to_reconstruct_ndarray[i_time] * 1e3,
                solution_found,
                gs_object.gs_error_calculated
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

    // Get owned versions for caclulating sensor values
    let coils_owned: Coils = coils.to_owned();
    let passives_owned: Passives = passives.to_owned();
    let plasma_owned: Plasma = plasma.to_owned();

    // Calculate sensor values
    bp_probes.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);
    flux_loops.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);
    rogowski_coils.calculate_sensor_values_rust(&coils_owned, &passives_owned, &plasma_owned);

    let chi_mag: Array1<f64> = epp_chi_sq_mag(&bp_probes, &flux_loops, &rogowski_coils, n_time);

    plasma.results.get_or_insert("global").insert("chi_mag", chi_mag);
}

pub fn epp_chi_sq_mag(bp_probes: &BpProbes, flux_loops: &FluxLoops, rogowski_coils: &RogowskiCoils, n_time: usize) -> Array1<f64> {
    let bp_probe_names: Vec<String> = bp_probes.results.keys();
    let bp_probes_measured: Array2<f64> = bp_probes.results.get("*").get("b").get("measured").unwrap_array2();
    let bp_probes_calculated: Array2<f64> = bp_probes.results.get("*").get("b").get("calculated").unwrap_array2();
    let bp_probes_weight: Array1<f64> = bp_probes.results.get("*").get("fit_settings").get("weight").unwrap_array1();
    let bp_probes_expected_value: Array1<f64> = bp_probes.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
    let bp_probes_include: Vec<bool> = bp_probes.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
    let n_bp_probes: usize = bp_probe_names.len();

    let flux_loop_names: Vec<String> = flux_loops.results.keys();
    let flux_loops_measured: Array2<f64> = flux_loops.results.get("*").get("psi").get("measured").unwrap_array2();
    let flux_loops_calculated: Array2<f64> = flux_loops.results.get("*").get("psi").get("calculated").unwrap_array2();
    let flux_loops_weight: Array1<f64> = flux_loops.results.get("*").get("fit_settings").get("weight").unwrap_array1();
    let flux_loops_expected_value: Array1<f64> = flux_loops.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
    let flux_loops_include: Vec<bool> = flux_loops.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
    let n_flux_loops: usize = flux_loop_names.len();

    let rogowski_coil_names: Vec<String> = rogowski_coils.results.keys();
    let rogowski_coils_measured: Array2<f64> = rogowski_coils.results.get("*").get("i").get("measured").unwrap_array2();
    let rogowski_coils_calculated: Array2<f64> = rogowski_coils.results.get("*").get("i").get("calculated").unwrap_array2();
    let rogowski_coils_weight: Array1<f64> = rogowski_coils.results.get("*").get("fit_settings").get("weight").unwrap_array1();
    let rogowski_coils_expected_value: Array1<f64> = rogowski_coils.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
    let rogowski_coils_include: Vec<bool> = rogowski_coils.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
    let n_rogowski_coils: usize = rogowski_coil_names.len();

    // Loop over all time-slices and calculate chi_sq_mag
    let mut chi_sq_mag_result: Array1<f64> = Array1::zeros(n_time);
    for i_time in 0..n_time {
        // bp_probes
        for i_bp_probe in 0..n_bp_probes {
            if bp_probes_include[i_bp_probe] == true {
                let sigma: f64 = bp_probes_expected_value[i_bp_probe] / bp_probes_weight[i_bp_probe];
                chi_sq_mag_result[i_time] += (bp_probes_measured[[i_time, i_bp_probe]] - bp_probes_calculated[[i_time, i_bp_probe]]).powi(2) / sigma.powi(2);
            }
        }

        // flux_loops
        for i_flux_loop in 0..n_flux_loops {
            if flux_loops_include[i_flux_loop] == true {
                let sigma: f64 = flux_loops_expected_value[i_flux_loop] / flux_loops_weight[i_flux_loop];
                chi_sq_mag_result[i_time] +=
                    (flux_loops_measured[[i_time, i_flux_loop]] - flux_loops_calculated[[i_time, i_flux_loop]]).powi(2) / sigma.powi(2);
            }
        }

        // rogowski_coils
        for i_rogowski_coil in 0..n_rogowski_coils {
            if rogowski_coils_include[i_rogowski_coil] == true {
                let sigma: f64 = rogowski_coils_expected_value[i_rogowski_coil] / rogowski_coils_weight[i_rogowski_coil];
                chi_sq_mag_result[i_time] +=
                    (rogowski_coils_measured[[i_time, i_rogowski_coil]] - rogowski_coils_calculated[[i_time, i_rogowski_coil]]).powi(2) / sigma.powi(2);
            }
        }
    }

    return chi_sq_mag_result;
}
