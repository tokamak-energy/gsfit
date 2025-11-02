use crate::sensors::{BpProbes, FluxLoops, RogowskiCoils};
use ndarray::{Array1, Array2};

pub fn epp_chi_sq_mag(bp_probes: &BpProbes, flux_loops: &FluxLoops, rogowski_coils: &RogowskiCoils, n_time: usize) -> Array1<f64> {
    // Loop over all time-slices and calculate `chi_sq_mag`
    let mut chi_sq_mag_result: Array1<f64> = Array1::zeros(n_time);

    let bp_probe_names: Vec<String> = bp_probes.results.keys();
    let n_bp_probes: usize = bp_probe_names.len();
    if n_bp_probes > 0 {
        let bp_probes_measured: Array2<f64> = bp_probes.results.get("*").get("b").get("measured").unwrap_array2();
        let bp_probes_calculated: Array2<f64> = bp_probes.results.get("*").get("b").get("calculated").unwrap_array2();
        let bp_probes_weight: Array1<f64> = bp_probes.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let bp_probes_expected_value: Array1<f64> = bp_probes.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let bp_probes_include: Vec<bool> = bp_probes.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
        for i_time in 0..n_time {
            // bp_probes
            for i_bp_probe in 0..n_bp_probes {
                if bp_probes_include[i_bp_probe] == true {
                    let sigma: f64 = bp_probes_expected_value[i_bp_probe] / bp_probes_weight[i_bp_probe];
                    chi_sq_mag_result[i_time] +=
                        (bp_probes_measured[(i_time, i_bp_probe)] - bp_probes_calculated[(i_time, i_bp_probe)]).powi(2) / sigma.powi(2);
                }
            }
        }
    }

    let flux_loop_names: Vec<String> = flux_loops.results.keys();
    let n_flux_loops: usize = flux_loop_names.len();
    if n_flux_loops > 0 {
        let flux_loops_measured: Array2<f64> = flux_loops.results.get("*").get("psi").get("measured").unwrap_array2();
        let flux_loops_calculated: Array2<f64> = flux_loops.results.get("*").get("psi").get("calculated").unwrap_array2();
        let flux_loops_weight: Array1<f64> = flux_loops.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let flux_loops_expected_value: Array1<f64> = flux_loops.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let flux_loops_include: Vec<bool> = flux_loops.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
        for i_time in 0..n_time {
            // flux_loops
            for i_flux_loop in 0..n_flux_loops {
                if flux_loops_include[i_flux_loop] == true {
                    let sigma: f64 = flux_loops_expected_value[i_flux_loop] / flux_loops_weight[i_flux_loop];
                    chi_sq_mag_result[i_time] +=
                        (flux_loops_measured[(i_time, i_flux_loop)] - flux_loops_calculated[(i_time, i_flux_loop)]).powi(2) / sigma.powi(2);
                }
            }
        }
    }

    let rogowski_coil_names: Vec<String> = rogowski_coils.results.keys();
    let n_rogowski_coils: usize = rogowski_coil_names.len();
    if n_rogowski_coils > 0 {
        let rogowski_coils_measured: Array2<f64> = rogowski_coils.results.get("*").get("i").get("measured").unwrap_array2();
        let rogowski_coils_calculated: Array2<f64> = rogowski_coils.results.get("*").get("i").get("calculated").unwrap_array2();
        let rogowski_coils_weight: Array1<f64> = rogowski_coils.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let rogowski_coils_expected_value: Array1<f64> = rogowski_coils.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let rogowski_coils_include: Vec<bool> = rogowski_coils.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();
        for i_time in 0..n_time {
            // rogowski_coils
            for i_rogowski_coil in 0..n_rogowski_coils {
                if rogowski_coils_include[i_rogowski_coil] == true {
                    let sigma: f64 = rogowski_coils_expected_value[i_rogowski_coil] / rogowski_coils_weight[i_rogowski_coil];
                    chi_sq_mag_result[i_time] +=
                        (rogowski_coils_measured[(i_time, i_rogowski_coil)] - rogowski_coils_calculated[(i_time, i_rogowski_coil)]).powi(2) / sigma.powi(2);
                }
            }
        }
    }

    return chi_sq_mag_result;
}
