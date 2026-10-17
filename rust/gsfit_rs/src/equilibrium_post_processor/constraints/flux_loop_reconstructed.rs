//! `time_slice(itime)/constraints/flux_loop(i1)/reconstructed`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::{EquilibriumGreensFluxLoop, EquilibriumTimeSlice};
use ndarray::{Array1, Array2};

/// Calculate the poloidal flux each flux loop measures in the reconstructed equilibrium, and store it as the `reconstructed` of the
/// loop's constraint.
///
/// Every flux loop in `constraints/flux_loop` gets one, whether or not it was fitted. It is the loop's Greens tables,
/// `greens/flux_loop(i1)`, found by the constraint's name, multiplied by the current in each source: the PF coil currents,
/// `constraints/pf_current(:)/measured`; the passive degrees of freedom, `passive_dof_values`; and the plasma current density,
/// `profiles_2d(0)/j_phi`, over the grid cells.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/flux_loop(:)/reconstructed` is written into it
/// * `constant_values` - the constant values; `greens`, the Greens tables, is read
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // The current in each source. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    let pf_currents: Array1<f64> = time_slice.constraints.pf_current(..).measured.to_array(); // shape = [n_coil]
    let passive_dof_values: &Array1<f64> = &time_slice.passive_dof_values; // shape = [n_dof_total]
    let n_dof_total: usize = passive_dof_values.len();
    let j_phi: &Array2<f64> = &time_slice.profiles_2d[0].j_phi; // shape = [n_z, n_r]
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area;

    let n_flux_loop: usize = time_slice.constraints.flux_loop.len();
    let mut reconstructed: Array1<f64> = Array1::from_elem(n_flux_loop, f64::NAN); // shape = [n_flux_loop]
    for i_flux_loop in 0..n_flux_loop {
        let greens: &EquilibriumGreensFluxLoop = constant_values.greens.flux_loop(&time_slice.constraints.flux_loop[i_flux_loop].name);

        let flux_from_pf_active: f64 = greens.pf_active.dot(&pf_currents);
        let mut flux_from_pf_passive: f64 = 0.0;
        for i_dof_total in 0..n_dof_total {
            flux_from_pf_passive += greens.pf_passive[i_dof_total] * passive_dof_values[i_dof_total];
        }
        let flux_from_plasma: f64 = (&greens.plasma * j_phi).sum() * d_area;

        reconstructed[i_flux_loop] = flux_from_pf_active + flux_from_pf_passive + flux_from_plasma;
    }

    for i_flux_loop in 0..n_flux_loop {
        time_slice.constraints.flux_loop[i_flux_loop].reconstructed = reconstructed[i_flux_loop];
    }
}
