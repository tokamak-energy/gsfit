//! `time_slice(itime)/constraints/b_field_pol_probe(i1)/reconstructed`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::{EquilibriumGreensBFieldPolProbe, EquilibriumTimeSlice};
use ndarray::{Array1, Array2};

/// Calculate the field each poloidal field probe measures in the reconstructed equilibrium, and store it as the `reconstructed` of
/// the probe's constraint.
///
/// Every probe in `constraints/b_field_pol_probe` gets one, whether or not it was fitted. It is the probe's Greens tables,
/// `greens/b_field_pol_probe(i1)`, found by the constraint's name, multiplied by the current in each source: the PF coil currents,
/// `constraints/pf_current(:)/measured`; the passive degrees of freedom, `passive_dof_values`; and the plasma current density,
/// `profiles_2d(0)/j_phi`, over the grid cells.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `constraints/b_field_pol_probe(:)/reconstructed` is written into it
/// * `constant_values` - the constant values; `greens`, the Greens tables, is read
/// * `_intermediate_values` - not used
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // The current in each source. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    let pf_currents: Array1<f64> = time_slice.constraints.pf_current(..).measured.to_array(); // shape = [n_coil]
    let passive_dof_values: &Array1<f64> = &time_slice.passive_dof_values; // shape = [n_dof_total]
    let n_dof_total: usize = passive_dof_values.len();
    let j_phi: &Array2<f64> = &time_slice.profiles_2d[0].j_phi; // shape = [n_z, n_r]
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area;

    let n_b_field_pol_probe: usize = time_slice.constraints.b_field_pol_probe.len();
    let mut reconstructed: Array1<f64> = Array1::from_elem(n_b_field_pol_probe, f64::NAN); // shape = [n_b_field_pol_probe]
    for i_b_field_pol_probe in 0..n_b_field_pol_probe {
        let greens: &EquilibriumGreensBFieldPolProbe = constant_values
            .greens
            .b_field_pol_probe(&time_slice.constraints.b_field_pol_probe[i_b_field_pol_probe].name);

        let field_from_pf_active: f64 = greens.pf_active.dot(&pf_currents);
        let mut field_from_pf_passive: f64 = 0.0;
        for i_dof_total in 0..n_dof_total {
            field_from_pf_passive += greens.pf_passive[i_dof_total] * passive_dof_values[i_dof_total];
        }
        let field_from_plasma: f64 = (&greens.plasma * j_phi).sum() * d_area;

        reconstructed[i_b_field_pol_probe] = field_from_pf_active + field_from_pf_passive + field_from_plasma;
    }

    for i_b_field_pol_probe in 0..n_b_field_pol_probe {
        time_slice.constraints.b_field_pol_probe[i_b_field_pol_probe].reconstructed = reconstructed[i_b_field_pol_probe];
    }
}
