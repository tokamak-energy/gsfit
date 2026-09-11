//! `time_slice(itime)/profiles_1d_r_midplane/r` and `.../pressure`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, array};

/// Calculate the pressure along the mid-plane, and store it in the time-slice.
///
/// The row used is `floor(n_z / 2)`, the middle row of the grid, rather than the row nearest the
/// magnetic axis, so this is a cut through the grid rather than through the plasma.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `profiles_1d_r_midplane` nodes are written into it
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let p_prime_source_function: &SharedSourceFunction = constant_values.p_prime_source_function;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
    let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
    let psi_n_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi_norm;
    let mask_2d: &Array2<f64> = &time_slice.profiles_2d[0].mask;

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;

    let p_prime_dof_values: &Array1<f64> = &time_slice.source_functions.p_prime.coefficients;

    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;

    let mut p_profile: Array1<f64> = Array1::from_elem(n_r, f64::NAN);

    // p = (dψ/dψ_N) · ∫_1^{ψ_N} p′(ψ_N′) dψ_N′,  where  dψ/dψ_N = ψ_B − ψ_A
    // See `profiles_1d::pressure::calculate` for the full derivation.
    let d_psi_d_psi_n: f64 = psi_b - psi_a;

    // TODO: change this to a slice
    for i_r in 0..n_r {
        let psi_n_here: f64 = psi_n_2d[(i_z_centre, i_r)];

        let pressure_local: f64 = p_prime_source_function.source_function_integral(&array![psi_n_here], p_prime_dof_values)[0];

        // Apply the mask, and store pressure
        p_profile[i_r] = pressure_local * mask_2d[(i_z_centre, i_r)] * d_psi_d_psi_n;
    }

    time_slice.profiles_1d_r_midplane.r = r.to_owned();
    time_slice.profiles_1d_r_midplane.pressure = p_profile;
}
