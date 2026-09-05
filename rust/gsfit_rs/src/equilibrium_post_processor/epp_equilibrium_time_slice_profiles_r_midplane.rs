//! `time_slice(itime)/profiles_r_midplane/r` and `.../pressure`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, array};

/// Calculate the pressure along the mid-plane, and store it in the time-slice.
///
/// The row used is `floor(n_z / 2)`, the middle row of the grid, rather than the row nearest the
/// magnetic axis, so this is a cut through the grid rather than through the plasma.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the two `profiles_r_midplane` nodes are written into it
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
pub fn epp_equilibrium_time_slice_profiles_r_midplane(time_slice: &mut EquilibriumTimeSlice, p_prime_source_function: &SharedSourceFunction) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_n_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();
    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    let p_prime_dof_values: &Array1<f64> = time_slice.source_functions.p_prime.coefficients.as_ref().unwrap();

    let n_r: usize = r.len();
    let n_z: usize = z.len();
    let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;

    let mut p_profile: Array1<f64> = Array1::from_elem(n_r, f64::NAN);

    // p = (dψ/dψ_N) · ∫_1^{ψ_N} p′(ψ_N′) dψ_N′,  where  dψ/dψ_N = ψ_B − ψ_A
    // See `epp_equilibrium_time_slice_profiles_1d_pressure` for the full derivation.
    let d_psi_d_psi_n: f64 = psi_b - psi_a;

    // TODO: change this to a slice
    for i_r in 0..n_r {
        let psi_n_here: f64 = psi_n_2d[(i_z_centre, i_r)];

        let pressure_local: f64 = p_prime_source_function.source_function_integral(&array![psi_n_here], p_prime_dof_values)[0];

        // Apply the mask, and store pressure
        p_profile[i_r] = pressure_local * mask_2d[(i_z_centre, i_r)] * d_psi_d_psi_n;
    }

    time_slice.profiles_r_midplane.r = Some(r.to_owned());
    time_slice.profiles_r_midplane.pressure = Some(p_profile);
}
