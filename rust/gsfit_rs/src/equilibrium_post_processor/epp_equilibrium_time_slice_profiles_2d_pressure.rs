//! `time_slice(itime)/profiles_2d(0)/pressure`

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, array};

/// Calculate the plasma pressure on the grid in the poloidal plane, and store it in the time-slice.
///
/// Ideal MHD equilibrium makes the pressure a flux function, `p = p(psi)`, so the 2D field is the
/// 1D profile evaluated at the normalised flux of each grid point:
///
/// ```text
/// p(R, Z) = (psi_b - psi_a) * integral_1^{psi_norm(R, Z)} p'(psi_norm') d(psi_norm')
/// ```
///
/// where `(psi_b - psi_a)` is `d(psi)/d(psi_norm)`, converting the source function's integral from
/// normalised to physical flux. The mask zeroes everything outside the plasma boundary, where
/// `p(psi)` has no meaning.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/pressure` is written into it
/// * `p_prime_source_function` - the p' source function the reconstruction was run with
///
/// A time-slice which failed to converge carries `NaN` in `psi_norm`, `mask`, `boundary/psi` and
/// `global_quantities/psi_magnetic_axis`, so the arithmetic below fills `pressure` with `NaN`
/// without needing a special case.
pub fn epp_equilibrium_time_slice_profiles_2d_pressure(time_slice: &mut EquilibriumTimeSlice, p_prime_source_function: &SharedSourceFunction) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let psi_norm_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();
    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();

    let (n_z, n_r): (usize, usize) = psi_norm_2d.dim();

    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    let psi_b: f64 = time_slice.boundary.psi.unwrap();

    // p = (d(psi)/d(psi_norm)) * integral_1^{psi_norm} p'(psi_norm') d(psi_norm'),
    // where d(psi)/d(psi_norm) = psi_b - psi_a
    let d_psi_d_psi_norm: f64 = psi_b - psi_a;

    let p_prime_dof_values: &Array1<f64> = time_slice.source_functions.p_prime.coefficients.as_ref().unwrap();

    let mut pressure_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            let psi_norm: f64 = psi_norm_2d[(i_z, i_r)];

            // TODO: evaluated one grid point at a time, which allocates twice per point. The source
            // functions take a whole array, so the entire grid could go through in one call
            let pressure_local_ndarray: Array1<f64> = p_prime_source_function.source_function_integral(&array![psi_norm], p_prime_dof_values);
            let pressure_local: f64 = pressure_local_ndarray[0];

            // Apply the mask, and store the pressure
            pressure_2d[(i_z, i_r)] = pressure_local * mask_2d[(i_z, i_r)] * d_psi_d_psi_norm;
        }
    }

    time_slice.profiles_2d[0].pressure = Some(pressure_2d);
}
