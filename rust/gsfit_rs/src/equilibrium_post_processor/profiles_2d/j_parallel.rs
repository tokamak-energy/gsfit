//! `time_slice(itime)/profiles_2d(0)/j_parallel`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the parallel current density on the grid in the poloidal plane, and store it in the
/// time-slice.
///
/// The data dictionary defines this as `j . b / b0`, with `b0` the vacuum toroidal field at the
/// machine's reference radius. It is deliberately **not** the component of `j` along `b`, which
/// would divide by `|b|` rather than by a fixed field; the data dictionary says so itself.
///
/// The toroidal part of `j . b` is `j_phi * b_phi`, both of which are already on the grid. The
/// poloidal part follows from the toroidal field function alone: `b_phi = f(psi) / r` and
/// `curl(b) = mu_0 * j` give
///
/// ```text
/// mu_0 * j_pol = 2 * pi * f'(psi) * b_pol
/// ```
///
/// where the `2 * pi` is GSFit's convention of `psi` being the total flux rather than the flux per
/// radian. So
///
/// ```text
/// j . b = 2 * pi * f'(psi) * |b_pol| ** 2 / mu_0 + j_phi * b_phi
/// ```
///
/// with `f'` recovered from the fitted source function as `ff'(psi) / f`, and `f = r * b_phi`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_2d(0)/j_parallel` is written into it
/// * `constant_values` - the constant values; `b0` and `ff_prime_source_function` are read
///
/// The mask zeroes the poloidal part outside the plasma boundary, where `ff'(psi)` has no meaning;
/// `j_phi` is already zero there, so `j_parallel` comes out zero too. A time-slice which failed to
/// converge carries `NaN` in `j_phi` and in the fields, so `j_parallel` comes out `NaN` without
/// needing a special case.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    let b0: f64 = constant_values.b0;
    let ff_prime_source_function: &SharedSourceFunction = constant_values.ff_prime_source_function;

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let mesh_r_2d: &Array2<f64> = time_slice.profiles_2d[0].r.as_ref().unwrap();
    let psi_norm_2d: &Array2<f64> = time_slice.profiles_2d[0].psi_norm.as_ref().unwrap();
    let mask_2d: &Array2<f64> = time_slice.profiles_2d[0].mask.as_ref().unwrap();
    let j_phi_2d: &Array2<f64> = time_slice.profiles_2d[0].j_phi.as_ref().unwrap();
    let b_field_r_2d: &Array2<f64> = time_slice.profiles_2d[0].b_field_r.as_ref().unwrap();
    let b_field_z_2d: &Array2<f64> = time_slice.profiles_2d[0].b_field_z.as_ref().unwrap();
    let b_field_phi_2d: &Array2<f64> = time_slice.profiles_2d[0].b_field_phi.as_ref().unwrap();

    let (n_z, n_r): (usize, usize) = psi_norm_2d.dim();

    let ff_prime_dof_values: &Array1<f64> = time_slice.source_functions.ff_prime.coefficients.as_ref().unwrap();
    let psi_norm_flat: Array1<f64> = Array1::from_iter(psi_norm_2d.iter().cloned());
    let ff_prime_2d: Array2<f64> = ff_prime_source_function
        .source_function_value(&psi_norm_flat, ff_prime_dof_values)
        .to_shape((n_z, n_r))
        .unwrap()
        .to_owned();

    let mut j_parallel_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            let b_field_phi: f64 = b_field_phi_2d[(i_z, i_r)];
            let b_pol_squared: f64 = b_field_r_2d[(i_z, i_r)].powi(2) + b_field_z_2d[(i_z, i_r)].powi(2);

            let f: f64 = mesh_r_2d[(i_z, i_r)] * b_field_phi;
            let f_prime: f64 = ff_prime_2d[(i_z, i_r)] * mask_2d[(i_z, i_r)] / f;

            let j_dot_b: f64 = 2.0 * PI * f_prime * b_pol_squared / MU_0 + j_phi_2d[(i_z, i_r)] * b_field_phi;

            j_parallel_2d[(i_z, i_r)] = j_dot_b / b0;
        }
    }

    time_slice.profiles_2d[0].j_parallel = Some(j_parallel_2d);
}
