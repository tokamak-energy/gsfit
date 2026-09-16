//! `time_slice(itime)/profiles_1d/j_parallel`

use super::super::constant_values::ConstantValues;
use super::super::flux_surface_average;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the flux-surface-averaged approximation to parallel current density.
///
/// `profiles_2d(0)/j_parallel` already contains `j.B / B0`, so this is its standard
/// `1 / b_p`-weighted flux-surface average.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let j_parallel_2d: &Array2<f64> = &time_slice.profiles_2d[0].j_parallel;
    let j_parallel: Array1<f64> = flux_surface_average::calculate(time_slice, flux_surfaces, j_parallel_2d, |_r_here| 1.0);
    time_slice.profiles_1d.j_parallel = j_parallel;
}
