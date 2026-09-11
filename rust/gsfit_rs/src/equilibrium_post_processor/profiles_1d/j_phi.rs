//! `time_slice(itime)/profiles_1d/j_phi`

use super::super::constant_values::ConstantValues;
use super::super::flux_surface_average;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Calculate the flux-surface-averaged toroidal current density.
///
/// The data dictionary defines this profile as `<<j_phi / R>> / <<1 / R>>`. The shared
/// `1 / b_p` denominator in those two standard flux-surface averages cancels, leaving an
/// inverse-radius-weighted average of the 2-D toroidal current density.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;
    let j_phi_2d: &Array2<f64> = &time_slice.profiles_2d[0].j_phi;
    let j_phi: Array1<f64> = flux_surface_average::calculate(time_slice, flux_surfaces, j_phi_2d, |r_here| 1.0 / r_here);
    time_slice.profiles_1d.j_phi = j_phi;
}
