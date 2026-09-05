//! `time_slice(itime)/global_quantities/energy_mhd`

use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calculate the plasma's stored (thermal) energy, and store it in the time-slice.
///
/// An isotropic plasma holds `(3 / 2) * p` of thermal energy per unit volume. On a rectangular
/// (R, Z) grid each cell sweeps a torus of volume `2 * pi * R * d_area`, so
///
/// ```text
/// energy_mhd = sum_grid( (3 / 2) * p(R, Z) * 2 * pi * R * d_area )
/// ```
///
/// The pressure is already masked to zero outside the plasma boundary, so summing over the whole
/// grid is an integral over the plasma alone.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `global_quantities/energy_mhd` is written into it
///
/// **Must run after `epp_equilibrium_time_slice_profiles_2d_pressure`**, which fills the pressure
/// this reads. A time-slice which failed to converge carries `NaN` there, so `energy_mhd` comes out
/// `NaN` without needing a special case.
pub fn epp_equilibrium_time_slice_global_quantities_energy_mhd(time_slice: &mut EquilibriumTimeSlice) {
    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let pressure_2d: &Array2<f64> = time_slice.profiles_2d[0].pressure.as_ref().unwrap();
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let d_area: f64 = time_slice.profiles_2d[0].grid.d_area.unwrap();

    let (n_z, n_r): (usize, usize) = pressure_2d.dim();

    let mut energy_mhd: f64 = 0.0;
    for i_r in 0..n_r {
        for i_z in 0..n_z {
            energy_mhd += (3.0 / 2.0) * pressure_2d[(i_z, i_r)] * 2.0 * PI * r[i_r] * d_area;
        }
    }

    time_slice.global_quantities.energy_mhd = Some(energy_mhd);
}
