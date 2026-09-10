//! Calculations written to `time_slice(itime)/profiles_1d_r_midplane`.
//!
//! Every profile here is a cut along the middle row of the (R, Z) grid, `floor(n_z / 2)`, rather
//! than the row nearest the magnetic axis, so these are cuts through the grid rather than through
//! the plasma. They all share one abscissa: `profiles_1d_r_midplane/r`, which
//! `pressure::calculate` writes.

pub(super) mod dpressure_dpsi;
pub(super) mod f;
pub(super) mod f_df_dpsi;
pub(super) mod j_phi;
pub(super) mod pressure;
pub(super) mod q;

#[cfg(test)]
use imas_rs::EquilibriumTimeSlice;

/// Build the time-slice the mid-plane calculators are tested against, for a unit test to overwrite
/// the one or two fields the calculator under test actually reads.
///
/// A converged time-slice on a 4-by-3 grid, whose mid-plane row is the one every calculator must
/// pick out. `floor(n_z / 2) = 1`, so that is the middle row; the rows either side hold values
/// which no correct answer contains, so that a calculator taking the wrong row fails.
///
/// Along the mid-plane row: `psi_norm = [0.0, 0.25, 0.5, 0.0]` with `mask = [0, 1, 1, 0]`, so the
/// two middle points are inside the plasma and the two ends are outside, carrying the zeroed
/// `psi_norm` the solver leaves there. `psi_magnetic_axis = 0` and `boundary/psi = -1` make
/// `d(psi)/d(psi_norm) = -1`.
#[cfg(test)]
pub(super) fn time_slice_for_test() -> EquilibriumTimeSlice {
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::{Array2, array};

    let psi_norm_2d: Array2<f64> = array![[0.9, 0.9, 0.9, 0.9], [0.0, 0.25, 0.5, 0.0], [0.9, 0.9, 0.9, 0.9]];
    let mask_2d: Array2<f64> = array![[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]];
    let j_phi_2d: Array2<f64> = array![[99.0, 99.0, 99.0, 99.0], [0.0, 3.0, 4.0, 0.0], [99.0, 99.0, 99.0, 99.0]];

    let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
    profiles_2d.grid.dim1 = Some(array![0.2, 0.4, 0.6, 0.8]);
    profiles_2d.grid.dim2 = Some(array![-0.5, 0.0, 0.5]);
    profiles_2d.psi_norm = Some(psi_norm_2d);
    profiles_2d.mask = Some(mask_2d);
    profiles_2d.j_phi = Some(j_phi_2d);

    let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
    time_slice.global_quantities.psi_magnetic_axis = Some(0.0);
    time_slice.boundary.psi = Some(-1.0);
    time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.5, 1.0]);
    time_slice.profiles_1d.q = Some(array![1.0, 2.0, 8.0]);
    time_slice.profiles_2d = vec![profiles_2d];

    return time_slice;
}
