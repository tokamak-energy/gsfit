//! Calculations written to `time_slice(itime)/profiles_1d_r_midplane_h`.
//!
//! One cut along the row through the magnetic axis, on a radial grid refined by
//! `N_R_SUBDIVISIONS_PER_CELL`, so that a normalised flux can be mapped back to a major radius in
//! the scrape-off layer to within a few millimetres. All three nodes are filled by the single
//! calculator here, because they share an abscissa and one sweep of the interpolator.
//!
//! Distinct from `profiles_1d_r_midplane` in both respects: that one cuts the grid's own middle
//! row, `floor(n_z / 2)`, and samples it only where the solver did.
//!
//! `profiles_1d/r_inboard` and `profiles_1d/r_outboard` cut the same row of the same grid, at the
//! same height, with the same bicubic. Reading a major radius off the profile here and reading one
//! off those therefore agree, to whatever accuracy the reader interpolates between the sampled
//! points with.

#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod r__psi__psi_norm;

/// Build the time-slice the calculator is tested against, for a unit test to overwrite the one or
/// two fields it is exercising.
///
/// `psi = 2 * r + 3 * z + 5 * r * z` on a 4-by-3 grid, with its derivatives filled in exactly. A
/// bicubic reproduces a bilinear function exactly, so every interpolated value has a closed form
/// to check against, and the `5 * r * z` term makes a calculator which mixed up the `r` and `z`
/// axes, or dropped the cross derivative, come out wrong.
///
/// `psi_magnetic_axis = 1` and `boundary/psi = 3`, so `psi_norm = (psi - 1) / 2`.
///
/// `profiles_2d/psi_norm` and `profiles_2d/mask` are deliberately set to zero: the calculator must
/// normalise `psi` itself rather than read the masked `psi_norm` the solver leaves behind, which
/// is zero throughout the scrape-off layer.
#[cfg(test)]
pub(super) fn time_slice_for_test() -> imas_rs::EquilibriumTimeSlice {
    use imas_rs::{EquilibriumProfiles2d, EquilibriumTimeSlice};
    use ndarray::{Array1, Array2, array};

    let r: Array1<f64> = array![0.2, 0.4, 0.6, 0.8];
    let z: Array1<f64> = array![-0.5, 0.0, 0.5];
    let n_r: usize = r.len();
    let n_z: usize = z.len();

    let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_z in 0..n_z {
        for i_r in 0..n_r {
            psi_2d[(i_z, i_r)] = psi_for_test(r[i_r], z[i_z]);
            d_psi_d_r_2d[(i_z, i_r)] = 2.0 + 5.0 * z[i_z];
            d_psi_d_z_2d[(i_z, i_r)] = 3.0 + 5.0 * r[i_r];
            d2_psi_d_r_d_z_2d[(i_z, i_r)] = 5.0;
        }
    }

    let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
    profiles_2d.grid.dim1 = r;
    profiles_2d.grid.dim2 = z;
    profiles_2d.psi = psi_2d;
    profiles_2d.d_psi_d_r = d_psi_d_r_2d;
    profiles_2d.d_psi_d_z = d_psi_d_z_2d;
    profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;
    profiles_2d.psi_norm = Array2::zeros((n_z, n_r));
    profiles_2d.mask = Array2::zeros((n_z, n_r));

    let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
    time_slice.global_quantities.magnetic_axis.z = 0.25;
    time_slice.global_quantities.psi_magnetic_axis = 1.0;
    time_slice.boundary.psi = 3.0;
    time_slice.profiles_2d = vec![profiles_2d];

    return time_slice;
}

/// The analytic `psi` the test fixture is built from, [weber]
#[cfg(test)]
pub(super) fn psi_for_test(r: f64, z: f64) -> f64 {
    return 2.0 * r + 3.0 * z + 5.0 * r * z;
}
