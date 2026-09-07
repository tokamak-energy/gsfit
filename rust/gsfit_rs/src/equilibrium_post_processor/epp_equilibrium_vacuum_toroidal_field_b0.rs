//! `vacuum_toroidal_field/b0`

use imas_rs::Equilibrium;
use ndarray::Array1;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the vacuum toroidal field at the reference radius, and store it on the IDS.
///
/// ```text
/// b0 = f_vac / r0,   where  f_vac = mu_0 * i_rod / (2 * pi)
/// ```
///
/// The data dictionary requires `r0 * b0` to equal the `tf` IDS's `b_field_phi_vacuum_r`, and it
/// does here by construction: `r0` is copied from `tf/r0` and `i_rod` is that same signal turned
/// back into a current, both by `solve_grad_shafranov`.
///
/// This is one value per time-slice, but it lives at the root of the IDS rather than inside
/// `time_slice`, so like `v_loop` it runs after the per-slice loop rather than inside it.
///
/// Not to be confused with `global_quantities/bt_vac_at_r_geo`, which is the same vacuum field
/// evaluated at each slice's own geometric axis rather than at the fixed machine reference
/// radius.
///
/// # Arguments
/// * `equilibrium_ids` - the solved equilibrium IDS; `vacuum_toroidal_field/b0` is written into it
///
/// A time-slice which failed to converge still has a rod current, because that is a measurement
/// rather than a reconstructed quantity, so `b0` is filled for every slice.
pub fn epp_equilibrium_vacuum_toroidal_field_b0(equilibrium_ids: &mut Equilibrium) {
    let r0: f64 = equilibrium_ids.vacuum_toroidal_field.r0.unwrap();
    let n_time: usize = equilibrium_ids.time_slice.len();

    let mut b0: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

    for i_time in 0..n_time {
        let i_rod: f64 = equilibrium_ids.time_slice[i_time].global_quantities.i_rod.unwrap();
        let f_vac: f64 = MU_0 * i_rod / (2.0 * PI);
        b0[i_time] = f_vac / r0;
    }

    equilibrium_ids.vacuum_toroidal_field.b0 = Some(b0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumTimeSlice;

    #[test]
    fn b0_times_r0_is_the_vacuum_field_times_major_radius() {
        let r0: f64 = 0.4;
        let i_rod: f64 = -1.2e6;
        let f_vac: f64 = MU_0 * i_rod / (2.0 * PI);

        let mut equilibrium_ids: Equilibrium = Equilibrium::default();
        equilibrium_ids.vacuum_toroidal_field.r0 = Some(r0);
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.i_rod = Some(i_rod);
        equilibrium_ids.time_slice = vec![time_slice];

        epp_equilibrium_vacuum_toroidal_field_b0(&mut equilibrium_ids);

        let b0: Array1<f64> = equilibrium_ids.vacuum_toroidal_field.b0.unwrap();

        // The data dictionary's consistency requirement: `r0 * b0` is the `tf` IDS's
        // `b_field_phi_vacuum_r`
        assert_abs_diff_eq!(r0 * b0[0], f_vac, epsilon = 1e-15);
        // A negative rod current gives a negative field, rather than a magnitude
        assert!(b0[0] < 0.0, "b0 = {}", b0[0]);
    }
}
