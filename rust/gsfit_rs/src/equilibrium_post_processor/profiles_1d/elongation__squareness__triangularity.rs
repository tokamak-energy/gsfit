//! `time_slice(itime)/profiles_1d/elongation`, `.../triangularity`, `.../triangularity_upper`,
//! `.../triangularity_lower`, `.../squareness_upper_inner`, `.../squareness_upper_outer`,
//! `.../squareness_lower_inner` and `.../squareness_lower_outer`

use super::super::boundary::geometry::epp_boundary_geometry;
use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use super::q::epp_hessian_matrix;
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;

/// Calculate the elongation, triangularity and squareness of every flux surface, and store them in
/// the time-slice.
///
/// Each surface is measured exactly as the plasma boundary is - the same `epp_boundary_geometry` is
/// called on each contour in turn - so the last point of each profile is the corresponding
/// `boundary` scalar, by construction rather than by coincidence.
///
/// The eight quantities are filled together because they are all properties of the contour alone,
/// and one pass over each contour produces all of them.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; the eight `profiles_1d` nodes above are written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
///
/// # The magnetic axis and missing surfaces
///
/// A surface which could not be traced is left as NaN, exactly as the enclosed volume is.
///
/// The magnetic axis is a point rather than a contour. `psi` is quadratic about it, so the flux
/// surfaces tend to ellipses in that limit, and the seven shape parameters which measure the
/// departure from an ellipse are set to zero there: triangularity is an up-down asymmetry an
/// ellipse does not have, and the squareness is defined to be zero for an ellipse.
///
/// The elongation of that limiting ellipse is not zero: it is finite, and set by the curvature of
/// `psi` at the axis, so it is taken from that curvature rather than from a contour. See
/// `epp_elongation_at_magnetic_axis`.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();

    let mut elongation_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut triang_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut triang_l_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut triang_u_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut square_l_i_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut square_l_o_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut square_u_i_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    let mut square_u_o_profile: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);

    // A slice which did not converge has no flux surfaces to measure
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    if psi_a.is_nan() {
        store(
            time_slice,
            elongation_profile,
            triang_profile,
            triang_l_profile,
            triang_u_profile,
            square_l_i_profile,
            square_l_o_profile,
            square_u_i_profile,
            square_u_o_profile,
        );
        return;
    }

    // The surfaces tend to ellipses at the magnetic axis; see the note in this function's
    // documentation. The elongation is filled after the loop, because it is extrapolated from the
    // surfaces the loop measures
    triang_profile[0] = 0.0;
    triang_l_profile[0] = 0.0;
    triang_u_profile[0] = 0.0;
    square_l_i_profile[0] = 0.0;
    square_l_o_profile[0] = 0.0;
    square_u_i_profile[0] = 0.0;
    square_u_o_profile[0] = 0.0;

    // Don't do the first point
    for i_psi_norm in 1..n_psi_norm {
        let fs_r: &Array1<f64> = &flux_surfaces[i_psi_norm].r;
        let fs_z: &Array1<f64> = &flux_surfaces[i_psi_norm].z;

        // A flux surface which could not be found is stored with zero points, which
        // `epp_boundary_geometry` returns NaN for
        let (elongation, triang, triang_l, triang_u, square_l_i, square_l_o, square_u_i, square_u_o): (f64, f64, f64, f64, f64, f64, f64, f64) =
            epp_boundary_geometry(fs_r, fs_z);

        elongation_profile[i_psi_norm] = elongation;
        triang_profile[i_psi_norm] = triang;
        triang_l_profile[i_psi_norm] = triang_l;
        triang_u_profile[i_psi_norm] = triang_u;
        square_l_i_profile[i_psi_norm] = square_l_i;
        square_l_o_profile[i_psi_norm] = square_l_o;
        square_u_i_profile[i_psi_norm] = square_u_i;
        square_u_o_profile[i_psi_norm] = square_u_o;
    }

    elongation_profile[0] = epp_elongation_at_magnetic_axis(time_slice);

    store(
        time_slice,
        elongation_profile,
        triang_profile,
        triang_l_profile,
        triang_u_profile,
        square_l_i_profile,
        square_l_o_profile,
        square_u_i_profile,
        square_u_o_profile,
    );
}

/// Calculate the elongation on the magnetic axis, from the curvature of `psi` there.
///
/// The flux surface degenerates to a point at the axis, so there is no contour to measure. `psi` is
/// quadratic about the axis though, which fixes the shape of the surfaces in that limit:
///
/// ```text
/// psi - psi_axis = (1 / 2) * (psi_rr * d_r ** 2 + 2 * psi_rz * d_r * d_z + psi_zz * d_z ** 2)
/// ```
///
/// A contour of that is an ellipse whose bounding box is `2 * sqrt(2 * k * psi_zz / determinant)`
/// wide and `2 * sqrt(2 * k * psi_rr / determinant)` tall, so the elongation the data dictionary
/// asks for - height over width - is
///
/// ```text
/// elongation = sqrt(psi_rr / psi_zz)
/// ```
///
/// The cross term cancels out of the ratio, so a tilted axis region is handled without a special
/// case. `psi_rr` and `psi_zz` share a sign at an extremum, so the ratio is positive.
///
/// # References
///
/// That the surfaces are ellipses set by the second derivatives of `psi` is standard near-axis
/// theory, going back to C. Mercier, "Equilibrium and stability of a toroidal magnetohydrodynamic
/// system in the neighbourhood of a magnetic axis", Nuclear Fusion 4 (1964) 213,
/// doi:10.1088/0029-5515/4/3/008. The quadratic form above, written in tokamak `(R, Z)`
/// coordinates and with the same cross term, is equation (1) of J. Ball and F. I. Parra,
/// "Intuition for the radial penetration of flux surface shaping in tokamaks", Plasma Phys.
/// Control. Fusion 57 (2015) 035006, doi:10.1088/0741-3335/57/3/035006.
///
/// Neither states `sqrt(psi_rr / psi_zz)` as such; that is the two lines of algebra given above,
/// and it is worth checking rather than taking on trust. `the_axis_elongation_comes_from_the_
/// curvature_of_psi` does exactly that, against an analytically quadratic `psi`.
///
/// # Consistency with `q` on the axis
///
/// `epp_q_axis` solves the same problem - the flux surface degenerates to a point, so the quantity
/// has to come from the local shape of `psi` - and it uses the same Hessian. The two agree by
/// construction: substituting `psi_rr = elongation ** 2 * psi_zz` into what it computes gives
///
/// ```text
/// |trace| / sqrt(determinant) = (psi_rr + psi_zz) / sqrt(psi_rr * psi_zz) = (1 + elongation ** 2) / elongation
/// ```
///
/// which turns its expression into the textbook on-axis safety factor,
/// `q_axis = (1 + elongation ** 2) / (2 * elongation) * 2 * b0 / (mu_0 * r0 * j_phi_axis)`.
///
/// # Why not extrapolate
///
/// Extrapolating from the innermost traced surfaces is much worse: those contours are only a few
/// grid cells across, so their measured elongation is the noisiest part of the profile, and
/// extrapolating that noise backwards can even give a negative answer.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// * `elongation_axis` - the elongation on the magnetic axis, or NaN when the axis sits on the edge
///   of the grid, where the Hessian cannot be differenced [dimensionless]
fn epp_elongation_at_magnetic_axis(time_slice: &EquilibriumTimeSlice) -> f64 {
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;

    let Some((hessian_matrix, _hessian_determinant, _hessian_trace)) = epp_hessian_matrix(time_slice, mag_r, mag_z) else {
        return f64::NAN;
    };

    let psi_rr: f64 = hessian_matrix[(0, 0)];
    let psi_zz: f64 = hessian_matrix[(1, 1)];

    (psi_rr / psi_zz).sqrt()
}

/// Write the eight profiles into the time-slice.
#[allow(clippy::too_many_arguments)]
fn store(
    time_slice: &mut EquilibriumTimeSlice,
    elongation_profile: Array1<f64>,
    triang_profile: Array1<f64>,
    triang_l_profile: Array1<f64>,
    triang_u_profile: Array1<f64>,
    square_l_i_profile: Array1<f64>,
    square_l_o_profile: Array1<f64>,
    square_u_i_profile: Array1<f64>,
    square_u_o_profile: Array1<f64>,
) {
    time_slice.profiles_1d.elongation = elongation_profile;
    time_slice.profiles_1d.triangularity = triang_profile;
    time_slice.profiles_1d.triangularity_lower = triang_l_profile;
    time_slice.profiles_1d.triangularity_upper = triang_u_profile;
    time_slice.profiles_1d.squareness_lower_inner = square_l_i_profile;
    time_slice.profiles_1d.squareness_lower_outer = square_l_o_profile;
    time_slice.profiles_1d.squareness_upper_inner = square_u_i_profile;
    time_slice.profiles_1d.squareness_upper_outer = square_u_o_profile;
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumProfiles2d;
    use ndarray::{Array2, array};
    use std::f64::consts::PI;

    /// A closed elliptical contour, as `flux_surfaces::calculate` would return it
    fn elliptical_flux_surface(r_geo: f64, z_geo: f64, r_minor: f64, kappa: f64) -> FluxSurface {
        // `n_theta` divisible by 4, so that the extremal points are exactly on the contour
        let n_theta: usize = 1000;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);

        return FluxSurface {
            r: r_geo + r_minor * theta.mapv(f64::cos),
            z: z_geo + kappa * r_minor * theta.mapv(f64::sin),
        };
    }

    /// An empty flux surface, as `flux_surfaces::calculate` stores one it could not trace
    fn untraced_flux_surface() -> FluxSurface {
        return FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };
    }

    /// A time-slice whose `psi` is exactly quadratic about `(0.5, 0.0)`, so that the elongation at
    /// the magnetic axis is `sqrt(psi_rr / psi_zz) = kappa_axis` analytically.
    ///
    /// A cross term is included, because the elongation must not depend on it
    fn time_slice_with_quadratic_psi(kappa_axis: f64) -> EquilibriumTimeSlice {
        let mag_r: f64 = 0.5;
        let mag_z: f64 = 0.0;
        let psi_zz: f64 = 3.0;
        let psi_rr: f64 = psi_zz * kappa_axis.powi(2);
        let psi_rz: f64 = 0.7;

        let n_r: usize = 41;
        let n_z: usize = 61;
        let r: Array1<f64> = Array1::linspace(0.2, 0.9, n_r);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_z);

        // `psi` is stored as `[n_z, n_r]`
        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - mag_r;
                let delta_z: f64 = z[i_z] - mag_z;
                psi_2d[(i_z, i_r)] = 0.5 * (psi_rr * delta_r.powi(2) + 2.0 * psi_rz * delta_r * delta_z + psi_zz * delta_z.powi(2));
                d_psi_d_r_2d[(i_z, i_r)] = psi_rr * delta_r + psi_rz * delta_z;
                d_psi_d_z_2d[(i_z, i_r)] = psi_rz * delta_r + psi_zz * delta_z;
            }
        }

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.psi_magnetic_axis = 0.0;
        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        time_slice.profiles_2d[0].grid.dim1 = r;
        time_slice.profiles_2d[0].grid.dim2 = z;
        time_slice.profiles_2d[0].psi = psi_2d;
        time_slice.profiles_2d[0].d_psi_d_r = d_psi_d_r_2d;
        time_slice.profiles_2d[0].d_psi_d_z = d_psi_d_z_2d;
        time_slice.profiles_2d[0].d2_psi_d_r2 = Array2::from_elem((n_z, n_r), psi_rr);
        time_slice.profiles_2d[0].d2_psi_d_r_d_z = Array2::from_elem((n_z, n_r), psi_rz);
        time_slice.profiles_2d[0].d2_psi_d_z2 = Array2::from_elem((n_z, n_r), psi_zz);

        return time_slice;
    }

    #[test]
    fn elliptical_surfaces_have_their_elongation_and_zero_triangularity_and_squareness() {
        let kappa: f64 = 1.7;

        let mut time_slice: EquilibriumTimeSlice = time_slice_with_quadratic_psi(kappa);
        time_slice.profiles_1d.psi_norm = array![0.0, 0.5, 1.0];

        // The magnetic axis, one surface which could not be traced, and one ellipse
        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), untraced_flux_surface(), elliptical_flux_surface(0.5, 0.02, 0.3, kappa)];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        let elongation_profile: Array1<f64> = time_slice.profiles_1d.elongation;
        let triang_profile: Array1<f64> = time_slice.profiles_1d.triangularity;
        let square_u_o_profile: Array1<f64> = time_slice.profiles_1d.squareness_upper_outer;

        // The magnetic axis is filled with the elliptical limit, and the elongation of that
        // ellipse comes from the curvature of `psi` rather than from any traced contour
        assert_abs_diff_eq!(triang_profile[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(square_u_o_profile[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(elongation_profile[0], kappa, epsilon = 1e-9);

        // A surface which could not be traced stays NaN
        assert!(elongation_profile[1].is_nan());
        assert!(triang_profile[1].is_nan());
        assert!(square_u_o_profile[1].is_nan());

        // An ellipse has zero triangularity, and is what the squareness is measured against
        assert_abs_diff_eq!(elongation_profile[2], kappa, epsilon = 1e-6);
        assert_abs_diff_eq!(triang_profile[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(square_u_o_profile[2], 0.0, epsilon = 1e-3);
    }

    /// The axis elongation must not care what the traced surfaces around it look like, and must
    /// not care about the cross term in the Hessian either
    #[test]
    fn the_axis_elongation_comes_from_the_curvature_of_psi() {
        for kappa_axis in [1.0, 1.5, 2.4] {
            let mut time_slice: EquilibriumTimeSlice = time_slice_with_quadratic_psi(kappa_axis);
            time_slice.profiles_1d.psi_norm = array![0.0, 0.25, 0.5, 1.0];

            // Deliberately nothing like `kappa_axis`, to show the axis value does not come from here
            let flux_surfaces: Vec<FluxSurface> = vec![
                untraced_flux_surface(),
                elliptical_flux_surface(0.5, 0.0, 0.1, 0.4),
                elliptical_flux_surface(0.5, 0.0, 0.2, 0.6),
                elliptical_flux_surface(0.5, 0.0, 0.3, 0.9),
            ];

            let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
            intermediate_values.flux_surfaces = flux_surfaces;
            calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

            let elongation_profile: Array1<f64> = time_slice.profiles_1d.elongation;

            // `psi` is exactly quadratic, so the differenced Hessian is exact too
            assert_abs_diff_eq!(elongation_profile[0], kappa_axis, epsilon = 1e-9);
            // Always a real, positive elongation, which extrapolating from those surfaces was not
            assert!(elongation_profile[0] > 0.0);
        }
    }

    /// Miller parameterisation: `r = r_geo + r_minor * cos(theta + arcsin(delta) * sin(theta))`.
    /// The top point is at `theta = pi / 2`, where `r = r_geo - r_minor * delta`, so the
    /// triangularity is `delta` exactly
    #[test]
    fn a_shaped_surface_gives_its_elongation_and_triangularity() {
        let r_geo: f64 = 0.9;
        let r_minor: f64 = 0.6;
        let kappa: f64 = 2.2;
        let delta: f64 = 0.4;

        let n_theta: usize = 1000;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI * (1.0 - 1.0 / (n_theta as f64)), n_theta);
        let flux_surface: FluxSurface = FluxSurface {
            r: r_geo + r_minor * theta.mapv(|theta_here| (theta_here + delta.asin() * theta_here.sin()).cos()),
            z: kappa * r_minor * theta.mapv(f64::sin),
        };

        let mut time_slice: EquilibriumTimeSlice = time_slice_with_quadratic_psi(1.6);
        time_slice.profiles_1d.psi_norm = array![0.0, 1.0];

        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), flux_surface];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        assert_abs_diff_eq!(time_slice.profiles_1d.elongation[1], kappa, epsilon = 1e-6);
        assert_abs_diff_eq!(time_slice.profiles_1d.triangularity[1], delta, epsilon = 1e-6);
        assert_abs_diff_eq!(time_slice.profiles_1d.triangularity_lower[1], delta, epsilon = 1e-6);
        assert_abs_diff_eq!(time_slice.profiles_1d.triangularity_upper[1], delta, epsilon = 1e-6);
    }

    #[test]
    fn a_slice_which_did_not_converge_is_all_nan() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.5, 1.0];
        time_slice.global_quantities.psi_magnetic_axis = f64::NAN;

        let flux_surfaces: Vec<FluxSurface> = vec![untraced_flux_surface(), untraced_flux_surface(), untraced_flux_surface()];

        let mut intermediate_values: IntermediateValues = intermediate_values_for_test();
        intermediate_values.flux_surfaces = flux_surfaces;
        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values);

        // Including the magnetic axis, which is only filled once there is a converged solution
        assert!(time_slice.profiles_1d.elongation.iter().all(|value| value.is_nan()));
        assert!(time_slice.profiles_1d.triangularity.iter().all(|value| value.is_nan()));
        assert!(time_slice.profiles_1d.squareness_lower_inner.iter().all(|value| value.is_nan()));
    }
}
