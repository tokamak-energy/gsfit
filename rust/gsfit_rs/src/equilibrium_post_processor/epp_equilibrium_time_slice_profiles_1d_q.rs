//! `time_slice(itime)/profiles_1d/q`

use super::epp_flux_surfaces::FluxSurface;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};
use ndarray_interp::interp2d::Interp2D;
use ndarray_stats::QuantileExt;
use std::f64::consts::PI;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Calculate the safety factor profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/q` is written into it
/// * `flux_surfaces` - the flux surfaces from `epp_flux_surfaces`, one per `psi_norm`
///
/// **Must run after `epp_flux_surfaces`**, which supplies the surfaces to integrate around, and
/// after `epp_equilibrium_time_slice_profiles_1d_f`, which supplies `f`.
pub fn epp_equilibrium_time_slice_profiles_1d_q(time_slice: &mut EquilibriumTimeSlice, flux_surfaces: &[FluxSurface]) {
    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.as_ref().unwrap().len();

    // A slice which did not converge has no flux surfaces to integrate around
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis.unwrap();
    if psi_a.is_nan() {
        time_slice.profiles_1d.q = Some(Array1::from_elem(n_psi_norm, f64::NAN));
        return;
    }

    let f_profile: &Array1<f64> = time_slice.profiles_1d.f.as_ref().unwrap();
    let q_profile: Array1<f64> = epp_q_profile(time_slice, flux_surfaces, f_profile);

    time_slice.profiles_1d.q = Some(q_profile);
}

/// Calculate the safety factor profile for an arbitrary `f` profile.
///
/// Kept separate from the writer above because it is evaluated twice: once with the reconstructed
/// `f`, and once with the vacuum `f`, which is what the diamagnetic flux is measured against.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `flux_surfaces` - the flux surfaces from `epp_flux_surfaces`, one per `psi_norm`
/// * `f_profile` - the `f = R * B_phi` profile to integrate with [tesla metre]
///
/// # Returns
/// * `q_profile` - the safety factor profile [dimensionless]
pub(super) fn epp_q_profile(time_slice: &EquilibriumTimeSlice, flux_surfaces: &[FluxSurface], f_profile: &Array1<f64>) -> Array1<f64> {
    // g3 = <1/R**2> = (2.0 / vol_prime) * integral(1 / (Bp * R**2) d_ell)
    // where: vol_prime = d(V)/d(psi)
    // where: <1/R**2> is notation for the flux surface average

    let n_psi_n: usize = flux_surfaces.len();

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi_2d: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap();
    let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
    let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();

    // `b_p = |grad(psi)| / (2 * PI * r)`
    let mesh_r_local: Array2<f64> = Array2::from_shape_fn(psi_2d.dim(), |(_i_z, i_r)| r[i_r]);
    let br: Array2<f64> = -d_psi_d_z_2d / (2.0 * PI * &mesh_r_local);
    let bz: Array2<f64> = d_psi_d_r_2d / (2.0 * PI * &mesh_r_local);
    let bp: Array2<f64> = (br.mapv(|x| x.powi(2)) + bz.mapv(|x| x.powi(2))).mapv(f64::sqrt);

    let bp_interpolator = Interp2D::builder(bp).x(z.clone()).y(r.clone()).build().unwrap();

    // Cumulative integral, so initialise with zeros
    let mut q_profile: Array1<f64> = Array1::zeros(n_psi_n);
    'fs_loop: for i_psi_n in 0..n_psi_n {
        let fs_r: Array1<f64> = flux_surfaces[i_psi_n].r.clone();
        let fs_z: Array1<f64> = flux_surfaces[i_psi_n].z.clone();
        let fs_n: usize = fs_r.len();

        if fs_n < 2 {
            continue 'fs_loop;
        }

        // TODO: temporary fix for invalid LCFS!!
        let invalid_lcfs: bool = fs_z.abs().max().map(|&fs_z_val| fs_z_val > *z.max().unwrap()).unwrap();
        if invalid_lcfs {
            continue 'fs_loop;
        }

        let mut ell: Array1<f64> = Array1::from_elem(fs_n, f64::NAN);
        ell[0] = 0.0;
        for i_fs in 1..fs_n {
            ell[i_fs] = ell[i_fs - 1] + (fs_r[i_fs] - fs_r[i_fs - 1]).hypot(fs_z[i_fs] - fs_z[i_fs - 1]);
        }
        let mut integrand: Array1<f64> = Array1::from_elem(fs_n, f64::NAN);
        // TODO: this **COULD** be wrong because I am calculating the integrand at the boundary point.
        // But the ell variable is not consistent, since it's between boundary points.
        // Look up "midpoint integral approximation" ??
        for i_fs in 0..fs_n {
            let bp_here: f64 = bp_interpolator.interp_scalar(fs_z[i_fs], fs_r[i_fs]).unwrap();

            integrand[i_fs] = f_profile[i_psi_n] / (2.0 * PI * bp_here * fs_r[i_fs].powi(2));
        }

        // Perform the integration
        for i_fs in 1..fs_n {
            q_profile[i_psi_n] += 0.5 * (ell[i_fs] - ell[i_fs - 1]) * (integrand[i_fs] + integrand[i_fs - 1]);
        }
    }

    // Central safety factor
    let q0: f64 = epp_q_axis(time_slice, f_profile);
    q_profile[0] = q0;

    return q_profile;
}

/// Calculate the safety factor on the magnetic axis.
///
/// The flux surface integral above degenerates to a point there, so `q` is instead taken from the
/// curvature of `psi` at the axis:
///
/// `q_axis = abs(tr(H(psiN=0))) / sqrt(det(H(psiN=0))) * f_profile(psiN=0) / (mu0 * r_mag**2 * j_phi)`
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `f_profile` - the `f = R * B_phi` profile to integrate with [tesla metre]
///
/// # Returns
/// * `q_axis` - the safety factor on the magnetic axis [dimensionless]
fn epp_q_axis(time_slice: &EquilibriumTimeSlice, f_profile: &Array1<f64>) -> f64 {
    // TODO: this works ok (ish). I think I will need to do a 2D interpolation for the Hessian matrix
    // I could do this by calculating the Hessian matrix at each point in the grid, and then doing 2D interpolation.
    // Or I could do 2D interpolation on psi, which is used to calculate the Hessian matrix?

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let j_2d: &Array2<f64> = time_slice.profiles_2d[0].j_phi.as_ref().unwrap();

    let r_mag: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let z_mag: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    // Find the nearest point to the magnetic axis
    let mut index_r_mag: usize = 0;
    let mut index_z_mag: usize = 0;
    let mut min_distance: f64 = f64::MAX;
    for i_r in 0..r.len() {
        for i_z in 0..z.len() {
            let distance: f64 = ((r[i_r] - r_mag).powi(2) + (z[i_z] - z_mag).powi(2)).sqrt();
            if distance < min_distance {
                min_distance = distance;
                index_r_mag = i_r;
                index_z_mag = i_z;
            }
        }
    }

    let (_hessian_matrix, hessian_determinant, hessian_trace): (Array2<f64>, f64, f64) = epp_hessian_matrix(time_slice, index_r_mag, index_z_mag);

    let j_phi: f64 = j_2d[(index_z_mag, index_r_mag)];
    let q_axis: f64 = hessian_trace.abs() / hessian_determinant.sqrt() * f_profile[0] / (MU_0 * r_mag.powi(2) * j_phi);

    return q_axis;
}

/// Calculate the Hessian matrix of `psi` at one grid point, by finite differences.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `i_r`, `i_z` - the grid indices to evaluate at
///
/// # Returns
/// * `hessian_matrix` - the 2x2 Hessian matrix [weber per metre ** 2]
/// * `hessian_determinant` - its determinant
/// * `hessian_trace` - its trace
fn epp_hessian_matrix(time_slice: &EquilibriumTimeSlice, i_r: usize, i_z: usize) -> (Array2<f64>, f64, f64) {
    // TODO: Perhaps I should 2D interpolate the Hessian matrix?

    // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
    // ever one entry in this array of structures
    let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
    let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
    let psi: &Array2<f64> = time_slice.profiles_2d[0].psi.as_ref().unwrap(); // shape = (n_z, n_r)

    let d_r: f64 = r[1] - r[0];
    let d_z: f64 = z[1] - z[0];

    let c: f64 = -2.0 * psi[(i_z, i_r)] + psi[(i_z, i_r + 1)] + psi[(i_z, i_r - 1)];
    let d: f64 = -2.0 * psi[(i_z, i_r)] + psi[(i_z + 1, i_r)] + psi[(i_z - 1, i_r)];
    let e: f64 = psi[(i_z, i_r)] - psi[(i_z, i_r + 1)] + psi[(i_z + 1, i_r + 1)] - psi[(i_z + 1, i_r)];

    let mut hessian_matrix: Array2<f64> = Array2::from_elem((2, 2), f64::NAN);
    hessian_matrix[(0, 0)] = c / d_r.powi(2);
    hessian_matrix[(0, 1)] = e / (d_r * d_z);
    hessian_matrix[(1, 0)] = e / (d_r * d_z);
    hessian_matrix[(1, 1)] = d / d_z.powi(2);

    // Calculate determinant and trace (as it's only 2x2 lets not use a library)
    let hessian_determinant: f64 = hessian_matrix[(0, 0)] * hessian_matrix[(1, 1)] - hessian_matrix[(0, 1)] * hessian_matrix[(1, 0)];
    let hessian_trace: f64 = hessian_matrix[(0, 0)] + hessian_matrix[(1, 1)];

    return (hessian_matrix, hessian_determinant, hessian_trace);
}
