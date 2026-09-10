//! `time_slice(itime)/profiles_1d/q`

use super::super::constant_values::ConstantValues;
use super::super::flux_surfaces::FluxSurface;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::{BicubicInterpolator, BicubicValueAndDerivatives};
use crate::plasma_geometry::hessian;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, s};

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;
const GAUSS_LEGENDRE_ABSCISSA: f64 = 0.5773502691896258;

/// Calculate the safety factor profile, and store it in the time-slice.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `profiles_1d/q` is written into it
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is read
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: &[FluxSurface] = &intermediate_values.flux_surfaces;

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
/// On each flux surface,
///
/// ```text
/// q = f / (2 * pi) * integral(d_ell / (r ** 2 * b_p))
///   = f * integral(d_ell / (r * |grad(psi)|)).
/// ```
///
/// The two components of `grad(psi)` are interpolated independently and combined only at each
/// quadrature point. This preserves the vector cancellation near magnetic nulls; interpolating
/// `|grad(psi)|` itself would systematically lose that cancellation. Each straight contour
/// segment is integrated with two-point Gauss-Legendre quadrature.
///
/// The exact `psi_norm = 1` point is deliberately left as NaN. For a diverted plasma the last
/// closed flux surface passes through an X-point, where `b_p = 0`, and its safety factor has a
/// logarithmic singularity. A finite number there would only be an implicit grid-dependent
/// regularisation.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `flux_surfaces` - the flux surfaces from `flux_surfaces::calculate`, one per `psi_norm`
/// * `f_profile` - the `f = R * B_phi` profile to integrate with [tesla metre]
///
/// # Returns
/// * `q_profile` - the safety factor profile [dimensionless]
pub(in crate::equilibrium_post_processor) fn epp_q_profile(
    time_slice: &EquilibriumTimeSlice,
    flux_surfaces: &[FluxSurface],
    f_profile: &Array1<f64>,
) -> Array1<f64> {
    let n_psi_n: usize = flux_surfaces.len();
    let psi_norm: &Array1<f64> = time_slice.profiles_1d.psi_norm.as_ref().unwrap();
    assert_eq!(psi_norm.len(), n_psi_n);
    assert_eq!(f_profile.len(), n_psi_n);

    let psi_gradient_interpolator: PsiGradientInterpolator = PsiGradientInterpolator::new(time_slice);
    let mut q_profile: Array1<f64> = Array1::from_elem(n_psi_n, f64::NAN);
    for i_psi_n in 1..n_psi_n {
        if psi_norm[i_psi_n] == 1.0 {
            continue;
        }

        q_profile[i_psi_n] = q_on_flux_surface(&flux_surfaces[i_psi_n], f_profile[i_psi_n], &psi_gradient_interpolator).unwrap_or(f64::NAN);
    }

    if n_psi_n > 0 && psi_norm[0] != 1.0 {
        q_profile[0] = epp_q_axis(time_slice, f_profile, &psi_gradient_interpolator);
    }

    return q_profile;
}

/// Calculate the safety factor directly on one supplied flux surface.
pub(in crate::equilibrium_post_processor) fn calculate_on_flux_surface(time_slice: &EquilibriumTimeSlice, flux_surface: &FluxSurface, f_here: f64) -> f64 {
    let psi_gradient_interpolator: PsiGradientInterpolator = PsiGradientInterpolator::new(time_slice);
    return q_on_flux_surface(flux_surface, f_here, &psi_gradient_interpolator).unwrap_or(f64::NAN);
}

/// Integrate `q` around one piecewise-linear flux surface.
///
/// The contour geometry is the sequence returned by marching squares. On every straight segment,
/// two-point Gauss-Legendre quadrature evaluates the field at interior points rather than at the
/// grid-edge crossings. For a smooth integrand this is fourth-order along the segment; the overall
/// accuracy can still be limited by representing the curved flux surface with straight chords.
fn q_on_flux_surface(flux_surface: &FluxSurface, f_here: f64, psi_gradient_interpolator: &PsiGradientInterpolator) -> Option<f64> {
    let fs_r: &Array1<f64> = &flux_surface.r;
    let fs_z: &Array1<f64> = &flux_surface.z;
    let n_fs: usize = fs_r.len();
    if n_fs < 4 || fs_z.len() != n_fs || !f_here.is_finite() {
        return None;
    }

    let mut contour_integral: f64 = 0.0;
    let mut compensation: f64 = 0.0;
    for i_fs in 1..n_fs {
        let delta_r: f64 = fs_r[i_fs] - fs_r[i_fs - 1];
        let delta_z: f64 = fs_z[i_fs] - fs_z[i_fs - 1];
        let delta_ell: f64 = delta_r.hypot(delta_z);
        if delta_ell == 0.0 {
            continue;
        }

        let midpoint_r: f64 = 0.5 * (fs_r[i_fs] + fs_r[i_fs - 1]);
        let midpoint_z: f64 = 0.5 * (fs_z[i_fs] + fs_z[i_fs - 1]);
        let offset_r: f64 = 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_r;
        let offset_z: f64 = 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_z;

        let integrand_minus: f64 = q_integrand_at(midpoint_r - offset_r, midpoint_z - offset_z, psi_gradient_interpolator)?;
        let integrand_plus: f64 = q_integrand_at(midpoint_r + offset_r, midpoint_z + offset_z, psi_gradient_interpolator)?;
        let contribution: f64 = 0.5 * delta_ell * (integrand_minus + integrand_plus);

        // Compensated summation keeps round-off below the interpolation and contour errors.
        let contribution_corrected: f64 = contribution - compensation;
        let contour_integral_updated: f64 = contour_integral + contribution_corrected;
        compensation = (contour_integral_updated - contour_integral) - contribution_corrected;
        contour_integral = contour_integral_updated;
    }

    return Some(f_here * contour_integral);
}

/// Evaluate `1 / (r * |grad(psi)|)` at one quadrature point.
fn q_integrand_at(r_here: f64, z_here: f64, psi_gradient_interpolator: &PsiGradientInterpolator) -> Option<f64> {
    let psi_derivatives: PsiDerivatives = psi_gradient_interpolator.interpolate(r_here, z_here)?;
    let grad_psi: f64 = psi_derivatives.d_psi_d_r.hypot(psi_derivatives.d_psi_d_z);
    if r_here <= 0.0 || !grad_psi.is_finite() || grad_psi == 0.0 {
        return None;
    }

    return Some(1.0 / (r_here * grad_psi));
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
fn epp_q_axis(time_slice: &EquilibriumTimeSlice, f_profile: &Array1<f64>, psi_gradient_interpolator: &PsiGradientInterpolator) -> f64 {
    let j_2d: &Array2<f64> = time_slice.profiles_2d[0].j_phi.as_ref().unwrap();

    let r_mag: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
    let z_mag: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();

    let Some((_hessian_matrix, hessian_determinant, hessian_trace)) = psi_gradient_interpolator.hessian_matrix(r_mag, z_mag) else {
        return f64::NAN;
    };
    let Some(j_phi) = psi_gradient_interpolator.interpolate_bilinear(j_2d, r_mag, z_mag) else {
        return f64::NAN;
    };
    if hessian_determinant <= 0.0 || j_phi == 0.0 {
        return f64::NAN;
    }

    let q_axis: f64 = hessian_trace.abs() / hessian_determinant.sqrt() * f_profile[0] / (MU_0 * r_mag.powi(2) * j_phi);

    return q_axis;
}

/// Calculate the Hessian matrix of `psi` at an arbitrary point on the grid.
///
/// The two components of `grad(psi)` are modelled independently with bicubic interpolation, then
/// differentiated at the requested point. This is the same construction used to refine stationary
/// points in the Grad-Shafranov solver and avoids differentiating an interpolated scalar `psi`
/// inside the discretised plasma-current region.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `r_point`, `z_point` - the position to evaluate [metre]
///
/// # Returns
/// * `hessian_matrix` - the 2x2 Hessian matrix [weber per metre ** 2]
/// * `hessian_determinant` - its determinant
/// * `hessian_trace` - its trace
pub(super) fn epp_hessian_matrix(time_slice: &EquilibriumTimeSlice, r_point: f64, z_point: f64) -> Option<(Array2<f64>, f64, f64)> {
    return PsiGradientInterpolator::new(time_slice).hessian_matrix(r_point, z_point);
}

/// Values of `grad(psi)` and the Hessian interpolated at one point.
struct PsiDerivatives {
    d_psi_d_r: f64,
    d_psi_d_z: f64,
    d2_psi_d_r2: f64,
    d2_psi_d_r_d_z: f64,
    d2_psi_d_z2: f64,
}

/// Cell coordinates for one point on the rectangular grid.
struct GridCellCoordinates {
    i_r_left: usize,
    i_z_lower: usize,
    x: f64,
    y: f64,
}

/// Independently bicubic models of `d(psi)/d(r)` and `d(psi)/d(z)` on the solved grid.
///
/// The bicubic model of each component needs that component's value, its two first derivatives,
/// and its mixed second derivative at every grid point. The first derivatives are the stored
/// Hessian of `psi`. The mixed derivatives are third derivatives of `psi`; they are obtained by
/// second-order finite differences of the Hessian in `z`, matching the established stationary-
/// point interpolation without requiring Green's tables at arbitrary `(r, z)` positions.
struct PsiGradientInterpolator<'a> {
    r: &'a Array1<f64>,
    z: &'a Array1<f64>,
    d_psi_d_r_2d: &'a Array2<f64>,
    d_psi_d_z_2d: &'a Array2<f64>,
    d2_psi_d_r2_2d: &'a Array2<f64>,
    d2_psi_d_r_d_z_2d: &'a Array2<f64>,
    d2_psi_d_z2_2d: &'a Array2<f64>,
    d3_psi_d_r2_d_z_2d: Array2<f64>,
    d3_psi_d_r_d_z2_2d: Array2<f64>,
    d_r: f64,
    d_z: f64,
}

impl<'a> PsiGradientInterpolator<'a> {
    fn new(time_slice: &'a EquilibriumTimeSlice) -> Self {
        // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
        let r: &Array1<f64> = time_slice.profiles_2d[0].grid.dim1.as_ref().unwrap();
        let z: &Array1<f64> = time_slice.profiles_2d[0].grid.dim2.as_ref().unwrap();
        let d_psi_d_r_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.as_ref().unwrap();
        let d_psi_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.as_ref().unwrap();
        let d2_psi_d_r2_2d: &Array2<f64> = time_slice.profiles_2d[0].d2_psi_d_r2.as_ref().unwrap();
        let d2_psi_d_r_d_z_2d: &Array2<f64> = time_slice.profiles_2d[0].d2_psi_d_r_d_z.as_ref().unwrap();
        let d2_psi_d_z2_2d: &Array2<f64> = time_slice.profiles_2d[0].d2_psi_d_z2.as_ref().unwrap();

        let n_r: usize = r.len();
        let n_z: usize = z.len();
        assert!(n_r >= 2);
        assert!(n_z >= 3);
        assert_eq!(d_psi_d_r_2d.dim(), (n_z, n_r));
        assert_eq!(d_psi_d_z_2d.dim(), (n_z, n_r));
        assert_eq!(d2_psi_d_r2_2d.dim(), (n_z, n_r));
        assert_eq!(d2_psi_d_r_d_z_2d.dim(), (n_z, n_r));
        assert_eq!(d2_psi_d_z2_2d.dim(), (n_z, n_r));

        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let d3_psi_d_r2_d_z_2d: Array2<f64> = derivative_in_z(d2_psi_d_r2_2d, d_z);
        let d3_psi_d_r_d_z2_2d: Array2<f64> = derivative_in_z(d2_psi_d_r_d_z_2d, d_z);

        return Self {
            r,
            z,
            d_psi_d_r_2d,
            d_psi_d_z_2d,
            d2_psi_d_r2_2d,
            d2_psi_d_r_d_z_2d,
            d2_psi_d_z2_2d,
            d3_psi_d_r2_d_z_2d,
            d3_psi_d_r_d_z2_2d,
            d_r,
            d_z,
        };
    }

    fn interpolate(&self, r_point: f64, z_point: f64) -> Option<PsiDerivatives> {
        let cell: GridCellCoordinates = self.cell_coordinates(r_point, z_point)?;
        let i_r_left: usize = cell.i_r_left;
        let i_z_lower: usize = cell.i_z_lower;

        let d_psi_d_r_interpolator: BicubicInterpolator = BicubicInterpolator::new(
            self.d_r,
            self.d_z,
            self.d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d2_psi_d_r2_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d3_psi_d_r2_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        );
        let d_psi_d_z_interpolator: BicubicInterpolator = BicubicInterpolator::new(
            self.d_r,
            self.d_z,
            self.d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d2_psi_d_z2_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d3_psi_d_r_d_z2_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        );

        let d_psi_d_r: BicubicValueAndDerivatives = d_psi_d_r_interpolator.value_and_derivatives(cell.x, cell.y);
        let d_psi_d_z: BicubicValueAndDerivatives = d_psi_d_z_interpolator.value_and_derivatives(cell.x, cell.y);

        return Some(PsiDerivatives {
            d_psi_d_r: d_psi_d_r.f,
            d_psi_d_z: d_psi_d_z.f,
            d2_psi_d_r2: d_psi_d_r.d_f_d_x / self.d_r,
            d2_psi_d_r_d_z: 0.5 * (d_psi_d_r.d_f_d_y / self.d_z + d_psi_d_z.d_f_d_x / self.d_r),
            d2_psi_d_z2: d_psi_d_z.d_f_d_y / self.d_z,
        });
    }

    fn hessian_matrix(&self, r_point: f64, z_point: f64) -> Option<(Array2<f64>, f64, f64)> {
        let psi_derivatives: PsiDerivatives = self.interpolate(r_point, z_point)?;
        let mut hessian_matrix: Array2<f64> = Array2::from_elem((2, 2), f64::NAN);
        hessian_matrix[(0, 0)] = psi_derivatives.d2_psi_d_r2;
        hessian_matrix[(0, 1)] = psi_derivatives.d2_psi_d_r_d_z;
        hessian_matrix[(1, 0)] = psi_derivatives.d2_psi_d_r_d_z;
        hessian_matrix[(1, 1)] = psi_derivatives.d2_psi_d_z2;

        let (hessian_determinant, hessian_trace): (f64, f64) =
            hessian(psi_derivatives.d2_psi_d_r2, psi_derivatives.d2_psi_d_r_d_z, psi_derivatives.d2_psi_d_z2);
        return Some((hessian_matrix, hessian_determinant, hessian_trace));
    }

    fn interpolate_bilinear(&self, values: &Array2<f64>, r_point: f64, z_point: f64) -> Option<f64> {
        let cell: GridCellCoordinates = self.cell_coordinates(r_point, z_point)?;
        let i_r_left: usize = cell.i_r_left;
        let i_z_lower: usize = cell.i_z_lower;

        let value_lower: f64 = (1.0 - cell.x) * values[(i_z_lower, i_r_left)] + cell.x * values[(i_z_lower, i_r_left + 1)];
        let value_upper: f64 = (1.0 - cell.x) * values[(i_z_lower + 1, i_r_left)] + cell.x * values[(i_z_lower + 1, i_r_left + 1)];
        return Some((1.0 - cell.y) * value_lower + cell.y * value_upper);
    }

    fn cell_coordinates(&self, r_point: f64, z_point: f64) -> Option<GridCellCoordinates> {
        if !r_point.is_finite()
            || !z_point.is_finite()
            || r_point < self.r[0]
            || r_point > self.r[self.r.len() - 1]
            || z_point < self.z[0]
            || z_point > self.z[self.z.len() - 1]
        {
            return None;
        }

        let i_r_left: usize = (((r_point - self.r[0]) / self.d_r).floor() as usize).min(self.r.len() - 2);
        let i_z_lower: usize = (((z_point - self.z[0]) / self.d_z).floor() as usize).min(self.z.len() - 2);
        let x: f64 = ((r_point - self.r[i_r_left]) / self.d_r).clamp(0.0, 1.0);
        let y: f64 = ((z_point - self.z[i_z_lower]) / self.d_z).clamp(0.0, 1.0);

        return Some(GridCellCoordinates { i_r_left, i_z_lower, x, y });
    }
}

/// Differentiate a grid quantity in `z` with second-order finite differences.
fn derivative_in_z(values: &Array2<f64>, d_z: f64) -> Array2<f64> {
    let (n_z, n_r): (usize, usize) = values.dim();
    assert!(n_z >= 3);

    let mut derivative: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
    for i_r in 0..n_r {
        derivative[(0, i_r)] = (-3.0 * values[(0, i_r)] + 4.0 * values[(1, i_r)] - values[(2, i_r)]) / (2.0 * d_z);
        for i_z in 1..n_z - 1 {
            derivative[(i_z, i_r)] = (values[(i_z + 1, i_r)] - values[(i_z - 1, i_r)]) / (2.0 * d_z);
        }
        derivative[(n_z - 1, i_r)] = (3.0 * values[(n_z - 1, i_r)] - 4.0 * values[(n_z - 2, i_r)] + values[(n_z - 3, i_r)]) / (2.0 * d_z);
    }

    return derivative;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::ids::equilibrium::EquilibriumProfiles2d;
    use ndarray::array;
    use std::f64::consts::PI;

    fn circular_flux_surface(r_axis: f64, z_axis: f64, minor_radius: f64) -> FluxSurface {
        let n_theta: usize = 2001;
        let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_theta);
        return FluxSurface {
            r: r_axis + minor_radius * theta.mapv(f64::cos),
            z: z_axis + minor_radius * theta.mapv(f64::sin),
        };
    }

    fn empty_flux_surface() -> FluxSurface {
        return FluxSurface {
            r: Array1::from_elem(0, f64::NAN),
            z: Array1::from_elem(0, f64::NAN),
        };
    }

    fn time_slice_with_circular_psi() -> EquilibriumTimeSlice {
        let r_axis: f64 = 1.0;
        let z_axis: f64 = 0.0;
        let psi_curvature: f64 = 2.5;
        let n_r: usize = 41;
        let n_z: usize = 41;
        let r: Array1<f64> = Array1::linspace(0.5, 1.5, n_r);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_z);

        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_r in 0..n_r {
            for i_z in 0..n_z {
                d_psi_d_r_2d[(i_z, i_r)] = psi_curvature * (r[i_r] - r_axis);
                d_psi_d_z_2d[(i_z, i_r)] = psi_curvature * (z[i_z] - z_axis);
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = Some(r);
        profiles_2d.grid.dim2 = Some(z);
        profiles_2d.d_psi_d_r = Some(d_psi_d_r_2d);
        profiles_2d.d_psi_d_z = Some(d_psi_d_z_2d);
        profiles_2d.d2_psi_d_r2 = Some(Array2::from_elem((n_z, n_r), psi_curvature));
        profiles_2d.d2_psi_d_r_d_z = Some(Array2::zeros((n_z, n_r)));
        profiles_2d.d2_psi_d_z2 = Some(Array2::from_elem((n_z, n_r), psi_curvature));
        profiles_2d.j_phi = Some(Array2::from_elem((n_z, n_r), 1.0e6));

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = Some(r_axis);
        time_slice.global_quantities.magnetic_axis.z = Some(z_axis);
        time_slice.profiles_1d.psi_norm = Some(array![0.0, 0.25, 0.5, 1.0]);
        time_slice.profiles_2d = vec![profiles_2d];
        return time_slice;
    }

    #[test]
    fn circular_surface_matches_the_analytic_q_and_boundary_is_nan() {
        let time_slice: EquilibriumTimeSlice = time_slice_with_circular_psi();
        let r_axis: f64 = time_slice.global_quantities.magnetic_axis.r.unwrap();
        let z_axis: f64 = time_slice.global_quantities.magnetic_axis.z.unwrap();
        let minor_radius: f64 = 0.3;
        let psi_curvature: f64 = 2.5;
        let f_here: f64 = 0.8;

        let flux_surfaces: Vec<FluxSurface> = vec![
            empty_flux_surface(),
            empty_flux_surface(),
            circular_flux_surface(r_axis, z_axis, minor_radius),
            circular_flux_surface(r_axis, z_axis, 0.4),
        ];
        let f_profile: Array1<f64> = Array1::from_elem(flux_surfaces.len(), f_here);
        let q_profile: Array1<f64> = epp_q_profile(&time_slice, &flux_surfaces, &f_profile);

        let q_expected: f64 = 2.0 * PI * f_here / (psi_curvature * (r_axis.powi(2) - minor_radius.powi(2)).sqrt());
        assert_abs_diff_eq!(q_profile[2], q_expected, epsilon = 2e-6);
        assert!(q_profile[1].is_nan());
        assert!(q_profile[3].is_nan());
    }

    #[test]
    fn q_axis_uses_the_curvature_at_the_exact_magnetic_axis() {
        let r: Array1<f64> = Array1::linspace(0.8, 1.2, 5);
        let z: Array1<f64> = Array1::linspace(-0.2, 0.2, 5);
        let mag_r: f64 = 1.049;
        let mag_z: f64 = 0.037;
        let psi_rr_axis: f64 = 2.0;
        let psi_zz_axis: f64 = 3.0;
        let psi_rrr: f64 = 5.0;
        let f_axis: f64 = 0.8;
        let j_phi_axis: f64 = 1.7e6;

        let n_r: usize = r.len();
        let n_z: usize = z.len();
        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let d2_psi_d_r_d_z_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        let d2_psi_d_z2_2d: Array2<f64> = Array2::from_elem((n_z, n_r), psi_zz_axis);
        for i_r in 0..n_r {
            for i_z in 0..n_z {
                let delta_r: f64 = r[i_r] - mag_r;
                let delta_z: f64 = z[i_z] - mag_z;
                psi_2d[(i_z, i_r)] = 0.5 * psi_rr_axis * delta_r.powi(2) + psi_rrr * delta_r.powi(3) / 6.0 + 0.5 * psi_zz_axis * delta_z.powi(2);
                d_psi_d_r_2d[(i_z, i_r)] = psi_rr_axis * delta_r + 0.5 * psi_rrr * delta_r.powi(2);
                d_psi_d_z_2d[(i_z, i_r)] = psi_zz_axis * delta_z;
                d2_psi_d_r2_2d[(i_z, i_r)] = psi_rr_axis + psi_rrr * delta_r;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = Some(r);
        profiles_2d.grid.dim2 = Some(z);
        profiles_2d.psi = Some(psi_2d);
        profiles_2d.d_psi_d_r = Some(d_psi_d_r_2d);
        profiles_2d.d_psi_d_z = Some(d_psi_d_z_2d);
        profiles_2d.d2_psi_d_r2 = Some(d2_psi_d_r2_2d);
        profiles_2d.d2_psi_d_r_d_z = Some(d2_psi_d_r_d_z_2d);
        profiles_2d.d2_psi_d_z2 = Some(d2_psi_d_z2_2d);
        profiles_2d.j_phi = Some(Array2::from_elem((n_z, n_r), j_phi_axis));

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.global_quantities.magnetic_axis.r = Some(mag_r);
        time_slice.global_quantities.magnetic_axis.z = Some(mag_z);
        time_slice.profiles_2d = vec![profiles_2d];

        let f_profile: Array1<f64> = Array1::from_vec(vec![f_axis]);
        let psi_gradient_interpolator: PsiGradientInterpolator = PsiGradientInterpolator::new(&time_slice);
        let q_axis: f64 = epp_q_axis(&time_slice, &f_profile, &psi_gradient_interpolator);
        let q_axis_expected: f64 = (psi_rr_axis + psi_zz_axis).abs() / (psi_rr_axis * psi_zz_axis).sqrt() * f_axis / (MU_0 * mag_r.powi(2) * j_phi_axis);

        assert_abs_diff_eq!(q_axis, q_axis_expected, epsilon = 1e-12);
    }
}
