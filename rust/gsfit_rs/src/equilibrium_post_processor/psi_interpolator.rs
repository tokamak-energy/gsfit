//! Bicubic interpolation of `psi`, and of its gradient, anywhere on the solved grid.
//!
//! Unlike most files here this fills no data-dictionary path: it is a helper shared by the
//! calculators which need `psi`, or its derivatives, between the grid points.

use crate::plasma_geometry::hessian;
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2};

/// Values of `grad(psi)` and the Hessian interpolated at one point.
pub(super) struct PsiDerivatives {
    pub(super) d_psi_d_r: f64,
    pub(super) d_psi_d_z: f64,
    pub(super) d2_psi_d_r2: f64,
    pub(super) d2_psi_d_r_d_z: f64,
    pub(super) d2_psi_d_z2: f64,
}

/// `psi` and its gradient at one point, both from the bicubic of `psi`.
pub(super) struct PsiValueAndGradient {
    pub(super) psi: f64,
    pub(super) d_psi_d_r: f64,
    pub(super) d_psi_d_z: f64,
}

/// Cell coordinates for one point on the rectangular grid.
struct GridCellCoordinates {
    i_r_left: usize,
    i_z_lower: usize,
    x: f64,
    y: f64,
}

/// Bicubic models of `psi`, and independently of `d(psi)/d(r)` and `d(psi)/d(z)`, on the solved grid.
///
/// `psi` itself is the bicubic built from `psi`, its two first derivatives and its mixed second
/// derivative at the corners of the cell. Along a grid line this is the cubic Hermite interpolation
/// `marching_squares` finds the boundary outline's crossings with, so its vertices lie exactly on
/// `psi(r, z) = psi_b`; the flux surfaces inside the boundary are found on the same bicubic, along rays
/// from the magnetic axis; see `flux_surfaces`.
///
/// The gradient is **not** the derivative of that bicubic: each component is a bicubic of its own.
/// Inside the discretised current region `psi` is C1 but not C2, so differentiating the bicubic of
/// `psi` can misplace where a component of the gradient vanishes by over half a cell; see the note
/// on `find_stationary_points_using_winding_number`, which finds the X-points the same way.
///
/// The bicubic model of each gradient component needs that component's value, its two first
/// derivatives, and its mixed second derivative at every grid point. The first derivatives are the
/// stored Hessian of `psi`. The mixed derivatives are third derivatives of `psi`; they are obtained
/// by second-order finite differences of the Hessian in `z`, matching the established
/// stationary-point interpolation without requiring Green's tables at arbitrary `(r, z)` positions.
pub(super) struct PsiInterpolator<'a> {
    r: &'a Array1<f64>,
    z: &'a Array1<f64>,
    psi_2d: &'a Array2<f64>,
    d_psi_d_r_2d: &'a Array2<f64>,
    d_psi_d_z_2d: &'a Array2<f64>,
    d2_psi_d_r2_2d: &'a Array2<f64>,
    d2_psi_d_r_d_z_2d: &'a Array2<f64>,
    d2_psi_d_z2_2d: &'a Array2<f64>,
    d3_psi_d_r2_d_z_2d: Array2<f64>,
    d3_psi_d_r_d_z2_2d: Array2<f64>,
    /// Grid spacing [metre]
    pub(super) d_r: f64,
    /// Grid spacing [metre]
    pub(super) d_z: f64,
}

impl<'a> PsiInterpolator<'a> {
    pub(super) fn new(time_slice: &'a EquilibriumTimeSlice) -> Self {
        // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
        let r: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim1;
        let z: &Array1<f64> = &time_slice.profiles_2d[0].grid.dim2;
        let psi_2d: &Array2<f64> = &time_slice.profiles_2d[0].psi;
        let d_psi_d_r_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_r;
        let d_psi_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d_psi_d_z;
        let d2_psi_d_r2_2d: &Array2<f64> = &time_slice.profiles_2d[0].d2_psi_d_r2;
        let d2_psi_d_r_d_z_2d: &Array2<f64> = &time_slice.profiles_2d[0].d2_psi_d_r_d_z;
        let d2_psi_d_z2_2d: &Array2<f64> = &time_slice.profiles_2d[0].d2_psi_d_z2;

        // `psi_2d` is checked where it is used, by `psi`, so that a caller which only needs the
        // gradient does not need `psi` at all
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

        Self {
            r,
            z,
            psi_2d,
            d_psi_d_r_2d,
            d_psi_d_z_2d,
            d2_psi_d_r2_2d,
            d2_psi_d_r_d_z_2d,
            d2_psi_d_z2_2d,
            d3_psi_d_r2_d_z_2d,
            d3_psi_d_r_d_z2_2d,
            d_r,
            d_z,
        }
    }

    /// Interpolate `psi` at one point.
    ///
    /// Only the value of the bicubic of `psi` is used; for a derivative of `psi`, use `derivatives`.
    ///
    /// # Arguments
    /// * `r_point`, `z_point` - the position to evaluate [metre]
    ///
    /// # Returns
    /// * `psi` - the poloidal flux, or `None` when the point is off the grid [weber]
    pub(super) fn psi(&self, r_point: f64, z_point: f64) -> Option<f64> {
        assert_eq!(self.psi_2d.dim(), (self.z.len(), self.r.len()));

        let cell: GridCellCoordinates = self.cell_coordinates(r_point, z_point)?;
        let psi: HermiteValueAndDerivatives = hermite_bicubic(
            self.psi_2d,
            self.d_psi_d_r_2d,
            self.d_psi_d_z_2d,
            self.d2_psi_d_r_d_z_2d,
            &cell,
            self.d_r,
            self.d_z,
        );

        Some(psi.f)
    }

    /// Interpolate `psi`, and the gradient of the same bicubic, at one point.
    ///
    /// The flux surfaces are contours of this bicubic, so this is the gradient to find them with. For the physics, e.g. the poloidal
    /// field, use `derivatives`, whose gradient comes from the bicubics of the gradient's components.
    ///
    /// # Arguments
    /// * `r_point`, `z_point` - the position to evaluate [metre]
    ///
    /// # Returns
    /// * `psi_value_and_gradient` - `psi` [weber] and its gradient [weber / metre], or `None` when the point is off the grid
    pub(super) fn psi_value_and_gradient(&self, r_point: f64, z_point: f64) -> Option<PsiValueAndGradient> {
        assert_eq!(self.psi_2d.dim(), (self.z.len(), self.r.len()));

        let cell: GridCellCoordinates = self.cell_coordinates(r_point, z_point)?;
        let psi: HermiteValueAndDerivatives = hermite_bicubic(
            self.psi_2d,
            self.d_psi_d_r_2d,
            self.d_psi_d_z_2d,
            self.d2_psi_d_r_d_z_2d,
            &cell,
            self.d_r,
            self.d_z,
        );

        Some(PsiValueAndGradient {
            psi: psi.f,
            d_psi_d_r: psi.d_f_d_x / self.d_r,
            d_psi_d_z: psi.d_f_d_y / self.d_z,
        })
    }

    /// Interpolate `grad(psi)` and the Hessian of `psi` at one point.
    ///
    /// # Arguments
    /// * `r_point`, `z_point` - the position to evaluate [metre]
    ///
    /// # Returns
    /// * `psi_derivatives` - the first derivatives [weber / metre] and the second derivatives
    ///   [weber / metre ** 2], or `None` when the point is off the grid
    pub(super) fn derivatives(&self, r_point: f64, z_point: f64) -> Option<PsiDerivatives> {
        let cell: GridCellCoordinates = self.cell_coordinates(r_point, z_point)?;
        let d_psi_d_r: HermiteValueAndDerivatives = hermite_bicubic(
            self.d_psi_d_r_2d,
            self.d2_psi_d_r2_2d,
            self.d2_psi_d_r_d_z_2d,
            &self.d3_psi_d_r2_d_z_2d,
            &cell,
            self.d_r,
            self.d_z,
        );
        let d_psi_d_z: HermiteValueAndDerivatives = hermite_bicubic(
            self.d_psi_d_z_2d,
            self.d2_psi_d_r_d_z_2d,
            self.d2_psi_d_z2_2d,
            &self.d3_psi_d_r_d_z2_2d,
            &cell,
            self.d_r,
            self.d_z,
        );

        Some(PsiDerivatives {
            d_psi_d_r: d_psi_d_r.f,
            d_psi_d_z: d_psi_d_z.f,
            d2_psi_d_r2: d_psi_d_r.d_f_d_x / self.d_r,
            d2_psi_d_r_d_z: 0.5 * (d_psi_d_r.d_f_d_y / self.d_z + d_psi_d_z.d_f_d_x / self.d_r),
            d2_psi_d_z2: d_psi_d_z.d_f_d_y / self.d_z,
        })
    }

    /// Calculate the Hessian matrix of `psi` at an arbitrary point on the grid.
    ///
    /// The two components of `grad(psi)` are modelled independently with bicubic interpolation, then
    /// differentiated at the requested point. This is the same construction used to refine stationary
    /// points in the Grad-Shafranov solver and avoids differentiating an interpolated scalar `psi`
    /// inside the discretised plasma-current region.
    ///
    /// # Arguments
    /// * `r_point`, `z_point` - the position to evaluate [metre]
    ///
    /// # Returns
    /// * `hessian_matrix` - the 2x2 Hessian matrix [weber per metre ** 2]
    /// * `hessian_determinant` - its determinant
    /// * `hessian_trace` - its trace
    pub(super) fn hessian_matrix(&self, r_point: f64, z_point: f64) -> Option<(Array2<f64>, f64, f64)> {
        let psi_derivatives: PsiDerivatives = self.derivatives(r_point, z_point)?;
        let mut hessian_matrix: Array2<f64> = Array2::from_elem((2, 2), f64::NAN);
        hessian_matrix[(0, 0)] = psi_derivatives.d2_psi_d_r2;
        hessian_matrix[(0, 1)] = psi_derivatives.d2_psi_d_r_d_z;
        hessian_matrix[(1, 0)] = psi_derivatives.d2_psi_d_r_d_z;
        hessian_matrix[(1, 1)] = psi_derivatives.d2_psi_d_z2;

        let (hessian_determinant, hessian_trace): (f64, f64) =
            hessian(psi_derivatives.d2_psi_d_r2, psi_derivatives.d2_psi_d_r_d_z, psi_derivatives.d2_psi_d_z2);
        Some((hessian_matrix, hessian_determinant, hessian_trace))
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

        Some(GridCellCoordinates { i_r_left, i_z_lower, x, y })
    }
}

/// A bicubic's value and first derivatives, with respect to the cell's normalised coordinates `x` and `y` in `[0, 1]`.
struct HermiteValueAndDerivatives {
    f: f64,
    d_f_d_x: f64,
    d_f_d_y: f64,
}

/// Evaluate the bicubic of one grid quantity in one cell, as the tensor product of cubic Hermite interpolations: from the quantity, its
/// two first derivatives and its mixed derivative at the cell's four corners. It is the same bicubic as `BicubicInterpolator`'s, which
/// builds its coefficient matrix from the same sixteen numbers; this evaluates it without allocating.
///
/// # Arguments
/// * `values_2d` - the quantity on the grid, shape (n_z, n_r) [any]
/// * `d_values_d_r_2d`, `d_values_d_z_2d` - its derivatives on the grid [any / metre]
/// * `d2_values_d_r_d_z_2d` - its mixed derivative on the grid [any / metre ** 2]
/// * `cell` - the cell, and the point's position in it
/// * `d_r`, `d_z` - the grid spacing [metre]
///
/// # Returns
/// * The value [any] and its derivatives with respect to the normalised cell coordinates [any]
fn hermite_bicubic(
    values_2d: &Array2<f64>,
    d_values_d_r_2d: &Array2<f64>,
    d_values_d_z_2d: &Array2<f64>,
    d2_values_d_r_d_z_2d: &Array2<f64>,
    cell: &GridCellCoordinates,
    d_r: f64,
    d_z: f64,
) -> HermiteValueAndDerivatives {
    let x: f64 = cell.x;
    let y: f64 = cell.y;
    // The cubic Hermite basis, [value at 0, value at 1, slope at 0, slope at 1], and its derivatives, in x and in y
    let basis_x: [f64; 4] = [
        (2.0 * x - 3.0) * x * x + 1.0,
        (3.0 - 2.0 * x) * x * x,
        ((x - 2.0) * x + 1.0) * x,
        (x - 1.0) * x * x,
    ];
    let d_basis_x: [f64; 4] = [6.0 * (x - 1.0) * x, 6.0 * (1.0 - x) * x, (3.0 * x - 4.0) * x + 1.0, (3.0 * x - 2.0) * x];
    let basis_y: [f64; 4] = [
        (2.0 * y - 3.0) * y * y + 1.0,
        (3.0 - 2.0 * y) * y * y,
        ((y - 2.0) * y + 1.0) * y,
        (y - 1.0) * y * y,
    ];
    let d_basis_y: [f64; 4] = [6.0 * (y - 1.0) * y, 6.0 * (1.0 - y) * y, (3.0 * y - 4.0) * y + 1.0, (3.0 * y - 2.0) * y];

    let mut f: f64 = 0.0;
    let mut d_f_d_x: f64 = 0.0;
    let mut d_f_d_y: f64 = 0.0;
    for i_corner_r in 0..2 {
        for i_corner_z in 0..2 {
            let i_r: usize = cell.i_r_left + i_corner_r;
            let i_z: usize = cell.i_z_lower + i_corner_z;
            let value: f64 = values_2d[(i_z, i_r)];
            let slope_x: f64 = d_r * d_values_d_r_2d[(i_z, i_r)];
            let slope_y: f64 = d_z * d_values_d_z_2d[(i_z, i_r)];
            let twist: f64 = d_r * d_z * d2_values_d_r_d_z_2d[(i_z, i_r)];
            // The value basis function of this corner is entry `i_corner`, its slope basis function entry `2 + i_corner`
            let value_x: f64 = basis_x[i_corner_r];
            let slope_basis_x: f64 = basis_x[2 + i_corner_r];
            let d_value_x: f64 = d_basis_x[i_corner_r];
            let d_slope_basis_x: f64 = d_basis_x[2 + i_corner_r];
            let value_y: f64 = basis_y[i_corner_z];
            let slope_basis_y: f64 = basis_y[2 + i_corner_z];
            let d_value_y: f64 = d_basis_y[i_corner_z];
            let d_slope_basis_y: f64 = d_basis_y[2 + i_corner_z];

            f += value * value_x * value_y + slope_x * slope_basis_x * value_y + slope_y * value_x * slope_basis_y + twist * slope_basis_x * slope_basis_y;
            d_f_d_x += value * d_value_x * value_y
                + slope_x * d_slope_basis_x * value_y
                + slope_y * d_value_x * slope_basis_y
                + twist * d_slope_basis_x * slope_basis_y;
            d_f_d_y += value * value_x * d_value_y
                + slope_x * slope_basis_x * d_value_y
                + slope_y * value_x * d_slope_basis_y
                + twist * slope_basis_x * d_slope_basis_y;
        }
    }

    HermiteValueAndDerivatives { f, d_f_d_x, d_f_d_y }
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

    derivative
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::ids::equilibrium::EquilibriumProfiles2d;
    use ndarray::array;

    /// A bicubic is exact for a polynomial of up to cubic order in each of `r` and `z`, so `psi`
    /// must be reproduced to round-off between the grid points, not only on them
    #[test]
    fn psi_reproduces_a_bicubic_polynomial_between_the_grid_points() {
        fn calculate_psi(r: f64, z: f64) -> f64 {
            0.3 + 1.2 * r.powi(3) - 0.7 * r * z.powi(2) + 0.4 * r.powi(2) * z + 0.9 * z.powi(3)
        }
        fn calculate_d_psi_d_r(r: f64, z: f64) -> f64 {
            3.6 * r.powi(2) - 0.7 * z.powi(2) + 0.8 * r * z
        }
        fn calculate_d_psi_d_z(r: f64, z: f64) -> f64 {
            -1.4 * r * z + 0.4 * r.powi(2) + 2.7 * z.powi(2)
        }
        fn calculate_d2_psi_d_r_d_z(r: f64, z: f64) -> f64 {
            -1.4 * z + 0.8 * r
        }

        let n_r: usize = 9;
        let n_z: usize = 11;
        let r: Array1<f64> = Array1::linspace(0.2, 1.0, n_r);
        let z: Array1<f64> = Array1::linspace(-0.5, 0.5, n_z);

        // Only what `psi` reads is filled with the polynomial; the rest only has to be on the grid
        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                psi_2d[(i_z, i_r)] = calculate_psi(r[i_r], z[i_z]);
                d_psi_d_r_2d[(i_z, i_r)] = calculate_d_psi_d_r(r[i_r], z[i_z]);
                d_psi_d_z_2d[(i_z, i_r)] = calculate_d_psi_d_z(r[i_r], z[i_z]);
                d2_psi_d_r_d_z_2d[(i_z, i_r)] = calculate_d2_psi_d_r_d_z(r[i_r], z[i_z]);
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;
        profiles_2d.psi = psi_2d;
        profiles_2d.d_psi_d_r = d_psi_d_r_2d;
        profiles_2d.d_psi_d_z = d_psi_d_z_2d;
        profiles_2d.d2_psi_d_r2 = Array2::zeros((n_z, n_r));
        profiles_2d.d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;
        profiles_2d.d2_psi_d_z2 = Array2::zeros((n_z, n_r));
        let time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice {
            profiles_2d: vec![profiles_2d],
            ..Default::default()
        };

        // Points inside cells, and one on a grid line
        let points_r: Array1<f64> = array![0.237, 0.555, 0.999, 0.6];
        let points_z: Array1<f64> = array![-0.481, 0.013, 0.431, 0.25];
        let n_point: usize = points_r.len();

        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);
        for i_point in 0..n_point {
            let psi: f64 = psi_interpolator.psi(points_r[i_point], points_z[i_point]).expect("the point is on the grid");
            assert_abs_diff_eq!(psi, calculate_psi(points_r[i_point], points_z[i_point]), epsilon = 1e-12);
        }

        // Off the grid there is nothing to interpolate
        assert!(psi_interpolator.psi(1.01, 0.0).is_none());
        assert!(psi_interpolator.psi(0.5, f64::NAN).is_none());
    }
}
