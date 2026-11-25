//! Plasma shapes and the conditions they put on `(1 - psi_N)` at the LCFS.
//!
//! Points are `(R, Z)` [metre]; once the shape is moved down by `z_geo` (`shifted_down`), `Z` is measured from `z_geo`, the height the
//! series is built about. The shapes are a model surface, centred on `(r_geo, z_geo)` with its innermost and outermost points at
//! `(r_geo -+ a_minor, z_geo)`, or a user-supplied LCFS:
//! * `SymmetricLimited` - the paper's Miller surface, Guazzotto and Freidberg's limited plasma;
//! * `DoubleNull` - Guazzotto and Freidberg's double null, two ellipses meeting at right angles at the X-points;
//! * `SingleNull` - Guazzotto and Freidberg's lower single null, a Miller upper half and the double null's two ellipses below it, with
//!   the X-point;
//! * `AntisymmetricLimited` - Miller upper and lower halves with different elongations and triangularities;
//! * `LcfsPoints` - a user-supplied LCFS, given as points `(R, Z)`, optionally with an X-point, up-down symmetric or not.
//!
//! The first two (and symmetric user points) are up-down symmetric, so `(1 - psi_N)` is even in `Z` and the conditions lie on the upper
//! half; the others are not, and `(1 - psi_N)` has odd terms too.
//!
//! A condition is a linear, homogeneous functional of `(1 - psi_N)` at a point. For the model surfaces they are the paper's seven
//! conditions (Eq. gf_conditions_vector; twelve for the up-down asymmetric shapes, with the X-point's in place of the highest point's
//! for a double or single null), in Guazzotto and Freidberg's order, then `(1 - psi_N) = 0` at more points of the model surface: the
//! value at the innermost, outermost and highest points, `d/dR` at the highest point, and the boundary's curvature,
//! `(d2/dZ2 + Lambda_in d/dR)` at the innermost and `(d2/dZ2 + Lambda_out d/dR)` at the outermost point, and
//! `(4 R**2 d2/d(R**2)**2 + Lambda_top d/dZ)` at the highest point (`4 R**2 d2/d(R**2)**2 = d2/dR2 - d/dR / R` is the paper's `d2/dR2`
//! where `d/dR` vanishes), with the `Lambda`s of Guazzotto and Freidberg's model surfaces. For a user LCFS they are `(1 - psi_N) = 0`
//! at every point. At an X-point, `(1 - psi_N) = d/dR = d/dZ = 0`; these conditions are marked exact: the least-squares fits impose them
//! exactly, so that the separatrix passes through the X-point, and fit the others.
//!
//! In the least-squares fits each condition is made dimensionless with the plasma's half-widths: a derivative in `R` is multiplied by
//! `R_geo a_minor / R` (the derivative in `R**2` times the half-width in `R**2`, `2 R_geo a_minor`), and one in `Z` by `a_minor`.

use super::lattice::FluxValues;
use super::settings_schema::{
    BuxtonProkopyszynSettingsAntisymmetricLimited, BuxtonProkopyszynSettingsConfiguration, BuxtonProkopyszynSettingsDoubleNull,
    BuxtonProkopyszynSettingsLcfsPoints, BuxtonProkopyszynSettingsSingleNull, BuxtonProkopyszynSettingsSymmetricLimited,
};
use ndarray::Axis;
use ndarray::{Array1, concatenate, s};
use std::f64::consts::PI;

/// A point `(R, Z)`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Major radius [metre]
    pub r: f64,
    /// Height (above `z_geo`, once the shape is shifted) [metre]
    pub z: f64,
}

/// Points, or a curve, `(R, Z)`
#[derive(Clone, Debug)]
pub struct Points {
    /// [metre]
    pub r: Array1<f64>,
    /// [metre]
    pub z: Array1<f64>,
}

/// The paper's matching data of a Miller surface: its highest point and the boundary's curvatures
#[derive(Clone, Copy, Debug)]
pub struct MillerMatchingConstants {
    /// The highest point's `R`, `R_geo - delta a_minor` [metre]
    pub r_top: f64,
    /// Its height above `z_geo`, `kappa a_minor` [metre]
    pub z_top: f64,
    /// `Lambda_in`, the curvature at the innermost point, `(1 - delta_hat)**2 / (kappa**2 a_minor)` [1 / metre]
    pub lambda_in: f64,
    /// `Lambda_out`, at the outermost point, `-(1 + delta_hat)**2 / (kappa**2 a_minor)` [1 / metre]
    pub lambda_out: f64,
    /// `Lambda_top`, at the highest point, `-kappa / (a_minor (1 - delta**2))` [1 / metre]
    pub lambda_top: f64,
}

/// Guazzotto and Freidberg's double null: two ellipses meeting at right angles at the X-points
#[derive(Clone, Copy, Debug)]
pub struct DoubleNullConstants {
    /// The upper X-point's `R`, `R_geo - delta_x a_minor` [metre]
    pub r_x_point: f64,
    /// Its height above `z_geo`, `kappa_x a_minor` [metre]
    pub z_x_point: f64,
    /// `Lambda_in`, `(1 - delta_x) (1 + xi) / (kappa_x**2 a_minor)` [1 / metre]
    pub lambda_in: f64,
    /// `Lambda_out`, `-(1 + delta_x) (1 + xi) / (kappa_x**2 a_minor)`, with Guazzotto and Freidberg's
    /// `xi = (1 - delta_x**2)**0.5 / (kappa_x - (1 - delta_x**2)**0.5)` [1 / metre]
    pub lambda_out: f64,
    /// The inboard ellipse's centre, `R` [metre]
    pub inboard_centre: f64,
    /// Its radial semi-axis [metre]
    pub inboard_semi_axis: f64,
    /// The outboard ellipse's centre, `R` [metre]
    pub outboard_centre: f64,
    /// Its radial semi-axis [metre]
    pub outboard_semi_axis: f64,
    /// Both ellipses' vertical semi-axis [metre]
    pub vertical_semi_axis: f64,
    /// The ellipses' parameter at the X-points, measured from the midplane [radian]
    pub theta_x_point: f64,
}

/// The constants of Guazzotto and Freidberg's twelve single-null conditions
#[derive(Clone, Copy, Debug)]
pub struct SingleNullConstants {
    /// The highest point's `R` [metre]
    pub r_top: f64,
    /// The X-point's `R` [metre]
    pub r_x_point: f64,
    /// The averages of the two halves' `Lambda_in` [1 / metre]
    pub lambda_in: f64,
    /// ... and of their `Lambda_out` [1 / metre]
    pub lambda_out: f64,
    /// `Lambda_top` of the upper (Miller) half [1 / metre]
    pub lambda_top: f64,
}

/// The radial size of a shape: given for a model surface, from the points for a user LCFS
#[derive(Clone, Copy, Debug)]
pub struct RadialScale {
    /// The middle of the radial extent, the paper's `R_geo` [metre]
    pub r_geo: f64,
    /// Half the radial extent, the paper's `a_minor` [metre]
    pub a_minor: f64,
}

/// The kind of a boundary condition, a linear functional of `(1 - psi_N)` at a point
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionKind {
    /// `(1 - psi_N)`
    Value,
    /// `d(1 - psi_N)/dR`
    DR,
    /// `d(1 - psi_N)/dZ`
    DZ,
    /// `(d2/dZ2 + Lambda d/dR)(1 - psi_N)`, the boundary's curvature at its innermost or outermost point
    ZCurvature,
    /// `(4 R**2 d2/d(R**2)**2 + Lambda d/dZ)(1 - psi_N)`, the boundary's curvature at its highest or lowest point
    RCurvature,
}

/// Linear homogeneous conditions on `(1 - psi_N)`, `condition_i(1 - psi_N) = 0`
#[derive(Clone, Debug)]
pub struct BoundaryConditions {
    /// [metre]
    pub r: Vec<f64>,
    /// [metre]
    pub z: Vec<f64>,
    pub kind: Vec<ConditionKind>,
    /// The paper's `Lambda`, for the curvature kinds [1 / metre]
    pub lambda: Vec<f64>,
    /// Relative weight of each condition in the least-squares fits [dimensionless]
    pub weight: Vec<f64>,
    /// Whether a condition is imposed exactly rather than by least squares: an X-point's, so that the separatrix passes through it
    pub is_exact: Vec<bool>,
    /// The paper's `R_geo`, which with `a_minor` makes the conditions dimensionless [metre]
    pub r_geo: f64,
    /// The paper's `a_minor` [metre]
    pub a_minor: f64,
}

impl BoundaryConditions {
    /// No conditions yet
    ///
    /// # Arguments
    /// * `r_geo` - the paper's `R_geo` [metre]
    /// * `a_minor` - the paper's `a_minor` [metre]
    pub fn new(r_geo: f64, a_minor: f64) -> BoundaryConditions {
        BoundaryConditions {
            r: Vec::new(),
            z: Vec::new(),
            kind: Vec::new(),
            lambda: Vec::new(),
            weight: Vec::new(),
            is_exact: Vec::new(),
            r_geo,
            a_minor,
        }
    }

    /// # Arguments
    /// * `r` - the point's `R` [metre]
    /// * `z` - the point's `Z` [metre]
    /// * `kind` - the condition's kind
    /// * `lambda` - the paper's `Lambda`, for the curvature kinds [1 / metre]
    /// * `weight` - relative weight in least-squares fits [dimensionless]
    /// * `is_exact` - impose the condition exactly in least-squares fits
    pub fn add(&mut self, r: f64, z: f64, kind: ConditionKind, lambda: f64, weight: f64, is_exact: bool) {
        self.r.push(r);
        self.z.push(z);
        self.kind.push(kind);
        self.lambda.push(lambda);
        self.weight.push(weight);
        self.is_exact.push(is_exact);
    }

    /// Number of conditions [dimensionless]
    pub fn n_condition(&self) -> usize {
        self.r.len()
    }

    /// # Arguments
    /// * `flux` - `(1 - psi_N)` and its derivatives at the condition points, as `lattice::series_evaluate` returns them
    ///
    /// # Returns
    /// * `condition_i(1 - psi_N)` for every condition, made dimensionless with `R_geo a_minor / R` per derivative in `R` and `a_minor`
    ///   per derivative in `Z` [dimensionless]
    pub fn residuals(&self, flux: &FluxValues) -> Array1<f64> {
        let n_condition: usize = self.n_condition();
        let a_minor: f64 = self.a_minor;
        let mut residuals: Array1<f64> = Array1::from_elem(n_condition, f64::NAN);
        for i_condition in 0..n_condition {
            let lambda: f64 = self.lambda[i_condition];
            let r: f64 = self.r[i_condition];
            // A derivative in R, as a derivative in R**2 times 2 R_geo a_minor
            let length_r: f64 = self.r_geo * a_minor / r;
            residuals[i_condition] = match self.kind[i_condition] {
                ConditionKind::Value => flux.one_minus_psi_n[i_condition],
                ConditionKind::DR => length_r * flux.d_d_r[i_condition],
                ConditionKind::DZ => a_minor * flux.d_d_z[i_condition],
                ConditionKind::ZCurvature => a_minor * a_minor * (flux.d2_d_z2[i_condition] + lambda * flux.d_d_r[i_condition]),
                ConditionKind::RCurvature => length_r * length_r * (flux.d2_d_r2[i_condition] - flux.d_d_r[i_condition] / r + lambda * flux.d_d_z[i_condition]),
            };
        }
        residuals
    }
}

/// The paper's Miller model surface, `R = R_geo + a_minor cos(theta + delta_hat sin(theta))`, `Z = kappa a_minor sin(theta)`,
/// `delta_hat = asin(delta)`, centred on `Z = 0`.
///
/// # Arguments
/// * `theta` - the surface's parameter [radian]
/// * `r_geo` - the paper's `R_geo` [metre]
/// * `a_minor` - the paper's `a_minor` [metre]
/// * `kappa` - elongation [dimensionless]
/// * `delta` - triangularity [dimensionless]
pub fn miller_surface(theta: &Array1<f64>, r_geo: f64, a_minor: f64, kappa: f64, delta: f64) -> Points {
    let delta_hat: f64 = delta.asin();
    let n_theta: usize = theta.len();
    let mut surface_r: Array1<f64> = Array1::from_elem(n_theta, f64::NAN);
    let mut surface_z: Array1<f64> = Array1::from_elem(n_theta, f64::NAN);
    for i_theta in 0..n_theta {
        let angle: f64 = theta[i_theta] + delta_hat * theta[i_theta].sin();
        surface_r[i_theta] = r_geo + a_minor * angle.cos();
        surface_z[i_theta] = kappa * a_minor * theta[i_theta].sin();
    }
    Points { r: surface_r, z: surface_z }
}

/// The paper's matching data of a Miller surface: the highest point and the curvatures `Lambda_in`, `Lambda_out` and `Lambda_top`
pub fn miller_matching_constants(r_geo: f64, a_minor: f64, kappa: f64, delta: f64) -> MillerMatchingConstants {
    let delta_hat: f64 = delta.asin();
    MillerMatchingConstants {
        r_top: r_geo - delta * a_minor,
        z_top: kappa * a_minor,
        lambda_in: (1.0 - delta_hat).powi(2) / (kappa * kappa * a_minor),
        lambda_out: -(1.0 + delta_hat).powi(2) / (kappa * kappa * a_minor),
        lambda_top: -kappa / (a_minor * (1.0 - delta * delta)),
    }
}

/// Guazzotto and Freidberg's double-null model surface: two ellipses meeting at right angles at the X-points.
///
/// # Arguments
/// * `n_point_per_ellipse` - points on each of the outboard and inboard ellipses [dimensionless]
/// * `r_geo` - the paper's `R_geo` [metre]
/// * `a_minor` - the paper's `a_minor` [metre]
/// * `kappa_x` - elongation at the X-point [dimensionless]
/// * `delta_x` - triangularity at the X-point [dimensionless]
///
/// # Returns
/// * the outboard ellipse from the lower to the upper X-point, then the inboard one from the upper to the lower X-point, and the
///   constants
///
/// # Errors
/// `kappa_x <= 2 (1 - delta_x**2)**0.5`
pub fn double_null_surface(n_point_per_ellipse: usize, r_geo: f64, a_minor: f64, kappa_x: f64, delta_x: f64) -> Result<(Points, DoubleNullConstants), String> {
    let root: f64 = (1.0 - delta_x * delta_x).sqrt();
    let xi: f64 = root / (kappa_x - root);
    // xi = 0 and xi = 1 divide by zero below
    if !(xi > 0.0 && xi < 1.0) {
        return Err(format!("double_null_surface: needs kappa_x > 2 (1 - delta_x**2)**0.5; got xi = {xi}"));
    }
    // The inboard ellipse passes through R_geo - a_minor, the outboard one through R_geo + a_minor, both through the X-points
    let inboard_offset: f64 = (xi - delta_x) / (1.0 - xi);
    let outboard_offset: f64 = (xi + delta_x) / (1.0 - xi);
    let constants: DoubleNullConstants = DoubleNullConstants {
        r_x_point: r_geo - delta_x * a_minor,
        z_x_point: kappa_x * a_minor,
        lambda_in: (1.0 - delta_x) * (1.0 + xi) / (kappa_x * kappa_x * a_minor),
        lambda_out: -(1.0 + delta_x) * (1.0 + xi) / (kappa_x * kappa_x * a_minor),
        inboard_centre: r_geo + inboard_offset * a_minor,
        inboard_semi_axis: (1.0 + inboard_offset) * a_minor,
        outboard_centre: r_geo - outboard_offset * a_minor,
        outboard_semi_axis: (1.0 + outboard_offset) * a_minor,
        vertical_semi_axis: kappa_x / (1.0 - xi * xi).sqrt() * a_minor,
        theta_x_point: ((1.0 - xi * xi).sqrt() / xi).atan(),
    };
    let theta_x_point: f64 = constants.theta_x_point;
    let theta_outboard: Array1<f64> = Array1::linspace(-theta_x_point, theta_x_point, n_point_per_ellipse);
    let theta_inboard: Array1<f64> = Array1::linspace(PI - theta_x_point, PI + theta_x_point, n_point_per_ellipse);
    let n_surface: usize = 2 * n_point_per_ellipse;
    let mut surface_r: Array1<f64> = Array1::from_elem(n_surface, f64::NAN);
    let mut surface_z: Array1<f64> = Array1::from_elem(n_surface, f64::NAN);
    for i_point in 0..n_point_per_ellipse {
        surface_r[i_point] = constants.outboard_centre + constants.outboard_semi_axis * theta_outboard[i_point].cos();
        surface_r[n_point_per_ellipse + i_point] = constants.inboard_centre + constants.inboard_semi_axis * theta_inboard[i_point].cos();
        surface_z[i_point] = constants.vertical_semi_axis * theta_outboard[i_point].sin();
        surface_z[n_point_per_ellipse + i_point] = constants.vertical_semi_axis * theta_inboard[i_point].sin();
    }
    Ok((Points { r: surface_r, z: surface_z }, constants))
}

/// Points inside a boundary, for the interior normalisation of the least-squares eigenproblem: the boundary points (the upper half of
/// an up-down symmetric boundary, or all of it) shrunk by 0.25, 0.5 and 0.75 towards the middle of their range of `R**2` at `centre_z`,
/// in `R**2` and `Z`.
pub fn shrunk_interior_points(surface: &Points, centre_z: f64) -> Points {
    let n_surface: usize = surface.r.len();
    let mut r_squared_minimum: f64 = f64::INFINITY;
    let mut r_squared_maximum: f64 = f64::NEG_INFINITY;
    for i_surface in 0..n_surface {
        let r_squared: f64 = surface.r[i_surface] * surface.r[i_surface];
        r_squared_minimum = r_squared_minimum.min(r_squared);
        r_squared_maximum = r_squared_maximum.max(r_squared);
    }
    let centre_r_squared: f64 = 0.5 * (r_squared_minimum + r_squared_maximum);
    let scales: [f64; 3] = [0.25, 0.5, 0.75];
    let n_scale: usize = scales.len();
    let mut interior_r: Array1<f64> = Array1::from_elem(n_scale * n_surface, f64::NAN);
    let mut interior_z: Array1<f64> = Array1::from_elem(n_scale * n_surface, f64::NAN);
    for i_scale in 0..n_scale {
        for i_surface in 0..n_surface {
            let r_squared: f64 = surface.r[i_surface] * surface.r[i_surface];
            interior_r[i_scale * n_surface + i_surface] = (centre_r_squared + scales[i_scale] * (r_squared - centre_r_squared)).sqrt();
            interior_z[i_scale * n_surface + i_surface] = centre_z + scales[i_scale] * (surface.z[i_surface] - centre_z);
        }
    }
    Points { r: interior_r, z: interior_z }
}

/// Guazzotto and Freidberg's lower single-null model surface, and the constants of their twelve conditions: the Miller surface above the
/// midplane, the double null's two ellipses below it, meeting at right angles at the X-point `(R_geo - delta_x a_minor,
/// -kappa_x a_minor)`.
///
/// # Arguments
/// * `n_point_upper` - points on the upper (Miller) half, with both midplane points [dimensionless]
/// * `n_point_per_ellipse` - points on each lower ellipse, with its midplane point and the X-point [dimensionless]
///
/// # Returns
/// * anticlockwise from the outer midplane over the top, down the inboard ellipse to the X-point and up the outboard one, the last point
///   repeating the first; and the constants
#[allow(clippy::too_many_arguments)]
pub fn single_null_surface(
    n_point_upper: usize,
    n_point_per_ellipse: usize,
    r_geo: f64,
    a_minor: f64,
    kappa: f64,
    delta: f64,
    kappa_x: f64,
    delta_x: f64,
) -> Result<(Points, SingleNullConstants), String> {
    let theta_upper: Array1<f64> = Array1::linspace(0.0, PI, n_point_upper);
    let upper: Points = miller_surface(&theta_upper, r_geo, a_minor, kappa, delta);
    let (_, two_ellipses): (Points, DoubleNullConstants) = double_null_surface(4, r_geo, a_minor, kappa_x, delta_x)?;
    let theta_x_point: f64 = two_ellipses.theta_x_point;
    let theta_inboard: Array1<f64> = Array1::linspace(PI, PI + theta_x_point, n_point_per_ellipse);
    let theta_outboard: Array1<f64> = Array1::linspace(-theta_x_point, 0.0, n_point_per_ellipse);
    let n_surface: usize = n_point_upper + 2 * (n_point_per_ellipse - 1);
    let mut surface_r: Array1<f64> = Array1::from_elem(n_surface, f64::NAN);
    let mut surface_z: Array1<f64> = Array1::from_elem(n_surface, f64::NAN);
    for i_point in 0..n_point_upper {
        surface_r[i_point] = upper.r[i_point];
        surface_z[i_point] = upper.z[i_point];
    }
    for i_point in 1..n_point_per_ellipse {
        let i_inboard: usize = n_point_upper + i_point - 1;
        let i_outboard: usize = n_point_upper + n_point_per_ellipse - 1 + i_point - 1;
        surface_r[i_inboard] = two_ellipses.inboard_centre + two_ellipses.inboard_semi_axis * theta_inboard[i_point].cos();
        surface_z[i_inboard] = two_ellipses.vertical_semi_axis * theta_inboard[i_point].sin();
        surface_r[i_outboard] = two_ellipses.outboard_centre + two_ellipses.outboard_semi_axis * theta_outboard[i_point].cos();
        surface_z[i_outboard] = two_ellipses.vertical_semi_axis * theta_outboard[i_point].sin();
    }
    let miller: MillerMatchingConstants = miller_matching_constants(r_geo, a_minor, kappa, delta);
    let constants: SingleNullConstants = SingleNullConstants {
        r_top: miller.r_top,
        r_x_point: two_ellipses.r_x_point,
        lambda_in: 0.5 * (miller.lambda_in + two_ellipses.lambda_in),
        lambda_out: 0.5 * (miller.lambda_out + two_ellipses.lambda_out),
        lambda_top: miller.lambda_top,
    };
    Ok((Points { r: surface_r, z: surface_z }, constants))
}

/// Extra LCFS points on the upper half of a Miller surface, away from the matching points: `theta_i = (pi / 2) i / (p + 1)` and
/// `pi - theta_i`, `i = 1..p`, `p = n_point / 2`.
pub fn miller_surface_points(r_geo: f64, a_minor: f64, kappa: f64, delta: f64, n_point: usize) -> Points {
    let n_half: usize = n_point / 2;
    let mut theta: Array1<f64> = Array1::from_elem(2 * n_half, f64::NAN);
    for i_half in 0..n_half {
        theta[i_half] = 0.5 * PI * (i_half as f64 + 1.0) / (n_half as f64 + 1.0);
        theta[n_half + i_half] = PI - theta[i_half];
    }
    miller_surface(&theta, r_geo, a_minor, kappa, delta)
}

/// Extra LCFS points on a double null: `p = n_point / 2` points equally spaced in the ellipse parameter on each of the outboard and
/// inboard upper arcs.
pub fn double_null_surface_points(r_geo: f64, a_minor: f64, kappa_x: f64, delta_x: f64, n_point: usize) -> Result<Points, String> {
    let n_half: usize = n_point / 2;
    // double_null_surface samples each ellipse at linspace(-theta_x_point, theta_x_point) about its midplane point; with 2 p + 3 samples,
    // the p after the midplane (outboard) and the p before it (inboard) are the interior points of the upper arcs
    let n_sample: usize = 2 * n_half + 3;
    let (surface, _): (Points, DoubleNullConstants) = double_null_surface(n_sample, r_geo, a_minor, kappa_x, delta_x)?;
    let mut point_r: Array1<f64> = Array1::from_elem(2 * n_half, f64::NAN);
    let mut point_z: Array1<f64> = Array1::from_elem(2 * n_half, f64::NAN);
    for i_half in 0..n_half {
        point_r[i_half] = surface.r[n_half + 2 + i_half];
        point_z[i_half] = surface.z[n_half + 2 + i_half];
        point_r[n_half + i_half] = surface.r[n_sample + 1 + i_half];
        point_z[n_half + i_half] = surface.z[n_sample + 1 + i_half];
    }
    Ok(Points { r: point_r, z: point_z })
}

/// Plasma boundary topology and the shape parameters it requires, made from the settings' `configuration` (`from_settings`).
///
/// Each variant carries exactly its own shape inputs, in the paper's notation. A model surface is centred on `(r_geo, z_geo)`, its
/// innermost and outermost points being `(r_geo -+ a_minor, z_geo)` [metre]; a user LCFS sets all three from its points. The series is built about `z_geo`. The up-down symmetry is implied by the variant, or given for
/// `LcfsPoints`.
///
/// The model surfaces are Guazzotto and Freidberg's (J. Plasma Phys. 87 (2021) 905870303).
#[derive(Clone, Debug)]
pub enum Configuration {
    /// Up-down symmetric smooth limiter: the paper's Miller surface (Guazzotto and Freidberg's Sec. 3)
    SymmetricLimited {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa: f64,
        delta: f64,
    },
    /// Up-down symmetric double null (Guazzotto and Freidberg's Sec. 4): two ellipses meeting at right angles at the X-points
    DoubleNull {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa_x: f64,
        delta_x: f64,
    },
    /// Up-down asymmetric lower single null (Guazzotto and Freidberg's Sec. 5): a Miller upper half (`kappa`, `delta`) and the double
    /// null's two ellipses below `z_geo` (`kappa_x`, `delta_x`), with the X-point at `(r_geo - delta_x a_minor, z_geo - kappa_x a_minor)`
    SingleNull {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa: f64,
        delta: f64,
        kappa_x: f64,
        delta_x: f64,
    },
    /// Up-down asymmetric limiter: Miller surfaces with `kappa_upper`, `delta_upper` above `z_geo` and `kappa_lower`, `delta_lower`
    /// below it (Guazzotto and Freidberg's Sec. 5 construction with a smooth lower half)
    AntisymmetricLimited {
        r_geo: f64,
        z_geo: f64,
        a_minor: f64,
        kappa_upper: f64,
        delta_upper: f64,
        kappa_lower: f64,
        delta_lower: f64,
    },
    /// A user-supplied LCFS, given as points `(R, Z)` in the machine [metre], with an X-point (`x_point_r` and `x_point_z`) or without
    /// one (`None` for both). Its `r_geo` and `z_geo` are the middle of the radial and vertical extent of the points and the X-point,
    /// and `a_minor` half the radial extent.
    ///
    /// Up-down symmetric (`is_up_down_symmetric = true`): `(1 - psi_N)` is even in `Z - z_geo`, the points are reflected onto the upper
    /// half, and an X-point is mirrored. Otherwise `(1 - psi_N)` has odd terms too, the points are used as given, and the X-point is
    /// where it is given (below `z_geo` for a lower single null).
    ///
    /// Each point gives `(1 - psi_N) = 0`; the X-point gives `(1 - psi_N) = 0`, `d/dR = 0` and `d/dZ = 0`, in place of the slope and
    /// curvature conditions of the model surfaces, weighted by `x_point_weight` relative to the points when they are fitted by least squares
    /// (`exact_x_points = false`). Use about `7 + 10 (n_wave - 1 + n_wave_cosh)` points (or more) on each half, spread over the whole
    /// LCFS, even for an up-down symmetric one: its `z_geo`, the middle of their vertical extent, is the plane it is reflected about.
    LcfsPoints {
        boundary_r: Vec<f64>,
        boundary_z: Vec<f64>,
        x_point_r: Option<f64>,
        x_point_z: Option<f64>,
        x_point_weight: f64,
        is_up_down_symmetric: bool,
    },
}

impl Configuration {
    /// The shape the settings' `configuration` describes: the variant its `method` names, with the shape inputs of the element of the
    /// same name.
    ///
    /// # Errors
    /// `method` names no variant, or the element holding its shape inputs is missing
    pub fn from_settings(settings: &BuxtonProkopyszynSettingsConfiguration) -> Result<Configuration, String> {
        let missing = |method: &str| -> String {
            format!(
                "`equilibrium_settings/configuration/method` is \"{method}\", but `equilibrium_settings/configuration/{method}`, which holds its shape, is missing"
            )
        };
        match settings.method.as_str() {
            "symmetric_limited" => {
                let shape: &BuxtonProkopyszynSettingsSymmetricLimited = settings.symmetric_limited.as_ref().ok_or_else(|| missing("symmetric_limited"))?;
                Ok(Configuration::SymmetricLimited {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa: shape.kappa,
                    delta: shape.delta,
                })
            }
            "double_null" => {
                let shape: &BuxtonProkopyszynSettingsDoubleNull = settings.double_null.as_ref().ok_or_else(|| missing("double_null"))?;
                Ok(Configuration::DoubleNull {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa_x: shape.kappa_x,
                    delta_x: shape.delta_x,
                })
            }
            "single_null" => {
                let shape: &BuxtonProkopyszynSettingsSingleNull = settings.single_null.as_ref().ok_or_else(|| missing("single_null"))?;
                Ok(Configuration::SingleNull {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa: shape.kappa,
                    delta: shape.delta,
                    kappa_x: shape.kappa_x,
                    delta_x: shape.delta_x,
                })
            }
            "antisymmetric_limited" => {
                let shape: &BuxtonProkopyszynSettingsAntisymmetricLimited =
                    settings.antisymmetric_limited.as_ref().ok_or_else(|| missing("antisymmetric_limited"))?;
                Ok(Configuration::AntisymmetricLimited {
                    r_geo: shape.r_geo,
                    z_geo: shape.z_geo,
                    a_minor: shape.a_minor,
                    kappa_upper: shape.kappa_upper,
                    delta_upper: shape.delta_upper,
                    kappa_lower: shape.kappa_lower,
                    delta_lower: shape.delta_lower,
                })
            }
            "lcfs_points" => {
                let points: &BuxtonProkopyszynSettingsLcfsPoints = settings.lcfs_points.as_ref().ok_or_else(|| missing("lcfs_points"))?;
                Ok(Configuration::LcfsPoints {
                    boundary_r: points.boundary_r.to_vec(),
                    boundary_z: points.boundary_z.to_vec(),
                    x_point_r: points.x_point_r,
                    x_point_z: points.x_point_z,
                    x_point_weight: points.x_point_weight,
                    is_up_down_symmetric: points.is_up_down_symmetric,
                })
            }
            method => Err(format!(
                "`equilibrium_settings/configuration/method` is \"{method}\", but it must be symmetric_limited, double_null, single_null, antisymmetric_limited or lcfs_points"
            )),
        }
    }

    /// The height about which the series is built, from which `Z` is measured: a model surface's, or the middle of the vertical extent
    /// of a user LCFS's points and X-point [metre]
    pub fn z_geo(&self) -> f64 {
        match self {
            Configuration::SymmetricLimited { z_geo, .. }
            | Configuration::DoubleNull { z_geo, .. }
            | Configuration::SingleNull { z_geo, .. }
            | Configuration::AntisymmetricLimited { z_geo, .. } => *z_geo,
            Configuration::LcfsPoints {
                boundary_z,
                x_point_r,
                x_point_z,
                ..
            } => {
                let mut z_min: f64 = f64::INFINITY;
                let mut z_max: f64 = f64::NEG_INFINITY;
                for boundary_z_point in boundary_z {
                    z_min = z_min.min(*boundary_z_point);
                    z_max = z_max.max(*boundary_z_point);
                }
                if let (Some(_), Some(x_point_z_value)) = (x_point_r, x_point_z) {
                    z_min = z_min.min(*x_point_z_value);
                    z_max = z_max.max(*x_point_z_value);
                }
                0.5 * (z_min + z_max)
            }
        }
    }

    /// The variant's name, as `options_used` records it
    pub fn name(&self) -> &'static str {
        match self {
            Configuration::SymmetricLimited { .. } => "SymmetricLimited",
            Configuration::DoubleNull { .. } => "DoubleNull",
            Configuration::SingleNull { .. } => "SingleNull",
            Configuration::AntisymmetricLimited { .. } => "AntisymmetricLimited",
            Configuration::LcfsPoints { .. } => "LcfsPoints",
        }
    }

    /// Check the inputs.
    ///
    /// # Errors
    /// A model surface without `0 < a_minor < r_geo`, with a non-finite `z_geo`, or with a shape Guazzotto and Freidberg's construction
    /// cannot make;
    /// `LcfsPoints` with `boundary_r` and `boundary_z` of different lengths or empty, or with only one of `x_point_r` and `x_point_z`
    pub fn validate(&self) -> Result<(), String> {
        // a_minor / r_geo must lie between 0 and 1, and the centre's height must be finite
        let check_size = |name: &str, r_geo: f64, z_geo: f64, a_minor: f64| -> Result<(), String> {
            if !(r_geo.is_finite() && a_minor > 0.0 && a_minor < r_geo) {
                return Err(format!("{name}: needs 0 < a_minor < r_geo; got r_geo = {r_geo}, a_minor = {a_minor}"));
            }
            if !z_geo.is_finite() {
                return Err(format!("{name}: needs a finite z_geo, not {z_geo}"));
            }
            Ok(())
        };
        // The Miller surfaces need kappa > 0 and |delta| < 1 (asin(delta), and 1 - delta**2 divides); the double null's ellipses meet
        // at the X-points for 0 < xi < 1, xi = (1 - delta_x**2)**0.5 / (kappa_x - (1 - delta_x**2)**0.5)
        let check_miller = |name: &str, kappa: f64, delta: f64| -> Result<(), String> {
            if !(kappa.is_finite() && kappa > 0.0 && delta.abs() < 1.0) {
                return Err(format!("{name}: needs kappa > 0 and |delta| < 1; got kappa = {kappa}, delta = {delta}"));
            }
            Ok(())
        };
        let check_double_null = |name: &str, kappa_x: f64, delta_x: f64| -> Result<(), String> {
            let root: f64 = (1.0 - delta_x * delta_x).sqrt();
            let xi: f64 = root / (kappa_x - root);
            if !(delta_x.abs() < 1.0 && xi > 0.0 && xi < 1.0) {
                return Err(format!(
                    "{name}: needs |delta_x| < 1 and kappa_x > 2 (1 - delta_x**2)**0.5; got kappa_x = {kappa_x}, delta_x = {delta_x} (xi = {xi})"
                ));
            }
            Ok(())
        };
        match self {
            Configuration::SymmetricLimited {
                r_geo,
                z_geo,
                a_minor,
                kappa,
                delta,
            } => {
                check_size("SymmetricLimited", *r_geo, *z_geo, *a_minor)?;
                check_miller("SymmetricLimited", *kappa, *delta)?;
            }
            Configuration::DoubleNull {
                r_geo,
                z_geo,
                a_minor,
                kappa_x,
                delta_x,
            } => {
                check_size("DoubleNull", *r_geo, *z_geo, *a_minor)?;
                check_double_null("DoubleNull", *kappa_x, *delta_x)?;
            }
            Configuration::SingleNull {
                r_geo,
                z_geo,
                a_minor,
                kappa,
                delta,
                kappa_x,
                delta_x,
            } => {
                check_size("SingleNull", *r_geo, *z_geo, *a_minor)?;
                check_miller("SingleNull", *kappa, *delta)?;
                check_double_null("SingleNull", *kappa_x, *delta_x)?;
            }
            Configuration::AntisymmetricLimited {
                r_geo,
                z_geo,
                a_minor,
                kappa_upper,
                delta_upper,
                kappa_lower,
                delta_lower,
            } => {
                check_size("AntisymmetricLimited", *r_geo, *z_geo, *a_minor)?;
                check_miller("AntisymmetricLimited (upper half)", *kappa_upper, *delta_upper)?;
                check_miller("AntisymmetricLimited (lower half)", *kappa_lower, *delta_lower)?;
            }
            Configuration::LcfsPoints {
                boundary_r,
                boundary_z,
                x_point_r,
                x_point_z,
                x_point_weight,
                ..
            } => {
                if boundary_r.len() != boundary_z.len() || boundary_r.is_empty() {
                    return Err("LcfsPoints: boundary_r and boundary_z must be 1d arrays of the same, non-zero, length".to_string());
                }
                if x_point_r.is_some() != x_point_z.is_some() {
                    return Err("LcfsPoints: give both x_point_r and x_point_z, or neither".to_string());
                }
                let n_point: usize = boundary_r.len();
                for i_point in 0..n_point {
                    if !(boundary_r[i_point].is_finite() && boundary_r[i_point] > 0.0 && boundary_z[i_point].is_finite()) {
                        return Err(format!(
                            "LcfsPoints: the points need R > 0 and finite Z; point {i_point} is ({}, {})",
                            boundary_r[i_point], boundary_z[i_point]
                        ));
                    }
                }
                if let (Some(r), Some(z)) = (x_point_r, x_point_z) {
                    if !(r.is_finite() && *r > 0.0 && z.is_finite()) {
                        return Err(format!("LcfsPoints: the X-point needs R > 0 and finite Z; got ({r}, {z})"));
                    }
                }
                if !(x_point_weight.is_finite() && *x_point_weight >= 0.0) {
                    return Err(format!("LcfsPoints: x_point_weight must be finite and not negative, not {x_point_weight}"));
                }
            }
        }
        Ok(())
    }

    /// The same shape moved down by `delta_z` [metre]: the solver works relative to `z_geo`, so it moves the shape down by it, after
    /// which `Z` is measured from `z_geo`.
    pub fn shifted_down(&self, delta_z: f64) -> Configuration {
        let mut shifted: Configuration = self.clone();
        match &mut shifted {
            Configuration::SymmetricLimited { z_geo, .. }
            | Configuration::DoubleNull { z_geo, .. }
            | Configuration::SingleNull { z_geo, .. }
            | Configuration::AntisymmetricLimited { z_geo, .. } => {
                *z_geo -= delta_z;
            }
            Configuration::LcfsPoints { boundary_z, x_point_z, .. } => {
                let n_point: usize = boundary_z.len();
                for i_point in 0..n_point {
                    boundary_z[i_point] -= delta_z;
                }
                if let Some(x_point_z_value) = x_point_z {
                    *x_point_z_value -= delta_z;
                }
            }
        }
        shifted
    }

    /// The paper's `R_geo` and `a_minor`: a model surface's; for a user LCFS, the middle and half the width of the radial extent of the
    /// points and the X-point
    pub fn scale(&self) -> RadialScale {
        match self {
            Configuration::SymmetricLimited { r_geo, a_minor, .. }
            | Configuration::DoubleNull { r_geo, a_minor, .. }
            | Configuration::SingleNull { r_geo, a_minor, .. }
            | Configuration::AntisymmetricLimited { r_geo, a_minor, .. } => RadialScale {
                r_geo: *r_geo,
                a_minor: *a_minor,
            },
            Configuration::LcfsPoints {
                boundary_r,
                x_point_r,
                x_point_z,
                ..
            } => {
                let mut r_min: f64 = f64::INFINITY;
                let mut r_max: f64 = f64::NEG_INFINITY;
                for boundary_r_point in boundary_r {
                    r_min = r_min.min(*boundary_r_point);
                    r_max = r_max.max(*boundary_r_point);
                }
                if let (Some(x_point_r_value), Some(_)) = (x_point_r, x_point_z) {
                    r_min = r_min.min(*x_point_r_value);
                    r_max = r_max.max(*x_point_r_value);
                }
                RadialScale {
                    r_geo: 0.5 * (r_min + r_max),
                    a_minor: 0.5 * (r_max - r_min),
                }
            }
        }
    }

    /// Whether `(1 - psi_N)` is even in `Z`: by the variant, or for `LcfsPoints` as given
    pub fn is_up_down_symmetric(&self) -> bool {
        match self {
            Configuration::SymmetricLimited { .. } | Configuration::DoubleNull { .. } => true,
            Configuration::SingleNull { .. } | Configuration::AntisymmetricLimited { .. } => false,
            Configuration::LcfsPoints { is_up_down_symmetric, .. } => *is_up_down_symmetric,
        }
    }

    /// A user LCFS's points, reflected to `Z >= 0` for an up-down symmetric LCFS
    fn lcfs_points(boundary_r: &[f64], boundary_z: &[f64], is_up_down_symmetric: bool) -> Points {
        let n_point: usize = boundary_r.len();
        let mut point_r: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut point_z: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        for i_point in 0..n_point {
            point_r[i_point] = boundary_r[i_point];
            point_z[i_point] = if is_up_down_symmetric {
                boundary_z[i_point].abs()
            } else {
                boundary_z[i_point]
            };
        }
        Points { r: point_r, z: point_z }
    }

    /// The boundary conditions.
    ///
    /// * Model surfaces: the paper's matching conditions, in Guazzotto and Freidberg's order; then `(1 - psi_N) = 0` at `n_point_extra`
    ///   more points on the model surface (on each half for an asymmetric one).
    /// * `LcfsPoints`: `(1 - psi_N) = 0` at every point, then `(1 - psi_N) = d/dR = d/dZ = 0` at the X-point.
    ///
    /// # Arguments
    /// * `r_geo` - the paper's `R_geo` [metre]
    /// * `a_minor` - the paper's `a_minor` [metre]
    /// * `n_point_extra` - number of extra points on a model surface, even (unused for a user LCFS) [dimensionless]
    ///
    /// # Errors
    /// A double-null shape with `kappa_x <= 2 (1 - delta_x**2)**0.5`
    pub fn conditions(&self, r_geo: f64, a_minor: f64, n_point_extra: usize) -> Result<BoundaryConditions, String> {
        let mut conditions: BoundaryConditions = BoundaryConditions::new(r_geo, a_minor);
        let r_in: f64 = r_geo - a_minor;
        let r_out: f64 = r_geo + a_minor;
        match self {
            Configuration::SymmetricLimited { kappa, delta, .. } => {
                let constants: MillerMatchingConstants = miller_matching_constants(r_geo, a_minor, *kappa, *delta);
                let r_top: f64 = constants.r_top;
                let z_top: f64 = constants.z_top;
                conditions.add(r_in, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::DR, 0.0, 1.0, false);
                conditions.add(r_in, 0.0, ConditionKind::ZCurvature, constants.lambda_in, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::ZCurvature, constants.lambda_out, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::RCurvature, constants.lambda_top, 1.0, false);
                if n_point_extra > 0 {
                    let points: Points = miller_surface_points(r_geo, a_minor, *kappa, *delta, n_point_extra);
                    let n_point: usize = points.r.len();
                    for i_point in 0..n_point {
                        conditions.add(points.r[i_point], points.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                }
            }
            Configuration::DoubleNull { kappa_x, delta_x, .. } => {
                // The last, d/dZ = 0 at the X-point, is Guazzotto and Freidberg's eigenvalue condition
                let (_, constants): (Points, DoubleNullConstants) = double_null_surface(4, r_geo, a_minor, *kappa_x, *delta_x)?;
                let r_x_point: f64 = constants.r_x_point;
                let z_x_point: f64 = constants.z_x_point;
                conditions.add(r_in, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_x_point, z_x_point, ConditionKind::Value, 0.0, 1.0, true);
                conditions.add(r_x_point, z_x_point, ConditionKind::DR, 0.0, 1.0, true);
                conditions.add(r_in, 0.0, ConditionKind::ZCurvature, constants.lambda_in, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::ZCurvature, constants.lambda_out, 1.0, false);
                conditions.add(r_x_point, z_x_point, ConditionKind::DZ, 0.0, 1.0, true);
                if n_point_extra > 0 {
                    let points: Points = double_null_surface_points(r_geo, a_minor, *kappa_x, *delta_x, n_point_extra)?;
                    let n_point: usize = points.r.len();
                    for i_point in 0..n_point {
                        conditions.add(points.r[i_point], points.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                }
            }
            Configuration::SingleNull {
                kappa,
                delta,
                kappa_x,
                delta_x,
                ..
            } => {
                // Guazzotto and Freidberg's twelve conditions, the last, the curvature at the highest point, being the eigenvalue
                // condition. They print the X-point's slope conditions at (-delta_x, -kappa_x) but its flux condition at the X-point
                // itself; all three act at the X-point, which reproduces the eigenvalues of their Table 4
                let (_, constants): (Points, SingleNullConstants) = single_null_surface(3, 3, r_geo, a_minor, *kappa, *delta, *kappa_x, *delta_x)?;
                let r_top: f64 = constants.r_top;
                let z_top: f64 = *kappa * a_minor;
                let r_x_point: f64 = constants.r_x_point;
                let z_x_point: f64 = -*kappa_x * a_minor;
                conditions.add(r_in, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_x_point, z_x_point, ConditionKind::Value, 0.0, 1.0, true);
                conditions.add(r_in, 0.0, ConditionKind::DZ, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::DZ, 0.0, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::DR, 0.0, 1.0, false);
                conditions.add(r_x_point, z_x_point, ConditionKind::DR, 0.0, 1.0, true);
                conditions.add(r_x_point, z_x_point, ConditionKind::DZ, 0.0, 1.0, true);
                conditions.add(r_in, 0.0, ConditionKind::ZCurvature, constants.lambda_in, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::ZCurvature, constants.lambda_out, 1.0, false);
                conditions.add(r_top, z_top, ConditionKind::RCurvature, constants.lambda_top, 1.0, false);
                if n_point_extra > 0 {
                    let upper: Points = miller_surface_points(r_geo, a_minor, *kappa, *delta, n_point_extra);
                    let lower: Points = double_null_surface_points(r_geo, a_minor, *kappa_x, *delta_x, n_point_extra)?;
                    let n_upper: usize = upper.r.len();
                    for i_point in 0..n_upper {
                        conditions.add(upper.r[i_point], upper.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                    let n_lower: usize = lower.r.len();
                    for i_point in 0..n_lower {
                        conditions.add(lower.r[i_point], -lower.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                }
            }
            Configuration::AntisymmetricLimited {
                kappa_upper,
                delta_upper,
                kappa_lower,
                delta_lower,
                ..
            } => {
                // The value at both midplane points, the top and the bottom; d/dZ = 0 at both midplane points and d/dR = 0 at the top and
                // the bottom; the curvature at the bottom; the midplane curvatures with the two halves' Lambdas averaged; and the
                // curvature at the top, the eigenvalue condition
                let upper: MillerMatchingConstants = miller_matching_constants(r_geo, a_minor, *kappa_upper, *delta_upper);
                let lower: MillerMatchingConstants = miller_matching_constants(r_geo, a_minor, *kappa_lower, *delta_lower);
                conditions.add(r_in, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(upper.r_top, upper.z_top, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(lower.r_top, -lower.z_top, ConditionKind::Value, 0.0, 1.0, false);
                conditions.add(r_in, 0.0, ConditionKind::DZ, 0.0, 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::DZ, 0.0, 1.0, false);
                conditions.add(upper.r_top, upper.z_top, ConditionKind::DR, 0.0, 1.0, false);
                conditions.add(lower.r_top, -lower.z_top, ConditionKind::DR, 0.0, 1.0, false);
                // The lowest point's curvature, the mirror image of the highest point's
                conditions.add(lower.r_top, -lower.z_top, ConditionKind::RCurvature, -lower.lambda_top, 1.0, false);
                conditions.add(r_in, 0.0, ConditionKind::ZCurvature, 0.5 * (upper.lambda_in + lower.lambda_in), 1.0, false);
                conditions.add(r_out, 0.0, ConditionKind::ZCurvature, 0.5 * (upper.lambda_out + lower.lambda_out), 1.0, false);
                conditions.add(upper.r_top, upper.z_top, ConditionKind::RCurvature, upper.lambda_top, 1.0, false);
                if n_point_extra > 0 {
                    let upper_points: Points = miller_surface_points(r_geo, a_minor, *kappa_upper, *delta_upper, n_point_extra);
                    let lower_points: Points = miller_surface_points(r_geo, a_minor, *kappa_lower, *delta_lower, n_point_extra);
                    let n_upper: usize = upper_points.r.len();
                    for i_point in 0..n_upper {
                        conditions.add(upper_points.r[i_point], upper_points.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                    let n_lower: usize = lower_points.r.len();
                    for i_point in 0..n_lower {
                        conditions.add(lower_points.r[i_point], -lower_points.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                    }
                }
            }
            Configuration::LcfsPoints {
                boundary_r,
                boundary_z,
                x_point_weight,
                ..
            } => {
                let points: Points = Configuration::lcfs_points(boundary_r, boundary_z, self.is_up_down_symmetric());
                let n_point: usize = points.r.len();
                for i_point in 0..n_point {
                    conditions.add(points.r[i_point], points.z[i_point], ConditionKind::Value, 0.0, 1.0, false);
                }
                let x_points: Vec<Point> = self.x_points(r_geo, a_minor)?;
                if !x_points.is_empty() {
                    // One X-point carries the conditions: the upper one of a symmetric pair, whose mirror image follows by symmetry
                    conditions.add(x_points[0].r, x_points[0].z, ConditionKind::Value, 0.0, *x_point_weight, true);
                    conditions.add(x_points[0].r, x_points[0].z, ConditionKind::DR, 0.0, *x_point_weight, true);
                    conditions.add(x_points[0].r, x_points[0].z, ConditionKind::DZ, 0.0, *x_point_weight, true);
                }
            }
        }
        Ok(conditions)
    }

    /// Interior points for the least-squares normalisation (see `shrunk_interior_points`)
    pub fn interior_points(&self, r_geo: f64, a_minor: f64) -> Result<Points, String> {
        match self {
            Configuration::SymmetricLimited { kappa, delta, .. } => {
                let theta: Array1<f64> = Array1::linspace(0.0, PI, 25);
                let upper: Points = miller_surface(&theta, r_geo, a_minor, *kappa, *delta);
                Ok(shrunk_interior_points(&upper, 0.0))
            }
            Configuration::DoubleNull { kappa_x, delta_x, .. } => {
                let (surface, _): (Points, DoubleNullConstants) = double_null_surface(25, r_geo, a_minor, *kappa_x, *delta_x)?;
                let n_surface: usize = surface.r.len();
                let mut upper_r: Vec<f64> = Vec::new();
                let mut upper_z: Vec<f64> = Vec::new();
                for i_surface in 0..n_surface {
                    if surface.z[i_surface] >= 0.0 {
                        upper_r.push(surface.r[i_surface]);
                        upper_z.push(surface.z[i_surface]);
                    }
                }
                let upper: Points = Points {
                    r: Array1::from(upper_r),
                    z: Array1::from(upper_z),
                };
                Ok(shrunk_interior_points(&upper, 0.0))
            }
            Configuration::SingleNull {
                kappa,
                delta,
                kappa_x,
                delta_x,
                ..
            } => {
                let (surface, _): (Points, SingleNullConstants) = single_null_surface(25, 13, r_geo, a_minor, *kappa, *delta, *kappa_x, *delta_x)?;
                let n_surface: usize = surface.r.len();
                let open: Points = Points {
                    r: surface.r.slice(s![..n_surface - 1]).to_owned(),
                    z: surface.z.slice(s![..n_surface - 1]).to_owned(),
                };
                Ok(shrunk_interior_points(&open, 0.5 * (kappa - kappa_x) * a_minor))
            }
            Configuration::AntisymmetricLimited { kappa_upper, kappa_lower, .. } => {
                let surface: Points = self.outline(r_geo, a_minor, 49)?;
                let n_surface: usize = surface.r.len();
                let open: Points = Points {
                    r: surface.r.slice(s![..n_surface - 1]).to_owned(),
                    z: surface.z.slice(s![..n_surface - 1]).to_owned(),
                };
                Ok(shrunk_interior_points(&open, 0.5 * (kappa_upper - kappa_lower) * a_minor))
            }
            Configuration::LcfsPoints { boundary_r, boundary_z, .. } => {
                let is_up_down_symmetric: bool = self.is_up_down_symmetric();
                let points: Points = Configuration::lcfs_points(boundary_r, boundary_z, is_up_down_symmetric);
                if is_up_down_symmetric {
                    return Ok(shrunk_interior_points(&points, 0.0));
                }
                let mut z_minimum: f64 = f64::INFINITY;
                let mut z_maximum: f64 = f64::NEG_INFINITY;
                for point_z in points.z.iter() {
                    z_minimum = z_minimum.min(*point_z);
                    z_maximum = z_maximum.max(*point_z);
                }
                Ok(shrunk_interior_points(&points, 0.5 * (z_minimum + z_maximum)))
            }
        }
    }

    /// The closed shape the LCFS is fitted to, last point = first: the model surface, anticlockwise from the outboard midplane, or the
    /// user's points (with their mirror images for a symmetric LCFS) and the X-points, ordered by angle about the middle of their extent
    /// in `R**2` and `Z`.
    ///
    /// # Arguments
    /// * `r_geo` - the paper's `R_geo` [metre]
    /// * `a_minor` - the paper's `a_minor` [metre]
    /// * `n_point` - about the number of points (unused for a user LCFS) [dimensionless]
    pub fn outline(&self, r_geo: f64, a_minor: f64, n_point: usize) -> Result<Points, String> {
        match self {
            Configuration::SymmetricLimited { kappa, delta, .. } => {
                let theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_point);
                Ok(miller_surface(&theta, r_geo, a_minor, *kappa, *delta))
            }
            Configuration::DoubleNull { kappa_x, delta_x, .. } => {
                let n_point_per_ellipse: usize = (n_point / 2).max(3);
                let (surface, _): (Points, DoubleNullConstants) = double_null_surface(n_point_per_ellipse, r_geo, a_minor, *kappa_x, *delta_x)?;
                let first_r: Array1<f64> = Array1::from_elem(1, surface.r[0]);
                let first_z: Array1<f64> = Array1::from_elem(1, surface.z[0]);
                Ok(Points {
                    r: concatenate(Axis(0), &[surface.r.view(), first_r.view()]).expect("outline: concatenate"),
                    z: concatenate(Axis(0), &[surface.z.view(), first_z.view()]).expect("outline: concatenate"),
                })
            }
            Configuration::SingleNull {
                kappa,
                delta,
                kappa_x,
                delta_x,
                ..
            } => {
                let n_point_upper: usize = (n_point / 2).max(3);
                let n_point_per_ellipse: usize = (n_point / 4).max(3);
                let (surface, _): (Points, SingleNullConstants) =
                    single_null_surface(n_point_upper, n_point_per_ellipse, r_geo, a_minor, *kappa, *delta, *kappa_x, *delta_x)?;
                Ok(surface)
            }
            Configuration::AntisymmetricLimited {
                kappa_upper,
                delta_upper,
                kappa_lower,
                delta_lower,
                ..
            } => {
                let n_half: usize = (n_point / 2).max(3);
                let theta_upper: Array1<f64> = Array1::linspace(0.0, PI, n_half + 1);
                let theta_lower: Array1<f64> = Array1::linspace(PI, 2.0 * PI, n_half + 1);
                let upper: Points = miller_surface(&theta_upper, r_geo, a_minor, *kappa_upper, *delta_upper);
                let lower: Points = miller_surface(&theta_lower, r_geo, a_minor, *kappa_lower, *delta_lower);
                Ok(Points {
                    r: concatenate(Axis(0), &[upper.r.view(), lower.r.slice(s![1..])]).expect("outline: concatenate"),
                    z: concatenate(Axis(0), &[upper.z.view(), lower.z.slice(s![1..])]).expect("outline: concatenate"),
                })
            }
            Configuration::LcfsPoints { boundary_r, boundary_z, .. } => {
                let is_up_down_symmetric: bool = self.is_up_down_symmetric();
                let points: Points = Configuration::lcfs_points(boundary_r, boundary_z, is_up_down_symmetric);
                let n_point_given: usize = points.r.len();
                let mut all_r: Vec<f64> = points.r.to_vec();
                let mut all_z: Vec<f64> = points.z.to_vec();
                if is_up_down_symmetric {
                    for i_point in 0..n_point_given {
                        all_r.push(points.r[i_point]);
                        all_z.push(-points.z[i_point]);
                    }
                }
                let x_points: Vec<Point> = self.x_points(r_geo, a_minor)?;
                for x_point in &x_points {
                    all_r.push(x_point.r);
                    all_z.push(x_point.z);
                }
                // The angles about the middle, in R**2 scaled by 2 R_geo a_minor and Z scaled by a_minor, the plasma's half-widths
                let r_squared_half_width: f64 = 2.0 * r_geo * a_minor;
                let n_all: usize = all_r.len();
                let mut r_squared_minimum: f64 = f64::INFINITY;
                let mut r_squared_maximum: f64 = f64::NEG_INFINITY;
                let mut z_minimum: f64 = f64::INFINITY;
                let mut z_maximum: f64 = f64::NEG_INFINITY;
                for i_all in 0..n_all {
                    let r_squared: f64 = all_r[i_all] * all_r[i_all];
                    r_squared_minimum = r_squared_minimum.min(r_squared);
                    r_squared_maximum = r_squared_maximum.max(r_squared);
                    z_minimum = z_minimum.min(all_z[i_all]);
                    z_maximum = z_maximum.max(all_z[i_all]);
                }
                let centre_r_squared: f64 = 0.5 * (r_squared_minimum + r_squared_maximum);
                let centre_z: f64 = 0.5 * (z_minimum + z_maximum);
                let mut angles: Vec<f64> = Vec::with_capacity(n_all);
                for i_all in 0..n_all {
                    let shift_r_squared: f64 = (all_r[i_all] * all_r[i_all] - centre_r_squared) / r_squared_half_width;
                    let shift_z: f64 = (all_z[i_all] - centre_z) / a_minor;
                    angles.push(shift_z.atan2(shift_r_squared).rem_euclid(2.0 * PI));
                }
                let order: Vec<usize> = argsort_stable(&angles);
                let mut outline_r: Array1<f64> = Array1::from_elem(n_all + 1, f64::NAN);
                let mut outline_z: Array1<f64> = Array1::from_elem(n_all + 1, f64::NAN);
                for i_all in 0..n_all {
                    outline_r[i_all] = all_r[order[i_all]];
                    outline_z[i_all] = all_z[order[i_all]];
                }
                outline_r[n_all] = outline_r[0];
                outline_z[n_all] = outline_z[0];
                Ok(Points { r: outline_r, z: outline_z })
            }
        }
    }

    /// The X-points: none (limited shapes), the upper and lower ones of a double null, the lower one of a single null, or for a user
    /// LCFS the given one, or for a symmetric one the upper one and its mirror image
    pub fn x_points(&self, r_geo: f64, a_minor: f64) -> Result<Vec<Point>, String> {
        match self {
            Configuration::SymmetricLimited { .. } | Configuration::AntisymmetricLimited { .. } => Ok(Vec::new()),
            Configuration::DoubleNull { kappa_x, delta_x, .. } => {
                let (_, constants): (Points, DoubleNullConstants) = double_null_surface(4, r_geo, a_minor, *kappa_x, *delta_x)?;
                Ok(vec![
                    Point {
                        r: constants.r_x_point,
                        z: constants.z_x_point,
                    },
                    Point {
                        r: constants.r_x_point,
                        z: -constants.z_x_point,
                    },
                ])
            }
            Configuration::SingleNull { kappa_x, delta_x, .. } => {
                let (_, constants): (Points, DoubleNullConstants) = double_null_surface(4, r_geo, a_minor, *kappa_x, *delta_x)?;
                Ok(vec![Point {
                    r: constants.r_x_point,
                    z: -constants.z_x_point,
                }])
            }
            Configuration::LcfsPoints { x_point_r, x_point_z, .. } => match (x_point_r, x_point_z) {
                (Some(x_point_r_value), Some(x_point_z_value)) => {
                    if self.is_up_down_symmetric() {
                        Ok(vec![
                            Point {
                                r: *x_point_r_value,
                                z: x_point_z_value.abs(),
                            },
                            Point {
                                r: *x_point_r_value,
                                z: -x_point_z_value.abs(),
                            },
                        ])
                    } else {
                        Ok(vec![Point {
                            r: *x_point_r_value,
                            z: *x_point_z_value,
                        }])
                    }
                }
                _ => Ok(Vec::new()),
            },
        }
    }
}

/// The indices that sort `values` ascending, ties in their original order
pub fn argsort_stable(values: &[f64]) -> Vec<usize> {
    let n_value: usize = values.len();
    let mut order: Vec<usize> = Vec::with_capacity(n_value);
    for i_value in 0..n_value {
        order.push(i_value);
    }
    order.sort_by(|i_first: &usize, i_second: &usize| values[*i_first].partial_cmp(&values[*i_second]).unwrap_or(std::cmp::Ordering::Equal));
    order
}
