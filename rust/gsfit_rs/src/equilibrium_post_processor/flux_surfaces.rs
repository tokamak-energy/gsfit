//! The flux surfaces, and the quadrature with which every contour integral over them is taken.
//!
//! Unlike most calculators this one fills no data-dictionary path. Many profiles are contour integrals over the flux surfaces,
//! `contour_integral(x d_l / |grad(psi)|)`, and so are the flux-surface averages, so the surfaces, with the points and the weights
//! of that quadrature, are found once and passed to the calculators which need them.
//!
//! # The surfaces inside the boundary, `0 < psi_N < 1`
//!
//! Each surface is found along rays from the magnetic axis. With polar coordinates about the axis,
//! `R = mag_r + r * cos(theta)` and `Z = mag_z + r * sin(theta)`, where the ray at `theta` meets the surface at `r(theta)`,
//!
//! ```text
//! contour_integral(x d_l / |grad(psi)|) = integral(x * r / |d(psi)/d(r)| d(theta)),    0 <= theta < 2 * pi,
//! ```
//!
//! because the element of the surface `d_l` and its projection on the circle about the axis, `r d(theta)`, are in the ratio of
//! `|grad(psi)|` to `|d(psi)/d(r)|`. The integrand is smooth and periodic in `theta`, so the trapezoidal rule on equally spaced rays
//! converges much faster than chords through the points where a surface crosses the grid lines. The quadrature weight of each ray is
//! `(2 * pi / n_ray) * r / |d(psi)/d(r)|`.
//!
//! The number of rays is not fixed. A surface starts with `N_RAY_START` rays, which are doubled, the new rays half-way between the
//! old ones, until the contour integrals `integral(d_l / (R * |grad(psi)|))`, which is `q / f`, and `integral(R * d_l / |grad(psi)|)`,
//! which is `d(volume)/d(psi) / (2 * pi)`, change by less than `RAY_RELATIVE_TOLERANCE`. The surfaces in the core take few rays;
//! a surface close to the separatrix of a diverted plasma, whose integrand is peaked towards the X-point, takes more.
//!
//! The rays need every surface to be star-shaped about the magnetic axis: each ray meets it once.
//!
//! # The magnetic axis, `psi_N = 0`
//!
//! The axis is a point, and the contour integrals over the surfaces next to it tend to a limit: `psi` is quadratic about the axis,
//! `r / |d(psi)/d(r)| -> 1 / |d2(psi)/d(r)2|`, and the integral of that over `theta` is `2 * pi / sqrt(det(H))`, with `H` the Hessian
//! of `psi` at the axis. So the axis is a surface with one quadrature point, on the axis, of weight `2 * pi / sqrt(det(H))`, and
//! every contour integral and flux-surface average comes out as its limit there.
//!
//! # The last closed flux surface, `psi_N = 1`
//!
//! The last surface is the boundary outline, which is traced with the X-point in it. A ray towards the X-point only touches the
//! separatrix there, so the rays cannot find it. Its contour integrals use two-point Gauss-Legendre quadrature on each straight segment
//! of the outline.

use super::constant_values::ConstantValues;
use super::intermediate_values::IntermediateValues;
use super::psi_interpolator::{PsiDerivatives, PsiInterpolator, PsiValueAndGradient};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// The number of rays each surface starts with [count]
const N_RAY_START: usize = 64;
/// Safety: the most rays on one surface [count]
const N_RAY_MAX: usize = 4096;
/// The rays are doubled until the contour integrals change by less than this fraction [dimensionless]
const RAY_RELATIVE_TOLERANCE: f64 = 1.0e-6;
/// The step with which a ray is marched to bracket its crossing with a surface, as a fraction of the smaller grid spacing
/// [dimensionless]
const MARCH_STEP_FRACTION: f64 = 0.25;
/// Safety: the most steps a ray is marched to bracket its crossing with a surface [count]
const N_MARCH_MAX: usize = 100_000;
/// Safety: the most Newton iterations for one crossing [count]
const N_NEWTON_MAX: usize = 100;
/// Newton's method has converged when the crossing moves by less than this fraction of its distance from the magnetic axis
/// [dimensionless]
const DISTANCE_RELATIVE_TOLERANCE: f64 = 1.0e-14;
/// The two-point Gauss-Legendre abscissa on `[-1, 1]`, `1 / sqrt(3)` [dimensionless]
const GAUSS_LEGENDRE_ABSCISSA: f64 = 0.5773502691896258;

/// One closed flux surface: its shape, and the quadrature of the contour integrals over it.
///
/// A surface which could not be found is stored empty, rather than being absent, so that `flux_surfaces[i_psi_norm]` always
/// corresponds to `psi_norm[i_psi_norm]`; every contour integral over it is NaN.
#[derive(Clone)]
pub struct FluxSurface {
    /// The surface as a closed polygon, its first point repeated at its end: where the rays cross it, in order of their angle, or the
    /// boundary outline for the last closed flux surface; the magnetic axis, for `psi_N = 0` [metre]
    pub r: Array1<f64>,
    /// [metre]
    pub z: Array1<f64>,
    /// The quadrature points on the surface [metre]
    pub quadrature_r: Array1<f64>,
    /// [metre]
    pub quadrature_z: Array1<f64>,
    /// `|grad(psi)|` at each quadrature point [weber / metre]
    pub quadrature_grad_psi: Array1<f64>,
    /// The weight of each quadrature point: `contour_integral(x d_l / |grad(psi)|) = sum(x * quadrature_weight)`
    /// [metre ** 2 / weber]
    pub quadrature_weight: Array1<f64>,
    /// The area the surface encloses in the poloidal plane [metre ** 2]
    pub area: f64,
    /// The volume the surface encloses [metre ** 3]
    pub volume: f64,
}

/// The largest and the smallest value of a quantity on a flux surface
pub struct ValueRange {
    /// [any]
    pub maximum: f64,
    /// [any]
    pub minimum: f64,
}

impl FluxSurface {
    /// The contour integral `contour_integral(x d_l / |grad(psi)|)` around the surface.
    ///
    /// # Arguments
    /// * `values` - `x`, as a function of the major radius [metre] and `|grad(psi)|` [weber / metre]
    ///
    /// # Returns
    /// * The integral; NaN for a surface which could not be found [any * metre ** 2 / weber]
    pub fn contour_integral<F: Fn(f64, f64) -> f64>(&self, values: F) -> f64 {
        let n_quadrature: usize = self.quadrature_weight.len();
        if n_quadrature == 0 {
            return f64::NAN;
        }
        let mut integral: f64 = 0.0;
        for i_quadrature in 0..n_quadrature {
            integral += values(self.quadrature_r[i_quadrature], self.quadrature_grad_psi[i_quadrature]) * self.quadrature_weight[i_quadrature];
        }

        integral
    }

    /// The flux-surface average, `<x> = contour_integral(x d_l / b_p) / contour_integral(d_l / b_p)`, with
    /// `d_l / b_p = 2 * pi * R * d_l / |grad(psi)|`.
    ///
    /// # Arguments
    /// * `values` - `x`, as a function of the major radius [metre] and `|grad(psi)|` [weber / metre]
    ///
    /// # Returns
    /// * `<x>`; NaN for a surface which could not be found [any]
    pub fn flux_surface_average<F: Fn(f64, f64) -> f64>(&self, values: F) -> f64 {
        let numerator: f64 = self.contour_integral(|r_here: f64, grad_psi_here: f64| values(r_here, grad_psi_here) * r_here);
        let denominator: f64 = self.contour_integral(|r_here: f64, _grad_psi_here: f64| r_here);

        numerator / denominator
    }

    /// The largest and the smallest value of a quantity on the surface.
    ///
    /// The largest and the smallest of the values at the quadrature points are each refined by the parabola through them and their two
    /// neighbours, in the length along the surface; a parabola whose vertex is not between the two neighbours is not used.
    ///
    /// # Arguments
    /// * `values` - the quantity, as a function of the major radius [metre] and `|grad(psi)|` [weber / metre]
    ///
    /// # Returns
    /// * The range; NaN for a surface which could not be found
    pub fn value_range<F: Fn(f64, f64) -> f64>(&self, values: F) -> ValueRange {
        let n_quadrature: usize = self.quadrature_weight.len();
        if n_quadrature == 0 {
            return ValueRange {
                maximum: f64::NAN,
                minimum: f64::NAN,
            };
        }
        let mut quadrature_values: Array1<f64> = Array1::from_elem(n_quadrature, f64::NAN);
        for i_quadrature in 0..n_quadrature {
            quadrature_values[i_quadrature] = values(self.quadrature_r[i_quadrature], self.quadrature_grad_psi[i_quadrature]);
        }
        let mut i_maximum: usize = 0;
        let mut i_minimum: usize = 0;
        for i_quadrature in 0..n_quadrature {
            if quadrature_values[i_quadrature] > quadrature_values[i_maximum] {
                i_maximum = i_quadrature;
            }
            if quadrature_values[i_quadrature] < quadrature_values[i_minimum] {
                i_minimum = i_quadrature;
            }
        }

        ValueRange {
            maximum: self.parabolic_extreme(&quadrature_values, i_maximum),
            minimum: self.parabolic_extreme(&quadrature_values, i_minimum),
        }
    }

    /// The extreme of the parabola through the values at quadrature point `i_centre` and its two neighbours, in the length along the
    /// surface; the value at `i_centre` when there are fewer than three points, or when the parabola's vertex is not between the two
    /// neighbours.
    fn parabolic_extreme(&self, quadrature_values: &Array1<f64>, i_centre: usize) -> f64 {
        let n_quadrature: usize = quadrature_values.len();
        let value_centre: f64 = quadrature_values[i_centre];
        if n_quadrature < 3 {
            return value_centre;
        }
        let i_before: usize = (i_centre + n_quadrature - 1) % n_quadrature;
        let i_after: usize = (i_centre + 1) % n_quadrature;
        let length_before: f64 = (self.quadrature_r[i_centre] - self.quadrature_r[i_before]).hypot(self.quadrature_z[i_centre] - self.quadrature_z[i_before]);
        let length_after: f64 = (self.quadrature_r[i_after] - self.quadrature_r[i_centre]).hypot(self.quadrature_z[i_after] - self.quadrature_z[i_centre]);
        if length_before <= 0.0 || length_after <= 0.0 {
            return value_centre;
        }
        let change_before: f64 = quadrature_values[i_before] - value_centre;
        let change_after: f64 = quadrature_values[i_after] - value_centre;
        // `value = value_centre + slope * s + curvature * s ** 2`, through `s = -length_before` and `s = length_after`
        let denominator: f64 = length_before * length_after * (length_before + length_after);
        let curvature: f64 = (length_before * change_after + length_after * change_before) / denominator;
        let slope: f64 = (length_before.powi(2) * change_after - length_after.powi(2) * change_before) / denominator;
        if curvature == 0.0 {
            return value_centre;
        }
        let s_vertex: f64 = -slope / (2.0 * curvature);
        if s_vertex < -length_before || s_vertex > length_after {
            return value_centre;
        }

        value_centre - slope.powi(2) / (4.0 * curvature)
    }
}

/// Find the flux surfaces and store them on the shared intermediate values.
///
/// This is the uniform-signature entry point the post-processor dispatches; the surfaces themselves are found by
/// `calculate_flux_surfaces`.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `intermediate_values` - the shared intermediate values; `flux_surfaces` is written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, intermediate_values: &mut IntermediateValues) {
    let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(time_slice);
    intermediate_values.flux_surfaces = flux_surfaces;
}

/// Find the flux surface at each `psi_norm`, and return them.
///
/// `psi_norm = 0` is the magnetic axis, the last `psi_norm` is the boundary outline, and every other surface is found along rays from
/// the magnetic axis, each searched for from where the rays crossed the surface inside it; see the module's description.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
///
/// # Returns
/// One `FluxSurface` per `psi_norm`, in the same order.
pub(crate) fn calculate_flux_surfaces(time_slice: &EquilibriumTimeSlice) -> Vec<FluxSurface> {
    let psi_norm: &Array1<f64> = &time_slice.profiles_1d.psi_norm;
    let n_psi_norm: usize = psi_norm.len();
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);

    let mut flux_surfaces: Vec<FluxSurface> = Vec::with_capacity(n_psi_norm);
    // Where the rays crossed the last surface found, which the next one, further out, is searched for from
    let mut distances_previous: Array1<f64> = Array1::zeros(0);
    let mut psi_norm_previous: f64 = 0.0;
    for i_psi_norm in 0..n_psi_norm {
        if i_psi_norm == n_psi_norm - 1 && n_psi_norm > 1 {
            flux_surfaces.push(polygon_flux_surface(
                &time_slice.boundary.outline.r,
                &time_slice.boundary.outline.z,
                &psi_interpolator,
            ));
        } else if psi_norm[i_psi_norm] == 0.0 {
            flux_surfaces.push(axis_flux_surface(time_slice, &psi_interpolator));
        } else {
            if psi_norm[i_psi_norm] < psi_norm_previous {
                distances_previous = Array1::zeros(0);
            }
            match ray_traced_flux_surface(time_slice, &psi_interpolator, psi_norm[i_psi_norm], &distances_previous) {
                Some(ray_traced) => {
                    distances_previous = ray_traced.distances;
                    psi_norm_previous = psi_norm[i_psi_norm];
                    flux_surfaces.push(ray_traced.flux_surface);
                }
                None => {
                    distances_previous = Array1::zeros(0);
                    psi_norm_previous = 0.0;
                    flux_surfaces.push(empty_flux_surface());
                }
            }
        }
    }

    flux_surfaces
}

/// Find the flux surface at one normalised poloidal flux strictly inside the boundary, along rays from the magnetic axis.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `psi_norm` - the flux surface, `0 < psi_N < 1` [dimensionless]
///
/// # Returns
/// * The surface; empty when it cannot be found, or `psi_norm` is not strictly inside the boundary
pub(in crate::equilibrium_post_processor) fn calculate_at_psi_norm(time_slice: &EquilibriumTimeSlice, psi_norm: f64) -> FluxSurface {
    if !psi_norm.is_finite() || psi_norm <= 0.0 || psi_norm >= 1.0 {
        return empty_flux_surface();
    }
    let psi_interpolator: PsiInterpolator = PsiInterpolator::new(time_slice);
    match ray_traced_flux_surface(time_slice, &psi_interpolator, psi_norm, &Array1::zeros(0)) {
        Some(ray_traced) => ray_traced.flux_surface,
        None => empty_flux_surface(),
    }
}

/// A surface found along rays, and how far along each ray it is.
struct RayTracedFluxSurface {
    flux_surface: FluxSurface,
    /// The distance from the magnetic axis along each ray, the rays equally spaced in angle from `theta = 0` [metre]
    distances: Array1<f64>,
}

/// One ray's crossing with a surface.
#[derive(Clone)]
struct RayCrossing {
    /// The distance from the magnetic axis [metre]
    distance: f64,
    /// [metre]
    r: f64,
    /// [metre]
    z: f64,
    /// `|grad(psi)|` [weber / metre]
    grad_psi: f64,
    /// `|d(psi)/d(distance)|` along the ray [weber / metre]
    d_psi_d_distance: f64,
}

/// Find one surface along rays from the magnetic axis, doubling the rays until its contour integrals converge.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `psi_interpolator` - the bicubic models of `psi` and `grad(psi)` built from `time_slice`
/// * `psi_norm` - the surface, `0 < psi_N < 1` [dimensionless]
/// * `distances_previous` - where the rays crossed a surface inside this one, the rays equally spaced in angle from `theta = 0`;
///   empty to search from the magnetic axis [metre]
///
/// # Returns
/// * The surface; `None` when a ray cannot find it, e.g. by leaving the grid
fn ray_traced_flux_surface(
    time_slice: &EquilibriumTimeSlice,
    psi_interpolator: &PsiInterpolator,
    psi_norm: f64,
    distances_previous: &Array1<f64>,
) -> Option<RayTracedFluxSurface> {
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;

    // The first rays, each searched for from where the ray at the same angle crossed the surface inside
    let mut crossings: Vec<RayCrossing> = Vec::with_capacity(N_RAY_START);
    for i_ray in 0..N_RAY_START {
        let theta: f64 = 2.0 * PI * (i_ray as f64) / (N_RAY_START as f64);
        let distance_guess: f64 = periodic_interpolation(distances_previous, theta);
        crossings.push(ray_crossing(time_slice, psi_interpolator, psi_norm, theta, distance_guess)?);
    }
    let mut integrals: [f64; 2] = ray_contour_integrals(&crossings);

    // Double the rays until the contour integrals converge, each new ray searched for from between its two neighbours
    let n_doubling_max: usize = (N_RAY_MAX / N_RAY_START).ilog2() as usize;
    let mut is_converged: bool = false;
    'doubling_loop: for _i_doubling in 0..n_doubling_max {
        let n_ray: usize = crossings.len();
        let mut crossings_doubled: Vec<RayCrossing> = Vec::with_capacity(2 * n_ray);
        for i_ray in 0..n_ray {
            let theta: f64 = 2.0 * PI * (2 * i_ray + 1) as f64 / (2 * n_ray) as f64;
            let distance_guess: f64 = 0.5 * (crossings[i_ray].distance + crossings[(i_ray + 1) % n_ray].distance);
            crossings_doubled.push(crossings[i_ray].clone());
            crossings_doubled.push(ray_crossing(time_slice, psi_interpolator, psi_norm, theta, distance_guess)?);
        }
        let integrals_doubled: [f64; 2] = ray_contour_integrals(&crossings_doubled);
        crossings = crossings_doubled;
        let mut change_largest: f64 = 0.0;
        for i_integral in 0..2 {
            change_largest = change_largest.max((integrals_doubled[i_integral] / integrals[i_integral] - 1.0).abs());
        }
        integrals = integrals_doubled;
        if change_largest <= RAY_RELATIVE_TOLERANCE {
            is_converged = true;
            break 'doubling_loop;
        }
    }
    if !is_converged {
        println!(
            "Warning: flux surface psi_N = {psi_norm}: its contour integrals had not converged to {RAY_RELATIVE_TOLERANCE:e} with {} rays",
            crossings.len()
        );
    }

    // The surface, its quadrature, and the area and volume it encloses:
    // area = integral(r ** 2 / 2 d(theta)), volume = 2 * pi * integral((mag_r * r ** 2 / 2 + r ** 3 * cos(theta) / 3) d(theta))
    let n_ray: usize = crossings.len();
    let d_theta: f64 = 2.0 * PI / (n_ray as f64);
    let mut polygon_r: Array1<f64> = Array1::from_elem(n_ray + 1, f64::NAN);
    let mut polygon_z: Array1<f64> = Array1::from_elem(n_ray + 1, f64::NAN);
    let mut quadrature_grad_psi: Array1<f64> = Array1::from_elem(n_ray, f64::NAN);
    let mut quadrature_weight: Array1<f64> = Array1::from_elem(n_ray, f64::NAN);
    let mut distances: Array1<f64> = Array1::from_elem(n_ray, f64::NAN);
    let mut area: f64 = 0.0;
    let mut volume: f64 = 0.0;
    for i_ray in 0..n_ray {
        let crossing: &RayCrossing = &crossings[i_ray];
        let theta: f64 = d_theta * (i_ray as f64);
        polygon_r[i_ray] = crossing.r;
        polygon_z[i_ray] = crossing.z;
        quadrature_grad_psi[i_ray] = crossing.grad_psi;
        quadrature_weight[i_ray] = d_theta * crossing.distance / crossing.d_psi_d_distance;
        distances[i_ray] = crossing.distance;
        area += d_theta * 0.5 * crossing.distance.powi(2);
        volume += 2.0 * PI * d_theta * (0.5 * mag_r * crossing.distance.powi(2) + crossing.distance.powi(3) * theta.cos() / 3.0);
    }
    polygon_r[n_ray] = polygon_r[0];
    polygon_z[n_ray] = polygon_z[0];
    let quadrature_r: Array1<f64> = polygon_r.slice(ndarray::s![..n_ray]).to_owned();
    let quadrature_z: Array1<f64> = polygon_z.slice(ndarray::s![..n_ray]).to_owned();

    Some(RayTracedFluxSurface {
        flux_surface: FluxSurface {
            r: polygon_r,
            z: polygon_z,
            quadrature_r,
            quadrature_z,
            quadrature_grad_psi,
            quadrature_weight,
            area,
            volume,
        },
        distances,
    })
}

/// The two contour integrals whose convergence decides the number of rays, `integral(d_l / (R * |grad(psi)|))` and
/// `integral(R * d_l / |grad(psi)|)`, by the trapezoidal rule on the rays, which are equally spaced in angle.
fn ray_contour_integrals(crossings: &[RayCrossing]) -> [f64; 2] {
    let n_ray: usize = crossings.len();
    let d_theta: f64 = 2.0 * PI / (n_ray as f64);
    let mut integral_over_r: f64 = 0.0;
    let mut integral_times_r: f64 = 0.0;
    for crossing in crossings {
        let weight: f64 = d_theta * crossing.distance / crossing.d_psi_d_distance;
        integral_over_r += weight / crossing.r;
        integral_times_r += weight * crossing.r;
    }

    [integral_over_r, integral_times_r]
}

/// Where one ray from the magnetic axis crosses a flux surface.
///
/// The crossing is bracketed by marching along the ray from `distance_guess`, outwards when `psi_N` there is inside the surface and
/// inwards when it is outside, in steps of `MARCH_STEP_FRACTION` of the grid spacing; it is then found with Newton's method on the
/// bicubic of `psi`, whose contours the flux surfaces are, kept inside the bracket by bisection. `|grad(psi)|` and `d(psi)/d(distance)`
/// at the crossing are from the bicubics of the components of `grad(psi)`, as the poloidal field is everywhere else.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `psi_interpolator` - the bicubic models of `psi` and `grad(psi)` built from `time_slice`
/// * `psi_norm` - the surface [dimensionless]
/// * `theta` - the ray's angle about the magnetic axis, from the outboard midplane towards the top [radian]
/// * `distance_guess` - where along the ray to start the search [metre]
///
/// # Returns
/// * The crossing; `None` when the ray leaves the grid, or reaches no crossing
fn ray_crossing(time_slice: &EquilibriumTimeSlice, psi_interpolator: &PsiInterpolator, psi_norm: f64, theta: f64, distance_guess: f64) -> Option<RayCrossing> {
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let psi_a: f64 = time_slice.global_quantities.psi_magnetic_axis;
    let psi_b: f64 = time_slice.boundary.psi;
    let cos_theta: f64 = theta.cos();
    let sin_theta: f64 = theta.sin();
    let march_step: f64 = MARCH_STEP_FRACTION * psi_interpolator.d_r.abs().min(psi_interpolator.d_z.abs());
    let psi_norm_at = |distance: f64| -> Option<f64> {
        let psi: f64 = psi_interpolator.psi(mag_r + distance * cos_theta, mag_z + distance * sin_theta)?;
        Some((psi - psi_a) / (psi_b - psi_a))
    };

    // Bracket the crossing: `psi_N` is below the surface's at `distance_low`, and at or above it at `distance_high`
    let distance_start: f64 = if distance_guess.is_finite() { distance_guess.max(0.0) } else { 0.0 };
    let mut distance_low: f64 = f64::NAN;
    let mut distance_high: f64 = f64::NAN;
    if psi_norm_at(distance_start)? < psi_norm {
        distance_low = distance_start;
        'march_out_loop: for _i_march in 0..N_MARCH_MAX {
            let distance_try: f64 = distance_low + march_step;
            if psi_norm_at(distance_try)? >= psi_norm {
                distance_high = distance_try;
                break 'march_out_loop;
            }
            distance_low = distance_try;
        }
    } else {
        distance_high = distance_start;
        'march_in_loop: for _i_march in 0..N_MARCH_MAX {
            if distance_high <= 0.0 {
                break 'march_in_loop;
            }
            let distance_try: f64 = (distance_high - march_step).max(0.0);
            if psi_norm_at(distance_try)? < psi_norm {
                distance_low = distance_try;
                break 'march_in_loop;
            }
            distance_high = distance_try;
        }
    }
    if !distance_low.is_finite() || !distance_high.is_finite() {
        return None;
    }

    // Newton's method on the bicubic of `psi`, falling back to bisection whenever a step would leave the bracket
    let mut distance: f64 = 0.5 * (distance_low + distance_high);
    'newton_loop: for _i_newton in 0..N_NEWTON_MAX {
        let psi_here: PsiValueAndGradient = psi_interpolator.psi_value_and_gradient(mag_r + distance * cos_theta, mag_z + distance * sin_theta)?;
        let residual: f64 = (psi_here.psi - psi_a) / (psi_b - psi_a) - psi_norm;
        let d_residual_d_distance: f64 = (psi_here.d_psi_d_r * cos_theta + psi_here.d_psi_d_z * sin_theta) / (psi_b - psi_a);
        if residual < 0.0 {
            distance_low = distance;
        } else {
            distance_high = distance;
        }
        let distance_newton: f64 = distance - residual / d_residual_d_distance;
        let distance_next: f64 = if distance_newton > distance_low && distance_newton < distance_high {
            distance_newton
        } else {
            0.5 * (distance_low + distance_high)
        };
        let distance_change: f64 = (distance_next - distance).abs();
        distance = distance_next;
        if distance_change <= DISTANCE_RELATIVE_TOLERANCE * distance {
            break 'newton_loop;
        }
    }

    let r_here: f64 = mag_r + distance * cos_theta;
    let z_here: f64 = mag_z + distance * sin_theta;
    let psi_derivatives: PsiDerivatives = psi_interpolator.derivatives(r_here, z_here)?;
    let grad_psi: f64 = psi_derivatives.d_psi_d_r.hypot(psi_derivatives.d_psi_d_z);
    let d_psi_d_distance: f64 = (psi_derivatives.d_psi_d_r * cos_theta + psi_derivatives.d_psi_d_z * sin_theta).abs();
    if r_here <= 0.0 || !grad_psi.is_finite() || !d_psi_d_distance.is_finite() || d_psi_d_distance == 0.0 {
        return None;
    }

    Some(RayCrossing {
        distance,
        r: r_here,
        z: z_here,
        grad_psi,
        d_psi_d_distance,
    })
}

/// Linear interpolation of a quantity on rays equally spaced in angle from `theta = 0`, which is periodic.
///
/// # Arguments
/// * `values` - the quantity on each ray; empty to interpolate nothing [any]
/// * `theta` - the angle [radian]
///
/// # Returns
/// * The quantity at `theta`; 0 when `values` is empty [any]
fn periodic_interpolation(values: &Array1<f64>, theta: f64) -> f64 {
    let n_value: usize = values.len();
    if n_value == 0 {
        return 0.0;
    }
    let position: f64 = theta.rem_euclid(2.0 * PI) / (2.0 * PI) * (n_value as f64);
    let i_before: usize = (position.floor() as usize) % n_value;
    let i_after: usize = (i_before + 1) % n_value;
    let fraction: f64 = position - position.floor();

    (1.0 - fraction) * values[i_before] + fraction * values[i_after]
}

/// The magnetic axis as a surface with one quadrature point, on the axis, whose weight is the limit of the contour integrals,
/// `2 * pi / sqrt(det(H))`, with `H` the Hessian of `psi` there.
///
/// # Arguments
/// * `time_slice` - the solved time-slice, read only
/// * `psi_interpolator` - the bicubic models of `psi` and `grad(psi)` built from `time_slice`
///
/// # Returns
/// * The axis; empty when the Hessian there is not that of an extremum
fn axis_flux_surface(time_slice: &EquilibriumTimeSlice, psi_interpolator: &PsiInterpolator) -> FluxSurface {
    let mag_r: f64 = time_slice.global_quantities.magnetic_axis.r;
    let mag_z: f64 = time_slice.global_quantities.magnetic_axis.z;
    let Some((_hessian_matrix, hessian_determinant, _hessian_trace)) = psi_interpolator.hessian_matrix(mag_r, mag_z) else {
        return empty_flux_surface();
    };
    if !hessian_determinant.is_finite() || hessian_determinant <= 0.0 || mag_r <= 0.0 {
        return empty_flux_surface();
    }

    FluxSurface {
        r: Array1::from_elem(1, mag_r),
        z: Array1::from_elem(1, mag_z),
        quadrature_r: Array1::from_elem(1, mag_r),
        quadrature_z: Array1::from_elem(1, mag_z),
        quadrature_grad_psi: Array1::zeros(1),
        quadrature_weight: Array1::from_elem(1, 2.0 * PI / hessian_determinant.sqrt()),
        area: 0.0,
        volume: 0.0,
    }
}

/// A surface given as a closed polygon, the boundary outline: two-point Gauss-Legendre quadrature on each of its segments, and the
/// area and volume the polygon encloses.
///
/// # Arguments
/// * `polygon_r`, `polygon_z` - the polygon [metre]
/// * `psi_interpolator` - the bicubic models of `psi` and `grad(psi)`
///
/// # Returns
/// * The surface; empty for fewer than four vertices, and without quadrature when `grad(psi)` cannot be found at a point
fn polygon_flux_surface(polygon_r: &Array1<f64>, polygon_z: &Array1<f64>, psi_interpolator: &PsiInterpolator) -> FluxSurface {
    let n_vertex: usize = polygon_r.len();
    if n_vertex < 4 || polygon_z.len() != n_vertex {
        return empty_flux_surface();
    }

    // The area and the centroid, by the shoelace formula, which hold for a polygon whose first point is repeated at its end or not;
    // the volume is the area times the length of the circle its centroid sweeps out
    let mut area_signed: f64 = 0.0;
    let mut centroid_r_moment: f64 = 0.0;
    for i_vertex in 0..n_vertex {
        let i_next: usize = (i_vertex + 1) % n_vertex;
        let cross_product: f64 = polygon_r[i_vertex] * polygon_z[i_next] - polygon_r[i_next] * polygon_z[i_vertex];
        area_signed += 0.5 * cross_product;
        centroid_r_moment += (polygon_r[i_vertex] + polygon_r[i_next]) * cross_product;
    }
    let centroid_r: f64 = centroid_r_moment / (6.0 * area_signed);
    let area: f64 = area_signed.abs();
    let volume: f64 = 2.0 * PI * centroid_r * area;

    let mut quadrature_r: Vec<f64> = Vec::with_capacity(2 * n_vertex);
    let mut quadrature_z: Vec<f64> = Vec::with_capacity(2 * n_vertex);
    let mut quadrature_grad_psi: Vec<f64> = Vec::with_capacity(2 * n_vertex);
    let mut quadrature_weight: Vec<f64> = Vec::with_capacity(2 * n_vertex);
    let mut is_quadrature_found: bool = true;
    'segment_loop: for i_vertex in 0..n_vertex {
        let i_next: usize = (i_vertex + 1) % n_vertex;
        let delta_r: f64 = polygon_r[i_next] - polygon_r[i_vertex];
        let delta_z: f64 = polygon_z[i_next] - polygon_z[i_vertex];
        let length: f64 = delta_r.hypot(delta_z);
        if length == 0.0 {
            continue 'segment_loop;
        }
        for sign in [-1.0, 1.0] {
            let point_r: f64 = 0.5 * (polygon_r[i_vertex] + polygon_r[i_next]) + sign * 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_r;
            let point_z: f64 = 0.5 * (polygon_z[i_vertex] + polygon_z[i_next]) + sign * 0.5 * GAUSS_LEGENDRE_ABSCISSA * delta_z;
            let grad_psi: f64 = match psi_interpolator.derivatives(point_r, point_z) {
                Some(psi_derivatives) => psi_derivatives.d_psi_d_r.hypot(psi_derivatives.d_psi_d_z),
                None => f64::NAN,
            };
            if !grad_psi.is_finite() || grad_psi == 0.0 || point_r <= 0.0 {
                is_quadrature_found = false;
                break 'segment_loop;
            }
            quadrature_r.push(point_r);
            quadrature_z.push(point_z);
            quadrature_grad_psi.push(grad_psi);
            quadrature_weight.push(0.5 * length / grad_psi);
        }
    }
    if !is_quadrature_found {
        quadrature_r.clear();
        quadrature_z.clear();
        quadrature_grad_psi.clear();
        quadrature_weight.clear();
    }

    FluxSurface {
        r: polygon_r.to_owned(),
        z: polygon_z.to_owned(),
        quadrature_r: Array1::from_vec(quadrature_r),
        quadrature_z: Array1::from_vec(quadrature_z),
        quadrature_grad_psi: Array1::from_vec(quadrature_grad_psi),
        quadrature_weight: Array1::from_vec(quadrature_weight),
        area,
        volume,
    }
}

/// A surface which could not be found
pub(in crate::equilibrium_post_processor) fn empty_flux_surface() -> FluxSurface {
    FluxSurface {
        r: Array1::zeros(0),
        z: Array1::zeros(0),
        quadrature_r: Array1::zeros(0),
        quadrature_z: Array1::zeros(0),
        quadrature_grad_psi: Array1::zeros(0),
        quadrature_weight: Array1::zeros(0),
        area: f64::NAN,
        volume: f64::NAN,
    }
}

/// A surface with only a shape, the closed polygon `r`, `z`, and no quadrature, for the tests of the calculators which only measure
/// the shape
#[cfg(test)]
pub(in crate::equilibrium_post_processor) fn shape_only_flux_surface(r: Array1<f64>, z: Array1<f64>) -> FluxSurface {
    FluxSurface {
        r,
        z,
        quadrature_r: Array1::zeros(0),
        quadrature_z: Array1::zeros(0),
        quadrature_grad_psi: Array1::zeros(0),
        quadrature_weight: Array1::zeros(0),
        area: f64::NAN,
        volume: f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::super::boundary::geometry::time_slice_with_elliptical_psi;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Circular surfaces, `psi = 1 - a * ((R - mag_r) ** 2 + (Z - mag_z) ** 2)`, about `(0.5, 0.0)`, with the boundary 0.2 m from the
    /// axis; `psi` is quadratic, so the bicubics reproduce it exactly
    fn time_slice_with_circular_psi() -> EquilibriumTimeSlice {
        let a: f64 = 25.0;
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(0.5, 0.0, a, 0.0, a);
        let minor_radius: f64 = 0.2;
        time_slice.boundary.psi = 1.0 - a * minor_radius.powi(2);
        time_slice
    }

    #[test]
    fn circular_surfaces_match_their_analytic_integrals_area_and_volume() {
        let time_slice: EquilibriumTimeSlice = time_slice_with_circular_psi();
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);
        let a: f64 = 25.0;
        let mag_r: f64 = 0.5;
        for psi_norm in [1.0e-8, 0.01, 0.25, 0.81] {
            let ray_traced: RayTracedFluxSurface =
                ray_traced_flux_surface(&time_slice, &psi_interpolator, psi_norm, &Array1::zeros(0)).expect("the surface is on the grid");
            let surface: FluxSurface = ray_traced.flux_surface;
            // The circle of radius rho about the axis, where |grad(psi)| = 2 a rho; to rounding, as psi, which is about 1, is rounded to
            // about 1e-16, which is 1e-16 / psi_norm of psi - psi_axis
            let rho: f64 = 0.2 * psi_norm.sqrt();
            let tolerance: f64 = 1.0e-12 + 1.0e-16 / psi_norm;
            // integral(d_l / (R |grad(psi)|)) = integral(d(theta) / (2 a (mag_r + rho cos(theta)))) = pi / (a sqrt(mag_r ** 2 - rho ** 2))
            let integral_over_r: f64 = surface.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);
            assert_abs_diff_eq!(
                integral_over_r,
                PI / (a * (mag_r.powi(2) - rho.powi(2)).sqrt()),
                epsilon = tolerance * integral_over_r
            );
            // integral(R d_l / |grad(psi)|) = pi mag_r / a, the derivative of the volume with respect to psi over 2 pi
            let integral_times_r: f64 = surface.contour_integral(|r_here: f64, _grad_psi_here: f64| r_here);
            assert_abs_diff_eq!(integral_times_r, PI * mag_r / a, epsilon = tolerance * integral_times_r);
            assert_abs_diff_eq!(surface.area, PI * rho.powi(2), epsilon = tolerance * surface.area);
            assert_abs_diff_eq!(surface.volume, 2.0 * PI * mag_r * PI * rho.powi(2), epsilon = tolerance * surface.volume);
            // The circle's average of R, integral(R ** 2 d(theta)) / integral(R d(theta)), is mag_r + rho ** 2 / (2 mag_r)
            let average_r: f64 = surface.flux_surface_average(|r_here: f64, _grad_psi_here: f64| r_here);
            assert_abs_diff_eq!(average_r, mag_r + rho.powi(2) / (2.0 * mag_r), epsilon = 1e-12);
        }
    }

    #[test]
    fn the_axis_is_the_limit_of_the_surfaces_next_to_it() {
        // A tilted ellipse
        let time_slice: EquilibriumTimeSlice = time_slice_with_elliptical_psi(0.5, 0.05, 30.0, 8.0, 12.0);
        let psi_interpolator: PsiInterpolator = PsiInterpolator::new(&time_slice);
        let axis: FluxSurface = axis_flux_surface(&time_slice, &psi_interpolator);
        let determinant: f64 = (2.0 * 30.0) * (2.0 * 12.0) - 8.0_f64.powi(2);
        assert_abs_diff_eq!(axis.quadrature_weight[0], 2.0 * PI / determinant.sqrt(), epsilon = 1e-14);

        let surface: FluxSurface = ray_traced_flux_surface(&time_slice, &psi_interpolator, 1.0e-10, &Array1::zeros(0))
            .expect("the surface is on the grid")
            .flux_surface;
        let integral_over_r_axis: f64 = axis.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);
        let integral_over_r: f64 = surface.contour_integral(|r_here: f64, _grad_psi_here: f64| 1.0 / r_here);
        assert_abs_diff_eq!(integral_over_r / integral_over_r_axis, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn the_surfaces_follow_the_profile_and_the_last_is_the_boundary_outline() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_with_circular_psi();
        time_slice.profiles_1d.psi_norm = array![0.0, 0.25, 0.5, 1.0];
        let n_outline: usize = 401;
        let outline_theta: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n_outline);
        time_slice.boundary.outline.r = outline_theta.mapv(|theta: f64| 0.5 + 0.2 * theta.cos());
        time_slice.boundary.outline.z = outline_theta.mapv(|theta: f64| 0.2 * theta.sin());

        let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(&time_slice);

        assert_eq!(flux_surfaces.len(), 4);
        assert_eq!(flux_surfaces[0].quadrature_weight.len(), 1);
        assert!(flux_surfaces[1].quadrature_weight.len() >= N_RAY_START);
        assert_eq!(flux_surfaces[3].r.len(), n_outline);
        // The outline is a polygon inscribed in the circle, so its area is slightly short of the circle's
        let area_circle: f64 = PI * 0.2_f64.powi(2);
        assert!(flux_surfaces[3].area < area_circle);
        assert_abs_diff_eq!(flux_surfaces[3].area / area_circle, 1.0, epsilon = 1e-4);
        // and its contour integrals are close to the circle's
        let a: f64 = 25.0;
        let integral_times_r: f64 = flux_surfaces[3].contour_integral(|r_here: f64, _grad_psi_here: f64| r_here);
        assert_abs_diff_eq!(integral_times_r / (PI * 0.5 / a), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn a_surface_off_the_grid_is_empty() {
        let time_slice: EquilibriumTimeSlice = time_slice_with_circular_psi();
        // psi = 1 - 25 rho ** 2 reaches psi_N = 1e4 far off the grid
        let surface: FluxSurface = calculate_at_psi_norm(&time_slice, 0.999_999);
        assert!(!surface.quadrature_weight.is_empty());
        let mut time_slice_far: EquilibriumTimeSlice = time_slice.clone();
        time_slice_far.boundary.psi = 1.0 - 25.0 * 2.0_f64.powi(2);
        let surface_far: FluxSurface = calculate_at_psi_norm(&time_slice_far, 0.9);
        assert_eq!(surface_far.quadrature_weight.len(), 0);
        assert!(surface_far.contour_integral(|r_here: f64, _grad_psi_here: f64| r_here).is_nan());
    }

    #[test]
    fn the_extremes_between_the_rays_are_refined() {
        let time_slice: EquilibriumTimeSlice = time_slice_with_circular_psi();
        let surface: FluxSurface = calculate_at_psi_norm(&time_slice, 0.25);
        // R on the circle of radius 0.1 about 0.5 runs from 0.4 to 0.6
        let range: ValueRange = surface.value_range(|r_here: f64, _grad_psi_here: f64| r_here);
        assert_abs_diff_eq!(range.maximum, 0.6, epsilon = 1e-9);
        assert_abs_diff_eq!(range.minimum, 0.4, epsilon = 1e-9);
    }
}
