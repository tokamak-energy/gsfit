//! The initial current-density guess the Grad-Shafranov solve starts from.
//!
//! Split out of `equilibrium_solve.rs` because this is not per-time-slice work: every input is the
//! same for every slice, so `grad_shafranov_solver` builds the seed once and hands the solver a
//! borrow of it. Profiling showed the old arrangement - one call per time-slice - spending 7% of
//! the whole run inside `geo`, testing a 4097-point ellipse against the vessel polygon 480 times.

use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Number of points the initial ellipse is sampled at when testing that it lies inside the vessel.
const INITIAL_ELLIPSE_BOUNDARY_POINTS: usize = 4096;

/// Create a smooth, axis-aligned quadratic current-density seed.
///
/// The ellipse is centred on the supplied initial magnetic-axis guess. Its
/// radial semi-axis is `initial_guess_minor_radius`, and its vertical semi-axis is
/// `initial_guess_minor_radius * initial_guess_elongation`. These plasma parameters are
/// independent of the vacuum-vessel shape. The complete sampled ellipse must
/// lie inside the vessel and computational grid, and no limiter point may lie
/// inside its support. The discrete current is normalised to `initial_guess_ip`
/// within floating-point precision.
pub fn quadratic_current_density_seed(
    r: &Array1<f64>,
    z: &Array1<f64>,
    limiter_r: &Array1<f64>,
    limiter_z: &Array1<f64>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    d_area: f64,
    initial_guess_ip: f64,
    initial_guess_cur_r: f64,
    initial_guess_cur_z: f64,
    initial_guess_minor_radius: f64,
    initial_guess_elongation: f64,
) -> Result<Array2<f64>, String> {
    if r.len() < 2 || z.len() < 2 {
        return Err("quadratic current initialisation requires at least two radial and vertical grid points".to_string());
    }
    if limiter_r.len() != limiter_z.len() || limiter_r.is_empty() {
        return Err("quadratic current initialisation requires matching, nonempty limiter R/Z arrays".to_string());
    }
    if vessel_r.len() != vessel_z.len() || vessel_r.len() < 3 {
        return Err("quadratic current initialisation requires matching vessel R/Z arrays with at least three points".to_string());
    }
    if r.iter()
        .chain(z.iter())
        .chain(limiter_r.iter())
        .chain(limiter_z.iter())
        .chain(vessel_r.iter())
        .chain(vessel_z.iter())
        .any(|value| !value.is_finite())
        || !d_area.is_finite()
        || d_area <= 0.0
        || !initial_guess_ip.is_finite()
        || initial_guess_ip == 0.0
        || !initial_guess_cur_r.is_finite()
        || !initial_guess_cur_z.is_finite()
        || !initial_guess_minor_radius.is_finite()
        || initial_guess_minor_radius <= 0.0
        || !initial_guess_elongation.is_finite()
        || initial_guess_elongation <= 0.0
    {
        return Err(
            "quadratic current initialisation requires finite geometry, positive cell area, finite nonzero initial current, positive minor radius, and positive kappa"
                .to_string(),
        );
    }

    let a_r = initial_guess_minor_radius;
    let b_z = initial_guess_minor_radius * initial_guess_elongation;
    if !b_z.is_finite() {
        return Err("quadratic current initialisation requires a finite vertical semi-axis".to_string());
    }

    let min_max = |values: &Array1<f64>| -> (f64, f64) {
        values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(minimum, maximum), &value| {
            (minimum.min(value), maximum.max(value))
        })
    };
    let (grid_r_min, grid_r_max) = min_max(r);
    let (grid_z_min, grid_z_max) = min_max(z);
    if initial_guess_cur_r - a_r < grid_r_min
        || initial_guess_cur_r + a_r > grid_r_max
        || initial_guess_cur_z - b_z < grid_z_min
        || initial_guess_cur_z + b_z > grid_z_max
    {
        return Err("initial current ellipse must lie inside the plasma grid".to_string());
    }

    // Some readers append discrete tile points to a closed limiter outline, so
    // the limiter is deliberately treated as a point set rather than a polygon.
    for (&r_value, &z_value) in limiter_r.iter().zip(limiter_z.iter()) {
        let s = ((r_value - initial_guess_cur_r) / a_r).powi(2) + ((z_value - initial_guess_cur_z) / b_z).powi(2);
        if s < 1.0 {
            return Err("initial current ellipse contains a limiter point".to_string());
        }
    }

    let vessel_coordinates: Vec<Coord<f64>> = vessel_r
        .iter()
        .zip(vessel_z.iter())
        .map(|(&r_value, &z_value)| Coord { x: r_value, y: z_value })
        .collect();
    let vessel_polygon = Polygon::new(LineString::new(vessel_coordinates), vec![]);
    let ellipse_coordinates: Vec<Coord<f64>> = (0..=INITIAL_ELLIPSE_BOUNDARY_POINTS)
        .map(|i_point| {
            let theta = 2.0 * PI * i_point as f64 / INITIAL_ELLIPSE_BOUNDARY_POINTS as f64;
            Coord {
                x: initial_guess_cur_r + a_r * theta.cos(),
                y: initial_guess_cur_z + b_z * theta.sin(),
            }
        })
        .collect();
    let ellipse_polygon = Polygon::new(LineString::new(ellipse_coordinates), vec![]);
    if !vessel_polygon.contains(&ellipse_polygon) {
        return Err("initial current ellipse must lie strictly inside the vessel".to_string());
    }

    // This edge is the support of the initial J_phi guess, not an LCFS; the
    // first-iteration plasma boundary is found separately from the total
    // poloidal flux, including PF-coil flux.
    let mut shape = Array2::zeros((z.len(), r.len()));
    for (i_z, &z_value) in z.iter().enumerate() {
        for (i_r, &r_value) in r.iter().enumerate() {
            let s = ((r_value - initial_guess_cur_r) / a_r).powi(2) + ((z_value - initial_guess_cur_z) / b_z).powi(2);
            let shape_value = (1.0 - s).max(0.0);
            if shape_value > 0.0 && !vessel_polygon.contains(&Point::new(r_value, z_value)) {
                return Err("quadratic current support contains a grid point outside the vessel".to_string());
            }
            shape[(i_z, i_r)] = shape_value;
        }
    }

    let shape_integral = shape.sum() * d_area;
    if !shape_integral.is_finite() || shape_integral <= 0.0 {
        return Err("quadratic current initialisation has empty support on the plasma grid".to_string());
    }
    let normalisation = initial_guess_ip / shape_integral;
    if !normalisation.is_finite() {
        return Err("quadratic current initialisation requires a finite current-density normalisation".to_string());
    }
    let j_2d = shape * normalisation;
    let achieved_current = j_2d.sum() * d_area;
    let relative_error = (achieved_current - initial_guess_ip).abs() / initial_guess_ip.abs();
    if j_2d.iter().any(|value| !value.is_finite()) || !relative_error.is_finite() || relative_error > 1.0e-10 {
        return Err("quadratic current initialisation could not produce a finite, normalised current density".to_string());
    }
    return Ok(j_2d);
}

#[cfg(test)]
mod tests {
    use super::quadratic_current_density_seed;
    use ndarray::{Array1, array};

    #[test]
    fn quadratic_current_seed_is_normalised_and_uses_explicit_shape() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];
        let d_area = (r[1] - r[0]) * (z[1] - z[0]);
        let initial_guess_ip = 120_000.0;

        let j_2d = quadratic_current_density_seed(
            &r,
            &z,
            &limiter_r,
            &limiter_z,
            &vessel_r,
            &vessel_z,
            d_area,
            initial_guess_ip,
            2.0,
            0.0,
            0.5,
            2.0,
        )
        .unwrap();

        assert!((j_2d.sum() * d_area - initial_guess_ip).abs() < 1.0e-10 * initial_guess_ip);
        assert!((j_2d[(6, 5)] / j_2d[(6, 4)] - 0.75).abs() < 1.0e-12);
        assert!((j_2d[(7, 4)] / j_2d[(6, 4)] - 0.9375).abs() < 1.0e-12);
        assert_eq!(j_2d[(6, 6)], 0.0);
        assert_eq!(j_2d[(10, 4)], 0.0);
    }

    #[test]
    fn quadratic_current_seed_is_independent_of_vessel_shape() {
        let r = Array1::linspace(1.0, 3.0, 17);
        let z = Array1::linspace(-1.5, 1.5, 25);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r_1 = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z_1 = array![-1.75, -1.75, 1.75, 1.75, -1.75];
        let vessel_r_2 = array![0.7, 3.3, 3.3, 2.6, 0.7, 0.7];
        let vessel_z_2 = array![-1.8, -1.8, 1.8, 1.65, 1.8, -1.8];
        let d_area = (r[1] - r[0]) * (z[1] - z[0]);

        let seed_1 = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r_1, &vessel_z_1, d_area, 100_000.0, 2.0, 0.0, 0.5, 2.0).unwrap();
        let seed_2 = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r_2, &vessel_z_2, d_area, 100_000.0, 2.0, 0.0, 0.5, 2.0).unwrap();

        assert_eq!(seed_1, seed_2);
    }

    #[test]
    fn quadratic_current_seed_preserves_negative_current_sign() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];
        let d_area = (r[1] - r[0]) * (z[1] - z[0]);

        let j_2d = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, d_area, -80_000.0, 2.0, 0.0, 0.5, 2.0).unwrap();

        assert!((j_2d.sum() * d_area + 80_000.0).abs() < 1.0e-8);
        assert!(j_2d.iter().all(|value| *value <= 0.0));
    }

    #[test]
    fn quadratic_current_seed_rejects_limiter_point_inside_support() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8, 2.25];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7, 0.0];
        let vessel_r = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];

        let error = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, 0.0625, 10_000.0, 2.0, 0.0, 0.5, 2.0).unwrap_err();

        assert!(error.contains("contains a limiter point"));
    }

    #[test]
    fn quadratic_current_seed_rejects_ellipse_outside_vessel() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r = array![1.75, 3.25, 3.25, 1.75, 1.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];

        let error = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, 0.0625, 10_000.0, 2.0, 0.0, 0.5, 2.0).unwrap_err();

        assert!(error.contains("strictly inside the vessel"));
    }

    #[test]
    fn quadratic_current_seed_rejects_nonfinite_normalisation() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];

        let error = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, 0.0625, f64::MAX, 2.0, 0.0, 0.5, 2.0).unwrap_err();

        assert!(error.contains("finite current-density normalisation"));
    }

    #[test]
    fn quadratic_current_seed_rejects_nonpositive_shape_parameters() {
        let r = Array1::linspace(1.0, 3.0, 9);
        let z = Array1::linspace(-1.5, 1.5, 13);
        let limiter_r = array![0.8, 3.2, 3.2, 0.8, 0.8];
        let limiter_z = array![-1.7, -1.7, 1.7, 1.7, -1.7];
        let vessel_r = array![0.75, 3.25, 3.25, 0.75, 0.75];
        let vessel_z = array![-1.75, -1.75, 1.75, 1.75, -1.75];

        let minor_radius_error =
            quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, 0.0625, 10_000.0, 2.0, 0.0, 0.0, 2.0).unwrap_err();
        let kappa_error =
            quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, 0.0625, 10_000.0, 2.0, 0.0, 0.5, 0.0).unwrap_err();

        assert!(minor_radius_error.contains("positive minor radius"));
        assert!(kappa_error.contains("positive kappa"));
    }
}
