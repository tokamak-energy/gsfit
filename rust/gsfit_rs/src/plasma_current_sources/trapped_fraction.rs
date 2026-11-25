//! The effective trapped particle fraction on each flux surface of an equilibrium time-slice.
//!
//! GSFit's post-processor does not fill `equilibrium/time_slice(itime)/profiles_1d/trapped_fraction`, so it is calculated here, from
//! the field sampled on the post-processor's own flux surfaces.

use super::flux_surface_samples::FluxSurfaceSamples;
use ndarray::Array1;

/// The effective trapped particle fraction, `f_t`, on each surface, from Lin-Liu and Miller, Physics of Plasmas 2, 1666 (1995), which is
/// the trapped fraction Sauter's and Redl's formulae are written in.
///
/// With `h = |B| / B_max` on a surface, and `<...>` its flux-surface average, Lin-Liu and Miller bound the trapped fraction,
/// `1 - (3 / 4) <h ** 2> integral_0^1 lambda d_lambda / <sqrt(1 - lambda h)>`, above and below by
///
/// ```text
/// f_t_upper = 1 - <h ** 2> / <h> ** 2 * (1 - sqrt(1 - <h>) * (1 + <h> / 2))
/// f_t_lower = 1 - <h ** 2> * <(1 - sqrt(1 - h) * (1 + h / 2)) / h ** 2>
/// ```
///
/// and take `f_t = 0.75 f_t_upper + 0.25 f_t_lower`, which on ST40 agrees with the integral to about 0.001. `B_max` is the largest
/// sample, so `h` is at most 1 at every one, which the square roots need.
///
/// # Arguments
/// * `flux_surface_samples` - the field sampled on each surface, `flux_surface_samples`, with the magnetic axis first
///
/// # Returns
/// `f_t` on each surface [dimensionless]: zero on the magnetic axis, where `|B|` is the same everywhere, and NaN on a surface which
/// could not be sampled
pub(super) fn trapped_fraction(flux_surface_samples: &[Option<FluxSurfaceSamples>]) -> Array1<f64> {
    let n_psi_norm: usize = flux_surface_samples.len();
    let mut trapped_fraction: Array1<f64> = Array1::from_elem(n_psi_norm, f64::NAN);
    for i_psi_norm in 0..n_psi_norm {
        if i_psi_norm == 0 {
            trapped_fraction[i_psi_norm] = 0.0;
        } else if let Some(samples) = &flux_surface_samples[i_psi_norm] {
            trapped_fraction[i_psi_norm] = lin_liu_miller(samples);
        }
    }

    trapped_fraction
}

/// Lin-Liu and Miller's effective trapped fraction on one surface; see `trapped_fraction`.
fn lin_liu_miller(samples: &FluxSurfaceSamples) -> f64 {
    let b_field_max: f64 = samples.b_field_max;
    let h_average: f64 = samples.average(|_r, b_field: f64, _b_field_p| b_field / b_field_max);
    let h_squared_average: f64 = samples.average(|_r, b_field: f64, _b_field_p| (b_field / b_field_max).powi(2));
    let lower_bound_average: f64 = samples.average(|_r, b_field: f64, _b_field_p| {
        let h: f64 = b_field / b_field_max;
        (1.0 - (1.0 - h).sqrt() * (1.0 + 0.5 * h)) / h.powi(2)
    });

    let trapped_fraction_upper: f64 = 1.0 - h_squared_average / h_average.powi(2) * (1.0 - (1.0 - h_average).sqrt() * (1.0 + 0.5 * h_average));
    let trapped_fraction_lower: f64 = 1.0 - h_squared_average * lower_bound_average;

    0.75 * trapped_fraction_upper + 0.25 * trapped_fraction_lower
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// `|B| = 1 / (1 + epsilon cos(theta))` on a large-aspect-ratio circle, at equally spaced `theta`, all equally weighted
    fn circular_samples(epsilon: f64) -> FluxSurfaceSamples {
        let n_theta: usize = 4000;
        let mut samples: FluxSurfaceSamples = FluxSurfaceSamples {
            r: Vec::with_capacity(n_theta),
            b_field: Vec::with_capacity(n_theta),
            b_field_p: Vec::with_capacity(n_theta),
            weight: Vec::with_capacity(n_theta),
            b_field_max: 1.0 / (1.0 - epsilon),
        };
        for i_theta in 0..n_theta {
            let theta: f64 = (i_theta as f64 + 0.5) * 2.0 * PI / n_theta as f64;
            samples.r.push(1.0 + epsilon * theta.cos());
            samples.b_field.push(1.0 / (1.0 + epsilon * theta.cos()));
            samples.b_field_p.push(0.1);
            samples.weight.push(1.0);
        }
        samples
    }

    #[test]
    fn a_uniform_field_traps_nothing() {
        let samples: FluxSurfaceSamples = FluxSurfaceSamples {
            r: vec![1.0; 100],
            b_field: vec![2.5; 100],
            b_field_p: vec![0.1; 100],
            weight: vec![1.0; 100],
            b_field_max: 2.5,
        };

        assert_eq!(lin_liu_miller(&samples), 0.0);
        assert_eq!(trapped_fraction(&[None, Some(samples), None]).to_vec()[..2], [0.0, 0.0]);
    }

    #[test]
    fn lin_liu_miller_agrees_with_the_exact_trapped_fraction_and_its_large_aspect_ratio_limit() {
        // The exact trapped fraction, `1 - (3 / 4) <h ** 2> integral_0^1 lambda d_lambda / <sqrt(1 - lambda h)>`, of the same field,
        // evaluated with adaptive quadrature (scipy's `quad`) on the same samples
        let cases: [(f64, f64); 3] = [(0.01, 0.145563), (0.1, 0.438772), (0.3, 0.685418)];
        for (epsilon, trapped_fraction_exact) in cases {
            assert_abs_diff_eq!(lin_liu_miller(&circular_samples(epsilon)), trapped_fraction_exact, epsilon = 1.0e-3);
        }

        // At small inverse aspect ratio `f_t = 1.46 sqrt(epsilon)`
        let epsilon: f64 = 1.0e-4;
        assert_abs_diff_eq!(lin_liu_miller(&circular_samples(epsilon)) / (1.46 * epsilon.sqrt()), 1.0, epsilon = 5.0e-3);
    }
}
