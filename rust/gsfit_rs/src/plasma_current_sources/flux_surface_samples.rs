//! Samples of the magnetic field on each flux surface of an equilibrium time-slice, from which any flux-surface average can be taken.
//!
//! The neoclassical formulae need averages the Equilibrium IDS does not hold, e.g. those in the trapped fraction, of functions of
//! `|B| / B_max`, and the Pfirsch-Schlüter current's `<1 / (R ** 2 B ** 2)>`. So the field is sampled once on each surface, and every
//! average is taken from the same samples, which keeps the near-cancellations in those currents free of discretisation error.

use crate::equilibrium_post_processor::flux_surfaces::{FluxSurface, calculate_flux_surfaces};
use imas_rs::EquilibriumTimeSlice;
use ndarray::Array1;
use std::f64::consts::PI;

/// Samples of the magnetic field on one flux surface, at the points of its quadrature, each with its weight `d_ell / b_p` in the
/// flux-surface average, `<x> = sum(x d_ell / b_p) / sum(d_ell / b_p)`.
pub(super) struct FluxSurfaceSamples {
    /// The major radius of each sample [metre]
    pub r: Vec<f64>,
    /// `|B|` at each sample [tesla]
    pub b_field: Vec<f64>,
    /// The poloidal field at each sample [tesla]
    pub b_field_p: Vec<f64>,
    /// The weight of each sample in the flux-surface average, `d_ell / b_p` [metre / tesla]
    pub weight: Vec<f64>,
    /// The largest `|B|` on the surface, refined between the samples, so no smaller than any of them [tesla]
    pub b_field_max: f64,
}

impl FluxSurfaceSamples {
    /// The flux-surface average of a function of the major radius, `|B|` and the poloidal field
    ///
    /// # Arguments
    /// * `values` - the function, of `(r, b_field, b_field_p)`, in [metre], [tesla] and [tesla]
    ///
    /// # Returns
    /// `<values(r, b_field, b_field_p)>`
    pub fn average<F: Fn(f64, f64, f64) -> f64>(&self, values: F) -> f64 {
        let n_sample: usize = self.r.len();
        let mut numerator: f64 = 0.0;
        let mut denominator: f64 = 0.0;
        for i_sample in 0..n_sample {
            numerator += self.weight[i_sample] * values(self.r[i_sample], self.b_field[i_sample], self.b_field_p[i_sample]);
            denominator += self.weight[i_sample];
        }

        numerator / denominator
    }
}

/// Sample the magnetic field on each flux surface of `profiles_1d`.
///
/// The surfaces are the post-processor's, `calculate_flux_surfaces`, sampled at the points of their quadrature, as
/// `b_field_average__b_field_max__b_field_min` samples them: `b_p = |grad(psi)| / (2 * pi * R)`, and `b_phi = f / R` exactly, so
/// `|B| = sqrt(b_p ** 2 + (f / R) ** 2)`. The weight `d_ell / b_p` of a point is `2 * pi * R` times its weight `d_ell / |grad(psi)|`.
///
/// # Arguments
/// * `time_slice` - a post-processed time-slice. Read for `profiles_1d/psi_norm` and `/f`, the `profiles_2d(0)` grid, `psi` and its
///   derivatives, the magnetic axis and the boundary
///
/// # Returns
/// The samples on each surface of `profiles_1d/psi_norm`. `None` on the magnetic axis, `psi_norm = 0`, which is a point, and on a
/// surface which cannot be found.
///
/// # Errors
/// `profiles_1d/f` is not on `profiles_1d/psi_norm`
pub(super) fn flux_surface_samples(time_slice: &EquilibriumTimeSlice) -> Result<Vec<Option<FluxSurfaceSamples>>, String> {
    let n_psi_norm: usize = time_slice.profiles_1d.psi_norm.len();
    let mut samples: Vec<Option<FluxSurfaceSamples>> = Vec::with_capacity(n_psi_norm);
    for _i_psi_norm in 0..n_psi_norm {
        samples.push(None);
    }
    if n_psi_norm < 2 {
        return Ok(samples);
    }

    let f_profile: &Array1<f64> = &time_slice.profiles_1d.f;
    if f_profile.len() != n_psi_norm {
        return Err(format!(
            "`profiles_1d/f` has {} values, but `profiles_1d/psi_norm` has {n_psi_norm}",
            f_profile.len()
        ));
    }

    let flux_surfaces: Vec<FluxSurface> = calculate_flux_surfaces(time_slice);

    'flux_surface_loop: for i_psi_norm in 1..n_psi_norm {
        let flux_surface: &FluxSurface = &flux_surfaces[i_psi_norm];
        let n_quadrature: usize = flux_surface.quadrature_weight.len();
        let f_here: f64 = f_profile[i_psi_norm];
        if n_quadrature == 0 || !f_here.is_finite() {
            continue 'flux_surface_loop;
        }

        let b_field = |r_here: f64, grad_psi_here: f64| -> f64 { (grad_psi_here / (2.0 * PI * r_here)).hypot(f_here / r_here) };
        let mut surface_samples: FluxSurfaceSamples = FluxSurfaceSamples {
            r: Vec::with_capacity(n_quadrature),
            b_field: Vec::with_capacity(n_quadrature),
            b_field_p: Vec::with_capacity(n_quadrature),
            weight: Vec::with_capacity(n_quadrature),
            b_field_max: flux_surface.value_range(b_field).maximum,
        };
        for i_quadrature in 0..n_quadrature {
            let r_here: f64 = flux_surface.quadrature_r[i_quadrature];
            let grad_psi_here: f64 = flux_surface.quadrature_grad_psi[i_quadrature];
            surface_samples.r.push(r_here);
            surface_samples.b_field.push(b_field(r_here, grad_psi_here));
            surface_samples.b_field_p.push(grad_psi_here / (2.0 * PI * r_here));
            surface_samples.weight.push(2.0 * PI * r_here * flux_surface.quadrature_weight[i_quadrature]);
        }
        samples[i_psi_norm] = Some(surface_samples);
    }

    Ok(samples)
}
