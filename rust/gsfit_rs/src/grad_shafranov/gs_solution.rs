use super::Error;
use crate::Plasma;
use crate::plasma_geometry;
use crate::plasma_geometry::BoundaryContour;
use crate::plasma_geometry::MagneticAxis;
use crate::plasma_geometry::StationaryPoint;
use crate::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
use crate::plasma_geometry::find_boundary;
use crate::plasma_geometry::find_magnetic_axis;
use crate::plasma_geometry::find_stationary_points_using_winding_number;
use crate::sensors::{SensorsDynamic, SensorsStatic};
use crate::source_functions::SourceFunctionTraits;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{SolveLstsq, Svd as FaerSvd};
use faer::mat::MatRef;
use faer::{Accum, Par};
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::Axis;
use ndarray::{Array1, Array2, Array3, ArrayView2, concatenate, s};
use ndarray_stats::QuantileExt;
use std::f64::consts::PI;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;
const INITIAL_ELLIPSE_BOUNDARY_POINTS: usize = 4096;

/// Create a smooth, axis-aligned quadratic current-density seed.
///
/// The ellipse is centred on the supplied initial magnetic-axis guess. Its
/// radial semi-axis is `initial_minor_radius`, and its vertical semi-axis is
/// `initial_minor_radius * initial_kappa`. These plasma parameters are
/// independent of the vacuum-vessel shape. The complete sampled ellipse must
/// lie inside the vessel and computational grid, and no limiter point may lie
/// inside its support. The discrete current is normalised to `initial_ip`
/// within floating-point precision.
fn quadratic_current_density_seed(
    r: &Array1<f64>,
    z: &Array1<f64>,
    limiter_r: &Array1<f64>,
    limiter_z: &Array1<f64>,
    vessel_r: &Array1<f64>,
    vessel_z: &Array1<f64>,
    d_area: f64,
    initial_ip: f64,
    initial_cur_r: f64,
    initial_cur_z: f64,
    initial_minor_radius: f64,
    initial_kappa: f64,
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
        || !initial_ip.is_finite()
        || initial_ip == 0.0
        || !initial_cur_r.is_finite()
        || !initial_cur_z.is_finite()
        || !initial_minor_radius.is_finite()
        || initial_minor_radius <= 0.0
        || !initial_kappa.is_finite()
        || initial_kappa <= 0.0
    {
        return Err(
            "quadratic current initialisation requires finite geometry, positive cell area, finite nonzero initial current, positive minor radius, and positive kappa"
                .to_string(),
        );
    }

    let a_r = initial_minor_radius;
    let b_z = initial_minor_radius * initial_kappa;
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
    if initial_cur_r - a_r < grid_r_min || initial_cur_r + a_r > grid_r_max || initial_cur_z - b_z < grid_z_min || initial_cur_z + b_z > grid_z_max {
        return Err("initial current ellipse must lie inside the plasma grid".to_string());
    }

    // Some readers append discrete tile points to a closed limiter outline, so
    // the limiter is deliberately treated as a point set rather than a polygon.
    for (&r_value, &z_value) in limiter_r.iter().zip(limiter_z.iter()) {
        let s = ((r_value - initial_cur_r) / a_r).powi(2) + ((z_value - initial_cur_z) / b_z).powi(2);
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
                x: initial_cur_r + a_r * theta.cos(),
                y: initial_cur_z + b_z * theta.sin(),
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
            let s = ((r_value - initial_cur_r) / a_r).powi(2) + ((z_value - initial_cur_z) / b_z).powi(2);
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
    let normalisation = initial_ip / shape_integral;
    if !normalisation.is_finite() {
        return Err("quadratic current initialisation requires a finite current-density normalisation".to_string());
    }
    let j_2d = shape * normalisation;
    let achieved_current = j_2d.sum() * d_area;
    let relative_error = (achieved_current - initial_ip).abs() / initial_ip.abs();
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
        let initial_ip = 120_000.0;

        let j_2d = quadratic_current_density_seed(&r, &z, &limiter_r, &limiter_z, &vessel_r, &vessel_z, d_area, initial_ip, 2.0, 0.0, 0.5, 2.0).unwrap();

        assert!((j_2d.sum() * d_area - initial_ip).abs() < 1.0e-10 * initial_ip);
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

/// Greens tables reorganised for `calculate_psi_and_derivatives`.
///
/// Precomputed **once per time-slice** (the tables do not change between Picard iterations),
/// while `calculate_psi_and_derivatives` is called **every** iteration.
///
/// The expensive part of `calculate_psi_and_derivatives` is the plasma grid-to-grid convolution:
/// ```text
///     field[(i_z, i_r)] = sum_{i_cur_z, i_cur_r} g[(|i_z - i_cur_z|, i_r, i_cur_r)] * j_2d[(i_cur_z, i_cur_r)] * d_area
/// ```
/// Because the Greens table only depends on the **vertical offset** `i_offset_z = |i_z - i_cur_z|`,
/// the convolution can be reorganised into a single matrix multiplication:
/// ```text
///     field = w @ g_plasma_by_offset
/// ```
/// where row `i_z` of `w` gathers, for each `(i_offset_z, i_cur_r)`, the (at most two) current
/// sources which see grid point `i_z` at that offset: `j_2d[(i_z - i_offset_z, i_cur_r)]` and
/// `j_2d[(i_z + i_offset_z, i_cur_r)]`. Kernels which are **even** in `z - z_current_source`
/// (`psi`, `d_psi_d_r`, `d2_psi_d_r2`, `d2_psi_d_z2`, `d3_psi_d_r_d_z2`) take the sum of the two
/// sources; kernels which are **odd** (`d_psi_d_z`, `d2_psi_d_r_d_z`, `d3_psi_d_r2_d_z`,
/// `d3_psi_d_z3`) take the difference. `w` is built fresh each iteration (it depends on `j_2d`), which is cheap;
/// the GEMM is done with `faer`.
///
/// The even kernels and the odd kernels are each concatenated column-wise, so the plasma
/// contribution to all nine fields costs exactly two GEMMs.
pub struct PsiAndDerivativesGreens {
    /// Plasma grid-to-grid, even kernels, concatenated column-wise in the order
    /// [`psi`, `d_psi_d_r`, `d2_psi_d_r2`, `d2_psi_d_z2`, `d3_psi_d_r_d_z2`];
    /// rows = (i_offset_z * n_r + i_cur_r); shape = (n_z * n_r, 5 * n_r)
    g_even_plasma_by_offset: Array2<f64>,
    /// Plasma grid-to-grid, odd kernels, concatenated column-wise in the order
    /// [`d_psi_d_z`, `d2_psi_d_r_d_z`, `d3_psi_d_r2_d_z`, `d3_psi_d_z3`];
    /// rows = (i_offset_z * n_r + i_cur_r); shape = (n_z * n_r, 4 * n_r)
    g_odd_plasma_by_offset: Array2<f64>,
    /// PF coils; each shape = (n_z * n_r, n_pf)
    g_d_psi_d_r_coils_matrix: Array2<f64>,
    g_d_psi_d_z_coils_matrix: Array2<f64>,
    g_d2_psi_d_r2_coils_matrix: Array2<f64>,
    g_d2_psi_d_r_d_z_coils_matrix: Array2<f64>,
    g_d2_psi_d_z2_coils_matrix: Array2<f64>,
    g_d3_psi_d_r2_d_z_coils_matrix: Array2<f64>,
    g_d3_psi_d_r_d_z2_coils_matrix: Array2<f64>,
    g_d3_psi_d_z3_coils_matrix: Array2<f64>,
    /// Passives; each shape = (n_z * n_r, n_passive_dof)
    g_psi_passives_matrix: Array2<f64>,
    g_d_psi_d_r_passives_matrix: Array2<f64>,
    g_d_psi_d_z_passives_matrix: Array2<f64>,
    g_d2_psi_d_r2_passives_matrix: Array2<f64>,
    g_d2_psi_d_r_d_z_passives_matrix: Array2<f64>,
    g_d2_psi_d_z2_passives_matrix: Array2<f64>,
    g_d3_psi_d_r2_d_z_passives_matrix: Array2<f64>,
    g_d3_psi_d_r_d_z2_passives_matrix: Array2<f64>,
    g_d3_psi_d_z3_passives_matrix: Array2<f64>,
}

impl PsiAndDerivativesGreens {
    pub fn new(plasma: &Plasma) -> Self {
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();

        // Plasma grid-to-grid tables; stored shape = (n_z * n_r, n_r), which unflattens to
        // (i_offset_z, i_r, i_cur_r). Permute to (i_offset_z, i_cur_r, i_r) and re-flatten so
        // that rows = (i_offset_z, i_cur_r) match the columns of `w`, and columns = i_r
        let permute_to_by_offset = |g_flat: Array2<f64>| -> Array2<f64> {
            let g_3d: Array3<f64> = g_flat
                .to_shape((n_z, n_r, n_r))
                .expect("PsiAndDerivativesGreens: failed to reshape grid_grid table into (n_z, n_r, n_r)")
                .to_owned();
            let g_3d_permuted: Array3<f64> = g_3d.permuted_axes([0, 2, 1]);
            let g_by_offset: Array2<f64> = g_3d_permuted
                .as_standard_layout()
                .to_shape((n_z * n_r, n_r))
                .expect("PsiAndDerivativesGreens: failed to flatten permuted table into (n_z * n_r, n_r)")
                .to_owned();
            return g_by_offset;
        };

        let grid_grid = |key: &str| -> Array2<f64> { permute_to_by_offset(plasma.results.get("greens").get("grid_grid").get(key).unwrap_array2()) };
        let g_psi_plasma_by_offset: Array2<f64> = grid_grid("psi");
        let g_d_psi_d_r_plasma_by_offset: Array2<f64> = grid_grid("d_psi_d_r");
        let g_d_psi_d_z_plasma_by_offset: Array2<f64> = grid_grid("d_psi_d_z");
        let g_d2_psi_d_r2_plasma_by_offset: Array2<f64> = grid_grid("d2_psi_d_r2");
        let g_d2_psi_d_r_d_z_plasma_by_offset: Array2<f64> = grid_grid("d2_psi_d_r_d_z");
        let g_d2_psi_d_z2_plasma_by_offset: Array2<f64> = grid_grid("d2_psi_d_z2");
        let g_d3_psi_d_r2_d_z_plasma_by_offset: Array2<f64> = grid_grid("d3_psi_d_r2_d_z");
        let g_d3_psi_d_r_d_z2_plasma_by_offset: Array2<f64> = grid_grid("d3_psi_d_r_d_z2");
        let g_d3_psi_d_z3_plasma_by_offset: Array2<f64> = grid_grid("d3_psi_d_z3");

        // Concatenate per parity, so each parity is a single GEMM
        // (`as_standard_layout` because `concatenate` does not guarantee a C-contiguous result)
        let g_even_plasma_by_offset: Array2<f64> = concatenate(
            Axis(1),
            &[
                g_psi_plasma_by_offset.view(),
                g_d_psi_d_r_plasma_by_offset.view(),
                g_d2_psi_d_r2_plasma_by_offset.view(),
                g_d2_psi_d_z2_plasma_by_offset.view(),
                g_d3_psi_d_r_d_z2_plasma_by_offset.view(),
            ],
        )
        .expect("PsiAndDerivativesGreens: failed to concatenate even kernels")
        .as_standard_layout()
        .to_owned();
        let g_odd_plasma_by_offset: Array2<f64> = concatenate(
            Axis(1),
            &[
                g_d_psi_d_z_plasma_by_offset.view(),
                g_d2_psi_d_r_d_z_plasma_by_offset.view(),
                g_d3_psi_d_r2_d_z_plasma_by_offset.view(),
                g_d3_psi_d_z3_plasma_by_offset.view(),
            ],
        )
        .expect("PsiAndDerivativesGreens: failed to concatenate odd kernels")
        .as_standard_layout()
        .to_owned();

        // PF coils: (n_z, n_r, n_pf) flattens to (n_z * n_r, n_pf)
        let coils_matrix = |key: &str| -> Array2<f64> {
            let g_coils: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get(key).unwrap_array3();
            let (_, _, n_pf): (usize, usize, usize) = g_coils.dim();
            return g_coils
                .to_shape((n_z * n_r, n_pf))
                .expect("PsiAndDerivativesGreens: failed to reshape PF coil table")
                .to_owned();
        };
        let g_d_psi_d_r_coils_matrix: Array2<f64> = coils_matrix("d_psi_d_r");
        let g_d_psi_d_z_coils_matrix: Array2<f64> = coils_matrix("d_psi_d_z");
        let g_d2_psi_d_r2_coils_matrix: Array2<f64> = coils_matrix("d2_psi_d_r2");
        let g_d2_psi_d_r_d_z_coils_matrix: Array2<f64> = coils_matrix("d2_psi_d_r_d_z");
        let g_d2_psi_d_z2_coils_matrix: Array2<f64> = coils_matrix("d2_psi_d_z2");
        let g_d3_psi_d_r2_d_z_coils_matrix: Array2<f64> = coils_matrix("d3_psi_d_r2_d_z");
        let g_d3_psi_d_r_d_z2_coils_matrix: Array2<f64> = coils_matrix("d3_psi_d_r_d_z2");
        let g_d3_psi_d_z3_coils_matrix: Array2<f64> = coils_matrix("d3_psi_d_z3");

        // Passives; already stored as (n_z * n_r, n_passive_dof)
        let g_psi_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid();
        let g_d_psi_d_r_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d_psi_d_r();
        let g_d_psi_d_z_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d_psi_d_z();
        let g_d2_psi_d_r2_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d2_psi_d_r2();
        let g_d2_psi_d_r_d_z_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d2_psi_d_r_d_z();
        let g_d2_psi_d_z2_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d2_psi_d_z2();
        let g_d3_psi_d_r2_d_z_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d3_psi_d_r2_d_z();
        let g_d3_psi_d_r_d_z2_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d3_psi_d_r_d_z2();
        let g_d3_psi_d_z3_passives_matrix: Array2<f64> = plasma.get_greens_passive_grid_d3_psi_d_z3();

        return Self {
            g_even_plasma_by_offset,
            g_odd_plasma_by_offset,
            g_d_psi_d_r_coils_matrix,
            g_d_psi_d_z_coils_matrix,
            g_d2_psi_d_r2_coils_matrix,
            g_d2_psi_d_r_d_z_coils_matrix,
            g_d2_psi_d_z2_coils_matrix,
            g_d3_psi_d_r2_d_z_coils_matrix,
            g_d3_psi_d_r_d_z2_coils_matrix,
            g_d3_psi_d_z3_coils_matrix,
            g_psi_passives_matrix,
            g_d_psi_d_r_passives_matrix,
            g_d_psi_d_z_passives_matrix,
            g_d2_psi_d_r2_passives_matrix,
            g_d2_psi_d_r_d_z_passives_matrix,
            g_d2_psi_d_z2_passives_matrix,
            g_d3_psi_d_r2_d_z_passives_matrix,
            g_d3_psi_d_r_d_z2_passives_matrix,
            g_d3_psi_d_z3_passives_matrix,
        };
    }
}

/// Grad-Shafranov solution, at single time-slice
pub struct GsSolution<'a> {
    // Object inputs
    plasma: &'a Plasma,
    coils_dynamic: &'a SensorsDynamic,
    bp_probes_static: &'a SensorsStatic,
    bp_probes_dynamic: &'a SensorsDynamic,
    flux_loops_static: &'a SensorsStatic,
    flux_loops_dynamic: &'a SensorsDynamic,
    dialoop_static: &'a SensorsStatic,
    dialoop_dynamic: &'a SensorsDynamic,
    rogowski_coils_static: &'a SensorsStatic,
    rogowski_coils_dynamic: &'a SensorsDynamic,
    isoflux_static: &'a SensorsStatic,
    isoflux_dynamic: &'a SensorsDynamic,
    isoflux_boundary_static: &'a SensorsStatic,
    isoflux_boundary_dynamic: &'a SensorsDynamic,
    pressure_sensors_static: &'a SensorsStatic,
    pressure_sensors_dynamic: &'a SensorsDynamic,
    magnetic_axis_static: &'a SensorsStatic,
    magnetic_axis_dynamic: &'a SensorsDynamic,
    n_iter_max: usize,
    n_iter_min: usize,
    n_iter_no_vertical_feedback: usize,
    gs_error_tolerence: f64,
    i_rod: f64,
    // Results
    pub gs_error_calculated: f64,
    pub ff_prime_dof_values: Array1<f64>,
    pub p_prime_dof_values: Array1<f64>,
    pub psi_2d_coils: Array2<f64>,
    pub passive_dof_values: Array1<f64>,
    pub psi_2d: Array2<f64>,
    pub d_psi_d_r_2d: Array2<f64>,
    pub d_psi_d_z_2d: Array2<f64>,
    pub d2_psi_d_r2_2d: Array2<f64>,
    pub d2_psi_d_r_d_z_2d: Array2<f64>,
    pub d2_psi_d_z2_2d: Array2<f64>,
    pub psi_n_2d: Array2<f64>,
    pub j_2d: Array2<f64>,
    pub mask: Array2<f64>,
    pub psi_b: f64,
    pub psi_a: f64,
    pub ip: f64,
    pub bounding_r: f64,
    pub bounding_z: f64,
    pub delta_z: f64,
    pub xpt_upper_r: f64,
    pub xpt_upper_z: f64,
    pub xpt_lower_r: f64,
    pub xpt_lower_z: f64,
    pub n_iter: usize,
    pub r_mag: f64,
    pub z_mag: f64,
    pub xpt_diverted: bool,
    pub stationary_points: Vec<StationaryPoint>,
    pub p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    passive_regularisations: Array2<f64>,
    passive_regularisations_weight: Array1<f64>,
    pub error_state: Option<Error>,
}

impl<'a> GsSolution<'a> {
    pub fn new(
        plasma: &'a Plasma,
        coils_dynamic: &'a SensorsDynamic,
        bp_probes_static: &'a SensorsStatic,
        bp_probes_dynamic: &'a SensorsDynamic,
        flux_loops_static: &'a SensorsStatic,
        flux_loops_dynamic: &'a SensorsDynamic,
        dialoop_static: &'a SensorsStatic,
        dialoop_dynamic: &'a SensorsDynamic,
        rogowski_coils_static: &'a SensorsStatic,
        rogowski_coils_dynamic: &'a SensorsDynamic,
        isoflux_static: &'a SensorsStatic,
        isoflux_dynamic: &'a SensorsDynamic,
        isoflux_boundary_static: &'a SensorsStatic,
        isoflux_boundary_dynamic: &'a SensorsDynamic,
        pressure_sensors_static: &'a SensorsStatic,
        pressure_sensors_dynamic: &'a SensorsDynamic,
        magnetic_axis_static: &'a SensorsStatic,
        magnetic_axis_dynamic: &'a SensorsDynamic,
        n_iter_max: usize,
        n_iter_min: usize,
        n_iter_no_vertical_feedback: usize,
        gs_error_tolerence: f64,
        i_rod: f64,
        p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
        ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
        passive_regularisations: Array2<f64>,
        passive_regularisations_weight: Array1<f64>,
    ) -> Self {
        GsSolution {
            // Object inputs
            plasma,
            coils_dynamic,
            bp_probes_static,
            bp_probes_dynamic,
            flux_loops_static,
            flux_loops_dynamic,
            dialoop_static,
            dialoop_dynamic,
            rogowski_coils_static,
            rogowski_coils_dynamic,
            isoflux_static,
            isoflux_dynamic,
            isoflux_boundary_static,
            isoflux_boundary_dynamic,
            pressure_sensors_static,
            pressure_sensors_dynamic,
            magnetic_axis_static,
            magnetic_axis_dynamic,
            n_iter_max,
            n_iter_min,
            n_iter_no_vertical_feedback,
            gs_error_tolerence,
            i_rod,
            // Results
            gs_error_calculated: f64::NAN,
            ff_prime_dof_values: Array1::zeros(0),
            p_prime_dof_values: Array1::zeros(0),
            passive_dof_values: Array1::zeros(0),
            psi_2d: Array2::zeros((0, 0)),
            d_psi_d_r_2d: Array2::zeros((0, 0)),
            d_psi_d_z_2d: Array2::zeros((0, 0)),
            d2_psi_d_r2_2d: Array2::zeros((0, 0)),
            d2_psi_d_r_d_z_2d: Array2::zeros((0, 0)),
            d2_psi_d_z2_2d: Array2::zeros((0, 0)),
            psi_n_2d: Array2::zeros((0, 0)),
            j_2d: Array2::zeros((0, 0)),
            mask: Array2::zeros((0, 0)),
            psi_2d_coils: Array2::zeros((0, 0)),
            psi_b: f64::NAN,
            psi_a: f64::NAN,
            ip: f64::NAN,
            bounding_r: f64::NAN,
            bounding_z: f64::NAN,
            delta_z: f64::NAN,
            xpt_upper_r: f64::NAN,
            xpt_upper_z: f64::NAN,
            xpt_lower_r: f64::NAN,
            xpt_lower_z: f64::NAN,
            n_iter: usize::MAX,
            r_mag: f64::NAN,
            z_mag: f64::NAN,
            xpt_diverted: false,
            stationary_points: Vec::new(),
            p_prime_source_function,
            ff_prime_source_function,
            passive_regularisations,
            passive_regularisations_weight,
            error_state: None,
        }
    }

    /// If the solver fails to converge, this function will set the solution to NAN values (but with the correct shape).
    fn set_to_failed_time_slice(&mut self) {
        self.gs_error_calculated = f64::NAN;
        self.ff_prime_dof_values = self.ff_prime_dof_values.to_owned() * f64::NAN;
        self.p_prime_dof_values = self.p_prime_dof_values.to_owned() * f64::NAN;
        self.passive_dof_values = self.passive_dof_values.to_owned() * f64::NAN;
        self.psi_2d = self.psi_2d.to_owned() * f64::NAN;
        self.d_psi_d_r_2d = self.d_psi_d_r_2d.to_owned() * f64::NAN;
        self.d_psi_d_z_2d = self.d_psi_d_z_2d.to_owned() * f64::NAN;
        self.d2_psi_d_r2_2d = self.d2_psi_d_r2_2d.to_owned() * f64::NAN;
        self.d2_psi_d_r_d_z_2d = self.d2_psi_d_r_d_z_2d.to_owned() * f64::NAN;
        self.d2_psi_d_z2_2d = self.d2_psi_d_z2_2d.to_owned() * f64::NAN;
        self.psi_n_2d = self.psi_n_2d.to_owned() * f64::NAN;
        self.j_2d = self.j_2d.to_owned() * f64::NAN;
        self.mask = self.mask.to_owned() * f64::NAN;
        self.psi_2d_coils = self.psi_2d_coils.to_owned() * f64::NAN;
        self.psi_b = f64::NAN;
        self.psi_a = f64::NAN;
        self.ip = f64::NAN;
        self.bounding_r = f64::NAN;
        self.bounding_z = f64::NAN;
        self.delta_z = f64::NAN;
        self.xpt_upper_r = f64::NAN;
        self.xpt_upper_z = f64::NAN;
        self.xpt_lower_r = f64::NAN;
        self.xpt_lower_z = f64::NAN;
        self.n_iter = usize::MAX;
        self.r_mag = f64::NAN;
        self.z_mag = f64::NAN;
        self.xpt_diverted = false;
    }

    /// Solve the inverse Grad-Shafranov problem
    pub fn solve(&mut self) {
        let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.p_prime_source_function.clone();
        let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.ff_prime_source_function.clone();

        // Unpack objects
        let coils_dynamic: &SensorsDynamic = self.coils_dynamic;
        let plasma: &Plasma = self.plasma;

        // Get sensors
        let bp_probes_static: &SensorsStatic = self.bp_probes_static;
        let bp_probes_dynamic: &SensorsDynamic = self.bp_probes_dynamic;
        let flux_loops_static: &SensorsStatic = self.flux_loops_static;
        let flux_loops_dynamic: &SensorsDynamic = self.flux_loops_dynamic;
        let dialoop_static: &SensorsStatic = self.dialoop_static;
        let dialoop_dynamic: &SensorsDynamic = self.dialoop_dynamic;
        let rogowski_coils_static: &SensorsStatic = self.rogowski_coils_static;
        let rogowski_coils_dynamic: &SensorsDynamic = self.rogowski_coils_dynamic;
        let isoflux_static: &SensorsStatic = self.isoflux_static;
        let isoflux_dynamic: &SensorsDynamic = self.isoflux_dynamic;
        let isoflux_boundary_static: &SensorsStatic = self.isoflux_boundary_static;
        let pressure_sensors_static: &SensorsStatic = self.pressure_sensors_static;
        let pressure_sensors_dynamic: &SensorsDynamic = self.pressure_sensors_dynamic;
        let isoflux_boundary_dynamic: &SensorsDynamic = self.isoflux_boundary_dynamic;
        let magnetic_axis_static: &SensorsStatic = self.magnetic_axis_static;
        let magnetic_axis_dynamic: &SensorsDynamic = self.magnetic_axis_dynamic;

        // Plasma grid
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let flat_r: Array1<f64> = plasma.results.get("grid").get("flat").get("r").unwrap_array1();
        let mesh_r: Array2<f64> = plasma.results.get("grid").get("mesh").get("r").unwrap_array2();
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();
        let limit_pts_r: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("r").unwrap_array1();
        let limit_pts_z: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("z").unwrap_array1();
        let vessel_r: Array1<f64> = plasma.results.get("vessel").get("r").unwrap_array1();
        let vessel_z: Array1<f64> = plasma.results.get("vessel").get("z").unwrap_array1();

        // Degrees of freedom
        let passives_shape: &[usize] = bp_probes_static.greens_with_passives.shape();
        let n_passive_dof: usize = passives_shape[0];
        let n_p_prime_dof: usize = p_prime_source_function.source_function_n_dof();
        let n_ff_prime_dof: usize = ff_prime_source_function.source_function_n_dof();
        let n_iter_no_vertical_feedback: usize = self.n_iter_no_vertical_feedback;

        // Constraints
        let n_bp: usize = bp_probes_dynamic.measured.len();
        let n_fl: usize = flux_loops_dynamic.measured.len();
        let n_dialoop: usize = dialoop_dynamic.measured.len();
        let n_rog: usize = rogowski_coils_dynamic.measured.len();
        let n_isoflux: usize = isoflux_dynamic.measured.len();
        let n_isoflux_boundary: usize = isoflux_boundary_dynamic.measured.len();
        let n_pressure_sensors: usize = pressure_sensors_dynamic.measured.len();
        let n_magnetic_axis_constraints: usize = magnetic_axis_dynamic.measured.len();
        let n_p_prime_regularisation: usize = p_prime_source_function.source_function_regularisation().shape()[0];
        let n_ff_prime_regularisation: usize = ff_prime_source_function.source_function_regularisation().shape()[0];
        let passive_regularisations: Array2<f64> = self.passive_regularisations.to_owned();
        let n_passive_regularisation: usize = passive_regularisations.shape()[0];
        let n_delta_z_regularisation: usize = 0; // initially set to 0 because we don't have previous iteration
        let n_constraints: usize = n_bp
            + n_fl
            + n_dialoop
            + n_rog
            + n_isoflux
            + n_isoflux_boundary
            + n_pressure_sensors
            + n_magnetic_axis_constraints
            + n_p_prime_regularisation
            + n_ff_prime_regularisation
            + n_passive_regularisation
            + n_delta_z_regularisation;

        // Magnetic sensor's Greens tables
        let greens_bp_probes_grid: Array2<f64> = bp_probes_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_bp_probes_dz: Array2<f64> = bp_probes_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_bp_probes_pf: Array2<f64> = bp_probes_static.greens_with_pf.to_owned(); // shape = [n_pf, n_sensors]
        let greens_bp_probes_passives: Array2<f64> = bp_probes_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_flux_loops_grid: Array2<f64> = flux_loops_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_flux_loops_dz: Array2<f64> = flux_loops_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_flux_loops_pf: Array2<f64> = flux_loops_static.greens_with_pf.to_owned(); // shape = [n_pf, n_sensors]
        let greens_flux_loops_passives: Array2<f64> = flux_loops_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_rogowski_coils_grid: Array2<f64> = rogowski_coils_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_rogowski_coils_dz: Array2<f64> = rogowski_coils_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_rogowski_coils_pf: Array2<f64> = rogowski_coils_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_rogowski_coils_passives: Array2<f64> = rogowski_coils_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_grid: Array2<f64> = isoflux_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_dz: Array2<f64> = isoflux_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_pf: Array2<f64> = isoflux_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_passives: Array2<f64> = isoflux_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_boundary_grid: Array2<f64> = isoflux_boundary_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_boundary_dz: Array2<f64> = isoflux_boundary_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_boundary_pf: Array2<f64> = isoflux_boundary_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_boundary_passives: Array2<f64> = isoflux_boundary_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_magnetic_axis_grid: Array2<f64> = magnetic_axis_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_magnetic_axis_dz: Array2<f64> = magnetic_axis_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_magnetic_axis_pf: Array2<f64> = magnetic_axis_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_magnetic_axis_passives: Array2<f64> = magnetic_axis_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        // pf_coil_currents
        let pf_coil_currents: Array1<f64> = coils_dynamic.measured.to_owned();

        // TODO: IDEA- change the normalisation so that it does represent current. But this won't work for the IVC eigenvalues
        self.passive_dof_values = Array1::zeros(n_passive_dof);

        // Initialise plasma with a smooth quadratic current-density distribution.
        if let Err(reason) = self.initialise_plasma_with_quadratic_current_density(
            plasma.initial_ip,
            plasma.initial_cur_r,
            plasma.initial_cur_z,
            plasma.initial_minor_radius,
            plasma.initial_kappa,
        ) {
            self.set_to_failed_time_slice();
            self.error_state = Some(Error::InvalidInitialCurrent(reason));
            println!("{:?}", self.error_state.as_ref().unwrap());
            return;
        }

        // Some variables we want to track between iterations
        let mut dof_values_previous: Array1<f64> = Array1::zeros(n_p_prime_dof + n_ff_prime_dof + n_passive_dof + 1);
        let mut psi_a_previous: f64 = 0.0; // needed to calculate gs-error

        // Precompute the reorganised Greens tables for `calculate_psi_and_derivatives`
        // (they do not change between iterations);  timing: 240ms, with [n_r, n_z]=[81, 321]
        let psi_and_derivatives_greens: PsiAndDerivativesGreens = PsiAndDerivativesGreens::new(plasma);

        // Iteration loop
        'iteration_loop: for i_iter in 0..self.n_iter_max {
            // println!("");
            // println!("Iteration {i_iter}");
            // From previous iteration
            let j_2d: Array2<f64> = self.j_2d.to_owned();

            // Updates `psi` and all of its derivatives (including the `delta_z` vertical stability correction);
            // timing: 350ms, with [n_r, n_z]=[81, 321]
            self.calculate_psi_and_derivatives(&psi_and_derivatives_greens);
            let psi_2d: Array2<f64> = self.psi_2d.to_owned();
            let d_psi_d_r_2d: Array2<f64> = self.d_psi_d_r_2d.to_owned();
            let d_psi_d_z_2d: Array2<f64> = self.d_psi_d_z_2d.to_owned();
            let d2_psi_d_r2_2d: Array2<f64> = self.d2_psi_d_r2_2d.to_owned();
            let d2_psi_d_r_d_z_2d: Array2<f64> = self.d2_psi_d_r_d_z_2d.to_owned();
            let d2_psi_d_z2_2d: Array2<f64> = self.d2_psi_d_z2_2d.to_owned();

            // Grid spacing
            let d_r: f64 = r[1] - r[0];
            let d_z: f64 = z[1] - z[0];

            // Find stationary points in `psi` (magnetic axis and x-points)
            let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
                r.view(),
                z.view(),
                psi_2d.view(),
                d_psi_d_r_2d.view(),
                d_psi_d_z_2d.view(),
                d2_psi_d_r2_2d.view(),
                d2_psi_d_r_d_z_2d.view(),
                d2_psi_d_z2_2d.view(),
            );
            // At a minimum we should have found the magnetic axis
            if stationary_points.is_empty() {
                // Set time-slice to failed
                self.set_to_failed_time_slice();

                // Store error state
                self.error_state = Some(Error::NoStationaryPointsFound);
                println!("{:?}", self.error_state.as_ref().unwrap());

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }

            // Store stationary points in the solution
            self.stationary_points = stationary_points.clone();

            // Find the magnetic axis (o-point)
            let magnetic_axis_or_error: Result<MagneticAxis, String> = find_magnetic_axis(&stationary_points, self.r_mag, self.z_mag, &vessel_r, &vessel_z);
            // Test if we have found the magnetic axis
            if magnetic_axis_or_error.is_err() {
                // Set time-slice to failed
                self.set_to_failed_time_slice();

                // Store error state
                self.error_state = Some(Error::NoMagneticAxisFound);
                println!("{:?}", self.error_state.as_ref().unwrap());

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }
            // Unwrap and get results out of `magnetic_axis_or_error`
            let magnetic_axis: MagneticAxis = magnetic_axis_or_error.expect("gs_solution: unwrapping magnetic_axis");
            let mag_r: f64 = magnetic_axis.r;
            let mag_z: f64 = magnetic_axis.z;
            let psi_a: f64 = magnetic_axis.psi;
            self.r_mag = mag_r;
            self.z_mag = mag_z;
            self.psi_a = psi_a;

            // Find boundary
            let plasma_boundary_or_error: Result<BoundaryContour, plasma_geometry::Error> = find_boundary(
                &r,
                &z,
                &psi_2d,
                &d_psi_d_r_2d,
                &d_psi_d_z_2d,
                &d2_psi_d_r_d_z_2d,
                &stationary_points,
                &limit_pts_r,
                &limit_pts_z,
                &vessel_r,
                &vessel_z,
                self.r_mag,
                self.z_mag,
            );
            // Test if we have found a plasma boundary
            if plasma_boundary_or_error.is_err() {
                // Set time-slice to failed
                self.set_to_failed_time_slice();

                // Extract the reasons for no boundary found
                let plasma_boundary_error: plasma_geometry::Error = plasma_boundary_or_error.err().unwrap();
                let (no_xpt_reason, no_limit_point_reason) = match plasma_boundary_error {
                    plasma_geometry::Error::NoBoundaryFound {
                        no_xpt_reason,
                        no_limit_point_reason,
                    } => (no_xpt_reason, no_limit_point_reason),
                };
                // Store error states in this module's own Error enum
                self.error_state = Some(Error::NoBoundaryFound {
                    no_xpt_reason,
                    no_limit_point_reason,
                });

                println!("{:?}", self.error_state.as_ref().unwrap());

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }
            // Unwrap and store the plasma boundary
            let plasma_boundary: BoundaryContour = plasma_boundary_or_error.expect("Failed to find plasma boundary");
            self.mask = plasma_boundary.mask.expect("Failed to unwrap mask");
            self.psi_b = plasma_boundary.bounding_psi;
            self.bounding_r = plasma_boundary.bounding_r;
            self.bounding_z = plasma_boundary.bounding_z;
            let mask: Array2<f64> = self.mask.to_owned();
            let psi_b: f64 = self.psi_b;
            self.xpt_diverted = plasma_boundary.xpt_diverted;

            // Calculate psi_n_2d
            let psi_n_2d: Array2<f64> = &mask * (&psi_2d - psi_a) / (psi_b - psi_a);
            self.psi_n_2d = psi_n_2d.clone();

            // Calculate GS error
            self.calculate_gs_error(psi_a_previous);
            psi_a_previous = psi_a; // needed to calculate gs-error in next iteration

            // Check for convergence
            let gs_error_calculated: f64 = self.gs_error_calculated;
            if gs_error_calculated < self.gs_error_tolerence && i_iter > self.n_iter_min {
                self.n_iter = i_iter;
                break 'iteration_loop; // Exit the iteration loop
            }

            // Check if we have reached the maximum number of iterations
            if i_iter == self.n_iter_max - 1 {
                // Set time-slice to failed
                self.set_to_failed_time_slice();

                // Store error state
                self.error_state = Some(Error::MaxIterReached);
                println!("{:?}", self.error_state.as_ref().unwrap());

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }

            // Flatten variables
            let mask_flat: Array1<f64> = Array1::from_iter(mask.iter().cloned());
            let psi_n_flat: Array1<f64> = Array1::from_iter(psi_n_2d.iter().cloned());
            let j_2d_flat: Array1<f64> = Array1::from_iter(j_2d.iter().cloned());

            let n_vertical_stabilisation: usize;
            if i_iter > n_iter_no_vertical_feedback {
                n_vertical_stabilisation = 1;
            } else {
                n_vertical_stabilisation = 0;
            }

            let n_dof: usize = n_p_prime_dof + n_ff_prime_dof + n_passive_dof + n_vertical_stabilisation;
            // Create the fitting matrix
            let mut fitting_matrix: Array2<f64> = Array2::zeros((n_constraints, n_dof));
            let mut constraint_weights: Array1<f64> = Array1::zeros(n_constraints);
            let mut constraint_values_from_coils: Array1<f64> = Array1::zeros(n_constraints);
            let mut s_measured: Array1<f64> = Array1::zeros(n_constraints);

            // Counter for the constraints
            let mut i_constraint: usize = 0;

            // Add bp_probes to fitting matrix
            for i_sensor in 0..n_bp {
                // j = 2.0 * pi * r * p_prime + 2.0 * pi * ff_prime / (mu_0 * r)

                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_bp_probes_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // j_2d is not consistent with mask. This inconsistency is how the plasma can "move" from iteration to iteration
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_bp_probes_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_bp_probes_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = bp_probes_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = bp_probes_static.fit_settings_weight[i_sensor] / bp_probes_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add flux_loops to fitting matrix
            for i_sensor in 0..n_fl {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_flux_loops_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_flux_loops_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                    //  * &mask_flat
                }

                // PF coil component
                let tmp: Array1<f64> = greens_flux_loops_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                // s_measured[i_constraint] = flux_loops_rs.all.psi.measured[i_sensor];
                s_measured[i_constraint] = flux_loops_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    2.0 * PI * flux_loops_static.fit_settings_weight[i_sensor] / flux_loops_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add dialoop (diamagnetic flux loop) to the fitting matrix.
            //
            // The diamagnetic loop responds to the toroidal flux function `f` (the poloidal-current
            // function), which depends only on the ff' source function — NOT on the toroidal
            // currents. The Green's tables relate toroidal currents to psi/B_p, so the dialoop uses
            // NO Green's functions, and no p', passive, coil or vertical-stabilisation terms.
            //
            // The diamagnetic flux is (Moret Eq. 41):
            //     Phi_t = integral( (f - f_vac) / R ) dA          (over the plasma mask)
            // where, as in `epp_bt_2d`, `f` is reconstructed from the ff' source function:
            //     f = sqrt( f_vac^2 + 2*(psi_b - psi_a)*G ),   G = sum_i ff'_dof[i]*ff'_integral_i(psi_n)
            // and f_vac = R0*B_phi0 = MU_0*i_rod/(2*PI).
            //
            // Linearising for small diamagnetism (|f - f_vac| << |f_vac|):
            //     f - f_vac ~= (psi_b - psi_a) * G / f_vac
            // so the response is linear in the ff' degrees of freedom:
            //     T[i] = ((psi_b - psi_a) / f_vac) * dA * sum_grid [ mask * ff'_integral_i(psi_n) / R ]
            //
            // Note on sign: this linearisation divides by the *signed* f_vac, so it already
            // preserves the correct sign for a negative TF rod current. Expanding the exact
            // f = sign(f_vac)*sqrt(f_vac^2 + 2*(psi_b-psi_a)*G) for small G gives
            // f - f_vac ~= (psi_b - psi_a)*G / f_vac, matching the term below without a separate
            // sign() factor.
            let f_vac: f64 = MU_0 * self.i_rod / (2.0 * PI);
            let d_psi: f64 = self.psi_b - self.psi_a;
            for i_sensor in 0..n_dialoop {
                // ff_prime degrees of freedom only (no p', no passives, no coils, no Green's)
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    let ff_prime_integral: Array1<f64> = ff_prime_source_function.source_function_integral_single_dof(&psi_n_flat, i_ff_prime_dof);
                    let integrand: Array1<f64> = &mask_flat * &ff_prime_integral / &flat_r;
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = (d_psi / f_vac) * d_area * integrand.sum();
                }

                // Store sensor value
                s_measured[i_constraint] = dialoop_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = dialoop_static.fit_settings_weight[i_sensor] / dialoop_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add rogowski_coils to fitting matrix
            for i_sensor in 0..n_rog {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_rogowski_coils_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_rogowski_coils_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_rogowski_coils_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = rogowski_coils_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    rogowski_coils_static.fit_settings_weight[i_sensor] / rogowski_coils_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux to fitting matrix
            for i_sensor in 0..n_isoflux {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_isoflux_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_isoflux_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = isoflux_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = isoflux_static.fit_settings_weight[i_sensor] / isoflux_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux_boundary to fitting matrix
            for i_sensor in 0..n_isoflux_boundary {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] =
                        greens_isoflux_boundary_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux_boundary!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_boundary_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_isoflux_boundary_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = psi_a;

                // Store weights
                constraint_weights[i_constraint] =
                    isoflux_boundary_static.fit_settings_weight[i_sensor] / isoflux_boundary_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add pressure_sensors to fitting matrix
            // d(psi)/d(psi_n)
            let d_psi_d_psi_n: f64 = 1.0 / (psi_b - psi_a);

            for i_sensor in 0..n_pressure_sensors {
                // Find the value of psi_n at the location of the pressure sensor
                let sensor_r: f64 = pressure_sensors_static.geometry_r[i_sensor];
                let sensor_z: f64 = pressure_sensors_static.geometry_z[i_sensor];

                // Find the nearest grid point to the sensor location
                let i_r_nearest: usize = (&r - sensor_r).abs().argmin().expect("find_viable_limit_point: unwrapping i_r_nearest");
                let i_z_nearest: usize = (&z - sensor_z).abs().argmin().expect("find_viable_limit_point: unwrapping i_z_nearest");

                // Find the four corner grid points surrounding the pressure sensor
                let i_r_nearest_left: usize;
                let i_r_nearest_right: usize;
                let i_z_nearest_lower: usize;
                let i_z_nearest_upper: usize;
                if pressure_sensors_static.geometry_r[i_sensor] > r[i_r_nearest] {
                    i_r_nearest_left = i_r_nearest;
                    i_r_nearest_right = i_r_nearest + 1;
                } else {
                    i_r_nearest_left = i_r_nearest - 1;
                    i_r_nearest_right = i_r_nearest;
                }
                if pressure_sensors_static.geometry_z[i_sensor] > z[i_z_nearest] {
                    i_z_nearest_lower = i_z_nearest;
                    i_z_nearest_upper = i_z_nearest + 1;
                } else {
                    i_z_nearest_lower = i_z_nearest - 1;
                    i_z_nearest_upper = i_z_nearest;
                }

                // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
                let f: ArrayView2<f64> = psi_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d_f_d_r: ArrayView2<f64> = d_psi_d_r_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d_f_d_z: ArrayView2<f64> = d_psi_d_z_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d2_f_d_r_d_z: ArrayView2<f64> = d2_psi_d_r_d_z_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);

                // Create a bicubic interpolator
                let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, f, d_f_d_r, d_f_d_z, d2_f_d_r_d_z);

                // Find psi at the pressure sensor
                let x: f64 = (pressure_sensors_static.geometry_r[i_sensor] - r[i_r_nearest_left]) / d_r;
                let y: f64 = (pressure_sensors_static.geometry_z[i_sensor] - z[i_z_nearest_lower]) / d_z;
                let psi_at_sensor: f64 = bicubic_interpolator.interpolate(x, y);

                let psi_n_at_sensor: f64 = (psi_at_sensor - psi_a) / (psi_b - psi_a);
                if !(0.0..=1.0).contains(&psi_n_at_sensor) {
                    println!(
                        "Warning: pressure sensor {} is outside of the plasma boundary (psi_n = {})",
                        i_sensor, psi_n_at_sensor
                    );
                    // Skip to the next sensor
                    continue;
                }

                let psi_n_from_sensor_to_boundary: Array1<f64> = Array1::from_vec(vec![psi_n_at_sensor, 1.0]);

                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    // Indefinitive integral of p_prime = pressure
                    let indefinite_integral_p_prime: Array1<f64> =
                        p_prime_source_function.source_function_integral_single_dof(&psi_n_from_sensor_to_boundary, i_p_prime_dof);

                    // The constant of integration is zero pressure at the boundary; or this can be thought of as a definite integral from the sensor to the boundary
                    // let definite_integral_p_prime: f64 = indefinite_integral_p_prime[1] - indefinite_integral_p_prime[0];
                    let definite_integral_p_prime: f64 = indefinite_integral_p_prime[0] - indefinite_integral_p_prime[1];

                    // Add to fitting matrix
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = definite_integral_p_prime / d_psi_d_psi_n;
                }

                // Vertical stability (not for pressure sensors)
                // TODO: should there be vertical stability for pressure sensors? I don't think so?

                // Store sensor values
                s_measured[i_constraint] = pressure_sensors_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    pressure_sensors_static.fit_settings_weight[i_sensor] / pressure_sensors_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add magnetic_axis to fitting matrix
            for i_sensor in 0..n_magnetic_axis_constraints {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_magnetic_axis_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_magnetic_axis_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_magnetic_axis_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_magnetic_axis_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_magnetic_axis_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = 0.0; // Magnetic axis value is always zero

                // Store weights
                constraint_weights[i_constraint] =
                    magnetic_axis_static.fit_settings_weight[i_sensor] / magnetic_axis_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Pressure sensor:
            // 1.) Find where the pressure sensors are located in `psi_n`
            // 2.) Calculate the "sensor" measurement matrix:
            //     `pressure[psi_n] = pressure_int_dof_01 * d(psi)/d(psi_n) + pressure_int_dof_02 * d(psi)/d(psi_n) + ... = measured_pressure`
            //     where `pressure_int_dof_xx` = integral from LCFS to psi_n of basis function xx

            // Add p_prime_regularisation to fitting matrix
            let p_prime_regularisation: Array2<f64> = p_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_p_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, 0..n_p_prime_dof])
                    .assign(&p_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add ff_prime_regularisation to fitting matrix
            let ff_prime_regularisation: Array2<f64> = ff_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_ff_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof])
                    .assign(&ff_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // // Add passive regularisation to the fitting matrix
            let regularisation_scaling: f64 = 0.001 * PI; // This regularisation_scaling factor need improving and explaining!

            let passive_regularisations_weight: Array1<f64> = self.passive_regularisations_weight.to_owned();
            for i_regularisation in 0..n_passive_regularisation {
                let passive_regularisation: Array1<f64> = passive_regularisations.slice(s![i_regularisation, ..]).to_owned();

                // Add passive degrees of freedom
                fitting_matrix
                    .slice_mut(s![
                        i_constraint,
                        n_p_prime_dof + n_ff_prime_dof..=n_p_prime_dof + n_ff_prime_dof + n_passive_dof - 1
                    ])
                    .assign(&passive_regularisation);

                // Add weight
                constraint_weights[i_constraint] = passive_regularisations_weight[i_regularisation] * regularisation_scaling;

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Solve for the least squares problem for the source function coefficients, passive currents, and vertical stability
            let a: Array2<f64> = Array2::from_diag(&constraint_weights).dot(&fitting_matrix); // matrix-matrix multiplication
            let b: Array1<f64> = &constraint_weights * &s_measured - &constraint_weights * &constraint_values_from_coils;

            fn l2_norm(v: &Array1<f64>) -> f64 {
                // Sum of squares of the elements in the vector
                let sum_of_squares: f64 = v.iter().map(|&x| x * x).sum();
                // Take the square root to get the L2 norm
                sum_of_squares.sqrt()
            }

            // Preconditioner
            let n_cols: usize = a.ncols();

            // Compute the L2 norm for each column and fill the diagonal of D
            let mut d: Array2<f64> = Array2::zeros((n_cols, n_cols)); // Initialize a square matrix D with zeros
            for i in 0..n_cols {
                let column: Array1<f64> = a.column(i).to_owned();
                let norm = l2_norm(&column);
                // Fill the diagonal of D with the inverse of the norm, or 0.0 if the norm is zero
                if norm > 0.0 {
                    d[(i, i)] = 1.0 / norm;
                } else {
                    println!("Warning: norm of column {} is zero, setting preconditioner to zero.", i);
                    d[(i, i)] = 0.0;
                }
            }

            let a_preconditioned: Array2<f64> = a.clone().dot(&d);

            // SVD-based least squares solve using faer (equivalent to LAPACK dgelss)
            let (m_usize, n_usize) = a_preconditioned.dim();
            let a_faer: faer::Mat<f64> = faer::Mat::from_fn(m_usize, n_usize, |i, j| a_preconditioned[(i, j)]);
            let b_faer: faer::Mat<f64> = faer::Mat::from_fn(m_usize, 1, |i, _| b[i]);

            let svd: FaerSvd<f64> = FaerSvd::new_thin(a_faer.as_ref()).expect("SVD decomposition failed");
            let x_faer: faer::Mat<f64> = svd.solve_lstsq(b_faer.as_ref());

            let mut d_new_vec: Vec<f64> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                d_new_vec.push(x_faer[(i_dof, 0)]);
            }
            let d_new: Array1<f64> = Array1::from_vec(d_new_vec);

            let mut dof_values: Array1<f64> = d.dot(&d_new); // `d` is the preconditioning matrix

            // // Could add Anderson mixing here??????????????
            // if i_iter > 3 {
            //     dof_values = 0.6 * &dof_values + 0.4 * &dof_values_previous;
            // }
            // let dof_values_old: Array1<f64> = dof_values.clone();

            // Compute the condition number from SVD singular values
            let s_col = svd.S().column_vector();
            let mut s: Vec<f64> = Vec::with_capacity(s_col.nrows());
            for i_singular_value in 0..s_col.nrows() {
                s.push(s_col[i_singular_value]);
            }
            if let (Some(&sigma_max), Some(&sigma_min)) = (s.first(), s.iter().filter(|&&x| x > 0.0).last()) {
                let _condition_number: f64 = sigma_max / sigma_min;
            } else {
                println!("Matrix is rank-deficient or singular, condition number is undefined.");
            }

            // // Add Anderson mixing. Will this help???  // NO: Anderson mixing does not seem to help!!
            // if i_iter > 3 {
            //     dof_values = 0.3 * &dof_values + 0.7 * &dof_values_previous;
            // }

            if i_iter > n_iter_no_vertical_feedback {
                dof_values_previous = dof_values.clone();
            }

            // Extract p_prime
            let p_prime_dof_values: Array1<f64> = dof_values.slice(s![0..n_p_prime_dof]).to_owned();
            self.p_prime_dof_values = p_prime_dof_values.clone();

            // Extract ff_prime
            let ff_prime_dof_values: Array1<f64> = dof_values.slice(s![n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof]).to_owned();
            self.ff_prime_dof_values = ff_prime_dof_values.clone();

            // Extract passive currents
            let passive_dof_values = dof_values
                .slice(s![n_p_prime_dof + n_ff_prime_dof..n_p_prime_dof + n_ff_prime_dof + n_passive_dof])
                .to_owned();
            self.passive_dof_values = passive_dof_values.clone();

            // Extract vertical stability
            let delta_z: f64;
            if i_iter > n_iter_no_vertical_feedback {
                delta_z = dof_values.last().expect("dof_values empty").to_owned();
            } else {
                delta_z = 0.0;
            }
            self.delta_z = delta_z;

            // Calculate j_2d
            self.calculate_j(&mesh_r); // TODO: I don't like having to pass mesh_r in
            let j_2d: Array2<f64> = self.j_2d.to_owned();

            // Total plasma current
            // TODO: do we actually need to calculate Ip at every iteration?
            let i_2d: Array2<f64> = &j_2d * d_area;
            let ip: f64 = i_2d.sum();
            self.ip = ip;

            // // Write the time-slice to numpy files for debugging
            // self._write_time_slice_to_file(i_iter);
        }
    }

    /// Calculate the poloidal flux, psi, in the 2d (r, z) grid.
    ///
    /// 1. Calculate the "unshifted" flux and the required derivatives (9 fields):
    ///     * `psi_unshifted`
    ///     * `d_psi_d_r_unshifted`
    ///     * `d_psi_d_z_unshifted`
    ///     * `d2_psi_d_r2_unshifted`
    ///     * `d2_psi_d_r_d_z_unshifted`
    ///     * `d2_psi_d_z2_unshifted`
    ///     * `d3_psi_d_r2_d_z_unshifted`
    ///     * `d3_psi_d_r_d_z2_unshifted`
    ///     * `d3_psi_d_z3_unshifted`
    /// 2. Apply the vertical stability correction (resulting in 6 fields):
    ///     * `psi = psi_unshifted + delta_z * d_psi_d_z_unshifted`
    ///     * `d_psi_d_r = d_psi_d_r_unshifted + delta_z * d2_psi_d_r_d_z_unshifted`
    ///     * `d_psi_d_z = d_psi_d_z_unshifted + delta_z * d2_psi_d_z2_unshifted`
    ///     * `d2_psi_d_r2 = d2_psi_d_r2_unshifted + delta_z * d3_psi_d_r2_d_z_unshifted`
    ///     * `d2_psi_d_r_d_z = d2_psi_d_r_d_z_unshifted + delta_z * d3_psi_d_r_d_z2_unshifted`
    ///     * `d2_psi_d_z2 = d2_psi_d_z2_unshifted + delta_z * d3_psi_d_z3_unshifted`
    /// These are the derivatives we require for the bicubic interpolation to find the x-point and magnetic axis
    /// 3. Store the shifted flux and derivatives in the class.
    ///
    /// Only `psi` and its derivatives are used; `br` and `bz` do not appear.
    ///
    /// The plasma contribution is calculated with two GEMMs; see `PsiAndDerivativesGreens` for the
    /// reorganisation of the convolution over current sources.
    pub fn calculate_psi_and_derivatives(&mut self, greens_tables: &PsiAndDerivativesGreens) {
        // Unpack from self
        let plasma: &Plasma = self.plasma;
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let j_2d: &Array2<f64> = &self.j_2d;
        let pf_coil_currents: &Array1<f64> = &self.coils_dynamic.measured;
        let passive_dof_values: &Array1<f64> = &self.passive_dof_values;
        let delta_z: f64 = self.delta_z;

        // ====================================================================
        // Part 1: the "unshifted" flux and derivatives
        // ====================================================================

        // Helper: contract a Greens matrix (n_z * n_r, n_dof) with a dof vector and reshape to (n_z, n_r)
        let contract = |g_matrix: &Array2<f64>, dof_values: &Array1<f64>| -> Array2<f64> {
            return g_matrix
                .dot(dof_values)
                .to_shape((n_z, n_r))
                .expect("calculate_psi_and_derivatives: failed to reshape contracted Greens matrix")
                .to_owned();
        };

        // PF coils
        // `psi` is precomputed (the PF currents are fixed within a time-slice);
        // the other fields are the Greens tables contracted with the PF currents
        let psi_2d_coils: &Array2<f64> = &self.psi_2d_coils;
        let d_psi_d_r_2d_coils: Array2<f64> = contract(&greens_tables.g_d_psi_d_r_coils_matrix, pf_coil_currents);
        let d_psi_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d_psi_d_z_coils_matrix, pf_coil_currents);
        let d2_psi_d_r2_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r2_coils_matrix, pf_coil_currents);
        let d2_psi_d_r_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r_d_z_coils_matrix, pf_coil_currents);
        let d2_psi_d_z2_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_z2_coils_matrix, pf_coil_currents);
        let d3_psi_d_r2_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r2_d_z_coils_matrix, pf_coil_currents);
        let d3_psi_d_r_d_z2_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r_d_z2_coils_matrix, pf_coil_currents);
        let d3_psi_d_z3_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_z3_coils_matrix, pf_coil_currents);

        // Passives: the Greens tables contracted with the passive degrees of freedom
        let psi_2d_passives: Array2<f64> = contract(&greens_tables.g_psi_passives_matrix, passive_dof_values);
        let d_psi_d_r_2d_passives: Array2<f64> = contract(&greens_tables.g_d_psi_d_r_passives_matrix, passive_dof_values);
        let d_psi_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d_psi_d_z_passives_matrix, passive_dof_values);
        let d2_psi_d_r2_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r2_passives_matrix, passive_dof_values);
        let d2_psi_d_r_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r_d_z_passives_matrix, passive_dof_values);
        let d2_psi_d_z2_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_z2_passives_matrix, passive_dof_values);
        let d3_psi_d_r2_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r2_d_z_passives_matrix, passive_dof_values);
        let d3_psi_d_r_d_z2_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r_d_z2_passives_matrix, passive_dof_values);
        let d3_psi_d_z3_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_z3_passives_matrix, passive_dof_values);

        // Plasma: two GEMMs over the reorganised tables (see `PsiAndDerivativesGreens`).
        // `w_even` and `w_odd` gather the current sources by (vertical offset, source radius):
        //     w_even[(i_z, i_offset_z * n_r + i_cur_r)] = d_area * (j_below + j_above)
        //     w_odd[(i_z, i_offset_z * n_r + i_cur_r)]  = d_area * (j_below - j_above)
        // where `j_below = j_2d[(i_z - i_offset_z, i_cur_r)]` (a source below the grid point) and
        // `j_above = j_2d[(i_z + i_offset_z, i_cur_r)]` (a source at or above the grid point).
        // The odd kernels (`d_psi_d_z`, `d2_psi_d_r_d_z`) change sign with the source side:
        // sources with `i_z <= i_cur_z` enter with -1
        let mut w_even: Array2<f64> = Array2::zeros((n_z, n_z * n_r));
        let mut w_odd: Array2<f64> = Array2::zeros((n_z, n_z * n_r));
        for i_z in 0..n_z {
            for i_offset_z in 0..n_z {
                let i_column_start: usize = i_offset_z * n_r;

                // Source below the grid point: i_cur_z = i_z - i_offset_z (excluding i_offset_z = 0)
                if i_offset_z > 0 && i_z >= i_offset_z {
                    let i_cur_z: usize = i_z - i_offset_z;
                    for i_cur_r in 0..n_r {
                        let j_this: f64 = d_area * j_2d[(i_cur_z, i_cur_r)];
                        w_even[(i_z, i_column_start + i_cur_r)] += j_this;
                        w_odd[(i_z, i_column_start + i_cur_r)] += j_this;
                    }
                }

                // Source at or above the grid point: i_cur_z = i_z + i_offset_z (including i_offset_z = 0)
                if i_z + i_offset_z < n_z {
                    let i_cur_z: usize = i_z + i_offset_z;
                    for i_cur_r in 0..n_r {
                        let j_this: f64 = d_area * j_2d[(i_cur_z, i_cur_r)];
                        w_even[(i_z, i_column_start + i_cur_r)] += j_this;
                        w_odd[(i_z, i_column_start + i_cur_r)] -= j_this;
                    }
                }
            }
        }

        // GEMMs, using `faer` (multi-threaded)
        let mut plasma_even: faer::Mat<f64> = faer::Mat::zeros(n_z, 5 * n_r);
        matmul(
            plasma_even.as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(
                w_even.as_slice().expect("calculate_psi_and_derivatives: `w_even` is not contiguous"),
                n_z,
                n_z * n_r,
            ),
            MatRef::from_row_major_slice(
                greens_tables
                    .g_even_plasma_by_offset
                    .as_slice()
                    .expect("calculate_psi_and_derivatives: `g_even_plasma_by_offset` is not contiguous"),
                n_z * n_r,
                5 * n_r,
            ),
            1.0,
            Par::rayon(0),
        );
        let mut plasma_odd: faer::Mat<f64> = faer::Mat::zeros(n_z, 4 * n_r);
        matmul(
            plasma_odd.as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(
                w_odd.as_slice().expect("calculate_psi_and_derivatives: `w_odd` is not contiguous"),
                n_z,
                n_z * n_r,
            ),
            MatRef::from_row_major_slice(
                greens_tables
                    .g_odd_plasma_by_offset
                    .as_slice()
                    .expect("calculate_psi_and_derivatives: `g_odd_plasma_by_offset` is not contiguous"),
                n_z * n_r,
                4 * n_r,
            ),
            1.0,
            Par::rayon(0),
        );

        // Assemble the unshifted fields (coils + passives + plasma).
        // The `plasma_even` / `plasma_odd` column blocks follow the concatenation order
        // documented on `PsiAndDerivativesGreens`
        let mut psi_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_z2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_r2_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_r_d_z2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_z3_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                psi_2d_unshifted[(i_z, i_r)] = psi_2d_coils[(i_z, i_r)] + psi_2d_passives[(i_z, i_r)] + plasma_even[(i_z, i_r)];
                d_psi_d_r_2d_unshifted[(i_z, i_r)] = d_psi_d_r_2d_coils[(i_z, i_r)] + d_psi_d_r_2d_passives[(i_z, i_r)] + plasma_even[(i_z, n_r + i_r)];
                d2_psi_d_r2_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_r2_2d_coils[(i_z, i_r)] + d2_psi_d_r2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 2 * n_r + i_r)];
                d2_psi_d_z2_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_z2_2d_coils[(i_z, i_r)] + d2_psi_d_z2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 3 * n_r + i_r)];
                d3_psi_d_r_d_z2_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_r_d_z2_2d_coils[(i_z, i_r)] + d3_psi_d_r_d_z2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 4 * n_r + i_r)];
                d_psi_d_z_2d_unshifted[(i_z, i_r)] = d_psi_d_z_2d_coils[(i_z, i_r)] + d_psi_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, i_r)];
                d2_psi_d_r_d_z_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_r_d_z_2d_coils[(i_z, i_r)] + d2_psi_d_r_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, n_r + i_r)];
                d3_psi_d_r2_d_z_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_r2_d_z_2d_coils[(i_z, i_r)] + d3_psi_d_r2_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, 2 * n_r + i_r)];
                d3_psi_d_z3_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_z3_2d_coils[(i_z, i_r)] + d3_psi_d_z3_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, 3 * n_r + i_r)];
            }
        }

        // ====================================================================
        // Part 2: apply the vertical stability correction
        // ====================================================================
        // `delta_z` is NaN before the first inverse solve (and 0.0 while the vertical feedback is off)
        let psi_2d: Array2<f64>;
        let d_psi_d_r_2d: Array2<f64>;
        let d_psi_d_z_2d: Array2<f64>;
        let d2_psi_d_r2_2d: Array2<f64>;
        let d2_psi_d_r_d_z_2d: Array2<f64>;
        let d2_psi_d_z2_2d: Array2<f64>;
        if delta_z.is_nan() {
            psi_2d = psi_2d_unshifted;
            d_psi_d_r_2d = d_psi_d_r_2d_unshifted;
            d_psi_d_z_2d = d_psi_d_z_2d_unshifted;
            d2_psi_d_r2_2d = d2_psi_d_r2_2d_unshifted;
            d2_psi_d_r_d_z_2d = d2_psi_d_r_d_z_2d_unshifted;
            d2_psi_d_z2_2d = d2_psi_d_z2_2d_unshifted;
        } else {
            psi_2d = psi_2d_unshifted + delta_z * &d_psi_d_z_2d_unshifted;
            d_psi_d_r_2d = d_psi_d_r_2d_unshifted + delta_z * &d2_psi_d_r_d_z_2d_unshifted;
            d_psi_d_z_2d = d_psi_d_z_2d_unshifted + delta_z * &d2_psi_d_z2_2d_unshifted;
            d2_psi_d_r2_2d = d2_psi_d_r2_2d_unshifted + delta_z * &d3_psi_d_r2_d_z_2d_unshifted;
            d2_psi_d_r_d_z_2d = d2_psi_d_r_d_z_2d_unshifted + delta_z * &d3_psi_d_r_d_z2_2d_unshifted;
            d2_psi_d_z2_2d = d2_psi_d_z2_2d_unshifted + delta_z * &d3_psi_d_z3_2d_unshifted;
        }

        // ====================================================================
        // Part 3: store the shifted flux and derivatives in the class
        // ====================================================================
        self.psi_2d = psi_2d;
        self.d_psi_d_r_2d = d_psi_d_r_2d;
        self.d_psi_d_z_2d = d_psi_d_z_2d;
        self.d2_psi_d_r2_2d = d2_psi_d_r2_2d;
        self.d2_psi_d_r_d_z_2d = d2_psi_d_r_d_z_2d;
        self.d2_psi_d_z2_2d = d2_psi_d_z2_2d;
    }

    fn calculate_j(&mut self, mesh_r: &Array2<f64>) {
        let psi_n_2d: Array2<f64> = self.psi_n_2d.to_owned();
        let (n_z, n_r) = psi_n_2d.dim();
        let mask: Array2<f64> = self.mask.to_owned();
        let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.p_prime_source_function.clone();
        let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.ff_prime_source_function.clone();

        // Calculate profiles
        let psi_n_flat: Array1<f64> = Array1::from_iter(psi_n_2d.iter().cloned());

        let p_prime_dof_values: Array1<f64> = self.p_prime_dof_values.to_owned();
        let ff_prime_dof_values: Array1<f64> = self.ff_prime_dof_values.to_owned();

        let p_prime_2d: Array2<f64> = p_prime_source_function
            .source_function_value(&psi_n_flat, &p_prime_dof_values.clone())
            .to_shape((n_z, n_r))
            .expect("gs_solution: error in p_prime_2d")
            .to_owned();
        let j_2d_p_prime: Array2<f64> = 2.0 * PI * mesh_r * p_prime_2d * &mask;

        let ff_prime_2d: Array2<f64> = ff_prime_source_function
            .source_function_value(&psi_n_flat, &ff_prime_dof_values.clone())
            .to_shape((n_z, n_r))
            .expect("gs_solution: error in ff_prime_2d")
            .to_owned();
        let j_2d_ff_prime: Array2<f64> = 2.0 * PI * ff_prime_2d * &mask / (MU_0 * mesh_r);

        // Calculate j_2d
        let j_2d: Array2<f64> = j_2d_p_prime + j_2d_ff_prime;
        self.j_2d = j_2d.clone();
    }

    pub fn initialise_plasma_with_quadratic_current_density(
        &mut self,
        initial_ip: f64,
        initial_cur_r: f64,
        initial_cur_z: f64,
        initial_minor_radius: f64,
        initial_kappa: f64,
    ) -> Result<(), String> {
        // Unpack objects
        let plasma: &Plasma = self.plasma;
        let coils_dynamic: &SensorsDynamic = self.coils_dynamic;

        // Extract stuff from Plasma
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let greens_pf_grid: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("psi").unwrap_array3();

        // Extract stuff from Coils
        let pf_currents: Array1<f64> = coils_dynamic.measured.to_owned();

        let (n_z, n_r, n_pf): (usize, usize, usize) = greens_pf_grid.dim();

        let mut psi_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_pf in 0..n_pf {
            psi_2d_coils = psi_2d_coils + &greens_pf_grid.slice(s![.., .., i_pf]) * pf_currents[i_pf];
        }

        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();
        let limiter_r: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("r").unwrap_array1();
        let limiter_z: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("z").unwrap_array1();
        let vessel_r: Array1<f64> = plasma.results.get("vessel").get("r").unwrap_array1();
        let vessel_z: Array1<f64> = plasma.results.get("vessel").get("z").unwrap_array1();
        let j_2d = quadratic_current_density_seed(
            &r,
            &z,
            &limiter_r,
            &limiter_z,
            &vessel_r,
            &vessel_z,
            d_area,
            initial_ip,
            initial_cur_r,
            initial_cur_z,
            initial_minor_radius,
            initial_kappa,
        )?;

        // Store in self
        self.j_2d = j_2d;
        self.psi_2d_coils = psi_2d_coils;
        self.r_mag = initial_cur_r;
        self.z_mag = initial_cur_z;
        Ok(())
    }

    /// Calculate the Grad-Shafranov "error"
    /// In the Picard iteration we change the solution by the error,
    /// so what we are doing here is checking to see how much the solutions
    /// is changing by
    fn calculate_gs_error(&mut self, psi_a_previous: f64) {
        // Get stuff out of self
        let psi_a: f64 = self.psi_a;
        let psi_b: f64 = self.psi_b;

        // Calculate the "error", in the same way EFIT does (called `cerror`)
        // Note, while this might "look" like a convergence test, it is in fact very similar
        // to a residule, since at each iteration the solution changes by the residule
        let gs_error_calculated: f64 = (psi_a - psi_a_previous).abs() / (psi_b - psi_a).abs();

        self.gs_error_calculated = gs_error_calculated;
    }

    /// Calculate the Grad Shafranov error by calcuating the LHS and RHS
    /// on the 2D (r, z) grid and seeing the difference = LHS - RHS.
    ///
    /// **This function is only used for development**
    fn _calculate_gs_error_numerical(&mut self) {
        // get stuff out of self
        let plasma: &Plasma = self.plasma;
        let psi_2d: Array2<f64> = self.psi_2d.to_owned();
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();

        // Define some variables
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        // Laplacian(psi)
        let mut laplacian_psi: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 1..(n_r - 1) {
            for i_z in 1..(n_z - 1) {
                let d2_psi_dz2: f64 = (psi_2d[(i_z + 1, i_r)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z - 1, i_r)]) / (d_z * d_z);
                let d2_psi_dr2: f64 = (psi_2d[(i_z, i_r + 1)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z, i_r - 1)]) / (d_r * d_r);
                let r_d_psi_dr: f64 = (1.0 / r[i_r]) * (psi_2d[(i_z, i_r + 1)] - psi_2d[(i_z, i_r - 1)]) / (2.0 * d_r);

                laplacian_psi[(i_z, i_r)] = d2_psi_dr2 - r_d_psi_dr + d2_psi_dz2;
            }
        }
        let mask: Array2<f64> = self.mask.to_owned();
        laplacian_psi = laplacian_psi * mask;

        // RHS of Grad-Shafranov equation
        // Eq. 3 in "Tokamak equilibrium reconstruction code LIUQE and its real time implementation", 2015
        let j_2d: Array2<f64> = self.j_2d.to_owned();
        let mut gs_rhs: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 0..n_r {
            let tmp: Array1<f64> = -2.0 * PI * MU_0 * r[i_r] * j_2d.slice(s![.., i_r]).to_owned();
            gs_rhs.slice_mut(s![.., i_r]).assign(&tmp);
        }

        // Calculate the residual
        // Note - there is high residual at the boundary
        // Perhaps we should make the mask larger??
        let residual_2d: Array2<f64> = laplacian_psi - gs_rhs;
        println!("{:?}", residual_2d);
    }

    /// Writes the current time slice to numpy files for debugging
    ///
    /// **This function is only used for development**
    fn _write_time_slice_to_file(&self, i_iter: usize) {
        use std::path::Path;

        // Equivalent to `mkdir -p tmp`
        std::fs::create_dir_all("tmp").expect("Failed to create 'tmp' directory");

        let d_psi_d_r_2d: Array2<f64> = self.d_psi_d_r_2d.to_owned();
        let d_psi_d_z_2d: Array2<f64> = self.d_psi_d_z_2d.to_owned();
        let psi_2d: Array2<f64> = self.psi_2d.to_owned();
        let psi_b: f64 = self.psi_b;
        let bounding_r: f64 = self.bounding_r;
        let bounding_z: f64 = self.bounding_z;
        let mag_r: f64 = self.r_mag;
        let mag_z: f64 = self.z_mag;

        // Filename has two leading zeros, e.g. i_iter=000, i_iter=001, ...
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_psi_2d.npy", i_iter)), &psi_2d);
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_d_psi_d_r_2d.npy", i_iter)), &d_psi_d_r_2d);
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_d_psi_d_z_2d.npy", i_iter)), &d_psi_d_z_2d);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_psi_b.npy", i_iter)), psi_b);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_bounding_r.npy", i_iter)), bounding_r);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_bounding_z.npy", i_iter)), bounding_z);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_mag_r.npy", i_iter)), mag_r);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_mag_z.npy", i_iter)), mag_z);
    }
}
