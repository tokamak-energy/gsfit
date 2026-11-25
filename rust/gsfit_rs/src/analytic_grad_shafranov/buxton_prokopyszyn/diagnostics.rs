//! Diagnostics of a summed series: the Grad-Shafranov residual inside the LCFS, the coefficient-growth flag of unstable series, and
//! whether the LCFS is a closed set of nested surfaces.

use super::configuration::{Point, Points};
use super::lattice::{FluxValues, Lattice, LatticeSeries, series_evaluate, series_size, series_values};
use super::profiles::{ProfileShapes, grad_shafranov_residual, source_term};
use super::radial_series::RadialVariable;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// The GS residual is also measured along rays from the magnetic axis towards this many points of the given LCFS. A series'
/// truncation error grows towards the LCFS, on ST40's shape within a fraction of a millimetre of it, where the grid has no points
/// [dimensionless]
pub const N_EDGE_RAY: usize = 128;

/// Fractions of the way from the axis to the LCFS at which each ray is sampled, clustered towards the LCFS [dimensionless]
const EDGE_FRACTIONS: [f64; 13] = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0];

/// The search for a ray's crossing of the solution's own LCFS: `N_CROSSING_SEARCH` samples from `CROSSING_SEARCH_START` to
/// `CROSSING_SEARCH_END` of the way to the given LCFS point (or to where the radial series stops being valid), then
/// `N_CROSSING_BISECTION` bisections of the first bracket [dimensionless]
const CROSSING_SEARCH_START: f64 = 0.3;
const CROSSING_SEARCH_END: f64 = 1.5;
const N_CROSSING_SEARCH: usize = 25;
const N_CROSSING_BISECTION: usize = 40;

/// The rays the GS residual is also measured along, and how far
pub struct EdgeRays<'a> {
    /// The rays' targets, points of the given LCFS and its X-points [metre]
    pub targets: &'a Points,
    /// Sample each ray up to its crossing of the solution's own LCFS, rather than up to its target
    pub to_own_lcfs: bool,
    /// With `to_own_lcfs`, the rays stop here: the inboard end of where the radial series is valid [metre]
    pub r_minimum: f64,
    /// ... and the outboard end [metre]
    pub r_maximum: f64,
}

/// A grid over the model box, and its points inside the LCFS
struct InteriorGrid {
    /// [metre]
    r: Vec<f64>,
    /// `(1 - psi_N)` and its derivatives, 1 at the axis
    flux: FluxValues,
    /// Inside the LCFS: `(1 - psi_N) > 0` and connected to the magnetic axis
    inside: Vec<bool>,
}

/// The result of `closure_check`
#[derive(Clone, Copy, Debug)]
pub struct Closure {
    /// All three checks pass
    pub closed: bool,
    /// The region `(1 - psi_N) > 0` connected to the axis reaches the edges of the box
    pub reaches_box_edge: bool,
    /// Local maxima in the region [dimensionless]
    pub n_maximum: usize,
    /// `(1 - psi_N)` falls monotonically along rays from the axis to the LCFS
    pub monotonic: bool,
}

/// The points of a 2-D boolean grid connected to `start` (4-connectivity), by flood fill
fn connected_region(positive: &Array2<bool>, start: (usize, usize)) -> Array2<bool> {
    let (n_row, n_col): (usize, usize) = positive.dim();
    let mut region: Array2<bool> = Array2::from_elem((n_row, n_col), false);
    if !positive[start] {
        return region;
    }
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    region[start] = true;
    queue.push_back(start);
    // Every point enters the queue at most once
    let n_point_max: usize = n_row * n_col;
    'fill_loop: for _i_point in 0..n_point_max {
        let (i_row, i_col): (usize, usize) = match queue.pop_front() {
            Some(point) => point,
            None => break 'fill_loop,
        };
        let mut neighbours: Vec<(usize, usize)> = Vec::with_capacity(4);
        if i_row > 0 {
            neighbours.push((i_row - 1, i_col));
        }
        if i_row + 1 < n_row {
            neighbours.push((i_row + 1, i_col));
        }
        if i_col > 0 {
            neighbours.push((i_row, i_col - 1));
        }
        if i_col + 1 < n_col {
            neighbours.push((i_row, i_col + 1));
        }
        for neighbour in neighbours {
            if positive[neighbour] && !region[neighbour] {
                region[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }
    region
}

/// The index of the value nearest `target`
fn nearest_index(values: &Array1<f64>, target: f64) -> usize {
    let n_value: usize = values.len();
    let mut i_nearest: usize = 0;
    let mut distance_min: f64 = f64::INFINITY;
    for i_value in 0..n_value {
        let distance: f64 = (values[i_value] - target).abs();
        if distance < distance_min {
            distance_min = distance;
            i_nearest = i_value;
        }
    }
    i_nearest
}

/// The targets of the rays of `gs_residual_ratio`: `N_EDGE_RAY` points of the given LCFS, equally spaced in arclength, and its X-points.
///
/// # Arguments
/// * `outline` - the given LCFS, closed (last point = first), e.g. `Configuration::outline` [metre]
/// * `x_points` - the X-points [metre]
pub fn edge_targets(outline: &Points, x_points: &[Point]) -> Points {
    let n_outline: usize = outline.r.len();
    let n_vertex: usize = n_outline - 1;
    // The vertex nearest each of N_EDGE_RAY equally spaced arclengths (the given points themselves, never chords between them)
    let mut arclength: Array1<f64> = Array1::from_elem(n_vertex, f64::NAN);
    let mut perimeter: f64 = 0.0;
    for i_vertex in 0..n_vertex {
        arclength[i_vertex] = perimeter;
        perimeter += (outline.r[i_vertex + 1] - outline.r[i_vertex]).hypot(outline.z[i_vertex + 1] - outline.z[i_vertex]);
    }
    let mut is_chosen: Vec<bool> = vec![false; n_vertex];
    for i_target in 0..N_EDGE_RAY {
        let target: f64 = (i_target as f64) * perimeter / (N_EDGE_RAY as f64);
        is_chosen[nearest_index(&arclength, target)] = true;
    }
    let mut edge_r: Vec<f64> = Vec::new();
    let mut edge_z: Vec<f64> = Vec::new();
    for i_vertex in 0..n_vertex {
        if is_chosen[i_vertex] {
            edge_r.push(outline.r[i_vertex]);
            edge_z.push(outline.z[i_vertex]);
        }
    }
    for x_point in x_points {
        edge_r.push(x_point.r);
        edge_z.push(x_point.z);
    }
    Points {
        r: Array1::from(edge_r),
        z: Array1::from(edge_z),
    }
}

/// A grid over the model box, `R_geo - a_minor <= R <= R_geo + a_minor`, `z_bottom <= Z <= z_top`, and its points inside the LCFS:
/// `(1 - psi_N) > 0` and connected to the magnetic axis.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - the sum of the orders, not yet divided by its value on the magnetic axis [dimensionless]
/// * `value_at_axis` - its value on the magnetic axis, by which it is divided [dimensionless]
/// * `axis` - the magnetic axis [metre]
/// * `z_bottom` - bottom of the box [metre]
/// * `z_top` - top of the box [metre]
/// * `n_grid` - grid points per direction [dimensionless]
#[allow(clippy::too_many_arguments)]
fn interior_grid(lattice: &Lattice, series: &LatticeSeries, value_at_axis: f64, axis: Point, z_bottom: f64, z_top: f64, n_grid: usize) -> InteriorGrid {
    let radial: &RadialVariable = &lattice.radial;
    let r_inner: f64 = (radial.r_c_squared - radial.r_squared_half_width).sqrt();
    let r_outer: f64 = (radial.r_c_squared + radial.r_squared_half_width).sqrt();
    let grid_r_1d: Array1<f64> = Array1::linspace(r_inner, r_outer, n_grid);
    let grid_z_1d: Array1<f64> = Array1::linspace(z_bottom, z_top, n_grid);
    let mut grid_r: Vec<f64> = Vec::with_capacity(n_grid * n_grid);
    let mut grid_z: Vec<f64> = Vec::with_capacity(n_grid * n_grid);
    for i_grid_r in 0..n_grid {
        for i_grid_z in 0..n_grid {
            grid_r.push(grid_r_1d[i_grid_r]);
            grid_z.push(grid_z_1d[i_grid_z]);
        }
    }
    let flux: FluxValues = series_evaluate(lattice, series, &grid_r, &grid_z).divided_by(value_at_axis);
    let mut positive: Array2<bool> = Array2::from_elem((n_grid, n_grid), false);
    for i_grid_r in 0..n_grid {
        for i_grid_z in 0..n_grid {
            positive[[i_grid_r, i_grid_z]] = flux.one_minus_psi_n[i_grid_r * n_grid + i_grid_z] > 0.0;
        }
    }
    let i_axis_r: usize = nearest_index(&grid_r_1d, axis.r);
    let i_axis_z: usize = nearest_index(&grid_z_1d, axis.z);
    let region: Array2<bool> = if positive[[i_axis_r, i_axis_z]] {
        connected_region(&positive, (i_axis_r, i_axis_z))
    } else {
        positive.clone()
    };
    let mut inside: Vec<bool> = vec![false; n_grid * n_grid];
    for i_grid_r in 0..n_grid {
        for i_grid_z in 0..n_grid {
            inside[i_grid_r * n_grid + i_grid_z] = region[[i_grid_r, i_grid_z]];
        }
    }
    InteriorGrid { r: grid_r, flux, inside }
}

/// The paper's Grad-Shafranov residual, the largest difference between the two sides of the Grad-Shafranov equation in `(1 - psi_N)`
/// divided by the largest value of its right-hand side, `k_max**2 [(R**2 / R_c**2) P + Q]`, over grid points inside the LCFS and, with
/// `edge_rays`, over points along rays from the magnetic axis towards their targets, which resolve the LCFS: `EDGE_FRACTIONS` of the way
/// to each target (as far as `(1 - psi_N) > 0`), or,
/// with `to_own_lcfs`, of the way to the ray's first crossing of the solution's own LCFS, wherever that is; with the analytic second
/// derivatives.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - the sum of the orders, not yet divided by its value on the magnetic axis [dimensionless]
/// * `value_at_axis` - its value on the magnetic axis [dimensionless]
/// * `axis` - the magnetic axis [metre]
/// * `z_bottom` - bottom of the box searched [metre]
/// * `z_top` - top of the box searched [metre]
/// * `k_max` - the paper's `k_max` [1 / metre]
/// * `shapes` - the profiles `P` and `Q`
/// * `n_grid` - grid points per direction [dimensionless]
/// * `edge_rays` - the rays, or `None` for the grid only
///
/// # Returns
/// * the relative residual, NaN if `k_max` is not finite [dimensionless]
#[allow(clippy::too_many_arguments)]
pub fn gs_residual_ratio(
    lattice: &Lattice,
    series: &LatticeSeries,
    value_at_axis: f64,
    axis: Point,
    z_bottom: f64,
    z_top: f64,
    k_max: f64,
    shapes: &ProfileShapes,
    n_grid: usize,
    edge_rays: Option<&EdgeRays>,
) -> f64 {
    if !k_max.is_finite() {
        return f64::NAN;
    }
    let r_c_squared: f64 = lattice.radial.r_c_squared;
    let grid: InteriorGrid = interior_grid(lattice, series, value_at_axis, axis, z_bottom, z_top, n_grid);
    let n_point: usize = grid.r.len();
    let mut residual_max: f64 = 0.0;
    let mut source_max: f64 = 0.0;
    for i_point in 0..n_point {
        if !grid.inside[i_point] {
            continue;
        }
        let residual: f64 = grad_shafranov_residual(
            grid.r[i_point],
            grid.flux.one_minus_psi_n[i_point],
            grid.flux.d_d_r[i_point],
            grid.flux.d2_d_r2[i_point],
            grid.flux.d2_d_z2[i_point],
            k_max,
            r_c_squared,
            shapes,
        );
        // NaN propagates: a residual that cannot be evaluated is not a small one
        if residual.is_nan() || residual.abs() > residual_max {
            residual_max = residual.abs();
        }
        let source: f64 = k_max * k_max * source_term(grid.r[i_point], grid.flux.one_minus_psi_n[i_point], r_c_squared, shapes);
        if source.is_nan() || source.abs() > source_max {
            source_max = source.abs();
        }
    }
    if let Some(rays) = edge_rays {
        let n_ray: usize = rays.targets.r.len();
        if n_ray > 0 {
            let n_fraction: usize = EDGE_FRACTIONS.len();
            // The length of each ray, as a fraction of the way to its target: 1, or its first crossing of the solution's own LCFS
            let ray_lengths: Array1<f64> = if rays.to_own_lcfs {
                own_lcfs_crossings(lattice, series, axis, rays.targets, rays.r_minimum, rays.r_maximum)
            } else {
                Array1::ones(n_ray)
            };
            let mut ray_r: Vec<f64> = Vec::with_capacity(n_ray * n_fraction);
            let mut ray_z: Vec<f64> = Vec::with_capacity(n_ray * n_fraction);
            for i_ray in 0..n_ray {
                for i_fraction in 0..n_fraction {
                    let fraction: f64 = ray_lengths[i_ray] * EDGE_FRACTIONS[i_fraction];
                    ray_r.push(axis.r + (rays.targets.r[i_ray] - axis.r) * fraction);
                    ray_z.push(axis.z + (rays.targets.z[i_ray] - axis.z) * fraction);
                }
            }
            let flux_ray: FluxValues = series_evaluate(lattice, series, &ray_r, &ray_z);
            for i_ray in 0..n_ray {
                // Inside the solution's own LCFS: (1 - psi_N) > 0 from the axis out (a ray's points beyond its first (1 - psi_N) <= 0 are
                // dropped)
                'fraction_loop: for i_fraction in 0..n_fraction {
                    let i_point: usize = i_ray * n_fraction + i_fraction;
                    let value_point: f64 = flux_ray.one_minus_psi_n[i_point] / value_at_axis;
                    if !(value_point > 0.0) {
                        break 'fraction_loop;
                    }
                    let residual: f64 = grad_shafranov_residual(
                        ray_r[i_point],
                        value_point,
                        flux_ray.d_d_r[i_point] / value_at_axis,
                        flux_ray.d2_d_r2[i_point] / value_at_axis,
                        flux_ray.d2_d_z2[i_point] / value_at_axis,
                        k_max,
                        r_c_squared,
                        shapes,
                    );
                    if residual.is_nan() || residual.abs() > residual_max {
                        residual_max = residual.abs();
                    }
                }
            }
        }
    }
    residual_max / source_max
}

/// Where rays from the magnetic axis towards points of the given LCFS first cross the solution's own LCFS, `(1 - psi_N) = 0`: sampled at
/// `N_CROSSING_SEARCH` fractions of the way to each point, from `CROSSING_SEARCH_START` to `CROSSING_SEARCH_END` or to where the ray
/// leaves `r_minimum <= R <= r_maximum` (the radial series' valid range, beyond which it is noise), then the first bracket bisected
/// `N_CROSSING_BISECTION` times. `(1 - psi_N)` that is not positive (or not finite) counts as outside. Nested surfaces have
/// `(1 - psi_N)` falling along the ray, so a ray also ends at its last sample before it stops falling: one through or beside an X-point
/// would otherwise go on into the private flux region, where `(1 - psi_N) > 0` again.
///
/// # Returns
/// * each crossing as a fraction of the way to the target [dimensionless]
fn own_lcfs_crossings(lattice: &Lattice, series: &LatticeSeries, axis: Point, targets: &Points, r_minimum: f64, r_maximum: f64) -> Array1<f64> {
    let n_ray: usize = targets.r.len();
    // Each ray's last fraction: CROSSING_SEARCH_END, or just short of where it leaves the valid range
    let mut fraction_ends: Array1<f64> = Array1::from_elem(n_ray, CROSSING_SEARCH_END);
    for i_ray in 0..n_ray {
        let step_r: f64 = targets.r[i_ray] - axis.r;
        if step_r < 0.0 && r_minimum.is_finite() {
            fraction_ends[i_ray] = CROSSING_SEARCH_END.min(0.999 * (r_minimum - axis.r) / step_r);
        } else if step_r > 0.0 && r_maximum.is_finite() {
            fraction_ends[i_ray] = CROSSING_SEARCH_END.min(0.999 * (r_maximum - axis.r) / step_r);
        }
    }
    let search_steps: Array1<f64> = Array1::linspace(0.0, 1.0, N_CROSSING_SEARCH);
    let mut search_fractions: Array2<f64> = Array2::from_elem((n_ray, N_CROSSING_SEARCH), f64::NAN);
    let mut search_r: Vec<f64> = Vec::with_capacity(n_ray * N_CROSSING_SEARCH);
    let mut search_z: Vec<f64> = Vec::with_capacity(n_ray * N_CROSSING_SEARCH);
    for i_ray in 0..n_ray {
        for i_search in 0..N_CROSSING_SEARCH {
            let fraction: f64 = CROSSING_SEARCH_START + (fraction_ends[i_ray] - CROSSING_SEARCH_START) * search_steps[i_search];
            search_fractions[[i_ray, i_search]] = fraction;
            search_r.push(axis.r + (targets.r[i_ray] - axis.r) * fraction);
            search_z.push(axis.z + (targets.z[i_ray] - axis.z) * fraction);
        }
    }
    let search_values: Array1<f64> = series_values(lattice, series, &search_r, &search_z);
    let mut lower: Array1<f64> = fraction_ends.clone();
    let mut upper: Array1<f64> = fraction_ends.clone();
    for i_ray in 0..n_ray {
        if !(search_values[i_ray * N_CROSSING_SEARCH] > 0.0) {
            lower[i_ray] = search_fractions[[i_ray, 0]];
            upper[i_ray] = search_fractions[[i_ray, 0]];
            continue;
        }
        'search_loop: for i_search in 1..N_CROSSING_SEARCH {
            let value_here: f64 = search_values[i_ray * N_CROSSING_SEARCH + i_search];
            if !(value_here > 0.0) {
                lower[i_ray] = search_fractions[[i_ray, i_search - 1]];
                upper[i_ray] = search_fractions[[i_ray, i_search]];
                break 'search_loop;
            }
            if value_here >= search_values[i_ray * N_CROSSING_SEARCH + i_search - 1] {
                // (1 - psi_N) stopped falling: past an X-point, or beside one; the ray ends at the sample before (nothing to bisect)
                lower[i_ray] = search_fractions[[i_ray, i_search - 1]];
                upper[i_ray] = search_fractions[[i_ray, i_search - 1]];
                break 'search_loop;
            }
        }
    }
    for _i_bisection in 0..N_CROSSING_BISECTION {
        let middle: Array1<f64> = (&lower + &upper) * 0.5;
        let mut middle_r: Vec<f64> = Vec::with_capacity(n_ray);
        let mut middle_z: Vec<f64> = Vec::with_capacity(n_ray);
        for i_ray in 0..n_ray {
            middle_r.push(axis.r + (targets.r[i_ray] - axis.r) * middle[i_ray]);
            middle_z.push(axis.z + (targets.z[i_ray] - axis.z) * middle[i_ray]);
        }
        let middle_values: Array1<f64> = series_values(lattice, series, &middle_r, &middle_z);
        for i_ray in 0..n_ray {
            if middle_values[i_ray] > 0.0 {
                lower[i_ray] = middle[i_ray];
            } else {
                upper[i_ray] = middle[i_ray];
            }
        }
    }
    lower
}

/// `sum |d_{j,m}|` of every order, a bound on `|(1 - psi_N)_l|` on the plasma [dimensionless]
pub fn coefficient_sizes(one_minus_psi_n_orders: &[LatticeSeries]) -> Array1<f64> {
    let n_taylor: usize = one_minus_psi_n_orders.len();
    let mut sizes: Array1<f64> = Array1::from_elem(n_taylor, f64::NAN);
    for i_order in 0..n_taylor {
        sizes[i_order] = series_size(&one_minus_psi_n_orders[i_order]);
    }
    sizes
}

/// The coefficient-growth flag. A well-conditioned series has coefficient sums that fall with order, or grow slowly; an unstable one
/// (for some combinations of shape terms and geometry, about x7 every two orders) has sums that grow geometrically. Flagged
/// when, over the second half of the orders, the sums grow faster than x1.5 per order, or are not finite.
pub fn is_growing(sizes: &Array1<f64>) -> bool {
    let n_size: usize = sizes.len();
    for i_size in 0..n_size {
        if !sizes[i_size].is_finite() {
            return true;
        }
    }
    if n_size < 5 {
        return false;
    }
    let i_middle: usize = n_size / 2;
    let rate: f64 = (sizes[n_size - 1] / sizes[i_middle]).powf(1.0 / ((n_size - 1 - i_middle) as f64));
    rate > 1.5 && sizes[n_size - 1] > 1.0e-13 * sizes[0]
}

/// Is the plasma a closed set of nested flux surfaces? On a grid over a box around the plasma, as far inboard (towards `R = 0`) as the
/// radial series can be evaluated, the region `(1 - psi_N) > 0` connected to the magnetic axis must not reach the edges of the box (the
/// top or bottom edge is an X-point's height, where the separatrix touches it, and is not checked), must hold a single maximum, and
/// `(1 - psi_N)` must fall monotonically along rays from the axis to the LCFS.
///
/// # Arguments
/// * `value_function` - `(1 - psi_N)(R, Z)`, 1 on the axis [dimensionless]
/// * `axis` - the magnetic axis [metre]
/// * `r_low` - inboard edge of the box [metre]
/// * `r_high` - outboard edge of the box [metre]
/// * `z_bottom` - bottom of the plasma; the box reaches 30 % of its half-height further, or the lower X-point [metre]
/// * `z_top` - top of the plasma; likewise [metre]
/// * `x_points` - the X-points, at most one above and one below the axis [metre]
/// * `boundary` - the LCFS, closed, from the rays or lines that traced it [metre]
#[allow(clippy::too_many_arguments)]
pub fn closure_check<F>(value_function: &F, axis: Point, r_low: f64, r_high: f64, z_bottom: f64, z_top: f64, x_points: &[Point], boundary: &Points) -> Closure
where
    F: Fn(&[f64], &[f64]) -> Array1<f64>,
{
    let n_grid: usize = 161;
    let centre_z: f64 = 0.5 * (z_top + z_bottom);
    let half_height: f64 = 0.5 * (z_top - z_bottom);
    let mut z_high: f64 = centre_z + 1.3 * half_height;
    let mut z_low: f64 = centre_z - 1.3 * half_height;
    let mut has_upper_x_point: bool = false;
    let mut has_lower_x_point: bool = false;
    for x_point in x_points {
        if x_point.z > axis.z {
            z_high = x_point.z;
            has_upper_x_point = true;
        } else {
            z_low = x_point.z;
            has_lower_x_point = true;
        }
    }
    let grid_r_1d: Array1<f64> = Array1::linspace(r_low, r_high, n_grid);
    let grid_z_1d: Array1<f64> = Array1::linspace(z_low, z_high, n_grid);
    let mut mesh_r: Vec<f64> = Vec::with_capacity(n_grid * n_grid);
    let mut mesh_z: Vec<f64> = Vec::with_capacity(n_grid * n_grid);
    for i_grid_r in 0..n_grid {
        for i_grid_z in 0..n_grid {
            mesh_r.push(grid_r_1d[i_grid_r]);
            mesh_z.push(grid_z_1d[i_grid_z]);
        }
    }
    let values_flat: Array1<f64> = value_function(&mesh_r, &mesh_z);
    let mut values: Array2<f64> = Array2::from_elem((n_grid, n_grid), f64::NAN);
    let mut positive: Array2<bool> = Array2::from_elem((n_grid, n_grid), false);
    for i_grid_r in 0..n_grid {
        for i_grid_z in 0..n_grid {
            let value: f64 = values_flat[i_grid_r * n_grid + i_grid_z];
            values[[i_grid_r, i_grid_z]] = value;
            positive[[i_grid_r, i_grid_z]] = value.is_finite() && value > 0.0;
        }
    }
    let i_axis_r: usize = nearest_index(&grid_r_1d, axis.r);
    let i_axis_z: usize = nearest_index(&grid_z_1d, axis.z);
    if !positive[[i_axis_r, i_axis_z]] {
        return Closure {
            closed: false,
            reaches_box_edge: true,
            n_maximum: 0,
            monotonic: false,
        };
    }
    let region: Array2<bool> = connected_region(&positive, (i_axis_r, i_axis_z));
    let mut reaches_box_edge: bool = false;
    for i_grid in 0..n_grid {
        reaches_box_edge |= region[[0, i_grid]] || region[[n_grid - 1, i_grid]];
        if !has_lower_x_point {
            reaches_box_edge |= region[[i_grid, 0]];
        }
        if !has_upper_x_point {
            reaches_box_edge |= region[[i_grid, n_grid - 1]];
        }
    }
    // Strict local maxima of (1 - psi_N) inside the region, 3 x 3 neighbourhoods
    let mut n_maximum: usize = 0;
    for i_grid_r in 1..n_grid - 1 {
        for i_grid_z in 1..n_grid - 1 {
            if !region[[i_grid_r, i_grid_z]] {
                continue;
            }
            let mut neighbourhood_max: f64 = f64::NEG_INFINITY;
            for i_offset_r in 0..3 {
                for i_offset_z in 0..3 {
                    let i_neighbour_r: usize = i_grid_r + i_offset_r - 1;
                    let i_neighbour_z: usize = i_grid_z + i_offset_z - 1;
                    if positive[[i_neighbour_r, i_neighbour_z]] {
                        neighbourhood_max = neighbourhood_max.max(values[[i_neighbour_r, i_neighbour_z]]);
                    }
                }
            }
            if values[[i_grid_r, i_grid_z]] >= neighbourhood_max {
                n_maximum += 1;
            }
        }
    }
    // (1 - psi_N) along rays from the axis to the LCFS points
    let n_sample: usize = 60;
    let fraction: Array1<f64> = Array1::linspace(0.0, 1.0, n_sample);
    let n_boundary: usize = boundary.r.len();
    let mut ray_r: Vec<f64> = Vec::with_capacity(n_boundary * n_sample);
    let mut ray_z: Vec<f64> = Vec::with_capacity(n_boundary * n_sample);
    for i_boundary in 0..n_boundary {
        for i_sample in 0..n_sample {
            ray_r.push(axis.r + (boundary.r[i_boundary] - axis.r) * fraction[i_sample]);
            ray_z.push(axis.z + (boundary.z[i_boundary] - axis.z) * fraction[i_sample]);
        }
    }
    let ray_values: Array1<f64> = value_function(&ray_r, &ray_z);
    let mut monotonic: bool = true;
    for i_boundary in 0..n_boundary {
        for i_sample in 0..n_sample - 1 {
            let difference: f64 = ray_values[i_boundary * n_sample + i_sample + 1] - ray_values[i_boundary * n_sample + i_sample];
            monotonic &= difference < 1.0e-9;
        }
    }
    Closure {
        closed: !reaches_box_edge && n_maximum == 1 && monotonic,
        reaches_box_edge,
        n_maximum,
        monotonic,
    }
}
