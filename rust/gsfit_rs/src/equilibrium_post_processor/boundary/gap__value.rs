//! `time_slice(itime)/boundary/gap(i1)/value`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use crate::plasma_geometry::bicubic_interpolator::{BicubicInterpolator, BicubicValueAndDerivatives};
use geo::{Contains, Coord, LineString, Point, Polygon};
use imas_rs::EquilibriumTimeSlice;
use ndarray::{Array1, Array2, s};

/// How far either side of the traced boundary's crossing the exact crossing is searched for, as a
/// fraction of the smaller grid spacing.
///
/// The traced boundary is within a fraction of a millimetre of the exact one on ST40's 12.5 mm
/// grid, so half a cell is generous. It is kept this small on purpose: near an X-point the divertor
/// legs sit at the same `psi` as the boundary, and a wider search could land on a leg instead
const SEARCH_HALF_WIDTH_IN_CELLS: f64 = 0.5;

/// The exact crossing is converged once a step moves it by less than this, [metre]
const CROSSING_TOLERANCE: f64 = 1e-10;

/// The most Newton-bisection iterations the exact crossing is allowed. Bisection alone halves the
/// search window every iteration, so this is far more than is ever needed
const N_ITER_MAX: usize = 100;

/// Calculate the value of every gap, and store them in the time-slice.
///
/// A gap is a reference point, `gap(i1)/r` and `gap(i1)/z`, and a direction, `gap(i1)/angle`. Its
/// value is the distance from the reference point to the plasma boundary along that direction. The
/// data dictionary measures `angle` clockwise from `grad(R)`, in the usual plot with `R` to the
/// right and `Z` upwards, so the gap points along `(cos(angle), -sin(angle))`.
///
/// The crossing is found in two steps:
/// 1. The gap's line is intersected with `boundary/outline`. This decides *which* crossing is
///    wanted: the traced boundary is the last closed flux surface alone, whereas `psi = psi_b` is
///    also satisfied along the divertor legs of a diverted plasma.
/// 2. That crossing is refined by solving `psi = psi_b` along the line, on the solver's own bicubic
///    model of `psi`. The outline is a polyline whose vertices come from *linear* interpolation
///    along the edges of the grid cells, so it is only accurate to a fraction of a millimetre; the
///    bicubic is built from the solver's analytic derivatives of `psi`. Should the refinement not
///    be possible - the crossing is off the grid, or the line is so nearly tangent to the boundary
///    that there is no change of sign to find - the outline's crossing is kept.
///
/// The value is signed. When the reference point is outside the plasma, which is where gaps are
/// normally defined from, the value is the distance to the nearest crossing ahead of it and is
/// positive. When the plasma has grown past the reference point, the value is the distance to the
/// nearest crossing *behind* it along the gap direction, and is negative by that much.
///
/// # Arguments
/// * `time_slice` - the solved time-slice; `boundary/gap(i1)/value` is written into it
///
/// A gap whose line does not reach the plasma, which includes one pointing away from it, is NaN.
/// A time-slice with no gaps has nothing to calculate, and is left alone.
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // Most runs have no gaps: they only come from a pulse schedule, which is optional. So this is
    // tested before anything else is read, and in particular before `profiles_2d[0]`
    let n_gaps: usize = time_slice.boundary.gap.len();
    if n_gaps == 0 {
        return;
    }

    let mut gap_values: Array1<f64> = Array1::from_elem(n_gaps, f64::NAN);

    let outline_r: &Array1<f64> = &time_slice.boundary.outline.r;
    let outline_z: &Array1<f64> = &time_slice.boundary.outline.z;
    let n_outline: usize = outline_r.len();

    // Three points is the fewest which enclose anything. A time-slice whose boundary could not be
    // traced has none, and so leaves every gap NaN
    if n_outline >= 3 && outline_z.len() == n_outline {
        let psi_b: f64 = time_slice.boundary.psi;

        // `profiles_2d[0]` because GSFit solves on a single rectangular (R, Z) grid, so there is only
        // ever one entry in this array of structures
        let flux_grid: FluxGrid = FluxGrid {
            r: &time_slice.profiles_2d[0].grid.dim1,
            z: &time_slice.profiles_2d[0].grid.dim2,
            psi_2d: &time_slice.profiles_2d[0].psi,
            d_psi_d_r_2d: &time_slice.profiles_2d[0].d_psi_d_r,
            d_psi_d_z_2d: &time_slice.profiles_2d[0].d_psi_d_z,
            d2_psi_d_r_d_z_2d: &time_slice.profiles_2d[0].d2_psi_d_r_d_z,
        };

        let mut outline_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_outline);
        for i_outline in 0..n_outline {
            outline_coordinates.push(Coord {
                x: outline_r[i_outline],
                y: outline_z[i_outline],
            });
        }
        let outline_polygon: Polygon = Polygon::new(
            LineString::from(outline_coordinates),
            vec![], // No holes
        );

        for i_gap in 0..n_gaps {
            let gap_r: f64 = time_slice.boundary.gap[i_gap].r;
            let gap_z: f64 = time_slice.boundary.gap[i_gap].z;
            let gap_angle: f64 = time_slice.boundary.gap[i_gap].angle;
            if !gap_r.is_finite() || !gap_z.is_finite() || !gap_angle.is_finite() {
                continue;
            }

            // Clockwise from `grad(R)`, so a positive angle turns the direction towards `-Z`
            let gap_direction_r: f64 = gap_angle.cos();
            let gap_direction_z: f64 = -gap_angle.sin();

            let reference_point_inside: bool = outline_polygon.contains(&Point::new(gap_r, gap_z));

            let s_outline: f64 = crossing_on_outline(outline_r, outline_z, gap_r, gap_z, gap_direction_r, gap_direction_z, reference_point_inside);
            if s_outline.is_nan() {
                continue;
            }

            gap_values[i_gap] = refine_crossing(&flux_grid, psi_b, gap_r, gap_z, gap_direction_r, gap_direction_z, s_outline);
        }
    }

    for i_gap in 0..n_gaps {
        time_slice.boundary.gap[i_gap].value = gap_values[i_gap];
    }
}

/// The signed distance along a gap's line to where it crosses the traced boundary.
///
/// The line is `(gap_r, gap_z) + s * (gap_direction_r, gap_direction_z)`, so with a unit direction
/// `s` is a distance, [metre]. The outline is treated as closed, joining its last point back to its
/// first, whether or not it repeats the first point; a repeated point only adds a segment of zero
/// length, which crosses nothing.
///
/// # Arguments
/// * `outline_r`, `outline_z` - the traced boundary, [metre]
/// * `gap_r`, `gap_z` - the gap's reference point, [metre]
/// * `gap_direction_r`, `gap_direction_z` - the unit vector the gap is measured along, [dimensionless]
/// * `reference_point_inside` - whether the reference point is inside the boundary
///
/// # Returns
/// * `s_crossing` - from outside, the nearest crossing ahead (`s >= 0`); from inside, the nearest
///   crossing behind (`s <= 0`). NaN when there is no such crossing, [metre]
fn crossing_on_outline(
    outline_r: &Array1<f64>,
    outline_z: &Array1<f64>,
    gap_r: f64,
    gap_z: f64,
    gap_direction_r: f64,
    gap_direction_z: f64,
    reference_point_inside: bool,
) -> f64 {
    let n_outline: usize = outline_r.len();

    let mut s_crossing: f64 = f64::NAN;
    for i_outline in 0..n_outline {
        let i_outline_next: usize = (i_outline + 1) % n_outline;

        let segment_delta_r: f64 = outline_r[i_outline_next] - outline_r[i_outline];
        let segment_delta_z: f64 = outline_z[i_outline_next] - outline_z[i_outline];

        // Solve `gap + s * direction = start + u * segment` for `s` and `u`, by taking the 2D cross
        // product of both sides with `segment` and then with `direction`
        let denominator: f64 = gap_direction_r * segment_delta_z - gap_direction_z * segment_delta_r;
        if denominator == 0.0 {
            // The segment is parallel to the gap, or has no length, so it cannot be crossed
            continue;
        }

        let start_delta_r: f64 = outline_r[i_outline] - gap_r;
        let start_delta_z: f64 = outline_z[i_outline] - gap_z;
        let s_here: f64 = (start_delta_r * segment_delta_z - start_delta_z * segment_delta_r) / denominator;
        let u_here: f64 = (start_delta_r * gap_direction_z - start_delta_z * gap_direction_r) / denominator;

        // The line crosses this segment's own extent, not its extension
        if !(0.0..=1.0).contains(&u_here) {
            continue;
        }

        let nearer: bool = if reference_point_inside {
            s_here <= 0.0 && (s_crossing.is_nan() || s_here > s_crossing)
        } else {
            s_here >= 0.0 && (s_crossing.is_nan() || s_here < s_crossing)
        };
        if nearer {
            s_crossing = s_here;
        }
    }

    s_crossing
}

/// Refine a crossing of the traced boundary onto the exact crossing of `psi = psi_b`.
///
/// `psi - psi_b` is bracketed within `SEARCH_HALF_WIDTH_IN_CELLS` of the outline's crossing, and its
/// root is found by Newton's method along the line, falling back to bisection whenever a Newton step
/// would leave the bracket. The bracket only ever shrinks, so the answer cannot wander off to a
/// different crossing further along the line.
///
/// # Arguments
/// * `flux_grid` - the solved `psi` and its derivatives
/// * `psi_b` - `psi` at the plasma boundary, [weber]
/// * `gap_r`, `gap_z` - the gap's reference point, [metre]
/// * `gap_direction_r`, `gap_direction_z` - the unit vector the gap is measured along, [dimensionless]
/// * `s_outline` - the crossing of the traced boundary, as a signed distance along the line, [metre]
///
/// # Returns
/// * `s_crossing` - the refined crossing, [metre]. `s_outline` itself when the crossing cannot be
///   bracketed: it is off the grid, or there is no change of sign of `psi - psi_b` across the
///   search window
fn refine_crossing(flux_grid: &FluxGrid, psi_b: f64, gap_r: f64, gap_z: f64, gap_direction_r: f64, gap_direction_z: f64, s_outline: f64) -> f64 {
    // `psi - psi_b`, and its derivative with respect to `s`, at a distance `s` along the line
    let residual_and_derivative = |s_here: f64| -> Option<(f64, f64)> {
        let (psi_here, d_psi_d_r_here, d_psi_d_z_here): (f64, f64, f64) =
            flux_grid.psi_and_gradient_at(gap_r + s_here * gap_direction_r, gap_z + s_here * gap_direction_z)?;
        let residual: f64 = psi_here - psi_b;
        let d_residual_d_s: f64 = d_psi_d_r_here * gap_direction_r + d_psi_d_z_here * gap_direction_z;
        if !residual.is_finite() || !d_residual_d_s.is_finite() {
            return None;
        }
        Some((residual, d_residual_d_s))
    };

    let search_half_width: f64 = SEARCH_HALF_WIDTH_IN_CELLS * flux_grid.d_r().min(flux_grid.d_z());
    let mut s_lower: f64 = s_outline - search_half_width;
    let mut s_upper: f64 = s_outline + search_half_width;

    let Some((residual_lower, _)) = residual_and_derivative(s_lower) else {
        return s_outline;
    };
    let Some((residual_upper, _)) = residual_and_derivative(s_upper) else {
        return s_outline;
    };
    let lower_is_positive: bool = residual_lower > 0.0;
    if lower_is_positive == (residual_upper > 0.0) {
        return s_outline;
    }

    let mut s_crossing: f64 = s_outline;
    'newton_loop: for _i_iter in 0..N_ITER_MAX {
        let Some((residual_here, d_residual_d_s_here)) = residual_and_derivative(s_crossing) else {
            return s_outline;
        };
        if residual_here == 0.0 {
            break 'newton_loop;
        }

        // Keep the half of the bracket which still holds the change of sign
        if (residual_here > 0.0) == lower_is_positive {
            s_lower = s_crossing;
        } else {
            s_upper = s_crossing;
        }

        let s_newton: f64 = s_crossing - residual_here / d_residual_d_s_here;
        let s_next: f64 = if s_newton > s_lower && s_newton < s_upper {
            s_newton
        } else {
            0.5 * (s_lower + s_upper)
        };

        let converged: bool = (s_next - s_crossing).abs() < CROSSING_TOLERANCE;
        s_crossing = s_next;
        if converged {
            break 'newton_loop;
        }
    }

    s_crossing
}

/// The solved poloidal flux on the grid, with the derivatives a bicubic model of a grid cell is built
/// from.
struct FluxGrid<'a> {
    /// R grid points, [metre]
    r: &'a Array1<f64>,
    /// Z grid points, [metre]
    z: &'a Array1<f64>,
    /// shape = (n_z, n_r), [weber]
    psi_2d: &'a Array2<f64>,
    /// shape = (n_z, n_r), [weber / metre]
    d_psi_d_r_2d: &'a Array2<f64>,
    /// shape = (n_z, n_r), [weber / metre]
    d_psi_d_z_2d: &'a Array2<f64>,
    /// shape = (n_z, n_r), [weber / metre ** 2]
    d2_psi_d_r_d_z_2d: &'a Array2<f64>,
}

impl FluxGrid<'_> {
    /// Radial grid spacing, [metre]
    fn d_r(&self) -> f64 {
        self.r[1] - self.r[0]
    }

    /// Vertical grid spacing, [metre]
    fn d_z(&self) -> f64 {
        self.z[1] - self.z[0]
    }

    /// `psi` and its gradient at a point, from the bicubic model of the grid cell containing it.
    ///
    /// # Arguments
    /// * `r_point`, `z_point` - where to evaluate, [metre]
    ///
    /// # Returns
    /// * `psi` - [weber]
    /// * `d_psi_d_r` - [weber / metre]
    /// * `d_psi_d_z` - [weber / metre]
    ///
    /// `None` when the point is off the grid, or is NaN.
    fn psi_and_gradient_at(&self, r_point: f64, z_point: f64) -> Option<(f64, f64, f64)> {
        let n_r: usize = self.r.len();
        let n_z: usize = self.z.len();
        if n_r < 2 || n_z < 2 {
            return None;
        }

        // Written so that a NaN point fails the test too. It has to, because the cell indices below
        // come from casting to `usize`, which would saturate a NaN or negative index to the first cell
        let on_grid: bool = r_point >= self.r[0] && r_point <= self.r[n_r - 1] && z_point >= self.z[0] && z_point <= self.z[n_z - 1];
        if !on_grid {
            return None;
        }

        let d_r: f64 = self.d_r();
        let d_z: f64 = self.d_z();

        // The cell containing the point, clamped so that a point on the last grid line uses the last cell
        let i_r_left: usize = (((r_point - self.r[0]) / d_r).floor() as usize).min(n_r - 2);
        let i_z_lower: usize = (((z_point - self.z[0]) / d_z).floor() as usize).min(n_z - 2);

        let psi_interpolator: BicubicInterpolator = BicubicInterpolator::new(
            d_r,
            d_z,
            self.psi_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d_psi_d_r_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d_psi_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
            self.d2_psi_d_r_d_z_2d.slice(s![i_z_lower..=i_z_lower + 1, i_r_left..=i_r_left + 1]),
        );

        let x: f64 = (r_point - self.r[i_r_left]) / d_r;
        let y: f64 = (z_point - self.z[i_z_lower]) / d_z;

        // `value_and_derivatives` differentiates with respect to the normalised cell coordinates
        let psi_value_and_derivatives: BicubicValueAndDerivatives = psi_interpolator.value_and_derivatives(x, y);

        Some((
            psi_value_and_derivatives.f,
            psi_value_and_derivatives.d_f_d_x / d_r,
            psi_value_and_derivatives.d_f_d_y / d_z,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use approx::assert_abs_diff_eq;
    use imas_rs::EquilibriumGap;
    use std::f64::consts::PI;

    /// Centre of the fixture's circular plasma, [metre]
    const CENTRE_R: f64 = 0.6;
    const CENTRE_Z: f64 = 0.0;
    /// Radius of the fixture's circular plasma, [metre]
    const MINOR_RADIUS: f64 = 0.3;
    /// Vertices on the fixture's traced boundary
    const N_OUTLINE: usize = 64;

    /// Build the time-slice this calculator is tested against: a circular plasma.
    ///
    /// ```text
    /// psi = (r - 0.6) ** 2 + z ** 2
    /// ```
    ///
    /// on a 0.05 metre grid, with the boundary at `psi_b = 0.09`, a circle of radius 0.3 metre.
    ///
    /// A bicubic reproduces this `psi` exactly, so the refined crossing is exact to rounding. The
    /// traced boundary is a 64-sided polygon inscribed in the circle, with its vertices half a step
    /// off the axes, so that where it crosses `z = 0` or `r = 0.6` it is a chord and not a vertex.
    /// A chord there sits `0.3 * (1 - cos(pi / 64))`, 0.36 mm, inside the circle: that is the error
    /// the refinement has to remove, and is far larger than the tolerance the tests use.
    fn time_slice_for_test(gaps: Vec<(f64, f64, f64)>) -> EquilibriumTimeSlice {
        use imas_rs::EquilibriumProfiles2d;

        let r: Array1<f64> = Array1::linspace(0.1, 1.2, 23);
        let z: Array1<f64> = Array1::linspace(-0.6, 0.6, 25);
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        let mut psi_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let d2_psi_d_r_d_z_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let delta_r: f64 = r[i_r] - CENTRE_R;
                let delta_z: f64 = z[i_z] - CENTRE_Z;
                psi_2d[(i_z, i_r)] = delta_r.powi(2) + delta_z.powi(2);
                d_psi_d_r_2d[(i_z, i_r)] = 2.0 * delta_r;
                d_psi_d_z_2d[(i_z, i_r)] = 2.0 * delta_z;
            }
        }

        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d {
            psi: psi_2d,
            d_psi_d_r: d_psi_d_r_2d,
            d_psi_d_z: d_psi_d_z_2d,
            d2_psi_d_r_d_z: d2_psi_d_r_d_z_2d,
            ..EquilibriumProfiles2d::default()
        };
        profiles_2d.grid.dim1 = r;
        profiles_2d.grid.dim2 = z;

        // Closed, repeating the first vertex at the end, as a traced contour is
        let mut outline_r: Array1<f64> = Array1::from_elem(N_OUTLINE + 1, f64::NAN);
        let mut outline_z: Array1<f64> = Array1::from_elem(N_OUTLINE + 1, f64::NAN);
        for i_outline in 0..=N_OUTLINE {
            let theta: f64 = 2.0 * PI * (i_outline as f64 + 0.5) / (N_OUTLINE as f64);
            outline_r[i_outline] = CENTRE_R + MINOR_RADIUS * theta.cos();
            outline_z[i_outline] = CENTRE_Z + MINOR_RADIUS * theta.sin();
        }

        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice {
            profiles_2d: vec![profiles_2d],
            ..EquilibriumTimeSlice::default()
        };
        time_slice.boundary.psi = MINOR_RADIUS.powi(2);
        time_slice.boundary.outline.r = outline_r;
        time_slice.boundary.outline.z = outline_z;

        for (gap_r, gap_z, gap_angle) in gaps {
            time_slice.boundary.gap.push(EquilibriumGap {
                r: gap_r,
                z: gap_z,
                angle: gap_angle,
                ..EquilibriumGap::default()
            });
        }

        time_slice
    }

    #[test]
    fn a_slice_with_no_gaps_is_left_alone() {
        // Deliberately empty: no `profiles_2d` and no boundary, which the calculator must not reach
        // for when there are no gaps to measure
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.boundary.gap.is_empty());
    }

    #[test]
    fn a_gap_is_the_exact_distance_to_the_boundary() {
        // From the inboard side of the circle, looking outwards along `+R`
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![(0.175, 0.0, 0.0)]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The boundary is at `r = 0.3`. The traced boundary alone would give 0.12536, so agreeing to
        // well under a micrometre shows the crossing was refined onto `psi = psi_b`
        assert_abs_diff_eq!(time_slice.boundary.gap[0].value, 0.125, epsilon = 1e-9);
    }

    #[test]
    fn the_angle_is_measured_clockwise_from_grad_r() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![
            // Above the plasma: clockwise by a quarter turn from `+R` is `-Z`, towards the plasma
            (CENTRE_R, 0.55, PI / 2.0),
            // The same point a quarter turn the other way looks upwards, away from the plasma
            (CENTRE_R, 0.55, -PI / 2.0),
            // Below and to the right, looking up and inwards along the diagonal through the centre
            (CENTRE_R + 0.4, CENTRE_Z - 0.4, -3.0 * PI / 4.0),
        ]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.boundary.gap[0].value, 0.55 - MINOR_RADIUS, epsilon = 1e-9);
        assert!(time_slice.boundary.gap[1].value.is_nan());
        assert_abs_diff_eq!(time_slice.boundary.gap[2].value, 0.4 * 2.0_f64.sqrt() - MINOR_RADIUS, epsilon = 1e-9);
    }

    #[test]
    fn a_reference_point_inside_the_plasma_gives_a_negative_gap() {
        // 0.1 metre inside the boundary, looking along `+R` towards the centre
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![(0.4, 0.0, 0.0)]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_abs_diff_eq!(time_slice.boundary.gap[0].value, -0.1, epsilon = 1e-9);
    }

    #[test]
    fn a_gap_which_misses_the_plasma_is_nan() {
        // Along `z = 0.4`, which passes above the circle
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![(0.175, 0.4, 0.0)]);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.boundary.gap[0].value.is_nan());
    }

    #[test]
    fn a_slice_whose_boundary_was_not_traced_gives_nan_gaps() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![(0.175, 0.0, 0.0), (CENTRE_R, 0.55, PI / 2.0)]);
        time_slice.boundary.outline.r = Array1::zeros(0);
        time_slice.boundary.outline.z = Array1::zeros(0);

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert!(time_slice.boundary.gap[0].value.is_nan());
        assert!(time_slice.boundary.gap[1].value.is_nan());
    }

    #[test]
    fn off_the_grid_the_traced_boundary_is_used() {
        let mut time_slice: EquilibriumTimeSlice = time_slice_for_test(vec![(0.175, 0.0, 0.0)]);

        // A grid which stops short of the inboard boundary, so the crossing cannot be refined
        let r: Array1<f64> = time_slice.profiles_2d[0].grid.dim1.slice(s![5..]).to_owned();
        let psi_2d: Array2<f64> = time_slice.profiles_2d[0].psi.slice(s![.., 5..]).to_owned();
        let d_psi_d_r_2d: Array2<f64> = time_slice.profiles_2d[0].d_psi_d_r.slice(s![.., 5..]).to_owned();
        let d_psi_d_z_2d: Array2<f64> = time_slice.profiles_2d[0].d_psi_d_z.slice(s![.., 5..]).to_owned();
        let d2_psi_d_r_d_z_2d: Array2<f64> = time_slice.profiles_2d[0].d2_psi_d_r_d_z.slice(s![.., 5..]).to_owned();
        time_slice.profiles_2d[0].grid.dim1 = r;
        time_slice.profiles_2d[0].psi = psi_2d;
        time_slice.profiles_2d[0].d_psi_d_r = d_psi_d_r_2d;
        time_slice.profiles_2d[0].d_psi_d_z = d_psi_d_z_2d;
        time_slice.profiles_2d[0].d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        // The chord of the traced boundary, 0.36 mm inside the circle
        let chord_r: f64 = CENTRE_R - MINOR_RADIUS * (PI / (N_OUTLINE as f64)).cos();
        assert_abs_diff_eq!(time_slice.boundary.gap[0].value, chord_r - 0.175, epsilon = 1e-12);
    }
}
