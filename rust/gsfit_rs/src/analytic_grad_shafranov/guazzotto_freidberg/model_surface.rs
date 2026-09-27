use ndarray::Array1;
use std::f64::consts::PI;

/// Upper half of a Miller profile: a smooth maximum (GF Eqs. 3.4, 3.6, 3.8).
///
/// Everything is in the normalised coordinates `x`, `y`. The lower half is the mirror image in `y = 0`.
///
/// References are to L. Guazzotto & J. P. Freidberg, J. Plasma Phys. 87 (2021) 905870303.
#[derive(Clone, Copy, Debug)]
pub struct MillerHalf {
    /// Inverse aspect ratio [dimensionless]
    eps: f64,
    /// Elongation [dimensionless]
    kappa: f64,
    /// `asin(delta)`, where `delta` is the triangularity [dimensionless]
    delta_hat: f64,
    /// The vertex, the smooth maximum, is at `(-x_delta, kappa)` (GF Eq. 3.5) [dimensionless]
    pub x_delta: f64,
    /// Curvature constant at the inner midplane (GF Eq. 3.8e) [dimensionless]
    pub lambda_1: f64,
    /// Curvature constant at the outer midplane (GF Eq. 3.8f) [dimensionless]
    pub lambda_2: f64,
    /// Curvature constant at the vertex (GF Eq. 3.8g) [dimensionless]
    pub lambda_3: f64,
}

impl MillerHalf {
    /// # Arguments
    /// * `eps` - inverse aspect ratio [dimensionless]
    /// * `kappa` - elongation [dimensionless]
    /// * `delta` - triangularity, `-1 < delta < 1` [dimensionless]
    pub fn new(eps: f64, kappa: f64, delta: f64) -> Result<Self, String> {
        if kappa <= 0.0 || delta.abs() >= 1.0 {
            return Err(format!(
                "GuazzottoFreidberg: a Miller profile needs kappa > 0 and |delta| < 1, got kappa = {kappa}, delta = {delta}"
            ));
        }
        let delta_hat: f64 = delta.asin();
        Ok(MillerHalf {
            eps,
            kappa,
            delta_hat,
            x_delta: delta + 0.5 * eps * (1.0 - delta * delta),
            lambda_1: (1.0 - eps) * (1.0 - delta_hat).powi(2) / (kappa * kappa),
            lambda_2: (1.0 + eps) * (1.0 + delta_hat).powi(2) / (kappa * kappa),
            lambda_3: kappa / ((1.0 - eps * delta).powi(2) * (1.0 - delta * delta)),
        })
    }

    /// The point at `theta` (GF Eq. 3.4): `theta` in `[0, pi]` is the upper half, and `[pi, 2 pi]` the lower.
    fn point(&self, theta: f64) -> (f64, f64) {
        let angle: f64 = theta + self.delta_hat * theta.sin();
        let x: f64 = angle.cos() - 0.5 * self.eps * angle.sin().powi(2);
        let y: f64 = self.kappa * theta.sin();
        (x, y)
    }
}

/// Upper half of a two-ellipse surface: two ellipses intersecting at right angles at an X-point
/// (GF Eqs. 4.1 - 4.3, 4.7).
///
/// Everything is in the normalised coordinates `x`, `y`. The lower half is the mirror image in `y = 0`.
#[derive(Clone, Copy, Debug)]
pub struct TwoEllipsesHalf {
    /// Inverse aspect ratio [dimensionless]
    eps: f64,
    /// X-point elongation [dimensionless]
    kappa_x: f64,
    /// Elongation of both ellipses, `kappa_0` (GF Eq. 4.3) [dimensionless]
    kappa_0: f64,
    /// Shift of the inboard ellipse, `x_1` (GF Eq. 4.3) [dimensionless]
    x_1: f64,
    /// Shift of the outboard ellipse, `x_2` (GF Eq. 4.3) [dimensionless]
    x_2: f64,
    /// Where the ellipses intersect, `theta_0` (GF Eq. 4.3) [radian]
    theta_0: f64,
    /// The vertex, the X-point, is at `(-x_x, kappa_x)` (GF Eq. 4.5) [dimensionless]
    pub x_x: f64,
    /// Curvature constant at the inner midplane (GF Eq. 4.7) [dimensionless]
    pub lambda_1: f64,
    /// Curvature constant at the outer midplane (GF Eq. 4.7) [dimensionless]
    pub lambda_2: f64,
}

impl TwoEllipsesHalf {
    /// # Arguments
    /// * `eps` - inverse aspect ratio [dimensionless]
    /// * `kappa_x` - X-point elongation, `kappa_x > 2 * sqrt(1 - delta_x ** 2)` (GF Eq. 4.4) [dimensionless]
    /// * `delta_x` - X-point triangularity, `-1 < delta_x < 1` [dimensionless]
    pub fn new(eps: f64, kappa_x: f64, delta_x: f64) -> Result<Self, String> {
        if delta_x.abs() >= 1.0 {
            return Err(format!(
                "GuazzottoFreidberg: a two-ellipse surface needs |delta_x| < 1, got delta_x = {delta_x}"
            ));
        }
        let root: f64 = (1.0 - delta_x * delta_x).sqrt();
        // GF Eq. 4.4: the ellipses can only meet at right angles when 0 < xi < 1
        if kappa_x <= 2.0 * root {
            return Err(format!(
                "GuazzottoFreidberg: a two-ellipse surface needs kappa_x > 2 * sqrt(1 - delta_x ** 2) = {:.6} (GF Eq. 4.4), got kappa_x = {kappa_x}",
                2.0 * root
            ));
        }
        let xi: f64 = root / (kappa_x - root);
        Ok(TwoEllipsesHalf {
            eps,
            kappa_x,
            kappa_0: kappa_x / (1.0 - xi * xi).sqrt(),
            x_1: (xi - delta_x) / (1.0 - xi),
            x_2: (xi + delta_x) / (1.0 - xi),
            // GF write `atan(sqrt(1 - xi ** 2) / xi)`, which is `acos(xi)` for 0 < xi < 1
            theta_0: xi.acos(),
            x_x: delta_x + 0.5 * eps * (1.0 - delta_x * delta_x),
            lambda_1: (1.0 - eps) * (1.0 - delta_x) * (1.0 + xi) / (kappa_x * kappa_x),
            lambda_2: (1.0 + eps) * (1.0 + delta_x) * (1.0 + xi) / (kappa_x * kappa_x),
        })
    }

    /// The point at `theta` on the inboard ellipse (GF Eq. 4.1), or on the outboard ellipse (GF Eq. 4.2).
    fn point(&self, theta: f64, is_inboard: bool) -> (f64, f64) {
        let rho: f64 = if is_inboard {
            self.x_1 + (1.0 + self.x_1) * theta.cos()
        } else {
            -self.x_2 + (1.0 + self.x_2) * theta.cos()
        };
        (normalised_x(self.eps, rho), self.kappa_0 * theta.sin())
    }
}

/// One half, upper or lower, of GF's model surface.
///
/// The model surface only sets the position, slope and curvature at the matching points: the analytic
/// plasma boundary passes through those points, but elsewhere it is close to the model surface rather
/// than on it.
#[derive(Clone, Copy, Debug)]
pub enum ModelSurfaceHalf {
    /// A smooth maximum
    Miller(MillerHalf),
    /// An X-point
    TwoEllipses(TwoEllipsesHalf),
}

impl ModelSurfaceHalf {
    /// The largest `|y|` on the half, `kappa` or `kappa_x`, reached at the vertex [dimensionless]
    pub fn height(&self) -> f64 {
        match self {
            ModelSurfaceHalf::Miller(miller) => miller.kappa,
            ModelSurfaceHalf::TwoEllipses(two_ellipses) => two_ellipses.kappa_x,
        }
    }

    /// `x` of the vertex: `-x_delta` or `-x_x` [dimensionless]
    pub fn vertex_x(&self) -> f64 {
        match self {
            ModelSurfaceHalf::Miller(miller) => -miller.x_delta,
            ModelSurfaceHalf::TwoEllipses(two_ellipses) => -two_ellipses.x_x,
        }
    }

    /// Whether the vertex is an X-point
    pub fn has_x_point(&self) -> bool {
        matches!(self, ModelSurfaceHalf::TwoEllipses(_))
    }

    /// Points along the half, anticlockwise in `(x, y)`.
    ///
    /// The upper half runs from the outer midplane, included, over the top to the inner midplane, excluded.
    /// The lower half runs from the inner midplane, included, under the bottom to the outer midplane, excluded.
    /// So the upper half followed by the lower half is the whole surface, with no point repeated. The vertex
    /// is always one of the points.
    ///
    /// # Arguments
    /// * `is_upper` - the upper half, or its mirror image the lower half
    /// * `n_point` - number of points; must be even, so a Miller vertex falls on a point
    ///
    /// # Returns
    /// * `(x, y)` - each of length `n_point` [dimensionless]
    pub fn outline(&self, is_upper: bool, n_point: usize) -> (Array1<f64>, Array1<f64>) {
        assert!(
            n_point >= 2 && n_point.is_multiple_of(2),
            "ModelSurfaceHalf.outline: n_point = {n_point} must be even and at least 2"
        );
        let mut x: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut y: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        match self {
            ModelSurfaceHalf::Miller(miller) => {
                let theta_start: f64 = if is_upper { 0.0 } else { PI };
                for i_point in 0..n_point {
                    let theta: f64 = theta_start + PI * (i_point as f64) / (n_point as f64);
                    (x[i_point], y[i_point]) = miller.point(theta);
                }
            }
            ModelSurfaceHalf::TwoEllipses(two_ellipses) => {
                // Half the points on each ellipse. Each ellipse's range is half open, and the X-point is the start
                // of the second ellipse's range
                let n_point_per_ellipse: usize = n_point / 2;
                let theta_0: f64 = two_ellipses.theta_0;
                for i_point in 0..n_point_per_ellipse {
                    let fraction: f64 = (i_point as f64) / (n_point_per_ellipse as f64);
                    let i_second: usize = n_point_per_ellipse + i_point;
                    if is_upper {
                        // Outboard ellipse, theta in [0, theta_0); then inboard ellipse, theta in [pi - theta_0, pi)
                        (x[i_point], y[i_point]) = two_ellipses.point(theta_0 * fraction, false);
                        (x[i_second], y[i_second]) = two_ellipses.point(PI - theta_0 + theta_0 * fraction, true);
                    } else {
                        // Inboard ellipse, theta in [pi, pi + theta_0); then outboard ellipse, theta in [2 pi - theta_0, 2 pi)
                        (x[i_point], y[i_point]) = two_ellipses.point(PI + theta_0 * fraction, true);
                        (x[i_second], y[i_second]) = two_ellipses.point(2.0 * PI - theta_0 + theta_0 * fraction, false);
                    }
                }
            }
        }
        (x, y)
    }
}

/// GF's normalised `x` of a point at `R = r_geo * (1 + eps * rho)`: `x = rho + (eps / 2) * (rho ** 2 - 1)`, from
/// `R ** 2 = r_geo ** 2 * (1 + eps ** 2 + 2 * eps * x)` (GF Eq. 2.8).
fn normalised_x(eps: f64, rho: f64) -> f64 {
    rho + 0.5 * eps * (rho * rho - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shortest distance from `(point_x, point_y)` to the points of an outline
    fn distance_to_outline(point_x: f64, point_y: f64, outline_x: &Array1<f64>, outline_y: &Array1<f64>) -> f64 {
        let n_point: usize = outline_x.len();
        let mut distance_min: f64 = f64::INFINITY;
        for i_point in 0..n_point {
            distance_min = distance_min.min((outline_x[i_point] - point_x).hypot(outline_y[i_point] - point_y));
        }
        distance_min
    }

    /// Each half passes through its matching points: the outer midplane `(1, 0)`, the inner midplane `(-1, 0)`, and
    /// its vertex, which is the highest point on the half
    #[test]
    fn each_half_passes_through_its_matching_points() {
        let eps: f64 = 0.25;
        let halves: [ModelSurfaceHalf; 2] = [
            ModelSurfaceHalf::Miller(MillerHalf::new(eps, 1.8, 0.6).unwrap()),
            ModelSurfaceHalf::TwoEllipses(TwoEllipsesHalf::new(eps, 2.1, 0.8).unwrap()),
        ];
        let n_point: usize = 400;
        for half in halves {
            for is_upper in [true, false] {
                let (outline_x, outline_y): (Array1<f64>, Array1<f64>) = half.outline(is_upper, n_point);
                let vertex_y: f64 = if is_upper { half.height() } else { -half.height() };
                // The midplane point each half starts at: outer for the upper half, inner for the lower half
                let midplane_x: f64 = if is_upper { 1.0 } else { -1.0 };
                assert!(distance_to_outline(midplane_x, 0.0, &outline_x, &outline_y) < 1.0e-14);
                assert!(distance_to_outline(half.vertex_x(), vertex_y, &outline_x, &outline_y) < 1.0e-14);
                let mut y_abs_max: f64 = 0.0;
                for i_point in 0..n_point {
                    y_abs_max = y_abs_max.max(outline_y[i_point].abs());
                }
                assert!((y_abs_max - half.height()).abs() < 1.0e-14, "the vertex is not the highest point on the half");
            }
        }
    }

    /// The two ellipses intersect at right angles at the X-point (GF Sec. 4.1). `R` and `Z` are both linear in the
    /// ellipse parameters, so the angle is checked with the tangents `(d R / d theta, d Z / d theta) / a`
    #[test]
    fn two_ellipses_intersect_at_right_angles() {
        for (kappa_x, delta_x) in [(2.1, 0.8), (2.4, 0.8), (2.0, 0.5), (3.0, -0.3)] {
            let two_ellipses: TwoEllipsesHalf = TwoEllipsesHalf::new(0.3, kappa_x, delta_x).unwrap();
            let theta_0: f64 = two_ellipses.theta_0;
            let theta_inboard: f64 = PI - theta_0;
            let tangent_inboard: (f64, f64) = (-(1.0 + two_ellipses.x_1) * theta_inboard.sin(), two_ellipses.kappa_0 * theta_inboard.cos());
            let tangent_outboard: (f64, f64) = (-(1.0 + two_ellipses.x_2) * theta_0.sin(), two_ellipses.kappa_0 * theta_0.cos());
            let cos_angle: f64 = (tangent_inboard.0 * tangent_outboard.0 + tangent_inboard.1 * tangent_outboard.1)
                / (tangent_inboard.0.hypot(tangent_inboard.1) * tangent_outboard.0.hypot(tangent_outboard.1));
            assert!(
                cos_angle.abs() < 1.0e-14,
                "kappa_x = {kappa_x}, delta_x = {delta_x}: cos(angle) = {cos_angle:e}"
            );
            // Both ellipses reach the X-point
            let (inboard_x, inboard_y): (f64, f64) = two_ellipses.point(theta_inboard, true);
            let (outboard_x, outboard_y): (f64, f64) = two_ellipses.point(theta_0, false);
            assert!((inboard_x - outboard_x).abs() < 1.0e-14 && (inboard_y - outboard_y).abs() < 1.0e-14);
            assert!((inboard_x + two_ellipses.x_x).abs() < 1.0e-14 && (inboard_y - kappa_x).abs() < 1.0e-14);
        }
    }

    #[test]
    fn two_ellipses_reject_kappa_x_below_the_limit() {
        // GF Eq. 4.4: kappa_x > 2 * sqrt(1 - 0.6 ** 2) = 1.6
        assert!(TwoEllipsesHalf::new(0.3, 1.59, 0.6).is_err());
        assert!(TwoEllipsesHalf::new(0.3, 1.61, 0.6).is_ok());
    }
}
