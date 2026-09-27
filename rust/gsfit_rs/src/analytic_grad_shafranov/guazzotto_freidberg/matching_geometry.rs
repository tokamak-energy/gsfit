use super::Configuration;
use super::flux::{ExpansionTerm, FluxDerivatives, expansion_terms};
use super::model_surface::{MillerHalf, ModelSurfaceHalf, TwoEllipsesHalf};

/// The expression a matching constraint sets to zero, in terms of the flux and its derivatives at the matching point
#[derive(Clone, Copy, Debug)]
pub enum ConstraintKind {
    /// `psi`: the point is on the plasma boundary
    Psi,
    /// `d psi / d x`: the boundary is horizontal there (`B_Z = 0` at an X-point)
    DPsiDX,
    /// `d psi / d y`: the boundary is vertical there (`B_R = 0` at an X-point)
    DPsiDY,
    /// `d2 psi / d y2 + lambda_1 * d psi / d x`: the curvature at the inner midplane (GF Eqs. 3.8e, 4.6e, 5.5j)
    InnerMidplaneCurvature { lambda_1: f64 },
    /// `d2 psi / d y2 - lambda_2 * d psi / d x`: the curvature at the outer midplane (GF Eqs. 3.8f, 4.6f, 5.5k)
    OuterMidplaneCurvature { lambda_2: f64 },
    /// `d2 psi / d x2 - lambda_3 * d psi / d y`: the curvature at the upper smooth maximum (GF Eqs. 3.8g, 5.5l)
    UpperMaximumCurvature { lambda_3: f64 },
}

/// One of GF's matching constraints: a linear condition on the flux at the matching point `(x, y)`
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    /// [dimensionless]
    pub x: f64,
    /// [dimensionless]
    pub y: f64,
    pub kind: ConstraintKind,
}

impl Constraint {
    fn new(x: f64, y: f64, kind: ConstraintKind) -> Self {
        Constraint { x, y, kind }
    }

    /// The expression the constraint sets to zero, for a flux with these derivatives at the matching point
    pub fn residual(&self, flux: &FluxDerivatives) -> f64 {
        match self.kind {
            ConstraintKind::Psi => flux.psi_hat,
            ConstraintKind::DPsiDX => flux.d_psi_hat_d_x,
            ConstraintKind::DPsiDY => flux.d_psi_hat_d_y,
            ConstraintKind::InnerMidplaneCurvature { lambda_1 } => flux.d2_psi_hat_d_y2 + lambda_1 * flux.d_psi_hat_d_x,
            ConstraintKind::OuterMidplaneCurvature { lambda_2 } => flux.d2_psi_hat_d_y2 - lambda_2 * flux.d_psi_hat_d_x,
            ConstraintKind::UpperMaximumCurvature { lambda_3 } => flux.d2_psi_hat_d_x2 - lambda_3 * flux.d_psi_hat_d_y,
        }
    }
}

/// Everything the solve needs from the boundary configuration: the flux expansion, the matching constraints,
/// and the model surface they come from.
pub struct MatchingGeometry {
    /// The terms of the flux expansion, 7 for an up-down symmetric configuration and 12 for an asymmetric one
    pub expansion_terms: Vec<ExpansionTerm>,
    /// One constraint per expansion term, in GF's order. The last is the eigenvalue condition, which is only
    /// satisfied for the right `alpha`; the others determine the expansion coefficients
    pub constraints: Vec<Constraint>,
    /// Upper half of the model surface
    pub upper_half: ModelSurfaceHalf,
    /// Lower half of the model surface, as its mirror image in `y = 0`
    pub lower_half: ModelSurfaceHalf,
}

impl MatchingGeometry {
    /// The highest `y` on the model surface [dimensionless]
    pub fn y_upper(&self) -> f64 {
        self.upper_half.height()
    }

    /// The lowest `y` on the model surface [dimensionless]
    pub fn y_lower(&self) -> f64 {
        -self.lower_half.height()
    }

    /// Whether `(x, y)` is inside the model box, `|x| < 1` and `y_lower < y < y_upper`, the rectangle which
    /// bounds the model surface. The truncated flux series is only reliable inside it.
    pub fn is_in_model_box(&self, x: f64, y: f64) -> bool {
        x.abs() < 1.0 && y > self.y_lower() && y < self.y_upper()
    }

    /// The X-points of the model surface, which the analytic flux also has, ordered upper before lower.
    ///
    /// # Returns
    /// * `(xpt_x, xpt_y)` - [dimensionless]
    pub fn xpts(&self) -> (Vec<f64>, Vec<f64>) {
        let mut xpt_x: Vec<f64> = Vec::new();
        let mut xpt_y: Vec<f64> = Vec::new();
        if self.upper_half.has_x_point() {
            xpt_x.push(self.upper_half.vertex_x());
            xpt_y.push(self.y_upper());
        }
        if self.lower_half.has_x_point() {
            xpt_x.push(self.lower_half.vertex_x());
            xpt_y.push(self.y_lower());
        }
        (xpt_x, xpt_y)
    }
}

/// Build the matching geometry for a boundary configuration.
///
/// References are to L. Guazzotto & J. P. Freidberg, J. Plasma Phys. 87 (2021) 905870303.
///
/// # Arguments
/// * `configuration` - the boundary topology and its shape parameters
/// * `eps` - inverse aspect ratio [dimensionless]
pub fn matching_geometry(configuration: &Configuration, eps: f64) -> Result<MatchingGeometry, String> {
    let constraints: Vec<Constraint>;
    let upper_half: ModelSurfaceHalf;
    let lower_half: ModelSurfaceHalf;
    match *configuration {
        Configuration::SymmetricLimited { kappa, delta } => {
            // GF Eqs. 3.5, 3.8
            let miller: MillerHalf = MillerHalf::new(eps, kappa, delta)?;
            let x_delta: f64 = miller.x_delta;
            let lambda_1: f64 = miller.lambda_1;
            let lambda_2: f64 = miller.lambda_2;
            let lambda_3: f64 = miller.lambda_3;
            constraints = vec![
                // (a) - (c) on the boundary: inner midplane, outer midplane, top
                Constraint::new(-1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(-x_delta, kappa, ConstraintKind::Psi),
                // (d) the top is a maximum
                Constraint::new(-x_delta, kappa, ConstraintKind::DPsiDX),
                // (e), (f) the midplane curvatures
                Constraint::new(-1.0, 0.0, ConstraintKind::InnerMidplaneCurvature { lambda_1 }),
                Constraint::new(1.0, 0.0, ConstraintKind::OuterMidplaneCurvature { lambda_2 }),
                // (g) the curvature at the top, the eigenvalue condition
                Constraint::new(-x_delta, kappa, ConstraintKind::UpperMaximumCurvature { lambda_3 }),
            ];
            upper_half = ModelSurfaceHalf::Miller(miller);
            lower_half = ModelSurfaceHalf::Miller(miller);
        }
        Configuration::DoubleNull { kappa_x, delta_x } => {
            // GF Eq. 4.6
            let two_ellipses: TwoEllipsesHalf = TwoEllipsesHalf::new(eps, kappa_x, delta_x)?;
            let x_x: f64 = two_ellipses.x_x;
            let lambda_1: f64 = two_ellipses.lambda_1;
            let lambda_2: f64 = two_ellipses.lambda_2;
            constraints = vec![
                // (a) - (c) on the boundary: inner midplane, outer midplane, upper X-point
                Constraint::new(-1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(-x_x, kappa_x, ConstraintKind::Psi),
                // (d) B_Z = 0 at the X-point
                Constraint::new(-x_x, kappa_x, ConstraintKind::DPsiDX),
                // (e), (f) the midplane curvatures
                Constraint::new(-1.0, 0.0, ConstraintKind::InnerMidplaneCurvature { lambda_1 }),
                Constraint::new(1.0, 0.0, ConstraintKind::OuterMidplaneCurvature { lambda_2 }),
                // (g) B_R = 0 at the X-point, the eigenvalue condition
                Constraint::new(-x_x, kappa_x, ConstraintKind::DPsiDY),
            ];
            upper_half = ModelSurfaceHalf::TwoEllipses(two_ellipses);
            lower_half = ModelSurfaceHalf::TwoEllipses(two_ellipses);
        }
        Configuration::SingleNull {
            kappa,
            delta,
            kappa_x,
            delta_x,
        } => {
            // GF Eqs. 5.3 - 5.6: a Miller upper half and a two-ellipse lower half, with the X-point below
            let miller: MillerHalf = MillerHalf::new(eps, kappa, delta)?;
            let two_ellipses: TwoEllipsesHalf = TwoEllipsesHalf::new(eps, kappa_x, delta_x)?;
            let x_delta: f64 = miller.x_delta;
            let x_x: f64 = two_ellipses.x_x;
            // GF Eq. 5.6: each midplane curvature is the average of the two halves'
            let lambda_1: f64 = 0.5 * (miller.lambda_1 + two_ellipses.lambda_1);
            let lambda_2: f64 = 0.5 * (miller.lambda_2 + two_ellipses.lambda_2);
            let lambda_3: f64 = miller.lambda_3;
            constraints = vec![
                // (a) - (d) on the boundary: inner midplane, outer midplane, upper maximum, lower X-point
                Constraint::new(-1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(1.0, 0.0, ConstraintKind::Psi),
                Constraint::new(-x_delta, kappa, ConstraintKind::Psi),
                Constraint::new(-x_x, -kappa_x, ConstraintKind::Psi),
                // (e), (f) the boundary is vertical at the midplanes; no longer automatic without up-down symmetry
                Constraint::new(-1.0, 0.0, ConstraintKind::DPsiDY),
                Constraint::new(1.0, 0.0, ConstraintKind::DPsiDY),
                // (g) the upper maximum
                Constraint::new(-x_delta, kappa, ConstraintKind::DPsiDX),
                // (h), (i) B_Z = 0 and B_R = 0 at the X-point. GF's Eq. 5.4 prints these at (-delta_X, -kappa_X), but
                // they must act at the same point as (d), (-x_X, -kappa_X), for the X-point to lie on the boundary;
                // with (-x_X, -kappa_X) the eigenvalues match GF's Table 4
                Constraint::new(-x_x, -kappa_x, ConstraintKind::DPsiDX),
                Constraint::new(-x_x, -kappa_x, ConstraintKind::DPsiDY),
                // (j), (k) the midplane curvatures
                Constraint::new(-1.0, 0.0, ConstraintKind::InnerMidplaneCurvature { lambda_1 }),
                Constraint::new(1.0, 0.0, ConstraintKind::OuterMidplaneCurvature { lambda_2 }),
                // (l) the curvature at the upper maximum, the eigenvalue condition
                Constraint::new(-x_delta, kappa, ConstraintKind::UpperMaximumCurvature { lambda_3 }),
            ];
            upper_half = ModelSurfaceHalf::Miller(miller);
            lower_half = ModelSurfaceHalf::TwoEllipses(two_ellipses);
        }
        Configuration::AntisymmetricLimited { .. } => {
            return Err("GuazzottoFreidberg: AntisymmetricLimited solve is not yet implemented".to_string());
        }
    }

    let expansion_terms: Vec<ExpansionTerm> = expansion_terms(configuration.symmetry());
    assert!(
        constraints.len() == expansion_terms.len(),
        "matching_geometry: {} constraints for {} expansion terms",
        constraints.len(),
        expansion_terms.len()
    );
    Ok(MatchingGeometry {
        expansion_terms,
        constraints,
        upper_half,
        lower_half,
    })
}
