use super::Symmetry;
use super::radial::RadialBasis;
use ndarray::Array1;

/// The normalised flux `psi_hat(x, y)` and its first and second derivatives in the normalised
/// coordinates `x`, `y` [dimensionless]
pub struct FluxDerivatives {
    pub psi_hat: f64,
    pub d_psi_hat_d_x: f64,
    pub d_psi_hat_d_y: f64,
    pub d2_psi_hat_d_x2: f64,
    pub d2_psi_hat_d_x_d_y: f64,
    pub d2_psi_hat_d_y2: f64,
}

/// The `y` dependence of an expansion term, `Y_n(y)` (GF Eq. 2.14)
#[derive(Clone, Copy, Debug)]
pub enum VerticalFunction {
    /// `cos(h_n y)`, even in `y`
    Cos,
    /// `sin(h_n y)`, odd in `y`
    Sin,
}

/// The `x` dependence of an expansion term, `X_n(x)` (GF Eq. 2.15)
#[derive(Clone, Copy, Debug)]
pub enum RadialFunction {
    /// The cosine-like solution, `C_n(x)`
    C,
    /// The sine-like solution, `S_n(x)`
    S,
}

/// One term of the flux expansion, `Y_n(y) X_n(x)`, whose coefficient is one of GF's unknowns `u`
#[derive(Clone, Copy, Debug)]
pub struct ExpansionTerm {
    /// Which separation constants, `h_n` and `k_n`, the term uses: `n = 1, ..., 4`, 1-indexed as in GF [dimensionless]
    pub mode: usize,
    pub vertical_function: VerticalFunction,
    pub radial_function: RadialFunction,
}

impl ExpansionTerm {
    /// The term's value and derivatives at `(x, y)`, with a unit coefficient.
    pub fn derivatives(&self, basis: &RadialBasis, x: f64, y: f64) -> FluxDerivatives {
        let h_n: f64 = basis.h(self.mode);
        let (radial, d_radial_d_x, d2_radial_d_x2): (f64, f64, f64) = match self.radial_function {
            RadialFunction::C => basis.c_all(x, self.mode),
            RadialFunction::S => basis.s_all(x, self.mode),
        };
        let cos_hy: f64 = (h_n * y).cos();
        let sin_hy: f64 = (h_n * y).sin();
        let (vertical, d_vertical_d_y, d2_vertical_d_y2): (f64, f64, f64) = match self.vertical_function {
            VerticalFunction::Cos => (cos_hy, -h_n * sin_hy, -h_n * h_n * cos_hy),
            VerticalFunction::Sin => (sin_hy, h_n * cos_hy, -h_n * h_n * sin_hy),
        };
        FluxDerivatives {
            psi_hat: vertical * radial,
            d_psi_hat_d_x: vertical * d_radial_d_x,
            d_psi_hat_d_y: d_vertical_d_y * radial,
            d2_psi_hat_d_x2: vertical * d2_radial_d_x2,
            d2_psi_hat_d_x_d_y: d_vertical_d_y * d_radial_d_x,
            d2_psi_hat_d_y2: d2_vertical_d_y2 * radial,
        }
    }
}

/// The terms of the flux expansion, in the order of GF's unknowns `u`.
///
/// The first term is `cos(h_1 y) C_1(x)`, whose coefficient `c_1 = 1` sets the scale of the solution.
///
/// * `Symmetric` - GF Eq. 3.2, 7 terms: `u = [c_1, c_2, s_2, c_3, s_3, c_4, s_4]`
/// * `Asymmetric` - GF Eq. 5.8, 12 terms: the 7 above, then the `sin(h_n y)` terms `[c_5, c_6, s_6, c_7, s_7]`
///
/// `S_1(x) = 0` because `k_1 = 0`, and `sin(h_4 y) = 0` because `h_4 = 0`, so neither has a term.
pub fn expansion_terms(symmetry: Symmetry) -> Vec<ExpansionTerm> {
    let term = |mode: usize, vertical_function: VerticalFunction, radial_function: RadialFunction| -> ExpansionTerm {
        ExpansionTerm {
            mode,
            vertical_function,
            radial_function,
        }
    };
    let mut terms: Vec<ExpansionTerm> = vec![
        term(1, VerticalFunction::Cos, RadialFunction::C),
        term(2, VerticalFunction::Cos, RadialFunction::C),
        term(2, VerticalFunction::Cos, RadialFunction::S),
        term(3, VerticalFunction::Cos, RadialFunction::C),
        term(3, VerticalFunction::Cos, RadialFunction::S),
        term(4, VerticalFunction::Cos, RadialFunction::C),
        term(4, VerticalFunction::Cos, RadialFunction::S),
    ];
    if symmetry == Symmetry::Asymmetric {
        terms.push(term(1, VerticalFunction::Sin, RadialFunction::C));
        terms.push(term(2, VerticalFunction::Sin, RadialFunction::C));
        terms.push(term(2, VerticalFunction::Sin, RadialFunction::S));
        terms.push(term(3, VerticalFunction::Sin, RadialFunction::C));
        terms.push(term(3, VerticalFunction::Sin, RadialFunction::S));
    }
    terms
}

/// Normalised flux `psi_hat(x, y)` and its derivatives, assembled from the solved expansion
/// coefficients (GF Eqs. 3.2, 5.8). Normalised so that `psi_hat(0, 0) = 1`.
pub struct FluxSolution {
    basis: RadialBasis,
    expansion_terms: Vec<ExpansionTerm>,
    /// One coefficient per expansion term, `c_1 = 1` first [dimensionless]
    coefficients: Array1<f64>,
    psi_hat_00: f64,
}

impl FluxSolution {
    pub fn new(basis: RadialBasis, expansion_terms: Vec<ExpansionTerm>, coefficients: Array1<f64>) -> Self {
        assert!(
            coefficients.len() == expansion_terms.len(),
            "FluxSolution.new: {} coefficients for {} expansion terms",
            coefficients.len(),
            expansion_terms.len()
        );
        let mut solution: FluxSolution = FluxSolution {
            basis,
            expansion_terms,
            coefficients,
            psi_hat_00: 1.0,
        };
        let flux_00: FluxDerivatives = solution.eval_all_raw(0.0, 0.0);
        solution.psi_hat_00 = flux_00.psi_hat;
        solution
    }

    /// The flux and its derivatives before the `psi_hat(0, 0) = 1` normalisation.
    fn eval_all_raw(&self, x: f64, y: f64) -> FluxDerivatives {
        let mut flux: FluxDerivatives = FluxDerivatives {
            psi_hat: 0.0,
            d_psi_hat_d_x: 0.0,
            d_psi_hat_d_y: 0.0,
            d2_psi_hat_d_x2: 0.0,
            d2_psi_hat_d_x_d_y: 0.0,
            d2_psi_hat_d_y2: 0.0,
        };
        let n_expansion_term: usize = self.expansion_terms.len();
        for i_expansion_term in 0..n_expansion_term {
            let term: FluxDerivatives = self.expansion_terms[i_expansion_term].derivatives(&self.basis, x, y);
            let coefficient: f64 = self.coefficients[i_expansion_term];
            flux.psi_hat += coefficient * term.psi_hat;
            flux.d_psi_hat_d_x += coefficient * term.d_psi_hat_d_x;
            flux.d_psi_hat_d_y += coefficient * term.d_psi_hat_d_y;
            flux.d2_psi_hat_d_x2 += coefficient * term.d2_psi_hat_d_x2;
            flux.d2_psi_hat_d_x_d_y += coefficient * term.d2_psi_hat_d_x_d_y;
            flux.d2_psi_hat_d_y2 += coefficient * term.d2_psi_hat_d_y2;
        }
        flux
    }

    /// The flux and its derivatives, normalised so that `psi_hat(0, 0) = 1`.
    pub fn eval_all(&self, x: f64, y: f64) -> FluxDerivatives {
        let flux: FluxDerivatives = self.eval_all_raw(x, y);
        let inv: f64 = 1.0 / self.psi_hat_00;
        FluxDerivatives {
            psi_hat: flux.psi_hat * inv,
            d_psi_hat_d_x: flux.d_psi_hat_d_x * inv,
            d_psi_hat_d_y: flux.d_psi_hat_d_y * inv,
            d2_psi_hat_d_x2: flux.d2_psi_hat_d_x2 * inv,
            d2_psi_hat_d_x_d_y: flux.d2_psi_hat_d_x_d_y * inv,
            d2_psi_hat_d_y2: flux.d2_psi_hat_d_y2 * inv,
        }
    }
}
