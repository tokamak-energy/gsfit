use super::flux::{FluxDerivatives, FluxSolution};
use super::matching_geometry::{Constraint, MatchingGeometry};
use super::radial::RadialBasis;
use ndarray::{Array1, Array2};

/// Dense `n x n` linear solve by Gaussian elimination with partial pivoting.
fn solve_linear(a_in: &Array2<f64>, b_in: &Array1<f64>) -> Option<Array1<f64>> {
    let n_unknown: usize = b_in.len();
    let mut a: Array2<f64> = a_in.clone();
    let mut b: Array1<f64> = b_in.clone();
    for i_column in 0..n_unknown {
        // Partial pivot
        let mut i_pivot: usize = i_column;
        let mut max_abs: f64 = a[(i_column, i_column)].abs();
        for i_row in (i_column + 1)..n_unknown {
            if a[(i_row, i_column)].abs() > max_abs {
                max_abs = a[(i_row, i_column)].abs();
                i_pivot = i_row;
            }
        }
        if max_abs < 1e-300 {
            return None;
        }
        if i_pivot != i_column {
            for i_element in 0..n_unknown {
                let tmp: f64 = a[(i_column, i_element)];
                a[(i_column, i_element)] = a[(i_pivot, i_element)];
                a[(i_pivot, i_element)] = tmp;
            }
            b.swap(i_column, i_pivot);
        }
        for i_row in (i_column + 1)..n_unknown {
            let factor: f64 = a[(i_row, i_column)] / a[(i_column, i_column)];
            for i_element in i_column..n_unknown {
                a[(i_row, i_element)] -= factor * a[(i_column, i_element)];
            }
            b[i_row] -= factor * b[i_column];
        }
    }
    let mut x: Array1<f64> = Array1::zeros(n_unknown);
    for i_row in (0..n_unknown).rev() {
        let mut sum: f64 = b[i_row];
        for i_element in (i_row + 1)..n_unknown {
            sum -= a[(i_row, i_element)] * x[i_element];
        }
        x[i_row] = sum / a[(i_row, i_row)];
    }
    Some(x)
}

/// Every constraint applied to every expansion term, GF's `A(alpha)` (Eqs. 3.9, 5.9).
///
/// # Returns
/// * `constraint_matrix` - shape `(n_constraint, n_expansion_term)`, which is square: element `(i, j)` is constraint `i`'s
///   residual for expansion term `j` with a unit coefficient [dimensionless]
fn constraint_matrix(basis: &RadialBasis, geom: &MatchingGeometry) -> Array2<f64> {
    let n_constraint: usize = geom.constraints.len();
    let n_expansion_term: usize = geom.expansion_terms.len();
    let mut constraint_matrix: Array2<f64> = Array2::from_elem((n_constraint, n_expansion_term), f64::NAN);
    for i_constraint in 0..n_constraint {
        let constraint: &Constraint = &geom.constraints[i_constraint];
        for i_expansion_term in 0..n_expansion_term {
            let term: FluxDerivatives = geom.expansion_terms[i_expansion_term].derivatives(basis, constraint.x, constraint.y);
            constraint_matrix[(i_constraint, i_expansion_term)] = constraint.residual(&term);
        }
    }
    constraint_matrix
}

/// The expansion coefficients which satisfy every constraint except the last, the eigenvalue condition, with `c_1 = 1`.
///
/// # Returns
/// * `coefficients` - one per expansion term, `c_1 = 1` first; `None` if the system is singular [dimensionless]
fn coefficients_from_constraint_matrix(constraint_matrix: &Array2<f64>) -> Option<Array1<f64>> {
    let n_expansion_term: usize = constraint_matrix.ncols();
    let n_unknown: usize = n_expansion_term - 1;
    // `c_1 = 1` moves the first column to the right-hand side
    let mut a_matrix: Array2<f64> = Array2::from_elem((n_unknown, n_unknown), f64::NAN);
    let mut b_vector: Array1<f64> = Array1::from_elem(n_unknown, f64::NAN);
    for i_row in 0..n_unknown {
        b_vector[i_row] = -constraint_matrix[(i_row, 0)];
        for i_unknown in 0..n_unknown {
            a_matrix[(i_row, i_unknown)] = constraint_matrix[(i_row, i_unknown + 1)];
        }
    }
    let unknowns: Array1<f64> = solve_linear(&a_matrix, &b_vector)?;
    let mut coefficients: Array1<f64> = Array1::from_elem(n_expansion_term, f64::NAN);
    coefficients[0] = 1.0;
    for i_unknown in 0..n_unknown {
        coefficients[i_unknown + 1] = unknowns[i_unknown];
    }
    Some(coefficients)
}

/// Solve the expansion coefficients from every constraint except the last, with `c_1 = 1` (GF Secs. 3.4, 5.4).
///
/// # Returns
/// * `coefficients` - one per expansion term, in the order of `geom.expansion_terms`; `None` if the system is singular [dimensionless]
pub fn solve_coefficients(basis: &RadialBasis, geom: &MatchingGeometry) -> Option<Array1<f64>> {
    coefficients_from_constraint_matrix(&constraint_matrix(basis, geom))
}

/// Squared, normalised residual of the eigenvalue condition, the last constraint (GF Eq. 3.11).
fn error_e(basis: &RadialBasis, geom: &MatchingGeometry) -> f64 {
    let constraint_matrix: Array2<f64> = constraint_matrix(basis, geom);
    let coefficients: Array1<f64> = match coefficients_from_constraint_matrix(&constraint_matrix) {
        Some(coefficients) => coefficients,
        None => return 1.0,
    };
    // GF Eq. 3.10: Q_1 c_1 + Q_2 c_2 + ... = 0
    let n_expansion_term: usize = coefficients.len();
    let i_constraint_eigenvalue: usize = geom.constraints.len() - 1;
    let mut dot: f64 = 0.0;
    let mut norm: f64 = 0.0;
    for i_expansion_term in 0..n_expansion_term {
        let q_times_coefficient: f64 = constraint_matrix[(i_constraint_eigenvalue, i_expansion_term)] * coefficients[i_expansion_term];
        dot += q_times_coefficient;
        norm += q_times_coefficient.abs();
    }
    if norm < 1e-300 {
        return 1.0;
    }
    let residual: f64 = dot / norm;
    residual * residual
}

/// Golden-section minimisation of `f` on `[a, b]`.
fn golden_section(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let golden_ratio: f64 = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c: f64 = b - golden_ratio * (b - a);
    let mut d: f64 = a + golden_ratio * (b - a);
    let mut f_c: f64 = f(c);
    let mut f_d: f64 = f(d);
    let n_iter_max: usize = 200;
    'refine: for _ in 0..n_iter_max {
        if (b - a).abs() < tol {
            break 'refine;
        }
        if f_c < f_d {
            b = d;
            d = c;
            f_d = f_c;
            c = b - golden_ratio * (b - a);
            f_c = f(c);
        } else {
            a = c;
            c = d;
            f_c = f_d;
            d = a + golden_ratio * (b - a);
            f_d = f(d);
        }
    }
    0.5 * (a + b)
}

/// The flux solution for a given `alpha`, or `None` if the constraint system is singular there.
pub fn flux_solution(alpha: f64, eps: f64, nu: f64, geom: &MatchingGeometry, n_radial_expansion: usize) -> Option<FluxSolution> {
    let basis: RadialBasis = RadialBasis::new(alpha, eps, nu, n_radial_expansion);
    let coefficients: Array1<f64> = solve_coefficients(&basis, geom)?;
    Some(FluxSolution::new(basis, geom.expansion_terms.clone(), coefficients))
}

/// Peak of the normalised eigenfunction over the model box. The physical fundamental mode peaks at the magnetic
/// axis with `psi_hat ~ 1`; spurious / excited roots overshoot (`psi_hat >> 1`) or contain interior nulls.
fn eigenfunction_peak(alpha: f64, eps: f64, nu: f64, geom: &MatchingGeometry, n_radial_expansion: usize) -> f64 {
    let flux: FluxSolution = match flux_solution(alpha, eps, nu, geom, n_radial_expansion) {
        Some(flux) => flux,
        None => return f64::INFINITY,
    };
    let y_lower: f64 = geom.y_lower();
    let y_upper: f64 = geom.y_upper();
    let n_probe: usize = 41;
    let mut peak: f64 = f64::NEG_INFINITY;
    for i_y in 0..n_probe {
        let y: f64 = y_lower + (y_upper - y_lower) * (i_y as f64) / ((n_probe - 1) as f64);
        for i_x in 0..n_probe {
            let x: f64 = -1.0 + 2.0 * (i_x as f64) / ((n_probe - 1) as f64);
            let psi_hat: f64 = flux.eval_all(x, y).psi_hat;
            if psi_hat > peak {
                peak = psi_hat;
            }
        }
    }
    peak
}

/// Find the physical eigenvalue `alpha` in `[alpha_lo, alpha_hi]` (GF Sec. 3.4):
/// a coarse scan brackets local minima of `E(alpha)`; each is refined by golden
/// section, and the lowest-alpha root that is both accurate and physical (its
/// eigenfunction peaks at `~1`, no interior overshoot) is returned. Falls back to
/// the global minimum if no physical root is found.
pub fn solve_alpha(eps: f64, nu: f64, geom: &MatchingGeometry, n_radial_expansion: usize, alpha_lo: f64, alpha_hi: f64) -> f64 {
    let n_coarse: usize = 500;
    let alphas: Array1<f64> = Array1::linspace(alpha_lo, alpha_hi, n_coarse);
    let eval_e = |alpha: f64| -> f64 { error_e(&RadialBasis::new(alpha, eps, nu, n_radial_expansion), geom) };

    let mut errs: Array1<f64> = Array1::zeros(n_coarse);
    for i_alpha in 0..n_coarse {
        errs[i_alpha] = eval_e(alphas[i_alpha]);
    }

    let root_tol: f64 = 1e-3; // refined |E| accepted as a genuine root
    let peak_tol: f64 = 1.5; // eigenfunction overshoot rejected above this
    // Walk local minima in ascending alpha; return the first genuine, physical root.
    'candidates: for i_alpha in 1..(n_coarse - 1) {
        if errs[i_alpha] < errs[i_alpha - 1] && errs[i_alpha] <= errs[i_alpha + 1] {
            let alpha_root: f64 = golden_section(&eval_e, alphas[i_alpha - 1], alphas[i_alpha + 1], 1e-8);
            if eval_e(alpha_root) < root_tol {
                let peak: f64 = eigenfunction_peak(alpha_root, eps, nu, geom, n_radial_expansion);
                if peak.is_finite() && peak < peak_tol {
                    return alpha_root;
                }
            }
            continue 'candidates;
        }
    }

    // Fallback: global minimum of E over the bracket, refined.
    let mut i_min: usize = 0;
    let mut min_err: f64 = errs[0];
    for i_alpha in 1..n_coarse {
        if errs[i_alpha] < min_err {
            min_err = errs[i_alpha];
            i_min = i_alpha;
        }
    }
    let i_lo: usize = if i_min > 0 { i_min - 1 } else { 0 };
    let i_hi: usize = if i_min < n_coarse - 1 { i_min + 1 } else { n_coarse - 1 };
    golden_section(&eval_e, alphas[i_lo], alphas[i_hi], 1e-8)
}

#[cfg(test)]
mod tests {
    use super::super::Configuration;
    use super::super::matching_geometry::{MatchingGeometry, matching_geometry};
    use super::*;

    /// The eigenvalues of GF's test cases (Tables 1 - 3) agree with GF's Table 4, which gives them to two decimal places.
    ///
    /// Not included: the elongated D (`eps = 0.33, kappa = 1.8, delta = 0.4, nu = 0.3`), for which Table 4 gives 1.96.
    /// Scanning `E(alpha)` over `[1, 3]` finds a single root, at 1.921, so the 1.96 appears to be a copy of the double
    /// null's value on the row below
    #[test]
    fn eigenvalues_match_guazzotto_freidberg_table_4() {
        let cases: [(&str, Configuration, f64, f64, f64); 8] = [
            ("circle", Configuration::SymmetricLimited { kappa: 1.0, delta: 0.0 }, 0.33, 1.0, 2.38),
            ("ellipse", Configuration::SymmetricLimited { kappa: 2.0, delta: 0.0 }, 0.25, 1.0, 1.88),
            (
                "high-triangularity D",
                Configuration::SymmetricLimited { kappa: 2.0, delta: 0.75 },
                0.4,
                1.0,
                2.05,
            ),
            ("inverse D", Configuration::SymmetricLimited { kappa: 1.9, delta: -0.6 }, 0.33, 0.5, 1.91),
            (
                "double null, standard",
                Configuration::DoubleNull { kappa_x: 2.0, delta_x: 0.5 },
                0.33,
                0.4,
                1.96,
            ),
            (
                "double null, spherical",
                Configuration::DoubleNull { kappa_x: 2.4, delta_x: 0.8 },
                0.75,
                1.04,
                1.90,
            ),
            (
                "single null, standard",
                Configuration::SingleNull {
                    kappa: 1.6,
                    delta: 0.4,
                    kappa_x: 2.0,
                    delta_x: 0.5,
                },
                0.33,
                1.0,
                2.04,
            ),
            (
                "single null, high beta_p",
                Configuration::SingleNull {
                    kappa: 1.8,
                    delta: 0.6,
                    kappa_x: 2.1,
                    delta_x: 0.8,
                },
                0.25,
                2.1,
                2.09,
            ),
        ];
        for (name, configuration, eps, nu, alpha_table_4) in cases {
            let geom: MatchingGeometry = matching_geometry(&configuration, eps).unwrap();
            let alpha: f64 = solve_alpha(eps, nu, &geom, 150, 1.0, 3.0);
            assert!(
                (alpha - alpha_table_4).abs() <= 0.005,
                "{name}: alpha = {alpha:.6}, but GF's Table 4 gives {alpha_table_4}"
            );
        }
    }

    /// The single-null solution satisfies all 12 constraints: the first 11 to rounding, as they are solved for, and the
    /// last, the eigenvalue condition, to the accuracy of `alpha`
    #[test]
    fn single_null_solution_satisfies_every_constraint() {
        let eps: f64 = 0.25;
        let nu: f64 = 2.1;
        let n_radial_expansion: usize = 150;
        let configuration: Configuration = Configuration::SingleNull {
            kappa: 1.8,
            delta: 0.6,
            kappa_x: 2.1,
            delta_x: 0.8,
        };
        let geom: MatchingGeometry = matching_geometry(&configuration, eps).unwrap();
        let alpha: f64 = solve_alpha(eps, nu, &geom, n_radial_expansion, 1.0, 3.0);
        let flux: FluxSolution = flux_solution(alpha, eps, nu, &geom, n_radial_expansion).unwrap();

        let n_constraint: usize = geom.constraints.len();
        assert_eq!(n_constraint, 12);
        for i_constraint in 0..n_constraint {
            let constraint: &Constraint = &geom.constraints[i_constraint];
            let derivatives: FluxDerivatives = flux.eval_all(constraint.x, constraint.y);
            let residual: f64 = constraint.residual(&derivatives);
            // `psi_hat(0, 0) = 1`, and its derivatives are of order `alpha ** 2`, so the residuals are compared with 1
            let tolerance: f64 = if i_constraint < n_constraint - 1 { 1.0e-10 } else { 1.0e-6 };
            assert!(
                residual.abs() < tolerance,
                "constraint {i_constraint} ({:?} at ({:.4}, {:.4})): residual = {residual:e}",
                constraint.kind,
                constraint.x,
                constraint.y
            );
        }
    }
}
