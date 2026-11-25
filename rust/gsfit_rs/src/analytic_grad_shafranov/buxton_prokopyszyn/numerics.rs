//! Numerical building blocks: Gauss-Legendre quadrature, Brent's method for a bounded minimum, the linear algebra (SVD,
//! pseudo-inverse) on `ndarray` arrays through `faer`, and small helpers.

use faer::c64;
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use ndarray::{Array1, Array2, ArrayView2};

/// The result of a singular value decomposition, `a = u diag(s) v^T`, singular values largest first.
pub struct SingularValueDecomposition {
    /// Left singular vectors, one per column [dimensionless]
    pub u: Array2<f64>,
    /// Singular values, non-increasing [same unit as the matrix]
    pub s: Array1<f64>,
    /// Right singular vectors, one per column (`v`, not `v^T`) [dimensionless]
    pub v: Array2<f64>,
}

/// Gauss-Legendre nodes and weights on `[-1, 1]`.
pub struct GaussLegendre {
    /// Nodes, ascending [dimensionless]
    pub nodes: Array1<f64>,
    /// Weights, summing to 2 [dimensionless]
    pub weights: Array1<f64>,
}

/// Gauss-Legendre quadrature of `n_node` points on `[-1, 1]`: the roots of `P_n` by Newton's method from Tricomi's estimate, the
/// weights `2 / ((1 - x**2) P_n'(x)**2)`, then symmetrised and scaled to sum to 2.
///
/// # Arguments
/// * `n_node` - number of nodes, at least 1 [dimensionless]
///
/// # Returns
/// * the nodes (ascending) and weights [dimensionless]
pub fn gauss_legendre(n_node: usize) -> GaussLegendre {
    let n_iteration_max: usize = 100;
    let mut nodes: Array1<f64> = Array1::from_elem(n_node, f64::NAN);
    let mut weights: Array1<f64> = Array1::from_elem(n_node, f64::NAN);
    for i_node in 0..n_node {
        // Descending estimate cos(pi (i + 3/4) / (n + 1/2)), stored ascending
        let mut node: f64 = (std::f64::consts::PI * (i_node as f64 + 0.75) / (n_node as f64 + 0.5)).cos();
        'newton_loop: for _i_iteration in 0..n_iteration_max {
            let (legendre, derivative): (f64, f64) = legendre_and_derivative(n_node, node);
            let step: f64 = legendre / derivative;
            node -= step;
            if step.abs() <= 1.0e-15 {
                break 'newton_loop;
            }
        }
        let (_, derivative): (f64, f64) = legendre_and_derivative(n_node, node);
        nodes[n_node - 1 - i_node] = node;
        weights[n_node - 1 - i_node] = 2.0 / ((1.0 - node * node) * derivative * derivative);
    }
    // Symmetrise, and scale the weights to sum to 2
    let mut nodes_symmetric: Array1<f64> = Array1::from_elem(n_node, f64::NAN);
    let mut weights_symmetric: Array1<f64> = Array1::from_elem(n_node, f64::NAN);
    for i_node in 0..n_node {
        nodes_symmetric[i_node] = 0.5 * (nodes[i_node] - nodes[n_node - 1 - i_node]);
        weights_symmetric[i_node] = 0.5 * (weights[i_node] + weights[n_node - 1 - i_node]);
    }
    let weight_sum: f64 = weights_symmetric.sum();
    for i_node in 0..n_node {
        weights_symmetric[i_node] *= 2.0 / weight_sum;
    }
    GaussLegendre {
        nodes: nodes_symmetric,
        weights: weights_symmetric,
    }
}

/// `(P_n(x), P_n'(x))`, the Legendre polynomial of degree `n_degree >= 1` and its derivative, by the three-term recurrence
fn legendre_and_derivative(n_degree: usize, x: f64) -> (f64, f64) {
    if n_degree == 1 {
        return (x, 1.0);
    }
    let mut legendre_previous: f64 = 1.0;
    let mut legendre: f64 = x;
    for i_degree in 2..=n_degree {
        let degree: f64 = i_degree as f64;
        let legendre_next: f64 = ((2.0 * degree - 1.0) * x * legendre - (degree - 1.0) * legendre_previous) / degree;
        legendre_previous = legendre;
        legendre = legendre_next;
    }
    let derivative: f64 = (n_degree as f64) * (x * legendre - legendre_previous) / (x * x - 1.0);
    (legendre, derivative)
}

/// The minimum of `function` on `[bound_low, bound_high]` by Brent's method: golden-section search with parabolic interpolation.
///
/// # Arguments
/// * `function` - the function to minimise
/// * `bound_low` - lower bound [any unit]
/// * `bound_high` - upper bound [same unit]
/// * `x_tolerance` - absolute tolerance on the minimiser, to which `sqrt(eps) |x|` is added [same unit]
/// * `n_function_max` - most function evaluations [dimensionless]
///
/// # Returns
/// * the minimiser [same unit]
pub fn minimize_scalar_bounded<F>(function: F, bound_low: f64, bound_high: f64, x_tolerance: f64, n_function_max: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let sqrt_eps: f64 = 2.2e-16_f64.sqrt();
    let golden_mean: f64 = 0.5 * (3.0 - 5.0_f64.sqrt());
    let mut a: f64 = bound_low;
    let mut b: f64 = bound_high;
    let mut fulc: f64 = a + golden_mean * (b - a);
    let mut nfc: f64 = fulc;
    let mut xf: f64 = fulc;
    let mut rat: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut fx: f64 = function(xf);
    let mut n_function: usize = 1;
    let mut ffulc: f64 = fx;
    let mut fnfc: f64 = fx;
    let mut xm: f64 = 0.5 * (a + b);
    let mut tol1: f64 = sqrt_eps * xf.abs() + x_tolerance / 3.0;
    let mut tol2: f64 = 2.0 * tol1;

    'search_loop: for _i_function in 0..n_function_max {
        if (xf - xm).abs() <= tol2 - 0.5 * (b - a) {
            break 'search_loop;
        }
        let mut golden: bool = true;
        // Check for a parabolic fit
        if e.abs() > tol1 {
            golden = false;
            let mut r: f64 = (xf - nfc) * (fx - ffulc);
            let mut q: f64 = (xf - fulc) * (fx - fnfc);
            let mut p: f64 = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            // Is the parabola acceptable?
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                let x_parabola: f64 = xf + rat;
                if (x_parabola - a) < tol2 || (b - x_parabola) < tol2 {
                    let si: f64 = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            // A golden-section step
            if xf >= xm {
                e = a - xf;
            } else {
                e = b - xf;
            }
            rat = golden_mean * e;
        }
        let si: f64 = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        let x: f64 = xf + si * rat.abs().max(tol1);
        let fu: f64 = function(x);
        n_function += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + x_tolerance / 3.0;
        tol2 = 2.0 * tol1;

        if n_function >= n_function_max {
            break 'search_loop;
        }
    }
    xf
}

/// The sign of `value`: -1, 0 or 1 (0 also for NaN)
fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `a / b` for complex numbers by Smith's algorithm (the textbook `a conj(b) / |b|**2` overflows and loses accuracy).
pub fn complex_divide(a: c64, b: c64) -> c64 {
    let b_re_abs: f64 = b.re.abs();
    let b_im_abs: f64 = b.im.abs();
    if b_re_abs >= b_im_abs {
        if b_re_abs == 0.0 && b_im_abs == 0.0 {
            return c64::new(a.re / b_re_abs, a.im / b_re_abs);
        }
        let ratio: f64 = b.im / b.re;
        let scale: f64 = 1.0 / (b.re + b.im * ratio);
        c64::new((a.re + a.im * ratio) * scale, (a.im - a.re * ratio) * scale)
    } else {
        let ratio: f64 = b.re / b.im;
        let scale: f64 = 1.0 / (b.im + b.re * ratio);
        c64::new((a.re * ratio + a.im) * scale, (a.im * ratio - a.re) * scale)
    }
}

/// The binomial coefficient `C(n, k)` in floating point, by the multiplicative formula (exact while it fits in 53 bits)
pub fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut value: f64 = 1.0;
    for i_factor in 1..=k {
        value = value * ((n - k + i_factor) as f64) / (i_factor as f64);
    }
    value
}

/// Copy an `ndarray` matrix into `faer`
pub fn to_faer(matrix: ArrayView2<f64>) -> faer::Mat<f64> {
    faer::Mat::from_fn(matrix.nrows(), matrix.ncols(), |i_row, i_col| matrix[[i_row, i_col]])
}

/// Copy a `faer` matrix into `ndarray`
pub fn from_faer(matrix: faer::MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(i_row, i_col)| matrix[(i_row, i_col)])
}

/// The thin SVD: `u` is `(m, k)`, `v` is `(n, k)`, `k = min(m, n)`.
///
/// # Panics
/// faer's SVD does not converge (non-finite entries)
pub fn svd_thin(matrix: ArrayView2<f64>) -> SingularValueDecomposition {
    let n_row: usize = matrix.nrows();
    let n_col: usize = matrix.ncols();
    let n_singular: usize = n_row.min(n_col);
    if n_singular == 0 {
        return SingularValueDecomposition {
            u: Array2::zeros((n_row, 0)),
            s: Array1::zeros(0),
            v: Array2::zeros((n_col, 0)),
        };
    }
    let matrix_faer: faer::Mat<f64> = to_faer(matrix);
    let svd: faer::linalg::solvers::Svd<f64> = faer::linalg::solvers::Svd::new_thin(matrix_faer.as_ref()).expect("svd_thin: the SVD did not converge");
    let s: Array1<f64> = Array1::from_shape_fn(n_singular, |i_singular| svd.S().column_vector()[i_singular]);
    SingularValueDecomposition {
        u: from_faer(svd.U()),
        s,
        v: from_faer(svd.V()),
    }
}

/// The full SVD: `u` is `(m, m)`, `v` is `(n, n)`, with `min(m, n)` singular values.
///
/// # Panics
/// faer's SVD does not converge (non-finite entries)
pub fn svd_full(matrix: ArrayView2<f64>) -> SingularValueDecomposition {
    let n_row: usize = matrix.nrows();
    let n_col: usize = matrix.ncols();
    let n_singular: usize = n_row.min(n_col);
    if n_row == 0 || n_col == 0 {
        return SingularValueDecomposition {
            u: Array2::eye(n_row),
            s: Array1::zeros(0),
            v: Array2::eye(n_col),
        };
    }
    let matrix_faer: faer::Mat<f64> = to_faer(matrix);
    let svd: faer::linalg::solvers::Svd<f64> = faer::linalg::solvers::Svd::new(matrix_faer.as_ref()).expect("svd_full: the SVD did not converge");
    let s: Array1<f64> = Array1::from_shape_fn(n_singular, |i_singular| svd.S().column_vector()[i_singular]);
    SingularValueDecomposition {
        u: from_faer(svd.U()),
        s,
        v: from_faer(svd.V()),
    }
}

/// The singular values, largest first, without the singular vectors
///
/// # Panics
/// faer's SVD does not converge (non-finite entries)
pub fn singular_values(matrix: ArrayView2<f64>) -> Array1<f64> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Array1::zeros(0);
    }
    let matrix_faer: faer::Mat<f64> = to_faer(matrix);
    let singular: Vec<f64> = matrix_faer.singular_values().expect("singular_values: the SVD did not converge");
    Array1::from_vec(singular)
}

/// The number of singular values above `relative_cutoff` times the largest
pub fn rank_above(s: &Array1<f64>, relative_cutoff: f64) -> usize {
    let n_singular: usize = s.len();
    if n_singular == 0 {
        return 0;
    }
    let mut rank: usize = 0;
    for i_singular in 0..n_singular {
        if s[i_singular] > relative_cutoff * s[0] {
            rank += 1;
        }
    }
    rank
}

/// The pseudo-inverse `V Sigma^-1 U^T`: singular values not above `rcond` times the largest are dropped.
///
/// # Arguments
/// * `matrix` - `(m, n)` [any unit]
/// * `rcond` - relative cut-off [dimensionless]
///
/// # Returns
/// * `(n, m)` [inverse unit]
///
/// # Panics
/// faer's SVD does not converge (non-finite entries)
pub fn pinv(matrix: ArrayView2<f64>, rcond: f64) -> Array2<f64> {
    let n_row: usize = matrix.nrows();
    let n_col: usize = matrix.ncols();
    if n_row == 0 || n_col == 0 {
        return Array2::zeros((n_col, n_row));
    }
    let matrix_faer: faer::Mat<f64> = to_faer(matrix);
    let svd: faer::linalg::solvers::Svd<f64> = faer::linalg::solvers::Svd::new_thin(matrix_faer.as_ref()).expect("pinv: the SVD did not converge");
    let singular: faer::ColRef<'_, f64> = svd.S().column_vector();
    let n_singular: usize = singular.nrows();
    // The singular values are sorted, largest first, so those above the cut-off are the leading ones
    let cutoff: f64 = rcond * singular[0];
    let mut n_kept: usize = 0;
    for i_singular in 0..n_singular {
        if singular[i_singular] > cutoff {
            n_kept += 1;
        }
    }
    let v_scaled: faer::Mat<f64> = faer::Mat::from_fn(n_col, n_kept, |i_col: usize, i_kept: usize| svd.V()[(i_col, i_kept)] * (1.0 / singular[i_kept]));
    let mut result: faer::Mat<f64> = faer::Mat::zeros(n_col, n_row);
    matmul(
        result.as_mut(),
        Accum::Replace,
        v_scaled.as_ref(),
        svd.U().subcols(0, n_kept).transpose(),
        1.0,
        Par::Seq,
    );
    from_faer(result.as_ref())
}
