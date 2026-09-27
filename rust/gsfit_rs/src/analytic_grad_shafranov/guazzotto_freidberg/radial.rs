use ndarray::Array1;

/// Coefficient recursion for the C_n / S_n power series (GF Eq. A3).
///
/// `a0`, `b0` seed the series: `(1, 0)` gives the cosine-like `C_n`, `(0, 1)` the
/// sine-like `S_n`. The first three coefficients are the seeds and zeros.
fn compute_coeffs(eps_hat: f64, lambda_sq: f64, k_n: f64, a0: f64, b0: f64, n_radial_expansion: usize) -> (Array1<f64>, Array1<f64>) {
    let mut a: Array1<f64> = Array1::zeros(n_radial_expansion);
    let mut b: Array1<f64> = Array1::zeros(n_radial_expansion);
    a[0] = a0;
    b[0] = b0;
    let coupling: f64 = lambda_sq - eps_hat * k_n * k_n;
    for i_term in 3..n_radial_expansion {
        let i_term_f64: f64 = i_term as f64;
        let pre: f64 = -1.0 / (i_term_f64 * (i_term_f64 - 1.0));
        a[i_term] = pre
            * (eps_hat * (i_term_f64 - 1.0) * (i_term_f64 - 2.0) * a[i_term - 1]
                + coupling * a[i_term - 3]
                + 2.0 * (i_term_f64 - 1.0) * k_n * b[i_term - 1]
                + 2.0 * eps_hat * (i_term_f64 - 2.0) * k_n * b[i_term - 2]);
        b[i_term] = pre
            * (eps_hat * (i_term_f64 - 1.0) * (i_term_f64 - 2.0) * b[i_term - 1] + coupling * b[i_term - 3]
                - 2.0 * (i_term_f64 - 1.0) * k_n * a[i_term - 1]
                - 2.0 * eps_hat * (i_term_f64 - 2.0) * k_n * a[i_term - 2]);
    }
    (a, b)
}

/// Precomputed radial basis functions `C_n(x)`, `S_n(x)` (and their first two
/// derivatives) for a fixed eigenvalue `alpha` (GF Appendix A). The separation
/// constants are from GF Eq. (3.3).
pub struct RadialBasis {
    eps_hat: f64,
    lambda_sq: f64,
    k: [f64; 4],
    h: [f64; 4],
    a_cos: Vec<Array1<f64>>,
    b_cos: Vec<Array1<f64>>,
    a_sin: Vec<Array1<f64>>,
    b_sin: Vec<Array1<f64>>,
    n_radial_expansion: usize,
}

impl RadialBasis {
    pub fn new(alpha: f64, eps: f64, nu: f64, n_radial_expansion: usize) -> Self {
        let eps_hat: f64 = 2.0 * eps / (1.0 + eps * eps);
        let lambda_sq: f64 = eps_hat * alpha * alpha * nu; // GF Eq. (A1)
        let root: f64 = (1.0 + eps * eps).sqrt();
        // GF Eq. (3.3): h_1..h_4 and k_1..k_4
        let h: [f64; 4] = [
            root * alpha,
            (35.0_f64 / 36.0).sqrt() * root * alpha,
            (13.0_f64 / 49.0).sqrt() * root * alpha,
            0.0,
        ];
        let k: [f64; 4] = [0.0, alpha / 6.0, 6.0 * alpha / 7.0, alpha];

        let mut a_cos: Vec<Array1<f64>> = Vec::with_capacity(4);
        let mut b_cos: Vec<Array1<f64>> = Vec::with_capacity(4);
        let mut a_sin: Vec<Array1<f64>> = Vec::with_capacity(4);
        let mut b_sin: Vec<Array1<f64>> = Vec::with_capacity(4);
        for i_n in 0..4 {
            let (a_c, b_c): (Array1<f64>, Array1<f64>) = compute_coeffs(eps_hat, lambda_sq, k[i_n], 1.0, 0.0, n_radial_expansion);
            let (a_s, b_s): (Array1<f64>, Array1<f64>) = compute_coeffs(eps_hat, lambda_sq, k[i_n], 0.0, 1.0, n_radial_expansion);
            a_cos.push(a_c);
            b_cos.push(b_c);
            a_sin.push(a_s);
            b_sin.push(b_s);
        }
        RadialBasis {
            eps_hat,
            lambda_sq,
            k,
            h,
            a_cos,
            b_cos,
            a_sin,
            b_sin,
            n_radial_expansion,
        }
    }

    /// Separation constant `h_n` (1-indexed).
    pub fn h(&self, n: usize) -> f64 {
        self.h[n - 1]
    }

    /// Evaluate a cosine/sine-like series and its first two `x`-derivatives at `x`
    /// (GF Eqs. A2, A4). `X'' = -(k_n^2 + lambda^2 x)/(1 + eps_hat x) X`.
    fn eval(&self, x: f64, a: &Array1<f64>, b: &Array1<f64>, k_n: f64) -> (f64, f64, f64) {
        let cos_kx: f64 = (k_n * x).cos();
        let sin_kx: f64 = (k_n * x).sin();
        let mut poly_a: f64 = 0.0;
        let mut poly_b: f64 = 0.0;
        let mut poly_a_deriv: f64 = 0.0;
        let mut poly_b_deriv: f64 = 0.0;
        let mut x_pow: f64 = 1.0; // x^i_term
        let mut x_pow_prev: f64 = 1.0; // x^(i_term-1); the i_term=0 derivative term is killed by the i_term factor
        for i_term in 0..self.n_radial_expansion {
            let i_term_f64: f64 = i_term as f64;
            poly_a += a[i_term] * x_pow;
            poly_b += b[i_term] * x_pow;
            poly_a_deriv += i_term_f64 * a[i_term] * x_pow_prev;
            poly_b_deriv += i_term_f64 * b[i_term] * x_pow_prev;
            x_pow_prev = x_pow;
            x_pow *= x;
        }
        let value: f64 = cos_kx * poly_a + sin_kx * poly_b;
        let value_prime: f64 = cos_kx * (k_n * poly_b + poly_a_deriv) + sin_kx * (poly_b_deriv - k_n * poly_a);
        let value_double_prime: f64 = -(k_n * k_n + self.lambda_sq * x) / (1.0 + self.eps_hat * x) * value;
        (value, value_prime, value_double_prime)
    }

    /// `(C_n(x), C_n'(x), C_n''(x))`.
    pub fn c_all(&self, x: f64, n: usize) -> (f64, f64, f64) {
        self.eval(x, &self.a_cos[n - 1], &self.b_cos[n - 1], self.k[n - 1])
    }

    /// `(S_n(x), S_n'(x), S_n''(x))`.
    pub fn s_all(&self, x: f64, n: usize) -> (f64, f64, f64) {
        self.eval(x, &self.a_sin[n - 1], &self.b_sin[n - 1], self.k[n - 1])
    }
}
