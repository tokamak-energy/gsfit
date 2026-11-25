//! Double-double arithmetic: a number is the unevaluated sum `high + low` of two `f64` with `|low| <= ulp(high) / 2`, about 32
//! significant digits, built from error-free transformations (Dekker, Knuth; Hida, Li and Bailey's QD library).
//!
//! Used for the minimum-norm particular solutions of the radial functions of strongly hyperbolic wavenumbers (`k_j` well above
//! `k_max`), whose projection cancels terms `exp(growth)` times larger than the result: double precision keeps `16 - growth / 2.3`
//! digits of it, double-double `32 - growth / 2.3`.
//!
//! Rust never contracts `a * b + c` into a fused multiply-add by itself, which would break Dekker's splitting in `two_product`.

/// Dekker's splitting constant, `2**27 + 1` [dimensionless]
const SPLITTER: f64 = 134217729.0;

/// `high + low`, with `|low| <= ulp(high) / 2`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleDouble {
    pub high: f64,
    pub low: f64,
}

impl DoubleDouble {
    pub const ZERO: DoubleDouble = DoubleDouble { high: 0.0, low: 0.0 };

    /// `(value, 0)`
    pub fn from_float(value: f64) -> DoubleDouble {
        DoubleDouble { high: value, low: 0.0 }
    }

    /// `high + low` rounded to `f64`
    pub fn to_float(self) -> f64 {
        self.high + self.low
    }

    /// `-self`
    pub fn negate(self) -> DoubleDouble {
        DoubleDouble {
            high: -self.high,
            low: -self.low,
        }
    }

    /// `self + other`
    pub fn add(self, other: DoubleDouble) -> DoubleDouble {
        let sum: DoubleDouble = two_sum(self.high, other.high);
        let sum_low: DoubleDouble = two_sum(self.low, other.low);
        let error: f64 = sum.low + sum_low.high;
        let renormalised: DoubleDouble = quick_two_sum(sum.high, error);
        let error_again: f64 = renormalised.low + sum_low.low;
        quick_two_sum(renormalised.high, error_again)
    }

    /// `self * other`
    pub fn multiply(self, other: DoubleDouble) -> DoubleDouble {
        let product: DoubleDouble = two_product(self.high, other.high);
        let error: f64 = product.low + (self.high * other.low + self.low * other.high);
        quick_two_sum(product.high, error)
    }

    /// `self * other`, `other` an `f64`
    pub fn multiply_float(self, other: f64) -> DoubleDouble {
        let product: DoubleDouble = two_product(self.high, other);
        let error: f64 = product.low + self.low * other;
        quick_two_sum(product.high, error)
    }

    /// `self / other`, by three `f64` quotient corrections
    pub fn divide(self, other: DoubleDouble) -> DoubleDouble {
        let quotient_1: f64 = self.high / other.high;
        let remainder: DoubleDouble = self.add(other.multiply_float(quotient_1).negate());
        let quotient_2: f64 = remainder.high / other.high;
        let remainder_again: DoubleDouble = remainder.add(other.multiply_float(quotient_2).negate());
        let quotient_3: f64 = remainder_again.high / other.high;
        let quotient: DoubleDouble = quick_two_sum(quotient_1, quotient_2);
        quotient.add(DoubleDouble::from_float(quotient_3))
    }

    /// `self / other`, `other` a non-zero `f64`
    pub fn divide_float(self, other: f64) -> DoubleDouble {
        self.divide(DoubleDouble::from_float(other))
    }

    /// `self ** 0.5`, `self >= 0`, by one Newton correction of the `f64` root
    pub fn square_root(self) -> DoubleDouble {
        let root: f64 = self.high.sqrt();
        let root_squared: DoubleDouble = two_product(root, root);
        let remainder: DoubleDouble = self.add(root_squared.negate());
        let correction: f64 = if root > 0.0 { remainder.high / (2.0 * root) } else { 0.0 };
        quick_two_sum(root, correction)
    }
}

/// `(s, e)` with `s = fl(a + b)` and `s + e = a + b` exactly
fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let sum: f64 = a + b;
    let b_virtual: f64 = sum - a;
    let error: f64 = (a - (sum - b_virtual)) + (b - b_virtual);
    DoubleDouble { high: sum, low: error }
}

/// `(s, e)` with `s = fl(a + b)` and `s + e = a + b` exactly, for `|a| >= |b|`
fn quick_two_sum(a: f64, b: f64) -> DoubleDouble {
    let sum: f64 = a + b;
    let error: f64 = b - (sum - a);
    DoubleDouble { high: sum, low: error }
}

/// `(p, e)` with `p = fl(a b)` and `p + e = a b` exactly, by Dekker's splitting (no overflow below about 1e300)
fn two_product(a: f64, b: f64) -> DoubleDouble {
    let product: f64 = a * b;
    let split_a: f64 = SPLITTER * a;
    let a_high: f64 = split_a - (split_a - a);
    let a_low: f64 = a - a_high;
    let split_b: f64 = SPLITTER * b;
    let b_high: f64 = split_b - (split_b - b);
    let b_low: f64 = b - b_high;
    let error: f64 = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
    DoubleDouble { high: product, low: error }
}

/// The sum of `values` by pairwise double-double additions: at every level an odd count is padded with a zero, then neighbours are
/// added, `(0, 1), (2, 3), ...`, until one value is left.
///
/// # Arguments
/// * `values` - the summands, at least one [any unit]
///
/// # Returns
/// * the sum [same unit]
pub fn sum_pairwise(values: &[DoubleDouble]) -> DoubleDouble {
    let n_value: usize = values.len();
    if n_value == 0 {
        return DoubleDouble::ZERO;
    }
    let mut level: Vec<DoubleDouble> = values.to_vec();
    let n_level: usize = (n_value as f64).log2().ceil() as usize;
    for _i_level in 0..n_level {
        if level.len() % 2 == 1 {
            level.push(DoubleDouble::ZERO);
        }
        let n_pair: usize = level.len() / 2;
        let mut next: Vec<DoubleDouble> = Vec::with_capacity(n_pair);
        for i_pair in 0..n_pair {
            next.push(level[2 * i_pair].add(level[2 * i_pair + 1]));
        }
        level = next;
    }
    level[0]
}
