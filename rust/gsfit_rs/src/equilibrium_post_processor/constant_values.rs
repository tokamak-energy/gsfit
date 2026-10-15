//! The quantities which are fixed for the whole of a time-slice's post-processing.

use crate::source_functions::SharedSourceFunction;
use imas_rs::EquilibriumLimiter;

/// Everything a post-processing calculator reads which is neither on the time-slice it is given
/// nor produced by an earlier calculator.
///
/// These are built once, before the first calculator runs, and handed to every calculator by
/// shared reference. That the reference is shared is the point: it is a compiler-checked proof
/// that no calculator can create an ordering dependency through them, which is what lets the
/// calculators be ordered from their declared dependencies alone.
///
/// Anything which *is* produced part-way through the sequence lives on `IntermediateValues` instead.
pub struct ConstantValues<'a> {
    /// Vacuum toroidal field at `r0`, for this time-slice [tesla]
    pub b0: f64,
    pub ff_prime_source_function: &'a SharedSourceFunction,
    /// Toroidal field rod current [ampere]
    pub i_rod: f64,
    /// The limiter, `equilibrium/limiter`, whose `unit(0)` is the vacuum vessel
    pub limiter: &'a EquilibriumLimiter,
    pub p_prime_source_function: &'a SharedSourceFunction,
    /// Vacuum toroidal field reference radius [metre]
    pub r0: f64,
}

/// Build a `ConstantValues` whose every field is a placeholder, for a unit test to overwrite the
/// one or two fields the calculator under test actually reads.
#[cfg(test)]
pub fn constant_values_for_test() -> ConstantValues<'static> {
    use crate::source_functions::EfitPolynomial;
    use ndarray::{Array1, Array2};
    use std::sync::{Arc, LazyLock};

    static SOURCE_FUNCTION: LazyLock<SharedSourceFunction> = LazyLock::new(|| {
        Arc::new(EfitPolynomial {
            n_dof: 1,
            regularisations: Array2::zeros((1, 1)),
            exact: false,
            dof_values: Array1::zeros(0),
        })
    });
    static LIMITER: LazyLock<EquilibriumLimiter> = LazyLock::new(EquilibriumLimiter::default);

    return ConstantValues {
        b0: f64::NAN,
        ff_prime_source_function: &SOURCE_FUNCTION,
        i_rod: f64::NAN,
        limiter: &LIMITER,
        p_prime_source_function: &SOURCE_FUNCTION,
        r0: f64::NAN,
    };
}
