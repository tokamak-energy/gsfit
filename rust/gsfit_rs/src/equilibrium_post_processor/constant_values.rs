//! The quantities which are fixed for the whole of a time-slice's post-processing.

use crate::source_functions::SharedSourceFunction;
use imas_rs::ids::wall::Wall as WallIds;

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
    pub p_prime_source_function: &'a SharedSourceFunction,
    /// Vacuum toroidal field reference radius [metre]
    pub r0: f64,
    pub wall_ids: &'a WallIds,
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
            dof_values: Array1::zeros(0),
        })
    });
    static WALL_IDS: LazyLock<WallIds> = LazyLock::new(WallIds::default);

    return ConstantValues {
        b0: f64::NAN,
        ff_prime_source_function: &SOURCE_FUNCTION,
        i_rod: f64::NAN,
        p_prime_source_function: &SOURCE_FUNCTION,
        r0: f64::NAN,
        wall_ids: &WALL_IDS,
    };
}
