//! The intermediate quantities the post-processing calculators produce for one another.

use super::flux_surfaces::FluxSurface;

/// The quantities calculated part-way through a time-slice's post-processing and read by later
/// calculators, which fill no data dictionary path and so cannot be carried on the time-slice.
///
/// Every calculator takes this by mutable reference, so that the ones which produce a field and
/// the ones which only read it share a single signature and can be held in one dispatch table.
/// The struct is kept as small as possible for exactly that reason: each field here is an
/// ordering dependency the compiler cannot check, so it has to be declared instead. Anything
/// fixed for the whole time-slice belongs on `ConstantValues`, where the shared reference proves
/// it is not a dependency at all.
///
/// Every field starts as a placeholder and holds it until the calculator which fills it has run.
pub struct IntermediateValues {
    /// Flux-surface-averaged `b_p ** 2` [tesla ** 2]
    pub bp_sq_fs_avg: f64,
    /// The closed flux surfaces, one per `psi_norm`
    pub flux_surfaces: Vec<FluxSurface>,
}

/// Build an `IntermediateValues` holding only placeholders, as it stands before the first calculator
/// has run.
pub fn intermediate_values_placeholders() -> IntermediateValues {
    IntermediateValues {
        bp_sq_fs_avg: f64::NAN,
        flux_surfaces: Vec::new(),
    }
}

/// Build an `IntermediateValues` whose every field is a placeholder, for a unit test to overwrite the
/// one or two fields the calculator under test actually reads.
#[cfg(test)]
pub fn intermediate_values_for_test() -> IntermediateValues {
    return intermediate_values_placeholders();
}
