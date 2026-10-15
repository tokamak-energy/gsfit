//! Custom (non-IMAS) keys added to the pf_passive IDS.
//!
//! This file is **not** compiled: it sits outside `src/`, as an input to
//! `../build_ids.py`, which splices the fields below into the matching generated structs in
//! `../../src/ids/pf_passive.rs`. It is written in ordinary Rust syntax so that it reads
//! exactly like the generated file, and so editors can still parse it.
//!
//! See `custom_equilibrium_keys.rs` for how the splice works.
//!
//! Note that these keys have no IMAS counterpart, so they cannot be written to an IMAS
//! backend. Anything written out over the standard data dictionary must skip them.

pub struct PfPassiveLoops {
    /// How the loop's current is distributed over its elements, which sets its degrees of freedom, `dof`:
    /// `constant_current_density` (`index` 1), one degree of freedom with the same current density in every element; or
    /// `eig` (`index` 2), one degree of freedom for each of the loop's slowest decaying eigenmodes
    pub current_distribution_type: IdentifierStatic,
    /// The loop's degrees of freedom: the current distributions whose amplitudes the equilibrium reconstruction fits
    pub dof: Vec<PfPassiveLoopsDof>,
    /// The regularisations of the degrees of freedom, one row of `n_dof` coefficients for each regularisation; no rows for none.
    /// Shape = `(n_regularisations, n_dof)`
    pub regularisations: FLT_2D,
    /// The weight of each regularisation in the fit, one for each row of `regularisations`
    /// Units: dimensionless
    pub regularisations_weight: FLT_1D,
}

/// One degree of freedom of a passive loop's current
pub struct PfPassiveLoopsDof {
    /// Name of the degree of freedom: `constant_current_density`, or `eig_01`, `eig_02`, ... for the eigenmodes, slowest decaying first
    pub name: STR_0D,
    /// The current in each element, in the order of `element`, for a unit amplitude of this degree of freedom.
    /// For `constant_current_density` it is the element's area, so that the amplitude is the current density
    pub current_distribution: FLT_1D,
    /// The eigenmode's time constant, the ratio of its inductance to its resistance. Only set for `eig`
    /// Units: s
    pub time_constant: FLT_0D,
}

pub struct PfCoilsElements {
    /// The current in this element, on the loop's timebase, `loop(i1)/time`. Only set by `solve_circuit_equations`, which simulates it
    /// Units: A
    pub current: FLT_1D,
}
