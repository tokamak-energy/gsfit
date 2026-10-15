//! Custom (non-IMAS) keys added to the pf_active IDS.
//!
//! This file is **not** compiled: it sits outside `src/`, as an input to
//! `../build_ids.py`, which splices the fields below into the matching generated structs in
//! `../../src/ids/pf_active.rs`. It is written in ordinary Rust syntax so that it reads
//! exactly like the generated file, and so editors can still parse it.
//!
//! See `custom_equilibrium_keys.rs` for how the splice works.
//!
//! Note that these keys have no IMAS counterpart, so they cannot be written to an IMAS
//! backend. Anything written out over the standard data dictionary must skip them.

pub struct PfCoils {
    /// Which of the coil's signals is prescribed: `current` (`index` 1) when `coil(i1)/current` is given, or `voltage` (`index` 2)
    /// when `coil(i1)/voltage` is given and the current is calculated
    pub prescribed: IdentifierStatic,
}
