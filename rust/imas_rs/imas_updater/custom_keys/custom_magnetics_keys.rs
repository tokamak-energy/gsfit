//! Custom (non-IMAS) keys added to the magnetics IDS.
//!
//! This file is **not** compiled: it sits outside `src/`, as an input to
//! `../build_ids.py`, which splices the fields below into the matching generated structs in
//! `../../src/ids/magnetics.rs`. It is written in ordinary Rust syntax so that it reads
//! exactly like the generated file, and so editors can still parse it.
//!
//! See `custom_equilibrium_keys.rs` for how the splice works.
//!
//! Note that these keys have no IMAS counterpart, so they cannot be written to an IMAS
//! backend. Anything written out over the standard data dictionary must skip them.

use crate::dd_base_types::{FLT_0D, STR_0D};

pub struct MagneticsFluxLoop {
    /// Greens tables: the poloidal flux at this flux loop per ampere flowing in each current source
    pub greens: MagneticsFluxLoopGreens,
}

/// Greens tables: the poloidal flux at one flux loop per ampere flowing in each current source.
///
/// These are geometry only - they depend on where the flux loop and the conductors are, never on
/// the plasma - so they are the same for every time.
pub struct MagneticsFluxLoopGreens {
    /// Active poloidal field coils, one entry per coil, in the same order as the `pf_active` IDS
    /// `coil` array of structures
    pub pf_active: Vec<MagneticsFluxLoopGreensPfActive>,
}

/// Greens table between one flux loop and one active poloidal field coil, summed over the coil's
/// filaments
pub struct MagneticsFluxLoopGreensPfActive {
    /// Name of the coil, matching `pf_active/coil/name`, e.g. `"BVL"`
    pub name: STR_0D,
    /// Poloidal flux at the flux loop, per ampere in the coil
    /// Units: Wb.A^-1
    pub value: FLT_0D,
}
