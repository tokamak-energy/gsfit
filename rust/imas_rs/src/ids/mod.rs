//! IMAS Interface Data Structures (IDS)
//!
//! This module contains the various IDS definitions for the IMAS data model.
//! Generated from IMAS Data Dictionary XSD schemas.

pub mod equilibrium;
pub mod magnetics;
pub mod pf_active;
pub mod pf_passive;
pub mod tf;
pub mod wall;

// Only `equilibrium` is re-exported at the crate root. Every IDS declares its own `Code`,
// `Library`, `IdentifierStatic`, ... so glob re-exporting a second one would make
// `imas_rs::Code` ambiguous. Reach the others through their module: `imas_rs::ids::wall::Wall`,
// `imas_rs::ids::pf_active::PfActive`, ...
pub use equilibrium::*;
