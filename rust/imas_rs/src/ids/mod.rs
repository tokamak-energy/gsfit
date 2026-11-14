//! IMAS Interface Data Structures (IDS)
//!
//! This module contains the various IDS definitions for the IMAS data model.
//! Generated from IMAS Data Dictionary XSD schemas.

pub mod core_profiles;
pub mod core_sources;
pub mod core_transport;
pub mod distribution_sources;
pub mod distributions;
pub mod equilibrium;
pub mod magnetics;
pub mod nbi;
pub mod pf_active;
pub mod pf_passive;
pub mod pulse_schedule;
pub mod tf;
pub mod thomson_scattering;
pub mod transport_solver_numerics;
pub mod wall;

// Only `equilibrium` is re-exported at the crate root. Every IDS declares its own `Code`,
// `Library`, `IdentifierStatic`, ... so glob re-exporting a second one would make
// `imas_rs::Code` ambiguous. Reach the others through their module: `imas_rs::ids::wall::Wall`,
// `imas_rs::ids::pf_active::PfActive`, ...
pub use equilibrium::*;
