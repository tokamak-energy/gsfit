//! IMAS Data Dictionary Library for Rust
//!
//! This library provides Rust implementations of the IMAS (Integrated Modelling
//! & Analysis Suite) data dictionary structures and types for tokamak fusion
//! data management.

pub mod dd_base_types;
pub mod ids;

/// The IMAS Data Dictionary version that the generated `src/ids/*.rs` files came from,
/// e.g. `"4.1.1-60-gf5d44e8"`.
///
/// `src/imas_dd_version.txt` is written by `imas_updater/build_ids.py` in the same run that
/// writes the generated files, so this constant cannot drift from the structs it describes.
pub const IMAS_DD_VERSION: &str = include_str!("imas_dd_version.txt").trim_ascii_end();

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types at crate root
pub use dd_base_types::*;
pub use ids::*;
