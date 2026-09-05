//! IMAS Data Dictionary Library for Rust
//!
//! This library provides Rust implementations of the IMAS (Integrated Modelling
//! & Analysis Suite) data dictionary structures and types for tokamak fusion
//! data management.

pub mod dd_base_types;
pub mod ids;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types at crate root
pub use dd_base_types::*;
pub use ids::*;
