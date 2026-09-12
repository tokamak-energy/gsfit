//! IMAS Data Dictionary Base Types
//!
//! This module defines the fundamental data types used in the IMAS
//! (Integrated Modelling & Analysis Suite) data dictionary schema.
//!
//! The naming convention follows the IMAS specification:
//! - `INT` = Integer
//! - `FLT` = Floating point (f64)
//! - `STR` = String
//! - `CPX` = Complex number
//! - `_0D` = Scalar (0-dimensional)
//! - `_1D` = 1-dimensional array
//! - `_2D` = 2-dimensional array

#![allow(non_camel_case_types)]

use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6};
use num_complex::Complex64;

// ============================================================================
// Scalar Types (0D)
// ============================================================================

/// Integer scalar
pub type INT_0D = i32;

/// The IMAS marker for an unset integer.
///
/// Every leaf of a freshly constructed IDS holds its "unset" value, and only holds a real value
/// once something has written one. The unset values follow the IMAS convention: a float is `NaN`,
/// a string or an array is empty, and an integer, which has no `NaN`, is this reserved value. They
/// are the same in Rust and in Python, so there is nothing to unwrap on the way in and nothing to
/// translate on the way out. Test for them with `is_nan()`, `== EMPTY_INT` or `is_empty()`; a
/// float can never be tested with `==`, because `NaN != NaN`.
pub const EMPTY_INT: INT_0D = -999999999;

/// Floating-point scalar (double precision)
pub type FLT_0D = f64;

/// String scalar
pub type STR_0D = String;

/// Complex scalar (double precision)
pub type CPX_0D = Complex64;

// ============================================================================
// 1D Array Types
// ============================================================================

/// 1D integer array
pub type INT_1D = Array1<i32>;

/// 1D floating-point array
pub type FLT_1D = Array1<f64>;

/// 1D string array
pub type STR_1D = Vec<String>;

/// 1D complex array
pub type CPX_1D = Array1<Complex64>;

// ============================================================================
// 2D Array Types
// ============================================================================

/// 2D integer array
pub type INT_2D = Array2<i32>;

/// 2D floating-point array
pub type FLT_2D = Array2<f64>;

/// 2D complex array
pub type CPX_2D = Array2<Complex64>;

// ============================================================================
// 3D Array Types
// ============================================================================

/// 3D integer array
pub type INT_3D = Array3<i32>;

/// 3D floating-point array
pub type FLT_3D = Array3<f64>;

/// 3D complex array
pub type CPX_3D = Array3<Complex64>;

// ============================================================================
// 4D Array Types
// ============================================================================

/// 4D integer array
pub type INT_4D = Array4<i32>;

/// 4D floating-point array
pub type FLT_4D = Array4<f64>;

/// 4D complex array
pub type CPX_4D = Array4<Complex64>;

// ============================================================================
// 5D Array Types
// ============================================================================

/// 5D integer array
pub type INT_5D = Array5<i32>;

/// 5D floating-point array
pub type FLT_5D = Array5<f64>;

/// 5D complex array
pub type CPX_5D = Array5<Complex64>;

// ============================================================================
// 6D Array Types
// ============================================================================

/// 6D integer array
pub type INT_6D = Array6<i32>;

/// 6D floating-point array
pub type FLT_6D = Array6<f64>;

/// 6D complex array
pub type CPX_6D = Array6<Complex64>;

// ============================================================================
// Structural Types
// ============================================================================

/// Marker trait for IMAS structures.
///
/// Implement this trait on any struct that represents an IMAS structure node.
/// This enables generic handling of nested data dictionary structures.
pub trait Structure: Clone + std::fmt::Debug {}

/// A dynamically-sized array of structures (e.g., time_slice, channel, etc.)
///
/// This is the Rust representation of IMAS `struct_array` - an array of
/// homogeneous structures that can grow dynamically.
pub type StructArray<T> = Vec<T>;

// ============================================================================
// Gathered Field Accessors
// ============================================================================

// When a leaf field is read across a whole array of structures - e.g.
// `equilibrium.time_slice(..).global_quantities.magnetic_axis.r` - the values are
// not contiguous in memory, so they have to be gathered into a fresh array.
//
// These two accumulators do that gathering. They are generic and hand-written
// once: the generated code supplies the field path as a projection closure, so
// the path is written exactly once and type-checked by the compiler, rather than
// being encoded in a generated type name (which previously allowed two different
// paths sharing a leaf name to collapse onto the same accumulator).

/// Lazily gathers one `Copy` scalar leaf field across a slice of IMAS structures.
///
/// `T` is the array-of-structures element type (e.g. `EquilibriumTimeSlice`) and
/// `U` the leaf's scalar type (`f64`, `i32`, `Complex64`). An unset element gathers
/// as its unset value (`NaN`, `EMPTY_INT`), like any other.
pub struct Accumulator<'a, T, U> {
    data: &'a [T],
    project: fn(&T) -> U,
}

impl<'a, T, U> Accumulator<'a, T, U> {
    pub fn new(data: &'a [T], project: fn(&T) -> U) -> Self {
        Self { data, project }
    }

    /// Number of elements gathered over.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gather every value into an `Array1`.
    pub fn to_array(&self) -> Array1<U> {
        Array1::from_iter(self.data.iter().map(|item| (self.project)(item)))
    }

    /// Gather every value into a `Vec`.
    pub fn to_vec(&self) -> Vec<U> {
        self.data.iter().map(|item| (self.project)(item)).collect()
    }
}

/// Lazily gathers one `STR_0D` leaf field across a slice of IMAS structures.
///
/// Separate from [`Accumulator`] because `String` is not `Copy`: values are
/// cloned by the projection, and the gathered form is a `Vec<String>` rather
/// than an `Array1`. An unset element gathers as an empty string.
pub struct StringAccumulator<'a, T> {
    data: &'a [T],
    project: fn(&T) -> String,
}

impl<'a, T> StringAccumulator<'a, T> {
    pub fn new(data: &'a [T], project: fn(&T) -> String) -> Self {
        Self { data, project }
    }

    /// Number of elements gathered over.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gather every value into a `Vec<String>`.
    pub fn to_vec(&self) -> Vec<String> {
        self.data.iter().map(|item| (self.project)(item)).collect()
    }
}

// ============================================================================
// Re-exports for convenience
// ============================================================================

pub use ndarray::{Array, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, array};
pub use num_complex::Complex64 as Complex;
