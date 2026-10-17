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

// `Array` itself comes in through the `pub use` at the bottom of this file
use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, Dimension, Ix1, IxDyn};
use num_complex::Complex64;
use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::slice::SliceIndex;
use std::sync::Arc;

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

// When a leaf field is read across an array of structures - e.g.
// `equilibrium.time_slice(..).global_quantities.magnetic_axis.r` - the values are
// not contiguous in memory, so they have to be gathered into a fresh array.
//
// Arrays of structures can be sliced at more than one level, e.g.
// `magnetics.flux_loop(..).greens.pf_active(..).value`. The dimensions are then gained from left to
// right: one per sliced level in the order they appear on the path, followed by the leaf's own
// dimensions, so that example is `(n_flux_loop, n_pf)`. See `arrange_left_to_right`, which is the
// one place that rule lives.
//
// The accumulator is generic and hand-written once: the generated code supplies the field path as
// a projection closure, so the path is written exactly once and type-checked by the compiler,
// rather than being encoded in a generated type name (which previously allowed two different
// paths sharing a leaf name to collapse onto the same accumulator).

/// Shape a gathered array so that its dimensions are gained from left to right.
///
/// `stacked` holds one value per gathered element along its first axis, in *path order* - the
/// first array of structures on the path varying slowest - followed by the leaf's own dimensions,
/// so its shape is `[n_values, leaf...]`. `gathered_shape` is the length of each sliced array of
/// structures, also in path order, and its product is `n_values`.
///
/// The result has shape `[n_level_1, ..., n_level_k, leaf...]`: one dimension per sliced level, in
/// the order they appear on the path, followed by the leaf's own dimensions. So
/// `flux_loop(..)/greens/pf_active(..)/value` is `(n_flux_loop, n_pf)`, and
/// `time_slice(..)/profiles_2d(0)/psi` is `(n_time, n_z, n_r)`, which is how IMAS lays out a time
/// series: time first. A signal stored inside an array of structures therefore has its time last,
/// e.g. `flux_loop(..)/flux/data` is `(n_flux_loop, n_time)`.
///
/// Because the values were collected in path order, this is only a relabelling of the first axis,
/// so nothing is copied and the result keeps the standard (C-contiguous) layout of `stacked`.
pub(crate) fn arrange_left_to_right<A>(stacked: ArrayD<A>, gathered_shape: &[usize]) -> ArrayD<A> {
    let mut shape: Vec<usize> = gathered_shape.to_vec();
    shape.extend_from_slice(&stacked.shape()[1..]);

    stacked
        .into_shape_with_order(IxDyn(&shape))
        .expect("the gathered values are in standard layout and their count is the product of the gathered lengths")
}

/// The elements of an array of structures which a slice view gathers over.
///
/// `items` points into the IDS, flattened in path order: the first array of structures on the path
/// varies slowest. `gathered_shape` is the length of each sliced level, also in path order. `D` is
/// the rank of the gathered result - one dimension per sliced level - carried in the type so that
/// `to_array` returns an `Array2` from a doubly sliced path without any runtime conversion.
///
/// Every accumulator and nested view of one slice view shares the same `items`, which is why they
/// are reference counted: building a view copies a pointer rather than the element list.
pub struct Elements<'a, T, D> {
    items: Arc<[&'a T]>,
    gathered_shape: Arc<[usize]>,
    rank: PhantomData<D>,
}

// Written by hand because `derive(Clone)` would demand `T: Clone` and `D: Clone`, which cloning
// two `Arc`s does not need
impl<T, D> Clone for Elements<'_, T, D> {
    fn clone(&self) -> Self {
        Self {
            items: Arc::clone(&self.items),
            gathered_shape: Arc::clone(&self.gathered_shape),
            rank: PhantomData,
        }
    }
}

impl<'a, T> Elements<'a, T, Ix1> {
    /// The elements of one array of structures, sliced once.
    pub fn from_slice(data: &'a [T]) -> Self {
        let n_items: usize = data.len();
        let mut items: Vec<&'a T> = Vec::with_capacity(n_items);
        for i_item in 0..n_items {
            items.push(&data[i_item]);
        }

        Self {
            items: items.into(),
            gathered_shape: vec![n_items].into(),
            rank: PhantomData,
        }
    }
}

impl<'a, T, D> Elements<'a, T, D> {
    /// Total number of elements, across every sliced level.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The length of each sliced level, in path order like the gathered arrays.
    pub fn shape(&self) -> Vec<usize> {
        self.gathered_shape.to_vec()
    }

    /// Every element, in path order.
    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        self.items.iter().copied()
    }
}

impl<'a, T, D: Dimension> Elements<'a, T, D> {
    /// Slice a nested array of structures under every element, adding one dimension.
    ///
    /// `get` reaches the nested array from one element, e.g. `|flux_loop| &flux_loop.greens.pf_active`,
    /// and `range` is applied under every element. `field_path` is only used to name the nested
    /// array if this panics.
    ///
    /// # Panics
    /// * If the nested arrays are not all the same length once `range` is applied, because the
    ///   result would be jagged and an array cannot hold that.
    /// * If `range` is out of bounds under any element, exactly as slicing a `Vec` does.
    pub fn nest<U, R>(&self, field_path: &'static str, get: fn(&T) -> &[U], range: R) -> Elements<'a, U, D::Larger>
    where
        R: SliceIndex<[U], Output = [U]> + Clone,
    {
        let n_items: usize = self.items.len();

        // With no outer elements there is nothing to measure, so the nested level is empty too
        let mut n_nested: usize = 0;
        let mut nested_items: Vec<&'a U> = Vec::new();

        for i_item in 0..n_items {
            let nested: &'a [U] = &get(self.items[i_item])[range.clone()];

            if i_item == 0 {
                n_nested = nested.len();
                nested_items.reserve(n_items * n_nested);
            } else if nested.len() != n_nested {
                panic!(
                    "cannot gather `{field_path}`: the element at {:?} has {} but the element at {:?} has {n_nested}, so the result \
                     would be jagged",
                    unravel(i_item, &self.gathered_shape),
                    nested.len(),
                    unravel(0, &self.gathered_shape),
                );
            }

            for i_nested in 0..nested.len() {
                nested_items.push(&nested[i_nested]);
            }
        }

        let mut gathered_shape: Vec<usize> = self.gathered_shape.to_vec();
        gathered_shape.push(n_nested);

        Elements {
            items: nested_items.into(),
            gathered_shape: gathered_shape.into(),
            rank: PhantomData,
        }
    }

    /// Select the element named `name` from a nested array of structures under every element, adding no dimension.
    ///
    /// `get` reaches the nested array from one element, as for [`Elements::nest`]. `field_path` is only used to name the
    /// nested array if this panics.
    ///
    /// # Panics
    /// If, under any element, no element or more than one is named `name`.
    pub fn nest_named<U: Named>(&self, field_path: &'static str, get: fn(&T) -> &[U], name: &str) -> Elements<'a, U, D> {
        let n_items: usize = self.items.len();
        let mut named_items: Vec<&'a U> = Vec::with_capacity(n_items);
        for i_item in 0..n_items {
            let nested: &'a [U] = get(self.items[i_item]);
            match index_of_name(nested, name) {
                Ok(i_named) => named_items.push(&nested[i_named]),
                Err(reason) => panic!(
                    "cannot select from `{field_path}` under the element at {:?}: {reason}",
                    unravel(i_item, &self.gathered_shape)
                ),
            }
        }

        // One element under every element, so the sliced levels are those above
        Elements {
            items: named_items.into(),
            gathered_shape: Arc::clone(&self.gathered_shape),
            rank: PhantomData,
        }
    }

    /// Read one leaf out of every element, and arrange the values from left to right.
    fn gather<U: Clone>(&self, project: fn(&T) -> U) -> Array<U, D> {
        let n_items: usize = self.items.len();
        let mut values: Vec<U> = Vec::with_capacity(n_items);
        for i_item in 0..n_items {
            values.push(project(self.items[i_item]));
        }

        let stacked: ArrayD<U> = ArrayD::from_shape_vec(IxDyn(&[n_items]), values).expect("one value was read per element");

        arrange_left_to_right(stacked, &self.gathered_shape)
            .into_dimensionality::<D>()
            .expect("an `Elements` of rank `D` has one sliced level per dimension")
    }
}

/// The index of the `flat`th element (in path order) in the gathered array. Only used to make a
/// panic message point at the elements which disagree.
fn unravel(flat: usize, gathered_shape: &[usize]) -> Vec<usize> {
    let n_gathered: usize = gathered_shape.len();
    let mut index: Vec<usize> = vec![0; n_gathered];
    let mut remainder: usize = flat;
    for i_gathered in (0..n_gathered).rev() {
        index[i_gathered] = remainder % gathered_shape[i_gathered].max(1);
        remainder /= gathered_shape[i_gathered].max(1);
    }
    index
}

/// Lazily gathers one leaf field across the elements of one or more sliced arrays of structures.
///
/// `T` is the array-of-structures element type (e.g. `EquilibriumTimeSlice`), `U` the leaf's type
/// (`f64`, `i32`, `String`, `Complex64`), and `D` the rank of the gathered result. An unset element
/// gathers as its unset value (`NaN`, `EMPTY_INT`, an empty string), like any other.
///
/// What a leaf gathers into depends on its type:
/// * a numeric leaf gathers with `to_array`, into an `ndarray` of any rank, or with `to_vec` for a
///   single sliced level;
/// * a string leaf gathers with `to_vec`, into a `Vec<String>`, for a single sliced level only. An
///   `ndarray` of strings supports none of the operations an array is for, while a `Vec<String>` is
///   what names are handled as everywhere else.
///
/// The split is made by the `Copy` bound on `to_array`: every numeric leaf is `Copy` and `String`
/// is not, so `.name.to_array()` is a compile error rather than an `Array1<String>`.
pub struct Accumulator<'a, T, U, D> {
    elements: Elements<'a, T, D>,
    project: fn(&T) -> U,
}

impl<'a, T, U, D> Accumulator<'a, T, U, D> {
    pub fn new(elements: Elements<'a, T, D>, project: fn(&T) -> U) -> Self {
        Self { elements, project }
    }

    /// Total number of elements gathered over, across every sliced level.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl<T, U: Copy, D: Dimension> Accumulator<'_, T, U, D> {
    /// Gather every value of a numeric leaf into an array whose dimensions are gained from left to
    /// right, e.g. `flux_loop(..).greens.pf_active(..).value.to_array()` is `(n_flux_loop, n_pf)`.
    ///
    /// A string leaf gathers with `to_vec` instead; see [`Accumulator`].
    pub fn to_array(&self) -> Array<U, D> {
        self.elements.gather(self.project)
    }
}

impl<T, U> Accumulator<'_, T, U, Ix1> {
    /// Gather every value into a `Vec`, e.g. `flux_loop(..).name.to_vec()` is a `Vec<String>`.
    ///
    /// This is how a string leaf is gathered. Only offered for a single sliced level, where there is
    /// no question of which order the values come in.
    pub fn to_vec(&self) -> Vec<U> {
        let n_items: usize = self.elements.items.len();
        let mut values: Vec<U> = Vec::with_capacity(n_items);
        for i_item in 0..n_items {
            values.push((self.project)(self.elements.items[i_item]));
        }
        values
    }
}

// ============================================================================
// Selecting from an Array of Structures
// ============================================================================

// Every array of structures has an accessor named after it, e.g. `pf_active.coil(...)`, which takes either
// of two selections:
// * a range of positions, e.g. `.coil(..)` or `.coil(0..2)`, which gives a slice view over those elements,
//   whose leaves gather one value per element;
// * the `name` of one element, e.g. `.coil("BVLT")`, which gives that element. It is only offered where the
//   element has a `name`, which the data dictionary defines as a "short string identifier (unique for a
//   given device)".
// A single position is not a selection: one element is reached by indexing the array itself, `.coil[0]`, so
// that there is only one way to write it.

/// An element of an array of structures: what a slice view gathers over.
///
/// Implemented by the generated code for every type which appears in an array of structures, so that a
/// [`Selection`] can build that type's slice view without knowing which type it is.
pub trait ArrayOfStructuresElement: Sized {
    /// The slice view which gathers over elements of this type, with `D` sliced levels
    type SliceView<'a, D>
    where
        Self: 'a;

    /// Build the slice view over `elements`
    fn slice_view<'a, D: Dimension>(elements: Elements<'a, Self, D>) -> Self::SliceView<'a, D>
    where
        Self: 'a;
}

/// An element of an array of structures which has a `name`, so can be selected by it.
///
/// Implemented by the generated code for every such type.
pub trait Named {
    /// The element's `name`
    fn name(&self) -> &str;
}

/// What an array-of-structures accessor, e.g. `.coil(...)`, can be given: a range of positions, or the `name` of
/// one element. See the section comment above.
///
/// `Output` is what the accessor gives from the structure which holds the array, e.g. `pf_active.coil(...)`.
/// `NestedOutput<D>` is what it gives from a view which has already sliced `D` levels above it, e.g.
/// `equilibrium.time_slice(..).constraints.pf_current(...)`: a range adds one dimension, and a name, which takes
/// the element with that name under every element above, adds none.
///
/// A single position is not a selection, so this does not compile; it is written `magnetics.flux_loop[0]`:
///
/// ```compile_fail,E0277
/// let magnetics: imas_rs::ids::magnetics::Magnetics = imas_rs::ids::magnetics::Magnetics::default();
/// let _flux_loop = magnetics.flux_loop(0);
/// ```
///
/// Nor does a name, where the elements have none:
///
/// ```compile_fail,E0277
/// let equilibrium: imas_rs::Equilibrium = imas_rs::Equilibrium::default();
/// let _time_slice = equilibrium.time_slice("t0");
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot select from an array of structures",
    label = "give a range, e.g. `..` or `0..2`, or the `name` of one element",
    note = "a single element is reached by indexing the array itself, e.g. `pf_active.coil[0]` rather than `pf_active.coil(0)`",
    note = "a name only selects where the elements have a `name`"
)]
pub trait Selection<'a, T: 'a> {
    /// What the accessor gives from the structure which holds the array
    type Output;
    /// What the accessor gives from a view which has already sliced `D` levels above the array
    type NestedOutput<D: Dimension>;

    /// Select from `items`, the array named `field_path`
    fn select(self, field_path: &'static str, items: &'a [T]) -> Self::Output;

    /// Select from the array which `get` reaches under every element of `elements`, named `field_path`
    fn select_nested<P, D: Dimension>(self, field_path: &'static str, elements: &Elements<'a, P, D>, get: fn(&P) -> &[T]) -> Self::NestedOutput<D>;
}

/// A range of positions selects a slice view over those elements, exactly as slicing a `Vec` does
macro_rules! impl_selection_for_range {
    ($($range:ty),+) => {
        $(
            impl<'a, T: ArrayOfStructuresElement + 'a> Selection<'a, T> for $range {
                type Output = T::SliceView<'a, Ix1>;
                type NestedOutput<D: Dimension> = T::SliceView<'a, D::Larger>;

                fn select(self, _field_path: &'static str, items: &'a [T]) -> Self::Output {
                    T::slice_view(Elements::from_slice(&items[self]))
                }

                fn select_nested<P, D: Dimension>(
                    self,
                    field_path: &'static str,
                    elements: &Elements<'a, P, D>,
                    get: fn(&P) -> &[T],
                ) -> Self::NestedOutput<D> {
                    T::slice_view(elements.nest(field_path, get, self))
                }
            }
        )+
    };
}

impl_selection_for_range!(
    RangeFull,
    Range<usize>,
    RangeFrom<usize>,
    RangeTo<usize>,
    RangeInclusive<usize>,
    RangeToInclusive<usize>
);

/// A name selects the one element with that `name`
impl<'a, T: ArrayOfStructuresElement + Named + 'a> Selection<'a, T> for &str {
    type Output = &'a T;
    type NestedOutput<D: Dimension> = T::SliceView<'a, D>;

    fn select(self, field_path: &'static str, items: &'a [T]) -> &'a T {
        match index_of_name(items, self) {
            Ok(i_named) => &items[i_named],
            Err(reason) => panic!("cannot select from `{field_path}`: {reason}"),
        }
    }

    fn select_nested<P, D: Dimension>(self, field_path: &'static str, elements: &Elements<'a, P, D>, get: fn(&P) -> &[T]) -> Self::NestedOutput<D> {
        T::slice_view(elements.nest_named(field_path, get, self))
    }
}

/// A name held in a `String`, so that `.coil(&name)` works as well as `.coil(name.as_str())`
impl<'a, T: ArrayOfStructuresElement + Named + 'a> Selection<'a, T> for &String {
    type Output = &'a T;
    type NestedOutput<D: Dimension> = T::SliceView<'a, D>;

    fn select(self, field_path: &'static str, items: &'a [T]) -> &'a T {
        self.as_str().select(field_path, items)
    }

    fn select_nested<P, D: Dimension>(self, field_path: &'static str, elements: &Elements<'a, P, D>, get: fn(&P) -> &[T]) -> Self::NestedOutput<D> {
        self.as_str().select_nested(field_path, elements, get)
    }
}

/// The position of the one element of `items` named `name`.
///
/// The data dictionary asks for a name to be unique, so a name which more than one element has is an error, like
/// one which no element has, rather than a choice of the first.
fn index_of_name<T: Named>(items: &[T], name: &str) -> Result<usize, String> {
    let n_items: usize = items.len();
    let mut i_named: Option<usize> = None;
    for i_item in 0..n_items {
        if items[i_item].name() == name {
            if let Some(i_first) = i_named {
                return Err(format!("more than one element is named `{name}`, at positions {i_first} and {i_item}"));
            }
            i_named = Some(i_item);
        }
    }

    match i_named {
        Some(i_item) => Ok(i_item),
        None => {
            let mut names: Vec<&str> = Vec::with_capacity(n_items);
            for i_item in 0..n_items {
                names.push(items[i_item].name());
            }
            Err(format!("no element is named `{name}`; the names are {names:?}"))
        }
    }
}

// ============================================================================
// Re-exports for convenience
// ============================================================================

pub use ndarray::{Array, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, array};
pub use num_complex::Complex64 as Complex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::equilibrium::Equilibrium;
    use crate::ids::magnetics::{Magnetics, MagneticsFluxLoop, MagneticsFluxLoopGreensPfActive};

    /// A distinct value for every (flux loop, PF coil) pair, so that a transposed or mis-ordered
    /// gather cannot pass by accident.
    fn greens_value(i_flux_loop: usize, i_pf: usize) -> f64 {
        10.0 * (i_flux_loop as f64) + (i_pf as f64)
    }

    /// `n_flux_loop` flux loops, each with `n_pf` PF coil Greens entries.
    fn magnetics_with_greens(n_flux_loop: usize, n_pf: usize) -> Magnetics {
        let mut magnetics: Magnetics = Magnetics {
            flux_loop: vec![MagneticsFluxLoop::default(); n_flux_loop],
            ..Magnetics::default()
        };
        for i_flux_loop in 0..n_flux_loop {
            magnetics.flux_loop[i_flux_loop].name = format!("L{i_flux_loop:03}");
            magnetics.flux_loop[i_flux_loop].greens.pf_active = vec![MagneticsFluxLoopGreensPfActive::default(); n_pf];
            for i_pf in 0..n_pf {
                magnetics.flux_loop[i_flux_loop].greens.pf_active[i_pf].name = format!("PF{i_pf}");
                magnetics.flux_loop[i_flux_loop].greens.pf_active[i_pf].value = greens_value(i_flux_loop, i_pf);
            }
        }
        magnetics
    }

    /// Two sliced levels are gained from left to right: the first on the path is the leftmost dimension.
    #[test]
    fn two_sliced_levels_are_gained_from_left_to_right() {
        let n_flux_loop: usize = 3;
        let n_pf: usize = 2;
        let magnetics: Magnetics = magnetics_with_greens(n_flux_loop, n_pf);

        let values: Array2<f64> = magnetics.flux_loop(..).greens.pf_active(..).value.to_array(); // shape = (n_flux_loop, n_pf)
        assert_eq!(values.dim(), (n_flux_loop, n_pf));
        for i_flux_loop in 0..n_flux_loop {
            for i_pf in 0..n_pf {
                assert_eq!(values[[i_flux_loop, i_pf]], greens_value(i_flux_loop, i_pf));
            }
        }

        // Strings gather into a `Vec<String>`, over a single sliced level
        let pf_names: Vec<String> = magnetics.flux_loop[2].greens.pf_active(..).name.to_vec();
        assert_eq!(pf_names, vec!["PF0".to_string(), "PF1".to_string()]);

        // A leaf above the second level is still gathered over the first level only
        let flux_loop_names: Vec<String> = magnetics.flux_loop(..).name.to_vec();
        assert_eq!(flux_loop_names, vec!["L000".to_string(), "L001".to_string(), "L002".to_string()]);

        assert_eq!(magnetics.flux_loop(..).greens.pf_active(..).shape(), vec![n_flux_loop, n_pf]);
        assert_eq!(magnetics.flux_loop(..).greens.pf_active(..).len(), n_flux_loop * n_pf);
    }

    /// A range at each level selects the same elements it would from a single `Vec`.
    #[test]
    fn ranges_apply_at_every_level() {
        let magnetics: Magnetics = magnetics_with_greens(4, 3);

        let values: Array2<f64> = magnetics.flux_loop(1..3).greens.pf_active(1..).value.to_array(); // shape = (n_flux_loop, n_pf)
        assert_eq!(values.dim(), (2, 2));
        assert_eq!(values[[0, 0]], greens_value(1, 1));
        assert_eq!(values[[0, 1]], greens_value(1, 2));
        assert_eq!(values[[1, 0]], greens_value(2, 1));
        assert_eq!(values[[1, 1]], greens_value(2, 2));

        // Slicing from a single element is one level, not two
        let values: Array1<f64> = magnetics.flux_loop[2].greens.pf_active(..).value.to_array();
        assert_eq!(values, array![greens_value(2, 0), greens_value(2, 1), greens_value(2, 2)]);
    }

    #[test]
    #[should_panic(expected = "cannot gather `greens.pf_active`")]
    fn a_jagged_nested_level_panics() {
        let mut magnetics: Magnetics = magnetics_with_greens(3, 2);
        magnetics.flux_loop[1].greens.pf_active.pop();

        let _values: Array2<f64> = magnetics.flux_loop(..).greens.pf_active(..).value.to_array();
    }

    /// A name selects the one element with that name, from the structure which holds the array.
    #[test]
    fn a_name_selects_one_element() {
        let magnetics: Magnetics = magnetics_with_greens(3, 2);

        assert_eq!(magnetics.flux_loop("L001").name, "L001");
        assert_eq!(magnetics.flux_loop("L002").greens.pf_active("PF1").value, greens_value(2, 1));

        // A name held in a `String`
        let flux_loop_name: String = "L000".to_string();
        assert_eq!(magnetics.flux_loop(&flux_loop_name).greens.pf_active("PF0").value, greens_value(0, 0));
    }

    /// A name under every element takes the element with that name under each of them, adding no dimension.
    #[test]
    fn a_name_under_every_element_adds_no_dimension() {
        let magnetics: Magnetics = magnetics_with_greens(3, 2);

        let values: Array1<f64> = magnetics.flux_loop(..).greens.pf_active("PF1").value.to_array(); // shape = (n_flux_loop)
        assert_eq!(values, array![greens_value(0, 1), greens_value(1, 1), greens_value(2, 1)]);

        // A range above a name still adds its own dimension
        let values: Array1<f64> = magnetics.flux_loop(1..).greens.pf_active("PF0").value.to_array(); // shape = (n_flux_loop)
        assert_eq!(values, array![greens_value(1, 0), greens_value(2, 0)]);
    }

    #[test]
    #[should_panic(expected = "cannot select from `flux_loop`: no element is named `L009`; the names are [\"L000\", \"L001\", \"L002\"]")]
    fn a_name_which_no_element_has_panics() {
        let magnetics: Magnetics = magnetics_with_greens(3, 2);

        let _flux_loop: &MagneticsFluxLoop = magnetics.flux_loop("L009");
    }

    /// The data dictionary asks for names to be unique, so a repeated name is an error rather than a choice of the first.
    #[test]
    #[should_panic(expected = "cannot select from `greens.pf_active` under the element at [1]: more than one element is named `PF0`, at positions 0 and 1")]
    fn a_name_which_more_than_one_element_has_panics() {
        let mut magnetics: Magnetics = magnetics_with_greens(3, 2);
        magnetics.flux_loop[1].greens.pf_active[1].name = "PF0".to_string();

        let _values: Array1<f64> = magnetics.flux_loop(..).greens.pf_active("PF0").value.to_array();
    }

    /// Gathering over no outer elements is empty at every level, rather than a panic.
    #[test]
    fn no_outer_elements_gathers_an_empty_array() {
        let magnetics: Magnetics = Magnetics::default();

        let values: Array2<f64> = magnetics.flux_loop(..).greens.pf_active(..).value.to_array();
        assert_eq!(values.dim(), (0, 0));
    }

    /// One sliced level is unchanged: a scalar leaf gathers into an `Array1`.
    #[test]
    fn one_sliced_level_gathers_into_an_array1() {
        let equilibrium: Equilibrium = Equilibrium::with_time(&array![0.1, 0.2, 0.3]);

        let time: Array1<f64> = equilibrium.time_slice(..).time.to_array();
        assert_eq!(time, array![0.1, 0.2, 0.3]);
        assert_eq!(equilibrium.time_slice(1..).time.to_vec(), vec![0.2, 0.3]);
    }

    /// The gathered levels come first, in path order, then the leaf's own dimensions.
    #[test]
    fn arrange_left_to_right_puts_leaf_dimensions_last() {
        let n_level_1: usize = 3;
        let n_level_2: usize = 2;
        let n_leaf: usize = 4;

        // One value per (level 1, level 2, leaf) triple, collected in path order
        let mut values: Vec<f64> = Vec::with_capacity(n_level_1 * n_level_2 * n_leaf);
        for i_level_1 in 0..n_level_1 {
            for i_level_2 in 0..n_level_2 {
                for i_leaf in 0..n_leaf {
                    values.push((100 * i_level_1 + 10 * i_level_2 + i_leaf) as f64);
                }
            }
        }
        let stacked: ArrayD<f64> = ArrayD::from_shape_vec(IxDyn(&[n_level_1 * n_level_2, n_leaf]), values).expect("the shape matches");

        let arranged: ArrayD<f64> = arrange_left_to_right(stacked, &[n_level_1, n_level_2]);

        assert_eq!(arranged.shape(), &[n_level_1, n_level_2, n_leaf]);
        assert!(arranged.is_standard_layout());
        for i_level_1 in 0..n_level_1 {
            for i_level_2 in 0..n_level_2 {
                for i_leaf in 0..n_leaf {
                    assert_eq!(arranged[[i_level_1, i_level_2, i_leaf]], (100 * i_level_1 + 10 * i_level_2 + i_leaf) as f64);
                }
            }
        }
    }
}
