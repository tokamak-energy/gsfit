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

impl<T, U: Clone, D: Dimension> Accumulator<'_, T, U, D> {
    /// Gather every value into an array whose dimensions are gained from left to right, e.g.
    /// `flux_loop(..).greens.pf_active(..).value.to_array()` is `(n_flux_loop, n_pf)`.
    pub fn to_array(&self) -> Array<U, D> {
        self.elements.gather(self.project)
    }
}

impl<T, U> Accumulator<'_, T, U, Ix1> {
    /// Gather every value into a `Vec`. Only offered for a single sliced level, where there is no
    /// question of which order the values come in.
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
// Re-exports for convenience
// ============================================================================

pub use ndarray::{Array, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, array};
pub use num_complex::Complex64 as Complex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::equilibrium::{Equilibrium, EquilibriumGreensPfPassive, EquilibriumGreensPfPassiveDof};
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

        // Strings gather the same way
        let names: Array2<String> = magnetics.flux_loop(..).greens.pf_active(..).name.to_array();
        assert_eq!(names.dim(), (n_flux_loop, n_pf));
        assert_eq!(names[[2, 1]], "PF1");

        // A leaf above the second level is still gathered over the first level only
        let flux_loop_names: Array1<String> = magnetics.flux_loop(..).name.to_array();
        assert_eq!(flux_loop_names, array!["L000".to_string(), "L001".to_string(), "L002".to_string()]);

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

    /// Gathering over no outer elements is empty at every level, rather than a panic.
    #[test]
    fn no_outer_elements_gathers_an_empty_array() {
        let magnetics: Magnetics = Magnetics::default();

        let values: Array2<f64> = magnetics.flux_loop(..).greens.pf_active(..).value.to_array();
        assert_eq!(values.dim(), (0, 0));
    }

    /// A nested array of structures under a nested array of structures, as in the equilibrium Greens tables.
    #[test]
    fn greens_passive_degrees_of_freedom_gather_over_two_levels() {
        let n_passive: usize = 2;
        let n_dof: usize = 3;
        let mut equilibrium: Equilibrium = Equilibrium::default();
        equilibrium.greens.pf_passive = vec![EquilibriumGreensPfPassive::default(); n_passive];
        for i_passive in 0..n_passive {
            equilibrium.greens.pf_passive[i_passive].dof = vec![EquilibriumGreensPfPassiveDof::default(); n_dof];
            for i_dof in 0..n_dof {
                equilibrium.greens.pf_passive[i_passive].dof[i_dof].name = format!("P{i_passive}_EIG_{i_dof:02}");
            }
        }

        let names: Array2<String> = equilibrium.greens.pf_passive(..).dof(..).name.to_array(); // shape = (n_passive, n_dof)
        assert_eq!(names.dim(), (n_passive, n_dof));
        assert_eq!(names[[1, 2]], "P1_EIG_02");
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
