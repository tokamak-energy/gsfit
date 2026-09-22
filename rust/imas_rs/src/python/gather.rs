//! Reading a leaf out of an IDS: the value it comes back as, and gathering one leaf across one or
//! more sliced arrays of structures.
//!
//! The generated `<ids>_paths.rs` tables call [`gather`] from every leaf's reader.
//!
//! The gathered dimensions are gained from left to right, exactly as the Rust accumulators do: one
//! per sliced array of structures, in the order they appear on the path, followed by the leaf's own
//! dimensions. So `flux_loop[:].greens.pf_active[:].value` is `(n_flux_loop, n_pf)` and
//! `time_slice[:].profiles_2d[0].psi` is `(n_time, n_z, n_r)`. The rule itself lives in
//! [`arrange_left_to_right`], which both share.

use super::IndexSpec;
use crate::dd_base_types::arrange_left_to_right;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayViewMutD, Axis, IxDyn, Slice};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};

// ============================================================================
// Values read back out of the IDS
// ============================================================================

/// A value gathered out of the IDS, before conversion to Python.
///
/// Arrays are held as `ArrayD` rather than a rank-specific type, because gathering a leaf across
/// sliced arrays of structures raises its rank by one per sliced level, and the data dictionary has
/// leaves from `FLT_0D` up to `FLT_4D`.
///
/// An unset leaf is not a special case here: it already holds its IMAS empty value (NaN,
/// `EMPTY_INT`, an empty string or an empty array), which is what Python should see.
pub enum Value {
    Flt0d(f64),
    FltNd(ArrayD<f64>),
    Int0d(i32),
    IntNd(ArrayD<i32>),
    Str0d(String),
    /// A single `STR_1D` leaf, which reads back as a `list[str]`
    StrList(Vec<String>),
    /// Strings gathered across sliced arrays of structures, which read back as a numpy array of
    /// `str`, so that they have the same shape as any other gathered leaf
    StrNd(ArrayD<String>),
}

impl Value {
    pub(super) fn into_python<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Value::Flt0d(value) => Ok(PyFloat::new(py, value).into_any()),
            Value::FltNd(values) => Ok(values.into_pyarray(py).into_any()),
            Value::Int0d(value) => value.into_bound_py_any(py),
            Value::IntNd(values) => Ok(values.into_pyarray(py).into_any()),
            Value::Str0d(value) => value.into_bound_py_any(py),
            Value::StrList(values) => values.into_bound_py_any(py),
            Value::StrNd(values) => string_array_into_python(py, values),
        }
    }
}

/// Build a numpy `str` array from a string array.
///
/// numpy's fixed-width unicode dtype has no Rust counterpart, so the strings go through a flat
/// Python list and numpy is asked to build and reshape it. Reshaping a flat list, rather than
/// handing numpy nested lists, keeps a zero-length dimension, which nested lists cannot express.
fn string_array_into_python<'py>(py: Python<'py>, values: ArrayD<String>) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = values.shape().to_vec();
    let flat: Vec<String> = values.as_standard_layout().iter().cloned().collect();

    let keyword_arguments: Bound<'py, PyDict> = PyDict::new(py);
    keyword_arguments.set_item("dtype", "str")?;

    let numpy_module: Bound<'py, PyModule> = py.import("numpy")?;
    let flat_array: Bound<'py, PyAny> = numpy_module.getattr("array")?.call((PyList::new(py, flat)?,), Some(&keyword_arguments))?;

    flat_array.call_method1("reshape", (shape,))
}

// ============================================================================
// Gathering
// ============================================================================

/// A leaf type that can be read out of the IDS, either singly or gathered across sliced arrays of
/// structures.
///
/// Gathering raises the rank by one per sliced level: an `FLT_0D` leaf gathered over `time_slice[:]`
/// becomes 1D, an `FLT_2D` leaf becomes 3D.
pub trait Gatherable: Clone {
    /// No array-of-structures level was sliced, so there is exactly one value.
    fn one(value: Self) -> Result<Value, String>;

    /// At least one level was sliced. `values` holds one value per gathered element, in path order
    /// (the first array of structures on the path varying slowest), and `gathered_shape` is the
    /// number of elements selected at each sliced level, also in path order.
    fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String>;
}

/// Lay scalar values out along one axis, ready for `arrange_left_to_right`.
fn scalars_along_one_axis<A>(values: Vec<A>) -> ArrayD<A> {
    let n_values: usize = values.len();
    ArrayD::from_shape_vec(IxDyn(&[n_values]), values).expect("a flat vector has exactly one axis")
}

impl Gatherable for f64 {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Flt0d(value))
    }

    fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String> {
        Ok(Value::FltNd(arrange_left_to_right(scalars_along_one_axis(values), gathered_shape)))
    }
}

impl Gatherable for i32 {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Int0d(value))
    }

    fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String> {
        Ok(Value::IntNd(arrange_left_to_right(scalars_along_one_axis(values), gathered_shape)))
    }
}

impl Gatherable for String {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Str0d(value))
    }

    fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String> {
        Ok(Value::StrNd(arrange_left_to_right(scalars_along_one_axis(values), gathered_shape)))
    }
}

impl Gatherable for Vec<String> {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::StrList(value))
    }

    /// Lists of different lengths are padded with empty strings, which is how IMAS marks an unset
    /// string, just as a float array is padded with NaN.
    fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String> {
        let n_values: usize = values.len();

        let mut n_strings_max: usize = 0;
        for i_value in 0..n_values {
            n_strings_max = n_strings_max.max(values[i_value].len());
        }

        let mut stacked: Array2<String> = Array2::from_elem((n_values, n_strings_max), String::new());
        for i_value in 0..n_values {
            let n_strings: usize = values[i_value].len();
            for i_string in 0..n_strings {
                stacked[[i_value, i_string]] = values[i_value][i_string].clone();
            }
        }

        Ok(Value::StrNd(arrange_left_to_right(stacked.into_dyn(), gathered_shape)))
    }
}

/// Stack float arrays along a new leading axis, padding with NaN where the gathered
/// arrays differ in shape (as `boundary/outline/r` does, having a different number of
/// points per time slice). An unset element is empty, so it comes out as all NaN.
///
/// `n_dimensions` is the rank of the leaf, which is known from its type, so that gathering over no
/// elements still gives an array of the right rank.
fn stack_float_arrays(values: Vec<ArrayD<f64>>, n_dimensions: usize) -> ArrayD<f64> {
    let n_values: usize = values.len();

    let mut max_shape: Vec<usize> = vec![0; n_dimensions];
    for array in &values {
        for i_dimension in 0..n_dimensions {
            max_shape[i_dimension] = max_shape[i_dimension].max(array.shape()[i_dimension]);
        }
    }

    let mut shape: Vec<usize> = Vec::with_capacity(n_dimensions + 1);
    shape.push(n_values);
    shape.extend_from_slice(&max_shape);

    let mut stacked: ArrayD<f64> = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
    for i_value in 0..n_values {
        let element_shape: Vec<usize> = values[i_value].shape().to_vec();
        let mut destination: ArrayViewMutD<f64> = stacked.index_axis_mut(Axis(0), i_value);
        let mut destination: ArrayViewMutD<f64> =
            destination.slice_each_axis_mut(|axis: ndarray::AxisDescription| Slice::from(0..element_shape[axis.axis.index()]));
        destination.assign(&values[i_value]);
    }

    stacked
}

/// Stack integer arrays along a new leading axis.
///
/// Unlike the float case there is no padding value, so the shapes must agree. An unset
/// element is empty, so one unset element among set ones is refused rather than guessed at.
fn stack_integer_arrays(values: Vec<ArrayD<i32>>, n_dimensions: usize) -> Result<ArrayD<i32>, String> {
    let n_values: usize = values.len();

    let mut shape: Vec<usize> = vec![0; n_dimensions];
    for i_value in 0..n_values {
        if i_value == 0 {
            shape = values[i_value].shape().to_vec();
        } else if values[i_value].shape() != shape.as_slice() {
            return Err(format!(
                "cannot be gathered: element {i_value} has shape {:?} but element 0 has shape {:?} \
                 (an unset element is empty), and an integer array has no NaN to pad with",
                values[i_value].shape(),
                shape
            ));
        }
    }

    let mut full_shape: Vec<usize> = Vec::with_capacity(shape.len() + 1);
    full_shape.push(n_values);
    full_shape.extend_from_slice(&shape);

    let mut stacked: ArrayD<i32> = ArrayD::zeros(IxDyn(&full_shape));
    for i_value in 0..n_values {
        stacked.index_axis_mut(Axis(0), i_value).assign(&values[i_value]);
    }

    Ok(stacked)
}

/// `Gatherable` for every array leaf type, which differ only in how they reach `ArrayD`, how they
/// are stacked, and their rank.
macro_rules! impl_gatherable_for_arrays {
    ($($array_type:ty => $value_variant:ident, $stack:expr, $n_dimensions:literal;)*) => {
        $(
            impl Gatherable for $array_type {
                fn one(value: Self) -> Result<Value, String> {
                    return Ok(Value::$value_variant(value.into_dyn()));
                }

                fn stack(values: Vec<Self>, gathered_shape: &[usize]) -> Result<Value, String> {
                    let mut as_dynamic: Vec<ArrayD<_>> = Vec::with_capacity(values.len());
                    for value in values {
                        as_dynamic.push(value.into_dyn());
                    }
                    let stacked = $stack(as_dynamic, $n_dimensions)?;
                    return Ok(Value::$value_variant(arrange_left_to_right(stacked, gathered_shape)));
                }
            }
        )*
    };
}

impl_gatherable_for_arrays! {
    Array1<f64> => FltNd, |values, n_dimensions| Ok::<_, String>(stack_float_arrays(values, n_dimensions)), 1;
    Array2<f64> => FltNd, |values, n_dimensions| Ok::<_, String>(stack_float_arrays(values, n_dimensions)), 2;
    Array3<f64> => FltNd, |values, n_dimensions| Ok::<_, String>(stack_float_arrays(values, n_dimensions)), 3;
    Array4<f64> => FltNd, |values, n_dimensions| Ok::<_, String>(stack_float_arrays(values, n_dimensions)), 4;
    Array1<i32> => IntNd, stack_integer_arrays, 1;
    Array2<i32> => IntNd, stack_integer_arrays, 2;
}

/// The length of one array-of-structures level, given the elements already chosen at
/// the levels above it. Generated per array-of-structures chain.
pub type LengthOf<I> = fn(&I, usize, &[usize]) -> Option<usize>;

/// Used by leaves that sit outside any array of structures.
pub fn no_levels<I>(_ids: &I, _level: usize, _at: &[usize]) -> Option<usize> {
    None
}

/// Read a leaf, applying one index selection per array-of-structures level on its path.
///
/// Any number of levels may keep their dimension (be indexed with a slice or a list). The result
/// gains its dimensions from left to right: the sliced levels in the order they appear on the path,
/// then the leaf's own dimensions. A sliced level must select the same number of elements under every
/// element of the levels above it, because otherwise the result would be jagged; that is refused
/// with an error naming the two elements which disagree.
pub fn gather<I, T: Gatherable>(
    ids: &I,
    indices: &[IndexSpec],
    n_levels: usize,
    length_of: LengthOf<I>,
    project: fn(&I, &[usize]) -> T,
) -> Result<Value, String> {
    if indices.len() != n_levels {
        return Err(format!(
            "expected {n_levels} array-of-structures index selection(s) along this path, found {}",
            indices.len()
        ));
    }

    let mut sliced_levels: Vec<usize> = Vec::with_capacity(n_levels);
    for i_level in 0..n_levels {
        if !indices[i_level].is_scalar() {
            sliced_levels.push(i_level);
        }
    }

    // The number of elements each sliced level selected, and the elements above it where that was
    // first seen, so that a jagged level can be reported against it
    let mut selected_lengths: Vec<Option<(usize, Vec<usize>)>> = vec![None; n_levels];
    let mut values: Vec<T> = Vec::new();
    let mut at: Vec<usize> = Vec::with_capacity(n_levels);
    collect_values(ids, indices, length_of, project, &mut at, &mut selected_lengths, &mut values)?;

    if sliced_levels.is_empty() {
        let value: T = values.pop().expect("with no sliced level exactly one element is selected");
        return T::one(value);
    }

    // A sliced level which was never reached, because a level above it selected nothing, is empty
    let n_sliced: usize = sliced_levels.len();
    let mut gathered_shape: Vec<usize> = Vec::with_capacity(n_sliced);
    for i_sliced in 0..n_sliced {
        let n_selected: usize = match &selected_lengths[sliced_levels[i_sliced]] {
            Some((n_selected, _)) => *n_selected,
            None => 0,
        };
        gathered_shape.push(n_selected);
    }

    T::stack(values, &gathered_shape)
}

/// Visit every selected element, depth first in path order, reading the leaf at the bottom.
///
/// Levels are resolved per element, because a nested array of structures can have a different
/// length under each element above it. The recursion is one call per level, so its depth is
/// bounded by the number of levels on the path.
fn collect_values<I, T>(
    ids: &I,
    indices: &[IndexSpec],
    length_of: LengthOf<I>,
    project: fn(&I, &[usize]) -> T,
    at: &mut Vec<usize>,
    selected_lengths: &mut [Option<(usize, Vec<usize>)>],
    values: &mut Vec<T>,
) -> Result<(), String> {
    let i_level: usize = at.len();
    if i_level == indices.len() {
        values.push(project(ids, at));
        return Ok(());
    }

    let n_elements: usize = match length_of(ids, i_level, at) {
        Some(n_elements) => n_elements,
        None => return Err(format!("array-of-structures level {i_level} could not be reached")),
    };
    let selected: Vec<usize> = indices[i_level].resolve(n_elements)?;
    let n_selected: usize = selected.len();

    if !indices[i_level].is_scalar() {
        match &selected_lengths[i_level] {
            None => selected_lengths[i_level] = Some((n_selected, at.clone())),
            Some((n_selected_first, at_first)) => {
                if n_selected != *n_selected_first {
                    return Err(format!(
                        "cannot be gathered: array-of-structures level {i_level} selects {n_selected} element(s) under {at:?} \
                         but {n_selected_first} under {at_first:?} (indices in path order), so the result would be jagged"
                    ));
                }
            }
        }
    }

    for i_selected in 0..n_selected {
        at.push(selected[i_selected]);
        collect_values(ids, indices, length_of, project, at, selected_lengths, values)?;
        at.pop();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::equilibrium::{Equilibrium, EquilibriumProfiles2d};

    /// `time_slice` is the only array-of-structures level used by these tests.
    fn time_slice_lengths(equilibrium: &Equilibrium, level: usize, _at: &[usize]) -> Option<usize> {
        match level {
            0 => return Some(equilibrium.time_slice.len()),
            _ => return None,
        }
    }

    /// `time_slice` and the `profiles_2d` under each time slice.
    fn time_slice_profiles_2d_lengths(equilibrium: &Equilibrium, level: usize, at: &[usize]) -> Option<usize> {
        match level {
            0 => return Some(equilibrium.time_slice.len()),
            1 => return Some(equilibrium.time_slice.get(at[0])?.profiles_2d.len()),
            _ => return None,
        }
    }

    fn project_ip(equilibrium: &Equilibrium, at: &[usize]) -> f64 {
        return equilibrium.time_slice[at[0]].global_quantities.ip;
    }

    fn gather_ip(equilibrium: &Equilibrium, index: IndexSpec) -> Result<Value, String> {
        return gather(equilibrium, &[index], 1, time_slice_lengths, project_ip);
    }

    fn everything() -> IndexSpec {
        return IndexSpec::Slice {
            start: None,
            stop: None,
            step: None,
        };
    }

    /// Read a gathered array, or fail loudly.
    fn expect_float_array(value: Value) -> ArrayD<f64> {
        match value {
            Value::FltNd(values) => return values,
            _ => panic!("expected a gathered float array"),
        }
    }

    /// An equilibrium whose `ip` is `1.0e6 * i_time`, except the last slice which is unset.
    fn equilibrium_with_ip(n_time: usize) -> Equilibrium {
        let mut equilibrium: Equilibrium = Equilibrium::with_size(n_time);
        for i_time in 0..n_time {
            if i_time + 1 == n_time {
                continue; // left unset, to check the NaN behaviour
            }
            equilibrium.time_slice[i_time].global_quantities.ip = 1.0e6 * (i_time as f64);
        }
        return equilibrium;
    }

    /// An equilibrium with `n_profiles_2d` entries in every time slice, each with a distinct `psi`
    /// value per (time slice, profile) pair in its single grid point.
    fn equilibrium_with_profiles_2d(n_time: usize, n_profiles_2d: usize) -> Equilibrium {
        let mut equilibrium: Equilibrium = Equilibrium::with_size(n_time);
        for i_time in 0..n_time {
            equilibrium.time_slice[i_time].profiles_2d = vec![EquilibriumProfiles2d::default(); n_profiles_2d];
            for i_profiles_2d in 0..n_profiles_2d {
                equilibrium.time_slice[i_time].profiles_2d[i_profiles_2d].psi = Array2::from_elem((1, 1), (10 * i_time + i_profiles_2d) as f64);
                equilibrium.time_slice[i_time].profiles_2d[i_profiles_2d].r#type.name = format!("T{i_time}P{i_profiles_2d}");
            }
        }
        return equilibrium;
    }

    fn project_psi_2d(equilibrium: &Equilibrium, at: &[usize]) -> Array2<f64> {
        return equilibrium.time_slice[at[0]].profiles_2d[at[1]].psi.clone();
    }

    fn project_profiles_2d_type_name(equilibrium: &Equilibrium, at: &[usize]) -> String {
        return equilibrium.time_slice[at[0]].profiles_2d[at[1]].r#type.name.clone();
    }

    /// A scalar index drops the dimension; every other selection keeps it.
    #[test]
    fn gather_follows_the_shape_of_the_index() {
        let equilibrium: Equilibrium = equilibrium_with_ip(5);

        match gather_ip(&equilibrium, IndexSpec::One(3)).expect("index 3 is in range") {
            Value::Flt0d(value) => assert_eq!(value, 3.0e6),
            _ => panic!("`time_slice[3]` should give a scalar"),
        }

        // A negative index counts from the end, as in Python.
        match gather_ip(&equilibrium, IndexSpec::One(-2)).expect("index -2 is in range") {
            Value::Flt0d(value) => assert_eq!(value, 3.0e6),
            _ => panic!("`time_slice[-2]` should give a scalar"),
        }

        let gathered: ArrayD<f64> = expect_float_array(gather_ip(&equilibrium, everything()).expect("`[:]` should resolve"));
        assert_eq!(gathered.shape(), &[5]);
        assert_eq!(gathered[[0]], 0.0);
        assert_eq!(gathered[[3]], 3.0e6);

        let gathered: ArrayD<f64> = expect_float_array(gather_ip(&equilibrium, IndexSpec::Many(vec![0, 2])).expect("a list index should resolve"));
        assert_eq!(gathered.shape(), &[2]);
        assert_eq!(gathered[[1]], 2.0e6);
    }

    /// An unset float reads back as NaN, so a failed time slice does not stop the gather.
    #[test]
    fn an_unset_leaf_gathers_as_nan() {
        let equilibrium: Equilibrium = equilibrium_with_ip(5);
        let gathered: ArrayD<f64> = expect_float_array(gather_ip(&equilibrium, everything()).expect("`[:]` should resolve"));

        assert!(gathered[[4]].is_nan(), "the unset last slice should read back as NaN");
        assert!(!gathered[[3]].is_nan(), "a set slice should not be NaN");
    }

    #[test]
    fn an_out_of_range_index_is_an_error_not_a_panic() {
        let equilibrium: Equilibrium = equilibrium_with_ip(5);

        assert!(gather_ip(&equilibrium, IndexSpec::One(99)).is_err());
        assert!(gather_ip(&equilibrium, IndexSpec::One(-99)).is_err());
        assert!(gather_ip(&equilibrium, IndexSpec::Many(vec![0, 99])).is_err());
    }

    /// A 1D leaf gathered over a sliced level becomes 2D, with the leaf's own dimension last,
    /// NaN-padded to the longest.
    #[test]
    fn ragged_array_leaves_are_nan_padded() {
        let mut equilibrium: Equilibrium = Equilibrium::with_size(3);
        equilibrium.time_slice[0].profiles_1d.psi = Array1::from(vec![1.0, 2.0]);
        equilibrium.time_slice[1].profiles_1d.psi = Array1::from(vec![3.0, 4.0, 5.0]);
        // time_slice[2] is left unset, so its profile is empty

        fn project_psi(equilibrium: &Equilibrium, at: &[usize]) -> Array1<f64> {
            return equilibrium.time_slice[at[0]].profiles_1d.psi.clone();
        }

        let gathered: ArrayD<f64> =
            expect_float_array(gather(&equilibrium, &[everything()], 1, time_slice_lengths, project_psi).expect("`[:]` should resolve"));

        // shape = [n_time, n_psi]
        assert_eq!(gathered.shape(), &[3, 3], "padded to the longest profile");
        assert_eq!(gathered[[0, 0]], 1.0);
        assert_eq!(gathered[[1, 2]], 5.0);
        assert!(gathered[[0, 2]].is_nan(), "the shorter profile is padded with NaN");
        assert!(gathered[[2, 0]].is_nan(), "the unset slice is all NaN");
    }

    /// Two sliced levels are gained from left to right, before the leaf's own dimensions.
    #[test]
    fn two_sliced_levels_are_gained_from_left_to_right() {
        let n_time: usize = 3;
        let n_profiles_2d: usize = 2;
        let equilibrium: Equilibrium = equilibrium_with_profiles_2d(n_time, n_profiles_2d);

        let gathered: ArrayD<f64> = expect_float_array(
            gather(&equilibrium, &[everything(), everything()], 2, time_slice_profiles_2d_lengths, project_psi_2d).expect("both levels have one length"),
        );

        // shape = [n_time, n_profiles_2d, n_z, n_r]
        assert_eq!(gathered.shape(), &[n_time, n_profiles_2d, 1, 1]);
        for i_time in 0..n_time {
            for i_profiles_2d in 0..n_profiles_2d {
                assert_eq!(gathered[[i_time, i_profiles_2d, 0, 0]], (10 * i_time + i_profiles_2d) as f64);
            }
        }

        // A selection at each level picks the same elements it would on its own
        let gathered: ArrayD<f64> = expect_float_array(
            gather(
                &equilibrium,
                &[IndexSpec::Many(vec![2, 0]), IndexSpec::One(1)],
                2,
                time_slice_profiles_2d_lengths,
                project_psi_2d,
            )
            .expect("the selections are in range"),
        );
        assert_eq!(gathered.shape(), &[2, 1, 1]);
        assert_eq!(gathered[[0, 0, 0]], 21.0);
        assert_eq!(gathered[[1, 0, 0]], 1.0);
    }

    /// Strings gathered over two levels come back with the same shape as any other leaf.
    #[test]
    fn strings_gather_over_two_levels() {
        let equilibrium: Equilibrium = equilibrium_with_profiles_2d(3, 2);

        let gathered: Value = gather(
            &equilibrium,
            &[everything(), everything()],
            2,
            time_slice_profiles_2d_lengths,
            project_profiles_2d_type_name,
        )
        .expect("both levels have one length");

        match gathered {
            Value::StrNd(names) => {
                // shape = [n_time, n_profiles_2d]
                assert_eq!(names.shape(), &[3, 2]);
                assert_eq!(names[[2, 1]], "T2P1");
            }
            _ => panic!("expected a gathered string array"),
        }
    }

    /// A level whose length differs from one element to the next cannot be gathered into an array.
    #[test]
    fn a_jagged_level_is_an_error_not_a_panic() {
        let mut equilibrium: Equilibrium = equilibrium_with_profiles_2d(3, 2);
        equilibrium.time_slice[1].profiles_2d.pop();

        let result: Result<Value, String> = gather(&equilibrium, &[everything(), everything()], 2, time_slice_profiles_2d_lengths, project_psi_2d);

        let message: String = result.err().expect("a jagged level should be refused");
        assert!(message.contains("would be jagged"), "unexpected message: {message}");
        assert!(message.contains("under [1]"), "the message should name the element which disagrees: {message}");

        // Indexing the jagged level with an integer is still fine
        assert!(
            gather(
                &equilibrium,
                &[everything(), IndexSpec::One(0)],
                2,
                time_slice_profiles_2d_lengths,
                project_psi_2d
            )
            .is_ok()
        );
    }
}
