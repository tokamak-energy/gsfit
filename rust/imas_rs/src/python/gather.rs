//! Reading a leaf out of an IDS: the value it comes back as, and gathering one leaf across a
//! sliced array of structures.
//!
//! The generated `<ids>_paths.rs` tables call [`gather`] from every leaf's reader.

use super::IndexSpec;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayViewMutD, Axis, IxDyn, Slice};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

// ============================================================================
// Values read back out of the IDS
// ============================================================================

/// A value gathered out of the IDS, before conversion to Python.
///
/// Arrays are held as `ArrayD` rather than a rank-specific type, because gathering a
/// leaf across a sliced array of structures raises its rank by one and the data
/// dictionary has leaves from `FLT_0D` up to `FLT_4D`.
///
/// An unset leaf is not a special case here: it already holds its IMAS empty value (NaN,
/// `EMPTY_INT`, an empty string or an empty array), which is what Python should see.
pub enum Value {
    Flt0d(f64),
    FltNd(ArrayD<f64>),
    Int0d(i32),
    IntNd(ArrayD<i32>),
    Str0d(String),
    StrList(Vec<String>),
    StrNested(Vec<Vec<String>>),
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
            Value::StrNested(values) => values.into_bound_py_any(py),
        }
    }
}

// ============================================================================
// Gathering
// ============================================================================

/// A leaf type that can be read out of the IDS, either singly or gathered across a
/// sliced array of structures.
///
/// Gathering raises the rank by one: an `FLT_0D` leaf gathered over `time_slice[:]`
/// becomes 1D, an `FLT_2D` leaf becomes 3D.
pub trait Gatherable: Clone {
    /// No array-of-structures level was sliced, so there is exactly one value.
    fn one(value: Self) -> Result<Value, String>;

    /// One level was sliced: stack the values along a new leading axis.
    fn stack(values: Vec<Self>) -> Result<Value, String>;
}

impl Gatherable for f64 {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Flt0d(value))
    }

    fn stack(values: Vec<Self>) -> Result<Value, String> {
        Ok(Value::FltNd(Array1::from_vec(values).into_dyn()))
    }
}

impl Gatherable for i32 {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Int0d(value))
    }

    fn stack(values: Vec<Self>) -> Result<Value, String> {
        Ok(Value::IntNd(Array1::from_vec(values).into_dyn()))
    }
}

impl Gatherable for String {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::Str0d(value))
    }

    fn stack(values: Vec<Self>) -> Result<Value, String> {
        Ok(Value::StrList(values))
    }
}

impl Gatherable for Vec<String> {
    fn one(value: Self) -> Result<Value, String> {
        Ok(Value::StrList(value))
    }

    fn stack(values: Vec<Self>) -> Result<Value, String> {
        // Strings cannot be padded, so a list of lists keeps the ragged shape.
        Ok(Value::StrNested(values))
    }
}

/// Stack float arrays along a new leading axis, padding with NaN where the gathered
/// arrays differ in shape (as `boundary/outline/r` does, having a different number of
/// points per time slice). An unset element is empty, so it comes out as all NaN.
fn stack_float_arrays(values: Vec<ArrayD<f64>>) -> Result<Value, String> {
    let n_values: usize = values.len();

    // Every element came from the same leaf type, so they all have the same rank.
    let n_dimensions: usize = match values.first() {
        Some(array) => array.ndim(),
        None => 0,
    };

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

    Ok(Value::FltNd(stacked))
}

/// Stack integer arrays along a new leading axis.
///
/// Unlike the float case there is no padding value, so the shapes must agree. An unset
/// element is empty, so one unset element among set ones is refused rather than guessed at.
fn stack_integer_arrays(values: Vec<ArrayD<i32>>) -> Result<Value, String> {
    let n_values: usize = values.len();

    let mut shape: Vec<usize> = Vec::new();
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

    Ok(Value::IntNd(stacked))
}

/// `Gatherable` for every array leaf type, which differ only in how they reach `ArrayD`.
macro_rules! impl_gatherable_for_arrays {
    ($($array_type:ty => $value_variant:ident, $stack_function:ident;)*) => {
        $(
            impl Gatherable for $array_type {
                fn one(value: Self) -> Result<Value, String> {
                    return Ok(Value::$value_variant(value.into_dyn()));
                }

                fn stack(values: Vec<Self>) -> Result<Value, String> {
                    let mut as_dynamic: Vec<ArrayD<_>> = Vec::with_capacity(values.len());
                    for value in values {
                        as_dynamic.push(value.into_dyn());
                    }
                    return $stack_function(as_dynamic);
                }
            }
        )*
    };
}

impl_gatherable_for_arrays! {
    Array1<f64> => FltNd, stack_float_arrays;
    Array2<f64> => FltNd, stack_float_arrays;
    Array3<f64> => FltNd, stack_float_arrays;
    Array4<f64> => FltNd, stack_float_arrays;
    Array1<i32> => IntNd, stack_integer_arrays;
    Array2<i32> => IntNd, stack_integer_arrays;
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
/// At most one level may keep its dimension (be indexed with a slice or a list). Allowing
/// two would make the result ragged in a way that has no single sensible shape - the
/// second level can have a different length under each element of the first - so it is
/// refused with an explanation instead of guessed at.
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

    // Find the one level, if any, that keeps its dimension.
    let mut sliced_level: Option<usize> = None;
    for i_level in 0..n_levels {
        if !indices[i_level].is_scalar() {
            if sliced_level.is_some() {
                return Err("only one array-of-structures level may be sliced at a time; index the others with an integer".to_string());
            }
            sliced_level = Some(i_level);
        }
    }

    /// Resolve one level that is known to select a single element.
    fn resolve_single<I>(ids: &I, indices: &[IndexSpec], i_level: usize, at: &[usize], length_of: LengthOf<I>) -> Result<usize, String> {
        let n_elements: usize = match length_of(ids, i_level, at) {
            Some(n_elements) => n_elements,
            None => return Err(format!("array-of-structures level {i_level} could not be reached")),
        };
        let resolved: Vec<usize> = indices[i_level].resolve(n_elements)?;
        Ok(resolved[0])
    }

    match sliced_level {
        None => {
            let mut at: Vec<usize> = Vec::with_capacity(n_levels);
            for i_level in 0..n_levels {
                let element: usize = resolve_single(ids, indices, i_level, &at, length_of)?;
                at.push(element);
            }
            T::one(project(ids, &at))
        }
        Some(sliced) => {
            let mut prefix: Vec<usize> = Vec::with_capacity(n_levels);
            for i_level in 0..sliced {
                let element: usize = resolve_single(ids, indices, i_level, &prefix, length_of)?;
                prefix.push(element);
            }

            let n_elements: usize = match length_of(ids, sliced, &prefix) {
                Some(n_elements) => n_elements,
                None => return Err(format!("array-of-structures level {sliced} could not be reached")),
            };
            let selected: Vec<usize> = indices[sliced].resolve(n_elements)?;

            let mut values: Vec<T> = Vec::with_capacity(selected.len());
            for i_selected in 0..selected.len() {
                let mut at: Vec<usize> = prefix.clone();
                at.push(selected[i_selected]);
                // Levels below the sliced one are resolved per element, because their
                // length can differ from one element to the next.
                for i_level in (sliced + 1)..n_levels {
                    let element: usize = resolve_single(ids, indices, i_level, &at, length_of)?;
                    at.push(element);
                }
                values.push(project(ids, &at));
            }

            T::stack(values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::equilibrium::Equilibrium;

    /// `time_slice` is the only array-of-structures level used by these tests.
    fn time_slice_lengths(equilibrium: &Equilibrium, level: usize, _at: &[usize]) -> Option<usize> {
        match level {
            0 => return Some(equilibrium.time_slice.len()),
            _ => return None,
        }
    }

    fn project_ip(equilibrium: &Equilibrium, at: &[usize]) -> f64 {
        return equilibrium.time_slice[at[0]].global_quantities.ip;
    }

    fn gather_ip(equilibrium: &Equilibrium, index: IndexSpec) -> Result<Value, String> {
        return gather(equilibrium, &[index], 1, time_slice_lengths, project_ip);
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

        let spec: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: None,
        };
        let gathered: ArrayD<f64> = expect_float_array(gather_ip(&equilibrium, spec).expect("`[:]` should resolve"));
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
        let spec: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: None,
        };
        let gathered: ArrayD<f64> = expect_float_array(gather_ip(&equilibrium, spec).expect("`[:]` should resolve"));

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

    /// A 1D leaf gathered over a sliced level becomes 2D, NaN-padded to the longest.
    #[test]
    fn ragged_array_leaves_are_nan_padded() {
        let mut equilibrium: Equilibrium = Equilibrium::with_size(3);
        equilibrium.time_slice[0].profiles_1d.psi = Array1::from(vec![1.0, 2.0]);
        equilibrium.time_slice[1].profiles_1d.psi = Array1::from(vec![3.0, 4.0, 5.0]);
        // time_slice[2] is left unset, so its profile is empty

        fn project_psi(equilibrium: &Equilibrium, at: &[usize]) -> Array1<f64> {
            return equilibrium.time_slice[at[0]].profiles_1d.psi.clone();
        }

        let spec: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: None,
        };
        let gathered: ArrayD<f64> = expect_float_array(gather(&equilibrium, &[spec], 1, time_slice_lengths, project_psi).expect("`[:]` should resolve"));

        assert_eq!(gathered.shape(), &[3, 3], "padded to the longest profile");
        assert_eq!(gathered[[0, 0]], 1.0);
        assert_eq!(gathered[[1, 2]], 5.0);
        assert!(gathered[[0, 2]].is_nan(), "the shorter profile is padded with NaN");
        assert!(gathered[[2, 0]].is_nan(), "the unset slice is all NaN");
    }

    /// Slicing two levels at once has no single sensible shape, so it is refused.
    #[test]
    fn slicing_two_levels_at_once_is_refused() {
        let equilibrium: Equilibrium = Equilibrium::with_size(2);

        fn two_levels(equilibrium: &Equilibrium, level: usize, at: &[usize]) -> Option<usize> {
            match level {
                0 => return Some(equilibrium.time_slice.len()),
                1 => return Some(equilibrium.time_slice.get(at[0])?.profiles_2d.len()),
                _ => return None,
            }
        }

        fn project_psi_2d(equilibrium: &Equilibrium, at: &[usize]) -> Array2<f64> {
            return equilibrium.time_slice[at[0]].profiles_2d[at[1]].psi.clone();
        }

        let everything: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: None,
        };
        let result: Result<Value, String> = gather(&equilibrium, &[everything.clone(), everything], 2, two_levels, project_psi_2d);

        let message: String = result.err().expect("slicing two levels should be refused");
        assert!(message.contains("only one array-of-structures level"), "unexpected message: {message}");
    }
}
