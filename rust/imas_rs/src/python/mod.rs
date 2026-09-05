//! Python bindings for the IMAS data dictionary.
//!
//! # Design
//!
//! The Python surface separates *where* the data lives from *what to read out of it*:
//!
//! ```python
//! from gsfit_rs.imas import equilibrium_paths
//! equilibrium_ids.get(equilibrium_paths.time_slice[:].global_quantities.ip)
//! ```
//!
//! `equilibrium_paths` holds no data at all. It is a cursor over a static description of
//! the data dictionary, so building a path is pure metadata navigation. This is what lets
//! a single [`PyPath`] class serve the whole tree: because there is nothing to borrow,
//! the pyo3 restriction that `#[pyclass]` cannot carry a lifetime never bites.
//!
//! A path is a value. It can be stored in a list, printed, passed around, and reused
//! against several IDSs - which is the point of the `get(path)` form over attribute
//! access that reads immediately.
//!
//! # Where the type hints come from
//!
//! At runtime there is one `Path` class reached through `__getattr__`, and `__dir__`
//! gives tab completion in the REPL. Editors and `mypy` read `python/gsfit_rs/imas.pyi`
//! instead, which declares one class per node and so type-checks every attribute and the
//! return type of `get`. Both the table below and that stub are meant to be emitted by
//! `imas_updater/build_ids.py`; see `equilibrium_paths.rs`.

mod equilibrium_paths;
mod wall_paths;

use crate::ids::equilibrium::Equilibrium;
use crate::ids::wall::Wall;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayViewMutD, Axis, IxDyn, Slice};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PySlice};
use std::any::Any;

pub use equilibrium_paths::EQUILIBRIUM_ROOT;
pub use wall_paths::WALL_ROOT;

// ============================================================================
// Path segments
// ============================================================================

/// One step along a data dictionary path.
#[derive(Clone, Debug)]
pub enum Segment {
    /// A named child, e.g. `global_quantities`.
    Field(&'static str),
    /// A selection within an array of structures, e.g. `[:]` or `[0]`.
    Index(IndexSpec),
}

/// How an array of structures was indexed.
///
/// Resolution is deferred: a path does not know how many time slices exist, so the
/// selection is stored as written and resolved against the real length inside `get`.
#[derive(Clone, Debug)]
pub enum IndexSpec {
    /// `path[3]` - selects one element and drops that dimension.
    One(isize),
    /// `path[:]`, `path[3:7]`, `path[::2]` - keeps the dimension.
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    /// `path[[0, 2, 5]]` - keeps the dimension.
    Many(Vec<isize>),
}

/// Resolve one possibly-negative index against a known length.
fn resolve_one_index(index: isize, n_elements: usize) -> Result<usize, String> {
    let n_elements_signed: isize = n_elements as isize;
    let resolved: isize = if index < 0 { index + n_elements_signed } else { index };

    if resolved < 0 || resolved >= n_elements_signed {
        return Err(format!("index {index} is out of range for an array of structures with {n_elements} element(s)"));
    }

    return Ok(resolved as usize);
}

impl IndexSpec {
    /// True when this selection drops the dimension (`path[3]` rather than `path[:]`).
    pub fn is_scalar(&self) -> bool {
        return matches!(self, IndexSpec::One(_));
    }

    /// Expand this selection into concrete element indices.
    pub fn resolve(&self, n_elements: usize) -> Result<Vec<usize>, String> {
        match self {
            IndexSpec::One(index) => {
                let resolved: usize = resolve_one_index(*index, n_elements)?;
                return Ok(vec![resolved]);
            }
            IndexSpec::Many(requested) => {
                let mut indices: Vec<usize> = Vec::with_capacity(requested.len());
                for index in requested {
                    indices.push(resolve_one_index(*index, n_elements)?);
                }
                return Ok(indices);
            }
            IndexSpec::Slice { start, stop, step } => {
                return resolve_slice(*start, *stop, *step, n_elements);
            }
        }
    }
}

/// Expand a Python slice into concrete indices, following CPython's `slice.indices` rules
/// (negative values count from the end, out-of-range endpoints clamp rather than raise).
fn resolve_slice(start: Option<isize>, stop: Option<isize>, step: Option<isize>, n_elements: usize) -> Result<Vec<usize>, String> {
    let n_elements_signed: isize = n_elements as isize;
    let step: isize = step.unwrap_or(1);

    if step == 0 {
        return Err("slice step cannot be zero".to_string());
    }

    // Clamping bounds differ by direction: a backwards slice may legitimately stop at -1.
    let (default_start, default_stop): (isize, isize) = if step > 0 { (0, n_elements_signed) } else { (n_elements_signed - 1, -1) };

    let start_resolved: isize = match start {
        None => default_start,
        Some(value) => {
            let shifted: isize = if value < 0 { value + n_elements_signed } else { value };
            if step > 0 {
                shifted.clamp(0, n_elements_signed)
            } else {
                shifted.clamp(-1, n_elements_signed - 1)
            }
        }
    };

    let stop_resolved: isize = match stop {
        None => default_stop,
        Some(value) => {
            let shifted: isize = if value < 0 { value + n_elements_signed } else { value };
            if step > 0 {
                shifted.clamp(0, n_elements_signed)
            } else {
                shifted.clamp(-1, n_elements_signed - 1)
            }
        }
    };

    // Compute the count up front so the loop is bounded.
    let n_indices: usize = if step > 0 {
        if stop_resolved > start_resolved {
            ((stop_resolved - start_resolved + step - 1) / step) as usize
        } else {
            0
        }
    } else {
        if stop_resolved < start_resolved {
            ((stop_resolved - start_resolved + step + 1) / step) as usize
        } else {
            0
        }
    };

    let mut indices: Vec<usize> = Vec::with_capacity(n_indices);
    for i_index in 0..n_indices {
        indices.push((start_resolved + (i_index as isize) * step) as usize);
    }

    return Ok(indices);
}

// ============================================================================
// Static description of the data dictionary
// ============================================================================

/// A node in the static data dictionary description.
pub struct Node {
    pub name: &'static str,
    pub documentation: &'static str,
    pub units: &'static str,
    pub kind: NodeKind,
}

pub enum NodeKind {
    /// A nested structure; navigate into its children by name.
    Structure(&'static [Node]),
    /// An array of structures; must be indexed before navigating further.
    ArrayOfStructures(&'static [Node]),
    /// A terminal data node.
    Leaf(Leaf),
}

/// A terminal data node, paired with the function that reads it out of the IDS.
///
/// `read` is a plain `fn` pointer so the whole description stays a `static`. It receives
/// the index selections gathered along the path, one per array-of-structures level
/// crossed, in order.
///
/// The IDS arrives as `&dyn Any` so that one `Node` type serves every IDS; the generated
/// reader downcasts it back. `Path` records which IDS it was built from and `read_path`
/// checks that before calling, so the downcast is a belt-and-braces failure rather than
/// the primary guard.
pub struct Leaf {
    /// The data dictionary type, e.g. `"FLT_0D"`.
    pub data_type: &'static str,
    pub read: fn(&dyn Any, &[IndexSpec]) -> Result<Value, String>,
}

impl Node {
    /// The children reachable from this node, or `None` for a leaf.
    ///
    /// An array of structures only exposes its children once indexed, so that
    /// `time_slice.global_quantities` fails with a clear message rather than silently
    /// meaning `time_slice[:]`.
    fn children(&self, indexed: bool) -> Option<&'static [Node]> {
        match &self.kind {
            NodeKind::Structure(children) => return Some(children),
            NodeKind::ArrayOfStructures(children) => {
                if indexed {
                    return Some(children);
                }
                return None;
            }
            NodeKind::Leaf(_) => return None,
        }
    }
}

// ============================================================================
// Values read back out of the IDS
// ============================================================================

/// A value gathered out of the IDS, before conversion to Python.
///
/// Arrays are held as `ArrayD` rather than a rank-specific type, because gathering a
/// leaf across a sliced array of structures raises its rank by one and the data
/// dictionary has leaves from `FLT_0D` up to `FLT_4D`.
pub enum Value {
    Flt0d(Option<f64>),
    FltNd(ArrayD<f64>),
    Int0d(Option<i32>),
    IntNd(ArrayD<i32>),
    Str0d(Option<String>),
    StrList(Vec<Option<String>>),
    StrNested(Vec<Option<Vec<String>>>),
}

impl Value {
    fn into_python<'py>(self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
        match self {
            // Unset floats become NaN, matching how a failed time slice is stored.
            Value::Flt0d(value) => return Ok(PyFloat::new(py, value.unwrap_or(f64::NAN)).into_any()),
            Value::FltNd(values) => return Ok(values.into_pyarray(py).into_any()),

            // Integers have no NaN, so an unset integer is an error rather than a silent
            // promotion to float.
            Value::Int0d(Some(value)) => return value.into_bound_py_any(py),
            Value::Int0d(None) => {
                return Err(PyValueError::new_err(format!("`{path}` is unset (None) and has no integer representation")));
            }
            Value::IntNd(values) => return Ok(values.into_pyarray(py).into_any()),

            // Strings gather to a list, where `None` is representable.
            Value::Str0d(value) => return value.into_bound_py_any(py),
            Value::StrList(values) => return values.into_bound_py_any(py),
            Value::StrNested(values) => return values.into_bound_py_any(py),
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
    fn one(value: Option<Self>) -> Result<Value, String>;

    /// One level was sliced: stack the values along a new leading axis.
    fn stack(values: Vec<Option<Self>>) -> Result<Value, String>;
}

impl Gatherable for f64 {
    fn one(value: Option<Self>) -> Result<Value, String> {
        return Ok(Value::Flt0d(value));
    }

    fn stack(values: Vec<Option<Self>>) -> Result<Value, String> {
        let mut stacked: Array1<f64> = Array1::from_elem(values.len(), f64::NAN);
        for i_value in 0..values.len() {
            if let Some(value) = values[i_value] {
                stacked[i_value] = value;
            }
        }
        return Ok(Value::FltNd(stacked.into_dyn()));
    }
}

impl Gatherable for i32 {
    fn one(value: Option<Self>) -> Result<Value, String> {
        return Ok(Value::Int0d(value));
    }

    fn stack(values: Vec<Option<Self>>) -> Result<Value, String> {
        // There is no integer NaN, so an unset element cannot be represented and is
        // reported rather than filled with a sentinel.
        let mut stacked: Array1<i32> = Array1::zeros(values.len());
        for i_value in 0..values.len() {
            match values[i_value] {
                Some(value) => stacked[i_value] = value,
                None => {
                    return Err(format!("is unset (None) at element {i_value}, and an integer has no NaN to stand in for it"));
                }
            }
        }
        return Ok(Value::IntNd(stacked.into_dyn()));
    }
}

impl Gatherable for String {
    fn one(value: Option<Self>) -> Result<Value, String> {
        return Ok(Value::Str0d(value));
    }

    fn stack(values: Vec<Option<Self>>) -> Result<Value, String> {
        return Ok(Value::StrList(values));
    }
}

impl Gatherable for Vec<String> {
    fn one(value: Option<Self>) -> Result<Value, String> {
        // An unset string list reads back empty, as an unset numeric array does.
        let values: Vec<String> = value.unwrap_or_default();
        return Ok(Value::StrList(values.into_iter().map(Some).collect()));
    }

    fn stack(values: Vec<Option<Self>>) -> Result<Value, String> {
        // Strings cannot be padded, so a list of lists keeps the ragged shape and an
        // unset element stays `None` rather than becoming an empty list.
        return Ok(Value::StrNested(values));
    }
}

/// Stack float arrays along a new leading axis, padding with NaN where the gathered
/// arrays differ in shape (as `boundary/outline/r` does, having a different number of
/// points per time slice).
fn stack_float_arrays(values: Vec<Option<ArrayD<f64>>>) -> Result<Value, String> {
    let n_values: usize = values.len();

    let mut n_dimensions: usize = 0;
    for value in &values {
        if let Some(array) = value {
            n_dimensions = array.ndim();
            break;
        }
    }

    let mut max_shape: Vec<usize> = vec![0; n_dimensions];
    for value in &values {
        if let Some(array) = value {
            if array.ndim() != n_dimensions {
                return Err(format!(
                    "cannot be gathered: the elements have different ranks ({} and {})",
                    n_dimensions,
                    array.ndim()
                ));
            }
            for i_dimension in 0..n_dimensions {
                max_shape[i_dimension] = max_shape[i_dimension].max(array.shape()[i_dimension]);
            }
        }
    }

    let mut shape: Vec<usize> = Vec::with_capacity(n_dimensions + 1);
    shape.push(n_values);
    shape.extend_from_slice(&max_shape);

    let mut stacked: ArrayD<f64> = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
    for i_value in 0..n_values {
        if let Some(array) = &values[i_value] {
            let element_shape: Vec<usize> = array.shape().to_vec();
            let mut destination: ArrayViewMutD<f64> = stacked.index_axis_mut(Axis(0), i_value);
            let mut destination: ArrayViewMutD<f64> =
                destination.slice_each_axis_mut(|axis: ndarray::AxisDescription| Slice::from(0..element_shape[axis.axis.index()]));
            destination.assign(array);
        }
    }

    return Ok(Value::FltNd(stacked));
}

/// Stack integer arrays along a new leading axis.
///
/// Unlike the float case there is no padding value, so every element must be set and
/// the shapes must agree.
fn stack_integer_arrays(values: Vec<Option<ArrayD<i32>>>) -> Result<Value, String> {
    let n_values: usize = values.len();

    let mut shape: Vec<usize> = Vec::new();
    for i_value in 0..n_values {
        match &values[i_value] {
            None => return Err(format!("is unset (None) at element {i_value}, and an integer array has no NaN to pad with")),
            Some(array) => {
                if i_value == 0 {
                    shape = array.shape().to_vec();
                } else if array.shape() != shape.as_slice() {
                    return Err(format!(
                        "cannot be gathered: element {i_value} has shape {:?} but element 0 has shape {:?}, \
                         and an integer array has no NaN to pad with",
                        array.shape(),
                        shape
                    ));
                }
            }
        }
    }

    let mut full_shape: Vec<usize> = Vec::with_capacity(shape.len() + 1);
    full_shape.push(n_values);
    full_shape.extend_from_slice(&shape);

    let mut stacked: ArrayD<i32> = ArrayD::zeros(IxDyn(&full_shape));
    for i_value in 0..n_values {
        if let Some(array) = &values[i_value] {
            stacked.index_axis_mut(Axis(0), i_value).assign(array);
        }
    }

    return Ok(Value::IntNd(stacked));
}

/// `Gatherable` for every array leaf type, which differ only in how they reach `ArrayD`.
macro_rules! impl_gatherable_for_arrays {
    ($($array_type:ty => $value_variant:ident, $stack_function:ident;)*) => {
        $(
            impl Gatherable for $array_type {
                fn one(value: Option<Self>) -> Result<Value, String> {
                    match value {
                        Some(array) => return Ok(Value::$value_variant(array.into_dyn())),
                        // An unset array has no shape to fill with NaN, so it reads back
                        // empty - the array equivalent of the node never having been written.
                        None => return Ok(Value::$value_variant(ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty array"))),
                    }
                }

                fn stack(values: Vec<Option<Self>>) -> Result<Value, String> {
                    let mut as_dynamic: Vec<Option<_>> = Vec::with_capacity(values.len());
                    for value in values {
                        as_dynamic.push(value.map(|array| array.into_dyn()));
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
    return None;
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
    project: fn(&I, &[usize]) -> Option<T>,
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
        return Ok(resolved[0]);
    }

    match sliced_level {
        None => {
            let mut at: Vec<usize> = Vec::with_capacity(n_levels);
            for i_level in 0..n_levels {
                let element: usize = resolve_single(ids, indices, i_level, &at, length_of)?;
                at.push(element);
            }
            return T::one(project(ids, &at));
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

            let mut values: Vec<Option<T>> = Vec::with_capacity(selected.len());
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

            return T::stack(values);
        }
    }
}

// ============================================================================
// Python: Path
// ============================================================================

/// A data dictionary path, built by attribute access and indexing. Holds no data.
#[pyclass(module = "gsfit_rs.imas", name = "Path", frozen)]
pub struct PyPath {
    node: &'static Node,
    segments: Vec<Segment>,
    /// Whether `node` - when it is an array of structures - has been indexed yet.
    indexed: bool,
    /// The IDS this path was built from, e.g. `"equilibrium"`. Carried down from the root
    /// so that passing a `wall` path to an `equilibrium` IDS is caught with a useful
    /// message rather than a failed downcast.
    root: &'static str,
}

impl PyPath {
    /// The root of one IDS's data dictionary.
    pub fn at_root(node: &'static Node) -> Self {
        return Self {
            node,
            segments: Vec::new(),
            indexed: false,
            root: node.name,
        };
    }

    /// The index selections along this path, one per array-of-structures level, in order.
    fn index_specs(&self) -> Vec<IndexSpec> {
        let mut specs: Vec<IndexSpec> = Vec::new();
        for segment in &self.segments {
            if let Segment::Index(spec) = segment {
                specs.push(spec.clone());
            }
        }
        return specs;
    }

    /// The path as an IMAS-style string, e.g. `time_slice(:)/global_quantities/ip`.
    fn as_imas_string(&self) -> String {
        let mut rendered: String = String::new();
        for segment in &self.segments {
            match segment {
                Segment::Field(name) => {
                    if !rendered.is_empty() {
                        rendered.push('/');
                    }
                    rendered.push_str(name);
                }
                Segment::Index(spec) => {
                    let inside: String = match spec {
                        IndexSpec::One(index) => index.to_string(),
                        IndexSpec::Slice { start, stop, step } => {
                            let start_text: String = start.map(|value| value.to_string()).unwrap_or_default();
                            let stop_text: String = stop.map(|value| value.to_string()).unwrap_or_default();
                            match step {
                                None => format!("{start_text}:{stop_text}"),
                                Some(step) => format!("{start_text}:{stop_text}:{step}"),
                            }
                        }
                        IndexSpec::Many(indices) => {
                            let rendered_indices: Vec<String> = indices.iter().map(|index| index.to_string()).collect();
                            rendered_indices.join(",")
                        }
                    };
                    rendered.push_str(&format!("({inside})"));
                }
            }
        }
        return rendered;
    }
}

#[pymethods]
impl PyPath {
    fn __getattr__(&self, name: &str) -> PyResult<PyPath> {
        // Python probes objects with dunder and private names (`__deepcopy__`,
        // `_ipython_canary_...`); no data dictionary name starts with an underscore, so
        // reject these before producing a "did you mean" message about them.
        if name.starts_with('_') {
            return Err(PyAttributeError::new_err(name.to_string()));
        }

        let children: &'static [Node] = match self.node.children(self.indexed) {
            Some(children) => children,
            None => match &self.node.kind {
                NodeKind::ArrayOfStructures(_) => {
                    let array_name: &str = self.node.name;
                    return Err(PyAttributeError::new_err(format!(
                        "`{array_name}` is an array of structures; index it before reading `{name}`, \
                         e.g. `.{array_name}[:].{name}` or `.{array_name}[0].{name}`"
                    )));
                }
                _ => {
                    return Err(PyAttributeError::new_err(format!(
                        "`{}` is a data node ({}); it has no child `{}`",
                        self.as_imas_string(),
                        self.node.name,
                        name
                    )));
                }
            },
        };

        for child in children {
            if child.name == name {
                let mut segments: Vec<Segment> = self.segments.clone();
                segments.push(Segment::Field(child.name));
                return Ok(PyPath {
                    node: child,
                    segments,
                    indexed: false,
                    root: self.root,
                });
            }
        }

        let available: Vec<&str> = children.iter().map(|child| child.name).collect();
        return Err(PyAttributeError::new_err(format!(
            "`{}` has no child `{}`. Available: {}",
            self.node.name,
            name,
            available.join(", ")
        )));
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyPath> {
        match &self.node.kind {
            NodeKind::ArrayOfStructures(_) => {}
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "`{}` is not an array of structures and cannot be indexed",
                    self.node.name
                )));
            }
        }

        if self.indexed {
            return Err(PyIndexError::new_err(format!("`{}` has already been indexed", self.node.name)));
        }

        let spec: IndexSpec = if let Ok(slice) = index.cast::<PySlice>() {
            IndexSpec::Slice {
                start: slice.getattr("start")?.extract()?,
                stop: slice.getattr("stop")?.extract()?,
                step: slice.getattr("step")?.extract()?,
            }
        } else if let Ok(single) = index.extract::<isize>() {
            IndexSpec::One(single)
        } else if let Ok(many) = index.extract::<Vec<isize>>() {
            IndexSpec::Many(many)
        } else {
            return Err(PyTypeError::new_err(
                "an array of structures must be indexed with an int, a slice, or a list of ints",
            ));
        };

        let mut segments: Vec<Segment> = self.segments.clone();
        segments.push(Segment::Index(spec));

        return Ok(PyPath {
            node: self.node,
            segments,
            indexed: true,
            root: self.root,
        });
    }

    /// The names reachable from here; drives tab completion in the REPL.
    fn __dir__(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        if let Some(children) = self.node.children(self.indexed) {
            for child in children {
                names.push(child.name.to_string());
            }
        }
        return names;
    }

    fn __repr__(&self) -> String {
        let rendered: String = self.as_imas_string();
        if rendered.is_empty() {
            return format!("Path({})", self.root);
        }
        return format!("Path({rendered})");
    }

    /// The units of this node, e.g. `"A"`. Empty when the data dictionary gives none.
    ///
    /// Named without a leading underscore because no equilibrium field is called `units`.
    /// `path` would collide (`grids_ggd/.../path`), which is why the path string is on
    /// `__repr__` instead.
    #[getter]
    fn units(&self) -> &'static str {
        return self.node.units;
    }

    /// The data dictionary description of this node.
    #[getter]
    fn documentation(&self) -> &'static str {
        return self.node.documentation;
    }

    /// The data dictionary type of this node, e.g. `"FLT_0D"`, or `""` for a structure.
    #[getter]
    fn data_type(&self) -> &'static str {
        match &self.node.kind {
            NodeKind::Leaf(leaf) => return leaf.data_type,
            _ => return "",
        }
    }
}

// ============================================================================
// Python: the IDSs
// ============================================================================

/// Read `path` out of `ids`. Shared by every IDS wrapper below.
fn read_path<'py, I: Any>(py: Python<'py>, ids: &I, ids_name: &str, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
    if path.root != ids_name {
        return Err(PyTypeError::new_err(format!(
            "this is a `{ids_name}` IDS but `{}` starts at `{}`; use `{ids_name}_paths`",
            path.__repr__(),
            path.root
        )));
    }

    let leaf: &Leaf = match &path.node.kind {
        NodeKind::Leaf(leaf) => leaf,
        _ => {
            let available: Vec<String> = path.__dir__();
            return Err(PyTypeError::new_err(format!(
                "`{}` is not a data node, so it cannot be read. Available children: {}",
                path.__repr__(),
                available.join(", ")
            )));
        }
    };

    let indices: Vec<IndexSpec> = path.index_specs();
    let value: Value = (leaf.read)(ids as &dyn Any, &indices).map_err(|message| {
        return PyIndexError::new_err(format!("{}: {}", path.as_imas_string(), message));
    })?;

    return value.into_python(py, &path.as_imas_string());
}

/// An equilibrium IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Equilibrium")]
pub struct PyEquilibrium {
    pub inner: Equilibrium,
}

impl PyEquilibrium {
    pub fn new(inner: Equilibrium) -> Self {
        return Self { inner };
    }
}

#[pymethods]
impl PyEquilibrium {
    /// Read the data at `path` out of this IDS.
    ///
    /// The shape of the result follows the shape of the index: `time_slice[3]` gives a
    /// scalar, `time_slice[:]` gives an array.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        return read_path(py, &self.inner, "equilibrium", path);
    }

    /// The number of time slices held by this IDS.
    fn __len__(&self) -> usize {
        return self.inner.time_slice.len();
    }

    fn __repr__(&self) -> String {
        return format!("Equilibrium(time_slice={} slice(s))", self.inner.time_slice.len());
    }
}

/// A wall IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Wall")]
pub struct PyWall {
    pub inner: Wall,
}

impl PyWall {
    pub fn new(inner: Wall) -> Self {
        return Self { inner };
    }
}

#[pymethods]
impl PyWall {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        return read_path(py, &self.inner, "wall", path);
    }

    fn __repr__(&self) -> String {
        return format!("Wall(description_2d={} description(s))", self.inner.description_2d.len());
    }
}

// ============================================================================
// Module registration
// ============================================================================

/// Populate an `imas` module with the data dictionary bindings.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPath>()?;
    module.add_class::<PyEquilibrium>()?;
    module.add_class::<PyWall>()?;
    module.add("equilibrium_paths", PyPath::at_root(&EQUILIBRIUM_ROOT).into_pyobject(module.py())?)?;
    module.add("wall_paths", PyPath::at_root(&WALL_ROOT).into_pyobject(module.py())?)?;
    return Ok(());
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// `time_slice` is the only array-of-structures level used by these tests.
    fn time_slice_lengths(equilibrium: &Equilibrium, level: usize, _at: &[usize]) -> Option<usize> {
        match level {
            0 => return Some(equilibrium.time_slice.len()),
            _ => return None,
        }
    }

    fn project_ip(equilibrium: &Equilibrium, at: &[usize]) -> Option<f64> {
        return equilibrium.time_slice.get(at[0])?.global_quantities.ip;
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
            equilibrium.time_slice[i_time].global_quantities.ip = Some(1.0e6 * (i_time as f64));
        }
        return equilibrium;
    }

    /// A slice must be handled exactly as CPython handles it, including negative bounds,
    /// negative steps and out-of-range endpoints clamping rather than raising.
    ///
    /// The expected values were produced by Python itself: `list(range(n))[start:stop:step]`.
    #[test]
    fn slice_semantics_match_python() {
        let cases: Vec<(Vec<usize>, usize, Option<isize>, Option<isize>, Option<isize>)> = vec![
            (vec![0, 1, 2, 3, 4], 5, None, None, None),
            (vec![1, 2], 5, Some(1), Some(3), None),
            (vec![0, 2, 4], 5, None, None, Some(2)),
            (vec![4, 3, 2, 1, 0], 5, None, None, Some(-1)),
            (vec![3, 4], 5, Some(-2), None, None),
            (vec![0, 1, 2, 3], 5, None, Some(-1), None),
            (Vec::<usize>::new(), 5, Some(10), Some(99), None),
            (vec![4, 3, 2], 5, Some(4), Some(1), Some(-1)),
            (vec![4, 2, 0], 5, None, None, Some(-2)),
            (vec![0, 1, 2, 3, 4], 5, Some(-99), Some(99), None),
            (Vec::<usize>::new(), 5, Some(3), Some(1), None),
            (Vec::<usize>::new(), 0, None, None, None),
            (vec![2, 1, 0], 3, None, None, Some(-1)),
            (Vec::<usize>::new(), 5, Some(2), Some(2), None),
        ];

        for (expected, n_elements, start, stop, step) in cases {
            let spec: IndexSpec = IndexSpec::Slice { start, stop, step };
            let resolved: Vec<usize> = spec.resolve(n_elements).expect("slice should resolve");
            assert_eq!(resolved, expected, "[{start:?}:{stop:?}:{step:?}] over {n_elements} element(s)");
        }
    }

    #[test]
    fn a_zero_step_is_rejected() {
        let spec: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: Some(0),
        };
        assert!(spec.resolve(5).is_err());
    }

    /// A scalar index drops the dimension; every other selection keeps it.
    #[test]
    fn gather_follows_the_shape_of_the_index() {
        let equilibrium: Equilibrium = equilibrium_with_ip(5);

        match gather_ip(&equilibrium, IndexSpec::One(3)).expect("index 3 is in range") {
            Value::Flt0d(value) => assert_eq!(value, Some(3.0e6)),
            _ => panic!("`time_slice[3]` should give a scalar"),
        }

        // A negative index counts from the end, as in Python.
        match gather_ip(&equilibrium, IndexSpec::One(-2)).expect("index -2 is in range") {
            Value::Flt0d(value) => assert_eq!(value, Some(3.0e6)),
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
        equilibrium.time_slice[0].profiles_1d.psi = Some(Array1::from(vec![1.0, 2.0]));
        equilibrium.time_slice[1].profiles_1d.psi = Some(Array1::from(vec![3.0, 4.0, 5.0]));
        // time_slice[2] is left unset

        fn project_psi(equilibrium: &Equilibrium, at: &[usize]) -> Option<Array1<f64>> {
            return equilibrium.time_slice.get(at[0])?.profiles_1d.psi.clone();
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

        fn project_psi_2d(equilibrium: &Equilibrium, at: &[usize]) -> Option<Array2<f64>> {
            return equilibrium.time_slice.get(at[0])?.profiles_2d.get(at[1])?.psi.clone();
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
