//! The `Path` cursor Python builds by attribute access and indexing, and the reader which
//! resolves one against an IDS.

use super::{IndexSpec, Leaf, Node, NodeKind, Value};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use std::any::Any;

/// One step along a data dictionary path.
#[derive(Clone, Debug)]
pub enum Segment {
    /// A named child, e.g. `global_quantities`.
    Field(&'static str),
    /// A selection within an array of structures, e.g. `[:]` or `[0]`.
    Index(IndexSpec),
}

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
        Self {
            node,
            segments: Vec::new(),
            indexed: false,
            root: node.name,
        }
    }

    /// The index selections along this path, one per array-of-structures level, in order.
    fn index_specs(&self) -> Vec<IndexSpec> {
        let mut specs: Vec<IndexSpec> = Vec::new();
        for segment in &self.segments {
            if let Segment::Index(spec) = segment {
                specs.push(spec.clone());
            }
        }
        specs
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
        rendered
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
        Err(PyAttributeError::new_err(format!(
            "`{}` has no child `{}`. Available: {}",
            self.node.name,
            name,
            available.join(", ")
        )))
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

        Ok(PyPath {
            node: self.node,
            segments,
            indexed: true,
            root: self.root,
        })
    }

    /// The names reachable from here; drives tab completion in the REPL.
    fn __dir__(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        if let Some(children) = self.node.children(self.indexed) {
            for child in children {
                names.push(child.name.to_string());
            }
        }
        names
    }

    fn __repr__(&self) -> String {
        let rendered: String = self.as_imas_string();
        if rendered.is_empty() {
            return format!("Path({})", self.root);
        }
        format!("Path({rendered})")
    }

    /// The units of this node, e.g. `"A"`. Empty when the data dictionary gives none.
    ///
    /// Named without a leading underscore because no equilibrium field is called `units`.
    /// `path` would collide (`grids_ggd/.../path`), which is why the path string is on
    /// `__repr__` instead.
    #[getter]
    fn units(&self) -> &'static str {
        self.node.units
    }

    /// The data dictionary description of this node.
    #[getter]
    fn documentation(&self) -> &'static str {
        self.node.documentation
    }

    /// The data dictionary type of this node, e.g. `"FLT_0D"`, or `""` for a structure.
    #[getter]
    fn data_type(&self) -> &'static str {
        match &self.node.kind {
            NodeKind::Leaf(leaf) => leaf.data_type,
            _ => "",
        }
    }
}

/// Read `path` out of `ids`. Shared by every IDS wrapper in `ids.rs`.
///
/// It lives here rather than with the wrappers because it needs the private parts of the
/// path: which IDS it was built from, and the index selections along it.
pub(super) fn read_path<'py, I: Any>(py: Python<'py>, ids: &I, ids_name: &str, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
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
    let value: Value = (leaf.read)(ids as &dyn Any, &indices).map_err(|message| PyIndexError::new_err(format!("{}: {}", path.as_imas_string(), message)))?;

    value.into_python(py)
}
