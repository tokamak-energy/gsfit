//! One `#[pyclass]` per IDS. Each holds a Rust IDS and reads it through a [`PyPath`].
//!
//! The wrappers are constructed on the Rust side only (`gsfit_rs` hands out a snapshot of
//! the IDS it holds); Python cannot build one from scratch.

use super::path::{PyPath, read_path};
use crate::ids::equilibrium::Equilibrium;
use crate::ids::magnetics::Magnetics;
use crate::ids::pf_active::PfActive;
use crate::ids::pf_passive::PfPassive;
use crate::ids::tf::Tf;
use crate::ids::wall::Wall;
use pyo3::prelude::*;

/// An equilibrium IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Equilibrium")]
pub struct PyEquilibrium {
    pub inner: Equilibrium,
}

impl PyEquilibrium {
    pub fn new(inner: Equilibrium) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEquilibrium {
    /// Read the data at `path` out of this IDS.
    ///
    /// The shape of the result follows the shape of the index: `time_slice[3]` gives a
    /// scalar, `time_slice[:]` gives an array.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "equilibrium", path)
    }

    /// The number of time slices held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.time_slice.len()
    }

    fn __repr__(&self) -> String {
        format!("Equilibrium(time_slice={} slice(s))", self.inner.time_slice.len())
    }
}

/// A magnetics IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Magnetics")]
pub struct PyMagnetics {
    pub inner: Magnetics,
}

impl PyMagnetics {
    pub fn new(inner: Magnetics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMagnetics {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "magnetics", path)
    }

    /// The three sensor arrays are listed rather than summed: a magnetics IDS has no single
    /// length, so there is no `__len__`.
    fn __repr__(&self) -> String {
        format!(
            "Magnetics(b_field_pol_probe={} probe(s), flux_loop={} loop(s), rogowski_coil={} coil(s))",
            self.inner.b_field_pol_probe.len(),
            self.inner.flux_loop.len(),
            self.inner.rogowski_coil.len()
        )
    }
}

/// A pf_active IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "PfActive")]
pub struct PyPfActive {
    pub inner: PfActive,
}

impl PyPfActive {
    pub fn new(inner: PfActive) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPfActive {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "pf_active", path)
    }

    /// The number of coils held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.coil.len()
    }

    fn __repr__(&self) -> String {
        format!("PfActive(coil={} coil(s))", self.inner.coil.len())
    }
}

/// A pf_passive IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "PfPassive")]
pub struct PyPfPassive {
    pub inner: PfPassive,
}

impl PyPfPassive {
    pub fn new(inner: PfPassive) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPfPassive {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "pf_passive", path)
    }

    /// The number of passive loops held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.r#loop.len()
    }

    fn __repr__(&self) -> String {
        format!("PfPassive(loop={} loop(s))", self.inner.r#loop.len())
    }
}

/// A tf IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Tf")]
pub struct PyTf {
    pub inner: Tf,
}

impl PyTf {
    pub fn new(inner: Tf) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTf {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "tf", path)
    }

    /// The number of coils held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.coil.len()
    }

    fn __repr__(&self) -> String {
        format!("Tf(coil={} coil(s))", self.inner.coil.len())
    }
}

/// A wall IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "Wall")]
pub struct PyWall {
    pub inner: Wall,
}

impl PyWall {
    pub fn new(inner: Wall) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWall {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "wall", path)
    }

    fn __repr__(&self) -> String {
        format!("Wall(description_2d={} description(s))", self.inner.description_2d.len())
    }
}
