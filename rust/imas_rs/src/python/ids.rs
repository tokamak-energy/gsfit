//! One `#[pyclass]` per IDS. Each holds a Rust IDS and reads it through a [`PyPath`].
//!
//! The wrappers are constructed on the Rust side only (`gsfit_rs` hands out a snapshot of
//! the IDS it holds); Python cannot build one from scratch. Each can be pickled, though: its
//! `__reduce__` gives the `dict` of the IDS's nodes, see [`PickleState`], which its static method
//! `_from_pickle_state` reads back.

use super::path::{PyPath, read_path};
use super::pickle::{PickleState, PickleStateError};
use crate::ids::core_profiles::CoreProfiles;
use crate::ids::core_sources::CoreSources;
use crate::ids::equilibrium::Equilibrium;
use crate::ids::magnetics::Magnetics;
use crate::ids::pf_active::PfActive;
use crate::ids::pf_passive::PfPassive;
use crate::ids::pulse_schedule::PulseSchedule;
use crate::ids::tf::Tf;
use crate::ids::wall::Wall;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};

/// A core_profiles IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "CoreProfiles")]
pub struct PyCoreProfiles {
    pub inner: CoreProfiles,
}

impl PyCoreProfiles {
    pub fn new(inner: CoreProfiles) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCoreProfiles {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "core_profiles", path)
    }

    /// The number of times of the 2D profiles held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.profiles_2d.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "CoreProfiles(profiles_1d={} time(s), profiles_2d={} time(s))",
            self.inner.profiles_1d.len(),
            self.inner.profiles_2d.len()
        )
    }

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `core_profiles`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("core_profiles", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `core_profiles` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `core_profiles` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("core_profiles")?
            .ok_or_else(|| PyValueError::new_err("The pickled `core_profiles` IDS has no `core_profiles`"))?;
        let inner: CoreProfiles =
            CoreProfiles::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `core_profiles` IDS"))?;
        Ok(Self::new(inner))
    }
}

/// A core_sources IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "CoreSources")]
pub struct PyCoreSources {
    pub inner: CoreSources,
}

impl PyCoreSources {
    pub fn new(inner: CoreSources) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCoreSources {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "core_sources", path)
    }

    /// The number of sources held by this IDS.
    fn __len__(&self) -> usize {
        self.inner.source.len()
    }

    fn __repr__(&self) -> String {
        let n_source: usize = self.inner.source.len();
        let mut source_names: Vec<String> = Vec::with_capacity(n_source);
        for i_source in 0..n_source {
            source_names.push(self.inner.source[i_source].identifier.name.clone());
        }
        format!("CoreSources(source={n_source}: [{}])", source_names.join(", "))
    }

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `core_sources`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("core_sources", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `core_sources` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `core_sources` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("core_sources")?
            .ok_or_else(|| PyValueError::new_err("The pickled `core_sources` IDS has no `core_sources`"))?;
        let inner: CoreSources = CoreSources::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `core_sources` IDS"))?;
        Ok(Self::new(inner))
    }
}

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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `equilibrium`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("equilibrium", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `equilibrium` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `equilibrium` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("equilibrium")?
            .ok_or_else(|| PyValueError::new_err("The pickled `equilibrium` IDS has no `equilibrium`"))?;
        let inner: Equilibrium = Equilibrium::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `equilibrium` IDS"))?;
        Ok(Self::new(inner))
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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `magnetics`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("magnetics", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `magnetics` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `magnetics` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("magnetics")?
            .ok_or_else(|| PyValueError::new_err("The pickled `magnetics` IDS has no `magnetics`"))?;
        let inner: Magnetics = Magnetics::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `magnetics` IDS"))?;
        Ok(Self::new(inner))
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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `pf_active`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pf_active", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pf_active` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `pf_active` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("pf_active")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pf_active` IDS has no `pf_active`"))?;
        let inner: PfActive = PfActive::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `pf_active` IDS"))?;
        Ok(Self::new(inner))
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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `pf_passive`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pf_passive", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pf_passive` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `pf_passive` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("pf_passive")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pf_passive` IDS has no `pf_passive`"))?;
        let inner: PfPassive = PfPassive::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `pf_passive` IDS"))?;
        Ok(Self::new(inner))
    }
}

/// A pulse_schedule IDS, readable from Python through paths.
#[pyclass(module = "gsfit_rs.imas", name = "PulseSchedule")]
pub struct PyPulseSchedule {
    pub inner: PulseSchedule,
}

impl PyPulseSchedule {
    pub fn new(inner: PulseSchedule) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPulseSchedule {
    /// Read the data at `path` out of this IDS.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.inner, "pulse_schedule", path)
    }

    /// A pulse schedule holds references for many unrelated systems, so it has no single length
    /// and there is no `__len__`. The gaps are what GSFit fills, so those are what is shown.
    fn __repr__(&self) -> String {
        format!("PulseSchedule(position_control.gap={} gap(s))", self.inner.position_control.gap.len())
    }

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `pulse_schedule`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pulse_schedule", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pulse_schedule` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `pulse_schedule` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("pulse_schedule")?
            .ok_or_else(|| PyValueError::new_err("The pickled `pulse_schedule` IDS has no `pulse_schedule`"))?;
        let inner: PulseSchedule =
            PulseSchedule::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `pulse_schedule` IDS"))?;
        Ok(Self::new(inner))
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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `tf`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("tf", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `tf` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `tf` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state.get_item("tf")?.ok_or_else(|| PyValueError::new_err("The pickled `tf` IDS has no `tf`"))?;
        let inner: Tf = Tf::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `tf` IDS"))?;
        Ok(Self::new(inner))
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

    /// Pickle this IDS. Its state is a `dict` of the version of `imas_rs` which pickled it, `version`, when, `pickled_datetime`,
    /// in seconds since 1970, and the `dict` of the IDS's nodes, `wall`, see `PickleState`; `_from_pickle_state` reads it
    /// back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("wall", self.inner.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled with a different data dictionary. A different
    /// version of `imas_rs` with the same data dictionary only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `wall` IDS has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `wall` IDS being unpickled was pickled by imas_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let ids_state: Bound<'_, PyAny> = state
            .get_item("wall")?
            .ok_or_else(|| PyValueError::new_err("The pickled `wall` IDS has no `wall`"))?;
        let inner: Wall = Wall::from_pickle_state(&ids_state).map_err(|error: PickleStateError| error.into_py_err("the `wall` IDS"))?;
        Ok(Self::new(inner))
    }
}
