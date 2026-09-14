use imas_rs::ids::magnetics::Magnetics as MagneticsIds;
use imas_rs::python::{PyMagnetics, PyPath, read_path};
use pyo3::prelude::*;

/// The machine's magnetic sensors, stored as an IMAS `magnetics` IDS.
///
/// Nothing is filled yet: this is constructed empty.
///
/// Read the data back through `magnetics_ids` and a path from `gsfit_rs.imas.magnetics_paths`;
/// there are no bespoke accessors.
#[pyclass(module = "gsfit_rs")]
pub struct Magnetics {
    pub magnetics_ids: MagneticsIds,
}

impl Default for Magnetics {
    fn default() -> Self {
        Self::new()
    }
}

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Construct an empty set of magnetic sensors.
    #[new]
    pub fn new() -> Self {
        Self {
            magnetics_ids: MagneticsIds::default(),
        }
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.magnetics_paths`, straight out of the
    /// magnetics IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.magnetics_ids, "magnetics", path)
    }

    /// A copy of the whole magnetics IDS, for reading with `gsfit_rs.imas.magnetics_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn magnetics_ids(&self) -> PyMagnetics {
        PyMagnetics::new(self.magnetics_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let n_b_field_pol_probe: usize = self.magnetics_ids.b_field_pol_probe.len();
        let n_flux_loop: usize = self.magnetics_ids.flux_loop.len();
        let n_rogowski_coil: usize = self.magnetics_ids.rogowski_coil.len();

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Magnetics>");
        string_output += &format!("║  {:<74} ║\n", version);
        string_output += &format!("║  {:<74} ║\n", format!(" b_field_pol_probe: {n_b_field_pol_probe} probe(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" flux_loop: {n_flux_loop} loop(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" rogowski_coil: {n_rogowski_coil} coil(s)"));
        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}
