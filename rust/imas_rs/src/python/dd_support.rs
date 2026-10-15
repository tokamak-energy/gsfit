//! One `#[pyclass]` per data dictionary structure from `dd_support.xsd` which is not an IDS.
//!
//! Unlike the IDS wrappers in `ids.rs`, these are built on the Python side and handed to
//! `gsfit_rs`, so each has a constructor.

use crate::dd_support::parameters_input::ParametersInput;
use pyo3::prelude::*;

/// The code parameters block passed from the wrapper to the code: the code parameters and their
/// defaults, each as XML, and the XML schema they follow.
#[pyclass(module = "gsfit_rs.dd_support", name = "ParametersInput")]
pub struct PyParametersInput {
    pub inner: ParametersInput,
}

impl PyParametersInput {
    pub fn new(inner: ParametersInput) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyParametersInput {
    /// Every argument is keyword-only, and defaults to "", the IMAS value of an unset string.
    #[new]
    #[pyo3(signature = (*, parameters_value = String::new(), parameters_default = String::new(), schema = String::new()))]
    fn py_new(parameters_value: String, parameters_default: String, schema: String) -> Self {
        Self::new(ParametersInput {
            parameters_value,
            parameters_default,
            schema,
        })
    }

    /// The code parameters, as XML.
    #[getter]
    fn parameters_value(&self) -> &str {
        &self.inner.parameters_value
    }

    /// The default code parameters, as XML.
    #[getter]
    fn parameters_default(&self) -> &str {
        &self.inner.parameters_default
    }

    /// The XML schema of the code parameters.
    #[getter]
    fn schema(&self) -> &str {
        &self.inner.schema
    }

    fn __repr__(&self) -> String {
        format!(
            "ParametersInput(parameters_value={} characters, parameters_default={} characters, schema={} characters)",
            self.inner.parameters_value.chars().count(),
            self.inner.parameters_default.chars().count(),
            self.inner.schema.chars().count()
        )
    }
}
