use crate::settings::settings_wall::SettingsWall;
use imas_rs::ids::wall::{Code, IdentifierStatic, Rz1dStatic, Wall as WallIds, Wall2d, Wall2dLimiterUnit};
use imas_rs::python::{PickleState, PickleStateError, PyParametersInput, PyPath, PyWall, read_path};
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's wall, stored as an IMAS `wall` IDS.
///
/// Only the limiter is filled so far:
/// * `wall/description_2d(0)/limiter/unit(i)/outline/r`
/// * `wall/description_2d(0)/limiter/unit(i)/outline/z`
///
/// This is the contour the data dictionary designates for equilibrium codes; its
/// documentation reads "Description of the immobile limiting surface(s) or plasma facing
/// components for defining the Last Closed Flux Surface".
///
/// # Unit ordering
///
/// `Equilibrium.copy_from_wall_ids` copies the units into the equilibrium IDS, as
/// `equilibrium/limiter/unit(i)`, in the same order, and the solver reads them from there by
/// position. So the order they are added in is part of the contract:
/// * `unit(0)` is the vacuum vessel contour: the closed polygon bounding the region the
///   plasma is allowed to occupy. The solver uses this, and only this, as the vessel.
/// * `unit(1)`, `unit(2)`, ... are additional plasma facing components, such as tiles which
///   protrude inside the vessel contour.
///
/// Every unit contributes its points to the set of candidate limit points, so a plasma may
/// be limited on the vessel or on any of the tiles.
///
/// Read the data back through `wall_ids` and a path from `gsfit_rs.imas.wall_paths`; there
/// are no bespoke accessors.
///
/// Note that `description_2d(i)/vessel` is a *different* node, holding the mechanical
/// structure of the vessel - nested metal layers with materials and resistivity - rather
/// than the region the plasma occupies. It is deliberately left empty.
///
/// # Settings
///
/// The wall is constructed with the settings it is built from, e.g. which run of which database
/// the limiter is read from, as an IMAS `parameters_input`. They are validated against the XML
/// schema in `parameters_input`, and only their XML is kept: it is written to `wall/code/parameters`,
/// where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_wall`, whose structs are generated from the XML schema
/// `wall.xsd`. They are kept here, not in the IDS, as the wall IDS can also be made by another code,
/// whose `wall/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct Wall {
    pub wall_ids: WallIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_wall: SettingsWall,
}

/// Python accessible methods
#[pymethods]
impl Wall {
    /// Construct a mostly empty wall IDS from its settings.
    /// Later it will be fully populated by reading from a database, with `add_limiter_unit`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the wall IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_wall` data types, e.g. a `pulseNo` too large for an `i64`
    ///
    /// `description_2d(0)` is created here, so that the units added later have somewhere to
    /// go. Its `type` is set to `multiple_units_no_vessel`, because we describe the limiter
    /// with a set of units and leave the mechanical vessel description empty.
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the wall's settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the wall's settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!("The wall's settings do not follow their XML schema:\n{report}")));
        }

        // Parse the XML string settings into a typed `SettingsWall` struct
        let settings_wall: SettingsWall = SettingsWall::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The wall's settings cannot be read into `SettingsWall`: {error}")))?;

        let description_2d: Wall2d = Wall2d {
            r#type: IdentifierStatic {
                name: "multiple_units_no_vessel".to_string(),
                index: 1,
                description: "Limiter is described with multiple units, no vessel description".to_string(),
            },
            ..Default::default()
        };

        // Construct the wall IDS from the parsed settings
        let wall_ids: WallIds = WallIds {
            description_2d: vec![description_2d],
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self { wall_ids, settings_wall })
    }

    /// Append a limiter unit to `wall/description_2d(0)/limiter/unit`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the unit, e.g. `"vacuum_vessel"`
    /// * `r` - outline radial points (1d array), [metre]
    /// * `z` - outline vertical points (1d array), [metre]
    ///
    /// The **first** unit added is the vacuum vessel contour; see the `Wall` documentation.
    ///
    /// The data dictionary asks for the first point to be repeated when the contour is
    /// closed; that is left to the caller, since a unit made of a handful of tile points is
    /// not a contour at all.
    pub fn add_limiter_unit(&mut self, name: &str, r: PyReadonlyArray1<f64>, z: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let unit_r: Array1<f64> = r.to_owned_array();
        let unit_z: Array1<f64> = z.to_owned_array();

        if unit_r.len() != unit_z.len() {
            return Err(PyValueError::new_err(format!(
                "limiter unit `{}`: `r` has {} point(s) but `z` has {}; an outline needs one `z` per `r`",
                name,
                unit_r.len(),
                unit_z.len()
            )));
        }

        if unit_r.is_empty() {
            return Err(PyValueError::new_err(format!("limiter unit `{name}`: the outline is empty")));
        }

        let limiter_unit: Wall2dLimiterUnit = Wall2dLimiterUnit {
            name: name.to_string(),
            outline: Rz1dStatic { r: unit_r, z: unit_z },
            ..Default::default()
        };

        let description_2d: &mut Wall2d = self
            .wall_ids
            .description_2d
            .first_mut()
            .ok_or_else(|| PyValueError::new_err("`wall/description_2d` is empty"))?;
        description_2d.limiter.unit.push(limiter_unit);

        Ok(())
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.wall_paths`, straight out of the wall
    /// IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.wall_ids, "wall", path)
    }

    /// A copy of the whole wall IDS, for reading with `gsfit_rs.imas.wall_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn wall_ids(&self) -> PyWall {
        PyWall::new(self.wall_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Wall>");
        string_output += &format!("║  {:<74} ║\n", version);

        // `description_2d(0)` always exists, because `Wall::new` creates it
        let units: &Vec<Wall2dLimiterUnit> = &self.wall_ids.description_2d[0].limiter.unit;
        let n_units: usize = units.len();
        if n_units == 0 {
            string_output += &format!("║  {:<74} ║\n", " `wall/description_2d(0)/limiter/unit` is empty");
        }
        for i_unit in 0..n_units {
            let unit_name: &str = if units[i_unit].name.is_empty() { "<unnamed>" } else { &units[i_unit].name };
            let n_points: usize = units[i_unit].outline.r.len();
            string_output += &format!("║  {:<74} ║\n", format!(" limiter unit({i_unit}) = {unit_name}; points = {n_points}"));
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `wall_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_wall`, are not pickled: they are read again from the XML which `wall_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("wall_ids", self.wall_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_wall`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Wall` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `Wall` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let wall_ids_state: Bound<'_, PyAny> = state
            .get_item("wall_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Wall` has no `wall_ids`"))?;
        let wall_ids: WallIds = PickleState::from_pickle_state(&wall_ids_state).map_err(|error: PickleStateError| error.into_py_err("`Wall.wall_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_wall: SettingsWall = SettingsWall::from_xml(&wall_ids.code.parameters)
            .map_err(|error: String| PyValueError::new_err(format!("The pickled `Wall`'s settings cannot be read into `SettingsWall`: {error}")))?;

        Ok(Self { wall_ids, settings_wall })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/wall/wall.xsd");

    /// `wall.xml` without its optional link to `workflow.xml`, so it follows `wall.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<wall_settings/>
"#;

    fn parameters_input_with_xml(parameters_value: &str) -> PyParametersInput {
        PyParametersInput::new(ParametersInput {
            parameters_value: parameters_value.to_string(),
            parameters_default: String::new(),
            schema: SCHEMA_XSD.to_string(),
        })
    }

    #[test]
    fn the_settings_xml_is_written_to_code_parameters() {
        let wall: Wall = Wall::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(wall.wall_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        let wall: Wall = Wall::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(wall.settings_wall.workflow, None);
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(Wall::new(&parameters_input_with_xml("")).is_err());
        // Another IDS's settings
        assert!(Wall::new(&parameters_input_with_xml("<tf_settings/>")).is_err());
        // An element which is not in the XML schema
        assert!(Wall::new(&parameters_input_with_xml("<wall_settings><extra/></wall_settings>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(Wall::new(&parameters_input).is_err());
    }
}
