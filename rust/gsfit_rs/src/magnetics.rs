use crate::settings::settings_magnetics::SettingsMagnetics;
use imas_rs::ids::magnetics::{Code, Magnetics as MagneticsIds};
use imas_rs::python::{PickleState, PickleStateError, PyMagnetics, PyParametersInput, PyPath, read_path, structure_nodes};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's magnetic sensors, stored as an IMAS `magnetics` IDS.
///
/// For each poloidal field probe, `b_field_pol_probe(i1)`, these are filled:
/// * `name`
/// * `position/r` and `position/z` - the probe's centre
/// * `poloidal_angle` - the direction of the probe's normal; see "Poloidal angle" below
/// * `field` - the measured field along the probe's normal, on the experimental timebase
///
/// For each flux loop, `flux_loop(i1)`:
/// * `name`
/// * `type` = 1, `poloidal_flux` - a loop all the way around the machine, at one major radius and height
/// * `position(0)/r` and `position(0)/z` - the loop's one point in the poloidal plane
/// * `flux` - the measured poloidal flux, on the experimental timebase
///
/// For each Rogowski coil, `rogowski_coil(i1)`:
/// * `name`
/// * `position(i2)/r` and `position(i2)/z` - the coil's contour in the poloidal plane
/// * `current` - the measured current, on the experimental timebase
///
/// The stretches of a Rogowski coil's contour which have no winding are kept in `rogowski_coil_gaps`, not in the IDS, as the data
/// dictionary has no node for them, and the magnetics IDS can also be made by another code.
///
/// For the diamagnetic flux, `diamagnetic_flux(0)`, and the plasma current, `ip(i1)`:
/// * `method_name`
/// * `data` and `time` - on the experimental timebase
///
/// The data dictionary allows one entry for each method of calculating them. GSFit takes the plasma current's first, generally
/// recommended, method, but accepts only one diamagnetic flux, as the equilibrium IDS has a single diamagnetic flux constraint, so the
/// signals of several diamagnetic loops are averaged into one before it is added.
///
/// The constraints at points which can move are custom keys, as the data dictionary has none, each a time trace on its own timebase,
/// with `name`, `description`, `validity_timed`, `validity` and `time`:
/// * `isoflux(i1)` - two points which lie on the same flux surface, `position_1` and `position_2`
/// * `isoflux_boundary(i1)` - a point which lies on the plasma boundary, `position`
/// * `stationary_point(i1)` - a point where both derivatives of the poloidal flux are zero, an X-point or the magnetic axis, `position`
///
/// Nothing else is filled, e.g. each position's `phi`, the probes' `toroidal_angle`, `area` and `turns`, and the loops' `area`:
/// GSFit does not use them.
///
/// # Poloidal angle
///
/// **For now, `b_field_pol_probe(i1)/poloidal_angle` keeps GSFit's present definition, COCOS 13, as MDSplus stores it.** It is
/// measured counter-clockwise from `grad(R)`, in the usual plot with `R` to the right and `Z` upwards, so the probe's normal is
/// `cos(poloidal_angle) * grad(R) + sin(poloidal_angle) * grad(Z)`, and an angle of `PI / 2` points towards `+Z`. It is the angle
/// GSFit has always called `angle_pol`, and is stored as given, without being wrapped into a range.
///
/// This is **not** the data dictionary's definition. Its COCOS 17 measures the angle clockwise, so that the normal is
/// `cos(poloidal_angle) * grad(R) - sin(poloidal_angle) * grad(Z)`, and keeps it within [0, 2 * PI].
/// TODO: convert to COCOS 17, `poloidal_angle = (-poloidal_angle) mod (2 * PI)`, before the magnetics IDS is written to an IMAS
/// backend.
///
/// `toroidal_angle` is left empty: every probe's normal is taken to lie in the poloidal plane.
///
/// # Sign
///
/// * `flux_loop(i1)/flux` is GSFit's `psi`, the total poloidal flux. GSFit's COCOS 13 and the data dictionary's COCOS 17 both have
///   `e_Bp = 1` and `sigma_Bp = -1`, so there is no factor of 2 * PI and no change of sign.
/// * `ip(i1)/data` and `diamagnetic_flux(i1)/data` are positive counter-clockwise viewed from above.
/// * `rogowski_coil(i1)/current` is positive along the coil's normal, which the order of the `position` points sets by the
///   right-hand rule. For the normal to be counter-clockwise viewed from above, like the toroidal currents the coil encloses, the
///   points run clockwise in the usual plot.
///
/// Read the data back through `magnetics_ids` and a path from `gsfit_rs.imas.magnetics_paths`; there are no bespoke accessors.
///
/// # Settings
///
/// The magnetic sensors are constructed with the settings they are built from, e.g. which run of which database the sensors are
/// read from, as an IMAS `parameters_input`. They are validated against the XML schema in `parameters_input`, and only their XML is
/// kept: it is written to `magnetics/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_magnetics`, whose structs are generated from the XML schema `magnetics.xsd`. They are kept here,
/// not in the IDS, as the magnetics IDS can also be made by another code, whose `magnetics/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct Magnetics {
    pub magnetics_ids: MagneticsIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_magnetics: SettingsMagnetics,
    /// The gaps in each Rogowski coil's winding, one `Vec` for each `rogowski_coil`, in the same order; empty for a coil with none
    pub rogowski_coil_gaps: Vec<Vec<RogowskiCoilGap>>,
}

/// One gap in a Rogowski coil's winding, where its contour, `rogowski_coil(i1)/position`, has no turns.
///
/// The poloidal field along a gap is not integrated, so the coil measures the current it encloses less the line integral of the
/// poloidal field along every gap, divided by MU_0.
#[derive(Clone, Debug, PartialEq)]
pub struct RogowskiCoilGap {
    /// Short string identifier for the gap, unique within the Rogowski coil
    pub name: String,
    /// Major radius of the start of the gap, [metre].
    /// The gap is the straight line from (`r_start`, `z_start`) to (`r_end`, `z_end`), and the line integral of the poloidal field
    /// along it is taken from its start to its end; so for it to be the part of the contour's line integral which the winding misses,
    /// the gap runs in the direction of the coil's contour, `position`
    pub r_start: f64,
    /// Major radius of the end of the gap, [metre]
    pub r_end: f64,
    /// Height of the start of the gap, [metre]
    pub z_start: f64,
    /// Height of the end of the gap, [metre]
    pub z_end: f64,
}

/// A gap is pickled as a structure of an IDS is, as the `dict` of its fields, see `imas_rs::python::PickleState`, but with every
/// field, as every field of a gap is set
impl PickleState for RogowskiCoilGap {
    fn is_unset(&self) -> bool {
        self.name.is_unset() && self.r_start.is_unset() && self.r_end.is_unset() && self.z_start.is_unset() && self.z_end.is_unset()
    }

    fn to_pickle_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("name", self.name.to_pickle_state(py)?)?;
        state.set_item("r_start", self.r_start.to_pickle_state(py)?)?;
        state.set_item("r_end", self.r_end.to_pickle_state(py)?)?;
        state.set_item("z_start", self.z_start.to_pickle_state(py)?)?;
        state.set_item("z_end", self.z_end.to_pickle_state(py)?)?;
        Ok(state.into_any())
    }

    fn from_pickle_state(state: &Bound<'_, PyAny>) -> Result<Self, PickleStateError> {
        let nodes: Vec<(String, Bound<'_, PyAny>)> = structure_nodes(state)?;
        let mut gap: Self = Self {
            name: String::new(),
            r_start: f64::NAN,
            r_end: f64::NAN,
            z_start: f64::NAN,
            z_end: f64::NAN,
        };
        let n_node: usize = nodes.len();
        for i_node in 0..n_node {
            let (name, node_state): &(String, Bound<'_, PyAny>) = &nodes[i_node];
            match name.as_str() {
                "name" => gap.name = PickleState::from_pickle_state(node_state).map_err(|error: PickleStateError| error.in_node("name"))?,
                "r_start" => gap.r_start = PickleState::from_pickle_state(node_state).map_err(|error: PickleStateError| error.in_node("r_start"))?,
                "r_end" => gap.r_end = PickleState::from_pickle_state(node_state).map_err(|error: PickleStateError| error.in_node("r_end"))?,
                "z_start" => gap.z_start = PickleState::from_pickle_state(node_state).map_err(|error: PickleStateError| error.in_node("z_start"))?,
                "z_end" => gap.z_end = PickleState::from_pickle_state(node_state).map_err(|error: PickleStateError| error.in_node("z_end"))?,
                _ => return Err(PickleStateError::unknown_node(name)),
            }
        }
        Ok(gap)
    }
}

// The methods which add each kind of sensor are in `magnetic_sensors/`, one file per sensor, each with its own `#[pymethods]` block

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Construct a mostly empty magnetics IDS from its settings.
    /// Later it will be fully populated by reading from a database, with `add_b_field_pol_probe`, `add_flux_loop`, `add_rogowski_coil`, `add_diamagnetic_flux`,
    /// `add_ip`, `add_isoflux`, `add_isoflux_boundary` and `add_stationary_point`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the magnetics IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_magnetics` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the magnetic sensors' settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the magnetic sensors' settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The magnetic sensors' settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsMagnetics` struct
        let settings_magnetics: SettingsMagnetics = SettingsMagnetics::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The magnetic sensors' settings cannot be read into `SettingsMagnetics`: {error}")))?;

        // Construct the magnetics IDS from the parsed settings
        let magnetics_ids: MagneticsIds = MagneticsIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            magnetics_ids,
            settings_magnetics,
            rogowski_coil_gaps: Vec::new(),
        })
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
        let n_diamagnetic_flux: usize = self.magnetics_ids.diamagnetic_flux.len();
        let n_ip: usize = self.magnetics_ids.ip.len();
        let n_isoflux: usize = self.magnetics_ids.isoflux.len();
        let n_isoflux_boundary: usize = self.magnetics_ids.isoflux_boundary.len();
        let n_stationary_point: usize = self.magnetics_ids.stationary_point.len();

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Magnetics>");
        string_output += &format!("║  {:<74} ║\n", version);
        string_output += &format!("║  {:<74} ║\n", format!(" b_field_pol_probe: {n_b_field_pol_probe} probe(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" flux_loop: {n_flux_loop} loop(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" rogowski_coil: {n_rogowski_coil} coil(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" diamagnetic_flux: {n_diamagnetic_flux} method(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" ip: {n_ip} method(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" isoflux: {n_isoflux} constraint(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" isoflux_boundary: {n_isoflux_boundary} constraint(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" stationary_point: {n_stationary_point} constraint(s)"));
        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `magnetics_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`, and
    /// `rogowski_coil_gaps`, a `list` for each Rogowski coil of the `dict` of each of its gaps. `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_magnetics`, are not pickled: they are read again from the XML which `magnetics_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("magnetics_ids", self.magnetics_ids.to_pickle_state(py)?)?;
        state.set_item("rogowski_coil_gaps", self.rogowski_coil_gaps.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_magnetics`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Magnetics` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `Magnetics` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let magnetics_ids_state: Bound<'_, PyAny> = state
            .get_item("magnetics_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Magnetics` has no `magnetics_ids`"))?;
        let magnetics_ids: MagneticsIds =
            PickleState::from_pickle_state(&magnetics_ids_state).map_err(|error: PickleStateError| error.into_py_err("`Magnetics.magnetics_ids`"))?;
        let rogowski_coil_gaps_state: Bound<'_, PyAny> = state
            .get_item("rogowski_coil_gaps")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Magnetics` has no `rogowski_coil_gaps`"))?;
        let rogowski_coil_gaps: Vec<Vec<RogowskiCoilGap>> =
            PickleState::from_pickle_state(&rogowski_coil_gaps_state).map_err(|error: PickleStateError| error.into_py_err("`Magnetics.rogowski_coil_gaps`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_magnetics: SettingsMagnetics = SettingsMagnetics::from_xml(&magnetics_ids.code.parameters)
            .map_err(|error: String| PyValueError::new_err(format!("The pickled `Magnetics`'s settings cannot be read into `SettingsMagnetics`: {error}")))?;

        Ok(Self {
            magnetics_ids,
            settings_magnetics,
            rogowski_coil_gaps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/magnetics/magnetics.xsd");

    /// `magnetics.xml` without its optional link to `workflow.xml`, so it follows `magnetics.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<magnetics_settings/>
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
        let magnetics: Magnetics = Magnetics::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(magnetics.magnetics_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        let magnetics: Magnetics = Magnetics::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(magnetics.settings_magnetics.workflow, None);
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(Magnetics::new(&parameters_input_with_xml("")).is_err());
        // Another IDS's settings
        assert!(Magnetics::new(&parameters_input_with_xml("<tf_settings/>")).is_err());
        // An element which is not in the XML schema
        assert!(Magnetics::new(&parameters_input_with_xml("<magnetics_settings><extra/></magnetics_settings>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(Magnetics::new(&parameters_input).is_err());
    }

    #[test]
    fn a_rogowski_coil_keeps_its_gaps_outside_the_ids() {
        let mut magnetics: Magnetics = Magnetics::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        magnetics
            .add_rogowski_coil_rs(
                "INIVC000",
                ndarray::Array1::from_vec(vec![0.2, 0.9, 0.9, 0.2, 0.2]),
                ndarray::Array1::from_vec(vec![-0.8, -0.8, 0.8, 0.8, -0.8]),
                ndarray::Array1::from_vec(vec![0.0, 1.0]),
                ndarray::Array1::from_vec(vec![1.0e5, 1.0e5]),
                vec!["GAP00".to_string()],
                ndarray::Array1::from_vec(vec![0.9]),
                ndarray::Array1::from_vec(vec![0.9]),
                ndarray::Array1::from_vec(vec![-0.1]),
                ndarray::Array1::from_vec(vec![0.1]),
            )
            .expect("the Rogowski coil is valid");

        assert_eq!(magnetics.rogowski_coil_gaps.len(), 1);
        assert_eq!(magnetics.rogowski_coil_gaps[0].len(), 1);
        assert_eq!(magnetics.rogowski_coil_gaps[0][0].name, "GAP00");
        assert_eq!(magnetics.rogowski_coil_gaps[0][0].z_end, 0.1);
    }
}
