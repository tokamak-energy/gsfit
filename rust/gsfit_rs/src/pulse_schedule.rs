use crate::settings::settings_pulse_schedule::SettingsPulseSchedule;
use imas_rs::ids::pulse_schedule::{Code, PulseSchedule as PulseScheduleIds, PulseScheduleGap};
use imas_rs::python::{PickleState, PickleStateError, PyParametersInput, PyPath, PyPulseSchedule, read_path};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's pulse schedule, stored as an IMAS `pulse_schedule` IDS.
///
/// Only the gap definitions are filled so far:
/// * `pulse_schedule/position_control/gap(i)/name`
/// * `pulse_schedule/position_control/gap(i)/r`
/// * `pulse_schedule/position_control/gap(i)/z`
/// * `pulse_schedule/position_control/gap(i)/angle`
///
/// A gap is a reference point and a direction. Its value is the distance from the reference point
/// to the plasma boundary along that direction, which the equilibrium post-processor calculates.
/// The reference waveforms, `gap(i)/value`, are deliberately left empty: GSFit measures the gaps,
/// it does not control them.
///
/// `Equilibrium.copy_from_pulse_schedule_ids` copies the definitions onto every time-slice's
/// `equilibrium/time_slice(itime)/boundary/gap`, which is where the calculated values go.
///
/// # Angle convention
///
/// The data dictionary gives `pulse_schedule/position_control/gap/angle` only as the "angle
/// between the direction in which the gap is measured (in the poloidal cross-section) and the
/// horizontal axis", without a sense of rotation. `equilibrium/time_slice/boundary/gap/angle` is
/// explicit: "measured clockwise from radial cylindrical vector (grad R) to gap vector (pointing
/// away from reference point)".
///
/// The angle here is stored in the equilibrium's convention, so that it is copied across
/// unchanged and the two IDSs cannot disagree. Clockwise is in the usual plot with `R` to the
/// right and `Z` upwards, the same sense as `profiles_2d/theta`, so the gap direction is
/// `(cos(angle), -sin(angle))` in `(R, Z)`. A database reader holding a counter-clockwise angle
/// must convert it.
///
/// Read the data back through `pulse_schedule_ids` and a path from
/// `gsfit_rs.imas.pulse_schedule_paths`; there are no bespoke accessors.
///
/// # Settings
///
/// The pulse schedule is constructed with the settings it is built from, e.g. which run of which
/// database the gaps are read from, as an IMAS `parameters_input`. They are validated against the
/// XML schema in `parameters_input`, and only their XML is kept: it is written to
/// `pulse_schedule/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_pulse_schedule`, whose structs are generated from the XML schema
/// `pulse_schedule.xsd`. They are kept here, not in the IDS, as the pulse_schedule IDS can also be made
/// by another code, whose `pulse_schedule/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct PulseSchedule {
    pub pulse_schedule_ids: PulseScheduleIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_pulse_schedule: SettingsPulseSchedule,
}

/// Python accessible methods
#[pymethods]
impl PulseSchedule {
    /// Construct a mostly empty pulse schedule IDS, with no gaps, from its settings.
    /// Later it will be fully populated by reading from a database, with `add_gap`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the pulse schedule IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_pulse_schedule` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the pulse schedule's settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the pulse schedule's settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The pulse schedule's settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsPulseSchedule` struct
        let settings_pulse_schedule: SettingsPulseSchedule = SettingsPulseSchedule::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The pulse schedule's settings cannot be read into `SettingsPulseSchedule`: {error}")))?;

        // Construct the pulse schedule IDS from the parsed settings
        let pulse_schedule_ids: PulseScheduleIds = PulseScheduleIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            pulse_schedule_ids,
            settings_pulse_schedule,
        })
    }

    /// Append a gap to `pulse_schedule/position_control/gap`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the gap, unique within the pulse schedule, e.g. `"IMGAP"`
    /// * `r` - major radius of the reference point, [metre]
    /// * `z` - height of the reference point, [metre]
    /// * `angle` - direction the gap is measured in, clockwise from `grad(R)`, [radian]; see the
    ///   `PulseSchedule` documentation
    pub fn add_gap(&mut self, name: &str, r: f64, z: f64, angle: f64) -> PyResult<()> {
        let gap: PulseScheduleGap = checked_gap(&self.pulse_schedule_ids.position_control.gap, name, r, z, angle).map_err(PyValueError::new_err)?;

        self.pulse_schedule_ids.position_control.gap.push(gap);

        Ok(())
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.pulse_schedule_paths`, straight out of
    /// the pulse_schedule IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.pulse_schedule_ids, "pulse_schedule", path)
    }

    /// A copy of the whole pulse_schedule IDS, for reading with `gsfit_rs.imas.pulse_schedule_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn pulse_schedule_ids(&self) -> PyPulseSchedule {
        PyPulseSchedule::new(self.pulse_schedule_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.PulseSchedule>");
        string_output += &format!("║  {:<74} ║\n", version);

        let gaps: &Vec<PulseScheduleGap> = &self.pulse_schedule_ids.position_control.gap;
        let n_gaps: usize = gaps.len();
        if n_gaps == 0 {
            string_output += &format!("║  {:<74} ║\n", " no gaps");
        }
        for i_gap in 0..n_gaps {
            let gap: &PulseScheduleGap = &gaps[i_gap];
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(
                    " gap({i_gap}) = {}; r = {} metre; z = {} metre; angle = {} radian",
                    gap.name, gap.r, gap.z, gap.angle
                )
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `pulse_schedule_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_pulse_schedule`, are not pickled: they are read again from the XML which `pulse_schedule_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pulse_schedule_ids", self.pulse_schedule_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_pulse_schedule`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PulseSchedule` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `PulseSchedule` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let pulse_schedule_ids_state: Bound<'_, PyAny> = state
            .get_item("pulse_schedule_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PulseSchedule` has no `pulse_schedule_ids`"))?;
        let pulse_schedule_ids: PulseScheduleIds = PickleState::from_pickle_state(&pulse_schedule_ids_state)
            .map_err(|error: PickleStateError| error.into_py_err("`PulseSchedule.pulse_schedule_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_pulse_schedule: SettingsPulseSchedule =
            SettingsPulseSchedule::from_xml(&pulse_schedule_ids.code.parameters).map_err(|error: String| {
                PyValueError::new_err(format!(
                    "The pickled `PulseSchedule`'s settings cannot be read into `SettingsPulseSchedule`: {error}"
                ))
            })?;

        Ok(Self {
            pulse_schedule_ids,
            settings_pulse_schedule,
        })
    }
}

/// Build one gap definition, checked for being usable and for not repeating an existing name.
///
/// # Arguments
/// * `gaps` - the gaps already in the pulse schedule
/// * `name` - short identifier for the gap
/// * `r` - major radius of the reference point, [metre]
/// * `z` - height of the reference point, [metre]
/// * `angle` - direction the gap is measured in, clockwise from `grad(R)`, [radian]
fn checked_gap(gaps: &[PulseScheduleGap], name: &str, r: f64, z: f64, angle: f64) -> Result<PulseScheduleGap, String> {
    if name.is_empty() {
        return Err("gap: `name` is empty; a gap needs a short identifier".to_string());
    }

    // The data dictionary asks for the name to be "unique for a given device", and a repeated name
    // would make the gaps impossible to tell apart downstream
    let n_gaps: usize = gaps.len();
    for i_gap in 0..n_gaps {
        if gaps[i_gap].name == name {
            return Err(format!("gap `{name}`: a gap with this name already exists, at `position_control/gap({i_gap})`"));
        }
    }

    if !r.is_finite() || r <= 0.0 {
        return Err(format!("gap `{name}`: `r` must be a positive major radius, got {r}"));
    }
    if !z.is_finite() {
        return Err(format!("gap `{name}`: `z` must be finite, got {z}"));
    }
    if !angle.is_finite() {
        return Err(format!("gap `{name}`: `angle` must be finite, got {angle}"));
    }

    Ok(PulseScheduleGap {
        name: name.to_string(),
        r,
        z,
        angle,
        ..PulseScheduleGap::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/pulse_schedule/pulse_schedule.xsd");

    /// `pulse_schedule.xml` without its optional link to `workflow.xml`, so it follows `pulse_schedule.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pulse_schedule_settings/>
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
        let pulse_schedule: PulseSchedule = PulseSchedule::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(pulse_schedule.pulse_schedule_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        let pulse_schedule: PulseSchedule = PulseSchedule::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(pulse_schedule.settings_pulse_schedule.workflow, None);
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(PulseSchedule::new(&parameters_input_with_xml("")).is_err());
        // Another IDS's settings
        assert!(PulseSchedule::new(&parameters_input_with_xml("<tf_settings/>")).is_err());
        // An element which is not in the XML schema
        assert!(PulseSchedule::new(&parameters_input_with_xml("<pulse_schedule_settings><extra/></pulse_schedule_settings>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(PulseSchedule::new(&parameters_input).is_err());
    }

    #[test]
    fn a_gap_is_stored_with_its_definition() {
        let gap: PulseScheduleGap = checked_gap(&[], "IMGAP", 0.175, 0.0, 0.0).unwrap();

        assert_eq!(gap.name, "IMGAP");
        assert_eq!(gap.r, 0.175);
        assert_eq!(gap.z, 0.0);
        assert_eq!(gap.angle, 0.0);
        // Only the definition is filled; the reference waveform is left empty
        assert!(gap.value.reference.is_empty());
    }

    #[test]
    fn a_repeated_name_is_an_error_naming_the_existing_gap() {
        let imgap: PulseScheduleGap = checked_gap(&[], "IMGAP", 0.175, 0.0, 0.0).unwrap();
        let mctgap: PulseScheduleGap = checked_gap(std::slice::from_ref(&imgap), "MCTGAP", 0.716, 0.32, 2.48).unwrap();
        let gaps: Vec<PulseScheduleGap> = vec![imgap, mctgap];

        let error: String = checked_gap(&gaps, "MCTGAP", 0.716, 0.32, 2.48).unwrap_err();

        assert!(error.contains("position_control/gap(1)"), "{error}");
    }

    #[test]
    fn an_unusable_definition_is_an_error() {
        assert!(checked_gap(&[], "", 0.175, 0.0, 0.0).is_err());
        assert!(checked_gap(&[], "IMGAP", 0.0, 0.0, 0.0).is_err());
        assert!(checked_gap(&[], "IMGAP", f64::NAN, 0.0, 0.0).is_err());
        assert!(checked_gap(&[], "IMGAP", 0.175, f64::INFINITY, 0.0).is_err());
        assert!(checked_gap(&[], "IMGAP", 0.175, 0.0, f64::NAN).is_err());
    }
}
