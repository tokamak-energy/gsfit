use crate::settings::settings_tf::SettingsTf;
use imas_rs::ids::tf::{Code, SignalFlt1d, Tf as TfIds};
use imas_rs::python::{PickleState, PickleStateError, PyParametersInput, PyPath, PyTf, read_path};
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's toroidal field, stored as an IMAS `tf` IDS.
///
/// Two things are filled:
/// * `tf/r0` - the reference major radius the vacuum toroidal field is quoted at, from the settings
/// * `tf/b_field_phi_vacuum_r` - the vacuum field times major radius, on the experimental
///   timebase
///
/// `b_field_phi_vacuum_r` is the poloidal-current function in vacuum, which GSFit writes as
/// `f_vac = R0 * B_phi0 = MU_0 * i_rod / (2 * PI)`. So the rod current is not stored
/// separately: it is `i_rod = 2 * PI * f_vac / MU_0`, and `solve_grad_shafranov` converts it
/// where it is needed. The data dictionary has no node for a rod current outside
/// `coil(i)/current`, and one signal cannot disagree with itself.
///
/// # Sign
///
/// `b_field_phi_vacuum_r` is signed: the data dictionary reads "Positive sign means
/// counter-clockwise when viewed from above". The sign carries through to `f_vac`, and from
/// there into `f`, `q`, `profiles_2d/b_field_phi` and the diamagnetic flux constraint, so a
/// database reader must supply it signed rather than as a magnitude.
///
/// Read the data back through `tf_ids` and a path from `gsfit_rs.imas.tf_paths`; there are no
/// bespoke accessors.
///
/// The coil geometry (`tf/coil`), the field map (`tf/field_map`) and the periodicity flags are
/// deliberately left empty: GSFit models the toroidal field as axisymmetric and needs only the
/// two nodes above.
///
/// # Settings
///
/// The toroidal field is constructed with the settings it is built from, e.g. the reference major
/// radius, and which run of which database the field is read from, as an IMAS `parameters_input`.
/// They are validated against the XML schema in `parameters_input`, and only their XML is kept: it is
/// written to `tf/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_tf`, whose structs are generated from the XML schema `tf.xsd`. They
/// are kept here, not in the IDS, as the tf IDS can also be made by another code, whose
/// `tf/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct Tf {
    pub tf_ids: TfIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_tf: SettingsTf,
}

/// Python accessible methods
#[pymethods]
impl Tf {
    /// Construct a mostly empty TF (toroidal field) IDS from its settings.
    /// Later it will be fully populated by reading from a database.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the TF IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.defaults` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_tf` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the toroidal field's settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the toroidal field's settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The toroidal field's settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsTf` struct
        let settings_tf: SettingsTf = SettingsTf::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The toroidal field's settings cannot be read into `SettingsTf`: {error}")))?;

        // Construct the TF IDS from the parsed settings
        let tf_ids: TfIds = TfIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            r0: settings_tf.vacuum_toroidal_field_reference_radius,
            ..Default::default()
        };

        Ok(Self { tf_ids, settings_tf })
    }

    /// Set `tf/b_field_phi_vacuum_r`, the vacuum field times major radius.
    ///
    /// # Arguments
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `data` - vacuum toroidal field times major radius (1d array), [tesla * metre]
    ///
    /// Store the **experimental** signal here, not one interpolated onto the reconstruction
    /// times: `Equilibrium.copy_from_tf_ids` interpolates it itself, so that changing the reconstruction
    /// times does not require re-reading the database.
    ///
    /// `data` is signed; see the `Tf` documentation. A reader holding a rod current instead
    /// should pass `MU_0 * i_rod / (2 * PI)`.
    pub fn set_b_field_phi_vacuum_r(&mut self, time: PyReadonlyArray1<f64>, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let signal_time: Array1<f64> = time.to_owned_array();
        let signal_data: Array1<f64> = data.to_owned_array();

        if signal_time.len() != signal_data.len() {
            return Err(PyValueError::new_err(format!(
                "`tf/b_field_phi_vacuum_r`: `time` has {} point(s) but `data` has {}; a signal needs one value per time",
                signal_time.len(),
                signal_data.len()
            )));
        }

        if signal_time.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "`tf/b_field_phi_vacuum_r`: {} point(s) is not enough to interpolate from; at least 2 are needed",
                signal_time.len()
            )));
        }

        self.tf_ids.b_field_phi_vacuum_r = SignalFlt1d {
            data: signal_data,
            time: signal_time,
        };

        Ok(())
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.tf_paths`, straight out of the tf IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.tf_ids, "tf", path)
    }

    /// A copy of the whole tf IDS, for reading with `gsfit_rs.imas.tf_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn tf_ids(&self) -> PyTf {
        PyTf::new(self.tf_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Tf>");
        string_output += &format!("║  {:<74} ║\n", version);

        let r0: f64 = self.tf_ids.r0;
        string_output += &format!("║  {:<74} ║\n", format!(" r0 = {r0} metre"));

        let signal_time: &Array1<f64> = &self.tf_ids.b_field_phi_vacuum_r.time;
        let signal_data: &Array1<f64> = &self.tf_ids.b_field_phi_vacuum_r.data;
        if signal_time.is_empty() || signal_data.is_empty() {
            string_output += &format!("║  {:<74} ║\n", " b_field_phi_vacuum_r is unset");
        } else {
            let n_time: usize = signal_time.len();
            string_output += &format!("║  {:<74} ║\n", format!(" b_field_phi_vacuum_r: {n_time} time point(s)"));
            string_output += &format!("║  {:<74} ║\n", format!(" time = [{}, {}] second", signal_time[0], signal_time[n_time - 1]));
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" data = [{}, {}] tesla * metre", signal_data[0], signal_data[n_time - 1])
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `tf_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_tf`, are not pickled: they are read again from the XML which `tf_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("tf_ids", self.tf_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_tf`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Tf` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `Tf` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let tf_ids_state: Bound<'_, PyAny> = state
            .get_item("tf_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Tf` has no `tf_ids`"))?;
        let tf_ids: TfIds = PickleState::from_pickle_state(&tf_ids_state).map_err(|error: PickleStateError| error.into_py_err("`Tf.tf_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_tf: SettingsTf = SettingsTf::from_xml(&tf_ids.code.parameters)
            .map_err(|error: String| PyValueError::new_err(format!("The pickled `Tf`'s settings cannot be read into `SettingsTf`: {error}")))?;

        Ok(Self { tf_ids, settings_tf })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::settings_tf::SettingsTfWorkflow;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/tf/tf.xsd");

    /// `tf.xml` without its optional link to `workflow.xml`, so it follows `tf.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<tf_settings>
    <vacuum_toroidal_field_reference_radius>0.4</vacuum_toroidal_field_reference_radius>
</tf_settings>
"#;

    /// `tf.xml`, with `workflow.xml` written in its link's place, as `serialise_python_dataclass_to_imas_parameters_input` writes it
    const SETTINGS_WITH_WORKFLOW_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<tf_settings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <vacuum_toroidal_field_reference_radius>0.4</vacuum_toroidal_field_reference_radius>
    <workflow>
        <tree>
            <tree_name>PSU2COIL</tree_name>
            <pulseNo xsi:nil="true" />
            <run_name>BEST</run_name>
            <usage>The currents</usage>
        </tree>
        <tree>
            <tree_name>MAG</tree_name>
            <pulseNo>-1</pulseNo>
            <run_name>RUN16</run_name>
            <usage>The magnetic sensors</usage>
        </tree>
    </workflow>
</tf_settings>
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
        let tf: Tf = Tf::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(tf.tf_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_into_structs() {
        let settings_tf: SettingsTf = SettingsTf::from_xml(SETTINGS_WITH_WORKFLOW_XML).expect("the settings can be read");

        assert_eq!(settings_tf.vacuum_toroidal_field_reference_radius, 0.4);

        let workflow: &SettingsTfWorkflow = settings_tf.workflow.as_ref().expect("the settings link to workflow.xml");
        assert_eq!(workflow.tree.len(), 2);
        assert_eq!(workflow.tree[0].tree_name, "PSU2COIL");
        assert_eq!(workflow.tree[0].pulseNo, None);
        assert_eq!(workflow.tree[0].run_name, "BEST");
        assert_eq!(workflow.tree[1].tree_name, "MAG");
        assert_eq!(workflow.tree[1].pulseNo, Some(-1));
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        let tf: Tf = Tf::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(tf.settings_tf.workflow, None);
    }

    #[test]
    fn the_reference_radius_in_the_settings_sets_r0() {
        let tf: Tf = Tf::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(tf.tf_ids.r0, 0.4);
    }

    #[test]
    fn a_reference_radius_which_is_not_a_positive_major_radius_is_an_error() {
        let r0_texts: [&str; 8] = ["0.0", "-1.0", "NaN", "INF", "abc", "0e0", "0.0e5", "1e-400"];
        for r0_text in r0_texts {
            let xml: String = format!("<tf_settings><vacuum_toroidal_field_reference_radius>{r0_text}</vacuum_toroidal_field_reference_radius></tf_settings>");

            assert!(Tf::new(&parameters_input_with_xml(&xml)).is_err(), "`{r0_text}` was accepted");
        }
    }

    #[test]
    fn a_reference_radius_can_be_written_in_any_way_a_number_can() {
        let r0_texts: [&str; 6] = ["0.4", "+0.4", "4e-1", "+4e-1", ".4e0", " 0.4 "];
        for r0_text in r0_texts {
            let xml: String = format!("<tf_settings><vacuum_toroidal_field_reference_radius>{r0_text}</vacuum_toroidal_field_reference_radius></tf_settings>");

            let tf: Tf = Tf::new(&parameters_input_with_xml(&xml)).expect("the settings follow their XML schema");

            assert_eq!(tf.tf_ids.r0, 0.4, "`{r0_text}` was not read as 0.4");
        }
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(Tf::new(&parameters_input_with_xml("")).is_err());
        // Missing the required reference radius
        assert!(Tf::new(&parameters_input_with_xml("<tf_settings/>")).is_err());
        // Another IDS's settings
        assert!(Tf::new(&parameters_input_with_xml("<wall_settings/>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(Tf::new(&parameters_input).is_err());
    }
}
