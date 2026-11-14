use crate::greens::Greens;
use crate::settings::settings_pf_active::SettingsPfActive;
use imas_rs::ids::pf_active::{Code, Outline2dGeometryStatic, PfActive as PfActiveIds, PfCoils, PfCoilsElements, RectangleStatic, SignalFlt1d};
use imas_rs::python::{PickleState, PickleStateError, PyParametersInput, PyPath, PyPfActive, read_path};
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's active poloidal field (PF) coils, stored as an IMAS `pf_active` IDS.
///
/// For each coil, `coil(i1)`, these are filled:
/// * `name`
/// * `element(i2)/geometry/rectangle/r`, `z`, `width` and `height` - one element per filament,
///   with `element(i2)/geometry/geometry_type` = 2, a rectangle
/// * `element(i2)/turns_with_sign` = 1 - every filament is one turn
/// * `inductance` - the self-inductance, calculated from the filaments
/// * `current` - the current per turn, on the experimental timebase
/// * `voltage` - only when the coil's voltage is prescribed
///
/// # Turns
///
/// The data dictionary gives `coil(i1)/current` for one turn, "to be multiplied by the number of
/// turns to obtain the generated magnetic field". Every filament is one turn, so every filament
/// carries the whole of `coil(i1)/current`.
///
/// # Sign
///
/// `coil(i1)/current` is signed: the data dictionary reads "Positive when the current flows
/// counter-clockwise in the coil when viewed from above".
///
/// # Prescribed current or voltage
///
/// A coil's current is prescribed, unless the coil has a voltage:
/// * `add_pf_coil` gives the coil its `coil(i1)/current`
/// * `set_pf_voltage_prescribed` gives the coil its `coil(i1)/voltage`, and `solve_circuit_equations`
///   then calculates its current
///
/// `Equilibrium.copy_from_pf_active_ids` copies each coil's current onto every time-slice of the
/// equilibrium IDS, as `constraints/pf_current(i1)`, interpolated onto the reconstruction times.
///
/// Read the data back through `pf_active_ids` and a path from `gsfit_rs.imas.pf_active_paths`;
/// there are no bespoke accessors.
///
/// The circuits (`pf_active/circuit`) and supplies (`pf_active/supply`) are left empty: a power
/// supply which feeds several coils in series is described by giving each coil its current.
///
/// # Settings
///
/// The PF coils are constructed with the settings they are built from, e.g. which run of which
/// database the coils are read from, as an IMAS `parameters_input`. They are validated against the
/// XML schema in `parameters_input`, and only their XML is kept: it is written to
/// `pf_active/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_pf_active`, whose structs are generated from the XML schema
/// `pf_active.xsd`. They are kept here, not in the IDS, as the pf_active IDS can also be made by
/// another code, whose `pf_active/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct PfActive {
    pub pf_active_ids: PfActiveIds,
    pub settings_pf_active: SettingsPfActive,
}

/// Python accessible methods
#[pymethods]
impl PfActive {
    /// Construct a mostly empty PF active IDS from its settings.
    /// Later it will be fully populated by reading from a database, with `add_pf_coil`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the PF active IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_pf_active` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the PF coils' settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the PF coils' settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The PF coils' settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsPfActive` struct
        let settings_pf_active: SettingsPfActive = SettingsPfActive::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The PF coils' settings cannot be read into `SettingsPfActive`: {error}")))?;

        // Construct the PF active IDS from the parsed settings
        let pf_active_ids: PfActiveIds = PfActiveIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            pf_active_ids,
            settings_pf_active,
        })
    }

    /// Append a coil to `pf_active/coil`, with its current prescribed.
    ///
    /// # Arguments
    /// * `name` - short identifier for the coil, unique within the PF coils, e.g. `"BVL"`
    /// * `r` - major radius of each filament's centre (1d array), [metre]
    /// * `z` - height of each filament's centre (1d array), [metre]
    /// * `width` - radial full width of each filament (1d array), [metre]
    /// * `height` - vertical full height of each filament (1d array), [metre]
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `current` - the current per turn (1d array), [ampere]
    ///
    /// Each filament becomes one `element(i2)`: a rectangle of one turn. The coil's self-inductance,
    /// `coil(i1)/inductance`, is calculated from them.
    ///
    /// Store the **experimental** current here, not one interpolated onto the reconstruction
    /// times: `Equilibrium.copy_from_pf_active_ids` interpolates it itself. It is signed; see the
    /// `PfActive` documentation. A single time holds the current at that time only.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pf_coil(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        width: PyReadonlyArray1<f64>,
        height: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        current: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let element_r: Array1<f64> = r.to_owned_array();
        let element_z: Array1<f64> = z.to_owned_array();
        let element_width: Array1<f64> = width.to_owned_array();
        let element_height: Array1<f64> = height.to_owned_array();
        let current_time: Array1<f64> = time.to_owned_array();
        let current_data: Array1<f64> = current.to_owned_array();

        // Run the Rust method
        self.add_pf_coil_rs(name, element_r, element_z, element_width, element_height, current_time, current_data)
            .map_err(PyValueError::new_err)
    }

    /// Prescribe a coil's voltage, rather than its current.
    ///
    /// # Arguments
    /// * `name` - the coil, as named in `add_pf_coil`
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `voltage` - voltage on the coil terminals (1d array), [volt]
    ///
    /// Writes `coil(i1)/voltage`, which is what makes the coil's voltage prescribed. The coil's
    /// `current` is kept.
    pub fn set_pf_voltage_prescribed(&mut self, name: &str, time: PyReadonlyArray1<f64>, voltage: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let voltage_time: Array1<f64> = time.to_owned_array();
        let voltage_data: Array1<f64> = voltage.to_owned_array();

        if voltage_time.len() != voltage_data.len() {
            return Err(PyValueError::new_err(format!(
                "PF coil `{name}`: `time` has {} point(s) but `voltage` has {}; a signal needs one value per time",
                voltage_time.len(),
                voltage_data.len()
            )));
        }

        if voltage_time.is_empty() {
            return Err(PyValueError::new_err(format!(
                "PF coil `{name}`: `time` is empty; the voltage needs at least one time"
            )));
        }

        let n_coil: usize = self.pf_active_ids.coil.len();
        for i_coil in 0..n_coil {
            if self.pf_active_ids.coil[i_coil].name == name {
                self.pf_active_ids.coil[i_coil].voltage = SignalFlt1d {
                    data: voltage_data,
                    time: voltage_time,
                };

                return Ok(());
            }
        }

        Err(PyValueError::new_err(format!(
            "PF coil `{name}`: there is no coil with this name in `pf_active/coil`; add it with `add_pf_coil` first"
        )))
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.pf_active_paths`, straight out of the
    /// pf_active IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.pf_active_ids, "pf_active", path)
    }

    /// A copy of the whole pf_active IDS, for reading with `gsfit_rs.imas.pf_active_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn pf_active_ids(&self) -> PyPfActive {
        PyPfActive::new(self.pf_active_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.PfActive>");
        string_output += &format!("║  {:<74} ║\n", version);

        let coils: &Vec<PfCoils> = &self.pf_active_ids.coil;
        let n_coil: usize = coils.len();
        if n_coil == 0 {
            string_output += &format!("║  {:<74} ║\n", " `pf_active/coil` is empty");
        }
        for i_coil in 0..n_coil {
            let n_element: usize = coils[i_coil].element.len();
            let prescribed: &str = if coils[i_coil].voltage.data.is_empty() { "current" } else { "voltage" };
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" coil({i_coil}) = {}; elements = {n_element}; prescribed = {prescribed}", coils[i_coil].name)
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `pf_active_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_pf_active`, are not pickled: they are read again from the XML which `pf_active_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pf_active_ids", self.pf_active_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_pf_active`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PfActive` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `PfActive` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let pf_active_ids_state: Bound<'_, PyAny> = state
            .get_item("pf_active_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PfActive` has no `pf_active_ids`"))?;
        let pf_active_ids: PfActiveIds =
            PickleState::from_pickle_state(&pf_active_ids_state).map_err(|error: PickleStateError| error.into_py_err("`PfActive.pf_active_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_pf_active: SettingsPfActive = SettingsPfActive::from_xml(&pf_active_ids.code.parameters)
            .map_err(|error: String| PyValueError::new_err(format!("The pickled `PfActive`'s settings cannot be read into `SettingsPfActive`: {error}")))?;

        Ok(Self {
            pf_active_ids,
            settings_pf_active,
        })
    }
}

/// Rust only methods
impl PfActive {
    /// Append a coil to `pf_active/coil`, with its current prescribed. This does the work of the Python method
    /// `add_pf_coil`, so that Rust code can add coils too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the coil, unique within the PF coils, e.g. `"BVL"`
    /// * `element_r` - major radius of each filament's centre, [metre]
    /// * `element_z` - height of each filament's centre, [metre]
    /// * `element_width` - radial full width of each filament, [metre]
    /// * `element_height` - vertical full height of each filament, [metre]
    /// * `current_time` - the experimental timebase, [second]
    /// * `current_data` - the current per turn, [ampere]
    ///
    /// # Errors
    /// Why the coil cannot be added: an empty or repeated name, arrays of different lengths, no filaments, a filament
    /// without a positive radius and size, or no current.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pf_coil_rs(
        &mut self,
        name: &str,
        element_r: Array1<f64>,
        element_z: Array1<f64>,
        element_width: Array1<f64>,
        element_height: Array1<f64>,
        current_time: Array1<f64>,
        current_data: Array1<f64>,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("PF coil: `name` is empty; a coil needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device", and
        // `set_pf_voltage_prescribed` finds the coil by it
        let n_coil: usize = self.pf_active_ids.coil.len();
        for i_coil in 0..n_coil {
            if self.pf_active_ids.coil[i_coil].name == name {
                return Err(format!("PF coil `{name}`: a coil with this name already exists, at `pf_active/coil({i_coil})`"));
            }
        }

        let n_element: usize = element_r.len();
        if element_z.len() != n_element || element_width.len() != n_element || element_height.len() != n_element {
            return Err(format!(
                "PF coil `{name}`: `r`, `z`, `width` and `height` have {}, {}, {} and {} point(s); each filament needs one of each",
                n_element,
                element_z.len(),
                element_width.len(),
                element_height.len()
            ));
        }

        if n_element == 0 {
            return Err(format!("PF coil `{name}`: there are no filaments"));
        }

        if current_time.len() != current_data.len() {
            return Err(format!(
                "PF coil `{name}`: `time` has {} point(s) but `current` has {}; a signal needs one value per time",
                current_time.len(),
                current_data.len()
            ));
        }

        if current_time.is_empty() {
            return Err(format!("PF coil `{name}`: `time` is empty; the current needs at least one time"));
        }

        // One element per filament: a rectangle of one turn
        let mut elements: Vec<PfCoilsElements> = Vec::with_capacity(n_element);
        for i_element in 0..n_element {
            // The Greens functions for the self-inductance need every filament at a positive radius, and of a positive size
            let is_usable_filament: bool =
                element_r[i_element] > 0.0 && element_z[i_element].is_finite() && element_width[i_element] > 0.0 && element_height[i_element] > 0.0;
            if !is_usable_filament {
                return Err(format!(
                    "PF coil `{name}`: filament {i_element} has r = {}, z = {}, width = {} and height = {} metre; r, width and height must be positive, and z finite",
                    element_r[i_element], element_z[i_element], element_width[i_element], element_height[i_element]
                ));
            }

            elements.push(PfCoilsElements {
                turns_with_sign: 1.0,
                geometry: Outline2dGeometryStatic {
                    geometry_type: 2, // rectangle
                    rectangle: RectangleStatic {
                        r: element_r[i_element],
                        z: element_z[i_element],
                        width: element_width[i_element],
                        height: element_height[i_element],
                    },
                    ..Default::default()
                },
                ..Default::default()
            });
        }

        // Self-inductance, [henry]: the flux linking the coil per ampere, summed over every pair of filaments, each one turn.
        // `solve_circuit_equations` uses the same method for the mutual inductances between the coils.
        // TODO: this should be re-visited. It is fast, but uses the flux at the centre of each filament, which overestimates a
        // self-inductance (by ~5% for a typical filament; see `Greens::psi`). `mutual_inductance_finite_size_to_finite_size` averages
        // the flux over each filament's cross-section, but splits every filament into 100 sub-filaments, so is far slower.
        let inductance: f64 = Greens::sensor_to_conductor(element_r.clone(), element_z.clone(), element_r, element_z, element_width, element_height)
            .psi()
            .sum();

        self.pf_active_ids.coil.push(PfCoils {
            name: name.to_string(),
            element: elements,
            inductance,
            current: SignalFlt1d {
                data: current_data,
                time: current_time,
            },
            ..Default::default()
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::settings_pf_active::SettingsPfActiveWorkflow;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/pf_active/pf_active.xsd");

    /// `pf_active.xml` without its optional link to `workflow.xml`, so it follows `pf_active.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pf_active_settings/>
"#;

    /// `pf_active.xml`, with `workflow.xml` written in its link's place, as `serialise_python_dataclass_to_imas_parameters_input` writes it
    const SETTINGS_WITH_WORKFLOW_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pf_active_settings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <workflow>
        <tree>
            <tree_name>ELMAG</tree_name>
            <pulseNo>-1</pulseNo>
            <run_name>RUN01</run_name>
            <usage>The geometry</usage>
        </tree>
        <tree>
            <tree_name>PSU2COIL</tree_name>
            <pulseNo xsi:nil="true" />
            <run_name>BEST</run_name>
            <usage>The currents</usage>
        </tree>
    </workflow>
</pf_active_settings>
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
        let pf_active: PfActive = PfActive::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(pf_active.pf_active_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_into_structs() {
        let settings_pf_active: SettingsPfActive = SettingsPfActive::from_xml(SETTINGS_WITH_WORKFLOW_XML).expect("the settings can be read");

        let workflow: &SettingsPfActiveWorkflow = settings_pf_active.workflow.as_ref().expect("the settings link to workflow.xml");
        assert_eq!(workflow.tree.len(), 2);
        assert_eq!(workflow.tree[0].tree_name, "ELMAG");
        assert_eq!(workflow.tree[0].pulseNo, Some(-1));
        assert_eq!(workflow.tree[1].tree_name, "PSU2COIL");
        assert_eq!(workflow.tree[1].pulseNo, None);
        assert_eq!(workflow.tree[1].run_name, "BEST");
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        let pf_active: PfActive = PfActive::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        assert_eq!(pf_active.settings_pf_active.workflow, None);
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(PfActive::new(&parameters_input_with_xml("")).is_err());
        // Another IDS's settings
        assert!(PfActive::new(&parameters_input_with_xml("<pf_passive_settings/>")).is_err());
        // An element which is not in the XML schema
        assert!(PfActive::new(&parameters_input_with_xml("<pf_active_settings><extra/></pf_active_settings>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(PfActive::new(&parameters_input).is_err());
    }

    #[test]
    fn a_new_coil_has_no_voltage_so_its_current_is_prescribed() {
        let mut pf_active: PfActive = PfActive::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema");

        pf_active
            .add_pf_coil_rs(
                "BVL",
                Array1::from_elem(1, 0.9),
                Array1::from_elem(1, 0.8),
                Array1::from_elem(1, 0.03),
                Array1::from_elem(1, 0.03),
                Array1::from_vec(vec![0.0, 1.0]),
                Array1::from_vec(vec![1.0e3, 1.0e3]),
            )
            .expect("the coil is valid");

        assert!(pf_active.pf_active_ids.coil[0].voltage.data.is_empty());
    }
}
