use crate::greens::FilamentGeometry;
use crate::settings::settings_pf_passive::SettingsPfPassive;
use imas_rs::ids::pf_passive::{Code, ObliqueStatic, Outline2dGeometryStatic, PfCoilsElements, PfPassive as PfPassiveIds, PfPassiveLoops};
use imas_rs::python::{PickleState, PickleStateError, PyParametersInput, PyPath, PyPfPassive, read_path};
use ndarray::{Array1, Array2};
use numpy::borrow::PyReadonlyArray1;
use numpy::{PyArray2, PyArrayMethods};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The machine's passive conductors, e.g. the vacuum vessel, the passive plates and the PF coil cases, stored as an IMAS
/// `pf_passive` IDS.
///
/// For each passive conductor, `loop(i1)`, these are filled:
/// * `name`
/// * `resistivity`
/// * `element(i2)/geometry/oblique/r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta` - one element per filament,
///   with `element(i2)/geometry/geometry_type` = 3, oblique (a parallelogram)
/// * `element(i2)/area` - the area of the parallelogram
/// * `element(i2)/turns_with_sign` = 1
///
/// # Geometry
///
/// The data dictionary describes a parallelogram from one of its corners, the reference point `oblique/r` and `z`. One side,
/// `length_alpha` long, leaves it at angle `alpha` counter-clockwise from the R axis; the other, `length_beta` long, leaves it at
/// angle `beta` counter-clockwise from the Z axis. The database readers convert into this.
///
/// GSFit's Greens functions describe a filament instead as EFIT does: by its centre, its radial width `d_r`, its vertical height
/// `d_z`, and the angles `angle_1` and `angle_2` of its sides from the R axis, where an angle of zero means the side is not sheared.
/// `passive_filaments` converts back into this.
///
/// # Machine data only
///
/// These are the conductors themselves, whichever code made the IDS. What the reconstruction fits their currents with is not: it is
/// the equilibrium's, from its fit settings, `equilibrium/pf_passive_dof.xml`, which say which passive conductors to fit, found here by
/// their `name`, and how. `Equilibrium.copy_from_pf_passive_ids` makes their degrees of freedom from these conductors' filaments.
///
/// Read the data back through `pf_passive_ids` and a path from `gsfit_rs.imas.pf_passive_paths`. The current `solve_circuit_equations`
/// simulates in each element, which the data dictionary has no node for, is kept outside the IDS, and read with `element_current`.
///
/// # Settings
///
/// The passive conductors are constructed with the settings they are built from, e.g. which run of which database they are read
/// from, as an IMAS `parameters_input`. They are validated against the XML schema in `parameters_input`, and only their XML is kept: it
/// is written to `pf_passive/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_pf_passive`, whose structs are generated from the XML schema `pf_passive.xsd`. They are kept here,
/// not in the IDS, as the pf_passive IDS can also be made by another code, whose `pf_passive/code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct PfPassive {
    pub pf_passive_ids: PfPassiveIds,
    pub settings_pf_passive: SettingsPfPassive,
    /// The current in each loop's elements, one for each `loop`, in the same order, on `loop(i1)/time`, shape = (n_element, n_time), [ampere].
    /// Only set by `solve_circuit_equations`, which simulates it; empty before then
    pub loop_element_currents: Vec<Array2<f64>>,
}

/// Python accessible methods
#[pymethods]
impl PfPassive {
    /// Construct a mostly empty PF passive IDS from its settings.
    /// Later it will be fully populated by reading from a database, with `add_loop`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the PF passive IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_pf_passive` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the passive conductors' settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the passive conductors' settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The passive conductors' settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsPfPassive` struct
        let settings_pf_passive: SettingsPfPassive = SettingsPfPassive::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The passive conductors' settings cannot be read into `SettingsPfPassive`: {error}")))?;

        // Construct the PF passive IDS from the parsed settings
        let pf_passive_ids: PfPassiveIds = PfPassiveIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            pf_passive_ids,
            settings_pf_passive,
            loop_element_currents: Vec::new(),
        })
    }

    /// The current in each of a passive conductor's elements, simulated by `solve_circuit_equations`, on `loop(i1)/time`,
    /// shape = [n_element, n_time], [ampere].
    ///
    /// # Arguments
    /// * `loop_name` - the passive conductor, as named in `add_loop`
    ///
    /// # Errors
    /// `ValueError` if there is no passive conductor with this name, or these passive conductors are not the ones
    /// `solve_circuit_equations` returned, which are the only ones to hold a simulated current
    fn element_current<'py>(&self, py: Python<'py>, loop_name: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let i_loop: usize = self.i_loop_named(loop_name).map_err(PyValueError::new_err)?;
        if self.loop_element_currents.is_empty() {
            return Err(PyValueError::new_err(format!(
                "passive conductor `{loop_name}`: there is no simulated current; read it from the copy of `PfPassive` which `solve_circuit_equations` returns"
            )));
        }

        Ok(PyArray2::from_array(py, &self.loop_element_currents[i_loop]))
    }

    /// Append a passive conductor to `pf_passive/loop`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the passive conductor, unique within the passive conductors, e.g. `"IVC"`
    /// * `r` - major radius of each filament's reference point, the corner which `alpha` and `beta` are measured from (1d array), [metre]
    /// * `z` - height of each filament's reference point (1d array), [metre]
    /// * `length_alpha` - length of each filament's side inclined at `alpha` (1d array), [metre]
    /// * `length_beta` - length of each filament's side inclined at `beta` (1d array), [metre]
    /// * `alpha` - inclination of each filament's first side, counter-clockwise from the R axis (1d array), [radian]
    /// * `beta` - inclination of each filament's second side, counter-clockwise from the Z axis (1d array), [radian]
    /// * `resistivity` - resistivity of the conductor, the same for every filament, [ohm * metre]
    ///
    /// Each filament becomes one `element(i2)`: a parallelogram of one turn, with its `area`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_loop(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        length_alpha: PyReadonlyArray1<f64>,
        length_beta: PyReadonlyArray1<f64>,
        alpha: PyReadonlyArray1<f64>,
        beta: PyReadonlyArray1<f64>,
        resistivity: f64,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let element_r: Array1<f64> = r.to_owned_array();
        let element_z: Array1<f64> = z.to_owned_array();
        let element_length_alpha: Array1<f64> = length_alpha.to_owned_array();
        let element_length_beta: Array1<f64> = length_beta.to_owned_array();
        let element_alpha: Array1<f64> = alpha.to_owned_array();
        let element_beta: Array1<f64> = beta.to_owned_array();

        // Run the Rust method
        self.add_loop_rs(
            name,
            element_r,
            element_z,
            element_length_alpha,
            element_length_beta,
            element_alpha,
            element_beta,
            resistivity,
        )
        .map_err(PyValueError::new_err)
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.pf_passive_paths`, straight out of the
    /// pf_passive IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.pf_passive_ids, "pf_passive", path)
    }

    /// A copy of the whole pf_passive IDS, for reading with `gsfit_rs.imas.pf_passive_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn pf_passive_ids(&self) -> PyPfPassive {
        PyPfPassive::new(self.pf_passive_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.PfPassive>");
        string_output += &format!("║  {:<74} ║\n", version);

        let loops: &Vec<PfPassiveLoops> = &self.pf_passive_ids.r#loop;
        let n_loop: usize = loops.len();
        if n_loop == 0 {
            string_output += &format!("║  {:<74} ║\n", " `pf_passive/loop` is empty");
        }
        for i_loop in 0..n_loop {
            let n_element: usize = loops[i_loop].element.len();
            string_output += &format!("║  {:<74} ║\n", format!(" loop({i_loop}) = {}; elements = {n_element}", loops[i_loop].name));
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `pf_passive_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`, and
    /// `loop_element_currents`, a `list` of NumPy arrays. `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_pf_passive`, are not pickled: they are read again from the XML which `pf_passive_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("pf_passive_ids", self.pf_passive_ids.to_pickle_state(py)?)?;
        state.set_item("loop_element_currents", self.loop_element_currents.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_pf_passive`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PfPassive` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `PfPassive` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let pf_passive_ids_state: Bound<'_, PyAny> = state
            .get_item("pf_passive_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PfPassive` has no `pf_passive_ids`"))?;
        let pf_passive_ids: PfPassiveIds =
            PickleState::from_pickle_state(&pf_passive_ids_state).map_err(|error: PickleStateError| error.into_py_err("`PfPassive.pf_passive_ids`"))?;
        let loop_element_currents_state: Bound<'_, PyAny> = state
            .get_item("loop_element_currents")?
            .ok_or_else(|| PyValueError::new_err("The pickled `PfPassive` has no `loop_element_currents`"))?;
        let loop_element_currents: Vec<Array2<f64>> = PickleState::from_pickle_state(&loop_element_currents_state)
            .map_err(|error: PickleStateError| error.into_py_err("`PfPassive.loop_element_currents`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_pf_passive: SettingsPfPassive = SettingsPfPassive::from_xml(&pf_passive_ids.code.parameters)
            .map_err(|error: String| PyValueError::new_err(format!("The pickled `PfPassive`'s settings cannot be read into `SettingsPfPassive`: {error}")))?;

        Ok(Self {
            pf_passive_ids,
            settings_pf_passive,
            loop_element_currents,
        })
    }
}

/// Rust only methods
impl PfPassive {
    /// The position in `pf_passive/loop` of the passive conductor named `loop_name`.
    ///
    /// # Errors
    /// There is no passive conductor with this name
    pub fn i_loop_named(&self, loop_name: &str) -> Result<usize, String> {
        let n_loop: usize = self.pf_passive_ids.r#loop.len();
        for i_loop in 0..n_loop {
            if self.pf_passive_ids.r#loop[i_loop].name == loop_name {
                return Ok(i_loop);
            }
        }

        Err(format!(
            "passive conductor `{loop_name}`: there is no passive conductor with this name in `pf_passive/loop`; add it with `add_loop` first"
        ))
    }

    /// Append a passive conductor to `pf_passive/loop`. This does the work of the Python method `add_loop`, so that Rust code can
    /// add passive conductors too; see it for the arguments.
    ///
    /// # Errors
    /// Why the passive conductor cannot be added: an empty or repeated name, arrays of different lengths, no filaments, or a filament
    /// without a positive radius and size.
    #[allow(clippy::too_many_arguments)]
    pub fn add_loop_rs(
        &mut self,
        name: &str,
        element_r: Array1<f64>,
        element_z: Array1<f64>,
        element_length_alpha: Array1<f64>,
        element_length_beta: Array1<f64>,
        element_alpha: Array1<f64>,
        element_beta: Array1<f64>,
        resistivity: f64,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("passive conductor: `name` is empty; a passive conductor needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device", and the sensors find their Greens with each
        // passive conductor by it
        let n_loop: usize = self.pf_passive_ids.r#loop.len();
        for i_loop in 0..n_loop {
            if self.pf_passive_ids.r#loop[i_loop].name == name {
                return Err(format!(
                    "passive conductor `{name}`: a passive conductor with this name already exists, at `pf_passive/loop({i_loop})`"
                ));
            }
        }

        let n_element: usize = element_r.len();
        if element_z.len() != n_element
            || element_length_alpha.len() != n_element
            || element_length_beta.len() != n_element
            || element_alpha.len() != n_element
            || element_beta.len() != n_element
        {
            return Err(format!(
                "passive conductor `{name}`: `r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta` have {}, {}, {}, {}, {} and {} point(s); each filament needs one of each",
                n_element,
                element_z.len(),
                element_length_alpha.len(),
                element_length_beta.len(),
                element_alpha.len(),
                element_beta.len()
            ));
        }

        if n_element == 0 {
            return Err(format!("passive conductor `{name}`: there are no filaments"));
        }

        // One element per filament: a parallelogram of one turn
        let mut elements: Vec<PfCoilsElements> = Vec::with_capacity(n_element);
        for i_element in 0..n_element {
            // The Greens functions need every filament at a positive radius, and of a positive size
            let is_usable_filament: bool = element_r[i_element] > 0.0
                && element_z[i_element].is_finite()
                && element_length_alpha[i_element] > 0.0
                && element_length_beta[i_element] > 0.0
                && element_alpha[i_element].is_finite()
                && element_beta[i_element].is_finite();
            if !is_usable_filament {
                return Err(format!(
                    "passive conductor `{name}`: filament {i_element} has r = {}, z = {} metre, length_alpha = {}, length_beta = {} metre, alpha = {} \
                     and beta = {} radian; r and the lengths must be positive, and z and the angles finite",
                    element_r[i_element],
                    element_z[i_element],
                    element_length_alpha[i_element],
                    element_length_beta[i_element],
                    element_alpha[i_element],
                    element_beta[i_element]
                ));
            }

            elements.push(PfCoilsElements {
                turns_with_sign: 1.0,
                geometry: Outline2dGeometryStatic {
                    geometry_type: 3, // oblique
                    oblique: ObliqueStatic {
                        r: element_r[i_element],
                        z: element_z[i_element],
                        length_alpha: element_length_alpha[i_element],
                        length_beta: element_length_beta[i_element],
                        alpha: element_alpha[i_element],
                        beta: element_beta[i_element],
                    },
                    ..Default::default()
                },
                ..Default::default()
            });
        }

        // The filaments as GSFit's Greens functions describe them
        let filaments: PassiveFilaments = passive_filaments(&elements);

        // The area of each parallelogram
        let mut area: Array1<f64> = Array1::from_elem(n_element, f64::NAN);
        for i_element in 0..n_element {
            let filament_geometry: FilamentGeometry = FilamentGeometry::new(
                filaments.angle_1[i_element],
                filaments.angle_2[i_element],
                filaments.d_r[i_element],
                filaments.d_z[i_element],
                filaments.r[i_element],
                filaments.z[i_element],
            );
            area[i_element] = filament_geometry.calculate_area();
            elements[i_element].area = area[i_element];
        }

        self.pf_passive_ids.r#loop.push(PfPassiveLoops {
            name: name.to_string(),
            resistivity,
            element: elements,
            ..Default::default()
        });

        Ok(())
    }
}

/// The filaments of one passive conductor, described as GSFit's Greens functions describe them, in the order of `element`.
///
/// See the `PfPassive` documentation for how this differs from the data dictionary's description.
pub struct PassiveFilaments {
    /// Major radius of each filament's centre, [metre]
    pub r: Array1<f64>,
    /// Height of each filament's centre, [metre]
    pub z: Array1<f64>,
    /// Radial width of each filament: the radial extent of its bottom and top sides, [metre]
    pub d_r: Array1<f64>,
    /// Vertical height of each filament: the vertical extent of its left and right sides, [metre]
    pub d_z: Array1<f64>,
    /// Angle between the R axis and each filament's left and right sides; zero when they are not sheared, so vertical, [radian]
    pub angle_1: Array1<f64>,
    /// Angle between the R axis and each filament's bottom and top sides; zero when they are not sheared, so horizontal, [radian]
    pub angle_2: Array1<f64>,
}

/// Describe a passive conductor's filaments as GSFit's Greens functions describe them, from their data dictionary `oblique`
/// description.
///
/// The reference point is the bottom left corner. The side inclined at `alpha` is the bottom side, and the side inclined at `beta`
/// is the left side, which is at angle `beta + pi / 2` from the R axis. An angle of zero means the side is not sheared.
///
/// # Arguments
/// * `elements` - the passive conductor's elements, `loop(i1)/element`
///
/// # Returns
/// * The filaments, in the order of `elements`
pub fn passive_filaments(elements: &[PfCoilsElements]) -> PassiveFilaments {
    let n_element: usize = elements.len();

    let mut filaments: PassiveFilaments = PassiveFilaments {
        r: Array1::from_elem(n_element, f64::NAN),
        z: Array1::from_elem(n_element, f64::NAN),
        d_r: Array1::from_elem(n_element, f64::NAN),
        d_z: Array1::from_elem(n_element, f64::NAN),
        angle_1: Array1::from_elem(n_element, f64::NAN),
        angle_2: Array1::from_elem(n_element, f64::NAN),
    };

    for i_element in 0..n_element {
        let oblique: &ObliqueStatic = &elements[i_element].geometry.oblique;

        // The bottom side, from the reference point, and the left side, from the reference point
        let side_alpha_r: f64 = oblique.length_alpha * oblique.alpha.cos();
        let side_alpha_z: f64 = oblique.length_alpha * oblique.alpha.sin();
        let side_beta_r: f64 = -oblique.length_beta * oblique.beta.sin();
        let side_beta_z: f64 = oblique.length_beta * oblique.beta.cos();

        // The centre is half of each side on from the reference point
        filaments.r[i_element] = oblique.r + 0.5 * (side_alpha_r + side_beta_r);
        filaments.z[i_element] = oblique.z + 0.5 * (side_alpha_z + side_beta_z);
        filaments.d_r[i_element] = side_alpha_r;
        filaments.d_z[i_element] = side_beta_z;
        filaments.angle_1[i_element] = if oblique.beta == 0.0 { 0.0 } else { oblique.beta + PI / 2.0 };
        filaments.angle_2[i_element] = if oblique.alpha == 0.0 { 0.0 } else { oblique.alpha };
    }

    filaments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::settings_pf_passive::SettingsPfPassiveWorkflow;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/pf_passive/pf_passive.xsd");

    /// `pf_passive.xml` without its optional link to `workflow.xml`, so it follows `pf_passive.xsd`, in which `workflow` is still a link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pf_passive_settings/>
"#;

    /// `pf_passive.xml`, with `workflow.xml` written in its link's place, as `serialise_python_dataclass_to_imas_parameters_input` writes it
    const SETTINGS_WITH_WORKFLOW_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pf_passive_settings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <workflow>
        <tree>
            <tree_name>ELMAG</tree_name>
            <pulseNo xsi:nil="true" />
            <run_name>RUN01</run_name>
            <usage>The geometry</usage>
        </tree>
    </workflow>
</pf_passive_settings>
"#;

    fn parameters_input_with_xml(parameters_value: &str) -> PyParametersInput {
        PyParametersInput::new(ParametersInput {
            parameters_value: parameters_value.to_string(),
            parameters_default: String::new(),
            schema: SCHEMA_XSD.to_string(),
        })
    }

    /// Empty passive conductors
    fn pf_passive_empty() -> PfPassive {
        PfPassive::new(&parameters_input_with_xml(SETTINGS_XML)).expect("the settings follow their XML schema")
    }

    /// Add a passive conductor, named `name`, of four square filaments, 1 centimetre across
    fn add_loop_of_four_filaments(pf_passive: &mut PfPassive, name: &str) -> Result<(), String> {
        let n_element: usize = 4;
        pf_passive.add_loop_rs(
            name,
            Array1::from_vec(vec![0.50, 0.52, 0.54, 0.56]),
            Array1::from_vec(vec![0.10, 0.12, 0.14, 0.16]),
            Array1::from_elem(n_element, 0.01),
            Array1::from_elem(n_element, 0.01),
            Array1::zeros(n_element),
            Array1::zeros(n_element),
            7.4e-7,
        )
    }

    #[test]
    fn the_settings_xml_is_written_to_code_parameters() {
        let pf_passive: PfPassive = pf_passive_empty();

        assert_eq!(pf_passive.pf_passive_ids.code.parameters, SETTINGS_XML);
    }

    #[test]
    fn the_settings_are_read_into_structs() {
        let settings_pf_passive: SettingsPfPassive = SettingsPfPassive::from_xml(SETTINGS_WITH_WORKFLOW_XML).expect("the settings can be read");

        let workflow: &SettingsPfPassiveWorkflow = settings_pf_passive.workflow.as_ref().expect("the settings link to workflow.xml");
        assert_eq!(workflow.tree.len(), 1);
        assert_eq!(workflow.tree[0].tree_name, "ELMAG");
        assert_eq!(workflow.tree[0].pulseNo, None);
        assert_eq!(workflow.tree[0].run_name, "RUN01");
    }

    #[test]
    fn the_settings_are_read_without_the_optional_link_to_the_workflow() {
        assert_eq!(pf_passive_empty().settings_pf_passive.workflow, None);
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(PfPassive::new(&parameters_input_with_xml("")).is_err());
        // Another IDS's settings
        assert!(PfPassive::new(&parameters_input_with_xml("<pf_active_settings/>")).is_err());
        // An element which is not in the XML schema
        assert!(PfPassive::new(&parameters_input_with_xml("<pf_passive_settings><extra/></pf_passive_settings>")).is_err());
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: String::new(),
        });

        assert!(PfPassive::new(&parameters_input).is_err());
    }

    #[test]
    fn a_loop_holds_one_parallelogram_of_one_turn_for_each_filament() {
        let mut pf_passive: PfPassive = pf_passive_empty();

        add_loop_of_four_filaments(&mut pf_passive, "VESSEL").expect("the loop is valid");

        let pf_passive_loop: &PfPassiveLoops = &pf_passive.pf_passive_ids.r#loop[0];
        assert_eq!(pf_passive_loop.name, "VESSEL");
        assert_eq!(pf_passive_loop.resistivity, 7.4e-7);
        let n_element: usize = pf_passive_loop.element.len();
        assert_eq!(n_element, 4);
        for i_element in 0..n_element {
            assert_eq!(pf_passive_loop.element[i_element].geometry.geometry_type, 3);
            assert_eq!(pf_passive_loop.element[i_element].turns_with_sign, 1.0);
            // A square, 1 centimetre across
            assert!((pf_passive_loop.element[i_element].area - 1.0e-4).abs() < 1.0e-16);
        }
    }

    #[test]
    fn a_loop_is_found_by_its_name() {
        let mut pf_passive: PfPassive = pf_passive_empty();
        add_loop_of_four_filaments(&mut pf_passive, "PLATE").expect("the loop is valid");
        add_loop_of_four_filaments(&mut pf_passive, "VESSEL").expect("the loop is valid");

        assert_eq!(pf_passive.i_loop_named("VESSEL"), Ok(1));
        assert!(pf_passive.i_loop_named("IVC").is_err());
    }

    #[test]
    fn a_loop_which_cannot_be_added_leaves_the_ids_unchanged() {
        let mut pf_passive: PfPassive = pf_passive_empty();
        add_loop_of_four_filaments(&mut pf_passive, "VESSEL").expect("the loop is valid");

        // A repeated name
        assert!(add_loop_of_four_filaments(&mut pf_passive, "VESSEL").is_err());
        // A filament without a positive size
        let n_element: usize = 2;
        let result: Result<(), String> = pf_passive.add_loop_rs(
            "PLATE",
            Array1::from_elem(n_element, 0.5),
            Array1::zeros(n_element),
            Array1::from_vec(vec![0.01, 0.0]),
            Array1::from_elem(n_element, 0.01),
            Array1::zeros(n_element),
            Array1::zeros(n_element),
            7.4e-7,
        );
        assert!(result.is_err());

        assert_eq!(pf_passive.pf_passive_ids.r#loop.len(), 1);
    }
}
