use crate::greens::Greens;
use imas_rs::dd_support::parameters_input::ParametersInput;
use imas_rs::ids::pf_active::{
    Code, IdentifierStatic, Outline2dGeometryStatic, PfActive as PfActiveIds, PfCoils, PfCoilsElements, RectangleStatic, SignalFlt1d,
};
use imas_rs::python::{PyParametersInput, PyPath, PyPfActive, read_path};
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
/// * `prescribed` - a custom key, saying whether the coil's current or its voltage is prescribed
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
/// `coil(i1)/prescribed` is an identifier, saying which of the coil's signals is given:
/// * `current` (`index` 1), set by `add_pf_coil`: `coil(i1)/current` is given
/// * `voltage` (`index` 2), set by `set_pf_voltage_prescribed`: `coil(i1)/voltage` is given, and
///   the current is calculated
///
/// `Plasma.copy_from_pf_active_ids` copies each coil's current onto every time-slice of the
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
/// database the coils are read from, as an IMAS `parameters_input`. They are kept in
/// `parameters_input`, and their XML is also written to `pf_active/code/parameters`, where IMAS
/// records the parameters of the code which made an IDS.
#[pyclass(module = "gsfit_rs")]
pub struct PfActive {
    pub pf_active_ids: PfActiveIds,
    /// The settings the PF coils are built from: their XML, the XML schema they follow, and the defaults
    pub parameters_input: ParametersInput,
}

/// Python accessible methods
#[pymethods]
impl PfActive {
    /// Construct empty PF coils, ready for `add_pf_coil` to be called.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the PF coils are built from, as a
    ///   `gsfit_rs.dd_support.ParametersInput`; their XML is written to `pf_active/code/parameters`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> Self {
        let pf_active_ids: PfActiveIds = PfActiveIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            pf_active_ids,
            parameters_input: parameters_input.inner.clone(),
        }
    }

    /// The settings the PF coils were built from, as a `gsfit_rs.dd_support.ParametersInput`.
    ///
    /// This is a copy: changing it does not change the PF coils.
    #[getter]
    fn parameters_input(&self) -> PyParametersInput {
        PyParametersInput::new(self.parameters_input.clone())
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
    /// times: `Plasma.copy_from_pf_active_ids` interpolates it itself. It is signed; see the
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
    /// Writes `coil(i1)/voltage`, and sets `coil(i1)/prescribed` to `voltage`. The coil's
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
                self.pf_active_ids.coil[i_coil].prescribed = IdentifierStatic {
                    name: "voltage".to_string(),
                    index: 2,
                    description: "The coil's voltage is prescribed, and its current is calculated".to_string(),
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
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(
                    " coil({i_coil}) = {}; elements = {n_element}; prescribed = {}",
                    coils[i_coil].name, coils[i_coil].prescribed.name
                )
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
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
            prescribed: IdentifierStatic {
                name: "current".to_string(),
                index: 1,
                description: "The coil's current is prescribed".to_string(),
            },
            ..Default::default()
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_kept_and_their_xml_written_to_code_parameters() {
        let parameters_input: ParametersInput = ParametersInput {
            parameters_value: "<pf_active_settings/>".to_string(),
            parameters_default: String::new(),
            schema: "<xs:schema/>".to_string(),
        };

        let pf_active: PfActive = PfActive::new(&PyParametersInput::new(parameters_input));

        assert_eq!(pf_active.pf_active_ids.code.parameters, "<pf_active_settings/>");
        assert_eq!(pf_active.parameters_input.parameters_value, "<pf_active_settings/>");
        assert_eq!(pf_active.parameters_input.schema, "<xs:schema/>");
    }
}
