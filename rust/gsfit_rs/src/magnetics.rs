use imas_rs::dd_support::parameters_input::ParametersInput;
use imas_rs::ids::magnetics::{Code, Magnetics as MagneticsIds};
use imas_rs::python::{PyMagnetics, PyParametersInput, PyPath, read_path};
use pyo3::prelude::*;

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
/// * `gaps(i2)/name`, `r_start`, `r_end`, `z_start` and `z_end` - a custom key, the stretches of the contour which have no winding
///
/// For the diamagnetic flux, `diamagnetic_flux(i1)`, and the plasma current, `ip(i1)`, one entry for each method of calculating
/// them:
/// * `method_name`
/// * `data` and `time` - on the experimental timebase
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
/// read from, as an IMAS `parameters_input`. They are kept in `parameters_input`, and their XML is also written to
/// `magnetics/code/parameters`, where IMAS records the parameters of the code which made an IDS.
#[pyclass(module = "gsfit_rs")]
pub struct Magnetics {
    pub magnetics_ids: MagneticsIds,
    /// The settings the magnetic sensors are built from: their XML, the XML schema they follow, and the defaults
    pub parameters_input: ParametersInput,
}

// The methods which add each kind of sensor are in `magnetic_sensors/`, one file per sensor, each with its own `#[pymethods]` block

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Construct empty magnetic sensors, ready for `add_b_field_pol_probe`, `add_flux_loop`, `add_rogowski_coil`,
    /// `add_diamagnetic_flux` and `add_ip` to be called.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the magnetic sensors are built from, as a
    ///   `gsfit_rs.dd_support.ParametersInput`; their XML is written to `magnetics/code/parameters`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> Self {
        let magnetics_ids: MagneticsIds = MagneticsIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            magnetics_ids,
            parameters_input: parameters_input.inner.clone(),
        }
    }

    /// The settings the magnetic sensors were built from, as a `gsfit_rs.dd_support.ParametersInput`.
    ///
    /// This is a copy: changing it does not change the magnetic sensors.
    #[getter]
    fn parameters_input(&self) -> PyParametersInput {
        PyParametersInput::new(self.parameters_input.clone())
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

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Magnetics>");
        string_output += &format!("║  {:<74} ║\n", version);
        string_output += &format!("║  {:<74} ║\n", format!(" b_field_pol_probe: {n_b_field_pol_probe} probe(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" flux_loop: {n_flux_loop} loop(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" rogowski_coil: {n_rogowski_coil} coil(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" diamagnetic_flux: {n_diamagnetic_flux} method(s)"));
        string_output += &format!("║  {:<74} ║\n", format!(" ip: {n_ip} method(s)"));
        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_kept_and_their_xml_written_to_code_parameters() {
        let parameters_input: ParametersInput = ParametersInput {
            parameters_value: "<magnetics_settings/>".to_string(),
            parameters_default: String::new(),
            schema: "<xs:schema/>".to_string(),
        };

        let magnetics: Magnetics = Magnetics::new(&PyParametersInput::new(parameters_input));

        assert_eq!(magnetics.magnetics_ids.code.parameters, "<magnetics_settings/>");
        assert_eq!(magnetics.parameters_input.parameters_value, "<magnetics_settings/>");
        assert_eq!(magnetics.parameters_input.schema, "<xs:schema/>");
    }
}
