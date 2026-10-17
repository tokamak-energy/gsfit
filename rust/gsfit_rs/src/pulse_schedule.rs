use imas_rs::dd_support::parameters_input::ParametersInput;
use imas_rs::ids::pulse_schedule::{Code, PulseSchedule as PulseScheduleIds, PulseScheduleGap};
use imas_rs::python::{PyParametersInput, PyPath, PyPulseSchedule, read_path};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
/// database the gaps are read from, as an IMAS `parameters_input`. They are kept in
/// `parameters_input`, and their XML is also written to `pulse_schedule/code/parameters`, where
/// IMAS records the parameters of the code which made an IDS.
#[pyclass(module = "gsfit_rs")]
pub struct PulseSchedule {
    pub pulse_schedule_ids: PulseScheduleIds,
    /// The settings the pulse schedule is built from: their XML, the XML schema they follow, and the defaults
    pub parameters_input: ParametersInput,
}

/// Python accessible methods
#[pymethods]
impl PulseSchedule {
    /// Construct an empty pulse schedule, with no gaps, ready for `add_gap` to be called.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the pulse schedule is built from, as a
    ///   `gsfit_rs.dd_support.ParametersInput`; their XML is written to `pulse_schedule/code/parameters`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> Self {
        let pulse_schedule_ids: PulseScheduleIds = PulseScheduleIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            pulse_schedule_ids,
            parameters_input: parameters_input.inner.clone(),
        }
    }

    /// The settings the pulse schedule was built from, as a `gsfit_rs.dd_support.ParametersInput`.
    ///
    /// This is a copy: changing it does not change the pulse schedule.
    #[getter]
    fn parameters_input(&self) -> PyParametersInput {
        PyParametersInput::new(self.parameters_input.clone())
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

    #[test]
    fn the_settings_are_kept_and_their_xml_written_to_code_parameters() {
        let parameters_input: ParametersInput = ParametersInput {
            parameters_value: "<pulse_schedule_settings/>".to_string(),
            parameters_default: String::new(),
            schema: "<xs:schema/>".to_string(),
        };

        let pulse_schedule: PulseSchedule = PulseSchedule::new(&PyParametersInput::new(parameters_input));

        assert_eq!(pulse_schedule.pulse_schedule_ids.code.parameters, "<pulse_schedule_settings/>");
        assert_eq!(pulse_schedule.parameters_input.parameters_value, "<pulse_schedule_settings/>");
        assert_eq!(pulse_schedule.parameters_input.schema, "<xs:schema/>");
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
