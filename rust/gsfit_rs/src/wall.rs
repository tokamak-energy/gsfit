use imas_rs::dd_support::parameters_input::ParametersInput;
use imas_rs::ids::wall::{Code, IdentifierStatic, Rz1dStatic, Wall as WallIds, Wall2d, Wall2dLimiterUnit};
use imas_rs::python::{PyParametersInput, PyPath, PyWall, read_path};
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
/// the limiter is read from, as an IMAS `parameters_input`. They are kept in `parameters_input`,
/// and their XML is also written to `wall/code/parameters`, where IMAS records the parameters of
/// the code which made an IDS.
#[pyclass(module = "gsfit_rs")]
pub struct Wall {
    pub wall_ids: WallIds,
    /// The settings the wall is built from: their XML, the XML schema they follow, and the defaults
    pub parameters_input: ParametersInput,
}

/// Python accessible methods
#[pymethods]
impl Wall {
    /// Construct an empty wall, ready for `add_limiter_unit` to be called.
    ///
    /// `description_2d(0)` is created here, so that the units added later have somewhere to
    /// go. Its `type` is set to `multiple_units_no_vessel`, because we describe the limiter
    /// with a set of units and leave the mechanical vessel description empty.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the wall is built from, as a
    ///   `gsfit_rs.dd_support.ParametersInput`; their XML is written to `wall/code/parameters`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> Self {
        let description_2d: Wall2d = Wall2d {
            r#type: IdentifierStatic {
                name: "multiple_units_no_vessel".to_string(),
                index: 1,
                description: "Limiter is described with multiple units, no vessel description".to_string(),
            },
            ..Default::default()
        };

        let wall_ids: WallIds = WallIds {
            description_2d: vec![description_2d],
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            wall_ids,
            parameters_input: parameters_input.inner.clone(),
        }
    }

    /// The settings the wall was built from, as a `gsfit_rs.dd_support.ParametersInput`.
    ///
    /// This is a copy: changing it does not change the wall.
    #[getter]
    fn parameters_input(&self) -> PyParametersInput {
        PyParametersInput::new(self.parameters_input.clone())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_kept_and_their_xml_written_to_code_parameters() {
        let parameters_input: ParametersInput = ParametersInput {
            parameters_value: "<wall_settings/>".to_string(),
            parameters_default: String::new(),
            schema: "<xs:schema/>".to_string(),
        };

        let wall: Wall = Wall::new(&PyParametersInput::new(parameters_input));

        assert_eq!(wall.wall_ids.code.parameters, "<wall_settings/>");
        assert_eq!(wall.parameters_input.parameters_value, "<wall_settings/>");
        assert_eq!(wall.parameters_input.schema, "<xs:schema/>");
    }
}
