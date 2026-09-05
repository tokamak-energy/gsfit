use imas_rs::ids::wall::{IdentifierStatic, Rz1dStatic, Wall as WallIds, Wall2d, Wall2dLimiterUnit};
use imas_rs::python::PyWall;
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
/// The solver reads the units by position, so the order they are added in is part of the
/// contract:
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
#[pyclass(module = "gsfit_rs")]
pub struct Wall {
    pub wall_ids: WallIds,
}

impl Default for Wall {
    fn default() -> Self {
        return Self::new();
    }
}

/// Python accessible methods
#[pymethods]
impl Wall {
    /// Construct an empty wall, ready for `add_limiter_unit` to be called.
    ///
    /// `description_2d(0)` is created here, so that the units added later have somewhere to
    /// go. Its `type` is set to `multiple_units_no_vessel`, because we describe the limiter
    /// with a set of units and leave the mechanical vessel description empty.
    #[new]
    pub fn new() -> Self {
        let mut description_2d: Wall2d = Wall2d::default();
        description_2d.r#type = IdentifierStatic {
            name: Some("multiple_units_no_vessel".to_string()),
            index: Some(1),
            description: Some("Limiter is described with multiple units, no vessel description".to_string()),
        };

        let mut wall_ids: WallIds = WallIds::default();
        wall_ids.description_2d = vec![description_2d];

        return Self { wall_ids };
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

        let mut limiter_unit: Wall2dLimiterUnit = Wall2dLimiterUnit::default();
        limiter_unit.name = Some(name.to_string());
        limiter_unit.outline = Rz1dStatic {
            r: Some(unit_r),
            z: Some(unit_z),
        };

        let description_2d: &mut Wall2d = self
            .wall_ids
            .description_2d
            .first_mut()
            .ok_or_else(|| PyValueError::new_err("`wall/description_2d` is empty"))?;
        description_2d.limiter.unit.push(limiter_unit);

        return Ok(());
    }

    /// The wall IDS, for reading with `gsfit_rs.imas.wall_paths`.
    ///
    /// This is the only way to read the data back out: there are no bespoke accessors, so
    /// every quantity is reached by its data dictionary path.
    ///
    /// The IDS is copied into the returned object, so it is a snapshot: changes made on
    /// the Rust side afterwards are not seen by it.
    #[getter]
    fn wall_ids(&self) -> PyWall {
        return PyWall::new(self.wall_ids.clone());
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Wall>");
        string_output += &format!("║  {:<74} ║\n", version);

        match limiter_units(&self.wall_ids) {
            Ok(units) => {
                let n_units: usize = units.len();
                for i_unit in 0..n_units {
                    let unit_name: &str = match &units[i_unit].name {
                        Some(unit_name) => unit_name,
                        None => "<unnamed>",
                    };
                    let n_points: usize = match &units[i_unit].outline.r {
                        Some(unit_r) => unit_r.len(),
                        None => 0,
                    };
                    string_output += &format!("║  {:<74} ║\n", format!(" limiter unit({i_unit}) = {unit_name}; points = {n_points}"));
                }
            }
            Err(error) => {
                string_output += &format!("║  {:<74} ║\n", format!(" {error}"));
            }
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}

/// The limiter units of `wall/description_2d(0)`, or an error naming the missing level.
fn limiter_units(wall_ids: &WallIds) -> Result<&Vec<Wall2dLimiterUnit>, String> {
    let description_2d: &Wall2d = wall_ids.description_2d.first().ok_or("`wall/description_2d` is empty")?;

    if description_2d.limiter.unit.is_empty() {
        return Err("`wall/description_2d(0)/limiter/unit` is empty".to_string());
    }

    return Ok(&description_2d.limiter.unit);
}

/// The outline of one limiter unit, checked for the two coordinates being present and the
/// same length.
fn unit_outline(unit: &Wall2dLimiterUnit, i_unit: usize) -> Result<(&Array1<f64>, &Array1<f64>), String> {
    let unit_r: &Array1<f64> = unit
        .outline
        .r
        .as_ref()
        .ok_or_else(|| format!("`wall/description_2d(0)/limiter/unit({i_unit})/outline/r` is unset"))?;
    let unit_z: &Array1<f64> = unit
        .outline
        .z
        .as_ref()
        .ok_or_else(|| format!("`wall/description_2d(0)/limiter/unit({i_unit})/outline/z` is unset"))?;

    if unit_r.len() != unit_z.len() {
        return Err(format!(
            "`wall/description_2d(0)/limiter/unit({i_unit})/outline` has {} `r` point(s) but {} `z`",
            unit_r.len(),
            unit_z.len()
        ));
    }

    return Ok((unit_r, unit_z));
}

/// Candidate limit points: every point of every limiter unit, in unit order.
///
/// A plasma can be limited anywhere it touches a plasma facing component, so all the units
/// are gathered together into one set of candidates. The vessel contour is `unit(0)`, so its
/// points come first.
///
/// # Arguments
/// * `wall_ids` - the `wall` IDS
///
/// # Returns
/// * `limit_pts_r` - radial coordinates, [metre]
/// * `limit_pts_z` - vertical coordinates, [metre]
pub fn limiter_points(wall_ids: &WallIds) -> Result<(Array1<f64>, Array1<f64>), String> {
    let units: &Vec<Wall2dLimiterUnit> = limiter_units(wall_ids)?;
    let n_units: usize = units.len();

    // Count first, so that the output is allocated once
    let mut n_limit_pts: usize = 0;
    for i_unit in 0..n_units {
        let (unit_r, _unit_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&units[i_unit], i_unit)?;
        n_limit_pts += unit_r.len();
    }

    let mut limit_pts_r: Array1<f64> = Array1::from_elem(n_limit_pts, f64::NAN);
    let mut limit_pts_z: Array1<f64> = Array1::from_elem(n_limit_pts, f64::NAN);

    let mut i_limit_pt: usize = 0;
    for i_unit in 0..n_units {
        let (unit_r, unit_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&units[i_unit], i_unit)?;
        let n_points_this_unit: usize = unit_r.len();
        for i_point in 0..n_points_this_unit {
            limit_pts_r[i_limit_pt] = unit_r[i_point];
            limit_pts_z[i_limit_pt] = unit_z[i_point];
            i_limit_pt += 1;
        }
    }

    return Ok((limit_pts_r, limit_pts_z));
}

/// The vacuum vessel contour: `wall/description_2d(0)/limiter/unit(0)/outline`.
///
/// This is the closed polygon bounding the region the plasma is allowed to occupy, used to
/// reject flux surfaces and magnetic axes which fall outside the machine. It is `unit(0)` by
/// convention; see the `Wall` documentation.
///
/// # Arguments
/// * `wall_ids` - the `wall` IDS
///
/// # Returns
/// * `vessel_r` - radial coordinates, [metre]
/// * `vessel_z` - vertical coordinates, [metre]
pub fn vacuum_vessel_outline(wall_ids: &WallIds) -> Result<(Array1<f64>, Array1<f64>), String> {
    let units: &Vec<Wall2dLimiterUnit> = limiter_units(wall_ids)?;
    let (vessel_r, vessel_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&units[0], 0)?;

    return Ok((vessel_r.to_owned(), vessel_z.to_owned()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Build a wall directly, without going through Python
    fn wall_with_units(units: Vec<(&str, Array1<f64>, Array1<f64>)>) -> WallIds {
        let mut wall: Wall = Wall::new();
        for (unit_name, unit_r, unit_z) in units {
            let mut limiter_unit: Wall2dLimiterUnit = Wall2dLimiterUnit::default();
            limiter_unit.name = Some(unit_name.to_string());
            limiter_unit.outline = Rz1dStatic {
                r: Some(unit_r),
                z: Some(unit_z),
            };
            wall.wall_ids.description_2d[0].limiter.unit.push(limiter_unit);
        }

        return wall.wall_ids;
    }

    #[test]
    fn limiter_points_gather_every_unit_in_order() {
        let wall_ids: WallIds = wall_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8, 0.8, 0.2, 0.2], array![-0.5, -0.5, 0.5, 0.5, -0.5]),
            ("mct_tiles", array![0.7103], array![0.3031]),
            ("mcb_tiles", array![0.7103], array![-0.3131]),
        ]);

        let (limit_pts_r, limit_pts_z): (Array1<f64>, Array1<f64>) = limiter_points(&wall_ids).unwrap();

        assert_eq!(limit_pts_r, array![0.2, 0.8, 0.8, 0.2, 0.2, 0.7103, 0.7103]);
        assert_eq!(limit_pts_z, array![-0.5, -0.5, 0.5, 0.5, -0.5, 0.3031, -0.3131]);
    }

    #[test]
    fn vacuum_vessel_is_the_first_unit_only() {
        let wall_ids: WallIds = wall_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8, 0.8, 0.2, 0.2], array![-0.5, -0.5, 0.5, 0.5, -0.5]),
            ("mct_tiles", array![0.7103], array![0.3031]),
        ]);

        let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(&wall_ids).unwrap();

        assert_eq!(vessel_r, array![0.2, 0.8, 0.8, 0.2, 0.2]);
        assert_eq!(vessel_z, array![-0.5, -0.5, 0.5, 0.5, -0.5]);
    }

    #[test]
    fn an_empty_limiter_is_an_error_naming_the_path() {
        let wall_ids: WallIds = wall_with_units(vec![]);

        let error: String = limiter_points(&wall_ids).unwrap_err();

        assert!(error.contains("`wall/description_2d(0)/limiter/unit` is empty"), "{error}");
    }

    #[test]
    fn a_ragged_unit_outline_is_an_error_naming_the_unit() {
        let wall_ids: WallIds = wall_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8], array![-0.5, -0.5]),
            ("mct_tiles", array![0.7103, 0.7103], array![0.3031]),
        ]);

        let error: String = limiter_points(&wall_ids).unwrap_err();

        assert!(error.contains("unit(1)/outline"), "{error}");
    }
}
