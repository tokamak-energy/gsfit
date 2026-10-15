//! The limiter an equilibrium is solved inside, as the equilibrium IDS holds it: `equilibrium/limiter`.
//!
//! The limiter is read from a database into the `wall` IDS, and copied from there into the
//! equilibrium IDS by `Plasma.copy_from_wall_ids`. The solver and the post-processor read it from the
//! equilibrium IDS alone, through the two functions here, and never from the `wall` IDS.
//!
//! The units are read by position: `unit(0)` is the vacuum vessel contour, and the others are plasma
//! facing components, such as tiles; see the documentation of `equilibrium/limiter/unit`.

use imas_rs::{EquilibriumLimiter, EquilibriumLimiterUnit};
use ndarray::Array1;

/// Candidate limit points: every point of every limiter unit, in unit order.
///
/// A plasma can be limited anywhere it touches a plasma facing component, so all the units
/// are gathered together into one set of candidates. The vessel contour is `unit(0)`, so its
/// points come first.
///
/// # Arguments
/// * `limiter` - the limiter, `equilibrium/limiter`
///
/// # Returns
/// * `limit_pts_r` - radial coordinates, [metre]
/// * `limit_pts_z` - vertical coordinates, [metre]
pub fn limiter_points(limiter: &EquilibriumLimiter) -> Result<(Array1<f64>, Array1<f64>), String> {
    let units: &[EquilibriumLimiterUnit] = &limiter.unit;
    let n_unit: usize = units.len();
    if n_unit == 0 {
        return Err("`equilibrium/limiter/unit` is empty: copy the limiter in from the `wall` IDS with `Plasma.copy_from_wall_ids`".to_string());
    }

    // Count first, so that the output is allocated once
    let mut n_limit_pts: usize = 0;
    for i_unit in 0..n_unit {
        let (unit_r, _unit_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&units[i_unit], i_unit)?;
        n_limit_pts += unit_r.len();
    }

    let mut limit_pts_r: Array1<f64> = Array1::from_elem(n_limit_pts, f64::NAN);
    let mut limit_pts_z: Array1<f64> = Array1::from_elem(n_limit_pts, f64::NAN);

    let mut i_limit_pt: usize = 0;
    for i_unit in 0..n_unit {
        let (unit_r, unit_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&units[i_unit], i_unit)?;
        let n_point_this_unit: usize = unit_r.len();
        for i_point in 0..n_point_this_unit {
            limit_pts_r[i_limit_pt] = unit_r[i_point];
            limit_pts_z[i_limit_pt] = unit_z[i_point];
            i_limit_pt += 1;
        }
    }

    Ok((limit_pts_r, limit_pts_z))
}

/// The vacuum vessel contour: `equilibrium/limiter/unit(0)/outline`.
///
/// This is the closed polygon bounding the region the plasma is allowed to occupy, used to
/// reject flux surfaces and magnetic axes which fall outside the machine. It is `unit(0)` by
/// convention; see the documentation of `equilibrium/limiter/unit`.
///
/// # Arguments
/// * `limiter` - the limiter, `equilibrium/limiter`
///
/// # Returns
/// * `vessel_r` - radial coordinates, [metre]
/// * `vessel_z` - vertical coordinates, [metre]
pub fn vacuum_vessel_outline(limiter: &EquilibriumLimiter) -> Result<(Array1<f64>, Array1<f64>), String> {
    if limiter.unit.is_empty() {
        return Err("`equilibrium/limiter/unit` is empty: copy the limiter in from the `wall` IDS with `Plasma.copy_from_wall_ids`".to_string());
    }
    let (vessel_r, vessel_z): (&Array1<f64>, &Array1<f64>) = unit_outline(&limiter.unit[0], 0)?;

    Ok((vessel_r.to_owned(), vessel_z.to_owned()))
}

/// The outline of one limiter unit, checked for the two coordinates being present and the
/// same length.
fn unit_outline(unit: &EquilibriumLimiterUnit, i_unit: usize) -> Result<(&Array1<f64>, &Array1<f64>), String> {
    let unit_r: &Array1<f64> = &unit.outline.r;
    let unit_z: &Array1<f64> = &unit.outline.z;
    if unit_r.is_empty() {
        return Err(format!("`equilibrium/limiter/unit({i_unit})/outline/r` is unset"));
    }
    if unit_z.is_empty() {
        return Err(format!("`equilibrium/limiter/unit({i_unit})/outline/z` is unset"));
    }

    if unit_r.len() != unit_z.len() {
        return Err(format!(
            "`equilibrium/limiter/unit({i_unit})/outline` has {} `r` point(s) but {} `z`",
            unit_r.len(),
            unit_z.len()
        ));
    }

    Ok((unit_r, unit_z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::EquilibriumLimiterUnitOutline;
    use ndarray::array;

    /// Build a limiter directly, without going through the `wall` IDS
    fn limiter_with_units(units: Vec<(&str, Array1<f64>, Array1<f64>)>) -> EquilibriumLimiter {
        let mut limiter: EquilibriumLimiter = EquilibriumLimiter::default();
        for (unit_name, unit_r, unit_z) in units {
            limiter.unit.push(EquilibriumLimiterUnit {
                name: unit_name.to_string(),
                outline: EquilibriumLimiterUnitOutline { r: unit_r, z: unit_z },
            });
        }

        limiter
    }

    #[test]
    fn limiter_points_gather_every_unit_in_order() {
        let limiter: EquilibriumLimiter = limiter_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8, 0.8, 0.2, 0.2], array![-0.5, -0.5, 0.5, 0.5, -0.5]),
            ("mct_tiles", array![0.7103], array![0.3031]),
            ("mcb_tiles", array![0.7103], array![-0.3131]),
        ]);

        let (limit_pts_r, limit_pts_z): (Array1<f64>, Array1<f64>) = limiter_points(&limiter).unwrap();

        assert_eq!(limit_pts_r, array![0.2, 0.8, 0.8, 0.2, 0.2, 0.7103, 0.7103]);
        assert_eq!(limit_pts_z, array![-0.5, -0.5, 0.5, 0.5, -0.5, 0.3031, -0.3131]);
    }

    #[test]
    fn vacuum_vessel_is_the_first_unit_only() {
        let limiter: EquilibriumLimiter = limiter_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8, 0.8, 0.2, 0.2], array![-0.5, -0.5, 0.5, 0.5, -0.5]),
            ("mct_tiles", array![0.7103], array![0.3031]),
        ]);

        let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(&limiter).unwrap();

        assert_eq!(vessel_r, array![0.2, 0.8, 0.8, 0.2, 0.2]);
        assert_eq!(vessel_z, array![-0.5, -0.5, 0.5, 0.5, -0.5]);
    }

    #[test]
    fn an_empty_limiter_is_an_error_saying_how_to_fill_it() {
        let limiter: EquilibriumLimiter = limiter_with_units(vec![]);

        let error: String = limiter_points(&limiter).unwrap_err();

        assert!(error.contains("`equilibrium/limiter/unit` is empty"), "{error}");
        assert!(error.contains("`Plasma.copy_from_wall_ids`"), "{error}");
    }

    #[test]
    fn a_ragged_unit_outline_is_an_error_naming_the_unit() {
        let limiter: EquilibriumLimiter = limiter_with_units(vec![
            ("vacuum_vessel", array![0.2, 0.8], array![-0.5, -0.5]),
            ("mct_tiles", array![0.7103, 0.7103], array![0.3031]),
        ]);

        let error: String = limiter_points(&limiter).unwrap_err();

        assert!(error.contains("`equilibrium/limiter/unit(1)/outline`"), "{error}");
    }
}
