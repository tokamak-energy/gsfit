//! `magnetics/stationary_point(i1)`: the stationary point constraints, each a point where both derivatives of the poloidal flux are
//! zero, an X-point or the magnetic axis; a custom key, as the data dictionary has none. The point can move, see `moving_point`.

use super::moving_point::{GreensQuantity, GreensSources, PointGreens, check_time_trace, interpolate_onto_times};
use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::greens_storage::{GreensStationaryPoint, GreensStationaryPointDPsi};
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::PfPassive;
use crate::settings::settings_equilibrium::SettingsEquilibriumStationaryPoint;
use imas_rs::ids::magnetics::{MagneticsPositionDynamic, MagneticsStationaryPoint};
use imas_rs::{EMPTY_INT, EquilibriumConstraintsStationaryPoint};
use ndarray::{Array1, Array2, Array3, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append a stationary point constraint to `magnetics/stationary_point`, a custom key: a point where both derivatives of the poloidal
    /// flux, `d(psi)/d(r)` and `d(psi)/d(z)`, are zero, an X-point or the magnetic axis.
    ///
    /// The point can move, so it is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
    /// `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where the point is NaN, or which is not valid, is not
    /// fitted.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the stationary point constraints, e.g. `"MAGNETIC_AXIS"` or
    ///   `"X_POINT_LOWER"`, matching its fit settings, `equilibrium_settings/constraints/stationary_point/stationary_point(i1)/name`
    /// * `description` - description, e.g. whether the point is the magnetic axis or an X-point, and where it is read from
    /// * `time` - the constraint's timebase, increasing (1d array), [second]
    /// * `position_r` - the point's major radius, positive, or NaN where it is not known (1d array), [metre]
    /// * `position_z` - the point's height (1d array), [metre]
    /// * `validity_timed` - (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid (1d array of
    ///   `int32`), [dimensionless]
    /// * `validity` - (optional) the validity of the whole time trace: negative is not valid, [dimensionless]
    #[pyo3(signature = (name, description, time, position_r, position_z, validity_timed = None, validity = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_stationary_point(
        &mut self,
        name: &str,
        description: &str,
        time: PyReadonlyArray1<f64>,
        position_r: PyReadonlyArray1<f64>,
        position_z: PyReadonlyArray1<f64>,
        validity_timed: Option<PyReadonlyArray1<i32>>,
        validity: Option<i32>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let validity_timed: Array1<i32> = match validity_timed {
            Some(validity_timed) => validity_timed.to_owned_array(),
            None => Array1::zeros(0),
        };

        // Run the Rust method
        self.add_stationary_point_rs(
            name,
            description,
            time.to_owned_array(),
            position_r.to_owned_array(),
            position_z.to_owned_array(),
            validity_timed,
            validity.unwrap_or(EMPTY_INT),
        )
        .map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append a stationary point constraint to `magnetics/stationary_point`. This does the work of the Python method `add_stationary_point`,
    /// so that Rust code can add stationary point constraints too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the stationary point constraints
    /// * `description` - description
    /// * `time` - the constraint's timebase, increasing, [second]
    /// * `position_r`, `position_z` - the point at each time, [metre]
    /// * `validity_timed` - the validity at each time; empty when every time is valid, [dimensionless]
    /// * `validity` - the validity of the whole time trace; `EMPTY_INT` when it is valid, [dimensionless]
    ///
    /// # Errors
    /// Why the constraint cannot be added: an empty or repeated name, or a time trace which `check_time_trace` rejects
    #[allow(clippy::too_many_arguments)]
    pub fn add_stationary_point_rs(
        &mut self,
        name: &str,
        description: &str,
        time: Array1<f64>,
        position_r: Array1<f64>,
        position_z: Array1<f64>,
        validity_timed: Array1<i32>,
        validity: i32,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("stationary point: `name` is empty; a stationary point constraint needs a short identifier".to_string());
        }
        let n_stationary_point: usize = self.magnetics_ids.stationary_point.len();
        for i_stationary_point in 0..n_stationary_point {
            if self.magnetics_ids.stationary_point[i_stationary_point].name == name {
                return Err(format!(
                    "stationary point `{name}`: a stationary point constraint with this name already exists, at `magnetics/stationary_point({i_stationary_point})`"
                ));
            }
        }
        check_time_trace(
            &format!("stationary point `{name}`"),
            &time,
            &[("position", &position_r, &position_z)],
            &validity_timed,
        )?;

        self.magnetics_ids.stationary_point.push(MagneticsStationaryPoint {
            name: name.to_string(),
            description: description.to_string(),
            position: MagneticsPositionDynamic { r: position_r, z: position_z },
            validity_timed,
            validity,
            time,
        });

        Ok(())
    }
}

/// Interpolate the point of `magnetics/stationary_point(i_stationary_point)` onto the reconstruction times, as (`position_r`,
/// `position_z`), each NaN where the point is not known [metre]
///
/// # Errors
/// The point cannot be interpolated, which `check_time_trace` stops when it is added
fn stationary_point_point(
    stationary_points: &[MagneticsStationaryPoint],
    i_stationary_point: usize,
    times_to_reconstruct: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let stationary_point: &MagneticsStationaryPoint = &stationary_points[i_stationary_point];
    let interpolate = |coordinate_name: &str, coordinate: &Array1<f64>| -> Result<Array1<f64>, String> {
        interpolate_onto_times(
            &stationary_point.time,
            coordinate,
            &stationary_point.validity_timed,
            stationary_point.validity,
            times_to_reconstruct,
        )
        .map_err(|error: String| {
            format!(
                "cannot interpolate `magnetics/stationary_point({i_stationary_point})/{coordinate_name}` ({}) onto the reconstruction times: {error}",
                stationary_point.name
            )
        })
    };
    Ok((
        interpolate("position/r", &stationary_point.position.r)?,
        interpolate("position/z", &stationary_point.position.z)?,
    ))
}

/// The names in the stationary point fit settings, `equilibrium_settings/constraints/stationary_point`, checked against the Magnetics IDS
///
/// # Errors
/// A name in the fit settings is not in the Magnetics IDS, `magnetics/stationary_point`
fn settings_names(settings: &[SettingsEquilibriumStationaryPoint], stationary_points: &[MagneticsStationaryPoint]) -> Result<Vec<String>, String> {
    let n_settings: usize = settings.len();
    let mut names: Vec<String> = Vec::with_capacity(n_settings);
    for i_settings in 0..n_settings {
        let settings_name: &str = &settings[i_settings].name;
        if !stationary_points
            .iter()
            .any(|stationary_point: &MagneticsStationaryPoint| stationary_point.name == settings_name)
        {
            return Err(format!(
                "`{settings_name}` is in the fit settings, `equilibrium_settings/constraints/stationary_point`, but not in the magnetics IDS, `magnetics/stationary_point`"
            ));
        }
        names.push(settings_name.to_string());
    }
    Ok(names)
}

/// Rust only methods, copying the stationary point constraints into the Equilibrium IDS
impl Equilibrium {
    /// Copy the stationary point constraints from the Magnetics IDS onto every time-slice of the Equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/stationary_point(i1)`, a custom key. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// Each constraint gets its point, `position`, interpolated onto the time-slice's time, and the `measured` of both derivatives,
    /// `d_psi_d_r` and `d_psi_d_z`, which are zero at a stationary point; or NaN, which is not fitted, when the point is not known at the
    /// time-slice.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the stationary point constraints in `magnetics/stationary_point`
    ///
    /// # Errors
    /// A stationary point constraint in the fit settings, `equilibrium_settings/constraints/stationary_point`, is not in the Magnetics IDS,
    /// or cannot be interpolated onto the reconstruction times.
    pub fn copy_from_magnetics_ids_stationary_point(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let stationary_points: &Vec<MagneticsStationaryPoint> = &magnetics.magnetics_ids.stationary_point;
        let n_stationary_point: usize = stationary_points.len();

        // Only the stationary point constraints in the fit settings are copied, and each of them has to be in the Magnetics IDS. Checked
        // before anything is copied
        let settings: &Vec<SettingsEquilibriumStationaryPoint> = &self.settings_equilibrium.constraints.stationary_point;
        let names: Vec<String> = settings_names(settings, stationary_points)?;

        // Each point is on the constraint's own timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut points: Vec<Option<(Array1<f64>, Array1<f64>)>> = Vec::with_capacity(n_stationary_point);
        for i_stationary_point in 0..n_stationary_point {
            if names.contains(&stationary_points[i_stationary_point].name) {
                points.push(Some(stationary_point_point(stationary_points, i_stationary_point, &times_to_reconstruct)?));
            } else {
                points.push(None);
            }
        }

        // Rebuilt from the stationary point constraints in the fit settings, in the order of the Magnetics IDS, so that calling this twice
        // replaces them instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight`
        // and `expected_value`, found by its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraintsStationaryPoint> =
                std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.stationary_point);
            let mut stationary_point_constraints: Vec<EquilibriumConstraintsStationaryPoint> = Vec::with_capacity(names.len());
            for i_stationary_point in 0..n_stationary_point {
                let Some((position_r, position_z)) = &points[i_stationary_point] else {
                    continue;
                };
                let stationary_point_name: &str = &stationary_points[i_stationary_point].name;
                let mut constraint: EquilibriumConstraintsStationaryPoint = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraintsStationaryPoint| settings_constraint.name == stationary_point_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => {
                        let i_settings: usize = settings
                            .iter()
                            .position(|settings_stationary_point: &SettingsEquilibriumStationaryPoint| settings_stationary_point.name == stationary_point_name)
                            .expect("only the stationary point constraints in the fit settings are copied");
                        EquilibriumConstraintsStationaryPoint {
                            name: stationary_point_name.to_string(),
                            weight: settings[i_settings].weight,
                            expected_value: settings[i_settings].expected_value,
                            ..Default::default()
                        }
                    }
                };
                constraint.position.r = position_r[i_time];
                constraint.position.z = position_z[i_time];
                let is_known: bool = position_r[i_time].is_finite() && position_z[i_time].is_finite();
                constraint.d_psi_d_r.measured = if is_known { 0.0 } else { f64::NAN };
                constraint.d_psi_d_z.measured = if is_known { 0.0 } else { f64::NAN };
                constraint.source = format!("magnetics/stationary_point({i_stationary_point})");
                constraint.time_measurement = times_to_reconstruct[i_time];
                stationary_point_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.stationary_point = stationary_point_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the stationary point constraints' Greens tables
impl Equilibrium {
    /// Calculate the stationary point constraints' Greens tables, `greens_tables/stationary_point(i1)`, one for each stationary point
    /// constraint in the fit settings, `equilibrium_settings/constraints/stationary_point`, in the order of `magnetics/stationary_point`.
    /// This does part of the work of the Python method `greens_with_magnetics`; see it for what is stored, and for the checks on
    /// `pf_active`, `pf_passive` and the fit settings.
    ///
    /// Each constraint has the tables of both derivatives of the poloidal flux at its point, `d_psi_d_r` and `d_psi_d_z`, and with the
    /// plasma, their derivatives with height, `d2(psi)/d(r)d(z)` and `d2(psi)/d(z)2`, for the vertical shift. The point can move, so there
    /// is a table for each time-slice, with the point interpolated onto its time as `copy_from_magnetics_ids` interpolates it; NaN where the
    /// point is not known.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the stationary point constraints in `magnetics/stationary_point`
    /// * `pf_active` - the PF coils, in the order of `greens_tables/grid/pf_active`
    /// * `pf_passive` - the passive conductors, holding the filaments of those the equilibrium fits
    ///
    /// # Errors
    /// A point cannot be interpolated onto the reconstruction times, or a passive conductor the equilibrium fits is not in `pf_passive`
    pub fn greens_with_magnetics_stationary_point(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) -> Result<(), String> {
        let stationary_points: &Vec<MagneticsStationaryPoint> = &magnetics.magnetics_ids.stationary_point;
        let n_stationary_point: usize = stationary_points.len();
        let names: Vec<String> = settings_names(&self.settings_equilibrium.constraints.stationary_point, stationary_points)?;

        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let sources: GreensSources = GreensSources::new(self, pf_active, pf_passive)?;
        let (n_r, n_z, n_coil, n_dof_total): (usize, usize, usize, usize) = (sources.n_r, sources.n_z, sources.n_coil, sources.n_dof_total);

        let mut greens_stationary_points: Vec<GreensStationaryPoint> = Vec::with_capacity(names.len());
        for i_stationary_point in 0..n_stationary_point {
            if !names.contains(&stationary_points[i_stationary_point].name) {
                continue;
            }
            let (position_r, position_z): (Array1<f64>, Array1<f64>) = stationary_point_point(stationary_points, i_stationary_point, &times_to_reconstruct)?;

            // The tables of `d(psi)/d(r)`, then of `d(psi)/d(z)`
            let mut g_plasma: [Array3<f64>; 2] = [Array3::from_elem((n_time, n_z, n_r), f64::NAN), Array3::from_elem((n_time, n_z, n_r), f64::NAN)];
            let mut g_d_plasma_d_z: [Array3<f64>; 2] = [Array3::from_elem((n_time, n_z, n_r), f64::NAN), Array3::from_elem((n_time, n_z, n_r), f64::NAN)];
            let mut g_pf_active: [Array2<f64>; 2] = [Array2::from_elem((n_time, n_coil), f64::NAN), Array2::from_elem((n_time, n_coil), f64::NAN)];
            let mut g_pf_passive: [Array2<f64>; 2] = [
                Array2::from_elem((n_time, n_dof_total), f64::NAN),
                Array2::from_elem((n_time, n_dof_total), f64::NAN),
            ];
            for i_time in 0..n_time {
                if !(position_r[i_time].is_finite() && position_z[i_time].is_finite()) {
                    continue;
                }
                // Both derivatives of the poloidal flux, and their derivatives with height for the vertical shift
                let quantities: [GreensQuantity; 4] = [Greens::d_psi_d_r, Greens::d_psi_d_z, Greens::d2_psi_d_r_d_z, Greens::d2_psi_d_z2];
                let greens: Vec<PointGreens> = sources.point_greens(position_r[i_time], position_z[i_time], &quantities)?;

                for i_derivative in 0..2 {
                    g_plasma[i_derivative].slice_mut(s![i_time, .., ..]).assign(
                        &greens[i_derivative]
                            .plasma
                            .view()
                            .into_shape_with_order((n_z, n_r))
                            .expect("greens_with_magnetics_stationary_point: failed to reshape `plasma` into (n_z, n_r)"),
                    );
                    g_d_plasma_d_z[i_derivative].slice_mut(s![i_time, .., ..]).assign(
                        &greens[2 + i_derivative]
                            .plasma
                            .view()
                            .into_shape_with_order((n_z, n_r))
                            .expect("greens_with_magnetics_stationary_point: failed to reshape `d_plasma_d_z` into (n_z, n_r)"),
                    );
                    g_pf_active[i_derivative].row_mut(i_time).assign(&greens[i_derivative].pf_active);
                    g_pf_passive[i_derivative].row_mut(i_time).assign(&greens[i_derivative].pf_passive);
                }
            }

            let [g_plasma_d_psi_d_r, g_plasma_d_psi_d_z] = g_plasma;
            let [g_d_plasma_d_z_d_psi_d_r, g_d_plasma_d_z_d_psi_d_z] = g_d_plasma_d_z;
            let [g_pf_active_d_psi_d_r, g_pf_active_d_psi_d_z] = g_pf_active;
            let [g_pf_passive_d_psi_d_r, g_pf_passive_d_psi_d_z] = g_pf_passive;
            greens_stationary_points.push(GreensStationaryPoint {
                name: stationary_points[i_stationary_point].name.clone(),
                d_psi_d_r: GreensStationaryPointDPsi {
                    plasma: g_plasma_d_psi_d_r,
                    d_plasma_d_z: g_d_plasma_d_z_d_psi_d_r,
                    pf_active: g_pf_active_d_psi_d_r,
                    pf_passive: g_pf_passive_d_psi_d_r,
                },
                d_psi_d_z: GreensStationaryPointDPsi {
                    plasma: g_plasma_d_psi_d_z,
                    d_plasma_d_z: g_d_plasma_d_z_d_psi_d_z,
                    pf_active: g_pf_active_d_psi_d_z,
                    pf_passive: g_pf_passive_d_psi_d_z,
                },
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the stationary point constraints' tables instead of repeating them
        self.greens_tables.stationary_point = greens_stationary_points;

        Ok(())
    }
}
