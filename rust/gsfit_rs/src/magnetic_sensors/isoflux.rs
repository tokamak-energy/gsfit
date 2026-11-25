//! `magnetics/isoflux(i1)`: the isoflux constraints, each two points which lie on the same flux surface; a custom key, as the data
//! dictionary has none. The points can move, see `moving_point`.

use super::moving_point::{GreensQuantity, GreensSources, PointGreens, check_time_trace, interpolate_onto_times};
use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::greens_storage::GreensIsoflux;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::PfPassive;
use crate::settings::settings_equilibrium::SettingsEquilibriumIsoflux;
use imas_rs::ids::magnetics::{MagneticsIsoflux, MagneticsPositionDynamic};
use imas_rs::{EMPTY_INT, EquilibriumConstraintsIsoflux};
use ndarray::{Array1, Array2, Array3, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append an isoflux constraint to `magnetics/isoflux`, a custom key: two points which lie on the same flux surface, so that the
    /// poloidal flux at the first minus the poloidal flux at the second is zero.
    ///
    /// The points can move, so each is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
    /// `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where either point is NaN, or which is not valid, is not
    /// fitted.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the isoflux constraints, e.g. `"ISOFLUX01"`, matching its fit
    ///   settings, `equilibrium_settings/constraints/isoflux/isoflux(i1)/name`
    /// * `description` - description, e.g. where the two points are read from
    /// * `time` - the constraint's timebase, increasing (1d array), [second]
    /// * `position_1_r` - the first point's major radius, positive, or NaN where it is not known (1d array), [metre]
    /// * `position_1_z` - the first point's height (1d array), [metre]
    /// * `position_2_r` - the second point's major radius, positive, or NaN where it is not known (1d array), [metre]
    /// * `position_2_z` - the second point's height (1d array), [metre]
    /// * `validity_timed` - (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid (1d array of
    ///   `int32`), [dimensionless]
    /// * `validity` - (optional) the validity of the whole time trace: negative is not valid, [dimensionless]
    #[pyo3(signature = (name, description, time, position_1_r, position_1_z, position_2_r, position_2_z, validity_timed = None, validity = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_isoflux(
        &mut self,
        name: &str,
        description: &str,
        time: PyReadonlyArray1<f64>,
        position_1_r: PyReadonlyArray1<f64>,
        position_1_z: PyReadonlyArray1<f64>,
        position_2_r: PyReadonlyArray1<f64>,
        position_2_z: PyReadonlyArray1<f64>,
        validity_timed: Option<PyReadonlyArray1<i32>>,
        validity: Option<i32>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let validity_timed: Array1<i32> = match validity_timed {
            Some(validity_timed) => validity_timed.to_owned_array(),
            None => Array1::zeros(0),
        };

        // Run the Rust method
        self.add_isoflux_rs(
            name,
            description,
            time.to_owned_array(),
            position_1_r.to_owned_array(),
            position_1_z.to_owned_array(),
            position_2_r.to_owned_array(),
            position_2_z.to_owned_array(),
            validity_timed,
            validity.unwrap_or(EMPTY_INT),
        )
        .map_err(PyValueError::new_err)
    }
}

/// Rust only methods
impl Magnetics {
    /// Append an isoflux constraint to `magnetics/isoflux`. This does the work of the Python method `add_isoflux`, so that Rust code can
    /// add isoflux constraints too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the isoflux constraints
    /// * `description` - description
    /// * `time` - the constraint's timebase, increasing, [second]
    /// * `position_1_r`, `position_1_z` - the first point at each time, [metre]
    /// * `position_2_r`, `position_2_z` - the second point at each time, [metre]
    /// * `validity_timed` - the validity at each time; empty when every time is valid, [dimensionless]
    /// * `validity` - the validity of the whole time trace; `EMPTY_INT` when it is valid, [dimensionless]
    ///
    /// # Errors
    /// Why the constraint cannot be added: an empty or repeated name, or a time trace which `check_time_trace` rejects
    #[allow(clippy::too_many_arguments)]
    pub fn add_isoflux_rs(
        &mut self,
        name: &str,
        description: &str,
        time: Array1<f64>,
        position_1_r: Array1<f64>,
        position_1_z: Array1<f64>,
        position_2_r: Array1<f64>,
        position_2_z: Array1<f64>,
        validity_timed: Array1<i32>,
        validity: i32,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("isoflux: `name` is empty; an isoflux constraint needs a short identifier".to_string());
        }
        let n_isoflux: usize = self.magnetics_ids.isoflux.len();
        for i_isoflux in 0..n_isoflux {
            if self.magnetics_ids.isoflux[i_isoflux].name == name {
                return Err(format!(
                    "isoflux `{name}`: an isoflux constraint with this name already exists, at `magnetics/isoflux({i_isoflux})`"
                ));
            }
        }
        check_time_trace(
            &format!("isoflux `{name}`"),
            &time,
            &[("position_1", &position_1_r, &position_1_z), ("position_2", &position_2_r, &position_2_z)],
            &validity_timed,
        )?;

        self.magnetics_ids.isoflux.push(MagneticsIsoflux {
            name: name.to_string(),
            description: description.to_string(),
            position_1: MagneticsPositionDynamic {
                r: position_1_r,
                z: position_1_z,
            },
            position_2: MagneticsPositionDynamic {
                r: position_2_r,
                z: position_2_z,
            },
            validity_timed,
            validity,
            time,
        });

        Ok(())
    }
}

/// The two points of an isoflux constraint at each reconstruction time, interpolated from `magnetics/isoflux(i1)`, as
/// (`position_1_r`, `position_1_z`, `position_2_r`, `position_2_z`), each NaN where its point is not known [metre]
type IsofluxPoints = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>);

/// Interpolate the two points of `magnetics/isoflux(i_isoflux)` onto the reconstruction times
///
/// # Errors
/// A point cannot be interpolated, which `check_time_trace` stops when it is added
fn isoflux_points(isofluxes: &[MagneticsIsoflux], i_isoflux: usize, times_to_reconstruct: &Array1<f64>) -> Result<IsofluxPoints, String> {
    let isoflux: &MagneticsIsoflux = &isofluxes[i_isoflux];
    let interpolate = |coordinate_name: &str, coordinate: &Array1<f64>| -> Result<Array1<f64>, String> {
        interpolate_onto_times(&isoflux.time, coordinate, &isoflux.validity_timed, isoflux.validity, times_to_reconstruct).map_err(|error: String| {
            format!(
                "cannot interpolate `magnetics/isoflux({i_isoflux})/{coordinate_name}` ({}) onto the reconstruction times: {error}",
                isoflux.name
            )
        })
    };
    Ok((
        interpolate("position_1/r", &isoflux.position_1.r)?,
        interpolate("position_1/z", &isoflux.position_1.z)?,
        interpolate("position_2/r", &isoflux.position_2.r)?,
        interpolate("position_2/z", &isoflux.position_2.z)?,
    ))
}

/// The names in the isoflux fit settings, `equilibrium_settings/constraints/isoflux`, checked against the Magnetics IDS
///
/// # Errors
/// A name in the fit settings is not in the Magnetics IDS, `magnetics/isoflux`
fn settings_names(settings: &[SettingsEquilibriumIsoflux], isofluxes: &[MagneticsIsoflux]) -> Result<Vec<String>, String> {
    let n_settings: usize = settings.len();
    let mut names: Vec<String> = Vec::with_capacity(n_settings);
    for i_settings in 0..n_settings {
        let settings_name: &str = &settings[i_settings].name;
        if !isofluxes.iter().any(|isoflux: &MagneticsIsoflux| isoflux.name == settings_name) {
            return Err(format!(
                "`{settings_name}` is in the fit settings, `equilibrium_settings/constraints/isoflux`, but not in the magnetics IDS, `magnetics/isoflux`"
            ));
        }
        names.push(settings_name.to_string());
    }
    Ok(names)
}

/// Rust only methods, copying the isoflux constraints into the Equilibrium IDS
impl Equilibrium {
    /// Copy the isoflux constraints from the Magnetics IDS onto every time-slice of the Equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/isoflux(i1)`, a custom key. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// Each constraint gets its two points, `position_1` and `position_2`, interpolated onto the time-slice's time, and `measured`, the
    /// poloidal flux at the first point minus that at the second, which is zero as they lie on the same flux surface; or NaN, which is
    /// not fitted, when either point is not known at the time-slice.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the isoflux constraints in `magnetics/isoflux`
    ///
    /// # Errors
    /// An isoflux constraint in the fit settings, `equilibrium_settings/constraints/isoflux`, is not in the Magnetics IDS, or cannot be
    /// interpolated onto the reconstruction times.
    pub fn copy_from_magnetics_ids_isoflux(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let isofluxes: &Vec<MagneticsIsoflux> = &magnetics.magnetics_ids.isoflux;
        let n_isoflux: usize = isofluxes.len();

        // Only the isoflux constraints in the fit settings are copied, and each of them has to be in the Magnetics IDS. Checked before
        // anything is copied
        let settings: &Vec<SettingsEquilibriumIsoflux> = &self.settings_equilibrium.constraints.isoflux;
        let names: Vec<String> = settings_names(settings, isofluxes)?;

        // Each point is on the constraint's own timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut points: Vec<Option<IsofluxPoints>> = Vec::with_capacity(n_isoflux);
        for i_isoflux in 0..n_isoflux {
            if names.contains(&isofluxes[i_isoflux].name) {
                points.push(Some(isoflux_points(isofluxes, i_isoflux, &times_to_reconstruct)?));
            } else {
                points.push(None);
            }
        }

        // Rebuilt from the isoflux constraints in the fit settings, in the order of the Magnetics IDS, so that calling this twice replaces
        // them instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight` and
        // `expected_value`, found by its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraintsIsoflux> = std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.isoflux);
            let mut isoflux_constraints: Vec<EquilibriumConstraintsIsoflux> = Vec::with_capacity(names.len());
            for i_isoflux in 0..n_isoflux {
                let Some((position_1_r, position_1_z, position_2_r, position_2_z)) = &points[i_isoflux] else {
                    continue;
                };
                let isoflux_name: &str = &isofluxes[i_isoflux].name;
                let mut constraint: EquilibriumConstraintsIsoflux = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraintsIsoflux| settings_constraint.name == isoflux_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => {
                        let i_settings: usize = settings
                            .iter()
                            .position(|settings_isoflux: &SettingsEquilibriumIsoflux| settings_isoflux.name == isoflux_name)
                            .expect("only the isoflux constraints in the fit settings are copied");
                        EquilibriumConstraintsIsoflux {
                            name: isoflux_name.to_string(),
                            weight: settings[i_settings].weight,
                            expected_value: settings[i_settings].expected_value,
                            ..Default::default()
                        }
                    }
                };
                constraint.position_1.r = position_1_r[i_time];
                constraint.position_1.z = position_1_z[i_time];
                constraint.position_2.r = position_2_r[i_time];
                constraint.position_2.z = position_2_z[i_time];
                let is_known: bool = position_1_r[i_time].is_finite()
                    && position_1_z[i_time].is_finite()
                    && position_2_r[i_time].is_finite()
                    && position_2_z[i_time].is_finite();
                constraint.measured = if is_known { 0.0 } else { f64::NAN };
                constraint.source = format!("magnetics/isoflux({i_isoflux})");
                constraint.time_measurement = times_to_reconstruct[i_time];
                isoflux_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.isoflux = isoflux_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the isoflux constraints' Greens tables
impl Equilibrium {
    /// Calculate the isoflux constraints' Greens tables, `greens_tables/isoflux(i1)`, one for each isoflux constraint in the fit settings,
    /// `equilibrium_settings/constraints/isoflux`, in the order of `magnetics/isoflux`. This does part of the work of the Python method
    /// `greens_with_magnetics`; see it for what is stored, and for the checks on `pf_active`, `pf_passive` and the fit settings.
    ///
    /// Each table is the poloidal flux at the first point minus that at the second, GSFit's `psi`. The points can move, so there is a
    /// table for each time-slice, with the points interpolated onto its time as `copy_from_magnetics_ids` interpolates them; NaN where
    /// either point is not known.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the isoflux constraints in `magnetics/isoflux`
    /// * `pf_active` - the PF coils, in the order of `greens_tables/grid/pf_active`
    /// * `pf_passive` - the passive conductors, holding the filaments of those the equilibrium fits
    ///
    /// # Errors
    /// A point cannot be interpolated onto the reconstruction times, or a passive conductor the equilibrium fits is not in `pf_passive`
    pub fn greens_with_magnetics_isoflux(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) -> Result<(), String> {
        let isofluxes: &Vec<MagneticsIsoflux> = &magnetics.magnetics_ids.isoflux;
        let n_isoflux: usize = isofluxes.len();
        let names: Vec<String> = settings_names(&self.settings_equilibrium.constraints.isoflux, isofluxes)?;

        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let sources: GreensSources = GreensSources::new(self, pf_active, pf_passive)?;
        let (n_r, n_z, n_coil, n_dof_total): (usize, usize, usize, usize) = (sources.n_r, sources.n_z, sources.n_coil, sources.n_dof_total);

        let mut greens_isofluxes: Vec<GreensIsoflux> = Vec::with_capacity(names.len());
        for i_isoflux in 0..n_isoflux {
            if !names.contains(&isofluxes[i_isoflux].name) {
                continue;
            }
            let (position_1_r, position_1_z, position_2_r, position_2_z): IsofluxPoints = isoflux_points(isofluxes, i_isoflux, &times_to_reconstruct)?;

            let mut g_plasma: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
            let mut g_d_plasma_d_z: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
            let mut g_pf_active: Array2<f64> = Array2::from_elem((n_time, n_coil), f64::NAN);
            let mut g_pf_passive: Array2<f64> = Array2::from_elem((n_time, n_dof_total), f64::NAN);
            for i_time in 0..n_time {
                let is_known: bool = position_1_r[i_time].is_finite()
                    && position_1_z[i_time].is_finite()
                    && position_2_r[i_time].is_finite()
                    && position_2_z[i_time].is_finite();
                if !is_known {
                    continue;
                }
                // The poloidal flux, and its derivative with height for the vertical shift, at each point
                let quantities: [GreensQuantity; 2] = [Greens::psi, Greens::d_psi_d_z];
                let greens_1: Vec<PointGreens> = sources.point_greens(position_1_r[i_time], position_1_z[i_time], &quantities)?;
                let greens_2: Vec<PointGreens> = sources.point_greens(position_2_r[i_time], position_2_z[i_time], &quantities)?;

                g_plasma.slice_mut(s![i_time, .., ..]).assign(
                    &(&greens_1[0].plasma - &greens_2[0].plasma)
                        .into_shape_with_order((n_z, n_r))
                        .expect("greens_with_magnetics_isoflux: failed to reshape `plasma` into (n_z, n_r)"),
                );
                g_d_plasma_d_z.slice_mut(s![i_time, .., ..]).assign(
                    &(&greens_1[1].plasma - &greens_2[1].plasma)
                        .into_shape_with_order((n_z, n_r))
                        .expect("greens_with_magnetics_isoflux: failed to reshape `d_plasma_d_z` into (n_z, n_r)"),
                );
                g_pf_active.row_mut(i_time).assign(&(&greens_1[0].pf_active - &greens_2[0].pf_active));
                g_pf_passive.row_mut(i_time).assign(&(&greens_1[0].pf_passive - &greens_2[0].pf_passive));
            }

            greens_isofluxes.push(GreensIsoflux {
                name: isofluxes[i_isoflux].name.clone(),
                plasma: g_plasma,
                d_plasma_d_z: g_d_plasma_d_z,
                pf_active: g_pf_active,
                pf_passive: g_pf_passive,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the isoflux constraints' tables instead of repeating them
        self.greens_tables.isoflux = greens_isofluxes;

        Ok(())
    }
}
