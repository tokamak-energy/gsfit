//! `magnetics/isoflux_boundary(i1)`: the isoflux boundary constraints, each a point which lies on the plasma boundary; a custom key, as
//! the data dictionary has none. The point can move, see `moving_point`.

use super::moving_point::{GreensQuantity, GreensSources, PointGreens, check_time_trace, interpolate_onto_times};
use crate::equilibrium::Equilibrium;
use crate::greens::Greens;
use crate::greens_storage::GreensIsofluxBoundary;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::PfPassive;
use crate::settings::settings_equilibrium::SettingsEquilibriumIsofluxBoundary;
use imas_rs::ids::magnetics::{MagneticsIsofluxBoundary, MagneticsPositionDynamic};
use imas_rs::{EMPTY_INT, EquilibriumConstraintsIsofluxBoundary};
use ndarray::{Array1, Array2, Array3, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python accessible methods
#[pymethods]
impl Magnetics {
    /// Append an isoflux boundary constraint to `magnetics/isoflux_boundary`, a custom key: a point which lies on the plasma boundary,
    /// so that the poloidal flux there minus the poloidal flux at the boundary is zero.
    ///
    /// The point can move, so it is a time trace, on the constraint's own timebase, which need not be the reconstruction times:
    /// `Equilibrium.copy_from_magnetics_ids` interpolates it onto them. A time where the point is NaN, or which is not valid, is not
    /// fitted.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the isoflux boundary constraints, e.g. `"BOUNDARY01"`, matching its
    ///   fit settings, `equilibrium_settings/constraints/isoflux_boundary/isoflux_boundary(i1)/name`
    /// * `description` - description, e.g. where the point is read from
    /// * `time` - the constraint's timebase, increasing (1d array), [second]
    /// * `position_r` - the point's major radius, positive, or NaN where it is not known (1d array), [metre]
    /// * `position_z` - the point's height (1d array), [metre]
    /// * `validity_timed` - (optional) the validity at each time, as a magnetic sensor's `validity_timed`: negative is not valid (1d array of
    ///   `int32`), [dimensionless]
    /// * `validity` - (optional) the validity of the whole time trace: negative is not valid, [dimensionless]
    #[pyo3(signature = (name, description, time, position_r, position_z, validity_timed = None, validity = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_isoflux_boundary(
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
        self.add_isoflux_boundary_rs(
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
    /// Append an isoflux boundary constraint to `magnetics/isoflux_boundary`. This does the work of the Python method
    /// `add_isoflux_boundary`, so that Rust code can add isoflux boundary constraints too; see it for what is stored.
    ///
    /// # Arguments
    /// * `name` - short identifier for the constraint, unique within the isoflux boundary constraints
    /// * `description` - description
    /// * `time` - the constraint's timebase, increasing, [second]
    /// * `position_r`, `position_z` - the point at each time, [metre]
    /// * `validity_timed` - the validity at each time; empty when every time is valid, [dimensionless]
    /// * `validity` - the validity of the whole time trace; `EMPTY_INT` when it is valid, [dimensionless]
    ///
    /// # Errors
    /// Why the constraint cannot be added: an empty or repeated name, or a time trace which `check_time_trace` rejects
    #[allow(clippy::too_many_arguments)]
    pub fn add_isoflux_boundary_rs(
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
            return Err("isoflux boundary: `name` is empty; an isoflux boundary constraint needs a short identifier".to_string());
        }
        let n_isoflux_boundary: usize = self.magnetics_ids.isoflux_boundary.len();
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            if self.magnetics_ids.isoflux_boundary[i_isoflux_boundary].name == name {
                return Err(format!(
                    "isoflux boundary `{name}`: an isoflux boundary constraint with this name already exists, at `magnetics/isoflux_boundary({i_isoflux_boundary})`"
                ));
            }
        }
        check_time_trace(
            &format!("isoflux boundary `{name}`"),
            &time,
            &[("position", &position_r, &position_z)],
            &validity_timed,
        )?;

        self.magnetics_ids.isoflux_boundary.push(MagneticsIsofluxBoundary {
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

/// Interpolate the point of `magnetics/isoflux_boundary(i_isoflux_boundary)` onto the reconstruction times, as (`position_r`,
/// `position_z`), each NaN where the point is not known [metre]
///
/// # Errors
/// The point cannot be interpolated, which `check_time_trace` stops when it is added
fn isoflux_boundary_point(
    isoflux_boundaries: &[MagneticsIsofluxBoundary],
    i_isoflux_boundary: usize,
    times_to_reconstruct: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let isoflux_boundary: &MagneticsIsofluxBoundary = &isoflux_boundaries[i_isoflux_boundary];
    let interpolate = |coordinate_name: &str, coordinate: &Array1<f64>| -> Result<Array1<f64>, String> {
        interpolate_onto_times(
            &isoflux_boundary.time,
            coordinate,
            &isoflux_boundary.validity_timed,
            isoflux_boundary.validity,
            times_to_reconstruct,
        )
        .map_err(|error: String| {
            format!(
                "cannot interpolate `magnetics/isoflux_boundary({i_isoflux_boundary})/{coordinate_name}` ({}) onto the reconstruction times: {error}",
                isoflux_boundary.name
            )
        })
    };
    Ok((
        interpolate("position/r", &isoflux_boundary.position.r)?,
        interpolate("position/z", &isoflux_boundary.position.z)?,
    ))
}

/// The names in the isoflux boundary fit settings, `equilibrium_settings/constraints/isoflux_boundary`, checked against the Magnetics IDS
///
/// # Errors
/// A name in the fit settings is not in the Magnetics IDS, `magnetics/isoflux_boundary`
fn settings_names(settings: &[SettingsEquilibriumIsofluxBoundary], isoflux_boundaries: &[MagneticsIsofluxBoundary]) -> Result<Vec<String>, String> {
    let n_settings: usize = settings.len();
    let mut names: Vec<String> = Vec::with_capacity(n_settings);
    for i_settings in 0..n_settings {
        let settings_name: &str = &settings[i_settings].name;
        if !isoflux_boundaries
            .iter()
            .any(|isoflux_boundary: &MagneticsIsofluxBoundary| isoflux_boundary.name == settings_name)
        {
            return Err(format!(
                "`{settings_name}` is in the fit settings, `equilibrium_settings/constraints/isoflux_boundary`, but not in the magnetics IDS, `magnetics/isoflux_boundary`"
            ));
        }
        names.push(settings_name.to_string());
    }
    Ok(names)
}

/// Rust only methods, copying the isoflux boundary constraints into the Equilibrium IDS
impl Equilibrium {
    /// Copy the isoflux boundary constraints from the Magnetics IDS onto every time-slice of the Equilibrium IDS, as
    /// `equilibrium/time_slice(itime)/constraints/isoflux_boundary(i1)`, a custom key. This does part of the work of the Python method
    /// `copy_from_magnetics_ids`; see it for what is stored.
    ///
    /// Each constraint gets its point, `position`, interpolated onto the time-slice's time, and `measured`, the poloidal flux there minus
    /// that at the boundary, which is zero as the point lies on the boundary; or NaN, which is not fitted, when the point is not known at
    /// the time-slice.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the isoflux boundary constraints in `magnetics/isoflux_boundary`
    ///
    /// # Errors
    /// An isoflux boundary constraint in the fit settings, `equilibrium_settings/constraints/isoflux_boundary`, is not in the magnetics
    /// IDS, or cannot be interpolated onto the reconstruction times.
    pub fn copy_from_magnetics_ids_isoflux_boundary(&mut self, magnetics: &Magnetics) -> Result<(), String> {
        let isoflux_boundaries: &Vec<MagneticsIsofluxBoundary> = &magnetics.magnetics_ids.isoflux_boundary;
        let n_isoflux_boundary: usize = isoflux_boundaries.len();

        // Only the isoflux boundary constraints in the fit settings are copied, and each of them has to be in the Magnetics IDS. Checked
        // before anything is copied
        let settings: &Vec<SettingsEquilibriumIsofluxBoundary> = &self.settings_equilibrium.constraints.isoflux_boundary;
        let names: Vec<String> = settings_names(settings, isoflux_boundaries)?;

        // Each point is on the constraint's own timebase, so it is interpolated onto the times of the time-slices `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut points: Vec<Option<(Array1<f64>, Array1<f64>)>> = Vec::with_capacity(n_isoflux_boundary);
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            if names.contains(&isoflux_boundaries[i_isoflux_boundary].name) {
                points.push(Some(isoflux_boundary_point(isoflux_boundaries, i_isoflux_boundary, &times_to_reconstruct)?));
            } else {
                points.push(None);
            }
        }

        // Rebuilt from the isoflux boundary constraints in the fit settings, in the order of the Magnetics IDS, so that calling this twice
        // replaces them instead of repeating them. Each keeps the constraint `Equilibrium::new` made from its fit settings, with its `weight`
        // and `expected_value`, found by its name
        for i_time in 0..n_time {
            let mut settings_constraints: Vec<EquilibriumConstraintsIsofluxBoundary> =
                std::mem::take(&mut self.equilibrium_ids.time_slice[i_time].constraints.isoflux_boundary);
            let mut isoflux_boundary_constraints: Vec<EquilibriumConstraintsIsofluxBoundary> = Vec::with_capacity(names.len());
            for i_isoflux_boundary in 0..n_isoflux_boundary {
                let Some((position_r, position_z)) = &points[i_isoflux_boundary] else {
                    continue;
                };
                let isoflux_boundary_name: &str = &isoflux_boundaries[i_isoflux_boundary].name;
                let mut constraint: EquilibriumConstraintsIsofluxBoundary = match settings_constraints
                    .iter()
                    .position(|settings_constraint: &EquilibriumConstraintsIsofluxBoundary| settings_constraint.name == isoflux_boundary_name)
                {
                    Some(i_settings_constraint) => settings_constraints.swap_remove(i_settings_constraint),
                    None => {
                        let i_settings: usize = settings
                            .iter()
                            .position(|settings_isoflux_boundary: &SettingsEquilibriumIsofluxBoundary| settings_isoflux_boundary.name == isoflux_boundary_name)
                            .expect("only the isoflux boundary constraints in the fit settings are copied");
                        EquilibriumConstraintsIsofluxBoundary {
                            name: isoflux_boundary_name.to_string(),
                            weight: settings[i_settings].weight,
                            expected_value: settings[i_settings].expected_value,
                            ..Default::default()
                        }
                    }
                };
                constraint.position.r = position_r[i_time];
                constraint.position.z = position_z[i_time];
                let is_known: bool = position_r[i_time].is_finite() && position_z[i_time].is_finite();
                constraint.measured = if is_known { 0.0 } else { f64::NAN };
                constraint.source = format!("magnetics/isoflux_boundary({i_isoflux_boundary})");
                constraint.time_measurement = times_to_reconstruct[i_time];
                isoflux_boundary_constraints.push(constraint);
            }
            self.equilibrium_ids.time_slice[i_time].constraints.isoflux_boundary = isoflux_boundary_constraints;
        }

        Ok(())
    }
}

/// Rust only methods, calculating the isoflux boundary constraints' Greens tables
impl Equilibrium {
    /// Calculate the isoflux boundary constraints' Greens tables, `greens_tables/isoflux_boundary(i1)`, one for each isoflux boundary
    /// constraint in the fit settings, `equilibrium_settings/constraints/isoflux_boundary`, in the order of `magnetics/isoflux_boundary`.
    /// This does part of the work of the Python method `greens_with_magnetics`; see it for what is stored, and for the checks on
    /// `pf_active`, `pf_passive` and the fit settings.
    ///
    /// Each table is the poloidal flux at the point, GSFit's `psi`. The point can move, so there is a table for each time-slice, with the
    /// point interpolated onto its time as `copy_from_magnetics_ids` interpolates it; NaN where the point is not known.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, holding the isoflux boundary constraints in `magnetics/isoflux_boundary`
    /// * `pf_active` - the PF coils, in the order of `greens_tables/grid/pf_active`
    /// * `pf_passive` - the passive conductors, holding the filaments of those the equilibrium fits
    ///
    /// # Errors
    /// A point cannot be interpolated onto the reconstruction times, or a passive conductor the equilibrium fits is not in `pf_passive`
    pub fn greens_with_magnetics_isoflux_boundary(&mut self, magnetics: &Magnetics, pf_active: &PfActive, pf_passive: &PfPassive) -> Result<(), String> {
        let isoflux_boundaries: &Vec<MagneticsIsofluxBoundary> = &magnetics.magnetics_ids.isoflux_boundary;
        let n_isoflux_boundary: usize = isoflux_boundaries.len();
        let names: Vec<String> = settings_names(&self.settings_equilibrium.constraints.isoflux_boundary, isoflux_boundaries)?;

        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let sources: GreensSources = GreensSources::new(self, pf_active, pf_passive)?;
        let (n_r, n_z, n_coil, n_dof_total): (usize, usize, usize, usize) = (sources.n_r, sources.n_z, sources.n_coil, sources.n_dof_total);

        let mut greens_isoflux_boundaries: Vec<GreensIsofluxBoundary> = Vec::with_capacity(names.len());
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            if !names.contains(&isoflux_boundaries[i_isoflux_boundary].name) {
                continue;
            }
            let (position_r, position_z): (Array1<f64>, Array1<f64>) = isoflux_boundary_point(isoflux_boundaries, i_isoflux_boundary, &times_to_reconstruct)?;

            let mut g_plasma: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
            let mut g_d_plasma_d_z: Array3<f64> = Array3::from_elem((n_time, n_z, n_r), f64::NAN);
            let mut g_pf_active: Array2<f64> = Array2::from_elem((n_time, n_coil), f64::NAN);
            let mut g_pf_passive: Array2<f64> = Array2::from_elem((n_time, n_dof_total), f64::NAN);
            for i_time in 0..n_time {
                if !(position_r[i_time].is_finite() && position_z[i_time].is_finite()) {
                    continue;
                }
                // The poloidal flux, and its derivative with height for the vertical shift
                let quantities: [GreensQuantity; 2] = [Greens::psi, Greens::d_psi_d_z];
                let greens: Vec<PointGreens> = sources.point_greens(position_r[i_time], position_z[i_time], &quantities)?;

                g_plasma.slice_mut(s![i_time, .., ..]).assign(
                    &greens[0]
                        .plasma
                        .view()
                        .into_shape_with_order((n_z, n_r))
                        .expect("greens_with_magnetics_isoflux_boundary: failed to reshape `plasma` into (n_z, n_r)"),
                );
                g_d_plasma_d_z.slice_mut(s![i_time, .., ..]).assign(
                    &greens[1]
                        .plasma
                        .view()
                        .into_shape_with_order((n_z, n_r))
                        .expect("greens_with_magnetics_isoflux_boundary: failed to reshape `d_plasma_d_z` into (n_z, n_r)"),
                );
                g_pf_active.row_mut(i_time).assign(&greens[0].pf_active);
                g_pf_passive.row_mut(i_time).assign(&greens[0].pf_passive);
            }

            greens_isoflux_boundaries.push(GreensIsofluxBoundary {
                name: isoflux_boundaries[i_isoflux_boundary].name.clone(),
                plasma: g_plasma,
                d_plasma_d_z: g_d_plasma_d_z,
                pf_active: g_pf_active,
                pf_passive: g_pf_passive,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the isoflux boundary constraints' tables instead of repeating them
        self.greens_tables.isoflux_boundary = greens_isoflux_boundaries;

        Ok(())
    }
}
