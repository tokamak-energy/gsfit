//! What the constraints at points which can move from one time to the next share: `magnetics/isoflux`, `magnetics/isoflux_boundary`
//! and `magnetics/stationary_point`, all custom keys.
//!
//! Each point is a time trace, `r` and `z`, on its constraint's own timebase, `time`, which need not be the reconstruction times: it can
//! hold more or fewer times. It is checked when it is added, and interpolated onto the reconstruction times both when it is copied into
//! the Equilibrium IDS, by `Equilibrium.copy_from_magnetics_ids`, and when its Greens tables are calculated, by
//! `Equilibrium.greens_with_magnetics`, so the two always agree, whichever is called first.

use crate::equilibrium::Equilibrium;
use crate::greens::{Conductor, ConductorModel, Greens, Sensor};
use crate::pf_active::PfActive;
use crate::pf_passive::PfPassive;
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::{EMPTY_INT, EquilibriumConstraints0d};
use ndarray::{Array1, Array2, array};

/// Check a moving point's time trace when its constraint is added to the Magnetics IDS.
///
/// # Arguments
/// * `constraint` - the constraint's kind and name, for the errors, e.g. "isoflux `ISOFLUX01`"
/// * `time` - the constraint's timebase [second]
/// * `points` - each point's name, for the errors, e.g. "position_1", and its major radius and height at each time [metre]
/// * `validity_timed` - the validity at each time; empty when every time is valid
///
/// # Errors
/// Why the time trace cannot be added: an empty timebase, one which is not finite and increasing, a point or `validity_timed` without
/// one value per time, or a major radius which is finite but not positive. A NaN is allowed: the point is not known at that time,
/// which is not fitted.
pub(super) fn check_time_trace(
    constraint: &str,
    time: &Array1<f64>,
    points: &[(&str, &Array1<f64>, &Array1<f64>)],
    validity_timed: &Array1<i32>,
) -> Result<(), String> {
    let n_time: usize = time.len();
    if n_time == 0 {
        return Err(format!("{constraint}: `time` is empty; the point needs at least one time"));
    }
    for i_time in 0..n_time {
        if !time[i_time].is_finite() {
            return Err(format!("{constraint}: `time[{i_time}]` is {}; every time must be finite", time[i_time]));
        }
        if i_time > 0 && time[i_time] <= time[i_time - 1] {
            return Err(format!(
                "{constraint}: `time` is not increasing at index {i_time}: {} follows {}",
                time[i_time],
                time[i_time - 1]
            ));
        }
    }

    for (point_name, point_r, point_z) in points {
        for (coordinate_name, coordinate) in [("r", point_r), ("z", point_z)] {
            if coordinate.len() != n_time {
                return Err(format!(
                    "{constraint}: `time` has {n_time} point(s) but `{point_name}/{coordinate_name}` has {}; a point needs one value per time",
                    coordinate.len()
                ));
            }
        }
        // `Greens` needs a positive major radius
        for i_time in 0..n_time {
            if point_r[i_time].is_finite() && point_r[i_time] <= 0.0 {
                return Err(format!(
                    "{constraint}: `{point_name}/r[{i_time}]` = {} metre; a major radius must be positive, or NaN where the point is not known",
                    point_r[i_time]
                ));
            }
        }
    }

    if !validity_timed.is_empty() && validity_timed.len() != n_time {
        return Err(format!(
            "{constraint}: `time` has {n_time} point(s) but `validity_timed` has {}; it needs one value per time, or none when every time is valid",
            validity_timed.len()
        ));
    }

    Ok(())
}

/// Interpolate one coordinate of a moving point onto the reconstruction times.
///
/// A time whose `validity_timed` is negative is not valid, and neither is any time when `validity` is negative: the data dictionary's
/// "problem identified" and "invalid data" values. `EMPTY_INT`, unset, is valid. An invalid time is NaN before the coordinate is
/// interpolated, so the reconstruction times next to it are NaN too.
///
/// The coordinate is interpolated linearly. A reconstruction time outside the timebase is NaN, as the point is not known there; the
/// timebase need not cover the reconstruction times.
///
/// # Arguments
/// * `time` - the constraint's timebase, checked by `check_time_trace` [second]
/// * `coordinate` - the coordinate at each time [metre]
/// * `validity_timed` - the validity at each time; empty when every time is valid
/// * `validity` - the validity of the whole time trace; `EMPTY_INT` when it is valid
/// * `times_to_reconstruct` - the reconstruction times [second]
///
/// # Returns
/// The coordinate at each reconstruction time; NaN where it is not known [metre]
///
/// # Errors
/// The timebase is not increasing, or has a different length to the coordinate, which `check_time_trace` stops when the point is added
pub(super) fn interpolate_onto_times(
    time: &Array1<f64>,
    coordinate: &Array1<f64>,
    validity_timed: &Array1<i32>,
    validity: i32,
    times_to_reconstruct: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let n_time_to_reconstruct: usize = times_to_reconstruct.len();
    let mut interpolated: Array1<f64> = Array1::from_elem(n_time_to_reconstruct, f64::NAN);

    if validity != EMPTY_INT && validity < 0 {
        return Ok(interpolated);
    }

    let mut coordinate_valid: Array1<f64> = coordinate.clone();
    let n_validity_timed: usize = validity_timed.len();
    for i_time in 0..n_validity_timed {
        if validity_timed[i_time] != EMPTY_INT && validity_timed[i_time] < 0 {
            coordinate_valid[i_time] = f64::NAN;
        }
    }

    let interpolator: interpolation::Dim1Linear =
        interpolation::Dim1Linear::new(time.clone(), coordinate_valid).map_err(|error: interpolation::Error| format!("{error:?}"))?;
    for i_time_to_reconstruct in 0..n_time_to_reconstruct {
        // Outside the timebase the point is not known, which is not an error: `interpolated` stays NaN
        if let Ok(value) = interpolator.interpolate_scalar(times_to_reconstruct[i_time_to_reconstruct]) {
            interpolated[i_time_to_reconstruct] = value;
        }
    }

    Ok(interpolated)
}

/// The current sources a moving point's Greens tables are calculated with, gathered once for every point and time-slice: the grid, the
/// PF coils, in the order of `greens_tables/grid/pf_active`, and the passive degrees of freedom, in the order of
/// `greens_tables/grid/pf_passive(:)/dof(:)`, which `Equilibrium.greens_with_magnetics` checked.
pub(super) struct GreensSources {
    /// Number of grid points in `r`
    pub(super) n_r: usize,
    /// Number of grid points in `z`
    pub(super) n_z: usize,
    /// The grid's cells, `d_r` by `d_z`, centred on the grid points, flattened from `profiles_2d(0)/r` and `/z`, n_z * n_r of them
    grid_cell: Conductor,
    /// Each PF coil's filaments, in the order of `pf_active/coil`
    coil_conductors: Vec<Conductor>,
    /// The elements of each passive conductor the equilibrium fits, in the order of `constraints/pf_passive_current`
    loop_conductors: Vec<Conductor>,
    /// How the conductors are treated, from `equilibrium_settings/numerics/greens`
    conductor_model: ConductorModel,
    /// Each passive degree of freedom's current distribution over its conductor's filaments, `[i_loop][i_dof]` [dimensionless]
    current_distributions: Vec<Vec<Array1<f64>>>,
    /// Number of PF coils
    pub(super) n_coil: usize,
    /// Number of passive degrees of freedom, over every passive conductor
    pub(super) n_dof_total: usize,
}

impl GreensSources {
    /// Gather the current sources.
    ///
    /// # Arguments
    /// * `equilibrium` - the equilibrium, holding the grid and the passive degrees of freedom, `constraints/pf_passive_current`
    /// * `pf_active` - the PF coils
    /// * `pf_passive` - the passive conductors, holding the filaments of those the equilibrium fits
    ///
    /// # Errors
    /// A passive conductor the equilibrium fits is not in `pf_passive`
    pub(super) fn new(equilibrium: &Equilibrium, pf_active: &PfActive, pf_passive: &PfPassive) -> Result<Self, String> {
        // The grid. `time_slice(0)` because the grid is the same on every time-slice, and `profiles_2d(0)` because GSFit solves on a single
        // rectangular (R, Z) grid. `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives the flattened grid
        let n_r: usize = equilibrium.settings_equilibrium.grid.n_r;
        let n_z: usize = equilibrium.settings_equilibrium.grid.n_z;
        let mesh_r: &Array2<f64> = &equilibrium.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &equilibrium.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let conductor_model: ConductorModel = ConductorModel::from_settings(&equilibrium.settings_equilibrium.numerics.greens)?;

        // The grid's cells, `d_r` by `d_z`, centred on the grid points; as obliques, each is described from its corner at the smallest r
        // and z
        let d_r: f64 = mesh_r[(0, 1)] - mesh_r[(0, 0)];
        let d_z: f64 = mesh_z[(1, 0)] - mesh_z[(0, 0)];
        let n_grid: usize = n_z * n_r;
        let grid_cell: Conductor = Conductor::from_oblique(
            &(Array1::from_iter(mesh_r.iter().copied()) - 0.5 * d_r),
            &(Array1::from_iter(mesh_z.iter().copied()) - 0.5 * d_z),
            &Array1::from_elem(n_grid, d_r),
            &Array1::from_elem(n_grid, d_z),
            &Array1::zeros(n_grid),
            &Array1::zeros(n_grid),
        )
        .map_err(|error: String| format!("the grid's cells: {error}"))?;

        // The PF coils, each coil's filaments, `element(i2)/geometry/rectangle`, described from their centres. As obliques they are
        // described from their corners at the smallest r and z, with both angles zero
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();
        let mut coil_conductors: Vec<Conductor> = Vec::with_capacity(n_coil);
        for i_coil in 0..n_coil {
            let coil_width: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.width.to_array();
            let coil_height: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.height.to_array();
            let n_coil_filaments: usize = coil_width.len();
            coil_conductors.push(
                Conductor::from_oblique(
                    &(pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array() - 0.5 * &coil_width),
                    &(pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array() - 0.5 * &coil_height),
                    &coil_width,
                    &coil_height,
                    &Array1::zeros(n_coil_filaments),
                    &Array1::zeros(n_coil_filaments),
                )
                .map_err(|error: String| format!("PF coil `{}`: {error}", pf_active_coils[i_coil].name))?,
            );
        }

        // The passive conductors the equilibrium fits, with their degrees of freedom. `time_slice(0)` because the degrees of freedom are
        // the same on every time-slice
        let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &equilibrium.equilibrium_ids.time_slice[0].constraints.pf_passive_current;
        let n_loop: usize = pf_passive_currents.len();
        let mut loop_conductors: Vec<Conductor> = Vec::with_capacity(n_loop);
        let mut current_distributions: Vec<Vec<Array1<f64>>> = Vec::with_capacity(n_loop);
        let mut n_dof_total: usize = 0;
        for i_loop in 0..n_loop {
            let i_pf_passive_loop: usize = pf_passive.i_loop_named(&pf_passive_currents[i_loop].name)?;
            let pf_passive_loop: &PfPassiveLoops = &pf_passive.pf_passive_ids.r#loop[i_pf_passive_loop];
            loop_conductors.push(
                Conductor::from_oblique(
                    &pf_passive_loop.element(..).geometry.oblique.r.to_array(),
                    &pf_passive_loop.element(..).geometry.oblique.z.to_array(),
                    &pf_passive_loop.element(..).geometry.oblique.length_alpha.to_array(),
                    &pf_passive_loop.element(..).geometry.oblique.length_beta.to_array(),
                    &pf_passive_loop.element(..).geometry.oblique.alpha.to_array(),
                    &pf_passive_loop.element(..).geometry.oblique.beta.to_array(),
                )
                .map_err(|error: String| format!("passive conductor `{}`: {error}", pf_passive_loop.name))?,
            );
            let n_dof: usize = pf_passive_currents[i_loop].dof.len();
            let mut current_distributions_this_loop: Vec<Array1<f64>> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                current_distributions_this_loop.push(pf_passive_currents[i_loop].dof[i_dof].current_distribution.clone());
            }
            current_distributions.push(current_distributions_this_loop);
            n_dof_total += n_dof;
        }

        Ok(Self {
            n_r,
            n_z,
            grid_cell,
            coil_conductors,
            loop_conductors,
            conductor_model,
            current_distributions,
            n_coil,
            n_dof_total,
        })
    }

    /// What one point sees, per ampere in each source, for each of `quantities`, e.g. `Greens::psi` and `Greens::d_psi_d_z`.
    ///
    /// # Arguments
    /// * `r` - the point's major radius, positive [metre]
    /// * `z` - the point's height [metre]
    /// * `quantities` - what the point sees, each a method of `Greens` giving an array of shape [1, n_source]
    ///
    /// # Returns
    /// For each of `quantities`, in the same order, the point's tables with every source
    ///
    /// # Errors
    /// The point is not at a positive major radius, or a table cannot be calculated with the settings' `numerics/greens/conductor_model`
    pub(super) fn point_greens(&self, r: f64, z: f64, quantities: &[GreensQuantity]) -> Result<Vec<PointGreens>, String> {
        let n_quantity: usize = quantities.len();
        let n_grid: usize = self.grid_cell.n_conductor;
        let mut point_greens: Vec<PointGreens> = Vec::with_capacity(n_quantity);
        for _ in 0..n_quantity {
            point_greens.push(PointGreens {
                plasma: Array1::from_elem(n_grid, f64::NAN),
                pf_active: Array1::from_elem(self.n_coil, f64::NAN),
                pf_passive: Array1::from_elem(self.n_dof_total, f64::NAN),
            });
        }
        let point_sensor: Sensor = Sensor::new(&array![r], &array![z])?;

        // With the plasma, one table per grid cell
        let greens_calculator: Greens = Greens::sensor_to_conductor(&point_sensor, &self.grid_cell, self.conductor_model.clone())?;
        for i_quantity in 0..n_quantity {
            point_greens[i_quantity].plasma.assign(&quantities[i_quantity](&greens_calculator)?.row(0));
        }

        // With each PF coil, summed over the coil's filaments
        for i_coil in 0..self.n_coil {
            let greens_calculator: Greens = Greens::sensor_to_conductor(&point_sensor, &self.coil_conductors[i_coil], self.conductor_model.clone())?;
            for i_quantity in 0..n_quantity {
                point_greens[i_quantity].pf_active[i_coil] = quantities[i_quantity](&greens_calculator)?.sum();
            }
        }

        // With each passive degree of freedom: its conductor's elements, weighted by the degree of freedom's current distribution
        let n_loop: usize = self.loop_conductors.len();
        let mut i_dof_total: usize = 0;
        for i_loop in 0..n_loop {
            let greens_calculator: Greens = Greens::sensor_to_conductor(&point_sensor, &self.loop_conductors[i_loop], self.conductor_model.clone())?;
            let mut quantities_filaments: Vec<Array2<f64>> = Vec::with_capacity(n_quantity); // each shape = [1, n_filament]
            for i_quantity in 0..n_quantity {
                quantities_filaments.push(quantities[i_quantity](&greens_calculator)?);
            }
            let n_dof: usize = self.current_distributions[i_loop].len();
            for i_dof in 0..n_dof {
                let current_distribution: &Array1<f64> = &self.current_distributions[i_loop][i_dof];
                for i_quantity in 0..n_quantity {
                    point_greens[i_quantity].pf_passive[i_dof_total] = (&quantities_filaments[i_quantity] * current_distribution).sum();
                }
                i_dof_total += 1;
            }
        }

        Ok(point_greens)
    }
}

/// One of the tables which a point sees, a method of `Greens`, e.g. `Greens::psi` or `Greens::d_psi_d_z`; see `GreensSources::point_greens`
pub(super) type GreensQuantity = fn(&Greens) -> Result<Array2<f64>, String>;

/// What one point sees, per ampere in each source, of one quantity, e.g. the poloidal flux; see `GreensSources::point_greens`
pub(super) struct PointGreens {
    /// The plasma current in each grid cell, flattened, shape = [n_z * n_r]
    pub(super) plasma: Array1<f64>,
    /// Each PF coil, shape = [n_coil]
    pub(super) pf_active: Array1<f64>,
    /// Each passive degree of freedom, shape = [n_dof_total]
    pub(super) pf_passive: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_interpolated_linearly_and_is_nan_outside_its_timebase() {
        let time: Array1<f64> = array![0.1, 0.2, 0.3];
        let coordinate: Array1<f64> = array![1.0, 2.0, 4.0];

        let interpolated: Array1<f64> = interpolate_onto_times(&time, &coordinate, &Array1::zeros(0), EMPTY_INT, &array![0.05, 0.1, 0.25, 0.3, 0.35]).unwrap();

        assert!(interpolated[0].is_nan());
        assert_eq!(interpolated[1], 1.0);
        assert!((interpolated[2] - 3.0).abs() < 1.0e-12);
        assert_eq!(interpolated[3], 4.0);
        assert!(interpolated[4].is_nan());
    }

    #[test]
    fn an_invalid_time_is_nan_and_so_are_the_reconstruction_times_next_to_it() {
        let time: Array1<f64> = array![0.1, 0.2, 0.3];
        let coordinate: Array1<f64> = array![1.0, 2.0, 4.0];
        let validity_timed: Array1<i32> = array![0, -2, 1];

        let interpolated: Array1<f64> = interpolate_onto_times(&time, &coordinate, &validity_timed, EMPTY_INT, &array![0.1, 0.15, 0.2, 0.3]).unwrap();

        assert_eq!(interpolated[0], 1.0);
        assert!(interpolated[1].is_nan());
        assert!(interpolated[2].is_nan());
        assert_eq!(interpolated[3], 4.0);
    }

    #[test]
    fn an_invalid_time_trace_is_nan_everywhere() {
        let time: Array1<f64> = array![0.1, 0.2];
        let coordinate: Array1<f64> = array![1.0, 2.0];

        let interpolated: Array1<f64> = interpolate_onto_times(&time, &coordinate, &Array1::zeros(0), -1, &array![0.1, 0.15]).unwrap();

        assert!(interpolated.iter().all(|value: &f64| value.is_nan()));
    }

    #[test]
    fn a_time_trace_which_cannot_be_interpolated_is_an_error() {
        let r: Array1<f64> = array![1.0, 2.0];
        let z: Array1<f64> = array![0.0, 0.1];
        let validity_timed: Array1<i32> = Array1::zeros(0);

        // Times which are not increasing, or not finite
        assert!(check_time_trace("isoflux `A`", &array![0.2, 0.1], &[("position_1", &r, &z)], &validity_timed).is_err());
        assert!(check_time_trace("isoflux `A`", &array![0.1, f64::NAN], &[("position_1", &r, &z)], &validity_timed).is_err());
        // A point without one value per time
        assert!(check_time_trace("isoflux `A`", &array![0.1, 0.2, 0.3], &[("position_1", &r, &z)], &validity_timed).is_err());
        assert!(check_time_trace("isoflux `A`", &array![0.1, 0.2], &[("position_1", &r, &array![0.0])], &validity_timed).is_err());
        // A major radius which is not positive
        assert!(check_time_trace("isoflux `A`", &array![0.1, 0.2], &[("position_1", &array![1.0, -1.0], &z)], &validity_timed).is_err());
        // `validity_timed` without one value per time
        assert!(check_time_trace("isoflux `A`", &array![0.1, 0.2], &[("position_1", &r, &z)], &array![0]).is_err());
        // A point which is not known at a time is allowed
        assert!(check_time_trace("isoflux `A`", &array![0.1, 0.2], &[("position_1", &array![1.0, f64::NAN], &z)], &validity_timed).is_ok());
    }
}
