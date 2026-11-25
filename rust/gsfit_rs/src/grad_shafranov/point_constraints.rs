//! The constraints at points which can move from one time-slice to the next, gathered for the solver at one time-slice: the isoflux,
//! isoflux boundary and stationary point constraints.
//!
//! Their points are copied onto each time-slice from the Magnetics IDS by `Equilibrium.copy_from_magnetics_ids`, and their Greens tables,
//! which have a table for each time-slice as the points move, are calculated by `Equilibrium.greens_with_magnetics`. `solve_grad_shafranov`
//! gathers them here, one `PointConstraints` per kind and time-slice, before the time-slices are solved, so that the solver, which does
//! not know which time-slice it is solving, is handed the Greens tables of its own.

use crate::greens_storage::{GreensIsoflux, GreensIsofluxBoundary, GreensStationaryPoint, GreensStationaryPointDPsi, GreensTables};
use crate::settings::settings_equilibrium::SettingsEquilibriumEquilibriumConstraints;
use imas_rs::{EquilibriumConstraintsIsoflux, EquilibriumConstraintsIsofluxBoundary, EquilibriumConstraintsStationaryPoint, EquilibriumTimeSlice};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

/// One kind of constraint at points which can move, at one time-slice: those fitted there, one column each.
///
/// A constraint is fitted at a time-slice when its fit settings include it and its points are known there, so `measured` is finite. A
/// stationary point is two constraints, `d(psi)/d(r) = 0` and `d(psi)/d(z) = 0`, so two columns, named `<name>/d_psi_d_r` and
/// `<name>/d_psi_d_z`.
#[derive(Clone, Debug)]
pub struct PointConstraints {
    /// The name of each constraint, in the order of the columns below
    pub names: Vec<String>,
    /// Greens with the plasma current in each grid cell, shape = [n_z * n_r, n_constraint]
    pub greens_with_grid: Array2<f64>,
    /// Greens with the plasma current in each grid cell, differentiated with respect to the points' height, for the vertical shift,
    /// shape = [n_z * n_r, n_constraint]
    pub greens_d_sensor_dz: Array2<f64>,
    /// Greens with each PF coil, in the order of `greens_tables/grid/pf_active`, shape = [n_pf, n_constraint]
    pub greens_with_pf: Array2<f64>,
    /// Greens with each passive degree of freedom, in the order of `greens_tables/grid/pf_passive(:)/dof(:)`, shape = [n_dof_total, n_constraint]
    pub greens_with_passives: Array2<f64>,
    /// Each constraint's `weight`, shape = [n_constraint] [dimensionless]
    pub weight: Array1<f64>,
    /// Each constraint's `expected_value`, shape = [n_constraint]
    pub expected_value: Array1<f64>,
    /// Each constraint's `measured`: zero, shape = [n_constraint]
    pub measured: Array1<f64>,
}

/// One column of a `PointConstraints`, borrowed from the Greens tables at the time-slice
struct Column<'a> {
    name: String,
    plasma: ArrayView2<'a, f64>,       // shape = [n_z, n_r]
    d_plasma_d_z: ArrayView2<'a, f64>, // shape = [n_z, n_r]
    pf_active: ArrayView1<'a, f64>,    // shape = [n_pf]
    pf_passive: ArrayView1<'a, f64>,   // shape = [n_dof_total]
    weight: f64,
    expected_value: f64,
    measured: f64,
}

/// The sizes of every `PointConstraints` at a time-slice, so that one with no constraints has the right number of rows
struct Sizes {
    n_grid: usize,
    n_pf: usize,
    n_dof_total: usize,
}

impl Sizes {
    fn new(time_slice: &EquilibriumTimeSlice, greens_tables: &GreensTables) -> Self {
        // `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
        let n_grid: usize = time_slice.profiles_2d[0].grid.dim1.len() * time_slice.profiles_2d[0].grid.dim2.len();
        let n_pf: usize = greens_tables.grid.pf_active.len();
        let mut n_dof_total: usize = 0;
        let n_greens_pf_passive: usize = greens_tables.grid.pf_passive.len();
        for i_greens_pf_passive in 0..n_greens_pf_passive {
            n_dof_total += greens_tables.grid.pf_passive[i_greens_pf_passive].dof.len();
        }
        Self { n_grid, n_pf, n_dof_total }
    }
}

impl PointConstraints {
    /// Put the columns side by side
    fn from_columns(columns: &[Column], sizes: &Sizes) -> Self {
        let n_constraint: usize = columns.len();
        let mut point_constraints: Self = Self {
            names: Vec::with_capacity(n_constraint),
            greens_with_grid: Array2::from_elem((sizes.n_grid, n_constraint), f64::NAN),
            greens_d_sensor_dz: Array2::from_elem((sizes.n_grid, n_constraint), f64::NAN),
            greens_with_pf: Array2::from_elem((sizes.n_pf, n_constraint), f64::NAN),
            greens_with_passives: Array2::from_elem((sizes.n_dof_total, n_constraint), f64::NAN),
            weight: Array1::from_elem(n_constraint, f64::NAN),
            expected_value: Array1::from_elem(n_constraint, f64::NAN),
            measured: Array1::from_elem(n_constraint, f64::NAN),
        };
        for i_constraint in 0..n_constraint {
            let column: &Column = &columns[i_constraint];
            point_constraints.names.push(column.name.clone());
            // Row-major, as the grid is flattened everywhere else
            point_constraints
                .greens_with_grid
                .column_mut(i_constraint)
                .assign(&Array1::from_iter(column.plasma.iter().copied()));
            point_constraints
                .greens_d_sensor_dz
                .column_mut(i_constraint)
                .assign(&Array1::from_iter(column.d_plasma_d_z.iter().copied()));
            point_constraints.greens_with_pf.column_mut(i_constraint).assign(&column.pf_active);
            point_constraints.greens_with_passives.column_mut(i_constraint).assign(&column.pf_passive);
            point_constraints.weight[i_constraint] = column.weight;
            point_constraints.expected_value[i_constraint] = column.expected_value;
            point_constraints.measured[i_constraint] = column.measured;
        }
        point_constraints
    }

    /// The isoflux constraints fitted at a time-slice, `time_slice(itime)/constraints/isoflux`, with their Greens tables at it,
    /// `greens_tables/isoflux(i1)`, found by the name of the constraint
    ///
    /// # Arguments
    /// * `time_slice` - the time-slice, with the constraints `Equilibrium.copy_from_magnetics_ids` copied onto it
    /// * `i_time` - the index of the time-slice, in `equilibrium/time_slice`, which the Greens tables are indexed by
    /// * `greens_tables` - the Greens tables, which `solve_grad_shafranov` checked have every constraint's, with one table per time-slice
    /// * `settings_constraints` - the fit settings, which say which constraints are fitted
    pub fn isoflux(
        time_slice: &EquilibriumTimeSlice,
        i_time: usize,
        greens_tables: &GreensTables,
        settings_constraints: &SettingsEquilibriumEquilibriumConstraints,
    ) -> Self {
        let mut columns: Vec<Column> = Vec::new();
        let n_isoflux: usize = time_slice.constraints.isoflux.len();
        for i_isoflux in 0..n_isoflux {
            let constraint: &EquilibriumConstraintsIsoflux = &time_slice.constraints.isoflux[i_isoflux];
            let is_included: bool = settings_constraints
                .isoflux
                .iter()
                .any(|settings_isoflux| settings_isoflux.name == constraint.name && settings_isoflux.include);
            if !is_included || !constraint.measured.is_finite() {
                continue;
            }
            let greens: &GreensIsoflux = greens_tables.isoflux(&constraint.name);
            columns.push(Column {
                name: constraint.name.clone(),
                plasma: greens.plasma.slice(s![i_time, .., ..]),
                d_plasma_d_z: greens.d_plasma_d_z.slice(s![i_time, .., ..]),
                pf_active: greens.pf_active.row(i_time),
                pf_passive: greens.pf_passive.row(i_time),
                weight: constraint.weight,
                expected_value: constraint.expected_value,
                measured: constraint.measured,
            });
        }
        Self::from_columns(&columns, &Sizes::new(time_slice, greens_tables))
    }

    /// The isoflux boundary constraints fitted at a time-slice, `time_slice(itime)/constraints/isoflux_boundary`, with their Greens tables
    /// at it, `greens_tables/isoflux_boundary(i1)`; see `isoflux` for the arguments
    pub fn isoflux_boundary(
        time_slice: &EquilibriumTimeSlice,
        i_time: usize,
        greens_tables: &GreensTables,
        settings_constraints: &SettingsEquilibriumEquilibriumConstraints,
    ) -> Self {
        let mut columns: Vec<Column> = Vec::new();
        let n_isoflux_boundary: usize = time_slice.constraints.isoflux_boundary.len();
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            let constraint: &EquilibriumConstraintsIsofluxBoundary = &time_slice.constraints.isoflux_boundary[i_isoflux_boundary];
            let is_included: bool = settings_constraints
                .isoflux_boundary
                .iter()
                .any(|settings_isoflux_boundary| settings_isoflux_boundary.name == constraint.name && settings_isoflux_boundary.include);
            if !is_included || !constraint.measured.is_finite() {
                continue;
            }
            let greens: &GreensIsofluxBoundary = greens_tables.isoflux_boundary(&constraint.name);
            columns.push(Column {
                name: constraint.name.clone(),
                plasma: greens.plasma.slice(s![i_time, .., ..]),
                d_plasma_d_z: greens.d_plasma_d_z.slice(s![i_time, .., ..]),
                pf_active: greens.pf_active.row(i_time),
                pf_passive: greens.pf_passive.row(i_time),
                weight: constraint.weight,
                expected_value: constraint.expected_value,
                measured: constraint.measured,
            });
        }
        Self::from_columns(&columns, &Sizes::new(time_slice, greens_tables))
    }

    /// The stationary point constraints fitted at a time-slice, `time_slice(itime)/constraints/stationary_point`, two columns each, with
    /// their Greens tables at it, `greens_tables/stationary_point(i1)`; see `isoflux` for the arguments
    pub fn stationary_point(
        time_slice: &EquilibriumTimeSlice,
        i_time: usize,
        greens_tables: &GreensTables,
        settings_constraints: &SettingsEquilibriumEquilibriumConstraints,
    ) -> Self {
        let mut columns: Vec<Column> = Vec::new();
        let n_stationary_point: usize = time_slice.constraints.stationary_point.len();
        for i_stationary_point in 0..n_stationary_point {
            let constraint: &EquilibriumConstraintsStationaryPoint = &time_slice.constraints.stationary_point[i_stationary_point];
            let is_included: bool = settings_constraints
                .stationary_point
                .iter()
                .any(|settings_stationary_point| settings_stationary_point.name == constraint.name && settings_stationary_point.include);
            if !is_included || !constraint.d_psi_d_r.measured.is_finite() || !constraint.d_psi_d_z.measured.is_finite() {
                continue;
            }
            let greens: &GreensStationaryPoint = greens_tables.stationary_point(&constraint.name);
            for (derivative_name, greens_derivative, measured) in [
                ("d_psi_d_r", &greens.d_psi_d_r, constraint.d_psi_d_r.measured),
                ("d_psi_d_z", &greens.d_psi_d_z, constraint.d_psi_d_z.measured),
            ] {
                let greens_derivative: &GreensStationaryPointDPsi = greens_derivative;
                columns.push(Column {
                    name: format!("{}/{derivative_name}", constraint.name),
                    plasma: greens_derivative.plasma.slice(s![i_time, .., ..]),
                    d_plasma_d_z: greens_derivative.d_plasma_d_z.slice(s![i_time, .., ..]),
                    pf_active: greens_derivative.pf_active.row(i_time),
                    pf_passive: greens_derivative.pf_passive.row(i_time),
                    weight: constraint.weight,
                    expected_value: constraint.expected_value,
                    measured,
                });
            }
        }
        Self::from_columns(&columns, &Sizes::new(time_slice, greens_tables))
    }
}
