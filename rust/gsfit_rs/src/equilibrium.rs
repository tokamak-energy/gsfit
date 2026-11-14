use crate::core_profiles::CoreProfiles;
use crate::grad_shafranov::{NewtonKrylov, NewtonPicard, Picard, SharedNonlinearSolver};
use crate::greens::Greens;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use crate::greens_storage::{GreensGrid, GreensGridPfActive, GreensGridPfPassive, GreensGridPfPassiveDof, GreensGridPlasma, GreensTables, PyGreensTables};
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use crate::pulse_schedule::PulseSchedule;
use crate::settings::settings_equilibrium::{
    SettingsEquilibrium, SettingsEquilibriumEfitPolynomial, SettingsEquilibriumEquilibriumConstraints, SettingsEquilibriumGrid, SettingsEquilibriumNumerics,
    SettingsEquilibriumNumericsNonlinearSolver, SettingsEquilibriumNumericsNonlinearSolverNewtonKrylov, SettingsEquilibriumNumericsNonlinearSolverNewtonPicard,
    SettingsEquilibriumNumericsNonlinearSolverPicard, SettingsEquilibriumPassive, SettingsEquilibriumPressure, SettingsEquilibriumRegularisationBuilder,
    SettingsEquilibriumTensionedCubicBSpline, SettingsEquilibriumTimeslices,
};
use crate::source_functions::{EfitPolynomial, SharedSourceFunction, TensionedCubicBSpline};
use crate::tf::Tf;
use crate::wall::Wall;
use faer::Side;
use imas_rs::ids::core_profiles::CoreProfilesProfiles2d;
use imas_rs::ids::equilibrium::{
    Code, Equilibrium as EquilibriumIds, EquilibriumConstraints0d, EquilibriumConstraints0dB0Like, EquilibriumConstraints0dDof, EquilibriumConstraints0dIpLike,
    EquilibriumConstraints0dOneLike, EquilibriumConstraints0dPosition, EquilibriumConstraintsIsoflux, EquilibriumConstraintsIsofluxBoundary,
    EquilibriumConstraintsStationaryPoint, EquilibriumGap, EquilibriumLimiterUnit, EquilibriumLimiterUnitOutline, EquilibriumProfiles2d,
    Rphizpsirho0dDynamicAos3,
};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::ids::pulse_schedule::PulseScheduleGap;
use imas_rs::ids::wall::Wall2dLimiterUnit;
use imas_rs::python::{PickleState, PickleStateError, PyEquilibrium, PyParametersInput, PyPath, read_path};
use ndarray::{Array1, Array2, ArrayView2, Axis, MeshIndex, meshgrid, s};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The equilibrium, stored as an IMAS `equilibrium` IDS.
///
/// It is built from GSFit's settings alone, `settings_equilibrium`:
/// * `code/parameters` - GSFit's settings, as XML text, which is where IMAS records the parameters of the code which made an IDS
/// * `time_slice(itime)` - one per time to reconstruct, each holding its `time`, the (R, Z) grid in `profiles_2d(0)`, the
///   normalised poloidal flux grid of the 1D profiles, `profiles_1d/psi_norm`, and the weight of each magnetic sensor in the
///   settings, in `constraints/<sensor>(i1)/weight`
///
/// The Greens tables are GSFit's own, which the data dictionary has no nodes for, so they are held alongside the IDS, in
/// `greens_tables`, whose structs are generated from the XSD schema `greens_storage.xsd`. Those between the grid points,
/// `greens_tables/grid/plasma`, are made from the settings too. The source functions, p' and ff', and the nonlinear solver are not
/// data, so are held alongside the IDS too.
///
/// The other IDSs are then copied in, by `copy_from_wall_ids`, `copy_from_tf_ids`, `copy_from_pf_active_ids`,
/// `copy_from_pf_passive_ids`, `copy_from_pulse_schedule_ids`, `copy_from_magnetics_ids` and `copy_from_core_profiles_ids`, and the Greens
/// tables with the PF coils,
/// the passive conductors and the magnetic sensors are calculated, by `greens_with_pf_active`, `greens_with_pf_passive` and
/// `greens_with_magnetics`, before `solve_grad_shafranov` solves it.
///
/// Read the IDS's data back through `get`, or `equilibrium_ids`, and a path from `gsfit_rs.imas.equilibrium_paths`; there are no
/// bespoke accessors. The Greens tables are read in Rust with their generated accessors, e.g. `greens_tables.flux_loop(name)`, and in
/// Python through the getter `greens_tables`, a copy, and a path from `gsfit_rs.imas.greens_tables_paths`.
///
/// # Settings
///
/// The equilibrium is constructed with GSFit's settings, `equilibrium.xml` with each file it links to written in its link's
/// place, as an IMAS `parameters_input`. They are validated against the XML schema in `parameters_input`, and only their XML is
/// kept: it is written to `code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_equilibrium`, whose structs are generated from the XML schema `equilibrium.xsd`, and
/// everything above is built from them. They are kept here, not in the IDS, as the equilibrium IDS can also be made by another
/// code, whose `code/parameters` would not hold GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct Equilibrium {
    /// The Equilibrium IDS data structure
    pub equilibrium_ids: EquilibriumIds,
    /// The Equilibrium XML settings passed into typed struct
    pub settings_equilibrium: SettingsEquilibrium,
    /// The p' source function, made from `settings_equilibrium.source_function_p_prime`
    pub p_prime_source_function: SharedSourceFunction,
    /// The ff' source function, made from `settings_equilibrium.source_function_ff_prime`
    pub ff_prime_source_function: SharedSourceFunction,
    /// The nonlinear solver, made from `settings_equilibrium.numerics`
    pub nonlinear_solver: SharedNonlinearSolver,
    /// The Greens tables
    pub greens_tables: GreensTables,
}

/// Python accessible methods
#[pymethods]
impl Equilibrium {
    /// Construct the equilibrium from GSFit's settings.
    ///
    /// The settings are read into `settings_equilibrium`, and everything else is made from them: the time-slices, the grid, the
    /// magnetic sensors' weights, the Greens tables between the grid points, and the source functions. See `Equilibrium`.
    ///
    /// # Arguments
    /// * `parameters_input` - GSFit's settings, the equilibrium IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string: `equilibrium.xml`, with each file it links to written in its link's place
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.parameters_default` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_equilibrium` data types, e.g. an integer too large for its type
    ///   - the settings cannot make an equilibrium, e.g. a radial grid which goes negative
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of GSFit's settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let parameters_value: &str = &parameters_input.inner.parameters_value;
        let settings_document: Document = oxml::parse(parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, GSFit's settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!("GSFit's settings do not follow their XML schema:\n{report}")));
        }

        // Parse the XML string settings into a typed `SettingsEquilibrium` struct
        let settings_equilibrium: SettingsEquilibrium = SettingsEquilibrium::from_xml(parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("GSFit's settings cannot be read into `SettingsEquilibrium`: {error}")))?;

        // The times to reconstruct
        let settings_timeslices: &SettingsEquilibriumTimeslices = &settings_equilibrium.timeslices;
        let times_to_reconstruct: Array1<f64> = match settings_timeslices.method.as_str() {
            "arange" => {
                let time_start: f64 = settings_timeslices.arange.time_start;
                let time_end: f64 = settings_timeslices.arange.time_end;
                // XSD v1.0 schema doesn't allow comparison between values, so we implement the check here:
                if time_end < time_start {
                    return Err(PyValueError::new_err(format!(
                        "`equilibrium_settings/timeslices/arange` has time_end = {time_end} before time_start = {time_start}"
                    )));
                }
                Array1::range(time_start, time_end, settings_timeslices.arange.dt)
            }
            "linspace" => Array1::linspace(
                settings_timeslices.linspace.time_start,
                settings_timeslices.linspace.time_end,
                settings_timeslices.linspace.n_time,
            ),
            "user_defined" => settings_timeslices.user_defined.clone(),
            // "good_pressure_sensors" is in the XML schema, but is not implemented yet
            method => {
                return Err(PyValueError::new_err(format!(
                    "`equilibrium_settings/timeslices/method` = \"{method}\" is not recognized"
                )));
            }
        };

        // The (R, Z) grid. The XML schema has each number of points at least 2, and the extent finite
        let grid: &SettingsEquilibriumGrid = &settings_equilibrium.grid;
        let n_r: usize = grid.n_r;
        let n_z: usize = grid.n_z;
        let r: Array1<f64> = Array1::linspace(grid.r_min, grid.r_max, n_r);
        let z: Array1<f64> = Array1::linspace(grid.z_min, grid.z_max, n_z);

        // Grid spacing
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let d_area: f64 = d_r * d_z;

        // Check that the R grid doesn't go negative
        // Note, we allow cells to touch the axis (R=0), which would be excluded by `r_min - d_r / 2.0 <= 0.0`
        // but we do not allow cells to go negative.
        if grid.r_min - d_r / 2.0 < 0.0 {
            return Err(PyValueError::new_err(format!(
                "`equilibrium_settings/grid` has r_min - d_r / 2 = {}, so the radial grid goes negative",
                grid.r_min - d_r / 2.0
            )));
        }

        // 2d (r, z) mesh
        let (mesh_z_view, mesh_r_view): (ArrayView2<f64>, ArrayView2<f64>) = meshgrid((&z, &r), MeshIndex::IJ);
        let mesh_z: Array2<f64> = mesh_z_view.to_owned(); // shape = (n_z, n_r)
        let mesh_r: Array2<f64> = mesh_r_view.to_owned(); // shape = (n_z, n_r)

        // The normalised poloidal flux grid of the 1D profiles, which the source functions are evaluated on
        let psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, grid.n_psi_norm);

        // The nonlinear solver
        let nonlinear_solver: SharedNonlinearSolver = nonlinear_solver_from_settings(&settings_equilibrium.numerics)?;

        // The source functions
        let p_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &settings_equilibrium.source_function_p_prime.method,
            settings_equilibrium.source_function_p_prime.efit_polynomial.as_ref(),
            settings_equilibrium.source_function_p_prime.tensioned_cubic_b_spline.as_ref(),
            "equilibrium_settings/source_function_p_prime",
        )?;
        let ff_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &settings_equilibrium.source_function_ff_prime.method,
            settings_equilibrium.source_function_ff_prime.efit_polynomial.as_ref(),
            settings_equilibrium.source_function_ff_prime.tensioned_cubic_b_spline.as_ref(),
            "equilibrium_settings/source_function_ff_prime",
        )?;

        // The Greens tables between the grid points, `greens_tables/grid/plasma`. The sensors are every grid point, row by row, and the
        // conductors the points of the lowest row, so each table is calculated with shape (n_z * n_r, n_r), and stored as
        // (i_z_sensor, i_r_sensor, i_r_conductor). A sensor's row is then its offset above the conductor
        let flat_r: Array1<f64> = mesh_r.flatten().to_owned();
        let flat_z: Array1<f64> = mesh_z.flatten().to_owned();
        let flat_d_r: Array1<f64> = &r * 0.0 + d_r;
        let flat_d_z: Array1<f64> = &r * 0.0 + d_z;
        let greens_calculator: Greens = Greens::sensor_to_conductor(flat_r, flat_z, r.clone(), 0.0 * r.clone() + z[0], flat_d_r, flat_d_z);
        let g_psi: Array2<f64> = greens_calculator.psi();
        let mut g_d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r();
        let mut g_d_psi_d_z: Array2<f64> = greens_calculator.d_psi_d_z();
        let mut g_d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2();
        let mut g_d2_psi_d_r_d_z: Array2<f64> = greens_calculator.d2_psi_d_r_d_z();
        let mut g_d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2();
        let mut g_d3_psi_d_r2_d_z: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z();
        let mut g_d3_psi_d_r_d_z2: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2();
        let mut g_d3_psi_d_z3: Array2<f64> = greens_calculator.d3_psi_d_z3();
        // A derivative which could not be calculated is NaN, and is taken as 0
        for g_derivative in [
            &mut g_d_psi_d_r,
            &mut g_d_psi_d_z,
            &mut g_d2_psi_d_r2,
            &mut g_d2_psi_d_r_d_z,
            &mut g_d2_psi_d_z2,
            &mut g_d3_psi_d_r2_d_z,
            &mut g_d3_psi_d_r_d_z2,
            &mut g_d3_psi_d_z3,
        ] {
            g_derivative.mapv_inplace(|value: f64| if value.is_nan() { 0.0 } else { value });
        }
        let greens_grid_plasma: GreensGridPlasma = GreensGridPlasma {
            psi: g_psi
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_psi` into (n_z, n_r, n_r)"),
            d_psi_d_r: g_d_psi_d_r
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d_psi_d_r` into (n_z, n_r, n_r)"),
            d_psi_d_z: g_d_psi_d_z
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d_psi_d_z` into (n_z, n_r, n_r)"),
            d2_psi_d_r2: g_d2_psi_d_r2
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d2_psi_d_r2` into (n_z, n_r, n_r)"),
            d2_psi_d_r_d_z: g_d2_psi_d_r_d_z
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d2_psi_d_r_d_z` into (n_z, n_r, n_r)"),
            d2_psi_d_z2: g_d2_psi_d_z2
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d2_psi_d_z2` into (n_z, n_r, n_r)"),
            d3_psi_d_r2_d_z: g_d3_psi_d_r2_d_z
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d3_psi_d_r2_d_z` into (n_z, n_r, n_r)"),
            d3_psi_d_r_d_z2: g_d3_psi_d_r_d_z2
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d3_psi_d_r_d_z2` into (n_z, n_r, n_r)"),
            d3_psi_d_z3: g_d3_psi_d_z3
                .into_shape_with_order((n_z, n_r, n_r))
                .expect("Equilibrium::new: failed to reshape `g_d3_psi_d_z3` into (n_z, n_r, n_r)"),
        };

        // The magnetic sensors' weights, from their fit settings, `equilibrium_settings/constraints`: one constraint for each sensor in
        // the settings, with its `name`, `weight` and `expected_value`. `copy_from_magnetics_ids` adds each sensor's measurement, found by its name
        let constraints_settings: &SettingsEquilibriumEquilibriumConstraints = &settings_equilibrium.constraints;
        let n_b_field_pol_probe: usize = constraints_settings.b_field_pol_probe.len();
        let mut b_field_pol_probe_constraints: Vec<EquilibriumConstraints0dOneLike> = Vec::with_capacity(n_b_field_pol_probe);
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            b_field_pol_probe_constraints.push(EquilibriumConstraints0dOneLike {
                name: constraints_settings.b_field_pol_probe[i_b_field_pol_probe].name.clone(),
                weight: constraints_settings.b_field_pol_probe[i_b_field_pol_probe].weight,
                expected_value: constraints_settings.b_field_pol_probe[i_b_field_pol_probe].expected_value,
                ..Default::default()
            });
        }
        let n_flux_loop: usize = constraints_settings.flux_loop.len();
        let mut flux_loop_constraints: Vec<EquilibriumConstraints0d> = Vec::with_capacity(n_flux_loop);
        for i_flux_loop in 0..n_flux_loop {
            flux_loop_constraints.push(EquilibriumConstraints0d {
                name: constraints_settings.flux_loop[i_flux_loop].name.clone(),
                weight: constraints_settings.flux_loop[i_flux_loop].weight,
                expected_value: constraints_settings.flux_loop[i_flux_loop].expected_value,
                ..Default::default()
            });
        }
        let n_rogowski_coil: usize = constraints_settings.rogowski_coil.len();
        let mut rogowski_coil_constraints: Vec<EquilibriumConstraints0dIpLike> = Vec::with_capacity(n_rogowski_coil);
        for i_rogowski_coil in 0..n_rogowski_coil {
            rogowski_coil_constraints.push(EquilibriumConstraints0dIpLike {
                name: constraints_settings.rogowski_coil[i_rogowski_coil].name.clone(),
                weight: constraints_settings.rogowski_coil[i_rogowski_coil].weight,
                expected_value: constraints_settings.rogowski_coil[i_rogowski_coil].expected_value,
                ..Default::default()
            });
        }
        // The data dictionary has a single diamagnetic flux constraint, which has no name. Its settings are optional: without them it is
        // left unset
        let mut diamagnetic_flux_constraint: EquilibriumConstraints0dB0Like = EquilibriumConstraints0dB0Like::default();
        if let Some(diamagnetic_flux_settings) = &constraints_settings.diamagnetic_flux {
            diamagnetic_flux_constraint.weight = diamagnetic_flux_settings.weight;
            diamagnetic_flux_constraint.expected_value = diamagnetic_flux_settings.expected_value;
        }
        // Likewise the constraints at points which can move, custom keys: `copy_from_magnetics_ids` adds each one's points, found by its name
        let n_isoflux: usize = constraints_settings.isoflux.len();
        let mut isoflux_constraints: Vec<EquilibriumConstraintsIsoflux> = Vec::with_capacity(n_isoflux);
        for i_isoflux in 0..n_isoflux {
            isoflux_constraints.push(EquilibriumConstraintsIsoflux {
                name: constraints_settings.isoflux[i_isoflux].name.clone(),
                weight: constraints_settings.isoflux[i_isoflux].weight,
                expected_value: constraints_settings.isoflux[i_isoflux].expected_value,
                ..Default::default()
            });
        }
        let n_isoflux_boundary: usize = constraints_settings.isoflux_boundary.len();
        let mut isoflux_boundary_constraints: Vec<EquilibriumConstraintsIsofluxBoundary> = Vec::with_capacity(n_isoflux_boundary);
        for i_isoflux_boundary in 0..n_isoflux_boundary {
            isoflux_boundary_constraints.push(EquilibriumConstraintsIsofluxBoundary {
                name: constraints_settings.isoflux_boundary[i_isoflux_boundary].name.clone(),
                weight: constraints_settings.isoflux_boundary[i_isoflux_boundary].weight,
                expected_value: constraints_settings.isoflux_boundary[i_isoflux_boundary].expected_value,
                ..Default::default()
            });
        }
        let n_stationary_point: usize = constraints_settings.stationary_point.len();
        let mut stationary_point_constraints: Vec<EquilibriumConstraintsStationaryPoint> = Vec::with_capacity(n_stationary_point);
        for i_stationary_point in 0..n_stationary_point {
            stationary_point_constraints.push(EquilibriumConstraintsStationaryPoint {
                name: constraints_settings.stationary_point[i_stationary_point].name.clone(),
                weight: constraints_settings.stationary_point[i_stationary_point].weight,
                expected_value: constraints_settings.stationary_point[i_stationary_point].expected_value,
                ..Default::default()
            });
        }

        let mut equilibrium_ids: EquilibriumIds = EquilibriumIds {
            code: Code {
                parameters: parameters_value.to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        // One time-slice per time to reconstruct, each holding the grid. The data dictionary has no IDS-level grid: `grid`,
        // `grid_type` and the (R, Z) positions all live inside `time_slice(itime)/profiles_2d`
        equilibrium_ids.allocate_time_slices(&times_to_reconstruct);
        for time_slice in equilibrium_ids.time_slice.iter_mut() {
            // GSFit solves on a single rectangular (R, Z) grid, so there is exactly one entry in this array of structures
            let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();

            // `rectangular` is index 1 of the data dictionary's poloidal plane coordinates
            // enumeration: "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)"
            profiles_2d.grid_type.name = "rectangular".to_string();
            profiles_2d.grid_type.index = 1;
            profiles_2d.grid_type.description = "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".to_string();

            profiles_2d.grid.dim1 = r.clone();
            profiles_2d.grid.dim2 = z.clone();
            profiles_2d.grid.d_area = d_area;
            profiles_2d.r = mesh_r.clone();
            profiles_2d.z = mesh_z.clone();
            time_slice.profiles_2d = vec![profiles_2d];

            // The psi_norm grid the source functions are defined on
            time_slice.profiles_1d.psi_norm = psi_norm.clone();

            // The magnetic sensors' weights
            time_slice.constraints.b_field_pol_probe = b_field_pol_probe_constraints.clone();
            time_slice.constraints.flux_loop = flux_loop_constraints.clone();
            time_slice.constraints.rogowski_coil = rogowski_coil_constraints.clone();
            time_slice.constraints.diamagnetic_flux = diamagnetic_flux_constraint.clone();
            time_slice.constraints.isoflux = isoflux_constraints.clone();
            time_slice.constraints.isoflux_boundary = isoflux_boundary_constraints.clone();
            time_slice.constraints.stationary_point = stationary_point_constraints.clone();
        }

        // The Greens tables, starting with those between the grid points. They are geometry, so they are the same for every time-slice
        let greens_tables: GreensTables = GreensTables {
            grid: GreensGrid {
                plasma: greens_grid_plasma,
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            equilibrium_ids,
            settings_equilibrium,
            p_prime_source_function,
            ff_prime_source_function,
            nonlinear_solver,
            greens_tables,
        })
    }

    /// Copy the limiter from the wall IDS into the equilibrium IDS.
    ///
    /// Copies `wall/description_2d(0)/limiter/unit(i)` into `equilibrium/limiter/unit(i)`, in the
    /// same order, so `unit(0)` is still the vacuum vessel contour. The solver and the post-processor
    /// read the limiter from the equilibrium IDS, so this has to be called before
    /// `solve_grad_shafranov`.
    ///
    /// Any limiter already in the equilibrium IDS is replaced, so calling this twice does not repeat
    /// the units.
    ///
    /// # Arguments
    /// * `wall` - The Wall object (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when the wall has no limiter units.
    pub fn copy_from_wall_ids(&mut self, wall: PyRef<Wall>) -> PyResult<()> {
        // `description_2d(0)` always exists, because `Wall::new` creates it. The outline of each unit
        // was checked by `Wall.add_limiter_unit` when the unit was added
        let wall_units: &Vec<Wall2dLimiterUnit> = &wall.wall_ids.description_2d[0].limiter.unit;
        let n_unit: usize = wall_units.len();
        if n_unit == 0 {
            return Err(PyValueError::new_err("`wall/description_2d(0)/limiter/unit` is empty"));
        }

        let mut equilibrium_units: Vec<EquilibriumLimiterUnit> = Vec::with_capacity(n_unit);
        for i_unit in 0..n_unit {
            equilibrium_units.push(EquilibriumLimiterUnit {
                name: wall_units[i_unit].name.clone(),
                outline: EquilibriumLimiterUnitOutline {
                    r: wall_units[i_unit].outline.r.clone(),
                    z: wall_units[i_unit].outline.z.clone(),
                },
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the units instead of
        // repeating them
        self.equilibrium_ids.limiter.unit = equilibrium_units;

        Ok(())
    }

    /// Copy the vacuum toroidal field from the tf IDS into the equilibrium IDS.
    ///
    /// Fills `equilibrium/vacuum_toroidal_field/r0` from `tf/r0`, and `equilibrium/vacuum_toroidal_field/b0`
    /// from `tf/b_field_phi_vacuum_r`. That is on the experimental timebase, so it is interpolated onto the
    /// reconstruction times, which are the times of the time-slices `Equilibrium::new` allocated, and divided by
    /// `r0`. The solver and the post-processor read the vacuum toroidal field from the equilibrium IDS, so
    /// this has to be called before `solve_grad_shafranov`.
    ///
    /// Any vacuum toroidal field already in the equilibrium IDS is replaced.
    ///
    /// # Arguments
    /// * `tf` - The Tf object (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when `tf/b_field_phi_vacuum_r` is unset, or when it cannot be interpolated onto the
    /// reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_tf_ids(&mut self, tf: PyRef<Tf>) -> PyResult<()> {
        // A positive major radius, as `Tf::new` sets it from the settings, which it validates against their XML schema
        let r0: f64 = tf.tf_ids.r0;
        let b_field_phi_vacuum_r_time: &Array1<f64> = &tf.tf_ids.b_field_phi_vacuum_r.time;
        let b_field_phi_vacuum_r_data: &Array1<f64> = &tf.tf_ids.b_field_phi_vacuum_r.data;
        if b_field_phi_vacuum_r_time.is_empty() || b_field_phi_vacuum_r_data.is_empty() {
            return Err(PyValueError::new_err("`tf/b_field_phi_vacuum_r` is unset"));
        }

        // `b_field_phi_vacuum_r` is on the experimental timebase, so it is interpolated onto the times of
        // the time-slices `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let interpolator: interpolation::Dim1Linear =
            interpolation::Dim1Linear::new(b_field_phi_vacuum_r_time.to_owned(), b_field_phi_vacuum_r_data.to_owned())
                .map_err(|error: interpolation::Error| PyValueError::new_err(format!("cannot interpolate `tf/b_field_phi_vacuum_r`: {error:?}")))?;
        let b_field_phi_vacuum_r: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
            PyValueError::new_err(format!("cannot interpolate `tf/b_field_phi_vacuum_r` onto the reconstruction times: {error:?}"))
        })?;

        // The data dictionary requires `vacuum_toroidal_field/r0 * b0` to equal `tf/b_field_phi_vacuum_r`,
        // so both are filled from that one signal. The sign carries through. The rod current is not a data
        // dictionary node, so it is recovered from these two wherever it is needed
        self.equilibrium_ids.vacuum_toroidal_field.r0 = r0;
        self.equilibrium_ids.vacuum_toroidal_field.b0 = b_field_phi_vacuum_r / r0;

        Ok(())
    }

    /// Copy the PF coil currents from the pf_active IDS onto every time-slice of the equilibrium IDS.
    ///
    /// Fills, for every coil in `pf_active/coil(i1)`, in the same order, on every time-slice `Equilibrium::new` allocated:
    /// * `equilibrium/time_slice(itime)/constraints/pf_current(i1)/name` - a custom key: `pf_active/coil(i1)/name`
    /// * `.../measured` - `pf_active/coil(i1)/current`, which is on the experimental timebase, interpolated onto the
    ///   time-slice's time, [ampere]
    /// * `.../source` - `pf_active/coil(i1)/current`, the path the measurement was read from
    /// * `.../time_measurement` - the time-slice's own time, because the current is interpolated rather than taken
    ///   at one of the times it was measured, [second]
    ///
    /// The solver reads the PF coil currents from the equilibrium IDS, so this has to be called before
    /// `solve_grad_shafranov`. The solver multiplies them by the Greens tables, `greens_tables/grid/pf_active(i1)`, so the coils
    /// have to be in the same order as those, which the solver checks by name.
    ///
    /// Any PF coil currents already on a time-slice are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `pf_active` - The PfActive object (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when `pf_active/coil` is empty, or when a coil's current cannot be interpolated onto the
    /// reconstruction times, e.g. because its timebase does not cover them.
    pub fn copy_from_pf_active_ids(&mut self, pf_active: PyRef<PfActive>) -> PyResult<()> {
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();
        if n_coil == 0 {
            return Err(PyValueError::new_err("`pf_active/coil` is empty"));
        }

        // Each coil's current is on the experimental timebase, so it is interpolated onto the times of the
        // time-slices `Equilibrium::new` allocated
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let n_time: usize = times_to_reconstruct.len();
        let mut pf_currents_measured: Array2<f64> = Array2::from_elem((n_time, n_coil), f64::NAN);
        for i_coil in 0..n_coil {
            let coil_name: &str = &pf_active_coils[i_coil].name;
            let interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(pf_active_coils[i_coil].current.time.clone(), pf_active_coils[i_coil].current.data.clone()).map_err(
                    |error: interpolation::Error| {
                        PyValueError::new_err(format!("cannot interpolate `pf_active/coil({i_coil})/current` ({coil_name}): {error:?}"))
                    },
                )?;
            let measured: Array1<f64> = interpolator.interpolate_array1(&times_to_reconstruct).map_err(|error: interpolation::Error| {
                PyValueError::new_err(format!(
                    "cannot interpolate `pf_active/coil({i_coil})/current` ({coil_name}) onto the reconstruction times: {error:?}"
                ))
            })?;
            pf_currents_measured.slice_mut(s![.., i_coil]).assign(&measured);
        }

        // Assigned rather than appended, so that calling this twice replaces the PF coil currents instead of
        // repeating them
        for i_time in 0..n_time {
            let mut pf_currents: Vec<EquilibriumConstraints0dIpLike> = Vec::with_capacity(n_coil);
            for i_coil in 0..n_coil {
                pf_currents.push(EquilibriumConstraints0dIpLike {
                    name: pf_active_coils[i_coil].name.clone(),
                    measured: pf_currents_measured[[i_time, i_coil]],
                    source: format!("pf_active/coil({i_coil})/current"),
                    time_measurement: times_to_reconstruct[i_time],
                    ..Default::default()
                });
            }
            self.equilibrium_ids.time_slice[i_time].constraints.pf_current = pf_currents;
        }

        Ok(())
    }

    /// Make the degrees of freedom the passive conductors' currents are fitted with, from the fit settings and the passive
    /// conductors' filaments, and copy them, with their regularisations, onto every time-slice of the equilibrium IDS.
    ///
    /// The fit settings, `equilibrium_settings/pf_passive_dof/passive`, say which passive conductors to fit, each found in
    /// `pf_passive/loop` by its `name`, and how. Fills, for each of them, in the order of the fit settings, on every time-slice
    /// `Equilibrium::new` allocated:
    /// * `equilibrium/time_slice(itime)/constraints/pf_passive_current(i1)/name` - a custom key: the passive conductor's `name`
    /// * `.../dof(i2)` - a custom key: one for each degree of freedom the conductor's current is fitted with, as its
    ///   `current_distribution_type` says:
    ///   - `constant_current_density`: one, `constant_current_density`, whose `current_distribution` is each element's `area`, so that its
    ///     amplitude is the current density
    ///   - `eig`: the conductor's `n_dof` slowest decaying eigenmodes, `eig_01`, `eig_02`, ..., from its mutual inductances and
    ///     resistances, each with its `current_distribution`, 100 times the eigenvector normalised to a length of 1, and its
    ///     `time_constant`, [second]
    /// * `.../regularisations` - a custom key: from the fit settings, shape = (n_regularisations, n_dof)
    /// * `.../regularisations_weight` - a custom key: from the fit settings, [dimensionless]
    ///
    /// `pf_passive_current(i1)` is where the data dictionary puts the current in a passive loop. GSFit does not measure that current:
    /// it fits it, as an amplitude for each of the loop's degrees of freedom, and the regularisations are what constrain the fit. So
    /// `measured`, `source` and `time_measurement` are left unset.
    ///
    /// `greens_with_pf_passive` reads the degrees of freedom, and the solver the regularisations, from the equilibrium IDS, so this has
    /// to be called before both. The solver fits the amplitudes with the Greens tables, `greens_tables/grid/pf_passive(i1)/dof(i2)`, so the
    /// loops and their degrees of freedom have to be in the same order as those, which the solver checks by name.
    ///
    /// Fit settings with no passive conductors are allowed: the equilibrium then has none. Any passive conductors already on a
    /// time-slice are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `pf_passive` - The PfPassive object, holding the passive conductors' filaments (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when a passive conductor in the fit settings is not in the pf_passive IDS, or is in the fit settings twice, when
    /// one of its elements has no positive `area`, or when its fit settings cannot make its degrees of freedom: an `n_dof` which is not
    /// 1 for `constant_current_density`, or is more than its elements for `eig`, or regularisations which are not of shape
    /// (n_regularisations, n_dof), with one weight each.
    pub fn copy_from_pf_passive_ids(&mut self, pf_passive: PyRef<PfPassive>) -> PyResult<()> {
        // Run the Rust method
        self.copy_from_pf_passive_ids_rs(&pf_passive).map_err(PyValueError::new_err)
    }

    /// Copy the gap definitions from the pulse_schedule IDS onto every time-slice of the equilibrium IDS.
    ///
    /// Fills `equilibrium/time_slice(itime)/boundary/gap(i)/name`, `.../description`, `.../r`, `.../z` and
    /// `.../angle` from `pulse_schedule/position_control/gap(i)`, in the same order, on every time-slice
    /// `Equilibrium::new` allocated. `.../value` is left unset, for the equilibrium post-processor to calculate.
    ///
    /// Every time-slice gets the definitions, including one which does not converge: where the gaps are
    /// measured from is known whether or not the plasma is found, and it keeps the gap array the same
    /// length on every time-slice. Only the value of such a slice is left unset.
    ///
    /// The pulse schedule is optional: without this call, or with a pulse schedule which has no gaps, the
    /// equilibrium has no gaps. Any gaps already on a time-slice are replaced, so calling this twice does
    /// not repeat them.
    ///
    /// # Arguments
    /// * `pulse_schedule` - The PulseSchedule object (a Rust implementation, initialised in Python)
    pub fn copy_from_pulse_schedule_ids(&mut self, pulse_schedule: PyRef<PulseSchedule>) {
        let pulse_schedule_gaps: &Vec<PulseScheduleGap> = &pulse_schedule.pulse_schedule_ids.position_control.gap;
        let n_gap: usize = pulse_schedule_gaps.len();

        let mut equilibrium_gaps: Vec<EquilibriumGap> = Vec::with_capacity(n_gap);
        for i_gap in 0..n_gap {
            equilibrium_gaps.push(EquilibriumGap {
                name: pulse_schedule_gaps[i_gap].name.clone(),
                description: pulse_schedule_gaps[i_gap].description.clone(),
                r: pulse_schedule_gaps[i_gap].r,
                z: pulse_schedule_gaps[i_gap].z,
                // Stored in the equilibrium's convention already; see the `PulseSchedule` documentation
                angle: pulse_schedule_gaps[i_gap].angle,
                // Left unset, for the equilibrium post-processor to calculate
                value: f64::NAN,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the gaps instead of
        // repeating them
        let n_time: usize = self.equilibrium_ids.time_slice.len();
        for i_time in 0..n_time {
            self.equilibrium_ids.time_slice[i_time].boundary.gap = equilibrium_gaps.clone();
        }
    }

    /// Copy the magnetic sensors' measurements from the magnetics IDS onto every time-slice of the equilibrium IDS.
    ///
    /// Only the sensors in the fit settings, `equilibrium_settings/constraints`, are copied, and each of them has to be in the magnetics
    /// IDS. Each measurement is on the experimental timebase, so it is interpolated onto the time-slice's time. On every time-slice
    /// `Equilibrium::new` allocated, for every sensor in the fit settings, in the order of the magnetics IDS:
    /// * `magnetics/b_field_pol_probe(i1)/field` fills `equilibrium/time_slice(itime)/constraints/b_field_pol_probe(i1)`, [tesla]
    /// * `magnetics/flux_loop(i1)/flux` fills `.../constraints/flux_loop(i1)`, [weber]
    /// * `magnetics/rogowski_coil(i1)/current` fills `.../constraints/rogowski_coil(i1)`, a custom key, as the data dictionary has
    ///   no constraint for Rogowski coils, [ampere]
    /// * `magnetics/diamagnetic_flux(0)/data` fills `.../constraints/diamagnetic_flux`, [weber]
    /// * `magnetics/ip(0)/data` fills `.../constraints/ip`, [ampere]
    ///
    /// And the constraints at points which can move, custom keys of both IDSs, each point interpolated onto the time-slice's time; a point
    /// not known at a time-slice, e.g. outside its timebase, is NaN there, and is not fitted:
    /// * `magnetics/isoflux(i1)/position_1` and `/position_2` fill `.../constraints/isoflux(i1)/position_1` and `/position_2`, with
    ///   `measured`, the flux difference between them, zero, [metre]
    /// * `magnetics/isoflux_boundary(i1)/position` fills `.../constraints/isoflux_boundary(i1)/position`, with `measured`, the flux there
    ///   minus that at the boundary, zero, [metre]
    /// * `magnetics/stationary_point(i1)/position` fills `.../constraints/stationary_point(i1)/position`, with the `measured` of
    ///   `d_psi_d_r` and `d_psi_d_z`, zero, [metre]
    ///
    /// Each constraint gets:
    /// * `name` - a custom key: the sensor's `name`, or the `method_name` for `ip`. `diamagnetic_flux` has none, as in the data
    ///   dictionary, because there is only one
    /// * `measured` - the measurement, interpolated onto the time-slice's time
    /// * `source` - the path the measurement was read from, e.g. `magnetics/b_field_pol_probe(i1)/field`
    /// * `time_measurement` - the time-slice's own time, because the measurement is interpolated rather than taken at one of the
    ///   times it was measured, [second]
    ///
    /// `diamagnetic_flux` and `ip` are single constraints. GSFit accepts a single diamagnetic flux, `magnetics/diamagnetic_flux(0)`, so
    /// the signals of several diamagnetic loops are averaged into one before it is added to the magnetics IDS. It is left unset without
    /// fit settings for it. The plasma current, which has no fit settings, is the magnetics IDS's generally recommended method of
    /// calculating it, the first, and is left unset when the magnetics IDS has none.
    ///
    /// Each constraint also keeps what `Equilibrium::new` set from the sensor's fit settings, found by the sensor's name, or for the
    /// diamagnetic flux, from its own: its `weight` and `expected_value`, a custom key.
    ///
    /// Any magnetic constraints already on a time-slice are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `magnetics` - The Magnetics object (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when a sensor in the fit settings is not in the magnetics IDS, when the magnetics IDS holds more than one
    /// diamagnetic flux, or when a measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not
    /// cover them. The diamagnetic flux is the exception: a time-slice it cannot be interpolated onto gets a `weight` of 0 instead.
    pub fn copy_from_magnetics_ids(&mut self, magnetics: PyRef<Magnetics>) -> PyResult<()> {
        // Change Python types into Rust types
        let magnetics_rs: &Magnetics = &magnetics;

        // Each kind of sensor is copied by its own method, in `magnetic_sensors/`
        self.copy_from_magnetics_ids_b_field_pol_probe(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_diamagnetic_flux(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_flux_loop(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_ip(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_rogowski_coil(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_isoflux(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_isoflux_boundary(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_stationary_point(magnetics_rs).map_err(PyValueError::new_err)?;

        Ok(())
    }

    /// Make the pressure constraints from the core_profiles IDS, on every time-slice of the equilibrium IDS.
    ///
    /// When the fit settings, `equilibrium_settings/constraints/pressure/pressure`, include pressure, there is one constraint for each
    /// point of `core_profiles/profiles_2d(itime)`, which must be on the same irregular (R, Z) grid, `irregular_rz_na`, at every time.
    /// `core_profiles/profiles_2d(itime)/pressure_thermal` is on the experimental timebase, so it is interpolated onto the time-slice's
    /// time. On every time-slice `Equilibrium::new` allocated, `equilibrium/time_slice(itime)/constraints/pressure(i1)` gets, for point i1:
    /// * `name` - a custom key: `sensor_name_prefix` followed by the point's two-digit number, from 1, e.g. `PRESSURE01`
    /// * `measured` - the thermal pressure, interpolated onto the time-slice's time; NaN when either of the times it is interpolated
    ///   between is NaN, or the time-slice is outside the times of `core_profiles/profiles_2d`. The solver does not fit a NaN
    /// * `position/r`, `position/z` - the point, `core_profiles/profiles_2d(itime)/grid/dim1(i1)` and `.../dim2(i1)`
    /// * `source` - the path the measurement was read from, `core_profiles/profiles_2d(:)/pressure_thermal(i1,0)`
    /// * `time_measurement` - the time-slice's own time, because the measurement is interpolated rather than taken at one of the times it
    ///   was measured, [second]
    /// * `weight` - the fit settings' `weight`
    /// * `expected_value` - a custom key: the fit settings' `expected_value`, the magnitude the measurement is normalised by, so that
    ///   the solver's least-squares weight is `weight / expected_value`
    ///
    /// When the fit settings do not include pressure, there are no pressure constraints. Any pressure constraints already on a time-slice
    /// are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `core_profiles` - the kinetic profiles, holding `core_profiles/profiles_2d`
    ///
    /// # Errors
    /// A `ValueError` when the fit settings include pressure but have no `sensor_name_prefix`, `weight` or `expected_value`, or when
    /// `core_profiles/profiles_2d` has fewer than two times, is not on one `irregular_rz_na` grid at every time, or has a
    /// `pressure_thermal` without one value per point.
    pub fn copy_from_core_profiles_ids(&mut self, core_profiles: PyRef<CoreProfiles>) -> PyResult<()> {
        let n_time: usize = self.equilibrium_ids.time_slice.len();

        // Without pressure in the fit settings there are no pressure constraints
        let settings_pressure: &SettingsEquilibriumPressure = &self.settings_equilibrium.constraints.pressure;
        if !settings_pressure.include {
            for i_time in 0..n_time {
                self.equilibrium_ids.time_slice[i_time].constraints.pressure = Vec::new();
            }
            return Ok(());
        }
        let (Some(sensor_name_prefix), Some(weight), Some(expected_value)) = (
            settings_pressure.sensor_name_prefix.clone(),
            settings_pressure.weight,
            settings_pressure.expected_value,
        ) else {
            return Err(PyValueError::new_err(
                "`equilibrium_settings/constraints/pressure/pressure` includes pressure, so needs `sensor_name_prefix`, `weight` and `expected_value`",
            ));
        };

        // The points and times of `core_profiles/profiles_2d`, which must be on the same irregular (R, Z) grid at every time
        let profiles_2d: &Vec<CoreProfilesProfiles2d> = &core_profiles.core_profiles_ids.profiles_2d;
        let n_profiles_2d: usize = profiles_2d.len();
        if n_profiles_2d < 2 {
            return Err(PyValueError::new_err(format!(
                "`core_profiles/profiles_2d` has {n_profiles_2d} time(s); at least 2 are needed to interpolate the pressure onto the reconstruction times"
            )));
        }
        let point_r: Array1<f64> = profiles_2d[0].grid.dim1.clone();
        let point_z: Array1<f64> = profiles_2d[0].grid.dim2.clone();
        let n_point: usize = point_r.len();
        let mut profiles_2d_time: Array1<f64> = Array1::from_elem(n_profiles_2d, f64::NAN);
        let mut pressure_thermal: Array2<f64> = Array2::from_elem((n_profiles_2d, n_point), f64::NAN); // shape = (n_profiles_2d, n_point)
        for i_profiles_2d in 0..n_profiles_2d {
            let profiles_2d_this_time: &CoreProfilesProfiles2d = &profiles_2d[i_profiles_2d];
            // `irregular_rz_na` is index 91 of the data dictionary's poloidal plane coordinates enumeration
            if profiles_2d_this_time.grid_type.index != 91 {
                return Err(PyValueError::new_err(format!(
                    "`core_profiles/profiles_2d({i_profiles_2d})/grid_type/index` is {}; the pressure constraints need the irregular (R, Z) grid `irregular_rz_na`, 91",
                    profiles_2d_this_time.grid_type.index
                )));
            }
            if profiles_2d_this_time.grid.dim1 != point_r || profiles_2d_this_time.grid.dim2 != point_z {
                return Err(PyValueError::new_err(format!(
                    "`core_profiles/profiles_2d({i_profiles_2d})/grid` has other points than `core_profiles/profiles_2d(0)/grid`; the pressure constraints need the same points at every time"
                )));
            }
            if profiles_2d_this_time.pressure_thermal.dim() != (n_point, 1) {
                return Err(PyValueError::new_err(format!(
                    "`core_profiles/profiles_2d({i_profiles_2d})/pressure_thermal` has shape {:?}, but the irregular grid has {n_point} point(s), so it needs ({n_point}, 1)",
                    profiles_2d_this_time.pressure_thermal.dim()
                )));
            }
            profiles_2d_time[i_profiles_2d] = profiles_2d_this_time.time;
            pressure_thermal
                .row_mut(i_profiles_2d)
                .assign(&profiles_2d_this_time.pressure_thermal.column(0));
        }

        // Each point's thermal pressure is on the experimental timebase, so it is interpolated onto the times of the time-slices
        // `Equilibrium::new` allocated. A time-slice it cannot be interpolated onto, e.g. one outside the experimental timebase, gets NaN,
        // which the solver does not fit
        let times_to_reconstruct: Array1<f64> = self.equilibrium_ids.time_slice(..).time.to_array();
        let mut pressure_measured: Array2<f64> = Array2::from_elem((n_time, n_point), f64::NAN); // shape = (n_time, n_point)
        for i_point in 0..n_point {
            let interpolator: interpolation::Dim1Linear = interpolation::Dim1Linear::new(profiles_2d_time.clone(), pressure_thermal.column(i_point).to_owned())
                .map_err(|error: interpolation::Error| {
                    PyValueError::new_err(format!(
                        "cannot interpolate `core_profiles/profiles_2d(:)/pressure_thermal({i_point},0)`: {error:?}"
                    ))
                })?;
            for i_time in 0..n_time {
                if let Ok(measured) = interpolator.interpolate_scalar(times_to_reconstruct[i_time]) {
                    pressure_measured[[i_time, i_point]] = measured;
                }
            }
        }

        // Rebuilt on every time-slice, so that calling this twice replaces them instead of repeating them
        for i_time in 0..n_time {
            let mut pressure_constraints: Vec<EquilibriumConstraints0dPosition> = Vec::with_capacity(n_point);
            for i_point in 0..n_point {
                pressure_constraints.push(EquilibriumConstraints0dPosition {
                    name: format!("{sensor_name_prefix}{:02}", i_point + 1),
                    measured: pressure_measured[[i_time, i_point]],
                    position: Rphizpsirho0dDynamicAos3 {
                        r: point_r[i_point],
                        z: point_z[i_point],
                        ..Default::default()
                    },
                    source: format!("core_profiles/profiles_2d(:)/pressure_thermal({i_point},0)"),
                    time_measurement: times_to_reconstruct[i_time],
                    weight,
                    expected_value,
                    ..Default::default()
                });
            }
            self.equilibrium_ids.time_slice[i_time].constraints.pressure = pressure_constraints;
        }

        Ok(())
    }

    /// Calculate the Greens function with the PF coils
    /// The Greens tables are stored within self. Example data structure:
    /// `greens_tables/grid/pf_active(i)/psi`, one for each `pf_active/coil(i)`, in the same order
    ///
    /// # Arguments
    /// * `pf_active` - The PfActive object (a Rust implementation, initialised in Python)
    ///
    fn greens_with_pf_active(&mut self, pf_active: PyRef<PfActive>) {
        // Unpack from self
        let n_r: usize = self.settings_equilibrium.grid.n_r;
        let n_z: usize = self.settings_equilibrium.grid.n_z;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables, one for each PF coil, in the order of `pf_active/coil`, which is
        // the order `Equilibrium.copy_from_pf_active_ids` copies the PF coil currents in
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();
        let mut greens_pf_active: Vec<GreensGridPfActive> = Vec::with_capacity(n_coil);

        for i_coil in 0..n_coil {
            // The coil's filaments
            let coil_r: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.r.to_array();
            let coil_z: Array1<f64> = pf_active_coils[i_coil].element(..).geometry.rectangle.z.to_array();
            let n_coil_filaments: usize = coil_r.len();

            // Greens function for flux
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                flat_r.clone(),
                flat_z.clone(),
                coil_r.clone(),
                coil_z.clone(),
                Array1::from_elem(n_coil_filaments, f64::NAN), // grid should not overlap with coils
                Array1::from_elem(n_coil_filaments, f64::NAN),
            );

            // Greens function for psi, and derivatives
            let g_psi_filaments: Array2<f64> = greens_calculator.psi(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_psi_d_r_filaments: Array2<f64> = greens_calculator.d_psi_d_r(); // shape = (n_z * n_r, n_coil_filaments)
            let g_d_psi_d_z_filaments: Array2<f64> = greens_calculator.d_psi_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_r2_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_r2(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_r_d_z_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_r_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d2_g_d_z2_all_filaments: Array2<f64> = greens_calculator.d2_psi_d_z2(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_r2_d_z_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_r_d_z2_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2(); // shape = (n_z * n_r, n_coil_filaments)
            let d3_g_d_z3_all_filaments: Array2<f64> = greens_calculator.d3_psi_d_z3(); // shape = (n_z * n_r, n_coil_filaments)

            // sum over all filaments and convert into shape = (n_z, n_r)
            let g_psi: Array2<f64> = g_psi_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `g_psi_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_psi_d_r: Array2<f64> = g_d_psi_d_r_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `g_d_psi_d_r_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d_psi_d_z: Array2<f64> = g_d_psi_d_z_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `g_d_psi_d_z_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_r2: Array2<f64> = d2_g_d_r2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d2_g_d_r2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_r_d_z: Array2<f64> = d2_g_d_r_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d2_g_d_r_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d2_psi_d_z2: Array2<f64> = d2_g_d_z2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d2_g_d_z2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_r2_d_z: Array2<f64> = d3_g_d_r2_d_z_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d3_g_d_r2_d_z_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_r_d_z2: Array2<f64> = d3_g_d_r_d_z2_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d3_g_d_r_d_z2_all_filaments` into (n_z, n_r)")
                .to_owned();
            let g_d3_psi_d_z3: Array2<f64> = d3_g_d_z3_all_filaments
                .sum_axis(Axis(1))
                .to_shape((n_z, n_r))
                .expect("Equilibrium.greens_with_pf_active: Failed to reshape `d3_g_d_z3_all_filaments` into (n_z, n_r)")
                .to_owned();

            // Store in the equilibrium IDS
            greens_pf_active.push(GreensGridPfActive {
                name: pf_active_coils[i_coil].name.clone(),
                psi: g_psi,
                d_psi_d_r: g_d_psi_d_r,
                d_psi_d_z: g_d_psi_d_z,
                d2_psi_d_r2: g_d2_psi_d_r2,
                d2_psi_d_r_d_z: g_d2_psi_d_r_d_z,
                d2_psi_d_z2: g_d2_psi_d_z2,
                d3_psi_d_r2_d_z: g_d3_psi_d_r2_d_z,
                d3_psi_d_r_d_z2: g_d3_psi_d_r_d_z2,
                d3_psi_d_z3: g_d3_psi_d_z3,
            });
        }

        self.greens_tables.grid.pf_active = greens_pf_active;
    }

    /// Calculate the Greens function with the passive conductors
    /// The Greens tables are stored within self. Example data structure:
    /// `greens_tables/grid/pf_passive(i)/dof(j)/psi`
    ///
    /// The degrees of freedom are the equilibrium's, `time_slice(itime)/constraints/pf_passive_current(i1)/dof(i2)`, which
    /// `copy_from_pf_passive_ids` makes from the fit settings, so it has to be called first, with the same `pf_passive`.
    ///
    /// The tables are in the order of `constraints/pf_passive_current`, and then of each one's `dof`. This is the order of the passive
    /// degrees of freedom everywhere in the solve: the sensors' Greens are gathered in it, the regularisations are assembled in it, and
    /// so `time_slice(itime)/passive_dof_values` follows it
    ///
    /// # Arguments
    /// * `pf_passive` - The passive conductors, a `PfPassive`, holding their filaments (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when the equilibrium's passive conductors are not the ones in the fit settings, as `copy_from_pf_passive_ids` has
    /// not been called, or when one of them is not in `pf_passive`
    fn greens_with_pf_passive(&mut self, pf_passive: PyRef<PfPassive>) -> PyResult<()> {
        // The passive conductors the equilibrium fits, with their degrees of freedom, which `copy_from_pf_passive_ids` made from the fit
        // settings. `time_slice(0)` because they are the same on every time-slice
        let mut settings_names: Vec<String> = Vec::new();
        for passive_settings in &self.settings_equilibrium.pf_passive_dof.passive {
            settings_names.push(passive_settings.name.clone());
        }
        let pf_passive_current_names: Vec<String> = self.equilibrium_ids.time_slice[0].constraints.pf_passive_current(..).name.to_vec();
        if pf_passive_current_names != settings_names {
            return Err(PyValueError::new_err(format!(
                "the passive conductors in the fit settings, `equilibrium_settings/pf_passive_dof/passive(:)/name`, are {settings_names:?}, but \
                 `time_slice(0)/constraints/pf_passive_current(:)/name` is {pf_passive_current_names:?}; call `Equilibrium.copy_from_pf_passive_ids` first"
            )));
        }
        let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &self.equilibrium_ids.time_slice[0].constraints.pf_passive_current;
        let n_loop: usize = pf_passive_currents.len();

        // Each one's filaments, found in `pf_passive` by its name
        let mut pf_passive_loops: Vec<&PfPassiveLoops> = Vec::with_capacity(n_loop);
        for i_loop in 0..n_loop {
            let i_pf_passive_loop: usize = pf_passive.i_loop_named(&pf_passive_currents[i_loop].name).map_err(PyValueError::new_err)?;
            pf_passive_loops.push(&pf_passive.pf_passive_ids.r#loop[i_pf_passive_loop]);
        }

        // Get variables out of self
        // `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives the
        // flattened grid
        let n_r: usize = self.settings_equilibrium.grid.n_r;
        let n_z: usize = self.settings_equilibrium.grid.n_z;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables, one for each passive conductor the equilibrium fits
        let mut greens_pf_passive: Vec<GreensGridPfPassive> = Vec::with_capacity(n_loop);

        // Calculate Greens with each passive degree of freedom
        for i_loop in 0..n_loop {
            let passive_name: &str = &pf_passive_loops[i_loop].name;
            let filaments: PassiveFilaments = passive_filaments(&pf_passive_loops[i_loop].element);
            let passive_r: Array1<f64> = filaments.r;
            let passive_z: Array1<f64> = filaments.z;

            // Green's tables. These depend only on the grid and this passive's filament geometry,
            // so they are built once per passive; the degree of freedom is applied to them below
            let greens_calculator: Greens = Greens::sensor_to_conductor(
                flat_r.clone(),
                flat_z.clone(),
                passive_r.clone(),
                passive_z.clone(),
                passive_r.clone() * f64::NAN, // d_r=0; as there will not be any points which coincide; using NaN as safety - if we get NaN's we know we have a problem
                passive_z.clone() * f64::NAN, // d_z=0; as there will not be any points which coincide; using NaN as safety - if we get NaN's we know we have a problem
            );

            // Green's functions for `psi`, and derivatives
            let g_psi_filaments: Array2<f64> = greens_calculator.psi(); // shape = [n_r * n_z, n_filament]
            let g_d_psi_d_r_coil_filaments: Array2<f64> = greens_calculator.d_psi_d_r(); // shape = [n_r * n_z, n_filament]
            let g_d_psi_d_z_coil_filaments: Array2<f64> = greens_calculator.d_psi_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_r2_filaments: Array2<f64> = greens_calculator.d2_psi_d_r2(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_r_d_z_filaments: Array2<f64> = greens_calculator.d2_psi_d_r_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d2_psi_d_z2_filaments: Array2<f64> = greens_calculator.d2_psi_d_z2(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_r2_d_z_filaments: Array2<f64> = greens_calculator.d3_psi_d_r2_d_z(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_r_d_z2_filaments: Array2<f64> = greens_calculator.d3_psi_d_r_d_z2(); // shape = [n_r * n_z, n_filament]
            let g_d3_psi_d_z3_filaments: Array2<f64> = greens_calculator.d3_psi_d_z3(); // shape = [n_r * n_z, n_filament]

            // The current distribution belonging to each degree of freedom, collected up-front so that
            // all of them can be applied to a Green's table in a single pass over it
            let n_dof: usize = pf_passive_currents[i_loop].dof.len();
            let mut current_distributions: Vec<Array1<f64>> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                current_distributions.push(pf_passive_currents[i_loop].dof[i_dof].current_distribution.clone());
            }

            // Apply the current distributions and sum over the filaments; one column per degree of freedom
            let g_psi_with_dof: Array2<f64> = apply_current_distributions(&g_psi_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_psi_d_r_with_dof: Array2<f64> = apply_current_distributions(&g_d_psi_d_r_coil_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d_psi_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d_psi_d_z_coil_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_r2_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_r2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_r_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_r_d_z_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d2_psi_d_z2_with_dof: Array2<f64> = apply_current_distributions(&g_d2_psi_d_z2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_r2_d_z_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_r2_d_z_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_r_d_z2_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_r_d_z2_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]
            let g_d3_psi_d_z3_with_dof: Array2<f64> = apply_current_distributions(&g_d3_psi_d_z3_filaments, &current_distributions); // shape = [n_r * n_z, n_dof]

            // One table over the grid, shape = (n_z, n_r), for each degree of freedom
            let mut greens_dof: Vec<GreensGridPfPassiveDof> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                greens_dof.push(GreensGridPfPassiveDof {
                    name: pf_passive_currents[i_loop].dof[i_dof].name.clone(),
                    psi: g_psi_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_psi_with_dof` into (n_z, n_r)"),
                    d_psi_d_r: g_d_psi_d_r_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d_psi_d_r_with_dof` into (n_z, n_r)"),
                    d_psi_d_z: g_d_psi_d_z_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d_psi_d_z_with_dof` into (n_z, n_r)"),
                    d2_psi_d_r2: g_d2_psi_d_r2_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d2_psi_d_r2_with_dof` into (n_z, n_r)"),
                    d2_psi_d_r_d_z: g_d2_psi_d_r_d_z_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d2_psi_d_r_d_z_with_dof` into (n_z, n_r)"),
                    d2_psi_d_z2: g_d2_psi_d_z2_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d2_psi_d_z2_with_dof` into (n_z, n_r)"),
                    d3_psi_d_r2_d_z: g_d3_psi_d_r2_d_z_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d3_psi_d_r2_d_z_with_dof` into (n_z, n_r)"),
                    d3_psi_d_r_d_z2: g_d3_psi_d_r_d_z2_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d3_psi_d_r_d_z2_with_dof` into (n_z, n_r)"),
                    d3_psi_d_z3: g_d3_psi_d_z3_with_dof
                        .column(i_dof)
                        .to_owned()
                        .into_shape_with_order((n_z, n_r))
                        .expect("Equilibrium.greens_with_pf_passive: failed to reshape `g_d3_psi_d_z3_with_dof` into (n_z, n_r)"),
                });
            }

            greens_pf_passive.push(GreensGridPfPassive {
                name: passive_name.to_string(),
                dof: greens_dof,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the tables instead of
        // silently doubling them up
        self.greens_tables.grid.pf_passive = greens_pf_passive;

        Ok(())
    }

    /// Calculate the magnetic sensors' Greens tables: what each sensor measures, per ampere in each current source.
    ///
    /// Fills, for every sensor in the fit settings, `equilibrium_settings/constraints`, which are the sensors `copy_from_magnetics_ids`
    /// copies, with the same name and in the order of the magnetics IDS, `greens_tables/b_field_pol_probe(i1)`, `greens_tables/flux_loop(i1)` and
    /// `greens_tables/rogowski_coil(i1)`, each with the sensor's tables with:
    /// * `plasma` - the plasma current in each grid cell, shape = (n_z, n_r)
    /// * `d_plasma_d_z` - the same, differentiated with respect to the sensor's height, shape = (n_z, n_r)
    /// * `pf_active` - each PF coil, in the order of `greens_tables/grid/pf_active`, shape = (n_coil)
    /// * `pf_passive` - each passive degree of freedom, in the order of `greens_tables/grid/pf_passive(:)/dof(:)`, shape = (n_dof_total)
    ///
    /// And for every constraint at points which can move in the fit settings, `greens_tables/isoflux(i1)`, `greens_tables/isoflux_boundary(i1)`
    /// and `greens_tables/stationary_point(i1)`, with the same tables, but one for each time-slice, as their points can move: `plasma` and
    /// `d_plasma_d_z` of shape = (n_time, n_z, n_r), `pf_active` of shape = (n_time, n_coil) and `pf_passive` of shape = (n_time, n_dof_total),
    /// NaN where the points are not known. Their points are interpolated onto the time-slices' times as `copy_from_magnetics_ids` interpolates
    /// them, so the two can be called in either order. A stationary point has the tables of both derivatives of the poloidal flux,
    /// `d_psi_d_r` and `d_psi_d_z`.
    ///
    /// `greens_tables/grid/pf_active` and `greens_tables/grid/pf_passive` name the PF coils and the passive degrees of freedom, so
    /// `Equilibrium.greens_with_pf_active` and `Equilibrium.greens_with_pf_passive` have to be called first, with the same `pf_active` and
    /// `pf_passive`. The passive degrees of freedom are the equilibrium's, `time_slice(itime)/constraints/pf_passive_current(:)/dof(:)`,
    /// which `copy_from_pf_passive_ids` makes.
    ///
    /// Any sensors' Greens tables already in `greens_tables` are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `magnetics` - The Magnetics object, holding where the sensors are (a Rust implementation, initialised in Python)
    /// * `pf_active` - The PfActive object, holding the PF coils' filaments (a Rust implementation, initialised in Python)
    /// * `pf_passive` - The PfPassive object, holding the passive conductors' filaments (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when a sensor in the fit settings is not in the magnetics IDS, when the PF coils, or the equilibrium's passive
    /// degrees of freedom, are not the ones `greens_tables/grid/pf_active` and `greens_tables/grid/pf_passive` were calculated with, in the same order,
    /// when a passive conductor the equilibrium fits is not in `pf_passive`, or when the points of a constraint at points which can move
    /// cannot be interpolated onto the time-slices' times.
    fn greens_with_magnetics(&mut self, magnetics: PyRef<Magnetics>, pf_active: PyRef<PfActive>, pf_passive: PyRef<PfPassive>) -> PyResult<()> {
        // Change Python types into Rust types
        let magnetics_rs: &Magnetics = &magnetics;
        let pf_active_rs: &PfActive = &pf_active;
        let pf_passive_rs: &PfPassive = &pf_passive;

        // Every sensor's `pf_active` is in the order of `greens_tables/grid/pf_active`, which names the PF coils, so the coils have to be
        // the ones those tables were calculated with
        let pf_active_names: Vec<String> = pf_active_rs.pf_active_ids.coil(..).name.to_vec();
        let greens_pf_active_names: Vec<String> = self.greens_tables.grid.pf_active(..).name.to_vec();
        if pf_active_names != greens_pf_active_names {
            return Err(PyValueError::new_err(format!(
                "`pf_active/coil(:)/name` is {pf_active_names:?}, but the Greens tables, `greens_tables/grid/pf_active(:)/name`, are for \
                 {greens_pf_active_names:?}; call `Equilibrium.greens_with_pf_active` with these PF coils first"
            )));
        }

        // Likewise, every sensor's `pf_passive` is in the order of `greens_tables/grid/pf_passive(:)/dof(:)`, which has to be the equilibrium's
        // passive degrees of freedom, which `copy_from_pf_passive_ids` made. `time_slice(0)` because they are the same on every time-slice
        let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &self.equilibrium_ids.time_slice[0].constraints.pf_passive_current;
        let n_loop: usize = pf_passive_currents.len();
        let mut pf_passive_dof_names: Vec<(String, String)> = Vec::new();
        for i_loop in 0..n_loop {
            let n_dof: usize = pf_passive_currents[i_loop].dof.len();
            for i_dof in 0..n_dof {
                pf_passive_dof_names.push((pf_passive_currents[i_loop].name.clone(), pf_passive_currents[i_loop].dof[i_dof].name.clone()));
            }
            // Each passive conductor has to be in `pf_passive`, which holds its filaments
            if let Err(error) = pf_passive_rs.i_loop_named(&pf_passive_currents[i_loop].name) {
                return Err(PyValueError::new_err(error));
            }
        }
        let n_greens_pf_passive: usize = self.greens_tables.grid.pf_passive.len();
        let mut greens_pf_passive_dof_names: Vec<(String, String)> = Vec::new();
        for i_greens_pf_passive in 0..n_greens_pf_passive {
            let n_dof: usize = self.greens_tables.grid.pf_passive[i_greens_pf_passive].dof.len();
            for i_dof in 0..n_dof {
                greens_pf_passive_dof_names.push((
                    self.greens_tables.grid.pf_passive[i_greens_pf_passive].name.clone(),
                    self.greens_tables.grid.pf_passive[i_greens_pf_passive].dof[i_dof].name.clone(),
                ));
            }
        }
        if pf_passive_dof_names != greens_pf_passive_dof_names {
            return Err(PyValueError::new_err(format!(
                "the passive degrees of freedom, `time_slice(0)/constraints/pf_passive_current(:)/dof(:)/name`, are {pf_passive_dof_names:?}, but \
                 the Greens tables, `greens_tables/grid/pf_passive(:)/dof(:)/name`, are for {greens_pf_passive_dof_names:?}; call \
                 `Equilibrium.greens_with_pf_passive` after `Equilibrium.copy_from_pf_passive_ids`"
            )));
        }

        // Only the sensors in the fit settings have Greens tables, and each of them has to be in the magnetics IDS
        let constraints_settings: &SettingsEquilibriumEquilibriumConstraints = &self.settings_equilibrium.constraints;
        let mut b_field_pol_probe_settings_names: Vec<String> = Vec::new();
        for b_field_pol_probe_settings in &constraints_settings.b_field_pol_probe {
            b_field_pol_probe_settings_names.push(b_field_pol_probe_settings.name.clone());
        }
        let mut flux_loop_settings_names: Vec<String> = Vec::new();
        for flux_loop_settings in &constraints_settings.flux_loop {
            flux_loop_settings_names.push(flux_loop_settings.name.clone());
        }
        let mut rogowski_coil_settings_names: Vec<String> = Vec::new();
        for rogowski_coil_settings in &constraints_settings.rogowski_coil {
            rogowski_coil_settings_names.push(rogowski_coil_settings.name.clone());
        }
        for (sensor, settings_names, magnetics_names) in [
            (
                "b_field_pol_probe",
                b_field_pol_probe_settings_names,
                magnetics_rs.magnetics_ids.b_field_pol_probe(..).name.to_vec(),
            ),
            ("flux_loop", flux_loop_settings_names, magnetics_rs.magnetics_ids.flux_loop(..).name.to_vec()),
            (
                "rogowski_coil",
                rogowski_coil_settings_names,
                magnetics_rs.magnetics_ids.rogowski_coil(..).name.to_vec(),
            ),
        ] {
            for settings_name in &settings_names {
                if !magnetics_names.contains(settings_name) {
                    return Err(PyValueError::new_err(format!(
                        "`{settings_name}` is in the fit settings, `equilibrium_settings/constraints/{sensor}`, but not in the magnetics IDS, `magnetics/{sensor}`"
                    )));
                }
            }
        }

        // Each kind of sensor is calculated by its own method, in `magnetic_sensors/`
        self.greens_with_magnetics_b_field_pol_probe(magnetics_rs, pf_active_rs, pf_passive_rs);
        self.greens_with_magnetics_flux_loop(magnetics_rs, pf_active_rs, pf_passive_rs);
        self.greens_with_magnetics_rogowski_coil(magnetics_rs, pf_active_rs, pf_passive_rs);
        self.greens_with_magnetics_isoflux(magnetics_rs, pf_active_rs, pf_passive_rs)
            .map_err(PyValueError::new_err)?;
        self.greens_with_magnetics_isoflux_boundary(magnetics_rs, pf_active_rs, pf_passive_rs)
            .map_err(PyValueError::new_err)?;
        self.greens_with_magnetics_stationary_point(magnetics_rs, pf_active_rs, pf_passive_rs)
            .map_err(PyValueError::new_err)?;

        Ok(())
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.equilibrium_paths`, straight out of the equilibrium IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached by its data dictionary path,
    /// e.g. `equilibrium_paths.time_slice[0].global_quantities.ip`. A path holds no data, so the IDS is only borrowed for the read,
    /// never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.equilibrium_ids, "equilibrium", path)
    }

    /// A copy of the whole equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`.
    ///
    /// Read the data with `get` instead. This copies everything on every access. It is for when a detached snapshot is wanted:
    /// changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        PyEquilibrium::new(self.equilibrium_ids.clone())
    }

    /// A copy of the Greens tables, for reading with `gsfit_rs.imas.greens_tables_paths`.
    ///
    /// This copies every table on every access, so take it once and read it with `get` as often as needed. It is a detached
    /// snapshot: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn greens_tables(&self) -> PyGreensTables {
        PyGreensTables::new(self.greens_tables.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Equilibrium>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_characters: usize = self.equilibrium_ids.code.parameters.chars().count();
        string_output += &format!("║  {:<74} ║\n", format!(" code/parameters: {n_characters} characters of XML"));

        let n_r: usize = self.settings_equilibrium.grid.n_r;
        let n_z: usize = self.settings_equilibrium.grid.n_z;
        string_output += &format!("║  {:<74} ║\n", format!(" grid: n_r = {n_r}, n_z = {n_z}"));

        let p_prime_n_dof: usize = self.p_prime_source_function.source_function_n_dof();
        let ff_prime_n_dof: usize = self.ff_prime_source_function.source_function_n_dof();
        string_output += &format!(
            "║  {:<74} ║\n",
            format!(" source functions: p' with {p_prime_n_dof} and ff' with {ff_prime_n_dof} degrees of freedom")
        );

        let n_time: usize = self.equilibrium_ids.time_slice.len();
        if n_time == 0 {
            string_output += &format!("║  {:<74} ║\n", " time_slice: none");
        } else {
            let time_first: f64 = self.equilibrium_ids.time_slice[0].time;
            let time_last: f64 = self.equilibrium_ids.time_slice[n_time - 1].time;
            string_output += &format!("║  {:<74} ║\n", format!(" time_slice: {n_time}, from {time_first} to {time_last} second"));
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `equilibrium_ids` and `greens_tables`, the `dict`s of their nodes, see
    /// `imas_rs::python::PickleState`. `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_equilibrium`, are not pickled: they are read again from the XML which `equilibrium_ids` holds in
    /// `code/parameters`, which they were read from, and the source functions and the nonlinear solver are made from them again,
    /// as `new` makes them.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("equilibrium_ids", self.equilibrium_ids.to_pickle_state(py)?)?;
        state.set_item("greens_tables", self.greens_tables.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_equilibrium`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Equilibrium` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `Equilibrium` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let equilibrium_ids_state: Bound<'_, PyAny> = state
            .get_item("equilibrium_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Equilibrium` has no `equilibrium_ids`"))?;
        let equilibrium_ids: EquilibriumIds =
            PickleState::from_pickle_state(&equilibrium_ids_state).map_err(|error: PickleStateError| error.into_py_err("`Equilibrium.equilibrium_ids`"))?;
        let greens_tables_state: Bound<'_, PyAny> = state
            .get_item("greens_tables")?
            .ok_or_else(|| PyValueError::new_err("The pickled `Equilibrium` has no `greens_tables`"))?;
        let greens_tables: GreensTables =
            PickleState::from_pickle_state(&greens_tables_state).map_err(|error: PickleStateError| error.into_py_err("`Equilibrium.greens_tables`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_equilibrium: SettingsEquilibrium = SettingsEquilibrium::from_xml(&equilibrium_ids.code.parameters).map_err(|error: String| {
            PyValueError::new_err(format!(
                "The pickled `Equilibrium`'s settings cannot be read into `SettingsEquilibrium`: {error}"
            ))
        })?;
        let nonlinear_solver: SharedNonlinearSolver = nonlinear_solver_from_settings(&settings_equilibrium.numerics)?;
        let p_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &settings_equilibrium.source_function_p_prime.method,
            settings_equilibrium.source_function_p_prime.efit_polynomial.as_ref(),
            settings_equilibrium.source_function_p_prime.tensioned_cubic_b_spline.as_ref(),
            "equilibrium_settings/source_function_p_prime",
        )?;
        let ff_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &settings_equilibrium.source_function_ff_prime.method,
            settings_equilibrium.source_function_ff_prime.efit_polynomial.as_ref(),
            settings_equilibrium.source_function_ff_prime.tensioned_cubic_b_spline.as_ref(),
            "equilibrium_settings/source_function_ff_prime",
        )?;

        Ok(Self {
            equilibrium_ids,
            settings_equilibrium,
            p_prime_source_function,
            ff_prime_source_function,
            nonlinear_solver,
            greens_tables,
        })
    }
}

/// Rust only methods
impl Equilibrium {
    /// Make the passive conductors' degrees of freedom, and copy them, with their regularisations, onto every time-slice. This does
    /// the work of the Python method `copy_from_pf_passive_ids`, so that Rust code can call it too; see it for what is filled.
    ///
    /// # Errors
    /// Why the degrees of freedom cannot be made; see `copy_from_pf_passive_ids`
    pub fn copy_from_pf_passive_ids_rs(&mut self, pf_passive: &PfPassive) -> Result<(), String> {
        let passives_settings: &Vec<SettingsEquilibriumPassive> = &self.settings_equilibrium.pf_passive_dof.passive;
        let n_passive: usize = passives_settings.len();

        let mut pf_passive_currents: Vec<EquilibriumConstraints0d> = Vec::with_capacity(n_passive);
        for i_passive in 0..n_passive {
            let passive_settings: &SettingsEquilibriumPassive = &passives_settings[i_passive];
            let passive_name: &str = &passive_settings.name;
            let settings_path: String = format!("equilibrium_settings/pf_passive_dof/passive({i_passive})");

            // A passive conductor is fitted with one set of degrees of freedom
            for j_passive in 0..i_passive {
                if passives_settings[j_passive].name == passive_settings.name {
                    return Err(format!(
                        "`{passive_name}` is in the fit settings twice, at `equilibrium_settings/pf_passive_dof/passive({j_passive})` and `{settings_path}`"
                    ));
                }
            }

            // Each passive conductor in the fit settings has to be in the pf_passive IDS, which holds its filaments
            let i_loop: usize = pf_passive
                .i_loop_named(passive_name)
                .map_err(|_| format!("`{passive_name}` is in the fit settings, `{settings_path}/name`, but not in the pf_passive IDS, `pf_passive/loop`"))?;
            let pf_passive_loop: &PfPassiveLoops = &pf_passive.pf_passive_ids.r#loop[i_loop];
            let n_element: usize = pf_passive_loop.element.len();

            // The regularisations: one row of `n_dof` coefficients for each. No regularisations is a matrix with no rows, which the settings
            // read with no columns either
            let n_dof: usize = passive_settings.n_dof;
            let n_regularisation: usize = passive_settings.regularisations.nrows();
            let regularisations: Array2<f64> = if n_regularisation == 0 {
                Array2::zeros((0, n_dof))
            } else {
                passive_settings.regularisations.clone()
            };
            if regularisations.ncols() != n_dof || passive_settings.regularisations_weight.len() != n_regularisation {
                return Err(format!(
                    "`{settings_path}` ({passive_name}): `regularisations` has shape {:?} and `regularisations_weight` has {} value(s); they need \
                     shape (n_regularisations, n_dof) = (n_regularisations, {n_dof}) and n_regularisations values",
                    regularisations.shape(),
                    passive_settings.regularisations_weight.len()
                ));
            }

            // The area of each element, which `PfPassive.add_loop` calculates from the element's geometry
            let mut area: Array1<f64> = Array1::from_elem(n_element, f64::NAN);
            for i_element in 0..n_element {
                area[i_element] = pf_passive_loop.element[i_element].area;
                if !(area[i_element] > 0.0 && area[i_element].is_finite()) {
                    return Err(format!(
                        "`pf_passive/loop({i_loop})/element({i_element})/area` ({passive_name}) is {}, but the degrees of freedom need it positive",
                        area[i_element]
                    ));
                }
            }

            // The degrees of freedom
            let current_distribution_type: &str = &passive_settings.current_distribution_type;
            let dofs: Vec<EquilibriumConstraints0dDof> = if current_distribution_type == "constant_current_density" {
                if n_dof != 1 {
                    return Err(format!(
                        "`{settings_path}` ({passive_name}): `current_distribution_type` is constant_current_density, which has 1 degree of freedom, \
                         but `n_dof` is {n_dof}"
                    ));
                }

                // The current in each element is its area, so that the degree of freedom is the current density
                vec![EquilibriumConstraints0dDof {
                    name: "constant_current_density".to_string(),
                    current_distribution: Array1::ones(n_element) * &area,
                    time_constant: f64::NAN,
                }]
            } else if current_distribution_type == "eig" {
                if n_dof > n_element {
                    return Err(format!(
                        "`{settings_path}` ({passive_name}): `n_dof` is {n_dof}, but `pf_passive/loop({i_loop})` has only {n_element} elements, and so \
                         only {n_element} eigenmodes"
                    ));
                }

                pf_passive_eigenmodes(pf_passive_loop, &area, n_dof).map_err(|error: String| format!("`{settings_path}` ({passive_name}): {error}"))?
            } else {
                return Err(format!(
                    "`{settings_path}/current_distribution_type` = \"{current_distribution_type}\" is not recognized"
                ));
            };

            pf_passive_currents.push(EquilibriumConstraints0d {
                name: passive_name.to_string(),
                dof: dofs,
                regularisations,
                regularisations_weight: passive_settings.regularisations_weight.clone(),
                ..Default::default()
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the passive conductors instead of
        // repeating them
        let n_time: usize = self.equilibrium_ids.time_slice.len();
        for i_time in 0..n_time {
            self.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current = pf_passive_currents.clone();
        }

        Ok(())
    }
}

/// The `n_dof` slowest decaying eigenmodes of the current in a passive conductor, as its degrees of freedom.
///
/// The eigenmodes are the solutions of `M x = tau R x`, with `M` the mutual inductances between the conductor's elements, `R` their
/// resistances, and `tau` the mode's time constant. This is solved as a symmetric eigenproblem, through the Cholesky factorisation of `R`.
///
/// # Arguments
/// * `pf_passive_loop` - the passive conductor, `pf_passive/loop(i1)`, one element per filament
/// * `area` - the area of each element, [metre ** 2]
/// * `n_dof` - how many eigenmodes, at most the number of elements
///
/// # Returns
/// The degrees of freedom `eig_01`, `eig_02`, ..., slowest decaying first, each with its current distribution, 100 times its
/// eigenvector normalised to a length of 1, and its time constant, [second]
///
/// # Errors
/// The Cholesky factorisation of the resistance matrix, or the eigendecomposition, fails
fn pf_passive_eigenmodes(pf_passive_loop: &PfPassiveLoops, area: &Array1<f64>, n_dof: usize) -> Result<Vec<EquilibriumConstraints0dDof>, String> {
    let filaments: PassiveFilaments = passive_filaments(&pf_passive_loop.element);
    let n_element: usize = pf_passive_loop.element.len();

    // Mutual inductance matrix
    let mutual_inductance_matrix: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
        &filaments.r,
        &filaments.z,
        &filaments.d_r,
        &filaments.d_z,
        &filaments.angle_1,
        &filaments.angle_2,
        &filaments.r,
        &filaments.z,
        &filaments.d_r,
        &filaments.d_z,
        &filaments.angle_1,
        &filaments.angle_2,
    );

    // Resistance matrix
    let length: Array1<f64> = 2.0 * PI * &filaments.r;
    let resistance_matrix: Array2<f64> = Array2::eye(n_element) * pf_passive_loop.resistivity * length / area;

    // Generalized symmetric eigenproblem: A x = λ B x
    // where A = mutual_inductance_matrix, B = resistance_matrix
    // Solved via: L L^T = B (Cholesky), then standard eigenproblem on L^{-1} A L^{-T}

    // Convert ndarray matrices to faer matrices
    let n: usize = n_element;
    let a_faer: faer::Mat<f64> = faer::Mat::from_fn(n, n, |i, j| mutual_inductance_matrix[[i, j]]);
    let b_faer: faer::Mat<f64> = faer::Mat::from_fn(n, n, |i, j| resistance_matrix[[i, j]]);

    // Cholesky factorization of B: B = L L^T
    let b_llt: faer::linalg::solvers::Llt<f64> = b_faer
        .llt(Side::Lower)
        .map_err(|error| format!("the Cholesky factorisation of the resistance matrix failed: {error:?}"))?;
    let l_factor: faer::MatRef<'_, f64> = b_llt.L();

    // Form C = L^{-1} A L^{-T} for the standard symmetric eigenproblem
    // Step 1: solve L Y = A  (forward substitution, Y = L^{-1} A)
    let mut y: faer::Mat<f64> = a_faer.clone();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l_factor, y.as_mut(), faer::Par::Seq);
    // Step 2: C = Y L^{-T}, equivalent to solving L^T C^T = Y^T
    let mut c: faer::Mat<f64> = y.transpose().to_owned();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l_factor, c.as_mut(), faer::Par::Seq);
    let c: faer::Mat<f64> = c.transpose().to_owned();

    // Symmetric eigendecomposition of C: eigenvalues in nondecreasing order
    let eigen: faer::linalg::solvers::SelfAdjointEigen<f64> = c
        .self_adjoint_eigen(Side::Lower)
        .map_err(|error| format!("the symmetric eigendecomposition failed: {error:?}"))?;
    let eigenvalues_faer: faer::Col<f64> = eigen.S().column_vector().to_owned();
    let eigenvectors_c: faer::Mat<f64> = eigen.U().to_owned();

    // Back-transform eigenvectors: x = L^{-T} u
    // Solve L^T X = U  (back substitution)
    let mut eigenvectors_faer: faer::Mat<f64> = eigenvectors_c;
    faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_factor.transpose(), eigenvectors_faer.as_mut(), faer::Par::Seq);

    // Eigenvalues are already sorted nondecreasing (smallest to largest)
    let mut eigenvalues: Array1<f64> = Array1::from_elem(n, f64::NAN);
    for i_eig in 0..n {
        eigenvalues[i_eig] = eigenvalues_faer[i_eig];
    }

    // Normalize eigenvectors using "2-norm" and store in ndarray (row-major: row i = eigenvector i)
    // This will panic if the eigenvector is itself zero, which should never happen, so let's keep the panic in place as it's the correct behaviour
    let mut eigenvectors: Array2<f64> = Array2::from_elem((n, n), f64::NAN);
    for i_eig in 0..n {
        let eigenvector_col: faer::ColRef<'_, f64> = eigenvectors_faer.col(i_eig);
        let norm: f64 = eigenvector_col.norm_l2();
        for j_elem in 0..n {
            eigenvectors[[i_eig, j_elem]] = eigenvector_col[j_elem] / norm;
        }
    }

    // The slowest decaying eigenmodes, which have the largest eigenvalues, first
    let mut dofs: Vec<EquilibriumConstraints0dDof> = Vec::with_capacity(n_dof);
    for i_eig in 0..n_dof {
        dofs.push(EquilibriumConstraints0dDof {
            name: format!("eig_{:02}", i_eig + 1),
            current_distribution: eigenvectors.slice(s![n_element - i_eig - 1, ..]).to_owned() * 100.0,
            time_constant: eigenvalues[n_element - i_eig - 1],
        });
    }

    Ok(dofs)
}

/// Make a source function, p' or ff', from its settings, as `make_source_function` in the database reader's `setup_plasma.py` does.
///
/// # Arguments
/// * `method` - the source function's `method`: "efit_polynomial" or "tensioned_cubic_b_spline"
/// * `efit_polynomial` - the source function's `efit_polynomial` settings, if they are given, used when `method` is "efit_polynomial"
/// * `tensioned_cubic_b_spline` - the source function's `tensioned_cubic_b_spline` settings, if they are given, used when `method` is
///   "tensioned_cubic_b_spline"
/// * `path` - the path of the source function's settings, e.g. `equilibrium_settings/source_function_p_prime`, for the errors
///
/// # Errors
/// `ValueError` if `method` is not a source function, its settings are left out, or they cannot make a source function
fn source_function_from_settings(
    method: &str,
    efit_polynomial: Option<&SettingsEquilibriumEfitPolynomial>,
    tensioned_cubic_b_spline: Option<&SettingsEquilibriumTensionedCubicBSpline>,
    path: &str,
) -> PyResult<SharedSourceFunction> {
    match method {
        "efit_polynomial" => {
            let efit_polynomial: &SettingsEquilibriumEfitPolynomial = efit_polynomial.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "`{path}/method` is \"efit_polynomial\", but `{path}/efit_polynomial` is not in the settings"
                ))
            })?;
            let n_dof: usize = efit_polynomial.n_dof;

            // No regularisations is a matrix with no rows, which the settings give as shape (0, 0).
            // The source function takes shape (n_regularisations, n_dof), so it needs shape (0, n_dof)
            let regularisations: Array2<f64> = if efit_polynomial.regularisations.nrows() == 0 {
                Array2::zeros((0, n_dof))
            } else {
                efit_polynomial.regularisations.clone()
            };
            if regularisations.ncols() != n_dof {
                return Err(PyValueError::new_err(format!(
                    "`{path}/efit_polynomial/regularisations` has {} columns, but there are n_dof = {n_dof} degrees of freedom",
                    regularisations.ncols()
                )));
            }

            // `exact` fixes the coefficients to `coefficients`, rather than fitting them. Both are left out when the coefficients
            // are fitted
            let exact: bool = efit_polynomial.exact.unwrap_or(false);
            let coefficients: Array1<f64> = efit_polynomial.coefficients.clone().unwrap_or_else(|| Array1::zeros(0));
            let source_function: EfitPolynomial = EfitPolynomial::new(n_dof, regularisations, exact, coefficients)
                .map_err(|error| PyValueError::new_err(format!("`{path}/efit_polynomial`: {error}")))?;

            Ok(Arc::new(source_function))
        }
        "tensioned_cubic_b_spline" => {
            let tensioned_cubic_b_spline: &SettingsEquilibriumTensionedCubicBSpline = tensioned_cubic_b_spline.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "`{path}/method` is \"tensioned_cubic_b_spline\", but `{path}/tensioned_cubic_b_spline` is not in the settings"
                ))
            })?;
            let interior_knots: &Array1<f64> = &tensioned_cubic_b_spline.interior_knots;
            let interval_tensions: &Array1<f64> = &tensioned_cubic_b_spline.interval_tensions;
            // There is one tension per interval between the knots, one more than the interior knots
            if interval_tensions.len() != interior_knots.len() + 1 {
                return Err(PyValueError::new_err(format!(
                    "`{path}/tensioned_cubic_b_spline/interval_tensions` has {} values, but there are {} intervals between the knots: one more than the {} interior knots",
                    interval_tensions.len(),
                    interior_knots.len() + 1,
                    interior_knots.len()
                )));
            }
            let n_dof: usize = interior_knots.len() + 4;

            // The regularisations are either built from `regularisation_builder` or given in `regularisations`.
            // `make_cubic_spline_regularisations` is false when left out
            let regularisations: Array2<f64> = if tensioned_cubic_b_spline.make_cubic_spline_regularisations.unwrap_or(false) {
                let regularisation_builder: &SettingsEquilibriumRegularisationBuilder =
                    tensioned_cubic_b_spline.regularisation_builder.as_ref().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "`{path}/tensioned_cubic_b_spline/make_cubic_spline_regularisations` is true, but `{path}/tensioned_cubic_b_spline/regularisation_builder` is not in the settings"
                        ))
                    })?;
                TensionedCubicBSpline::make_regularisations(interior_knots, interval_tensions, regularisation_builder)
                    .map_err(|error| PyValueError::new_err(format!("`{path}/tensioned_cubic_b_spline/regularisation_builder`: {error}")))?
            } else {
                match &tensioned_cubic_b_spline.regularisations {
                    Some(regularisations) if regularisations.nrows() > 0 => regularisations.clone(),
                    // No regularisations is a matrix with no rows, which the settings give as shape (0, 0), or leave out.
                    // The source function takes shape (n_regularisations, n_dof), so it needs shape (0, n_dof)
                    _ => Array2::zeros((0, n_dof)),
                }
            };
            // The source function takes its number of degrees of freedom from the regularisations
            if regularisations.ncols() != n_dof {
                return Err(PyValueError::new_err(format!(
                    "`{path}/tensioned_cubic_b_spline/regularisations` has {} columns, but there are n_dof = (number of interior knots) + 4 = {n_dof} degrees of freedom",
                    regularisations.ncols()
                )));
            }

            Ok(Arc::new(TensionedCubicBSpline::new(
                regularisations,
                interior_knots.clone(),
                interval_tensions.clone(),
            )))
        }
        _ => Err(PyValueError::new_err(format!("`{path}/method` = \"{method}\" is not recognized"))),
    }
}

/// Make the nonlinear solver chosen by `numerics/nonlinear_solver/method`, with its settings.
///
/// The settings it reads are not checked here: the XML schema they were validated against has each count at least 1 and each
/// tolerance, step and mixing fraction positive and finite.
///
/// # Arguments
/// * `settings_numerics` - `equilibrium_settings/numerics`
///
/// # Errors
/// `ValueError` if `method` is not a nonlinear solver
fn nonlinear_solver_from_settings(settings_numerics: &SettingsEquilibriumNumerics) -> PyResult<SharedNonlinearSolver> {
    // The settings every nonlinear solver reads
    let n_iter_max: usize = settings_numerics.n_iter_max;
    let n_iter_min: usize = settings_numerics.n_iter_min;
    let grad_shafranov_deviation_tolerance: f64 = settings_numerics.grad_shafranov_deviation_tolerance;

    let settings_nonlinear_solver: &SettingsEquilibriumNumericsNonlinearSolver = &settings_numerics.nonlinear_solver;
    match settings_nonlinear_solver.method.as_str() {
        "picard" => {
            let settings_picard: &SettingsEquilibriumNumericsNonlinearSolverPicard = &settings_nonlinear_solver.picard;
            Ok(Arc::new(Picard {
                n_iter_max,
                n_iter_min,
                grad_shafranov_deviation_tolerance,
                n_iter_no_vertical_feedback: settings_picard.n_iter_no_vertical_feedback,
                apply_anderson_mixing: settings_picard.apply_anderson_mixing,
                anderson_n_history: settings_picard.anderson_n_history,
                anderson_mixing: settings_picard.anderson_mixing,
            }))
        }
        "newton_krylov" => {
            let settings_newton_krylov: &SettingsEquilibriumNumericsNonlinearSolverNewtonKrylov = &settings_nonlinear_solver.newton_krylov;
            Ok(Arc::new(NewtonKrylov {
                n_iter_max,
                n_iter_min,
                grad_shafranov_deviation_tolerance,
                n_iter_no_vertical_feedback: settings_newton_krylov.n_iter_no_vertical_feedback,
                picard_handover: settings_newton_krylov.picard_handover,
                n_krylov_max: settings_newton_krylov.n_krylov_max,
                krylov_tolerance: settings_newton_krylov.krylov_tolerance,
                finite_difference_step: settings_newton_krylov.finite_difference_step,
                verbose: settings_newton_krylov.verbose,
            }))
        }
        "newton_picard" => {
            let settings_newton_picard: &SettingsEquilibriumNumericsNonlinearSolverNewtonPicard = &settings_nonlinear_solver.newton_picard;
            Ok(Arc::new(NewtonPicard {
                n_iter_max,
                n_iter_min,
                grad_shafranov_deviation_tolerance,
                n_basis_max: settings_newton_picard.n_basis_max,
                contraction_threshold: settings_newton_picard.contraction_threshold,
                finite_difference_step: settings_newton_picard.finite_difference_step,
                verbose: settings_newton_picard.verbose,
            }))
        }
        method => Err(PyValueError::new_err(format!(
            "`equilibrium_settings/numerics/nonlinear_solver/method` = \"{method}\" is not recognized"
        ))),
    }
}

/// Apply every degree of freedom's current distribution to a Green's table, and sum over the passive
/// filaments.
///
/// The straightforward way to write this is one pass per degree of freedom:
/// ```ignore
/// let with_dof: Array2<f64> = &greens_filaments * &current_distribution;  // a table-sized temporary
/// let g: Array1<f64> = with_dof.sum_axis(Axis(1));
/// ```
/// which re-reads the whole `[n_grid, n_filament]` table once per degree of freedom, and allocates
/// and streams a table-sized temporary each time. For ST40's inner vessel that is 15 degrees of
/// freedom against a 56 MB table, for each of 13 tables.
///
/// This does all the degrees of freedom in a single pass over the table instead: one grid point's
/// row is loaded, and every degree of freedom is applied to it while it is still in cache. Nothing
/// table-sized is allocated.
///
/// The arithmetic is unchanged. The products are formed in the same order and handed to the same
/// `ndarray` summation that `sum_axis` used, so the answer is identical bit for bit.
///
/// # Arguments
/// * `greens_filaments` - Green's table between the grid and this passive's filaments, shape = [n_grid, n_filament]
/// * `current_distributions` - one current distribution per degree of freedom, each of length `n_filament`, [dimensionless]
///
/// # Returns
/// * `greens_with_dof` - shape = [n_grid, n_dof]
fn apply_current_distributions(greens_filaments: &Array2<f64>, current_distributions: &[Array1<f64>]) -> Array2<f64> {
    let n_grid: usize = greens_filaments.nrows();
    let n_filament: usize = greens_filaments.ncols();
    let n_dof: usize = current_distributions.len();

    let mut greens_with_dof: Array2<f64> = Array2::from_elem((n_grid, n_dof), f64::NAN);
    if n_dof == 0 || n_filament == 0 || n_grid == 0 {
        return greens_with_dof;
    }

    // Both arrays are freshly allocated, and so are contiguous and row-major
    let greens_filaments_flat: &[f64] = greens_filaments
        .as_slice()
        .expect("apply_current_distributions: `greens_filaments` must be contiguous");
    let mut current_distributions_flat: Vec<&[f64]> = Vec::with_capacity(n_dof);
    for i_dof in 0..n_dof {
        current_distributions_flat.push(
            current_distributions[i_dof]
                .as_slice()
                .expect("apply_current_distributions: `current_distribution` must be contiguous"),
        );
    }

    // One rayon task per block of grid points, rather than per grid point, so that the work in a
    // task is large compared with the cost of handing it to a worker thread
    const N_GRID_PER_TASK: usize = 64;

    {
        let greens_with_dof_flat: &mut [f64] = greens_with_dof
            .as_slice_mut()
            .expect("apply_current_distributions: `greens_with_dof` must be contiguous");

        greens_filaments_flat
            .par_chunks(N_GRID_PER_TASK * n_filament)
            .zip(greens_with_dof_flat.par_chunks_mut(N_GRID_PER_TASK * n_dof))
            .for_each(|(greens_filaments_block, greens_with_dof_block): (&[f64], &mut [f64])| {
                let n_grid_in_block: usize = greens_filaments_block.len() / n_filament;

                // Scratch space for one grid point's row, re-used by every degree of freedom and
                // every grid point in this block
                let mut products: Array1<f64> = Array1::from_elem(n_filament, f64::NAN);

                for i_grid_in_block in 0..n_grid_in_block {
                    let greens_filaments_row: &[f64] = &greens_filaments_block[i_grid_in_block * n_filament..(i_grid_in_block + 1) * n_filament];

                    for i_dof in 0..n_dof {
                        let current_distribution: &[f64] = current_distributions_flat[i_dof];
                        for i_filament in 0..n_filament {
                            products[i_filament] = greens_filaments_row[i_filament] * current_distribution[i_filament];
                        }

                        // `Array1::sum()` on contiguous data is the same eight-accumulator reduction
                        // that `sum_axis(Axis(1))` applied to each row, so this is bit-identical
                        greens_with_dof_block[i_grid_in_block * n_dof + i_dof] = products.sum();
                    }
                }
            });
    }

    greens_with_dof
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;

    /// An XML schema accepting any `<equilibrium_settings>`. The real one is `equilibrium.xsd`, with the XML schema of each file it links to merged in,
    /// which `serialise_python_dataclass_to_imas_parameters_input` does in Python; these tests are of what `Equilibrium::new` makes from the settings
    const ANY_EQUILIBRIUM_SETTINGS_XSD: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xs:element name="equilibrium_settings">
        <xs:complexType>
            <xs:sequence>
                <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
            </xs:sequence>
        </xs:complexType>
    </xs:element>
</xs:schema>"#;

    /// Settings with every required element, on a small grid, without the optional diamagnetic flux
    const SETTINGS_XML: &str = "<equilibrium_settings>
        <timeslices>
            <method>user_defined</method>
            <arange><time_start>0.0</time_start><time_end>0.1</time_end><dt>0.01</dt></arange>
            <linspace><time_start>0.0</time_start><time_end>0.1</time_end><n_time>3</n_time></linspace>
            <user_defined>0.1 0.2</user_defined>
        </timeslices>
        <numerics>
            <RAYON_NUM_THREADS>1</RAYON_NUM_THREADS><n_iter_max>100</n_iter_max><grad_shafranov_deviation_tolerance>5.0e-6</grad_shafranov_deviation_tolerance>
            <n_iter_min>0</n_iter_min><save_unconverged>false</save_unconverged>
            <nonlinear_solver>
                <method>picard</method>
                <picard>
                    <n_iter_no_vertical_feedback>1</n_iter_no_vertical_feedback><apply_anderson_mixing>false</apply_anderson_mixing>
                    <anderson_n_history>5</anderson_n_history><anderson_mixing>1.0</anderson_mixing>
                </picard>
                <newton_krylov>
                    <n_iter_no_vertical_feedback>1</n_iter_no_vertical_feedback><picard_handover>0.01</picard_handover><n_krylov_max>10</n_krylov_max><krylov_tolerance>0.1</krylov_tolerance>
                    <finite_difference_step>1.0</finite_difference_step><verbose>false</verbose>
                </newton_krylov>
                <newton_picard>
                    <n_basis_max>10</n_basis_max><contraction_threshold>0.5</contraction_threshold><finite_difference_step>1.0</finite_difference_step>
                    <verbose>false</verbose>
                </newton_picard>
            </nonlinear_solver>
        </numerics>
        <grid><n_r>5</n_r><n_z>4</n_z><r_min>0.2</r_min><r_max>1.0</r_max><z_min>-1.0</z_min><z_max>1.0</z_max><n_psi_norm>11</n_psi_norm></grid>
        <initial_guess><ip>3.0e5</ip><cur_r>0.5</cur_r><cur_z>0.0</cur_z><minor_radius>0.2</minor_radius><elongation>1.2</elongation></initial_guess>
        <source_function_p_prime><method>efit_polynomial</method><efit_polynomial><n_dof>2</n_dof><regularisations/></efit_polynomial></source_function_p_prime>
        <source_function_ff_prime>
            <method>efit_polynomial</method><efit_polynomial><n_dof>3</n_dof><regularisations><row>1.0 0.0 0.0</row></regularisations></efit_polynomial>
        </source_function_ff_prime>
        <pf_passive_dof/>
        <constraints>
            <b_field_pol_probe/><flux_loop/><rogowski_coil/><diamagnetic_flux/><isoflux/><isoflux_boundary/><stationary_point/>
            <pressure><pressure><include>false</include></pressure></pressure>
        </constraints>
    </equilibrium_settings>";

    fn parameters_input_with(parameters_value: &str, schema: &str) -> PyParametersInput {
        PyParametersInput::new(ParametersInput {
            parameters_value: parameters_value.to_string(),
            parameters_default: String::new(),
            schema: schema.to_string(),
        })
    }

    #[test]
    fn the_equilibrium_is_built_from_its_settings() {
        let equilibrium: Equilibrium =
            Equilibrium::new(&parameters_input_with(SETTINGS_XML, ANY_EQUILIBRIUM_SETTINGS_XSD)).expect("the settings make an equilibrium");

        // The settings, as XML and parsed
        assert_eq!(equilibrium.equilibrium_ids.code.parameters, SETTINGS_XML);
        assert_eq!(equilibrium.settings_equilibrium.grid.n_r, 5);
        // Left out of the settings
        assert_eq!(equilibrium.settings_equilibrium.constraints.diamagnetic_flux, None);

        // One time-slice per time, each holding the grid
        assert_eq!(equilibrium.equilibrium_ids.time_slice.len(), 2);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].time, 0.2);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_2d[0].grid.dim1.len(), 5);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_2d[0].r.shape(), &[4, 5]);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_1d.psi_norm.len(), 11);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_1d.psi_norm[10], 1.0);

        // The Greens tables between the grid points
        assert_eq!(equilibrium.greens_tables.grid.plasma.psi.shape(), &[4, 5, 5]);

        // The source functions; p' has no regularisations
        assert_eq!(equilibrium.p_prime_source_function.source_function_n_dof(), 2);
        assert_eq!(equilibrium.p_prime_source_function.source_function_regularisation().shape(), &[0, 2]);
        assert_eq!(equilibrium.ff_prime_source_function.source_function_n_dof(), 3);
        assert_eq!(equilibrium.ff_prime_source_function.source_function_regularisation().shape(), &[1, 3]);
    }

    #[test]
    fn settings_without_their_xml_schema_are_an_error() {
        assert!(Equilibrium::new(&parameters_input_with(SETTINGS_XML, "")).is_err());
    }

    #[test]
    fn settings_which_do_not_follow_their_xml_schema_are_an_error() {
        // Not XML at all, which is what an empty `ParametersInput()` holds
        assert!(Equilibrium::new(&parameters_input_with("", ANY_EQUILIBRIUM_SETTINGS_XSD)).is_err());
        // Another IDS's settings
        assert!(Equilibrium::new(&parameters_input_with("<wall_settings/>", ANY_EQUILIBRIUM_SETTINGS_XSD)).is_err());
    }

    #[test]
    fn settings_missing_a_required_element_are_an_error() {
        let without_initial_guess: String = SETTINGS_XML.replace(
            "<initial_guess><ip>3.0e5</ip><cur_r>0.5</cur_r><cur_z>0.0</cur_z><minor_radius>0.2</minor_radius><elongation>1.2</elongation></initial_guess>",
            "",
        );
        assert_ne!(without_initial_guess, SETTINGS_XML);

        assert!(Equilibrium::new(&parameters_input_with(&without_initial_guess, ANY_EQUILIBRIUM_SETTINGS_XSD)).is_err());
    }

    /// The XML schema of the passive conductors' settings
    const PF_PASSIVE_SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/pf_passive/pf_passive.xsd");

    /// The fit settings of two passive conductors, in the other order to `pf_passive_with_two_loops`: the vessel's two slowest decaying
    /// eigenmodes, regularised, and the plate's constant current density
    const PF_PASSIVE_DOF_XML: &str = "<pf_passive_dof>
            <passive>
                <name>VESSEL</name><current_distribution_type>eig</current_distribution_type><include>true</include><n_dof>2</n_dof>
                <regularisations><row>1.0 -1.0</row></regularisations><regularisations_weight>0.5</regularisations_weight>
            </passive>
            <passive>
                <name>PLATE</name><current_distribution_type>constant_current_density</current_distribution_type><include>true</include>
                <n_dof>1</n_dof><regularisations/><regularisations_weight/>
            </passive>
        </pf_passive_dof>";

    /// An equilibrium whose fit settings hold `pf_passive_dof_xml` in place of no passive conductors
    fn equilibrium_with_pf_passive_dof(pf_passive_dof_xml: &str) -> Equilibrium {
        let settings_xml: String = SETTINGS_XML.replace("<pf_passive_dof/>", pf_passive_dof_xml);
        assert_ne!(settings_xml, SETTINGS_XML);

        Equilibrium::new(&parameters_input_with(&settings_xml, ANY_EQUILIBRIUM_SETTINGS_XSD)).expect("the settings make an equilibrium")
    }

    /// Two passive conductors, `PLATE` and `VESSEL`, each a row of four square filaments, 1 centimetre across
    fn pf_passive_with_two_loops() -> PfPassive {
        let mut pf_passive: PfPassive =
            PfPassive::new(&parameters_input_with("<pf_passive_settings/>", PF_PASSIVE_SCHEMA_XSD)).expect("the settings follow their XML schema");
        let loop_names: [&str; 2] = ["PLATE", "VESSEL"];
        let loop_z: [f64; 2] = [-0.5, 0.5];
        let n_loop: usize = loop_names.len();
        let n_element: usize = 4;
        for i_loop in 0..n_loop {
            pf_passive
                .add_loop_rs(
                    loop_names[i_loop],
                    Array1::from_vec(vec![0.50, 0.52, 0.54, 0.56]),
                    Array1::from_elem(n_element, loop_z[i_loop]),
                    Array1::from_elem(n_element, 0.01),
                    Array1::from_elem(n_element, 0.01),
                    Array1::zeros(n_element),
                    Array1::zeros(n_element),
                    7.4e-7,
                )
                .expect("the loop is valid");
        }

        pf_passive
    }

    #[test]
    fn the_passive_conductors_in_the_fit_settings_get_their_degrees_of_freedom_in_the_fit_settings_order() {
        let mut equilibrium: Equilibrium = equilibrium_with_pf_passive_dof(PF_PASSIVE_DOF_XML);
        let pf_passive: PfPassive = pf_passive_with_two_loops();

        equilibrium
            .copy_from_pf_passive_ids_rs(&pf_passive)
            .expect("the fit settings make degrees of freedom");
        // Calling it again replaces them, rather than repeating them
        equilibrium
            .copy_from_pf_passive_ids_rs(&pf_passive)
            .expect("the fit settings make degrees of freedom");

        let n_time: usize = equilibrium.equilibrium_ids.time_slice.len();
        assert_eq!(n_time, 2);
        for i_time in 0..n_time {
            let pf_passive_currents: &Vec<EquilibriumConstraints0d> = &equilibrium.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current;
            assert_eq!(pf_passive_currents.len(), 2);

            // The vessel's eigenmodes, slowest decaying first
            let vessel: &EquilibriumConstraints0d = &pf_passive_currents[0];
            assert_eq!(vessel.name, "VESSEL");
            assert_eq!(vessel.dof.len(), 2);
            assert_eq!(vessel.dof[0].name, "eig_01");
            assert_eq!(vessel.dof[1].name, "eig_02");
            assert!(vessel.dof[0].time_constant > vessel.dof[1].time_constant);
            assert_eq!(vessel.dof[0].current_distribution.len(), 4);
            assert_eq!(vessel.regularisations.shape(), &[1, 2]);
            assert_eq!(vessel.regularisations_weight[0], 0.5);

            // The plate's constant current density, whose current distribution is the elements' areas
            let plate: &EquilibriumConstraints0d = &pf_passive_currents[1];
            assert_eq!(plate.name, "PLATE");
            assert_eq!(plate.dof.len(), 1);
            assert_eq!(plate.dof[0].name, "constant_current_density");
            assert!(plate.dof[0].time_constant.is_nan());
            let n_element: usize = 4;
            for i_element in 0..n_element {
                assert_eq!(
                    plate.dof[0].current_distribution[i_element],
                    pf_passive.pf_passive_ids.r#loop[0].element[i_element].area
                );
            }
            // No rows, with a column for each degree of freedom
            assert_eq!(plate.regularisations.shape(), &[0, 1]);
        }
    }

    #[test]
    fn a_passive_conductor_in_the_fit_settings_which_is_not_in_pf_passive_is_an_error() {
        let mut equilibrium: Equilibrium = equilibrium_with_pf_passive_dof(&PF_PASSIVE_DOF_XML.replace("<name>PLATE</name>", "<name>IVC</name>"));

        let result: Result<(), String> = equilibrium.copy_from_pf_passive_ids_rs(&pf_passive_with_two_loops());

        assert!(result.expect_err("IVC is not in pf_passive").contains("`IVC`"));
    }

    #[test]
    fn fit_settings_which_cannot_make_degrees_of_freedom_are_an_error() {
        let pf_passive: PfPassive = pf_passive_with_two_loops();
        let pf_passive_dof_xmls: Vec<String> = vec![
            // More eigenmodes than elements
            PF_PASSIVE_DOF_XML
                .replace("<n_dof>2</n_dof>", "<n_dof>5</n_dof>")
                .replace("<row>1.0 -1.0</row>", "<row>1.0 -1.0 0.0 0.0 0.0</row>"),
            // A constant current density has one degree of freedom
            PF_PASSIVE_DOF_XML.replace("<n_dof>1</n_dof>", "<n_dof>2</n_dof>"),
            // A regularisation with the wrong number of coefficients
            PF_PASSIVE_DOF_XML.replace("<row>1.0 -1.0</row>", "<row>1.0 -1.0 0.0</row>"),
            // A regularisation without a weight
            PF_PASSIVE_DOF_XML.replace("<regularisations_weight>0.5</regularisations_weight>", "<regularisations_weight/>"),
            // The same passive conductor twice
            PF_PASSIVE_DOF_XML.replace("<name>PLATE</name>", "<name>VESSEL</name>"),
        ];
        let n_case: usize = pf_passive_dof_xmls.len();
        for i_case in 0..n_case {
            assert_ne!(pf_passive_dof_xmls[i_case], PF_PASSIVE_DOF_XML);
            let mut equilibrium: Equilibrium = equilibrium_with_pf_passive_dof(&pf_passive_dof_xmls[i_case]);

            assert!(equilibrium.copy_from_pf_passive_ids_rs(&pf_passive).is_err(), "{}", pf_passive_dof_xmls[i_case]);
        }
    }
}
