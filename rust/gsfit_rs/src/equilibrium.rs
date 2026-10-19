use crate::greens::Greens;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use crate::pulse_schedule::PulseSchedule;
use crate::source_functions::{EfitPolynomial, SharedSourceFunction, TensionedCubicBSpline};
use crate::tf::Tf;
use crate::wall::Wall;
use imas_rs::EMPTY_INT;
use imas_rs::ids::equilibrium::{
    Code, Equilibrium as EquilibriumIds, EquilibriumCodeParametersParsed, EquilibriumCodeParametersParsedEfitPolynomial,
    EquilibriumCodeParametersParsedEquilibriumConstraints, EquilibriumCodeParametersParsedGrid, EquilibriumCodeParametersParsedTensionedCubicBSpline,
    EquilibriumCodeParametersParsedTimeslices, EquilibriumConstraints0d, EquilibriumConstraints0dB0Like, EquilibriumConstraints0dDof,
    EquilibriumConstraints0dIpLike, EquilibriumConstraints0dOneLike, EquilibriumGap, EquilibriumGreensGridPfActive, EquilibriumGreensGridPfPassive,
    EquilibriumGreensGridPfPassiveDof, EquilibriumGreensGridPlasma, EquilibriumLimiterUnit, EquilibriumLimiterUnitOutline, EquilibriumProfiles2d,
};
use imas_rs::ids::pf_active::PfCoils;
use imas_rs::ids::pf_passive::PfPassiveLoops;
use imas_rs::ids::pulse_schedule::PulseScheduleGap;
use imas_rs::ids::wall::Wall2dLimiterUnit;
use imas_rs::python::{PyEquilibrium, PyParametersInput, PyPath, read_path};
use ndarray::{Array1, Array2, ArrayView2, Axis, MeshIndex, meshgrid, s};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// The equilibrium, stored as an IMAS `equilibrium` IDS.
///
/// It is built from GSFit's settings alone:
/// * `code/parameters` - GSFit's settings, as XML text, which is where IMAS records the parameters of the code which made an IDS
/// * `code/parameters_parsed` - the same settings read into structs, e.g. `code/parameters_parsed/grid/n_r`, which everything
///   below is built from
/// * `time_slice(itime)` - one per time to reconstruct, each holding its `time`, the (R, Z) grid in `profiles_2d(0)`, the
///   normalised poloidal flux grid of the 1D profiles, `profiles_1d/psi_norm`, and the weight of each magnetic sensor in the
///   settings, in `constraints/<sensor>(i1)/weight`
/// * `greens/grid/plasma` - the Greens tables between the grid points
///
/// The source functions, p' and ff', are not data, so are held alongside the IDS rather than in it.
///
/// The other IDSs are then copied in, by `copy_from_wall_ids`, `copy_from_tf_ids`, `copy_from_pf_active_ids`,
/// `copy_from_pf_passive_ids`, `copy_from_pulse_schedule_ids` and `copy_from_magnetics_ids`, and the Greens tables with the PF coils,
/// the passive conductors and the magnetic sensors are calculated, by `greens_with_pf_active`, `greens_with_pf_passive` and
/// `greens_with_magnetics`, before `solve_grad_shafranov` solves it.
///
/// Read the data back through `get`, or `equilibrium_ids`, and a path from `gsfit_rs.imas.equilibrium_paths`; there are no
/// bespoke accessors.
///
/// # Settings
///
/// The equilibrium is constructed with GSFit's settings, `equilibrium.xml` with each file it links to written in its link's
/// place, as an IMAS `parameters_input`. Only their XML is kept: it is written to `code/parameters`, and read into
/// `code/parameters_parsed`, whose structs are generated from the settings' XML schema,
/// `python/gsfit/settings_schema/xsd/equilibrium/equilibrium.xsd`. The rest of the `parameters_input`, the XML schema and the
/// defaults, which are empty, is not kept: the data dictionary has no node for either.
#[pyclass(module = "gsfit_rs")]
pub struct Equilibrium {
    pub equilibrium_ids: EquilibriumIds,
    /// The p' source function, made from `code/parameters_parsed/source_function_p_prime`
    pub p_prime_source_function: SharedSourceFunction,
    /// The ff' source function, made from `code/parameters_parsed/source_function_ff_prime`
    pub ff_prime_source_function: SharedSourceFunction,
}

/// Python accessible methods
#[pymethods]
impl Equilibrium {
    /// Construct the equilibrium from GSFit's settings.
    ///
    /// The settings are read into `code/parameters_parsed`, and everything else is made from them: the time-slices, the grid, the
    /// magnetic sensors' weights, the Greens tables between the grid points, and the source functions. See `Equilibrium`.
    ///
    /// # Arguments
    /// * `parameters_input` - GSFit's settings, as a `gsfit_rs.dd_support.ParametersInput`, whose XML is `equilibrium.xml` with
    ///   each file it links to written in its link's place; it is written to `code/parameters`, and read into
    ///   `code/parameters_parsed`
    ///
    /// # Errors
    /// `ValueError` if the XML cannot be read into `code/parameters_parsed`: it is not well-formed, it is not `equilibrium.xml`,
    /// it holds an element which is not in the XML schema `code/parameters_parsed` was generated from, or a value cannot be read
    /// as its type. Or if the settings cannot make an equilibrium, e.g. a grid with fewer than two points in a direction
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // GSFit's settings, read into `code/parameters_parsed`
        let parameters_value: &str = &parameters_input.inner.parameters_value;
        let parameters_parsed: EquilibriumCodeParametersParsed = EquilibriumCodeParametersParsed::from_xml(parameters_value)
            .map_err(|error| PyValueError::new_err(format!("GSFit's settings cannot be read into `equilibrium/code/parameters_parsed`: {error}")))?;

        // The times to reconstruct. They are calculated step by step as numpy calculates them in `Gsfit.setup_timeslices`, so that
        // they are the same to the bit
        let timeslices: &EquilibriumCodeParametersParsedTimeslices = &parameters_parsed.timeslices;
        let times_to_reconstruct: Array1<f64> = match timeslices.method.as_str() {
            "arange" => {
                // `numpy.arange(time_start, time_end, dt)`: ceil((time_end - time_start) / dt) times, of which the first two are
                // `time_start` and `time_start + dt`, and the rest `time_start + i_time * delta`, with `delta` the difference
                // between the first two
                let time_start: f64 = timeslices.arange.time_start;
                let time_end: f64 = timeslices.arange.time_end;
                let dt: f64 = timeslices.arange.dt;
                let n_time_ceil: f64 = ((time_end - time_start) / dt).ceil();
                if !n_time_ceil.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "`code/parameters_parsed/timeslices/arange` gives no number of times: time_start = {time_start}, time_end = {time_end}, dt = {dt}"
                    )));
                }
                let n_time: usize = n_time_ceil.max(0.0) as usize;
                let mut times: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                if n_time > 0 {
                    times[0] = time_start;
                }
                if n_time > 1 {
                    times[1] = time_start + dt;
                    let delta: f64 = times[1] - times[0];
                    for i_time in 2..n_time {
                        times[i_time] = time_start + (i_time as f64) * delta;
                    }
                }
                times
            }
            "linspace" => {
                // `numpy.linspace(time_start, time_end, n_time)`, which, unlike `Array1::linspace`, ends at exactly `time_end`
                let time_start: f64 = timeslices.linspace.time_start;
                let time_end: f64 = timeslices.linspace.time_end;
                let n_time: usize = usize::try_from(timeslices.linspace.n_time).map_err(|_| {
                    PyValueError::new_err(format!(
                        "`code/parameters_parsed/timeslices/linspace/n_time` is {}, but it must be a number of times",
                        timeslices.linspace.n_time
                    ))
                })?;
                let mut times: Array1<f64> = Array1::linspace(time_start, time_end, n_time);
                if n_time > 1 {
                    times[n_time - 1] = time_end;
                }
                times
            }
            "user_defined" => timeslices.user_defined.clone(),
            // "good_pressure_sensors" is in the XML schema, but, as in `Gsfit.setup_timeslices`, is not implemented yet
            method => {
                return Err(PyValueError::new_err(format!(
                    "`code/parameters_parsed/timeslices/method` is \"{method}\", but only \"arange\", \"linspace\" and \"user_defined\" are implemented"
                )));
            }
        };

        // The (R, Z) grid. An unset number of points is negative
        let grid: &EquilibriumCodeParametersParsedGrid = &parameters_parsed.grid;
        if grid.n_r < 2 || grid.n_z < 2 || grid.n_psi_norm < 2 {
            return Err(PyValueError::new_err(format!(
                "`code/parameters_parsed/grid` has n_r = {}, n_z = {} and n_psi_norm = {}, but each must be at least 2",
                grid.n_r, grid.n_z, grid.n_psi_norm
            )));
        }
        if !(grid.r_min.is_finite() && grid.r_max.is_finite() && grid.z_min.is_finite() && grid.z_max.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "`code/parameters_parsed/grid` has r_min = {}, r_max = {}, z_min = {} and z_max = {}, but each must be a number",
                grid.r_min, grid.r_max, grid.z_min, grid.z_max
            )));
        }
        let n_r: usize = grid.n_r as usize;
        let n_z: usize = grid.n_z as usize;
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
                "`code/parameters_parsed/grid` has r_min - d_r / 2 = {}, so the radial grid goes negative",
                grid.r_min - d_r / 2.0
            )));
        }

        // 2d (r, z) mesh
        let (mesh_z_view, mesh_r_view): (ArrayView2<f64>, ArrayView2<f64>) = meshgrid((&z, &r), MeshIndex::IJ);
        let mesh_z: Array2<f64> = mesh_z_view.to_owned(); // shape = (n_z, n_r)
        let mesh_r: Array2<f64> = mesh_r_view.to_owned(); // shape = (n_z, n_r)

        // The normalised poloidal flux grid of the 1D profiles, which the source functions are evaluated on. It is
        // `numpy.linspace(0.0, 1.0, n_psi_norm)`, which, unlike `Array1::linspace`, ends at exactly 1.0
        let n_psi_norm: usize = grid.n_psi_norm as usize;
        let mut psi_norm: Array1<f64> = Array1::linspace(0.0, 1.0, n_psi_norm);
        psi_norm[n_psi_norm - 1] = 1.0;

        // Check the nonlinear solver is one `EquilibriumSolver::solve` knows
        let nonlinear_solver_method: &str = &parameters_parsed.numerics.nonlinear_solver.method;
        if !["picard", "newton_krylov", "newton_picard"].contains(&nonlinear_solver_method) {
            return Err(PyValueError::new_err(format!(
                "`code/parameters_parsed/numerics/nonlinear_solver/method` is \"{nonlinear_solver_method}\", but it must be \"picard\", \"newton_krylov\" or \"newton_picard\""
            )));
        }

        // Anderson mixing takes a positive fraction of the residual
        let anderson_mixing: f64 = parameters_parsed.numerics.nonlinear_solver.picard.anderson_mixing;
        if anderson_mixing.is_nan() || anderson_mixing <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "`code/parameters_parsed/numerics/nonlinear_solver/picard/anderson_mixing` is {anderson_mixing}, but it must be positive"
            )));
        }

        // The source functions
        let p_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &parameters_parsed.source_function_p_prime.method,
            &parameters_parsed.source_function_p_prime.efit_polynomial,
            &parameters_parsed.source_function_p_prime.tensioned_cubic_b_spline,
            "code/parameters_parsed/source_function_p_prime",
        )?;
        let ff_prime_source_function: SharedSourceFunction = source_function_from_settings(
            &parameters_parsed.source_function_ff_prime.method,
            &parameters_parsed.source_function_ff_prime.efit_polynomial,
            &parameters_parsed.source_function_ff_prime.tensioned_cubic_b_spline,
            "code/parameters_parsed/source_function_ff_prime",
        )?;

        // The Greens tables between the grid points, `greens/grid/plasma`. The observers are every grid point, row by row, and the
        // sources the points of the lowest row, so each table is calculated with shape (n_z * n_r, n_r). An observer's row is its
        // offset above the source, so the table is (i_offset_z, i_r, i_r_source)
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
        let greens_grid_plasma: EquilibriumGreensGridPlasma = EquilibriumGreensGridPlasma {
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

        // The magnetic sensors' weights, from their fit settings, `code/parameters_parsed/constraints`: one constraint for each sensor in
        // the settings, with its `name` and `weight`. `copy_from_magnetics_ids` adds each sensor's measurement, found by its name
        let constraints_settings: &EquilibriumCodeParametersParsedEquilibriumConstraints = &parameters_parsed.constraints;
        let n_b_field_pol_probe: usize = constraints_settings.b_field_pol_probe.len();
        let mut b_field_pol_probe_constraints: Vec<EquilibriumConstraints0dOneLike> = Vec::with_capacity(n_b_field_pol_probe);
        for i_b_field_pol_probe in 0..n_b_field_pol_probe {
            b_field_pol_probe_constraints.push(EquilibriumConstraints0dOneLike {
                name: constraints_settings.b_field_pol_probe[i_b_field_pol_probe].name.clone(),
                weight: constraints_settings.b_field_pol_probe[i_b_field_pol_probe].weight,
                ..Default::default()
            });
        }
        let n_flux_loop: usize = constraints_settings.flux_loop.len();
        let mut flux_loop_constraints: Vec<EquilibriumConstraints0d> = Vec::with_capacity(n_flux_loop);
        for i_flux_loop in 0..n_flux_loop {
            flux_loop_constraints.push(EquilibriumConstraints0d {
                name: constraints_settings.flux_loop[i_flux_loop].name.clone(),
                weight: constraints_settings.flux_loop[i_flux_loop].weight,
                ..Default::default()
            });
        }
        let n_rogowski_coil: usize = constraints_settings.rogowski_coil.len();
        let mut rogowski_coil_constraints: Vec<EquilibriumConstraints0dIpLike> = Vec::with_capacity(n_rogowski_coil);
        for i_rogowski_coil in 0..n_rogowski_coil {
            rogowski_coil_constraints.push(EquilibriumConstraints0dIpLike {
                name: constraints_settings.rogowski_coil[i_rogowski_coil].name.clone(),
                weight: constraints_settings.rogowski_coil[i_rogowski_coil].weight,
                ..Default::default()
            });
        }
        // The data dictionary has a single diamagnetic flux constraint, which has no name. Its settings are optional, and their `include` is
        // required, so it is unset only when they are left out
        let mut diamagnetic_flux_constraint: EquilibriumConstraints0dB0Like = EquilibriumConstraints0dB0Like::default();
        if constraints_settings.diamagnetic_flux.include != EMPTY_INT {
            diamagnetic_flux_constraint.weight = constraints_settings.diamagnetic_flux.weight;
        }

        let mut equilibrium_ids: EquilibriumIds = EquilibriumIds {
            code: Code {
                parameters: parameters_value.to_string(),
                parameters_parsed,
                ..Default::default()
            },
            ..Default::default()
        };
        // They are geometry, so they are the same for every time-slice, and hang off the IDS root rather than off `time_slice`
        equilibrium_ids.greens.grid.plasma = greens_grid_plasma;

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
        }

        Ok(Self {
            equilibrium_ids,
            p_prime_source_function,
            ff_prime_source_function,
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
    /// A `ValueError` when `tf/r0` or `tf/b_field_phi_vacuum_r` is unset, or when
    /// `tf/b_field_phi_vacuum_r` cannot be interpolated onto the reconstruction times, e.g. because its
    /// timebase does not cover them.
    pub fn copy_from_tf_ids(&mut self, tf: PyRef<Tf>) -> PyResult<()> {
        let r0: f64 = tf.tf_ids.r0;
        if r0.is_nan() {
            return Err(PyValueError::new_err("`tf/r0` is unset"));
        }
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
    /// `solve_grad_shafranov`. The solver multiplies them by the Greens tables, `greens/grid/pf_active(i1)`, so the coils
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

    /// Copy the passive loops' degrees of freedom, and their regularisations, from the pf_passive IDS onto every time-slice of
    /// the equilibrium IDS.
    ///
    /// Fills, for every loop in `pf_passive/loop(i1)`, in the same order, on every time-slice `Equilibrium::new` allocated:
    /// * `equilibrium/time_slice(itime)/constraints/pf_passive_current(i1)/name` - a custom key: `pf_passive/loop(i1)/name`
    /// * `.../dof(i2)/name` - a custom key: `pf_passive/loop(i1)/dof(i2)/name`, one for each degree of freedom the loop's current
    ///   is fitted with
    /// * `.../regularisations` - a custom key: `pf_passive/loop(i1)/regularisations`, shape = (n_regularisations, n_dof)
    /// * `.../regularisations_weight` - a custom key: `pf_passive/loop(i1)/regularisations_weight`, [dimensionless]
    ///
    /// `pf_passive_current(i1)` is where the data dictionary puts the current in `pf_passive/loop(i1)`. GSFit does not measure that
    /// current: it fits it, as an amplitude for each of the loop's degrees of freedom, and the regularisations are what constrain the
    /// fit. So `measured`, `source` and `time_measurement` are left unset.
    ///
    /// The solver reads the regularisations from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`. It fits
    /// the amplitudes with the Greens tables, `greens/grid/pf_passive(i1)/dof(i2)`, so the loops and their degrees of freedom have to be in
    /// the same order as those, which the solver checks by name.
    ///
    /// A pf_passive IDS with no loops is allowed: the equilibrium then has no passive conductors. Any passive loops already on a
    /// time-slice are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `pf_passive` - The PfPassive object (a Rust implementation, initialised in Python)
    pub fn copy_from_pf_passive_ids(&mut self, pf_passive: PyRef<PfPassive>) {
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();

        // The regularisations were checked against the degrees of freedom by `PfPassive.add_loop` when the loop was added
        let mut pf_passive_currents: Vec<EquilibriumConstraints0d> = Vec::with_capacity(n_loop);
        for i_loop in 0..n_loop {
            let n_dof: usize = pf_passive_loops[i_loop].dof.len();
            let mut dofs: Vec<EquilibriumConstraints0dDof> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                dofs.push(EquilibriumConstraints0dDof {
                    name: pf_passive_loops[i_loop].dof[i_dof].name.clone(),
                });
            }

            pf_passive_currents.push(EquilibriumConstraints0d {
                name: pf_passive_loops[i_loop].name.clone(),
                dof: dofs,
                regularisations: pf_passive_loops[i_loop].regularisations.clone(),
                regularisations_weight: pf_passive_loops[i_loop].regularisations_weight.clone(),
                ..Default::default()
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the passive loops instead of
        // repeating them
        let n_time: usize = self.equilibrium_ids.time_slice.len();
        for i_time in 0..n_time {
            self.equilibrium_ids.time_slice[i_time].constraints.pf_passive_current = pf_passive_currents.clone();
        }
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
    /// Only the sensors in the fit settings, `code/parameters_parsed/constraints`, are copied, and each of them has to be in the magnetics
    /// IDS. Each measurement is on the experimental timebase, so it is interpolated onto the time-slice's time. On every time-slice
    /// `Equilibrium::new` allocated, for every sensor in the fit settings, in the order of the magnetics IDS:
    /// * `magnetics/b_field_pol_probe(i1)/field` fills `equilibrium/time_slice(itime)/constraints/b_field_pol_probe(i1)`, [tesla]
    /// * `magnetics/flux_loop(i1)/flux` fills `.../constraints/flux_loop(i1)`, [weber]
    /// * `magnetics/rogowski_coil(i1)/current` fills `.../constraints/rogowski_coil(i1)`, a custom key, as the data dictionary has
    ///   no constraint for Rogowski coils, [ampere]
    /// * `magnetics/diamagnetic_flux(0)/data` fills `.../constraints/diamagnetic_flux`, [weber]
    /// * `magnetics/ip(0)/data` fills `.../constraints/ip`, [ampere]
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
    /// diamagnetic flux, from its own: its `weight`.
    ///
    /// Any magnetic constraints already on a time-slice are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `magnetics` - The Magnetics object (a Rust implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when a sensor in the fit settings is not in the magnetics IDS, when the magnetics IDS holds more than one
    /// diamagnetic flux, or when a measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not
    /// cover them.
    pub fn copy_from_magnetics_ids(&mut self, magnetics: PyRef<Magnetics>) -> PyResult<()> {
        // Change Python types into Rust types
        let magnetics_rs: &Magnetics = &magnetics;

        // Each kind of sensor is copied by its own method, in `magnetic_sensors/`
        self.copy_from_magnetics_ids_b_field_pol_probe(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_diamagnetic_flux(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_flux_loop(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_ip(magnetics_rs).map_err(PyValueError::new_err)?;
        self.copy_from_magnetics_ids_rogowski_coil(magnetics_rs).map_err(PyValueError::new_err)?;

        Ok(())
    }

    /// Calculate the Greens function with the PF coils
    /// The Greens tables are stored within self. Example data structure:
    /// `greens/grid/pf_active(i)/psi` on the equilibrium IDS, one for each `pf_active/coil(i)`, in the same order
    ///
    /// # Arguments
    /// * `pf_active` - The PfActive object (a Rust implementation, initialised in Python)
    ///
    fn greens_with_pf_active(&mut self, pf_active: PyRef<PfActive>) {
        // Unpack from self
        let n_r: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_r as usize;
        let n_z: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_z as usize;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables for the equilibrium IDS, one for each PF coil, in the order of `pf_active/coil`, which is
        // the order `Equilibrium.copy_from_pf_active_ids` copies the PF coil currents in
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_coil: usize = pf_active_coils.len();
        let mut greens_pf_active: Vec<EquilibriumGreensGridPfActive> = Vec::with_capacity(n_coil);

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
            greens_pf_active.push(EquilibriumGreensGridPfActive {
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

        self.equilibrium_ids.greens.grid.pf_active = greens_pf_active;
    }

    /// Calculate the Greens function with the passive conductors
    /// The Greens tables are stored within self. Example data structure:
    /// `greens/grid/pf_passive(i)/dof(j)/psi` on the equilibrium IDS
    /// Note: when adding a passive conductor to `pf_passive` we selected how to represent its degrees of freedom through
    /// `loop(i1)/current_distribution_type` (e.g. `constant_current_density` or `eig`)
    ///
    /// The tables are in the order of `pf_passive/loop`, and then of `loop(i1)/dof`. This is the order of the passive degrees of
    /// freedom everywhere in the solve: the sensors' Greens are gathered in it, the regularisations are assembled in it, and so
    /// `time_slice(itime)/passive_dof_values` follows it
    ///
    /// # Arguments
    /// * `pf_passive` - The passive conductors, a `PfPassive` (a Rust implementation, initialised in Python)
    ///
    fn greens_with_pf_passive(&mut self, pf_passive: PyRef<PfPassive>) {
        // The passive conductors, in the order of `pf_passive/loop`
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();

        // Get variables out of self
        // `profiles_2d/r` and `/z` are the (R, Z) mesh, so iterating them row-major gives the
        // flattened grid
        let n_r: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_r as usize;
        let n_z: usize = self.equilibrium_ids.code.parameters_parsed.grid.n_z as usize;
        let mesh_r: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].r;
        let mesh_z: &Array2<f64> = &self.equilibrium_ids.time_slice[0].profiles_2d[0].z;
        let flat_r: Array1<f64> = Array1::from_iter(mesh_r.iter().copied());
        let flat_z: Array1<f64> = Array1::from_iter(mesh_z.iter().copied());

        // Greens tables for the equilibrium IDS
        let mut greens_pf_passive: Vec<EquilibriumGreensGridPfPassive> = Vec::with_capacity(n_loop);

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
            let n_dof: usize = pf_passive_loops[i_loop].dof.len();
            let mut current_distributions: Vec<Array1<f64>> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                current_distributions.push(pf_passive_loops[i_loop].dof[i_dof].current_distribution.clone());
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
            let mut greens_dof: Vec<EquilibriumGreensGridPfPassiveDof> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                greens_dof.push(EquilibriumGreensGridPfPassiveDof {
                    name: pf_passive_loops[i_loop].dof[i_dof].name.clone(),
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

            greens_pf_passive.push(EquilibriumGreensGridPfPassive {
                name: passive_name.to_string(),
                dof: greens_dof,
            });
        }

        // Assigned rather than appended, so that calling this twice replaces the tables instead of
        // silently doubling them up
        self.equilibrium_ids.greens.grid.pf_passive = greens_pf_passive;
    }

    /// Calculate the magnetic sensors' Greens tables: what each sensor measures, per ampere in each current source.
    ///
    /// Fills, for every sensor in the fit settings, `code/parameters_parsed/constraints`, which are the sensors `copy_from_magnetics_ids`
    /// copies, with the same name and in the order of the magnetics IDS, `greens/b_field_pol_probe(i1)`, `greens/flux_loop(i1)` and
    /// `greens/rogowski_coil(i1)` on the equilibrium IDS, each with the sensor's tables with:
    /// * `plasma` - the plasma current in each grid cell, shape = (n_z, n_r)
    /// * `d_plasma_d_z` - the same, differentiated with respect to the sensor's height, shape = (n_z, n_r)
    /// * `pf_active` - each PF coil, in the order of `greens/grid/pf_active`, shape = (n_coil)
    /// * `pf_passive` - each passive degree of freedom, in the order of `greens/grid/pf_passive(:)/dof(:)`, shape = (n_dof_total)
    ///
    /// `greens/grid/pf_active` and `greens/grid/pf_passive` name the PF coils and the passive degrees of freedom, so
    /// `Equilibrium.greens_with_pf_active` and `Equilibrium.greens_with_pf_passive` have to be called first, with the same `pf_active` and
    /// `pf_passive`.
    ///
    /// Any sensors' Greens tables already on the IDS are replaced, so calling this twice does not repeat them.
    ///
    /// # Arguments
    /// * `magnetics` - The Magnetics object, holding where the sensors are (a Rust implementation, initialised in Python)
    /// * `pf_active` - The PfActive object, holding the PF coils' filaments (a Rust implementation, initialised in Python)
    /// * `pf_passive` - The PfPassive object, holding the passive conductors' filaments and degrees of freedom (a Rust
    ///   implementation, initialised in Python)
    ///
    /// # Errors
    /// A `ValueError` when a sensor in the fit settings is not in the magnetics IDS, or when the PF coils, or the passive loops and their
    /// degrees of freedom, are not the ones `greens/grid/pf_active` and `greens/grid/pf_passive` were calculated with, in the same order.
    fn greens_with_magnetics(&mut self, magnetics: PyRef<Magnetics>, pf_active: PyRef<PfActive>, pf_passive: PyRef<PfPassive>) -> PyResult<()> {
        // Change Python types into Rust types
        let magnetics_rs: &Magnetics = &magnetics;
        let pf_active_rs: &PfActive = &pf_active;
        let pf_passive_rs: &PfPassive = &pf_passive;

        // Every sensor's `pf_active` is in the order of `greens/grid/pf_active`, which names the PF coils, so the coils have to be
        // the ones those tables were calculated with
        let pf_active_names: Vec<String> = pf_active_rs.pf_active_ids.coil(..).name.to_vec();
        let greens_pf_active_names: Vec<String> = self.equilibrium_ids.greens.grid.pf_active(..).name.to_vec();
        if pf_active_names != greens_pf_active_names {
            return Err(PyValueError::new_err(format!(
                "`pf_active/coil(:)/name` is {pf_active_names:?}, but the Greens tables, `greens/grid/pf_active(:)/name`, are for \
                 {greens_pf_active_names:?}; call `Equilibrium.greens_with_pf_active` with these PF coils first"
            )));
        }

        // Likewise, every sensor's `pf_passive` is in the order of `greens/grid/pf_passive(:)/dof(:)`
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive_rs.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut pf_passive_dof_names: Vec<(String, String)> = Vec::new();
        for i_loop in 0..n_loop {
            let n_dof: usize = pf_passive_loops[i_loop].dof.len();
            for i_dof in 0..n_dof {
                pf_passive_dof_names.push((pf_passive_loops[i_loop].name.clone(), pf_passive_loops[i_loop].dof[i_dof].name.clone()));
            }
        }
        let n_greens_pf_passive: usize = self.equilibrium_ids.greens.grid.pf_passive.len();
        let mut greens_pf_passive_dof_names: Vec<(String, String)> = Vec::new();
        for i_greens_pf_passive in 0..n_greens_pf_passive {
            let n_dof: usize = self.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].dof.len();
            for i_dof in 0..n_dof {
                greens_pf_passive_dof_names.push((
                    self.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].name.clone(),
                    self.equilibrium_ids.greens.grid.pf_passive[i_greens_pf_passive].dof[i_dof].name.clone(),
                ));
            }
        }
        if pf_passive_dof_names != greens_pf_passive_dof_names {
            return Err(PyValueError::new_err(format!(
                "the passive degrees of freedom, `pf_passive/loop(:)/dof(:)/name`, are {pf_passive_dof_names:?}, but the Greens tables, \
                 `greens/grid/pf_passive(:)/dof(:)/name`, are for {greens_pf_passive_dof_names:?}; call `Equilibrium.greens_with_pf_passive` \
                 with these passive loops first"
            )));
        }

        // Only the sensors in the fit settings have Greens tables, and each of them has to be in the magnetics IDS
        let constraints_settings: &EquilibriumCodeParametersParsedEquilibriumConstraints = &self.equilibrium_ids.code.parameters_parsed.constraints;
        for (sensor, settings_names, magnetics_names) in [
            (
                "b_field_pol_probe",
                constraints_settings.b_field_pol_probe(..).name.to_vec(),
                magnetics_rs.magnetics_ids.b_field_pol_probe(..).name.to_vec(),
            ),
            (
                "flux_loop",
                constraints_settings.flux_loop(..).name.to_vec(),
                magnetics_rs.magnetics_ids.flux_loop(..).name.to_vec(),
            ),
            (
                "rogowski_coil",
                constraints_settings.rogowski_coil(..).name.to_vec(),
                magnetics_rs.magnetics_ids.rogowski_coil(..).name.to_vec(),
            ),
        ] {
            for settings_name in &settings_names {
                if !magnetics_names.contains(settings_name) {
                    return Err(PyValueError::new_err(format!(
                        "`{settings_name}` is in the fit settings, `code/parameters_parsed/constraints/{sensor}`, but not in the magnetics IDS, `magnetics/{sensor}`"
                    )));
                }
            }
        }

        // Each kind of sensor is calculated by its own method, in `magnetic_sensors/`
        self.greens_with_magnetics_b_field_pol_probe(magnetics_rs, pf_active_rs, pf_passive_rs);
        self.greens_with_magnetics_flux_loop(magnetics_rs, pf_active_rs, pf_passive_rs);
        self.greens_with_magnetics_rogowski_coil(magnetics_rs, pf_active_rs, pf_passive_rs);

        Ok(())
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.equilibrium_paths`, straight out of the equilibrium IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached by its data dictionary path,
    /// e.g. `equilibrium_paths.code.parameters_parsed.grid.n_r`. A path holds no data, so the IDS is only borrowed for the read,
    /// never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.equilibrium_ids, "equilibrium", path)
    }

    /// A copy of the whole equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`.
    ///
    /// Read the data with `get` instead. This copies everything, the Greens tables included, on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        PyEquilibrium::new(self.equilibrium_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Equilibrium>");
        string_output += &format!("║  {:<74} ║\n", version);

        let n_characters: usize = self.equilibrium_ids.code.parameters.chars().count();
        string_output += &format!("║  {:<74} ║\n", format!(" code/parameters: {n_characters} characters of XML"));

        let n_r: i32 = self.equilibrium_ids.code.parameters_parsed.grid.n_r;
        let n_z: i32 = self.equilibrium_ids.code.parameters_parsed.grid.n_z;
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
}

/// Make a source function, p' or ff', from its settings, as `make_source_function` in the database reader's `setup_plasma.py` does.
///
/// # Arguments
/// * `method` - the source function's `method`: "efit_polynomial" or "tensioned_cubic_b_spline"
/// * `efit_polynomial` - the source function's `efit_polynomial` settings, used when `method` is "efit_polynomial"
/// * `tensioned_cubic_b_spline` - the source function's `tensioned_cubic_b_spline` settings, used when `method` is
///   "tensioned_cubic_b_spline"
/// * `path` - the path of the source function's settings, e.g. `code/parameters_parsed/source_function_p_prime`, for the errors
///
/// # Errors
/// `ValueError` if `method` is not a source function, its settings are left out, or they cannot make a source function
fn source_function_from_settings(
    method: &str,
    efit_polynomial: &EquilibriumCodeParametersParsedEfitPolynomial,
    tensioned_cubic_b_spline: &EquilibriumCodeParametersParsedTensionedCubicBSpline,
    path: &str,
) -> PyResult<SharedSourceFunction> {
    match method {
        "efit_polynomial" => {
            // `n_dof` is required, so is unset, and negative, only when `efit_polynomial` is left out
            let n_dof: usize = usize::try_from(efit_polynomial.n_dof).map_err(|_| {
                PyValueError::new_err(format!(
                    "`{path}/method` is \"efit_polynomial\", but `{path}/efit_polynomial` is not in the settings"
                ))
            })?;

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
            // are fitted, and an unset `exact` is neither 0 nor 1
            let exact: bool = efit_polynomial.exact == 1;
            let source_function: EfitPolynomial = EfitPolynomial::new(n_dof, regularisations, exact, efit_polynomial.coefficients.clone())
                .map_err(|error| PyValueError::new_err(format!("`{path}/efit_polynomial`: {error}")))?;

            Ok(Arc::new(source_function))
        }
        "tensioned_cubic_b_spline" => {
            let interior_knots: &Array1<f64> = &tensioned_cubic_b_spline.interior_knots;
            let interval_tensions: &Array1<f64> = &tensioned_cubic_b_spline.interval_tensions;
            // There is one tension per interval between the knots, one more than the interior knots. Neither has any value only when
            // `tensioned_cubic_b_spline` is left out
            if interior_knots.is_empty() && interval_tensions.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "`{path}/method` is \"tensioned_cubic_b_spline\", but `{path}/tensioned_cubic_b_spline` is not in the settings"
                )));
            }
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
            // `make_cubic_spline_regularisations` is false when left out, when it is unset, so neither 0 nor 1
            let regularisations: Array2<f64> = if tensioned_cubic_b_spline.make_cubic_spline_regularisations == 1 {
                // Each of `regularisation_builder`'s elements is required, so `regularisation_mode` is empty only when it is left out
                if tensioned_cubic_b_spline.regularisation_builder.regularisation_mode.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "`{path}/tensioned_cubic_b_spline/make_cubic_spline_regularisations` is true, but `{path}/tensioned_cubic_b_spline/regularisation_builder` is not in the settings"
                    )));
                }
                TensionedCubicBSpline::make_regularisations(interior_knots, interval_tensions, &tensioned_cubic_b_spline.regularisation_builder)
                    .map_err(|error| PyValueError::new_err(format!("`{path}/tensioned_cubic_b_spline/regularisation_builder`: {error}")))?
            } else if tensioned_cubic_b_spline.regularisations.nrows() == 0 {
                // No regularisations is a matrix with no rows, which the settings give as shape (0, 0), or leave out.
                // The source function takes shape (n_regularisations, n_dof), so it needs shape (0, n_dof)
                Array2::zeros((0, n_dof))
            } else {
                tensioned_cubic_b_spline.regularisations.clone()
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
        _ => Err(PyValueError::new_err(format!(
            "`{path}/method` is \"{method}\", but it must be \"efit_polynomial\" or \"tensioned_cubic_b_spline\""
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

    #[test]
    fn the_equilibrium_is_built_from_its_settings() {
        let parameters_value: String = "<equilibrium_settings>
            <timeslices><method>user_defined</method><user_defined>0.1 0.2</user_defined></timeslices>
            <numerics><nonlinear_solver><method>picard</method><picard><anderson_mixing>1.0</anderson_mixing></picard></nonlinear_solver></numerics>
            <grid><n_r>5</n_r><n_z>4</n_z><r_min>0.2</r_min><r_max>1.0</r_max><z_min>-1.0</z_min><z_max>1.0</z_max><n_psi_norm>11</n_psi_norm></grid>
            <source_function_p_prime><method>efit_polynomial</method><efit_polynomial><n_dof>2</n_dof><regularisations/></efit_polynomial></source_function_p_prime>
            <source_function_ff_prime><method>efit_polynomial</method><efit_polynomial><n_dof>3</n_dof><regularisations><row>1.0 0.0 0.0</row></regularisations></efit_polynomial></source_function_ff_prime>
        </equilibrium_settings>"
            .to_string();
        let parameters_input: ParametersInput = ParametersInput {
            parameters_value: parameters_value.clone(),
            parameters_default: String::new(),
            schema: "<xs:schema/>".to_string(),
        };

        let equilibrium: Equilibrium = Equilibrium::new(&PyParametersInput::new(parameters_input)).expect("the settings make an equilibrium");

        // The settings, as XML and parsed
        assert_eq!(equilibrium.equilibrium_ids.code.parameters, parameters_value);
        assert_eq!(equilibrium.equilibrium_ids.code.parameters_parsed.grid.n_r, 5);
        // Left out of the settings, so unset
        assert!(equilibrium.equilibrium_ids.code.parameters_parsed.initial_guess.ip.is_nan());

        // One time-slice per time, each holding the grid
        assert_eq!(equilibrium.equilibrium_ids.time_slice.len(), 2);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].time, 0.2);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_2d[0].grid.dim1.len(), 5);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_2d[0].r.shape(), &[4, 5]);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_1d.psi_norm.len(), 11);
        assert_eq!(equilibrium.equilibrium_ids.time_slice[1].profiles_1d.psi_norm[10], 1.0);

        // The Greens tables between the grid points
        assert_eq!(equilibrium.equilibrium_ids.greens.grid.plasma.psi.shape(), &[4, 5, 5]);

        // The source functions; p' has no regularisations
        assert_eq!(equilibrium.p_prime_source_function.source_function_n_dof(), 2);
        assert_eq!(equilibrium.p_prime_source_function.source_function_regularisation().shape(), &[0, 2]);
        assert_eq!(equilibrium.ff_prime_source_function.source_function_n_dof(), 3);
        assert_eq!(equilibrium.ff_prime_source_function.source_function_regularisation().shape(), &[1, 3]);
    }
}
