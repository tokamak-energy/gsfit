use super::Configuration;
use super::coils_fit::{BoundaryCoilFit, fit_boundary_coils, plasma_current_density, plasma_current_filaments};
use super::eigenvalue::{flux_solution, solve_alpha};
use super::flux::{FluxDerivatives, FluxSolution};
use super::matching_geometry::{MatchingGeometry, matching_geometry};
use super::settings_schema::{GuazzottoFreidbergSettings, GuazzottoFreidbergSettingsGrid};
use crate::equilibrium_post_processor::equilibrium_post_processor;
use crate::grad_shafranov::CONVERGENCE_STATUS_CONVERGED;
use crate::grad_shafranov::contour_tree_nodes as grad_shafranov_contour_tree_nodes;
use crate::greens::{Conductor, ConductorModel, Greens, Sensor};
use crate::greens_storage::GreensTables;
use crate::magnetics::Magnetics;
use crate::pf_active::PfActive;
use crate::plasma_geometry::{BoundaryContour, MagneticAxis, StationaryPoint, find_boundary, find_magnetic_axis, find_stationary_points_using_winding_number};
use crate::settings::settings_pf_active::SettingsPfActive;
use crate::source_functions::{EfitPolynomial, SharedSourceFunction};
use crate::wall::Wall;
use imas_rs::ids::equilibrium::Code;
use imas_rs::ids::magnetics::SignalFlt1dValidity;
use imas_rs::ids::pf_active::{PfActive as PfActiveIds, PfCoils};
use imas_rs::ids::wall::{Wall as WallIds, Wall2dLimiterUnit};
use imas_rs::python::{PyEquilibrium, PyParametersInput};
use imas_rs::{
    Equilibrium, EquilibriumContourTreeNode, EquilibriumLimiter, EquilibriumLimiterUnit, EquilibriumLimiterUnitOutline, EquilibriumProfiles2d,
    EquilibriumTimeSlice,
};
use ndarray::{Array1, Array2, array};
use numpy::borrow::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Lower end of the range the eigenvalue `alpha` is searched for in: the eigenvalues of GF's Table 4 are all between 1.88 and 2.38
/// [dimensionless]
const ALPHA_SEARCH_MIN: f64 = 1.0;

/// Upper end of the range the eigenvalue `alpha` is searched for in [dimensionless]
const ALPHA_SEARCH_MAX: f64 = 3.0;

/// Maximum number of Newton iterations when locating the magnetic axis
const N_ITER_MAGNETIC_AXIS_MAX: usize = 50;

/// Newton step, in the normalised coordinates `x`, `y`, below which the magnetic axis is converged
const MAGNETIC_AXIS_TOLERANCE: f64 = 1.0e-12;

/// The data dictionary's `boundary/type`
const BOUNDARY_TYPE_LIMITED: i32 = 0;

/// The data dictionary's `contour_tree/node/critical_type`
const CRITICAL_TYPE_MAXIMUM: i32 = 2;

/// Guazzotto–Freidberg analytic Grad–Shafranov equilibrium.
///
/// Reference: L. Guazzotto & J. P. Freidberg, "Simple, general, realistic, robust,
/// analytic tokamak equilibria. Part 1", J. Plasma Phys. 87 (2021) 905870303.
///
/// The equilibrium is solved analytically by `solve`; `get_pf_active` then produces a [`PfActive`] object whose
/// PF-coil currents (placed on the `(R, Z)` grid boundary) reproduce the vacuum field
/// consistent with the analytic solution, and fills the Equilibrium IDS with the free-boundary equilibrium. This yields a
/// self-consistent free-boundary test case that GSFit can be asked to reconstruct.
///
/// # Settings
///
/// It is constructed with its settings, `equilibrium.xml` with each file it links to written in its link's place, as an IMAS
/// `parameters_input`, as GSFit's `Equilibrium` is. Their XML is written to `equilibrium_ids`' `code/parameters`, and read into
/// `guazzotto_freidberg_settings`, whose structs are generated from the settings' XML schema,
/// `python/analytic_grad_shafranov/guazzotto_freidberg/settings_schema/xsd/equilibrium/equilibrium.xsd`.
///
/// # Flux convention
///
/// GF write the flux per radian, `Psi = R * A_phi`. Everything stored here follows the IMAS
/// convention instead, the total poloidal flux `psi = 2 * pi * R * A_phi` [weber], so `psi_0` and the
/// Equilibrium IDS are both `2 * pi` times GF's values, and `dpressure_dpsi` and `f_df_dpsi` are
/// `1 / (2 * pi)` times GF's `p'` and `ff'`.
///
/// # Normalised flux
///
/// GF's normalised flux, `psi_hat = psi / psi_0`, is 1 at `(x, y) = (0, 0)` and 0 on the plasma
/// boundary. It is **not** the IMAS `psi_norm`, which is 0 at the magnetic axis and 1 on the
/// boundary.
#[pyclass(module = "gsfit_rs.analytic_grad_shafranov.guazzotto_freidberg")]
pub struct GuazzottoFreidberg {
    /// The Equilibrium IDS: `code/parameters` holds the settings from construction, and `get_pf_active` fills and post-processes the rest
    equilibrium_ids: Equilibrium,
    /// What the solve works from and makes, which is not IDS data: the problem resolved from the settings, and the solution
    guazzotto_freidberg_temporary_storage: GuazzottoFreidbergTemporaryStorage,
    /// The settings, the XML of `code/parameters` read into structs
    guazzotto_freidberg_settings: GuazzottoFreidbergSettings,
}

/// What `GuazzottoFreidberg` holds while it solves, alongside its IDS
struct GuazzottoFreidbergTemporaryStorage {
    /// The analytic problem, resolved from the settings when constructed
    problem: Problem,
    /// The result of `solve`; `None` until it has run
    solution: Option<Solution>,
}

/// The analytic problem, resolved from the settings
struct Problem {
    /// The boundary topology, and the model surface's shape, size and position
    configuration: Configuration,
    /// The flux expansion, the matching constraints, and the model surface they come from
    geometry: MatchingGeometry,
    /// GF's profile parameter, approximately the poloidal beta [dimensionless]
    nu: f64,
    /// GF's pressure normalisation, the pressure where `psi_hat = 1` [pascal]
    p_0: f64,
    /// GF's `B_0`, the vacuum toroidal field at `r_geo`, `b0 * r0 / r_geo` [tesla]
    bt_vac_at_r_geo: f64,
    /// The grid's radial axis [metre]
    r: Array1<f64>,
    /// The grid's vertical axis [metre]
    z: Array1<f64>,
}

/// The result of `solve`
struct Solution {
    /// The eigenvalue [dimensionless]
    alpha: f64,
    /// Beta at `psi_hat = 1`, `2 * mu_0 * p_0 / bt_vac_at_r_geo ** 2` (GF Eq. 6.4) [dimensionless]
    beta_0: f64,
    /// Diamagnetic change in the toroidal field (GF Eq. 6.4) [tesla]
    bt_diamagnetic_shift: f64,
    /// Flux normalisation, `psi = psi_0 * psi_hat`: GF's `Psi_0` (Eq. 6.5) times `2 * pi` [weber]
    psi_0: f64,
    /// The flux solution for `alpha`: the expansion and its coefficients
    flux: FluxSolution,
    /// Analytic flux, `psi_0 * psi_hat`, on every grid cell, shape `(n_z, n_r)` [weber].
    /// Outside the model box `|x| < 1`, `y_lower < y < y_upper` the truncated series is unphysical and
    /// can diverge; it is kept there anyway, so that it can be plotted for comparison
    psi_analytic: Array2<f64>,
    /// 1 inside the analytic plasma, which is the model box where `psi_hat > 0`, and 0 elsewhere;
    /// shape `(n_z, n_r)` [dimensionless]
    mask: Array2<f64>,
}

/// A point in GF's normalised coordinates
struct NormalisedPoint {
    /// `x = (R ** 2 / r_geo ** 2 - 1 - eps ** 2) / (2 * eps)` [dimensionless]
    x: f64,
    /// `y = (Z - z_geo) / a_minor` [dimensionless]
    y: f64,
}

/// A point in the machine
struct MachinePoint {
    /// [metre]
    r: f64,
    /// [metre]
    z: f64,
}

/// The analytic magnetic axis, the maximum of `psi_hat`
struct AnalyticMagneticAxis {
    /// Where it is, in GF's normalised coordinates
    point: NormalisedPoint,
    /// The normalised flux there [dimensionless]
    psi_hat: f64,
}

#[pymethods]
impl GuazzottoFreidberg {
    /// Construct the analytic equilibrium from its settings; `solve` solves it.
    ///
    /// The settings are read into `guazzotto_freidberg_settings`, and the analytic problem is resolved from them. Every setting is
    /// described, in GF's notation, in the XML schemas in `python/analytic_grad_shafranov/guazzotto_freidberg/settings_schema/xsd/equilibrium/`.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings, as a `gsfit_rs.dd_support.ParametersInput`, whose XML is `equilibrium.xml` with each file
    ///   it links to written in its link's place; it is written to `equilibrium_ids`' `code/parameters`, and read into
    ///   `guazzotto_freidberg_settings`
    ///
    /// # Errors
    /// A `ValueError` if the XML cannot be read into `guazzotto_freidberg_settings`: it is not well-formed, it is not `equilibrium.xml`,
    /// it holds an element which is not in the XML schema the structs were generated from, or lacks one, or a value cannot be read as
    /// its type. Or when the settings cannot be solved: a model surface GF's construction rejects; `a_minor` not between 0 and `r_geo`;
    /// a non-positive `nu`, `p_0` or `r0`; a zero `b0`; a grid with fewer than 4 points either way, `r_min <= 0`, an empty extent, or
    /// which does not contain the model surface; `n_psi_norm < 3`; or `n_radial_expansion < 1`.
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        let error = |message: String| PyValueError::new_err(format!("GuazzottoFreidberg: {message}"));
        // The settings, read into `guazzotto_freidberg_settings`
        let parameters_value: &str = &parameters_input.inner.parameters_value;
        let settings: GuazzottoFreidbergSettings =
            GuazzottoFreidbergSettings::from_xml(parameters_value).map_err(|message: String| error(format!("the settings cannot be read: {message}")))?;

        // The model surface
        let configuration: Configuration = Configuration::from_settings(&settings.configuration).map_err(error)?;
        let r_geo: f64 = configuration.r_geo();
        let z_geo: f64 = configuration.z_geo();
        let a_minor: f64 = configuration.a_minor();
        if !(r_geo.is_finite() && r_geo > 0.0) {
            return Err(error(format!("r_geo must be finite and positive, not {r_geo}")));
        }
        if !z_geo.is_finite() {
            return Err(error(format!("z_geo must be finite, not {z_geo}")));
        }
        if !(a_minor > 0.0 && a_minor < r_geo) {
            return Err(error(format!("a_minor must lie between 0 and r_geo = {r_geo}, not {a_minor}")));
        }
        let geometry: MatchingGeometry = matching_geometry(&configuration).map_err(error)?;

        // The profiles and the vacuum toroidal field
        let nu: f64 = settings.profiles.nu;
        let p_0: f64 = settings.profiles.p_0;
        let r0: f64 = settings.vacuum_toroidal_field.r0;
        let b0: f64 = settings.vacuum_toroidal_field.b0;
        if !(nu.is_finite() && nu > 0.0) {
            return Err(error(format!("nu must be finite and positive, not {nu}")));
        }
        if !(p_0.is_finite() && p_0 > 0.0) {
            return Err(error(format!("p_0 must be finite and positive, not {p_0}")));
        }
        if !(r0.is_finite() && r0 > 0.0) {
            return Err(error(format!("r0 must be finite and positive, not {r0}")));
        }
        if !(b0.is_finite() && b0 != 0.0) {
            return Err(error(format!("b0 must be finite and non-zero, not {b0}")));
        }
        // The vacuum field falls off as 1 / R. `r0 / r_geo` is 1 when they are the same, so that `bt_vac_at_r_geo` is then `b0` exactly
        let bt_vac_at_r_geo: f64 = b0 * (r0 / r_geo);

        // The grid, which must contain the model plasma. The GF model surface spans R in [r_geo - a_minor, r_geo + a_minor] and Z in
        // [z_geo + a_minor * y_lower, z_geo + a_minor * y_upper] (maxima / X-points)
        let grid: &GuazzottoFreidbergSettingsGrid = &settings.grid;
        let n_r: usize = grid.n_r;
        let n_z: usize = grid.n_z;
        let r_min: f64 = grid.r_min;
        let r_max: f64 = grid.r_max;
        let z_min: f64 = grid.z_min;
        let z_max: f64 = grid.z_max;
        if n_r < 4 || n_z < 4 || !(r_max > r_min) || !(z_max > z_min) || r_min <= 0.0 {
            return Err(error("the grid needs n_r, n_z >= 4, 0 < r_min < r_max and z_min < z_max".to_string()));
        }
        if grid.n_psi_norm < 3 {
            return Err(error("n_psi_norm must be at least 3".to_string()));
        }
        if settings.numerics.n_radial_expansion < 1 {
            return Err(error("n_radial_expansion must be at least 1".to_string()));
        }
        let plasma_r_inner: f64 = r_geo - a_minor;
        let plasma_r_outer: f64 = r_geo + a_minor;
        let plasma_z_lower: f64 = z_geo + a_minor * geometry.y_lower();
        let plasma_z_upper: f64 = z_geo + a_minor * geometry.y_upper();
        if !(r_min <= plasma_r_inner && r_max >= plasma_r_outer && z_min <= plasma_z_lower && z_max >= plasma_z_upper) {
            return Err(error(format!(
                "the plasma (R in [{plasma_r_inner:.4}, {plasma_r_outer:.4}] m, Z in [{plasma_z_lower:.4}, {plasma_z_upper:.4}] m) is not contained in the grid (R in [{r_min:.4}, {r_max:.4}] m, Z in [{z_min:.4}, {z_max:.4}] m); enlarge the r_min/r_max/z_min/z_max bounds"
            )));
        }
        let r: Array1<f64> = Array1::linspace(r_min, r_max, n_r);
        let z: Array1<f64> = Array1::linspace(z_min, z_max, n_z);

        let problem: Problem = Problem {
            configuration,
            geometry,
            nu,
            p_0,
            bt_vac_at_r_geo,
            r,
            z,
        };
        // The IDS records the settings it is made from in `code/parameters`, as GSFit's `Equilibrium` does; `get_pf_active` fills the rest
        let equilibrium_ids: Equilibrium = Equilibrium {
            code: Code {
                parameters: parameters_value.to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(GuazzottoFreidberg {
            equilibrium_ids,
            guazzotto_freidberg_temporary_storage: GuazzottoFreidbergTemporaryStorage { problem, solution: None },
            guazzotto_freidberg_settings: settings,
        })
    }

    /// Solve the eigenvalue `alpha` and evaluate the analytic flux and plasma mask over the `(R, Z)` grid.
    ///
    /// # Errors
    /// A `RuntimeError` if the matching constraints are singular at the eigenvalue found.
    pub fn solve(&mut self) -> PyResult<()> {
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;
        let eps: f64 = problem.configuration.eps();
        let r_geo: f64 = problem.configuration.r_geo();
        let nu: f64 = problem.nu;
        let p_0: f64 = problem.p_0;
        let bt_vac_at_r_geo: f64 = problem.bt_vac_at_r_geo;
        let n_radial_expansion: usize = self.guazzotto_freidberg_settings.numerics.n_radial_expansion;

        // 1. The eigenvalue alpha, and the final expansion coefficients and flux solution
        let alpha: f64 = solve_alpha(eps, nu, &problem.geometry, n_radial_expansion, ALPHA_SEARCH_MIN, ALPHA_SEARCH_MAX);
        let flux: FluxSolution = flux_solution(alpha, eps, nu, &problem.geometry, n_radial_expansion)
            .ok_or_else(|| PyRuntimeError::new_err(format!("GuazzottoFreidberg: the matching constraints are singular at alpha = {alpha}")))?;

        // 2. Dimensional scalars (GF Eqs. 6.4, 6.5). GF's `Psi_0` is per radian, so it is multiplied by `2 * pi` to give the total flux
        let beta_0: f64 = 2.0 * MU_0 * p_0 / (bt_vac_at_r_geo * bt_vac_at_r_geo);
        let bt_diamagnetic_shift: f64 = bt_vac_at_r_geo * 0.5 * beta_0 * (1.0 + eps * eps) * (1.0 - nu) / nu;
        let psi_0: f64 = 2.0 * PI * eps * bt_vac_at_r_geo * r_geo * r_geo / alpha * (beta_0 / nu).sqrt();

        // 3. The analytic flux on every cell, and the plasma mask. The plasma lies inside the
        //    model box |x| < 1, y_lower < y < y_upper, where the C_n / S_n series is reliable. Outside
        //    it the truncated series is unphysical (it can diverge near |x| = 1 / eps_hat or oscillate
        //    beyond the maxima / X-points), so those points are never part of the plasma. The box also
        //    cuts off the private flux region below an X-point, where psi_hat > 0 too.
        let n_r: usize = problem.r.len();
        let n_z: usize = problem.z.len();
        let mut psi_analytic: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut mask: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                let point: NormalisedPoint = self.normalised_coordinates(problem.r[i_r], problem.z[i_z]);
                let psi_hat: f64 = flux.eval_all(point.x, point.y).psi_hat;
                psi_analytic[(i_z, i_r)] = psi_0 * psi_hat;
                if problem.geometry.is_in_model_box(point.x, point.y) && psi_hat > 0.0 {
                    mask[(i_z, i_r)] = 1.0;
                }
            }
        }

        self.guazzotto_freidberg_temporary_storage.solution = Some(Solution {
            alpha,
            beta_0,
            bt_diamagnetic_shift,
            psi_0,
            flux,
            psi_analytic,
            mask,
        });
        Ok(())
    }

    /// Fit the PF-coil currents on the `(R, Z)` grid boundary so their vacuum field, added to the
    /// plasma's own field, reproduces the analytic flux at the `control_points`. Then store the
    /// resulting free-boundary equilibrium in the Equilibrium IDS, and post-process it. Requires
    /// `solve` first.
    ///
    /// The IDS holds the total flux, plasma plus coils, which is valid in the vacuum as well as
    /// inside the plasma. For a diverted plasma the magnetic axis, the X-points, the boundary flux
    /// and the plasma mask are found in that flux, as the Grad-Shafranov solver finds them; for a
    /// limited one they are the analytic ones. See `build_equilibrium_ids`.
    ///
    /// # Arguments
    /// * `coil_regularisation_weight` - Tikhonov weight `lambda^2` on the coil-current magnitude
    /// * `control_points` - `(n_control_point, 2)` array of `(r, z)` points where the flux is matched [metre]
    /// * `wall` - the wall: its limiter is copied into the Equilibrium IDS, where its limiter and vacuum vessel
    ///   bound a diverted plasma, and the post-processor traces the scrape-off layer legs up to it
    ///
    /// # Returns
    /// * `pf_active` - one single-filament PF coil per boundary node, carrying the fitted current, at the settings' `time`
    ///
    /// # Errors
    /// A `ValueError` if `control_points` is not `(n_control_point, 2)`. A `RuntimeError` if `solve` has not run, or if the free-boundary
    /// equilibrium cannot be built: no vacuum vessel in the wall, or no magnetic axis or boundary found in the total flux.
    pub fn get_pf_active(&mut self, coil_regularisation_weight: f64, control_points: PyReadonlyArray2<f64>, wall: PyRef<Wall>) -> PyResult<PfActive> {
        let solution: &Solution = self.solved()?;
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;

        let control_points_ndarray: Array2<f64> = control_points.to_owned_array();
        if control_points_ndarray.ncols() != 2 {
            return Err(PyValueError::new_err(
                "GuazzottoFreidberg.get_pf_active: control_points must have shape (n_control_point, 2)",
            ));
        }
        let control_r: Array1<f64> = control_points_ndarray.column(0).to_owned();
        let control_z: Array1<f64> = control_points_ndarray.column(1).to_owned();
        let n_control: usize = control_r.len();

        // Target flux at the control points: psi_0 * psi_hat inside the model box, 0 on the LCFS and outside.
        let mut psi_control: Array1<f64> = Array1::zeros(n_control);
        for i_control in 0..n_control {
            let point: NormalisedPoint = self.normalised_coordinates(control_r[i_control], control_z[i_control]);
            if problem.geometry.is_in_model_box(point.x, point.y) {
                psi_control[i_control] = solution.psi_0 * solution.flux.eval_all(point.x, point.y).psi_hat;
            }
        }

        // Fit the boundary coil currents and recompute the total flux (plasma + coils).
        let j_phi_2d: Array2<f64> = self.plasma_current_density(solution);
        let fit: BoundaryCoilFit = fit_boundary_coils(
            &problem.r,
            &problem.z,
            &j_phi_2d,
            &solution.mask,
            &control_r,
            &control_z,
            &psi_control,
            coil_regularisation_weight,
        );

        // Assemble the PF coils: one single-filament coil per boundary node, at the time of the equilibrium. The analytic equilibrium has no
        // settings to record in `pf_active/code/parameters`
        let mut pf_active: PfActive = PfActive {
            pf_active_ids: PfActiveIds::default(),
            settings_pf_active: SettingsPfActive { workflow: None },
        };
        let n_coil: usize = fit.coil_r.len();
        let time_single: Array1<f64> = Array1::from_elem(1, self.guazzotto_freidberg_settings.time);
        for i_coil in 0..n_coil {
            let name: String = format!("GF_BND_{i_coil:04}");
            let r_single: Array1<f64> = Array1::from_elem(1, fit.coil_r[i_coil]);
            let z_single: Array1<f64> = Array1::from_elem(1, fit.coil_z[i_coil]);
            let d_r_single: Array1<f64> = Array1::from_elem(1, fit.coil_d_r[i_coil]);
            let d_z_single: Array1<f64> = Array1::from_elem(1, fit.coil_d_z[i_coil]);
            let current_single: Array1<f64> = Array1::from_elem(1, fit.coil_current[i_coil]);
            pf_active
                .add_pf_coil_rs(&name, r_single, z_single, d_r_single, d_z_single, time_single.clone(), current_single)
                .map_err(|reason: String| PyRuntimeError::new_err(format!("GuazzottoFreidberg.get_pf_active: {reason}")))?;
        }

        // Store the free-boundary equilibrium, and post-process it exactly as a reconstruction is
        let equilibrium_ids: Equilibrium = self
            .build_equilibrium_ids(solution, fit, j_phi_2d, &wall.wall_ids)
            .map_err(|reason: String| PyRuntimeError::new_err(format!("GuazzottoFreidberg.get_pf_active: {reason}")))?;
        self.equilibrium_ids = equilibrium_ids;
        let (p_prime_source_function, ff_prime_source_function): (SharedSourceFunction, SharedSourceFunction) = source_functions();
        // It has no magnetic sensors to reconstruct, and so no Greens tables
        equilibrium_post_processor(
            &mut self.equilibrium_ids,
            &GreensTables::default(),
            &p_prime_source_function,
            &ff_prime_source_function,
        );

        Ok(pf_active)
    }

    /// Fill the magnetic sensors' measurements with what the analytic plasma current plus `pf_active` produce at each sensor. Requires
    /// `solve` first. The sensors must already be in `magnetics`, e.g. added with `Magnetics.add_flux_loop` and
    /// `Magnetics.add_b_field_pol_probe`; their measurements are overwritten.
    ///
    /// Each coil filament carries its coil's current at the first time of its experimental timebase, so each measurement is a signal
    /// with that one time, the first coil's, or the settings' `time` without any coils.
    ///
    /// The values are physical, as GSFit's own Greens tables give them: `magnetics/flux_loop(i1)/flux` is `Greens::psi`
    /// (= `2 pi R A_phi`, weber) at `position(0)`, and `magnetics/b_field_pol_probe(i1)/field` is
    /// `B_r cos(poloidal_angle) + B_z sin(poloidal_angle)` (tesla) at `position`, with GSFit's COCOS 13 `poloidal_angle`.
    ///
    /// # Arguments
    /// * `magnetics` - the magnetic sensors, with their geometry already set; their measurements are overwritten
    /// * `pf_active` - the PF coils, typically the ones `get_pf_active` returned
    ///
    /// # Errors
    /// A `RuntimeError` if `solve` has not run, or if the Greens functions between the sensors and the plasma or the coils cannot be
    /// calculated.
    pub fn get_sensor_values(&self, mut magnetics: PyRefMut<Magnetics>, pf_active: PyRef<PfActive>) -> PyResult<()> {
        let solution: &Solution = self.solved()?;
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;
        let error = |reason: String| PyRuntimeError::new_err(format!("GuazzottoFreidberg.get_sensor_values: {reason}"));

        // Plasma current filaments + the coils are the free-boundary field the sensors see.
        let r: &Array1<f64> = &problem.r;
        let z: &Array1<f64> = &problem.z;
        let j_phi_2d: Array2<f64> = self.plasma_current_density(solution);
        let (cell_r, cell_z, cell_current): (Array1<f64>, Array1<f64>, Array1<f64>) = plasma_current_filaments(r, z, &j_phi_2d, &solution.mask);
        let n_cell: usize = cell_r.len();

        // Every filament of every PF coil, each carrying its coil's current
        let mut coil_r_vec: Vec<f64> = Vec::new();
        let mut coil_z_vec: Vec<f64> = Vec::new();
        let mut coil_width_vec: Vec<f64> = Vec::new();
        let mut coil_height_vec: Vec<f64> = Vec::new();
        let mut coil_current_vec: Vec<f64> = Vec::new();
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_pf_coil: usize = pf_active_coils.len();
        for i_pf_coil in 0..n_pf_coil {
            let filament_r: Array1<f64> = pf_active_coils[i_pf_coil].element(..).geometry.rectangle.r.to_array();
            let filament_z: Array1<f64> = pf_active_coils[i_pf_coil].element(..).geometry.rectangle.z.to_array();
            let filament_width: Array1<f64> = pf_active_coils[i_pf_coil].element(..).geometry.rectangle.width.to_array();
            let filament_height: Array1<f64> = pf_active_coils[i_pf_coil].element(..).geometry.rectangle.height.to_array();
            let coil_current: f64 = pf_active_coils[i_pf_coil].current.data[0];
            let n_filament: usize = filament_r.len();
            for i_filament in 0..n_filament {
                coil_r_vec.push(filament_r[i_filament]);
                coil_z_vec.push(filament_z[i_filament]);
                coil_width_vec.push(filament_width[i_filament]);
                coil_height_vec.push(filament_height[i_filament]);
                coil_current_vec.push(coil_current);
            }
        }
        let coil_width: Array1<f64> = Array1::from(coil_width_vec);
        let coil_height: Array1<f64> = Array1::from(coil_height_vec);
        let coil_current: Array1<f64> = Array1::from(coil_current_vec);
        let n_coil: usize = coil_width.len();

        // Sensors sit away from the current sources, so the sources are points, `ConductorModel::Point`, matching
        // GSFit's own sensor path. As obliques, the cells and the coils' filaments are described from their corners
        // at the smallest r and z
        let d_r_cell: f64 = r[1] - r[0];
        let d_z_cell: f64 = z[1] - z[0];
        let cells: Conductor = Conductor::from_oblique(
            &(&cell_r - 0.5 * d_r_cell),
            &(&cell_z - 0.5 * d_z_cell),
            &Array1::from_elem(n_cell, d_r_cell),
            &Array1::from_elem(n_cell, d_z_cell),
            &Array1::zeros(n_cell),
            &Array1::zeros(n_cell),
        )
        .map_err(|reason: String| error(format!("the plasma's cells: {reason}")))?;
        let coils: Conductor = Conductor::from_oblique(
            &(Array1::from(coil_r_vec) - 0.5 * &coil_width),
            &(Array1::from(coil_z_vec) - 0.5 * &coil_height),
            &coil_width,
            &coil_height,
            &Array1::zeros(n_coil),
            &Array1::zeros(n_coil),
        )
        .map_err(|reason: String| error(format!("the PF coils' filaments: {reason}")))?;

        // The measurements are at the time the coils' currents are taken at
        let time: f64 = if n_pf_coil > 0 {
            pf_active_coils[0].current.time[0]
        } else {
            self.guazzotto_freidberg_settings.time
        };

        // --- Flux loops: measured psi = Greens::psi() @ (plasma cells + coils)  [weber] ---
        let n_flux_loop: usize = magnetics.magnetics_ids.flux_loop.len();
        if n_flux_loop > 0 {
            let mut flux_loop_r: Array1<f64> = Array1::zeros(n_flux_loop);
            let mut flux_loop_z: Array1<f64> = Array1::zeros(n_flux_loop);
            for i_flux_loop in 0..n_flux_loop {
                flux_loop_r[i_flux_loop] = magnetics.magnetics_ids.flux_loop[i_flux_loop].position[0].r;
                flux_loop_z[i_flux_loop] = magnetics.magnetics_ids.flux_loop[i_flux_loop].position[0].z;
            }
            let flux_loops: Sensor = Sensor::new(&flux_loop_r, &flux_loop_z).map_err(|reason: String| error(format!("the flux loops: {reason}")))?;
            let psi_from_cells: Array1<f64> = Greens::sensor_to_conductor(&flux_loops, &cells, ConductorModel::Point)
                .and_then(|greens_calculator: Greens| greens_calculator.psi())
                .map_err(|reason: String| error(format!("the flux loops' Greens with the plasma: {reason}")))?
                .dot(&cell_current);
            let psi_from_coils: Array1<f64> = Greens::sensor_to_conductor(&flux_loops, &coils, ConductorModel::Point)
                .and_then(|greens_calculator: Greens| greens_calculator.psi())
                .map_err(|reason: String| error(format!("the flux loops' Greens with the PF coils: {reason}")))?
                .dot(&coil_current);
            for i_flux_loop in 0..n_flux_loop {
                magnetics.magnetics_ids.flux_loop[i_flux_loop].flux = SignalFlt1dValidity {
                    data: array![psi_from_cells[i_flux_loop] + psi_from_coils[i_flux_loop]],
                    time: array![time],
                    ..Default::default()
                };
            }
        }

        // --- BP probes: measured B = B_r cos(poloidal_angle) + B_z sin(poloidal_angle)  [tesla] ---
        let n_bp_probe: usize = magnetics.magnetics_ids.b_field_pol_probe.len();
        if n_bp_probe > 0 {
            let mut bp_probe_r: Array1<f64> = Array1::zeros(n_bp_probe);
            let mut bp_probe_z: Array1<f64> = Array1::zeros(n_bp_probe);
            let mut bp_probe_poloidal_angle: Array1<f64> = Array1::zeros(n_bp_probe);
            for i_bp_probe in 0..n_bp_probe {
                bp_probe_r[i_bp_probe] = magnetics.magnetics_ids.b_field_pol_probe[i_bp_probe].position.r;
                bp_probe_z[i_bp_probe] = magnetics.magnetics_ids.b_field_pol_probe[i_bp_probe].position.z;
                bp_probe_poloidal_angle[i_bp_probe] = magnetics.magnetics_ids.b_field_pol_probe[i_bp_probe].poloidal_angle;
            }
            let bp_probes: Sensor = Sensor::new(&bp_probe_r, &bp_probe_z).map_err(|reason: String| error(format!("the BP probes: {reason}")))?;
            let greens_cells: Greens = Greens::sensor_to_conductor(&bp_probes, &cells, ConductorModel::Point)
                .map_err(|reason: String| error(format!("the BP probes' Greens with the plasma: {reason}")))?;
            let greens_coils: Greens = Greens::sensor_to_conductor(&bp_probes, &coils, ConductorModel::Point)
                .map_err(|reason: String| error(format!("the BP probes' Greens with the PF coils: {reason}")))?;
            let b_r_total: Array1<f64> = greens_cells.b_r().map_err(|reason: String| error(format!("b_r: {reason}")))?.dot(&cell_current)
                + greens_coils.b_r().map_err(|reason: String| error(format!("b_r: {reason}")))?.dot(&coil_current);
            let b_z_total: Array1<f64> = greens_cells.b_z().map_err(|reason: String| error(format!("b_z: {reason}")))?.dot(&cell_current)
                + greens_coils.b_z().map_err(|reason: String| error(format!("b_z: {reason}")))?.dot(&coil_current);
            for i_bp_probe in 0..n_bp_probe {
                let value: f64 =
                    b_r_total[i_bp_probe] * bp_probe_poloidal_angle[i_bp_probe].cos() + b_z_total[i_bp_probe] * bp_probe_poloidal_angle[i_bp_probe].sin();
                magnetics.magnetics_ids.b_field_pol_probe[i_bp_probe].field = SignalFlt1dValidity {
                    data: array![value],
                    time: array![time],
                    ..Default::default()
                };
            }
        }
        Ok(())
    }

    /// GF's model surface, the shape whose position, slope and curvature the analytic boundary is
    /// matched to at the matching points: the red curve in GF's figures. Elsewhere the analytic
    /// boundary is close to the model surface, but not on it. Does not need `solve`.
    ///
    /// The upper half is sampled uniformly in GF's angle-like parameter `theta`, from the outer
    /// midplane over the top to the inner midplane, and the lower half likewise back to the outer
    /// midplane. A half with an X-point is two ellipses, each sampled with half of the half's points.
    ///
    /// # Arguments
    /// * `n_point_per_half` - number of points on each of the upper and lower halves; must be even
    ///
    /// # Returns
    /// * `(r, z)` - the closed outline, anticlockwise from the outer midplane; its last point repeats
    ///   the first, so each is `2 * n_point_per_half + 1` long [metre]
    fn model_surface<'py>(&self, py: Python<'py>, n_point_per_half: usize) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let geometry: &MatchingGeometry = &self.guazzotto_freidberg_temporary_storage.problem.geometry;
        let (upper_x, upper_y): (Array1<f64>, Array1<f64>) = geometry.upper_half.outline(true, n_point_per_half);
        let (lower_x, lower_y): (Array1<f64>, Array1<f64>) = geometry.lower_half.outline(false, n_point_per_half);

        let n_point: usize = 2 * n_point_per_half + 1;
        let mut r: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        let mut z: Array1<f64> = Array1::from_elem(n_point, f64::NAN);
        for i_point in 0..n_point_per_half {
            let upper_point: MachinePoint = self.physical_coordinates(upper_x[i_point], upper_y[i_point]);
            r[i_point] = upper_point.r;
            z[i_point] = upper_point.z;
            let lower_point: MachinePoint = self.physical_coordinates(lower_x[i_point], lower_y[i_point]);
            r[n_point_per_half + i_point] = lower_point.r;
            z[n_point_per_half + i_point] = lower_point.z;
        }
        r[n_point - 1] = r[0];
        z[n_point - 1] = z[0];
        (r.into_pyarray(py), z.into_pyarray(py))
    }

    /// The eigenvalue [dimensionless]; a `RuntimeError` until `solve` has run
    #[getter]
    fn alpha(&self) -> PyResult<f64> {
        Ok(self.solved()?.alpha)
    }

    /// Beta at `psi_hat = 1`, `2 * mu_0 * p_0 / B_0 ** 2` (GF Eq. 6.4), with GF's `B_0` the vacuum toroidal field at `r_geo`
    /// [dimensionless]; a `RuntimeError` until `solve` has run
    #[getter]
    fn beta_0(&self) -> PyResult<f64> {
        Ok(self.solved()?.beta_0)
    }

    /// Diamagnetic change in the toroidal field (GF Eq. 6.4) [tesla]; a `RuntimeError` until `solve` has run
    #[getter]
    fn bt_diamagnetic_shift(&self) -> PyResult<f64> {
        Ok(self.solved()?.bt_diamagnetic_shift)
    }

    /// Flux normalisation, `psi = psi_0 * psi_hat`: GF's `Psi_0` (Eq. 6.5) times `2 * pi` [weber]; a `RuntimeError` until `solve` has run
    #[getter]
    fn psi_0(&self) -> PyResult<f64> {
        Ok(self.solved()?.psi_0)
    }

    /// The analytic flux, `psi_0 * psi_hat`, on every grid cell, shape `(n_z, n_r)` [weber]; a `RuntimeError` until `solve` has run.
    ///
    /// Outside the model box the truncated series is unphysical and can diverge; use `mask` to
    /// select the plasma.
    #[getter]
    fn psi_analytic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.solved()?.psi_analytic.clone().into_pyarray(py))
    }

    /// 1 inside the analytic plasma and 0 elsewhere, shape `(n_z, n_r)` [dimensionless]; a `RuntimeError` until `solve` has run
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.solved()?.mask.clone().into_pyarray(py))
    }

    /// The Equilibrium IDS, for reading with `gsfit_rs.imas.equilibrium_paths`. Only `code/parameters`, the settings, until
    /// `get_pf_active` has run.
    ///
    /// The IDS is copied into the returned object, so it is a snapshot: changes made on the Rust
    /// side afterwards are not seen by it.
    #[getter]
    fn equilibrium_ids(&self) -> PyEquilibrium {
        PyEquilibrium::new(self.equilibrium_ids.clone())
    }
}

// Rust only methods
impl GuazzottoFreidberg {
    /// The solution, or a `RuntimeError` naming `solve` when it has not run
    fn solved(&self) -> PyResult<&Solution> {
        self.guazzotto_freidberg_temporary_storage
            .solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GuazzottoFreidberg: call solve() first"))
    }

    /// GF's normalised coordinates of a point in the machine, `(R, Z)` [metre]
    fn normalised_coordinates(&self, r: f64, z: f64) -> NormalisedPoint {
        let configuration: &Configuration = &self.guazzotto_freidberg_temporary_storage.problem.configuration;
        let r_geo: f64 = configuration.r_geo();
        let eps: f64 = configuration.eps();
        NormalisedPoint {
            x: (r * r / (r_geo * r_geo) - 1.0 - eps * eps) / (2.0 * eps),
            y: (z - configuration.z_geo()) / configuration.a_minor(),
        }
    }

    /// The inverse of `normalised_coordinates`: the point in the machine at `(x, y)` [dimensionless]
    fn physical_coordinates(&self, x: f64, y: f64) -> MachinePoint {
        let configuration: &Configuration = &self.guazzotto_freidberg_temporary_storage.problem.configuration;
        let eps: f64 = configuration.eps();
        MachinePoint {
            r: configuration.r_geo() * (1.0 + eps * eps + 2.0 * eps * x).sqrt(),
            z: configuration.z_geo() + configuration.a_minor() * y,
        }
    }

    /// The analytic plasma's toroidal current density on the grid, shape `(n_z, n_r)` [ampere / metre ** 2].
    fn plasma_current_density(&self, solution: &Solution) -> Array2<f64> {
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;
        let psi_hat_2d: Array2<f64> = &solution.psi_analytic / solution.psi_0;
        plasma_current_density(
            &problem.r,
            &psi_hat_2d,
            &solution.mask,
            solution.psi_0,
            problem.p_0,
            problem.configuration.r_geo(),
            problem.bt_vac_at_r_geo,
            solution.bt_diamagnetic_shift,
        )
    }

    /// The analytic magnetic axis, the maximum of `psi_hat`, found by Newton iteration on
    /// `grad(psi_hat) = 0` from the plasma's highest grid cell.
    ///
    /// # Errors
    /// When the Newton iteration does not converge within `N_ITER_MAGNETIC_AXIS_MAX` iterations.
    fn magnetic_axis(&self, solution: &Solution) -> Result<AnalyticMagneticAxis, String> {
        // Start from the highest cell inside the plasma
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;
        let n_r: usize = problem.r.len();
        let n_z: usize = problem.z.len();
        let mut psi_max: f64 = f64::NEG_INFINITY;
        let mut r_start: f64 = f64::NAN;
        let mut z_start: f64 = f64::NAN;
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                if solution.mask[(i_z, i_r)] > 0.0 && solution.psi_analytic[(i_z, i_r)] > psi_max {
                    psi_max = solution.psi_analytic[(i_z, i_r)];
                    r_start = problem.r[i_r];
                    z_start = problem.z[i_z];
                }
            }
        }
        let start: NormalisedPoint = self.normalised_coordinates(r_start, z_start);
        let mut x: f64 = start.x;
        let mut y: f64 = start.y;

        // The map from (R, Z) to (x, y) is one-to-one, so the stationary point is the same in both
        let mut converged: bool = false;
        'newton_loop: for _i_iter in 0..N_ITER_MAGNETIC_AXIS_MAX {
            let derivatives: FluxDerivatives = solution.flux.eval_all(x, y);
            let hessian_determinant: f64 =
                derivatives.d2_psi_hat_d_x2 * derivatives.d2_psi_hat_d_y2 - derivatives.d2_psi_hat_d_x_d_y * derivatives.d2_psi_hat_d_x_d_y;
            let step_x: f64 =
                (derivatives.d2_psi_hat_d_y2 * derivatives.d_psi_hat_d_x - derivatives.d2_psi_hat_d_x_d_y * derivatives.d_psi_hat_d_y) / hessian_determinant;
            let step_y: f64 =
                (derivatives.d2_psi_hat_d_x2 * derivatives.d_psi_hat_d_y - derivatives.d2_psi_hat_d_x_d_y * derivatives.d_psi_hat_d_x) / hessian_determinant;
            x -= step_x;
            y -= step_y;
            if step_x.hypot(step_y) < MAGNETIC_AXIS_TOLERANCE {
                converged = true;
                break 'newton_loop;
            }
        }
        if !converged {
            return Err(format!(
                "the Newton iteration for the analytic magnetic axis did not converge within {N_ITER_MAGNETIC_AXIS_MAX} iterations"
            ));
        }

        Ok(AnalyticMagneticAxis {
            point: NormalisedPoint { x, y },
            psi_hat: solution.flux.eval_all(x, y).psi_hat,
        })
    }

    /// Build the Equilibrium IDS for the free-boundary equilibrium: one time-slice, at the settings' `time`.
    ///
    /// Filled with everything the equilibrium post-processor reads, the same as the Grad-Shafranov
    /// solver leaves it. The flux and its derivatives are the total, plasma plus coils.
    ///
    /// For a diverted plasma the magnetic axis, the X-points, the boundary flux and the mask are
    /// found in the total flux, by the same functions the Grad-Shafranov solver uses, so that they
    /// agree with the flux the post-processor traces the boundary in. The analytic ones do not quite:
    /// the total flux differs from the analytic flux by the coil fit's residual, and near an X-point,
    /// where the flux is flat, that is as large as the flux itself, so the traced boundary would
    /// not pass through the analytic X-points. A limited GF plasma touches no wall and has no
    /// X-point, so there is nothing in the flux to find its boundary by, and the analytic one is
    /// kept: `psi_hat = 0`, with the analytic mask.
    ///
    /// The source functions are one-degree-of-freedom `EfitPolynomial`s, whose basis `1 - psi_norm` is
    /// exactly the GF profile shape, so the post-processor's `pressure`, `f`, ... are the analytic
    /// profiles. See `source_functions`.
    ///
    /// # Errors
    /// For a diverted plasma, when the wall has no limiter units, or no magnetic axis or boundary is found in the total flux; and
    /// when the analytic magnetic axis cannot be found.
    fn build_equilibrium_ids(&self, solution: &Solution, fit: BoundaryCoilFit, j_phi_2d: Array2<f64>, wall_ids: &WallIds) -> Result<Equilibrium, String> {
        let problem: &Problem = &self.guazzotto_freidberg_temporary_storage.problem;
        let settings: &GuazzottoFreidbergSettings = &self.guazzotto_freidberg_settings;
        let r: &Array1<f64> = &problem.r;
        let z: &Array1<f64> = &problem.z;
        let n_r: usize = r.len();
        let n_z: usize = z.len();
        let d_area: f64 = (r[1] - r[0]) * (z[1] - z[0]);

        // The IDS starts with the settings it was constructed with, kept in `code/parameters`, and the limiter, copied in from the wall as
        // `Equilibrium.copy_from_wall_ids` copies it into a reconstruction's, because a diverted plasma's boundary is found inside it below
        let mut equilibrium_ids: Equilibrium = Equilibrium::default();
        equilibrium_ids.code.parameters = self.equilibrium_ids.code.parameters.clone();
        let wall_units: &Vec<Wall2dLimiterUnit> = &wall_ids.description_2d[0].limiter.unit;
        let n_unit: usize = wall_units.len();
        for i_unit in 0..n_unit {
            equilibrium_ids.limiter.unit.push(EquilibriumLimiterUnit {
                name: wall_units[i_unit].name.clone(),
                outline: EquilibriumLimiterUnitOutline {
                    r: wall_units[i_unit].outline.r.clone(),
                    z: wall_units[i_unit].outline.z.clone(),
                },
            });
        }

        // Analytic magnetic axis
        let analytic_axis: AnalyticMagneticAxis = self.magnetic_axis(solution)?;
        let analytic_axis_machine: MachinePoint = self.physical_coordinates(analytic_axis.point.x, analytic_axis.point.y);
        let mag_r_analytic: f64 = analytic_axis_machine.r;
        let mag_z_analytic: f64 = analytic_axis_machine.z;
        let psi_a_analytic: f64 = solution.psi_0 * analytic_axis.psi_hat;

        // Magnetic axis, boundary and stationary points
        let mag_r: f64;
        let mag_z: f64;
        let psi_a: f64;
        let psi_b: f64;
        let boundary_type: i32;
        let bounding_r: f64;
        let bounding_z: f64;
        let mask_2d: Array2<f64>;
        let contour_tree_nodes: Vec<EquilibriumContourTreeNode>;
        if !problem.geometry.xpts().0.is_empty() {
            // Diverted: found in the total flux, as `EquilibriumSolver::update_psi_and_geometry` finds them, inside the limiter just
            // copied in. Its `unit(0)` is the vacuum vessel, and the candidate limit points are every point of every unit, in unit order
            let limiter: &EquilibriumLimiter = &equilibrium_ids.limiter;
            if limiter.unit.is_empty() {
                return Err("`wall/description_2d(0)/limiter/unit` is empty, so there is no vacuum vessel to find the boundary inside".to_string());
            }
            let vessel_r: &Array1<f64> = &limiter.unit[0].outline.r;
            let vessel_z: &Array1<f64> = &limiter.unit[0].outline.z;
            let n_limiter_unit: usize = limiter.unit.len();
            let mut n_limit_pt: usize = 0;
            for i_unit in 0..n_limiter_unit {
                n_limit_pt += limiter.unit[i_unit].outline.r.len();
            }
            let mut limit_pts_r: Array1<f64> = Array1::from_elem(n_limit_pt, f64::NAN);
            let mut limit_pts_z: Array1<f64> = Array1::from_elem(n_limit_pt, f64::NAN);
            let mut i_limit_pt: usize = 0;
            for i_unit in 0..n_limiter_unit {
                let unit_outline: &EquilibriumLimiterUnitOutline = &limiter.unit[i_unit].outline;
                let n_point_this_unit: usize = unit_outline.r.len();
                for i_point in 0..n_point_this_unit {
                    limit_pts_r[i_limit_pt] = unit_outline.r[i_point];
                    limit_pts_z[i_limit_pt] = unit_outline.z[i_point];
                    i_limit_pt += 1;
                }
            }
            let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
                r.view(),
                z.view(),
                fit.flux_total.psi.view(),
                fit.flux_total.d_psi_d_r.view(),
                fit.flux_total.d_psi_d_z.view(),
                fit.flux_total.d2_psi_d_r2.view(),
                fit.flux_total.d2_psi_d_r_d_z.view(),
                fit.flux_total.d2_psi_d_z2.view(),
            );
            // The search for the magnetic axis starts from the analytic one
            let magnetic_axis: MagneticAxis = find_magnetic_axis(&stationary_points, mag_r_analytic, mag_z_analytic, vessel_r, vessel_z)
                .map_err(|reason: String| format!("no magnetic axis found in the total flux: {reason}"))?;
            let plasma_boundary: BoundaryContour = find_boundary(
                r,
                z,
                &fit.flux_total.psi,
                &fit.flux_total.d_psi_d_r,
                &fit.flux_total.d_psi_d_z,
                &fit.flux_total.d2_psi_d_r_d_z,
                &stationary_points,
                &limit_pts_r,
                &limit_pts_z,
                vessel_r,
                vessel_z,
                magnetic_axis.r,
                magnetic_axis.z,
            )
            .map_err(|reason| format!("no plasma boundary found in the total flux: {reason:?}"))?;
            mag_r = magnetic_axis.r;
            mag_z = magnetic_axis.z;
            psi_a = magnetic_axis.psi;
            psi_b = plasma_boundary.bounding_psi;
            boundary_type = plasma_boundary.xpt_diverted as i32;
            bounding_r = plasma_boundary.bounding_r;
            bounding_z = plasma_boundary.bounding_z;
            mask_2d = plasma_boundary
                .mask
                .ok_or_else(|| "the plasma boundary found in the total flux has no mask".to_string())?;
            contour_tree_nodes = grad_shafranov_contour_tree_nodes(&stationary_points);
        } else {
            // Limited: the analytic boundary, `psi_hat = 0`. The inboard mid-plane, `(x, y) = (-1, 0)`, is
            // one of GF's matching points on the boundary
            mag_r = mag_r_analytic;
            mag_z = mag_z_analytic;
            psi_a = psi_a_analytic;
            psi_b = 0.0;
            boundary_type = BOUNDARY_TYPE_LIMITED;
            let bounding_point: MachinePoint = self.physical_coordinates(-1.0, 0.0);
            bounding_r = bounding_point.r;
            bounding_z = bounding_point.z;
            mask_2d = solution.mask.clone();
            contour_tree_nodes = vec![EquilibriumContourTreeNode {
                critical_type: CRITICAL_TYPE_MAXIMUM,
                r: mag_r,
                z: mag_z,
                psi: psi_a,
                ..Default::default()
            }];
        }

        // The IDS layout, the same as a reconstruction's: one time-slice, holding one rectangular grid
        equilibrium_ids.allocate_time_slices(&array![settings.time]);
        // An analytic solution is always usable
        equilibrium_ids.code.output_flag = array![0];
        // `r0 * b0` is the vacuum `f`, which is what the post-processor recovers the rod current from
        equilibrium_ids.vacuum_toroidal_field.r0 = settings.vacuum_toroidal_field.r0;
        equilibrium_ids.vacuum_toroidal_field.b0 = array![settings.vacuum_toroidal_field.b0];

        let time_slice: &mut EquilibriumTimeSlice = &mut equilibrium_ids.time_slice[0];
        // The normalised flux the source functions are evaluated on
        time_slice.profiles_1d.psi_norm = Array1::linspace(0.0, 1.0, settings.grid.n_psi_norm);

        // `EfitPolynomial` coefficients which reproduce the GF profiles. With `psi = psi_a * (1 - psi_norm)`:
        // p' = 2 * p_0 * psi / psi_0 ** 2 = (2 * p_0 * psi_a / psi_0 ** 2) * (1 - psi_norm)
        // ff' = 2 * r_geo ** 2 * b_0 * d_b * psi / psi_0 ** 2 = (2 * r_geo ** 2 * b_0 * d_b * psi_a / psi_0 ** 2) * (1 - psi_norm)
        // `psi_a` is the analytic one, as these are the analytic profiles
        let r_geo: f64 = problem.configuration.r_geo();
        let psi_0_squared: f64 = solution.psi_0 * solution.psi_0;
        time_slice.source_functions.p_prime.coefficients = array![2.0 * problem.p_0 * psi_a_analytic / psi_0_squared];
        time_slice.source_functions.ff_prime.coefficients =
            array![2.0 * r_geo * r_geo * problem.bt_vac_at_r_geo * solution.bt_diamagnetic_shift * psi_a_analytic / psi_0_squared];

        time_slice.global_quantities.magnetic_axis.r = mag_r;
        time_slice.global_quantities.magnetic_axis.z = mag_z;
        time_slice.global_quantities.psi_magnetic_axis = psi_a;
        time_slice.global_quantities.ip = j_phi_2d.sum() * d_area;
        time_slice.boundary.psi = psi_b;
        time_slice.boundary.bounding.r = bounding_r;
        time_slice.boundary.bounding.z = bounding_z;
        time_slice.boundary.r#type = boundary_type;
        time_slice.contour_tree.node = contour_tree_nodes;
        time_slice.convergence.result.name = "converged".to_string();
        time_slice.convergence.result.index = CONVERGENCE_STATUS_CONVERGED;
        time_slice.convergence.result.description = "analytic Guazzotto-Freidberg equilibrium".to_string();

        // `rectangular` is index 1 of the data dictionary's poloidal plane coordinates enumeration
        let mut profiles_2d: EquilibriumProfiles2d = EquilibriumProfiles2d::default();
        profiles_2d.grid_type.name = "rectangular".to_string();
        profiles_2d.grid_type.index = 1;
        profiles_2d.grid_type.description = "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".to_string();
        profiles_2d.grid.dim1 = r.to_owned();
        profiles_2d.grid.dim2 = z.to_owned();
        profiles_2d.grid.d_area = d_area;
        profiles_2d.r = Array2::from_shape_fn((n_z, n_r), |(_i_z, i_r)| r[i_r]);
        profiles_2d.z = Array2::from_shape_fn((n_z, n_r), |(i_z, _i_r)| z[i_z]);
        profiles_2d.psi_norm = &mask_2d * (&fit.flux_total.psi - psi_a) / (psi_b - psi_a);
        profiles_2d.psi = fit.flux_total.psi;
        profiles_2d.d_psi_d_r = fit.flux_total.d_psi_d_r;
        profiles_2d.d_psi_d_z = fit.flux_total.d_psi_d_z;
        profiles_2d.d2_psi_d_r2 = fit.flux_total.d2_psi_d_r2;
        profiles_2d.d2_psi_d_r_d_z = fit.flux_total.d2_psi_d_r_d_z;
        profiles_2d.d2_psi_d_z2 = fit.flux_total.d2_psi_d_z2;
        profiles_2d.psi_coils = fit.psi_coils;
        profiles_2d.mask = mask_2d;
        profiles_2d.j_phi = j_phi_2d;
        time_slice.profiles_2d = vec![profiles_2d];

        Ok(equilibrium_ids)
    }
}

/// The p' and ff' source functions the Equilibrium IDS's coefficients are written for.
///
/// Both are a one-degree-of-freedom `EfitPolynomial`, whose single basis function is `1 - psi_norm`.
/// Neither is fitted, so neither has a regularisation.
fn source_functions() -> (SharedSourceFunction, SharedSourceFunction) {
    let p_prime_source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
        n_dof: 1,
        regularisations: Array2::zeros((0, 1)),
        dof_values: Array1::zeros(0),
        exact: false,
    });
    let ff_prime_source_function: SharedSourceFunction = Arc::new(EfitPolynomial {
        n_dof: 1,
        regularisations: Array2::zeros((0, 1)),
        dof_values: Array1::zeros(0),
        exact: false,
    });
    (p_prime_source_function, ff_prime_source_function)
}
