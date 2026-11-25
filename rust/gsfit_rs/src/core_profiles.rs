use crate::settings::settings_core_profiles::SettingsCoreProfiles;
use imas_rs::ids::core_profiles::{
    Code, CoreProfileIons, CoreProfiles as CoreProfilesIds, CoreProfilesProfiles1d, CoreProfilesProfiles2d, CoreProfilesProfiles2dElectrons, CoreRadialGrid,
    EquilibriumProfiles2dGrid, IdentifierDynamicAos3, PlasmaCompositionNeutralElement,
};
use imas_rs::python::{PickleState, PickleStateError, PyCoreProfiles, PyParametersInput, PyPath, read_path};
use ndarray::{Array1, Array2, Axis};
use numpy::PyArrayMethods;
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The kinetic profiles, stored as an IMAS `core_profiles` IDS.
///
/// Two parts are filled, each by its own setter.
///
/// `profiles_2d(itime)`, by `set_profiles_2d_irregular_rz`, at a list of points in the poloidal cross-section: the irregular (R, Z) grid
/// `irregular_rz_na`, index 91 of the data dictionary's poloidal plane coordinates, whose points are `grid/dim1(i)`, `grid/dim2(i)`, and
/// whose fields are on `values(i, 0)`. One entry per measurement time, each holding:
/// * `time` - the measurement time [second]
/// * `grid_type`, `grid/dim1`, `grid/dim2` - the points: the major radius and height of each [metre]
/// * `electrons/pressure` - the electron pressure, a custom key, as the data dictionary has no electrons in `profiles_2d` [pascal]
/// * `pressure_thermal` - the thermal pressure, electrons and ions [pascal]
///
/// A value which was not measured, or is not physical, is NaN; the equilibrium does not fit it.
///
/// `Equilibrium.copy_from_core_profiles_ids` makes the pressure constraints, `equilibrium/time_slice(itime)/constraints/pressure(i1)`, from
/// `pressure_thermal`, one per point.
///
/// `profiles_1d(itime)`, the kinetic profiles, one entry per time, on the radial grid `grid/rho_pol_norm = sqrt(psi_norm)`, which, with
/// `time`, every quantity of a `profiles_1d(itime)` shares. Each part has its own setter, and they can be set in any order, as long as
/// each is at the same times and on the same grid:
/// * `set_profiles_1d_electrons` - `electrons/density_thermal` [1 / metre ** 3] and `electrons/temperature` [electron_volt]
/// * `set_profiles_1d_zeff` - `zeff` [dimensionless]
/// * `set_profiles_1d_thermal_ions` - one species' `ion(i1)/density_thermal` [1 / metre ** 3] and `ion(i1)/temperature` [electron_volt]
/// * `set_profiles_1d_fast_ions` - one species' fast ions, e.g. the beam ions from ASCOT5: their density, `ion(i1)/density_fast`
///   [1 / metre ** 3], and their pressures along and across the field, `ion(i1)/pressure_fast_parallel = sum(m v_par ** 2)` and
///   `ion(i1)/pressure_fast_perpendicular = sum(m v_perp ** 2 / 2)` [pascal]
/// * `set_profiles_1d_e_field_parallel` - the parallel electric field, `e_field/parallel = <E . B> / B0` [volt / metre], and the
///   `vacuum_toroidal_field` whose `b0` is `B0`
///
/// A species' thermal and fast ions share its `ion(i1)`, which is found by its `name`, and holds the species: `/name`, `/element(1)/a`,
/// `/element(1)/z_n`, `/element(1)/atoms_n` and `/z_ion`. A species which is not there yet is added after the others. The bootstrap
/// current takes `ion(0)` as the main ion, so set the main ion first.
///
/// The plasma current's components, `gsfit_rs.calculate_plasma_current_components`, read `profiles_1d`: the bootstrap current the
/// electrons, `zeff` and the thermal ions; the ohmic current the electrons, `zeff` and `e_field/parallel`; and the shielding of the
/// beam-driven current the electrons and `zeff`. The fast ions are read by nothing yet.
///
/// Read the data back through `get`, or `core_profiles_ids`, and a path from `gsfit_rs.imas.core_profiles_paths`; there are no bespoke
/// accessors.
///
/// # Settings
///
/// The kinetic profiles are constructed with the settings they are built from, e.g. which run of which database they are read from, as an
/// IMAS `parameters_input`. They are validated against the XML schema in `parameters_input`, and only their XML is kept: it is written to
/// `core_profiles/code/parameters`, where IMAS records the parameters of the code which made an IDS.
///
/// They are also read into `settings_core_profiles`, whose structs are generated from the XML schema `core_profiles.xsd`. They are kept
/// here, not in the IDS, as the core_profiles IDS can also be made by another code, whose `core_profiles/code/parameters` would not hold
/// GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct CoreProfiles {
    pub core_profiles_ids: CoreProfilesIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_core_profiles: SettingsCoreProfiles,
}

/// Python accessible methods
#[pymethods]
impl CoreProfiles {
    /// Construct an empty core_profiles IDS from its settings.
    /// Later it will be populated by reading from a database.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the core_profiles IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.defaults` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_core_profiles` data types, e.g. a `pulseNo` too large for an `i64`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the kinetic profiles' settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the kinetic profiles' settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The kinetic profiles' settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsCoreProfiles` struct
        let settings_core_profiles: SettingsCoreProfiles = SettingsCoreProfiles::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The kinetic profiles' settings cannot be read into `SettingsCoreProfiles`: {error}")))?;

        // Construct the core_profiles IDS from the parsed settings
        let core_profiles_ids: CoreProfilesIds = CoreProfilesIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            core_profiles_ids,
            settings_core_profiles,
        })
    }

    /// Set `core_profiles/profiles_2d(itime)`, one per measurement time, at a list of points: the irregular (R, Z) grid
    /// `irregular_rz_na`. Any `profiles_2d` already set is replaced.
    ///
    /// # Arguments
    /// * `time` - the measurement times (1d array), [second]
    /// * `r` - the major radius of each point, `grid/dim1` (1d array), [metre]
    /// * `z` - the height of each point, `grid/dim2` (1d array), [metre]
    /// * `electrons_pressure` - the electron pressure, `electrons/pressure` (2d array, shape = [n_time, n_point]), [pascal]
    /// * `pressure_thermal` - the thermal pressure, electrons and ions, `pressure_thermal` (2d array, shape = [n_time, n_point]), [pascal]
    ///
    /// Store the **experimental** times here, not the reconstruction times: `Equilibrium.copy_from_core_profiles_ids` interpolates onto
    /// them itself. A value which was not measured, or is not physical, is NaN.
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no point, a time is not finite or the times do not increase, a point is not at a positive
    /// finite major radius and a finite height, or a pressure does not have one value per time and point.
    pub fn set_profiles_2d_irregular_rz(
        &mut self,
        time: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        electrons_pressure: PyReadonlyArray2<f64>,
        pressure_thermal: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let point_r: Array1<f64> = r.to_owned_array();
        let point_z: Array1<f64> = z.to_owned_array();
        let electrons_pressure_rs: Array2<f64> = electrons_pressure.to_owned_array();
        let pressure_thermal_rs: Array2<f64> = pressure_thermal.to_owned_array();

        // Run the Rust method
        self.set_profiles_2d_irregular_rz_rs(time_rs, point_r, point_z, electrons_pressure_rs, pressure_thermal_rs)
            .map_err(PyValueError::new_err)
    }

    /// Set the fast ions of one species on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
    /// `ion(i1)/density_fast`, `ion(i1)/pressure_fast_parallel` and `ion(i1)/pressure_fast_perpendicular`.
    ///
    /// The fast ions of a species share its `ion(i1)` with its thermal ions, so `ion(i1)` is found by its `name`, and a species which is not
    /// there yet is added after the others. Its fast ions already set are replaced.
    ///
    /// Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
    /// `rho_pol_norm`; when it is set, e.g. by the fast ions of another species, the fast ions must be at its times and on its
    /// `grid/rho_pol_norm`.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `ion_name` - the species, `ion(i1)/name`, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, `ion(i1)/element(1)/a`, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, `ion(i1)/element(1)/z_n`, [dimensionless]
    /// * `ion_z_ion` - its charge, `ion(i1)/z_ion`, [dimensionless]
    /// * `density_fast` - the density of its fast ions, `ion(i1)/density_fast` (2d array, shape = [n_time, n_rho_pol_norm]),
    ///   [1 / metre ** 3]
    /// * `pressure_fast_parallel` - their pressure along the field, `sum(m v_par ** 2)`, `ion(i1)/pressure_fast_parallel` (2d array,
    ///   shape = [n_time, n_rho_pol_norm]), [pascal]
    /// * `pressure_fast_perpendicular` - their pressure across the field, `sum(m v_perp ** 2 / 2)`, `ion(i1)/pressure_fast_perpendicular`
    ///   (2d array, shape = [n_time, n_rho_pol_norm]), [pascal]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, the species is not named or not physical, a profile does not have one
    /// value per time and radial point, `profiles_1d` is set at other times or on another `grid/rho_pol_norm`, or it holds a species of the
    /// same name which is another species.
    #[allow(clippy::too_many_arguments)]
    pub fn set_profiles_1d_fast_ions(
        &mut self,
        time: PyReadonlyArray1<f64>,
        rho_pol_norm: PyReadonlyArray1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        density_fast: PyReadonlyArray2<f64>,
        pressure_fast_parallel: PyReadonlyArray2<f64>,
        pressure_fast_perpendicular: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let density_fast_rs: Array2<f64> = density_fast.to_owned_array();
        let pressure_fast_parallel_rs: Array2<f64> = pressure_fast_parallel.to_owned_array();
        let pressure_fast_perpendicular_rs: Array2<f64> = pressure_fast_perpendicular.to_owned_array();

        // Run the Rust method
        self.set_profiles_1d_fast_ions_rs(
            time_rs,
            rho_pol_norm_rs,
            ion_name,
            ion_element_a,
            ion_element_z_n,
            ion_z_ion,
            density_fast_rs,
            pressure_fast_parallel_rs,
            pressure_fast_perpendicular_rs,
        )
        .map_err(PyValueError::new_err)
    }

    /// Set the thermal electrons on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
    /// `electrons/density_thermal` and `electrons/temperature`. Any already set are replaced.
    ///
    /// Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
    /// `rho_pol_norm`; when it is set, the electrons must be at its times and on its `grid/rho_pol_norm`.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `density_thermal` - the electron density, `electrons/density_thermal` (2d array, shape = [n_time, n_rho_pol_norm]),
    ///   [1 / metre ** 3]
    /// * `temperature` - the electron temperature, `electrons/temperature` (2d array, shape = [n_time, n_rho_pol_norm]), [electron_volt]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, a profile does not have one value per time and radial point, or
    /// `profiles_1d` is set at other times or on another `grid/rho_pol_norm`.
    pub fn set_profiles_1d_electrons(
        &mut self,
        time: PyReadonlyArray1<f64>,
        rho_pol_norm: PyReadonlyArray1<f64>,
        density_thermal: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let density_thermal_rs: Array2<f64> = density_thermal.to_owned_array();
        let temperature_rs: Array2<f64> = temperature.to_owned_array();

        // Run the Rust method
        self.set_profiles_1d_electrons_rs(time_rs, rho_pol_norm_rs, density_thermal_rs, temperature_rs)
            .map_err(PyValueError::new_err)
    }

    /// Set the effective charge on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`: `zeff`. Any
    /// already set is replaced.
    ///
    /// Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
    /// `rho_pol_norm`; when it is set, `zeff` must be at its times and on its `grid/rho_pol_norm`.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `zeff` - the effective charge, `zeff` (2d array, shape = [n_time, n_rho_pol_norm]), [dimensionless]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, `zeff` does not have one value per time and radial point, or is below 1
    /// somewhere (NaN is allowed), or `profiles_1d` is set at other times or on another `grid/rho_pol_norm`.
    pub fn set_profiles_1d_zeff(&mut self, time: PyReadonlyArray1<f64>, rho_pol_norm: PyReadonlyArray1<f64>, zeff: PyReadonlyArray2<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let zeff_rs: Array2<f64> = zeff.to_owned_array();

        // Run the Rust method
        self.set_profiles_1d_zeff_rs(time_rs, rho_pol_norm_rs, zeff_rs).map_err(PyValueError::new_err)
    }

    /// Set the thermal ions of one species on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
    /// `ion(i1)/density_thermal` and `ion(i1)/temperature`.
    ///
    /// The thermal ions of a species share its `ion(i1)` with its fast ions, so `ion(i1)` is found by its `name`, and a species which is not
    /// there yet is added after the others. Its thermal ions already set are replaced. The bootstrap current takes `ion(0)` as the main
    /// ion, so set the main ion first.
    ///
    /// Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
    /// `rho_pol_norm`; when it is set, the ions must be at its times and on its `grid/rho_pol_norm`.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `ion_name` - the species, `ion(i1)/name`, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, `ion(i1)/element(1)/a`, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, `ion(i1)/element(1)/z_n`, [dimensionless]
    /// * `ion_z_ion` - its charge, `ion(i1)/z_ion`, [dimensionless]
    /// * `density_thermal` - the density of its thermal ions, `ion(i1)/density_thermal` (2d array, shape = [n_time, n_rho_pol_norm]),
    ///   [1 / metre ** 3]
    /// * `temperature` - their temperature, `ion(i1)/temperature` (2d array, shape = [n_time, n_rho_pol_norm]), [electron_volt]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, the species is not named or not physical, a profile does not have one
    /// value per time and radial point, `profiles_1d` is set at other times or on another `grid/rho_pol_norm`, or it holds a species of the
    /// same name which is another species.
    #[allow(clippy::too_many_arguments)]
    pub fn set_profiles_1d_thermal_ions(
        &mut self,
        time: PyReadonlyArray1<f64>,
        rho_pol_norm: PyReadonlyArray1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        density_thermal: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let density_thermal_rs: Array2<f64> = density_thermal.to_owned_array();
        let temperature_rs: Array2<f64> = temperature.to_owned_array();

        // Run the Rust method
        self.set_profiles_1d_thermal_ions_rs(
            time_rs,
            rho_pol_norm_rs,
            ion_name,
            ion_element_a,
            ion_element_z_n,
            ion_z_ion,
            density_thermal_rs,
            temperature_rs,
        )
        .map_err(PyValueError::new_err)
    }

    /// Set the parallel electric field on `core_profiles/profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm`:
    /// `e_field/parallel = <E . B> / B0`, and `core_profiles/vacuum_toroidal_field`, whose `b0` is `B0`. Any already set are replaced.
    ///
    /// Every quantity of `profiles_1d(itime)` shares its time and `grid`. So when `profiles_1d` is unset, it is made, at `time` and on
    /// `rho_pol_norm`; when it is set, the field must be at its times and on its `grid/rho_pol_norm`. `vacuum_toroidal_field/b0` is on the
    /// same times.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `e_field_parallel` - the flux-surface averaged parallel electric field, `<E . B> / B0`, `e_field/parallel` (2d array,
    ///   shape = [n_time, n_rho_pol_norm]), [volt / metre]
    /// * `vacuum_toroidal_field_r0` - the major radius at which `vacuum_toroidal_field_b0` is given, `vacuum_toroidal_field/r0`, [metre]
    /// * `vacuum_toroidal_field_b0` - the vacuum toroidal field at `vacuum_toroidal_field_r0`, `B0`, which `e_field_parallel` is normalised
    ///   by, `vacuum_toroidal_field/b0` (1d array, shape = [n_time]), [tesla]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, `e_field_parallel` does not have one value per time and radial point,
    /// the vacuum toroidal field is not a positive finite radius and one finite, non-zero field per time, or `profiles_1d` is set at other
    /// times or on another `grid/rho_pol_norm`.
    pub fn set_profiles_1d_e_field_parallel(
        &mut self,
        time: PyReadonlyArray1<f64>,
        rho_pol_norm: PyReadonlyArray1<f64>,
        e_field_parallel: PyReadonlyArray2<f64>,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let e_field_parallel_rs: Array2<f64> = e_field_parallel.to_owned_array();
        let vacuum_toroidal_field_b0_rs: Array1<f64> = vacuum_toroidal_field_b0.to_owned_array();

        // Run the Rust method
        self.set_profiles_1d_e_field_parallel_rs(
            time_rs,
            rho_pol_norm_rs,
            e_field_parallel_rs,
            vacuum_toroidal_field_r0,
            vacuum_toroidal_field_b0_rs,
        )
        .map_err(PyValueError::new_err)
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.core_profiles_paths`, straight out of the core_profiles IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached by its data dictionary path. A path
    /// holds no data, so the IDS is only borrowed for the read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.core_profiles_ids, "core_profiles", path)
    }

    /// A copy of the whole core_profiles IDS, for reading with `gsfit_rs.imas.core_profiles_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a detached snapshot is wanted: changes made
    /// on the Rust side afterwards are not seen by it.
    #[getter]
    fn core_profiles_ids(&self) -> PyCoreProfiles {
        PyCoreProfiles::new(self.core_profiles_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.CoreProfiles>");
        string_output += &format!("║  {:<74} ║\n", version);

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &self.core_profiles_ids.profiles_1d;
        let n_time_1d: usize = profiles_1d.len();
        if n_time_1d == 0 {
            string_output += &format!("║  {:<74} ║\n", " profiles_1d is unset");
        } else {
            let n_rho_pol_norm: usize = profiles_1d[0].grid.rho_pol_norm.len();
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" profiles_1d: {n_time_1d} time(s), {n_rho_pol_norm} point(s) in rho_pol_norm")
            );
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" time = [{}, {}] second", profiles_1d[0].time, profiles_1d[n_time_1d - 1].time)
            );
            // What the first `profiles_1d` holds; every one holds the same, as each setter sets all of them
            let profiles_1d_first: &CoreProfilesProfiles1d = &profiles_1d[0];
            let mut quantity_names: Vec<&str> = Vec::new();
            if !profiles_1d_first.electrons.density_thermal.is_empty() {
                quantity_names.push("electrons");
            }
            if !profiles_1d_first.zeff.is_empty() {
                quantity_names.push("zeff");
            }
            if !profiles_1d_first.e_field.parallel.is_empty() {
                quantity_names.push("e_field/parallel");
            }
            string_output += &format!("║  {:<74} ║\n", format!(" set = [{}]", quantity_names.join(", ")));
            let n_ion: usize = profiles_1d_first.ion.len();
            let mut ion_descriptions: Vec<String> = Vec::with_capacity(n_ion);
            for i_ion in 0..n_ion {
                let ion: &CoreProfileIons = &profiles_1d_first.ion[i_ion];
                let mut ion_parts: Vec<&str> = Vec::new();
                if !ion.density_thermal.is_empty() {
                    ion_parts.push("thermal");
                }
                if !ion.density_fast.is_empty() {
                    ion_parts.push("fast");
                }
                ion_descriptions.push(format!("{} ({})", ion.name, ion_parts.join(", ")));
            }
            string_output += &format!("║  {:<74} ║\n", format!(" ion(:) = [{}]", ion_descriptions.join(", ")));
        }

        let profiles_2d: &Vec<CoreProfilesProfiles2d> = &self.core_profiles_ids.profiles_2d;
        let n_time: usize = profiles_2d.len();
        if n_time == 0 {
            string_output += &format!("║  {:<74} ║\n", " profiles_2d is unset");
        } else {
            let n_point: usize = profiles_2d[0].grid.dim1.len();
            string_output += &format!("║  {:<74} ║\n", format!(" profiles_2d: {n_time} time(s), {n_point} point(s)"));
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" time = [{}, {}] second", profiles_2d[0].time, profiles_2d[n_time - 1].time)
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `core_profiles_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_core_profiles`, are not pickled: they are read again from the XML which `core_profiles_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("core_profiles_ids", self.core_profiles_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_core_profiles`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `CoreProfiles` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `CoreProfiles` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let core_profiles_ids_state: Bound<'_, PyAny> = state
            .get_item("core_profiles_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `CoreProfiles` has no `core_profiles_ids`"))?;
        let core_profiles_ids: CoreProfilesIds = PickleState::from_pickle_state(&core_profiles_ids_state)
            .map_err(|error: PickleStateError| error.into_py_err("`CoreProfiles.core_profiles_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_core_profiles: SettingsCoreProfiles = SettingsCoreProfiles::from_xml(&core_profiles_ids.code.parameters).map_err(|error: String| {
            PyValueError::new_err(format!(
                "The pickled `CoreProfiles`'s settings cannot be read into `SettingsCoreProfiles`: {error}"
            ))
        })?;

        Ok(Self {
            core_profiles_ids,
            settings_core_profiles,
        })
    }
}

/// Rust only methods
impl CoreProfiles {
    /// Set `core_profiles/profiles_2d(itime)` at a list of points. This does the work of the Python method
    /// `set_profiles_2d_irregular_rz`, so that Rust code can set them too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the measurement times, [second]
    /// * `point_r` - the major radius of each point, [metre]
    /// * `point_z` - the height of each point, [metre]
    /// * `electrons_pressure` - the electron pressure, shape = [n_time, n_point], [pascal]
    /// * `pressure_thermal` - the thermal pressure, electrons and ions, shape = [n_time, n_point], [pascal]
    ///
    /// # Errors
    /// Why the profiles cannot be set: no time or no point, a time which is not finite or times which do not increase, a point not at
    /// a positive finite major radius and a finite height, or a pressure without one value per time and point.
    pub fn set_profiles_2d_irregular_rz_rs(
        &mut self,
        time: Array1<f64>,
        point_r: Array1<f64>,
        point_z: Array1<f64>,
        electrons_pressure: Array2<f64>,
        pressure_thermal: Array2<f64>,
    ) -> Result<(), String> {
        let n_time: usize = time.len();
        let n_point: usize = point_r.len();

        if n_time == 0 {
            return Err("`core_profiles/profiles_2d`: `time` is empty; the profiles need at least one time".to_string());
        }
        for i_time in 0..n_time {
            if !time[i_time].is_finite() {
                return Err(format!("`core_profiles/profiles_2d`: `time[{i_time}]` = {} is not finite", time[i_time]));
            }
        }
        for i_time in 1..n_time {
            if time[i_time] <= time[i_time - 1] {
                return Err(format!(
                    "`core_profiles/profiles_2d`: the times must increase, but `time[{i_time}]` = {} follows `time[{}]` = {}",
                    time[i_time],
                    i_time - 1,
                    time[i_time - 1]
                ));
            }
        }

        if n_point == 0 {
            return Err("`core_profiles/profiles_2d`: `r` is empty; the profiles need at least one point".to_string());
        }
        if point_z.len() != n_point {
            return Err(format!(
                "`core_profiles/profiles_2d`: `r` has {n_point} point(s) but `z` has {}; each point needs both",
                point_z.len()
            ));
        }
        for i_point in 0..n_point {
            let is_usable_position: bool = point_r[i_point] > 0.0 && point_r[i_point].is_finite() && point_z[i_point].is_finite();
            if !is_usable_position {
                return Err(format!(
                    "`core_profiles/profiles_2d`: point {i_point} has r = {} and z = {} metre; r must be positive, and both finite",
                    point_r[i_point], point_z[i_point]
                ));
            }
        }

        if electrons_pressure.dim() != (n_time, n_point) {
            return Err(format!(
                "`core_profiles/profiles_2d`: `electrons_pressure` has shape {:?}, but there are {n_time} time(s) and {n_point} point(s)",
                electrons_pressure.dim()
            ));
        }
        if pressure_thermal.dim() != (n_time, n_point) {
            return Err(format!(
                "`core_profiles/profiles_2d`: `pressure_thermal` has shape {:?}, but there are {n_time} time(s) and {n_point} point(s)",
                pressure_thermal.dim()
            ));
        }

        // One `profiles_2d` per time. On the irregular grid each field is on `values(i_point, 0)`, so it has a second dimension of 1
        let mut profiles_2d: Vec<CoreProfilesProfiles2d> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            profiles_2d.push(CoreProfilesProfiles2d {
                // `irregular_rz_na` is index 91 of the data dictionary's poloidal plane coordinates enumeration
                grid_type: IdentifierDynamicAos3 {
                    name: "irregular_rz_na".to_string(),
                    index: 91,
                    description: "Irregular grid, thus give list of vertices in dim1(1:ndim1), dim2(1:ndim1) and then all fields are on values(1:ndim1,1)"
                        .to_string(),
                },
                grid: EquilibriumProfiles2dGrid {
                    dim1: point_r.clone(),
                    dim2: point_z.clone(),
                    ..Default::default()
                },
                electrons: CoreProfilesProfiles2dElectrons {
                    pressure: electrons_pressure.row(i_time).to_owned().insert_axis(Axis(1)),
                },
                pressure_thermal: pressure_thermal.row(i_time).to_owned().insert_axis(Axis(1)),
                time: time[i_time],
                ..Default::default()
            });
        }
        self.core_profiles_ids.profiles_2d = profiles_2d;

        Ok(())
    }

    /// Set the fast ions of one species on `core_profiles/profiles_1d(itime)`. This does the work of the Python method
    /// `set_profiles_1d_fast_ions`, so that Rust code can set them too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `ion_name` - the species, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, [dimensionless]
    /// * `ion_z_ion` - its charge, [dimensionless]
    /// * `density_fast` - the density of its fast ions, shape = [n_time, n_rho_pol_norm], [1 / metre ** 3]
    /// * `pressure_fast_parallel` - their pressure along the field, shape = [n_time, n_rho_pol_norm], [pascal]
    /// * `pressure_fast_perpendicular` - their pressure across the field, shape = [n_time, n_rho_pol_norm], [pascal]
    ///
    /// # Errors
    /// Why the fast ions cannot be set: see `set_profiles_1d_fast_ions`. Nothing is changed when they cannot.
    #[allow(clippy::too_many_arguments)]
    pub fn set_profiles_1d_fast_ions_rs(
        &mut self,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        density_fast: Array2<f64>,
        pressure_fast_parallel: Array2<f64>,
        pressure_fast_perpendicular: Array2<f64>,
    ) -> Result<(), String> {
        check_time_and_rho_pol_norm(PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        check_ion_species(PROFILES_1D_PATH, &ion_name, ion_element_a, ion_element_z_n, ion_z_ion)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(PROFILES_1D_PATH, "density_fast", &density_fast, n_time, n_rho_pol_norm)?;
        check_profile_shape(PROFILES_1D_PATH, "pressure_fast_parallel", &pressure_fast_parallel, n_time, n_rho_pol_norm)?;
        check_profile_shape(
            PROFILES_1D_PATH,
            "pressure_fast_perpendicular",
            &pressure_fast_perpendicular,
            n_time,
            n_rho_pol_norm,
        )?;
        // Checked, and the species found, before anything is changed, so that nothing is when the fast ions cannot be set
        self.check_profiles_1d_coordinates(&time, &rho_pol_norm)?;
        let i_ion_species: Vec<Option<usize>> = self.ion_species_indices(n_time, &ion_name, ion_element_a, ion_element_z_n, ion_z_ion)?;

        self.make_profiles_1d_if_unset(&time, &rho_pol_norm);
        for i_time in 0..n_time {
            let ion: &mut CoreProfileIons = ion_species_mut(
                &mut self.core_profiles_ids.profiles_1d[i_time],
                i_ion_species[i_time],
                &ion_name,
                ion_element_a,
                ion_element_z_n,
                ion_z_ion,
            );
            ion.density_fast = density_fast.row(i_time).to_owned();
            ion.pressure_fast_parallel = pressure_fast_parallel.row(i_time).to_owned();
            ion.pressure_fast_perpendicular = pressure_fast_perpendicular.row(i_time).to_owned();
        }

        Ok(())
    }

    /// Set the thermal electrons on `core_profiles/profiles_1d(itime)`. This does the work of the Python method `set_profiles_1d_electrons`,
    /// so that Rust code can set them too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `density_thermal` - the electron density, shape = [n_time, n_rho_pol_norm], [1 / metre ** 3]
    /// * `temperature` - the electron temperature, shape = [n_time, n_rho_pol_norm], [electron_volt]
    ///
    /// # Errors
    /// Why the electrons cannot be set: see `set_profiles_1d_electrons`. Nothing is changed when they cannot.
    pub fn set_profiles_1d_electrons_rs(
        &mut self,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        density_thermal: Array2<f64>,
        temperature: Array2<f64>,
    ) -> Result<(), String> {
        check_time_and_rho_pol_norm(PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(PROFILES_1D_PATH, "density_thermal", &density_thermal, n_time, n_rho_pol_norm)?;
        check_profile_shape(PROFILES_1D_PATH, "temperature", &temperature, n_time, n_rho_pol_norm)?;
        self.check_profiles_1d_coordinates(&time, &rho_pol_norm)?;

        self.make_profiles_1d_if_unset(&time, &rho_pol_norm);
        for i_time in 0..n_time {
            let profiles_1d: &mut CoreProfilesProfiles1d = &mut self.core_profiles_ids.profiles_1d[i_time];
            profiles_1d.electrons.density_thermal = density_thermal.row(i_time).to_owned();
            profiles_1d.electrons.temperature = temperature.row(i_time).to_owned();
        }

        Ok(())
    }

    /// Set the effective charge on `core_profiles/profiles_1d(itime)`. This does the work of the Python method `set_profiles_1d_zeff`, so
    /// that Rust code can set it too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `zeff` - the effective charge, shape = [n_time, n_rho_pol_norm], [dimensionless]
    ///
    /// # Errors
    /// Why the effective charge cannot be set: see `set_profiles_1d_zeff`. Nothing is changed when it cannot.
    pub fn set_profiles_1d_zeff_rs(&mut self, time: Array1<f64>, rho_pol_norm: Array1<f64>, zeff: Array2<f64>) -> Result<(), String> {
        check_time_and_rho_pol_norm(PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(PROFILES_1D_PATH, "zeff", &zeff, n_time, n_rho_pol_norm)?;
        // The neoclassical formulae need `Z_eff >= 1`; NaN, a value which is not known, is allowed
        for i_time in 0..n_time {
            for i_rho_pol_norm in 0..n_rho_pol_norm {
                let zeff_here: f64 = zeff[(i_time, i_rho_pol_norm)];
                let is_usable_zeff: bool = zeff_here.is_nan() || (zeff_here.is_finite() && zeff_here >= 1.0);
                if !is_usable_zeff {
                    return Err(format!(
                        "{PROFILES_1D_PATH}: `zeff[{i_time}, {i_rho_pol_norm}]` = {zeff_here} must be finite and at least 1, or NaN"
                    ));
                }
            }
        }
        self.check_profiles_1d_coordinates(&time, &rho_pol_norm)?;

        self.make_profiles_1d_if_unset(&time, &rho_pol_norm);
        for i_time in 0..n_time {
            self.core_profiles_ids.profiles_1d[i_time].zeff = zeff.row(i_time).to_owned();
        }

        Ok(())
    }

    /// Set the thermal ions of one species on `core_profiles/profiles_1d(itime)`. This does the work of the Python method
    /// `set_profiles_1d_thermal_ions`, so that Rust code can set them too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `ion_name` - the species, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, [dimensionless]
    /// * `ion_z_ion` - its charge, [dimensionless]
    /// * `density_thermal` - the density of its thermal ions, shape = [n_time, n_rho_pol_norm], [1 / metre ** 3]
    /// * `temperature` - their temperature, shape = [n_time, n_rho_pol_norm], [electron_volt]
    ///
    /// # Errors
    /// Why the thermal ions cannot be set: see `set_profiles_1d_thermal_ions`. Nothing is changed when they cannot.
    #[allow(clippy::too_many_arguments)]
    pub fn set_profiles_1d_thermal_ions_rs(
        &mut self,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        density_thermal: Array2<f64>,
        temperature: Array2<f64>,
    ) -> Result<(), String> {
        check_time_and_rho_pol_norm(PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        check_ion_species(PROFILES_1D_PATH, &ion_name, ion_element_a, ion_element_z_n, ion_z_ion)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(PROFILES_1D_PATH, "density_thermal", &density_thermal, n_time, n_rho_pol_norm)?;
        check_profile_shape(PROFILES_1D_PATH, "temperature", &temperature, n_time, n_rho_pol_norm)?;
        // Checked, and the species found, before anything is changed, so that nothing is when the thermal ions cannot be set
        self.check_profiles_1d_coordinates(&time, &rho_pol_norm)?;
        let i_ion_species: Vec<Option<usize>> = self.ion_species_indices(n_time, &ion_name, ion_element_a, ion_element_z_n, ion_z_ion)?;

        self.make_profiles_1d_if_unset(&time, &rho_pol_norm);
        for i_time in 0..n_time {
            let ion: &mut CoreProfileIons = ion_species_mut(
                &mut self.core_profiles_ids.profiles_1d[i_time],
                i_ion_species[i_time],
                &ion_name,
                ion_element_a,
                ion_element_z_n,
                ion_z_ion,
            );
            ion.density_thermal = density_thermal.row(i_time).to_owned();
            ion.temperature = temperature.row(i_time).to_owned();
        }

        Ok(())
    }

    /// Set the parallel electric field on `core_profiles/profiles_1d(itime)`, and `core_profiles/vacuum_toroidal_field`. This does the work
    /// of the Python method `set_profiles_1d_e_field_parallel`, so that Rust code can set them too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `e_field_parallel` - `<E . B> / B0`, shape = [n_time, n_rho_pol_norm], [volt / metre]
    /// * `vacuum_toroidal_field_r0` - the major radius at which `vacuum_toroidal_field_b0` is given, [metre]
    /// * `vacuum_toroidal_field_b0` - the vacuum toroidal field at `vacuum_toroidal_field_r0`, `B0`, shape = [n_time], [tesla]
    ///
    /// # Errors
    /// Why the parallel electric field cannot be set: see `set_profiles_1d_e_field_parallel`. Nothing is changed when it cannot.
    pub fn set_profiles_1d_e_field_parallel_rs(
        &mut self,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        e_field_parallel: Array2<f64>,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: Array1<f64>,
    ) -> Result<(), String> {
        check_time_and_rho_pol_norm(PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(PROFILES_1D_PATH, "e_field_parallel", &e_field_parallel, n_time, n_rho_pol_norm)?;
        check_vacuum_toroidal_field(
            "core_profiles/vacuum_toroidal_field",
            vacuum_toroidal_field_r0,
            &vacuum_toroidal_field_b0,
            n_time,
        )?;
        self.check_profiles_1d_coordinates(&time, &rho_pol_norm)?;

        self.make_profiles_1d_if_unset(&time, &rho_pol_norm);
        for i_time in 0..n_time {
            self.core_profiles_ids.profiles_1d[i_time].e_field.parallel = e_field_parallel.row(i_time).to_owned();
        }
        self.core_profiles_ids.vacuum_toroidal_field.r0 = vacuum_toroidal_field_r0;
        self.core_profiles_ids.vacuum_toroidal_field.b0 = vacuum_toroidal_field_b0;

        Ok(())
    }

    /// Check that `profiles_1d`, when it is set, is at `time` and on `rho_pol_norm`, as every quantity of a `profiles_1d(itime)` shares its
    /// time and grid. When it is unset there is nothing to check: it is made at them by `make_profiles_1d_if_unset`.
    ///
    /// # Arguments
    /// * `time` - the times of the profiles being set, [second]
    /// * `rho_pol_norm` - their radial grid, [dimensionless]
    ///
    /// # Errors
    /// `profiles_1d` is set at other times, or on another `grid/rho_pol_norm`
    fn check_profiles_1d_coordinates(&self, time: &Array1<f64>, rho_pol_norm: &Array1<f64>) -> Result<(), String> {
        let n_time: usize = time.len();
        let n_time_set: usize = self.core_profiles_ids.profiles_1d.len();
        if n_time_set == 0 {
            return Ok(());
        }
        if n_time_set != n_time {
            return Err(format!(
                "{PROFILES_1D_PATH} is set at {n_time_set} time(s), but these profiles are at {n_time}; they must be at its times"
            ));
        }
        for i_time in 0..n_time {
            let profiles_1d: &CoreProfilesProfiles1d = &self.core_profiles_ids.profiles_1d[i_time];
            if profiles_1d.time != time[i_time] {
                return Err(format!(
                    "`core_profiles/profiles_1d({i_time})/time` = {} second, but these profiles' `time[{i_time}]` = {} second; they must be at its times",
                    profiles_1d.time, time[i_time]
                ));
            }
            if profiles_1d.grid.rho_pol_norm != rho_pol_norm {
                return Err(format!(
                    "`core_profiles/profiles_1d({i_time})/grid/rho_pol_norm` is not these profiles' `rho_pol_norm`; every quantity of `profiles_1d({i_time})` is on its grid"
                ));
            }
        }

        Ok(())
    }

    /// Make `profiles_1d`, one per time, each on the radial grid `rho_pol_norm`, when it is unset. Check it first with
    /// `check_profiles_1d_coordinates`.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, [dimensionless]
    fn make_profiles_1d_if_unset(&mut self, time: &Array1<f64>, rho_pol_norm: &Array1<f64>) {
        if !self.core_profiles_ids.profiles_1d.is_empty() {
            return;
        }
        let n_time: usize = time.len();
        let mut profiles_1d: Vec<CoreProfilesProfiles1d> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            profiles_1d.push(CoreProfilesProfiles1d {
                grid: CoreRadialGrid {
                    rho_pol_norm: rho_pol_norm.clone(),
                    ..Default::default()
                },
                time: time[i_time],
                ..Default::default()
            });
        }
        self.core_profiles_ids.profiles_1d = profiles_1d;
    }

    /// Which `ion(i1)` of each `profiles_1d(itime)` is a species, found by its `name`; each is checked to be that species. Check
    /// `profiles_1d` first with `check_profiles_1d_coordinates`.
    ///
    /// # Arguments
    /// * `n_time` - the number of times of the profiles being set
    /// * `ion_name` - the species' `ion(i1)/name`
    /// * `ion_element_a` - the mass of its nucleus, `ion(i1)/element(1)/a`, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, `ion(i1)/element(1)/z_n`, [dimensionless]
    /// * `ion_z_ion` - its charge, `ion(i1)/z_ion`, [dimensionless]
    ///
    /// # Returns
    /// One per time: the index of the species' `ion(i1)`, or `None` when it is not there yet, as at every time when `profiles_1d` is unset
    ///
    /// # Errors
    /// An `ion(i1)` of the same name is another species
    fn ion_species_indices(
        &self,
        n_time: usize,
        ion_name: &str,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
    ) -> Result<Vec<Option<usize>>, String> {
        let mut i_ion_species: Vec<Option<usize>> = vec![None; n_time];
        let n_time_set: usize = self.core_profiles_ids.profiles_1d.len();
        for i_time in 0..n_time_set {
            let ions: &Vec<CoreProfileIons> = &self.core_profiles_ids.profiles_1d[i_time].ion;
            let n_ion: usize = ions.len();
            for i_ion in 0..n_ion {
                let ion: &CoreProfileIons = &ions[i_ion];
                if ion.name != ion_name {
                    continue;
                }
                let is_same_species: bool =
                    ion.element.len() == 1 && ion.element[0].a == ion_element_a && ion.element[0].z_n == ion_element_z_n && ion.z_ion == ion_z_ion;
                if !is_same_species {
                    return Err(format!(
                        "`core_profiles/profiles_1d({i_time})/ion({i_ion})` is named \"{ion_name}\", but is another species: its `element`, or `z_ion` = {}, differs",
                        ion.z_ion
                    ));
                }
                i_ion_species[i_time] = Some(i_ion);
            }
        }

        Ok(i_ion_species)
    }
}

/// Where the 1D profiles are, for the errors
const PROFILES_1D_PATH: &str = "`core_profiles/profiles_1d`";

/// The `ion(i1)` of `profiles_1d` which is a species: `i_ion` when the species is there, or a new `ion(i1)` holding it, added after the
/// others, when it is not.
///
/// # Arguments
/// * `profiles_1d` - one `core_profiles/profiles_1d(itime)`
/// * `i_ion` - the index of the species' `ion(i1)`, from `CoreProfiles::ion_species_indices`, or `None` when it is not there yet
/// * `ion_name` - the species' `ion(i1)/name`
/// * `ion_element_a` - the mass of its nucleus, `ion(i1)/element(1)/a`, [atomic_mass_unit]
/// * `ion_element_z_n` - the charge of its nucleus, `ion(i1)/element(1)/z_n`, [dimensionless]
/// * `ion_z_ion` - its charge, `ion(i1)/z_ion`, [dimensionless]
fn ion_species_mut<'a>(
    profiles_1d: &'a mut CoreProfilesProfiles1d,
    i_ion: Option<usize>,
    ion_name: &str,
    ion_element_a: f64,
    ion_element_z_n: i32,
    ion_z_ion: f64,
) -> &'a mut CoreProfileIons {
    let i_ion_species: usize = match i_ion {
        Some(i_ion_here) => i_ion_here,
        None => {
            profiles_1d.ion.push(CoreProfileIons {
                element: vec![PlasmaCompositionNeutralElement {
                    a: ion_element_a,
                    z_n: ion_element_z_n,
                    atoms_n: 1,
                }],
                z_ion: ion_z_ion,
                name: ion_name.to_string(),
                ..Default::default()
            });
            profiles_1d.ion.len() - 1
        }
    };

    &mut profiles_1d.ion[i_ion_species]
}

/// Check the coordinates of 1D profiles: their times, which must be finite and increase, and their radial grid, `rho_pol_norm =
/// sqrt(psi_norm)`, which must be finite, at least 0, and increase.
///
/// # Arguments
/// * `path` - where the profiles go, for the errors, e.g. "`core_profiles/profiles_1d`"
/// * `time` - the times, [second]
/// * `rho_pol_norm` - the radial grid, [dimensionless]
///
/// # Errors
/// There is no time or no radial point, a time is not finite or the times do not increase, or a radial point is not finite and at least 0
/// or the radial points do not increase
pub(crate) fn check_time_and_rho_pol_norm(path: &str, time: &Array1<f64>, rho_pol_norm: &Array1<f64>) -> Result<(), String> {
    let n_time: usize = time.len();
    let n_rho_pol_norm: usize = rho_pol_norm.len();

    if n_time == 0 {
        return Err(format!("{path}: `time` is empty; the profiles need at least one time"));
    }
    for i_time in 0..n_time {
        if !time[i_time].is_finite() {
            return Err(format!("{path}: `time[{i_time}]` = {} is not finite", time[i_time]));
        }
    }
    for i_time in 1..n_time {
        if time[i_time] <= time[i_time - 1] {
            return Err(format!(
                "{path}: the times must increase, but `time[{i_time}]` = {} follows `time[{}]` = {}",
                time[i_time],
                i_time - 1,
                time[i_time - 1]
            ));
        }
    }

    if n_rho_pol_norm == 0 {
        return Err(format!("{path}: `rho_pol_norm` is empty; the profiles need at least one radial point"));
    }
    for i_rho_pol_norm in 0..n_rho_pol_norm {
        let is_usable_rho_pol_norm: bool = rho_pol_norm[i_rho_pol_norm].is_finite() && rho_pol_norm[i_rho_pol_norm] >= 0.0;
        if !is_usable_rho_pol_norm {
            return Err(format!(
                "{path}: `rho_pol_norm[{i_rho_pol_norm}]` = {} must be finite and at least 0",
                rho_pol_norm[i_rho_pol_norm]
            ));
        }
    }
    for i_rho_pol_norm in 1..n_rho_pol_norm {
        if rho_pol_norm[i_rho_pol_norm] <= rho_pol_norm[i_rho_pol_norm - 1] {
            return Err(format!(
                "{path}: `rho_pol_norm` must increase, but `rho_pol_norm[{i_rho_pol_norm}]` = {} follows `rho_pol_norm[{}]` = {}",
                rho_pol_norm[i_rho_pol_norm],
                i_rho_pol_norm - 1,
                rho_pol_norm[i_rho_pol_norm - 1]
            ));
        }
    }

    Ok(())
}

/// Check that a profile has one value per time and radial point.
///
/// # Arguments
/// * `path` - where the profile goes, for the errors, e.g. "`core_profiles/profiles_1d`"
/// * `profile_name` - the profile's name, for the errors
/// * `profile` - the profile, shape = [n_time, n_rho_pol_norm]
/// * `n_time` - the number of times
/// * `n_rho_pol_norm` - the number of radial points
///
/// # Errors
/// It does not
pub(crate) fn check_profile_shape(path: &str, profile_name: &str, profile: &Array2<f64>, n_time: usize, n_rho_pol_norm: usize) -> Result<(), String> {
    if profile.dim() != (n_time, n_rho_pol_norm) {
        return Err(format!(
            "{path}: `{profile_name}` has shape {:?}, but there are {n_time} time(s) and {n_rho_pol_norm} radial point(s)",
            profile.dim()
        ));
    }

    Ok(())
}

/// Check that an ion species is named, and physical: the mass and charge of its nucleus, and its charge, positive and finite.
///
/// # Arguments
/// * `path` - where the species goes, for the errors, e.g. "`core_profiles/profiles_1d`"
/// * `ion_name` - the species' name, e.g. "D"
/// * `ion_element_a` - the mass of its nucleus, [atomic_mass_unit]
/// * `ion_element_z_n` - the charge of its nucleus, [dimensionless]
/// * `ion_z_ion` - its charge, [dimensionless]
///
/// # Errors
/// It is not named, or not physical
pub(crate) fn check_ion_species(path: &str, ion_name: &str, ion_element_a: f64, ion_element_z_n: i32, ion_z_ion: f64) -> Result<(), String> {
    if ion_name.is_empty() {
        return Err(format!("{path}: `ion_name` is empty; the species needs a name, e.g. \"D\""));
    }
    let is_physical_species: bool = ion_element_a.is_finite() && ion_element_a > 0.0 && ion_element_z_n >= 1 && ion_z_ion.is_finite() && ion_z_ion > 0.0;
    if !is_physical_species {
        return Err(format!(
            "{path}: the species \"{ion_name}\" has `ion_element_a` = {ion_element_a}, `ion_element_z_n` = {ion_element_z_n} and `ion_z_ion` = {ion_z_ion}; all must be positive, and finite"
        ));
    }

    Ok(())
}

/// Check a vacuum toroidal field, which normalises a flux-surface averaged quantity: a positive finite major radius, and one finite,
/// non-zero field per time.
///
/// # Arguments
/// * `path` - where the field goes, for the errors, e.g. "core_profiles/vacuum_toroidal_field"
/// * `vacuum_toroidal_field_r0` - the major radius at which the field is given, [metre]
/// * `vacuum_toroidal_field_b0` - the field at `vacuum_toroidal_field_r0`, one per time, [tesla]
/// * `n_time` - the number of times
///
/// # Errors
/// It is not
pub(crate) fn check_vacuum_toroidal_field(
    path: &str,
    vacuum_toroidal_field_r0: f64,
    vacuum_toroidal_field_b0: &Array1<f64>,
    n_time: usize,
) -> Result<(), String> {
    if !(vacuum_toroidal_field_r0.is_finite() && vacuum_toroidal_field_r0 > 0.0) {
        return Err(format!("`{path}/r0` = {vacuum_toroidal_field_r0} metre must be positive, and finite"));
    }
    if vacuum_toroidal_field_b0.len() != n_time {
        return Err(format!(
            "`{path}/b0` has {} value(s), but there are {n_time} time(s); it needs one per time",
            vacuum_toroidal_field_b0.len()
        ));
    }
    for i_time in 0..n_time {
        let is_usable_b0: bool = vacuum_toroidal_field_b0[i_time].is_finite() && vacuum_toroidal_field_b0[i_time] != 0.0;
        if !is_usable_b0 {
            return Err(format!(
                "`{path}/b0[{i_time}]` = {} tesla must be finite and not 0, as it normalises the profiles",
                vacuum_toroidal_field_b0[i_time]
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;
    use ndarray::array;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/core_profiles/core_profiles.xsd");

    /// `core_profiles.xml` without its optional link to `workflow.xml`, so it follows `core_profiles.xsd`, in which `workflow` is still a
    /// link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<core_profiles_settings>
    <thomson_scattering>
        <pressure_multiplier>2.0</pressure_multiplier>
        <minimum_pressure>500.0</minimum_pressure>
    </thomson_scattering>
</core_profiles_settings>
"#;

    fn core_profiles() -> CoreProfiles {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: SCHEMA_XSD.to_string(),
        });
        CoreProfiles::new(&parameters_input).expect("the settings follow their XML schema")
    }

    #[test]
    fn the_settings_are_read_into_structs_and_written_to_code_parameters() {
        let core_profiles: CoreProfiles = core_profiles();

        assert_eq!(core_profiles.core_profiles_ids.code.parameters, SETTINGS_XML);
        let thomson_scattering = core_profiles
            .settings_core_profiles
            .thomson_scattering
            .as_ref()
            .expect("the settings hold the Thomson scattering");
        assert_eq!(thomson_scattering.pressure_multiplier, 2.0);
        assert_eq!(thomson_scattering.minimum_pressure, 500.0);
    }

    #[test]
    fn profiles_2d_are_set_on_the_irregular_grid_one_per_time() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let electrons_pressure: Array2<f64> = array![[1.0, 2.0, f64::NAN], [4.0, 5.0, 6.0]];
        let pressure_thermal: Array2<f64> = &electrons_pressure * 2.0;

        core_profiles
            .set_profiles_2d_irregular_rz_rs(
                array![0.1, 0.2],
                array![0.3, 0.4, 0.5],
                array![0.0, 0.01, -0.01],
                electrons_pressure,
                pressure_thermal,
            )
            .expect("the profiles are consistent");

        let profiles_2d: &Vec<CoreProfilesProfiles2d> = &core_profiles.core_profiles_ids.profiles_2d;
        assert_eq!(profiles_2d.len(), 2);
        assert_eq!(profiles_2d[1].time, 0.2);
        assert_eq!(profiles_2d[1].grid_type.index, 91);
        assert_eq!(profiles_2d[1].grid.dim1, array![0.3, 0.4, 0.5]);
        assert_eq!(profiles_2d[1].grid.dim2, array![0.0, 0.01, -0.01]);
        assert_eq!(profiles_2d[1].pressure_thermal.dim(), (3, 1));
        assert_eq!(profiles_2d[1].pressure_thermal[[2, 0]], 12.0);
        assert_eq!(profiles_2d[1].electrons.pressure[[0, 0]], 4.0);
        assert!(profiles_2d[0].electrons.pressure[[2, 0]].is_nan());
    }

    #[test]
    fn inconsistent_profiles_are_an_error() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let pressure: Array2<f64> = Array2::from_elem((2, 3), 1.0);

        // Times which do not increase
        assert!(
            core_profiles
                .set_profiles_2d_irregular_rz_rs(
                    array![0.2, 0.1],
                    array![0.3, 0.4, 0.5],
                    array![0.0, 0.0, 0.0],
                    pressure.clone(),
                    pressure.clone()
                )
                .is_err()
        );
        // A point at a negative major radius
        assert!(
            core_profiles
                .set_profiles_2d_irregular_rz_rs(
                    array![0.1, 0.2],
                    array![0.3, -0.4, 0.5],
                    array![0.0, 0.0, 0.0],
                    pressure.clone(),
                    pressure.clone()
                )
                .is_err()
        );
        // A pressure without one value per time and point
        assert!(
            core_profiles
                .set_profiles_2d_irregular_rz_rs(array![0.1, 0.2], array![0.3, 0.4], array![0.0, 0.0], pressure.clone(), pressure.clone())
                .is_err()
        );
    }

    /// Set the fast ions of the species `ion_name` at the times 0.1 and 0.2 second on `rho_pol_norm` = [0.0, 0.5, 1.0], with the pressures
    /// in proportion to `density_fast`
    fn set_fast_ions(
        core_profiles: &mut CoreProfiles,
        ion_name: &str,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        density_fast: &Array2<f64>,
    ) -> Result<(), String> {
        core_profiles.set_profiles_1d_fast_ions_rs(
            array![0.1, 0.2],
            array![0.0, 0.5, 1.0],
            ion_name.to_string(),
            ion_element_a,
            ion_element_z_n,
            ion_z_ion,
            density_fast.clone(),
            density_fast * 1.0e-15,
            density_fast * 2.0e-15,
        )
    }

    #[test]
    fn fast_ions_are_set_on_profiles_1d_one_ion_per_species() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let density_fast: Array2<f64> = array![[1.0e18, 2.0e18, 3.0e18], [4.0e18, 5.0e18, 6.0e18]];

        set_fast_ions(&mut core_profiles, "D", 2.0141, 1, 1.0, &density_fast).expect("the fast ions are consistent");

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d.len(), 2);
        assert_eq!(profiles_1d[1].time, 0.2);
        assert_eq!(profiles_1d[1].grid.rho_pol_norm, array![0.0, 0.5, 1.0]);
        assert!(profiles_1d[1].grid.rho_tor_norm.is_empty());
        assert_eq!(profiles_1d[1].ion.len(), 1);
        let ion: &CoreProfileIons = &profiles_1d[1].ion[0];
        assert_eq!(ion.name, "D");
        assert_eq!(ion.element.len(), 1);
        assert_eq!(ion.element[0].a, 2.0141);
        assert_eq!(ion.element[0].z_n, 1);
        assert_eq!(ion.element[0].atoms_n, 1);
        assert_eq!(ion.z_ion, 1.0);
        assert_eq!(ion.density_fast, density_fast.row(1));
        assert_eq!(ion.pressure_fast_parallel, (&density_fast * 1.0e-15).row(1));
        assert_eq!(ion.pressure_fast_perpendicular, (&density_fast * 2.0e-15).row(1));
        assert!(ion.density_thermal.is_empty());

        // Another species is added after the first
        set_fast_ions(&mut core_profiles, "He", 4.0026, 2, 2.0, &density_fast).expect("the fast helium is consistent");
        // The fast ions of a species already there are replaced
        let density_fast_doubled: Array2<f64> = &density_fast * 2.0;
        set_fast_ions(&mut core_profiles, "D", 2.0141, 1, 1.0, &density_fast_doubled).expect("the fast ions are consistent");

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d[0].ion.len(), 2);
        assert_eq!(profiles_1d[0].ion[0].name, "D");
        assert_eq!(profiles_1d[0].ion[1].name, "He");
        assert_eq!(profiles_1d[0].ion[0].density_fast, density_fast_doubled.row(0));
        assert_eq!(profiles_1d[0].ion[1].density_fast, density_fast.row(0));
    }

    #[test]
    fn fast_ions_inconsistent_with_profiles_1d_are_an_error_and_change_nothing() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let density_fast: Array2<f64> = Array2::from_elem((2, 3), 1.0e18);
        set_fast_ions(&mut core_profiles, "D", 2.0141, 1, 1.0, &density_fast).expect("the fast ions are consistent");
        let density_fast_other: Array2<f64> = Array2::from_elem((2, 3), 7.0e18);

        // A species of the same name which is another species
        assert!(set_fast_ions(&mut core_profiles, "D", 3.016, 1, 1.0, &density_fast_other).is_err());
        // A species which is not physical
        assert!(set_fast_ions(&mut core_profiles, "X", 2.0141, 0, 1.0, &density_fast_other).is_err());
        // A profile without one value per time and radial point
        assert!(set_fast_ions(&mut core_profiles, "D", 2.0141, 1, 1.0, &Array2::from_elem((2, 2), 1.0e18)).is_err());
        // Another grid than `profiles_1d`'s
        assert!(
            core_profiles
                .set_profiles_1d_fast_ions_rs(
                    array![0.1, 0.2],
                    array![0.0, 0.4, 1.0],
                    "He".to_string(),
                    4.0026,
                    2,
                    2.0,
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                )
                .is_err()
        );
        // Other times than `profiles_1d`'s
        assert!(
            core_profiles
                .set_profiles_1d_fast_ions_rs(
                    array![0.1, 0.3],
                    array![0.0, 0.5, 1.0],
                    "He".to_string(),
                    4.0026,
                    2,
                    2.0,
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                )
                .is_err()
        );
        // A radial grid which does not increase
        assert!(
            core_profiles
                .set_profiles_1d_fast_ions_rs(
                    array![0.1, 0.2],
                    array![0.0, 1.0, 0.5],
                    "He".to_string(),
                    4.0026,
                    2,
                    2.0,
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                    density_fast_other.clone(),
                )
                .is_err()
        );

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d[0].ion.len(), 1);
        assert_eq!(profiles_1d[0].ion[0].density_fast, density_fast.row(0));
    }
    #[test]
    fn thermal_profiles_are_set_on_profiles_1d_and_share_the_ions_with_the_fast_ions() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let time: Array1<f64> = array![0.1, 0.2];
        let rho_pol_norm: Array1<f64> = array![0.0, 0.5, 1.0];
        let electrons_density: Array2<f64> = array![[4.0e19, 3.0e19, 1.0e19], [5.0e19, 4.0e19, 2.0e19]];
        let electrons_temperature: Array2<f64> = array![[1000.0, 500.0, 50.0], [1200.0, 600.0, 60.0]];
        let zeff: Array2<f64> = array![[2.0, 2.0, f64::NAN], [2.0, 1.5, 1.0]];
        let e_field_parallel: Array2<f64> = Array2::from_elem((2, 3), 0.05);

        core_profiles
            .set_profiles_1d_electrons_rs(time.clone(), rho_pol_norm.clone(), electrons_density.clone(), electrons_temperature.clone())
            .expect("the electrons are consistent");
        core_profiles
            .set_profiles_1d_zeff_rs(time.clone(), rho_pol_norm.clone(), zeff.clone())
            .expect("the effective charge is consistent, NaN included");
        core_profiles
            .set_profiles_1d_thermal_ions_rs(
                time.clone(),
                rho_pol_norm.clone(),
                "D".to_string(),
                2.0141,
                1,
                1.0,
                &electrons_density * 0.8,
                electrons_temperature.clone(),
            )
            .expect("the deuterium is consistent");
        core_profiles
            .set_profiles_1d_thermal_ions_rs(
                time.clone(),
                rho_pol_norm.clone(),
                "C".to_string(),
                12.011,
                6,
                6.0,
                &electrons_density / 30.0,
                electrons_temperature.clone(),
            )
            .expect("the carbon is consistent");
        set_fast_ions(&mut core_profiles, "D", 2.0141, 1, 1.0, &(&electrons_density * 0.1)).expect("the fast deuterium is consistent");
        core_profiles
            .set_profiles_1d_e_field_parallel_rs(time.clone(), rho_pol_norm.clone(), e_field_parallel.clone(), 0.45, array![1.8, 1.7])
            .expect("the parallel electric field is consistent");

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d.len(), 2);
        assert_eq!(profiles_1d[1].time, 0.2);
        assert_eq!(profiles_1d[1].grid.rho_pol_norm, rho_pol_norm);
        assert_eq!(profiles_1d[1].electrons.density_thermal, electrons_density.row(1));
        assert_eq!(profiles_1d[1].electrons.temperature, electrons_temperature.row(1));
        assert_eq!(profiles_1d[1].zeff, zeff.row(1));
        assert!(profiles_1d[0].zeff[2].is_nan());
        assert_eq!(profiles_1d[1].e_field.parallel, e_field_parallel.row(1));
        assert_eq!(core_profiles.core_profiles_ids.vacuum_toroidal_field.r0, 0.45);
        assert_eq!(core_profiles.core_profiles_ids.vacuum_toroidal_field.b0, array![1.8, 1.7]);
        // The fast deuterium shares the thermal deuterium's `ion(0)`, and the carbon is `ion(1)`
        assert_eq!(profiles_1d[1].ion.len(), 2);
        assert_eq!(profiles_1d[1].ion[0].name, "D");
        assert_eq!(profiles_1d[1].ion[0].density_thermal, (&electrons_density * 0.8).row(1));
        assert_eq!(profiles_1d[1].ion[0].temperature, electrons_temperature.row(1));
        assert_eq!(profiles_1d[1].ion[0].density_fast, (&electrons_density * 0.1).row(1));
        assert_eq!(profiles_1d[1].ion[1].name, "C");
        assert_eq!(profiles_1d[1].ion[1].z_ion, 6.0);
        assert_eq!(profiles_1d[1].ion[1].element[0].z_n, 6);
        assert_eq!(profiles_1d[1].ion[1].element[0].atoms_n, 1);
        assert!(profiles_1d[1].ion[1].density_fast.is_empty());
    }

    #[test]
    fn thermal_profiles_set_again_are_replaced() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let time: Array1<f64> = array![0.1, 0.2];
        let rho_pol_norm: Array1<f64> = array![0.0, 0.5, 1.0];
        let density: Array2<f64> = Array2::from_elem((2, 3), 1.0e19);
        let temperature: Array2<f64> = Array2::from_elem((2, 3), 100.0);

        for multiplier in [1.0, 3.0] {
            core_profiles
                .set_profiles_1d_electrons_rs(time.clone(), rho_pol_norm.clone(), &density * multiplier, temperature.clone())
                .expect("the electrons are consistent");
            core_profiles
                .set_profiles_1d_thermal_ions_rs(
                    time.clone(),
                    rho_pol_norm.clone(),
                    "D".to_string(),
                    2.0141,
                    1,
                    1.0,
                    &density * multiplier,
                    temperature.clone(),
                )
                .expect("the deuterium is consistent");
        }

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d[0].ion.len(), 1);
        assert_eq!(profiles_1d[0].ion[0].density_thermal, (&density * 3.0).row(0));
        assert_eq!(profiles_1d[0].electrons.density_thermal, (&density * 3.0).row(0));
    }

    #[test]
    fn inconsistent_thermal_profiles_are_an_error_and_change_nothing() {
        let mut core_profiles: CoreProfiles = core_profiles();
        let time: Array1<f64> = array![0.1, 0.2];
        let rho_pol_norm: Array1<f64> = array![0.0, 0.5, 1.0];
        let density: Array2<f64> = Array2::from_elem((2, 3), 1.0e19);
        let temperature: Array2<f64> = Array2::from_elem((2, 3), 100.0);
        core_profiles
            .set_profiles_1d_electrons_rs(time.clone(), rho_pol_norm.clone(), density.clone(), temperature.clone())
            .expect("the electrons are consistent");
        core_profiles
            .set_profiles_1d_thermal_ions_rs(
                time.clone(),
                rho_pol_norm.clone(),
                "D".to_string(),
                2.0141,
                1,
                1.0,
                density.clone(),
                temperature.clone(),
            )
            .expect("the deuterium is consistent");
        let density_other: Array2<f64> = Array2::from_elem((2, 3), 7.0e19);

        // An effective charge below 1
        assert!(
            core_profiles
                .set_profiles_1d_zeff_rs(time.clone(), rho_pol_norm.clone(), Array2::from_elem((2, 3), 0.9))
                .is_err()
        );
        // An infinite effective charge
        assert!(
            core_profiles
                .set_profiles_1d_zeff_rs(time.clone(), rho_pol_norm.clone(), Array2::from_elem((2, 3), f64::INFINITY))
                .is_err()
        );
        // Electrons on another grid than `profiles_1d`'s
        assert!(
            core_profiles
                .set_profiles_1d_electrons_rs(time.clone(), array![0.0, 0.4, 1.0], density_other.clone(), temperature.clone())
                .is_err()
        );
        // Electrons at other times than `profiles_1d`'s
        assert!(
            core_profiles
                .set_profiles_1d_electrons_rs(array![0.1, 0.3], rho_pol_norm.clone(), density_other.clone(), temperature.clone())
                .is_err()
        );
        // A temperature without one value per time and radial point
        assert!(
            core_profiles
                .set_profiles_1d_electrons_rs(time.clone(), rho_pol_norm.clone(), density_other.clone(), Array2::from_elem((2, 2), 100.0))
                .is_err()
        );
        // A species of the same name which is another species
        assert!(
            core_profiles
                .set_profiles_1d_thermal_ions_rs(
                    time.clone(),
                    rho_pol_norm.clone(),
                    "D".to_string(),
                    3.016,
                    1,
                    1.0,
                    density_other.clone(),
                    temperature.clone(),
                )
                .is_err()
        );
        // A vacuum toroidal field without one value per time, of 0, or at a negative radius
        let e_field_parallel: Array2<f64> = Array2::from_elem((2, 3), 0.05);
        assert!(
            core_profiles
                .set_profiles_1d_e_field_parallel_rs(time.clone(), rho_pol_norm.clone(), e_field_parallel.clone(), 0.45, array![1.8])
                .is_err()
        );
        assert!(
            core_profiles
                .set_profiles_1d_e_field_parallel_rs(time.clone(), rho_pol_norm.clone(), e_field_parallel.clone(), 0.45, array![1.8, 0.0])
                .is_err()
        );
        assert!(
            core_profiles
                .set_profiles_1d_e_field_parallel_rs(time.clone(), rho_pol_norm.clone(), e_field_parallel.clone(), -0.45, array![1.8, 1.8])
                .is_err()
        );

        let profiles_1d: &Vec<CoreProfilesProfiles1d> = &core_profiles.core_profiles_ids.profiles_1d;
        assert_eq!(profiles_1d[0].electrons.density_thermal, density.row(0));
        assert!(profiles_1d[0].zeff.is_empty());
        assert_eq!(profiles_1d[0].ion.len(), 1);
        assert_eq!(profiles_1d[0].ion[0].density_thermal, density.row(0));
        assert!(profiles_1d[0].e_field.parallel.is_empty());
        assert!(core_profiles.core_profiles_ids.vacuum_toroidal_field.b0.is_empty());
    }
}
