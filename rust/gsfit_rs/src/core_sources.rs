use crate::core_profiles::{check_ion_species, check_profile_shape, check_time_and_rho_pol_norm, check_vacuum_toroidal_field};
use crate::settings::settings_core_sources::SettingsCoreSources;
use imas_rs::ids::core_sources::{
    Code, CoreRadialGrid, CoreSources as CoreSourcesIds, CoreSourcesSource, CoreSourcesSourceProfiles1d, DistributionSpecies, Identifier,
    PlasmaCompositionIonsConstant, PlasmaCompositionNeutralElementConstant,
};
use imas_rs::python::{PickleState, PickleStateError, PyCoreSources, PyParametersInput, PyPath, read_path};
use ndarray::{Array1, Array2};
use numpy::PyArrayMethods;
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use oxml::Document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};
use xmlschema::{Report, Schema, SchemaError};

/// The neutral beams' `core_sources/source(i1)/identifier`, from the data dictionary's `core_source_identifier.xml`
const NBI_IDENTIFIER_NAME: &str = "NBI";
const NBI_IDENTIFIER_INDEX: i32 = 2;
const NBI_IDENTIFIER_DESCRIPTION: &str = "Source from Neutral Beam Injection";

/// The neutral beams' `core_sources/source(i1)/species/type`, an ion, from the data dictionary's `species_reference_identifier.xml`
const SPECIES_TYPE_ION_NAME: &str = "ion";
const SPECIES_TYPE_ION_INDEX: i32 = 2;
const SPECIES_TYPE_ION_DESCRIPTION: &str = "Ion species in a single/average state; refer to ion-structure";

/// Where the neutral beams' 1D profiles are, for the errors
const NBI_PROFILES_1D_PATH: &str = "`core_sources/source(NBI)/profiles_1d`";

/// The sources of the plasma, stored as an IMAS `core_sources` IDS.
///
/// The neutral beams' source is set from Python, by `set_source_nbi_profiles_1d`: `source(i1)`, with `identifier/name` "NBI", holding
/// * `species/type` - an ion, and `species/ion/name`, `/element(1)/a`, `/element(1)/z_n`, `/element(1)/atoms_n` and `/z_ion` - the beam's
///   species, whose charge, `Z_b`, the electrons' shielding needs
/// * `profiles_1d(itime)`, one per time, on the radial grid `grid/rho_pol_norm = sqrt(psi_norm)`, e.g. from ASCOT5: `time` [second],
///   `grid/rho_pol_norm` [dimensionless], and `j_parallel`, the fast ions' parallel current, `<J_fast . B> / B0`, with `B0` =
///   `core_sources/vacuum_toroidal_field/b0` [ampere / metre ** 2]
///
/// `j_parallel` is the current of the fast ions alone, before the electrons shield it. The data dictionary's `j_parallel` is the current
/// a source drives, which for the neutral beams is after the shielding, `1 - (Z_b / Z_eff) (1 - L31)`. But GSFit applies the shielding
/// itself, with the same trapped fraction and `L31` as the bootstrap current, so it keeps the current the fast-ion code gives:
/// `gsfit_rs.calculate_plasma_current_components` makes the beam-driven current from it, and does not write that back here.
///
/// `gsfit_rs.calculate_plasma_current_components` also writes the bootstrap and ohmic currents here, as the sources
/// "bootstrap_current" and "ohmic", on the equilibrium's flux surfaces and times.
///
/// Every source shares `core_sources/vacuum_toroidal_field`, which normalises `j_parallel`, and the times of its `profiles_1d`.
///
/// Read the data back through `get`, or `core_sources_ids`, and a path from `gsfit_rs.imas.core_sources_paths`; there are no bespoke
/// accessors.
///
/// # Settings
///
/// The sources are constructed with the settings they are built from, as an IMAS `parameters_input`. They are validated against the XML
/// schema in `parameters_input`, and only their XML is kept: it is written to `core_sources/code/parameters`, where IMAS records the
/// parameters of the code which made an IDS.
///
/// They are also read into `settings_core_sources`, whose structs are generated from the XML schema `core_sources.xsd`. They are kept
/// here, not in the IDS, as the core_sources IDS can also be made by another code, whose `core_sources/code/parameters` would not hold
/// GSFit's settings.
#[pyclass(module = "gsfit_rs")]
pub struct CoreSources {
    pub core_sources_ids: CoreSourcesIds,
    #[allow(dead_code, reason = "Standard design: the struct keeps its typed settings, even if nothing reads them yet")]
    pub settings_core_sources: SettingsCoreSources,
}

/// Python accessible methods
#[pymethods]
impl CoreSources {
    /// Construct an empty core_sources IDS from its settings.
    /// Later its sources will be set, e.g. from a fast-ion code.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the core_sources IDS is built from:
    ///   - `parameters_input.parameters_value` = XML data as a string
    ///   - `parameters_input.schema` = XML schema as a string
    ///   - `parameters_input.defaults` = default XML values as a string
    ///
    /// # Errors:
    /// * `ValueError` if:
    ///   - there is no XSD schema, or it cannot be read
    ///   - there is no XML, or it is not well-formed
    ///   - the XML does not follow the XSD schema
    ///   - the XML cannot be read into `settings_core_sources` data types
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> PyResult<Self> {
        // Create schema from the XSD string
        let schema: Schema = xmlschema::parse_schema(&parameters_input.inner.schema).map_err(|error: SchemaError| {
            PyValueError::new_err(format!(
                "`parameters_input.schema`, the XML schema of the sources' settings, cannot be read: {error}"
            ))
        })?;
        // Create XML document from the XML string
        let settings_document: Document = oxml::parse(&parameters_input.inner.parameters_value).map_err(|error: oxml::Error| {
            PyValueError::new_err(format!(
                "`parameters_input.parameters_value`, the sources' settings, is not well-formed XML: {error}"
            ))
        })?;
        // Validate the XML document against the schema
        let report: Report = xmlschema::validate(&settings_document, &schema);
        if !report.is_valid() {
            return Err(PyValueError::new_err(format!(
                "The sources' settings do not follow their XML schema:\n{report}"
            )));
        }

        // Parse the XML string settings into a typed `SettingsCoreSources` struct
        let settings_core_sources: SettingsCoreSources = SettingsCoreSources::from_xml(&parameters_input.inner.parameters_value)
            .map_err(|error: String| PyValueError::new_err(format!("The sources' settings cannot be read into `SettingsCoreSources`: {error}")))?;

        // Construct the core_sources IDS from the parsed settings
        let core_sources_ids: CoreSourcesIds = CoreSourcesIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            core_sources_ids,
            settings_core_sources,
        })
    }

    /// Set the neutral beams' source, `core_sources/source(i1)` with `identifier/name` "NBI": the beam's species, `species`, and its
    /// `profiles_1d(itime)/j_parallel`, one per time, on the radial grid `grid/rho_pol_norm`, and `core_sources/vacuum_toroidal_field`,
    /// which normalises it. Any neutral beams' source already set is replaced; otherwise it is added after the other sources.
    ///
    /// Every source shares the times and `vacuum_toroidal_field` of `core_sources`. To have the bootstrap and ohmic currents written beside
    /// it by `gsfit_rs.calculate_plasma_current_components`, set it at the equilibrium's times, with NaN at a time with no fast ions, and
    /// with the equilibrium's `vacuum_toroidal_field`.
    ///
    /// # Arguments
    /// * `time` - the times (1d array), [second]
    /// * `rho_pol_norm` - the radial grid, `grid/rho_pol_norm = sqrt(psi_norm)` (1d array), [dimensionless]
    /// * `ion_name` - the beam's species, `species/ion/name`, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, `species/ion/element(1)/a`, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, `species/ion/element(1)/z_n`, [dimensionless]
    /// * `ion_z_ion` - its charge, `Z_b`, `species/ion/z_ion`, [dimensionless]
    /// * `j_parallel` - the fast ions' parallel current, `<J_fast . B> / B0`, before the electrons shield it, `profiles_1d(itime)/j_parallel`
    ///   (2d array, shape = [n_time, n_rho_pol_norm]), [ampere / metre ** 2]
    /// * `vacuum_toroidal_field_r0` - the major radius at which `vacuum_toroidal_field_b0` is given, `vacuum_toroidal_field/r0`, [metre]
    /// * `vacuum_toroidal_field_b0` - the vacuum toroidal field at `vacuum_toroidal_field_r0`, `B0`, which `j_parallel` is normalised by,
    ///   `vacuum_toroidal_field/b0` (1d array, shape = [n_time]), [tesla]
    ///
    /// # Errors
    /// A `ValueError` when there is no time or no radial point, a time is not finite or the times do not increase, a radial point is not
    /// finite and at least 0 or the radial points do not increase, the species is not named or not physical, `j_parallel` does not have one
    /// value per time and radial point, the vacuum toroidal field is not a positive finite radius and one finite, non-zero field per time,
    /// or another source is at other times or normalised by another vacuum toroidal field.
    #[allow(clippy::too_many_arguments)]
    pub fn set_source_nbi_profiles_1d(
        &mut self,
        time: PyReadonlyArray1<f64>,
        rho_pol_norm: PyReadonlyArray1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        j_parallel: PyReadonlyArray2<f64>,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let time_rs: Array1<f64> = time.to_owned_array();
        let rho_pol_norm_rs: Array1<f64> = rho_pol_norm.to_owned_array();
        let j_parallel_rs: Array2<f64> = j_parallel.to_owned_array();
        let vacuum_toroidal_field_b0_rs: Array1<f64> = vacuum_toroidal_field_b0.to_owned_array();

        // Run the Rust method
        self.set_source_nbi_profiles_1d_rs(
            time_rs,
            rho_pol_norm_rs,
            ion_name,
            ion_element_a,
            ion_element_z_n,
            ion_z_ion,
            j_parallel_rs,
            vacuum_toroidal_field_r0,
            vacuum_toroidal_field_b0_rs,
        )
        .map_err(PyValueError::new_err)
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.core_sources_paths`, straight out of the core_sources IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached by its data dictionary path. A path
    /// holds no data, so the IDS is only borrowed for the read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.core_sources_ids, "core_sources", path)
    }

    /// A copy of the whole core_sources IDS, for reading with `gsfit_rs.imas.core_sources_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a detached snapshot is wanted: changes made
    /// on the Rust side afterwards are not seen by it.
    #[getter]
    fn core_sources_ids(&self) -> PyCoreSources {
        PyCoreSources::new(self.core_sources_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.CoreSources>");
        string_output += &format!("║  {:<74} ║\n", version);

        let sources: &Vec<CoreSourcesSource> = &self.core_sources_ids.source;
        let n_source: usize = sources.len();
        if n_source == 0 {
            string_output += &format!("║  {:<74} ║\n", " source is unset");
        }
        for i_source in 0..n_source {
            let profiles_1d: &Vec<CoreSourcesSourceProfiles1d> = &sources[i_source].profiles_1d;
            let n_time: usize = profiles_1d.len();
            let n_rho_pol_norm: usize = if n_time > 0 { profiles_1d[0].grid.rho_pol_norm.len() } else { 0 };
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(
                    " source({i_source}) = {}: {n_time} time(s), {n_rho_pol_norm} point(s) in rho_pol_norm",
                    sources[i_source].identifier.name
                )
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }

    /// Pickle this. Its state is a `dict` of the version of `gsfit_rs` which pickled it, `version`, when, `pickled_datetime`, in
    /// seconds since 1970, and its data: `core_sources_ids`, the `dict` of the IDS's nodes, see `imas_rs::python::PickleState`.
    /// `_from_pickle_state` reads it back.
    ///
    /// The settings, `settings_core_sources`, are not pickled: they are read again from the XML which `core_sources_ids` holds in
    /// `code/parameters`, which they were read from.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state: Bound<'py, PyDict> = PyDict::new(py);
        state.set_item("version", env!("CARGO_PKG_VERSION"))?;
        let pickled_datetime: f64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?
            .as_secs_f64();
        state.set_item("pickled_datetime", pickled_datetime)?;
        state.set_item("core_sources_ids", self.core_sources_ids.to_pickle_state(py)?)?;
        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))
    }

    /// Unpickle: read back the state which `__reduce__` made.
    ///
    /// # Errors
    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of `gsfit_rs` with a different
    /// data dictionary, or its settings cannot be read into `settings_core_sources`. A different version of `gsfit_rs` whose state is
    /// the same only gives a warning
    #[staticmethod]
    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let version: String = state
            .get_item("version")?
            .ok_or_else(|| PyValueError::new_err("The pickled `CoreSources` has no `version`"))?
            .extract()?;
        if version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "Warning: the `CoreSources` being unpickled was pickled by gsfit_rs v{version}, but this is v{}",
                env!("CARGO_PKG_VERSION")
            );
        }
        let core_sources_ids_state: Bound<'_, PyAny> = state
            .get_item("core_sources_ids")?
            .ok_or_else(|| PyValueError::new_err("The pickled `CoreSources` has no `core_sources_ids`"))?;
        let core_sources_ids: CoreSourcesIds =
            PickleState::from_pickle_state(&core_sources_ids_state).map_err(|error: PickleStateError| error.into_py_err("`CoreSources.core_sources_ids`"))?;
        // The settings are read from the XML which they were read from by `new`, which the IDS holds in `code/parameters`
        let settings_core_sources: SettingsCoreSources = SettingsCoreSources::from_xml(&core_sources_ids.code.parameters).map_err(|error: String| {
            PyValueError::new_err(format!(
                "The pickled `CoreSources`'s settings cannot be read into `SettingsCoreSources`: {error}"
            ))
        })?;

        Ok(Self {
            core_sources_ids,
            settings_core_sources,
        })
    }
}

/// Rust only methods
impl CoreSources {
    /// Set the neutral beams' source. This does the work of the Python method `set_source_nbi_profiles_1d`, so that Rust code can set it
    /// too; see it for what is stored.
    ///
    /// # Arguments
    /// * `time` - the times, [second]
    /// * `rho_pol_norm` - the radial grid, `sqrt(psi_norm)`, [dimensionless]
    /// * `ion_name` - the beam's species, e.g. "D"
    /// * `ion_element_a` - the mass of its nucleus, [atomic_mass_unit]
    /// * `ion_element_z_n` - the charge of its nucleus, [dimensionless]
    /// * `ion_z_ion` - its charge, `Z_b`, [dimensionless]
    /// * `j_parallel` - the fast ions' parallel current, `<J_fast . B> / B0`, before the electrons shield it, shape = [n_time, n_rho_pol_norm],
    ///   [ampere / metre ** 2]
    /// * `vacuum_toroidal_field_r0` - the major radius at which `vacuum_toroidal_field_b0` is given, [metre]
    /// * `vacuum_toroidal_field_b0` - the vacuum toroidal field at `vacuum_toroidal_field_r0`, shape = [n_time], [tesla]
    ///
    /// # Errors
    /// Why the source cannot be set: see `set_source_nbi_profiles_1d`. Nothing is changed when it cannot.
    #[allow(clippy::too_many_arguments)]
    pub fn set_source_nbi_profiles_1d_rs(
        &mut self,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        ion_name: String,
        ion_element_a: f64,
        ion_element_z_n: i32,
        ion_z_ion: f64,
        j_parallel: Array2<f64>,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: Array1<f64>,
    ) -> Result<(), String> {
        check_time_and_rho_pol_norm(NBI_PROFILES_1D_PATH, &time, &rho_pol_norm)?;
        check_ion_species("`core_sources/source(NBI)/species`", &ion_name, ion_element_a, ion_element_z_n, ion_z_ion)?;
        let n_time: usize = time.len();
        let n_rho_pol_norm: usize = rho_pol_norm.len();
        check_profile_shape(NBI_PROFILES_1D_PATH, "j_parallel", &j_parallel, n_time, n_rho_pol_norm)?;
        check_vacuum_toroidal_field(
            "core_sources/vacuum_toroidal_field",
            vacuum_toroidal_field_r0,
            &vacuum_toroidal_field_b0,
            n_time,
        )?;

        // Every source shares the times and `vacuum_toroidal_field` of `core_sources`, so the other sources must be at the neutral beams'
        // times, normalised by their vacuum toroidal field
        let n_source: usize = self.core_sources_ids.source.len();
        let mut i_source_nbi: Option<usize> = None;
        let mut n_source_other: usize = 0;
        for i_source in 0..n_source {
            let source: &CoreSourcesSource = &self.core_sources_ids.source[i_source];
            if source.identifier.index == NBI_IDENTIFIER_INDEX {
                i_source_nbi = Some(i_source);
                continue;
            }
            n_source_other += 1;
            let n_time_source: usize = source.profiles_1d.len();
            let mut is_at_same_times: bool = n_time_source == n_time;
            if is_at_same_times {
                for i_time in 0..n_time {
                    if source.profiles_1d[i_time].time != time[i_time] {
                        is_at_same_times = false;
                    }
                }
            }
            if !is_at_same_times {
                return Err(format!(
                    "`core_sources/source({i_source})`, `{}`, is not at the neutral beams' times; every source shares the times of `core_sources`",
                    source.identifier.name
                ));
            }
        }
        if n_source_other > 0
            && (self.core_sources_ids.vacuum_toroidal_field.r0 != vacuum_toroidal_field_r0
                || self.core_sources_ids.vacuum_toroidal_field.b0 != vacuum_toroidal_field_b0)
        {
            return Err("`core_sources/vacuum_toroidal_field` is not the neutral beams', so the sources would be normalised differently".to_string());
        }

        let mut profiles_1d: Vec<CoreSourcesSourceProfiles1d> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            profiles_1d.push(CoreSourcesSourceProfiles1d {
                grid: CoreRadialGrid {
                    rho_pol_norm: rho_pol_norm.clone(),
                    ..Default::default()
                },
                j_parallel: j_parallel.row(i_time).to_owned(),
                time: time[i_time],
                ..Default::default()
            });
        }
        let source_nbi: CoreSourcesSource = CoreSourcesSource {
            identifier: Identifier {
                name: NBI_IDENTIFIER_NAME.to_string(),
                index: NBI_IDENTIFIER_INDEX,
                description: NBI_IDENTIFIER_DESCRIPTION.to_string(),
            },
            species: DistributionSpecies {
                r#type: Identifier {
                    name: SPECIES_TYPE_ION_NAME.to_string(),
                    index: SPECIES_TYPE_ION_INDEX,
                    description: SPECIES_TYPE_ION_DESCRIPTION.to_string(),
                },
                ion: PlasmaCompositionIonsConstant {
                    element: vec![PlasmaCompositionNeutralElementConstant {
                        a: ion_element_a,
                        z_n: ion_element_z_n,
                        atoms_n: 1,
                    }],
                    z_ion: ion_z_ion,
                    name: ion_name,
                    ..Default::default()
                },
                ..Default::default()
            },
            profiles_1d,
            ..Default::default()
        };
        match i_source_nbi {
            Some(i_source) => self.core_sources_ids.source[i_source] = source_nbi,
            None => self.core_sources_ids.source.push(source_nbi),
        }
        self.core_sources_ids.vacuum_toroidal_field.r0 = vacuum_toroidal_field_r0;
        self.core_sources_ids.vacuum_toroidal_field.b0 = vacuum_toroidal_field_b0;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imas_rs::dd_support::parameters_input::ParametersInput;
    use ndarray::array;

    /// The XML schema the settings' structs are generated from
    const SCHEMA_XSD: &str = include_str!("../../../python/gsfit/settings_schema/xsd/core_sources/core_sources.xsd");

    /// `core_sources.xml` without its optional link to `workflow.xml`, so it follows `core_sources.xsd`, in which `workflow` is still a
    /// link
    const SETTINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<core_sources_settings/>
"#;

    fn core_sources() -> CoreSources {
        let parameters_input: PyParametersInput = PyParametersInput::new(ParametersInput {
            parameters_value: SETTINGS_XML.to_string(),
            parameters_default: String::new(),
            schema: SCHEMA_XSD.to_string(),
        });
        CoreSources::new(&parameters_input).expect("the settings follow their XML schema")
    }

    /// Set the neutral beams' source, of deuterium, at the times 0.1 and 0.2 second on `rho_pol_norm` = [0.0, 0.5, 1.0], normalised by
    /// 1.8 tesla at 0.45 metre
    fn set_source_nbi(core_sources: &mut CoreSources, j_parallel: &Array2<f64>) -> Result<(), String> {
        core_sources.set_source_nbi_profiles_1d_rs(
            array![0.1, 0.2],
            array![0.0, 0.5, 1.0],
            "D".to_string(),
            2.0141,
            1,
            1.0,
            j_parallel.clone(),
            0.45,
            array![1.8, 1.8],
        )
    }

    /// Set the neutral beams' source, of deuterium, with any coordinates and vacuum toroidal field
    fn set_source_nbi_at(
        core_sources: &mut CoreSources,
        time: Array1<f64>,
        rho_pol_norm: Array1<f64>,
        j_parallel: &Array2<f64>,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: Array1<f64>,
    ) -> Result<(), String> {
        core_sources.set_source_nbi_profiles_1d_rs(
            time,
            rho_pol_norm,
            "D".to_string(),
            2.0141,
            1,
            1.0,
            j_parallel.clone(),
            vacuum_toroidal_field_r0,
            vacuum_toroidal_field_b0,
        )
    }

    #[test]
    fn the_settings_are_written_to_code_parameters() {
        let core_sources: CoreSources = core_sources();

        assert_eq!(core_sources.core_sources_ids.code.parameters, SETTINGS_XML);
        assert!(core_sources.settings_core_sources.workflow.is_none());
        assert!(core_sources.core_sources_ids.source.is_empty());
    }

    #[test]
    fn the_nbi_source_is_set_one_profiles_1d_per_time() {
        let mut core_sources: CoreSources = core_sources();
        let j_parallel: Array2<f64> = array![[1.0e5, 2.0e5, 0.0], [3.0e5, 4.0e5, f64::NAN]];

        set_source_nbi(&mut core_sources, &j_parallel).expect("the source is consistent");

        assert_eq!(core_sources.core_sources_ids.source.len(), 1);
        let source: &CoreSourcesSource = &core_sources.core_sources_ids.source[0];
        assert_eq!(source.identifier.name, "NBI");
        assert_eq!(source.identifier.index, 2);
        assert_eq!(source.species.r#type.name, "ion");
        assert_eq!(source.species.r#type.index, 2);
        assert_eq!(source.species.ion.name, "D");
        assert_eq!(source.species.ion.element.len(), 1);
        assert_eq!(source.species.ion.element[0].a, 2.0141);
        assert_eq!(source.species.ion.element[0].z_n, 1);
        assert_eq!(source.species.ion.element[0].atoms_n, 1);
        assert_eq!(source.species.ion.z_ion, 1.0);
        assert_eq!(source.profiles_1d.len(), 2);
        assert_eq!(source.profiles_1d[1].time, 0.2);
        assert_eq!(source.profiles_1d[1].grid.rho_pol_norm, array![0.0, 0.5, 1.0]);
        assert!(source.profiles_1d[1].grid.rho_tor_norm.is_empty());
        assert_eq!(source.profiles_1d[0].j_parallel, j_parallel.row(0));
        assert_eq!(source.profiles_1d[1].j_parallel[1], 4.0e5);
        assert!(source.profiles_1d[1].j_parallel[2].is_nan());
        assert_eq!(core_sources.core_sources_ids.vacuum_toroidal_field.r0, 0.45);
        assert_eq!(core_sources.core_sources_ids.vacuum_toroidal_field.b0, array![1.8, 1.8]);

        // Setting it again replaces it
        let j_parallel_doubled: Array2<f64> = &j_parallel * 2.0;
        set_source_nbi(&mut core_sources, &j_parallel_doubled).expect("the source is consistent");
        assert_eq!(core_sources.core_sources_ids.source.len(), 1);
        assert_eq!(core_sources.core_sources_ids.source[0].profiles_1d[1].j_parallel[1], 8.0e5);
    }

    #[test]
    fn an_inconsistent_nbi_source_is_an_error_and_changes_nothing() {
        let mut core_sources: CoreSources = core_sources();
        let j_parallel: Array2<f64> = Array2::from_elem((2, 3), 1.0e5);

        // Times which do not increase
        assert!(set_source_nbi_at(&mut core_sources, array![0.2, 0.1], array![0.0, 0.5, 1.0], &j_parallel, 0.45, array![1.8, 1.8]).is_err());
        // A current without one value per time and radial point
        assert!(set_source_nbi_at(&mut core_sources, array![0.1, 0.2], array![0.0, 1.0], &j_parallel, 0.45, array![1.8, 1.8]).is_err());
        // A vacuum toroidal field without one value per time
        assert!(set_source_nbi_at(&mut core_sources, array![0.1, 0.2], array![0.0, 0.5, 1.0], &j_parallel, 0.45, array![1.8]).is_err());
        // A vacuum toroidal field of 0, which cannot normalise the current
        assert!(set_source_nbi_at(&mut core_sources, array![0.1, 0.2], array![0.0, 0.5, 1.0], &j_parallel, 0.45, array![1.8, 0.0]).is_err());
        // A species which is not physical
        assert!(
            core_sources
                .set_source_nbi_profiles_1d_rs(
                    array![0.1, 0.2],
                    array![0.0, 0.5, 1.0],
                    "D".to_string(),
                    2.0141,
                    1,
                    0.0,
                    j_parallel.clone(),
                    0.45,
                    array![1.8, 1.8],
                )
                .is_err()
        );
        assert!(core_sources.core_sources_ids.source.is_empty());

        // Another source at other times
        core_sources.core_sources_ids.source.push(CoreSourcesSource {
            identifier: Identifier {
                name: "bootstrap_current".to_string(),
                index: 13,
                description: String::new(),
            },
            profiles_1d: vec![CoreSourcesSourceProfiles1d {
                time: 0.1,
                ..Default::default()
            }],
            ..Default::default()
        });
        assert!(set_source_nbi(&mut core_sources, &j_parallel).is_err());
        assert_eq!(core_sources.core_sources_ids.source.len(), 1);
    }
}
