//! The plasma current's components from Python: `calculate_plasma_current_components`, which calculates every component from the IDSs, and
//! the classes it returns them in, `PlasmaCurrentComponents` and `PlasmaCurrentComponent`.

use super::collisionality::electrons_collisionality_norm;
use super::current_beam_driven::current_beam_driven;
use super::current_bootstrap::current_bootstrap;
use super::current_diamagnetic::current_diamagnetic;
use super::current_ohmic::current_ohmic;
use super::current_pfirsch_schluter::current_pfirsch_schluter;
use super::plasma_current_components::{PlasmaCurrentComponent, PlasmaCurrentComponents, PlasmaCurrentComponentsGrid, PlasmaCurrentComponentsTimeSlice};
use crate::core_profiles::CoreProfiles;
use crate::core_sources::CoreSources;
use crate::equilibrium::Equilibrium;
use imas_rs::Equilibrium as EquilibriumIds;
use imas_rs::ids::core_profiles::CoreProfiles as CoreProfilesIds;
use imas_rs::ids::core_sources::CoreSources as CoreSourcesIds;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The neutral beams' `core_sources/source(i1)/identifier/index`, from the data dictionary's `core_source_identifier.xml`
const NBI_IDENTIFIER_INDEX: i32 = 2;

/// Calculate the components of the toroidal plasma current on every time-slice of the equilibrium, from the IDSs.
///
/// In this order:
/// * the diamagnetic and Pfirsch-Schlüter currents, from the equilibrium's `p'` and flux surfaces
/// * the bootstrap current, from the kinetic profiles, written to `core_sources` as the source "bootstrap_current"
/// * the ohmic current, from the kinetic profiles' parallel electric field, written to `core_sources` as the source "ohmic", with the
///   neoclassical conductivity
/// * the beam-driven current, only when `core_sources` has the neutral beams' source ("NBI"): its fast ions' current, shielded by the
///   electrons, which is not written to `core_sources`
///
/// and the electron collisionality, which the bootstrap current and the neoclassical conductivity take, is written to the kinetic profiles,
/// as `core_profiles/profiles_1d(:)/electrons/collisionality_norm`, at the points of their own grid.
///
/// A time-slice the solver could not use, or without the kinetic profiles, or the neutral beams' source, which a component needs at its
/// time, has NaN in that component. The sources written are on the equilibrium's flux surfaces and times, and every source of `core_sources`
/// shares its times and `vacuum_toroidal_field`, so any source already in `core_sources`, such as the neutral beams', must be at the
/// equilibrium's times, with the equilibrium's `vacuum_toroidal_field`.
///
/// # Arguments
/// * `equilibrium` - the post-processed equilibrium; only its `equilibrium_ids` is read
/// * `core_profiles` - the kinetic profiles, on `core_profiles/profiles_1d(:)`, see `gsfit_rs.CoreProfiles`, which the electron
///   collisionality is written into
/// * `core_sources` - the plasma's sources, which the bootstrap and ohmic currents are written into
///
/// # Returns
/// The components, `PlasmaCurrentComponents`
///
/// # Errors
/// A `ValueError` when a component or the electron collisionality cannot be calculated, with the reason; `core_profiles` and `core_sources`
/// are not changed then.
#[pyfunction]
pub fn calculate_plasma_current_components(
    equilibrium: PyRef<'_, Equilibrium>,
    mut core_profiles: PyRefMut<'_, CoreProfiles>,
    mut core_sources: PyRefMut<'_, CoreSources>,
) -> PyResult<PyPlasmaCurrentComponents> {
    let equilibrium_ids: &EquilibriumIds = &equilibrium.equilibrium_ids;
    // The kinetic profiles and the sources are written into copies, which replace `core_profiles` and `core_sources` once everything has
    // been calculated, so that they are not changed when something cannot be
    let mut core_profiles_ids: CoreProfilesIds = core_profiles.core_profiles_ids.clone();
    let mut core_sources_ids: CoreSourcesIds = core_sources.core_sources_ids.clone();
    let mut plasma_current_components: PlasmaCurrentComponents = PlasmaCurrentComponents::new(equilibrium_ids);

    current_diamagnetic(equilibrium_ids, &mut plasma_current_components).map_err(PyValueError::new_err)?;
    current_pfirsch_schluter(equilibrium_ids, &mut plasma_current_components).map_err(PyValueError::new_err)?;
    current_bootstrap(equilibrium_ids, &core_profiles_ids, &mut core_sources_ids, &mut plasma_current_components).map_err(PyValueError::new_err)?;
    current_ohmic(equilibrium_ids, &core_profiles_ids, &mut core_sources_ids, &mut plasma_current_components).map_err(PyValueError::new_err)?;
    let n_source: usize = core_sources_ids.source.len();
    let mut with_source_nbi: bool = false;
    for i_source in 0..n_source {
        if core_sources_ids.source[i_source].identifier.index == NBI_IDENTIFIER_INDEX {
            with_source_nbi = true;
        }
    }
    if with_source_nbi {
        current_beam_driven(equilibrium_ids, &core_profiles_ids, &core_sources_ids, &mut plasma_current_components).map_err(PyValueError::new_err)?;
    }
    electrons_collisionality_norm(equilibrium_ids, &mut core_profiles_ids).map_err(PyValueError::new_err)?;

    core_profiles.core_profiles_ids = core_profiles_ids;
    core_sources.core_sources_ids = core_sources_ids;

    Ok(PyPlasmaCurrentComponents {
        inner: plasma_current_components,
    })
}

/// The toroidal plasma current, split into its components, on the flux surfaces of each equilibrium time-slice, as
/// `calculate_plasma_current_components` makes it.
///
/// Every quantity is an array of shape [n_time, n_psi_norm], one row per equilibrium time-slice, on its flux surfaces, `psi_norm`. A
/// component which was not calculated, e.g. the beam-driven current without the neutral beams' source, is NaN.
#[pyclass(module = "gsfit_rs", name = "PlasmaCurrentComponents")]
pub struct PyPlasmaCurrentComponents {
    pub inner: PlasmaCurrentComponents,
}

/// One component of the toroidal plasma current, on the flux surfaces of each equilibrium time-slice.
#[pyclass(module = "gsfit_rs", name = "PlasmaCurrentComponent")]
pub struct PyPlasmaCurrentComponent {
    /// The component's name, e.g. "bootstrap"
    name: String,
    /// The toroidal current density on each surface, `<j_phi / R> / <1 / R>`, shape = [n_time, n_psi_norm] [ampere / metre ** 2]
    j_phi: Array2<f64>,
    /// The toroidal current inside each surface, shape = [n_time, n_psi_norm] [ampere]
    current_phi_inside: Array2<f64>,
    /// The flux-surface averaged parallel current on each surface, `<J . B> / B0`, shape = [n_time, n_psi_norm] [ampere / metre ** 2]
    j_parallel: Array2<f64>,
}

/// Python accessible methods
#[pymethods]
impl PyPlasmaCurrentComponents {
    /// The equilibrium time-slices' times, shape = [n_time] [second]
    #[getter]
    fn time<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n_time: usize = self.inner.time_slice.len();
        let mut time: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
        for i_time in 0..n_time {
            time[i_time] = self.inner.time_slice[i_time].time;
        }

        time.into_pyarray(py)
    }

    /// The poloidal flux of each flux surface, the equilibrium's `profiles_1d/psi`, shape = [n_time, n_psi_norm] [weber]
    #[getter]
    fn psi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.grid_quantity("psi", |grid: &PlasmaCurrentComponentsGrid| &grid.psi)?.into_pyarray(py))
    }

    /// The normalised poloidal flux of each flux surface, the equilibrium's `profiles_1d/psi_norm`, shape = [n_time, n_psi_norm]
    /// [dimensionless]
    #[getter]
    fn psi_norm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .grid_quantity("psi_norm", |grid: &PlasmaCurrentComponentsGrid| &grid.psi_norm)?
            .into_pyarray(py))
    }

    /// The normalised toroidal flux coordinate of each flux surface, the equilibrium's `profiles_1d/rho_tor_norm`,
    /// shape = [n_time, n_psi_norm] [dimensionless]
    #[getter]
    fn rho_tor_norm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .grid_quantity("rho_tor_norm", |grid: &PlasmaCurrentComponentsGrid| &grid.rho_tor_norm)?
            .into_pyarray(py))
    }

    /// The poloidal cross-section inside each flux surface, the equilibrium's `profiles_1d/area`, shape = [n_time, n_psi_norm] [metre ** 2]
    #[getter]
    fn area<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.grid_quantity("area", |grid: &PlasmaCurrentComponentsGrid| &grid.area)?.into_pyarray(py))
    }

    /// The beam-driven current: the neutral beams' fast ions' current, shielded by the electrons
    #[getter]
    fn beam_driven(&self) -> PyResult<PyPlasmaCurrentComponent> {
        self.component("beam_driven", |time_slice: &PlasmaCurrentComponentsTimeSlice| &time_slice.beam_driven)
    }

    /// The bootstrap current
    #[getter]
    fn bootstrap(&self) -> PyResult<PyPlasmaCurrentComponent> {
        self.component("bootstrap", |time_slice: &PlasmaCurrentComponentsTimeSlice| &time_slice.bootstrap)
    }

    /// The diamagnetic current
    #[getter]
    fn diamagnetic(&self) -> PyResult<PyPlasmaCurrentComponent> {
        self.component("diamagnetic", |time_slice: &PlasmaCurrentComponentsTimeSlice| &time_slice.diamagnetic)
    }

    /// The ohmic current
    #[getter]
    fn ohmic(&self) -> PyResult<PyPlasmaCurrentComponent> {
        self.component("ohmic", |time_slice: &PlasmaCurrentComponentsTimeSlice| &time_slice.ohmic)
    }

    /// The Pfirsch-Schlüter current
    #[getter]
    fn pfirsch_schluter(&self) -> PyResult<PyPlasmaCurrentComponent> {
        self.component("pfirsch_schluter", |time_slice: &PlasmaCurrentComponentsTimeSlice| &time_slice.pfirsch_schluter)
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let n_time: usize = self.inner.time_slice.len();

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.PlasmaCurrentComponents>");
        if n_time == 0 {
            string_output += &format!("║  {:<74} ║\n", " no time-slice");
        } else {
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(
                    " {n_time} time-slice(s), time = [{}, {}] second",
                    self.inner.time_slice[0].time,
                    self.inner.time_slice[n_time - 1].time
                )
            );
            // Which components were calculated, on any time-slice
            let mut component_names: Vec<&str> = Vec::new();
            let mut with_beam_driven: bool = false;
            let mut with_bootstrap: bool = false;
            let mut with_diamagnetic: bool = false;
            let mut with_ohmic: bool = false;
            let mut with_pfirsch_schluter: bool = false;
            for i_time in 0..n_time {
                let time_slice: &PlasmaCurrentComponentsTimeSlice = &self.inner.time_slice[i_time];
                with_beam_driven = with_beam_driven || !time_slice.beam_driven.j_phi.is_empty();
                with_bootstrap = with_bootstrap || !time_slice.bootstrap.j_phi.is_empty();
                with_diamagnetic = with_diamagnetic || !time_slice.diamagnetic.j_phi.is_empty();
                with_ohmic = with_ohmic || !time_slice.ohmic.j_phi.is_empty();
                with_pfirsch_schluter = with_pfirsch_schluter || !time_slice.pfirsch_schluter.j_phi.is_empty();
            }
            if with_beam_driven {
                component_names.push("beam_driven");
            }
            if with_bootstrap {
                component_names.push("bootstrap");
            }
            if with_diamagnetic {
                component_names.push("diamagnetic");
            }
            if with_ohmic {
                component_names.push("ohmic");
            }
            if with_pfirsch_schluter {
                component_names.push("pfirsch_schluter");
            }
            string_output += &format!("║  {:<74} ║\n", format!(" calculated = [{}]", component_names.join(", ")));
        }
        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}

/// Rust only methods
impl PyPlasmaCurrentComponents {
    /// The number of flux surfaces of every time-slice: the most any has; a time-slice without flux surfaces is NaN.
    ///
    /// # Errors
    /// Two time-slices have different numbers of flux surfaces, so their quantities cannot be rows of one array
    fn n_psi_norm(&self) -> PyResult<usize> {
        let n_time: usize = self.inner.time_slice.len();
        let mut n_psi_norm: usize = 0;
        for i_time in 0..n_time {
            let n_psi_norm_here: usize = self.inner.time_slice[i_time].grid.psi_norm.len();
            if n_psi_norm_here == 0 {
                continue;
            }
            if n_psi_norm > 0 && n_psi_norm_here != n_psi_norm {
                return Err(PyValueError::new_err(format!(
                    "the plasma current components' time-slice {i_time} has {n_psi_norm_here} flux surfaces, but another has {n_psi_norm}, so they \
                     cannot be the rows of one array"
                )));
            }
            n_psi_norm = n_psi_norm_here;
        }

        Ok(n_psi_norm)
    }

    /// A quantity of the flux surfaces, one row per time-slice.
    ///
    /// # Arguments
    /// * `name` - the quantity's name, for the errors
    /// * `quantity_of` - the quantity, of a time-slice's flux surfaces
    ///
    /// # Errors
    /// The time-slices' flux surfaces cannot be the rows of one array
    fn grid_quantity(&self, name: &str, quantity_of: fn(&PlasmaCurrentComponentsGrid) -> &Array1<f64>) -> PyResult<Array2<f64>> {
        let n_time: usize = self.inner.time_slice.len();
        let mut rows: Vec<&Array1<f64>> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            rows.push(quantity_of(&self.inner.time_slice[i_time].grid));
        }

        stack_rows(name, &rows, self.n_psi_norm()?).map_err(PyValueError::new_err)
    }

    /// A component, one row per time-slice.
    ///
    /// # Arguments
    /// * `name` - the component's name
    /// * `component_of` - the component, of a time-slice
    ///
    /// # Errors
    /// The time-slices' flux surfaces cannot be the rows of one array
    fn component(&self, name: &str, component_of: fn(&PlasmaCurrentComponentsTimeSlice) -> &PlasmaCurrentComponent) -> PyResult<PyPlasmaCurrentComponent> {
        let n_time: usize = self.inner.time_slice.len();
        let n_psi_norm: usize = self.n_psi_norm()?;
        let mut j_phi_rows: Vec<&Array1<f64>> = Vec::with_capacity(n_time);
        let mut current_phi_inside_rows: Vec<&Array1<f64>> = Vec::with_capacity(n_time);
        let mut j_parallel_rows: Vec<&Array1<f64>> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            let component: &PlasmaCurrentComponent = component_of(&self.inner.time_slice[i_time]);
            j_phi_rows.push(&component.j_phi);
            current_phi_inside_rows.push(&component.current_phi_inside);
            j_parallel_rows.push(&component.j_parallel);
        }

        Ok(PyPlasmaCurrentComponent {
            name: name.to_string(),
            j_phi: stack_rows(&format!("{name}.j_phi"), &j_phi_rows, n_psi_norm).map_err(PyValueError::new_err)?,
            current_phi_inside: stack_rows(&format!("{name}.current_phi_inside"), &current_phi_inside_rows, n_psi_norm).map_err(PyValueError::new_err)?,
            j_parallel: stack_rows(&format!("{name}.j_parallel"), &j_parallel_rows, n_psi_norm).map_err(PyValueError::new_err)?,
        })
    }
}

/// Python accessible methods
#[pymethods]
impl PyPlasmaCurrentComponent {
    /// The toroidal current density on each flux surface, `<j_phi / R> / <1 / R>`, as the data dictionary defines
    /// `core_profiles/profiles_1d/j_phi`, so that its integral over the cross-section is the toroidal current, shape = [n_time, n_psi_norm]
    /// [ampere / metre ** 2]
    #[getter]
    fn j_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.j_phi.clone().into_pyarray(py)
    }

    /// The toroidal current inside each flux surface, the integral of `j_phi` over the cross-section, by the trapezium rule in `area`,
    /// shape = [n_time, n_psi_norm] [ampere]
    #[getter]
    fn current_phi_inside<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.current_phi_inside.clone().into_pyarray(py)
    }

    /// The flux-surface averaged parallel current on each flux surface, `<J . B> / B0`, with `B0` the equilibrium's
    /// `vacuum_toroidal_field/b0`, as the data dictionary defines `core_profiles/profiles_1d/j_total`; zero for the diamagnetic and
    /// Pfirsch-Schlüter currents, shape = [n_time, n_psi_norm] [ampere / metre ** 2]
    #[getter]
    fn j_parallel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.j_parallel.clone().into_pyarray(py)
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let (n_time, n_psi_norm): (usize, usize) = self.j_phi.dim();
        format!("PlasmaCurrentComponent({}: {n_time} time-slice(s), {n_psi_norm} flux surface(s))", self.name)
    }
}

/// Rows of one length stacked into an array, one per time-slice; an empty row, of a time-slice without the quantity, is NaN.
///
/// # Arguments
/// * `name` - the quantity's name, for the errors
/// * `rows` - one row per time-slice, each empty or of `n_psi_norm` values
/// * `n_psi_norm` - the number of flux surfaces
///
/// # Returns
/// The rows, shape = [n_time, n_psi_norm]
///
/// # Errors
/// A row is neither empty nor of `n_psi_norm` values
fn stack_rows(name: &str, rows: &[&Array1<f64>], n_psi_norm: usize) -> Result<Array2<f64>, String> {
    let n_time: usize = rows.len();
    let mut stacked: Array2<f64> = Array2::from_elem((n_time, n_psi_norm), f64::NAN);
    for i_time in 0..n_time {
        let row: &Array1<f64> = rows[i_time];
        if row.is_empty() {
            continue;
        }
        if row.len() != n_psi_norm {
            return Err(format!(
                "the plasma current components' `{name}` has {} values on time-slice {i_time}, but its flux surfaces are {n_psi_norm}",
                row.len()
            ));
        }
        for i_psi_norm in 0..n_psi_norm {
            stacked[(i_time, i_psi_norm)] = row[i_psi_norm];
        }
    }

    Ok(stacked)
}
