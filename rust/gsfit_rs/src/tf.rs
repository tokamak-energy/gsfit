use imas_rs::ids::tf::{SignalFlt1d, Tf as TfIds};
use imas_rs::python::PyTf;
use ndarray::Array1;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The machine's toroidal field, stored as an IMAS `tf` IDS.
///
/// Two things are filled:
/// * `tf/r0` - the reference major radius the vacuum toroidal field is quoted at
/// * `tf/b_field_phi_vacuum_r` - the vacuum field times major radius, on the experimental
///   timebase
///
/// `b_field_phi_vacuum_r` is the poloidal-current function in vacuum, which GSFit writes as
/// `f_vac = R0 * B_phi0 = MU_0 * i_rod / (2 * PI)`. So the rod current is not stored
/// separately: it is `i_rod = 2 * PI * f_vac / MU_0`, and `solve_grad_shafranov` converts it
/// where it is needed. The data dictionary has no node for a rod current outside
/// `coil(i)/current`, and one signal cannot disagree with itself.
///
/// # Sign
///
/// `b_field_phi_vacuum_r` is signed: the data dictionary reads "Positive sign means
/// counter-clockwise when viewed from above". The sign carries through to `f_vac`, and from
/// there into `f`, `q`, `profiles_2d/b_field_phi` and the diamagnetic flux constraint, so a
/// database reader must supply it signed rather than as a magnitude.
///
/// Read the data back through `tf_ids` and a path from `gsfit_rs.imas.tf_paths`; there are no
/// bespoke accessors.
///
/// The coil geometry (`tf/coil`), the field map (`tf/field_map`) and the periodicity flags are
/// deliberately left empty: GSFit models the toroidal field as axisymmetric and needs only the
/// two nodes above.
#[pyclass(module = "gsfit_rs")]
pub struct Tf {
    pub tf_ids: TfIds,
}

impl Default for Tf {
    fn default() -> Self {
        return Self::new();
    }
}

/// Python accessible methods
#[pymethods]
impl Tf {
    /// Construct an empty toroidal field, ready for `set_r0` and `set_b_field_phi_vacuum_r`.
    #[new]
    pub fn new() -> Self {
        return Self { tf_ids: TfIds::default() };
    }

    /// Set `tf/r0`, the reference major radius.
    ///
    /// # Arguments
    /// * `r0` - reference major radius the vacuum toroidal field is quoted at, [metre]
    ///
    /// This is the machine's official reference radius, typically the middle of the vessel at
    /// the equatorial midplane. It is copied onto `equilibrium/vacuum_toroidal_field/r0` by the
    /// solver, so that the two IDSs cannot disagree.
    pub fn set_r0(&mut self, r0: f64) -> PyResult<()> {
        if !r0.is_finite() || r0 <= 0.0 {
            return Err(PyValueError::new_err(format!("`tf/r0` must be a positive major radius, got {r0}")));
        }

        self.tf_ids.r0 = r0;

        return Ok(());
    }

    /// Set `tf/b_field_phi_vacuum_r`, the vacuum field times major radius.
    ///
    /// # Arguments
    /// * `time` - the experimental timebase (1d array), [second]
    /// * `data` - vacuum toroidal field times major radius (1d array), [tesla * metre]
    ///
    /// Store the **experimental** signal here, not one interpolated onto the reconstruction
    /// times: `solve_grad_shafranov` interpolates it itself, so that changing the reconstruction
    /// times does not require re-reading the database.
    ///
    /// `data` is signed; see the `Tf` documentation. A reader holding a rod current instead
    /// should pass `MU_0 * i_rod / (2 * PI)`.
    pub fn set_b_field_phi_vacuum_r(&mut self, time: PyReadonlyArray1<f64>, data: PyReadonlyArray1<f64>) -> PyResult<()> {
        // Change Python types into Rust types
        let signal_time: Array1<f64> = time.to_owned_array();
        let signal_data: Array1<f64> = data.to_owned_array();

        if signal_time.len() != signal_data.len() {
            return Err(PyValueError::new_err(format!(
                "`tf/b_field_phi_vacuum_r`: `time` has {} point(s) but `data` has {}; a signal needs one value per time",
                signal_time.len(),
                signal_data.len()
            )));
        }

        if signal_time.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "`tf/b_field_phi_vacuum_r`: {} point(s) is not enough to interpolate from; at least 2 are needed",
                signal_time.len()
            )));
        }

        self.tf_ids.b_field_phi_vacuum_r = SignalFlt1d {
            data: signal_data,
            time: signal_time,
        };

        return Ok(());
    }

    /// The tf IDS, for reading with `gsfit_rs.imas.tf_paths`.
    ///
    /// This is the only way to read the data back out: there are no bespoke accessors, so
    /// every quantity is reached by its data dictionary path.
    ///
    /// The IDS is copied into the returned object, so it is a snapshot: changes made on the
    /// Rust side afterwards are not seen by it.
    #[getter]
    fn tf_ids(&self) -> PyTf {
        return PyTf::new(self.tf_ids.clone());
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.Tf>");
        string_output += &format!("║  {:<74} ║\n", version);

        let r0: f64 = self.tf_ids.r0;
        if r0.is_nan() {
            string_output += &format!("║  {:<74} ║\n", " r0 is unset");
        } else {
            string_output += &format!("║  {:<74} ║\n", format!(" r0 = {r0} metre"));
        }

        let signal_time: &Array1<f64> = &self.tf_ids.b_field_phi_vacuum_r.time;
        let signal_data: &Array1<f64> = &self.tf_ids.b_field_phi_vacuum_r.data;
        if signal_time.is_empty() || signal_data.is_empty() {
            string_output += &format!("║  {:<74} ║\n", " b_field_phi_vacuum_r is unset");
        } else {
            let n_time: usize = signal_time.len();
            string_output += &format!("║  {:<74} ║\n", format!(" b_field_phi_vacuum_r: {n_time} time point(s)"));
            string_output += &format!("║  {:<74} ║\n", format!(" time = [{}, {}] second", signal_time[0], signal_time[n_time - 1]));
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(" data = [{}, {}] tesla * metre", signal_data[0], signal_data[n_time - 1])
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }
}
