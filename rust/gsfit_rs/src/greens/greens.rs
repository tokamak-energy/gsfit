//! `Greens`: the Green's functions between sensors and conductors, and between conductors, which the rest of GSFit uses.
//!
//! Each table holds the flux per ampere, or one of its derivatives with respect to the sensor's position, from each conductor at each
//! sensor; between conductors it holds their self- and mutual inductances. How every conductor is treated is chosen by
//! `ConductorModel`, from the settings `equilibrium_settings/numerics/greens`:
//! * `ConductorModel::Point` - each conductor is a point in the (R, Z) plane, a filament at its centroid (`point_to_point.rs`)
//! * `ConductorModel::FiniteSize` - a uniform current density over each conductor's parallelogram cross-section, from analytic series
//!   (`finite_size_to_point.rs`, `finite_size_to_finite_size.rs`)
//!
//! If the selected model cannot calculate a table, the method returns an error; it never falls back to another model.

use super::finite_size_to_finite_size::FiniteSizeToFiniteSize;
use super::finite_size_to_point::FiniteSizeToPoint;
use super::geometry::{Conductor, Sensor};
use super::point_to_point::PointToPoint;
use crate::settings::settings_equilibrium::{SettingsEquilibriumNumericsGreens, SettingsEquilibriumNumericsGreensFiniteSize};
use approx::abs_diff_eq;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// The error for a derivative of a table between conductors
const DERIVATIVE_BETWEEN_CONDUCTORS: &str =
    "the derivatives of the flux are only available at sensors, from `Greens::sensor_to_conductor`; a table between conductors holds only `psi`";

/// Safety: a sensor which is closer than this to an edge or a corner of a conductor is taken to be on it, [metre]. It is far larger than the
/// rounding of a position, about 1e-16 metre at 1 metre, and far smaller than `SENSOR_SHIFT`, so a sensor which has been moved is clearly off
/// the edge
const SENSOR_ON_EDGE_TOLERANCE: f64 = 1.0e-12;

/// Safety: how far `Greens::sensor_to_conductor` moves a sensor which is on an edge or at a corner of a conductor, in both R and Z, with
/// `ConductorModel::FiniteSize`, [metre]. The 1 nm is arbitrary, but shouldn't cause us any problems
const SENSOR_SHIFT: f64 = 1.0e-9;

/// How the Green's functions treat every conductor.
#[derive(Clone, Debug)]
pub enum ConductorModel {
    /// Each conductor is a point in the (R, Z) plane: a filament at its centroid. A sensor on a conductor's centroid sees the conductor's
    /// own field instead, from closed-form expressions for a rectangle `length_alpha * cos(alpha)` by `length_beta * cos(beta)`
    Point,
    /// A uniform current density over each conductor's parallelogram cross-section, from analytic series, with the orders and the accuracy
    /// of `equilibrium_settings/numerics/greens/finite_size`
    FiniteSize(SettingsEquilibriumNumericsGreensFiniteSize),
}

impl ConductorModel {
    /// The model which the settings select.
    ///
    /// # Arguments
    /// * `settings_greens` - `equilibrium_settings/numerics/greens`
    ///
    /// # Returns
    /// * `conductor_model` - `ConductorModel::Point` for `point`, or `ConductorModel::FiniteSize` with the settings' `finite_size` for
    ///   `finite_size`
    ///
    /// # Errors
    /// * `conductor_model` is not recognized
    pub fn from_settings(settings_greens: &SettingsEquilibriumNumericsGreens) -> Result<Self, String> {
        match settings_greens.conductor_model.as_str() {
            "point" => Ok(ConductorModel::Point),
            "finite_size" => Ok(ConductorModel::FiniteSize(settings_greens.finite_size.clone())),
            conductor_model => Err(format!(
                "`equilibrium_settings/numerics/greens/conductor_model` = \"{conductor_model}\" is not recognized"
            )),
        }
    }
}

/// The Green's tables between sensors and conductors, or between conductors.
///
/// Built with one of three constructors, each with the `ConductorModel` to use:
/// * `sensor_to_conductor` - at points, e.g. magnetic sensors or the grid: the flux and its derivatives
/// * `conductor_to_conductor` - between two sets of conductors: the mutual inductances, `psi` only
/// * `conductor_to_self` - within one set of conductors: the self- and mutual inductances, `psi` only, calculated once for each pair
///
/// Every table has shape [n_row, n_column]: the sensors, or the first set of conductors, are the rows, and the conductors, or the second
/// set, are the columns.
///
/// # Example
/// ```
/// use gsfit_rs::greens::{Conductor, ConductorModel, Greens, Sensor};
/// use ndarray::{Array1, Array2, array};
///
/// // A flux loop
/// let sensor: Sensor = Sensor::new(&array![0.12345], &array![0.0]).unwrap();
///
/// // A pair of Helmholtz coils, each 1 mm square, described as IMAS describes an oblique: from the corner at the smallest r and z
/// let coil_r: Array1<f64> = array![1.23456789, 1.23456789];
/// let coil_z: Array1<f64> = array![-1.23456789 / 2.0, 1.23456789 / 2.0];
/// let coil_width: f64 = 1.0e-3;
/// let conductor: Conductor = Conductor::from_oblique(
///     &(&coil_r - 0.5 * coil_width),
///     &(&coil_z - 0.5 * coil_width),
///     &array![coil_width, coil_width],
///     &array![coil_width, coil_width],
///     &array![0.0, 0.0],
///     &array![0.0, 0.0],
/// )
/// .unwrap();
///
/// let greens_calculator: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::Point).unwrap();
/// let g_psi: Array2<f64> = greens_calculator.psi().unwrap(); // shape = [n_sensor, n_conductor]
/// println!("g_psi = {g_psi:#?}");
/// ```
pub struct Greens {
    /// Safety: whether each row's sensor was moved by `SENSOR_SHIFT` (1 nm) in both R and Z, off an edge or a corner of a conductor, before the
    /// tables were calculated; see `Greens::sensor_to_conductor`. Always false with `ConductorModel::Point`, and between conductors,
    /// shape = [n_row]
    pub is_sensor_shifted: Array1<bool>,
    /// The major radius of each row: the sensors, where they were used (after any safety shift), or the centroids of the first set of
    /// conductors, shape = [n_row], [metre]
    row_r: Array1<f64>,
    /// How the tables are calculated
    calculation: Calculation,
}

/// How the tables of a `Greens` are calculated: by the constructor and the `ConductorModel` it was built with
enum Calculation {
    /// `sensor_to_conductor` with `ConductorModel::Point`
    PointToPoint(PointToPoint),
    /// `sensor_to_conductor` with `ConductorModel::FiniteSize`; boxed, as it holds its tables once they are calculated
    FiniteSizeToPoint(Box<FiniteSizeToPoint>),
    /// `conductor_to_conductor` and `conductor_to_self` with `ConductorModel::Point`: `psi` only, between the centroids
    PointToPointBetweenConductors(PointToPoint),
    /// `conductor_to_conductor` and `conductor_to_self` with `ConductorModel::FiniteSize`: `psi` only; boxed, as it holds two sets of
    /// conductors
    FiniteSizeToFiniteSize(Box<FiniteSizeToFiniteSize>),
}

impl Greens {
    /// The Green's tables between sensors and conductors: the flux, and its derivatives with respect to the sensor's position.
    ///
    /// Safety: with `ConductorModel::FiniteSize`, a sensor which is on an edge or at a corner of a conductor, to within
    /// `SENSOR_ON_EDGE_TOLERANCE`, is first moved by `SENSOR_SHIFT` (1 nm) in both R and Z, and `is_sensor_shifted` records which were.
    ///
    /// # Arguments
    /// * `sensor` - the points where the flux is evaluated, the rows
    /// * `conductor` - the conductors, the columns
    /// * `conductor_model` - how the conductors are treated
    ///
    /// # Returns
    /// * `greens_calculator` - from which `psi`, `d_psi_d_r`, ... are calculated, each with shape = [n_sensor, n_conductor]
    ///
    /// # Errors
    /// * with `ConductorModel::FiniteSize`, a sensor which was on an edge or at a corner of a conductor is still on one after it is moved,
    ///   which needs an edge parallel to the shift, or two conductors closer together than the shift; the calculations themselves can also
    ///   fail, in the methods
    pub fn sensor_to_conductor(sensor: &Sensor, conductor: &Conductor, conductor_model: ConductorModel) -> Result<Self, String> {
        let n_sensor: usize = sensor.n_sensor;
        let n_conductor: usize = conductor.n_conductor;
        match conductor_model {
            ConductorModel::Point => {
                // The rectangle whose self-field a sensor on the conductor's centroid sees
                let mut conductor_d_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut conductor_d_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                for i_conductor in 0..n_conductor {
                    conductor_d_r[i_conductor] = conductor.length_alpha[i_conductor] * conductor.alpha[i_conductor].cos();
                    conductor_d_z[i_conductor] = conductor.length_beta[i_conductor] * conductor.beta[i_conductor].cos();
                }

                Ok(Greens {
                    is_sensor_shifted: Array1::from_elem(n_sensor, false),
                    row_r: sensor.r.clone(),
                    calculation: Calculation::PointToPoint(PointToPoint::new(
                        sensor.r.clone(),
                        sensor.z.clone(),
                        conductor.centroid_r.clone(),
                        conductor.centroid_z.clone(),
                        conductor_d_r,
                        conductor_d_z,
                        false,
                    )),
                })
            }
            ConductorModel::FiniteSize(settings_greens_finite_size) => {
                // Safety: a sensor exactly on an edge or at a corner of a conductor, e.g. a point typed onto a line of GSFit's grid, is moved
                // by `SENSOR_SHIFT` (1 nm) in both R and Z, here, before any conductor sees it. On an edge the second derivatives of the flux
                // jump, and at a corner they are infinite, and each conductor would decide from its own rounding whether the sensor is inside
                // it, so two conductors which share an edge could both count it, or neither. Once moved, the sensor is inside exactly one of
                // them, for every conductor, and its second derivatives are those of the side it was moved to; the flux and its first
                // derivatives are continuous, so they hardly change.
                // The 1 nm is arbitrary, but shouldn't cause us any problems: it is far smaller than any position we know, and far larger
                // than rounding.
                // TODO: let's revisit this later to see if this can be done without shift

                // Each conductor's two sides, as `Conductor` describes them, and the signed area they span
                let mut side_alpha_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut side_alpha_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut side_beta_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut side_beta_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut area_signed: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                for i_conductor in 0..n_conductor {
                    side_alpha_r[i_conductor] = conductor.length_alpha[i_conductor] * conductor.alpha[i_conductor].cos();
                    side_alpha_z[i_conductor] = conductor.length_alpha[i_conductor] * conductor.alpha[i_conductor].sin();
                    side_beta_r[i_conductor] = -conductor.length_beta[i_conductor] * conductor.beta[i_conductor].sin();
                    side_beta_z[i_conductor] = conductor.length_beta[i_conductor] * conductor.beta[i_conductor].cos();
                    area_signed[i_conductor] = side_alpha_r[i_conductor] * side_beta_z[i_conductor] - side_alpha_z[i_conductor] * side_beta_r[i_conductor];
                }

                // The first conductor, if any, which has the point `(point_r, point_z)` on one of its edges or at one of its corners, to
                // within `SENSOR_ON_EDGE_TOLERANCE`
                let conductor_with_point_on_edge = |point_r: f64, point_z: f64| -> Option<usize> {
                    for i_conductor in 0..n_conductor {
                        // The point's position from the reference corner as fractions of the two sides, `(lambda_alpha, lambda_beta)` with
                        // `point = corner + lambda_alpha * side_alpha + lambda_beta * side_beta`, and from them its distance from each of the
                        // four edges, positive on the inside
                        let point_from_corner_r: f64 = point_r - conductor.r[i_conductor];
                        let point_from_corner_z: f64 = point_z - conductor.z[i_conductor];
                        let lambda_alpha: f64 =
                            (point_from_corner_r * side_beta_z[i_conductor] - point_from_corner_z * side_beta_r[i_conductor]) / area_signed[i_conductor];
                        let lambda_beta: f64 =
                            (side_alpha_r[i_conductor] * point_from_corner_z - side_alpha_z[i_conductor] * point_from_corner_r) / area_signed[i_conductor];
                        let distance_between_sides_alpha: f64 = conductor.area[i_conductor] / conductor.length_alpha[i_conductor];
                        let distance_between_sides_beta: f64 = conductor.area[i_conductor] / conductor.length_beta[i_conductor];
                        let n_edge: usize = 4;
                        let edges_distance: [f64; 4] = [
                            lambda_beta * distance_between_sides_alpha,         // the side alpha through the reference corner
                            (1.0 - lambda_beta) * distance_between_sides_alpha, // the other side alpha
                            lambda_alpha * distance_between_sides_beta,         // the side beta through the reference corner
                            (1.0 - lambda_alpha) * distance_between_sides_beta, // the other side beta
                        ];

                        // On an edge or at a corner: not outside the conductor by more than the tolerance, and within it of an edge
                        let mut is_near_conductor: bool = true;
                        let mut is_on_edge: bool = false;
                        for i_edge in 0..n_edge {
                            if edges_distance[i_edge] < -SENSOR_ON_EDGE_TOLERANCE {
                                is_near_conductor = false;
                            }
                            if abs_diff_eq!(edges_distance[i_edge], 0.0, epsilon = SENSOR_ON_EDGE_TOLERANCE) {
                                is_on_edge = true;
                            }
                        }
                        if is_near_conductor && is_on_edge {
                            return Some(i_conductor);
                        }
                    }
                    None
                };

                // Move each sensor which is on an edge or at a corner; all the conductors then see it at the same place
                let mut sensor_shifted_r: Array1<f64> = sensor.r.clone();
                let mut sensor_shifted_z: Array1<f64> = sensor.z.clone();
                let mut is_sensor_shifted: Array1<bool> = Array1::from_elem(n_sensor, false);
                for i_sensor in 0..n_sensor {
                    if conductor_with_point_on_edge(sensor.r[i_sensor], sensor.z[i_sensor]).is_some() {
                        sensor_shifted_r[i_sensor] = sensor.r[i_sensor] + SENSOR_SHIFT;
                        sensor_shifted_z[i_sensor] = sensor.z[i_sensor] + SENSOR_SHIFT;
                        is_sensor_shifted[i_sensor] = true;

                        // Only an edge parallel to the shift, or two conductors closer together than the shift, can leave it on an edge
                        if let Some(i_conductor) = conductor_with_point_on_edge(sensor_shifted_r[i_sensor], sensor_shifted_z[i_sensor]) {
                            return Err(format!(
                                "sensor {i_sensor}, at r = {} and z = {} metre, is on an edge or at a corner of a conductor, and still is after the \
                                 safety shift of {SENSOR_SHIFT:e} metre in both r and z, of conductor {i_conductor}: an edge is parallel to the \
                                 shift, or two conductors are closer together than it",
                                sensor.r[i_sensor], sensor.z[i_sensor]
                            ));
                        }
                    }
                }
                let sensor_shifted: Sensor = Sensor::new(&sensor_shifted_r, &sensor_shifted_z)?;

                Ok(Greens {
                    is_sensor_shifted,
                    row_r: sensor_shifted.r.clone(),
                    calculation: Calculation::FiniteSizeToPoint(Box::new(FiniteSizeToPoint {
                        sensor: sensor_shifted,
                        conductor: conductor.clone(),
                        settings_greens_finite_size,
                        tables: OnceLock::new(),
                    })),
                })
            }
        }
    }

    /// The mutual inductances between two sets of conductors: the flux linking each conductor of the first set, per ampere in each conductor
    /// of the second.
    ///
    /// With `ConductorModel::Point` the flux is taken at the centroids of the first set, from filaments at the centroids of the second; a
    /// centroid of the first set which is on a centroid of the second sees the second's own field, from closed-form expressions for a
    /// rectangle. With `ConductorModel::FiniteSize` the flux is averaged over the cross-sections of both.
    ///
    /// # Arguments
    /// * `conductor_1` - the first set of conductors, the rows
    /// * `conductor_2` - the second set of conductors, the columns
    /// * `conductor_model` - how the conductors are treated
    ///
    /// # Returns
    /// * `greens_calculator` - from which `psi` is calculated, with shape = [n_conductor_1, n_conductor_2], [henry]; the derivatives are not
    ///   available
    ///
    /// # Errors
    /// None at present; the calculations themselves can fail, in the methods
    pub fn conductor_to_conductor(conductor_1: &Conductor, conductor_2: &Conductor, conductor_model: ConductorModel) -> Result<Self, String> {
        let calculation: Calculation = match conductor_model {
            ConductorModel::Point => {
                // The rectangle whose self-field a centroid of the first set on a centroid of the second sees
                let n_conductor_2: usize = conductor_2.n_conductor;
                let mut conductor_2_d_r: Array1<f64> = Array1::from_elem(n_conductor_2, f64::NAN);
                let mut conductor_2_d_z: Array1<f64> = Array1::from_elem(n_conductor_2, f64::NAN);
                for i_conductor_2 in 0..n_conductor_2 {
                    conductor_2_d_r[i_conductor_2] = conductor_2.length_alpha[i_conductor_2] * conductor_2.alpha[i_conductor_2].cos();
                    conductor_2_d_z[i_conductor_2] = conductor_2.length_beta[i_conductor_2] * conductor_2.beta[i_conductor_2].cos();
                }
                Calculation::PointToPointBetweenConductors(PointToPoint::new(
                    conductor_1.centroid_r.clone(),
                    conductor_1.centroid_z.clone(),
                    conductor_2.centroid_r.clone(),
                    conductor_2.centroid_z.clone(),
                    conductor_2_d_r,
                    conductor_2_d_z,
                    false,
                ))
            }
            ConductorModel::FiniteSize(settings_greens_finite_size) => Calculation::FiniteSizeToFiniteSize(Box::new(FiniteSizeToFiniteSize {
                conductor_1: conductor_1.clone(),
                conductor_2: conductor_2.clone(),
                is_symmetric: false,
                settings_greens_finite_size,
            })),
        };

        Ok(Greens {
            is_sensor_shifted: Array1::from_elem(conductor_1.n_conductor, false),
            row_r: conductor_1.centroid_r.clone(),
            calculation,
        })
    }

    /// The self- and mutual inductances within one set of conductors: the flux linking each conductor per ampere in each conductor.
    ///
    /// The matrix is symmetric, so each pair is calculated once, and the matrix is exactly symmetric. Otherwise the same as
    /// `conductor_to_conductor` with the same set twice.
    ///
    /// # Arguments
    /// * `conductor` - the conductors, both the rows and the columns
    /// * `conductor_model` - how the conductors are treated
    ///
    /// # Returns
    /// * `greens_calculator` - from which `psi` is calculated, with shape = [n_conductor, n_conductor], [henry]; the derivatives are not
    ///   available
    ///
    /// # Errors
    /// None at present; the calculations themselves can fail, in the methods
    pub fn conductor_to_self(conductor: &Conductor, conductor_model: ConductorModel) -> Result<Self, String> {
        let calculation: Calculation = match conductor_model {
            ConductorModel::Point => {
                // The rectangle whose self-field each centroid sees from its own conductor
                let n_conductor: usize = conductor.n_conductor;
                let mut conductor_d_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                let mut conductor_d_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
                for i_conductor in 0..n_conductor {
                    conductor_d_r[i_conductor] = conductor.length_alpha[i_conductor] * conductor.alpha[i_conductor].cos();
                    conductor_d_z[i_conductor] = conductor.length_beta[i_conductor] * conductor.beta[i_conductor].cos();
                }
                Calculation::PointToPointBetweenConductors(PointToPoint::new(
                    conductor.centroid_r.clone(),
                    conductor.centroid_z.clone(),
                    conductor.centroid_r.clone(),
                    conductor.centroid_z.clone(),
                    conductor_d_r,
                    conductor_d_z,
                    true,
                ))
            }
            ConductorModel::FiniteSize(settings_greens_finite_size) => Calculation::FiniteSizeToFiniteSize(Box::new(FiniteSizeToFiniteSize {
                conductor_1: conductor.clone(),
                conductor_2: conductor.clone(),
                is_symmetric: true,
                settings_greens_finite_size,
            })),
        };

        Ok(Greens {
            is_sensor_shifted: Array1::from_elem(conductor.n_conductor, false),
            row_r: conductor.centroid_r.clone(),
            calculation,
        })
    }

    /// The flux per ampere, `psi = g_psi * current`; between conductors, the mutual inductance.
    ///
    /// # Returns
    /// * `g_psi[(i_row, i_column)]` - shape = [n_row, n_column], [henry]
    ///
    /// # Errors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn psi(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.psi()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => finite_size_to_point.psi(),
            Calculation::PointToPointBetweenConductors(point_to_point) => Ok(point_to_point.psi()),
            Calculation::FiniteSizeToFiniteSize(finite_size_to_finite_size) => finite_size_to_finite_size.psi(),
        }
    }

    /// `d(psi)/d(r) = g_d_psi_d_r * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d_psi_d_r[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d_psi_d_r(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d_psi_d_r()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d_psi_d_r.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d(psi)/d(z) = g_d_psi_d_z * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d_psi_d_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d_psi_d_z(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d_psi_d_z()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d_psi_d_z.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d2(psi)/d(r2) = g_d2_psi_d_r2 * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d2_psi_d_r2[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 2]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d2_psi_d_r2(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d2_psi_d_r2()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d2_psi_d_r2.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d2(psi)/(d(r) d(z)) = g_d2_psi_d_r_d_z * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d2_psi_d_r_d_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 2]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d2_psi_d_r_d_z(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d2_psi_d_r_d_z()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d2_psi_d_r_d_z.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d2(psi)/d(z2) = g_d2_psi_d_z2 * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d2_psi_d_z2[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 2]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d2_psi_d_z2(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d2_psi_d_z2()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d2_psi_d_z2.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d3(psi)/(d(r2) d(z)) = g_d3_psi_d_r2_d_z * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d3_psi_d_r2_d_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 3]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d3_psi_d_r2_d_z(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d3_psi_d_r2_d_z()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d3_psi_d_r2_d_z.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d3(psi)/(d(r) d(z2)) = g_d3_psi_d_r_d_z2 * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d3_psi_d_r_d_z2[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 3]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d3_psi_d_r_d_z2(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d3_psi_d_r_d_z2()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d3_psi_d_r_d_z2.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `d3(psi)/d(z3) = g_d3_psi_d_z3 * current`, with respect to the sensor's position.
    ///
    /// # Returns
    /// * `g_d3_psi_d_z3[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [henry / metre ** 3]
    ///
    /// # Errors
    /// * the table is between conductors
    /// * the finite-size series cannot be calculated to the settings' accuracy, e.g. a conductor may not be halved often enough
    pub fn d3_psi_d_z3(&self) -> Result<Array2<f64>, String> {
        match &self.calculation {
            Calculation::PointToPoint(point_to_point) => Ok(point_to_point.d3_psi_d_z3()),
            Calculation::FiniteSizeToPoint(finite_size_to_point) => Ok(finite_size_to_point.tables()?.d3_psi_d_z3.clone()),
            Calculation::PointToPointBetweenConductors(_) | Calculation::FiniteSizeToFiniteSize(_) => Err(DERIVATIVE_BETWEEN_CONDUCTORS.to_string()),
        }
    }

    /// `b_r = g_b_r * current`, from `b_r = -d(psi)/d(z) / (2 * PI * r)`.
    ///
    /// # Returns
    /// * `g_b_r[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [tesla / ampere]
    ///
    /// # Errors
    /// * as `d_psi_d_z`
    pub fn b_r(&self) -> Result<Array2<f64>, String> {
        let mut g_b_r: Array2<f64> = self.d_psi_d_z()?;
        let n_row: usize = g_b_r.nrows();
        let n_column: usize = g_b_r.ncols();
        for i_row in 0..n_row {
            for i_column in 0..n_column {
                g_b_r[(i_row, i_column)] = -g_b_r[(i_row, i_column)] / (2.0 * PI * self.row_r[i_row]);
            }
        }

        Ok(g_b_r)
    }

    /// `b_z = g_b_z * current`, from `b_z = d(psi)/d(r) / (2 * PI * r)`.
    ///
    /// # Returns
    /// * `g_b_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [tesla / ampere]
    ///
    /// # Errors
    /// * as `d_psi_d_r`
    pub fn b_z(&self) -> Result<Array2<f64>, String> {
        let mut g_b_z: Array2<f64> = self.d_psi_d_r()?;
        let n_row: usize = g_b_z.nrows();
        let n_column: usize = g_b_z.ncols();
        for i_row in 0..n_row {
            for i_column in 0..n_column {
                g_b_z[(i_row, i_column)] /= 2.0 * PI * self.row_r[i_row];
            }
        }

        Ok(g_b_z)
    }

    /// `d(b_r)/d(z) = g_d_b_r_d_z * current`, from `d(b_r)/d(z) = -d2(psi)/d(z2) / (2 * PI * r)`.
    ///
    /// # Returns
    /// * `g_d_b_r_d_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [tesla / (ampere * metre)]
    ///
    /// # Errors
    /// * as `d2_psi_d_z2`
    pub fn d_b_r_d_z(&self) -> Result<Array2<f64>, String> {
        let mut g_d_b_r_d_z: Array2<f64> = self.d2_psi_d_z2()?;
        let n_row: usize = g_d_b_r_d_z.nrows();
        let n_column: usize = g_d_b_r_d_z.ncols();
        for i_row in 0..n_row {
            for i_column in 0..n_column {
                g_d_b_r_d_z[(i_row, i_column)] = -g_d_b_r_d_z[(i_row, i_column)] / (2.0 * PI * self.row_r[i_row]);
            }
        }

        Ok(g_d_b_r_d_z)
    }

    /// `d(b_z)/d(z) = g_d_b_z_d_z * current`, from `d(b_z)/d(z) = d2(psi)/(d(r) d(z)) / (2 * PI * r)`.
    ///
    /// # Returns
    /// * `g_d_b_z_d_z[(i_sensor, i_conductor)]` - shape = [n_sensor, n_conductor], [tesla / (ampere * metre)]
    ///
    /// # Errors
    /// * as `d2_psi_d_r_d_z`
    pub fn d_b_z_d_z(&self) -> Result<Array2<f64>, String> {
        let mut g_d_b_z_d_z: Array2<f64> = self.d2_psi_d_r_d_z()?;
        let n_row: usize = g_d_b_z_d_z.nrows();
        let n_column: usize = g_d_b_z_d_z.ncols();
        for i_row in 0..n_row {
            for i_column in 0..n_column {
                g_d_b_z_d_z[(i_row, i_column)] /= 2.0 * PI * self.row_r[i_row];
            }
        }

        Ok(g_d_b_z_d_z)
    }
}

#[cfg(test)]
mod tests {
    use super::super::finite_size_to_point::mutual_inductance_finite_size_to_point_source;
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::array;

    /// The settings the ST40 settings use
    fn settings_greens_finite_size() -> SettingsEquilibriumNumericsGreensFiniteSize {
        SettingsEquilibriumNumericsGreensFiniteSize {
            n_elliptic: 6,
            n_thin_ring: 12,
            n_multipole: 12,
            relative_tolerance: 1.0e-9,
            n_subdivision_max: 30,
        }
    }

    /// Three squares of 10 mm, one sheared, from their corners
    fn three_conductors() -> Conductor {
        Conductor::from_oblique(
            &array![0.50, 0.53, 0.70],
            &array![0.00, 0.02, -0.30],
            &array![0.010, 0.010, 0.010],
            &array![0.010, 0.010, 0.010],
            &array![0.0, 0.0, 0.1],
            &array![0.0, 0.0, -0.2],
        )
        .unwrap()
    }

    #[test]
    fn the_settings_select_the_model() {
        let mut settings_greens: SettingsEquilibriumNumericsGreens = SettingsEquilibriumNumericsGreens {
            conductor_model: "point".to_string(),
            finite_size: settings_greens_finite_size(),
        };
        assert!(matches!(ConductorModel::from_settings(&settings_greens), Ok(ConductorModel::Point)));

        settings_greens.conductor_model = "finite_size".to_string();
        assert!(matches!(ConductorModel::from_settings(&settings_greens), Ok(ConductorModel::FiniteSize(_))));

        settings_greens.conductor_model = "filament".to_string();
        assert!(ConductorModel::from_settings(&settings_greens).is_err());
    }

    #[test]
    fn conductor_to_self_is_conductor_to_conductor_with_the_same_set_and_exactly_symmetric() {
        let conductor: Conductor = three_conductors();
        for conductor_model in [ConductorModel::Point, ConductorModel::FiniteSize(settings_greens_finite_size())] {
            let g_self: Array2<f64> = Greens::conductor_to_self(&conductor, conductor_model.clone()).unwrap().psi().unwrap();
            let g_pair: Array2<f64> = Greens::conductor_to_conductor(&conductor, &conductor, conductor_model).unwrap().psi().unwrap();
            let n_conductor: usize = conductor.n_conductor;
            for i_conductor in 0..n_conductor {
                for j_conductor in 0..n_conductor {
                    assert_eq!(g_self[(i_conductor, j_conductor)], g_self[(j_conductor, i_conductor)]);
                    assert_relative_eq!(g_self[(i_conductor, j_conductor)], g_pair[(i_conductor, j_conductor)], max_relative = 1e-12);
                }
            }
        }
    }

    #[test]
    fn a_derivative_between_conductors_is_an_error() {
        let conductor: Conductor = three_conductors();
        for conductor_model in [ConductorModel::Point, ConductorModel::FiniteSize(settings_greens_finite_size())] {
            let greens_calculator: Greens = Greens::conductor_to_self(&conductor, conductor_model).unwrap();
            assert!(greens_calculator.psi().is_ok());
            assert!(greens_calculator.d_psi_d_r().is_err());
            assert!(greens_calculator.d3_psi_d_z3().is_err());
            assert!(greens_calculator.b_r().is_err());
        }
    }

    #[test]
    fn far_from_the_conductors_the_two_models_agree() {
        // 10 mm conductors seen from about 0.5 m: the finite size changes the flux by about (10 mm / 0.5 m) ** 2 / 24
        let conductor: Conductor = three_conductors();
        let sensor: Sensor = Sensor::new(&array![1.1, 0.2], &array![0.4, -0.5]).unwrap();
        let g_point: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::Point).unwrap();
        let g_finite_size: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::FiniteSize(settings_greens_finite_size())).unwrap();

        let tables_point: [Array2<f64>; 3] = [g_point.psi().unwrap(), g_point.d_psi_d_r().unwrap(), g_point.d2_psi_d_z2().unwrap()];
        let tables_finite_size: [Array2<f64>; 3] = [
            g_finite_size.psi().unwrap(),
            g_finite_size.d_psi_d_r().unwrap(),
            g_finite_size.d2_psi_d_z2().unwrap(),
        ];
        for i_table in 0..3 {
            for i_sensor in 0..sensor.n_sensor {
                for i_conductor in 0..conductor.n_conductor {
                    assert_relative_eq!(
                        tables_point[i_table][(i_sensor, i_conductor)],
                        tables_finite_size[i_table][(i_sensor, i_conductor)],
                        max_relative = 1e-3
                    );
                }
            }
        }
    }

    #[test]
    fn the_fields_are_the_derivatives_of_the_flux_over_two_pi_r() {
        let conductor: Conductor = three_conductors();
        let sensor: Sensor = Sensor::new(&array![0.9, 0.3], &array![0.25, -0.1]).unwrap();
        for conductor_model in [ConductorModel::Point, ConductorModel::FiniteSize(settings_greens_finite_size())] {
            let greens_calculator: Greens = Greens::sensor_to_conductor(&sensor, &conductor, conductor_model).unwrap();
            let d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r().unwrap();
            let d_psi_d_z: Array2<f64> = greens_calculator.d_psi_d_z().unwrap();
            let d2_psi_d_r_d_z: Array2<f64> = greens_calculator.d2_psi_d_r_d_z().unwrap();
            let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2().unwrap();
            let b_r: Array2<f64> = greens_calculator.b_r().unwrap();
            let b_z: Array2<f64> = greens_calculator.b_z().unwrap();
            let d_b_r_d_z: Array2<f64> = greens_calculator.d_b_r_d_z().unwrap();
            let d_b_z_d_z: Array2<f64> = greens_calculator.d_b_z_d_z().unwrap();
            for i_sensor in 0..sensor.n_sensor {
                let two_pi_r: f64 = 2.0 * PI * sensor.r[i_sensor];
                for i_conductor in 0..conductor.n_conductor {
                    assert_abs_diff_eq!(b_r[(i_sensor, i_conductor)], -d_psi_d_z[(i_sensor, i_conductor)] / two_pi_r, epsilon = 1e-20);
                    assert_abs_diff_eq!(b_z[(i_sensor, i_conductor)], d_psi_d_r[(i_sensor, i_conductor)] / two_pi_r, epsilon = 1e-20);
                    assert_abs_diff_eq!(
                        d_b_r_d_z[(i_sensor, i_conductor)],
                        -d2_psi_d_z2[(i_sensor, i_conductor)] / two_pi_r,
                        epsilon = 1e-20
                    );
                    assert_abs_diff_eq!(
                        d_b_z_d_z[(i_sensor, i_conductor)],
                        d2_psi_d_r_d_z[(i_sensor, i_conductor)] / two_pi_r,
                        epsilon = 1e-20
                    );
                }
            }
        }
    }

    #[test]
    fn a_sensor_on_an_edge_which_two_cells_share_is_moved_into_exactly_one_of_them() {
        // Two cells of ST40's grid, one above the other, as `GreensSources::new` builds them: from their corners, their centres minus half a
        // cell. The first sensor is typed onto the line between them, the second is inside the lower cell
        let d_r: f64 = 0.0125;
        let d_z: f64 = 0.0125;
        let conductor: Conductor = Conductor::from_oblique(
            &(array![0.41, 0.41] - 0.5 * d_r),
            &(array![0.0, 0.0125] - 0.5 * d_z),
            &array![d_r, d_r],
            &array![d_z, d_z],
            &array![0.0, 0.0],
            &array![0.0, 0.0],
        )
        .unwrap();
        let sensor: Sensor = Sensor::new(&array![0.41, 0.41], &array![0.00625, 0.003]).unwrap();

        let greens_calculator: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::FiniteSize(settings_greens_finite_size())).unwrap();
        assert_eq!(greens_calculator.is_sensor_shifted, array![true, false]);
        let greens_calculator_point: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::Point).unwrap();
        assert_eq!(greens_calculator_point.is_sensor_shifted, array![false, false]);

        // Exactly one of the two cells counts the sensor as inside it: with 1 ampere in each cell, Delta* psi is that of 1 ampere over one
        // cell, where both counting it would double it, and neither would make it zero
        let d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r().unwrap();
        let d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2().unwrap();
        let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2().unwrap();
        let sensor_shifted_r: f64 = sensor.r[0] + SENSOR_SHIFT;
        let mut delta_star: f64 = 0.0;
        for i_conductor in 0..conductor.n_conductor {
            delta_star += d2_psi_d_r2[(0, i_conductor)] - d_psi_d_r[(0, i_conductor)] / sensor_shifted_r + d2_psi_d_z2[(0, i_conductor)];
        }
        let delta_star_one_cell: f64 = -2.0 * PI * physical_constants::VACUUM_MAG_PERMEABILITY * sensor_shifted_r / (d_r * d_z);
        assert_relative_eq!(delta_star, delta_star_one_cell, max_relative = 1e-9);

        // The flux hardly changes: compared with the flux alone at the sensor where it was typed, which needs no shift
        let g_psi: Array2<f64> = greens_calculator.psi().unwrap();
        let g_psi_typed: Array2<f64> = mutual_inductance_finite_size_to_point_source(
            &sensor.r,
            &sensor.z,
            &conductor.r,
            &conductor.z,
            &conductor.length_alpha,
            &conductor.length_beta,
            &conductor.alpha,
            &conductor.beta,
            &settings_greens_finite_size(),
        )
        .unwrap();
        for i_conductor in 0..conductor.n_conductor {
            assert_relative_eq!(g_psi[(0, i_conductor)], g_psi_typed[(0, i_conductor)], max_relative = 1e-7);
            assert_eq!(g_psi[(1, i_conductor)], g_psi_typed[(1, i_conductor)]);
        }
    }

    #[test]
    fn a_sensor_at_a_corner_which_four_cells_share_is_moved_into_exactly_one_of_them() {
        // Four cells of ST40's grid around one of its nodes, and a sensor typed onto the node
        let d_r: f64 = 0.0125;
        let d_z: f64 = 0.0125;
        let conductor: Conductor = Conductor::from_oblique(
            &(array![0.41, 0.4225, 0.41, 0.4225] - 0.5 * d_r),
            &(array![0.0, 0.0, 0.0125, 0.0125] - 0.5 * d_z),
            &array![d_r, d_r, d_r, d_r],
            &array![d_z, d_z, d_z, d_z],
            &array![0.0, 0.0, 0.0, 0.0],
            &array![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let sensor: Sensor = Sensor::new(&array![0.41625], &array![0.00625]).unwrap();

        let greens_calculator: Greens = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::FiniteSize(settings_greens_finite_size())).unwrap();
        assert_eq!(greens_calculator.is_sensor_shifted, array![true]);

        // Every derivative is finite, and with 1 ampere in each cell Delta* psi is that of 1 ampere over one cell
        let d_psi_d_r: Array2<f64> = greens_calculator.d_psi_d_r().unwrap();
        let d2_psi_d_r2: Array2<f64> = greens_calculator.d2_psi_d_r2().unwrap();
        let d2_psi_d_r_d_z: Array2<f64> = greens_calculator.d2_psi_d_r_d_z().unwrap();
        let d2_psi_d_z2: Array2<f64> = greens_calculator.d2_psi_d_z2().unwrap();
        let d3_psi_d_z3: Array2<f64> = greens_calculator.d3_psi_d_z3().unwrap();
        let sensor_shifted_r: f64 = sensor.r[0] + SENSOR_SHIFT;
        let mut delta_star: f64 = 0.0;
        for i_conductor in 0..conductor.n_conductor {
            assert!(d2_psi_d_r_d_z[(0, i_conductor)].is_finite() && d3_psi_d_z3[(0, i_conductor)].is_finite());
            delta_star += d2_psi_d_r2[(0, i_conductor)] - d_psi_d_r[(0, i_conductor)] / sensor_shifted_r + d2_psi_d_z2[(0, i_conductor)];
        }
        let delta_star_one_cell: f64 = -2.0 * PI * physical_constants::VACUUM_MAG_PERMEABILITY * sensor_shifted_r / (d_r * d_z);
        assert_relative_eq!(delta_star, delta_star_one_cell, max_relative = 1e-9);
    }

    #[test]
    fn a_sensor_on_an_edge_parallel_to_the_shift_is_an_error() {
        // A parallelogram whose first side is inclined at 45 degrees, along the shift, and a sensor half way along that side, which the shift
        // moves along the edge rather than off it
        let length_alpha: f64 = 0.01;
        let alpha: f64 = PI / 4.0;
        let conductor: Conductor =
            Conductor::from_oblique(&array![0.5], &array![0.0], &array![length_alpha], &array![0.01], &array![alpha], &array![0.0]).unwrap();
        let sensor: Sensor = Sensor::new(&array![0.5 + 0.5 * length_alpha * alpha.cos()], &array![0.5 * length_alpha * alpha.sin()]).unwrap();

        let error: String = Greens::sensor_to_conductor(&sensor, &conductor, ConductorModel::FiniteSize(settings_greens_finite_size()))
            .err()
            .expect("the sensor is still on the edge after the shift");
        assert!(error.contains("safety shift"), "{error}");
    }
}
