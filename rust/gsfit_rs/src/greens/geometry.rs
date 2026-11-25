//! What the Green's functions are calculated between: points where the flux is evaluated, `Sensor`, and toroidal conductors with a
//! finite cross-section, `Conductor`.
//!
//! Both are checked once, when they are built, so `Greens` can rely on them.

use ndarray::Array1;

/// The two sides of a parallelogram are taken as parallel, so that it has no area, when the sine of the angle between them is below this.
const PARALLEL_SIDES_TOLERANCE: f64 = 1.0e-12;

/// Points where the flux and its derivatives are evaluated, e.g. magnetic sensors, or the points of GSFit's grid.
///
/// Each point is a toroidal filament of no size, at `(r, z)`.
#[derive(Clone, Debug)]
pub struct Sensor {
    /// R coordinate of each point, shape = [n_sensor], [metre]
    pub r: Array1<f64>,
    /// Z coordinate of each point, shape = [n_sensor], [metre]
    pub z: Array1<f64>,
    /// The number of points, [count]
    pub n_sensor: usize,
}

impl Sensor {
    /// Points where the flux and its derivatives are evaluated.
    ///
    /// # Arguments
    /// * `r` - R coordinate of each point, shape = [n_sensor], [metre]
    /// * `z` - Z coordinate of each point, shape = [n_sensor], [metre]
    ///
    /// # Returns
    /// * `sensor` - The points, with copies of `r` and `z`
    ///
    /// # Errors
    /// * `r` and `z` do not have the same length
    /// * a point is not at a positive major radius, or is not finite
    pub fn new(r: &Array1<f64>, z: &Array1<f64>) -> Result<Self, String> {
        let n_sensor: usize = r.len();
        if z.len() != n_sensor {
            return Err(format!("`r` and `z` have {} and {} point(s); each point needs one of each", n_sensor, z.len()));
        }

        // The Green's function of a filament is written for a point at a positive major radius
        for i_sensor in 0..n_sensor {
            let is_usable_point: bool = r[i_sensor] > 0.0 && r[i_sensor].is_finite() && z[i_sensor].is_finite();
            if !is_usable_point {
                return Err(format!(
                    "point {i_sensor} is at r = {} and z = {} metre; r must be positive, and both finite",
                    r[i_sensor], z[i_sensor]
                ));
            }
        }

        Ok(Sensor {
            r: r.clone(),
            z: z.clone(),
            n_sensor,
        })
    }
}

/// Toroidal conductors, each with a parallelogram cross-section which carries a uniform current density.
///
/// Each conductor is described as the IMAS data dictionary describes `geometry/oblique`, e.g. `pf_passive/loop(i1)/element(i2)/geometry/oblique`:
/// its reference corner `(r, z)`, the lengths of the two sides which meet there, and their inclinations. The two sides are:
///
/// ```text
/// side_alpha = length_alpha * ( cos(alpha), sin(alpha))
/// side_beta  = length_beta  * (-sin(beta),  cos(beta))
/// ```
///
/// A rectangle, e.g. `pf_active/coil(i1)/element(i2)/geometry/rectangle`, is an oblique with `alpha = beta = 0`, whose reference corner is
/// its centre minus half of its width and height: IMAS describes a rectangle from its centre, but an oblique from a corner.
///
/// The fields are public so they can be read. `centroid_r`, `centroid_z` and `area` are worked out from the others by `from_oblique`, so
/// they are not updated if the others are changed afterwards.
#[derive(Clone, Debug)]
pub struct Conductor {
    /// R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
    pub r: Array1<f64>,
    /// Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
    pub z: Array1<f64>,
    /// Length of each conductor's side which is inclined at `alpha`, shape = [n_conductor], [metre]
    pub length_alpha: Array1<f64>,
    /// Length of each conductor's side which is inclined at `beta`, shape = [n_conductor], [metre]
    pub length_beta: Array1<f64>,
    /// Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
    pub alpha: Array1<f64>,
    /// Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
    pub beta: Array1<f64>,
    /// R coordinate of each conductor's centroid, half of each side on from the reference corner, shape = [n_conductor], [metre]
    pub centroid_r: Array1<f64>,
    /// Z coordinate of each conductor's centroid, shape = [n_conductor], [metre]
    pub centroid_z: Array1<f64>,
    /// The area of each conductor's cross-section, shape = [n_conductor], [metre ** 2]
    pub area: Array1<f64>,
    /// The number of conductors, [count]
    pub n_conductor: usize,
}

impl Conductor {
    /// Toroidal conductors, from their IMAS `geometry/oblique` description.
    ///
    /// # Arguments
    /// * `r` - R coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
    /// * `z` - Z coordinate of each conductor's reference corner, shape = [n_conductor], [metre]
    /// * `length_alpha` - Length of each conductor's side which is inclined at `alpha`, shape = [n_conductor], [metre]
    /// * `length_beta` - Length of each conductor's side which is inclined at `beta`, shape = [n_conductor], [metre]
    /// * `alpha` - Inclination of each conductor's first side, counter-clockwise from the R axis, shape = [n_conductor], [radian]
    /// * `beta` - Inclination of each conductor's second side, counter-clockwise from the Z axis, shape = [n_conductor], [radian]
    ///
    /// # Returns
    /// * `conductor` - The conductors, with copies of the six arrays, and each conductor's centroid and area
    ///
    /// # Errors
    /// * the six arrays do not have the same length
    /// * a conductor has no area: a side without a positive, finite length, or two sides which are parallel
    /// * a conductor is not wholly at a major radius of zero or more; a conductor which touches the axis, as a cell of GSFit's grid may, is
    ///   allowed here, but `ConductorModel::FiniteSize` needs every conductor at a positive major radius
    pub fn from_oblique(
        r: &Array1<f64>,
        z: &Array1<f64>,
        length_alpha: &Array1<f64>,
        length_beta: &Array1<f64>,
        alpha: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Result<Self, String> {
        let n_conductor: usize = r.len();
        if z.len() != n_conductor
            || length_alpha.len() != n_conductor
            || length_beta.len() != n_conductor
            || alpha.len() != n_conductor
            || beta.len() != n_conductor
        {
            return Err(format!(
                "`r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta` have {}, {}, {}, {}, {} and {} conductor(s); each conductor needs \
                 one of each",
                n_conductor,
                z.len(),
                length_alpha.len(),
                length_beta.len(),
                alpha.len(),
                beta.len()
            ));
        }

        let mut centroid_r: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut centroid_z: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        let mut area: Array1<f64> = Array1::from_elem(n_conductor, f64::NAN);
        for i_conductor in 0..n_conductor {
            // The directions of the two sides which meet at the reference corner
            let side_alpha_unit_r: f64 = alpha[i_conductor].cos();
            let side_alpha_unit_z: f64 = alpha[i_conductor].sin();
            let side_beta_unit_r: f64 = -beta[i_conductor].sin();
            let side_beta_unit_z: f64 = beta[i_conductor].cos();

            // A parallelogram with an area has two sides of positive length which are not parallel
            let sine_between_sides: f64 = (side_alpha_unit_r * side_beta_unit_z - side_alpha_unit_z * side_beta_unit_r).abs();
            let has_area: bool = length_alpha[i_conductor] > 0.0
                && length_alpha[i_conductor].is_finite()
                && length_beta[i_conductor] > 0.0
                && length_beta[i_conductor].is_finite()
                && sine_between_sides >= PARALLEL_SIDES_TOLERANCE;
            if !has_area {
                return Err(format!(
                    "conductor {i_conductor} has no area: `length_alpha` = {} metre, `length_beta` = {} metre, `alpha` = {} radian and `beta` = {} \
                     radian; the lengths must be positive, and the two sides not parallel",
                    length_alpha[i_conductor], length_beta[i_conductor], alpha[i_conductor], beta[i_conductor]
                ));
            }

            // The innermost of the four corners
            let corner_r_min: f64 =
                r[i_conductor] + (length_alpha[i_conductor] * side_alpha_unit_r).min(0.0) + (length_beta[i_conductor] * side_beta_unit_r).min(0.0);
            let is_usable_position: bool = corner_r_min >= 0.0 && corner_r_min.is_finite() && z[i_conductor].is_finite();
            if !is_usable_position {
                return Err(format!(
                    "conductor {i_conductor} has its reference corner at `r` = {} and `z` = {} metre, and its innermost corner at r = {corner_r_min} \
                     metre; the whole conductor must be at a major radius of zero or more",
                    r[i_conductor], z[i_conductor]
                ));
            }

            // The two sides, and the centroid half of each side on from the reference corner. Written as `passive_filaments` writes it, so
            // the filaments are where they were
            let side_alpha_r: f64 = length_alpha[i_conductor] * side_alpha_unit_r;
            let side_alpha_z: f64 = length_alpha[i_conductor] * side_alpha_unit_z;
            let side_beta_r: f64 = length_beta[i_conductor] * side_beta_unit_r;
            let side_beta_z: f64 = length_beta[i_conductor] * side_beta_unit_z;
            centroid_r[i_conductor] = r[i_conductor] + 0.5 * (side_alpha_r + side_beta_r);
            centroid_z[i_conductor] = z[i_conductor] + 0.5 * (side_alpha_z + side_beta_z);
            area[i_conductor] = (side_alpha_r * side_beta_z - side_alpha_z * side_beta_r).abs();
        }

        Ok(Conductor {
            r: r.clone(),
            z: z.clone(),
            length_alpha: length_alpha.clone(),
            length_beta: length_beta.clone(),
            alpha: alpha.clone(),
            beta: beta.clone(),
            centroid_r,
            centroid_z,
            area,
            n_conductor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn a_rectangle_has_its_centroid_at_its_centre() {
        let conductor: Conductor = Conductor::from_oblique(&array![1.0], &array![-0.5], &array![0.2], &array![0.4], &array![0.0], &array![0.0]).unwrap();

        assert_abs_diff_eq!(conductor.centroid_r[0], 1.1, epsilon = 1e-15);
        assert_abs_diff_eq!(conductor.centroid_z[0], -0.3, epsilon = 1e-15);
        assert_abs_diff_eq!(conductor.area[0], 0.08, epsilon = 1e-16);
        assert_eq!(conductor.n_conductor, 1);
    }

    #[test]
    fn a_sheared_parallelogram_has_the_area_of_its_two_sides() {
        // The first side along R, the second inclined by 30 degrees from the Z axis: the area is base times height
        let length_alpha: f64 = 0.3;
        let length_beta: f64 = 0.5;
        let beta: f64 = PI / 6.0;
        let conductor: Conductor = Conductor::from_oblique(
            &array![2.0],
            &array![0.0],
            &array![length_alpha],
            &array![length_beta],
            &array![0.0],
            &array![beta],
        )
        .unwrap();

        assert_abs_diff_eq!(conductor.area[0], length_alpha * length_beta * beta.cos(), epsilon = 1e-15);
        assert_abs_diff_eq!(conductor.centroid_r[0], 2.0 + 0.5 * (length_alpha - length_beta * beta.sin()), epsilon = 1e-15);
        assert_abs_diff_eq!(conductor.centroid_z[0], 0.5 * length_beta * beta.cos(), epsilon = 1e-15);
    }

    #[test]
    fn arrays_of_different_lengths_are_an_error() {
        let result: Result<Conductor, String> = Conductor::from_oblique(
            &array![1.0, 2.0],
            &array![0.0],
            &array![0.1, 0.1],
            &array![0.1, 0.1],
            &array![0.0, 0.0],
            &array![0.0, 0.0],
        );
        assert!(result.is_err());

        let result_sensor: Result<Sensor, String> = Sensor::new(&array![1.0, 2.0], &array![0.0]);
        assert!(result_sensor.is_err());
    }

    #[test]
    fn a_conductor_with_no_area_is_an_error() {
        // A side of zero length
        let result: Result<Conductor, String> = Conductor::from_oblique(&array![1.0], &array![0.0], &array![0.0], &array![0.1], &array![0.0], &array![0.0]);
        assert!(result.is_err());

        // Two parallel sides: the second side is along R, like the first
        let result: Result<Conductor, String> =
            Conductor::from_oblique(&array![1.0], &array![0.0], &array![0.1], &array![0.1], &array![0.0], &array![-PI / 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn a_conductor_which_crosses_the_axis_is_an_error() {
        let result: Result<Conductor, String> =
            Conductor::from_oblique(&array![0.05], &array![0.0], &array![0.1], &array![0.1], &array![0.0], &array![PI / 4.0]);
        assert!(result.is_err());

        // A grid cell which touches the axis is allowed
        let result: Result<Conductor, String> = Conductor::from_oblique(&array![0.0], &array![0.0], &array![0.1], &array![0.1], &array![0.0], &array![0.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn a_point_which_is_not_at_a_positive_major_radius_is_an_error() {
        assert!(Sensor::new(&array![0.0], &array![0.0]).is_err());
        assert!(Sensor::new(&array![1.0], &array![f64::NAN]).is_err());
        assert!(Sensor::new(&array![1.0], &array![0.0]).is_ok());
    }
}
