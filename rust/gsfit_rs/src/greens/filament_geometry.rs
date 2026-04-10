use approx::abs_diff_eq;

// TODO: IMAS has a different parallelogram definition, using angles `alpha` and `beta` instead of `angle_1` and `angle_2`!!
pub struct FilamentGeometry {
    pub r_1: f64,
    pub z_1: f64,
    pub r_2: f64,
    pub z_2: f64,
    pub r_3: f64,
    pub z_3: f64,
    pub r_4: f64,
    pub z_4: f64,
}

impl FilamentGeometry {
    /// New instance of `FilamentGeometry` for a filament
    ///
    /// # Arguments
    /// * `angle_1` -
    /// * `angle_2` -
    /// * `d_r` - The radial width of the filament
    /// * `d_z` - The vertical width of the filament
    /// * `r` - The radial coordinate of the filament center
    /// * `z` - The vertical coordinate of the filament center
    ///
    /// # Returns
    /// * `filament_geometry` - The `FilamentGeometry` instance containing the coordinates of the four vertices of the parallelogram
    ///
    pub fn new(angle_1: f64, angle_2: f64, d_r: f64, d_z: f64, r: f64, z: f64) -> Self {
        // First coordinate in the parallelogram (bottom; left)
        let r_1: f64;
        let z_1: f64;
        if abs_diff_eq!(angle_1, 0.0) {
            r_1 = r - d_r / 2.0;
        } else {
            r_1 = r - d_r / 2.0 - d_z / (2.0 * angle_1.tan());
        }
        if abs_diff_eq!(angle_2, 0.0) {
            z_1 = z - d_z / 2.0;
        } else {
            z_1 = z - d_z / 2.0 - d_r * angle_2.tan() / 2.0;
        }

        // Second coordinate in the parallelogram (bottom; right)
        let r_2: f64;
        let z_2: f64;
        if abs_diff_eq!(angle_1, 0.0) {
            r_2 = r + d_r / 2.0;
        } else {
            r_2 = r + d_r / 2.0 - d_z / (2.0 * angle_1.tan());
        }
        if abs_diff_eq!(angle_2, 0.0) {
            z_2 = z - d_z / 2.0;
        } else {
            z_2 = z - d_z / 2.0 + d_r * angle_2.tan() / 2.0;
        }

        // Third coordinate in the parallelogram (top; right)
        let r_3: f64;
        let z_3: f64;
        if abs_diff_eq!(angle_1, 0.0) {
            r_3 = r + d_r / 2.0;
        } else {
            r_3 = r + d_r / 2.0 + d_z / (2.0 * angle_1.tan());
        }
        if abs_diff_eq!(angle_2, 0.0) {
            z_3 = z + d_z / 2.0;
        } else {
            z_3 = z + d_z / 2.0 + d_r * angle_2.tan() / 2.0;
        }

        // Fourth coordinate in the parallelogram (top; left)
        let r_4: f64;
        let z_4: f64;
        if abs_diff_eq!(angle_1, 0.0) {
            r_4 = r - d_r / 2.0;
        } else {
            r_4 = r - d_r / 2.0 + d_z / (2.0 * angle_1.tan());
        }
        if abs_diff_eq!(angle_2, 0.0) {
            z_4 = z + d_z / 2.0;
        } else {
            z_4 = z + d_z / 2.0 - d_r * angle_2.tan() / 2.0;
        }

        Self {
            r_1,
            z_1,
            r_2,
            z_2,
            r_3,
            z_3,
            r_4,
            z_4,
        }
    }

    /// Shoelace formula for polygon area
    ///
    /// Note, Shoelace formula gives:
    /// * Positive for counter-clockwise vertex ordering
    /// * Negative for clockwise vertex ordering
    /// to avoid this we have added `.abs()`
    ///
    /// # Arguments
    /// * `self` - The `FilamentGeometry` instance
    ///
    /// # Returns
    /// * `area` - The area of the parallelogram
    pub fn calculate_area(&self) -> f64 {
        let area: f64 = (self.r_1 * self.z_2 - self.r_2 * self.z_1 + self.r_2 * self.z_3 - self.r_3 * self.z_2 + self.r_3 * self.z_4 - self.r_4 * self.z_3
            + self.r_4 * self.z_1
            - self.r_1 * self.z_4)
            .abs()
            / 2.0;

        area
    }
}

#[test]
fn test_filament_geometry() {
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    // Random test data
    let d_r: f64 = 1.2345e-3;
    let d_z: f64 = 2.3456e-3;
    let r: f64 = 0.34567;
    let z: f64 = 0.45678;

    // bottom; left
    let expected_r_1: f64 = r - d_r / 2.0;
    let expected_z_1: f64 = z - d_z / 2.0;
    // bottom; right
    let expected_r_2: f64 = r + d_r / 2.0;
    let expected_z_2: f64 = z - d_z / 2.0;
    // top; right
    let expected_r_3: f64 = r + d_r / 2.0;
    let expected_z_3: f64 = z + d_z / 2.0;
    // top; left
    let expected_r_4: f64 = r - d_r / 2.0;
    let expected_z_4: f64 = z + d_z / 2.0;

    // 1. Test rectangle filament (i.e. zero angles)
    let angle_1: f64 = 0.0;
    let angle_2: f64 = 0.0;

    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);
    let area: f64 = filament_geometry.calculate_area();
    assert_abs_diff_eq!(area, d_r * d_z, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.r_1, expected_r_1, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.z_1, expected_z_1, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.r_2, expected_r_2, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.z_2, expected_z_2, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.r_3, expected_r_3, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.z_3, expected_z_3, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.r_4, expected_r_4, epsilon = 1e-12);
    assert_abs_diff_eq!(filament_geometry.z_4, expected_z_4, epsilon = 1e-12);

    // 2. Test with non-zero `angle_1`
    let angle_1: f64 = PI / 2.0 - 1.0e-8; // A very small angle, should be close to a rectangle
    let angle_2: f64 = 0.0;

    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);
    let area: f64 = filament_geometry.calculate_area();
    assert_abs_diff_eq!(area, d_r * d_z, epsilon = 1e-8); // lower precision, as the area is not exactly d_r * d_z
    assert_abs_diff_eq!(filament_geometry.r_1, expected_r_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_1, expected_z_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_2, expected_r_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_2, expected_z_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_3, expected_r_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_3, expected_z_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_4, expected_r_4, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_4, expected_z_4, epsilon = 1e-8);

    // 3. Test with non-zero `angle_2`
    let angle_1: f64 = 0.0;
    let angle_2: f64 = PI - 1.0e-8; // A very small angle, should be close to a rectangle

    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);
    let area: f64 = filament_geometry.calculate_area();
    assert_abs_diff_eq!(area, d_r * d_z, epsilon = 1e-8); // lower precision, as the area is not exactly d_r * d_z
    assert_abs_diff_eq!(filament_geometry.r_1, expected_r_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_1, expected_z_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_2, expected_r_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_2, expected_z_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_3, expected_r_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_3, expected_z_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_4, expected_r_4, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_4, expected_z_4, epsilon = 1e-8);

    // 4. Test with non-zero `angle_1` and `angle_2`
    let angle_1: f64 = PI / 2.0 - 1.0e-8;
    let angle_2: f64 = PI - 1.0e-8;

    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);
    let area: f64 = filament_geometry.calculate_area();
    assert_abs_diff_eq!(area, d_r * d_z, epsilon = 1e-8); // lower precision, as the area is not exactly d_r * d_z
    assert_abs_diff_eq!(filament_geometry.r_1, expected_r_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_1, expected_z_1, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_2, expected_r_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_2, expected_z_2, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_3, expected_r_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_3, expected_z_3, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.r_4, expected_r_4, epsilon = 1e-8);
    assert_abs_diff_eq!(filament_geometry.z_4, expected_z_4, epsilon = 1e-8);

    // 5. Move the filament to a different location, the area should be the same
    let r: f64 = 8.56789;
    let z: f64 = -2.67890;
    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);
    let area_different_location: f64 = filament_geometry.calculate_area();
    assert_abs_diff_eq!(area_different_location, area, epsilon = 1e-8);
}
