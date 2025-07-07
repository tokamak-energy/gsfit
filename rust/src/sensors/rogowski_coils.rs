use crate::Plasma;
use crate::coils::Coils;
use crate::nested_dict::NestedDict;
use crate::nested_dict::NestedDictAccumulator;
use crate::passives::Passives;
use crate::sensors::static_and_dynamic_data_types::{SensorsDynamic, SensorsStatic};
use geo::{Contains, Coord, LineString, Point, Polygon};
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_interp::interp1d::Interp1D;
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;
const PI: f64 = std::f64::consts::PI;

use super::BpProbes;

#[derive(Clone)]
#[pyclass]
pub struct RogowskiCoils {
    pub results: NestedDict,
}

/// Python accessible methods
#[pymethods]
impl RogowskiCoils {
    #[new]
    pub fn new() -> Self {
        Self { results: NestedDict::new() }
    }

    /// Data structure:
    ///
    /// # Examples
    ///
    /// ```
    /// [probe_name]["psi"]["calculated"]                               = Array1<f64>;  shape=[n_time]
    /// [probe_name]["psi"]["measured"]                                 = Array1<f64>;  shape=[n_time]
    /// [probe_name]["fit_settings"]["comment"]                         = str
    /// [probe_name]["fit_settings"]["expected_value"]                  = f64
    /// [probe_name]["fit_settings"]["include"]                         = bool
    /// [probe_name]["fit_settings"]["weight"]                          = f64
    /// [probe_name]["geometry"]["r"]                                   = f64
    /// [probe_name]["geometry"]["z"]                                   = f64
    /// [probe_name]["greens"]["d_plasma_d_z"]                          = Array1;  shape=[n_z * n_r]
    /// [probe_name]["greens"]["passives"][passive_name][dof_name]      = f64
    /// [probe_name]["greens"]["pf"][pf_name]                           = f64
    /// [probe_name]["greens"]["plasma"]                                = Array1;  shape=[n_z * n_r]
    /// ```
    ///
    pub fn add_sensor(
        &mut self,
        name: &str,
        r: &Bound<'_, PyArray1<f64>>,
        z: &Bound<'_, PyArray1<f64>>,
        fit_settings_comment: String,
        fit_settings_expected_value: f64,
        fit_settings_include: bool,
        fit_settings_weight: f64,
        time: &Bound<'_, PyArray1<f64>>,
        measured: &Bound<'_, PyArray1<f64>>,
        gaps_r: &Bound<'_, PyArray1<f64>>,
        gaps_z: &Bound<'_, PyArray1<f64>>,
        gaps_d_r: &Bound<'_, PyArray1<f64>>,
        gaps_d_z: &Bound<'_, PyArray1<f64>>,
        gaps_name: &Bound<'_, PyList>,
    ) {
        // Convert to Rust data types
        let r_ndarray: Array1<f64> = Array1::from(unsafe { r.as_array() }.to_vec());
        let z_ndarray: Array1<f64> = Array1::from(unsafe { z.as_array() }.to_vec());
        let time_ndarray: Array1<f64> = Array1::from(unsafe { time.as_array() }.to_vec());
        let measured_ndarray: Array1<f64> = Array1::from(unsafe { measured.as_array() }.to_vec());
        let gaps_r_ndarray: Array1<f64> = Array1::from(unsafe { gaps_r.as_array() }.to_vec());
        let gaps_z_ndarray: Array1<f64> = Array1::from(unsafe { gaps_z.as_array() }.to_vec());
        let gaps_d_r_ndarray: Array1<f64> = Array1::from(unsafe { gaps_d_r.as_array() }.to_vec());
        let gaps_d_z_ndarray: Array1<f64> = Array1::from(unsafe { gaps_d_z.as_array() }.to_vec());
        let n_gaps: usize = gaps_r_ndarray.len();

        // Add gaps
        for i_gap in 0..n_gaps {
            let gap_r_start: f64 = gaps_r_ndarray[i_gap] - gaps_d_r_ndarray[i_gap];
            let gap_r_end: f64 = gaps_r_ndarray[i_gap] + gaps_d_r_ndarray[i_gap];
            let gap_z_start: f64 = gaps_z_ndarray[i_gap] - gaps_d_z_ndarray[i_gap];
            let gap_z_end: f64 = gaps_z_ndarray[i_gap] + gaps_d_z_ndarray[i_gap];
            let gap_name: String = gaps_name.get_item(i_gap).unwrap().extract().unwrap();

            self.results
                .get_or_insert(name)
                .get_or_insert("geometry")
                .get_or_insert("gaps")
                .get_or_insert(&gap_name)
                .insert("r_start", gap_r_start);
            self.results
                .get_or_insert(name)
                .get_or_insert("geometry")
                .get_or_insert("gaps")
                .get_or_insert(&gap_name)
                .insert("r_end", gap_r_end);
            self.results
                .get_or_insert(name)
                .get_or_insert("geometry")
                .get_or_insert("gaps")
                .get_or_insert(&gap_name)
                .insert("z_start", gap_z_start);
            self.results
                .get_or_insert(name)
                .get_or_insert("geometry")
                .get_or_insert("gaps")
                .get_or_insert(&gap_name)
                .insert("z_end", gap_z_end);
        }

        // Geometry
        self.results.get_or_insert(name).get_or_insert("geometry").insert("r", r_ndarray);
        self.results.get_or_insert(name).get_or_insert("geometry").insert("z", z_ndarray);

        // // Gaps
        // self.results.get_or_insert(name).get_or_insert("geometry").get_or_insert("gaps").insert("r", gap_r_ndarray);
        // self.results.get_or_insert(name).get_or_insert("geometry").get_or_insert("gaps").insert("z", gap_z_ndarray);
        // self.results.get_or_insert(name).get_or_insert("geometry").get_or_insert("gaps").insert("d_r", gap_d_r_ndarray);
        // self.results.get_or_insert(name).get_or_insert("geometry").get_or_insert("gaps").insert("d_z", gap_d_z_ndarray);

        // Fit settings
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("comment", fit_settings_comment);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("expected_value", fit_settings_expected_value);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("include", fit_settings_include);
        self.results
            .get_or_insert(name)
            .get_or_insert("fit_settings")
            .insert("weight", fit_settings_weight);

        // Measurements
        self.results.get_or_insert(name).get_or_insert("i").insert("time_experimental", time_ndarray);
        self.results
            .get_or_insert(name)
            .get_or_insert("i")
            .insert("measured_experimental", measured_ndarray);
    }

    ///
    fn greens_with_coils(&mut self, coils: PyRef<Coils>) {
        // Change Python type into Rust
        let coils_local: &Coils = &*coils;

        for sensor_name in self.results.keys() {
            // Rogowski path
            let sensor_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").unwrap_array1();
            let sensor_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").unwrap_array1();
            let n_sensor_points: usize = sensor_r.len();

            // Gap names
            let gap_names: Vec<String> = self.results.get(&sensor_name).get("geometry").get("gaps").keys();
            let n_gaps: usize = gap_names.len();

            // Collect the coordinates
            let mut rogowski_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_sensor_points);
            for i_sensor_point in 0..n_sensor_points {
                rogowski_coordinates.push(Coord {
                    x: sensor_r[i_sensor_point],
                    y: sensor_z[i_sensor_point],
                });
            }

            // Construct the contour
            let rogowski_contour: Polygon = Polygon::new(
                LineString::new(rogowski_coordinates),
                vec![], // No holes
            );

            let pf_coil_names: Vec<String> = coils_local.results.get("pf").keys();
            let n_pf: usize = pf_coil_names.len();
            for i_pf in 0..n_pf {
                let pf_coil_name: String = pf_coil_names[i_pf].clone();

                let coil_r: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("r").unwrap_array1();
                let coil_z: Array1<f64> = coils.results.get("pf").get(&pf_coil_name).get("geometry").get("z").unwrap_array1();
                let n_filaments: usize = coil_r.len();

                let mut g_with_pf: f64 = 0.0;
                for i_filament in 0..n_filaments {
                    let test_point: Point = Point::new(coil_r[i_filament], coil_z[i_filament]);
                    let inside: bool = rogowski_contour.contains(&test_point);
                    if inside {
                        g_with_pf += 1.0;
                    }
                }

                // Calculate the greens for the gaps (needed for later)
                // TODO: this is wasteful as we are re-calculating the same thing
                // (at least there are not many PF coils. but still not good...)
                let mut g_gap: f64 = 0.0;
                for i_gap in 0..n_gaps {
                    let gap_name: String = gap_names[i_gap].clone();

                    // Construct virtual bp probes
                    let (mut virtual_b_r_probes, mut virtual_b_z_probes, gap_virtual_d_r, gap_virtual_d_z) =
                        self.clone().construct_virtual_bp_probes(&sensor_name, &gap_name);

                    // Calculate Greens betwen the virtual bp-probes and the coils
                    virtual_b_r_probes.greens_with_coils_rs(coils.clone());
                    virtual_b_z_probes.greens_with_coils_rs(coils.clone());

                    let g_gaps_b_r: Array1<f64> = virtual_b_r_probes.results.get("*").get("greens").get("pf").get(&pf_coil_name).unwrap_array1(); // shape = [n_virtual_bp_probes]
                    let g_gaps_b_z: Array1<f64> = virtual_b_z_probes.results.get("*").get("greens").get("pf").get(&pf_coil_name).unwrap_array1();

                    let n_virtual_bp_probes: usize = g_gaps_b_r.len();

                    g_gap = g_gaps_b_r[0] * 0.5 * gap_virtual_d_r + g_gaps_b_z[0] * 0.5 * gap_virtual_d_z;
                    for i_virtual_bp_probe in 1..n_virtual_bp_probes - 1 {
                        g_gap += g_gaps_b_r[i_virtual_bp_probe] * gap_virtual_d_r + g_gaps_b_z[i_virtual_bp_probe] * gap_virtual_d_z;
                    }
                    g_gap += g_gaps_b_r[n_virtual_bp_probes - 1] * 0.5 * gap_virtual_d_r + g_gaps_b_z[n_virtual_bp_probes - 1] * 0.5 * gap_virtual_d_z;
                }

                // Add the gap contribution
                g_with_pf = g_with_pf - g_gap / MU_0;

                // Store
                self.results
                    .get_or_insert(&sensor_name)
                    .get_or_insert("greens")
                    .get_or_insert("pf")
                    .insert(&pf_coil_name, g_with_pf);
            }
        }
    }

    ///
    fn greens_with_passives(&mut self, passives: PyRef<Passives>) {
        // Change Python type into Rust
        let passives_local: &Passives = &*passives;

        for sensor_name in self.results.keys() {
            // Get variables out of self
            let sensor_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").unwrap_array1();
            let sensor_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").unwrap_array1();
            let n_sensor_points: usize = sensor_r.len();

            // Gap names
            let gap_names: Vec<String> = self.results.get(&sensor_name).get("geometry").get("gaps").keys();

            // Collect the coordinates
            let mut rogowski_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_sensor_points);
            for i_sensor_point in 0..n_sensor_points {
                rogowski_coordinates.push(Coord {
                    x: sensor_r[i_sensor_point],
                    y: sensor_z[i_sensor_point],
                });
            }

            // Construct the contour
            let rogowski_contour: Polygon = Polygon::new(
                LineString::new(rogowski_coordinates),
                vec![], // No holes
            );

            // Calculate Greens with each passive degree of freedom
            for passive_name in passives_local.results.keys() {
                let _tmp: NestedDictAccumulator<'_> = passives_local.results.get(&passive_name).get("dof");
                let dof_names: Vec<String> = _tmp.keys();
                let passive_r: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("r").unwrap_array1();
                let passive_z: Array1<f64> = passives_local.results.get(&passive_name).get("geometry").get("z").unwrap_array1();

                let n_filaments: usize = passive_r.len();

                let mut inside_vec: Array1<f64> = Array1::zeros(n_filaments);
                // Loop over all passive filaments
                for i_filament in 0..n_filaments {
                    let test_point: Point = Point::new(passive_r[i_filament], passive_z[i_filament]);
                    let inside: bool = rogowski_contour.contains(&test_point);
                    if inside {
                        inside_vec[i_filament] = 1.0;
                    }
                }

                for dof_name in dof_names {
                    let current_distribution: Array1<f64> = passives_local
                        .results
                        .get(&passive_name)
                        .get("dof")
                        .get(&dof_name)
                        .get("current_distribution")
                        .unwrap_array1();

                    let g_all: Array1<f64> = &inside_vec * current_distribution;

                    // Calculate the greens for the gaps (needed for later)
                    // TODO: this is wasteful as we are re-calculating the same thing
                    // (at least there are not many PF coils. but still not good...)
                    let mut g_gap: f64 = 0.0;
                    for gap_name in &gap_names {
                        // Construct virtual bp probes
                        let (mut virtual_b_r_probes, mut virtual_b_z_probes, gap_virtual_d_r, gap_virtual_d_z) =
                            self.clone().construct_virtual_bp_probes(&sensor_name, &gap_name);

                        // Calculate Greens betwen the virtual bp-probes and the coils
                        virtual_b_r_probes.greens_with_passives_rs(passives_local.clone());
                        virtual_b_z_probes.greens_with_passives_rs(passives_local.clone());

                        let g_gaps_b_r: Array1<f64> = virtual_b_r_probes
                            .results
                            .get("*")
                            .get("greens")
                            .get("passives")
                            .get(&passive_name)
                            .get(&dof_name)
                            .unwrap_array1(); // shape = [n_virtual_bp_probes]
                        let g_gaps_b_z: Array1<f64> = virtual_b_z_probes
                            .results
                            .get("*")
                            .get("greens")
                            .get("passives")
                            .get(&passive_name)
                            .get(&dof_name)
                            .unwrap_array1(); // shape = [n_virtual_bp_probes]

                        let n_virtual_bp_probes: usize = virtual_b_z_probes.results.keys().len();

                        g_gap = g_gaps_b_r[0] * 0.5 * gap_virtual_d_r + g_gaps_b_z[0] * 0.5 * gap_virtual_d_z;
                        for i_virtual_bp_probe in 1..n_virtual_bp_probes - 1 {
                            g_gap = g_gap + g_gaps_b_r[i_virtual_bp_probe] * gap_virtual_d_r + g_gaps_b_z[i_virtual_bp_probe] * gap_virtual_d_z;
                        }
                        g_gap =
                            g_gap + g_gaps_b_r[n_virtual_bp_probes - 1] * 0.5 * gap_virtual_d_r + g_gaps_b_z[n_virtual_bp_probes - 1] * 0.5 * gap_virtual_d_z;
                    }

                    // Sum over all pasive filaments
                    let g_with_passives: f64 = g_all.sum() - g_gap / MU_0;

                    // Store
                    self.results
                        .get_or_insert(&sensor_name)
                        .get_or_insert("greens")
                        .get_or_insert("passives")
                        .get_or_insert(&passive_name)
                        .insert(&dof_name, g_with_passives);
                }
            }
        }
    }

    fn greens_with_plasma(&mut self, plasma: PyRef<Plasma>) {
        // Change Python type into Rust
        let plasma_local: &Plasma = &*plasma;

        let plasma_r: Array1<f64> = plasma_local.results.get("grid").get("flat").get("r").unwrap_array1();
        let plasma_z: Array1<f64> = plasma_local.results.get("grid").get("flat").get("z").unwrap_array1();
        let n_rz: usize = plasma_r.len();

        for sensor_name in self.results.keys() {
            let sensor_r: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("r").unwrap_array1();
            let sensor_z: Array1<f64> = self.results.get(&sensor_name).get("geometry").get("z").unwrap_array1();
            let n_sensor_points: usize = sensor_r.len();

            // Gap names
            let gap_names: Vec<String> = self.results.get(&sensor_name).get("geometry").get("gaps").keys();

            // Collect the coordinates
            let mut rogowski_coordinates: Vec<Coord<f64>> = Vec::with_capacity(n_sensor_points);
            for i_sensor_point in 0..n_sensor_points {
                rogowski_coordinates.push(Coord {
                    x: sensor_r[i_sensor_point],
                    y: sensor_z[i_sensor_point],
                });
            }

            // Construct the contour
            let rogowski_contour: Polygon = Polygon::new(
                LineString::new(rogowski_coordinates),
                vec![], // No holes
            );

            // Loop over plasma grid to see which grid points are inside the Rogowski
            let mut g_with_plasma: Array1<f64> = Array1::zeros(n_rz);
            for i_rz in 0..n_rz {
                let test_point: Point = Point::new(plasma_r[i_rz], plasma_z[i_rz]);
                let inside: bool = rogowski_contour.contains(&test_point);
                if inside {
                    g_with_plasma[i_rz] = 1.0;
                }
            }

            // Calculate the greens for the gaps (needed for later)
            // TODO: this is wasteful as we are re-calculating the same thing
            // (at least there are not many PF coils. but still not good...)
            let mut g_gap: Array1<f64> = Array1::zeros(n_rz);
            let mut g_gap_d_plasma_d_z: Array1<f64> = Array1::zeros(n_rz);
            for gap_name in gap_names {
                // Construct virtual bp probes
                let (mut virtual_b_r_probes, mut virtual_b_z_probes, gap_virtual_d_r, gap_virtual_d_z) =
                    self.clone().construct_virtual_bp_probes(&sensor_name, &gap_name);

                // Calculate Greens betwen the virtual bp-probes and the coils
                virtual_b_r_probes.greens_with_plasma_rs(plasma.clone());
                virtual_b_z_probes.greens_with_plasma_rs(plasma.clone());

                let g_gaps_b_r: Array2<f64> = virtual_b_r_probes.results.get("*").get("greens").get("plasma").unwrap_array2(); // shape = [n_z * n_r, n_virtual_bp_probes]
                let g_gaps_b_z: Array2<f64> = virtual_b_z_probes.results.get("*").get("greens").get("plasma").unwrap_array2();

                let n_virtual_bp_probes: usize = virtual_b_z_probes.results.keys().len();

                g_gap = g_gaps_b_r.slice(s![.., 0]).to_owned() * 0.5 * gap_virtual_d_r + g_gaps_b_z.slice(s![.., 0]).to_owned() * 0.5 * gap_virtual_d_z;
                for i_virtual_bp_probe in 1..n_virtual_bp_probes - 1 {
                    g_gap = g_gap
                        + g_gaps_b_r.slice(s![.., i_virtual_bp_probe]).to_owned() * gap_virtual_d_r
                        + g_gaps_b_z.slice(s![.., i_virtual_bp_probe]).to_owned() * gap_virtual_d_z;
                }
                g_gap = g_gap
                    + g_gaps_b_r.slice(s![.., n_virtual_bp_probes - 1]).to_owned() * 0.5 * gap_virtual_d_r
                    + g_gaps_b_z.slice(s![.., n_virtual_bp_probes - 1]).to_owned() * 0.5 * gap_virtual_d_z;

                // d(plasma)/d(z) for vertical stability
                let g_gap_d_plasma_d_z_radial: Array2<f64> = virtual_b_r_probes.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2(); // shape = [n_z * n_r, n_virtual_bp_probes]
                let g_gap_d_plasma_d_z_vertical: Array2<f64> = virtual_b_z_probes.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2();

                g_gap_d_plasma_d_z = g_gap_d_plasma_d_z_radial.slice(s![.., 0]).to_owned() * 0.5 * gap_virtual_d_r
                    + g_gap_d_plasma_d_z_vertical.slice(s![.., 0]).to_owned() * 0.5 * gap_virtual_d_z;
                for i_virtual_bp_probe in 1..n_virtual_bp_probes - 1 {
                    g_gap_d_plasma_d_z = g_gap_d_plasma_d_z
                        + g_gap_d_plasma_d_z_radial.slice(s![.., i_virtual_bp_probe]).to_owned() * gap_virtual_d_r
                        + g_gap_d_plasma_d_z_vertical.slice(s![.., i_virtual_bp_probe]).to_owned() * gap_virtual_d_z;
                }
                g_gap_d_plasma_d_z = g_gap_d_plasma_d_z
                    + g_gap_d_plasma_d_z_radial.slice(s![.., n_virtual_bp_probes - 1]).to_owned() * 0.5 * gap_virtual_d_r
                    + g_gap_d_plasma_d_z_vertical.slice(s![.., n_virtual_bp_probes - 1]).to_owned() * 0.5 * gap_virtual_d_z;
            }

            // Add the gap contribution
            g_with_plasma = g_with_plasma - g_gap / MU_0;

            // Store greens
            self.results.get_or_insert(&sensor_name).get_or_insert("greens").insert("plasma", g_with_plasma);

            // Vertical stability
            let g_d_plasma_d_z: Array1<f64> = Array1::zeros(n_rz) - g_gap_d_plasma_d_z / MU_0;

            // Store
            // TODO: should I reshape to Array2 [n_z, n_r] ????
            self.results
                .get_or_insert(&sensor_name)
                .get_or_insert("greens")
                .insert("d_plasma_d_z", g_d_plasma_d_z);
        }
    }

    fn calculate_sensor_values(&mut self, coils: PyRef<Coils>, passives: PyRef<Passives>, plasma: PyRef<Plasma>) {
        // Convert Python types into Rust
        let coils_rs: &Coils = &*coils;
        let passives_rs: &Passives = &*passives;
        let plasma_rs: &Plasma = &*plasma;

        // Run the Rust method
        self.calculate_sensor_values_rust(coils_rs, passives_rs, plasma_rs);
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", " <gsfit_rs.RogowskiCoils>");
        string_output += &format!("║  {:<74} ║\n", version);

        // n_sensors = self.results
        let n_sensors: usize = self.results.keys().len();
        string_output += &format!("║ {:<74} ║\n", format!(" n_sensors = {}", n_sensors.to_string()));

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        return string_output;
    }

    /// Get Array1<f64> and return a numpy.ndarray
    pub fn get_array1(&self, keys: Vec<String>, py: Python) -> Py<PyArray1<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array1: Array1<f64> = result_accumulator.unwrap_array1();
        return array1.into_pyarray(py).into();
    }

    /// Get Array2<f64> and return a numpy.ndarray
    pub fn get_array2(&self, keys: Vec<String>, py: Python) -> Py<PyArray2<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array2: Array2<f64> = result_accumulator.unwrap_array2();
        return array2.into_pyarray(py).into();
    }

    /// Get Array3<f64> and return a numpy.ndarray
    pub fn get_array3(&self, keys: Vec<String>, py: Python) -> Py<PyArray3<f64>> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to NumPy, and return
        let array3: Array3<f64> = result_accumulator.unwrap_array3();
        return array3.into_pyarray(py).into();
    }

    /// Get Vec<bool> and return a Python list[bool]
    pub fn get_bool(&self, keys: Vec<String>) -> bool {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<bool>, and return as a Python list
        let bool_value: bool = result_accumulator.unwrap_bool();

        return bool_value;
    }

    /// Get f64 value and return a f64
    pub fn get_f64(&self, keys: Vec<String>) -> f64 {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap the f64, and return
        return result_accumulator.unwrap_f64();
    }

    /// Get usize value and return a int
    pub fn get_usize(&self, keys: Vec<String>) -> usize {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap the f64, and return
        return result_accumulator.unwrap_usize();
    }

    /// Get Vec<bool> and return a Python list[bool]
    pub fn get_vec_bool(&self, keys: Vec<String>, py: Python) -> Py<PyList> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<bool>, and return as a Python list
        let vec_bool: Vec<bool> = result_accumulator.unwrap_vec_bool();
        let result: Py<PyList> = PyList::new(py, vec_bool).unwrap().into();
        return result;
    }

    /// Get Vec<usize> and return a Python list[int]
    pub fn get_vec_usize(&self, keys: Vec<String>, py: Python) -> Py<PyList> {
        // Start with the root accumulator
        let mut result_accumulator: NestedDictAccumulator<'_> = self.results.get(&keys[0]);

        // Traverse the keys to reach the desired value
        for key in &keys[1..] {
            result_accumulator = result_accumulator.get(key);
        }

        // Unwrap, convert to a Vec<usize>, and return as a Python list of integers
        let vec_usize: Vec<usize> = result_accumulator.unwrap_vec_usize();
        let vec_i64: Vec<i64> = vec_usize.into_iter().map(|x| x as i64).collect();
        let result: Py<PyList> = PyList::new(py, vec_i64).unwrap().into();
        return result;
    }

    /// Get the keys from results and return a Python list of strings
    #[pyo3(signature = (key_path=None))]
    pub fn keys(&self, py: Python, key_path: Option<&Bound<'_, PyList>>) -> Py<PyList> {
        let keys: Vec<String> = if let Some(key_path) = key_path {
            if key_path.len() == 0 {
                self.results.keys()
            } else {
                // Convert PyList to Vec<String> and traverse NestedDictAccumulator
                let keys: Vec<String> = key_path.extract().expect("Failed to extract key_path as Vec<String>");
                let mut result_accumulator = self.results.get(&keys[0]);
                // Skip the first key and traverse the rest
                for key in &keys[1..] {
                    result_accumulator = result_accumulator.get(key);
                }
                result_accumulator.keys()
            }
        } else {
            // if `sub_keys = None` (when not supplied)
            self.results.keys()
        };
        let result: Py<PyList> = PyList::new(py, keys).unwrap().into();
        return result;
    }

    /// Print keys to screen, to be used within Python
    pub fn print_keys(&self) {
        self.results.print_keys();
    }
}

// Rust only methods
impl RogowskiCoils {
    pub fn construct_virtual_bp_probes(self, rogowski_name: &str, gap_name: &str) -> (BpProbes, BpProbes, f64, f64) {
        // Return structures
        let mut virtual_b_r_probes: BpProbes = BpProbes::new();
        let mut virtual_b_z_probes: BpProbes = BpProbes::new();

        // Get the number of gaps
        let gap_names: Vec<String> = self.results.get(rogowski_name).get("geometry").get("gaps").keys();

        // Exit if there are no gaps for the Rogowski coil
        let n_gaps: usize = gap_names.len();
        if n_gaps == 0 {
            return (virtual_b_r_probes, virtual_b_z_probes, 0.0, 0.0);
        }

        // Gap start and end
        let gap_r_start: f64 = self
            .results
            .get(rogowski_name)
            .get("geometry")
            .get("gaps")
            .get(gap_name)
            .get("r_start")
            .unwrap_f64();
        let gap_r_end: f64 = self
            .results
            .get(rogowski_name)
            .get("geometry")
            .get("gaps")
            .get(gap_name)
            .get("r_end")
            .unwrap_f64();
        let gap_z_start: f64 = self
            .results
            .get(rogowski_name)
            .get("geometry")
            .get("gaps")
            .get(gap_name)
            .get("z_start")
            .unwrap_f64();
        let gap_z_end: f64 = self
            .results
            .get(rogowski_name)
            .get("geometry")
            .get("gaps")
            .get(gap_name)
            .get("z_end")
            .unwrap_f64();

        // Number of virtual bp-probes to calculate the integral `b*d_l`
        let n_virtual_bp_probes: usize = 50;

        // Location to create virtual bp_probes
        let chord_r: Array1<f64> = Array1::linspace(gap_r_start, gap_r_end, n_virtual_bp_probes);
        let chord_z: Array1<f64> = Array1::linspace(gap_z_start, gap_z_end, n_virtual_bp_probes);

        let chord_d_r: f64 = chord_r[1] - chord_r[0];
        let chord_d_z: f64 = chord_z[1] - chord_z[0];

        // Add to virtual b_r probes
        for i_vitual_bp_probe in 0..n_virtual_bp_probes {
            let virtual_b_r_probe_name: String = format!("virtual_b_r_gap={}_sensor={}", gap_name, i_vitual_bp_probe);
            virtual_b_r_probes.add_sensor_rs(
                &virtual_b_r_probe_name,
                0.0, // horizontal = b_r
                chord_r[i_vitual_bp_probe],
                chord_z[i_vitual_bp_probe],
                "no comment".to_string(),
                0.0,
                true,
                1.0,
                Array1::zeros(0),
                Array1::zeros(0),
            );
        }

        // Add to virtual b_z probes
        for i_vitual_bp_probe in 0..n_virtual_bp_probes {
            let virtual_b_z_probe_name: String = format!("virtual_b_r_gap={}_sensor={}", gap_name, i_vitual_bp_probe);
            virtual_b_z_probes.add_sensor_rs(
                &virtual_b_z_probe_name,
                PI / 2.0, // vertical = b_z
                chord_r[i_vitual_bp_probe],
                chord_z[i_vitual_bp_probe],
                "no comment".to_string(),
                0.0,
                true,
                1.0,
                Array1::zeros(0),
                Array1::zeros(0),
            );
        }

        return (virtual_b_r_probes, virtual_b_z_probes, chord_d_r, chord_d_z);
    }

    /// This splits the RogowskiCoils into:
    /// 1.) Static (non time-dependent) object. Note, it is here that the sensors are down-selected, based on ["fit_settings"]["include"]
    /// 2.) A Vec of time-dependent ojbects. Note, the length of the Vec is the number of time-slices we want to reconstruct
    pub fn split_into_static_and_dynamic(&mut self, times_to_reconstruct: &Array1<f64>) -> (SensorsStatic, Vec<SensorsDynamic>) {
        // Vector of boolean's to say if we use the sensor or not
        let include: Vec<bool> = self.results.get("*").get("fit_settings").get("include").unwrap_vec_bool();

        // Convert from boolean to indices
        let include_indices: Vec<usize> = include
            .iter()
            .enumerate()
            .filter(|(_, include)| **include) // Filter for `true` values
            .map(|(i, _)| i) // Extract the indices
            .collect(); // Collect into a Vec

        // Sensor names
        let sensor_names_all: Vec<String> = self.results.keys();
        let n_sensors_all: usize = sensor_names_all.len();
        // Down select sensor names
        let sensor_names: Vec<String> = include_indices.iter().map(|&index| sensor_names_all[index].clone()).collect();
        let n_sensors: usize = sensor_names.len();

        // Fit settings
        // Weight
        let fit_settings_weight: Array1<f64> = self.results.get("*").get("fit_settings").get("weight").unwrap_array1();
        let fit_settings_weight: Array1<f64> = fit_settings_weight.select(Axis(0), &include_indices);
        // Expected value
        let fit_settings_expected_value: Array1<f64> = self.results.get("*").get("fit_settings").get("expected_value").unwrap_array1();
        let fit_settings_expected_value: Array1<f64> = fit_settings_expected_value.select(Axis(0), &include_indices);

        // Greens
        // With PF coils
        let greens_with_pf: Array2<f64> = self.results.get("*").get("greens").get("pf").get("*").unwrap_array2(); // shape = [n_pf, n_sensors]
        let greens_with_pf: Array2<f64> = greens_with_pf.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_pf, n_sensors]
        // With plasma
        let greens_with_grid: Array2<f64> = self.results.get("*").get("greens").get("plasma").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
        let greens_with_grid: Array2<f64> = greens_with_grid.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_pf, n_sensors]
        // With d_sensor_dz (for vertical stability)
        let greens_d_sensor_dz: Array2<f64> = self.results.get("*").get("greens").get("d_plasma_d_z").unwrap_array2(); // shape = [n_z * n_r, n_sensors]
        let greens_d_sensor_dz: Array2<f64> = greens_d_sensor_dz.select(Axis(1), &include_indices); // downselect to only the sensors needed; shape = [n_z * n_r, n_sensors]

        // With passives
        let passive_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").keys();
        let n_passives: usize = passive_names.len();

        // Count the number of degrees of freedom
        let mut n_dof_total: usize = 0;
        for passive_name in &passive_names {
            let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys();
            n_dof_total += dof_names.len();
        }

        let mut greens_with_passives: Array2<f64> = Array2::zeros((n_dof_total, n_sensors));

        // let mut dof_names_total: Vec<String> = Vec::with_capacity(n_dof_total);
        for i_sensor in 0..n_sensors {
            let mut i_dof_total: usize = 0;
            for i_passive in 0..n_passives {
                let passive_name: &str = &passive_names[i_passive];
                let dof_names: Vec<String> = self.results.get(&sensor_names[0]).get("greens").get("passives").get(passive_name).keys(); // something like ["eig01", "eig02", ...]
                for dof_name in dof_names {
                    greens_with_passives[[i_dof_total, i_sensor]] = self
                        .results
                        .get(&sensor_names[i_sensor])
                        .get("greens")
                        .get("passives")
                        .get(&passive_name)
                        .get(&dof_name)
                        .unwrap_f64();

                    // Store the name
                    // dof_names_total[i_dof_total] = dof_name;

                    // Keep count
                    i_dof_total += 1;
                }
            }
        }

        // Create the `SensorsStatic` data
        let results_static: SensorsStatic = SensorsStatic {
            greens_with_grid,
            greens_with_pf,
            greens_with_passives,
            greens_d_sensor_dz,
            fit_settings_weight,
            fit_settings_expected_value,
        };

        // Time dependent
        // Interpolate all sensors to `times_to_reconstruct`
        let n_time: usize = times_to_reconstruct.len();
        let mut measured: Array2<f64> = Array2::zeros((n_sensors_all, n_time));
        for i_sensor in 0..n_sensors_all {
            // Sensor names
            let sensor_name: &str = &sensor_names_all[i_sensor];

            // Measured values
            let time_experimental: Array1<f64> = self.results.get(sensor_name).get("i").get("time_experimental").unwrap_array1();
            let measured_experimental: Array1<f64> = self.results.get(sensor_name).get("i").get("measured_experimental").unwrap_array1();

            // Create the interpolator
            let interpolator = Interp1D::builder(measured_experimental)
                .x(time_experimental.clone())
                .build()
                .expect("RogowskiCoils.split_into_static_and_dynamic: Can't make Interp1D");

            // Do the interpolation
            let measured_this_coil: Array1<f64> = interpolator
                .interp_array(&times_to_reconstruct)
                .expect("RogowskiCoils.split_into_static_and_dynamic: Can't do interpolation");

            // Store for later
            measured.slice_mut(s![i_sensor, ..]).assign(&measured_this_coil);

            // Store in self
            self.results
                .get_or_insert(sensor_name)
                .get_or_insert("i")
                .insert("measured", measured_this_coil);
        }

        // MDSplus is "Sensor-Major", but we want to rearrange the data to be "Time-Major"
        let mut results_dynamic: Vec<SensorsDynamic> = Vec::with_capacity(n_time);
        for i_time in 0..n_time {
            // Select time-slide and the sensors we use in reconstruction
            let measured_this_time_slice_and_sensors: Array1<f64> = measured.slice(s![.., i_time]).select(Axis(0), &include_indices).to_owned();
            // Create new `SensorsDynamic` instance and store
            let results_dynamic_this_time_slice: SensorsDynamic = SensorsDynamic {
                measured: measured_this_time_slice_and_sensors,
            };
            results_dynamic.push(results_dynamic_this_time_slice);
        }

        return (results_static, results_dynamic);
    }

    pub fn calculate_sensor_values_rust(&mut self, coils: &Coils, passives: &Passives, plasma: &Plasma) {
        for sensor_name in self.results.keys() {
            // Coils
            let g_with_coils: Array1<f64> = self.results.get(&sensor_name).get("greens").get("pf").get("*").unwrap_array1(); // shape = [n_pf]
            let coil_currents: Array2<f64> = coils.results.get("pf").get("*").get("i").get("measured").unwrap_array2(); // shape = [n_time, n_pf]
            let n_time: usize = coil_currents.len_of(Axis(0));

            // Plasma
            let g_with_plasma: Array1<f64> = self.results.get(&sensor_name).get("greens").get("plasma").unwrap_array1(); // shape = [n_z*n_r]
            let j_2d: Array3<f64> = plasma.results.get("two_d").get("j").unwrap_array3(); // shape = [n_time, n_z, n_r]
            let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();

            // Loop over time
            let mut sensor_values: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                // PF coil
                let sensor_values_from_coils: f64 = g_with_coils.dot(&coil_currents.slice(s![i_time, ..]));

                // Passives
                // TODO: this looks slow!
                let mut sensor_values_from_passives: f64 = 0.0;
                let passive_names: Vec<String> = self.results.get(&sensor_name).get("greens").get("passives").keys();
                for passive_name in passive_names {
                    let dof_names: Vec<String> = self.results.get(&sensor_name).get("greens").get("passives").get(&passive_name).keys();
                    for dof_name in dof_names {
                        let g_with_passive: f64 = self
                            .results
                            .get(&sensor_name)
                            .get("greens")
                            .get("passives")
                            .get(&passive_name)
                            .get(&dof_name)
                            .unwrap_f64();
                        let passive_dof_value: f64 = passives.results.get(&passive_name).get("dof").get(&dof_name).get("calculated").unwrap_array1()[i_time];
                        sensor_values_from_passives += g_with_passive * passive_dof_value;
                    }
                }

                // Plasma
                let j_2d_flat: Array1<f64> = Array1::from_iter(j_2d.slice(s![i_time, .., ..]).iter().copied());
                let sensor_values_from_plasma: f64 = (&g_with_plasma * j_2d_flat).sum() * d_area;

                // Total
                sensor_values[i_time] = sensor_values_from_coils + sensor_values_from_passives + sensor_values_from_plasma;
            }

            // Store this sensor
            self.results.get_or_insert(&sensor_name).get_or_insert("i").insert("calculated", sensor_values);
        }
    }
}
