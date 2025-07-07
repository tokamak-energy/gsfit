use crate::Plasma;
use crate::plasma_geometry::BoundaryContour;
use crate::plasma_geometry::find_boundary;
use crate::plasma_geometry::find_magnetic_axis;
use crate::sensors::{SensorsDynamic, SensorsStatic};
use crate::source_functions::SourceFunctionTraits;
use core::f64;
use ndarray::Axis;
use ndarray::{Array1, Array2, Array3, s};
use std::sync::Arc;
extern crate blas_src;
use lapack::*;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;
const PI: f64 = std::f64::consts::PI;

/// Grad-Shafranov solution, at single time-slice
pub struct GsSolution<'a> {
    // Object inputs
    plasma: &'a Plasma,
    coils_dynamic: &'a SensorsDynamic,
    bp_probes_static: &'a SensorsStatic,
    bp_probes_dynamic: &'a SensorsDynamic,
    flux_loops_static: &'a SensorsStatic,
    flux_loops_dynamic: &'a SensorsDynamic,
    rogowski_coils_static: &'a SensorsStatic,
    rogowski_coils_dynamic: &'a SensorsDynamic,
    isoflux_static: &'a SensorsStatic,
    isoflux_dynamic: &'a SensorsDynamic,
    isoflux_boundary_static: &'a SensorsStatic,
    isoflux_boundary_dynamic: &'a SensorsDynamic,
    n_iter_max: usize,
    n_iter_min: usize,
    n_iter_no_vertical_feedback: usize,
    gs_error_tolerence: f64,
    // Results
    pub gs_error_calculated: f64,
    pub ff_prime_dof_values: Array1<f64>,
    pub p_prime_dof_values: Array1<f64>,
    pub psi_2d_coils: Array2<f64>,
    pub passive_dof_values: Array1<f64>,
    pub br_2d: Array2<f64>,
    pub bz_2d: Array2<f64>,
    pub psi_2d: Array2<f64>,
    pub psi_n_2d: Array2<f64>,
    pub j_2d: Array2<f64>,
    pub mask: Array2<f64>,
    pub psi_b: f64,
    pub psi_a: f64,
    pub ip: f64,
    pub boundary_r: Array1<f64>,
    pub boundary_z: Array1<f64>,
    pub bounding_r: f64,
    pub bounding_z: f64,
    pub delta_z: f64,
    // pub n_iter: usize,
    pub xpt_upper_r: f64,
    pub xpt_upper_z: f64,
    pub xpt_lower_r: f64,
    pub xpt_lower_z: f64,
    pub n_iter: usize,
    pub r_mag: f64,
    pub z_mag: f64,
    pub xpt_diverted: bool,
    pub p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    passive_regularisations: Array2<f64>,
    passive_regularisations_weight: Array1<f64>,
}

impl<'a> GsSolution<'a> {
    pub fn new(
        plasma: &'a Plasma,
        coils_dynamic: &'a SensorsDynamic,
        bp_probes_static: &'a SensorsStatic,
        bp_probes_dynamic: &'a SensorsDynamic,
        flux_loops_static: &'a SensorsStatic,
        flux_loops_dynamic: &'a SensorsDynamic,
        rogowski_coils_static: &'a SensorsStatic,
        rogowski_coils_dynamic: &'a SensorsDynamic,
        isoflux_static: &'a SensorsStatic,
        isoflux_dynamic: &'a SensorsDynamic,
        isoflux_boundary_static: &'a SensorsStatic,
        isoflux_boundary_dynamic: &'a SensorsDynamic,
        n_iter_max: usize,
        n_iter_min: usize,
        n_iter_no_vertical_feedback: usize,
        gs_error_tolerence: f64,
        p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
        ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
        passive_regularisations: Array2<f64>,
        passive_regularisations_weight: Array1<f64>,
    ) -> Self {
        GsSolution {
            // Object inputs
            plasma,
            coils_dynamic,
            bp_probes_static,
            bp_probes_dynamic,
            flux_loops_static,
            flux_loops_dynamic,
            rogowski_coils_static,
            rogowski_coils_dynamic,
            isoflux_static,
            isoflux_dynamic,
            isoflux_boundary_static,
            isoflux_boundary_dynamic,
            n_iter_max,
            n_iter_min,
            n_iter_no_vertical_feedback,
            gs_error_tolerence,
            // Results
            gs_error_calculated: f64::NAN,
            ff_prime_dof_values: Array1::zeros(0),
            p_prime_dof_values: Array1::zeros(0),
            passive_dof_values: Array1::zeros(0),
            br_2d: Array2::zeros((0, 0)),
            bz_2d: Array2::zeros((0, 0)),
            psi_2d: Array2::zeros((0, 0)),
            psi_n_2d: Array2::zeros((0, 0)),
            j_2d: Array2::zeros((0, 0)),
            mask: Array2::zeros((0, 0)),
            psi_2d_coils: Array2::zeros((0, 0)),
            psi_b: f64::NAN,
            psi_a: f64::NAN,
            ip: f64::NAN,
            boundary_r: Array1::zeros(0),
            boundary_z: Array1::zeros(0),
            bounding_r: f64::NAN,
            bounding_z: f64::NAN,
            delta_z: f64::NAN,
            xpt_upper_r: f64::NAN,
            xpt_upper_z: f64::NAN,
            xpt_lower_r: f64::NAN,
            xpt_lower_z: f64::NAN,
            n_iter: usize::MAX,
            r_mag: f64::NAN,
            z_mag: f64::NAN,
            xpt_diverted: false,
            p_prime_source_function,
            ff_prime_source_function,
            passive_regularisations,
            passive_regularisations_weight,
        }
    }

    /// If the solver fails to converge, this function will set the solution to NAN values (but with the correct shape).
    fn set_to_failed_time_slice(&mut self) {
        self.gs_error_calculated = f64::NAN;
        self.ff_prime_dof_values = self.ff_prime_dof_values.to_owned() * f64::NAN;
        self.p_prime_dof_values = self.p_prime_dof_values.to_owned() * f64::NAN;
        self.passive_dof_values = self.passive_dof_values.to_owned() * f64::NAN;
        self.br_2d = self.br_2d.to_owned() * f64::NAN;
        self.bz_2d = self.bz_2d.to_owned() * f64::NAN;
        self.psi_2d = self.psi_2d.to_owned() * f64::NAN;
        self.psi_n_2d = self.psi_n_2d.to_owned() * f64::NAN;
        self.j_2d = self.j_2d.to_owned() * f64::NAN;
        self.mask = self.mask.to_owned() * f64::NAN;
        self.psi_2d_coils = self.psi_2d_coils.to_owned() * f64::NAN;
        self.psi_b = f64::NAN;
        self.psi_a = f64::NAN;
        self.ip = f64::NAN;
        self.boundary_r = Array1::zeros(0);
        self.boundary_z = Array1::zeros(0);
        self.bounding_r = f64::NAN;
        self.bounding_z = f64::NAN;
        self.delta_z = f64::NAN;
        self.xpt_upper_r = f64::NAN;
        self.xpt_upper_z = f64::NAN;
        self.xpt_lower_r = f64::NAN;
        self.xpt_lower_z = f64::NAN;
        self.n_iter = usize::MAX;
        self.r_mag = f64::NAN;
        self.z_mag = f64::NAN;
        self.xpt_diverted = false;
    }

    /// Solve the inverse Grad-Shafranov problem
    pub fn solve(&mut self) {
        let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.p_prime_source_function.clone();
        let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.ff_prime_source_function.clone();

        // Unpack objects
        let coils_dynamic: &SensorsDynamic = self.coils_dynamic;
        let plasma: &Plasma = &self.plasma;

        // Get sensors
        let bp_probes_static: &SensorsStatic = self.bp_probes_static;
        let bp_probes_dynamic: &SensorsDynamic = self.bp_probes_dynamic;
        let flux_loops_static: &SensorsStatic = self.flux_loops_static;
        let flux_loops_dynamic: &SensorsDynamic = self.flux_loops_dynamic;
        let rogowski_coils_static: &SensorsStatic = self.rogowski_coils_static;
        let rogowski_coils_dynamic: &SensorsDynamic = self.rogowski_coils_dynamic;
        let isoflux_static: &SensorsStatic = self.isoflux_static;
        let isoflux_dynamic: &SensorsDynamic = self.isoflux_dynamic;
        let isoflux_boundary_static: &SensorsStatic = self.isoflux_boundary_static;
        let isoflux_boundary_dynamic: &SensorsDynamic = self.isoflux_boundary_dynamic;

        // Plasma grid
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();
        let flat_r: Array1<f64> = plasma.results.get("grid").get("flat").get("r").unwrap_array1();
        let mesh_r: Array2<f64> = plasma.results.get("grid").get("mesh").get("r").unwrap_array2();
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();
        let limit_pts_r: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("r").unwrap_array1();
        let limit_pts_z: Array1<f64> = plasma.results.get("limiter").get("limit_pts").get("z").unwrap_array1();
        let vessel_r: Array1<f64> = plasma.results.get("vessel").get("r").unwrap_array1();
        let vessel_z: Array1<f64> = plasma.results.get("vessel").get("z").unwrap_array1();

        // Degrees of freedom
        let passives_shape: &[usize] = bp_probes_static.greens_with_passives.shape();
        let n_passive_dof: usize = passives_shape[0];
        let n_p_prime_dof: usize = ff_prime_source_function.source_function_n_dof();
        let n_ff_prime_dof: usize = ff_prime_source_function.source_function_n_dof();
        let n_iter_no_vertical_feedback: usize = self.n_iter_no_vertical_feedback;

        // Constraints
        let n_bp: usize = bp_probes_dynamic.measured.len();
        let n_fl: usize = flux_loops_dynamic.measured.len();
        let n_rog: usize = rogowski_coils_dynamic.measured.len();
        let n_isoflux: usize = isoflux_dynamic.measured.len();
        let n_isoflux_boundary: usize = isoflux_boundary_dynamic.measured.len();
        let n_p_prime_regularisation: usize = p_prime_source_function.source_function_regularisation().shape()[0];
        let n_ff_prime_regularisation: usize = ff_prime_source_function.source_function_regularisation().shape()[0];
        let n_source_function_reg: usize = n_p_prime_regularisation + n_ff_prime_regularisation;
        let passive_regularisations: Array2<f64> = self.passive_regularisations.to_owned();
        let n_passive_regularisation: usize = passive_regularisations.shape()[0];
        let n_delta_z_regularisation: usize = 0; // initially set to 0 because we don't have previous iteration
        let n_constraints: usize =
            n_bp + n_fl + n_rog + n_isoflux + n_isoflux_boundary + n_source_function_reg + n_passive_regularisation + n_delta_z_regularisation;

        // Magnetic sensor's Greens tables
        let greens_bp_probes_grid: Array2<f64> = bp_probes_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_bp_probes_dz: Array2<f64> = bp_probes_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_bp_probes_pf: Array2<f64> = bp_probes_static.greens_with_pf.to_owned(); // shape = [n_pf, n_sensors]
        let greens_bp_probes_passives: Array2<f64> = bp_probes_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_flux_loops_grid: Array2<f64> = flux_loops_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_flux_loops_dz: Array2<f64> = flux_loops_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_flux_loops_pf: Array2<f64> = flux_loops_static.greens_with_pf.to_owned(); // shape = [n_pf, n_sensors]
        let greens_flux_loops_passives: Array2<f64> = flux_loops_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_rogowski_coils_grid: Array2<f64> = rogowski_coils_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_rogowski_coils_dz: Array2<f64> = rogowski_coils_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]  TO ADD!!!!!
        let greens_rogowski_coils_pf: Array2<f64> = rogowski_coils_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_rogowski_coils_passives: Array2<f64> = rogowski_coils_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_grid: Array2<f64> = isoflux_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_dz: Array2<f64> = isoflux_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]  TO ADD!!!!!
        let greens_isoflux_pf: Array2<f64> = isoflux_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_passives: Array2<f64> = isoflux_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_boundary_grid: Array2<f64> = isoflux_boundary_static.greens_with_grid.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_boundary_dz: Array2<f64> = isoflux_boundary_static.greens_d_sensor_dz.to_owned(); // shape = [n_z*n_r, n_sensors]  TO ADD!!!!!
        let greens_isoflux_boundary_pf: Array2<f64> = isoflux_boundary_static.greens_with_pf.to_owned(); // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_boundary_passives: Array2<f64> = isoflux_boundary_static.greens_with_passives.to_owned(); // shape = [n_passive_dof, n_sensors]

        // pf_coil_currents
        let pf_coil_currents: Array1<f64> = coils_dynamic.measured.to_owned();

        // TODO: IDEA- change the normalisation so that it does represent current. But this won't work for the IVC eigenvalues
        self.passive_dof_values = Array1::zeros(n_passive_dof);

        // Initialise plasma with point source current
        self.initialise_plasma_with_point_source_current();

        // Some variables we want to track between iterations
        let mut dof_values_previous: Array1<f64> = Array1::zeros(n_p_prime_dof + n_ff_prime_dof + n_passive_dof + 1);
        let mut psi_a_previous: f64 = 0.0; // needed to calculate gs-error

        // Iteration loop
        'iteration_loop: for i_iter in 0..self.n_iter_max {
            // From previous iteration
            let j_2d: Array2<f64> = self.j_2d.to_owned();

            // Calculate br and bz
            // Note, `self.calculate_b()` needs to be before `self.calculate_psi()`, because `br` will be used to calculate
            // the delta_z numerical stabilisation, which is added to `psi`
            let (d_br_d_z_2d, d_bz_d_z_2d): (Array2<f64>, Array2<f64>) = self.calculate_b();

            // Updates psi
            self.calculate_psi();
            let psi_2d: Array2<f64> = self.psi_2d.to_owned();

            // Apply the delta_z stabilisation
            if i_iter > n_iter_no_vertical_feedback + 1 {
                // Make `br` and `bz` consistent with `psi`
                self.br_2d = self.br_2d.to_owned() + self.delta_z * &d_br_d_z_2d;
                self.bz_2d = self.bz_2d.to_owned() + self.delta_z * &d_bz_d_z_2d;
            }

            // Get br and bz
            let br_2d: Array2<f64> = self.br_2d.to_owned();
            let bz_2d: Array2<f64> = self.bz_2d.to_owned();

            // Find boundary
            let plasma_boundary: Result<BoundaryContour, String> = find_boundary(
                r.clone(),
                z.clone(),
                psi_2d.clone(),
                br_2d.clone(),
                bz_2d.clone(),
                d_br_d_z_2d.clone(),
                d_bz_d_z_2d.clone(),
                limit_pts_r.clone(),
                limit_pts_z.clone(),
                vessel_r.clone(),
                vessel_z.clone(),
                self.r_mag, // previous iteration
                self.z_mag, // previous iteration
            );

            // Test if we have found a plasma boundary
            if plasma_boundary.is_err() {
                self.set_to_failed_time_slice();
                break 'iteration_loop; // exit the iteration loop for this time-slice
            }

            // Unwrap and store the plasma boundary
            let plasma_boundary_unwrapped: BoundaryContour = plasma_boundary.expect("Failed to find plasma boundary");
            self.mask = plasma_boundary_unwrapped.mask.expect("Failed to unwrap mask");
            self.psi_b = plasma_boundary_unwrapped.bounding_psi;
            self.boundary_r = plasma_boundary_unwrapped.boundary_r;
            self.boundary_z = plasma_boundary_unwrapped.boundary_z;
            self.bounding_r = plasma_boundary_unwrapped.bounding_r;
            self.bounding_z = plasma_boundary_unwrapped.bounding_z;
            let mask: Array2<f64> = self.mask.to_owned();
            let psi_b: f64 = self.psi_b;
            self.xpt_diverted = plasma_boundary_unwrapped.xpt_diverted;

            // Find the magnetic axis (o-point)
            // TODO: we have calculated the turning points (x-points and o-points) twice! Once in `find_boundary` and once here.
            // TODO: should add an exception if the magnetic axis is not found
            let (mag_r, mag_z, psi_a): (f64, f64, f64) = find_magnetic_axis(&r, &z, &br_2d, &bz_2d, &d_bz_d_z_2d, &psi_2d, self.r_mag, self.z_mag);
            self.r_mag = mag_r;
            self.z_mag = mag_z;
            self.psi_a = psi_a;

            // Calculate psi_n_2d
            let psi_n_2d: Array2<f64> = &mask * (&psi_2d - psi_a) / (psi_b - psi_a);
            self.psi_n_2d = psi_n_2d.clone();

            // Calculate GS error
            self.calculate_gs_error(psi_a_previous);
            psi_a_previous = psi_a; // needed to calculate gs-error in next iteration

            // Check for convergence
            let gs_error_calculated: f64 = self.gs_error_calculated;
            if gs_error_calculated < self.gs_error_tolerence && i_iter > self.n_iter_min {
                self.n_iter = i_iter;
                // println!("Found GS solution.");
                break 'iteration_loop; // Exit the iteration loop
            }

            // Check if we have reached the maximum number of iterations
            if i_iter == self.n_iter_max - 1 {
                // Ensure that failed time-slices are excluded
                self.set_to_failed_time_slice();
                break 'iteration_loop; // Exit the iteration loop
            }

            // Flatten variables
            let mask_flat: Array1<f64> = Array1::from_iter(mask.iter().cloned());
            let psi_n_flat: Array1<f64> = Array1::from_iter(psi_n_2d.iter().cloned());
            let j_2d_flat: Array1<f64> = Array1::from_iter(j_2d.iter().cloned());

            let n_vertical_stabilisation: usize;
            if i_iter > n_iter_no_vertical_feedback {
                n_vertical_stabilisation = 1;
            } else {
                n_vertical_stabilisation = 0;
            }

            let n_dof: usize = n_p_prime_dof + n_ff_prime_dof + n_passive_dof + n_vertical_stabilisation;
            // Create the fitting matrix
            let mut fitting_matrix: Array2<f64> = Array2::zeros((n_constraints, n_dof));
            let mut constraint_weights: Array1<f64> = Array1::zeros(n_constraints);
            let mut constraint_values_from_coils: Array1<f64> = Array1::zeros(n_constraints);
            let mut s_measured: Array1<f64> = Array1::zeros(n_constraints);

            // Counter for the constraints
            let mut i_constraint: usize = 0;

            // Add bp_probes to fitting matrix
            for i_sensor in 0..n_bp {
                // j = 2.0 * pi * r * p_prime + 2.0 * pi * ff_prime / (mu_0 * r)

                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_bp_probes_passives[[i_passive_dof, i_sensor]];
                }

                // Vertical stability (using previous iteration)
                // j_2d is not consistent with mask. This inconsistency is how the plasma can "move" from iteration to iteration
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_bp_probes_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                    //  * &mask_flat
                }

                // PF coil component
                let tmp: Array1<f64> = greens_bp_probes_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = bp_probes_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = bp_probes_static.fit_settings_weight[i_sensor] / bp_probes_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add flux_loops to fitting matrix
            for i_sensor in 0..n_fl {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_flux_loops_passives[[i_passive_dof, i_sensor]];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_flux_loops_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                    //  * &mask_flat
                }

                // PF coil component
                let tmp: Array1<f64> = greens_flux_loops_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                // s_measured[i_constraint] = flux_loops_rs.all.psi.measured[i_sensor];
                s_measured[i_constraint] = flux_loops_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    2.0 * PI * flux_loops_static.fit_settings_weight[i_sensor] / flux_loops_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add rogowski_coils to fitting matrix
            for i_sensor in 0..n_rog {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_rogowski_coils_passives[[i_passive_dof, i_sensor]];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_rogowski_coils_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                // TODO: I checked and `greens_rogowski_coils_pf["bvlb", "BVLBCASE"] = 16`
                let tmp: Array1<f64> = greens_rogowski_coils_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = rogowski_coils_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    rogowski_coils_static.fit_settings_weight[i_sensor] / rogowski_coils_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux to fitting matrix
            for i_sensor in 0..n_isoflux {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_isoflux_passives[[i_passive_dof, i_sensor]];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_isoflux_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = isoflux_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = isoflux_static.fit_settings_weight[i_sensor] / isoflux_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux_boundary to fitting matrix
            for i_sensor in 0..n_isoflux_boundary {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_n_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] =
                        greens_isoflux_boundary_passives[[i_passive_dof, i_sensor]];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux_boundary!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_boundary_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = greens_isoflux_boundary_pf.slice(s![.., i_sensor]).to_owned() * &pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = psi_a;

                // Store weights
                constraint_weights[i_constraint] =
                    isoflux_boundary_static.fit_settings_weight[i_sensor] / isoflux_boundary_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add p_prime_regularisation to fitting matrix
            let p_prime_regularisation: Array2<f64> = p_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_p_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, 0..n_p_prime_dof])
                    .assign(&p_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add ff_prime_regularisation to fitting matrix
            let ff_prime_regularisation: Array2<f64> = ff_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_ff_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof])
                    .assign(&ff_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // // Add passive regularisation to the fitting matrix
            let regularisation_scaling: f64 = 0.001 * PI; // This regularisation_scaling factor need improving and explaining!

            let passive_regularisations_weight: Array1<f64> = self.passive_regularisations_weight.to_owned();
            for i_regularisation in 0..n_passive_regularisation {
                let passive_regularisation: Array1<f64> = passive_regularisations.slice(s![i_regularisation, ..]).to_owned();

                // Add passive degrees of freedom
                fitting_matrix
                    .slice_mut(s![
                        i_constraint,
                        n_p_prime_dof + n_ff_prime_dof..=n_p_prime_dof + n_ff_prime_dof + n_passive_dof - 1
                    ])
                    .assign(&passive_regularisation);

                // Add weight
                constraint_weights[i_constraint] = passive_regularisations_weight[i_regularisation] * regularisation_scaling;

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Solve for the least squares problem for the source function coefficients, passive currents, and vertical stability
            let a: Array2<f64> = Array2::from_diag(&constraint_weights).dot(&fitting_matrix); // matrix-matrix multiplication
            let b: Array1<f64> = &constraint_weights * &s_measured - &constraint_weights * &constraint_values_from_coils;

            fn l2_norm(v: &Array1<f64>) -> f64 {
                // Sum of squares of the elements in the vector
                let sum_of_squares: f64 = v.iter().map(|&x| x * x).sum();
                // Take the square root to get the L2 norm
                sum_of_squares.sqrt()
            }

            // Preconditioner
            let n_cols: usize = a.ncols();

            // Compute the L2 norm for each column and fill the diagonal of D
            let mut d: Array2<f64> = Array2::zeros((n_cols, n_cols)); // Initialize a square matrix D with zeros
            for i in 0..n_cols {
                let column: Array1<f64> = a.column(i).to_owned();
                let norm = l2_norm(&column);
                // Fill the diagonal of D with the inverse of the norm, or 0.0 if the norm is zero
                if norm > 0.0 {
                    d[[i, i]] = 1.0 / norm;
                } else {
                    println!("Warning: norm of column {} is zero, setting preconditioner to zero.", i);
                    d[[i, i]] = 0.0;
                }
            }

            let a_preconditioned: Array2<f64> = a.clone().dot(&d);

            // This uses LAPACK; I found that ndarray_linlg internally sets `rcond = -1`.
            // "If RCOND < 0, machine precision is used instead."
            // `ndarray_linlg` uses "dgelsd"
            // EFIT uses LAPACK's "la_gelss" (https://www.netlib.org/lapack/explore-html/da/d55/group__gelss_gac6159de3953ae0386c2799294745ac90.html)
            // let dof_values: Array1<f64> = a.least_squares(&b).unwrap().solution;
            // let dof_values_preconditioned: Array1<f64> = a_preconditioned.least_squares(&b).expect("failed least squares").solution;

            // Get dimensions of the input
            let (m_usize, n_usize) = a.dim();
            let m: i32 = m_usize as i32;
            let n: i32 = n_usize as i32;
            let nrhs: i32 = 1; // Number of right-hand sides

            // Convert A and B to column-major layout for LAPACK
            let mut a_vec: Vec<f64> = a_preconditioned.t().iter().cloned().collect();
            let mut b_vec: Vec<f64> = b.to_vec();

            // Allocate outputs
            let mut s: Vec<f64> = vec![0.0; n_usize.min(m_usize)]; // Singular values
            let mut rank: i32 = 0; // Effective rank
            let rcond: f64 = -1.0; // Use machine precision
            let mut info: i32 = 999;

            // Workspace query
            let lwork: i32 = 4218; // TODO: what is this number?????????????????????
            let mut work = vec![0.0; lwork as usize];

            // using the same as EFIT
            unsafe {
                dgelss(
                    m,          // `m` the number of rows of the matrix `a`
                    n,          // `n` the number of columns of the matrix `a`
                    nrhs,       // `nrhs` the number of right hand sides, i.e., the number of columns of the matrices `b`
                    &mut a_vec, // `a` is DOUBLE PRECISION array
                    m,          // `lda` the leading dimension of the array `a`
                    &mut b_vec, // `b` vector
                    m,          // `ldb` the leading dimension of the array `b`
                    &mut s,     // `s` the singular values of A in decreasing order
                    rcond,      // `rcond` if RCOND < 0, machine precision is used
                    &mut rank,  // `rank`
                    &mut work,  // `work`
                    lwork,      // `lwork` Workspace query
                    &mut info,  // `info`
                );
            }

            let d_new: Array1<f64> = Array1::from_vec(b_vec[0..n_dof].to_vec());

            let mut dof_values: Array1<f64> = d.dot(&d_new); // `d` is the preconditioning matrix

            // // Could add Anderson mixing here??????????????
            // if i_iter > 3 {
            //     dof_values = 0.6 * &dof_values + 0.4 * &dof_values_previous;
            // }
            // let dof_values_old: Array1<f64> = dof_values.clone();

            // Compute the condition number
            if let (Some(&sigma_max), Some(&sigma_min)) = (s.first(), s.iter().filter(|&&x| x > 0.0).last()) {
                let condition_number = sigma_max / sigma_min;
            } else {
                println!("Matrix is rank-deficient or singular, condition number is undefined.");
            }

            // // Add Anderson mixing. Will this help???  // NO: Anderson mixing does not seem to help!!
            // if i_iter > 3 {
            //     dof_values = 0.3 * &dof_values + 0.7 * &dof_values_previous;
            // }

            if i_iter > n_iter_no_vertical_feedback {
                dof_values_previous = dof_values.clone();
            }

            // Extract p_prime
            let p_prime_dof_values: Array1<f64> = dof_values.slice(s![0..n_p_prime_dof]).to_owned();
            self.p_prime_dof_values = p_prime_dof_values.clone();

            // Extract ff_prime
            let ff_prime_dof_values: Array1<f64> = dof_values.slice(s![n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof]).to_owned();
            self.ff_prime_dof_values = ff_prime_dof_values.clone();

            // Extract passive currents
            let passive_dof_values = dof_values
                .slice(s![n_p_prime_dof + n_ff_prime_dof..n_p_prime_dof + n_ff_prime_dof + n_passive_dof])
                .to_owned();
            self.passive_dof_values = passive_dof_values.clone();

            // Extract vertical stability
            let delta_z: f64;
            if i_iter > n_iter_no_vertical_feedback {
                delta_z = dof_values.last().expect("dof_values empty").to_owned();
            } else {
                delta_z = 0.0;
            }
            self.delta_z = delta_z.clone();

            // Calculate profiles
            let psi_n_flat: Array1<f64> = Array1::from_iter(psi_n_2d.iter().cloned());

            let p_prime_2d: Array2<f64> = p_prime_source_function
                .source_function_value(&psi_n_flat, &p_prime_dof_values.clone())
                .to_shape((n_z, n_r))
                .expect("error in p_prime_2d")
                .to_owned();
            let j_2d_p_prime: Array2<f64> = 2.0 * PI * &mesh_r * p_prime_2d * &mask;

            let ff_prime_2d: Array2<f64> = ff_prime_source_function
                .source_function_value(&psi_n_flat, &ff_prime_dof_values.clone())
                .to_shape((n_z, n_r))
                .expect("error in ff_prime_2d")
                .to_owned();
            let j_2d_ff_prime: Array2<f64> = 2.0 * PI * ff_prime_2d * &mask / (MU_0 * &mesh_r);

            // Calculate j_2d
            let j_2d: Array2<f64> = j_2d_p_prime + j_2d_ff_prime;
            self.j_2d = j_2d.clone();

            // Total plasma current
            let i_2d: Array2<f64> = &j_2d * d_area;
            let ip: f64 = i_2d.sum();
            self.ip = ip;
        }
    }

    /// Calculate the poloidal flux, psi, in the 2d (r, z) grid.
    pub fn calculate_psi(&mut self) {
        // Unpack from self
        let plasma: &Plasma = &self.plasma;

        // Get stuff out of class
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let j_2d: Array2<f64> = self.j_2d.to_owned();
        let g_grid_grid: Array2<f64> = plasma.results.get("greens").get("grid_grid").get("psi").unwrap_array2(); // shape = [n_z*n_r, n_r]
        let g_passive_grid: Array2<f64> = plasma.get_greens_passive_grid(); // shape = [n_z*n_r, n_passive_dof]
        let passive_dof_values: Array1<f64> = self.passive_dof_values.to_owned();
        let delta_z: f64 = self.delta_z;

        // Calculate some sizes
        let passives_shape: &[usize] = g_passive_grid.shape();
        let n_passive_dof: usize = passives_shape[1];

        // psi from coils
        let psi_2d_coils: Array2<f64> = self.psi_2d_coils.clone();

        // psi from passives
        let mut psi_2d_passives: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_passive_dof in 0..n_passive_dof {
            let psi_2d_passives_this_slice: Array2<f64> = g_passive_grid
                .slice(s![.., i_passive_dof])
                .to_owned()
                .to_shape((n_z, n_r))
                .expect("psi_2d_passives: can't change shape")
                .to_owned()
                * passive_dof_values[i_passive_dof];
            psi_2d_passives = &psi_2d_passives + psi_2d_passives_this_slice;
        }

        // Do some re-shaping
        let (g_grid_grid_flat, _): (Vec<f64>, Option<usize>) = g_grid_grid.into_raw_vec_and_offset();
        let g_grid_grid_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_grid_grid_flat).expect("Failed to reshape into Array3");

        // Conceptually, we are looping over the current's and modifying the Green's table for the current
        let mut psi_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_cur_z in 0..n_z {
            // Cyclic indexing for the z-axis
            let z_indexer: Vec<usize> = (0..n_z).map(|i_z| i_cur_z.abs_diff(i_z)).collect();

            for i_cur_r in 0..n_r {
                // Performance improvement: a lot of the grid doesn't have plasma current
                if j_2d[(i_cur_z, i_cur_r)].abs() > 0.0 {
                    // Select the Green's table for the radial current source location
                    // selecting the r-axis and re-ordering in one operation, might be fastest
                    let g_grid_grid_2d_reordered: Array2<f64> = g_grid_grid_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer);

                    // Calculate the contribution to psi from this current source
                    let psi_2d_plasma_this_j: Array2<f64> = g_grid_grid_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;

                    // Add contribution from current
                    psi_2d_plasma += &psi_2d_plasma_this_j;
                }
            }
        }

        // Calculate psi
        let mut psi_2d: Array2<f64> = &psi_2d_coils + &psi_2d_passives + &psi_2d_plasma;

        // Apply delta_z correction
        // Only apply delta_z correction if it is not NaN
        if !delta_z.is_nan() {
            let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();

            // Calculate d(psi)/d(z)
            let mut d_psi_d_z: Array2<f64> = Array2::zeros((n_z, n_r));
            for i_z in 0..n_z {
                let tmp: Array1<f64> = -2.0 * PI * &r * self.br_2d.slice(s![i_z, ..]);
                d_psi_d_z.slice_mut(s![i_z, ..]).assign(&tmp);
            }

            // Apply vertical stability correction
            psi_2d = psi_2d + delta_z * d_psi_d_z;
        }

        // Add to class
        self.psi_2d = psi_2d;
    }

    /// Calcuates br and bz in the 2d (r, z) grid.
    /// Called prior to x-point finding, where br=0 and bz=0
    pub fn calculate_b(&mut self) -> (Array2<f64>, Array2<f64>) {
        // Unpack from self
        let plasma: &Plasma = self.plasma;
        let coils_dynamic: &SensorsDynamic = &self.coils_dynamic;

        // Unpacking everything;  timing: 15ms, with [n_r, n_z]=[100, 201]
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();
        let g_br_coils: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("br").unwrap_array3(); // shape = (n_z, n_r, n_pf)
        let g_bz_coils: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("bz").unwrap_array3(); // shape = (n_z, n_r, n_pf)
        let g_br_plasma: Array2<f64> = plasma.results.get("greens").get("grid_grid").get("br").unwrap_array2(); // shape = (n_z * n_r, n_r)
        let g_bz_plasma: Array2<f64> = plasma.results.get("greens").get("grid_grid").get("bz").unwrap_array2(); // shape = (n_z * n_r, n_r)
        let g_br_passives: Array2<f64> = plasma.get_greens_passive_grid_br();
        let g_bz_passives: Array2<f64> = plasma.get_greens_passive_grid_bz(); // shape = (n_z * n_r, n_passive_dofs)

        let g_d_br_d_z_coils: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("d_br_d_z").unwrap_array3(); // shape = (n_z, n_r, n_pf)
        let g_d_bz_d_z_coils: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("d_bz_d_z").unwrap_array3(); // shape = (n_z, n_r, n_pf)
        let g_d_br_d_z_plasma: Array2<f64> = plasma.results.get("greens").get("grid_grid").get("d_br_d_z").unwrap_array2(); // shape = (n_z * n_r, n_r)
        let g_d_bz_d_z_plasma: Array2<f64> = plasma.results.get("greens").get("grid_grid").get("d_bz_d_z").unwrap_array2(); // shape = (n_z * n_r, n_r)
        let g_d_br_d_z_passives: Array2<f64> = plasma.get_greens_passive_grid_d_br_d_z();
        let g_d_bz_d_z_passives: Array2<f64> = plasma.get_greens_passive_grid_d_bz_d_z(); // shape = (n_z * n_r, n_passive_dofs)

        let j_2d: Array2<f64> = self.j_2d.to_owned();
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let pf_currents: Array1<f64> = coils_dynamic.measured.to_owned();
        let passive_dof_values: Array1<f64> = self.passive_dof_values.to_owned();

        let (_1, _2, n_pf): (usize, usize, usize) = g_br_coils.dim();

        // Coils;  timing: 1.9ms, with [n_r, n_z]=[100,201]
        let mut br_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut bz_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_pf in 0..n_pf {
            br_2d_coils = br_2d_coils + &g_br_coils.slice(s![.., .., i_pf]) * pf_currents[i_pf];
            bz_2d_coils = bz_2d_coils + &g_bz_coils.slice(s![.., .., i_pf]) * pf_currents[i_pf];
        }

        // Passives;  timing: 4ms, with [n_r, n_z]=[100,201]
        let passives_shape: &[usize] = g_br_passives.shape(); // shape = (n_z * n_r, n_passive_dof)
        let n_passive_dof: usize = passives_shape[1];

        let mut br_2d_passives: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut bz_2d_passives: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_passive_dof in 0..n_passive_dof {
            // br
            let br_2d_passives_this_slice: Array2<f64> = g_br_passives
                .slice(s![.., i_passive_dof])
                .to_owned()
                .to_shape((n_z, n_r))
                .expect("variable: br_2d_passives_this_slice;  probably wrong array dimensions")
                .to_owned()
                * passive_dof_values[i_passive_dof];
            br_2d_passives = br_2d_passives + br_2d_passives_this_slice;

            // bz
            let bz_2d_passives_this_slice: Array2<f64> = g_bz_passives
                .slice(s![.., i_passive_dof])
                .to_owned()
                .to_shape((n_z, n_r))
                .expect("variable: bz_2d_passives_this_slice;  probably wrong array dimensions")
                .to_owned()
                * passive_dof_values[i_passive_dof];
            bz_2d_passives = bz_2d_passives + bz_2d_passives_this_slice;
        }

        // Plasma br and bz;  timing: 921ms, with [n_r, n_z]=[100, 201]
        let (g_br_plasma_flat, _): (Vec<f64>, Option<usize>) = g_br_plasma.into_raw_vec_and_offset(); // tested with BZ and can confirm that it is working...
        let (g_bz_plasma_flat, _): (Vec<f64>, Option<usize>) = g_bz_plasma.into_raw_vec_and_offset();
        let g_br_plasma_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_br_plasma_flat).expect("Failed to reshape into Array3");
        let g_bz_plasma_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_bz_plasma_flat).expect("Failed to reshape into Array3");

        let mut br_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut bz_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));

        // Conceptually, we are looping over the current's and modifying the Green's table for the current
        for i_cur_z in 0..n_z {
            // Cyclic indexing for the z-axis (a current filament "looks" the same in z, but not r)
            let z_indexer: Vec<usize> = (0..n_z).map(|i_z| i_cur_z.abs_diff(i_z)).collect();

            // br it is up/down asymmetric; bz is up/down symmetric
            let sign: Vec<f64> = (0..n_z).map(|i_z| if i_z <= i_cur_z { -1.0 } else { 1.0 }).collect();
            let sign_ndarray: Array1<f64> = Array1::from(sign);
            let sign_broadcast: Array2<f64> = sign_ndarray.insert_axis(Axis(1)); // shape = [n_z, 1] (column-vector)

            for i_cur_r in 0..n_r {
                // Performance improvement: a lot of the grid doesn't have plasma current
                if j_2d[(i_cur_z, i_cur_r)].abs() > 0.0 {
                    // Select and reorder the Green's table for the radial current source location
                    let g_br_plasma_2d_reordered: Array2<f64> = g_br_plasma_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer) * &sign_broadcast;
                    let g_bz_plasma_2d_reordered: Array2<f64> = g_bz_plasma_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer);

                    // Calculate the contribution to br from this current source
                    let br_2d_plasma_this_j: Array2<f64> = g_br_plasma_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;
                    let bz_2d_plasma_this_j: Array2<f64> = g_bz_plasma_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;

                    // Add contribution from current
                    br_2d_plasma += &br_2d_plasma_this_j;
                    bz_2d_plasma += &bz_2d_plasma_this_j;
                }
            }
        }

        // Add up all the components
        let br_2d: Array2<f64> = &br_2d_coils + &br_2d_passives + &br_2d_plasma;
        let bz_2d: Array2<f64> = &bz_2d_coils + &bz_2d_passives + &bz_2d_plasma;

        // Store in self
        self.br_2d = br_2d.clone();
        self.bz_2d = bz_2d.clone();

        // d_br_d_z and d_bz_d_z
        // Coils;  timing: 1.9ms, with [n_r, n_z]=[100,201]
        let mut d_br_d_z_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut d_bz_d_z_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_pf in 0..n_pf {
            d_br_d_z_2d_coils = d_br_d_z_2d_coils + &g_d_br_d_z_coils.slice(s![.., .., i_pf]) * pf_currents[i_pf];
            d_bz_d_z_2d_coils = d_bz_d_z_2d_coils + &g_d_bz_d_z_coils.slice(s![.., .., i_pf]) * pf_currents[i_pf];
        }

        // Passives;  timing: 4ms, with [n_r, n_z]=[100,201]
        let passives_shape: &[usize] = g_d_br_d_z_passives.shape(); // shape = (n_z * n_r, n_passive_dof)
        let n_passive_dof: usize = passives_shape[1];

        let mut d_br_d_z_2d_passives: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut d_bz_d_z_2d_passives: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_passive_dof in 0..n_passive_dof {
            // d_br_d_z
            let d_br_d_z_2d_passives_this_slice: Array2<f64> = g_d_br_d_z_passives
                .slice(s![.., i_passive_dof])
                .to_owned()
                .to_shape((n_z, n_r))
                .expect("variable: d_br_d_z_2d_passives_this_slice;  probably wrong array dimensions")
                .to_owned()
                * passive_dof_values[i_passive_dof];
            d_br_d_z_2d_passives = d_br_d_z_2d_passives + d_br_d_z_2d_passives_this_slice;

            // d_bz_d_z
            let d_bz_d_z_2d_passives_this_slice: Array2<f64> = g_d_bz_d_z_passives
                .slice(s![.., i_passive_dof])
                .to_owned()
                .to_shape((n_z, n_r))
                .expect("variable: d_bz_d_z_2d_passives_this_slice;  probably wrong array dimensions")
                .to_owned()
                * passive_dof_values[i_passive_dof];
            d_bz_d_z_2d_passives = d_bz_d_z_2d_passives + d_bz_d_z_2d_passives_this_slice;
        }

        // Plasma d_br_d_z and d_bz_d_z;  timing: 921ms, with [n_r, n_z]=[100,201]
        let (g_d_br_d_z_plasma_flat, _): (Vec<f64>, Option<usize>) = g_d_br_d_z_plasma.into_raw_vec_and_offset(); // tested with d_bz_d_z and can confirm that it is working...
        let (g_d_bz_d_z_plasma_flat, _): (Vec<f64>, Option<usize>) = g_d_bz_d_z_plasma.into_raw_vec_and_offset();
        let g_d_br_d_z_plasma_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_d_br_d_z_plasma_flat).expect("Failed to reshape into Array3");
        let g_d_bz_d_z_plasma_3d: Array3<f64> = Array3::from_shape_vec((n_z, n_r, n_r), g_d_bz_d_z_plasma_flat).expect("Failed to reshape into Array3");

        let mut d_br_d_z_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));
        let mut d_bz_d_z_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));

        // Conceptually, we are looping over the current's and modifying the Green's table for the current
        for i_cur_z in 0..n_z {
            // Cyclic indexing for the z-axis (a current filament "looks" the same in z, but not r)
            let z_indexer: Vec<usize> = (0..n_z).map(|i_z| i_cur_z.abs_diff(i_z)).collect();

            // d_br_d_z it is up/down symmetric; d_bz_d_z is up/down asymmetric
            let sign: Vec<f64> = (0..n_z).map(|i_z| if i_z <= i_cur_z { -1.0 } else { 1.0 }).collect();
            let sign_ndarray: Array1<f64> = Array1::from(sign);
            let sign_broadcast: Array2<f64> = sign_ndarray.insert_axis(Axis(1)); // shape = [n_z, 1] (column-vector)

            for i_cur_r in 0..n_r {
                // Performance improvement: a lot of the grid doesn't have plasma current
                if j_2d[(i_cur_z, i_cur_r)].abs() > 0.0 {
                    // Select and reorder the Green's table for the radial current source location
                    let g_d_br_d_z_plasma_2d_reordered: Array2<f64> = g_d_br_d_z_plasma_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer);
                    let g_d_bz_d_z_plasma_2d_reordered: Array2<f64> =
                        g_d_bz_d_z_plasma_3d.index_axis(Axis(2), i_cur_r).select(Axis(0), &z_indexer) * &sign_broadcast;

                    // Calculate the contribution to d_br_d_z from this current source
                    let d_br_d_z_2d_plasma_this_j: Array2<f64> = g_d_br_d_z_plasma_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;
                    let d_bz_d_z_2d_plasma_this_j: Array2<f64> = g_d_bz_d_z_plasma_2d_reordered * j_2d[(i_cur_z, i_cur_r)] * d_area;

                    // Add contribution from current
                    d_br_d_z_2d_plasma += &d_br_d_z_2d_plasma_this_j;
                    d_bz_d_z_2d_plasma += &d_bz_d_z_2d_plasma_this_j;
                }
            }
        }

        // TODO: this is a numerical fudge. The reason is that there is a jump in the derivate of d(br)/d(z) at the filament location.
        // if you imagine the field above and below the filament of interst you see that bz=0, so d(bz)/d(z)=0. However, br changes sign
        // above and below the filament of interest. When you take the difference (derivative) you see d(br)/d(z) is not zero!
        // Jump condition at the filament!!
        // Numerically differentiate br_2d_plasma with respect to z
        let mut d_br_d_z_2d_plasma: Array2<f64> = Array2::zeros((n_z, n_r));
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();
        let d_z: f64 = z[1] - z[0];
        for i_r in 0..n_r {
            for i_z in 1..(n_z - 1) {
                d_br_d_z_2d_plasma[[i_z, i_r]] = (br_2d_plasma[[i_z + 1, i_r]] - br_2d_plasma[[i_z - 1, i_r]]) / (2.0 * d_z);
            }
            // Handle boundary conditions with forward and backward differences
            d_br_d_z_2d_plasma[[0, i_r]] = (br_2d_plasma[[1, i_r]] - br_2d_plasma[[0, i_r]]) / d_z;
            d_br_d_z_2d_plasma[[n_z - 1, i_r]] = (br_2d_plasma[[n_z - 1, i_r]] - br_2d_plasma[[n_z - 2, i_r]]) / d_z;
        }

        // SOMETHING LIKE THIS. BUT THIS IS WRONG!!
        // for i_r in 0..n_r {
        //     for i_z in 0..n_z {
        //         let jump_value: f64 = MU_0 * j_2d[[i_z, i_r]] * d_area / (2.0 * r[i_r]);
        //         println!("jump_value={}", jump_value);
        //         d_br_d_z_2d_plasma[[i_z, i_r]] = d_br_d_z_2d_plasma[[i_z, i_r]] + MU_0 * j_2d[[i_z, i_r]] * d_area / (2.0 * r[i_r]);
        //     }
        // }

        // Add up all the components
        let d_br_d_z_2d: Array2<f64> = &d_br_d_z_2d_coils + &d_br_d_z_2d_passives + &d_br_d_z_2d_plasma;
        let d_bz_d_z_2d: Array2<f64> = &d_bz_d_z_2d_coils + &d_bz_d_z_2d_passives + &d_bz_d_z_2d_plasma;

        return (d_br_d_z_2d, d_bz_d_z_2d);
    }

    pub fn initialise_plasma_with_point_source_current(&mut self) {
        // Unpack objects
        let plasma: &Plasma = &self.plasma;
        let coils_dynamic: &SensorsDynamic = &self.coils_dynamic;

        // TODO: Need to fix this!!
        let ip_guess: f64 = 425.0e3;

        // Extract stuff from Plasma
        let d_area: f64 = plasma.results.get("grid").get("d_area").unwrap_f64();
        let greens_pf_grid: Array3<f64> = plasma.results.get("greens").get("pf").get("*").get("psi").unwrap_array3();

        // Extract stuff from Coils
        let pf_currents: Array1<f64> = coils_dynamic.measured.to_owned();

        let (n_z, n_r, n_pf): (usize, usize, usize) = greens_pf_grid.dim();

        let mut psi_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_pf in 0..n_pf {
            psi_2d_coils = psi_2d_coils + &greens_pf_grid.slice(s![.., .., i_pf]) * pf_currents[i_pf];
        }

        // Initial plasma (single filament)
        let mut j_2d: Array2<f64> = Array2::zeros((n_z, n_r));

        // Find where to initialise the plasma
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let r_target: f64 = 0.45;
        let mut i_r_centre: usize = 0;
        let mut smallest_diff: f64 = f64::MAX;
        for (i, &value) in r.iter().enumerate() {
            let diff: f64 = (value - r_target).abs();
            if diff < smallest_diff {
                smallest_diff = diff;
                i_r_centre = i;
            }
        }

        // Initialise `i_z_centre` to be centre of Z grid, assuming this is Z=0
        let i_z_centre: usize = (n_z as f64 / 2.0).floor() as usize;
        j_2d[(i_z_centre, i_r_centre)] = 3.0;

        j_2d[(i_z_centre - 1, i_r_centre)] = 2.0;
        j_2d[(i_z_centre + 1, i_r_centre)] = 2.0;
        j_2d[(i_z_centre, i_r_centre - 1)] = 2.0;
        j_2d[(i_z_centre, i_r_centre + 1)] = 2.0;

        j_2d[(i_z_centre + 2, i_r_centre)] = 1.0;
        j_2d[(i_z_centre - 2, i_r_centre)] = 1.0;
        j_2d[(i_z_centre + 1, i_r_centre + 1)] = 1.0;
        j_2d[(i_z_centre - 1, i_r_centre + 1)] = 1.0;
        j_2d[(i_z_centre, i_r_centre + 2)] = 1.0;
        j_2d[(i_z_centre + 1, i_r_centre - 1)] = 1.0;
        j_2d[(i_z_centre - 1, i_r_centre - 1)] = 1.0;
        j_2d[(i_z_centre, i_r_centre - 2)] = 1.0;

        j_2d = j_2d * ip_guess / d_area / 19.0;

        // Store in self
        self.j_2d = j_2d;
        self.psi_2d_coils = psi_2d_coils;
        self.r_mag = r[i_r_centre]; // `r_mag` is not really correct; but as good as we can do for the initial guess
        self.z_mag = 0.0;

        self.calculate_psi();
    }

    /// Calculate the Grad-Shafranov "error"
    /// In the Picard iteration we change the solution by the error,
    /// so what we are doing here is checking to see how much the solutions
    /// is changing by
    fn calculate_gs_error(&mut self, psi_a_previous: f64) {
        // Get stuff out of self
        let psi_a: f64 = self.psi_a;
        let psi_b: f64 = self.psi_b;

        // Calculate the "error", in the same way EFIT does (called `cerror`)
        // Note, while this might "look" like a convergence test, it is in fact very similar
        // to a residule, since at each iteration the solution changes by the residule
        let gs_error_calculated: f64 = (psi_a - psi_a_previous).abs() / (psi_b - psi_a).abs();

        self.gs_error_calculated = gs_error_calculated;
    }

    /// Calculate the Grad Shafranov error by calcuating the LHS and RHS
    /// on the 2D (r, z) grid and seeing the difference = LHS - RHS.
    /// **This function is only used for development**
    fn _calculate_gs_error_numerical(&mut self) {
        // get stuff out of self
        let plasma: &Plasma = self.plasma;
        let psi_2d: Array2<f64> = self.psi_2d.to_owned();
        let r: Array1<f64> = plasma.results.get("grid").get("r").unwrap_array1();
        let z: Array1<f64> = plasma.results.get("grid").get("z").unwrap_array1();

        // Define some variables
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let n_r: usize = r.len();
        let n_z: usize = z.len();

        // Laplacian(psi)
        let mut laplacian_psi: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 1..(n_r - 1) {
            for i_z in 1..(n_z - 1) {
                let d2_psi_dz2: f64 = (psi_2d[(i_z + 1, i_r)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z - 1, i_r)]) / (d_z * d_z);
                let d2_psi_dr2: f64 = (psi_2d[(i_z, i_r + 1)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z, i_r - 1)]) / (d_r * d_r);
                let r_d_psi_dr: f64 = (1.0 / r[i_r]) * (psi_2d[(i_z, i_r + 1)] - psi_2d[(i_z, i_r - 1)]) / (2.0 * d_r);

                laplacian_psi[(i_z, i_r)] = d2_psi_dr2 - r_d_psi_dr + d2_psi_dz2;
            }
        }
        let mask: Array2<f64> = self.mask.to_owned();
        laplacian_psi = laplacian_psi * mask;

        // RHS of Grad-Shafranov equation
        // Eq. 3 in "Tokamak equilibrium reconstruction code LIUQE and its real time implementation", 2015
        let j_2d: Array2<f64> = self.j_2d.to_owned();
        let mut gs_rhs: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 0..n_r {
            let tmp: Array1<f64> = -2.0 * PI * MU_0 * r[i_r] * j_2d.slice(s![.., i_r]).to_owned();
            gs_rhs.slice_mut(s![.., i_r]).assign(&tmp);
        }

        // // TEMPORARY printing
        // let lhs: f64 = laplacian_psi[(50, 25)];
        // let rhs: f64 = gs_rhs[(50, 25)];
        // println!("lhs={lhs}, rhs={rhs}");
        // let tmp_r: f64 = r[25];
        // let tmp_z: f64 = z[50];
        // println!("r={tmp_r}, z={tmp_z}");
        // // write `laplacian_psi` to file
        // let file = File::create("gs_lhs.txt").expect("can't make file");
        // let mut writer = BufWriter::new(file);
        // for row in laplacian_psi.rows() {
        //     let line: String = row.iter()
        //         .map(|&value| value.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ");
        //     writeln!(writer, "{}", line).expect("can't write line");
        // }
        // writer.flush().expect("can't flush writer");
        // // write `gs_rhs` to file
        // let file = File::create("gs_rhs.txt").expect("can't make file");
        // let mut writer = BufWriter::new(file);
        // for row in gs_rhs.rows() {
        //     let line: String = row.iter()
        //         .map(|&value| value.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ");
        //     writeln!(writer, "{}", line).expect("can't write line");
        // }
        // writer.flush().expect("can't flush writer");

        // Calculate the residual
        // Note - there is high residual at the boundary
        // Perhaps we should make the mask larger??
        let residual_2d: Array2<f64> = laplacian_psi - gs_rhs;
        println!("{:?}", residual_2d);
    }
}
