use crate::coils::Coils;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use crate::passives::PassiveGeometryAll;
use crate::passives::Passives;
use crate::sensors::{BpProbes, FluxLoops, RogowskiCoils};
use diffsol::{NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod, Vector, VectorHost};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use ndarray_linalg::Solve;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;

// Choice of ODE solver:
//
// * Explicit methods (e.g. Runge-Kutta):
//   `y(t + d_t) = y(t) + d_t * y_prime(t)`
//
// * Implicit methods (e.g. BDF):
//   `y(t + d_t) = y(t) + d_t * y_prime(t + d_t)`
//
// The important difference is that implicit methods have y(t + d_t) on the RHS, which have to be solved iteratively.
//
// Explicit methods are generally simpler and faster. But implicit methods are more stable for stiff ODEs.
//
// The BDF (Backward Differentiation Formula) method switches between BDF1, BDF2, BDF3, BDF4, and BDF5 depending on stiffness.
// It is **very** complicated to implement BDF, so we use the excellent `diffsol` crate.

const PI: f64 = std::f64::consts::PI;

#[derive(Clone, PartialEq)]
enum CurrentSourceType {
    PF,
    Alpha,
    Beta,
    Passive,
}

#[derive(Clone)]
struct StateIdentifier {
    current_source_type: CurrentSourceType,
    circuit_name: String,                  // e.g., "SOL", "BVLT", "IVC", "plasma"
    passive_filament_index: Option<usize>, // only used for passive filaments
}

#[derive(Clone)]
struct CircuitEquationModel {
    state_space_matrix_a: Array2<f64>,
    state_space_matrix_b: Array2<f64>,
    pf_current_controlled_current_interpolators: Vec<interpolation::Dim1Linear>, // I_PF_I
    pf_current_controlled_current_derivative_interpolators: Vec<interpolation::Dim1Linear>, // d(I_PF_I)/d(t)
    n_current_controlled: usize,
    n_voltage_controlled: usize,
    n_passive_filaments: usize,
    n_plasma: usize,
    n_states: usize,
    state_identifiers: Vec<StateIdentifier>,
}

impl CircuitEquationModel {
    /// Constructor for CircuitEquationModel
    fn new(coils: Coils, passives: Passives) -> Self {
        // Find which PF coils are current controlled and which are voltage controlled
        let pf_names: Vec<String> = coils.results.get("pf").keys();
        let mut pf_current_controlled_names: Vec<String> = Vec::new();
        let mut pf_voltage_controlled_names: Vec<String> = Vec::new();
        for pf_name in pf_names {
            let controlled_by: String = coils.results.get("pf").get(&pf_name).get("controlled_by").unwrap_string();
            if controlled_by == "current" {
                pf_current_controlled_names.push(pf_name.to_owned());
            } else if controlled_by == "voltage" {
                pf_voltage_controlled_names.push(pf_name.to_owned());
            } else {
                panic!("Unknown PF coil control type, controlled_by = {controlled_by}");
            }
        }
        let n_current_controlled: usize = pf_current_controlled_names.len();
        let n_voltage_controlled: usize = pf_voltage_controlled_names.len();

        // Collect I_PF_I (current controlled PF coils)
        let mut pf_current_controlled_current_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_controlled);
        for pf_name in &pf_current_controlled_names {
            let times: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("experimental").get("time").unwrap_array1();
            let currents: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("experimental").get("value").unwrap_array1();

            // Construct and store the interpolator, I_PF_I
            let current_interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(times.clone(), currents.clone()).expect("Failed to create interpolator for: I_PF_I");
            pf_current_controlled_current_interpolators.push(current_interpolator);
        }

        // Collect d(I_PF_I)/d(t) (current controlled PF coils)
        let mut pf_current_controlled_current_derivative_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_controlled);
        for pf_name in &pf_current_controlled_names {
            let pf_exp_time: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("experimental").get("time").unwrap_array1();
            let pf_exp_current: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("experimental").get("value").unwrap_array1();

            let n_time: usize = pf_exp_time.len();
            let mut pf_exp_current_derivatives: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            // Users will send in non-uniform time, when designing a waveform for the PF coils
            // Forward difference for first point
            if n_time > 1 {
                pf_exp_current_derivatives[0] = (pf_exp_current[1] - pf_exp_current[0]) / (pf_exp_time[1] - pf_exp_time[0]);
            }
            // Central difference for interior points
            for i_time in 1..n_time - 1 {
                pf_exp_current_derivatives[i_time] =
                    (pf_exp_current[i_time + 1] - pf_exp_current[i_time - 1]) / (pf_exp_time[i_time + 1] - pf_exp_time[i_time - 1]);
            }
            // Backward difference for last point
            if n_time > 1 {
                let last: usize = n_time - 1;
                pf_exp_current_derivatives[last] = (pf_exp_current[last] - pf_exp_current[last - 1]) / (pf_exp_time[last] - pf_exp_time[last - 1]);
            }

            // Construct and store the interpolator, d(I_PF_I)/d(t)
            let current_derivatives_interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(pf_exp_time.clone(), pf_exp_current_derivatives.clone())
                    .expect("Failed to create interpolator for: d(I_PF_I)/d(t)");
            pf_current_controlled_current_derivative_interpolators.push(current_derivatives_interpolator);
        }

        // Get the number of passive filaments, note the data structure with each passive containing a different number of filaments makes this a bit tricky hence the method
        let n_passive_filaments: usize = passives.get_n_passive_filaments();

        // Find if we are solving for the plasma current state or not
        let n_plasma: usize = 0; // TODO: temporarily **NOT** including plasma

        // Store the variable names we are solving for
        let n_variables_solving: usize = n_voltage_controlled + n_passive_filaments + n_current_controlled + n_plasma;
        let mut state_identifiers: Vec<StateIdentifier> = Vec::with_capacity(n_variables_solving);
        for pf_name in &pf_voltage_controlled_names {
            state_identifiers.push(StateIdentifier {
                current_source_type: CurrentSourceType::PF,
                circuit_name: pf_name.to_string(),
                passive_filament_index: None,
            });
        }
        for passive_name in &passives.results.keys() {
            let n_filaments: usize = passives.results.get(passive_name).get("geometry").get("r").unwrap_array1().len();
            for i_filament in 0..n_filaments {
                state_identifiers.push(StateIdentifier {
                    current_source_type: CurrentSourceType::Passive,
                    circuit_name: passive_name.to_string(),
                    passive_filament_index: Some(i_filament),
                });
            }
        }
        for pf_name in &pf_current_controlled_names {
            state_identifiers.push(StateIdentifier {
                current_source_type: CurrentSourceType::Alpha,
                circuit_name: pf_name.to_string(),
                passive_filament_index: None,
            });
        }
        for _ in 0..n_plasma {
            // will only add if plasma is included
            state_identifiers.push(StateIdentifier {
                current_source_type: CurrentSourceType::Beta,
                circuit_name: "plasma".to_string(),
                passive_filament_index: None,
            });
        }

        // Matrix dimensions are: (vertical, horizontal)
        let mut mutual_11: Array2<f64> = Array2::from_elem((n_current_controlled, n_current_controlled), f64::NAN);
        let mut mutual_12: Array2<f64> = Array2::from_elem((n_current_controlled, n_voltage_controlled), f64::NAN);
        let mut mutual_13: Array2<f64> = Array2::from_elem((n_current_controlled, n_passive_filaments), f64::NAN);
        let mut mutual_14: Array2<f64> = Array2::from_elem((n_current_controlled, n_plasma), f64::NAN); // plasma

        let mut mutual_21: Array2<f64> = Array2::from_elem((n_voltage_controlled, n_current_controlled), f64::NAN);
        let mut mutual_22: Array2<f64> = Array2::from_elem((n_voltage_controlled, n_voltage_controlled), f64::NAN);
        let mut mutual_23: Array2<f64> = Array2::from_elem((n_voltage_controlled, n_passive_filaments), f64::NAN);
        let mut mutual_24: Array2<f64> = Array2::from_elem((n_voltage_controlled, n_plasma), f64::NAN); // plasma

        let mutual_31: Array2<f64>; // symmetric with mutual_13; shape = (n_passive_filaments, n_current_controlled)
        let mutual_32: Array2<f64>; // symmetric with mutual_23; shape = (n_passive_filaments, n_voltage_controlled)
        let mutual_33: Array2<f64> = passives.greens_with_self(); // shape = (n_passive_filaments, n_passive_filaments)
        let mut mutual_34: Array2<f64> = Array2::from_elem((n_passive_filaments, n_plasma), f64::NAN); // plasma

        let mut mutual_41: Array2<f64> = Array2::from_elem((n_plasma, n_current_controlled), f64::NAN);
        let mut mutual_42: Array2<f64> = Array2::from_elem((n_plasma, n_voltage_controlled), f64::NAN);
        let mut mutual_43: Array2<f64> = Array2::from_elem((n_plasma, n_passive_filaments), f64::NAN);
        let mut mutual_44: Array2<f64> = Array2::from_elem((n_plasma, n_plasma), f64::NAN); // plasma

        let mut res_1: Array2<f64> = Array2::zeros((n_current_controlled, n_current_controlled)); // PF_I
        let mut res_2: Array2<f64> = Array2::zeros((n_voltage_controlled, n_voltage_controlled)); // PF_V
        let mut res_3: Array2<f64> = Array2::zeros((n_passive_filaments, n_passive_filaments)); // passives
        let mut res_4: Array2<f64> = Array2::zeros((n_plasma, n_plasma)); // plasma

        // Loop over current controlled PF coils (vertical index)
        for (i_pf_current_controlled, pf_current_controlled_name) in pf_current_controlled_names.iter().enumerate() {
            // Loop over current controlled PF coils (horizontal index)
            for (j_pf_current_controlled, pf_current_controlled_name_2) in pf_current_controlled_names.iter().enumerate() {
                mutual_11[[i_pf_current_controlled, j_pf_current_controlled]] = coils
                    .results
                    .get("pf")
                    .get(pf_current_controlled_name)
                    .get("greens")
                    .get(pf_current_controlled_name_2)
                    .unwrap_f64();
            }

            // Loop over voltage controlled PF coils (horizontal index)
            for (j_pf_voltage_controlled, pf_voltage_controlled_name) in pf_voltage_controlled_names.iter().enumerate() {
                mutual_12[[i_pf_current_controlled, j_pf_voltage_controlled]] = coils
                    .results
                    .get("pf")
                    .get(pf_current_controlled_name)
                    .get("greens")
                    .get(pf_voltage_controlled_name)
                    .unwrap_f64();
            }

            // Resistance matrix
            res_1[(i_pf_current_controlled, i_pf_current_controlled)] = coils.results.get("pf").get(pf_current_controlled_name).get("resistance").unwrap_f64();
        }

        // Loop over voltage controlled PF coils
        for (i_pf_voltage_controlled, pf_voltage_controlled_name) in pf_voltage_controlled_names.iter().enumerate() {
            // Loop over current controlled PF coils
            for (j_pf_current_controlled, pf_current_controlled_name) in pf_current_controlled_names.iter().enumerate() {
                mutual_21[[i_pf_voltage_controlled, j_pf_current_controlled]] = coils
                    .results
                    .get("pf")
                    .get(pf_voltage_controlled_name)
                    .get("greens")
                    .get(pf_current_controlled_name)
                    .unwrap_f64();
            }

            // Loop over voltage controlled PF coils
            for (j_pf_voltage_controlled, pf_voltage_controlled_name_2) in pf_voltage_controlled_names.iter().enumerate() {
                mutual_22[[i_pf_voltage_controlled, j_pf_voltage_controlled]] = coils
                    .results
                    .get("pf")
                    .get(pf_voltage_controlled_name)
                    .get("greens")
                    .get(pf_voltage_controlled_name_2)
                    .unwrap_f64();
            }

            // Resistance matrix
            res_2[(i_pf_voltage_controlled, i_pf_voltage_controlled)] = coils.results.get("pf").get(pf_voltage_controlled_name).get("resistance").unwrap_f64();
        }

        // Get the passive filament geometry
        let passive_geometry_all: PassiveGeometryAll = passives.get_all_passive_filament_geometry();
        let passive_r: Array1<f64> = passive_geometry_all.r;
        let passive_z: Array1<f64> = passive_geometry_all.z;
        let passive_d_r: Array1<f64> = passive_geometry_all.d_r;
        let passive_d_z: Array1<f64> = passive_geometry_all.d_z;
        let passive_angle_1: Array1<f64> = passive_geometry_all.angle_1;
        let passive_angle_2: Array1<f64> = passive_geometry_all.angle_2;
        let passive_resistivity: Array1<f64> = passive_geometry_all.resistivity;

        // Mutual inductance between I_PF_I and passives
        // mutual_13.shape = (n_current_controlled, n_passive_filaments)
        for i_current_controlled in 0..n_current_controlled {
            // Get the coil geometry
            let coil_name: &String = &pf_current_controlled_names[i_current_controlled];
            let coil_r: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("z").unwrap_array1();
            let coil_d_r: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("d_r").unwrap_array1();
            let coil_d_z: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("d_z").unwrap_array1();
            let n_coil_filaments: usize = coil_r.len();
            let coil_angle_1: Array1<f64> = Array1::from_elem(n_coil_filaments, 0.0);
            let coil_angle_2: Array1<f64> = Array1::from_elem(n_coil_filaments, 0.0);

            let g_coil_filaments_passive_filaments: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                &coil_r,
                &coil_z,
                &coil_d_r,
                &coil_d_z,
                &coil_angle_1,
                &coil_angle_2,
                &passive_r,
                &passive_z,
                &passive_d_r,
                &passive_d_z,
                &passive_angle_1,
                &passive_angle_2,
            ); // shape = [n_coil_filaments, n_passive_filaments]

            // Sum over all coil filaments
            let g_coil_passive_filaments = g_coil_filaments_passive_filaments.sum_axis(Axis(0)); // shape = [n_passive_filaments]

            // Store in mutual_13
            mutual_13.slice_mut(s![i_current_controlled, ..]).assign(&g_coil_passive_filaments);
        }
        // mutual_31 is the transpose of mutual_13
        mutual_31 = mutual_13.t().to_owned();

        // Mutual inductance between I_PF_V and passives
        // mutual_23.shape = (n_voltage_controlled, n_passive_filaments)
        for i_voltage_controlled in 0..n_voltage_controlled {
            // Get the coil geometry
            let coil_name: &String = &pf_voltage_controlled_names[i_voltage_controlled];
            let coil_r: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("r").unwrap_array1();
            let coil_z: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("z").unwrap_array1();
            let coil_d_r: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("d_r").unwrap_array1();
            let coil_d_z: Array1<f64> = coils.results.get("pf").get(coil_name).get("geometry").get("d_z").unwrap_array1();
            let n_coil_filaments: usize = coil_r.len();
            let coil_angle_1: Array1<f64> = Array1::from_elem(n_coil_filaments, 0.0);
            let coil_angle_2: Array1<f64> = Array1::from_elem(n_coil_filaments, 0.0);

            let g_coil_filaments_passive_filaments: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                &coil_r,
                &coil_z,
                &coil_d_r,
                &coil_d_z,
                &coil_angle_1,
                &coil_angle_2,
                &passive_r,
                &passive_z,
                &passive_d_r,
                &passive_d_z,
                &passive_angle_1,
                &passive_angle_2,
            ); // shape = [n_coil_filaments, n_passive_filaments]

            // Sum over all coil filaments
            let g_coil_passive_filaments = g_coil_filaments_passive_filaments.sum_axis(Axis(0)); // shape = [n_passive_filaments]

            // Store in mutual_23
            mutual_23.slice_mut(s![i_voltage_controlled, ..]).assign(&g_coil_passive_filaments);
        }
        // mutual_32 is the transpose of mutual_23
        mutual_32 = mutual_23.t().to_owned();

        // Passive resistance matrix
        for i_passive in 0..n_passive_filaments {
            let area: f64 = passive_d_r[i_passive] * passive_d_z[i_passive];
            let length: f64 = 2.0 * PI * passive_r[i_passive];

            res_3[(i_passive, i_passive)] = passive_resistivity[i_passive] * length / area;
        }

        // The circuit equation is:
        // mass_matrix * d(states)/d(t) = stiffness_matrix * states + source_matrix * u
        // circuit_equation_matrix_1 * d(states)/d(t) = circuit_equation_matrix_2 * states + circuit_equation_matrix_3 * u
        //
        // Later we will rearrange this to state-space form:
        // d(states)/d(t) = a_matrix * states + b_matrix * u

        // circuit_equation_matrix_1 = "mass matrix"
        #[rustfmt::skip]
        let circuit_equation_matrix_1: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[(-1.0 * &mutual_12).view(),   (-1.0 * &mutual_13).view(),   Array2::eye(n_current_controlled).view(),                       Array2::zeros((n_current_controlled, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_22).view(),   (-1.0 * &mutual_23).view(),   Array2::zeros((n_voltage_controlled, n_current_controlled)).view(), Array2::zeros((n_voltage_controlled, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_32).view(),   (-1.0 * &mutual_33).view(),   Array2::zeros((n_passive_filaments, n_current_controlled)).view(),       Array2::zeros((n_passive_filaments, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_42).view(),   (-1.0 * &mutual_43).view(),   Array2::zeros((n_plasma, n_current_controlled)).view(),         Array2::eye(n_plasma).view()]).unwrap().view(),
            ]
        ).unwrap();

        // circuit_equation_matrix_2 = "stiffness matrix"
        #[rustfmt::skip]
        let circuit_equation_matrix_2: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[Array2::zeros((n_current_controlled, n_voltage_controlled)).view(),   Array2::zeros((n_current_controlled, n_passive_filaments)).view(),       Array2::zeros((n_current_controlled, n_current_controlled)).view(),       Array2::zeros((n_current_controlled, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[res_2.view(),                                                 Array2::zeros((n_voltage_controlled, n_passive_filaments)).view(),       Array2::zeros((n_voltage_controlled, n_current_controlled)).view(),       Array2::zeros((n_voltage_controlled, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passive_filaments, n_voltage_controlled)).view(),         res_3.view(),                                               Array2::zeros((n_passive_filaments, n_current_controlled)).view(),             Array2::zeros((n_passive_filaments, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_voltage_controlled)).view(),           Array2::zeros((n_plasma, n_passive_filaments)).view(),               Array2::zeros((n_plasma, n_current_controlled)).view(),               Array2::zeros((n_plasma, n_plasma)).view()]).unwrap().view(),
            ]
        ).unwrap();

        // circuit_equation_matrix_3 = "source matrix"
        #[rustfmt::skip]
        let circuit_equation_matrix_3: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[res_1.view(),                                               Array2::zeros((n_current_controlled, n_voltage_controlled)).view(),     Array2::zeros((n_current_controlled, n_plasma)).view(),       mutual_11.view(),   mutual_14.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_voltage_controlled, n_current_controlled)).view(), (-1.0f64 * Array2::eye(n_voltage_controlled)).view(),               Array2::zeros((n_voltage_controlled, n_plasma)).view(),       mutual_21.view(),   mutual_24.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passive_filaments, n_current_controlled)).view(),       Array2::zeros((n_passive_filaments, n_voltage_controlled)).view(),           Array2::zeros((n_passive_filaments, n_plasma)).view(),             mutual_31.view(),   mutual_34.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_current_controlled)).view(),         Array2::zeros((n_plasma, n_voltage_controlled)).view(),             res_4.view(),                                             mutual_41.view(),   mutual_44.view()]).unwrap().view(),
            ]
        ).unwrap();

        // In state space notation, `A` = "state matrix"
        // We could compute it directly by taking the inverse:
        // `let state_space_matrix_a_old_method: Array2<f64> = circuit_equation_matrix_1.clone().inv().expect("circuit_equation_matrix_1 inversion failed").dot(&circuit_equation_matrix_2);`
        // However, this is numerically less stable.
        // It is better to solve the linear system for each column than taking the inverse:
        let n_cols: usize = circuit_equation_matrix_2.ncols();
        let n_rows: usize = circuit_equation_matrix_1.nrows();
        let mut state_space_matrix_a: Array2<f64> = Array2::from_elem((n_rows, n_cols), f64::NAN);
        for i_col in 0..n_cols {
            let col: Array1<f64> = circuit_equation_matrix_2.column(i_col).to_owned();
            let solution: Array1<f64> = circuit_equation_matrix_1
                .solve(&col)
                .expect(&format!("Failed to solve for `state_space_matrix_a` i_col={i_col}"));
            state_space_matrix_a.column_mut(i_col).assign(&solution);
        }

        // In "state space" notation, `B` = "input matrix"
        let n_cols: usize = circuit_equation_matrix_3.ncols();
        let mut state_space_matrix_b: Array2<f64> = Array2::from_elem((n_rows, n_cols), f64::NAN);
        for i_col in 0..n_cols {
            let col: Array1<f64> = circuit_equation_matrix_3.column(i_col).to_owned();
            let solution: Array1<f64> = circuit_equation_matrix_1
                .solve(&col)
                .expect(&format!("Failed to solve for `state_space_matrix_b` i_col={i_col}"));
            state_space_matrix_b.column_mut(i_col).assign(&solution);
        }

        // TODO: by taking the eigenvalues of `state_space_matrix_a`, we can check if the system is stiff or not.
        // * If the eigenvalues have widely different magnitudes, then the "stiffness ratio" is large
        // * `stiffness_ratio = max(|Re(λ)|) / min(|Re(λ)|)`, if `stiffness_ratio >> 1` then the system is stiff

        CircuitEquationModel {
            n_current_controlled,
            n_voltage_controlled,
            n_passive_filaments,
            n_plasma,
            n_states: n_voltage_controlled + n_passive_filaments + n_current_controlled + n_plasma,
            state_space_matrix_a,
            state_space_matrix_b,
            pf_current_controlled_current_interpolators,
            pf_current_controlled_current_derivative_interpolators,
            state_identifiers,
        }
    }

    /// Compute the control input vector u(t) at the given time [various units]
    /// u = [I_PF_I; V_PF_V; Ip; d(I_PF_I)/d(t); d(Ip)/d(t)]
    fn compute_input_vector(&self, time_now: f64) -> Array1<f64> {
        // Get sizes
        let n_pf_current_controlled: usize = self.n_current_controlled;
        let n_pf_voltage_controlled: usize = self.n_voltage_controlled;
        let n_plasma: usize = self.n_plasma;

        // The control input vector u
        // `u = [I_PF_I; V_PF_V; Ip; d(I_PF_I)/d(t); d(Ip)/d(t)]`
        let n_u: usize = n_pf_current_controlled + n_pf_voltage_controlled + n_plasma + n_pf_current_controlled + n_plasma;
        let mut u: Array1<f64> = Array1::from_elem(n_u, f64::NAN);

        // Add I_PF_I to `u` vector
        let mut pf_current_controlled_values: Array1<f64> = Array1::from_elem(n_pf_current_controlled, f64::NAN);
        for i_pf in 0..n_pf_current_controlled {
            let pf_current_controlled_interpolator: &interpolation::Dim1Linear = &self.pf_current_controlled_current_interpolators[i_pf];

            // Interpolate to get current at present time [ampere]
            pf_current_controlled_values[i_pf] = pf_current_controlled_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current");
        }
        u.slice_mut(s![0..n_pf_current_controlled]).assign(&pf_current_controlled_values);

        // Add V_PF_V to `u` vector
        let pf_voltage_controlled_v: Array1<f64> = Array1::from_elem(n_pf_voltage_controlled, 0.0); // TODO: replace with actual values [volt]
        u.slice_mut(s![n_pf_current_controlled..n_pf_current_controlled + n_pf_voltage_controlled])
            .assign(&pf_voltage_controlled_v);

        // Add Ip to `u` vector
        // TODO: add plasma state

        // Add d(I_PF_I)/d(t) to `u` vector
        let mut pf_current_controlled_derivative_values: Array1<f64> = Array1::from_elem(n_pf_current_controlled, f64::NAN);
        for i_pf in 0..n_pf_current_controlled {
            let pf_current_controlled_derivative_interpolator: &interpolation::Dim1Linear = &self.pf_current_controlled_current_derivative_interpolators[i_pf];

            // Interpolate to get derivative of current at present time [ampere / second]
            pf_current_controlled_derivative_values[i_pf] = pf_current_controlled_derivative_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current derivative");
        }
        u.slice_mut(s![n_pf_current_controlled + n_pf_voltage_controlled + n_plasma
            ..n_pf_current_controlled + n_pf_voltage_controlled + n_plasma + n_pf_current_controlled])
            .assign(&pf_current_controlled_derivative_values);

        // Add d(Ip)/d(t) to `u` vector
        // TODO: Add plasma state

        u
    }

    /// Returns a function which computes the right-hand side (RHS) of the ODE system:
    /// d(states)/dt = A*states + B*u(t)
    /// where A is the state matrix, B is the input matrix, and u(t) is the control input vector.
    ///
    /// This function is run inside `diffsol` during integration.
    fn rhs(&self) -> impl Fn(&NalgebraVec<f64>, &NalgebraVec<f64>, f64, &mut NalgebraVec<f64>) {
        let state_space_matrix_a: Array2<f64> = self.state_space_matrix_a.clone();
        let state_space_matrix_b: Array2<f64> = self.state_space_matrix_b.clone();
        let model: CircuitEquationModel = self.clone();

        // Note: `diffsol` has defined the function argumens for `rhs`. But we need to pass in additional data, specifically the state-space matrices.
        // The easiest way to do this is to use `move` "closures", which produce a function with the correct arguments.
        move |states: &NalgebraVec<f64>, _ode_params: &NalgebraVec<f64>, time_now: f64, d_states_d_t: &mut NalgebraVec<f64>| {
            // Compute control input u(t)
            let u: Array1<f64> = model.compute_input_vector(time_now);

            // Convert DVector to Array1
            let states_ndarray: Array1<f64> = Array1::from_vec(states.as_slice().to_vec());

            // Calculate d(states)/d(t) = A*states + B*u [ampere / second]
            let d_states_d_t_ndarray: Array1<f64> = state_space_matrix_a.dot(&states_ndarray) + state_space_matrix_b.dot(&u);

            // Convert back to DVector
            for (i, &value) in d_states_d_t_ndarray.iter().enumerate() {
                d_states_d_t[i] = value;
            }
        }
    }

    /// Returns a function which computes the Jacobian-vector product for the ODE system.
    /// The Jacobian-vector product is: jac_v = A * v
    ///
    /// This function is run inside `diffsol` during implicit integration steps.
    fn rhs_jacobian(&self) -> impl Fn(&NalgebraVec<f64>, &NalgebraVec<f64>, f64, &NalgebraVec<f64>, &mut NalgebraVec<f64>) {
        let state_space_matrix_a: Array2<f64> = self.state_space_matrix_a.clone();

        move |_states: &NalgebraVec<f64>, _ode_params: &NalgebraVec<f64>, _time_now: f64, v: &NalgebraVec<f64>, jac_v: &mut NalgebraVec<f64>| {
            // Convert input vector to ndarray
            let v_ndarray: Array1<f64> = Array1::from_vec(v.as_slice().to_vec());

            // Compute Jacobian-vector product: jac_v = A * v
            let jac_v_ndarray: Array1<f64> = state_space_matrix_a.dot(&v_ndarray);

            // Copy result back to output vector
            for (i, &value) in jac_v_ndarray.iter().enumerate() {
                jac_v[i] = value;
            }
        }
    }

    /// Returns a function, which produces the `initial_states`.
    /// This function is run inside `diffsol`
    fn initial_states(&self) -> impl Fn(&NalgebraVec<f64>, f64, &mut NalgebraVec<f64>) {
        /// Initialize the state vector at the start of integration.
        ///
        /// Present implementation fills the initial state vector with zeros.
        ///
        /// # Arguments
        /// * `_ode_params` - ODE parameters, shape = (0), [unknown]
        /// * `_time_initial` - The initial time, [second]
        /// * `initial_states` - Initial state vector to fill, shape=(n_states), [ampere]
        ///
        fn initial_states_function(_ode_params: &NalgebraVec<f64>, _time_initial: f64, initial_states: &mut NalgebraVec<f64>) {
            initial_states.fill(0.0);
        }

        return initial_states_function;
    }
}

/// Solves the circuit equations
///
/// Note: the adaptive time-stepping can produce a lot of simulated time-points,
/// so `adaptive_time_stepping = false` should be used to avoid unwieldly large outputs
///
/// # Arguments
/// * `coils` - Coils data structure containing either current or voltage waveforms
/// * `passives` - Passives
/// * `times_to_solve` - Times to solve the circuit equations at, [second]
/// * `adaptive_time_stepping` - The calculation is always uses adaptive time stepping, this flat interpolates results onto `times_to_solve`
///
/// # Returns
/// - None, but mutates `coils` and `passives` to add the simulated current and voltage waveforms
///
#[pyfunction]
pub fn solve_circuit_equations(
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    times_to_solve: PyReadonlyArray1<f64>,
    adaptive_time_stepping: bool,
) {
    let times_to_solve_ndarray: Array1<f64> = times_to_solve.to_owned_array();
    let time_start: f64 = times_to_solve_ndarray[0];
    let time_end: f64 = times_to_solve_ndarray.last().unwrap().to_owned();

    // Construct the circuit equation model
    let model: CircuitEquationModel = CircuitEquationModel::new(coils.clone(), passives.clone());

    // Solver tolerances
    let rtol: f64 = 1e-3; // Relative tolerance [dimensionless]
    // TODO: atol can be either a vector per state or a vector with a single value used for all states
    // I think this might be the reason why diffsol is slow - some states will have large value, some small value
    // So we might need to set atol differently for different states?
    let atol: f64 = 1e-2; // Absolute tolerance [multiple_units]

    // Build the ODE problem with both RHS and Jacobian
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .p(vec![]) // No parameters
        .rhs_implicit(
            model.rhs(),          // a function which computes the RHS of the ODE system
            model.rhs_jacobian(), // a function which computes the Jacobian-vector product of the ODE system
        )
        .init(
            model.initial_states(), // a function which initializes the state vector
            model.n_states,         // the number of states, usize
        )
        .rtol(rtol)
        .atol(vec![atol]) // single absolute tolerance for all states
        .t0(time_start) // initial time [second]
        .build()
        .expect("Failed to build ODE problem");

    // Create BDF solver with specified tolerances using dense nalgebra matrices
    type LS = diffsol::NalgebraLU<f64>;
    let mut solver = problem.bdf::<LS>().expect("Failed to create BDF solver");

    // Solve the ODE system
    let (calculated_states, calculated_times_vec): (NalgebraMat<f64>, Vec<f64>) = solver.solve(time_end).expect("Failed to solve ODE system");
    println!("Integration completed");

    // Extract results
    let calculated_times: Array1<f64> = Array1::from_vec(calculated_times_vec.clone());
    let n_time: usize = calculated_times.len();

    // Allocate passive currents
    for passive_name in passives.results.keys() {
        let n_filaments: usize = passives.results.get(&passive_name).get("geometry").get("r").unwrap_array1().len();
        let n_time_store: usize;
        if adaptive_time_stepping {
            n_time_store = n_time;
        } else {
            n_time_store = times_to_solve_ndarray.len();
        }
        let current_simulated: Array2<f64> = Array2::from_elem((n_time_store, n_filaments), f64::NAN);
        passives
            .results
            .get_or_insert(&passive_name)
            .get_or_insert("i_filaments")
            .get_or_insert("simulated")
            .insert("value", current_simulated);

        if adaptive_time_stepping {
            passives
                .results
                .get_or_insert(&passive_name)
                .get_or_insert("i_filaments")
                .get_or_insert("simulated")
                .insert("time", calculated_times.clone());
        } else {
            passives
                .results
                .get_or_insert(&passive_name)
                .get_or_insert("i_filaments")
                .get_or_insert("simulated")
                .insert("time", times_to_solve_ndarray.clone());
        }
    }

    // Store results into `Coils` and `Passives` structures
    let mut index_pf_current_counter: usize = 0;
    for (i_state, state_identifier) in model.state_identifiers.iter().enumerate() {
        match state_identifier.current_source_type {
            CurrentSourceType::PF => {
                let circuit_name: &String = &state_identifier.circuit_name;

                // Extract simulated current
                let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                for i_time in 0..n_time {
                    current_simulated[i_time] = calculated_states[(i_state, i_time)];
                }

                // Non-adaptive time-stepping means we need to interpolate results onto `times_to_solve`
                if !adaptive_time_stepping {
                    let current_simulated_interpolated: Array1<f64> = interpolation::Dim1Linear::new(calculated_times.clone(), current_simulated.clone())
                        .expect("Failed to create interpolator for non-adaptive time stepping")
                        .interpolate_array1(&times_to_solve_ndarray)
                        .expect("Failed to interpolate results onto `times_to_solve` for non-adaptive time-stepping");

                    // Overwrite `current_simulated` with the interpolated version
                    current_simulated = current_simulated_interpolated;
                }

                // Store simulated current back into Coils
                coils
                    .results
                    .get_or_insert("pf")
                    .get_or_insert(circuit_name)
                    .get_or_insert("i")
                    .get_or_insert("simulated")
                    .insert("value", current_simulated);

                if adaptive_time_stepping {
                    // Store the calculated times
                    coils
                        .results
                        .get_or_insert("pf")
                        .get_or_insert(circuit_name)
                        .get_or_insert("i")
                        .get_or_insert("simulated")
                        .insert("time", calculated_times.clone());
                } else {
                    // Store the original `times_to_solve` since we have interpolated results onto this time base
                    coils
                        .results
                        .get_or_insert("pf")
                        .get_or_insert(circuit_name)
                        .get_or_insert("i")
                        .get_or_insert("simulated")
                        .insert("time", times_to_solve_ndarray.clone());
                }

                // TODO: store voltage
            }
            CurrentSourceType::Passive => {
                let circuit_name: &String = &state_identifier.circuit_name;

                // Extract simulated current
                let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                for i_time in 0..n_time {
                    current_simulated[i_time] = calculated_states[(i_state, i_time)];
                }

                if !adaptive_time_stepping {
                    let current_simulated_interpolated = interpolation::Dim1Linear::new(calculated_times.clone(), current_simulated.clone())
                        .expect("Failed to create interpolator for non-adaptive time stepping")
                        .interpolate_array1(&times_to_solve_ndarray)
                        .expect("Failed to interpolate results onto `times_to_solve` for non-adaptive time-stepping");

                    // Overwrite `current_simulated` with the interpolated version
                    current_simulated = current_simulated_interpolated;
                }

                // TODO: this looks slow because we are taking data out, modifying a relatively small amount, and then putting it all back in.
                // Real world impact is actually quite small. Solving the ODE is still dominant.
                let mut simulated_currents: Array2<f64> = passives
                    .results
                    .get(circuit_name)
                    .get("i_filaments")
                    .get("simulated")
                    .get("value")
                    .unwrap_array2();
                simulated_currents
                    .slice_mut(s![.., state_identifier.passive_filament_index.unwrap()])
                    .assign(&current_simulated.to_owned());

                passives
                    .results
                    .get_or_insert(circuit_name)
                    .get_or_insert("i_filaments")
                    .get_or_insert("simulated")
                    .insert("value", simulated_currents);

                if adaptive_time_stepping {
                    // Store the calculated times
                    passives
                        .results
                        .get_or_insert(circuit_name)
                        .get_or_insert("i_filaments")
                        .get_or_insert("simulated")
                        .insert("time", calculated_times.clone());
                } else {
                    // Store the original `times_to_solve` since we have interpolated results onto this time base
                    passives
                        .results
                        .get_or_insert(circuit_name)
                        .get_or_insert("i_filaments")
                        .get_or_insert("simulated")
                        .insert("time", times_to_solve_ndarray.clone());
                }
            }
            CurrentSourceType::Alpha => {
                // This is for current-controlled PF coils
                let circuit_name: &String = &state_identifier.circuit_name;

                // Calculate the "simulated" current
                // Note, the current is actually imposed, but we still want it on the simulation time-base
                let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                for i_time in 0..n_time {
                    current_simulated[i_time] = model.pf_current_controlled_current_interpolators[index_pf_current_counter]
                        .interpolate_scalar(calculated_times[i_time])
                        .expect("Failed to interpolate PF current");
                }

                if !adaptive_time_stepping {
                    let current_simulated_interpolated: Array1<f64> = interpolation::Dim1Linear::new(calculated_times.clone(), current_simulated.clone())
                        .expect("Failed to create interpolator for non-adaptive time stepping")
                        .interpolate_array1(&times_to_solve_ndarray)
                        .expect("Failed to interpolate results onto `times_to_solve` for non-adaptive time-stepping");

                    // Overwrite `current_simulated` with the interpolated version
                    current_simulated = current_simulated_interpolated;
                }

                // Store simulated current into Coils
                coils
                    .results
                    .get_or_insert("pf")
                    .get_or_insert(circuit_name)
                    .get_or_insert("i")
                    .get_or_insert("simulated")
                    .insert("value", current_simulated);

                if adaptive_time_stepping {
                    // Store the calculated times
                    coils
                        .results
                        .get_or_insert("pf")
                        .get_or_insert(circuit_name)
                        .get_or_insert("i")
                        .get_or_insert("simulated")
                        .insert("time", calculated_times.clone());
                } else {
                    // Store the original `times_to_solve` since we have interpolated results onto this time base
                    coils
                        .results
                        .get_or_insert("pf")
                        .get_or_insert(circuit_name)
                        .get_or_insert("i")
                        .get_or_insert("simulated")
                        .insert("time", times_to_solve_ndarray.clone());
                }

                // TODO: calculate and store voltage

                // Increment counter
                index_pf_current_counter += 1;
            }
            CurrentSourceType::Beta => {
                let circuit_name: &String = &state_identifier.circuit_name;
                // println!("Storing results for Beta: {}", circuit_name);

                // TODO
            }
        }
    }
}

// IDEAS FOR TESTING:
// * If we have some PF coils voltage controlled = 0V; this will be the same as passives.
// * Control a PF coil using current, then switch to voltage controlled using voltage from current controlled case.
