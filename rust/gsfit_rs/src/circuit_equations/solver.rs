use crate::greens::{Greens, mutual_inductance_finite_size_to_finite_size};
use crate::material_properties::copper_resistivity;
use crate::pf_active::PfActive;
use crate::pf_passive::{PassiveFilaments, PfPassive, passive_filaments};
use diffsol::{NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod, OdeSolverStopReason, Vector, VectorHost};
use faer::linalg::solvers::Solve;
use imas_rs::ids::pf_active::{PfCoils, SignalFlt1d};
use imas_rs::ids::pf_passive::PfPassiveLoops;
use ndarray::{Array1, Array2, Axis, concatenate, s};
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;
use std::f64::consts::PI;

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
    pf_current_prescribed_current_interpolators: Vec<interpolation::Dim1Linear>, // I_PF_I
    pf_current_prescribed_current_derivative_interpolators: Vec<interpolation::Dim1Linear>, // d(I_PF_I)/d(t)
    n_current_prescribed: usize,
    n_voltage_prescribed: usize,
    n_passive_filaments: usize,
    n_plasma: usize,
    n_states: usize,
    state_identifiers: Vec<StateIdentifier>,
}

impl CircuitEquationModel {
    /// Constructor for CircuitEquationModel
    fn new(pf_active: &PfActive, pf_passive: &PfPassive) -> Self {
        // Find which PF coils have their current prescribed and which their voltage, by name and by position in `pf_active/coil`
        let pf_active_coils: &Vec<PfCoils> = &pf_active.pf_active_ids.coil;
        let n_pf: usize = pf_active_coils.len();
        let mut pf_current_prescribed_names: Vec<String> = Vec::new();
        let mut pf_current_prescribed_indices: Vec<usize> = Vec::new();
        let mut pf_voltage_prescribed_names: Vec<String> = Vec::new();
        let mut pf_voltage_prescribed_indices: Vec<usize> = Vec::new();
        for i_pf in 0..n_pf {
            let prescribed: &str = &pf_active_coils[i_pf].prescribed.name;
            if prescribed == "current" {
                pf_current_prescribed_names.push(pf_active_coils[i_pf].name.clone());
                pf_current_prescribed_indices.push(i_pf);
            } else if prescribed == "voltage" {
                pf_voltage_prescribed_names.push(pf_active_coils[i_pf].name.clone());
                pf_voltage_prescribed_indices.push(i_pf);
            } else {
                panic!("Unknown PF coil prescription, `pf_active/coil({i_pf})/prescribed/name` = {prescribed}");
            }
        }
        let n_current_prescribed: usize = pf_current_prescribed_names.len();
        let n_voltage_prescribed: usize = pf_voltage_prescribed_names.len();

        // Collect I_PF_I (current prescribed PF coils)
        let mut pf_current_prescribed_current_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_prescribed);
        for pf_name in &pf_current_prescribed_names {
            let times: Array1<f64> = pf_active.pf_active_ids.coil(pf_name).current.time.clone();
            let currents: Array1<f64> = pf_active.pf_active_ids.coil(pf_name).current.data.clone();

            // Construct and store the interpolator, I_PF_I
            let current_interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(times.clone(), currents.clone()).expect("Failed to create interpolator for: I_PF_I");
            pf_current_prescribed_current_interpolators.push(current_interpolator);
        }

        // Collect d(I_PF_I)/d(t) (current prescribed PF coils)
        let mut pf_current_prescribed_current_derivative_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_prescribed);
        for pf_name in &pf_current_prescribed_names {
            let pf_exp_time: Array1<f64> = pf_active.pf_active_ids.coil(pf_name).current.time.clone();
            let pf_exp_current: Array1<f64> = pf_active.pf_active_ids.coil(pf_name).current.data.clone();

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
            pf_current_prescribed_current_derivative_interpolators.push(current_derivatives_interpolator);
        }

        // Every passive filament, in the order of `pf_passive/loop` and then of `loop(i1)/element`, described as the Greens functions
        // describe them, with the resistivity of its passive conductor
        let pf_passive_loops: &Vec<PfPassiveLoops> = &pf_passive.pf_passive_ids.r#loop;
        let n_loop: usize = pf_passive_loops.len();
        let mut passive_r_vec: Vec<f64> = Vec::new();
        let mut passive_z_vec: Vec<f64> = Vec::new();
        let mut passive_d_r_vec: Vec<f64> = Vec::new();
        let mut passive_d_z_vec: Vec<f64> = Vec::new();
        let mut passive_angle_1_vec: Vec<f64> = Vec::new();
        let mut passive_angle_2_vec: Vec<f64> = Vec::new();
        let mut passive_resistivity_vec: Vec<f64> = Vec::new();
        for i_loop in 0..n_loop {
            let filaments: PassiveFilaments = passive_filaments(&pf_passive_loops[i_loop].element);
            passive_r_vec.extend(filaments.r.iter());
            passive_z_vec.extend(filaments.z.iter());
            passive_d_r_vec.extend(filaments.d_r.iter());
            passive_d_z_vec.extend(filaments.d_z.iter());
            passive_angle_1_vec.extend(filaments.angle_1.iter());
            passive_angle_2_vec.extend(filaments.angle_2.iter());
            let n_element: usize = pf_passive_loops[i_loop].element.len();
            for _i_element in 0..n_element {
                passive_resistivity_vec.push(pf_passive_loops[i_loop].resistivity);
            }
        }
        let passive_r: Array1<f64> = Array1::from_vec(passive_r_vec);
        let passive_z: Array1<f64> = Array1::from_vec(passive_z_vec);
        let passive_d_r: Array1<f64> = Array1::from_vec(passive_d_r_vec);
        let passive_d_z: Array1<f64> = Array1::from_vec(passive_d_z_vec);
        let passive_angle_1: Array1<f64> = Array1::from_vec(passive_angle_1_vec);
        let passive_angle_2: Array1<f64> = Array1::from_vec(passive_angle_2_vec);
        let passive_resistivity: Array1<f64> = Array1::from_vec(passive_resistivity_vec);
        let n_passive_filaments: usize = passive_r.len();

        // Find if we are solving for the plasma current state or not
        let n_plasma: usize = 0; // TODO: temporarily **NOT** including plasma

        // Store the variable names we are solving for
        let n_variables_solving: usize = n_voltage_prescribed + n_passive_filaments + n_current_prescribed + n_plasma;
        let mut state_identifiers: Vec<StateIdentifier> = Vec::with_capacity(n_variables_solving);
        for pf_name in &pf_voltage_prescribed_names {
            state_identifiers.push(StateIdentifier {
                current_source_type: CurrentSourceType::PF,
                circuit_name: pf_name.to_string(),
                passive_filament_index: None,
            });
        }
        for i_loop in 0..n_loop {
            let n_element: usize = pf_passive_loops[i_loop].element.len();
            for i_element in 0..n_element {
                state_identifiers.push(StateIdentifier {
                    current_source_type: CurrentSourceType::Passive,
                    circuit_name: pf_passive_loops[i_loop].name.clone(),
                    passive_filament_index: Some(i_element),
                });
            }
        }
        for pf_name in &pf_current_prescribed_names {
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
        let mut mutual_11: Array2<f64> = Array2::from_elem((n_current_prescribed, n_current_prescribed), f64::NAN);
        let mut mutual_12: Array2<f64> = Array2::from_elem((n_current_prescribed, n_voltage_prescribed), f64::NAN);
        let mut mutual_13: Array2<f64> = Array2::from_elem((n_current_prescribed, n_passive_filaments), f64::NAN);
        let mut mutual_14: Array2<f64> = Array2::from_elem((n_current_prescribed, n_plasma), f64::NAN); // plasma

        let mut mutual_21: Array2<f64> = Array2::from_elem((n_voltage_prescribed, n_current_prescribed), f64::NAN);
        let mut mutual_22: Array2<f64> = Array2::from_elem((n_voltage_prescribed, n_voltage_prescribed), f64::NAN);
        let mut mutual_23: Array2<f64> = Array2::from_elem((n_voltage_prescribed, n_passive_filaments), f64::NAN);
        let mut mutual_24: Array2<f64> = Array2::from_elem((n_voltage_prescribed, n_plasma), f64::NAN); // plasma

        let mutual_31: Array2<f64>; // symmetric with mutual_13; shape = (n_passive_filaments, n_current_prescribed)
        let mutual_32: Array2<f64>; // symmetric with mutual_23; shape = (n_passive_filaments, n_voltage_prescribed)
        let mutual_33: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
            &passive_r,
            &passive_z,
            &passive_d_r,
            &passive_d_z,
            &passive_angle_1,
            &passive_angle_2,
            &passive_r,
            &passive_z,
            &passive_d_r,
            &passive_d_z,
            &passive_angle_1,
            &passive_angle_2,
        ); // shape = (n_passive_filaments, n_passive_filaments)
        let mut mutual_34: Array2<f64> = Array2::from_elem((n_passive_filaments, n_plasma), f64::NAN); // plasma

        let mut mutual_41: Array2<f64> = Array2::from_elem((n_plasma, n_current_prescribed), f64::NAN);
        let mut mutual_42: Array2<f64> = Array2::from_elem((n_plasma, n_voltage_prescribed), f64::NAN);
        let mut mutual_43: Array2<f64> = Array2::from_elem((n_plasma, n_passive_filaments), f64::NAN);
        let mut mutual_44: Array2<f64> = Array2::from_elem((n_plasma, n_plasma), f64::NAN); // plasma

        let mut res_1: Array2<f64> = Array2::zeros((n_current_prescribed, n_current_prescribed)); // PF_I
        let mut res_2: Array2<f64> = Array2::zeros((n_voltage_prescribed, n_voltage_prescribed)); // PF_V
        let mut res_3: Array2<f64> = Array2::zeros((n_passive_filaments, n_passive_filaments)); // passives
        let mut res_4: Array2<f64> = Array2::zeros((n_plasma, n_plasma)); // plasma

        // Mutual inductance between every pair of PF coils, [henry]: the flux at the filaments of one coil, per ampere in the
        // other's filaments, summed over both. It is the method of `coil(i1)/inductance`, which the diagonal repeats
        let mut mutual_pf: Array2<f64> = Array2::from_elem((n_pf, n_pf), f64::NAN);
        for i_pf in 0..n_pf {
            let coil_r: Array1<f64> = pf_active_coils[i_pf].element(..).geometry.rectangle.r.to_array();
            let coil_z: Array1<f64> = pf_active_coils[i_pf].element(..).geometry.rectangle.z.to_array();
            for j_pf in 0..n_pf {
                let greens_calculator: Greens = Greens::sensor_to_conductor(
                    coil_r.clone(),
                    coil_z.clone(),
                    pf_active_coils[j_pf].element(..).geometry.rectangle.r.to_array(),
                    pf_active_coils[j_pf].element(..).geometry.rectangle.z.to_array(),
                    pf_active_coils[j_pf].element(..).geometry.rectangle.width.to_array(),
                    pf_active_coils[j_pf].element(..).geometry.rectangle.height.to_array(),
                );
                mutual_pf[[i_pf, j_pf]] = greens_calculator.psi().sum();
            }
        }

        // Resistance of each PF coil, [ohm]: its filaments in series, each a copper loop at 20 degrees C
        let resistivity_copper_20c: f64 = copper_resistivity(293.15); // ~ 1.68e-8 ohm * metre
        let mut resistance_pf: Array1<f64> = Array1::from_elem(n_pf, f64::NAN);
        for i_pf in 0..n_pf {
            let area: Array1<f64> =
                pf_active_coils[i_pf].element(..).geometry.rectangle.width.to_array() * pf_active_coils[i_pf].element(..).geometry.rectangle.height.to_array();
            let length: Array1<f64> = 2.0 * PI * pf_active_coils[i_pf].element(..).geometry.rectangle.r.to_array();
            resistance_pf[i_pf] = (resistivity_copper_20c * length / area).sum();
        }

        // Loop over current prescribed PF coils (vertical index)
        for i_pf_current_prescribed in 0..n_current_prescribed {
            let i_pf: usize = pf_current_prescribed_indices[i_pf_current_prescribed];

            // Loop over current prescribed PF coils (horizontal index)
            for j_pf_current_prescribed in 0..n_current_prescribed {
                mutual_11[[i_pf_current_prescribed, j_pf_current_prescribed]] = mutual_pf[[i_pf, pf_current_prescribed_indices[j_pf_current_prescribed]]];
            }

            // Loop over voltage prescribed PF coils (horizontal index)
            for j_pf_voltage_prescribed in 0..n_voltage_prescribed {
                mutual_12[[i_pf_current_prescribed, j_pf_voltage_prescribed]] = mutual_pf[[i_pf, pf_voltage_prescribed_indices[j_pf_voltage_prescribed]]];
            }

            // Resistance matrix
            res_1[(i_pf_current_prescribed, i_pf_current_prescribed)] = resistance_pf[i_pf];
        }

        // Loop over voltage prescribed PF coils
        for i_pf_voltage_prescribed in 0..n_voltage_prescribed {
            let i_pf: usize = pf_voltage_prescribed_indices[i_pf_voltage_prescribed];

            // Loop over current prescribed PF coils
            for j_pf_current_prescribed in 0..n_current_prescribed {
                mutual_21[[i_pf_voltage_prescribed, j_pf_current_prescribed]] = mutual_pf[[i_pf, pf_current_prescribed_indices[j_pf_current_prescribed]]];
            }

            // Loop over voltage prescribed PF coils
            for j_pf_voltage_prescribed in 0..n_voltage_prescribed {
                mutual_22[[i_pf_voltage_prescribed, j_pf_voltage_prescribed]] = mutual_pf[[i_pf, pf_voltage_prescribed_indices[j_pf_voltage_prescribed]]];
            }

            // Resistance matrix
            res_2[(i_pf_voltage_prescribed, i_pf_voltage_prescribed)] = resistance_pf[i_pf];
        }

        // Mutual inductance between I_PF_I and passives
        // mutual_13.shape = (n_current_prescribed, n_passive_filaments)
        for i_current_prescribed in 0..n_current_prescribed {
            // Get the coil geometry
            let coil: &PfCoils = pf_active.pf_active_ids.coil(&pf_current_prescribed_names[i_current_prescribed]);
            let coil_r: Array1<f64> = coil.element(..).geometry.rectangle.r.to_array();
            let coil_z: Array1<f64> = coil.element(..).geometry.rectangle.z.to_array();
            let coil_d_r: Array1<f64> = coil.element(..).geometry.rectangle.width.to_array();
            let coil_d_z: Array1<f64> = coil.element(..).geometry.rectangle.height.to_array();
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
            mutual_13.slice_mut(s![i_current_prescribed, ..]).assign(&g_coil_passive_filaments);
        }
        // mutual_31 is the transpose of mutual_13
        mutual_31 = mutual_13.t().to_owned();

        // Mutual inductance between I_PF_V and passives
        // mutual_23.shape = (n_voltage_prescribed, n_passive_filaments)
        for i_voltage_prescribed in 0..n_voltage_prescribed {
            // Get the coil geometry
            let coil: &PfCoils = pf_active.pf_active_ids.coil(&pf_voltage_prescribed_names[i_voltage_prescribed]);
            let coil_r: Array1<f64> = coil.element(..).geometry.rectangle.r.to_array();
            let coil_z: Array1<f64> = coil.element(..).geometry.rectangle.z.to_array();
            let coil_d_r: Array1<f64> = coil.element(..).geometry.rectangle.width.to_array();
            let coil_d_z: Array1<f64> = coil.element(..).geometry.rectangle.height.to_array();
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
            mutual_23.slice_mut(s![i_voltage_prescribed, ..]).assign(&g_coil_passive_filaments);
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
                concatenate(Axis(1), &[(-1.0 * &mutual_12).view(),   (-1.0 * &mutual_13).view(),   Array2::eye(n_current_prescribed).view(),                       Array2::zeros((n_current_prescribed, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_22).view(),   (-1.0 * &mutual_23).view(),   Array2::zeros((n_voltage_prescribed, n_current_prescribed)).view(), Array2::zeros((n_voltage_prescribed, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_32).view(),   (-1.0 * &mutual_33).view(),   Array2::zeros((n_passive_filaments, n_current_prescribed)).view(),       Array2::zeros((n_passive_filaments, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_42).view(),   (-1.0 * &mutual_43).view(),   Array2::zeros((n_plasma, n_current_prescribed)).view(),         Array2::eye(n_plasma).view()]).unwrap().view(),
            ]
        ).unwrap();

        // circuit_equation_matrix_2 = "stiffness matrix"
        #[rustfmt::skip]
        let circuit_equation_matrix_2: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[Array2::zeros((n_current_prescribed, n_voltage_prescribed)).view(),   Array2::zeros((n_current_prescribed, n_passive_filaments)).view(),       Array2::zeros((n_current_prescribed, n_current_prescribed)).view(),       Array2::zeros((n_current_prescribed, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[res_2.view(),                                                 Array2::zeros((n_voltage_prescribed, n_passive_filaments)).view(),       Array2::zeros((n_voltage_prescribed, n_current_prescribed)).view(),       Array2::zeros((n_voltage_prescribed, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passive_filaments, n_voltage_prescribed)).view(),         res_3.view(),                                               Array2::zeros((n_passive_filaments, n_current_prescribed)).view(),             Array2::zeros((n_passive_filaments, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_voltage_prescribed)).view(),           Array2::zeros((n_plasma, n_passive_filaments)).view(),               Array2::zeros((n_plasma, n_current_prescribed)).view(),               Array2::zeros((n_plasma, n_plasma)).view()]).unwrap().view(),
            ]
        ).unwrap();

        // circuit_equation_matrix_3 = "source matrix"
        #[rustfmt::skip]
        let circuit_equation_matrix_3: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[res_1.view(),                                               Array2::zeros((n_current_prescribed, n_voltage_prescribed)).view(),     Array2::zeros((n_current_prescribed, n_plasma)).view(),       mutual_11.view(),   mutual_14.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_voltage_prescribed, n_current_prescribed)).view(), (-1.0f64 * Array2::eye(n_voltage_prescribed)).view(),               Array2::zeros((n_voltage_prescribed, n_plasma)).view(),       mutual_21.view(),   mutual_24.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passive_filaments, n_current_prescribed)).view(),       Array2::zeros((n_passive_filaments, n_voltage_prescribed)).view(),           Array2::zeros((n_passive_filaments, n_plasma)).view(),             mutual_31.view(),   mutual_34.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_current_prescribed)).view(),         Array2::zeros((n_plasma, n_voltage_prescribed)).view(),             res_4.view(),                                             mutual_41.view(),   mutual_44.view()]).unwrap().view(),
            ]
        ).unwrap();

        // In state space notation, `A` = "state matrix"
        // We could compute it directly by taking the inverse, but solving the linear system is more stable.
        //
        // We use faer's partial pivoting LU decomposition plus 3 iterations of iterative refinement:
        //   1. Solve M1 * X0 = RHS using LU
        //   2. Compute residual r = RHS - M1 * X0
        //   3. Solve M1 * correction = r using the same LU factors
        //   4. X = X0 + correction
        // This typically recovers near-full precision for moderate condition numbers (~1e7).
        let n_rows: usize = circuit_equation_matrix_1.nrows();
        let n_cols_1: usize = circuit_equation_matrix_1.ncols();
        let n_cols_2: usize = circuit_equation_matrix_2.ncols();
        let n_cols_3: usize = circuit_equation_matrix_3.ncols();

        // Convert to faer and compute LU decomposition once
        let matrix_1_faer: faer::Mat<f64> = faer::Mat::from_fn(n_rows, n_cols_1, |i, j| circuit_equation_matrix_1[[i, j]]);
        let rhs_a_faer: faer::Mat<f64> = faer::Mat::from_fn(n_rows, n_cols_2, |i, j| circuit_equation_matrix_2[[i, j]]);
        let rhs_b_faer: faer::Mat<f64> = faer::Mat::from_fn(n_rows, n_cols_3, |i, j| circuit_equation_matrix_3[[i, j]]);
        let lu: faer::linalg::solvers::PartialPivLu<f64> = matrix_1_faer.partial_piv_lu();

        // Solve for state_space_matrix_a: circuit_equation_matrix_1 * X = circuit_equation_matrix_2
        let mut solution_a: faer::Mat<f64> = lu.solve(&rhs_a_faer);
        for _ in 0..3 {
            let residual: faer::Mat<f64> = &rhs_a_faer - &matrix_1_faer * &solution_a;
            let correction: faer::Mat<f64> = lu.solve(&residual);
            solution_a = &solution_a + &correction;
        }
        let state_space_matrix_a: Array2<f64> = Array2::from_shape_fn((n_rows, n_cols_2), |(i, j)| solution_a[(i, j)]);

        // Solve for state_space_matrix_b: circuit_equation_matrix_1 * X = circuit_equation_matrix_3
        let mut solution_b: faer::Mat<f64> = lu.solve(&rhs_b_faer);
        for _ in 0..3 {
            let residual: faer::Mat<f64> = &rhs_b_faer - &matrix_1_faer * &solution_b;
            let correction: faer::Mat<f64> = lu.solve(&residual);
            solution_b = &solution_b + &correction;
        }
        let state_space_matrix_b: Array2<f64> = Array2::from_shape_fn((n_rows, n_cols_3), |(i, j)| solution_b[(i, j)]);

        // TODO: by taking the eigenvalues of `state_space_matrix_a`, we can check if the system is stiff or not.
        // * If the eigenvalues have widely different magnitudes, then the "stiffness ratio" is large
        // * `stiffness_ratio = max(|Re(λ)|) / min(|Re(λ)|)`, if `stiffness_ratio >> 1` then the system is stiff

        CircuitEquationModel {
            n_current_prescribed,
            n_voltage_prescribed,
            n_passive_filaments,
            n_plasma,
            n_states: n_voltage_prescribed + n_passive_filaments + n_current_prescribed + n_plasma,
            state_space_matrix_a,
            state_space_matrix_b,
            pf_current_prescribed_current_interpolators,
            pf_current_prescribed_current_derivative_interpolators,
            state_identifiers,
        }
    }

    /// Compute the control input vector u(t) at the given time [various units]
    /// u = [I_PF_I; V_PF_V; Ip; d(I_PF_I)/d(t); d(Ip)/d(t)]
    fn compute_input_vector(&self, time_now: f64) -> Array1<f64> {
        // Get sizes
        let n_pf_current_prescribed: usize = self.n_current_prescribed;
        let n_pf_voltage_prescribed: usize = self.n_voltage_prescribed;
        let n_plasma: usize = self.n_plasma;

        // The control input vector u
        // `u = [I_PF_I; V_PF_V; Ip; d(I_PF_I)/d(t); d(Ip)/d(t)]`
        let n_u: usize = n_pf_current_prescribed + n_pf_voltage_prescribed + n_plasma + n_pf_current_prescribed + n_plasma;
        let mut u: Array1<f64> = Array1::from_elem(n_u, f64::NAN);

        // Add I_PF_I to `u` vector
        let mut pf_current_prescribed_values: Array1<f64> = Array1::from_elem(n_pf_current_prescribed, f64::NAN);
        for i_pf in 0..n_pf_current_prescribed {
            let pf_current_prescribed_interpolator: &interpolation::Dim1Linear = &self.pf_current_prescribed_current_interpolators[i_pf];

            // Interpolate to get current at present time [ampere]
            pf_current_prescribed_values[i_pf] = pf_current_prescribed_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current");
        }
        u.slice_mut(s![0..n_pf_current_prescribed]).assign(&pf_current_prescribed_values);

        // Add V_PF_V to `u` vector
        let pf_voltage_prescribed_v: Array1<f64> = Array1::from_elem(n_pf_voltage_prescribed, 0.0); // TODO: replace with actual values [volt]
        u.slice_mut(s![n_pf_current_prescribed..n_pf_current_prescribed + n_pf_voltage_prescribed])
            .assign(&pf_voltage_prescribed_v);

        // Add Ip to `u` vector
        // TODO: add plasma state

        // Add d(I_PF_I)/d(t) to `u` vector
        let mut pf_current_prescribed_derivative_values: Array1<f64> = Array1::from_elem(n_pf_current_prescribed, f64::NAN);
        for i_pf in 0..n_pf_current_prescribed {
            let pf_current_prescribed_derivative_interpolator: &interpolation::Dim1Linear = &self.pf_current_prescribed_current_derivative_interpolators[i_pf];

            // Interpolate to get derivative of current at present time [ampere / second]
            pf_current_prescribed_derivative_values[i_pf] = pf_current_prescribed_derivative_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current derivative");
        }
        u.slice_mut(s![n_pf_current_prescribed + n_pf_voltage_prescribed + n_plasma
            ..n_pf_current_prescribed + n_pf_voltage_prescribed + n_plasma + n_pf_current_prescribed])
            .assign(&pf_current_prescribed_derivative_values);

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

        initial_states_function
    }
}

/// Solves the circuit equations
///
/// Note: the adaptive time-stepping can produce a lot of simulated time-points,
/// so `adaptive_time_stepping = false` should be used to avoid unwieldly large outputs
///
/// # Arguments
/// * `pf_active` - The PF coils, each with its current or its voltage prescribed (`coil(i1)/prescribed`)
/// * `pf_passive` - The passive conductors, whose filaments, `loop(i1)/element(i2)`, carry eddy currents
/// * `times_to_solve` - Times to solve the circuit equations at, [second]
/// * `adaptive_time_stepping` - The calculation is always uses adaptive time stepping, this flat interpolates results onto `times_to_solve`
///
/// # Returns
/// - A copy of `pf_active`, whose `coil(i1)/current` is the simulated current, on the simulated timebase
/// - A copy of `pf_passive`, whose `loop(i1)/element(i2)/current` is the simulated current, on the simulated timebase, `loop(i1)/time`
///
#[pyfunction]
pub fn solve_circuit_equations(
    pf_active: PyRef<PfActive>,
    pf_passive: PyRef<PfPassive>,
    times_to_solve: PyReadonlyArray1<f64>,
    adaptive_time_stepping: bool,
) -> (PfActive, PfPassive) {
    let times_to_solve_ndarray: Array1<f64> = times_to_solve.to_owned_array();
    let time_start: f64 = times_to_solve_ndarray[0];
    let time_end: f64 = times_to_solve_ndarray.last().unwrap().to_owned();

    // Construct the circuit equation model
    let model: CircuitEquationModel = CircuitEquationModel::new(&pf_active, &pf_passive);

    // Solver tolerances [dimensionless]
    let rtol: f64 = 0.005; // Relative tolerance = 0.5% [dimensionless]

    // Absolute tolerance per state [multiple_units]
    // Different states have very different magnitudes:
    //   - Voltage-prescribed PF coil currents: typical scale ~10's of kA, so atol ~ 10 A is fine
    //   - Passive filament eddy currents. In totality the vessel currents can be large, but since we have
    //     discretized into many filaments, each filament will be ~500A, so atol ~ 0.5 A
    //   - d(Alpha)/d(time) = V_PF; checking Alpha ~6 volt * second, so atol ~0.06
    //   - Beta (plasma state): not yet implemented
    let mut atol: Vec<f64> = Vec::with_capacity(model.n_states);
    for state_identifier in &model.state_identifiers {
        match state_identifier.current_source_type {
            CurrentSourceType::PF => atol.push(100.0),    // voltage-prescribed PF coil current [ampere]
            CurrentSourceType::Passive => atol.push(0.5), // passive eddy current [ampere]
            CurrentSourceType::Alpha => atol.push(0.06),  // auxiliary state for current-prescribed PF [volt * second]
            CurrentSourceType::Beta => atol.push(10.0),   // auxiliary state for plasma [TODO: set appropriately]
        }
    }

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
        .atol(atol) // per-state absolute tolerances
        .t0(time_start) // initial time [second]
        .build()
        .expect("Failed to build ODE problem");

    // Create BDF solver with specified tolerances using dense nalgebra matrices
    type LS = diffsol::NalgebraLU<f64>;
    let mut solver = problem.bdf::<LS>().expect("Failed to create BDF solver");

    // Solve the ODE system
    let (calculated_states, calculated_times_vec, _ode_stop_reason): (NalgebraMat<f64>, Vec<f64>, OdeSolverStopReason<f64>) =
        solver.solve(time_end).expect("Failed to solve ODE system");
    println!("Integration completed");

    // Extract results
    let calculated_times: Array1<f64> = Array1::from_vec(calculated_times_vec.clone());
    let n_time: usize = calculated_times.len();

    // The simulated PF coils: a copy of `pf_active`, whose currents are replaced with the simulated ones below
    let mut pf_active_simulated: PfActive = PfActive {
        pf_active_ids: pf_active.pf_active_ids.clone(),
        parameters_input: pf_active.parameters_input.clone(),
    };
    let n_pf: usize = pf_active_simulated.pf_active_ids.coil.len();
    let simulated_time: Array1<f64> = if adaptive_time_stepping {
        calculated_times.clone()
    } else {
        times_to_solve_ndarray.clone()
    };

    // The simulated passive conductors: a copy of `pf_passive`, whose filaments are given the simulated currents below, on the
    // simulated timebase
    let mut pf_passive_simulated: PfPassive = PfPassive {
        pf_passive_ids: pf_passive.pf_passive_ids.clone(),
        parameters_input: pf_passive.parameters_input.clone(),
    };
    let n_loop: usize = pf_passive_simulated.pf_passive_ids.r#loop.len();
    for i_loop in 0..n_loop {
        pf_passive_simulated.pf_passive_ids.r#loop[i_loop].time = simulated_time.clone();
    }

    // Store results into the simulated PF coils and passive conductors
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

                // Store the simulated current in the simulated PF coils, on the calculated times, or on `times_to_solve` when the
                // results have been interpolated onto them
                for i_pf in 0..n_pf {
                    if pf_active_simulated.pf_active_ids.coil[i_pf].name == *circuit_name {
                        pf_active_simulated.pf_active_ids.coil[i_pf].current = SignalFlt1d {
                            data: current_simulated.clone(),
                            time: simulated_time.clone(),
                        };
                    }
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

                // Store the simulated current in the simulated passive conductor's filament, on `loop(i1)/time`, which is the calculated
                // times, or `times_to_solve` when the results have been interpolated onto them
                let i_element: usize = state_identifier.passive_filament_index.unwrap();
                for i_loop in 0..n_loop {
                    if pf_passive_simulated.pf_passive_ids.r#loop[i_loop].name == *circuit_name {
                        pf_passive_simulated.pf_passive_ids.r#loop[i_loop].element[i_element].current = current_simulated.clone();
                    }
                }
            }
            CurrentSourceType::Alpha => {
                // This is for current-prescribed PF coils
                let circuit_name: &String = &state_identifier.circuit_name;

                // Calculate the "simulated" current
                // Note, the current is actually imposed, but we still want it on the simulation time-base
                let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
                for i_time in 0..n_time {
                    current_simulated[i_time] = model.pf_current_prescribed_current_interpolators[index_pf_current_counter]
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

                // Store the simulated current in the simulated PF coils, on the calculated times, or on `times_to_solve` when the
                // results have been interpolated onto them
                for i_pf in 0..n_pf {
                    if pf_active_simulated.pf_active_ids.coil[i_pf].name == *circuit_name {
                        pf_active_simulated.pf_active_ids.coil[i_pf].current = SignalFlt1d {
                            data: current_simulated.clone(),
                            time: simulated_time.clone(),
                        };
                    }
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

    (pf_active_simulated, pf_passive_simulated)
}

// IDEAS FOR TESTING:
// * If we have some PF coils voltage prescribed = 0V; this will be the same as passives.
// * Control a PF coil using current, then switch to voltage prescribed using voltage from current prescribed case.
