use crate::coils::Coils;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use crate::passives::PassiveGeometryAll;
use crate::passives::Passives;
use crate::sensors::{BpProbes, FluxLoops, RogowskiCoils};
use dop_shared::OutputType;
use interpolation;
use ndarray::{Array1, Array2, s};
use ndarray::{Axis, concatenate};
use ndarray_linalg::Inverse;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use ode_solvers::*;
use pyo3::prelude::*;

const PI: f64 = std::f64::consts::PI;

#[derive(Clone)]
pub struct CircuitEquationModel {
    pub a_matrix: Array2<f64>,
    pub b_matrix: Array2<f64>,
    pub pf_current_driven_current_interpolators: Vec<interpolation::Dim1Linear>, // I_PF_I
    pub pf_current_driven_current_derivative_interpolators: Vec<interpolation::Dim1Linear>, // d(I_PF_I)/d(t)
    pub n_current_driven: usize,
    pub n_voltage_driven: usize,
    pub n_passives: usize,
    pub n_plasma: usize,
    pub state_names: Vec<(String, String)>, // data type, circuit name; e.g., ("pf", "SOL"), ("alpha", "BVLT"), etc.
}

impl CircuitEquationModel {
    /// Constructor for CircuitEquationModel
    pub fn new(coils: Coils, passives: Passives) -> Self {
        // Find which PF coils are current driven and which are voltage driven
        let pf_names: Vec<String> = coils.results.get("pf").keys();
        let mut pf_current_driven_names: Vec<String> = Vec::new();
        let mut pf_voltage_driven_names: Vec<String> = Vec::new();
        for pf_name in pf_names {
            let is_current_driven: String = coils
                .results
                .get("pf")
                .get(&pf_name)
                .get("driven_by")
                .unwrap_string();
            println!("pf_name={}, is_current_driven={}", pf_name, is_current_driven);
            let mut is_current_driven: bool = true; // TODO: replace with actual check
            // println!("pf_name={}", pf_name);
            if pf_name == "BVLT" || pf_name == "BVLB" || pf_name == "DIVT" {
                // TODO: some random coils are current driven to test code
                is_current_driven = false;
            }
            if is_current_driven {
                pf_current_driven_names.push(pf_name.to_owned());
            } else {
                pf_voltage_driven_names.push(pf_name.to_owned());
            }
        }
        let n_current_driven: usize = pf_current_driven_names.len();
        let n_voltage_driven: usize = pf_voltage_driven_names.len();

        // Collect I_PF_I (current driven PF coils)
        let mut pf_current_driven_current_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_driven);
        for pf_name in &pf_current_driven_names {
            let times: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("time_experimental").unwrap_array1();
            let currents: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("measured_experimental").unwrap_array1();

            // Construct and store the interpolator, I_PF_I
            let current_interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(times.clone(), currents.clone()).expect("Failed to create interpolator for: I_PF_I");
            pf_current_driven_current_interpolators.push(current_interpolator);
        }

        // Collect d(I_PF_I)/d(t) (current driven PF coils)
        let mut pf_current_driven_current_derivative_interpolators: Vec<interpolation::Dim1Linear> = Vec::with_capacity(n_current_driven);
        for pf_name in &pf_current_driven_names {
            let times: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("time_experimental").unwrap_array1();
            let currents: Array1<f64> = coils.results.get("pf").get(pf_name).get("i").get("measured_experimental").unwrap_array1();

            let mut current_derivatives: Array1<f64> = Array1::from_elem(currents.len(), f64::NAN);
            // Forward difference for first point
            if currents.len() > 1 {
                current_derivatives[0] = (currents[1] - currents[0]) / (times[1] - times[0]);
            }
            // Central difference for interior points
            for i in 1..currents.len() - 1 {
                current_derivatives[i] = (currents[i + 1] - currents[i - 1]) / (times[i + 1] - times[i - 1]);
            }
            // Backward difference for last point
            if currents.len() > 1 {
                let last = currents.len() - 1;
                current_derivatives[last] = (currents[last] - currents[last - 1]) / (times[last] - times[last - 1]);
            }

            // Construct and store the interpolator, d(I_PF_I)/d(t)
            let current_derivatives_interpolator: interpolation::Dim1Linear =
                interpolation::Dim1Linear::new(times.clone(), current_derivatives.clone()).expect("Failed to create interpolator for: d(I_PF_I)/d(t)");
            pf_current_driven_current_derivative_interpolators.push(current_derivatives_interpolator);
        }

        // let n_passives: usize = passive_locations_r.len();
        let n_passives: usize = passives.get_n_passive_filaments();
        // println!("n_passives={}", n_passives);

        // Find if we are solving for the plasma current state or not
        let n_plasma: usize = 0; // TODO: temporarily **NOT** including plasma

        // Store the variable names we are solving for
        let n_variables_solving: usize = n_voltage_driven + n_passives + n_current_driven + n_plasma;
        let mut state_names: Vec<(String, String)> = Vec::with_capacity(n_variables_solving);
        for pf_name in &pf_voltage_driven_names {
            state_names.push(("pf".to_string(), pf_name.to_string()));
        }
        for i_passive in 0..n_passives {
            state_names.push(("passive".to_string(), format!("{i_passive}")));
        }
        for pf_name in &pf_current_driven_names {
            state_names.push(("alpha".to_string(), pf_name.to_string()));
        }
        for _ in 0..n_plasma {
            // will only add if plasma is included
            state_names.push(("beta".to_string(), "plasma".to_string()));
        }

        // Matrix dimensions are: (vertical, horizontal)
        let mut mutual_11: Array2<f64> = Array2::from_elem((n_current_driven, n_current_driven), f64::NAN);
        let mut mutual_12: Array2<f64> = Array2::from_elem((n_current_driven, n_voltage_driven), f64::NAN);
        let mut mutual_13: Array2<f64> = Array2::from_elem((n_current_driven, n_passives), f64::NAN);
        let mut mutual_14: Array2<f64> = Array2::from_elem((n_current_driven, n_plasma), f64::NAN); // plasma

        let mut mutual_21: Array2<f64> = Array2::from_elem((n_voltage_driven, n_current_driven), f64::NAN);
        let mut mutual_22: Array2<f64> = Array2::from_elem((n_voltage_driven, n_voltage_driven), f64::NAN);
        let mut mutual_23: Array2<f64> = Array2::from_elem((n_voltage_driven, n_passives), f64::NAN);
        let mut mutual_24: Array2<f64> = Array2::from_elem((n_voltage_driven, n_plasma), f64::NAN); // plasma

        let mut mutual_31: Array2<f64> = Array2::from_elem((n_passives, n_current_driven), f64::NAN);
        let mut mutual_32: Array2<f64> = Array2::from_elem((n_passives, n_voltage_driven), f64::NAN);
        // let mut mutual_33: Array2<f64> = Array2::from_elem((n_passives, n_passives), f64::NAN);
        let mutual_33: Array2<f64> = passives.greens_with_self();
        let mut mutual_34: Array2<f64> = Array2::from_elem((n_passives, n_plasma), f64::NAN); // plasma

        let mut mutual_41: Array2<f64> = Array2::from_elem((n_plasma, n_current_driven), f64::NAN);
        let mut mutual_42: Array2<f64> = Array2::from_elem((n_plasma, n_voltage_driven), f64::NAN);
        let mut mutual_43: Array2<f64> = Array2::from_elem((n_plasma, n_passives), f64::NAN);
        let mut mutual_44: Array2<f64> = Array2::from_elem((n_plasma, n_plasma), f64::NAN); // plasma

        let mut res_1: Array2<f64> = Array2::zeros((n_current_driven, n_current_driven)); // I_PF_I
        let mut res_2: Array2<f64> = Array2::zeros((n_voltage_driven, n_voltage_driven)); // V_PF_V
        let mut res_3: Array2<f64> = Array2::zeros((n_passives, n_passives)); // passives
        let mut res_4: Array2<f64> = Array2::zeros((n_plasma, n_plasma)); // plasma

        // Loop over current driven PF coils (vertical index)
        for (i_pf_current_driven, pf_current_driven_name) in pf_current_driven_names.iter().enumerate() {
            // Loop over current driven PF coils (horizontal index)
            for (j_pf_current_driven, pf_current_driven_name_2) in pf_current_driven_names.iter().enumerate() {
                mutual_11[[i_pf_current_driven, j_pf_current_driven]] = coils
                    .results
                    .get("pf")
                    .get(pf_current_driven_name)
                    .get("greens")
                    .get(pf_current_driven_name_2)
                    .unwrap_f64();
            }

            // Loop over voltage driven PF coils (horizontal index)
            for (j_pf_voltage_driven, pf_voltage_driven_name) in pf_voltage_driven_names.iter().enumerate() {
                mutual_12[[i_pf_current_driven, j_pf_voltage_driven]] = coils
                    .results
                    .get("pf")
                    .get(pf_current_driven_name)
                    .get("greens")
                    .get(pf_voltage_driven_name)
                    .unwrap_f64();
            }

            // Resistance matrix
            res_1[(i_pf_current_driven, i_pf_current_driven)] = coils.results.get("pf").get(pf_current_driven_name).get("resistance").unwrap_f64();
        }

        // Loop over voltage driven PF coils
        for (i_pf_voltage_driven, pf_voltage_driven_name) in pf_voltage_driven_names.iter().enumerate() {
            // Loop over current driven PF coils
            for (j_pf_current_driven, pf_current_driven_name) in pf_current_driven_names.iter().enumerate() {
                mutual_21[[i_pf_voltage_driven, j_pf_current_driven]] = coils
                    .results
                    .get("pf")
                    .get(pf_voltage_driven_name)
                    .get("greens")
                    .get(pf_current_driven_name)
                    .unwrap_f64();
            }

            // Loop over voltage driven PF coils
            for (j_pf_voltage_driven, pf_voltage_driven_name_2) in pf_voltage_driven_names.iter().enumerate() {
                mutual_22[[i_pf_voltage_driven, j_pf_voltage_driven]] = coils
                    .results
                    .get("pf")
                    .get(pf_voltage_driven_name)
                    .get("greens")
                    .get(pf_voltage_driven_name_2)
                    .unwrap_f64();
            }

            // Resistance matrix
            res_2[(i_pf_voltage_driven, i_pf_voltage_driven)] = coils.results.get("pf").get(pf_voltage_driven_name).get("resistance").unwrap_f64();
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
        // mutual_13.shape = (n_current_driven, n_passives)
        for i_current_driven in 0..n_current_driven {
            // Get the coil geometry
            let coil_name: &String = &pf_current_driven_names[i_current_driven];
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
            ); // shape = [n_coil_filaments, n_passives]

            // Sum over all coil filaments
            let g_coil_passive_filaments = g_coil_filaments_passive_filaments.sum_axis(Axis(0)); // shape = [n_passives]

            // Store in mutual_13
            mutual_13.slice_mut(s![i_current_driven, ..]).assign(&g_coil_passive_filaments);
        }
        // mutual_31 is the transpose of mutual_13
        mutual_31 = mutual_13.t().to_owned();

        // Mutual inductance between I_PF_V and passives
        // mutual_23.shape = (n_voltage_driven, n_passives)
        for i_voltage_driven in 0..n_voltage_driven {
            // Get the coil geometry
            let coil_name: &String = &pf_voltage_driven_names[i_voltage_driven];
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
            ); // shape = [n_coil_filaments, n_passives]

            // Sum over all coil filaments
            let g_coil_passive_filaments = g_coil_filaments_passive_filaments.sum_axis(Axis(0)); // shape = [n_passives]

            // Store in mutual_23
            mutual_23.slice_mut(s![i_voltage_driven, ..]).assign(&g_coil_passive_filaments);
        }
        // mutual_32 is the transpose of mutual_23
        mutual_32 = mutual_23.t().to_owned();

        // Passive resistance matrix
        for i_passive in 0..n_passives {
            let area: f64 = passive_d_r[i_passive] * passive_d_z[i_passive];
            let length: f64 = 2.0 * PI * passive_r[i_passive];

            res_3[(i_passive, i_passive)] = passive_resistivity[i_passive] * length / area;
        }

        // println!("mutual_21 = {:?}", mutual_21);
        // println!("mutual_12 = {:?}", mutual_12);

        #[rustfmt::skip]
        let temp_1: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[(-1.0 * &mutual_12).view(),   (-1.0 * &mutual_13).view(),   Array2::eye(n_current_driven).view(),                       Array2::zeros((n_current_driven, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_22).view(),   (-1.0 * &mutual_23).view(),   Array2::zeros((n_voltage_driven, n_current_driven)).view(), Array2::zeros((n_voltage_driven, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_32).view(),   (-1.0 * &mutual_33).view(),   Array2::zeros((n_passives, n_current_driven)).view(),       Array2::zeros((n_passives, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[(-1.0 * &mutual_42).view(),   (-1.0 * &mutual_43).view(),   Array2::zeros((n_plasma, n_current_driven)).view(),         Array2::eye(n_plasma).view()]).unwrap().view(),
            ]
        ).unwrap();

        #[rustfmt::skip]
        let temp_2: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[Array2::zeros((n_current_driven, n_voltage_driven)).view(),   Array2::zeros((n_current_driven, n_passives)).view(),       Array2::zeros((n_current_driven, n_current_driven)).view(),       Array2::zeros((n_current_driven, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[res_2.view(),                                                 Array2::zeros((n_voltage_driven, n_passives)).view(),       Array2::zeros((n_voltage_driven, n_current_driven)).view(),       Array2::zeros((n_voltage_driven, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passives, n_voltage_driven)).view(),         res_3.view(),                                               Array2::zeros((n_passives, n_current_driven)).view(),             Array2::zeros((n_passives, n_plasma)).view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_voltage_driven)).view(),           Array2::zeros((n_plasma, n_passives)).view(),               Array2::zeros((n_plasma, n_current_driven)).view(),               Array2::zeros((n_plasma, n_plasma)).view()]).unwrap().view(),
            ]
        ).unwrap();

        #[rustfmt::skip]
        let temp_3: Array2<f64> = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[res_1.view(),                                               Array2::zeros((n_current_driven, n_voltage_driven)).view(),     Array2::zeros((n_current_driven, n_plasma)).view(),       mutual_11.view(),   mutual_14.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_voltage_driven, n_current_driven)).view(), (-1.0f64 * Array2::eye(n_voltage_driven)).view(),               Array2::zeros((n_voltage_driven, n_plasma)).view(),       mutual_21.view(),   mutual_24.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_passives, n_current_driven)).view(),       Array2::zeros((n_passives, n_voltage_driven)).view(),           Array2::zeros((n_passives, n_plasma)).view(),             mutual_31.view(),   mutual_34.view()]).unwrap().view(),
                concatenate(Axis(1), &[Array2::zeros((n_plasma, n_current_driven)).view(),         Array2::zeros((n_plasma, n_voltage_driven)).view(),             res_4.view(),                                             mutual_41.view(),   mutual_44.view()]).unwrap().view(),
            ]
        ).unwrap();

        let a_matrix: Array2<f64> = temp_1.clone().inv().expect("temp_1 inversion failed").dot(&temp_2);
        let b_matrix: Array2<f64> = temp_1.clone().inv().expect("temp_1 inversion failed").dot(&temp_3);

        // println!("state_names={:#?}", state_names);
        // println!("state_names.len()={:#?}", state_names.len());

        // println!("temp_1={:#?}", temp_1);
        // println!("temp_2={:#?}", temp_2);
        // println!("temp_3={:#?}", temp_3);
        // println!("a_matrix={:#?}", a_matrix);
        // println!("b_matrix={:#?}", b_matrix);

        CircuitEquationModel {
            n_current_driven,
            n_voltage_driven,
            n_passives,
            n_plasma,
            a_matrix,
            b_matrix,
            pf_current_driven_current_interpolators,
            pf_current_driven_current_derivative_interpolators,
            state_names,
        }
    }
}

impl ode_solvers::System<f64, DVector<f64>> for CircuitEquationModel {
    /// System of ordinary differential equations.
    /// Calculate the derivative of the state vector at the current time.
    /// Note, the function signature is defined by the `ode_solvers` crate.
    ///
    /// # Arguments
    /// - `time_now`: Current time.
    /// - `states`: Current state vector.
    /// - `d_states_d_t`: Derivative of state vector to be filled in. This is a mutable output argument.
    ///
    fn system(&self, time_now: f64, states: &DVector<f64>, d_states_d_t: &mut DVector<f64>) {
        println!("time_now = {}", time_now);
        // Get sizes
        let n_pf_current_driven: usize = self.n_current_driven;
        let n_pf_voltage_driven: usize = self.n_voltage_driven;
        let n_plasma: usize = self.n_plasma;

        // The "states" we are solving for are:
        // `u = [I_PF_I; V_PF_V; Ip; d(I_PF_I)/d(t); d(Ip)/d(t)]`
        let n_u: usize = n_pf_current_driven + n_pf_voltage_driven + n_plasma + n_pf_current_driven + n_plasma;
        let mut u: Array1<f64> = Array1::from_elem(n_u, f64::NAN);

        // Add I_PF_I to `u` states
        let mut pf_current_driven_values: Array1<f64> = Array1::from_elem(n_pf_current_driven, f64::NAN);
        for i_pf in 0..n_pf_current_driven {
            let pf_current_driven_interpolator: &interpolation::Dim1Linear = &self.pf_current_driven_current_interpolators[i_pf];

            // Interpolate to get current at present time
            pf_current_driven_values[i_pf] = pf_current_driven_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current");
        }
        u.slice_mut(s![0..n_pf_current_driven]).assign(&pf_current_driven_values);

        // Add V_PF_V to `u` states
        let pf_voltage_driven_v: Array1<f64> = Array1::from_elem(n_pf_voltage_driven, 0.0); // TODO: replace with actual values
        u.slice_mut(s![n_pf_current_driven..n_pf_current_driven + n_pf_voltage_driven])
            .assign(&pf_voltage_driven_v);

        // Add Ip to `u` states
        // TODO: add plasma state
        // println!("u.shape() = {:?}", u.shape());

        // Add d(I_PF_I)/d(t) to `u` states
        let mut pf_current_driven_derivative_values: Array1<f64> = Array1::from_elem(n_pf_current_driven, f64::NAN);
        for i_pf in 0..n_pf_current_driven {
            let pf_current_driven_derivative_interpolator: &interpolation::Dim1Linear = &self.pf_current_driven_current_derivative_interpolators[i_pf];

            // Interpolate to get derivative of current at present time
            pf_current_driven_derivative_values[i_pf] = pf_current_driven_derivative_interpolator
                .interpolate_scalar(time_now)
                .expect("Failed to interpolate PF current derivative");
        }
        u.slice_mut(s![
            n_pf_current_driven + n_pf_voltage_driven + n_plasma..n_pf_current_driven + n_pf_voltage_driven + n_plasma + n_pf_current_driven
        ])
        .assign(&pf_current_driven_derivative_values);

        // Add d(Ip)/d(t) to `u` states
        // TODO: Add plasma state

        // Convert DVector to Array1
        let states_ndarray: Array1<f64> = Array1::from_vec(states.as_slice().to_vec());
        // println!("states_ndarray.shape() = {:?}", states_ndarray.shape());

        // Calculate d(states)/d(t), what we need to return
        let d_states_d_t_ndarray: Array1<f64> = self.a_matrix.dot(&states_ndarray) + self.b_matrix.dot(&u);

        // Convert back to DVector, and modify the mutable variable
        let n_states: usize = d_states_d_t_ndarray.len();
        for i_state in 0..n_states {
            d_states_d_t[i_state] = d_states_d_t_ndarray[i_state];
        }

        // println!("time_now={}", time_now);
        // println!("states={:#?}", states);
        // println!("d_states_d_t={:#?}", d_states_d_t);
    }

    /// "Stop function" which is called at every successful integration step.
    /// The integration is stopped when this function returns true.
    fn solout(&mut self, _x: f64, _y: &DVector<f64>, _dy: &DVector<f64>) -> bool {
        false
    }
}

#[pyfunction]
/// Python wrapper for `solve_circuit_equations_rs`
pub fn solve_circuit_equations(
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    mut bp_probes: PyRefMut<BpProbes>,
    mut flux_loops: PyRefMut<FluxLoops>,
    mut rogowski_coils: PyRefMut<RogowskiCoils>,
    times_to_solve: PyReadonlyArray1<f64>,
) {
    // let coils_rs: Coils = coils.to_owned();
    let passives_rs: Passives = passives.to_owned();
    let bp_probes_rs: BpProbes = bp_probes.to_owned();
    let flux_loops_rs: FluxLoops = flux_loops.to_owned();
    let rogowski_coils_rs: RogowskiCoils = rogowski_coils.to_owned();
    let times_to_solve_ndarray: Array1<f64> = times_to_solve.to_owned_array();

    solve_circuit_equations_rs(coils, passives_rs, bp_probes_rs, flux_loops_rs, rogowski_coils_rs, times_to_solve_ndarray);
    // coils_rs.results.get_or_insert("pf").get_or_insert("testing").insert("data", 123.456);
}

fn solve_circuit_equations_rs(
    mut coils: PyRefMut<Coils>,
    mut passives: Passives,
    mut bp_probes: BpProbes,
    mut flux_loops: FluxLoops,
    mut rogowski_coils: RogowskiCoils,
    times_to_solve: Array1<f64>,
) {
    let n_time: usize = times_to_solve.len();

    let coil_names: Vec<String> = coils.results.get("pf").keys();
    let n_coils: usize = coil_names.len();

    // Find which PF coils are current driven and which are voltage driven
    let pf_names: Vec<String> = coils.results.get("pf").keys();
    let mut pf_current_driven_names: Vec<String> = Vec::new();
    let mut pf_voltage_driven_names: Vec<String> = Vec::new();
    for pf_name in pf_names {
        // let is_current_driven: bool = coils
        //     .results
        //     .get("pf")
        //     .get(&pf_name)
        //     .get("is_current_driven")
        //     .unwrap_bool();
        let mut is_current_driven: bool = true; // TODO: replace with actual check
        // println!("pf_name={}", pf_name);
        if pf_name == "SOL" || pf_name == "BVLT" || pf_name == "BVLB" || pf_name == "DIVT" {
            // TODO: give us a challenge
            is_current_driven = false;
        }
        if is_current_driven {
            pf_current_driven_names.push(pf_name.to_owned());
        } else {
            pf_voltage_driven_names.push(pf_name.to_owned());
        }
    }

    let n_current_driven: usize = pf_current_driven_names.len();
    let n_voltage_driven: usize = pf_voltage_driven_names.len();
    let n_passives: usize = passives.get_n_passive_filaments();

    // println!("pf_current_driven_names: {:?}", pf_current_driven_names);
    // println!("pf_voltage_driven_names: {:?}", pf_voltage_driven_names);

    let model: CircuitEquationModel = CircuitEquationModel::new(coils.to_owned(), passives.to_owned());

    // println!("here 01");
    let time_start: f64 = times_to_solve[0];
    let time_end: f64 = times_to_solve.last().unwrap().to_owned();
    let d_time: f64 = times_to_solve[1] - times_to_solve[0];

    // Solver tolerances - looser tolerances can help with stiffness detection
    // Try: 1e-3 to 1e-6 for tighter control, or 1e-1 to 1.0 for looser control
    let rtol: f64 = 1e-3; // Relative tolerance
    let atol: f64 = 1e-3; // Absolute tolerance

    // println!("here 02");
    let n_plasma: usize = 0; // TODO: need to add plasma

    // `states = [I_PF_V; I_passives; alpha; beta]`
    let n_states: usize = n_voltage_driven + n_passives + n_current_driven + n_plasma;
    let initial_states: DVector<f64> = DVector::zeros(n_states);
    // println!("here 03");

    // Configure Dopri5 with advanced parameters to handle stiffness better
    // Use from_param to set all configuration options
    let safety_factor: f64 = 0.9; // Safety factor for step size control
    let beta: f64 = 0.04; // Beta coefficient for PI controller
    let fac_min: f64 = 0.2; // Min factor between successive steps
    let fac_max: f64 = 10.0; // Max factor between successive steps
    let h_max: f64 = time_end - time_start; // Maximum step size (full range)
    let h_initial: f64 = 0.0; // Let solver auto-compute initial step (0.0 = auto)
    let n_max: u32 = 500000; // Max number of steps (increased from default 100000)
    let n_stiff: u32 = 100000; // Check stiffness every n_stiff steps (increased from default 1000)

    let mut stepper = Dopri5::from_param(
        model.clone(),
        time_start,
        time_end,
        d_time,
        initial_states,
        rtol,
        atol,
        safety_factor,
        beta,
        fac_min,
        fac_max,
        h_max,
        h_initial,
        n_max,
        n_stiff,
        OutputType::Dense,
    );

    println!("Solver configuration:");
    println!("  rtol={}, atol={}", rtol, atol);
    println!("  Max steps: {}", n_max);
    println!("  Stiffness check interval: {}", n_stiff);
    println!("  Step size factors: [{}, {}]", fac_min, fac_max);
    println!("  Max step size: {}", h_max);

    // println!("here 04");

    let integral_results_or_error: Result<dop_shared::Stats, dop_shared::IntegrationError> = stepper.integrate();

    // Check if integration succeeded or failed
    match &integral_results_or_error {
        Ok(stats) => {
            println!("Integration succeeded!");
            println!("  Number of function evaluations: {}", stats.num_eval);
            println!("  Accepted steps: {}", stats.accepted_steps);
            println!("  Rejected steps: {}", stats.rejected_steps);
        }
        Err(error) => {
            println!("Integration FAILED with error: {:?}", error);
            println!("  Error details: {}", error);
        }
    }

    // stepper.integrate_with_continuous_output_model(continuous_output)

    // integral_results_or_error.
    // let integral_results: dop_shared::Stats = integral_results_or_error.expect("Integration failed in solve_circuit_equations_rs");

    // println!("here 05");

    let time_out: &Vec<f64> = stepper.x_out();
    let time_out: Array1<f64> = Array1::from_vec(time_out.to_owned());
    // println!("here 06");
    let states_out: &Vec<DVector<f64>> = stepper.y_out(); // shape=(n_time, n_states)
    // println!("here 07");

    // println!("states_out.len()={:?}", states_out.len());
    // println!("time_out.len()={:?}", time_out.len());
    // println!("states_out[0]={:?}", states_out[0]);
    // println!("n_time_old = {}", n_time);
    let n_time: usize = states_out.len();
    // println!("n_time_new = {}", n_time);

    // println!("state_names={:#?}", model.state_names);
    for (i_state, (circuit_type, circuit_name)) in model.state_names.iter().enumerate() {
        // println!("circuit_type={}, circuit_name={}", circuit_type, circuit_name);
        if circuit_type == "pf" {
            // Extract simulated current
            let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            for i_time in 0..n_time {
                current_simulated[i_time] = states_out[i_time][i_state];
            }

            // Store simulated current back into coils
            coils
                .results
                .get_or_insert("pf")
                .get_or_insert(circuit_name)
                .get_or_insert("i")
                .insert("simulated", current_simulated);
            coils
                .results
                .get_or_insert("pf")
                .get_or_insert(circuit_name)
                .get_or_insert("i")
                .insert("time_simulated", time_out.clone());
        } else if circuit_type == "passive" {
            // let i_passive: usize = circuit_name.parse::<usize>().unwrap();
            // // Extract simulated current
            // let mut current_simulated: Array1<f64> = Array1::from_elem(n_time, f64::NAN);
            // for i_time in 0..n_time {
            //     current_simulated[i_time] = states_out[i_time][i_state];
            // }
            // passives
            //     .results
            //     .get_or_insert(passive)
        }
    }
}

// IDEAS FOR TESTING:
// * If we have some PF coils voltage driven = 0V; this will be the same as passives.
