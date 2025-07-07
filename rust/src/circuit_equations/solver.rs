use core::time;

use crate::coils::Coils;
use crate::passives::Passives;
use crate::sensors::{BpProbes, FluxLoops, RogowskiCoils};
use ndarray::{Array, Array1, Array2, Array3, s};
use ndarray_interp::interp1d::Interp1D;
use ndarray_linalg::Inverse; // Import the Inverse trait
use numpy::IntoPyArray; // converting to python data types
use numpy::PyArrayMethods; // used in to convert python data into ndarray
use numpy::{PyArray1, PyArray2};
use ode_solvers::dopri5::*;
use ode_solvers::*;
use pyo3::prelude::*;

pub struct CircuitEquationModel {
    pub something: f64,
    pub resistance_matrix: Array2<f64>,
    pub inductance_matrix: Array2<f64>,
    pub voltages: Array2<f64>,
    pub time: Array1<f64>,
}

impl CircuitEquationModel {
    /// Constructor for CircuitEquationModel
    pub fn new(coils: Coils) -> Self {
        let coil_names: Vec<String> = coils.results.get("pf").keys();
        let n_coils: usize = coil_names.len();

        let time: Array1<f64> = coils.results.get("pf").get(&coil_names[0]).get("i").get("time_experimental").unwrap_array1();
        let n_time: usize = time.len();

        let mut voltages: Array2<f64> = Array2::zeros((n_time, n_coils));

        // for i_coil in 0..n_coils {
        //     let interpolator = Interp1D::builder(voltages.slice(s![.., i_coil]))
        //         .x(time.clone())
        //         .build()
        //         .expect("Coils.split_into_static_and_dynamic: Can't make Interp1D");

        //     let voltage_this_coil: f64 = interpolator.interp_scalar(time).unwrap();
        // }

        CircuitEquationModel {
            something: 0.0,
            resistance_matrix: Array2::zeros((0, 0)),
            inductance_matrix: Array2::zeros((0, 0)),
            voltages,
            time,
        }
    }
}

impl ode_solvers::System<f64, DVector<f64>> for CircuitEquationModel {
    /// System of ordinary differential equations.
    fn system(&self, time_now: f64, states: &DVector<f64>, d_states_d_t: &mut DVector<f64>) {
        let states_ndarray: Array1<f64> = Array1::from(states.as_slice().to_vec());

        let n_states: usize = states.len();

        let resistance_matrix: Array2<f64> = self.resistance_matrix.to_owned();
        let inductance_matrix: Array2<f64> = self.inductance_matrix.to_owned();

        // Need to interpolate the voltages for each PF coil at the current time
        let time: Array1<f64> = self.time.to_owned();
        let voltages_vs_time: Array2<f64> = self.voltages.to_owned();
        let mut voltages: Array1<f64> = Array1::zeros(n_states);

        for i_state in 0..n_states {
            let interpolator = Interp1D::builder(voltages_vs_time.slice(s![.., i_state]))
                .x(time.clone())
                .build()
                .expect("Coils.split_into_static_and_dynamic: Can't make Interp1D");

            let voltage_this_coil: f64 = interpolator.interp_scalar(time_now).unwrap();
            voltages[i_state] = voltage_this_coil;
        }

        let inv_inductance_matrix: Array2<f64> = inductance_matrix.inv().unwrap();
        let resistance_matrix_dot_states: Array1<f64> = resistance_matrix.dot(&states_ndarray);
        let u: Array1<f64> = voltages - resistance_matrix_dot_states;
        let d_states_d_t_tmp: Array1<f64> = inv_inductance_matrix.dot(&u);
        // Copy the results from temporary array into results array
        for i_state in 0..n_states {
            d_states_d_t[i_state] = d_states_d_t_tmp[i_state];
        }
    }

    /// Stop function called at every successful integration step. The integration is stopped when this function returns true.
    fn solout(&mut self, _x: f64, _y: &DVector<f64>, _dy: &DVector<f64>) -> bool {
        false
    }
}

#[pyfunction]
pub fn solve_circuit_equations(
    mut coils: PyRefMut<Coils>,
    mut passives: PyRefMut<Passives>,
    mut bp_probes: PyRefMut<BpProbes>,
    mut flux_loops: PyRefMut<FluxLoops>,
    mut rogowski_coils: PyRefMut<RogowskiCoils>,
    times_to_solve: &Bound<'_, PyArray1<f64>>,
) {
    let coil_names: Vec<String> = coils.results.get("pf").keys();
    let n_coils: usize = coil_names.len();

    let times_to_solve_ndarray: Array1<f64> = Array1::from(unsafe { times_to_solve.as_array() }.to_vec());

    let model: CircuitEquationModel = CircuitEquationModel::new(coils.to_owned());

    let time_start: f64 = times_to_solve_ndarray[0];
    let time_end: f64 = times_to_solve_ndarray.last().unwrap().to_owned();
    let d_time: f64 = times_to_solve_ndarray[1] - times_to_solve_ndarray[0];
    let rtol: f64 = 1e-6;
    let atol: f64 = 1e-6;

    let initial_states: DVector<f64> = DVector::zeros(n_coils);

    let mut stepper = Dopri5::new(model, time_start, time_end, d_time, initial_states, rtol, atol);

    let res = stepper.integrate();

    let x_out = stepper.x_out();
    let y_out = stepper.y_out();

    // print to remove warnings
    coils.print_keys();
    passives.print_keys();
    bp_probes.print_keys();
    flux_loops.print_keys();
    rogowski_coils.print_keys();
    println!("times_to_solve: {:?}", times_to_solve_ndarray);

    println!("solve_circuit_equations");
}
