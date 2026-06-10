use super::greens::Greens;
use ndarray::Array1;
use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use numpy::borrow::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (r, z, r_prime, z_prime, d_r=None, d_z=None))]
pub fn greens_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: Option<PyReadonlyArray1<f64>>,
    d_z: Option<PyReadonlyArray1<f64>>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();

    // Fallback when option not supplied
    let n_prime: usize = r_prime_ndarray.len();
    let d_r: Array1<f64> = match d_r {
        Some(d_r_readonly) => d_r_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };
    let d_z: Array1<f64> = match d_z {
        Some(d_z_readonly) => d_z_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r, d_z);

    greens_table.psi().into_pyarray(py).into()
}

#[pyfunction]
pub fn d_psi_d_r_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: PyReadonlyArray1<f64>,
    d_z: PyReadonlyArray1<f64>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();
    let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
    let d_z_ndarray: Array1<f64> = d_z.to_owned_array();

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r_ndarray, d_z_ndarray);

    greens_table.d_psi_d_r().into_pyarray(py).into()
}

#[pyfunction]
pub fn d_psi_d_z_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: PyReadonlyArray1<f64>,
    d_z: PyReadonlyArray1<f64>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();
    let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
    let d_z_ndarray: Array1<f64> = d_z.to_owned_array();

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r_ndarray, d_z_ndarray);

    greens_table.d_psi_d_z().into_pyarray(py).into()
}

#[pyfunction]
pub fn d2_psi_d_r2_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: PyReadonlyArray1<f64>,
    d_z: PyReadonlyArray1<f64>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();
    let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
    let d_z_ndarray: Array1<f64> = d_z.to_owned_array();

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r_ndarray, d_z_ndarray);

    greens_table.d2_psi_d_r2().into_pyarray(py).into()
}

#[pyfunction]
pub fn d2_psi_d_r_d_z_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: PyReadonlyArray1<f64>,
    d_z: PyReadonlyArray1<f64>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();
    let d_r_ndarray: Array1<f64> = d_r.to_owned_array();
    let d_z_ndarray: Array1<f64> = d_z.to_owned_array();

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r_ndarray, d_z_ndarray);

    greens_table.d2_psi_d_r_d_z().into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (r, z, r_prime, z_prime, d_r=None, d_z=None))]
pub fn d2_psi_d_z2_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: Option<PyReadonlyArray1<f64>>,
    d_z: Option<PyReadonlyArray1<f64>>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();

    // Fallback when option not supplied
    let n_prime: usize = r_prime_ndarray.len();
    let d_r: Array1<f64> = match d_r {
        Some(d_r_readonly) => d_r_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };
    let d_z: Array1<f64> = match d_z {
        Some(d_z_readonly) => d_z_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r, d_z);

    greens_table.d2_psi_d_z2().into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (r, z, r_prime, z_prime, d_r=None, d_z=None))]
pub fn d3_psi_d_r_d_z2_py(
    py: Python,
    r: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    r_prime: PyReadonlyArray1<f64>,
    z_prime: PyReadonlyArray1<f64>,
    d_r: Option<PyReadonlyArray1<f64>>,
    d_z: Option<PyReadonlyArray1<f64>>,
) -> Py<PyArray2<f64>> {
    let r_ndarray: Array1<f64> = r.to_owned_array();
    let z_ndarray: Array1<f64> = z.to_owned_array();
    let r_prime_ndarray: Array1<f64> = r_prime.to_owned_array();
    let z_prime_ndarray: Array1<f64> = z_prime.to_owned_array();

    // Fallback when option not supplied
    let n_prime: usize = r_prime_ndarray.len();
    let d_r: Array1<f64> = match d_r {
        Some(d_r_readonly) => d_r_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };
    let d_z: Array1<f64> = match d_z {
        Some(d_z_readonly) => d_z_readonly.to_owned_array(),
        None => Array1::from_elem(n_prime, f64::NAN),
    };

    let greens_table: Greens = Greens::sensor_to_conductor(r_ndarray, z_ndarray, r_prime_ndarray, z_prime_ndarray, d_r, d_z);

    greens_table.d3_psi_d_r_d_z2().into_pyarray(py).into()
}
