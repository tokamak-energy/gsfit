use data_tree::{DataTree, DataValue};
use ndarray::{Array1, Array2, Array3};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList};

/// Convert a DataTree to a Python dictionary (recursive).
///
/// Use this in `__getstate__` method to serialize a DataTree for Python pickling.
///
/// # Example
/// ```ignore
/// fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
///     let state_dict = PyDict::new(py);
///     let results_dict = data_tree_to_py_dict(py, &self.results)?;
///     state_dict.set_item("results", results_dict)?;
///     state_dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
///     Ok(state_dict.into())
/// }
/// ```
pub fn data_tree_to_py_dict(py: Python<'_>, data_tree: &DataTree) -> PyResult<Py<PyDict>> {
    let py_dictionary: Bound<'_, PyDict> = PyDict::new(py);

    let keys_store: Vec<String> = data_tree.keys();
    for key in keys_store {
        let data_value: &DataValue = data_tree.data.get(&key).expect("Key not found in DataTree");

        match data_value {
            DataValue::F64(f64_val) => {
                py_dictionary.set_item(&key, *f64_val)?;
            }
            DataValue::String(string_val) => {
                py_dictionary.set_item(&key, string_val)?;
            }
            DataValue::Usize(usize_val) => {
                py_dictionary.set_item(&key, *usize_val)?;
            }
            DataValue::Bool(bool_val) => {
                py_dictionary.set_item(&key, *bool_val)?;
            }
            DataValue::Array1(array1_val) => {
                let array1_py: Bound<'_, PyArray1<f64>> = array1_val.to_owned().into_pyarray(py);
                py_dictionary.set_item(&key, array1_py)?;
            }
            DataValue::Array2(array2_val) => {
                let array2_py: Bound<'_, PyArray2<f64>> = array2_val.to_owned().into_pyarray(py);
                py_dictionary.set_item(&key, array2_py)?;
            }
            DataValue::Array3(array3_val) => {
                let array3_py: Bound<'_, PyArray3<f64>> = array3_val.to_owned().into_pyarray(py);
                py_dictionary.set_item(&key, array3_py)?;
            }
            DataValue::VecUsize(vec_usize_val) => {
                py_dictionary.set_item(&key, vec_usize_val)?;
            }
            DataValue::VecBool(vec_bool_val) => {
                py_dictionary.set_item(&key, vec_bool_val)?;
            }
            DataValue::DataTree(child_data_tree) => {
                let child_py_dictionary: Py<PyDict> = data_tree_to_py_dict(py, child_data_tree)?;
                py_dictionary.set_item(&key, child_py_dictionary)?;
            }
        }
    }

    Ok(py_dictionary.unbind())
}

/// Convert a Python dictionary to a DataTree (recursive).
///
/// Use this in `__setstate__` method to deserialize a DataTree from Python unpickling.
///
/// # Example
/// ```ignore
/// fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
///     let results_dict = state.get_item("results")?
///         .ok_or_else(|| PyTypeError::new_err("Missing 'results' key"))?;
///     let results_dict_bound = results_dict.downcast::<PyDict>()?;
///     self.results = py_dict_to_data_tree(results_dict_bound)?;
///     Ok(())
/// }
/// ```
pub fn py_dict_to_data_tree(state_dictionary: &Bound<'_, PyDict>) -> PyResult<DataTree> {
    let mut out_data_tree: DataTree = DataTree::new();

    for (key_py, value_py) in state_dictionary.iter() {
        let key: String = key_py.extract()?;
        let value: DataValue = py_any_to_data_value(&value_py)?;
        out_data_tree.data.insert(key, value);
    }

    Ok(out_data_tree)
}

/// Convert a Python object to a DataValue (handles all supported types).
///
/// This is used internally by `py_dict_to_data_tree` but can also be used standalone
/// if you need to convert individual Python values.
pub fn py_any_to_data_value(value_py: &Bound<'_, PyAny>) -> PyResult<DataValue> {
    // Try dictionary first (for nested DataTrees)
    if let Ok(dict_val) = value_py.downcast::<PyDict>() {
        let nested_data_tree: DataTree = py_dict_to_data_tree(&dict_val)?;
        return Ok(DataValue::DataTree(nested_data_tree));
    }

    // Try numpy arrays
    if let Ok(array1_val) = value_py.downcast::<PyArray1<f64>>() {
        let array1: Array1<f64> = array1_val.to_owned_array();
        return Ok(DataValue::Array1(array1));
    }

    if let Ok(array2_val) = value_py.downcast::<PyArray2<f64>>() {
        let array2: Array2<f64> = array2_val.to_owned_array();
        return Ok(DataValue::Array2(array2));
    }

    if let Ok(array3_val) = value_py.downcast::<PyArray3<f64>>() {
        let array3: Array3<f64> = array3_val.to_owned_array();
        return Ok(DataValue::Array3(array3));
    }

    // Try Python lists (for Vec<usize> or Vec<bool>)
    if let Ok(list_val) = value_py.downcast::<PyList>() {
        let n_items: usize = list_val.len();
        if n_items == 0 {
            let vec_usize: Vec<usize> = list_val.extract()?;
            return Ok(DataValue::VecUsize(vec_usize));
        }

        let first_item: Bound<'_, PyAny> = list_val.get_item(0)?;
        if first_item.is_instance_of::<PyBool>() {
            let vec_bool: Vec<bool> = list_val.extract()?;
            return Ok(DataValue::VecBool(vec_bool));
        }

        let vec_usize: Vec<usize> = list_val.extract()?;
        return Ok(DataValue::VecUsize(vec_usize));
    }

    // Try primitive types (order matters: bool before usize/f64)
    if let Ok(bool_val) = value_py.extract::<bool>() {
        return Ok(DataValue::Bool(bool_val));
    }

    if let Ok(usize_val) = value_py.extract::<usize>() {
        return Ok(DataValue::Usize(usize_val));
    }

    if let Ok(f64_val) = value_py.extract::<f64>() {
        return Ok(DataValue::F64(f64_val));
    }

    if let Ok(string_val) = value_py.extract::<String>() {
        return Ok(DataValue::String(string_val));
    }

    Err(PyTypeError::new_err("Unsupported type in Python unpickling"))
}
