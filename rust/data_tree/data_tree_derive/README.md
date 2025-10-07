# Procedural Macro: `AddDataTreeGetters`

The macro generates the following methods automatically.
Note, all methods are wrapped in a `#[pymethods]` block, making them accessible from Python.

- `get_array1(&self, keys: Vec<String>, py: Python) -> Py<PyArray1<f64>>`
- `get_array2(&self, keys: Vec<String>, py: Python) -> Py<PyArray2<f64>>`
- `get_array3(&self, keys: Vec<String>, py: Python) -> Py<PyArray3<f64>>`
- `get_f64(&self, keys: Vec<String>) -> f64`
- `get_usize(&self, keys: Vec<String>) -> usize`
- `get_bool(&self, keys: Vec<String>) -> bool`
- `get_vec_bool(&self, keys: Vec<String>, py: Python) -> Py<PyList>`
- `get_vec_usize(&self, keys: Vec<String>, py: Python) -> Py<PyList>`
- `keys(&self, py: Python, key_path: Option<&Bound<'_, PyList>>) -> Py<PyList>`
- `print_keys(&self)`

