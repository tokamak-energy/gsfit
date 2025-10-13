use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive macro to automatically generate DataTree getter methods
///
/// This macro generates the following methods for structs containing a `results: DataTree` field:
/// - get_array1(&self, keys: Vec<String>, py: Python) -> Py<PyArray1<f64>>
/// - get_array2(&self, keys: Vec<String>, py: Python) -> Py<PyArray2<f64>>
/// - get_array3(&self, keys: Vec<String>, py: Python) -> Py<PyArray3<f64>>
/// - get_f64(&self, keys: Vec<String>) -> f64
/// - get_usize(&self, keys: Vec<String>) -> usize
/// - get_bool(&self, keys: Vec<String>) -> bool
/// - get_vec_bool(&self, keys: Vec<String>, py: Python) -> Py<PyList>
/// - get_vec_usize(&self, keys: Vec<String>, py: Python) -> Py<PyList>
/// - keys(&self, py: Python, key_path: Option<&Bound<'_, PyList>>) -> Py<PyList>
/// - print_keys(&self)
#[proc_macro_derive(AddDataTreeGetters)]
pub fn add_data_tree_getters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident;

    // Verify the struct has a field named "results" of type DataTree
    let has_results_field = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => fields_named
                .named
                .iter()
                .any(|field| field.ident.as_ref().map(|ident| ident == "results").unwrap_or(false)),
            _ => false,
        },
        _ => false,
    };

    if !has_results_field {
        panic!("AddDataTreeGetters can only be derived for structs with a 'results: DataTree' field");
    }

    // Generate the implementation
    let expanded = quote! {
        // PyO3 supports multiple #[pymethods] blocks, so we generate a separate one
        #[pymethods]
        impl #struct_name {
            /// Get Array1<f64> and return a numpy.ndarray
            pub fn get_array1(&self, keys: Vec<String>, py: Python) -> Py<PyArray1<f64>> {
                // Start with the root accumulator
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                let mut result_accumulator: DataTreeAccumulator<'_> = self.results.get(&keys[0]);

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
                        // Convert PyList to Vec<String> and traverse DataTreeAccumulator
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
    };

    TokenStream::from(expanded)
}
