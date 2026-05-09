/// Writer for NumPy `.npy` files (format version 1.0)
///
/// Only supports `<f8` (little-endian float64) arrays with C order (fortran_order = False).
///
/// Reference: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
use crate::header::build_npy_header;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;

/// Write a scalar `f64` to a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file to write
/// * `value` - The scalar value
///
/// # Panics
/// Panics if the file cannot be written
pub fn write_npy_0d(path: &Path, value: f64) {
    let shape_str: String = String::from("()");
    let header_bytes: Vec<u8> = build_npy_header(&shape_str);

    let mut file_bytes: Vec<u8> = Vec::with_capacity(header_bytes.len() + 8);
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&value.to_le_bytes());

    fs::write(path, &file_bytes).expect("npy_writer: Failed to write .npy file");
}

/// Write a 1D `f64` array to a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file to write
/// * `array` - The 1D array data
///
/// # Panics
/// Panics if the file cannot be written
pub fn write_npy_1d(path: &Path, array: &Array1<f64>) {
    let n_elements: usize = array.len();
    let shape_str: String = format!("({},)", n_elements);
    let header_bytes: Vec<u8> = build_npy_header(&shape_str);

    let mut file_bytes: Vec<u8> = Vec::with_capacity(header_bytes.len() + n_elements * 8);
    file_bytes.extend_from_slice(&header_bytes);

    for i_element in 0..n_elements {
        file_bytes.extend_from_slice(&array[i_element].to_le_bytes());
    }

    fs::write(path, &file_bytes).expect("npy_writer: Failed to write .npy file");
}

/// Write a 2D `f64` array to a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file to write
/// * `array` - The 2D array data, shape = (n_rows, n_cols)
///
/// # Panics
/// Panics if the file cannot be written
pub fn write_npy_2d(path: &Path, array: &Array2<f64>) {
    let n_rows: usize = array.nrows();
    let n_cols: usize = array.ncols();
    let n_elements: usize = n_rows * n_cols;
    let shape_str: String = format!("({}, {})", n_rows, n_cols);
    let header_bytes: Vec<u8> = build_npy_header(&shape_str);

    let mut file_bytes: Vec<u8> = Vec::with_capacity(header_bytes.len() + n_elements * 8);
    file_bytes.extend_from_slice(&header_bytes);

    for i_row in 0..n_rows {
        for j_col in 0..n_cols {
            file_bytes.extend_from_slice(&array[[i_row, j_col]].to_le_bytes());
        }
    }

    fs::write(path, &file_bytes).expect("npy_writer: Failed to write .npy file");
}
