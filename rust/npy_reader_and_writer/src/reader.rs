/// Reader for NumPy `.npy` files (format version 1.0 and 2.0)
///
/// Only supports `<f8` (little-endian float64) arrays with C order (fortran_order = False).
///
/// Reference: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
use crate::header::parse_npy_header;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;

/// Read a scalar `f64` from a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file
///
/// # Returns
/// * `f64` - The scalar value
///
/// # Panics
/// Panics if the file cannot be read, the format is invalid, or the dtype is not `<f8`
pub fn read_npy_0d(path: &Path) -> f64 {
    let bytes: Vec<u8> = fs::read(path).expect("npy_reader: Failed to read .npy file");
    let (shape, data_offset): (Vec<usize>, usize) = parse_npy_header(&bytes);

    assert!(
        shape.is_empty(),
        "npy_reader: Expected a 0D (scalar) array with shape (), got shape {:?}",
        shape
    );

    let data_bytes: &[u8] = &bytes[data_offset..];
    assert_eq!(
        data_bytes.len(),
        8,
        "npy_reader: Data size mismatch: expected 8 bytes, got {}",
        data_bytes.len()
    );

    f64::from_le_bytes(data_bytes[0..8].try_into().unwrap())
}

/// Read a 1D `f64` array from a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file
///
/// # Returns
/// * `Array1<f64>` - The array data
///
/// # Panics
/// Panics if the file cannot be read, the format is invalid, or the dtype is not `<f8`
pub fn read_npy_1d(path: &Path) -> Array1<f64> {
    let bytes: Vec<u8> = fs::read(path).expect("npy_reader: Failed to read .npy file");
    let (shape, data_offset): (Vec<usize>, usize) = parse_npy_header(&bytes);

    assert_eq!(shape.len(), 1, "npy_reader: Expected a 1D array, got shape {:?}", shape);

    let n_elements: usize = shape[0];
    let data_bytes: &[u8] = &bytes[data_offset..];
    assert_eq!(
        data_bytes.len(),
        n_elements * 8,
        "npy_reader: Data size mismatch: expected {} bytes, got {}",
        n_elements * 8,
        data_bytes.len()
    );

    let mut v: Vec<f64> = Vec::with_capacity(n_elements);
    for i_element in 0..n_elements {
        let byte_offset: usize = i_element * 8;
        let value: f64 = f64::from_le_bytes(data_bytes[byte_offset..byte_offset + 8].try_into().unwrap());
        v.push(value);
    }

    Array1::from_vec(v)
}

/// Read a 2D `f64` array from a `.npy` file
///
/// # Arguments
/// * `path` - Path to the `.npy` file
///
/// # Returns
/// * `Array2<f64>` - The array data, shape = (n_rows, n_cols)
///
/// # Panics
/// Panics if the file cannot be read, the format is invalid, or the dtype is not `<f8`
pub fn read_npy_2d(path: &Path) -> Array2<f64> {
    let bytes: Vec<u8> = fs::read(path).expect("npy_reader: Failed to read .npy file");
    let (shape, data_offset): (Vec<usize>, usize) = parse_npy_header(&bytes);

    assert_eq!(shape.len(), 2, "npy_reader: Expected a 2D array, got shape {:?}", shape);

    let n_rows: usize = shape[0];
    let n_cols: usize = shape[1];
    let n_elements: usize = n_rows * n_cols;
    let data_bytes: &[u8] = &bytes[data_offset..];
    assert_eq!(
        data_bytes.len(),
        n_elements * 8,
        "npy_reader: Data size mismatch: expected {} bytes, got {}",
        n_elements * 8,
        data_bytes.len()
    );

    let mut v: Vec<f64> = Vec::with_capacity(n_elements);
    for i_element in 0..n_elements {
        let byte_offset: usize = i_element * 8;
        let value: f64 = f64::from_le_bytes(data_bytes[byte_offset..byte_offset + 8].try_into().unwrap());
        v.push(value);
    }

    Array2::from_shape_vec((n_rows, n_cols), v).expect("npy_reader: Failed to create Array2 from data")
}
