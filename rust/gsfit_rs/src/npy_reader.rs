/// Reader for NumPy `.npy` files (format version 1.0 and 2.0)
///
/// Only supports `<f8` (little-endian float64) arrays with C order (fortran_order = False).
/// Uses only the standard library — no external crates required.
///
/// Reference: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;

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

/// Parse the `.npy` header and return (shape, data_offset)
///
/// Supports NPY format version 1.0 (2-byte header length) and 2.0 (4-byte header length)
///
/// # Arguments
/// * `bytes` - The entire file contents as bytes
///
/// # Returns
/// * `(Vec<usize>, usize)` - (shape dimensions, byte offset where data begins)
fn parse_npy_header(bytes: &[u8]) -> (Vec<usize>, usize) {
    // Check magic number: \x93NUMPY
    assert!(bytes.len() >= 10, "npy_reader: File too small to be a valid .npy file");
    assert_eq!(bytes[0], 0x93, "npy_reader: Invalid magic number");
    assert_eq!(&bytes[1..6], b"NUMPY", "npy_reader: Invalid magic string");

    // Read version
    let major_version: u8 = bytes[6];
    let _minor_version: u8 = bytes[7];

    // Read header length (little-endian)
    let (header_len, header_start): (usize, usize) = match major_version {
        1 => {
            let len: usize = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        2 => {
            let len: usize = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => panic!("npy_reader: Unsupported .npy format version {}", major_version),
    };

    let data_offset: usize = header_start + header_len;
    let header_bytes: &[u8] = &bytes[header_start..data_offset];
    let header_str: &str = std::str::from_utf8(header_bytes).expect("npy_reader: Header is not valid UTF-8");

    // Parse 'descr' — must be '<f8' (little-endian float64)
    assert!(
        header_str.contains("'<f8'") || header_str.contains("'=f8'"),
        "npy_reader: Only little-endian float64 ('<f8') is supported, got header: {}",
        header_str.trim()
    );

    // Parse 'fortran_order' — must be False (C order)
    assert!(
        header_str.contains("'fortran_order': False"),
        "npy_reader: Only C-order (fortran_order: False) is supported, got header: {}",
        header_str.trim()
    );

    // Parse 'shape' — extract the tuple contents
    let shape: Vec<usize> = parse_shape_from_header(header_str);

    (shape, data_offset)
}

/// Extract the shape tuple from the header string
///
/// Handles formats like:
/// * `'shape': (150,)` — 1D
/// * `'shape': (161, 80)` — 2D
/// * `'shape': (161, 80, 3)` — 3D (etc.)
///
/// # Arguments
/// * `header` - The header string from the `.npy` file
///
/// # Returns
/// * `Vec<usize>` - The shape dimensions
fn parse_shape_from_header(header: &str) -> Vec<usize> {
    // Find "'shape': (" and extract until ")"
    let shape_key: &str = "'shape': (";
    let shape_start: usize = header.find(shape_key).expect("npy_reader: Could not find 'shape' key in header") + shape_key.len();
    let shape_end: usize = header[shape_start..].find(')').expect("npy_reader: Could not find closing ')' for shape tuple") + shape_start;

    let shape_str: &str = &header[shape_start..shape_end];

    // Parse comma-separated values, ignoring trailing comma for 1D case like "(150,)"
    let mut shape: Vec<usize> = Vec::new();
    for part in shape_str.split(',') {
        let trimmed: &str = part.trim();
        if !trimmed.is_empty() {
            let dim: usize = trimmed.parse::<usize>().expect("npy_reader: Failed to parse shape dimension");
            shape.push(dim);
        }
    }

    assert!(!shape.is_empty(), "npy_reader: Shape is empty");

    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shape_1d() {
        let header: &str = "{'descr': '<f8', 'fortran_order': False, 'shape': (150,), }";
        let shape: Vec<usize> = parse_shape_from_header(header);
        assert_eq!(shape, vec![150]);
    }

    #[test]
    fn test_parse_shape_2d() {
        let header: &str = "{'descr': '<f8', 'fortran_order': False, 'shape': (161, 80), }";
        let shape: Vec<usize> = parse_shape_from_header(header);
        assert_eq!(shape, vec![161, 80]);
    }

    #[test]
    fn test_parse_npy_header() {
        // Construct a minimal valid v1.0 .npy file with a 1D shape (3,)
        let header_str: &str = "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }";
        let header_bytes: &[u8] = header_str.as_bytes();
        // Pad header to align total (10 + header_len) to multiple of 64
        let total_without_pad: usize = 10 + header_bytes.len();
        let padded_len: usize = ((total_without_pad + 63) / 64) * 64 - 10;
        let mut header_padded: Vec<u8> = header_bytes.to_vec();
        header_padded.resize(padded_len, b' ');

        let mut file_bytes: Vec<u8> = Vec::new();
        file_bytes.push(0x93);
        file_bytes.extend_from_slice(b"NUMPY");
        file_bytes.push(1); // major version
        file_bytes.push(0); // minor version
        let header_len: u16 = header_padded.len() as u16;
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(&header_padded);

        // Append 3 f64 values
        for val in [1.0_f64, 2.0_f64, 3.0_f64] {
            file_bytes.extend_from_slice(&val.to_le_bytes());
        }

        let (shape, data_offset): (Vec<usize>, usize) = parse_npy_header(&file_bytes);
        assert_eq!(shape, vec![3]);
        assert_eq!(data_offset, 10 + header_padded.len());

        // Verify we can read the data
        let data_bytes: &[u8] = &file_bytes[data_offset..];
        assert_eq!(data_bytes.len(), 24); // 3 * 8 bytes
    }

    #[test]
    fn test_read_npy_1d_from_test_data() {
        let test_data_path: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/masking/vessel_r.npy");
        let path: &Path = Path::new(test_data_path);
        if path.exists() {
            let v: Array1<f64> = read_npy_1d(path);
            assert!(v.len() > 0, "npy_reader: vessel_r.npy should not be empty");
        }
    }
}
