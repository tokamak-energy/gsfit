//! Internal header parsing and building for `.npy` files

/// Parse the `.npy` header and return (shape, data_offset)
///
/// Supports NPY format version 1.0 (2-byte header length) and 2.0 (4-byte header length)
///
/// # Arguments
/// * `bytes` - The entire file contents as bytes
///
/// # Returns
/// * `(Vec<usize>, usize)` - (shape dimensions, byte offset where data begins)
pub(crate) fn parse_npy_header(bytes: &[u8]) -> (Vec<usize>, usize) {
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
            assert!(
                bytes.len() >= 12,
                "npy_reader: File too small for .npy v2.0 header (need at least 12 bytes, got {})",
                bytes.len()
            );
            let len: usize = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => panic!("npy_reader: Unsupported .npy format version {}", major_version),
    };

    let data_offset: usize = header_start + header_len;
    assert!(
        data_offset <= bytes.len(),
        "npy_reader: File is truncated: header indicates data starts at byte {}, but file is only {} bytes",
        data_offset,
        bytes.len()
    );
    let header_bytes: &[u8] = &bytes[header_start..data_offset];
    let header_str: &str = std::str::from_utf8(header_bytes).expect("npy_reader: Header is not valid UTF-8");

    // Parse 'descr' — must be '<f8' (little-endian float64)
    assert!(
        header_str.contains("'<f8'"),
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

/// Build a complete NPY v1.0 header (magic + version + header length + header string)
///
/// The header is padded with spaces so that the total (magic + version + header_len + header)
/// is a multiple of 64 bytes, as recommended by the format specification.
///
/// # Arguments
/// * `shape_str` - The shape as a Python tuple string, e.g. `"(3,)"` or `"(161, 80)"`
///
/// # Returns
/// * `Vec<u8>` - The complete header bytes ready to be written before the data
pub(crate) fn build_npy_header(shape_str: &str) -> Vec<u8> {
    let dict_str: String = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}", shape_str);

    // Prefix: magic (6 bytes) + version (2 bytes) + header_len (2 bytes) = 10 bytes
    let prefix_len: usize = 10;

    // We need: prefix_len + dict_str.len() + 1 (for trailing newline) to be padded to a multiple of 64
    let unpadded_len: usize = prefix_len + dict_str.len() + 1; // +1 for '\n'
    let padded_total: usize = unpadded_len.div_ceil(64) * 64;
    let n_padding: usize = padded_total - unpadded_len;
    let header_len: u16 = (dict_str.len() + n_padding + 1) as u16; // dict + spaces + newline

    let mut header_bytes: Vec<u8> = Vec::with_capacity(padded_total);

    // Magic number and version
    header_bytes.push(0x93);
    header_bytes.extend_from_slice(b"NUMPY");
    header_bytes.push(1); // major version
    header_bytes.push(0); // minor version

    // Header length (little-endian u16)
    header_bytes.extend_from_slice(&header_len.to_le_bytes());

    // Header string (dict + padding spaces + newline)
    header_bytes.extend_from_slice(dict_str.as_bytes());
    for _i_pad in 0..n_padding {
        header_bytes.push(b' ');
    }
    header_bytes.push(b'\n');

    header_bytes
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
pub(crate) fn parse_shape_from_header(header: &str) -> Vec<usize> {
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
        let mut header_padded: Vec<u8> = header_str.as_bytes().to_vec();
        header_padded.push(b'\n');
        // Pad header to align total (10 + header_len) to multiple of 64
        let total_without_pad: usize = 10 + header_padded.len();
        let padded_len: usize = ((total_without_pad + 63) / 64) * 64 - 10;
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
    fn test_header_alignment() {
        // Header total should be a multiple of 64 bytes
        let header: Vec<u8> = build_npy_header("(100,)");
        assert_eq!(header.len() % 64, 0, "npy_writer: Header length should be a multiple of 64");
    }
}
