/// Reader and writer for NumPy `.npy` files (format version 1.0 and 2.0)
///
/// Only supports `<f8` (little-endian float64) arrays with C order (fortran_order = False).
///
/// Reference: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
mod header;
mod reader;
mod writer;

pub use reader::{read_npy_0d, read_npy_1d, read_npy_2d};
pub use writer::{write_npy_0d, write_npy_1d, write_npy_2d};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Generate a unique suffix for temporary test file names.
    /// Uses PID + atomic counter to avoid filename collisions,
    /// which could happen if Rust runs tests in parallel across threads.
    fn unique_suffix() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let count: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("[pid={},atomic={}]", std::process::id(), count)
    }

    #[test]
    fn test_write_and_read_npy_1d() {
        let file_path: std::path::PathBuf = std::env::temp_dir().join(format!("gsfit_testing_npy_writer_1d_{}.npy", unique_suffix()));

        let original: Array1<f64> = array![1.0, 2.5, 3.7, -4.2, 0.0];
        write_npy_1d(&file_path, &original);

        let loaded: Array1<f64> = read_npy_1d(&file_path);
        assert_eq!(original, loaded);

        fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_write_and_read_npy_2d() {
        let file_path: std::path::PathBuf = std::env::temp_dir().join(format!("gsfit_testing_npy_writer_2d_{}.npy", unique_suffix()));

        let original: Array2<f64> = Array2::from_shape_vec((3, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        write_npy_2d(&file_path, &original);

        let loaded: Array2<f64> = read_npy_2d(&file_path);
        assert_eq!(original, loaded);

        fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_write_and_read_npy_0d() {
        let file_path: std::path::PathBuf = std::env::temp_dir().join(format!("gsfit_testing_npy_writer_0d_{}.npy", unique_suffix()));

        let original: f64 = 3.14159;
        write_npy_0d(&file_path, original);

        let loaded: f64 = read_npy_0d(&file_path);
        assert_eq!(original, loaded);

        fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_write_npy_1d_single_element() {
        let file_path: std::path::PathBuf = std::env::temp_dir().join(format!("gsfit_testing_npy_writer_1d_single_{}.npy", unique_suffix()));

        let original: Array1<f64> = array![42.0];
        write_npy_1d(&file_path, &original);

        let loaded: Array1<f64> = read_npy_1d(&file_path);
        assert_eq!(original, loaded);

        fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_write_npy_2d_single_row() {
        let file_path: std::path::PathBuf = std::env::temp_dir().join(format!("gsfit_testing_npy_writer_2d_single_row_{}.npy", unique_suffix()));

        let original: Array2<f64> = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        write_npy_2d(&file_path, &original);

        let loaded: Array2<f64> = read_npy_2d(&file_path);
        assert_eq!(original, loaded);

        fs::remove_file(&file_path).ok();
    }
}
