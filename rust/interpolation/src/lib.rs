// Private modules
mod errors;

// Public modules
pub mod dim_1 {
    pub mod linear;
}

// Public flattened exports
pub use dim_1::linear::Dim1Linear;
pub use errors::Error;
