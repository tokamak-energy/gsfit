# AddDataTreeGetters Procedural Macro

## Summary

Successfully created a procedural macro `AddDataTreeGetters` that automatically generates DataTree getter methods for structs containing a `results: DataTree` field.

## What Was Done

### 1. Created a new procedural macro crate: `data_tree_derive`

**Location:** `/rust/data_tree_derive/`

**Files created:**
- `Cargo.toml` - Defines the crate as a proc-macro with dependencies on `syn`, `quote`, and `proc-macro2`
- `src/lib.rs` - Implements the `AddDataTreeGetters` derive macro

### 2. Implemented the derive macro

The macro generates the following methods automatically:
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

All methods are wrapped in a `#[pymethods]` block, making them accessible from Python.

### 3. Updated the data_tree crate

**File:** `/rust/data_tree/Cargo.toml`
- Added dependency: `data_tree_derive = { path = "../data_tree_derive" }`

**File:** `/rust/data_tree/src/lib.rs`
- Added re-export: `pub use data_tree_derive::AddDataTreeGetters;`

This allows users to import the macro from `data_tree` instead of needing to import from `data_tree_derive`.

### 4. Updated the workspace

**File:** `/rust/Cargo.toml`
- Added `data_tree_derive` to workspace members

### 5. Applied the macro to Coils struct

**File:** `/rust/gsfit_rs/src/coils.rs`
- Added `AddDataTreeGetters` to the derive list: `#[derive(Clone, AddDataTreeGetters)]`
- Removed 150+ lines of repetitive getter method implementations
- Updated imports to include `AddDataTreeGetters`

### 6. Enabled PyO3 multiple-pymethods feature

**File:** `/rust/gsfit_rs/Cargo.toml`
- Updated PyO3 dependency to include the `multiple-pymethods` feature
- Changed from: `pyo3 = { version = "0.26.0", features = ["extension-module"] }`
- Changed to: `pyo3 = { version = "0.26.0", features = ["extension-module", "multiple-pymethods"] }`

This feature is required for PyO3 0.26+ to support multiple `#[pymethods]` blocks on the same struct.

## Usage

To use the macro on any struct with a `results: DataTree` field:

```rust
use data_tree::{DataTree, AddDataTreeGetters};
use pyo3::prelude::*;

#[derive(Clone, AddDataTreeGetters)]
#[pyclass]
pub struct MyStruct {
    pub results: DataTree,
}

#[pymethods]
impl MyStruct {
    // Your custom methods here
    #[new]
    pub fn new() -> Self {
        Self { results: DataTree::new() }
    }
}
```

The macro will automatically generate all the getter methods in a separate `#[pymethods]` block.

## Benefits

1. **Reduced code duplication:** Eliminated 150+ lines of repetitive boilerplate code
2. **Maintainability:** Changes to getter logic only need to be made in one place (the macro)
3. **Consistency:** All structs with `DataTree` fields get the same interface
4. **Type safety:** The macro validates at compile-time that the struct has a `results` field

## Technical Details

- **Macro type:** Derive procedural macro
- **Dependencies:** Uses `syn` for parsing, `quote` for code generation
- **PyO3 integration:** Generates PyO3-compatible `#[pymethods]` blocks
- **Requirement:** PyO3 0.21+ with `multiple-pymethods` feature enabled

## Compilation Status

✅ All crates compile successfully
✅ Release build completes without errors
✅ The macro is ready to use
