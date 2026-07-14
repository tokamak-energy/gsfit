use std::env;

fn main() {
    // Rebuild if the Python interpreter selection changes
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    // Test binaries (`cargo test`, `cargo test --doc`, `cargo llvm-cov`) embed a Python
    // interpreter, so they must link against libpython. These always use the "debug" profile.
    // Release builds are made by maturin for the Python extension module, which must NOT link
    // libpython: the symbols are provided by the Python process which loads the module.
    let profile: String = env::var("PROFILE").unwrap_or_default();
    if profile != "debug" {
        return;
    }

    // Discover the Python interpreter using pyo3's own logic (checks `PYO3_PYTHON`, then
    // `VIRTUAL_ENV`, then `python3`/`python` on PATH), which handles uv, venv, conda, and
    // system Pythons on Linux, macOS, and Windows
    let config: &pyo3_build_config::InterpreterConfig = pyo3_build_config::get();

    if let (Some(lib_dir), Some(lib_name)) = (config.lib_dir(), config.lib_name()) {
        // Link-time: where libpython lives and what it is called
        // (Linux "python3.14"; macOS "python3.14"; Windows "python314")
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib={lib_name}");

        // Run-time: embed an rpath so test binaries can find libpython without needing
        // `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to be set.
        // Windows has no rpath; the DLL is found via PATH instead.
        let target_os: String = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "linux" || target_os == "macos" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
    }
}
