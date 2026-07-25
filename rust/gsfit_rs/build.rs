use std::env;
use std::fs;
use std::path::Path;

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

        // Run-time: make sure test binaries can find libpython without the user having to
        // set any environment variables.
        let target_os: String = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        match target_os.as_str() {
            // Linux/macOS: embed an rpath so the loader finds libpython without needing
            // `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to be set.
            "linux" | "macos" => {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
            // Windows has no rpath. Instead, hard-link `pythonXY.dll` next to the test binaries
            // so the loader finds it (Windows searches the executable's own directory first).
            // A hard-link shares the same bytes as the original DLL, so nothing is duplicated on
            // disk. This avoids requiring the Python install directory to be on `PATH`, which
            // matters for tools like `cargo test`, `cargo llvm-cov`, and `cargo mutants` (the
            // latter copies the tree to a temp dir and runs the standalone test executable there).
            "windows" => {
                link_python_dll_next_to_binaries(&lib_dir, &lib_name);
            }
            _ => {}
        }
    }
}

/// Hard-link `pythonXY.dll` into the profile output directory and its `deps/` subdirectory so
/// that Windows test executables can locate libpython at run-time. Falls back to a copy if the
/// link cannot be created (e.g. the target directory is on a different volume).
fn link_python_dll_next_to_binaries(lib_dir: &str, lib_name: &str) {
    // On Windows, `lib_dir` points at the `libs` folder (containing `pythonXY.lib`); the DLL
    // itself lives one level up, next to `python.exe`.
    let Some(install_root) = Path::new(lib_dir).parent() else {
        return;
    };
    let dll_source: std::path::PathBuf = install_root.join(format!("{lib_name}.dll"));
    if !dll_source.exists() {
        println!("cargo:warning={} not found; test executables may fail to load libpython", dll_source.display());
        return;
    }

    // Rebuild if the DLL changes (e.g. after a Python upgrade).
    println!("cargo:rerun-if-changed={}", dll_source.display());

    // `OUT_DIR` is `<target>/<profile>/build/<crate>-<hash>/out`; the test binaries live in
    // `<target>/<profile>/deps/` (and integration binaries in `<target>/<profile>/`).
    let out_dir: String = env::var("OUT_DIR").unwrap_or_default();
    let Some(profile_dir) = Path::new(&out_dir).ancestors().nth(3) else {
        return;
    };
    let dll_file_name: &std::ffi::OsStr = dll_source.file_name().unwrap_or_default();

    for dest_dir in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
        if fs::create_dir_all(&dest_dir).is_err() {
            continue;
        }
        let dll_dest: std::path::PathBuf = dest_dir.join(dll_file_name);
        // Only (re)link when missing or out of date, to avoid needless work on incremental builds.
        let needs_link: bool = match (fs::metadata(&dll_source), fs::metadata(&dll_dest)) {
            (Ok(src_meta), Ok(dst_meta)) => match (src_meta.modified(), dst_meta.modified()) {
                (Ok(src_time), Ok(dst_time)) => src_time > dst_time,
                _ => true,
            },
            _ => true,
        };
        if needs_link {
            // `hard_link` fails if the destination already exists, so clear any stale entry first.
            let _ = fs::remove_file(&dll_dest);
            // Prefer a hard-link (no data duplication); fall back to a copy across volumes.
            if fs::hard_link(&dll_source, &dll_dest).is_err() {
                if let Err(copy_error) = fs::copy(&dll_source, &dll_dest) {
                    println!(
                        "cargo:warning=failed to place {} next to the test binaries at {}: {copy_error}; test executables may fail to load libpython",
                        dll_source.display(),
                        dll_dest.display()
                    );
                }
            }
        }
    }
}
