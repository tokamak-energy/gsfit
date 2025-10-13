use std::env;
use std::process::Command;

fn main() {
    // Tell the linker where to find the libraries
    #[cfg(windows)]
    {
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=lapack");
    }

    // When building for tests (debug profile), add Python library path for linking
    // This allows tests to link against libpython without extension-module feature
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "test" || profile == "debug" {
        // Try Python from PYO3_PYTHON env var first, then VIRTUAL_ENV, then python3
        let python_cmd = if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
            pyo3_python
        } else if let Ok(venv) = env::var("VIRTUAL_ENV") {
            format!("{}/bin/python", venv)
        } else {
            "python3".to_string()
        };

        // Try to get Python library directory
        if let Ok(output) = Command::new(&python_cmd)
            .args(&["-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"])
            .output()
        {
            if output.status.success() {
                if let Ok(libdir) = String::from_utf8(output.stdout) {
                    let libdir = libdir.trim();
                    println!("cargo:rustc-link-search=native={}", libdir);

                    // Get the Python version for the library name
                    if let Ok(version_output) = Command::new(&python_cmd)
                        .args(&["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
                        .output()
                    {
                        if let Ok(version) = String::from_utf8(version_output.stdout) {
                            let version = version.trim();
                            println!("cargo:rustc-link-lib=dylib=python{}", version);
                        }
                    }
                }
            }
        }
    }
}
