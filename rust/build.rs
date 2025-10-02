fn main() {
    // Tell the linker where to find the libraries
    #[cfg(windows)]
    {
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=lapack");
    }
}
