import os
from pathlib import Path


def add_vcpkg_dll_directory() -> None:
    """
    This function must be called **before** importing `gsfit_rs`.

    On Windows, Python 3.8+ does not use the `PATH` environment variable to find
    DLLs for extension modules. This is a security feature.
    Instead, Windows only checks:
    1. The directory containing the .pyd file
    2. System directories
    3. Directories explicitly added with `os.add_dll_directory("path_to_dlls")`

    This function adds the `VCPKG_ROOT` path to the DLL search directories.

    OpenBLAS and LAPACK should be installed via vcpkg:
    ```powershell
    vcpkg install openblas:x64-windows-static-md
    vcpkg install lapack:x64-windows-static-md
    ```

    Raises:
        RuntimeError: If `VCPKG_ROOT` is not set or OpenBLAS/LAPACK are not installed.
    """

    # This is needed for mypy type checking, as `os.add_dll_directory` is specific to Windows
    if not hasattr(os, 'add_dll_directory'):
        return

    # Check for `VCPKG_ROOT` environment variable (standard vcpkg variable)
    vcpkg_root = os.environ.get("VCPKG_ROOT")
    if not vcpkg_root:
        raise RuntimeError(
            "`VCPKG_ROOT` environment variable is not set.\n"
            "Please set it to your vcpkg installation directory."
        )

    # Construct the path to the OpenBLAS DLLs
    dll_path = Path(vcpkg_root) / "installed" / "x64-windows-static-md" / "bin"

    # Verify the path exists and contains OpenBLAS
    if not dll_path.exists():
        raise RuntimeError(
            f"vcpkg bin directory not found: {dll_path}"
        )

    # Check for OpenBLAS and LAPACK DLLs
    has_openblas_and_lapack = (dll_path / "libblas.dll").exists() and (dll_path / "liblapack.dll").exists()
    if not has_openblas_and_lapack:
        raise RuntimeError(
            f"OpenBLAS and LAPACK DLLs not found in: {dll_path}\n"
            "Please install OpenBLAS and LAPACK via vcpkg:\n"
            "```powershell\n"
            "  vcpkg install openblas:x64-windows-static-md\n"
            "  vcpkg install lapack:x64-windows-static-md\n"
            "```"
        )

    # Add the DLL directory to the search path
    os.add_dll_directory(str(dll_path))
