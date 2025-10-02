import sys

# On Windows, we need to add the DLL directory before importing the Rust extension
if sys.platform == "win32":
    from .windows_dll_loader import add_vcpkg_dll_directory
    add_vcpkg_dll_directory()

from . import gsfit_rs
from .gsfit_rs import *

__doc__ = gsfit_rs.__doc__
if hasattr(gsfit_rs, "__all__"):
    __all__ = gsfit_rs.__all__