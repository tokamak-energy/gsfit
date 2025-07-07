import os
import pathlib
from importlib import metadata as metadata_py

import setuptools_git_versioning  # type: ignore

from .. import version_storage


def get_version(package_name: str) -> tuple[bool, str, str]:
    __version__ = metadata_py.version(package_name)
    __git_is_dirty__ = False

    __git_short_hash__ = __version__.split("git.")[1].replace(".dirty", "") if "git." in __version__ else ""

    version_storage.__version__ = __version__
    version_storage.__git_is_dirty__ = __git_is_dirty__
    version_storage.__git_short_hash__ = __git_short_hash__

    # # Get "__version__"
    # # BUXTON: THIS WON'T WORK AS __file__ is no longer this file
    # path_to_pyproject_toml = os.path.dirname(__file__) + '/../../'
    # path_to_pyproject_toml = str(pathlib.Path(path_to_pyproject_toml).resolve())
    # use_pyproject_toml = os.path.isfile(path_to_pyproject_toml + '/pyproject.toml')
    # if use_pyproject_toml:
    #     # get "__version__" from "pyproject.toml" if "pyproject.toml" exists
    #     # this will work if running locally, i.e. not installed
    #     pwd = os.getcwd()
    #     os.chdir(path_to_pyproject_toml)
    #     __version__ = setuptools_git_versioning.version_from_git()
    #     os.chdir(pwd)
    # else:
    #     # Get "__version__" from python's package metadata
    #     # This will work for all pip installation settings:
    #     #    pip install gas
    #     #    pip install .
    #     #    pip install -e .
    #     __version__ = metadata_py.version(package_name)

    # # Get git short hash
    # __git_short_hash__ = __version__.split("git.")[1].replace('.dirty', '')
    # __git_is_dirty__ = '.dirty' in __version__

    return __git_is_dirty__, __git_short_hash__, __version__
