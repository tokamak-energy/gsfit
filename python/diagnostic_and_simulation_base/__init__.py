# get_version("diagnostic_and_simulation_base")  # needs to be run before importing version_storage
import diagnostic_and_simulation_base.version_storage as version_storage

from .diagnostic_and_simulation_base import DiagnosticAndSimulationBase
from .nested_dictionary import NestedDict
from .utility.get_version import get_version
from .utility.logging import logger

__datetime__ = version_storage.__datetime__
__git_is_dirty__ = version_storage.__git_is_dirty__
__git_short_hash__ = version_storage.__git_short_hash__
__python__ = version_storage.__python__
__python_library__ = version_storage.__python_library__
__user__ = version_storage.__user__
__version__ = version_storage.__version__

# Define interfaces
__all__ = [
    "DiagnosticAndSimulationBase",
    "NestedDict",
    "get_version",
    "logger",
    "version_storage",
]
