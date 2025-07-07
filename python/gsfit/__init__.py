from diagnostic_and_simulation_base import get_version

get_version("gsfit")  # needs to be run before importing version_storage
from diagnostic_and_simulation_base import version_storage

from .gsfit import Gsfit

__datetime__ = version_storage.__datetime__
__git_is_dirty__ = version_storage.__git_is_dirty__
__git_short_hash__ = version_storage.__git_short_hash__
__python__ = version_storage.__python__
__python_library__ = version_storage.__python_library__
__user__ = version_storage.__user__
__version__ = version_storage.__version__

# Define interfaces
__all__ = [
    "__datetime__",
    "__git_is_dirty__",
    "__git_short_hash__",
    "__python__",
    "__python_library__",
    "__user__",
    "__version__",
    "Gsfit",
]
