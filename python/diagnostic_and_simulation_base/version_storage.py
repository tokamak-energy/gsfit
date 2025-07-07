import datetime
import getpass
import platform
import sys

from .utility.get_python_library import get_python_library

__version__: str | None = None
__git_is_dirty__: bool | None = None
__git_short_hash__: str | None = None
__python_library__ = get_python_library()
__python__ = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
__user__ = getpass.getuser()
__computer__ = platform.node()
__datetime__ = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
