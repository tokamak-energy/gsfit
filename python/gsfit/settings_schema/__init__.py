"""
GSFit's settings schema.

* `xsd/`: the XML schemas of the settings XML files, which are the source of truth
* `python/`: the Python types of the settings, generated from `xsd/` by `_generate_python_schema.py`
"""

from pathlib import Path

# The directory of XML schemas, which each root element's `XSD_PATH` is relative to
XSD_DIRECTORY_PATH: Path = Path(__file__).resolve().parent / "xsd"
