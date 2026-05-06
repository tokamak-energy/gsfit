"""Reusable pytest wrapper that runs plain Python scripts as test cases.

Each script's top-level code (including any ``assert`` statements) is
executed inside a proper pytest test function, so failures are reported
with full tracebacks and coverage tools attribute the executed lines
correctly.

Usage
-----
Run a single script::

    python -m pytest -s tests/test_wrapper.py --script examples/example_02__mastu__freegsnke_data.py

Run multiple scripts (each becomes its own parametrised test case)::

    python -m pytest -s tests/test_wrapper.py \\
        --script examples/example_02__mastu__freegsnke_data.py \\
        --script examples/example_05__st40__setup_for_rtgsfit.py

Typical CI workflow::

    jupyter nbconvert --to script examples/my_notebook.ipynb
    python -m pytest -s tests/test_wrapper.py --script examples/my_notebook.py
"""

import runpy
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--script`` command-line option with pytest."""
    parser.addoption("--script", action="append", default=[], help="Path(s) to script(s) to run as tests")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise ``test_script`` once per ``--script`` path provided."""
    if "script" in metafunc.fixturenames:
        scripts = metafunc.config.getoption("script")
        metafunc.parametrize("script", scripts, ids=[Path(s).stem for s in scripts])


def test_script(script: str) -> None:
    """Execute *script* as top-level ``__main__`` code and let any exception propagate."""
    path = Path(script).resolve()
    assert path.exists(), f"Script not found: {path}"
    runpy.run_path(str(path), run_name="__main__")
