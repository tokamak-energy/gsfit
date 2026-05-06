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

Note: The ``--script`` option and parametrisation hooks live in
``tests/conftest.py`` so that pytest registers them before argument parsing.
"""

import runpy
from pathlib import Path


def test_script(script: str) -> None:
    """Execute *script* as top-level ``__main__`` code and let any exception propagate."""
    path = Path(script).resolve()
    assert path.exists(), f"Script not found: {path}"
    runpy.run_path(str(path), run_name="__main__")
