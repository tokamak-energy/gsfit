"""Shared pytest configuration for the test suite."""

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
