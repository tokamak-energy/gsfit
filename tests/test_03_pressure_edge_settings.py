import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gsfit.database_readers.mock_st40_mdsplus.setup_plasma import setup_plasma


SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "python" / "gsfit" / "settings"


@pytest.mark.parametrize("directory", sorted(path for path in SETTINGS_ROOT.iterdir() if (path / "source_function_p_prime.json").exists()))
def test_pressure_edge_settings_in_p_prime_file(directory: Path) -> None:
    p_prime = json.loads((directory / "source_function_p_prime.json").read_text())
    code = json.loads((directory / "GSFIT_code_settings.json").read_text())

    assert p_prime["pressure_edge"]["free"] is False
    assert p_prime["pressure_edge"]["regularisation_weight"] == 1.0e-4
    assert "pressure_edge" not in code["numerics"]


@pytest.mark.parametrize(
    ("pressure_edge", "legacy_pressure_edge", "expected"),
    [
        ({"free": True, "regularisation_weight": 2.0e-4}, None, (True, 2.0e-4)),
        ({"free": False, "regularisation_weight": 1.0e-4}, {"free": True, "regularisation_weight": 3.0e-4}, (False, 1.0e-4)),
        (None, {"free": True, "regularisation_weight": 3.0e-4}, (True, 3.0e-4)),
        (None, None, (False, 0.0)),
    ],
)
def test_plasma_reads_pressure_edge_from_p_prime_settings(
    pressure_edge: dict[str, float | bool] | None,
    legacy_pressure_edge: dict[str, float | bool] | None,
    expected: tuple[bool, float],
) -> None:
    settings = {
        name: json.loads((SETTINGS_ROOT / "default" / name).read_text())
        for name in ("GSFIT_code_settings.json", "source_function_p_prime.json", "source_function_ff_prime.json")
    }
    if legacy_pressure_edge is not None:
        settings["GSFIT_code_settings.json"]["numerics"]["pressure_edge"] = legacy_pressure_edge
    if pressure_edge is None:
        del settings["source_function_p_prime.json"]["pressure_edge"]
    else:
        settings["source_function_p_prime.json"]["pressure_edge"] = pressure_edge

    with patch("gsfit.database_readers.mock_st40_mdsplus.setup_plasma.Plasma") as plasma:
        setup_plasma(None, 0, settings, np.array([0.1]))

    assert plasma.call_args.args[-2:] == expected
