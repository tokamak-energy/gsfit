import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gsfit_rs import EfitPolynomial, Plasma
from gsfit_rs.imas import equilibrium_paths as ep

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
        ({"free": True}, None, (True, 1.0e-4)),
        ({"free": False, "regularisation_weight": 1.0e-4}, {"free": True, "regularisation_weight": 3.0e-4}, (False, 1.0e-4)),
        (None, {"free": True, "regularisation_weight": 3.0e-4}, (True, 3.0e-4)),
        (None, None, (False, 1.0e-4)),
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


def make_plasma(**edge_settings: float | bool) -> Plasma:
    p_prime = EfitPolynomial(1, np.zeros((0, 1)))
    ff_prime = EfitPolynomial(1, np.zeros((0, 1)))
    return Plasma(
        n_r=3,
        n_z=3,
        r_min=0.5,
        r_max=0.7,
        z_min=-0.1,
        z_max=0.1,
        psi_norm=np.linspace(0.0, 1.0, 3),
        p_prime_source_function=p_prime,
        ff_prime_source_function=ff_prime,
        initial_guess_ip=1.0e4,
        initial_guess_cur_r=0.6,
        initial_guess_cur_z=0.0,
        initial_guess_minor_radius=0.05,
        initial_guess_elongation=1.0,
        n_iter_max=5,
        n_iter_min=1,
        n_iter_no_vertical_feedback=0,
        grad_shafranov_deviation_tolerance=1.0e-5,
        use_anderson_mixing=False,
        anderson_mixing_from_previous_iter=0.0,
        times_to_reconstruct=np.array([0.1]),
        **edge_settings,
    )


def test_free_edge_pressure_uses_positive_constructor_default() -> None:
    plasma = make_plasma(pressure_edge_free=True)
    assert plasma.get(ep.code.numerics.pressure_edge.free) == 1
    assert plasma.get(ep.code.numerics.pressure_edge.regularisation_weight) == 1.0e-4


@pytest.mark.parametrize("weight", [0.0, -1.0e-4, float("nan"), float("inf"), -float("inf")])
def test_free_edge_pressure_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="pressure_edge_regularisation_weight must be positive and finite"):
        make_plasma(pressure_edge_free=True, pressure_edge_regularisation_weight=weight)


def test_fixed_edge_pressure_does_not_require_prior() -> None:
    plasma = make_plasma(pressure_edge_free=False, pressure_edge_regularisation_weight=0.0)
    assert plasma.get(ep.code.numerics.pressure_edge.free) == 0
