from typing import Any

import pytest

from gsfit import Gsfit
from gsfit import Gsfit_Ts_1
from gsfit import Gsfit_Ts_2


def make_controller_settings() -> dict[str, Any]:
    return {
        "GSFIT_code_settings.json": {"timeslices": {"method": "all"}},
        "sensor_weights_pressure.json": {"include": False},
        "source_function_p_prime.json": {
            "method": "tensioned_cubic_b_spline",
            "pressure_edge": {"free": False, "regularisation_weight": 1.0e-4},
        },
    }


def test_gsfit_ts_classes_configure_efit_and_edge_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    init_kwargs: list[dict[str, object]] = []

    def initialize(self: Gsfit, **kwargs: object) -> None:
        init_kwargs.append(kwargs)
        self.settings = make_controller_settings()

    monkeypatch.setattr(Gsfit, "__init__", initialize)

    fixed_edge = Gsfit_Ts_1(pulseNo=12345, run_name="RUN01")
    free_edge = Gsfit_Ts_2(pulseNo=12345, run_name="RUN02")

    assert [kwargs["analysis_name"] for kwargs in init_kwargs] == ["GSFIT_TS", "GSFIT_TS"]
    for controller in (fixed_edge, free_edge):
        assert controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] == "good_pressure_sensors"
        assert controller.settings["sensor_weights_pressure.json"]["include"] is True
        assert controller.settings["source_function_p_prime.json"]["method"] == "efit_polynomial"

    assert fixed_edge.settings["source_function_p_prime.json"]["pressure_edge"]["free"] is False
    assert free_edge.settings["source_function_p_prime.json"]["pressure_edge"]["free"] is True
