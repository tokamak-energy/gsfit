from typing import Any

import pytest
from gsfit import Gsfit

test_parameters: list[dict[str, Any]] = [
    {  # 1st test - not much
        "pulseNo": 12050,
        "run_name": "TEST0",
        "run_description": "",
        "write_to_mds": False,
    },
]


@pytest.mark.parametrize("test_parameter", test_parameters)
def test_initialisation(test_parameter: dict[str, Any]) -> None:
    gsfit_controller = Gsfit(
        pulseNo=test_parameter["pulseNo"],
        run_name=test_parameter["run_name"],
        run_description=test_parameter["run_description"],
        write_to_mds=test_parameter["write_to_mds"],
    )

    # Run
    gsfit_controller.run()
