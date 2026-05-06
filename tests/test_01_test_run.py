from typing import Any

import pytest
from gsfit import Gsfit
import gsfit_rs

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
    coils = gsfit_rs.Coils()
    print(coils)
