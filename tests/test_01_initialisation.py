from typing import Any

import gsfit_rs

def test_01_initialisation(test_parameter: dict[str, Any]) -> None:
    coils = gsfit_rs.Coils()
    print(coils)
