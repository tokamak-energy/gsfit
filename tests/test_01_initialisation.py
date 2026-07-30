from typing import Any

import gsfit_rs


def test_01_initialisation() -> None:
    coils = gsfit_rs.Coils()
    print(coils)
