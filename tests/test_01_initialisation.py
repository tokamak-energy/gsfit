import gsfit_rs
from gsfit_rs.dd_support import ParametersInput


def test_01_initialisation() -> None:
    pf_active = gsfit_rs.PfActive(ParametersInput())
    print(pf_active)
