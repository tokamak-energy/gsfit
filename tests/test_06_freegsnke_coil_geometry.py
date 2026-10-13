"""FreeGSNKE PF filament geometry is preserved when importing coils."""

from unittest.mock import Mock

import freegs4e
import gsfit_rs
import numpy as np
import pytest
from freegs4e.multi_coil import MultiCoil
from freegsnke.equilibrium_update import Equilibrium

from gsfit.database_readers.freegsnke import DatabaseReader


def make_equilibrium(coil: MultiCoil, current: float) -> Mock:
    circuit = freegs4e.machine.Circuit([("PF", coil, 1.0)], current=current)
    equilibrium = Mock(spec=Equilibrium)
    equilibrium.tokamak = Mock(coils=[("circuit", circuit)])
    return equilibrium


def make_coil() -> MultiCoil:
    coil = MultiCoil([1.0, 1.02], [0.1, 0.12])
    coil.dR = 0.012
    coil.dZ = 0.018
    return coil


def test_physical_filament_dimensions_and_currents_are_preserved() -> None:
    coil = make_coil()
    coils = DatabaseReader().setup_coils(
        pulseNo=0,
        settings={},
        time=np.array([0.0, 1.0]),
        freegsnke_eqs=[make_equilibrium(coil, 100.0), make_equilibrium(coil, 200.0)],
    )
    coil_r = coils.get_array1(["pf", "PF", "geometry", "r"])
    coil_z = coils.get_array1(["pf", "PF", "geometry", "z"])
    coil_d_r = coils.get_array1(["pf", "PF", "geometry", "d_r"])
    coil_d_z = coils.get_array1(["pf", "PF", "geometry", "d_z"])
    np.testing.assert_array_equal(coil_r, coil.Rfil)
    np.testing.assert_array_equal(coil_z, coil.Zfil)
    np.testing.assert_array_equal(coil_d_r, np.full(2, 0.012))
    np.testing.assert_array_equal(coil_d_z, np.full(2, 0.018))
    np.testing.assert_array_equal(coils.get_array1(["pf", "PF", "i", "experimental", "value"]), np.array([100.0, 200.0]))
    greens = gsfit_rs.greens_py(coil_r, coil_z, coil_r, coil_z, coil_d_r, coil_d_z)
    assert np.isfinite(greens).all()


@pytest.mark.parametrize("dimension", ["dR", "dZ"])
@pytest.mark.parametrize("value", [0.0, -0.01, np.nan, np.inf])
def test_invalid_filament_dimensions_are_rejected(dimension: str, value: float) -> None:
    coil = make_coil()
    setattr(coil, dimension, value)
    with pytest.raises(ValueError, match="PF.*finite, positive filament dimensions"):
        DatabaseReader().setup_coils(
            pulseNo=0, settings={}, time=np.array([0.0]), freegsnke_eqs=[make_equilibrium(coil, 100.0)]
        )


@pytest.mark.parametrize("dimension", ["dR", "dZ"])
def test_missing_filament_dimensions_are_rejected(dimension: str) -> None:
    coil = make_coil()
    delattr(coil, dimension)
    with pytest.raises(ValueError, match="PF.*missing physical filament dimensions"):
        DatabaseReader().setup_coils(
            pulseNo=0, settings={}, time=np.array([0.0]), freegsnke_eqs=[make_equilibrium(coil, 100.0)]
        )
