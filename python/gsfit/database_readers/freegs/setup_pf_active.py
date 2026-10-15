import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from gsfit_rs import PfActive
from gsfit_rs.dd_support import ParametersInput

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to FreeGS.**

    One `coil(i1)` is added for each FreeGS coil, with one `element(i2)` per filament, carrying the coil's current from the first equilibrium.
    FreeGS does not store the size of a filament, so each is given a nominal 1 mm square.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the PfActive Rust class
    # This reader still reads the JSON settings, so has no XML settings to record in `pf_active/code/parameters` yet
    pf_active = PfActive(ParametersInput())

    for pf_coil in freegs_eqs[0].tokamak.coils:
        # if isinstance(pf_coil, freegs.multi_coil.MultiCoil):
        coil_name = pf_coil[0]
        coil_r = pf_coil[1].Rfil
        coil_z = pf_coil[1].Zfil
        coil_width = pf_coil[1].Rfil * 0.0 + 1e-3
        coil_height = pf_coil[1].Zfil * 0.0 + 1e-3
        current = pf_coil[1].current
        # elif isinstance(pf_coil[1], freegs.machine.Coil):
        #     coil_name = pf_coil[0]
        #     coil_r = np.array([pf_coil[1].R])
        #     coil_z = np.array([pf_coil[1].Z])
        #     coil_width = np.array([1e-3])
        #     coil_height = np.array([1e-3])
        #     current = pf_coil[1].current * pf_coil[1].turns
        # elif isinstance(pf_coil[1], freegs.machine.Solenoid):
        #     coil_name = pf_coil[0]
        #     Rs = pf_coil[1].Rs
        #     Zsmin = pf_coil[1].Zsmin
        #     Zsmax = pf_coil[1].Zsmax
        #     n_turns = pf_coil[1].Ns
        #     coil_r = np.full(n_turns, Rs)
        #     coil_z = np.linspace(Zsmin, Zsmax, n_turns)
        #     coil_width = np.full(n_turns, 1e-3)
        #     coil_height = np.full(n_turns, 1e-3)
        #     current = pf_coil[1].current
        # else:
        #     raise ValueError(f"Unknown coil type: {pf_coil[1]}")

        pf_active.add_pf_coil(
            name=coil_name,
            r=coil_r,
            z=coil_z,
            width=coil_width,
            height=coil_height,
            time=np.array([0.0, 1.0]),
            current=np.array([current, current]),
        )

    return pf_active
