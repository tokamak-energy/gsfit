import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_coils(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> Coils:
    """
    This method initialises the Rust `Coils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the coils class
    coils = Coils()

    for pf_coil in freegs_eqs[0].tokamak.coils:
        # if isinstance(pf_coil, freegs.multi_coil.MultiCoil):
        coil_name = pf_coil[0]
        coil_r = pf_coil[1].Rfil
        coil_z = pf_coil[1].Zfil
        coil_d_r = pf_coil[1].Rfil * 0.0 + 1e-3
        coil_d_z = pf_coil[1].Zfil * 0.0 + 1e-3
        current = pf_coil[1].current
        # elif isinstance(pf_coil[1], freegs.machine.Coil):
        #     coil_name = pf_coil[0]
        #     coil_r = np.array([pf_coil[1].R])
        #     coil_z = np.array([pf_coil[1].Z])
        #     coil_d_r = np.array([1e-3])
        #     coil_d_z = np.array([1e-3])
        #     current = pf_coil[1].current * pf_coil[1].turns
        # elif isinstance(pf_coil[1], freegs.machine.Solenoid):
        #     coil_name = pf_coil[0]
        #     Rs = pf_coil[1].Rs
        #     Zsmin = pf_coil[1].Zsmin
        #     Zsmax = pf_coil[1].Zsmax
        #     n_turns = pf_coil[1].Ns
        #     coil_r = np.full(n_turns, Rs)
        #     coil_z = np.linspace(Zsmin, Zsmax, n_turns)
        #     coil_d_r = np.full(n_turns, 1e-3)
        #     coil_d_z = np.full(n_turns, 1e-3)
        #     current = pf_coil[1].current
        # else:
        #     raise ValueError(f"Unknown coil type: {pf_coil[1]}")

        coils.add_pf_coil(
            name=coil_name,
            r=coil_r,
            z=coil_z,
            d_r=coil_d_r,
            d_z=coil_d_z,
            time=np.array([0.0, 1.0]),
            measured=np.array([current, current]),
        )

    # Add TF coil
    coils.add_tf_coil(
        time=np.array([0.0, 1.0]),
        measured=np.array([1.0e6, 1.0e6]),
    )

    return coils
