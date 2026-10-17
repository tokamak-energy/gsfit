import typing

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Magnetics
from st40_database import GetData


def add_to_magnetics_diamagnetic_flux(
    magnetics: Magnetics,
    dialoop: GetData,
) -> None:
    """
    Add the diamagnetic flux to `magnetics/diamagnetic_flux`, as the method `DIALOOP`.

    :param magnetics: the `Magnetics` the diamagnetic flux is added to
    :param dialoop: the DIALOOP run the diamagnetic flux is read from, opened with `GetData`

    **This method is specific to ST40's experimental MDSplus database.**

    Not every pulse has a DIALOOP run. When MDSplus cannot read it, nothing is added, and the reconstruction runs without the diamagnetic flux.

    The diamagnetic loop's path, which `Dialoop` keeps for reference only, is not added: `magnetics/diamagnetic_flux` has no node for it.
    """

    # Measured signal: \DIALOOP::TOP.<run_name>.GLOBAL:PHI_DIA
    try:
        time: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float64], dialoop.get("TIME")).astype(np.float64)  # [second]
        data: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float64], dialoop.get("GLOBAL.PHI_DIA")).astype(np.float64)  # [weber]
    except mdsthin.MdsException as exception:
        print(f"DIALOOP could not be read, so `magnetics/diamagnetic_flux` is left empty: {exception}")
        return

    magnetics.add_diamagnetic_flux(
        method_name="DIALOOP",
        time=time,
        data=data,
    )
