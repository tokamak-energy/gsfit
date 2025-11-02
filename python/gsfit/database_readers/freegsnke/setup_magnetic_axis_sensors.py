import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium  # type: ignore
from gsfit_rs import MagneticAxis

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_magnetic_axis_sensors(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
) -> MagneticAxis:
    """
    This method initialises the Rust `MagneticAxis` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

        **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the MagneticAxis Rust class
    magnetic_axis = MagneticAxis()

    return magnetic_axis
