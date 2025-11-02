import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import MagneticAxis

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_magnetic_axis_sensors(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> MagneticAxis:
    """
    This method initialises the Rust `MagneticAxis` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the MagneticAxis Rust class
    magnetic_axis = MagneticAxis()

    return magnetic_axis
