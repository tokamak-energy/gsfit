import typing
from typing import TYPE_CHECKING

import freegsnke  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import Dialoop

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_dialoop(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> Dialoop:
    """
    This method initialises the Rust `Dialoop` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Isoflux Rust class
    dialoop = Dialoop()

    # TODO: Implement this method

    return dialoop
