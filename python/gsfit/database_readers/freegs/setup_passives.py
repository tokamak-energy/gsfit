import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from gsfit_rs import Passives
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_passives(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> Passives:
    """
    This method initialises the Rust `Passives` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegs_eqs: List of FreeGS equilibrium objects, one for each time-slice

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Passives Rust class
    passives = Passives()

    return passives
