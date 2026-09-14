import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
from gsfit_rs import Wall

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_wall(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
) -> Wall:
    """
    This method initialises the Rust `Wall` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Vacuum vessel contour, which bounds the region the plasma is allowed to occupy.
    # FreeGSNKE describes the wall with a single contour, so there are no separate tile units
    freegsnke_tokamak = freegsnke_eqs[0].tokamak  # assume static data does not change in time
    vacuum_vessel_r = np.array(freegsnke_tokamak.limiter.R)
    vacuum_vessel_z = np.array(freegsnke_tokamak.limiter.Z)

    # Initialise the Wall Rust class
    wall = Wall()

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)

    return wall
