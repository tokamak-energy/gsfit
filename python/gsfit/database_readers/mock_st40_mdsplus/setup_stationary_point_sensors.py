import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import StationaryPoint

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_stationary_point_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> StationaryPoint:
    """
    This method initialises the Rust `StationaryPoint` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param times_to_reconstruct: Times to reconstruct the equilibrium

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    The mock only stores magnetics, so an empty `StationaryPoint` class is returned.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the StationaryPoint Rust class
    stationary_points = StationaryPoint()

    return stationary_points
