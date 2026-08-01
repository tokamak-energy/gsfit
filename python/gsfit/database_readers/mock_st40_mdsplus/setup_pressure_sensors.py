import typing
from typing import TYPE_CHECKING

from gsfit_rs import Pressure

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    The mock only stores magnetics, so an empty `Pressure` class is returned.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    return pressure
