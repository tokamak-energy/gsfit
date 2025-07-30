import typing
from typing import TYPE_CHECKING

from gsfit_rs import Pressure
from st40_database import GetData  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_pressure_sensors(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    # TODO: Implement this method

    return pressure
