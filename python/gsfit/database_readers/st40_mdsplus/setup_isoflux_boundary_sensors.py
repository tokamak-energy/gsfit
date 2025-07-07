import typing
from typing import TYPE_CHECKING

import numpy as np
from gsfit_rs import IsofluxBoundary
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_isoflux_boundary_sensors(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> IsofluxBoundary:
    """
    This method initialises the Rust `IsofluxBoundary` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the IsofluxBoundary Rust class
    isoflux_boundary = IsofluxBoundary()

    # TODO: implment the method

    return isoflux_boundary
