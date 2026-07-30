import typing
from typing import TYPE_CHECKING

from gsfit_rs import IsofluxBoundary

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_isoflux_boundary_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> IsofluxBoundary:
    """
    This method initialises the Rust `IsofluxBoundary` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    The mock only stores magnetics, so an empty `IsofluxBoundary` class is returned.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the IsofluxBoundary Rust class
    isoflux_boundary = IsofluxBoundary()

    return isoflux_boundary
