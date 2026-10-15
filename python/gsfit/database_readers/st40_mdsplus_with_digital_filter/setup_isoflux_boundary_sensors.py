from typing import TYPE_CHECKING

from gsfit_rs import IsofluxBoundary

from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_isoflux_boundary_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
) -> IsofluxBoundary:
    """
    This method initialises the Rust `IsofluxBoundary` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object, read from `magnetics/magnetics.xml`; not used

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the IsofluxBoundary Rust class
    isoflux_boundary = IsofluxBoundary()

    # TODO: implment the method

    return isoflux_boundary
