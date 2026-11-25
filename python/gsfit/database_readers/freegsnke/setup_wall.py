import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
from gsfit_rs import Wall
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.wall.wall import WallSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_wall(
    self: "DatabaseReader",
    pulseNo: int,
    settings: WallSettings,
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
    **kwargs: typing.Any,
) -> Wall:
    """
    This method initialises the Rust `Wall` class, which holds an IMAS `wall` IDS.

    :param pulseNo: Pulse number. Not used: the limiter is read from the FreeGSNKE equilibria
    :param settings: `WallSettings` object containing the XML settings read from the `settings` directory
    :param time: The timebase of `freegsnke_eqs` [second]
    :param freegsnke_eqs: The FreeGSNKE equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGSNKE.**

    FreeGSNKE describes the limiter with a single contour, so it is the one limiter unit, the vacuum vessel contour.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the wall records in `wall/code/parameters`
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=settings,
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Vacuum vessel contour, which bounds the region the plasma is allowed to occupy. The machine does not change in time, so it is read from
    # the first equilibrium
    freegsnke_tokamak = freegsnke_eqs[0].tokamak
    vacuum_vessel_r: npt.NDArray[np.float64] = np.asarray(freegsnke_tokamak.limiter.R, dtype=np.float64)  # [metre]
    vacuum_vessel_z: npt.NDArray[np.float64] = np.asarray(freegsnke_tokamak.limiter.Z, dtype=np.float64)  # [metre]

    # Initialise the Wall Rust class
    wall: Wall = Wall(settings_xml)

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)

    return wall
