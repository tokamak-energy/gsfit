import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
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
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
    **kwargs: typing.Any,
) -> Wall:
    """
    This method initialises the Rust `Wall` class, which holds an IMAS `wall` IDS.

    :param pulseNo: Pulse number. Not used: the limiter is read from the FreeGS equilibria
    :param settings: `WallSettings` object containing the XML settings read from the `settings` directory. Not used: the limiter is read from
        the FreeGS equilibria
    :param time: The timebase of `freegs_eqs` [second]
    :param freegs_eqs: The FreeGS equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGS.**

    FreeGS describes the limiter with a single contour, its wall, so it is the one limiter unit, the vacuum vessel contour.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the wall records in `wall/code/parameters`: none, as the wall is read from the FreeGS equilibria
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=WallSettings(),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Vacuum vessel contour, which bounds the region the plasma is allowed to occupy. The machine does not change in time, so it is read from
    # the first equilibrium
    freegs_tokamak = freegs_eqs[0].tokamak
    vacuum_vessel_r: npt.NDArray[np.float64] = np.asarray(freegs_tokamak.wall.R, dtype=np.float64)  # [metre]
    vacuum_vessel_z: npt.NDArray[np.float64] = np.asarray(freegs_tokamak.wall.Z, dtype=np.float64)  # [metre]

    # Initialise the Wall Rust class
    wall: Wall = Wall(settings_xml)

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)

    return wall
