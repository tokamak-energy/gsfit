import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Wall
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReader


# Merging-compression tiles. These protrude inside the vacuum vessel contour, so the plasma
# can be limited on them. `MCT` is the top tile and `MCB` the bottom, named after the
# merging-compression coils they sit in front of
MCT_TILES_R = np.array([0.7103])  # [metre]
MCT_TILES_Z = np.array([0.3031])  # [metre]
MCB_TILES_R = np.array([0.7103])  # [metre]
MCB_TILES_Z = np.array([-0.3131])  # [metre]


def setup_wall(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Wall:
    """
    This method initialises the Rust `Wall` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Vacuum vessel contour, which bounds the region the plasma is allowed to occupy
    elmag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["elmag"]["run_name"]
    elmag = GetData(pulseNo, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)
    vacuum_vessel_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.R"))
    vacuum_vessel_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.Z"))

    # Initialise the Wall Rust class
    wall = Wall()

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)
    wall.add_limiter_unit(name="mct_tiles", r=MCT_TILES_R, z=MCT_TILES_Z)
    wall.add_limiter_unit(name="mcb_tiles", r=MCB_TILES_R, z=MCB_TILES_Z)

    return wall
