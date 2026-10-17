import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Wall
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.wall.wall import WallSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


# Merging-compression tiles. These protrude inside the vacuum vessel contour, so the plasma
# can be limited on them. `MCT` is the top tile and `MCB` the bottom, named after the
# merging-compression coils they sit in front of
# TODO: need to add these into MDSplus.
# But first need to re-formulate MDSplus schema to allow different limiter units
MCT_TILES_R = np.array([0.7103])  # [metre]
MCT_TILES_Z = np.array([0.3031])  # [metre]
MCB_TILES_R = np.array([0.7103])  # [metre]
MCB_TILES_Z = np.array([-0.3131])  # [metre]


def setup_wall(
    self: "DatabaseReader",
    pulseNo: int,
    settings: WallSettings,
) -> Wall:
    """
    This method initialises the Rust `Wall` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `WallSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ELMAG run to read

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ELMAG run to read, from the `workflow.xml` which `wall.xml` links to
    workflow = settings.workflow
    if workflow is None or "ELMAG" not in workflow.tree:
        raise ValueError("`wall.xml` must link to a `workflow.xml` holding the ELMAG `tree`, which says which ELMAG run to read the limiter from")
    # A nil `pulseNo` reads the pulse being reconstructed
    elmag_settings = workflow.tree["ELMAG"]
    if elmag_settings.pulseNo is None:
        elmag_pulseNo = pulseNo
    else:
        elmag_pulseNo = elmag_settings.pulseNo

    # The wall's settings are those of `wall.xml` holding only the run which built the wall
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=WallSettings(workflow=Workflow(tree=KeyedList([elmag_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Vacuum vessel contour, which bounds the region the plasma is allowed to occupy
    elmag = GetData(elmag_pulseNo, f"ELMAG#{elmag_settings.run_name}", is_fail_quiet=False)
    vacuum_vessel_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.R"))
    vacuum_vessel_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.Z"))

    # Initialise the Wall Rust class
    wall = Wall(settings_xml)

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=vacuum_vessel_r, z=vacuum_vessel_z)
    wall.add_limiter_unit(name="mct_tiles", r=MCT_TILES_R, z=MCT_TILES_Z)
    wall.add_limiter_unit(name="mcb_tiles", r=MCB_TILES_R, z=MCB_TILES_Z)

    return wall
