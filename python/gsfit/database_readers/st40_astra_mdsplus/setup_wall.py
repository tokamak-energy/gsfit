from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Wall
from gsfit_rs.dd_support import ParametersInput
from matplotlib.path import Path as MplPath

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.wall.wall import WallSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_wall(
    self: "DatabaseReader",
    pulseNo: int,
    settings: WallSettings,
) -> Wall:
    """
    This method initialises the Rust `Wall` class, which holds an IMAS `wall` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `WallSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ASTRA run to read

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    The wall is ASTRA's own limiter, `P_BOUNDARY.LIMITER.VESSEL`, which its equilibrium, SPIDER, limited the plasma with, from ASTRA's
    `exp/equ/MCVC/limpnt.dat`. It lists any separate limit points first, then the vacuum vessel's contour, which starts on the inboard midplane:
    * `unit(0)` is the contour, closed, as the solver takes it as the region the plasma is allowed to occupy;
    * `unit(1)` holds the separate limit points which are inside the contour. Those outside it, e.g. at (0.971, +-0.31) m, could never limit a
      plasma inside the contour, and are left out, as the solver takes every point of every unit as a candidate limit point.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA run to read, from the `workflow.xml` which `wall.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree:
        raise ValueError("`wall.xml` must link to a `workflow.xml` holding the ASTRA `tree`, which says which ASTRA run to read the limiter from")
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo

    # The wall's settings are those of `wall.xml` holding only the run which built the wall
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=WallSettings(workflow=Workflow(tree=KeyedList([astra_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    conn: mdsthin.Connection = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", astra_pulseNo)
    run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
    # ASTRA stores these as float32; GSFit uses float64
    limiter_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.P_BOUNDARY.LIMITER.VESSEL:R").data().astype(np.float64)  # [metre]
    limiter_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.P_BOUNDARY.LIMITER.VESSEL:Z").data().astype(np.float64)  # [metre]

    # The contour starts at the first point on the inboard midplane: at the smallest major radius, and `Z = 0`
    n_limiter_point: int = len(limiter_r)
    i_contour_start: int | None = None
    for i_limiter_point in range(n_limiter_point):
        if i_contour_start is None and limiter_r[i_limiter_point] == np.min(limiter_r) and limiter_z[i_limiter_point] == 0.0:
            i_contour_start = i_limiter_point
    if i_contour_start is None:
        raise ValueError(f"`{run_path}.P_BOUNDARY.LIMITER.VESSEL` has no point on the inboard midplane, where the vacuum vessel's contour starts")
    contour_r: npt.NDArray[np.float64] = np.append(limiter_r[i_contour_start:], limiter_r[i_contour_start])  # [metre]
    contour_z: npt.NDArray[np.float64] = np.append(limiter_z[i_contour_start:], limiter_z[i_contour_start])  # [metre]

    # The separate limit points which are inside the contour
    separate_points_r: npt.NDArray[np.float64] = limiter_r[:i_contour_start]  # [metre]
    separate_points_z: npt.NDArray[np.float64] = limiter_z[:i_contour_start]  # [metre]
    inside_contour: npt.NDArray[np.bool_] = MplPath(np.column_stack((contour_r, contour_z))).contains_points(
        np.column_stack((separate_points_r, separate_points_z))
    )

    # Initialise the Wall Rust class
    wall: Wall = Wall(settings_xml)

    # `unit(0)` must be the vacuum vessel contour: the solver reads the limiter units by
    # position, and uses the first one, and only the first one, as the region the plasma is
    # allowed to occupy. Every unit contributes candidate limit points
    wall.add_limiter_unit(name="vacuum_vessel", r=contour_r, z=contour_z)
    if np.any(inside_contour):
        wall.add_limiter_unit(name="limit_points", r=separate_points_r[inside_contour], z=separate_points_z[inside_contour])

    return wall
