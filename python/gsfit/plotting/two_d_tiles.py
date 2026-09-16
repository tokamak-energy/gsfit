import typing

import matplotlib.axes
import numpy as np
import numpy.typing as npt

from ..gsfit import Gsfit
from .filament_corners import filament_corners


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float | None = None,
) -> None:
    """
    Plot the plasma facing tiles.

    The tile geometry is not stored in any IDS, so it is read directly from ST40's ELMAG MDSplus tree
    under `LIMITER.TILES`, using the `elmag` workflow of the selected `database_reader`. The tiles are
    (possibly sheared) parallelograms; the corner construction and colours follow the ST40 physics
    viewer (`plot_elmag.py`).

    **This plotter is specific to ST40's experimental MDSplus database.**

    :param gsfit_controller: the Gsfit controller, whose `pulseNo` and settings select the ELMAG tree
    :param ax: the matplotlib axes to draw on
    :param time_desired: unused; kept for API symmetry with the other `two_d_*` plotters
    """
    # `st40_database` is only installed with the `with_st40_mdsplus` extra, and every plotter is
    # imported by `gsfit.plotting`, so it is imported here rather than at the top of the file
    from st40_database import GetData

    database_reader_settings = gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]
    database_reader_method = database_reader_settings["method"]
    workflow = database_reader_settings.get(database_reader_method, {}).get("workflow", {})
    if "elmag" not in workflow:
        raise ValueError(f"The tiles are read from ST40's ELMAG tree, but database_reader={database_reader_method} has no `elmag` workflow")

    # `pulseNo = null` means "use the shot's pulse"
    if workflow["elmag"]["pulseNo"] is None:
        elmag_pulseNo = gsfit_controller.pulseNo
    else:
        elmag_pulseNo = workflow["elmag"]["pulseNo"]
    elmag_run_name = workflow["elmag"]["run_name"]
    elmag = GetData(elmag_pulseNo, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)

    tiles_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.R"))  # [metre]
    tiles_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.Z"))  # [metre]
    tiles_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.DR"))  # [metre]
    tiles_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.DZ"))  # [metre]
    tiles_angle_1 = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.ANGLE1"))  # [radian]
    tiles_angle_2 = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.TILES.ANGLE2"))  # [radian]
    tile_names = typing.cast(list[str], elmag.get("LIMITER.TILES.NAMES"))

    # The geometry can have more entries than there are names: ELMAG#RUN16 has 54 geometries but
    # 53 names, and the unnamed last geometry is a copy of `MCB1`. Only the named tiles are drawn,
    # which is also what the physics viewer ends up drawing
    n_tiles: int = len(tile_names)
    for i_tile in range(n_tiles):
        tile_name: str = tile_names[i_tile]

        corner_r, corner_z = filament_corners(
            filament_r=tiles_r[i_tile],
            filament_z=tiles_z[i_tile],
            filament_d_r=tiles_d_r[i_tile],
            filament_d_z=tiles_d_z[i_tile],
            filament_angle_1=tiles_angle_1[i_tile],
            filament_angle_2=tiles_angle_2[i_tile],
        )

        # Colours match `plot_elmag.py`
        facecolor: str
        if "INNER" in tile_name:
            # Inner divertor tiles
            facecolor = "#657B00"
        elif "OUTER" in tile_name:
            # Outer divertor tiles
            facecolor = "#810D8E"
        elif "CP" in tile_name:
            # Centre post tiles
            facecolor = "#4C7478"
        elif "MCT" in tile_name or "MCB" in tile_name:
            # Merging-compression tiles, top and bottom
            facecolor = "#855784"
        else:
            # High field side tiles and molybdenum plates
            facecolor = "#CF9966"

        # The molybdenum plates are too thin to see when only filled, so they are also outlined.
        # The other tiles have no edge, so that they are not drawn larger than reality
        edgecolor: str
        if "MOLY" in tile_name:
            edgecolor = facecolor
        else:
            edgecolor = "none"

        ax.fill(
            corner_r,
            corner_z,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            zorder=2,
        )
