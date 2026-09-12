import matplotlib.axes
import numpy as np
from gsfit_rs.imas import wall_paths as wp

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes) -> None:
    wall_ids = gsfit_controller.wall.wall_ids

    # Every limiter unit is a plasma facing surface; `unit(0)` is the vacuum vessel contour and the
    # rest are tiles, so they are drawn one at a time rather than concatenated into one line
    n_unit = len(np.atleast_1d(wall_ids.get(wp.description_2d[0].limiter.unit[:].outline.r)))
    for i_unit in range(n_unit):
        limit_pts_r = wall_ids.get(wp.description_2d[0].limiter.unit[i_unit].outline.r)
        limit_pts_z = wall_ids.get(wp.description_2d[0].limiter.unit[i_unit].outline.z)
        ax.plot(limit_pts_r, limit_pts_z, color="black", linewidth=0.5)
