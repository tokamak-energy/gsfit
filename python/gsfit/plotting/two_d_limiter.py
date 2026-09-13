import matplotlib.axes
from gsfit_rs.imas import wall_paths as wp

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes) -> None:
    # Every limiter unit is a plasma facing surface; `unit(0)` is the vacuum vessel contour and the
    # rest are tiles, so they are drawn one at a time rather than concatenated into one line
    n_unit = len(gsfit_controller.wall.get(wp.description_2d[0].limiter.unit[:].name))
    for i_unit in range(n_unit):
        limit_pts_r = gsfit_controller.wall.get(wp.description_2d[0].limiter.unit[i_unit].outline.r)
        limit_pts_z = gsfit_controller.wall.get(wp.description_2d[0].limiter.unit[i_unit].outline.z)
        ax.plot(limit_pts_r, limit_pts_z, color="black", linewidth=0.5)
