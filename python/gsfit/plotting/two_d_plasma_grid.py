import matplotlib.axes
import numpy as np

from gsfit_rs.imas import equilibrium_paths as ep

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes) -> None:
    plasma = gsfit_controller.plasma

    equilibrium_ids = gsfit_controller.plasma.equilibrium_ids

    # `profiles_2d/r` and `/z` are the (R, Z) mesh; flattening them row-major gives the flat grid
    grid_r = np.asarray(equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].r)).flatten()
    grid_z = np.asarray(equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].z)).flatten()

    ax.plot(grid_r, grid_z, linestyle="", marker="o", color="black", markersize=0.75)
