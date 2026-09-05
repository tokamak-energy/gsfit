import matplotlib.axes
import numpy as np

from gsfit_rs.imas import equilibrium_paths as ep

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
) -> None:
    plasma = gsfit_controller.plasma

    equilibrium_ids = gsfit_controller.plasma.equilibrium_ids

    # `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    r = equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].grid.dim1)
    z = equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].grid.dim2)
    n_r = len(r)
    n_z = len(z)

    for i_z in range(n_z):
        ax.plot([r[0], r[-1]], [z[i_z], z[i_z]], color="black", linestyle="solid", linewidth=0.5)
    for i_r in range(n_r):
        ax.plot([r[i_r], r[i_r]], [z[0], z[-1]], color="black", linestyle="solid", linewidth=0.5)

    # `profiles_2d/r` and `/z` are the (R, Z) mesh; flattening them row-major gives the flat grid
    flat_r = np.asarray(equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].r)).flatten()
    flat_z = np.asarray(equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].z)).flatten()
    ax.plot(flat_r, flat_z, color="black", linestyle="none", marker="o", markersize=0.5)
