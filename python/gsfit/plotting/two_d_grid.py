import matplotlib.axes

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
) -> None:
    plasma = gsfit_controller.plasma

    r = plasma.get_array1(["grid", "r"])
    z = plasma.get_array1(["grid", "z"])
    n_r = len(r)
    n_z = len(z)

    for i_z in range(n_z):
        ax.plot([r[0], r[-1]], [z[i_z], z[i_z]], color="black", linestyle="solid", linewidth=0.5)
    for i_r in range(n_r):
        ax.plot([r[i_r], r[i_r]], [z[0], z[-1]], color="black", linestyle="solid", linewidth=0.5)

    flat_r = plasma.get_array1(["grid", "flat", "r"])
    flat_z = plasma.get_array1(["grid", "flat", "z"])
    ax.plot(flat_r, flat_z, color="black", linestyle="none", marker="o", markersize=0.5)
