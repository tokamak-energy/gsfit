import matplotlib.axes
import matplotlib.pyplot as plt

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes) -> None:
    ax.set_aspect("equal")

    plasma = gsfit_controller.plasma

    grid_r = plasma.get_array1(["grid", "flat", "r"])
    grid_z = plasma.get_array1(["grid", "flat", "z"])

    ax.plot(grid_r, grid_z, linestyle="", marker="o", color="black", markersize=0.75)
