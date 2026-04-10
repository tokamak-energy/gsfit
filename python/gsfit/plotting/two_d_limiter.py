import matplotlib.axes
import matplotlib.pyplot as plt

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes) -> None:
    ax.set_aspect("equal")

    plasma = gsfit_controller.plasma
    limit_pts_r = plasma.get_array1(["limiter", "limit_pts", "r"])
    limit_pts_z = plasma.get_array1(["limiter", "limit_pts", "z"])

    ax.plot(limit_pts_r, limit_pts_z, color="black", linewidth=0.5)
