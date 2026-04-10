import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes, time_desired: float, color: str = "blue", linestyle: str = "solid") -> None:
    plasma = gsfit_controller.plasma

    time = plasma.get_array1(["time"])
    i_time = np.argmin(np.abs(time - time_desired))

    psi_n = plasma.get_array1(["profiles", "psi_n"])
    ff_prime = plasma.get_array2(["profiles", "ff_prime"])[i_time, :] / mu_0

    ax.plot(psi_n, ff_prime, color=color, linestyle=linestyle, label="GSFit", linewidth=1.0)
