import matplotlib.axes
import numpy as np

from gsfit_rs.imas import equilibrium_paths as ep

from ..gsfit import Gsfit


def plot(gsfit_controller: Gsfit, ax: matplotlib.axes.Axes, time_desired: float, color: str = "blue", linestyle: str = "solid") -> None:
    plasma = gsfit_controller.plasma

    equilibrium_ids = gsfit_controller.plasma.equilibrium_ids

    time = equilibrium_ids.get(ep.time_slice[:].time)
    i_time = int(np.argmin(np.abs(time - time_desired)))

    psi_n = equilibrium_ids.get(ep.time_slice[0].profiles_1d.psi_norm)
    p_prime = equilibrium_ids.get(ep.time_slice[i_time].profiles_1d.dpressure_dpsi)

    ax.plot(psi_n, p_prime, color=color, linestyle=linestyle, label="GSFit", linewidth=1.0)
