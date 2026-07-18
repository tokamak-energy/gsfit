import matplotlib.axes
import numpy as np
import numpy.typing as npt

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float,
    color: str = "blue",
    linestyle: str = "dashed",
    psi_n_levels: npt.NDArray[np.float64] | None = None,
) -> None:
    plasma = gsfit_controller.plasma

    time = plasma.get_array1(["time"])
    i_time = np.argmin(np.abs(time - time_desired))

    gsfit_r = plasma.get_array1(["grid", "r"])
    gsfit_z = plasma.get_array1(["grid", "z"])
    gsfit_psi = plasma.get_array3(["profiles_2d", "r_z", "psi"])[i_time, :, :]

    gsfit_nbnd = plasma.get_vec_usize(["boundary", "outline", "n"])[i_time]
    gsfit_boundary_r = plasma.get_array2(["boundary", "outline", "r"])[i_time, :gsfit_nbnd]
    gsfit_boundary_z = plasma.get_array2(["boundary", "outline", "z"])[i_time, :gsfit_nbnd]

    # Default to 35 levels if not provided
    if psi_n_levels is None:
        psi_n_levels = np.linspace(np.min(gsfit_psi), np.max(gsfit_psi), 35)

    ax.contour(gsfit_r, gsfit_z, gsfit_psi, levels=psi_n_levels, colors=color, linewidths=0.6, linestyles=linestyle)
    ax.plot(gsfit_boundary_r, gsfit_boundary_z, color=color, linestyle=linestyle, label="GSFit", linewidth=1.0)
