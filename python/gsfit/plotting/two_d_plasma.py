import matplotlib.axes
import numpy as np
import numpy.typing as npt

from gsfit_rs.imas import equilibrium_paths as ep

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float,
    color: str = "blue",
    linestyle: str = "dashed",
    psi_n_levels: npt.NDArray[np.float64] | None = None,
) -> None:
    equilibrium_ids = gsfit_controller.plasma.equilibrium_ids

    time = equilibrium_ids.get(ep.time_slice[:].time)
    i_time = int(np.argmin(np.abs(time - time_desired)))

    # `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    gsfit_r = equilibrium_ids.get(ep.time_slice[i_time].profiles_2d[0].grid.dim1)
    gsfit_z = equilibrium_ids.get(ep.time_slice[i_time].profiles_2d[0].grid.dim2)
    gsfit_psi = equilibrium_ids.get(ep.time_slice[i_time].profiles_2d[0].psi)

    # The boundary is stored at its own length for each time-slice, so there is no padding to trim
    gsfit_boundary_r = equilibrium_ids.get(ep.time_slice[i_time].boundary.outline.r)
    gsfit_boundary_z = equilibrium_ids.get(ep.time_slice[i_time].boundary.outline.z)

    # Default to 35 levels if not provided
    if psi_n_levels is None:
        psi_n_levels = np.linspace(np.min(gsfit_psi), np.max(gsfit_psi), 35)

    ax.contour(gsfit_r, gsfit_z, gsfit_psi, levels=psi_n_levels, colors=color, linewidths=0.6, linestyles=linestyle)
    ax.plot(gsfit_boundary_r, gsfit_boundary_z, color=color, linestyle=linestyle, label="GSFit", linewidth=1.0)
