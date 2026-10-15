import matplotlib.axes
import numpy as np
from gsfit_rs.imas import pf_active_paths as pp

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float | None = None,
    facecolor: str = "#F1C40F",
    edgecolor: str = "#7D6608",
    alpha: float = 1.0,
    linewidth: float = 0.5,
) -> None:
    """
    Plot the poloidal-field (PF) coil cross-sections.

    The PF coils are stored in the `pf_active` IDS, as one rectangular filament per
    `coil(i1)/element(i2)`, at `geometry/rectangle/r`, `z`, `width` and `height` [metre].
    Colours follow the ST40 physics viewer (`plot_elmag.py`).

    :param gsfit_controller: the Gsfit controller holding the populated `pf_active` object
    :param ax: the matplotlib axes to draw on
    :param time_desired: unused; kept for API symmetry with the other `two_d_*` plotters
    :param facecolor: the colour the coil cross-sections are filled with
    :param edgecolor: the colour the coil cross-sections are outlined in
    :param alpha: opacity of the coils, where 1 is opaque and 0 is invisible; useful when one set of
        coils is drawn over another, so that the set underneath still shows through [dimensionless]
    :param linewidth: width of the outline around each filament [point]
    """
    pf_active = gsfit_controller.pf_active

    n_coils: int = len(pf_active.get(pp.coil[:].name))
    for i_coil in range(n_coils):
        filaments_r = pf_active.get(pp.coil[i_coil].element[:].geometry.rectangle.r)  # [metre]
        filaments_z = pf_active.get(pp.coil[i_coil].element[:].geometry.rectangle.z)  # [metre]
        filaments_width = pf_active.get(pp.coil[i_coil].element[:].geometry.rectangle.width)  # [metre]
        filaments_height = pf_active.get(pp.coil[i_coil].element[:].geometry.rectangle.height)  # [metre]

        n_filaments: int = filaments_r.shape[0]
        for i_filament in range(n_filaments):
            corner_r = filaments_r[i_filament] + filaments_width[i_filament] * np.array([-0.5, 0.5, 0.5, -0.5])
            corner_z = filaments_z[i_filament] + filaments_height[i_filament] * np.array([-0.5, -0.5, 0.5, 0.5])
            ax.fill(
                corner_r,
                corner_z,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=2,
            )
