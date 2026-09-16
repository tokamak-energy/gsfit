import matplotlib.axes
import numpy as np

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float | None = None,
) -> None:
    """
    Plot the poloidal-field (PF) coil cross-sections.

    The PF coils are stored as axis-aligned rectangular filaments in the `coils` Rust data
    structure under `["pf", <coil_name>, "geometry", "r"/"z"/"d_r"/"d_z"]` [metre].
    Colours follow the ST40 physics viewer (`plot_elmag.py`).

    :param gsfit_controller: the Gsfit controller holding the populated `coils` object
    :param ax: the matplotlib axes to draw on
    :param time_desired: unused; kept for API symmetry with the other `two_d_*` plotters
    """
    coils = gsfit_controller.coils

    # Fixed PF-coil colours (match plot_elmag.py)
    facecolor: str = "#F1C40F"
    edgecolor: str = "#7D6608"

    for pf_coil_name in coils.keys(["pf"]):
        filaments_r = coils.get_array1(["pf", pf_coil_name, "geometry", "r"])  # [metre]
        filaments_z = coils.get_array1(["pf", pf_coil_name, "geometry", "z"])  # [metre]
        filaments_d_r = coils.get_array1(["pf", pf_coil_name, "geometry", "d_r"])  # [metre]
        filaments_d_z = coils.get_array1(["pf", pf_coil_name, "geometry", "d_z"])  # [metre]

        n_filaments: int = filaments_r.shape[0]
        for i_filament in range(n_filaments):
            corner_r = filaments_r[i_filament] + filaments_d_r[i_filament] * np.array([-0.5, 0.5, 0.5, -0.5])
            corner_z = filaments_z[i_filament] + filaments_d_z[i_filament] * np.array([-0.5, -0.5, 0.5, 0.5])
            ax.fill(
                corner_r,
                corner_z,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.5,
                zorder=2,
            )
