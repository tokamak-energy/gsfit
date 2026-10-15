import matplotlib.axes
import numpy as np
import numpy.typing as npt
from gsfit_rs.imas import pf_passive_paths as pp

from ..gsfit import Gsfit


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float | None = None,
) -> None:
    """
    Plot the passive conducting structures (e.g. the vacuum vessel / IVC).

    Each passive is discretised into filaments, the elements of a `pf_passive/loop(i1)`. Each filament is a (possibly sheared) parallelogram,
    which the data dictionary describes from one corner, the reference point `oblique/r` and `z`: one side, `length_alpha` long, leaves it at
    `alpha` counter-clockwise from the R axis, and the other, `length_beta` long, at `beta` counter-clockwise from the Z axis.
    The colours follow the ST40 physics viewer (`plot_elmag.py`).

    :param gsfit_controller: the Gsfit controller holding the populated `pf_passive` object
    :param ax: the matplotlib axes to draw on
    :param time_desired: unused; kept for API symmetry with the other `two_d_*` plotters
    """
    pf_passive = gsfit_controller.pf_passive

    # Fixed vessel grey (matches plot_elmag.py); drawn with no edge so filaments are not larger than reality
    facecolor: str = "#979A9A"

    n_loop: int = len(pf_passive.get(pp.loop[:].name))
    for i_loop in range(n_loop):
        filaments_r: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.r)  # [metre]
        filaments_z: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.z)  # [metre]
        filaments_length_alpha: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.length_alpha)  # [metre]
        filaments_length_beta: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.length_beta)  # [metre]
        filaments_alpha: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.alpha)  # [radian]
        filaments_beta: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.beta)  # [radian]

        n_filaments: int = filaments_r.shape[0]
        for i_filament in range(n_filaments):
            # The two sides which meet at the reference point
            side_alpha_r: float = filaments_length_alpha[i_filament] * np.cos(filaments_alpha[i_filament])  # [metre]
            side_alpha_z: float = filaments_length_alpha[i_filament] * np.sin(filaments_alpha[i_filament])  # [metre]
            side_beta_r: float = -filaments_length_beta[i_filament] * np.sin(filaments_beta[i_filament])  # [metre]
            side_beta_z: float = filaments_length_beta[i_filament] * np.cos(filaments_beta[i_filament])  # [metre]

            # The four corners, going round from the reference point
            corner_r: npt.NDArray[np.float64] = filaments_r[i_filament] + np.array([0.0, side_alpha_r, side_alpha_r + side_beta_r, side_beta_r])  # [metre]
            corner_z: npt.NDArray[np.float64] = filaments_z[i_filament] + np.array([0.0, side_alpha_z, side_alpha_z + side_beta_z, side_beta_z])  # [metre]

            ax.fill(
                corner_r,
                corner_z,
                facecolor=facecolor,
                edgecolor="none",
                zorder=1,
            )
