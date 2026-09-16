import matplotlib.axes

from ..gsfit import Gsfit
from .filament_corners import filament_corners


def plot(
    gsfit_controller: Gsfit,
    ax: matplotlib.axes.Axes,
    time_desired: float | None = None,
) -> None:
    """
    Plot the passive conducting structures (e.g. the vacuum vessel / IVC).

    Each passive is discretised into filaments stored in the `passives` Rust data structure
    under `[<passive_name>, "geometry", "r"/"z"/"d_r"/"d_z"/"angle_1"/"angle_2"]`. The filaments
    are (possibly sheared) parallelograms; the corner construction and colours follow the ST40
    physics viewer (`plot_elmag.py`).

    :param gsfit_controller: the Gsfit controller holding the populated `passives` object
    :param ax: the matplotlib axes to draw on
    :param time_desired: unused; kept for API symmetry with the other `two_d_*` plotters
    """
    passives = gsfit_controller.passives

    # Fixed vessel grey (matches plot_elmag.py); drawn with no edge so filaments are not larger than reality
    facecolor: str = "#979A9A"

    for passive_name in passives.keys():
        filaments_r = passives.get_array1([passive_name, "geometry", "r"])  # [metre]
        filaments_z = passives.get_array1([passive_name, "geometry", "z"])  # [metre]
        filaments_d_r = passives.get_array1([passive_name, "geometry", "d_r"])  # [metre]
        filaments_d_z = passives.get_array1([passive_name, "geometry", "d_z"])  # [metre]
        filaments_angle_1 = passives.get_array1([passive_name, "geometry", "angle_1"])  # [radian]
        filaments_angle_2 = passives.get_array1([passive_name, "geometry", "angle_2"])  # [radian]

        n_filaments: int = filaments_r.shape[0]
        for i_filament in range(n_filaments):
            corner_r, corner_z = filament_corners(
                filament_r=filaments_r[i_filament],
                filament_z=filaments_z[i_filament],
                filament_d_r=filaments_d_r[i_filament],
                filament_d_z=filaments_d_z[i_filament],
                filament_angle_1=filaments_angle_1[i_filament],
                filament_angle_2=filaments_angle_2[i_filament],
            )
            ax.fill(
                corner_r,
                corner_z,
                facecolor=facecolor,
                edgecolor="none",
                zorder=1,
            )
