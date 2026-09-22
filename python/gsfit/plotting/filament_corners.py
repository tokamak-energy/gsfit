import numpy as np
import numpy.typing as npt


def filament_corners(
    filament_r: float,
    filament_z: float,
    filament_d_r: float,
    filament_d_z: float,
    filament_angle_1: float,
    filament_angle_2: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    The four corners of a (possibly sheared) parallelogram filament.

    The corner construction follows the ST40 physics viewer (`plot_elmag.py`).
    An angle of exactly zero means that pair of sides is not sheared.

    :param filament_r: radial position of the filament centre [metre]
    :param filament_z: vertical position of the filament centre [metre]
    :param filament_d_r: radial width of the filament [metre]
    :param filament_d_z: vertical height of the filament [metre]
    :param filament_angle_1: angle between the R axis and the left / right sides [radian]
    :param filament_angle_2: angle between the R axis and the top / bottom sides [radian]
    :return: the radial [metre] and vertical [metre] positions of the corners
    """

    delta_r = filament_d_r * np.array([-0.5, 0.5, 0.5, -0.5])  # [metre]
    if filament_angle_1 != 0.0:
        delta_r = delta_r + (filament_d_z / np.tan(filament_angle_1)) * np.array([-0.5, -0.5, 0.5, 0.5])

    delta_z = filament_d_z * np.array([-0.5, -0.5, 0.5, 0.5])  # [metre]
    if filament_angle_2 != 0.0:
        delta_z = delta_z + filament_d_r * np.tan(filament_angle_2) * np.array([-0.5, 0.5, 0.5, -0.5])

    corner_r = filament_r + delta_r  # [metre]
    corner_z = filament_z + delta_z  # [metre]

    return corner_r, corner_z
