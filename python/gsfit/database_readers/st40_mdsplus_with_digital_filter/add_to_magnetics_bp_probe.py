import typing

import numpy as np
import numpy.typing as npt
from gsfit_rs import Magnetics
from st40_database import GetData


def add_to_magnetics_bp_probe(
    magnetics: Magnetics,
    mag: GetData,
) -> None:
    """
    Add every poloidal field probe (Mirnov coil) to `magnetics/b_field_pol_probe`.

    :param magnetics: the `Magnetics` the probes are added to
    :param mag: the MAG run the probes are read from, opened with `GetData`

    **This method is specific to ST40's experimental MDSplus database.**

    **The angle keeps GSFit's present definition, COCOS 13, as MDSplus stores it.**
    MAG stores `BPPROBE.ALL.THETA` counter-clockwise from `grad(R)`, in the usual plot with `R` to the right and `Z` upwards, which is the
    angle GSFit has always called `angle_pol`.
    It is stored in `poloidal_angle` unchanged, although the data dictionary's COCOS 17 measures it clockwise; see the `Magnetics` documentation.
    """

    names_long: list[str] = typing.cast(list[str], mag.get("BPPROBE.ALL.NAMES"))
    sensors_name: npt.NDArray[np.str_] = np.char.replace(names_long, "BPPROBE_", "P")
    # MAG stores these as float32; GSFit uses float64
    sensors_r: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("BPPROBE.ALL.R")).astype(np.float64)  # [metre]
    sensors_z: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("BPPROBE.ALL.Z")).astype(np.float64)  # [metre]
    sensors_poloidal_angle: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("BPPROBE.ALL.THETA")).astype(np.float64)  # [radian]

    # Every probe is measured on MAG's timebase
    time: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)  # [second]

    n_sensors: int = len(sensors_name)
    for i_sensor in range(n_sensors):
        sensor_name: str = str(sensors_name[i_sensor])
        field_mds_path: str = f"BPPROBE.{sensor_name.upper()}.B_PROCESSED"
        field: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get(field_mds_path)).astype(np.float64)  # [tesla]

        magnetics.add_b_field_pol_probe(
            name=sensor_name,
            r=float(sensors_r[i_sensor]),
            z=float(sensors_z[i_sensor]),
            poloidal_angle=float(sensors_poloidal_angle[i_sensor]),
            time=time,
            field=field,
        )
