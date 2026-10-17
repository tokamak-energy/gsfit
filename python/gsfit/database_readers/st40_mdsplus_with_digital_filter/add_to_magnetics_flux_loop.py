import typing

import numpy as np
import numpy.typing as npt
from gsfit_rs import Magnetics
from st40_database import GetData


def add_to_magnetics_flux_loop(
    magnetics: Magnetics,
    mag: GetData,
) -> None:
    """
    Add every flux loop to `magnetics/flux_loop`.

    :param magnetics: the `Magnetics` the flux loops are added to
    :param mag: the MAG run the flux loops are read from, opened with `GetData`

    **This method is specific to ST40's experimental MDSplus database.**

    Every ST40 flux loop goes all the way around the machine, at one major radius and height.
    MAG stores the flux as GSFit's `psi`, the total poloidal flux, which is what the data dictionary's `flux_loop(i1)/flux` holds.
    """

    names_long: list[str] = typing.cast(list[str], mag.get("FLOOP.ALL.NAMES"))
    sensors_name: npt.NDArray[np.str_] = np.char.replace(names_long, "FLOOP_", "L")
    # MAG stores these as float32; GSFit uses float64
    sensors_r: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("FLOOP.ALL.R")).astype(np.float64)  # [metre]
    sensors_z: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("FLOOP.ALL.Z")).astype(np.float64)  # [metre]

    # Every flux loop is measured on MAG's timebase
    time: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)  # [second]

    n_sensors: int = len(sensors_name)
    for i_sensor in range(n_sensors):
        sensor_name: str = str(sensors_name[i_sensor])
        flux_mds_path: str = f"FLOOP.{sensor_name.upper()}.PSI_PROCESSED"
        flux: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get(flux_mds_path)).astype(np.float64)  # [weber]

        magnetics.add_flux_loop(
            name=sensor_name,
            r=float(sensors_r[i_sensor]),
            z=float(sensors_z[i_sensor]),
            time=time,
            flux=flux,
        )
