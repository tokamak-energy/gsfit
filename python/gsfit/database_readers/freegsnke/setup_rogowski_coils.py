import typing
from typing import TYPE_CHECKING

import freegsnke  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_rogowski_coils(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    # No Rogowski coils are included in FreeGSNKE, so let's use the limiter as
    # the plasma current Rogowski
    sensor_name = "plasma_current_rogowski"
    freegsnke_tokamak = freegsnke_eqs[0].tokamak
    limit_pts_r = np.array(freegsnke_tokamak.limiter.R)
    limit_pts_z = np.array(freegsnke_tokamak.limiter.Z)

    # Get the plasma current
    n_time = len(time)
    ip = np.full((n_time), np.nan)
    for i_time in range(0, n_time):
        ip[i_time] = freegsnke_eqs[i_time].plasmaCurrent()

    # Get the Rogowski coil data
    if sensor_name in settings["sensor_weights_rogowski_coils.json"]:
        fit_settings_comment = settings["sensor_weights_rogowski_coils.json"][sensor_name]["fit_settings"]["comment"]
        fit_settings_expected_value = settings["sensor_weights_rogowski_coils.json"][sensor_name]["fit_settings"]["expected_value"]
        fit_settings_include = settings["sensor_weights_rogowski_coils.json"][sensor_name]["fit_settings"]["include"]
        fit_settings_weight = settings["sensor_weights_rogowski_coils.json"][sensor_name]["fit_settings"]["weight"]
    else:
        fit_settings_comment = ""
        fit_settings_expected_value = np.nan
        fit_settings_include = False
        fit_settings_weight = np.nan

    # Add Rogowski coil to the Rust class
    rogowski_coils.add_sensor(
        name=sensor_name,
        r=limit_pts_r,
        z=limit_pts_z,
        fit_settings_comment=fit_settings_comment,
        fit_settings_expected_value=fit_settings_expected_value,
        fit_settings_include=fit_settings_include,
        fit_settings_weight=fit_settings_weight,
        time=time,
        measured=ip,
        gaps_r=np.array([]),
        gaps_z=np.array([]),
        gaps_d_r=np.array([]),
        gaps_d_z=np.array([]),
        gaps_name=[],
    )

    return rogowski_coils
