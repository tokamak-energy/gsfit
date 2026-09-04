import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
from gsfit_rs import Pressure

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class using synthetic pressure sensors,
    generated from the FreeGSNKE "truth" equilibrium.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    The pressure sensors are only added when `sensor_weights_pressure.json["include"]` is `True`.
    Synthetic sensors are placed along the outboard midplane (at the magnetic axis height),
    from `synthetic.r_start` to `synthetic.r_end` [metre], measured relative to the magnetic axis
    radius. The measured pressure at each sensor is read directly from the FreeGSNKE equilibrium's
    `pressure(psi_n)` profile, so it is a "perfect" synthetic measurement (no noise added).
    Sensors landing outside the last closed flux surface, where the pressure is (by definition)
    below `synthetic.minimum_pressure` [pascal], are discarded.

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    pressure_settings = settings["sensor_weights_pressure.json"]

    # Pressure sensors are opt-in
    if not pressure_settings.get("include", False):
        return pressure

    synthetic_settings = pressure_settings["synthetic"]
    sensor_r_start: float = synthetic_settings["r_start"]
    sensor_r_end: float = synthetic_settings["r_end"]
    n_sensors: int = synthetic_settings["n_sensors"]
    minimum_pressure: float = synthetic_settings["minimum_pressure"]

    sensor_name_prefix = pressure_settings["sensor_name_prefix"]
    fit_settings_comment = pressure_settings["fit_settings"]["comment"]
    fit_settings_weight = pressure_settings["fit_settings"]["weight"]

    n_time: int = len(time)

    # We assume the sensor geometry is not changing in time, so we use the first time-slice
    mag_r: float = freegsnke_eqs[0].Rmagnetic()
    mag_z: float = freegsnke_eqs[0].Zmagnetic()
    sensors_geometry_r: npt.NDArray[np.float64] = mag_r + np.linspace(sensor_r_start, sensor_r_end, n_sensors)
    sensors_geometry_z: npt.NDArray[np.float64] = np.full(n_sensors, mag_z)

    # Get the "measured" pressure at each sensor, for each time-slice
    pressure_measured = np.full((n_time, n_sensors), np.nan)
    for i_time in range(n_time):
        psi_n_sensors = freegsnke_eqs[i_time].psiNRZ(sensors_geometry_r, sensors_geometry_z)
        pressure_measured[i_time, :] = freegsnke_eqs[i_time].pressure(psi_n_sensors)

    for i_sensor in range(n_sensors):
        measured = pressure_measured[:, i_sensor].copy()

        # Ignore (set to NaN) any non-physical / below-threshold measurements, e.g. sensors
        # which landed outside the last closed flux surface
        measured[measured <= minimum_pressure] = np.nan

        # Only add the sensor if it has at least one valid measurement
        if np.any(~np.isnan(measured)):
            pressure.add_sensor(
                name=f"{sensor_name_prefix}{i_sensor + 1:02d}",
                geometry_r=sensors_geometry_r[i_sensor],
                geometry_z=sensors_geometry_z[i_sensor],
                fit_settings_comment=fit_settings_comment,
                fit_settings_expected_value=float(np.nanmax(measured)),
                fit_settings_include=True,
                fit_settings_weight=fit_settings_weight,
                time=time,
                measured=measured,
            )

    return pressure
