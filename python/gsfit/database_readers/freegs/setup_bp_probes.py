import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_bp_probes(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> BpProbes:
    """
    This method initialises the Rust `BpProbes` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegs_eqs: List of FreeGS equilibrium objects, one for each time-slice

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the BpProbes Rust class
    bp_probes = BpProbes()

    n_time = len(time)

    # Let's see how many flux loops there are
    n_bp_probes = 0
    for sensor in freegs_eqs[0].tokamak.sensors:
        if isinstance(sensor, freegs.machine.PoloidalFieldSensor):
            n_bp_probes += 1

    # Collect the measured values
    measured_values = np.full((n_time, n_bp_probes), np.nan)
    for i_time in range(n_time):
        i_sensor = 0
        for sensor in freegs_eqs[i_time].tokamak.sensors:
            if isinstance(sensor, freegs.machine.PoloidalFieldSensor):
                measured_values[i_time, i_sensor] = sensor.measurement
                i_sensor += 1

    # Add the sensors to the BpProbes class
    i_sensor = 0
    for sensor in freegs_eqs[0].tokamak.sensors:
        if isinstance(sensor, freegs.machine.PoloidalFieldSensor):
            # Get the sensor name
            sensor_name = sensor.name

            # Get the sensor position
            r = sensor.R
            z = sensor.Z
            angle = sensor.theta

            # Get the sensor data
            measured = sensor.measurement

            # Only add sesnors which are included in the settings file
            if sensor_name in settings["sensor_weights_bp_probe.json"]:
                fit_settings_comment = settings["sensor_weights_bp_probe.json"][sensor_name]["fit_settings"]["comment"]
                fit_settings_expected_value = settings["sensor_weights_bp_probe.json"][sensor_name]["fit_settings"]["expected_value"]
                fit_settings_include = settings["sensor_weights_bp_probe.json"][sensor_name]["fit_settings"]["include"]
                fit_settings_weight = settings["sensor_weights_bp_probe.json"][sensor_name]["fit_settings"]["weight"]
            else:
                fit_settings_comment = ""
                fit_settings_expected_value = np.nan
                fit_settings_include = False
                fit_settings_weight = np.nan

            # Add the sensor to the BpProbes class
            bp_probes.add_sensor(
                name=sensor_name,
                geometry_angle_pol=angle,
                geometry_r=r,
                geometry_z=z,
                fit_settings_comment=fit_settings_comment,
                fit_settings_expected_value=fit_settings_expected_value,
                fit_settings_include=fit_settings_include,
                fit_settings_weight=fit_settings_weight,
                time=time,
                measured=measured_values[:, i_sensor],
            )

            # Increment the sensor index
            i_sensor += 1

    return bp_probes
