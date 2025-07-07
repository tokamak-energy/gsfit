import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import FluxLoops

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_flux_loops(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> FluxLoops:
    """
    This method initialises the Rust `FluxLoops` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGS equilibrium objects, one for each time-slice

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the FluxLoops Rust class
    flux_loops = FluxLoops()

    n_time = len(time)

    # Let's see how many flux loops there are
    n_flux_loops = 0
    for sensor in freegs_eqs[0].tokamak.sensors:
        if isinstance(sensor, freegs.machine.FluxLoopSensor):
            n_flux_loops += 1

    # Collect the measured values
    measured_values = np.full((n_time, n_flux_loops), np.nan)
    for i_time in range(n_time):
        i_sensor = 0
        for sensor in freegs_eqs[i_time].tokamak.sensors:
            if isinstance(sensor, freegs.machine.FluxLoopSensor):
                measured_values[i_time, i_sensor] = sensor.measurement * 2.0 * np.pi
                i_sensor += 1

    # Add the sensors to the FluxLoops class
    i_sensor = 0
    for sensor in freegs_eqs[0].tokamak.sensors:
        if isinstance(sensor, freegs.machine.FluxLoopSensor):
            sensor_name = sensor.name
            r = sensor.R
            z = sensor.Z

            if sensor_name in settings["sensor_weights_flux_loops.json"]:
                fit_settings_comment = settings["sensor_weights_flux_loops.json"][sensor_name]["fit_settings"]["comment"]
                fit_settings_expected_value = settings["sensor_weights_flux_loops.json"][sensor_name]["fit_settings"]["expected_value"]
                fit_settings_include = settings["sensor_weights_flux_loops.json"][sensor_name]["fit_settings"]["include"]
                fit_settings_weight = settings["sensor_weights_flux_loops.json"][sensor_name]["fit_settings"]["weight"] / (2.0 * np.pi)
            else:
                fit_settings_comment = ""
                fit_settings_expected_value = np.nan
                fit_settings_include = False
                fit_settings_weight = np.nan

            # Add the sensor to the BpProbes class
            flux_loops.add_sensor(
                name=sensor_name,
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

    return flux_loops
