import typing
from typing import TYPE_CHECKING

# import freegs  # type: ignore
import freegsnke  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import FluxLoops

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_flux_loops(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> FluxLoops:
    """
    This method initialises the Rust `FluxLoops` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the FluxLoops Rust class
    flux_loops = FluxLoops()

    # We assume that the static data is not changing in time
    freegsnke_tokamak = freegsnke_eqs[0].tokamak

    # Get lengths
    n_time = len(time)
    n_sensors = len(freegsnke_tokamak.probes.floops)

    # Loop over time and calculate the pickup values
    pickups_vals = np.full((n_time, n_sensors), np.nan)
    for i_time in range(n_time):
        pickups_vals[i_time, :] = freegsnke_tokamak.probes.calculate_fluxloop_value(freegsnke_eqs[i_time])

    for i_sensor, sensor in enumerate(freegsnke_tokamak.probes.floops):
        # Get the sensor name
        sensor_name = sensor["name"]

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

        # Get the sensor position
        r = sensor["position"][0]
        z = sensor["position"][1]

        # Get the sensor data, all times
        measured = pickups_vals[:, i_sensor] * 2.0 * np.pi  # convert from "weber / (2.0 * pi)" to "weber"

        # Add the sensor to the FluxLoop class
        flux_loops.add_sensor(
            name=sensor_name,
            geometry_r=r,
            geometry_z=z,
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            measured=measured,
            time=np.array([0.0, 1.0]),
        )

    return flux_loops
