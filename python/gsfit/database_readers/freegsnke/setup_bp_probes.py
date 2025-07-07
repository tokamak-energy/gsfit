import typing
from typing import TYPE_CHECKING

import freegsnke  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_bp_probes(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> BpProbes:
    """
    This method initialises the Rust `BpProbes` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the BpProbes Rust class
    bp_probes = BpProbes()

    # We assume that the static data is not changing in time
    freegsnke_tokamak = freegsnke_eqs[0].tokamak

    # Get lengths
    n_time = len(time)
    n_sensors = len(freegsnke_tokamak.probes.pickups)

    # Loop over time and calculate the pickup values
    pickups_vals = np.full((n_time, n_sensors), np.nan)
    for i_time in range(n_time):
        pickups_vals[i_time, :] = freegsnke_tokamak.probes.calculate_pickup_value(freegsnke_eqs[i_time])

    # Loop over sensors and add them to the BpProbes class
    for i_sensor, sensor in enumerate(freegsnke_tokamak.probes.pickups):
        # Only use sensors in the poloidal plane
        if sensor["orientation"] == "PARALLEL":
            # Get the sensor name
            sensor_name = sensor["name"]

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

            # Get the sensor position
            r = sensor["position"][0]
            z = sensor["position"][2]

            orientation_vector = sensor["orientation_vector"]
            orientation_r = orientation_vector[0]
            orientation_z = orientation_vector[2]
            if orientation_r == 0.0 and orientation_z == 0.0:
                raise ValueError(f"Sensor {sensor_name} has zero orientation vector. Perhaps it's pointing in the toroidal direction?")
            angle = np.arctan2(orientation_z, orientation_r)

            # Get the sensor data, all times
            measured = pickups_vals[:, i_sensor]

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
                measured=measured,
            )

    return bp_probes
