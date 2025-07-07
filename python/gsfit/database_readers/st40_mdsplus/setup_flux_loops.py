import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import FluxLoops
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_flux_loops(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> FluxLoops:
    """
    This method initialises the Rust `FluxLoops` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the FluxLoops Rust class
    flux_loops = FluxLoops()

    # read in mag data from MDSplus
    mag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["mag"]["run_name"]
    mag = GetData(pulseNo, f"MAG#{mag_run_name}", is_fail_quiet=False)

    # FL-probes
    names_long = typing.cast(list[str], mag.get("FLOOP.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "FLOOP_", "L")
    sensors_r = typing.cast(npt.NDArray[np.float64], mag.get("FLOOP.ALL.R"))
    sensors_z = typing.cast(npt.NDArray[np.float64], mag.get("FLOOP.ALL.Z"))

    n_sensors = len(sensors_names)
    for i_sensor in range(0, n_sensors):
        sensor_name = sensors_names[i_sensor]
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

        # Measured signals
        time = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)
        measured = typing.cast(npt.NDArray[np.float32], mag.get(f"FLOOP.{sensor_name}.PSI")).astype(np.float64)

        # Add the sensor to the Rust class
        flux_loops.add_sensor(
            name=sensors_names[i_sensor],
            geometry_r=sensors_r[i_sensor],
            geometry_z=sensors_z[i_sensor],
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )

    return flux_loops
