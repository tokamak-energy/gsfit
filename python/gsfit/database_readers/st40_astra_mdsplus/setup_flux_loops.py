import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import FluxLoops
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_flux_loops(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> FluxLoops:
    """
    This method initialises the Rust `FluxLoops` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the FluxLoops Rust class
    flux_loops = FluxLoops()

    # Extract the astra_run_name from settings
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["astra_run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA bp_probes
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.FLOOP.ALL:PSI").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.FLOOP.ALL:NAME").data().astype(str)

    # read in mag data from MDSplus
    mag = GetData(12050, "MAG#BEST", is_fail_quiet=False)

    # FL-probes
    names_long = mag.get("FLOOP.ALL.NAMES")
    sensors_names = np.char.replace(names_long, "FLOOP_", "L")
    sensors_r = mag.get("FLOOP.ALL.R")
    sensors_z = mag.get("FLOOP.ALL.Z")

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

        # Measured values
        index = names == sensor_name.replace("L", "FLOOP_")
        measured = measurements[:, index].squeeze()

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
