import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_rogowski_coils(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    # Extract the astra_run_name from settings
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["astra_run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA rogowki_coils from MDSplus
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.ROG.ALL:I").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.ROG.ALL:NAME").data().astype(str)

    # Read in mag data from MDSplus
    # FIXME: using a fixed shot is not a good idea!
    mag = GetData(12050, "MAG#BEST", is_fail_quiet=False)

    names_long = mag.get("ROG.ALL.NAMES")
    sensors_names = np.char.replace(names_long, "ROG_", "")
    paths_r = mag.get("ROG.ALL.R_PATH").astype(np.float64)  # BUXTON: need to fix these data types in MDSplus!
    paths_z = mag.get("ROG.ALL.Z_PATH").astype(np.float64)

    n_sensors = len(sensors_names)
    for i_sensor in range(0, n_sensors):
        sensor_name = sensors_names[i_sensor]
        path_r = paths_r[i_sensor, :]
        path_z = paths_z[i_sensor, :]

        # Remove nan's
        # This is because in MDSplus "ALL" does not allow jagged arrays
        path_r = path_r[~np.isnan(path_r)]
        path_z = path_z[~np.isnan(path_z)]

        # Don't store the "fake" Rogowski coils (e.g. the MC supports)
        if len(path_r) > 4:
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

            # Measured values
            index = names == "ROG_" + sensor_name
            measured = measurements[:, index].squeeze()

            # ASTRA does not include gaps in Rogowski coils
            gaps_r = np.array([])
            gaps_z = np.array([])
            gaps_d_r = np.array([])
            gaps_d_z = np.array([])
            gaps_name = []

            # Add Rogowski coil to the Rust class
            rogowski_coils.add_sensor(
                name=sensor_name,
                r=path_r,
                z=path_z,
                fit_settings_comment=fit_settings_comment,
                fit_settings_expected_value=fit_settings_expected_value,
                fit_settings_include=fit_settings_include,
                fit_settings_weight=fit_settings_weight,
                time=time,
                measured=measured,
                gaps_r=gaps_r,
                gaps_z=gaps_z,
                gaps_d_r=gaps_d_r,
                gaps_d_z=gaps_d_z,
                gaps_name=gaps_name,
            )

    return rogowski_coils
