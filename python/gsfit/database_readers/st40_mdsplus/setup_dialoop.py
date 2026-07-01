import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Dialoop
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_dialoop(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Dialoop:
    """
    This method initialises the Rust `Dialoop` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Dialoop Rust class
    dialoop = Dialoop()

    # Read in the diamagnetic flux from MDSplus
    dialoop_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["dialoop"]["run_name"]
    mag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["mag"]["run_name"]
    dialoop_data = GetData(pulseNo, f"DIALOOP#{dialoop_run_name}", is_fail_quiet=False)

    # We use a single diamagnetic flux loop
    sensor_name = "DIALOOP"

    # Get geometry (path of integration) directly via mdsthin from the MAG tree
    # (the geometry is stored in MAG, not in the DIALOOP tree)
    conn = mdsthin.Connection("smaug")
    conn.openTree("MAG", pulseNo)
    path_r = conn.get(f"\\MAG::TOP.{mag_run_name}.DIALOOP.L000:R_PATH").data().astype(np.float64)
    path_z = conn.get(f"\\MAG::TOP.{mag_run_name}.DIALOOP.L000:Z_PATH").data().astype(np.float64)

    if sensor_name in settings["sensor_weights_dialoop.json"]:
        fit_settings_comment = settings["sensor_weights_dialoop.json"][sensor_name]["fit_settings"]["comment"]
        fit_settings_expected_value = settings["sensor_weights_dialoop.json"][sensor_name]["fit_settings"]["expected_value"]
        fit_settings_include = settings["sensor_weights_dialoop.json"][sensor_name]["fit_settings"]["include"]
        fit_settings_weight = settings["sensor_weights_dialoop.json"][sensor_name]["fit_settings"]["weight"]
    else:
        fit_settings_comment = ""
        fit_settings_expected_value = np.nan
        fit_settings_include = False
        fit_settings_weight = np.nan

    # Measured signal: \DIALOOP::TOP.<run_name>.GLOBAL:PHI_DIA
    time = typing.cast(npt.NDArray[np.float64], dialoop_data.get("TIME")).astype(np.float64)
    measured = typing.cast(npt.NDArray[np.float64], dialoop_data.get("GLOBAL.PHI_DIA")).astype(np.float64)

    # Add the sensor to the Rust class
    dialoop.add_sensor(
        name=sensor_name,
        r=path_r,
        z=path_z,
        fit_settings_comment=fit_settings_comment,
        fit_settings_expected_value=fit_settings_expected_value,
        fit_settings_include=fit_settings_include,
        fit_settings_weight=fit_settings_weight,
        time=time,
        measured=measured,
    )

    return dialoop
