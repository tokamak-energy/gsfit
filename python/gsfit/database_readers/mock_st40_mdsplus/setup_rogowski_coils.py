import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils

from .mock_get_data import MockGetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_rogowski_coils(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    mag = MockGetData.from_workflow(settings, pulseNo, "mag")

    names_long = typing.cast(list[str], mag.get("ROG.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "ROG_", "")
    paths_r = typing.cast(npt.NDArray[np.float64], mag.get("ROG.ALL.R_PATH")).astype(np.float64)
    paths_z = typing.cast(npt.NDArray[np.float64], mag.get("ROG.ALL.Z_PATH")).astype(np.float64)

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

            # Measured signals
            time = typing.cast(npt.NDArray[np.float64], mag.get("TIME")).astype(np.float64)
            measured = typing.cast(npt.NDArray[np.float64], mag.get(f"ROG.{sensor_name}.I")).astype(np.float64)

            # By default we don't have any gaps
            gaps_r: npt.NDArray[np.float64] = np.array([])
            gaps_z: npt.NDArray[np.float64] = np.array([])
            gaps_d_r: npt.NDArray[np.float64] = np.array([])
            gaps_d_z: npt.NDArray[np.float64] = np.array([])
            gaps_name: list[str] = []

            # Only INIVC000 has gaps
            if sensor_name == "INIVC000":
                gaps = MockGetData.from_workflow(settings, pulseNo, "rog_gaps")
                gaps_r = typing.cast(npt.NDArray[np.float64], gaps.get("ROG.INIVC000.GAPS.R"))
                gaps_z = typing.cast(npt.NDArray[np.float64], gaps.get("ROG.INIVC000.GAPS.Z"))
                gaps_d_r = typing.cast(npt.NDArray[np.float64], gaps.get("ROG.INIVC000.GAPS.DR"))
                gaps_d_z = typing.cast(npt.NDArray[np.float64], gaps.get("ROG.INIVC000.GAPS.DZ"))
                gaps_name = typing.cast(list[str], gaps.get("ROG.INIVC000.GAPS.NAME"))

            # Add Rogowski coil to the Rust class
            rogowski_coils.add_sensor(
                sensor_name,
                path_r,
                path_z,
                fit_settings_comment,
                fit_settings_expected_value,
                fit_settings_include,
                fit_settings_weight,
                time,
                measured,
                gaps_r=gaps_r,
                gaps_z=gaps_z,
                gaps_d_r=gaps_d_r,
                gaps_d_z=gaps_d_z,
                gaps_name=gaps_name,
            )

    return rogowski_coils
