import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_rogowski_coils(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    # Read in mag data from MDSplus
    mag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["mag"]["run_name"]
    mag = GetData(pulseNo, f"MAG#{mag_run_name}", is_fail_quiet=False)

    names_long = typing.cast(list[str], mag.get("ROG.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "ROG_", "")
    paths_r = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.R_PATH")).astype(np.float64)
    paths_z = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.Z_PATH")).astype(np.float64)

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
            time = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)
            measured = typing.cast(npt.NDArray[np.float32], mag.get(f"ROG.{sensor_name}.I")).astype(np.float64)

            # By default we don't have any gaps
            gaps_r = np.array([])
            gaps_z = np.array([])
            gaps_d_r = np.array([])
            gaps_d_z = np.array([])
            gaps_name = []

            # Only INIVC000 has gaps
            if sensor_name == "INIVC000":
                from MDSplus import Connection  # type: ignore

                conn = Connection("smaug")
                conn.openTree("st40", 11010605)
                gaps_r = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:R").data().astype(np.float64)
                gaps_z = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:Z").data().astype(np.float64)
                gaps_d_r = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DR").data().astype(np.float64)
                gaps_d_z = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DZ").data().astype(np.float64)
                gaps_name = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:NAME").data()
                gaps_name = [str(gap_name).replace(" ", "")[2:-1] for gap_name in gaps_name]

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
