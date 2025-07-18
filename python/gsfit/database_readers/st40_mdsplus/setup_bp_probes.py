import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_bp_probes(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> BpProbes:
    """
    This method initialises the Rust `BpProbes` class (Mirnov coils).

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the BpProbes Rust class
    bp_probes = BpProbes()

    # read in mag data from MDSplus
    mag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["mag"]["run_name"]
    mag = GetData(pulseNo, f"MAG#{mag_run_name}", is_fail_quiet=False)

    # Bp-probes
    names_long = typing.cast(list[str], mag.get("BPPROBE.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "BPPROBE_", "P")
    sensors_r = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.R"))
    sensors_z = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.Z"))
    sensors_angle_pol = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.THETA"))

    n_sensors = len(sensors_names)
    for i_sensor in range(0, n_sensors):
        sensor_name = sensors_names[i_sensor]

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

        # Measured values
        time = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)
        measured = typing.cast(npt.NDArray[np.float32], mag.get(f"BPPROBE.{sensor_name}.B")).astype(np.float64)

        # Add the sensor to the Rust class
        bp_probes.add_sensor(
            name=sensor_name,
            geometry_angle_pol=sensors_angle_pol[i_sensor],
            geometry_r=sensors_r[i_sensor],
            geometry_z=sensors_z[i_sensor],
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )

    return bp_probes
