import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_bp_probes(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> BpProbes:
    """
    This method initialises the Rust `BpProbes` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the BpProbes Rust class
    bp_probes = BpProbes()

    # Extract the astra_run_name from settings
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["astra_run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA bp_probes
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.BPPROBE.ALL:B").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.BPPROBE.ALL:NAME").data().astype(str)

    # Read mag data from MDSplus
    # FIXME: using a fixed shot is not a good idea!
    mag = GetData(12050, "MAG#BEST", is_fail_quiet=False)

    # Bp-probes
    names_long = mag.get("BPPROBE.ALL.NAMES")
    sensors_names = np.char.replace(names_long, "BPPROBE_", "P")
    sensors_r = mag.get("BPPROBE.ALL.R")
    sensors_z = mag.get("BPPROBE.ALL.Z")
    sensors_angle_pol = mag.get("BPPROBE.ALL.THETA")

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
        index = names == sensor_name.replace("P", "BPPROBE_")
        measured = measurements[:, index].squeeze()

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
