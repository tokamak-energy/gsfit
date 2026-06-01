# mypy: ignore-errors
# TODO: need to fix mypy errors

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
from gsfit_rs import BpProbes

from .astra_bp_probe_reader import astra_bp_probe_reader

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_bp_probes(
    self: "DatabaseReader",
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
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["workflow"]["astra"]["run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA bp_probes
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.BPPROBE.ALL:B").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.BPPROBE.ALL:NAME").data().astype(str)

    # Read Bp probe geometry from pf_probe.dat
    bp_probes_data = astra_bp_probe_reader()

    for bp_probe_name, data in bp_probes_data.items():
        sensor_name = bp_probe_name.replace("BPPROBE_", "P")

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
        index = names == bp_probe_name
        measured = measurements[:, index].squeeze()

        # Add the sensor to the Rust class
        bp_probes.add_sensor(
            name=sensor_name,
            geometry_angle_pol=data["angle_pol"],
            geometry_r=data["r"],
            geometry_z=data["z"],
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )

    return bp_probes
