# mypy: ignore-errors
# TODO: need to fix mypy errors

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import FluxLoops

from .astra_flux_loop_reader import astra_flux_loop_reader

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_flux_loops(
    self: "DatabaseReader",
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
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["workflow"]["astra"]["run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA flux_loops
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.FLOOP.ALL:PSI").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.FLOOP.ALL:NAME").data().astype(str)

    # Read flux loop geometry from fl_loop.dat
    flux_loops_data = astra_flux_loop_reader()

    for flux_loop_name, data in flux_loops_data.items():
        sensor_name = flux_loop_name.replace("FLOOP_", "L")

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
        index = names == flux_loop_name
        measured = measurements[:, index].squeeze()

        # Add the sensor to the Rust class
        flux_loops.add_sensor(
            name=sensor_name,
            geometry_r=data["r"],
            geometry_z=data["z"],
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )

    return flux_loops
