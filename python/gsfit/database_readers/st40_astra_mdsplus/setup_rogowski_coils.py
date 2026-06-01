# mypy: ignore-errors
# TODO: need to fix mypy errors

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
from gsfit_rs import RogowskiCoils

from .astra_rogowski_coils_reader import astra_rogowski_coils_reader

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_rogowski_coils(
    self: "DatabaseReader",
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
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["workflow"]["astra"]["run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA rogowki_coils from MDSplus
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    measurements = conn.get(f"\\ASTRA::TOP.{astra_run_name}.ROG.ALL:I").data().astype(np.float64)
    names = conn.get(f"\\ASTRA::TOP.{astra_run_name}.ROG.ALL:NAME").data().astype(str)

    # Read Rogowski coil geometry from rogpath.dat
    rogowski_coils_data = astra_rogowski_coils_reader()

    for rog_name, data in rogowski_coils_data.items():
        path_r = data["r"]
        path_z = data["z"]

        # Don't store the "fake" Rogowski coils (e.g. the MC supports)
        if len(path_r) > 4:
            sensor_name = rog_name.replace("ROG_", "")

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
            index = names == rog_name
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
