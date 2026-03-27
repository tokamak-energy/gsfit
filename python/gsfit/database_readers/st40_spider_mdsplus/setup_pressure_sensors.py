import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Pressure
from st40_database import GetData  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    pressure.add_sensor(
        name="pressure_01",
        geometry_r=0.4,
        geometry_z=0.0,
        fit_settings_comment="",
        fit_settings_expected_value=10e3,
        fit_settings_include=True,
        fit_settings_weight=100.0,
        time=np.array([-1.0, 1.0]).astype(np.float64),
        measured=np.array([15e3, 15e3]).astype(np.float64),
    )

    # TODO: Implement this method

    return pressure
