import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import StationaryPoint

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_stationary_point_sensors(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> StationaryPoint:
    """
    This method initialises the Rust `StationaryPoint` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the StationaryPoint Rust class
    stationary_points = StationaryPoint()

    # magnetic_axis.add_sensor(
    #     "magnetic_axis",
    #     fit_settings_comment="",
    #     fit_settings_expected_value=1.0e-3, # value of `br`.
    #     fit_settings_include=True,
    #     fit_settings_weight=100.0,
    #     time=np.linspace(0.0, 1.0, 100).astype(np.float64),
    #     mag_axis_r=np.full(100, 54.0e-2).astype(np.float64),
    #     mag_axis_z=np.full(100, 0.0).astype(np.float64),
    #     times_to_reconstruct=times_to_reconstruct,
    # )

    return stationary_points
