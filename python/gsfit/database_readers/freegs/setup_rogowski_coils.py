import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_rogowski_coils(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGS equilibrium objects, one for each time-slice

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    # Loop over time and store the plasma current
    n_time = len(time)
    measured_ip = np.full(n_time, np.nan)
    for i_time in range(n_time):
        measured_ip[i_time] = freegs_eqs[i_time].plasmaCurrent()

    # Add the Rogowski coil for the plasma current
    rogowski_coils.add_sensor(
        name="rog_01",
        r=freegs_eqs[0].tokamak.wall.R,
        z=freegs_eqs[0].tokamak.wall.Z,
        fit_settings_comment="",
        fit_settings_expected_value=1e3,
        fit_settings_include=True,
        fit_settings_weight=100.0,
        time=time,
        measured=measured_ip,
        gaps_r=np.array([]),
        gaps_z=np.array([]),
        gaps_d_r=np.array([]),
        gaps_d_z=np.array([]),
        gaps_name=[],
    )

    # TODO: need to add some Rogowski coils for the passive plates!!!

    return rogowski_coils
