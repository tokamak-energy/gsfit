import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Tf
from scipy.constants import mu_0
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on the **experimental**
       timebase. `solve_grad_shafranov` interpolates it onto the reconstruction times itself, so
       do not interpolate it here.

    `b_field_phi_vacuum_r` is signed: positive means counter-clockwise viewed from above.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Tf Rust class
    tf = Tf()

    tf.set_r0(settings["GSFIT_code_settings.json"]["vacuum_toroidal_field_reference_radius"])

    psu2coil_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus_with_digital_filter"]["workflow"]["psu2coil"]["run_name"]
    psu2coil = GetData(pulseNo, f"PSU2COIL#{psu2coil_run_name}", is_fail_quiet=False, use_redis=False)
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    i_rod = typing.cast(npt.NDArray[np.float64], psu2coil.get("TF.I_ROD"))

    # `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`, the poloidal-current function in vacuum
    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=mu_0 * i_rod / (2.0 * np.pi),
    )

    return tf
