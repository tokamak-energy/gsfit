import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
from gsfit_rs import Tf
from gsfit_rs.dd_support import ParametersInput

if TYPE_CHECKING:
    from . import DatabaseReader

# The major radius ASTRA quotes `GLOBAL:BTVAC` at
ASTRA_BTVAC_REFERENCE_RADIUS = 0.5  # [metre]


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA MDSplus database.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on the **experimental**
       timebase. `Plasma.copy_from_tf_ids` interpolates it onto the reconstruction times itself, so
       do not interpolate it here.

    `b_field_phi_vacuum_r` is signed: positive means counter-clockwise viewed from above.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Tf Rust class
    # This reader still reads the JSON settings, so has no XML settings to record in `tf/code/parameters` yet
    tf = Tf(ParametersInput())

    tf.set_r0(settings["GSFIT_code_settings.json"]["vacuum_toroidal_field_reference_radius"])

    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["workflow"]["astra"]["run_name"]
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)
    bt_vac = conn.get(f"\\ASTRA::TOP.{astra_run_name}.GLOBAL:BTVAC").data().astype(np.float64)

    # `b_field_phi_vacuum_r = R * B_phi` is invariant with R in vacuum, so the radius ASTRA
    # quotes `BTVAC` at cancels out and no `mu_0` round trip through a rod current is needed
    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=bt_vac * ASTRA_BTVAC_REFERENCE_RADIUS,
    )

    return tf
