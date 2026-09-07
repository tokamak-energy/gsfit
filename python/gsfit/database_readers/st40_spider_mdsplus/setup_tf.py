import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
from gsfit_rs import Tf

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

    **This method is specific to ST40's SPIDER MDSplus database.**

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

    spider_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_spider_mdsplus"]["workflow"]["spider"]["run_name"]
    conn = mdsthin.Connection("smaug")
    conn.openTree("SPIDER", pulseNo)
    time = conn.get(f"\\SPIDER::TOP.{spider_run_name}:TIME").data().astype(np.float64)
    bt_vac = conn.get(f"\\SPIDER::TOP.{spider_run_name}.GLOBAL:BTVAC_GEO").data().astype(np.float64)
    r_geo = conn.get(f"\\SPIDER::TOP.{spider_run_name}.GLOBAL:RGEO").data().astype(np.float64)

    # `b_field_phi_vacuum_r = R * B_phi` is invariant with R in vacuum, so SPIDER's field at the
    # geometric axis times that axis is the same quantity, with no `mu_0` round trip
    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=bt_vac * r_geo,
    )

    return tf
