import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import PulseSchedule
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pulse_schedule(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> PulseSchedule:
    """
    This method initialises the Rust `PulseSchedule` class, which holds an IMAS `pulse_schedule` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    Only the gap definitions are filled, `pulse_schedule/position_control/gap(i)/name`, `.../r`, `.../z` and `.../angle`.
    `solve_grad_shafranov` copies them onto the equilibrium IDS and calculates the value of each gap.

    **The angle is converted.**
    ELMAG stores `GAPS.ANGLE` counter-clockwise from the outward major radius, in the usual plot with `R` to the right and `Z` upwards.
    For example `MCTGAP`, on the upper merging-compression tile, is at 3.8 radian, which points down and inwards, into the plasma.
    GSFit stores the angle in the equilibrium IDS's convention, clockwise from `grad(R)`, so it is negated here.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    elmag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus_with_digital_filter"]["workflow"]["elmag"]["run_name"]
    elmag = GetData(pulseNo, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)
    gaps_angle_counter_clockwise = typing.cast(npt.NDArray[np.float64], elmag.get("GAPS.ANGLE"))  # [radian]
    gaps_name = typing.cast(list[str], elmag.get("GAPS.NAMES"))
    gaps_r = typing.cast(npt.NDArray[np.float64], elmag.get("GAPS.R_ORIGIN"))  # [metre]
    gaps_z = typing.cast(npt.NDArray[np.float64], elmag.get("GAPS.Z_ORIGIN"))  # [metre]

    # Counter-clockwise to clockwise, kept within [0, 2 * pi) as ELMAG stores it
    gaps_angle = np.mod(-gaps_angle_counter_clockwise, 2.0 * np.pi)  # [radian]

    n_gaps = len(gaps_name)
    if not (len(gaps_angle) == len(gaps_r) == len(gaps_z) == n_gaps):
        raise ValueError(
            f"ELMAG#{elmag_run_name} GAPS: the nodes disagree on the number of gaps; "
            f"NAMES={n_gaps}, ANGLE={len(gaps_angle)}, R_ORIGIN={len(gaps_r)}, Z_ORIGIN={len(gaps_z)}"
        )

    # Initialise the PulseSchedule Rust class
    pulse_schedule = PulseSchedule()

    # A run with no gaps leaves the pulse schedule without any, which is allowed
    for i_gap in range(n_gaps):
        pulse_schedule.add_gap(
            name=gaps_name[i_gap],
            r=float(gaps_r[i_gap]),
            z=float(gaps_z[i_gap]),
            angle=float(gaps_angle[i_gap]),
        )

    return pulse_schedule
