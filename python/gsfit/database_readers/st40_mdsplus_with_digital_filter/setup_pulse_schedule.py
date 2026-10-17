import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PulseSchedule
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pulse_schedule.pulse_schedule import PulseScheduleSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pulse_schedule(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PulseScheduleSettings,
) -> PulseSchedule:
    """
    This method initialises the Rust `PulseSchedule` class, which holds an IMAS `pulse_schedule` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `PulseScheduleSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ELMAG
        run to read

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    Only the gap definitions are filled, `pulse_schedule/position_control/gap(i)/name`, `.../r`, `.../z` and `.../angle`.
    `Plasma.copy_from_pulse_schedule_ids` copies them onto the equilibrium IDS, and `solve_grad_shafranov` calculates the value of each gap.

    **The angle is converted.**
    ELMAG stores `GAPS.ANGLE` counter-clockwise from the outward major radius, in the usual plot with `R` to the right and `Z` upwards.
    For example `MCTGAP`, on the upper merging-compression tile, is at 3.8 radian, which points down and inwards, into the plasma.
    GSFit stores the angle in the equilibrium IDS's convention, clockwise from `grad(R)`, so it is negated here.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ELMAG run to read the gaps from, from the `workflow.xml` which `pulse_schedule.xml` links to
    workflow = settings.workflow
    if workflow is None or "ELMAG" not in workflow.tree:
        raise ValueError("`pulse_schedule.xml` must link to a `workflow.xml` holding the ELMAG `tree`, which says which ELMAG run to read the gaps from")
    # A nil `pulseNo` reads the pulse being reconstructed
    elmag_settings = workflow.tree["ELMAG"]
    if elmag_settings.pulseNo is None:
        elmag_pulseNo = pulseNo
    else:
        elmag_pulseNo = elmag_settings.pulseNo
    elmag_run_name = elmag_settings.run_name

    # Serialise the settings, which the pulse schedule records in `pulse_schedule/code/parameters`: `pulse_schedule.xml` holding only the run which
    # built the gaps
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=PulseScheduleSettings(workflow=Workflow(tree=KeyedList([elmag_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    elmag = GetData(elmag_pulseNo, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)
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
    pulse_schedule = PulseSchedule(settings_xml)

    # A run with no gaps leaves the pulse schedule without any, which is allowed
    for i_gap in range(n_gaps):
        pulse_schedule.add_gap(
            name=gaps_name[i_gap],
            r=float(gaps_r[i_gap]),
            z=float(gaps_z[i_gap]),
            angle=float(gaps_angle[i_gap]),
        )

    return pulse_schedule
