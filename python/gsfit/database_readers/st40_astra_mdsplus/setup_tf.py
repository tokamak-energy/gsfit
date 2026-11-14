from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Tf
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.tf.tf import TfSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree

if TYPE_CHECKING:
    from . import DatabaseReader

# The major radius ASTRA quotes `GLOBAL:BTVAC` at
ASTRA_BTVAC_REFERENCE_RADIUS: float = 0.5  # [metre]


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: TfSettings,
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `TfSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ASTRA run to read

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at, which `Tf` sets from the settings
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on ASTRA's timebase. `Equilibrium.copy_from_tf_ids` interpolates it onto the
       reconstruction times itself, so it is not interpolated here.

    ASTRA's vacuum toroidal field, `GLOBAL:BTVAC`, is quoted at `R = 0.5 m`. `b_field_phi_vacuum_r = R * B_phi` is the same at every `R` in vacuum,
    so it is `BTVAC * 0.5 m`, whatever `tf/r0` is. It is positive, counter-clockwise viewed from above, as on ST40.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA run to read, from the `workflow.xml` which `tf.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree:
        raise ValueError("`tf.xml` must link to a `workflow.xml` holding the ASTRA `tree`, which says which ASTRA run to read the toroidal field from")
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo

    # Serialise the settings to be stored inside the IDS: `tf.xml` holding only the settings which built the toroidal field
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=TfSettings(
            vacuum_toroidal_field_reference_radius=settings.vacuum_toroidal_field_reference_radius,
            workflow=Workflow(tree=KeyedList([astra_settings])),
        ),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Tf Rust class, which sets the reference major radius for the vacuum toroidal field from the settings
    tf: Tf = Tf(settings_xml)

    conn: mdsthin.Connection = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", astra_pulseNo)
    run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
    # ASTRA stores these as float32; GSFit uses float64
    b_field_phi_vacuum: npt.NDArray[np.float64] = conn.get(f"{run_path}.GLOBAL:BTVAC").data().astype(np.float64)  # [tesla]
    n_time: int = len(b_field_phi_vacuum)
    # ASTRA's `TIME` holds every time it was to run to, and each signal only the times it was run to, which are the first of them
    time: npt.NDArray[np.float64] = conn.get(f"{run_path}:TIME").data().astype(np.float64)[:n_time]  # [second]

    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=b_field_phi_vacuum * ASTRA_BTVAC_REFERENCE_RADIUS,
    )

    return tf
