import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Tf
from scipy.constants import mu_0
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.tf.tf import TfSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: TfSettings,
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `TfSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which PSU2COIL run to read

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on the **experimental**
       timebase. `Equilibrium.copy_from_tf_ids` interpolates it onto the reconstruction times itself, so
       do not interpolate it here.

    `b_field_phi_vacuum_r` is signed: positive means counter-clockwise viewed from above.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.

    """

    # The PSU2COIL run to read, from the `workflow.xml` which `tf.xml` links to
    workflow = settings.workflow
    if workflow is None or "PSU2COIL" not in workflow.tree:
        raise ValueError("`tf.xml` must link to a `workflow.xml` holding the PSU2COIL `tree`, which says which PSU2COIL run to read the TF rod's current from")
    # A nil `pulseNo` reads the pulse being reconstructed
    psu2coil_settings = workflow.tree["PSU2COIL"]
    if psu2coil_settings.pulseNo is None:
        psu2coil_pulseNo = pulseNo
    else:
        psu2coil_pulseNo = psu2coil_settings.pulseNo
    vacuum_toroidal_field_reference_radius = settings.vacuum_toroidal_field_reference_radius

    # Serialise the settings to be stored inside the IDS: `tf.xml` holding only the settings which built the toroidal field
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=TfSettings(
            vacuum_toroidal_field_reference_radius=vacuum_toroidal_field_reference_radius,
            workflow=Workflow(tree=KeyedList([psu2coil_settings])),
        ),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Tf Rust class
    tf = Tf(settings_xml)

    # Set the reference major radius for the vacuum toroidal field
    tf.set_r0(vacuum_toroidal_field_reference_radius)

    # Store the TF `f_vac`
    psu2coil = GetData(psu2coil_pulseNo, f"PSU2COIL#{psu2coil_settings.run_name}", is_fail_quiet=False, use_redis=False)
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    i_rod = typing.cast(npt.NDArray[np.float64], psu2coil.get("TF.I_ROD"))

    # `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`, the poloidal-current function in vacuum
    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=mu_0 * i_rod / (2.0 * np.pi),
    )

    return tf
