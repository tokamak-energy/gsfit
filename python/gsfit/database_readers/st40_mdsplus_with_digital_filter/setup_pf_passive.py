import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfPassive
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.equilibrium.equilibrium import EquilibriumSettings
from ...settings_schema.python.pf_passive.pf_passive import PfPassiveSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_passive(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfPassiveSettings,
    settings_equilibrium: EquilibriumSettings,
) -> PfPassive:
    """
    This method initialises the Rust `PfPassive` class, which holds an IMAS `pf_passive` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `PfPassiveSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ELMAG run to read
    :param settings_equilibrium: `EquilibriumSettings` object, read from `equilibrium/equilibrium.xml`, whose `passive_dof_regularisation` says which
        passive structures to fit, and how

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    One `loop(i1)` is added for each passive structure listed in `passive_dof_regularisation`, in that order, with one `element(i2)` per filament.

    ELMAG describes each filament as EFIT does: by its centre, its radial width and vertical height, and the angles of its sides from the R axis,
    where an angle of zero means that pair of sides is not sheared (see `python/gsfit/plotting/filament_corners.py`). The data dictionary describes
    a parallelogram instead from one of its corners, the reference point, with the lengths of the two sides which meet there and their angles,
    `oblique`. So each filament is converted here.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ELMAG run to read, from the `workflow.xml` which `pf_passive.xml` links to
    workflow = settings.workflow
    if workflow is None or workflow.elmag is None:
        raise ValueError("`pf_passive.xml` must link to a `workflow.xml` holding `elmag`, which says which ELMAG run to read the passive structures from")
    # A nil `pulseNo` reads the pulse being reconstructed
    elmag_settings = workflow.elmag
    if elmag_settings.pulseNo is None:
        elmag_pulseNo = pulseNo
    else:
        elmag_pulseNo = elmag_settings.pulseNo

    # Serialise the settings, which the passive structures record in `pf_passive/code/parameters`: `pf_passive.xml` holding only the run which
    # built the passive structures. The equilibrium's `passive_dof_regularisation` is not recorded there; what it sets is stored in the IDS itself,
    # as each loop's `current_distribution_type`, `dof`, `regularisations` and `regularisations_weight`
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfPassiveSettings(workflow=Workflow(elmag=elmag_settings)),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfPassive Rust class
    pf_passive = PfPassive(settings_xml)

    elmag = GetData(elmag_pulseNo, f"ELMAG#{elmag_settings.run_name}", is_fail_quiet=False)

    vessel_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.R"))  # [metre]
    vessel_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.Z"))  # [metre]
    vessel_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DR"))  # [metre]
    vessel_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DZ"))  # [metre]
    vessel_angle_1 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE1"))  # [radian]
    vessel_angle_2 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE2"))  # [radian]
    vessel_resistivity = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.RESISTIVITY"))  # [ohm * metre]
    vessel_filaments_to_passives = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.FILS2PASSIVE"))
    passive_names = typing.cast(list[str], elmag.get("VESSEL.PASSIVE_NAME"))

    # Convert each filament into the data dictionary's `oblique` description.
    # The reference point is the bottom left corner, corner 1 in `filament_corners`. The side at angle `alpha`, counter-clockwise from the R axis,
    # is the bottom side, from corner 1 to corner 2; the side at angle `beta`, counter-clockwise from the Z axis, is the left side, from corner 1 to
    # corner 4
    n_filament: int = len(vessel_r)
    oblique_r: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [metre]
    oblique_z: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [metre]
    oblique_length_alpha: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [metre]
    oblique_length_beta: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [metre]
    oblique_alpha: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [radian]
    oblique_beta: npt.NDArray[np.float64] = np.full(n_filament, np.nan)  # [radian]
    for i_filament in range(n_filament):
        # How far the left side leans in R over its height, `angle_1` being its angle from the R axis
        if vessel_angle_1[i_filament] == 0.0:
            side_beta_r: float = 0.0  # [metre]
        else:
            side_beta_r = vessel_d_z[i_filament] / np.tan(vessel_angle_1[i_filament])

        # How far the bottom side rises in Z over its width, `angle_2` being its angle from the R axis
        if vessel_angle_2[i_filament] == 0.0:
            side_alpha_z: float = 0.0  # [metre]
        else:
            side_alpha_z = vessel_d_r[i_filament] * np.tan(vessel_angle_2[i_filament])

        # The bottom left corner, half of each side back from the centre
        oblique_r[i_filament] = vessel_r[i_filament] - vessel_d_r[i_filament] / 2.0 - side_beta_r / 2.0
        oblique_z[i_filament] = vessel_z[i_filament] - vessel_d_z[i_filament] / 2.0 - side_alpha_z / 2.0

        # The bottom side, from (d_r, side_alpha_z)
        oblique_length_alpha[i_filament] = np.hypot(vessel_d_r[i_filament], side_alpha_z)
        if vessel_angle_2[i_filament] == 0.0:
            oblique_alpha[i_filament] = 0.0
        else:
            oblique_alpha[i_filament] = np.arctan2(side_alpha_z, vessel_d_r[i_filament])

        # The left side, from (side_beta_r, d_z): `beta` is measured from the Z axis, so a side leaning outwards has a negative `beta`
        oblique_length_beta[i_filament] = np.hypot(side_beta_r, vessel_d_z[i_filament])
        if vessel_angle_1[i_filament] == 0.0:
            oblique_beta[i_filament] = 0.0
        else:
            oblique_beta[i_filament] = np.arctan2(-side_beta_r, vessel_d_z[i_filament])

    for pf_passive_to_fit in settings_equilibrium.passive_dof_regularisation.passive:
        passive_name = pf_passive_to_fit.name
        current_distribution_type = pf_passive_to_fit.current_distribution_type
        n_dof = pf_passive_to_fit.n_dof
        regularisations = np.array(pf_passive_to_fit.regularisations).astype(np.float64)
        regularisations_weight = np.array(pf_passive_to_fit.regularisations_weight).astype(np.float64)

        # Find where this passive is in the vessel_filaments_to_passives matrix
        i_passive_in_vessel = passive_names.index(passive_name)

        # Extract the filaments corresponding to this passive
        i_filaments = vessel_filaments_to_passives[:, i_passive_in_vessel].astype(bool)

        pf_passive.add_loop(
            name=passive_name,
            r=oblique_r[i_filaments],
            z=oblique_z[i_filaments],
            length_alpha=oblique_length_alpha[i_filaments],
            length_beta=oblique_length_beta[i_filaments],
            alpha=oblique_alpha[i_filaments],
            beta=oblique_beta[i_filaments],
            resistivity=vessel_resistivity[i_passive_in_vessel],
            current_distribution_type=current_distribution_type,
            n_dof=n_dof,
            regularisations=regularisations,
            regularisations_weight=regularisations_weight,
        )

    return pf_passive
