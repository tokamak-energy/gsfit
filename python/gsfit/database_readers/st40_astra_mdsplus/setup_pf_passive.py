from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfPassive
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_passive.pf_passive import PfPassiveSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree

if TYPE_CHECKING:
    from . import DatabaseReader

# ASTRA's passive structures which ST40's own machine description, ELMAG, counts as part of the in-vessel coil, IVC
PASSIVES_IN_IVC: list[str] = ["GASBFLB", "GASBFLT"]


def setup_pf_passive(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfPassiveSettings,
) -> PfPassive:
    """
    This method initialises the Rust `PfPassive` class, which holds an IMAS `pf_passive` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `PfPassiveSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ASTRA run to read

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    The passive structures are ASTRA's filaments, `PASSIVES`, which its equilibrium, SPIDER, ran with, from ASTRA's `exp/equ/MCVC/fires.dat`.
    One `loop(i1)` is added for each passive structure, in the order its first filament comes in, with one `element(i2)` per filament: these are
    machine data only, and which of them to fit, and how, is the equilibrium's, `equilibrium/pf_passive_dof.xml`, which
    `Equilibrium.copy_from_pf_passive_ids` reads.

    The gas baffles, GASBFLB and GASBFLT, are part of the in-vessel coil, IVC, as they are in ST40's ELMAG, which the settings are made for.

    Each filament is a rectangle, from its centre, `RP` and `ZP`, and its full width and height, `DR` and `DZ`, which the data dictionary's `oblique`
    describes from its bottom left corner, with sides along R and Z. ASTRA stores each filament's resistance, `RESP`, from which its resistivity is
    `RESP * DR * DZ / (2 pi RP)`; a loop's resistivity is the mean of its filaments', as the IVC's gas baffles may differ from the rest of it.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA run to read, from the `workflow.xml` which `pf_passive.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree:
        raise ValueError(
            "`pf_passive.xml` must link to a `workflow.xml` holding the ASTRA `tree`, which says which ASTRA run to read the passive structures from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo

    # Serialise the settings, which the passive structures record in `pf_passive/code/parameters`: `pf_passive.xml` holding only the run which
    # built the passive structures
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfPassiveSettings(workflow=Workflow(tree=KeyedList([astra_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfPassive Rust class
    pf_passive: PfPassive = PfPassive(settings_xml)

    conn: mdsthin.Connection = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", astra_pulseNo)
    run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
    filaments_name_raw: npt.NDArray[np.str_] = np.asarray(conn.get(f"{run_path}.PASSIVES:NAME").data())
    # ASTRA stores these as float32; GSFit uses float64
    filaments_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.PASSIVES:RP").data().astype(np.float64)  # [metre]
    filaments_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.PASSIVES:ZP").data().astype(np.float64)  # [metre]
    filaments_d_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.PASSIVES:DR").data().astype(np.float64)  # [metre]
    filaments_d_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.PASSIVES:DZ").data().astype(np.float64)  # [metre]
    filaments_resistance: npt.NDArray[np.float64] = conn.get(f"{run_path}.PASSIVES:RESP").data().astype(np.float64)  # [ohm]
    filaments_resistivity: npt.NDArray[np.float64] = filaments_resistance * filaments_d_r * filaments_d_z / (2.0 * np.pi * filaments_r)  # [ohm * metre]

    # The passive structure each filament belongs to, and the passive structures in the order their first filaments come in
    n_filament: int = len(filaments_r)
    filament_passive_names: list[str] = []
    passive_names: list[str] = []
    for i_filament in range(n_filament):
        # mdsthin returns the names padded with spaces
        passive_name: str = str(filaments_name_raw[i_filament]).strip()
        if passive_name in PASSIVES_IN_IVC:
            passive_name = "IVC"
        filament_passive_names.append(passive_name)
        if passive_name not in passive_names:
            passive_names.append(passive_name)
    filaments_passive_name: npt.NDArray[np.str_] = np.array(filament_passive_names)

    for passive_name in passive_names:
        in_passive: npt.NDArray[np.bool_] = filaments_passive_name == passive_name
        n_filament_passive: int = int(np.sum(in_passive))

        pf_passive.add_loop(
            name=passive_name,
            r=filaments_r[in_passive] - filaments_d_r[in_passive] / 2.0,
            z=filaments_z[in_passive] - filaments_d_z[in_passive] / 2.0,
            length_alpha=filaments_d_r[in_passive],
            length_beta=filaments_d_z[in_passive],
            alpha=np.zeros(n_filament_passive),
            beta=np.zeros(n_filament_passive),
            resistivity=float(np.mean(filaments_resistivity[in_passive])),
        )

    return pf_passive
