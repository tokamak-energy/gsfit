import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfActive
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_active.pf_active import PfActiveSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfActiveSettings,
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `PfActiveSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ELMAG and
        PSU2COIL runs to read

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    One `coil(i1)` is added for each PF coil, with one `element(i2)` per filament, and its current on the **experimental** timebase.
    Every coil's current is prescribed.
    A power supply which feeds several coils in series gives each of them its current.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ELMAG and PSU2COIL runs to read, from the `workflow.xml` which `pf_active.xml` links to
    workflow = settings.workflow
    if workflow is None or "ELMAG" not in workflow.tree or "PSU2COIL" not in workflow.tree:
        raise ValueError(
            "`pf_active.xml` must link to a `workflow.xml` holding the ELMAG and PSU2COIL `tree`s, which say which ELMAG and PSU2COIL runs to read the PF coils from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    elmag_settings = workflow.tree["ELMAG"]
    if elmag_settings.pulseNo is None:
        elmag_pulseNo = pulseNo
    else:
        elmag_pulseNo = elmag_settings.pulseNo
    psu2coil_settings = workflow.tree["PSU2COIL"]
    if psu2coil_settings.pulseNo is None:
        psu2coil_pulseNo = pulseNo
    else:
        psu2coil_pulseNo = psu2coil_settings.pulseNo

    # Serialise the settings, which the PF coils record in `pf_active/code/parameters`: `pf_active.xml` holding only the runs which built the PF coils
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfActiveSettings(workflow=Workflow(tree=KeyedList([elmag_settings, psu2coil_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfActive Rust class
    pf_active = PfActive(settings_xml)

    # Read the filaments of every PF coil from ELMAG; `fils2coils` says which coil each filament belongs to
    elmag = GetData(elmag_pulseNo, f"ELMAG#{elmag_settings.run_name}")
    coils_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.R"))
    coils_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.Z"))
    coils_width = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DR"))
    coils_height = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DZ"))
    coil_names = typing.cast(list[str], elmag.get("COILS.COIL_NAMES"))
    fils2coils = typing.cast(npt.NDArray[np.bool_], elmag.get("COILS.FILS2COILS") == 1.0)

    # Read the current in each PSU, and the coils connected to it, from PSU2COIL
    psu2coil = GetData(psu2coil_pulseNo, f"PSU2COIL#{psu2coil_settings.run_name}", is_fail_quiet=False, use_redis=False)
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    pf_i = typing.cast(npt.NDArray[np.float64], psu2coil.get("PF.ALL.I"))  # shape = (n_time, n_psu)
    coils_connected_to_psus = typing.cast(list[list[str]], psu2coil.get("PF.ALL.COILS"))

    n_psu = pf_i.shape[1]
    for i_psu in range(n_psu):
        current_this_psu = pf_i[:, i_psu]
        coils_connected_to_this_psu = coils_connected_to_psus[i_psu]

        for coil_name in coils_connected_to_this_psu:
            # Skip empty coil names (MDSplus does not allow jagged array)
            if coil_name != "":
                i_coil = coil_names.index(coil_name)
                i_filaments = fils2coils[:, i_coil]

                # Every coil connected to this PSU carries its current
                pf_active.add_pf_coil(
                    name=coil_name,
                    r=coils_r[i_filaments],
                    z=coils_z[i_filaments],
                    width=coils_width[i_filaments],
                    height=coils_height[i_filaments],
                    time=time,
                    current=current_this_psu,
                )

    return pf_active
