import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfPassive
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_passive.pf_passive import PfPassiveSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_passive(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfPassiveSettings,
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
    **kwargs: typing.Any,
) -> PfPassive:
    """
    This method initialises the Rust `PfPassive` class, which holds an IMAS `pf_passive` IDS.

    :param pulseNo: Pulse number. Not used: there is nothing to read
    :param settings: `PfPassiveSettings` object containing the XML settings read from the `settings` directory. Not used: there is nothing to read
    :param time: The timebase of `freegs_eqs` [second]
    :param freegs_eqs: The FreeGS equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGS.**

    FreeGS has no passive structures, so the pf_passive IDS is empty, and `equilibrium/pf_passive_dof.xml` must list no passive conductors.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the passive structures record in `pf_passive/code/parameters`: none, as there is nothing to read
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfPassiveSettings(),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfPassive Rust class
    pf_passive: PfPassive = PfPassive(settings_xml)

    return pf_passive
