from typing import TYPE_CHECKING

from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Equilibrium
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.equilibrium.equilibrium import EquilibriumSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_equilibrium(
    self: "DatabaseReader",
    pulseNo: int,
    settings: EquilibriumSettings,
) -> Equilibrium:
    """
    This method initialises the Rust `Equilibrium` class, which holds an IMAS `equilibrium` IDS.

    :param pulseNo: Pulse number. Not used: the equilibrium is built from its settings alone
    :param settings: `EquilibriumSettings` object containing the XML settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    The equilibrium is built from its settings alone:
    1. `equilibrium/code/parameters`, the settings as XML: `equilibrium.xml`, with each file it links to written in its link's place
    2. `equilibrium/code/parameters_parsed`, the same settings read into structs, e.g. `equilibrium/code/parameters_parsed/grid/n_r`,
       which `Equilibrium` makes everything else from: the time-slices, the grid, the magnetic sensors' weights, the Greens tables between
       the grid points, and the p' and ff' source functions

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings to be stored inside the IDS: all of them, as GSFit's own settings are what make the equilibrium
    parameters_input: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=settings,
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Equilibrium Rust class, which reads the settings into `equilibrium/code/parameters_parsed`
    equilibrium: Equilibrium = Equilibrium(parameters_input)

    return equilibrium
