from typing import TYPE_CHECKING

from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Magnetics
from gsfit_rs.dd_support import ParametersInput
from st40_database import GetData

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree
from .add_to_magnetics_bp_probe import add_to_magnetics_bp_probe
from .add_to_magnetics_diamagnetic_flux import add_to_magnetics_diamagnetic_flux
from .add_to_magnetics_flux_loop import add_to_magnetics_flux_loop
from .add_to_magnetics_rogowski_coil import add_to_magnetics_rogowski_coil

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_magnetics(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
) -> Magnetics:
    """
    This method initialises the Rust `Magnetics` class, which holds an IMAS `magnetics` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which MAG run
        to read the poloidal field probes, flux loops and Rogowski coils from, and which DIALOOP run to read the diamagnetic flux from

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    Each kind of sensor is added by its own function:
    1. `add_to_magnetics_bp_probe` fills `magnetics/b_field_pol_probe`
    2. `add_to_magnetics_flux_loop` fills `magnetics/flux_loop`
    3. `add_to_magnetics_rogowski_coil` fills `magnetics/rogowski_coil`
    4. `add_to_magnetics_diamagnetic_flux` fills `magnetics/diamagnetic_flux`, when the pulse has a DIALOOP run

    Every signal is stored on its **experimental** timebase.
    Every sensor is added, whether or not it is fitted: which sensors are fitted, and their weights, are the equilibrium's constraints.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The DIALOOP and MAG runs to read, from the `workflow.xml` which `magnetics.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "DIALOOP" not in workflow.tree or "MAG" not in workflow.tree:
        raise ValueError(
            "`magnetics.xml` must link to a `workflow.xml` holding the DIALOOP and MAG `tree`s, which say which DIALOOP run to read the diamagnetic flux from, "
            "and which MAG run to read the poloidal field probes, flux loops and Rogowski coils from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    dialoop_settings: WorkflowTree = workflow.tree["DIALOOP"]
    if dialoop_settings.pulseNo is None:
        dialoop_pulseNo: int = pulseNo
    else:
        dialoop_pulseNo = dialoop_settings.pulseNo
    mag_settings: WorkflowTree = workflow.tree["MAG"]
    if mag_settings.pulseNo is None:
        mag_pulseNo: int = pulseNo
    else:
        mag_pulseNo = mag_settings.pulseNo

    # Serialise the settings, which the magnetic sensors record in `magnetics/code/parameters`: `magnetics.xml` holding only the runs which built
    # the magnetic sensors
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=MagneticsSettings(workflow=Workflow(tree=KeyedList([dialoop_settings, mag_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Magnetics Rust class
    magnetics: Magnetics = Magnetics(settings_xml)

    # The poloidal field probes, flux loops and Rogowski coils are all read from the one MAG run
    mag: GetData = GetData(mag_pulseNo, f"MAG#{mag_settings.run_name}", is_fail_quiet=False)
    add_to_magnetics_bp_probe(magnetics, mag)
    add_to_magnetics_flux_loop(magnetics, mag)
    add_to_magnetics_rogowski_coil(magnetics, mag)

    # The diamagnetic flux
    dialoop: GetData = GetData(dialoop_pulseNo, f"DIALOOP#{dialoop_settings.run_name}", is_fail_quiet=False)
    add_to_magnetics_diamagnetic_flux(magnetics, dialoop)

    return magnetics
