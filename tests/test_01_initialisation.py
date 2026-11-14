import gsfit_rs
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit.settings_schema import XSD_DIRECTORY_PATH
from gsfit.settings_schema.python.pf_active.pf_active import PfActiveSettings


def test_01_initialisation() -> None:
    pf_active = gsfit_rs.PfActive(
        parameters_input=serialise_python_dataclass_to_imas_parameters_input(
            settings=PfActiveSettings(),
            xsd_directory_path=XSD_DIRECTORY_PATH,
        ),
    )
    print(pf_active)
