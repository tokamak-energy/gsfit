import typing
from typing import TYPE_CHECKING

from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import CoreProfiles
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.core_profiles.core_profiles import CoreProfilesSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_core_profiles(
    self: "DatabaseReader",
    pulseNo: int,
    settings: CoreProfilesSettings,
    profiles_2d_irregular_rz: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> CoreProfiles:
    """
    This method initialises the Rust `CoreProfiles` class, which holds an IMAS `core_profiles` IDS: the kinetic profiles which the
    pressure constraints are made from.

    :param pulseNo: Pulse number. Not used: the kinetic profiles are passed to `Gsfit.run`
    :param settings: `CoreProfilesSettings` object containing the XML settings read from the `settings` directory. Not used: the kinetic
        profiles are passed to `Gsfit.run`
    :param profiles_2d_irregular_rz: The kinetic profiles, a dictionary of the keyword arguments of `CoreProfiles.set_profiles_2d_irregular_rz`
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `freegs_eqs`, which this method does not use

    **This method is specific to FreeGS.**

    `Gsfit` only calls this method when the equilibrium fits pressure.
    FreeGS has no kinetic diagnostics, so the kinetic profiles are passed to `Gsfit.run`, e.g. FreeGS's pressure at the Thomson scattering's
    channels: `gsfit_controller.run(time=time, freegs_eqs=freegs_eqs, profiles_2d_irregular_rz=profiles_2d_irregular_rz)`.
    FreeGS only has the total pressure, so the electron pressure is NaN, as it was not measured.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    if profiles_2d_irregular_rz is None:
        raise ValueError(
            "The equilibrium fits pressure, but `Gsfit.run` was not given `profiles_2d_irregular_rz`, the kinetic profiles which the pressure constraints are made from"
        )

    # Serialise the settings, which the kinetic profiles record in `core_profiles/code/parameters`: none, as the profiles are passed to `Gsfit.run`
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=CoreProfilesSettings(),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the CoreProfiles Rust class
    core_profiles: CoreProfiles = CoreProfiles(settings_xml)
    core_profiles.set_profiles_2d_irregular_rz(**profiles_2d_irregular_rz)

    return core_profiles
