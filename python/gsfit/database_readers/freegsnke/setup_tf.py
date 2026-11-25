import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
from gsfit_rs import Tf
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.tf.tf import TfSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: TfSettings,
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
    **kwargs: typing.Any,
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number. Not used: the toroidal field is read from the FreeGSNKE equilibria
    :param settings: `TfSettings` object containing the XML settings read from the `settings` directory
    :param time: The timebase of `freegsnke_eqs` [second]
    :param freegsnke_eqs: The FreeGSNKE equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGSNKE.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at, which `Tf` sets from the settings
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on the timebase of `freegsnke_eqs`. `Equilibrium.copy_from_tf_ids`
       interpolates it onto the reconstruction times itself, so do not interpolate it here.

    `b_field_phi_vacuum_r` is each equilibrium's vacuum poloidal-current function, `f_vac = R * B_phi`, which FreeGSNKE's profiles hold. It
    is signed: positive means counter-clockwise viewed from above, as in FreeGSNKE.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the toroidal field records in `tf/code/parameters`
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=settings,
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Tf Rust class, which sets the reference major radius for the vacuum toroidal field from the settings
    tf: Tf = Tf(settings_xml)

    # The vacuum `f_vac = R * B_phi` of each equilibrium
    n_time: int = len(time)
    f_vac: npt.NDArray[np.float64] = np.full(n_time, np.nan)  # [tesla * metre]
    for i_time in range(n_time):
        f_vac[i_time] = freegsnke_eqs[i_time].fvac()

    tf.set_b_field_phi_vacuum_r(
        time=time,
        data=f_vac,
    )

    return tf
