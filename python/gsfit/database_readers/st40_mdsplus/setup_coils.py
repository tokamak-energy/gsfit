import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_coils(
    self: "DatabaseReaderSt40MDSplus",
    settings: dict[str, typing.Any],
    pulseNo: int,
) -> Coils:
    """
    This method initialises the Rust `Coils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Coils Rust class
    coils = Coils()

    elmag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["elmag"]["run_name"]
    # elmag = GetData(pulseNo, f"ELMAG#{elmag_run_name}")
    elmag = GetData(11012050, f"ELMAG#{elmag_run_name}")

    coils_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.R"))
    coils_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.Z"))
    coils_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DR"))
    coils_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DZ"))
    coil_names = typing.cast(list[str], elmag.get("COILS.COIL_NAMES"))
    fils2coils = typing.cast(npt.NDArray[np.bool_], elmag.get("COILS.FILS2COILS") == 1.0)

    psu2coil_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["psu2coil"]["run_name"]
    psu2coil = GetData(pulseNo, f"PSU2COIL#{psu2coil_run_name}", is_fail_quiet=False, use_redis=False)
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    pf_i = typing.cast(npt.NDArray[np.float64], psu2coil.get("PF.ALL.I"))

    # Retrieve the coils which are connected to each PSU
    coils_connected_to_psus = typing.cast(list[list[str]], psu2coil.get("PF.ALL.COILS"))

    n_time, n_psu = pf_i.shape

    for i_psu in range(0, n_psu):
        current_this_psu = pf_i[:, i_psu]
        coils_connected_to_this_psu = coils_connected_to_psus[i_psu]

        for coil_name in coils_connected_to_this_psu:
            # Skip empty coil names (MDSplus does not allow jagged array)
            if coil_name != "":
                i_pf = coil_names.index(coil_name)
                i_filaments = fils2coils[:, i_pf]
                coil_r = coils_r[i_filaments]
                coil_z = coils_z[i_filaments]
                coil_d_r = coils_d_r[i_filaments]
                coil_d_z = coils_d_z[i_filaments]

                # Add coil to Rust class
                coils.add_pf_coil(
                    coil_name,
                    coil_r,
                    coil_z,
                    coil_d_r,
                    coil_d_z,
                    time=time,
                    measured=current_this_psu,
                )

    # Add TF coil
    i_rod = typing.cast(npt.NDArray[np.float64], psu2coil.get("TF.I_ROD"))
    coils.add_tf_coil(
        time,
        i_rod,
    )

    return coils
