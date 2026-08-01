import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils

from .mock_get_data import MockGetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_coils(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Coils:
    """
    This method initialises the Rust `Coils` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Coils Rust class
    coils = Coils()

    # Coil geometry, from the "machine description" pulse
    elmag = MockGetData.from_workflow(settings, pulseNo, "elmag_coils")
    coils_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.R"))
    coils_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.Z"))
    coils_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DR"))
    coils_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DZ"))
    coil_names = typing.cast(list[str], elmag.get("COILS.COIL_NAMES"))
    fils2coils = typing.cast(npt.NDArray[np.bool_], np.asarray(elmag.get("COILS.FILS2COILS")) == 1.0)

    # Measured power-supply currents
    psu2coil = MockGetData.from_workflow(settings, pulseNo, "psu2coil")
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    pf_i = typing.cast(npt.NDArray[np.float64], psu2coil.get("PF.ALL.I"))
    coils_connected_to_psus = typing.cast(list[list[str]], psu2coil.get("PF.ALL.COILS"))

    n_time, n_psu = pf_i.shape
    for i_psu in range(0, n_psu):
        current_this_psu = pf_i[:, i_psu]
        coils_connected_to_this_psu = coils_connected_to_psus[i_psu]

        # A single power supply can drive several coils in series
        for coil_name in coils_connected_to_this_psu:
            if coil_name != "":
                i_pf = coil_names.index(coil_name)
                i_filaments = fils2coils[:, i_pf]

                # Add the PF coil to the Rust class
                coils.add_pf_coil(
                    coil_name,
                    coils_r[i_filaments],
                    coils_z[i_filaments],
                    coils_d_r[i_filaments],
                    coils_d_z[i_filaments],
                    time=time,
                    measured=current_this_psu,
                )

    # Add the TF coil to the Rust class
    i_rod = typing.cast(npt.NDArray[np.float64], psu2coil.get("TF.I_ROD"))
    coils.add_tf_coil(time, i_rod)

    return coils
