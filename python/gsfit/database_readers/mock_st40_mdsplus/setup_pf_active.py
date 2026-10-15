import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import PfActive
from gsfit_rs.dd_support import ParametersInput

from .mock_get_data import MockGetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    One `coil(i1)` is added for each PF coil, with one `element(i2)` per filament, and its current on the **experimental** timebase.
    Every coil's current is prescribed.
    A power supply which feeds several coils in series gives each of them its current.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the PfActive Rust class
    # This reader still reads the JSON settings, so has no XML settings to record in `pf_active/code/parameters` yet
    pf_active = PfActive(ParametersInput())

    # Coil geometry, from the "machine description" pulse
    elmag = MockGetData.from_workflow(settings, pulseNo, "elmag_coils")
    coils_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.R"))
    coils_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.Z"))
    coils_width = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DR"))
    coils_height = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DZ"))
    coil_names = typing.cast(list[str], elmag.get("COILS.COIL_NAMES"))
    fils2coils = typing.cast(npt.NDArray[np.bool_], np.asarray(elmag.get("COILS.FILS2COILS")) == 1.0)

    # Measured power-supply currents
    psu2coil = MockGetData.from_workflow(settings, pulseNo, "psu2coil")
    time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
    pf_i = typing.cast(npt.NDArray[np.float64], psu2coil.get("PF.ALL.I"))  # shape = (n_time, n_psu)
    coils_connected_to_psus = typing.cast(list[list[str]], psu2coil.get("PF.ALL.COILS"))

    n_psu = pf_i.shape[1]
    for i_psu in range(n_psu):
        current_this_psu = pf_i[:, i_psu]
        coils_connected_to_this_psu = coils_connected_to_psus[i_psu]

        # A single power supply can drive several coils in series
        for coil_name in coils_connected_to_this_psu:
            if coil_name != "":
                i_coil = coil_names.index(coil_name)
                i_filaments = fils2coils[:, i_coil]

                # Add the PF coil to the Rust class
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
