import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfActive
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_active.pf_active import PfActiveSettings

if TYPE_CHECKING:
    from . import DatabaseReader

# FreeGS's filaments are points, which GSFit's filaments cannot be: a small square gives the same flux, except very close to it
FILAMENT_SIZE: float = 1.0e-3  # [metre]


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfActiveSettings,
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
    **kwargs: typing.Any,
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number. Not used: the PF coils are read from the FreeGS equilibria
    :param settings: `PfActiveSettings` object containing the XML settings read from the `settings` directory. Not used: the PF coils are read
        from the FreeGS equilibria
    :param time: The timebase of `freegs_eqs` [second]
    :param freegs_eqs: The FreeGS equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGS.**

    One `coil(i1)` is added for each of FreeGS's coils, which must be `freegs.multi_coil.MultiCoil`s, with one `element(i2)` per filament.
    Each filament is one turn, carrying the coil's current at each time, as in FreeGS. FreeGS's filaments are points, so each is given a
    small size, `FILAMENT_SIZE`.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the PF coils record in `pf_active/code/parameters`: none, as the PF coils are read from the FreeGS equilibria
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfActiveSettings(),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfActive Rust class
    pf_active: PfActive = PfActive(settings_xml)

    # The machine does not change in time, so the coils are read from the first equilibrium, and only their currents from each one
    n_time: int = len(time)
    for coil_name, coil in freegs_eqs[0].tokamak.coils:
        if not isinstance(coil, freegs.multi_coil.MultiCoil):
            raise TypeError(f"FreeGS's coil `{coil_name}` is a `{type(coil).__name__}`, which is not read: only `MultiCoil`s are")

        # GSFit's filaments are single turns, carrying the coil's current, so they cannot be mirrored or have their current reversed
        if coil.mirror or coil.polarity[0] != 1.0:
            raise ValueError(f"FreeGS's coil `{coil_name}` is mirrored, or has a polarity other than 1, which GSFit's PF coils cannot hold")

        # The current in each turn, at each time
        current: npt.NDArray[np.float64] = np.full(n_time, np.nan)  # [ampere]
        for i_time in range(n_time):
            current[i_time] = freegs_eqs[i_time].tokamak[coil_name].current

        coil_r: npt.NDArray[np.float64] = np.asarray(coil.Rfil, dtype=np.float64)  # [metre]
        coil_z: npt.NDArray[np.float64] = np.asarray(coil.Zfil, dtype=np.float64)  # [metre]
        n_filament: int = len(coil_r)
        pf_active.add_pf_coil(
            name=coil_name,
            r=coil_r,
            z=coil_z,
            width=np.full(n_filament, FILAMENT_SIZE),
            height=np.full(n_filament, FILAMENT_SIZE),
            time=time,
            current=current,
        )

    return pf_active
