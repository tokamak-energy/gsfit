import typing
from typing import TYPE_CHECKING

import freegs4e
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
from gsfit_rs import PfActive
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_active.pf_active import PfActiveSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfActiveSettings,
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
    **kwargs: typing.Any,
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number. Not used: the PF coils are read from the FreeGSNKE equilibria
    :param settings: `PfActiveSettings` object containing the XML settings read from the `settings` directory
    :param time: The timebase of `freegsnke_eqs` [second]
    :param freegsnke_eqs: The FreeGSNKE equilibria the synthetic measurements are made from, one for each time
    :param kwargs: The other keyword arguments of `Gsfit.run`, e.g. `isofluxes`, which only `setup_magnetics` reads

    **This method is specific to FreeGSNKE.**

    FreeGSNKE groups its PF coils into circuits, `freegs4e.machine.Circuit`, each a list of `(coil_name, coil, multiplier)`.
    One `coil(i1)` is added for each coil in a circuit, with one `element(i2)` per filament, whose size is the coil's `dR` and `dZ`, which
    FreeGSNKE's `build_machine` sets. Each filament is one turn, carrying the circuit's current times the coil's multiplier, at each time.
    The other current carrying objects in `tokamak.coils` are FreeGSNKE's passive structures, which `setup_pf_passive` does not add yet.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the PF coils record in `pf_active/code/parameters`
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=settings,
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfActive Rust class
    pf_active: PfActive = PfActive(settings_xml)

    # The machine does not change in time, so the coils are read from the first equilibrium, and only their currents from each one
    n_time: int = len(time)
    for circuit_name, circuit in freegsnke_eqs[0].tokamak.coils:
        if not isinstance(circuit, freegs4e.machine.Circuit):
            continue

        n_coil: int = len(circuit.coils)
        for i_coil in range(n_coil):
            coil_name: str = circuit.coils[i_coil][0]
            coil = circuit.coils[i_coil][1]
            coil_multiplier: float = float(circuit.coils[i_coil][2])

            # GSFit's filaments are single turns. FreeGSNKE's polarity reverses a filament's current, which they cannot
            if np.any(np.asarray(coil.polarity) != 1.0):
                raise ValueError(f"FreeGSNKE's coil `{coil_name}` has filaments with a polarity other than 1, which GSFit's PF coils cannot hold")

            # The circuit's current, at each time. FreeGSNKE does not keep each coil's own `current` up to date, so it is not used
            current: npt.NDArray[np.float64] = np.full(n_time, np.nan)  # [ampere]
            for i_time in range(n_time):
                current[i_time] = coil_multiplier * freegsnke_eqs[i_time].tokamak[circuit_name].current

            # Every filament in a FreeGSNKE coil is the same size
            coil_r: npt.NDArray[np.float64] = np.asarray(coil.Rfil, dtype=np.float64)  # [metre]
            coil_z: npt.NDArray[np.float64] = np.asarray(coil.Zfil, dtype=np.float64)  # [metre]
            n_filament: int = len(coil_r)
            pf_active.add_pf_coil(
                name=coil_name,
                r=coil_r,
                z=coil_z,
                width=np.full(n_filament, coil.dR, dtype=np.float64),
                height=np.full(n_filament, coil.dZ, dtype=np.float64),
                time=time,
                current=current,
            )

    return pf_active
