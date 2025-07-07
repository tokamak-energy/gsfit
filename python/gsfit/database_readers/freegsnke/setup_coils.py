import typing
from typing import TYPE_CHECKING

import freegs4e  # type: ignore
import freegsnke  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_coils(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> Coils:
    """
    This method initialises the Rust `Coils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGSNKE equilibrium objects, one for each time-slice

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the coils class
    coils = Coils()

    # We assume that the static data is not changing in time
    freegsnke_tokamak = freegsnke_eqs[0].tokamak

    # Get lengths
    n_time = len(time)

    # Count the number of PF coils
    n_pf = 0
    # `freegsnke_tokamak.coils` is a list of tuples
    for current_carrying_object in freegsnke_tokamak.coils:
        # Each `current_carrying_object` tuple contains the `circuit_name` and the `circuit`.
        # the `current_carrying_object` can be a PF coil or passive.
        circuit_name = current_carrying_object[0]
        circuit = current_carrying_object[1]
        if isinstance(circuit, freegs4e.machine.Circuit):
            pf_coils = circuit.coils
            n_pf = n_pf + len(pf_coils)

    # Loop over time and get the currents
    pf_coil_currents = np.full((n_time, n_pf), np.nan)
    for i_time in range(0, n_time):
        print(f"i_time = {i_time}")
        i_pf = 0
        for current_carrying_object in freegsnke_eqs[i_time].tokamak.coils:
            circuit_name = current_carrying_object[0]
            circuit = current_carrying_object[1]
            if isinstance(circuit, freegs4e.machine.Circuit):
                pf_coils = circuit.coils
                for pf_coil in pf_coils:
                    coil_name = pf_coil[0]
                    coil_object = pf_coil[1]

                    # Store the current
                    pf_coil_currents[i_time, i_pf] = coil_object.current

                    # set index for the next PF coil
                    i_pf = i_pf + 1

    # Add the PF coils
    i_pf = 0
    for current_carrying_object in freegsnke_tokamak.coils:
        circuit_name = current_carrying_object[0]
        circuit = current_carrying_object[1]
        if isinstance(circuit, freegs4e.machine.Circuit):
            pf_coils = circuit.coils
            for pf_coil in pf_coils:
                coil_name = pf_coil[0]
                coil_object = pf_coil[1]

                coil_r = np.array(coil_object.Rfil).astype(np.float64)
                coil_z = np.array(coil_object.Zfil).astype(np.float64)
                coil_d_r = 0.0 * coil_r
                coil_d_z = 0.0 * coil_z

                current = coil_object.current

                coils.add_pf_coil(
                    name=coil_name,
                    r=coil_r,
                    z=coil_z,
                    d_r=coil_d_r,
                    d_z=coil_d_z,
                    time=time,
                    measured=pf_coil_currents[:, i_pf],
                )

                # set index for the next PF coil
                i_pf = i_pf + 1

    # Add TF coil
    coils.add_tf_coil(
        time=np.array([0.0, 1.0]),
        measured=np.array([1.0e6, 1.0e6]),
    )

    return coils
