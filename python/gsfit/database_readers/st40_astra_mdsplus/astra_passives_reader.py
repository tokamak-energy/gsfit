"""Passives reader for ASTRA simulations using the fires.dat file.

This module reads passive conductor geometry and properties from the fires.dat file
which contains filament data in the following format:
    ignore, r, z, d_r, d_z, resistivity, ignore, name
"""

import os
from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt


def _parse_fortran_float(value: str) -> float:
    """
    Parse Fortran-style floating point notation.

    Converts strings like '.155222855D0' or '690D-9' to Python floats.

    :param value: String containing Fortran-style float notation
    :return: Python float value
    """
    # Replace Fortran 'D' notation with Python 'e' notation
    value = value.replace("D", "e").replace("d", "e")
    return float(value)


def astra_passives_reader() -> dict[str, dict[str, Any]]:
    """
    Read passive conductor data from the fires.dat file.

    The fires.dat file contains filament data with columns:
        - Column 0: ignored
        - Column 1: r (radial position in meters)
        - Column 2: z (vertical position in meters)
        - Column 3: d_r (radial width in meters)
        - Column 4: d_z (vertical height in meters)
        - Column 5: resistivity (in Ohm-meters, Fortran notation)
        - Column 6: ignored
        - Column 7: name (passive conductor name, e.g., IVC, OVC, GASBFLT)

    :return: Dictionary mapping passive names to their properties:
        - r: numpy array of radial positions
        - z: numpy array of vertical positions
        - d_r: numpy array of radial widths
        - d_z: numpy array of vertical heights
        - angle_1: numpy array of zeros (rectangular filaments)
        - angle_2: numpy array of zeros (rectangular filaments)
        - resistivity: mean resistivity of the passive
    """
    # Get the path to the fires.dat file (in the same directory as this module)
    # Taken from:
    # https://tokamak-devlin.tokamak.local/gitlab/physics/astra_te/-/blob/master/exp/equ/MCVC/fires.dat
    fires_dat_path = os.path.join(os.path.dirname(__file__), " fires.dat")

    # Read and parse the fires.dat file
    filaments: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"r": [], "z": [], "d_r": [], "d_z": [], "resistivity": []}
    )

    with open(fires_dat_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # Parse columns: ignore, r, z, d_r, d_z, resistivity, ignore, name
            r = _parse_fortran_float(parts[1])
            z = _parse_fortran_float(parts[2])
            d_r = _parse_fortran_float(parts[3])
            d_z = _parse_fortran_float(parts[4])
            resistivity = _parse_fortran_float(parts[5])
            name = parts[7].strip()

            filaments[name]["r"].append(r)
            filaments[name]["z"].append(z)
            filaments[name]["d_r"].append(d_r)
            filaments[name]["d_z"].append(d_z)
            filaments[name]["resistivity"].append(resistivity)

    # Convert to numpy arrays and compute derived quantities
    passives_data: dict[str, dict[str, Any]] = {}

    for passive_name, data in filaments.items():
        r: npt.NDArray[np.float64] = np.array(data["r"])
        z: npt.NDArray[np.float64] = np.array(data["z"])
        d_r: npt.NDArray[np.float64] = np.array(data["d_r"])
        d_z: npt.NDArray[np.float64] = np.array(data["d_z"])
        n_filaments: int = len(r)

        # Use mean resistivity for the passive (all filaments in a passive have the same resistivity)
        resistivity: float = float(np.mean(data["resistivity"]))

        # Angles are zero for rectangular filaments (not provided in fires.dat)
        angle_1: npt.NDArray[np.float64] = np.zeros(n_filaments)
        angle_2: npt.NDArray[np.float64] = np.zeros(n_filaments)

        passives_data[passive_name] = {
            "r": r,
            "z": z,
            "d_r": d_r,
            "d_z": d_z,
            "angle_1": angle_1,
            "angle_2": angle_2,
            "resistivity": resistivity,
        }

    return passives_data