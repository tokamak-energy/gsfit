"""Rogowski coils reader for ASTRA simulations using the rogpath.dat file.

This module reads Rogowski coil geometry from the rogpath.dat file
which contains coil path data in the following format:
    index name
    r values (space-separated)
    z values (space-separated)
"""

import os
from typing import Any

import numpy as np
import numpy.typing as npt


def astra_rogowski_coils_reader() -> dict[str, dict[str, Any]]:
    """
    Read Rogowski coil data from the rogpath.dat file.

    The rogpath.dat file contains Rogowski coil data with lines:
        - Line 1: index name (e.g., "1 ROG_BVLB")
        - Line 2: r values (space-separated floats)
        - Line 3: z values (space-separated floats)

    This pattern repeats for each Rogowski coil.

    :return: Dictionary mapping Rogowski coil names to their properties:
        - r: numpy array of radial positions
        - z: numpy array of vertical positions
    """
    # Get the path to the rogpath.dat file (in the same directory as this module)
    # Taken from:
    # https://tokamak-devlin.tokamak.local/gitlab/physics/astra_te/-/blob/master/exp/equ/MCVC/rogpath.dat
    rogpath_dat_path = os.path.join(os.path.dirname(__file__), "rogpath.dat")

    # Read and parse the rogpath.dat file
    rogowski_coils_data: dict[str, dict[str, Any]] = {}

    with open(rogpath_dat_path, "r") as f:
        lines = f.readlines()

    n_lines: int = len(lines)
    i_line: int = 0

    while i_line < n_lines:
        # Skip empty lines
        line = lines[i_line].strip()
        if not line:
            i_line += 1
            continue

        # Parse header line: index name
        parts = line.split()
        if len(parts) < 2:
            i_line += 1
            continue

        name: str = parts[1].strip()

        # Parse r values (next line)
        i_line += 1
        if i_line >= n_lines:
            break
        r_line = lines[i_line].strip()
        r: npt.NDArray[np.float64] = np.array([float(x) for x in r_line.split()], dtype=np.float64)

        # Parse z values (next line)
        i_line += 1
        if i_line >= n_lines:
            break
        z_line = lines[i_line].strip()
        z: npt.NDArray[np.float64] = np.array([float(x) for x in z_line.split()], dtype=np.float64)

        rogowski_coils_data[name] = {
            "r": r,
            "z": z,
        }

        i_line += 1

    return rogowski_coils_data
