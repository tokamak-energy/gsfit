"""Coils reader for ASTRA simulations using the coil.dat file.

This module reads coil geometry from the coil.dat file
which contains coil data in the following format:
    ndi (number of divisions)
    rc zc wc hc awc ahc ... name ...
"""

import os
from typing import Any


def astra_coils_reader() -> dict[str, dict[str, Any]]:
    """
    Read coil data from the coil.dat file.

    The coil.dat file contains coil data with pairs of lines:
        - Line 1: ndi (number of divisions)
        - Line 2: rc zc wc hc awc ahc unknown unknown unknown name unknown

    Where:
        - rc: radial position of center (m)
        - zc: vertical position of center (m)
        - wc: projection of first side on R (m)
        - hc: projection of second side on Z (m)
        - awc: angle of first side with R axis (degrees)
        - ahc: angle of second side with R axis (degrees)
        - ndi: approximate number of cells for dividing

    :return: Dictionary mapping coil names to their properties:
        - rc: radial center position (float)
        - zc: vertical center position (float)
        - wc: radial width (float)
        - hc: vertical height (float)
        - awc: angle of first side (float, degrees)
        - ahc: angle of second side (float, degrees)
        - ndi: number of divisions (int)
    """
    # Get the path to the coil.dat file (in the same directory as this module)
    # Taken from:
    # https://tokamak-devlin.tokamak.local/gitlab/physics/astra_te/-/blob/master/exp/equ/MCVC/coil.dat
    coil_dat_path = os.path.join(os.path.dirname(__file__), "coil.dat")

    # Read and parse the coil.dat file
    coils_data: dict[str, dict[str, Any]] = {}

    with open(coil_dat_path, "r") as f:
        lines = f.readlines()

    n_lines: int = len(lines)
    i_line: int = 0

    while i_line < n_lines:
        # Skip empty lines
        line = lines[i_line].strip()
        if not line:
            i_line += 1
            continue

        # Parse ndi line (first line of pair)
        try:
            ndi: int = int(line.split()[0])
        except (ValueError, IndexError):
            i_line += 1
            continue

        # Parse coil parameters line (second line of pair)
        i_line += 1
        if i_line >= n_lines:
            break

        params_line = lines[i_line].strip()
        parts = params_line.split()
        if len(parts) < 10:
            i_line += 1
            continue

        # Parse: rc zc wc hc awc ahc unknown unknown unknown name
        rc: float = float(parts[0])
        zc: float = float(parts[1])
        wc: float = float(parts[2])
        hc: float = float(parts[3])
        awc: float = float(parts[4])
        ahc: float = float(parts[5])
        name: str = parts[9].strip()

        # Map CS to SOL (ASTRA uses CS, but we use SOL)
        if name == "CS":
            name = "SOL"

        coils_data[name] = {
            "rc": rc,
            "zc": zc,
            "wc": wc,
            "hc": hc,
            "awc": awc,
            "ahc": ahc,
            "ndi": ndi,
        }

        i_line += 1

    return coils_data
