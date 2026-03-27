"""Flux loop reader for ASTRA simulations using the fl_loop.dat file.

This module reads flux loop geometry from the fl_loop.dat file
which contains flux loop data in the following format:
    r, z, name
"""

import os
from typing import Any


def astra_flux_loop_reader() -> dict[str, dict[str, Any]]:
    """
    Read flux loop data from the fl_loop.dat file.

    The fl_loop.dat file contains flux loop data with columns:
        - Column 0: r (radial position in meters)
        - Column 1: z (vertical position in meters)
        - Column 2: name (flux loop name, e.g., FLOOP_001)

    :return: Dictionary mapping flux loop names to their properties:
        - r: radial position (float)
        - z: vertical position (float)
    """
    # Get the path to the fl_loop.dat file (in the same directory as this module)
    fl_loop_dat_path = os.path.join(os.path.dirname(__file__), "fl_loop.dat")

    # Read and parse the fl_loop.dat file
    flux_loops_data: dict[str, dict[str, Any]] = {}

    with open(fl_loop_dat_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # Parse columns: r, z, name
            r: float = float(parts[0])
            z: float = float(parts[1])
            name: str = parts[2].strip()

            flux_loops_data[name] = {
                "r": r,
                "z": z,
            }

    return flux_loops_data
