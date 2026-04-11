"""Bp probe reader for ASTRA simulations using the pf_probe.dat file.

This module reads Bp probe geometry from the pf_probe.dat file
which contains probe data in the following format:
    r, z, angle_pol, name
"""

import os
from typing import Any


def astra_bp_probe_reader() -> dict[str, dict[str, Any]]:
    """
    Read Bp probe data from the pf_probe.dat file.

    The pf_probe.dat file contains Bp probe data with columns:
        - Column 0: r (radial position in meters)
        - Column 1: z (vertical position in meters)
        - Column 2: angle_pol (poloidal angle in radians)
        - Column 3: name (probe name, e.g., BPPROBE_101)

    :return: Dictionary mapping Bp probe names to their properties:
        - r: radial position (float)
        - z: vertical position (float)
        - angle_pol: poloidal angle in radians (float)
    """
    # Get the path to the pf_probe.dat file (in the same directory as this module)
    pf_probe_dat_path = os.path.join(os.path.dirname(__file__), "pf_probe.dat")

    # Read and parse the pf_probe.dat file
    bp_probes_data: dict[str, dict[str, Any]] = {}

    with open(pf_probe_dat_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            # Parse columns: r, z, angle_pol, name
            r: float = float(parts[0])
            z: float = float(parts[1])
            angle_pol: float = float(parts[2])
            name: str = parts[3].strip()

            bp_probes_data[name] = {
                "r": r,
                "z": z,
                "angle_pol": angle_pol,
            }

    return bp_probes_data
