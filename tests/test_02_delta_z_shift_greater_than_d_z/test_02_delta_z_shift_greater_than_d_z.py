"""
Regression test:
A vertical-feedback shift larger than the grid cell height must not destroy the
magnetic axis.

Background
----------
GSFit stabilises the vertical instability using a Taylor series expansion of the poloidal flux:
``
    psi_shifted = psi_unshifted + delta_z * d_psi_unshifted_d_z
```

A bug has been reported that when the vertical-feedback shift is greater than the cell height:
    `delta_z > d_z`
causes the magnetic-axis O-point to not be found, ending the iteration with `NoMagneticAxisFound`.

This test uses data from a real ST40 experimental shot (12050), but to reproduce the bug we have
deliberately created a fine vertical grid (`d_z = 0.00625 m = 6.25 mm`).
"""

from pathlib import Path

import numpy as np
import pytest
from gsfit import Gsfit


def test_delta_z_shift_greater_than_d_z() -> None:
    gsfit_controller = Gsfit(
        pulseNo=12050,
        run_name="TEST_DELTA_Z",
        run_description="delta_z vertical shift greater than cell height",
        settings_path="default",
        write_to_mds=False,
    )

    code_settings = gsfit_controller.settings["GSFIT_code_settings.json"]

    # Build the objects from the frozen snapshot
    code_settings["database_reader"]["method"] = "npy_snapshot"
    code_settings["database_reader"]["npy_snapshot"] = {"snapshot_dir": str(Path(__file__).parent / "data"), "workflow": {}}

    # Fine vertical grid so the first vertical-feedback step exceeds one cell
    code_settings["grid"]["n_z"] = 321

    # Reconstruct only the single frozen time-slice
    code_settings["timeslices"]["method"] = "user_defined"
    code_settings["timeslices"]["user_defined"] = [130e-3] # 130 ms

    # Run the reconstruction, which should converge
    gsfit_controller.run()

    plasma = gsfit_controller.plasma
    grid_z = plasma.get_array1(["grid", "z"])
    d_z = grid_z[1] - grid_z[0]
    print(f"d_z = {d_z} m")
    gs_error = plasma.get_array1(["global", "gs_error"])[0]
    r_mag = plasma.get_array1(["global", "r_mag"])[0]
    z_mag = plasma.get_array1(["global", "z_mag"])[0]

    assert np.isfinite(gs_error), "GS reconstruction failed, should have converged"
    assert np.isfinite(r_mag) and np.isfinite(z_mag), "magnetic axis position is not finite"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
