"""
Regression test:
A vertical-feedback shift larger than the grid cell height must not destroy the
magnetic axis.

Background
----------
GSFit stabilises the vertical instability using a Taylor series expansion of the poloidal flux:
    `psi_shifted = psi_unshifted + delta_z * d_psi_unshifted_d_z`

A bug has been reported that when the vertical-feedback shift is greater than the cell height:
    `delta_z > d_z`
causes the magnetic-axis O-point to not be found, ending the iteration with `NoMagneticAxisFound`.

This test uses data from a real ST40 experimental shot (12050), but to reproduce the bug we have
deliberately created a fine vertical grid (`d_z = 0.00625 m = 6.25 mm`).

The shot data is read from the mocked MDSplus trees in `data/` (see the `mock_st40_mdsplus`
database_reader), so the test needs no database, network, or `st40_database` dependency.
"""

from pathlib import Path

import numpy as np
import pytest
from gsfit import Gsfit
from gsfit_rs.imas import equilibrium_paths as ep


def test_02_delta_z_shift_greater_than_d_z() -> None:
    gsfit_controller = Gsfit(
        pulseNo=12050,
        run_name="TEST_DELTA_Z",
        run_description="delta_z vertical shift greater than cell height",
        settings_path="default",
        write_to_mds=False,
    )

    code_settings = gsfit_controller.settings["GSFIT_code_settings.json"]

    # Build the objects from the mocked MDSplus trees in `data/`, so no database is needed.
    # `workflow` maps each read onto a mocked tree, exactly as it does for `st40_mdsplus`.
    # `pulseNo = None` means "use the shot's pulseNo"; the machine-description reads pin
    # their own fixed pulse, and are held under a separate workflow name because they use
    # the same tree at a different pulse.
    code_settings["database_reader"]["method"] = "mock_st40_mdsplus"
    mock_dir = str(Path(__file__).parent / "data")
    workflow = {
        "elmag": {
            "tree_name": "ELMAG",
            "pulseNo": None,
            "run_name": "RUN16",
            "usage": "Vessel and limiter geometry and resistance",
        },
        "elmag_coils": {
            "tree_name": "ELMAG",
            "pulseNo": 11012050,
            "run_name": "RUN16",
            "usage": "PF coil geometry, from the machine description pulse",
        },
        "mag": {
            "tree_name": "MAG",
            "pulseNo": None,
            "run_name": "BEST",
            "usage": "Magnetic sensors geometry and measured values",
        },
        "psu2coil": {
            "tree_name": "PSU2COIL",
            "pulseNo": None,
            "run_name": "RUN02",
            "usage": "PF and TF coil currents",
        },
        "rog_gaps": {
            "tree_name": "MAG",
            "pulseNo": 11010605,
            "run_name": "RUN14C",
            "usage": "INIVC000 Rogowski coil gaps",
        },
    }
    code_settings["database_reader"]["mock_st40_mdsplus"] = {"mock_dir": mock_dir, "workflow": workflow}

    # Fine vertical grid so the first vertical-feedback step exceeds one cell height
    code_settings["grid"]["n_z"] = 321

    # Reconstruct only the single frozen time-slice
    code_settings["timeslices"]["method"] = "user_defined"
    code_settings["timeslices"]["user_defined"] = [130e-3]  # 130 ms

    # Run the reconstruction, which should converge
    # Note: The only way we know that the first iteration's vertical shift exceeds the cell height is by adding print statements
    # into the Rust code.
    # TODO: This test checks that the reconstruction works when `delta_z > d_z`, but we don't check this!
    # To fix this we need to add more instrumentation into the Rust code "observability".
    gsfit_controller.run()

    # `plasma.equilibrium_ids` copies the whole IDS out of Rust, so read it once and reuse it
    equilibrium_ids = gsfit_controller.plasma.equilibrium_ids

    grid_z = equilibrium_ids.get(ep.time_slice[0].profiles_2d[0].grid.dim2)
    d_z = grid_z[1] - grid_z[0]
    print(f"test_02_delta_z_shift_greater_than_d_z:  d_z = {d_z} m")
    gs_error = equilibrium_ids.get(ep.time_slice[0].convergence.grad_shafranov_deviation_value)
    r_mag = equilibrium_ids.get(ep.time_slice[0].global_quantities.magnetic_axis.r)
    z_mag = equilibrium_ids.get(ep.time_slice[0].global_quantities.magnetic_axis.z)

    assert np.isfinite(gs_error), "GS reconstruction failed, should have converged"
    assert np.isfinite(r_mag) and np.isfinite(z_mag), "magnetic axis position is not finite"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--capture=no", "--verbose"]))
