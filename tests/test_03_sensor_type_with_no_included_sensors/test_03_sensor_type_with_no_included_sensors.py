"""
Regression test:
A reconstruction must still run when a whole sensor type has no sensors included in the fit.

Background
----------
Excluding every bp probe (or every flux loop) from the fit is a legitimate way to test the
sensitivity of a reconstruction to a diagnostic. Two failures used to occur:

* the number of passive degrees of freedom was read from the bp-probe Green's table, which is
  empty when no bp probe is included, so the passive currents had the wrong length and the
  solver panicked in a matrix product;
* the measured values of an excluded sensor type were never stored, so the sensor
  post-processing (calculated values, chi_mag) panicked on a missing key.

The shot data is read from the mocked MDSplus trees used by `test_02`, so the test needs no
database, network, or `st40_database` dependency.
"""

from pathlib import Path

import numpy as np
import pytest
from gsfit import Gsfit

MOCK_DIR = str(Path(__file__).parent.parent / "test_02_delta_z_shift_greater_than_d_z" / "data")
WORKFLOW = {
    "elmag": {"tree_name": "ELMAG", "pulseNo": None, "run_name": "RUN16", "usage": "Vessel and limiter geometry and resistance"},
    "elmag_coils": {"tree_name": "ELMAG", "pulseNo": 11012050, "run_name": "RUN16", "usage": "PF coil geometry, from the machine description pulse"},
    "mag": {"tree_name": "MAG", "pulseNo": None, "run_name": "BEST", "usage": "Magnetic sensors geometry and measured values"},
    "psu2coil": {"tree_name": "PSU2COIL", "pulseNo": None, "run_name": "RUN02", "usage": "PF and TF coil currents"},
    "rog_gaps": {"tree_name": "MAG", "pulseNo": 11010605, "run_name": "RUN14C", "usage": "INIVC000 Rogowski coil gaps"},
}


def make_controller() -> Gsfit:
    gsfit_controller = Gsfit(
        pulseNo=12050,
        run_name="TEST_NO_SENSORS",
        run_description="sensor type with no included sensors",
        settings_path="default",
        write_to_mds=False,
    )
    code_settings = gsfit_controller.settings["GSFIT_code_settings.json"]
    code_settings["database_reader"]["method"] = "mock_st40_mdsplus"
    code_settings["database_reader"]["mock_st40_mdsplus"] = {"mock_dir": MOCK_DIR, "workflow": WORKFLOW}
    code_settings["timeslices"]["method"] = "user_defined"
    code_settings["timeslices"]["user_defined"] = [130e-3]  # 130 ms
    return gsfit_controller


@pytest.mark.parametrize("sensor_weights_file", ["sensor_weights_bp_probe.json", "sensor_weights_flux_loops.json"])
def test_03_sensor_type_with_no_included_sensors(sensor_weights_file: str) -> None:
    gsfit_controller = make_controller()

    # Exclude every sensor of this type from the fit
    for sensor_settings in gsfit_controller.settings[sensor_weights_file].values():
        if isinstance(sensor_settings, dict) and "fit_settings" in sensor_settings:
            sensor_settings["fit_settings"]["include"] = False

    gsfit_controller.run()

    plasma = gsfit_controller.plasma
    ip = plasma.get_array1(["global", "ip"])[0]
    chi_mag = plasma.get_array1(["global", "chi_mag"])[0]
    assert np.isfinite(ip), "GS reconstruction failed, should have converged without this sensor type"
    assert np.isfinite(chi_mag), "chi_mag should still be calculated from the remaining sensor types"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--capture=no", "--verbose"]))
