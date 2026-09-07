# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit
import numpy as np

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=14477,
    run_name="RUN03",
    run_description="2026-07-21: digital filter applied to magnetics",
    write_to_mds=False,
    settings_path="st40_P2p3_calibrations",
)

gsfit_controller.settings["GSFIT_code_settings.json"]["database_writer"]["method"] = "imas"
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [100.0e-3, 125.0e-3, 150.0e-3, 175.0e-3]

# Run
gsfit_controller.run()

equilibrium_ids = gsfit_controller.equilibrium_ids

# Example getting plasma current
ip = np.array([ts.global_quantities.ip.value for ts in equilibrium_ids.time_slice])
print(f"ip = {ip}")
