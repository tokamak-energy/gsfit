import numpy as np
from gsfit import Gsfit

pulseNo = 13343
pulseNo_write = pulseNo + 11_000_000

gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name="LIKE_RTGS",
    run_description="Using a single degree of freedom for p_prime and ff_prime; and 5 eigenmodes for the IVC",
    write_to_mds=True,
    pulseNo_write=pulseNo_write,
    settings_path="st40_setup_for_rtgsfit",
)

gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "arange"
gsfit_controller.settings["GSFIT_code_settings.json"]["database_writer"]["method"] = "tokamak_energy_mdsplus"

gsfit_controller.run()
