# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

pulseNo = 14437#13565  # A real experimental shot
# pulseNo_write = pulseNo + 11_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name="RUN04",
    run_description="Same as RUN03, more equilibrium post-processing",
    write_to_mds=True,
    settings_path="st40_P2p3_calibrations"
    # pulseNo_write=pulseNo_write,
)
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [143.0e-3]
# gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["psu2coil"]["run_name"] = "RUN02"

# Run
gsfit_controller.run()
