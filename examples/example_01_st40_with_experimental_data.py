# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

pulseNo = 14477  # A real experimental shot
pulseNo_write = pulseNo + 11_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name="TEST04",
    run_description="Test run",
    write_to_mds=True,
    pulseNo_write=pulseNo_write,
)
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [185e-3]

# Run
gsfit_controller.run()
