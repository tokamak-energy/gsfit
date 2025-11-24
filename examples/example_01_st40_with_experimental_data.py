# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

# pulseNo = 14841  # A real experimental shot
pulseNo = 14844  # A real experimental shot
pulseNo_write = pulseNo + 11_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name="TEST03",
    run_description="Standard run",
    write_to_mds=True,
    pulseNo_write=pulseNo_write,
)
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [80.0e-3, 95.5e-3, 96.0e-3, 96.5e-3]
# gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [95.5e-3]

# Run
gsfit_controller.run()
