# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

pulseNo = 14_681  # A real experimental shot
pulseNo_write = pulseNo + 52_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object
gsfit_controller = Gsfit(
    pulseNo=pulseNo,
    run_name="ITER07",
    run_description="Testing include=True for DIALOOP sensor, weight=1e2",
    write_to_mds=True,
    pulseNo_write=pulseNo_write,
)
gsfit_controller.settings["sensor_weights_dialoop.json"]["DIALOOP"]["fit_settings"]["weight"] = 1e2
gsfit_controller.settings["sensor_weights_dialoop.json"]["DIALOOP"]["fit_settings"]["include"] = True
# Need to lower 
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["time_end"] = 249.0e-3

# Run
gsfit_controller.run()
