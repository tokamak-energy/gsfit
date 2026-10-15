# Note, this example will only run inside Tokamak Energy's network

from gsfit import Gsfit

# pulseNo = 13662
# 12739 = no ELMAG ??
# 12738 = no DIALOOP ??
for pulseNo in [11842]:
    # pulseNo = 13542 # type-I ELMs 94 -> 112ms !!!
    pulseNo_write = pulseNo + 11_000_000  # Write to a "million" modelling pulse number

    # Construct the GSFit object
    gsfit_controller = Gsfit(
        pulseNo=pulseNo,
        run_name="RUN04",
        run_description="Same as RUN03, more equilibrium post-processing",
        # run_description="Same as RUN04, higher time resolution",
        # run_description="Same as RUN04, on magnetics time scale",
        write_to_mds=True,
        settings_path="st40_P2p3_calibrations",
        pulseNo_write=pulseNo_write,
    )
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = [143.0e-3]

    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["time_start"] = 20.0e-03
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["time_end"] = 230.0e-03
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["dt"] = 0.05e-03

    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["time_start"] = 94.0e-03
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["time_end"] = 112.0e-03
    # gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["arange"]["dt"] = 0.01e-03

    gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["method"] = "st40_mdsplus"
    gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["psu2coil"]["run_name"] = "RUN02"

    # Run
    gsfit_controller.run()
