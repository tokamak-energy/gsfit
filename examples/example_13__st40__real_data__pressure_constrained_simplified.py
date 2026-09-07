# Note, this example will only run inside Tokamak Energy's network
#
# Example 13: pressure-constrained GSFit for ST40, using the new simplified workflow.
#
# This reproduces `scan_gsfit__st40__real_data_pressure_constrained.py` and
# `example_10__st40__real_data__pressure_constrained.ipynb`, but almost all of the boiler-plate now lives
# inside GSFit and is driven purely by settings:
#   * Reading the Thomson scattering (TS) tree and building the pressure sensors is done by
#     `database_readers/st40_mdsplus_with_digital_filter/setup_pressure_sensors.py`.
#   * Finding the "good" time-slices (via the PPTS `BAD_MA` flag) is done by the
#     `good_pressure_sensors` timeslices method.
#   * Building the tensioned cubic B-spline regularisations is done by
#     `tensioned_cubic_splines_regularisations.py`, driven by the `make_cubic_spline_regularisations: true`
#     settings.
#
# So this example is now just a handful of settings changes followed by `run()`.

from gsfit import Gsfit

pulse_num = 14_685  # A real experimental shot
pulse_num_write = pulse_num + 52_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object (uses the "default" settings, i.e. the `st40_mdsplus_with_digital_filter` reader)
gsfit_controller = Gsfit(
    pulseNo=pulse_num,
    run_name="EX13",
    run_description="Pressure constrained GSFit using the simplified settings-driven workflow.",
    write_to_mds=True,
    pulseNo_write=pulse_num_write,
    analysis_name="GSFIT_TS",
)

# 1. Only reconstruct the time-slices where the pressure (Thomson scattering) sensors are "good".
#    This reads the `BAD_MA` flag from the PPTS tree (the PPTS run_name is set in the active method's
#    "workflow" section in GSFIT_code_settings.json).
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "good_pressure_sensors"

# 2. Turn on the pressure sensors (Thomson scattering).
gsfit_controller.settings["sensor_weights_pressure.json"]["include"] = True

# 3. Use a tensioned cubic B-spline for p'. Its regularisation matrix is built automatically from
#    source_function_p_prime.json["tensioned_cubic_b_spline"]["make_cubic_spline_regularisations"] = true.
gsfit_controller.settings["source_function_p_prime.json"]["method"] = "tensioned_cubic_b_spline"

# Run all of GSFit (read data & initialise, solve the Grad-Shafranov equation, and write the results).
gsfit_controller.run()
