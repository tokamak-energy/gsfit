# Note, this example will only run inside Tokamak Energy's network
#
# Example 14: pressure-constrained GSFit for ST40 with the **edge pressure as a free parameter**.
#
# This is example_13 plus one change: `numerics/pressure_edge/free`.
#
# The Grad-Shafranov equation only ever involves `p'`, so the pressure is recovered by integrating
# it, which leaves it defined only up to a constant of integration. GSFit has historically fixed
# that constant by demanding `p(psi_norm = 1) = 0`, i.e. pinning the pressure to zero at the plasma
# boundary. That is a modelling choice, not a measurement, and the Thomson scattering profile
# frequently has real pressure out at the boundary.
#
# Setting `numerics/pressure_edge/free` makes that constant an extra degree of freedom of the
# least-squares fit instead, so the pressure sensors choose the edge pressure rather than having it
# imposed on them. The fitted value is written to `PROFILES_1D.PSI_NORM:PRESSURE`, whose last point
# (`psi_norm = 1`) is the edge pressure: exactly zero in example_13, and whatever the fit chose here.
#
# What the fit chooses on this pulse (`p_axis` is this fit's own on-axis pressure):
#
#     t [ms]   p_edge     p_axis    p_edge / p_axis
#         40   -461 Pa    34.4 kPa      -1.3 %
#        100    423 Pa    24.5 kPa       1.7 %
#        170    979 Pa    16.2 kPa       6.0 %
#        220   5172 Pa    15.4 kPa      33.6 %
#
# Early in the pulse the fitted edge pressure is close to zero, so it makes little difference
# whether it is pinned there or not. By the end of the pulse it is a third of the on-axis pressure,
# so the two choices give noticeably different profiles. Whether such a large edge pressure is
# physical, or is the extra degree of freedom absorbing something else, is for the user to judge:
# the `regularisation_weight` below controls how strongly it is pulled back towards zero.

from gsfit import Gsfit

pulse_num = 14_681
pulse_num_write = pulse_num + 52_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object (uses the "default" settings, i.e. the `st40_mdsplus_with_digital_filter` reader)
gsfit_controller = Gsfit(
    pulseNo=pulse_num,
    run_name="EX14",
    run_description="Pressure constrained GSFit with the edge pressure as a free parameter of the fit.",
    write_to_mds=True,
    pulseNo_write=pulse_num_write,
    analysis_name="GSFIT_TS",
)

# 1. Only reconstruct the time-slices where the pressure (Thomson scattering) sensors are "good".
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "good_pressure_sensors"

# 2. Turn on the pressure sensors (Thomson scattering). Without these the edge pressure below has
#    nothing to inform it, and its prior would simply pull it back to zero.
gsfit_controller.settings["sensor_weights_pressure.json"]["include"] = True

# 3. Use the EFIT polynomial (quadratic, n_dof=2) for p'. This is already the library default, but
#    is set explicitly here so the demonstration does not depend on that default not changing.
gsfit_controller.settings["source_function_p_prime.json"]["method"] = "efit_polynomial"

# 4. The change this example is about: let the fit choose the pressure at the plasma boundary
#    instead of pinning it to zero.
pressure_edge_settings = gsfit_controller.settings["GSFIT_code_settings.json"]["numerics"]["pressure_edge"]
pressure_edge_settings["free"] = True

# The edge pressure is unconstrained whenever no pressure sensor lies inside the plasma, so it
# always carries a prior: one extra row, `weight * pressure_edge = 0`. The weight is in units of
# 1/Pa, and `1e-4` (the default, set explicitly here) is deliberately weak: a 1 kPa edge pressure
# costs a residual of 0.1, against ~10 for a typical 2 kPa pressure sensor. Raise it to pull the
# result back towards example_13's pinned-to-zero behaviour.
pressure_edge_settings["regularisation_weight"] = 1.0e-4

# Run all of GSFit (read data & initialise, solve the Grad-Shafranov equation, and write the results).
gsfit_controller.run()
