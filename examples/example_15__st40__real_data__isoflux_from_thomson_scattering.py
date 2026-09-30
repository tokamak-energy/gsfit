# Note, this example will only run inside Tokamak Energy's network
#
# Example 15: GSFit for ST40 constrained by magnetics plus **isoflux** sensors derived from Thomson
# scattering (TS), instead of by TS pressure sensors.
#
# A pressure sensor (example_13, example_14) tells GSFit "the total pressure at this (R, Z) is p". That is a
# strong statement: it needs the TS electron pressure to be absolutely calibrated, and it needs an assumed ion
# contribution to turn it into the total pressure that GSFit fits.
#
# An isoflux sensor says something much weaker: "these two points lie on the same flux surface",
#
#     psi(r_1, z_1) - psi(r_2, z_2) = 0
#
# which is linear in every degree of freedom and has a right-hand side of exactly zero. No absolute
# calibration and no ion-pressure assumption enter anywhere, because the constraint is unchanged if the
# measured profile is scaled by any constant. What it constrains is the *shape* of the flux surfaces, not the
# magnitude of the pressure.
#
# The pairs come from the TS profile itself: if the measured quantity is a flux function, the two radii at
# which the profile crosses a given level are on the same flux surface. Cutting the profile at several levels
# gives several pairs per time-slice.

from gsfit import Gsfit

pulse_num = 14_681
pulse_num_write = pulse_num + 52_000_000  # Write to a "million" modelling pulse number

# Construct the GSFit object (uses the "default" settings, i.e. the `st40_mdsplus_with_digital_filter` reader)
gsfit_controller = Gsfit(
    pulseNo=pulse_num,
    run_name="EX15",
    run_description="Magnetics plus Thomson scattering isoflux constraints; no pressure sensors.",
    write_to_mds=True,
    pulseNo_write=pulse_num_write,
    analysis_name="GSFIT_TS",
)

# 1. Only reconstruct the time-slices where the Thomson scattering data is "good". The isoflux pairs come from
#    the same diagnostic as the pressure sensors, so the same time-slice selection applies.
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "good_pressure_sensors"

# 2. Turn the isoflux sensors on. One sensor is created per entry in `level_fractions` below.
gsfit_controller.settings["sensor_weights_isoflux.json"]["include"] = True

# 3. Leave the pressure sensors off (the default), so the only Thomson scattering information reaching the fit
#    is the isoflux constraint. Set this to True to use both.
gsfit_controller.settings["sensor_weights_pressure.json"]["include"] = False

# 4. Pair on the electron temperature.
#
#    "TE" is the default despite being the noisiest of the three available quantities. Force balance gives
#    `B . grad p = 0` only for the *total* pressure and only without rotation; with toroidal rotation the
#    centrifugal term pushes density outboard on a flux surface, so "NE" -- and hence "PE" -- are not exactly
#    flux functions. Fast parallel heat conduction keeps "TE" one regardless.
#
#    The character of the error matters more than its size: the centrifugal term is anti-symmetric between the
#    high-field and low-field sides, so it biases every pair the same way and averaging over pairs does not
#    reduce it. The larger "TE" error is random, so it does average down. Running this again with "NE" is a
#    useful cross-check -- a disagreement has the known functional form above.
thomson_scattering_settings = gsfit_controller.settings["sensor_weights_isoflux.json"]["thomson_scattering"]
thomson_scattering_settings["quantity"] = "TE"

# 5. Levels at which to cut the profile, as fractions of its peak value. Levels near 1.0 are useless (both
#    crossings collapse onto the peak, so the pair carries almost no information) and levels near 0.0 land in
#    the edge, where the TS uncertainties blow up.
thomson_scattering_settings["level_fractions"] = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

# 6. Guards on the pair-finding. A level is rejected -- leaving that (time, level) out of the fit entirely --
#    rather than guessed at, whenever the data cannot support it:
#      * `max_relative_error`: drop TS channels whose relative uncertainty exceeds this.
#      * `max_gap`: refuse to place a crossing by interpolating across a radial hole wider than this [metre],
#        such as one left behind by the uncertainty filter.
#      * `min_points_per_branch`: how many usable channels each of the two branches needs.
#      * `r_limits`: discard pairs straying outside this radial range [metre]. A flat-cored profile puts its
#        crossings far out in the scrape-off layer, where the two points are no longer meaningfully on the
#        same closed flux surface.
thomson_scattering_settings["max_relative_error"] = 0.1
thomson_scattering_settings["max_gap"] = 0.08
thomson_scattering_settings["min_points_per_branch"] = 3
thomson_scattering_settings["r_limits"] = [0.25, 0.8]

# 7. Weight given to each isoflux row. The residual is a flux difference in Wb and the expected value is 1.0,
#    so the effective weight is this number directly, in 1/Wb. For comparison, the flux loops work out at
#    `2 * pi * weight / expected_value`, i.e. roughly 2e3 to 4e3 with the default settings, so 1e3 puts an
#    isoflux row somewhat below a single flux loop.
gsfit_controller.settings["sensor_weights_isoflux.json"]["fit_settings"]["weight"] = 1.0e3

# Run all of GSFit (read data & initialise, solve the Grad-Shafranov equation, and write the results).
gsfit_controller.run()
