# Note, this example will only run inside Tokamak Energy's network
#
# Scan: pressure-constrained GSFit for ST40, using the new simplified workflow.
#
# All of the boiler-plate that this script used to contain (reading the Thomson scattering tree, finding
# the good time-slices via the PPTS `BAD_MA` flag, building the pressure sensors, and constructing the
# tensioned cubic B-spline regularisations) now lives inside GSFit and is driven purely by settings:
#   * Pressure sensors:    database_readers/st40_mdsplus_with_digital_filter/setup_pressure_sensors.py
#   * Good time-slices:    the `good_pressure_sensors` timeslices method
#   * p' regularisations:  tensioned_cubic_splines_regularisations.py (via `make_cubic_spline_regularisations: true`)
#
# So a scan is now just a list of settings changes.

import time

from gsfit import Gsfit


def print_timing(section_name: str, section_start_time: float) -> float:
    section_elapsed_time = time.perf_counter() - section_start_time
    print(f"[TIMING] {section_name}: {section_elapsed_time:.3f} s")
    return time.perf_counter()


script_start_time = time.perf_counter()

# -- Pulse numbers -------------------------------------------------------------
pulse_num = 14_685
pulse_num_write = pulse_num + 52_000_000
psu2coil_run_name = "RUN05"

# -- Scan configurations -------------------------------------------------------
# regularisation_scale: multiplies the "second derivative = 0" regularisation rows.
# right_boundary_condition: "free" (p' unconstrained at psi_n = 1) or "dirichlet" (p' = 0 at psi_n = 1).
# use_dialoop: include the diamagnetic loop sensor in the fit.
scan_configs = [
    {
        "run_name": "SCAN04_01",
        "run_description": "Magnetics only, no pressure constraints.",
        "use_pressure": False,
        "use_dialoop": False,
        "regularisation_scale": 1e-6,
        "right_boundary_condition": "free",
    },
    {
        "run_name": "SCAN04_02",
        "run_description": "Pressure constrained, high regs (x1e-6), free BC.",
        "use_pressure": True,
        "use_dialoop": False,
        "regularisation_scale": 1e-6,
        "right_boundary_condition": "free",
    },
    {
        "run_name": "SCAN04_03",
        "run_description": "Pressure constrained, high regs (x1e-6), Dirichlet BC.",
        "use_pressure": True,
        "use_dialoop": False,
        "regularisation_scale": 1e-6,
        "right_boundary_condition": "dirichlet",
    },
    {
        "run_name": "SCAN04_04",
        "run_description": "Pressure constrained, low regs (x1e-7), free BC.",
        "use_pressure": True,
        "use_dialoop": False,
        "regularisation_scale": 1e-7,
        "right_boundary_condition": "free",
    },
    {
        "run_name": "SCAN04_05",
        "run_description": "Pressure constrained, high regs (x1e-6), free BC, with diamagnetic loop.",
        "use_pressure": True,
        "use_dialoop": True,
        "regularisation_scale": 1e-6,
        "right_boundary_condition": "free",
    },
]

# -- Run scan ------------------------------------------------------------------
n_configs = len(scan_configs)
for i_config, config in enumerate(scan_configs):
    print(f"\n{'=' * 60}")
    print(f"Running config {i_config + 1}/{n_configs}: {config['run_name']}")
    print(f"  {config['run_description']}")
    print(f"{'=' * 60}\n")

    config_start_time = time.perf_counter()

    gsfit_controller = Gsfit(
        pulseNo=pulse_num,
        run_name=config["run_name"],
        run_description=config["run_description"],
        write_to_mds=True,
        pulseNo_write=pulse_num_write,
    )

    # Reconstruct only the time-slices where the pressure sensors are good (PPTS `BAD_MA` flag)
    gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "good_pressure_sensors"
    database_reader_settings = gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus_with_digital_filter"]
    database_reader_settings["workflow"]["psu2coil"]["run_name"] = psu2coil_run_name

    # Turn the pressure sensors (Thomson scattering) on / off
    gsfit_controller.settings["sensor_weights_pressure.json"]["include"] = config["use_pressure"]

    # Turn the diamagnetic loop on / off
    gsfit_controller.settings["sensor_weights_dialoop.json"]["DIALOOP"]["fit_settings"]["include"] = config["use_dialoop"]

    if config["use_pressure"]:
        # Use a tensioned cubic B-spline for p', with regularisations built from settings
        gsfit_controller.settings["source_function_p_prime.json"]["method"] = "tensioned_cubic_b_spline"
        p_prime_settings = gsfit_controller.settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]
        regularisation_builder = p_prime_settings["regularisation_builder"]
        regularisation_builder["regularisation_scale"] = config["regularisation_scale"]
        regularisation_builder["right_boundary_condition"] = config["right_boundary_condition"]

    gsfit_controller.run()

    config_elapsed_time = time.perf_counter() - config_start_time
    print(f"[TIMING] {config['run_name']} total: {config_elapsed_time:.3f} s")

total_elapsed_time = time.perf_counter() - script_start_time
print(f"\n[TIMING] Total scan runtime: {total_elapsed_time:.3f} s")
