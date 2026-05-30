import os
import time

# Set RAYON_NUM_THREADS before any gsfit_rs import so Rayon picks it up on first initialisation
os.environ["RAYON_NUM_THREADS"] = "32"

import gsfit_rs
import mdsthin
import numpy as np
from gsfit import Gsfit


def print_timing(section_name: str, section_start_time: float) -> float:
    section_elapsed_time = time.perf_counter() - section_start_time
    print(f"[TIMING] {section_name}: {section_elapsed_time:.3f} s")
    return time.perf_counter()


script_start_time = time.perf_counter()
section_start_time = script_start_time

# ── Pulse numbers ─────────────────────────────────────────────────────────────
# pulse_num = 13_560
pulse_num = 13_599
pulse_num_write = pulse_num + 52_000_000
# psu2coil_run_name = "RUN06"
psu2coil_run_name = "RUN05"

# ── Read TS sensor geometry ───────────────────────────────────────────────────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("TS", pulse_num)
    ts_r = conn.get("\\TS::TOP.BEST:R").data().astype(np.float64)
    ts_z = conn.get("\\TS::TOP.BEST:Z").data().astype(np.float64)

section_start_time = print_timing("Read TS sensor geometry", section_start_time)

# ── Find good time slices via BAD_MA flag ─────────────────────────────────────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("PPTS", pulse_num)
    ts_time_vector = conn.get("\\PPTS::TOP.BEST:TIME").data().astype(np.float64)
    ts_bad_ma_times = conn.get("\\PPTS::TOP.BEST.GLOBAL:BAD_MA").data().astype(bool)

for i, (time_value, bad_ma) in enumerate(zip(ts_time_vector, ts_bad_ma_times)):
    if bad_ma:
        print(f"\033[91m{i:02d} {time_value:6.3f} s\033[0m", end="  ")  # Red: bad
    else:
        print(f"\033[92m{i:02d} {time_value:6.3f} s\033[0m", end="  ")  # Green: good
    if (i + 1) % 10 == 0:
        print()

good_indices = np.where(~ts_bad_ma_times)[0]
print(f"\nGood time slice indices: {good_indices}")
print(f"Number of good time slices: {len(good_indices)}")

ts_good_times = ts_time_vector[good_indices]

section_start_time = print_timing("Find good time slices", section_start_time)

# ── Read TS pressure profiles ─────────────────────────────────────────────────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("TS", pulse_num)
    ts_pe = conn.get("\\TS::TOP.BEST.PROFILES:PE").data().astype(np.float64)

section_start_time = print_timing("Read TS pressure profiles", section_start_time)

# ── Build pressure sensor object ──────────────────────────────────────────────
pressure_sensors = gsfit_rs.Pressure()
for i_ts_sensor in range(len(ts_r)):
    measured_pressure_full = 2 * ts_pe[good_indices, i_ts_sensor]
    measured_pressure_full = measured_pressure_full.copy()
    measured_pressure_full[measured_pressure_full <= 0.5e3] = np.nan
    if np.any(~np.isnan(measured_pressure_full)):
        pressure_sensors.add_sensor(
            name=f"TS_{i_ts_sensor + 1}",
            geometry_r=ts_r[i_ts_sensor],
            geometry_z=ts_z[i_ts_sensor],
            fit_settings_comment="",
            fit_settings_expected_value=2e3,
            fit_settings_include=True,
            fit_settings_weight=10.0,
            time=ts_good_times,
            measured=measured_pressure_full,
        )

section_start_time = print_timing("Build pressure sensor object", section_start_time)

# ── Build base p' regularisation array ───────────────────────────────────────
interior_knots = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98])
n_dof = len(interior_knots) + 4
interval_tensions = np.ones(len(interior_knots) + 1, dtype=np.float64)

regularisations_dummy = np.zeros((1, n_dof), dtype=np.float64)
p_prime_source_function_dummy = gsfit_rs.TensionedCubicBSpline(regularisations_dummy, interior_knots, interval_tensions)
sigma1_array = p_prime_source_function_dummy.get_array1("sigma1_array")
sigma2_array = p_prime_source_function_dummy.get_array1("sigma2_array")

knots = np.concatenate(([0], interior_knots, [1]))
p_prime_regularisations_base = np.zeros((len(knots) + 2, n_dof), dtype=np.float64)
for i in range(len(knots)):
    j = i + 3
    p_prime_regularisations_base[i + 1, j - 3] = 1.0 / (sigma2_array[j - 2] * sigma1_array[j - 1])
    p_prime_regularisations_base[i + 1, j - 2] = -(1.0 / sigma2_array[j - 2] + 1.0 / sigma2_array[j - 1]) / sigma1_array[j - 1]
    p_prime_regularisations_base[i + 1, j - 1] = 1.0 / (sigma2_array[j - 1] * sigma1_array[j - 1])

# Impose dp'/dpsi_n = 0 at psi_n = 0
p_prime_regularisations_base[0, 0] = 1e10
p_prime_regularisations_base[0, 1] = -1e10
p_prime_regularisations_base = p_prime_regularisations_base[1:, :]  # remove first row

# Impose p' = 0 at psi_n = 1 (Dirichlet BC row — optionally removed per config below)
p_prime_regularisations_base[-1, -1] = 1e10

section_start_time = print_timing("Build p' regularisation base", section_start_time)

# ── Scan configurations ───────────────────────────────────────────────────────
# reg_scale: None = linear (no scaling), 1e-6 = high regs, 1e-7 = low regs
# remove_last_row: False = keep Dirichlet BC (p'=0 at psi_n=1), True = free BC
scan_configs = [
    {
        "run_name": "SCAN02_01",
        "run_description": "Magnetics only, no pressure constraints.",
        "use_pressure": False,
        "remove_last_row": False,
        "reg_scale": None,
    },
    {
        "run_name": "SCAN02_02",
        "run_description": "Pressure constrained, linear regs, Dirichlet BC.",
        "use_pressure": True,
        "remove_last_row": False,
        "reg_scale": None,
    },
    {
        "run_name": "SCAN02_03",
        "run_description": "Pressure constrained, linear regs, free BC.",
        "use_pressure": True,
        "remove_last_row": True,
        "reg_scale": None,
    },
    {
        "run_name": "SCAN02_04",
        "run_description": "Pressure constrained, high regs (x1e-6), Dirichlet BC.",
        "use_pressure": True,
        "remove_last_row": False,
        "reg_scale": 1e-6,
    },
    {
        "run_name": "SCAN02_05",
        "run_description": "Pressure constrained, high regs (x1e-6), free BC.",
        "use_pressure": True,
        "remove_last_row": True,
        "reg_scale": 1e-6,
    },
    {
        "run_name": "SCAN02_06",
        "run_description": "Pressure constrained, low regs (x1e-7), Dirichlet BC.",
        "use_pressure": True,
        "remove_last_row": False,
        "reg_scale": 1e-7,
    },
    {
        "run_name": "SCAN02_07",
        "run_description": "Pressure constrained, low regs (x1e-7), free BC.",
        "use_pressure": True,
        "remove_last_row": True,
        "reg_scale": 1e-7,
    },
]

# ── Run scan ──────────────────────────────────────────────────────────────────
n_configs = len(scan_configs)
for i_config, config in enumerate(scan_configs):
    print(f"\n{'=' * 60}")
    print(f"Running config {i_config + 1}/{n_configs}: {config['run_name']}")
    print(f"  {config['run_description']}")
    print(f"{'=' * 60}\n")

    config_start_time = time.perf_counter()
    section_start_time = config_start_time

    # Build p' regularisation array for this config
    p_prime_reg = p_prime_regularisations_base.copy()
    if config["reg_scale"] is not None:
        p_prime_reg[:-1, :] *= config["reg_scale"]  # do not scale the Dirichlet row
    if config["remove_last_row"]:
        p_prime_reg = p_prime_reg[:-1, :]  # remove Dirichlet BC row

    # Initialise GSFit controller
    gsfit_controller = Gsfit(
        pulseNo=pulse_num,
        run_name=config["run_name"],
        run_description=config["run_description"],
        write_to_mds=True,
        pulseNo_write=pulse_num_write,
    )
    gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
    gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = list(ts_good_times)
    gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["psu2coil"]["run_name"] = psu2coil_run_name

    if config["use_pressure"]:
        gsfit_controller.settings["source_function_p_prime.json"]["method"] = "tensioned_cubic_b_spline"
        gsfit_controller.settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["regularizations"] = p_prime_reg.tolist()

    section_start_time = print_timing(f"[{config['run_name']}] Initialise controller", section_start_time)

    gsfit_controller.set_environment_variables()
    gsfit_controller.setup_timeslices()
    gsfit_controller.setup_objects()

    if config["use_pressure"]:
        gsfit_controller.pressure_sensors = pressure_sensors
    else:
        gsfit_controller.pressure_sensors = gsfit_rs.Pressure()

    section_start_time = print_timing(f"[{config['run_name']}] Set up objects", section_start_time)

    gsfit_controller.calculate_greens()
    section_start_time = print_timing(f"[{config['run_name']}] Calculate Greens", section_start_time)

    gsfit_controller.inverse_solver_rust()
    section_start_time = print_timing(f"[{config['run_name']}] Run inverse solver", section_start_time)

    gsfit_controller.write_results_to_mdsplus()
    section_start_time = print_timing(f"[{config['run_name']}] Write results", section_start_time)

    config_elapsed_time = time.perf_counter() - config_start_time
    print(f"[TIMING] {config['run_name']} total: {config_elapsed_time:.3f} s")

total_elapsed_time = time.perf_counter() - script_start_time
print(f"\n[TIMING] Total scan runtime: {total_elapsed_time:.3f} s")
