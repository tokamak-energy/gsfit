import os
import time

# Set RAYON_NUM_THREADS before any gsfit_rs import so Rayon picks it up on first initialisation
os.environ["RAYON_NUM_THREADS"] = "32"

import gsfit_rs
import mdsthin
import numpy as np
from gsfit import Gsfit
from scipy.interpolate import RectBivariateSpline


def print_timing(section_name: str, section_start_time: float) -> float:
    section_elapsed_time = time.perf_counter() - section_start_time
    print(f"[TIMING] {section_name}: {section_elapsed_time:.3f} s")
    return time.perf_counter()


script_start_time = time.perf_counter()
section_start_time = script_start_time

# ── Pulse numbers ─────────────────────────────────────────────────────────────
pulse_num = 13_560
pulse_num_read = pulse_num
pulse_num_write = pulse_num + 52_000_000

# ── Read TS sensor geometry ───────────────────────────────────────────────────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("TS", pulse_num)
    ts_r = conn.get("\\TS::TOP.BEST:R").data().astype(np.float64)
    ts_z = conn.get("\\TS::TOP.BEST:Z").data().astype(np.float64)

section_start_time = print_timing("Read TS sensor geometry", section_start_time)

# ── Initialise GSFit controller ───────────────────────────────────────────────
gsfit_controller = Gsfit(
    pulseNo=pulse_num,
    run_name="TEST039",
    run_description="Linear but stop at 2 (ignore 36 37 38)",
    write_to_mds=True,
    pulseNo_write=pulse_num_write,
)
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["method"] = "user_defined"
gsfit_controller.settings["GSFIT_code_settings.json"]["database_reader"]["st40_mdsplus"]["workflow"]["psu2coil"]["run_name"] = "RUN05"

section_start_time = print_timing("Initialise GSFit controller", section_start_time)

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

section_start_time = print_timing("Find good time slices", section_start_time)

# ── Read TS pressure profiles ─────────────────────────────────────────────────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("TS", pulse_num)
    ts_r = conn.get("\\TS::TOP.BEST:R").data().astype(np.float64)
    ts_z = conn.get("\\TS::TOP.BEST:Z").data().astype(np.float64)
    ts_pe = conn.get("\\TS::TOP.BEST.PROFILES:PE").data().astype(np.float64)

section_start_time = print_timing("Read TS pressure profiles", section_start_time)

# ── Map TS sensors onto normalised poloidal flux from a prior GSFit run ───────
with mdsthin.Connection("smaug") as conn:
    conn.openTree("GSFIT", pulse_num_read)
    gsfit_ppac_r = conn.get("\\GSFIT::TOP.BEST.TWO_D:RGRID").data().astype(np.float64)
    gsfit_ppac_z = conn.get("\\GSFIT::TOP.BEST.TWO_D:ZGRID").data().astype(np.float64)
    gsfit_ppac_psi = conn.get("\\GSFIT::TOP.BEST.TWO_D:PSI").data().astype(np.float64)
    gsfit_ppac_time = conn.get("\\GSFIT::TOP.BEST:TIME").data().astype(np.float64)
    gsfit_ppac_psi_a = conn.get("\\GSFIT::TOP.BEST.GLOBAL:PSI_A").data().astype(np.float64)
    gsfit_ppac_psi_b = conn.get("\\GSFIT::TOP.BEST.GLOBAL:PSI_B").data().astype(np.float64)

gsfit_ppac_psin = np.zeros_like(gsfit_ppac_psi)
for i_time, time_value in enumerate(gsfit_ppac_time):
    gsfit_ppac_psin[i_time, :, :] = (
        (gsfit_ppac_psi_a[i_time] - gsfit_ppac_psi[i_time, :, :])
        / (gsfit_ppac_psi_a[i_time] - gsfit_ppac_psi_b[i_time])
    )

# PSI is stored as (n_time, n_z, n_r); transpose to (n_r, n_z) for RectBivariateSpline
ts_psin_values = np.zeros((len(ts_time_vector), len(ts_r)))
for i_time, time_value in enumerate(ts_time_vector):
    nearest_time_index = np.argmin(np.abs(gsfit_ppac_time - time_value))
    spline = RectBivariateSpline(gsfit_ppac_r, gsfit_ppac_z, gsfit_ppac_psin[nearest_time_index, :, :].T)
    for i_ts in range(len(ts_r)):
        ts_psin_values[i_time, i_ts] = spline(ts_r[i_ts], ts_z[i_ts], grid=False)

section_start_time = print_timing("Map TS sensors onto normalised flux", section_start_time)

# ── Build pressure sensor objects ─────────────────────────────────────────────
pressure_sensors = gsfit_rs.Pressure()
ts_good_times = ts_time_vector[good_indices]
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

section_start_time = print_timing("Build pressure sensor objects", section_start_time)

# ── Set up tensioned cubic B-spline regularisation for p' ─────────────────────
interior_knots = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98])
n_dof = len(interior_knots) + 4
interval_tensions = np.ones(len(interior_knots) + 1, dtype=np.float64)

regularisations_dummy = np.zeros((1, n_dof), dtype=np.float64)
p_prime_source_function_dummy = gsfit_rs.TensionedCubicBSpline(regularisations_dummy, interior_knots, interval_tensions)
sigma1_array = p_prime_source_function_dummy.get_array1("sigma1_array")
sigma2_array = p_prime_source_function_dummy.get_array1("sigma2_array")

knots = np.concatenate(([0], interior_knots, [1]))
p_prime_regularisations_array = np.zeros((len(knots) + 2, n_dof), dtype=np.float64)
for i in range(len(knots)):
    j = i + 3
    p_prime_regularisations_array[i + 1, j - 3] = 1.0 / (sigma2_array[j - 2] * sigma1_array[j - 1])
    p_prime_regularisations_array[i + 1, j - 2] = -(1.0 / sigma2_array[j - 2] + 1.0 / sigma2_array[j - 1]) / sigma1_array[j - 1]
    p_prime_regularisations_array[i + 1, j - 1] = 1.0 / (sigma2_array[j - 1] * sigma1_array[j - 1])

# Impose dp'/dpsi_n = 0 at psi_n = 0
p_prime_regularisations_array[0, 0] = 1e2
p_prime_regularisations_array[0, 1] = -1e2
p_prime_regularisations_array = p_prime_regularisations_array[1:, :]  # remove first row

# Impose p' = 0 at psi_n = 1
p_prime_regularisations_array[-1, -1] = 1e10
# p_prime_regularisations_array = p_prime_regularisations_array[:-1, :]  # remove last row

# Reduce regs
# p_prime_regularisations_array[:-1, :] *= 1e-6

# print("p' regularisation array:")
for i in range(p_prime_regularisations_array.shape[0]):
    print("  ", end="")
    for j in range(p_prime_regularisations_array.shape[1]):
        print(f"{p_prime_regularisations_array[i, j]:9.2e}", end=" ")
    print()

section_start_time = print_timing("Set up p' regularisation", section_start_time)

# ── Configure and run GSFit ───────────────────────────────────────────────────
gsfit_controller.settings["source_function_p_prime.json"]["method"] = "tensioned_cubic_b_spline"
gsfit_controller.settings["GSFIT_code_settings.json"]["timeslices"]["user_defined"] = list(ts_good_times)
gsfit_controller.settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["regularizations"] = (
    p_prime_regularisations_array.tolist()
)

section_start_time = print_timing("Configure GSFit settings", section_start_time)

gsfit_controller.set_environment_variables()
gsfit_controller.setup_timeslices()
gsfit_controller.setup_objects()

gsfit_controller.pressure_sensors = pressure_sensors

section_start_time = print_timing("Set up GSFit objects", section_start_time)

gsfit_controller.calculate_greens()
section_start_time = print_timing("Calculate greens", section_start_time)

gsfit_controller.inverse_solver_rust()
section_start_time = print_timing("Run inverse solver", section_start_time)

gsfit_controller.write_results_to_mdsplus()
section_start_time = print_timing("Write results to MDSplus", section_start_time)

total_elapsed_time = time.perf_counter() - script_start_time
print(f"[TIMING] Total runtime: {total_elapsed_time:.3f} s")
