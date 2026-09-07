"""Robustness matrix on the public ST40 mock workload.

Runs a set of nominal and degenerate configurations and records, per case, the outcome
(converged slices, iterations, Ip, error messages printed by the solver, exceptions).
Run once per build (reference and candidate) and diff the JSON outputs.

Usage: robustness.py --tag ref --repo ../ref
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--repo", default=str(HERE.parent.parent.parent))
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--out", default=str(HERE.parent.parent.parent / "benchmark_results"))
args = parser.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(args.threads)
repo = Path(args.repo)

import numpy as np
from gsfit import Gsfit

mock_dir = str(repo / "tests/test_02_delta_z_shift_greater_than_d_z/data")
WORKFLOW = {
    "elmag": {"tree_name": "ELMAG", "pulseNo": None, "run_name": "RUN16", "usage": ""},
    "elmag_coils": {"tree_name": "ELMAG", "pulseNo": 11012050, "run_name": "RUN16", "usage": ""},
    "mag": {"tree_name": "MAG", "pulseNo": None, "run_name": "BEST", "usage": ""},
    "psu2coil": {"tree_name": "PSU2COIL", "pulseNo": None, "run_name": "RUN02", "usage": ""},
    "rog_gaps": {"tree_name": "MAG", "pulseNo": 11010605, "run_name": "RUN14C", "usage": ""},
}


def make_controller(times):
    c = Gsfit(pulseNo=12050, run_name="ROBUST", run_description="robustness", settings_path="default", write_to_mds=False)
    cs = c.settings["GSFIT_code_settings.json"]
    cs["RAYON_NUM_THREADS"] = args.threads
    cs["database_reader"]["method"] = "mock_st40_mdsplus"
    cs["database_reader"]["mock_st40_mdsplus"] = {"mock_dir": mock_dir, "workflow": WORKFLOW}
    cs["timeslices"]["method"] = "user_defined"
    cs["timeslices"]["user_defined"] = [float(t) for t in times]
    return c


def case_nominal(c):
    pass


def case_zero_initial_current(c):
    c.settings["GSFIT_code_settings.json"]["initial_guess"]["ip"] = 0.0


def case_negative_initial_current(c):
    c.settings["GSFIT_code_settings.json"]["initial_guess"]["ip"] = -425.0e3


def case_initial_ellipse_outside_vessel(c):
    c.settings["GSFIT_code_settings.json"]["initial_guess"]["r_cur"] = 0.95


def case_initial_ellipse_far_off_axis(c):
    ig = c.settings["GSFIT_code_settings.json"]["initial_guess"]
    ig["r_cur"] = 0.30
    ig["z_cur"] = 0.45
    ig["minor_radius"] = 0.08


def case_max_iter_3(c):
    c.settings["GSFIT_code_settings.json"]["numerics"]["n_iter_max"] = 3


def case_tight_tolerance(c):
    c.settings["GSFIT_code_settings.json"]["numerics"]["gs_error"] = 1.0e-9


def case_no_bp_probes(c):
    for name, v in c.settings["sensor_weights_bp_probe.json"].items():
        if isinstance(v, dict) and "fit_settings" in v:
            v["fit_settings"]["include"] = False


def case_no_flux_loops(c):
    for name, v in c.settings["sensor_weights_flux_loops.json"].items():
        if isinstance(v, dict) and "fit_settings" in v:
            v["fit_settings"]["include"] = False


def case_magnetics_minimal(c):
    # keep only the first four bp probes and two flux loops (diagnostic degeneracy)
    kept = 0
    for name, v in c.settings["sensor_weights_bp_probe.json"].items():
        if isinstance(v, dict) and "fit_settings" in v:
            if v["fit_settings"]["include"] and kept < 4:
                kept += 1
            else:
                v["fit_settings"]["include"] = False
    kept = 0
    for name, v in c.settings["sensor_weights_flux_loops.json"].items():
        if isinstance(v, dict) and "fit_settings" in v:
            if v["fit_settings"]["include"] and kept < 2:
                kept += 1
            else:
                v["fit_settings"]["include"] = False


def case_coarse_grid(c):
    c.settings["GSFIT_code_settings.json"]["grid"]["n_r"] = 61
    c.settings["GSFIT_code_settings.json"]["grid"]["n_z"] = 121


def case_fine_grid_321(c):
    c.settings["GSFIT_code_settings.json"]["grid"]["n_z"] = 321


def case_no_vertical_feedback(c):
    c.settings["GSFIT_code_settings.json"]["numerics"]["n_iter_no_vertical_feedback"] = 1000


def case_higher_order_source_functions(c):
    c.settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"] = 4
    c.settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"] = [[0.0, 0.0, 0.0, 4.467344e-05]]
    c.settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"] = 3
    c.settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"] = [[0.0, 0.0, 177.734375009]]


CASES = {
    "nominal_3_slices": (case_nominal, [129.9e-3, 130.0e-3, 130.1e-3]),
    "zero_initial_current": (case_zero_initial_current, [130e-3]),
    "negative_initial_current": (case_negative_initial_current, [130e-3]),
    "initial_ellipse_outside_vessel": (case_initial_ellipse_outside_vessel, [130e-3]),
    "initial_ellipse_far_off_axis": (case_initial_ellipse_far_off_axis, [130e-3]),
    "max_iter_3": (case_max_iter_3, [130e-3]),
    "tight_tolerance_1e-9": (case_tight_tolerance, [130e-3]),
    "no_bp_probes": (case_no_bp_probes, [130e-3]),
    "no_flux_loops": (case_no_flux_loops, [130e-3]),
    "magnetics_minimal_4bp_2fl": (case_magnetics_minimal, [130e-3]),
    "coarse_grid_61x121": (case_coarse_grid, [130e-3]),
    "fine_grid_81x321": (case_fine_grid_321, [130e-3]),
    "no_vertical_feedback": (case_no_vertical_feedback, [130e-3]),
    "higher_order_source_functions": (case_higher_order_source_functions, [130e-3]),
}

results = {}
for name, (fn, times) in CASES.items():
    record = {"times": times}
    # Capture at the file-descriptor level so the Rust solver's println! messages are recorded as well
    capture = tempfile.TemporaryFile(mode="w+")
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    sys.stdout.flush()
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)
    try:
        c = make_controller(times)
        fn(c)
        c.run()
        plasma = c.plasma
        ip = np.asarray(plasma.get_array1(["global", "ip"]))
        n_iter = [int(n) for n in plasma.get_vec_usize(["global", "n_iter"])]
        gs_err = np.asarray(plasma.get_array1(["global", "gs_error"]))
        chi = np.asarray(plasma.get_array1(["global", "chi_mag"]))
        record.update(
            {
                "outcome": "ran",
                "n_converged": int(np.isfinite(ip).sum()),
                "n_time": len(ip),
                "ip": [float(x) for x in ip],
                "n_iter": n_iter,
                "gs_error": [float(x) for x in gs_err],
                "chi_mag": [float(x) for x in chi],
                "r_mag": [float(x) for x in plasma.get_array1(["global", "r_mag"])],
                "z_mag": [float(x) for x in plasma.get_array1(["global", "z_mag"])],
                "psi_a": [float(x) for x in plasma.get_array1(["global", "psi_a"])],
                "psi_b": [float(x) for x in plasma.get_array1(["global", "psi_b"])],
            }
        )
    except BaseException as exc:  # pyo3 PanicException derives from BaseException
        record.update({"outcome": "exception", "exception": f"{type(exc).__name__}: {str(exc)[:300]}"})
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
    capture.seek(0)
    out = capture.read()
    capture.close()
    record["solver_messages"] = sorted(
        set(
            l.strip()
            for l in out.splitlines()
            if any(
                k in l
                for k in [
                    "Error",
                    "error_state",
                    "NoBoundary",
                    "NoMagnetic",
                    "NoStationary",
                    "MaxIter",
                    "Invalid",
                    "Warning",
                    "rank-deficient",
                    "norm of column",
                ]
            )
        )
    )[:12]
    record["solution_lines"] = [l.strip() for l in out.splitlines() if "solution_found" in l][:5]
    results[name] = record
    print(
        f"{name:34s} {record.get('outcome')}  converged={record.get('n_converged')}/{record.get('n_time')}  n_iter={record.get('n_iter')}  msgs={record['solver_messages'][:2]}",
        flush=True,
    )

out_path = Path(args.out) / f"robustness_{args.tag}.json"
benchlib.write_json(out_path, {"provenance": benchlib.provenance(repo), "cases": results})
print("wrote", out_path)
