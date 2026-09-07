"""ST40 mock-MDSplus benchmark: real pulse 12050 magnetics (mock trees from tests/),
default settings (81 x 161 grid, PF coils, 8 passives incl. IVC eigenmodes), N time-slices.

This is the public proxy for GSFit's standard post-shot ST40 workflow.
Usage: bench_st40_mock.py --n-time 31 --threads 8 --tag baseline --out results/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib

parser = argparse.ArgumentParser()
parser.add_argument("--n-time", type=int, default=31, help="number of time-slices (<= 31 uses distinct mock times; more repeats the window)")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--n-z", type=int, default=None, help="override grid n_z (default from settings: 161)")
parser.add_argument("--tag", default="run")
parser.add_argument("--out", default=str(HERE.parent.parent.parent / "benchmark_results"))
parser.add_argument("--repo", default=str(HERE.parent.parent.parent))
args = parser.parse_args()

os.environ["RAYON_NUM_THREADS"] = str(args.threads)
repo = Path(args.repo)
out_dir = Path(args.out)

import numpy as np

timer = benchlib.PhaseTimer()
from gsfit import Gsfit

timer.mark("import")

mock_dir = str(repo / "tests/test_02_delta_z_shift_greater_than_d_z/data")
# The mock trees hold 31 real samples in a 0.3 ms window around 130 ms (MAG float32, PSU2COIL float64);
# pick reconstruction times strictly inside the intersection of both time axes so every
# sensor and coil signal is interpolated rather than extrapolated.
mag_t = np.load(f"{mock_dir}/mdsplus_mock_mag_12050.npz")["TIME"].astype(np.float64)
psu_t = np.load(f"{mock_dir}/mdsplus_mock_psu2coil_12050.npz")["TIME"].astype(np.float64)
t_lo = max(mag_t.min(), psu_t.min()) + 1e-6
t_hi = min(mag_t.max(), psu_t.max()) - 1e-6
times = np.array([130e-3]) if args.n_time == 1 else np.linspace(t_lo, t_hi, args.n_time)

controller = Gsfit(pulseNo=12050, run_name="BENCH", run_description="benchmark", settings_path="default", write_to_mds=False)
cs = controller.settings["GSFIT_code_settings.json"]
cs["RAYON_NUM_THREADS"] = args.threads
cs["database_reader"]["method"] = "mock_st40_mdsplus"
cs["database_reader"]["mock_st40_mdsplus"] = {
    "mock_dir": mock_dir,
    "workflow": {
        "elmag": {"tree_name": "ELMAG", "pulseNo": None, "run_name": "RUN16", "usage": ""},
        "elmag_coils": {"tree_name": "ELMAG", "pulseNo": 11012050, "run_name": "RUN16", "usage": ""},
        "mag": {"tree_name": "MAG", "pulseNo": None, "run_name": "BEST", "usage": ""},
        "psu2coil": {"tree_name": "PSU2COIL", "pulseNo": None, "run_name": "RUN02", "usage": ""},
        "rog_gaps": {"tree_name": "MAG", "pulseNo": 11010605, "run_name": "RUN14C", "usage": ""},
    },
}
if args.n_z:
    cs["grid"]["n_z"] = args.n_z
cs["timeslices"]["method"] = "user_defined"
cs["timeslices"]["user_defined"] = [float(t) for t in times]
timer.mark("settings")

# Replicate Gsfit.run() phase by phase so each phase is timed separately
controller.set_environment_variables()
controller.setup_timeslices()
controller.setup_objects()
timer.mark("setup_objects")
controller.calculate_greens()
timer.mark("calculate_greens")
controller.inverse_solver_rust()
timer.mark("inverse_solver")
controller.write_results_to_mdsplus()
timer.mark("map_results")

summary = benchlib.dump_results(controller, out_dir / f"st40mock_{args.tag}.npz")
timer.mark("dump")
payload = {
    "workload": "st40_mock_12050",
    "n_time": len(times),
    "grid": {"n_r": cs["grid"]["n_r"], "n_z": cs["grid"]["n_z"]},
    "threads": args.threads,
    "phases_s": timer.as_dict(),
    "summary": summary,
    "provenance": benchlib.provenance(repo),
    "loadavg_end": os.getloadavg(),
}
benchlib.write_json(out_dir / f"st40mock_{args.tag}.json", payload)
print("PHASES", {k: round(v, 3) for k, v in payload["phases_s"].items()})
print("SUMMARY", summary)
