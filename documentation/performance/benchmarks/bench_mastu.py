"""MAST-U FreeGSNKE synthetic benchmark (reproduces examples/example_02 without plotting).

Phase timing: FreeGSNKE forward solve (not GSFit), then GSFit setup, Greens, inverse solve,
result mapping, plus the isoflux re-solve from the notebook's section 2.3.
Usage: bench_mastu.py --n-time 2 --threads 8 --tag baseline
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib

parser = argparse.ArgumentParser()
parser.add_argument("--n-time", type=int, default=2)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--tag", default="run")
parser.add_argument("--out", default=str(HERE.parent.parent.parent / "benchmark_results"))
parser.add_argument("--repo", default=str(HERE.parent.parent.parent))
parser.add_argument("--no-isoflux", action="store_true")
parser.add_argument("--perturb-psi", type=float, default=0.0, help="scale the FreeGSNKE plasma flux by (1 + perturb_psi) after the forward solve; a rounding-sensitivity control")
parser.add_argument("--perturb-weights", type=float, default=0.0, help="scale every sensor fit weight by (1 + perturb_weights); a rounding-sensitivity control")
parser.add_argument("--perturb", type=float, default=0.0, help="scale every coil current by (1 + perturb) before the forward solve; a rounding-sensitivity control")
args = parser.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(args.threads)
repo = Path(args.repo)
out_dir = Path(args.out)

import numpy as np

timer = benchlib.PhaseTimer()
import gsfit_rs
from freegsnke import GSstaticsolver
from freegsnke import build_machine
from freegsnke import equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp
from gsfit import Gsfit

timer.mark("import")

cfg = Path(os.path.expanduser("~/github/freegsnke/machine_configs/MAST-U"))
tok = build_machine.tokamak(
    active_coils_path=str(cfg / "MAST-U_like_active_coils.pickle"),
    passive_coils_path=str(cfg / "MAST-U_like_passive_coils.pickle"),
    limiter_path=str(cfg / "MAST-U_like_limiter.pickle"),
    wall_path=str(cfg / "MAST-U_like_wall.pickle"),
    magnetic_probe_path=str(cfg / "MAST-U_like_magnetic_probes.pickle"),
)
eq = equilibrium_update.Equilibrium(tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129)
profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=600.0e3, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
solver = GSstaticsolver.NKGSsolver(eq)
with open(os.path.expanduser("~/github/freegsnke/examples/data/simple_diverted_currents_PaxisIp.pk"), "rb") as f:
    currents = pickle.load(f)
for k, v in currents.items():
    eq.tokamak[k].current = v * (1.0 + args.perturb)
solver.solve(eq=eq, profiles=profiles, constrain=None, target_relative_tolerance=1e-9)
eq._updatePlasmaPsi(eq.plasma_psi * (1.0 + args.perturb_psi))
tok.probes.initialise_setup(eq)
timer.mark("freegsnke_forward")

controller = Gsfit(pulseNo=0, run_name="bench", settings_path="mastu_with_synthetic_data_from_freegsnke", write_to_mds=False)
controller.settings["GSFIT_code_settings.json"]["RAYON_NUM_THREADS"] = args.threads
if args.perturb_weights != 0.0:
    for weights_file in ["sensor_weights_bp_probe.json", "sensor_weights_flux_loops.json"]:
        for sensor_settings in controller.settings[weights_file].values():
            if isinstance(sensor_settings, dict) and "fit_settings" in sensor_settings:
                sensor_settings["fit_settings"]["weight"] = sensor_settings["fit_settings"]["weight"] * (1.0 + args.perturb_weights)
# As in the notebook: sensor data at time = [0.0, 1.0] from two (identical) equilibria;
# reconstruct at the settings' user_defined [0.5] for one slice, else n_time slices across the window
time = np.array([0.0, 1.0])
eqs = [eq, eq]
kwargs = dict(time=time, freegsnke_eqs=eqs)
controller.set_environment_variables()
controller.setup_timeslices()
if args.n_time != 1:
    controller.results["TIME"] = np.linspace(0.0, 1.0, args.n_time)
controller.setup_objects(**kwargs)
timer.mark("setup_objects")
controller.calculate_greens()
timer.mark("calculate_greens")
controller.inverse_solver_rust()
timer.mark("inverse_solver")
controller.write_results_to_mdsplus()
timer.mark("map_results")
ip = controller["GLOBAL"]["IP"]
summary = benchlib.dump_results(controller, out_dir / f"mastu_{args.tag}.npz", {"ip_err_abs_max": float(np.nanmax(np.abs(ip - 600e3)))})
timer.mark("dump")

if not args.no_isoflux:
    # Section 2.3 of the notebook: add two isoflux constraints on the mid-plane and re-solve
    import shapely

    r1 = np.array([0.6, 0.8])
    mid_r = eq.R_1D
    mid_psi_n = eq.psiNRZ(mid_r, 0.0 * mid_r)
    mid_p = eq.pressure(mid_psi_n)
    r2 = np.full(2, np.nan)
    for i in range(2):
        pline = shapely.geometry.LineString(np.column_stack((mid_r, mid_p)))
        line = shapely.geometry.LineString(np.column_stack(([r1[i], r1[i]], [0.0, 1e8])))
        p_i = pline.intersection(line).y
        iso = shapely.geometry.LineString(np.column_stack(([0.0, 10.0], [p_i, p_i])))
        r2[i] = iso.intersection(pline).geoms[1].x
    isoflux = gsfit_rs.Isoflux()
    ttr = controller.results["TIME"]
    for i in range(2):
        isoflux.add_sensor(
            name=f"isoflux_constraint_{i + 1}",
            fit_settings_comment="",
            fit_settings_include=True,
            fit_settings_weight=1.0,
            time=time,
            location_1_r=np.full(len(time), r1[i]),
            location_1_z=np.zeros(len(time)),
            location_2_r=np.full(len(time), r2[i]),
            location_2_z=np.zeros(len(time)),
            times_to_reconstruct=ttr,
        )
    controller.isoflux = isoflux
    controller.calculate_greens()
    timer.mark("isoflux_greens")
    controller.inverse_solver_rust()
    timer.mark("isoflux_solve")
    controller.write_results_to_mdsplus()
    summary["isoflux"] = benchlib.dump_results(controller, out_dir / f"mastu_isoflux_{args.tag}.npz")
    timer.mark("isoflux_map_dump")

payload = {
    "workload": "mastu_freegsnke_example02",
    "n_time": args.n_time,
    "threads": args.threads,
    "phases_s": timer.as_dict(),
    "summary": summary,
    "provenance": benchlib.provenance(repo),
    "loadavg_end": os.getloadavg(),
}
benchlib.write_json(out_dir / f"mastu_{args.tag}.json", payload)
print("PHASES", {k: round(v, 3) for k, v in payload["phases_s"].items()})
print("SUMMARY", summary)
