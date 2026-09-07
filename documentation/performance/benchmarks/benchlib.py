"""Shared helpers for the GSFit offline benchmarks.

Every benchmark records: phase wall-clock times, environment provenance
(git commit, threads, host), and a compact dump of physically meaningful
outputs for accuracy comparison between builds.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class PhaseTimer:
    def __init__(self) -> None:
        self.phases: list[tuple[str, float]] = []
        self._t0 = time.perf_counter()
        self._last = self._t0

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self.phases.append((name, dt))
        self._last = now
        return dt

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float]:
        d = {name: dt for name, dt in self.phases}
        d["total"] = self.total()
        return d


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(repo), "diff", "--quiet"]) != 0
        return out + ("-dirty" if dirty else "")
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def provenance(repo: Path) -> dict:
    import gsfit_rs

    return {
        "git_commit": git_commit(repo),
        "gsfit_rs_file": getattr(gsfit_rs, "__file__", "?"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "(unset)"),
        "loadavg_start": os.getloadavg(),
        "argv": sys.argv,
    }


def dump_results(controller, out_npz: Path, extra: dict | None = None) -> dict:
    """Dump the physically meaningful observables of a finished reconstruction."""
    plasma = controller.plasma
    d: dict[str, np.ndarray] = {}

    def grab(obj, keys, getter):
        try:
            return getattr(obj, getter)(keys)
        except BaseException:  # pyo3 PanicException derives from BaseException
            return None

    d["time"] = plasma.get_array1(["time"])
    for key in [
        "ip",
        "psi_a",
        "psi_b",
        "r_mag",
        "z_mag",
        "gs_error",
        "chi_mag",
        "r_geo",
        "z_geo",
        "elongation",
        "r_minor",
        "w_mhd",
        "li_3",
        "beta_p_1",
        "q95",
        "q0",
        "delta_z",
        "bounding_r",
        "bounding_z",
    ]:
        v = grab(plasma, ["global", key], "get_array1")
        if v is not None:
            d[f"global.{key}"] = np.asarray(v)
    d["n_iter"] = np.asarray(grab(plasma, ["global", "n_iter"], "get_vec_usize") or [], dtype=np.int64)
    d["psi_2d"] = plasma.get_array3(["profiles_2d", "r_z", "psi"])
    d["j_2d"] = plasma.get_array3(["profiles_2d", "r_z", "j"])
    d["grid_r"] = plasma.get_array1(["grid", "r"])
    d["grid_z"] = plasma.get_array1(["grid", "z"])
    d["boundary_r"] = plasma.get_array2(["boundary", "outline", "r"])
    d["boundary_z"] = plasma.get_array2(["boundary", "outline", "z"])
    d["boundary_n"] = np.asarray(plasma.get_vec_usize(["boundary", "outline", "n"]), dtype=np.int64)
    d["p_prime_coefs"] = plasma.get_array2(["source_functions", "p_prime", "coefficients"])
    d["ff_prime_coefs"] = plasma.get_array2(["source_functions", "ff_prime", "coefficients"])
    for name in ["p", "q", "f", "ff_prime", "p_prime"]:
        v = grab(plasma, ["profiles_1d", "psi_norm", name], "get_array2")
        if v is not None:
            d[f"profiles_1d.{name}"] = v
    v = grab(plasma, ["profiles_1d", "r_midplane", "p"], "get_array2")
    if v is not None:
        d["profiles_1d.r_midplane.p"] = v
    for sensors, name in [(controller.bp_probes, "bp_probes"), (controller.flux_loops, "flux_loops"), (controller.rogowski_coils, "rogowski_coils")]:
        for kind in ["b", "psi", "i"]:
            m = grab(sensors, ["*", kind, "measured", "value"], "get_array2")
            c = grab(sensors, ["*", kind, "calculated", "value"], "get_array2")
            if m is not None and c is not None:
                d[f"{name}.measured"] = m
                d[f"{name}.calculated"] = c
                break
    # passive currents
    try:
        pas = controller.passives
        for pname in pas.keys():
            v = grab(pas, [pname, "i"], "get_array1")
            if v is not None:
                d[f"passives.{pname}.i"] = np.asarray(v)
    except Exception:
        pass
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{k: v for k, v in d.items() if v is not None})
    summary = {
        "n_time": len(d["time"]),
        "n_converged": int(np.isfinite(d.get("global.ip", np.array([np.nan]))).sum()),
        "n_iter_mean": float(np.mean([n for n in d["n_iter"] if n < 10**9])) if len(d["n_iter"]) else None,
        "ip_mean": float(np.nanmean(d.get("global.ip", np.array([np.nan])))),
    }
    if extra:
        summary.update(extra)
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
