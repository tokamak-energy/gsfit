"""Execute a GSFit example notebook's code cells headlessly and dump the reconstruction results.

Usage: run_notebook.py examples/example_07__....ipynb --tag baseline [--threads 8] [--out results/]
The notebook is executed as one script (IPython magics stripped, matplotlib Agg backend), the total
wall-clock is recorded, and `gsfit_controller` results are dumped with benchlib.dump_results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib

parser = argparse.ArgumentParser()
parser.add_argument("notebook")
parser.add_argument("--tag", required=True)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--out", default=str(HERE.parent.parent.parent / "benchmark_results"))
parser.add_argument("--repo", default=None)
args = parser.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(args.threads)
os.environ["MPLBACKEND"] = "Agg"
nb_path = Path(args.notebook).resolve()
repo = Path(args.repo) if args.repo else nb_path.parent.parent
out_dir = Path(args.out)
os.chdir(nb_path.parent)

with open(nb_path) as fh:
    nb = json.load(fh)
lines: list[str] = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    for line in "".join(cell["source"]).splitlines():
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
source = "\n".join(lines)

namespace: dict = {"__name__": "__main__"}
t0 = time.perf_counter()
exec(compile(source, str(nb_path), "exec"), namespace)
wall = time.perf_counter() - t0
controller = namespace.get("gsfit_controller")
summary = benchlib.dump_results(controller, out_dir / f"{nb_path.stem}_{args.tag}.npz") if controller is not None else {}
payload = {
    "workload": nb_path.name,
    "threads": args.threads,
    "wall_s": wall,
    "summary": summary,
    "provenance": benchlib.provenance(repo),
    "loadavg_end": os.getloadavg(),
}
benchlib.write_json(out_dir / f"{nb_path.stem}_{args.tag}.json", payload)
print("NOTEBOOK", nb_path.name, "wall_s", round(wall, 2), "SUMMARY", summary)
