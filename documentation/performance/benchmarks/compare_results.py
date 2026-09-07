"""Compare two benchmark result dumps (npz) key by key.

Reports, per key: max |a-b|, max |a-b| / max|a|, and whether they are bitwise identical.
Exit code 1 if any key exceeds the relative tolerance (default 1e-9) or a key is missing.
"""

import sys

import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
a = np.load(a_path, allow_pickle=True)
b = np.load(b_path, allow_pickle=True)
keys = sorted(set(a.files) | set(b.files))
worst = 0.0
bad = []
print(f"{'key':40s} {'shape':>18s} {'max|a-b|':>12s} {'rel':>10s}  note")
for k in keys:
    if k not in a.files or k not in b.files:
        print(f"{k:40s} MISSING in {'a' if k not in a.files else 'b'}")
        bad.append(k)
        continue
    x, y = np.asarray(a[k]), np.asarray(b[k])
    if x.shape != y.shape:
        print(f"{k:40s} shape mismatch {x.shape} vs {y.shape}")
        bad.append(k)
        continue
    if x.dtype.kind not in "fc" or y.dtype.kind not in "fc":
        same = np.array_equal(x, y)
        print(f"{k:40s} {x.shape!s:>18s} {'':>12s} {'':>10s}  {'identical' if same else 'DIFFERENT'}")
        if not same:
            bad.append(k)
        continue
    fx, fy = np.isfinite(x), np.isfinite(y)
    if not np.array_equal(fx, fy):
        print(f"{k:40s} {x.shape!s:>18s} finite-pattern differs ({(~fx).sum()} vs {(~fy).sum()} non-finite)")
        bad.append(k)
        continue
    d = np.abs(x[fx] - y[fy])
    if d.size == 0:
        print(f"{k:40s} {x.shape!s:>18s} (no finite values)")
        continue
    mx = float(np.max(np.abs(x[fx]))) if fx.any() else 0.0
    md = float(d.max())
    rel = md / mx if mx > 0 else md
    note = "bitwise" if md == 0.0 else ("ok" if rel <= rtol else "EXCEEDS")
    if rel > rtol:
        bad.append(k)
    worst = max(worst, rel)
    print(f"{k:40s} {x.shape!s:>18s} {md:12.3e} {rel:10.2e}  {note}")
print(f"worst relative difference: {worst:.3e}; tolerance {rtol:.1e}; {'FAIL: ' + ', '.join(bad) if bad else 'PASS'}")
sys.exit(1 if bad else 0)
