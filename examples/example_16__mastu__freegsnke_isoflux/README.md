# Example 16: FreeGSNKE reference with isoflux constraints

Open [example_16.ipynb](example_16.ipynb) in VS Code or Jupyter for the
walkthrough, code, inline figures and results. Saved outputs can be read without
running it. Supporting routines are in [helpers.py](helpers.py).

FreeGSNKE provides a known MAST-U equilibrium. Compare three GSFit reconstructions:

1. Magnetics only (polynomial): two EFIT coefficients per source.
2. Magnetics + isoflux (polynomial): the same basis with 24 equal-flux pairs.
3. Magnetics + isoflux (spline): eleven tensioned cubic B-spline coefficients per source.

The pairs lie on horizontal and vertical chords through the reference axis,
at twelve flux levels from `psi_N = 0.01` to `1.0`. **No pressure measurements
are fitted**; pressure recovery is a diagnostic only. Labels such as `2 + 2`
and `11 + 11` mean p-prime + FF-prime degrees of freedom, not total fit parameters.
No experimental database connection is needed.

## Set up and run

From the GSFit repository root:

```bash
uv venv --python 3.13
```

Follow the activation instructions printed by `uv` for your shell and operating
system, then install:

```bash
uv pip install -e ".[with_freegs_and_freegsnke]" ipykernel
```

Use the development environment with its Rust extension rebuilt after solver
changes. Select `.venv` in the notebook's kernel picker.

The first code cell defaults to `~/github/freegsnke`
(`/home/alex.prokopyszyn/github/freegsnke` on this machine). Use a **trusted**
checkout containing the MAST-U configuration and example current files:
these inputs are pickle files and must not come from an untrusted source.

Adjust that path if needed, then select **Restart Kernel and Run All**.
The kernel can start in this directory or the GSFit repository root.
For Intel MKL OpenMP symbol-loading errors, set
`MKL_THREADING_LAYER=SEQUENTIAL` before starting the kernel.

All three cases use the same initial current-density seed: Ip = 500 kA,
centroid R = 0.8 m and Z = 0 m, minor radius 0.5 m and elongation 2.
The earlier deliberate +8 cm vertical offset could lead to
`NoMagneticAxisFound` in the spline case. Removing that offset allowed all three
cases to converge in three fresh-kernel runs with the earlier seven-coefficient
spline basis. The eleven-coefficient notebook has also been rerun successfully
without solution mixing; this is not a general guarantee of convergence.

## Figures and results

Figures appear inline and are also saved in the ignored `plots/` directory:

- `reference_pairs.png` and `magnetic_sensors.png`: pair geometry and diagnostics.
- `flux_surfaces.png`: all three reconstructions, with thick LCFS contours.
- `pair_residuals.png`: flux differences for H1-H12 and V1-V12.
- `source_profiles.png`: p-prime and FF-prime versus sqrt(psi_N).
- `midplane_pressure.png`: pressure at Z = 0, stopping at each curve's LCFS.
- `metrics.json`: convergence and reconstruction errors.

Rerunning replaces these generated files. The notebook saves its inline outputs.

## CI regression checks

The Linux Python-test job runs this notebook once in a fresh kernel, alongside
Example 02, using FreeGSNKE input files pinned to
`06d1361a692b8c18b288f80c50273d14fcc896c9`. macOS and Windows do not run Example 16.
The notebook checks:

- 24 equal-flux pairs within `1e-9 Wb`, at the intended flux levels and chord positions.
- All three reconstructions converge within their configured iteration and
  GS-deviation limits, with finite metrics and source profiles.
- Two coefficients per polynomial source and eleven per spline source, with
  0/24/24 isoflux pairs and no pressure sensors.
- Both isoflux cases reduce maximum pair-flux error relative to magnetics-only.
- Spline axis error below 2 mm, maximum pair residual below 1 mWb, and
  sample-pressure RMS below 1000 Pa.

These are broad regression bounds, not exact outputs or a requirement that
splines outperform polynomials on every metric. CI uploads the executed
notebook, figures and metrics in `linux_jupyter_notebooks`, including partial
outputs on failure.

## Reference-profile choice and pressure RMS

The notebook deliberately uses `alpha_m = 2`, `alpha_n = 3`, giving source
shapes proportional to `(1 - psi_N**2)**3`. This challenges the two-coefficient
polynomial basis; it is a selected demonstration, not an unbiased benchmark.
Reference current is 600 kA and axis pressure is 8000 Pa.

Exploratory results rerun with the current Z = 0 m seed and no solution mixing
(eleven coefficients per spline source; pressure RMS rounded to the nearest Pa):

| alpha_m | alpha_n | Polynomial magnetics pressure RMS [Pa] | Polynomial + isoflux pressure RMS [Pa] | Spline + isoflux pressure RMS [Pa] |
|---:|---:|---:|---:|---:|
| 1.8 | 1.2 | 298 | 176 | 113 |
| 4 | 2 | 492 | 280 | 150 |
| **2** | **3** | **4541** | **649** | **499** |

An additional (2, 3) trial increased both spline sources from 7 to 9 and 11
coefficients: pressure RMS was 706, 593 and 499 Pa respectively, with all three
converging in 9 iterations. `reconstruct(..., spline_n_interior=5)` selects 9
coefficients per spline source; `7` selects 11, as used explicitly by the main
notebook. The helper's default remains 7 coefficients for existing callers.
The same regularization recipe was used, without
rescaling its penalties for knot spacing, so this is not a pure resolution study.

**Pressure RMS is the RMS pressure difference at all 48 spatial sample locations**
making up the 24 pairs, with equal weight per point:

```text
sqrt(mean((p_GSFit(R_i, Z_i) - p_FreeGSNKE(R_i, Z_i))**2))
```

GSFit pressure is linearly interpolated from its returned 2D grid. This metric
is in Pa, not a sum over pairs or a whole-grid/midplane-only error. Percentages
divide it by the reference axis pressure (8000 Pa), not local pressure.

Table entries were refreshed using the shared helper; these are not
exact CI targets or guarantees of convergence. At (2, 3), the eleven-coefficient
spline has lower axis, pair-flux and pressure errors than polynomial + isoflux.
The polynomial baseline also develops unphysical negative pressure
despite numerical convergence. Inspect the profiles, not convergence status alone.
