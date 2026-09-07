# Offline reconstruction throughput, September 2026

This note documents the measurements behind the pull request "Faster offline reconstruction at unchanged accuracy". It records what a GSFit "run" consists of, where the time went, what was changed, and the before/after evidence on public workloads.

## 1. What a run consists of

`Gsfit.run()` performs, in order:

| Phase | What happens | Scales with |
| --- | --- | --- |
| `setup_objects` | Database reader builds the Rust objects. For ST40 this includes the passive eigenmode decomposition (finite-size mutual inductance between all vessel filaments). | vessel filaments squared |
| `calculate_greens` | Green's tables between coils, passives and the plasma grid, and between every current source and every sensor. | grid points, filaments, sensors, passive degrees of freedom |
| `inverse_solver_rust` | Picard iteration per time-slice (parallel over slices with rayon), then the equilibrium post-processor and the sensor post-processing. | time-slices, iterations, grid points |
| `write_results_to_mdsplus` | Maps results to the MDSplus tree layout (and writes to MDSplus inside Tokamak Energy's network). | output size |

The default ST40 settings reconstruct `arange(10 ms, 250 ms, 0.5 ms)` = 480 time-slices on an 81 x 161 grid with 7 PF coils, 8 passive conductors (538 filaments; the IVC with 15 eigenmode degrees of freedom) and 59 magnetic sensors. The per-slice Picard iteration converges in about 14 iterations to `gs_error = 5e-6`.

A reported "about ten minutes" for a post-shot run was therefore compared with the sum of setup, Green's tables and the 480-slice solve on public data, not with a single per-slice update. MDSplus reads and writes are not part of the public workloads and were not measured.

## 2. Workloads and acceptance criteria (frozen before tuning)

All workloads are public and run without Tokamak Energy's network:

* **ST40 mock, 31 slices**: real ST40 pulse 12050 magnetics from the mocked MDSplus trees in `tests/test_02_delta_z_shift_greater_than_d_z/data` (31 samples in a 0.3 ms window around 130 ms), default settings. Representative of the standard ST40 configuration (passives with eigenmodes, all sensor types).
* **ST40 mock, 480 slices**: the same window tiled to the default workflow length, to measure a full-length run.
* **MAST-U FreeGSNKE example 02**: `examples/example_02__mastu__freegsnke_data.ipynb` (synthetic magnetics from FreeGSNKE, 65 x 129 grid, no passives), magnetics-only solve and the isoflux re-solve.
* **MAST-U FreeGSNKE example 07**: tensioned cubic B-spline source functions.
* **Rust unit tests, doc tests, `pytest tests/`, `mypy`, `ty`**.

Acceptance: identical requested diagnostics, grid, tolerances and iteration limits; per time-slice, the flux map, boundary, magnetic axis, Ip, fitted coefficients, chi_mag, profiles and the iteration count are compared with the reference build. Changes at the floating-point rounding level (about 1e-10 relative) are accepted; any change in convergence outcome or iteration count is not.

## 3. Where the time went (reference build, ST40 mock, 31 slices, 8 threads, Apple M4)

| Phase | Time | Cause |
| --- | --- | --- |
| passives setup | 20.8 s | finite-size mutual inductance of 388 IVC filaments, serial, each 100 x 100 sub-filament block built through the thread pool |
| Green's with passives | 27.5 s | plasma-grid and sensor Green's tables recomputed for every passive degree of freedom (15x for the IVC); Rogowski virtual probes rebuilt per degree of freedom |
| Green's with coils, plasma | 5.1 s | |
| solve, 31 slices | 12.8 s | per Picard iteration: dense Green's matrix product 61%, boundary flood fill 14%, stationary-point search 9%, per-slice table reorganisation 8% |

## 4. What changed

1. **Passive Green's tables once per passive** (commit "Calculate passive Green's tables once per passive"): the filament tables are calculated once and contracted with each degree of freedom; the mutual-inductance quadrature runs in parallel with sequential inner blocks. Bitwise identical.
2. **Per-run tables shared across slices** (commit "Share per-run solver tables"): reorganised Green's tables, inside-vessel mask and static sensor tables built once per run instead of per slice or per iteration; dense edge arrays in the stationary-point search. Bitwise identical.
3. **FFT convolution along z** (commit "Evaluate the plasma Green's convolution with FFTs along z"): the plasma contribution to psi and its derivatives is a convolution along the uniform z axis for every radial pair, evaluated with real FFTs instead of a 3 GFLOP dense product per iteration. Same double sum, different rounding order.
4. **Per-iteration and per-slice reuse** (commit "Reuse per-iteration and per-slice quantities"): source-function basis functions evaluated once per iteration, PF coil fields once per slice, boolean grid in the flood fill. Bitwise identical.
5. **Parallel post-processing** (commit "Post-process the time-slices in parallel"): the boundary contour, flux surfaces, profiles and scrape-off-layer legs are calculated in parallel over the slices. Bitwise identical.
6. **Stationary-point search** (commit "Skip cells without edge crossings"): cells with no zero crossing on any edge are skipped before any allocation. Bitwise identical.
7. **Robustness** (commit "Allow a sensor type with no included sensors"): excluding every bp probe or every flux loop no longer panics; regression test added.

The reference for "bitwise identical" is the ST40 mock workload (2 and 31 slices) compared with `compare_results.py` after each commit; the FFT commit is the only one which changes results, at the rounding level.

## 5. Results

### 5.1 Timing (medians of 3 alternating runs; the 480-slice runs once each)

| Metric and workload | Before (s) | After (s) | Improvement | n |
| --- | --- | --- | --- | --- |
| ST40 mock, 31 slices, total (setup + Green's + solve + map) | 71.00 | 15.78 | 4.5x | 3/3 |
|   setup_objects | 21.73 | 8.99 | 2.4x | 3/3 |
|   calculate_greens | 34.33 | 4.86 | 7.1x | 3/3 |
|   inverse_solver (31 slices incl. post-processing) | 15.24 | 1.91 | 8.0x | 3/3 |
| ST40 mock, 480 slices (default workflow length), total | 293.43 | 38.70 | 7.6x | 1/1 |
|   inverse_solver (480 slices) | 233.81 | 26.55 | 8.8x | 1/1 |
| MAST-U FreeGSNKE example 02, GSFit part (setup + Green's + 2 solves) | 4.45 | 3.07 | 1.5x | 3/3 |
|   inverse_solver (1 slice) | 0.51 | 0.19 | 2.8x | 3/3 |
| Notebook example 02 end to end (incl. FreeGSNKE forward solve, plots) | 12.11 | 9.80 | 1.2x | 3/3 |
| Notebook example 07 end to end (B-spline source functions) | 11.10 | 8.15 | 1.4x | 3/3 |

The "total" rows exclude the Python import and the result dump of the benchmark script. Reference: upstream `main` at `2f6fd0c`; candidate: this branch at `39bd7b3` (the later commits change tests, warnings and documentation only). For the ST40 workloads the solver phase is now a minority of the run; the remaining setup cost is dominated by the finite-size mutual-inductance quadrature of the vessel filaments (parallel, but still 100 x 100 elliptic-integral blocks per filament pair) and the Green's tables of the PF coils.

### 5.2 Accuracy

Accuracy (candidate vs reference, first repeat):
| Workload | psi max rel diff | boundary max abs diff (m) | Ip max rel diff | chi_mag max rel diff | n_iter identical | converged before/after |
| --- | --- | --- | --- | --- | --- | --- |
| ST40 mock 31 slices | 1.8e-11 | 1.9e-10 | 2.0e-12 | 1.1e-10 | True | 31/31 vs 31/31 |
| ST40 mock 480 slices | 1.4e-06 | 3.1e-06 | 1.3e-07 | 3.1e-06 | False | 480/480 vs 480/480 |
| MAST-U example 02 (magnetics) | 9.7e-15 | 2.9e-14 | 3.9e-16 | 6.2e-13 | True | 1/1 vs 1/1 |
| MAST-U example 02 (with isoflux) | 5.0e-15 | 2.7e-14 | 3.9e-16 | 1.1e-13 | True | 1/1 vs 1/1 |
| MAST-U example 07 (B-splines) | 6.6e-04 | 5.9e-02 | 1.5e-05 | 3.0e-03 | False | 1/1 vs 1/1 |

The "boundary max abs diff" column compares the boundary outlines point by point; for example 07 the two outlines have different point counts and start points because the bounding x-point differs (see below), so that entry measures ordering, not distance. The extents of the two outlines agree to 1e-4 m except for the upper boundary (1.099 m vs 1.134 m), which is the x-point alternation described below.

* ST40 mock, 31 slices: rounding-level agreement on every slice, identical iteration counts.
* ST40 mock, 480 slices: 479 of 480 slices agree to 3e-11 relative in flux. Slice 64 converged on the reference build at iteration 13 with `gs_error = 4.99999999e-6`, i.e. within rounding of the 5e-6 tolerance, and on this branch at iteration 14; that slice differs by 1.3e-6 relative, the tolerance itself.
* MAST-U example 02: 1e-14 (bitwise apart from the last digits), with and without the isoflux constraints.
* MAST-U example 07 (tensioned cubic B-splines, 11 p' degrees of freedom, near double null): the two builds converge along different Picard paths (16 vs 26 iterations) to solutions 3.5e-4 apart in flux, with Ip within 1.5e-5 relative and chi_mag within 0.3%. This case is path-sensitive on the reference build itself: scaling every fit weight by (1 + 1e-14) changes the reference's iteration count to 19, by (1 + 1e-12) to 21, by (1 + 1e-10) to 12, with flux differences of 4e-5 to 1.5e-4 and the same alternation between the upper and lower x-point as the bounding point. The difference on this branch is of that nature (the FFT commit changes floating-point rounding), not a change of the algorithm; every other commit is bitwise identical to its predecessor on this case.


## 6. Robustness matrix (ST40 mock, default settings unless stated)

| Case | Before (upstream main) | After (this branch) |
| --- | --- | --- |
| nominal, 3 slices | 3/3 converged, [14, 14, 14] iterations | 3/3 converged, [14, 14, 14] iterations |
| initial Ip = 0 | 0/1 converged (`InvalidInitialCurrent`) | 0/1 converged (`InvalidInitialCurrent`) |
| initial Ip = -425 kA | 0/1 converged (`NoStationaryPointsFound, Warning:`) | 0/1 converged (`NoStationaryPointsFound, Warning:`) |
| initial current ellipse outside the vessel (r_cur = 0.95 m) | 0/1 converged (`InvalidInitialCurrent`) | 0/1 converged (`InvalidInitialCurrent`) |
| initial ellipse far off axis (r_cur = 0.30 m, z_cur = 0.45 m, a = 0.08 m) | 0/1 converged (`InvalidInitialCurrent`) | 0/1 converged (`InvalidInitialCurrent`) |
| n_iter_max = 3 | 0/1 converged (`MaxIterReached`) | 0/1 converged (`MaxIterReached`) |
| gs_error = 1e-9 | 1/1 converged, [24] iterations | 1/1 converged, [24] iterations |
| no bp probes included | crash: `PanicException` | 1/1 converged, [13] iterations |
| no flux loops included | crash: `PanicException` | 1/1 converged, [14] iterations |
| only 4 bp probes and 2 flux loops | 1/1 converged, [15] iterations | 1/1 converged, [15] iterations |
| coarse grid 61 x 121 | 1/1 converged, [13] iterations | 1/1 converged, [13] iterations |
| fine grid 81 x 321 | 1/1 converged, [14] iterations | 1/1 converged, [14] iterations |
| vertical feedback disabled | 0/1 converged (`MaxIterReached`) | 0/1 converged (`MaxIterReached`) |
| p' with 4 and ff' with 3 polynomial degrees of freedom | 0/1 converged (`NoStationaryPointsFound`) | 0/1 converged (`NoStationaryPointsFound`) |

All outcomes, error classes and iteration counts are unchanged except the two cases which crashed before and now run; in the converging cases Ip agrees to better than 2e-12 relative. The remaining failures are pre-existing behaviour of the reconstruction algorithm (a poor initial guess is rejected by the initial-current validation, and the negative-current, vertical-feedback-off and higher-order source-function cases do not converge on this ST40 slice) and are reported here so that they stay visible.

## 7. Reproduction

```shell
# candidate: this branch; reference: upstream main 2f6fd0c, each in its own clone and venv
uv pip install -e ".[dev,with_freegs_and_freegsnke]"   # or: maturin develop --release
git clone https://github.com/FusionComputingLab/freegsnke.git ~/github/freegsnke   # MAST-U machine description
B=documentation/performance/benchmarks
python $B/bench_st40_mock.py --n-time 31 --threads 8 --tag ref      # in the reference clone
python $B/bench_st40_mock.py --n-time 31 --threads 8 --tag new      # in this clone
python $B/compare_results.py benchmark_results/st40mock_ref.npz benchmark_results/st40mock_new.npz 1e-9
python $B/bench_mastu.py --n-time 1 --threads 8 --tag new
python $B/run_notebook.py examples/example_07__mastu__freegsnke_data__tensioned_cubic_b_spline_test.ipynb --tag new
python $B/robustness.py --tag new
```

Both clones must use the same numpy and scipy versions: the FreeGSNKE forward solve which produces the synthetic MAST-U measurements gives values which differ by up to 1e-4 T between numpy 1.26 / scipy 1.15 and numpy 2.5 / scipy 1.18, which then shows up as a 1e-4 relative difference in the reconstruction that has nothing to do with GSFit. The numbers below were taken with numpy 1.26.4 and scipy 1.15.3 in both environments (a first pass with mismatched environments was discarded for that reason).

Hardware for the numbers above: Apple M4 (4 performance and 6 efficiency cores), 16 GiB, macOS 26.5, Rust 1.98.1, Python 3.13, `RAYON_NUM_THREADS=8`, release profile (`lto = true`, `codegen-units = 1`). The machine was shared with other work; runs alternated between reference and candidate and medians of three are reported. These are not measurements on Tokamak Energy's hardware.
