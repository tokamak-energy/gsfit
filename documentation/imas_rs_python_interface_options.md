# Python interface for `imas_rs` — design options

**Status:** decided; prototype implemented for `time_slice/global_quantities`.
**Goal:** read data out of the `Equilibrium` IDS from Python with editor/REPL completion,
no string keys in calling code, and wildcard gathering across arrays of structures.
**Constraint:** the interfacing code lives in `imas_rs`, not `gsfit_rs`.

---

## 0. Decision

**Surface S3** (detached path objects), because a path is then a value that can be stored
in a mapping table:

```python
from gsfit_rs.imas import equilibrium_paths
equilibrium_ids.get(equilibrium_paths.time_slice[:].global_quantities.ip)
```

**Implementation I2** (one navigator class over a generated static table). Choosing S3
made this decisive rather than a toss-up, and the reason is worth writing down:

> Under S1, a leaf getter had to *read data*, so it needed the root handle and an exact
> return type - which is what made a generated `#[pyclass]` per node (I1) worth its
> compile cost. Under S3 the path holds **no data at all**, so navigation is pure
> metadata. Generating ~565 pyo3 getters that only append a path segment buys nothing
> that a static table does not already give.

Two consequences follow, both good:

- The pyo3 "`#[pyclass]` cannot have a lifetime" constraint (§1) **stops mattering**.
  There is nothing to borrow while building a path, so the awkward
  `Arc`/`Py<...>`-plus-selection dance is confined to the single `get` call.
- The `Item`/`Many` split that would have doubled the Rust class count under I1b
  (§3) is **free**, because it lives only in the `.pyi`. Static types stay exact:

  ```python
  def get(self, path: Path[_T]) -> _T: ...
  # time_slice[0].global_quantities.ip  ->  Path[float]
  # time_slice[:].global_quantities.ip  ->  Path[npt.NDArray[np.float64]]
  ```

The compile-time measurement proposed in §5 is therefore moot and was not carried out;
the prototype added **no** new compiler warnings and one navigator class.

### What was built

| file | role |
|---|---|
| [`rust/imas_rs/src/python/mod.rs`](../rust/imas_rs/src/python/mod.rs) | hand-written: `Path`, `Equilibrium`, index resolution, gathering |
| [`rust/imas_rs/imas_updater/build_ids.py`](../rust/imas_rs/imas_updater/build_ids.py) | **generates** the two files below, from one walk of the schema |
| `rust/imas_rs/src/python/equilibrium_paths.rs` | generated: the static table (10,460 lines) |
| `python/gsfit_rs/imas.pyi` | generated: the type stub (4,260 lines) |
| `imas_rs/Cargo.toml` | `python` feature, so `imas_rs` stays a plain Rust crate by default |
| [`rust/gsfit_rs/src/lib.rs`](../rust/gsfit_rs/src/lib.rs) | mounts the classes as the `gsfit_rs.imas` submodule |
| [`rust/gsfit_rs/src/plasma.rs`](../rust/gsfit_rs/src/plasma.rs) | `plasma.equilibrium_ids` getter |
| [`python/gsfit_rs/gsfit_rs.pyi`](../python/gsfit_rs/gsfit_rs.pyi) | `Plasma.equilibrium_ids` declaration |

**Scope is the whole equilibrium IDS**: 557 nodes, 463 leaves, 41 array-of-structures
chains, to a maximum nesting depth of 6
(`grids_ggd/grid/space/objects_per_dimension/object/boundary`).

Both generated files carry a "do not edit" warning naming `build_ids.py`, and both come
from the same `build_path_tree` walk, so they cannot drift from each other or from
`equilibrium.rs`.

§4.3 (unset floats read back as NaN, unset integers raise) and §4.5 (the IDS is copied
into the Python object, so it is a snapshot) are implemented as recommended. §4.4 (ragged
arrays) is resolved - see the rule below.

### One rule worth knowing

**At most one array-of-structures level may be sliced at a time.**

```python
equilibrium_ids.get(paths.time_slice[:].profiles_2d[0].psi)   # fine  -> (n_time, n_r, n_z)
equilibrium_ids.get(paths.time_slice[0].profiles_2d[:].psi)   # fine
equilibrium_ids.get(paths.time_slice[:].profiles_2d[:].psi)   # refused
```

Two sliced levels have no single sensible shape, because the inner level can have a
different length under each element of the outer one. Rather than guess, it is refused -
and refused **statically**, because an array reached from a `Many` class only accepts an
integer index, so `mypy` rejects it before it ever runs.

Within that rule, gathering raises the rank by one and pads ragged results with NaN, so
`time_slice[:].boundary.outline.r` comes back rectangular even though each slice has a
different number of boundary points. That reproduces what GSFit already does by hand in
[plasma.rs](../rust/gsfit_rs/src/plasma.rs#L1505).

---

## 1. What is already there

Facts established by reading the code, so the options below rest on something real.

**The Rust API you want already exists.** `build_ids.py` generates a view/accumulator
layer, so this compiles today:

```rust
equilibrium.time_slice(..).global_quantities.ip.unwrap()   // -> Array1<f64>, one value per slice
equilibrium.time_slice(0).global_quantities.ip             // -> Option<f64>
equilibrium.time_slice(3..7).boundary.bounding.r.unwrap()  // -> Array1<f64>
```

`time_slice(i)` is generic over [`EquilibriumTimeSliceIndex`](../rust/imas_rs/src/ids/equilibrium.rs#L6154),
which is implemented for `usize` (returns one element) and for every range type
(returns an `EquilibriumTimeSliceSliceView`). `..` is the wildcard.

So the Python work is **exposing an API that exists**, not inventing one.

**Scale of the generated file** ([equilibrium.rs](../rust/imas_rs/src/ids/equilibrium.rs)):

| | count |
|---|---|
| structs total | 136 |
| of which view/accumulator types | 89 |
| data structs (would need a Python class) | 47 |
| fields total | 565 |
| leaf fields (`Option<base_type>`) | 239 |
| array-of-structures fields (`Vec<T>`) | 43 |

**`imas_rs` has no pyo3 dependency today.** Its only dependencies are `ndarray` and
`num-complex` ([Cargo.toml](../rust/imas_rs/Cargo.toml)). `gsfit_rs` is the `cdylib`
that owns `#[pymodule] fn gsfit_rs` ([lib.rs:45](../rust/gsfit_rs/src/lib.rs#L45)).
Putting the bindings in `imas_rs` means adding `pyo3` + `numpy` there — ideally behind
a `python` feature so `imas_rs` stays usable as a plain Rust crate.

**Two constraints that shape everything below:**

1. **`#[pyclass]` cannot have a lifetime parameter.** The whole existing view layer is
   `EquilibriumTimeSliceSliceView<'a>` — borrow-based. None of it can be handed to
   Python directly. Every Python-facing node must own a `'static` handle to the root
   (`Py<PyEquilibrium>` or `Arc<Equilibrium>`) plus a selection, and re-borrow on each
   access.

2. **The accumulators only gather 0D scalars.** [`is_scalar`](../rust/imas_rs/imas_updater/build_ids.py#L77)
   gates the view machinery, so `time_slice[:].profiles_1d.psi` (`FLT_1D`) and
   `time_slice[:].profiles_2d[0].psi` (`FLT_2D`) have **no** accumulator today.
   The DataTree did stack these (`get_array2`, `get_array3`). Whichever option is
   chosen, `imas_rs` needs a new array-stacking accumulator. See §4.4.

---

## 2. Part one — the Python surface (what you actually type)

These are independent of how it is built. Pick the surface first.

### S1 — implicit materialisation *(closest to what you asked for)*

```python
eq = plasma.equilibrium

eq.time_slice[:].global_quantities.ip          # -> ndarray, shape (n_time,)
eq.time_slice[0].global_quantities.ip          # -> float
eq.time_slice[3:7].global_quantities.ip        # -> ndarray, shape (4,)
eq.time_slice[[0, 2, 5]].global_quantities.ip  # -> ndarray, shape (3,)
eq.time_slice[:].boundary.bounding.r           # -> ndarray, shape (n_time,)
eq.time_slice[:].profiles_2d[0].psi            # -> ndarray, shape (n_time, n_r, n_z)
len(eq.time_slice)                             # -> n_time
```

**Governing rule:** *the shape of the result is the shape of the index, with the leaf's
own dimensions appended.* An `int` index drops a dimension, a slice keeps it. This is
exactly numpy's rule, so it needs no explanation.

- Shortest possible. No `unwrap()`, no `()`, no `get_`.
- A leaf is a terminal — you cannot ask it for its units or its DD path afterwards.
- Every read materialises. `eq.time_slice[:].profiles_2d[0].psi` allocates the full
  3D array whether or not you wanted all of it.

*Variant S1b:* return an `np.ndarray` **subclass** carrying `.units` and `.path`.
`isinstance(x, np.ndarray)` still passes and every numpy operation still works, but
the database writer can read `x.units`. Costs a thin Python shim, as numpy subclasses
are awkward to construct from Rust.

### S2 — explicit terminal, mirroring the Rust

```python
eq.time_slice[:].global_quantities.ip.unwrap()          # -> ndarray, raises on unset
eq.time_slice[:].global_quantities.ip.to_numpy()        # -> ndarray, NaN for unset
eq.time_slice[:].global_quantities.ip.to_list()         # -> list[float | None]
eq.time_slice[:].global_quantities.ip.units             # -> "A"
eq.time_slice[:].global_quantities.ip.path              # -> "global_quantities.ip"
```

- 1:1 with `Accumulator::unwrap()` / `to_vec()`, so Rust and Python read the same.
- Lets you *choose* the unset policy per call site instead of baking one in (§4.3).
- Lazy: the accumulator is cheap, materialisation is explicit.
- Noisier. `.unwrap()` on ~200 lines of `map_results_to_database.py` is real clutter.
- Two ways to spell the common case invites inconsistency.

### S3 — detached path objects

```python
from gsfit_rs.imas import equilibrium as path

eq.get(path.time_slice[:].global_quantities.ip)
```

You called this ugly, and for a single read it is. It has one real advantage worth
weighing: **the path becomes a value you can put in a table.**
`map_results_to_database.py` is a 200-line mapping table, and this makes it data:

```python
MAPPING = [
    ("GLOBAL:IP",        path.time_slice[:].global_quantities.ip),
    ("BOUNDARY.GEO_AXIS:R", path.time_slice[:].boundary.geometric_axis.r),
]
for node, p in MAPPING:
    results[node] = eq.get(p)
```

- Paths are reusable, storable, printable, testable independent of any data.
- Verbose at every call site; two objects to import; the indirection has to be learned.
- Could be added **later** on top of S1 or S2 without changing them.

### S4 — flat generated methods

```python
eq.time_slice_global_quantities_ip()      # -> ndarray
eq.time_slice_boundary_bounding_r()
```

- Trivial to generate, trivial to complete, no intermediate objects, smallest runtime.
- No wildcard flexibility — the index position is baked into the name, so
  `time_slice[3:7]` and `profiles_2d[0]` need separate arguments or separate methods.
- Does not look like an IMAS path, which defeats the point of moving to IMAS.

**Recommendation: S1**, with S3 kept in reserve as an additive layer for the database
writer if the mapping table turns out to want it.

---

## 3. Part two — how it is built behind the surface

**Important: S1's surface is identical under all three.** The implementation choice is
reversible without touching a single line of calling code. That makes this a
performance/compile-time decision, not an API decision.

### I1 — a generated `#[pyclass]` per node

Every data struct gets a Python class with a real getter per field, generated by
`build_ids.py` alongside the Rust it already emits.

```rust
#[pyclass(module = "gsfit_rs.imas", name = "Equilibrium")]
pub struct PyEquilibrium {
    pub inner: Equilibrium,
}

#[pymethods]
impl PyEquilibrium {
    #[getter]
    fn time_slice(slf: Py<Self>) -> PyEquilibriumTimeSliceArray {
        PyEquilibriumTimeSliceArray { root: slf }
    }
}

/// Total plasma current, toroidal component.
#[pymethods]
impl PyEquilibriumTimeSliceGlobalQuantitiesView {
    /// Units: A
    #[getter]
    fn ip<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let eq = self.root.borrow(py);
        let values: Array1<f64> = self.indices.iter()
            .map(|&i| eq.inner.time_slice[i].global_quantities.ip.unwrap_or(f64::NAN))
            .collect();
        values.into_pyarray(py)
    }
}
```

Note the getter doc comment becomes the Python property docstring, so the **DD
documentation and units already parsed by
[`extract_documentation`](../rust/imas_rs/imas_updater/build_ids.py#L190) and
[`extract_units`](../rust/imas_rs/imas_updater/build_ids.py#L200) flow straight into
`help()`** — that is free and it is nice.

- Genuine attributes: `dir()`, `help()`, IPython completion all work with no extra work.
- A typo is an immediate `AttributeError` at the Rust boundary.
- Exact static types per leaf.
- **~47 classes and ~565 getters** through the pyo3 proc macro. Compile time and
  binary size will both grow noticeably. *I have not measured this* — see §5.

**Sub-decision (matters for typing):**

- **I1a — one class per node**, carrying a `Selection { One(usize), Many(Vec<usize>) }`.
  ~47 classes. But `ip` must return `float` *or* `ndarray` depending on the selection,
  so the stub says `float | npt.NDArray[np.float64]` and every call site needs a cast.
- **I1b — two classes per node** (element and view). ~94 classes, ~1130 getters, but
  `__getitem__` overloads give exact types:

  ```python
  class TimeSliceArray:
      @overload
      def __getitem__(self, i: int) -> TimeSlice: ...
      @overload
      def __getitem__(self, i: slice | list[int]) -> TimeSliceView: ...
  # TimeSlice.global_quantities.ip     -> float
  # TimeSliceView.global_quantities.ip -> npt.NDArray[np.float64]
  ```

  Given your standard of always writing the type out, **I1b is the one that fits** —
  at roughly double the compile cost.

### I2 — one navigator class + a generated metadata table

A single hand-written `#[pyclass]` walks a generated `static` description of the tree.

```rust
pub enum NodeKind { Struct(&'static [Child]), Aos(&'static [Child]), Leaf(LeafKind) }
pub struct Child { pub name: &'static str, pub node: &'static Node }

#[pyclass(module = "gsfit_rs.imas", name = "Node")]
pub struct PyNode {
    root: Py<PyEquilibrium>,
    path: Vec<Segment>,
    node: &'static Node,
}

#[pymethods]
impl PyNode {
    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> { /* consult node */ }
    fn __getitem__(&self, py: Python, index: &Bound<'_, PyAny>) -> PyResult<PyNode> { ... }
    fn __dir__(&self) -> Vec<&'static str> { /* children, for tab completion */ }
}
```

Generated: the `static` tree (cheap), plus one `read_leaf` match arm per leaf (239) to
actually reach the field. Plain match arms, not macro expansions.

- Small binary, fast compile, one place to fix navigation bugs.
- `__dir__` gives REPL/IPython tab completion for free.
- Error messages can be **better** than a real `AttributeError`, because the table
  knows the valid siblings: *"`ipp` is not a child of `global_quantities`; did you mean
  `ip`? Available: beta_pol, beta_tor, ..."*
- Strings exist internally. Never in your calling code, but they are in the crate.
- Without a stub, static type checkers see nothing. **With** the generated stub (§4.1)
  they see and enforce everything — so this is not the weakness it first appears.

### I3 — flat generated leaf functions

Only viable under surface S4. Listed for completeness; rejected with S4.

---

## 4. Part three — decisions that apply whichever option wins

### 4.1 The `.pyi` stub is required either way

A compiled extension module is opaque to mypy and pyright. **Neither I1 nor I2 gives
you editor completion without a generated `.pyi`.** I1 gives you *runtime* completion
free; the editor still needs the stub.

Since the stub must be generated anyway, it should come from `build_ids.py` from the
same schema, next to the Rust. This levels I1 and I2 considerably and is the single
biggest reason I2 stays competitive.

The repo already relies on stubs and checks them — [`gsfit_rs.pyi`](../python/gsfit_rs/gsfit_rs.pyi),
with `mypy`, `ty` and `pytest-mypy` in the dev dependencies — so this fits the existing
workflow rather than adding to it.

### 4.2 Where the module lives

`imas_rs` cannot be a `cdylib` of its own without becoming a second Python package.
Cleanest is for `imas_rs` to expose a registration function that `gsfit_rs` mounts as a
**submodule**:

```rust
// imas_rs/src/python/mod.rs
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> { ... }
```
```python
from gsfit_rs.imas import Equilibrium
```

`imas_rs` then owns its own generated stub, and `gsfit_rs` owns one line.

### 4.3 What an unset (`None`) leaf becomes in Python

This is the direct continuation of the failed-time-slice discussion. `Accumulator::unwrap()`
panics on `None`, which is why failed slices are NaN-filled rather than left unset.

| leaf type | option | note |
|---|---|---|
| `FLT_0D` | NaN | consistent with `set_to_failed_time_slice`; matches the current DataTree behaviour |
| `FLT_0D` | raise | safe but makes whole-array reads fragile |
| `INT_0D` | ? | **no NaN exists.** `convergence.iterations_n` and `boundary.type` are unset on failure |
| `STR_0D` | `None` in a `list[str \| None]` | strings gather to a list, so `None` is natural |

The integer case is the open one. Choices: raise; promote to `float64` with NaN; return
a masked array; or return `object` dtype with `None`. **Recommend raising for integers**,
since silently promoting an iteration count to a float is worse than a loud failure —
but this is your call, and it argues for surface S2's per-call-site `.unwrap()` vs
`.to_numpy()` if you would rather not decide globally.

### 4.4 Ragged arrays — resolved by NaN padding

`boundary.outline.r` has a different point count per time slice. Today GSFit **pre-pads
to `max_n_boundary` with NaN before storing** ([plasma.rs:1505](../rust/gsfit_rs/src/plasma.rs#L1505)),
so the DataTree only ever saw a rectangular `Array2`. In IMAS the natural storage is a
true-length `Array1` per slice, so the gather has to decide:

- NaN-pad to the longest — rectangular, matches today's output exactly, wastes memory;
- return `list[ndarray]` — honest, but breaks the "shape of the index" rule;
- return a masked array — correct, but pulls `numpy.ma` into the contract.

**NaN-padding was chosen and implemented** (`stack_float_arrays` in
[mod.rs](../rust/imas_rs/src/python/mod.rs)), because it reproduces the current MDSplus
output and the `outline.n` count is stored alongside to trim with. Integer arrays cannot
be padded this way, so gathering one requires every element to be set and the shapes to
agree; otherwise it reports why, rather than inventing a fill value.

### 4.5 How Python gets hold of the `Equilibrium`

`imas_rs` cannot reference `Plasma` (that is the circular dependency again), so the
Python root must own its data:

```rust
#[pyclass] pub struct PyEquilibrium { pub inner: Equilibrium }
```

`plasma.equilibrium` then **clones the IDS once** into the pyclass, and every child node
holds a cheap `Py<PyEquilibrium>` handle. For 100 slices of 65×65 profiles_2d that is
roughly 27 MB and a few ms, once per script — acceptable. The alternative, storing
`Py<PyEquilibrium>` inside `Plasma` to avoid the copy, drags GIL tokens into the solver
and through `par_iter_mut`. **Recommend the clone**, and cache the `Py` so repeated
`plasma.equilibrium` access is free.

Snapshot semantics follow: the Python object does not see later Rust-side mutation.
That is a feature for a results object, but it should be documented.

### 4.6 Read-only, or writable too?

Everything above is read. Writing (`eq.time_slice[0].global_quantities.ip = 1.0e6`)
roughly doubles the generated surface and raises the snapshot question sharply.
**Recommend read-only for now** — the writers live in Rust, and Python only consumes.

---

## 5. Recommendation

**Surface S1** — `eq.time_slice[:].global_quantities.ip` returning ndarray directly.
It is what you asked for and the numpy shape rule makes it self-explanatory.

**Implementation: start with I1b, but measure before committing.** ~1130 pyo3 getters
is the one number in this document I cannot predict, and it is the deciding factor. The
honest path:

1. Add the `python` feature and `PyEquilibrium` to `imas_rs`.
2. Generate the classes for **`global_quantities` only** — one AoS level, ~22 leaves.
3. Time a clean `cargo build --release` before and after.
4. If the cost is acceptable, generate the rest. If not, switch to **I2** — the Python
   surface, the stub, and every line of calling code stay identical.

**Then, in order:**

5. Generate the `.pyi` from `build_ids.py` (§4.1) — needed for the completion you asked for.
6. Add the array-stacking accumulator (§4.4) so 1D and 2D leaves work at all.
7. Port `map_results_to_database.py` one section at a time, checking against the
   DataTree output as you go, the same way the solver migration was verified.

**Deferred:** S3 path objects (additive, only if the mapping table wants them),
write access (§4.6).

---

## 6. Summary

| | S1 implicit | S2 explicit | S3 paths | S4 flat |
|---|---|---|---|---|
| brevity | best | fair | poor | good |
| mirrors Rust | partly | exactly | no | no |
| units/path available | S1b only | yes | yes | no |
| paths storable as data | no | no | **yes** | no |
| wildcards | yes | yes | yes | **no** |
| looks like IMAS | yes | yes | yes | **no** |

| | I1a one class | I1b two classes | I2 navigator |
|---|---|---|---|
| generated pyclasses | ~47 | ~94 | 1 |
| generated getters | ~565 | ~1130 | 0 (239 match arms) |
| compile time / binary | high | **highest** | low |
| exact static types | no (union) | **yes** | yes (via stub) |
| REPL completion | free | free | via `__dir__` |
| editor completion | needs stub | needs stub | needs stub |
| typo caught | at Rust boundary | at Rust boundary | in navigator, better message |
| needs `.pyi` | yes | yes | yes |

All three are interchangeable behind an identical Python surface.
