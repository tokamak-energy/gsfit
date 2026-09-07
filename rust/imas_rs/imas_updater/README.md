# `imas_updater`

Generates the Rust IDS structs in `../src/ids/` from the IMAS Data Dictionary XSD schemas.

This is **not** run at build time. There is deliberately no `build.rs`: the generated
`.rs` files are committed, so `cargo build` needs neither Python nor a copy of the Data
Dictionary. Regeneration happens only when you run the script below.

## Pinned Data Dictionary version

The committed `../src/ids/*.rs` files were generated from
[IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary), at the
version recorded in [`../src/imas_dd_version.txt`](../src/imas_dd_version.txt).

That file is the single source of truth for the version, and is written by `build_ids.py`
in the same run that writes the generated files, so it cannot drift from them. Do not copy
the version anywhere else: Rust reads it as `imas_rs::IMAS_DD_VERSION`, and the badge in the
repository's top-level `README.md` reads it from GitHub. That badge trims the commit hash
off, so it shows `4.1.1-60` where the file holds `4.1.1-60-gf5d44e8`.

The trimmed number is what the neighbouring `IMAS DD develop: commits ahead` badge counts:
how far the Data Dictionary's `develop` branch has moved past its latest release. While the
two agree, `../src/ids/*.rs` was generated from the tip of `develop`; once the badge reads
higher, `develop` has commits that the generated files do not have.

The version is `git describe --tags --dirty` output, e.g. `4.1.1-60-gf5d44e8` for the 60th
commit after tag `4.1.1`, at commit `f5d44e8`. A `-dirty` suffix means the clone had
uncommitted changes, so the generated files cannot be reproduced from the Data Dictionary
repository alone.

## Updating the IDS structs

1. Clone the Data Dictionary next to this file (it is git-ignored):

   ```bash
   git clone git@github.com:iterorganization/IMAS-Data-Dictionary.git rust/imas_rs/imas_updater/IMAS-Data-Dictionary
   ```

2. Check out the version you want to generate from. Anything `git describe` can name works,
   e.g. a tag, a commit, or the tip of a branch:

   ```bash
   git -C rust/imas_rs/imas_updater/IMAS-Data-Dictionary checkout 4.1.1
   ```

3. Run the generator. It writes one file per IDS into `../src/ids/` and then runs
   `rustfmt` over each, using the workspace `rustfmt.toml`:

   ```bash
   python rust/imas_rs/imas_updater/build_ids.py
   ```

4. Review the diff, then `cargo check -p imas_rs` and `cargo check -p gsfit_rs`.

5. Commit `../src/imas_dd_version.txt` alongside the regenerated `.rs` files. Step 3 rewrites
   it for you, so there is nothing to update by hand.

## Adding another IDS

Add its name to `ids_names` at the bottom of `build_ids.py`, add a `pub mod` line to
`../src/ids/mod.rs`, then regenerate. The schema directory must exist as
`IMAS-Data-Dictionary/schemas/<ids_name>/dd_<ids_name>.xsd`.

To reach it from Python as well, add the name to `ids_names_with_python_paths` too, and in
`../src/python/mod.rs` add: `mod <ids_name>_paths;`, a `pub use <ids_name>_paths::<IDS_NAME>_ROOT;`,
a `Py<IdsName>` `#[pyclass]` wrapper modelled on `PyEquilibrium`, and two lines in `register`
(`add_class` and `add("<ids_name>_paths", ...)`).

The IDSs currently generated are `equilibrium`, `pf_active`, `pf_passive`, `tf` and `wall`.

## Hand-written code

Only `../src/ids/*.rs` is generated. Everything else in the crate is hand-written and is
safe to edit:

* `../src/dd_base_types.rs` — the `FLT_0D`/`INT_0D`/… aliases and the `Accumulator` types.
* `../src/lib.rs`, `../src/ids/mod.rs` — module wiring.
* The `Equilibrium::with_size` / `Equilibrium::with_time` constructors, which the
  generator emits into `equilibrium.rs`.

Note that anything added by hand *inside* a generated file is lost on the next
regeneration. Physics belongs in `gsfit_rs` instead — see
`gsfit_rs/src/grad_shafranov/equilibrium_solve.rs`, which attaches methods to these
structs via an extension trait.
