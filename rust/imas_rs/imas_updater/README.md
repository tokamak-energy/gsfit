# `imas_updater`

Generates the Rust IDS structs in `../src/ids/` from the IMAS Data Dictionary XSD schemas.

This is **not** run at build time. There is deliberately no `build.rs`: the generated
`.rs` files are committed, so `cargo build` needs neither Python nor a copy of the Data
Dictionary. Regeneration happens only when you run the script below.

## Pinned Data Dictionary version

The committed `../src/ids/*.rs` files were generated from:

| | |
| --- | --- |
| Repository | `git@github.com:iterorganization/IMAS-Data-Dictionary.git` |
| Version | `4.1.1-58-g4542d30` |
| Commit | `4542d30` |

## Updating the IDS structs

1. Clone the Data Dictionary next to this file (it is git-ignored):

   ```bash
   git clone git@github.com:iterorganization/IMAS-Data-Dictionary.git rust/imas_rs/imas_updater/IMAS-Data-Dictionary
   ```

2. Check out the version you want to generate from:

   ```bash
   git -C rust/imas_rs/imas_updater/IMAS-Data-Dictionary checkout 4542d30
   ```

3. Run the generator. It writes one file per IDS into `../src/ids/` and then runs
   `rustfmt` over each, using the workspace `rustfmt.toml`:

   ```bash
   python rust/imas_rs/imas_updater/build_ids.py
   ```

4. Review the diff, then `cargo check -p imas_rs` and `cargo check -p gsfit_rs`.

5. If you moved to a newer Data Dictionary, update the pinned version in the table above
   in the same commit as the regenerated `.rs` files.

## Adding another IDS

Add its name to `ids_names` at the bottom of `build_ids.py`, add a `pub mod` line to
`../src/ids/mod.rs`, then regenerate. The schema directory must exist as
`IMAS-Data-Dictionary/schemas/<ids_name>/dd_<ids_name>.xsd`.

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
