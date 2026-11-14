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
`../src/ids/mod.rs`, add a `linguist-generated=true` line for the new
`../src/ids/<ids_name>.rs` to the repository's top-level `.gitattributes`, then regenerate.
The schema directory must exist as
`IMAS-Data-Dictionary/schemas/<ids_name>/dd_<ids_name>.xsd`.

To reach it from Python as well, add the name to `ids_names_with_python_paths` too, which also
generates its pickling, `../src/python/<ids_name>_pickle.rs`, then wire the generated files in by hand,
in two files:

* `../src/python/ids.rs`: a `Py<IdsName>` `#[pyclass]` wrapper modelled on `PyEquilibrium`, with its
  `__reduce__` and `_from_pickle_state`.
* `../src/python/mod.rs`: `mod <ids_name>_paths;`, `mod <ids_name>_pickle;`,
  `pub use <ids_name>_paths::<IDS_NAME>_ROOT;`, the wrapper's name in the `pub use ids::{...}` line,
  and two lines in `register` (`add_class` and `add("<ids_name>_paths", ...)`).

## Pickling

Each generated struct implements `PickleState` (in `../src/python/pickle.rs`), which converts it to
and from the `dict` of its nodes which Python's `pickle` stores, keyed by their names in the data
dictionary, leaving out the nodes which are unset. `pickle.rs` implements it for the leaves and for
`Vec`; the implementations for the structs are generated, into `../src/python/<ids_name>_pickle.rs`.
The classes holding a structure, the IDS wrappers in `ids.rs`, `GreensTables`, and `gsfit_rs`'s classes,
pickle it through `__reduce__`, and read it back with their static method `_from_pickle_state`.

The IDSs currently generated are `core_profiles`, `core_sources`, `core_transport`, `distribution_sources`, `distributions`,
`equilibrium`, `magnetics`, `nbi`, `pf_active`, `pf_passive`, `pulse_schedule`, `tf`, `thomson_scattering`,
`transport_solver_numerics` and `wall`.
Of these, `core_profiles`, `core_sources`, `equilibrium`, `magnetics`, `pf_active`, `pf_passive`, `pulse_schedule`, `tf` and
`wall` also have Python bindings and pickling; the others are Rust only so far.

## Custom (non-IMAS) keys

`custom_keys/custom_<ids_name>_keys.xsd` holds the hand-written keys GSFit needs which the
Data Dictionary does not define. `build_ids.py` merges them into the matching generated
structs, so they come out flat alongside the IMAS ones (`profiles_2d.d_psi_d_r`, not
`profiles_2d.custom.d_psi_d_r`). They are an *input* to the generator, which is why they live
here rather than in `../src/ids/`. An IDS without such a file simply has no custom keys.

Each is an XSD schema laid out as the Data Dictionary's own `dd_<ids_name>.xsd` is, holding
only GSFit's additions, so every input to the generator is an XSD schema:

* an `xs:complexType` named as one of `dd_<ids_name>.xsd`, e.g. `equilibrium_profiles_2d`,
  adds its elements to it;
* an `xs:complexType` with any other name is a new structure, e.g. `equilibrium_greens`;
* the `xs:element` named as the IDS adds its elements to the IDS's root;
* inside any of these, an element named as one the Data Dictionary already has there, holding
  an `xs:complexType`, adds that complexType's elements to it. This is how a structure declared
  in `dd_support.xsd` gets keys, e.g. `code`.

The keys use the Data Dictionary's types, written as it writes its own: a leaf holds
`<xs:group ref="FLT_1D"/>`, or another of its base types, and a structure names a Data
Dictionary complexType in `type`, e.g. `identifier_static`, or a new structure. An array of
structures has `minOccurs="0"`, as it can be empty: the Data Dictionary never writes
`minOccurs`, so its own arrays read as one or more, though in IMAS any can be empty. Units are as
the Data Dictionary writes them, e.g. `Wb.m^-1`, and `1` for dimensionless. Each schema
includes `dd_support.xsd`, so it is a valid schema, every type it uses resolves to the Data
Dictionary's, and an editor can show their definitions. `build_ids.py` rejects anything else,
e.g. `xs:double` or a simple type of the schema's own. `custom_equilibrium_keys.xsd`
describes the layout in full.

GSFit's settings are not kept in an IDS, as an IDS can also be made by another code: `build_ids.py`
rejects a custom key naming a settings XSD schema. Their structs are generated from their own XSD
schemas by `python/gsfit/settings_schema/_generate_rust_schema.py`, into `gsfit_rs::settings`, and
kept with the Rust class they configure.

## Structures of GSFit's own, not in any IDS

Data GSFit keeps which the Data Dictionary has no node for, and which is not part of an IDS, is
still generated as an IDS is, so that it has the same accessors: a range, e.g. `.flux_loop(..)`,
or a name, e.g. `.flux_loop(name)`. `greens_storage.xsd` describes the Greens tables, which
`build_ids.py` generates, through `build_structure`, into one module, `../../gsfit_rs/src/greens_storage.rs`,
and which `gsfit_rs::Equilibrium` holds, in `greens_tables`. The module holds their structs, and their Python
bindings, which are an IDS's too: the paths, which `gsfit_rs` mounts as
`gsfit_rs.imas.greens_tables_paths`, the class holding a copy, `GreensTables`, which the getter
`Equilibrium.greens_tables` returns, and the pickling of the structs; their section of the type stub is in `imas.pyi`.
An IDS's structs, Python bindings and pickling are three files, as this crate has its Python bindings behind
its optional `python` feature, while `gsfit_rs` always has them, so a structure's are one. The schema is written
as the custom keys are, but declares the whole structure, its root being its one `xs:element`; the generated
module reaches `dd_base_types.rs` and the Python bindings through `imas_rs`. Adding another means adding a
`build_structure` call at the bottom of `build_ids.py`, the generated file to `.gitattributes`, and
mounting its class and paths in `gsfit_rs`'s `lib.rs`.

## Hand-written code

Only `../src/ids/*.rs`, `../src/python/*_paths.rs`, `../src/python/*_pickle.rs`,
`../src/dd_support/parameters_input.rs` and, in `gsfit_rs`, `src/greens_storage.rs` are generated. Everything else in the
crate is hand-written and is safe to edit:

* `../src/dd_base_types.rs` — the `FLT_0D`/`INT_0D`/… aliases, and the gathering machinery the
  generated views are built on: `Elements`, `Accumulator`, and `arrange_left_to_right`, which is
  the one place the order of gathered dimensions is decided (shared with the Python bindings).
* `../src/lib.rs`, `../src/ids/mod.rs` — module wiring.
* The `Equilibrium::with_size` / `Equilibrium::with_time` constructors, which the
  generator emits into `equilibrium.rs`.

Note that anything added by hand *inside* a generated file is lost on the next
regeneration. Physics belongs in `gsfit_rs` instead — see
`gsfit_rs/src/grad_shafranov/equilibrium_solve.rs`, which attaches methods to these
structs via an extension trait.
