// The settings each machine description is built with, read from their XML into structs.
// The XML schema is the source of truth: the structs are generated from it by `python/gsfit/settings_schema/_generate_rust_schema.py`.
// They are kept with the Rust class they configure, not in its IDS, as an IDS can also be made by another code.

// Load submodules
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_core_profiles;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_core_sources;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_equilibrium;
pub(crate) mod settings_leaf_readers;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_magnetics;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_pf_active;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_pf_passive;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_pulse_schedule;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_tf;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_wall;
#[allow(dead_code, reason = "generated from the XML schema, so holds settings the Rust code may not read")]
pub mod settings_workflow;
