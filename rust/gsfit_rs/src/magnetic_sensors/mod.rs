// Load modules
// Each file adds one kind of sensor to the `Magnetics` struct, with its own `#[pymethods]` block and Rust only `impl Magnetics` block,
// and copies that kind of sensor into the equilibrium IDS, with a Rust only `impl Plasma` block; so there is nothing to expose
mod b_field_pol_probe;
mod diamagnetic_flux;
mod flux_loop;
mod ip;
mod rogowski_coil;
