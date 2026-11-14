// Load modules
// Each file adds one kind of sensor to the `Magnetics` struct, with its own `#[pymethods]` block and Rust only `impl Magnetics` block,
// and copies that kind of sensor into the equilibrium IDS, with a Rust only `impl Equilibrium` block; so there is nothing to expose.
// `isoflux`, `isoflux_boundary` and `stationary_point` are the constraints at points which can move, custom keys of the magnetics IDS,
// which share `moving_point`
mod b_field_pol_probe;
mod diamagnetic_flux;
mod flux_loop;
// mod hall_probes; // Hall probes are not yet implemented
mod ip;
mod isoflux;
mod isoflux_boundary;
mod moving_point;
mod rogowski_coil;
mod stationary_point;
