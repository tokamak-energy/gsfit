// Guazzotto–Freidberg analytic Grad–Shafranov equilibrium.

// Load submodules
mod coils_fit;
mod configuration;
mod eigenvalue;
mod flux;
mod guazzotto_freidberg;
mod matching_geometry;
mod model_surface;
mod radial;
mod symmetry;

// Public flattened exports
pub use configuration::Configuration;
pub use guazzotto_freidberg::GuazzottoFreidberg;
pub use symmetry::Symmetry;
