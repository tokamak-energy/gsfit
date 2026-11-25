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
mod settings_schema;
mod symmetry;

// Flattened for the submodules
use configuration::Configuration;
use symmetry::Symmetry;

// Public flattened exports
pub use guazzotto_freidberg::GuazzottoFreidberg;
