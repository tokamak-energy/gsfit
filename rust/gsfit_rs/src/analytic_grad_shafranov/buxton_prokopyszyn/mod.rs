// Buxton-Prokopyszyn analytic Grad-Shafranov equilibrium: Guazzotto-Freidberg generalised to EFIT-polynomial p' and ff' of any order,
// with optional constant terms.

// Load submodules
mod buxton_prokopyszyn;
mod configuration;
mod diagnostics;
mod double_double;
mod eigenvalue;
mod flux_geometry;
mod gsfit_units;
mod lattice;
mod numerics;
mod perturbation;
mod profiles;
mod radial_series;
mod reference_operator;
mod settings_schema;
mod shape_basis;
mod source_function;

// Public flattened exports
pub use buxton_prokopyszyn::BuxtonProkopyszyn;
