//! The components of the toroidal plasma current: beam-driven, bootstrap, diamagnetic, ohmic and Pfirsch-Schlüter, each in its own
//! `current_*.rs`. From Python, `calculate_plasma_current_components` calculates them all.
//!
//! They fall into two groups. The diamagnetic and Pfirsch-Schlüter currents are driven by the pressure: they are `p'` times coefficients
//! which depend only on `F` and the shape of the flux surfaces, so the Grad-Shafranov equation contains them once `p'` is fixed. The
//! bootstrap, ohmic and beam-driven currents flow along the field lines, and are what neoclassical theory and current drive supply, as
//! `<J . B>`.
//!
//! Every component is calculated from IDSs only: the `equilibrium` IDS, with `profiles_1d` and `profiles_2d` filled by the post-processor,
//! and, for the bootstrap, ohmic and beam-driven currents, the `core_profiles` IDS, holding the kinetic profiles on `profiles_1d(itime)`.
//! The beam-driven current is made from the fast ions' current in the neutral beams' source of the `core_sources` IDS. The bootstrap and
//! ohmic currents are written to the `core_sources` IDS, as `<J . B> / B0`; the beam-driven current is not, as the neutral beams' source
//! keeps the fast ions' own current. The data dictionary has nowhere for the toroidal current of each component, which is what adds up to
//! the plasma current, so every component writes that to `PlasmaCurrentComponents`. The electron collisionality, which the bootstrap and
//! ohmic currents take, is written to the `core_profiles` IDS, as `profiles_1d(itime)/electrons/collisionality_norm`.
//!
//! The helpers they share are here too: the field sampled on the flux surfaces, `flux_surface_samples.rs`, the trapped fraction,
//! `trapped_fraction.rs`, the collisionalities, `collisionality.rs`, the kinetic profiles mapped onto the flux surfaces,
//! `kinetic_profiles.rs`, and the writing of a source into `core_sources`, `core_sources_source.rs`. The choices which will become settings
//! are in `settings.rs`.

mod collisionality;
mod core_sources_source;
mod current_beam_driven;
mod current_bootstrap;
mod current_diamagnetic;
mod current_ohmic;
mod current_pfirsch_schluter;
mod flux_surface_samples;
mod kinetic_profiles;
mod plasma_current_components;
mod plasma_current_components_python_interface;
mod settings;
#[cfg(test)]
mod test_equilibrium;
mod trapped_fraction;

pub use plasma_current_components_python_interface::{PyPlasmaCurrentComponent, PyPlasmaCurrentComponents, calculate_plasma_current_components};
