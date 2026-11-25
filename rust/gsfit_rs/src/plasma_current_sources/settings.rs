//! The settings of the plasma current sources, all in one place.
//!
//! They are fixed here for now, with values which suit ST40, and will become XML settings, read into a generated settings struct
//! like the other classes' (see `python/gsfit/settings_schema/xsd/`).

/// The settings of the plasma current sources.
pub(super) struct SettingsPlasmaCurrentSources {
    /// How far apart [second] the time of a `core_profiles/profiles_1d(itime)` can be from an equilibrium time-slice's, and still be the
    /// kinetic profiles for it. The kinetic profiles are not interpolated in time: they are expected at the equilibrium's times
    pub time_tolerance: f64,
    /// Which of `core_profiles/profiles_1d(itime)/ion(:)` is the main ion. Its charge, `z_ion`, and density set the ion collisionality,
    /// and its temperature is taken as every ion's, as Redl's formulae assume one ion temperature
    pub ion_main_index: usize,
}

/// The settings, until they are read from XML.
///
/// * `time_tolerance` - 1 microsecond: far below the spacing of any reconstruction, and far above the rounding of a time written by
///   another code
/// * `ion_main_index` - 0: the first ion, e.g. deuterium, with the impurities after it
pub(super) const SETTINGS: SettingsPlasmaCurrentSources = SettingsPlasmaCurrentSources {
    time_tolerance: 1.0e-6,
    ion_main_index: 0,
};
