//! IMAS Wall IDS
//!
//! This module defines the wall Interface Data Structure (IDS)
//! Auto-generated from IMAS Data Dictionary XSD schema.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::dd_base_types::{Accumulator, FLT_0D, FLT_1D, FLT_2D, FLT_3D, INT_0D, INT_1D, STR_0D, STR_1D, StringAccumulator};

// ============================================================================
// Complex Types
// ============================================================================

/// Simple 0D description of plasma-wall interaction, related to electrons
#[derive(Debug, Clone, Default)]
pub struct WallGlobalQuantititesElectrons {
    /// Pumped particle flux (in equivalent electrons)
    /// Units: s^-1
    pub pumping_speed: Option<FLT_1D>,
    /// Particle flux from the plasma (in equivalent electrons)
    /// Units: s^-1
    pub particle_flux_from_plasma: Option<FLT_1D>,
    /// Particle flux from the wall corresponding to the conversion into various neutral types (first dimension: 1: cold; 2: thermal; 3: fast), in equivalent electrons
    /// Units: s^-1
    pub particle_flux_from_wall: Option<FLT_2D>,
    /// Gas puff rate (in equivalent electrons)
    /// Units: s^-1
    pub gas_puff: Option<FLT_1D>,
}

/// This structure allows distinguishing the species causing the sputtering
#[derive(Debug, Clone, Default)]
pub struct WallGlobalQuantititesNeutralOrigin {
    /// List of elements forming the atom or molecule of the incident species
    pub element: Vec<PlasmaCompositionNeutralElementConstant>,
    /// String identifying the incident species (e.g. H, D, CD4, ...)
    pub name: Option<STR_0D>,
    /// Array of incident angles of this incident species, on which the physical sputtering coefficient is tabulated
    /// Units: rad
    pub angles: Option<FLT_1D>,
    /// Array of energies of this incident species, on which the physical sputtering coefficient is tabulated
    /// Units: eV
    pub energies: Option<FLT_1D>,
    /// Effective coefficient of physical sputtering due to this incident species. It is assumed that all sputtered neutrals from the wall have the wall temperature (cold neutrals)
    pub sputtering_physical: Option<FLT_3D>,
    /// Effective coefficient of chemical sputtering due to this incident species. It is assumed that all sputtered neutrals from the wall have the wall temperature (cold neutrals)
    pub sputtering_chemical: Option<FLT_1D>,
}

/// Simple 0D description of plasma-wall interaction, related to a given neutral species
#[derive(Debug, Clone, Default)]
pub struct WallGlobalQuantititesNeutral {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElementConstant>,
    /// String identifying the species (e.g. H, D, CD4, ...)
    pub name: Option<STR_0D>,
    /// Pumped particle flux for that species
    /// Units: s^-1
    pub pumping_speed: Option<FLT_1D>,
    /// Particle flux from the plasma for that species
    /// Units: s^-1
    pub particle_flux_from_plasma: Option<FLT_1D>,
    /// Particle flux from the wall corresponding to the conversion into various neutral types (first dimension: 1: cold; 2: thermal; 3: fast)
    /// Units: s^-1
    pub particle_flux_from_wall: Option<FLT_2D>,
    /// Gas puff rate for that species
    /// Units: s^-1
    pub gas_puff: Option<FLT_1D>,
    /// Wall inventory, i.e. cumulated exchange of neutral species between plasma and wall from t = 0, positive if a species has gone to the wall, for that species
    pub wall_inventory: Option<FLT_1D>,
    /// Particle recycling coefficient corresponding to the conversion into various neutral types (first dimension: 1: cold; 2: thermal; 3: fast)
    pub recycling_particles_coefficient: Option<FLT_2D>,
    /// Energy recycling coefficient corresponding to the conversion into various neutral types (first dimension: 1: cold; 2: thermal; 3: fast)
    pub recycling_energy_coefficient: Option<FLT_2D>,
    /// Sputtering coefficients due to a set of incident species
    pub incident_species: Vec<WallGlobalQuantititesNeutralOrigin>,
}

/// Simple 0D description of plasma-wall interaction
#[derive(Debug, Clone, Default)]
pub struct WallGlobalQuantitites {
    /// Quantities related to electrons
    pub electrons: WallGlobalQuantititesElectrons,
    /// Quantities related to the various neutral species
    pub neutral: Vec<WallGlobalQuantititesNeutral>,
    /// Wall temperature
    /// Units: K
    pub temperature: Option<FLT_1D>,
    /// Total power incident on the wall. This power is split in the various physical categories listed below
    /// Units: W
    pub power_incident: Option<FLT_1D>,
    /// Power conducted by the plasma onto the wall
    /// Units: W
    pub power_conducted: Option<FLT_1D>,
    /// Power convected by the plasma onto the wall
    /// Units: W
    pub power_convected: Option<FLT_1D>,
    /// Net radiated power from plasma onto the wall (incident-reflected)
    /// Units: W
    pub power_radiated: Option<FLT_1D>,
    /// Black body radiated power emitted from the wall (emissivity is included)
    /// Units: W
    pub power_black_body: Option<FLT_1D>,
    /// Net power from neutrals on the wall  (positive means power is deposited on the wall)
    /// Units: W
    pub power_neutrals: Option<FLT_1D>,
    /// Power deposited on the wall due to recombination of plasma ions
    /// Units: W
    pub power_recombination_plasma: Option<FLT_1D>,
    /// Power deposited on the wall due to recombination of neutrals into a ground state (e.g. molecules)
    /// Units: W
    pub power_recombination_neutrals: Option<FLT_1D>,
    /// Power deposited on the wall due to electric currents (positive means power is deposited on the target)
    /// Units: W
    pub power_currents: Option<FLT_1D>,
    /// Power to cooling systems
    /// Units: W
    pub power_to_cooling: Option<FLT_1D>,
    /// Toroidal current flowing in the vacuum vessel
    /// Units: A
    pub current_phi: Option<FLT_1D>,
}

/// 2D limiter unit description
#[derive(Debug, Clone, Default)]
pub struct Wall2dLimiterUnit {
    /// Short string identifier (unique for a given device). Although the details may be machine-specific, a tree-like syntax must be followed, listing first top level components, then going down to finer element description. The tree levels are separated by a /, using a number of levels relevant to the granularity of the description. Example : ic_antenna/a1/bumpers refers to the bumpers of the a1 IC antenna
    pub name: Option<STR_0D>,
    /// Description, e.g. “channel viewing the upper divertor”
    pub description: Option<STR_0D>,
    /// Type of component of this unit
    pub component_type: IdentifierStatic,
    /// Irregular outline of the limiting surface. Repeat the first point in case of a closed contour
    pub outline: Rz1dStatic,
    /// Simplified description of toroidal angle extensions of the unit, by a list of zones defined by their centre and full width (in toroidal angle).  In each of these zones, the unit outline remains the same. Leave this node empty for an axisymmetric unit. The first dimension gives the centre and full width toroidal angle values for the unit. The second dimension represents the toroidal occurrences of the unit countour (i.e. the number of toroidal zones).
    /// Units: rad
    pub phi_extensions: Option<FLT_2D>,
    /// Resistivity of the limiter unit
    /// Units: ohm.m
    pub resistivity: Option<FLT_0D>,
    /// Thickness of this unit evaluated at the midplane (whenever relevant)
    /// Units: m
    pub midplane_thickness: Option<FLT_0D>,
    /// Nuclear power density heating this unit evaluated at the midplane (whenever relevant)
    /// Units: W.m^-3
    pub midplane_power_density_nuclear: SignalFlt1d,
    /// Nuclear power heating this unit
    /// Units: W
    pub power_nuclear: SignalFlt1d,
}

/// 2D limiter description
#[derive(Debug, Clone, Default)]
pub struct Wall2dLimiter {
    /// Type of the limiter description. index = 0 for the official single contour limiter and 1 for the official disjoint PFC structure like first wall. Additional representations needed on a code-by-code basis follow same incremental pair tagging starting on index =2
    pub r#type: IdentifierStatic,
    /// Set of limiter units. Whenever relevant, multiple units should be ordered so that they define contiguous sections, clockwise in the poloidal direction.
    pub unit: Vec<Wall2dLimiterUnit>,
}

/// 2D mobile parts description
#[derive(Debug, Clone, Default)]
pub struct Wall2dMobileUnit {
    /// Name of the mobile unit
    pub name: Option<STR_0D>,
    /// Irregular outline of the mobile unit, for a set of time slices. Repeat the first point in case of a closed contour
    pub outline: Vec<Rz1dDynamicAosTime>,
    /// Simplified description of toroidal angle extensions of the unit, by a list of zones defined by their centre and full width (in toroidal angle).  In each of these zones, the unit outline remains the same. Leave this node empty for an axisymmetric unit. The first dimension gives the centre and full width toroidal angle values for the unit. The second dimension represents the toroidal occurrences of the unit countour (i.e. the number of toroidal zones).
    /// Units: rad
    pub phi_extensions: Option<FLT_2D>,
    /// Resistivity of the mobile unit
    /// Units: ohm.m
    pub resistivity: Option<FLT_0D>,
}

/// 2D mobile parts description
#[derive(Debug, Clone, Default)]
pub struct Wall2dMobile {
    /// Type of the description
    pub r#type: IdentifierStatic,
    /// Set of mobile units
    pub unit: Vec<Wall2dMobileUnit>,
}

/// 2D wall description
#[derive(Debug, Clone, Default)]
pub struct Wall2d {
    /// Type of the description
    pub r#type: IdentifierStatic,
    /// Description of the immobile limiting surface(s) or plasma facing components for defining the Last Closed Flux Surface.
    pub limiter: Wall2dLimiter,
    /// In case of mobile plasma facing components, use the time-dependent description below this node to provide the full outline of the closest PFC surfaces to the plasma. Even in such a case, the 'limiter' structure is still used to provide the outermost limiting surface (can be used e.g. to define the boundary of the mesh of equilibrium reconstruction codes)
    pub mobile: Wall2dMobile,
    /// Mechanical structure of the vacuum vessel. The vessel is described as a set of nested layers with given physics properties; Two representations are admitted for each vessel unit : annular (two contours) or block elements.
    pub vessel: Vessel2d,
}

/// Incident and emitted energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergySimple {
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub emitted: Vec<GenericGridScalar>,
}

/// Neutral state energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergyNeutralState {
    /// String identifying state
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Neutral type, in terms of energy. ID =1: cold; 2: thermal; 3: fast; 4: NBI
    pub neutral_type: IdentifierDynamicAos3,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub emitted: Vec<GenericGridScalar>,
}

/// Neutral energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergyNeutral {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// String identifying neutral (e.g. H, D, T, He, C, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding ion species in the ../../ion array
    pub ion_index: Option<INT_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub emitted: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdEnergyNeutralState>,
}

/// Ion state energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergyIonState {
    /// Minimum Z of the charge state bundle
    /// Units: e
    pub z_min: Option<FLT_0D>,
    /// Maximum Z of the charge state bundle
    /// Units: e
    pub z_max: Option<FLT_0D>,
    /// String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...)
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub emitted: Vec<GenericGridScalar>,
}

/// Ion energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergyIon {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// Ion charge (of the dominant ionization state; lumped ions are allowed)
    /// Units: e
    pub z_ion: Option<FLT_0D>,
    /// String identifying ion (e.g. H, D, T, He, C, D2, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding neutral species in the ../../neutral array
    pub neutral_index: Option<INT_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: W.m^-2
    pub emitted: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only the 'ion' level is considered and the 'state' array of structure is empty; 1-Ion states are considered and are described in the 'state' array of structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdEnergyIonState>,
}

/// Energy fluxes due to kinetic energy of particles related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdKinetic {
    /// Electron fluxes. Fluxes are given at the wall, after the sheath.
    pub electrons: WallDescriptionGgdEnergySimple,
    /// Fluxes related to the various ion species, in the sense of isonuclear or isomolecular sequences. Ionization states (and other types of states) must be differentiated at the state level below. Fluxes are given at the wall, after the sheath.
    pub ion: Vec<WallDescriptionGgdEnergyIon>,
    /// Neutral species fluxes
    pub neutral: Vec<WallDescriptionGgdEnergyNeutral>,
}

/// Energy fluxes due to recombination related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecombination {
    /// Fluxes related to the various ion species, in the sense of isonuclear or isomolecular sequences. Ionization states (and other types of states) must be differentiated at the state level below
    pub ion: Vec<WallDescriptionGgdEnergyIon>,
    /// Neutral species fluxes
    pub neutral: Vec<WallDescriptionGgdEnergyNeutral>,
}

/// Patricle energy fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdEnergy {
    /// Total radiation, not split by process
    pub radiation: WallDescriptionGgdEnergySimple,
    /// Current energy fluxes
    pub current: WallDescriptionGgdEnergySimple,
    /// Wall recombination
    pub recombination: WallDescriptionGgdRecombination,
    /// Energy fluxes due to the kinetic energy of particles
    pub kinetic: WallDescriptionGgdKinetic,
}

/// Neutral state fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticleNeutralState {
    /// String identifying state
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Neutral type, in terms of energy. ID =1: cold; 2: thermal; 3: fast; 4: NBI
    pub neutral_type: IdentifierDynamicAos3,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub emitted: Vec<GenericGridScalar>,
}

/// Neutral fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticleNeutral {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// String identifying neutral (e.g. H, D, T, He, C, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding ion species in the ../../ion array
    pub ion_index: Option<INT_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub emitted: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdParticleNeutralState>,
}

/// Ion state fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticleIonState {
    /// Minimum Z of the charge state bundle
    /// Units: e
    pub z_min: Option<FLT_0D>,
    /// Maximum Z of the charge state bundle
    /// Units: e
    pub z_max: Option<FLT_0D>,
    /// String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...)
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub emitted: Vec<GenericGridScalar>,
}

/// Ion fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticleIon {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// Ion charge (of the dominant ionization state; lumped ions are allowed)
    /// Units: e
    pub z_ion: Option<FLT_0D>,
    /// String identifying ion (e.g. H, D, T, He, C, D2, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding neutral species in the ../../neutral array
    pub neutral_index: Option<INT_0D>,
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub emitted: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only the 'ion' level is considered and the 'state' array of structure is empty; 1-Ion states are considered and are described in the 'state' array of structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdParticleIonState>,
}

/// Electron fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticleEl {
    /// Incident fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub incident: Vec<GenericGridScalar>,
    /// Emitted fluxes for various wall components (grid subsets)
    /// Units: m^-2.s^-1
    pub emitted: Vec<GenericGridScalar>,
}

/// Patricle fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdParticle {
    /// Electron fluxes
    pub electrons: WallDescriptionGgdParticleEl,
    /// Fluxes related to the various ion species, in the sense of isonuclear or isomolecular sequences. Ionization states (and other types of states) must be differentiated at the state level below
    pub ion: Vec<WallDescriptionGgdParticleIon>,
    /// Neutral species fluxes
    pub neutral: Vec<WallDescriptionGgdParticleNeutral>,
}

/// Neutral state fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecyclingNeutralState {
    /// String identifying state
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Neutral type, in terms of energy. ID =1: cold; 2: thermal; 3: fast; 4: NBI
    pub neutral_type: IdentifierDynamicAos3,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Recycling coefficient for various wall components (grid subsets)
    pub coefficient: Vec<GenericGridScalar>,
}

/// Neutral fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecyclingNeutral {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// String identifying neutral (e.g. H, D, T, He, C, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding ion species in the ../../ion array
    pub ion_index: Option<INT_0D>,
    /// Recycling coefficient for various wall components (grid subsets)
    pub coefficient: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdRecyclingNeutralState>,
}

/// Ion state fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecyclingIonState {
    /// Minimum Z of the charge state bundle
    /// Units: e
    pub z_min: Option<FLT_0D>,
    /// Maximum Z of the charge state bundle
    /// Units: e
    pub z_max: Option<FLT_0D>,
    /// String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...)
    pub name: Option<STR_0D>,
    /// Vibrational level (can be bundled)
    /// Units: e
    pub vibrational_level: Option<FLT_0D>,
    /// Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature.
    pub vibrational_mode: Option<STR_0D>,
    /// Configuration of atomic orbitals of this state, e.g. 1s2-2s1
    pub electron_configuration: Option<STR_0D>,
    /// Recycling coefficient for various wall components (grid subsets)
    pub coefficient: Vec<GenericGridScalar>,
}

/// Ion fluxes related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecyclingIon {
    /// List of elements forming the atom or molecule
    pub element: Vec<PlasmaCompositionNeutralElement>,
    /// Ion charge (of the dominant ionization state; lumped ions are allowed)
    /// Units: e
    pub z_ion: Option<FLT_0D>,
    /// String identifying ion (e.g. H, D, T, He, C, D2, ...)
    pub name: Option<STR_0D>,
    /// Index of the corresponding neutral species in the ../../neutral array
    pub neutral_index: Option<INT_0D>,
    /// Recycling coefficient for various wall components (grid subsets)
    pub coefficient: Vec<GenericGridScalar>,
    /// Multiple states calculation flag : 0-Only the 'ion' level is considered and the 'state' array of structure is empty; 1-Ion states are considered and are described in the 'state' array of structure
    pub multiple_states_flag: Option<INT_0D>,
    /// Fluxes related to the different states of the species
    pub state: Vec<WallDescriptionGgdRecyclingIonState>,
}

/// Recycling coefficients in the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdRecycling {
    /// Recycling coefficients for the various ion species, in the sense of isonuclear or isomolecular sequences. Ionization states (and other types of states) must be differentiated at the state level below
    pub ion: Vec<WallDescriptionGgdRecyclingIon>,
    /// Recycling coefficients for the various neutral species
    pub neutral: Vec<WallDescriptionGgdRecyclingNeutral>,
}

/// Physics quantities related to the 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdGgd {
    /// Net power density arriving on the wall surface, for various wall components (grid subsets)
    /// Units: W.m^-2
    pub power_density: Vec<GenericGridScalar>,
    /// Temperature of the wall, for various wall components (grid subsets)
    /// Units: K
    pub temperature: Vec<GenericGridScalar>,
    /// Electric potential applied to the wall element by outside means, for various wall components (grid subsets). Different from the plasma electric potential or the sheath potential drop.
    /// Units: V
    pub v_biasing: Vec<GenericGridScalar>,
    /// Fraction of incoming particles that is reflected back to the vacuum chamber
    pub recycling: WallDescriptionGgdRecycling,
    /// Particle fluxes. The incident and emitted components are distinguished. The net flux received by the wall is equal to incident - emitted
    pub particle_fluxes: WallDescriptionGgdParticle,
    /// Energy fluxes. The incident and emitted components are distinguished. The net flux received by the wall is equal to incident - emitted
    pub energy_fluxes: WallDescriptionGgdEnergy,
    /// Total current density, given on various grid subsets
    /// Units: A.m^-2
    pub j_total: Vec<GenericGridVectorComponentsRphiz>,
    /// Magnetic field, given on various grid subsets
    /// Units: T
    pub b_field: Vec<GenericGridVectorComponentsRphiz>,
    /// Electromagnetic force density computed by the cross-product of j_total x b_field and given on various grid subsets
    /// Units: N.m^-3
    pub em_force_density: Vec<GenericGridVectorComponentsRphiz>,
    /// Electric field, given on various grid subsets
    /// Units: V.m^-1
    pub e_field: Vec<GenericGridVectorComponentsRphiz>,
    /// Magnetic vector potential, given on various grid subsets
    /// Units: T.m
    pub a_field: Vec<GenericGridVectorComponentsRphiz>,
    /// Poloidal flux, given on various grid subsets. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi: Vec<GenericGridScalar>,
    /// Electric potential, given on various grid subsets
    /// Units: V
    pub phi_potential: Vec<GenericGridScalar>,
    /// Resistivity, given on various grid subsets
    /// Units: ohm.m
    pub resistivity: Vec<GenericGridScalar>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Thickness of a thin wall with GGD description
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdThickness {
    /// The thickness is given for various wall components (grid subsets)
    /// Units: m
    pub grid_subset: Vec<GenericGridScalar>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Bidirectional Reflectance Distribution Function of each wall surface element with GGD description
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdBrdf {
    /// The BRDF model type is described for various wall components (grid subsets), using the identifier convention below
    pub r#type: Vec<GenericGridIdentifier>,
    /// Parameters of the BRDF model for various wall components (grid subsets)
    /// Units: mixed
    pub parameters: Vec<GenericGridVector>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Material forming the wall with GGD description
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdMaterial {
    /// Material is described for various wall components (grid subsets), using the identifier convention below
    pub grid_subset: Vec<GenericGridIdentifier>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Component type for GGD description
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgdComponent {
    /// Identifiers of the components (described in the various grid_subsets). Although the details may be machine-specific, a tree-like syntax must be followed, listing first top level components, then going down to finer element description. The tree levels are separated by a /, using a number of levels relevant to the granularity of the description. Example : ic_antenna/a1/bumpers refers to the bumpers of the a1 IC antenna
    pub identifiers: Option<STR_1D>,
    /// The component type is given for various grid_subsets, using the identifier convention below
    pub r#type: Vec<GenericGridIdentifierSingle>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// 3D wall description using the GGD
#[derive(Debug, Clone, Default)]
pub struct WallDescriptionGgd {
    /// Type of wall: index = 0 for gas tight, 1 for a wall with holes/open ports, 2 for a thin wall description
    pub r#type: IdentifierStatic,
    /// Wall geometry described using the Generic Grid Description, for various time slices (in case of mobile wall elements). The timebase of this array of structure must be a subset of the timebase on which physical quantities are described (../ggd structure). Grid_subsets are used to describe various  wall components in a modular way.
    pub grid_ggd: Vec<GenericGridAos3Root>,
    /// Material of each grid_ggd object, given for each slice of the grid_ggd time base (the material is not supposed to change, but grid_ggd may evolve with time)
    pub material: Vec<WallDescriptionGgdMaterial>,
    /// Description of the components represented by various subsets, given for each slice of the grid_ggd time base (the component description is not supposed to change, but grid_ggd may evolve with time)
    pub component: Vec<WallDescriptionGgdComponent>,
    /// In the case of a thin wall description, effective thickness of each surface element of grid_ggd, given for each slice of the grid_ggd time base (the thickness is not supposed to change, but grid_ggd may evolve with time)
    pub thickness: Vec<WallDescriptionGgdThickness>,
    /// Bidirectional Reflectance Distribution Function, given for each slice of the grid_ggd time base (the component description is not supposed to change, but grid_ggd may evolve with time)
    pub brdf: Vec<WallDescriptionGgdBrdf>,
    /// Wall physics quantities represented using the general grid description, for various time slices.
    pub ggd: Vec<WallDescriptionGgdGgd>,
}

/// Element entering in the composition of the neutral atom or molecule (constant)
#[derive(Debug, Clone, Default)]
pub struct PlasmaCompositionNeutralElementConstant {
    /// Mass of atom
    /// Units: u
    pub a: Option<FLT_0D>,
    /// Nuclear charge
    /// Units: e
    pub z_n: Option<INT_0D>,
    /// Number of atoms of this element in the molecule
    pub atoms_n: Option<INT_0D>,
}

/// Standard type for identifiers (static). The three fields: name, index and description are all representations of the same information. Associated with each application of this identifier-type, there should be a translation table defining the three fields for all objects to be identified.
#[derive(Debug, Clone, Default)]
pub struct IdentifierStatic {
    /// Short string identifier
    pub name: Option<STR_0D>,
    /// Integer identifier (enumeration index within a list). Private identifier values must be indicated by a negative index.
    pub index: Option<INT_0D>,
    /// Verbose description
    pub description: Option<STR_0D>,
}

/// Structure for list of R, Z positions (1D, constant)
#[derive(Debug, Clone, Default)]
pub struct Rz1dStatic {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_1D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_1D>,
}

/// Signal (FLT_1D) with its time base
#[derive(Debug, Clone, Default)]
pub struct SignalFlt1d {
    /// Data
    /// Units: as_parent
    pub data: Option<FLT_1D>,
    /// Time
    /// Units: s
    pub time: Option<FLT_1D>,
}

/// Structure for list of R, Z positions (1D list of Npoints, dynamic within a type 3 array of structures (index on time), with time as sibling)
#[derive(Debug, Clone, Default)]
pub struct Rz1dDynamicAosTime {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_1D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_1D>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// 2D vessel description
#[derive(Debug, Clone, Default)]
pub struct Vessel2d {
    /// Type of the description. index = 0 for the official single/multiple annular representation and 1 for the official block element representation for each unit. Additional representations needed on a code-by-code basis follow same incremental pair tagging starting on index=2
    pub r#type: IdentifierStatic,
    /// Set of units
    pub unit: Vec<Vessel2dUnit>,
}

/// Scalar real values on a generic grid (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridScalar {
    /// Index of the grid used to represent this quantity
    pub grid_index: Option<INT_0D>,
    /// Index of the grid subset the data is provided on. Corresponds to the index used in the grid subset definition: grid_subset(:)/identifier/index
    pub grid_subset_index: Option<INT_0D>,
    /// One scalar value is provided per element in the grid subset.
    /// Units: as_parent
    pub values: Option<FLT_1D>,
    /// Interpolation coefficients, to be used for a high precision evaluation of the physical quantity with finite elements, provided per element in the grid subset (first dimension).
    /// Units: as_parent
    pub coefficients: Option<FLT_2D>,
}

/// Standard type for identifiers (dynamic within type 3 array of structures (index on time)). The three fields: name, index and description are all representations of the same information. Associated with each application of this identifier-type, there should be a translation table defining the three fields for all objects to be identified.
#[derive(Debug, Clone, Default)]
pub struct IdentifierDynamicAos3 {
    /// Short string identifier
    pub name: Option<STR_0D>,
    /// Integer identifier (enumeration index within a list). Private identifier values must be indicated by a negative index.
    pub index: Option<INT_0D>,
    /// Verbose description
    pub description: Option<STR_0D>,
}

/// Element entering in the composition of the neutral atom or molecule (within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct PlasmaCompositionNeutralElement {
    /// Mass of atom
    /// Units: u
    pub a: Option<FLT_0D>,
    /// Nuclear charge
    /// Units: e
    pub z_n: Option<INT_0D>,
    /// Number of atoms of this element in the molecule
    pub atoms_n: Option<INT_0D>,
}

/// Vector components in predefined directions on a generic grid, R, Z and toroidal directions only (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridVectorComponentsRphiz {
    /// Index of the grid used to represent this quantity
    pub grid_index: Option<INT_0D>,
    /// Index of the grid subset the data is provided on. Corresponds to the index used in the grid subset definition: grid_subset(:)/identifier/index
    pub grid_subset_index: Option<INT_0D>,
    /// Component along the major radius axis, one scalar value is provided per element in the grid subset.
    /// Units: as_parent
    pub r: Option<FLT_1D>,
    /// Interpolation coefficients for the component along the major radius axis, to be used for a high precision evaluation of the physical quantity with finite elements, provided per element in the grid subset (first dimension).
    /// Units: as_parent
    pub r_coefficients: Option<FLT_2D>,
    /// Toroidal component, one scalar value is provided per element in the grid subset.
    /// Units: as_parent
    pub phi: Option<FLT_1D>,
    /// Interpolation coefficients for the toroidal component, to be used for a high precision evaluation of the physical quantity with finite elements, provided per element in the grid subset (first dimension).
    /// Units: as_parent
    pub phi_coefficients: Option<FLT_2D>,
    /// Component along the height axis, one scalar value is provided per element in the grid subset.
    /// Units: as_parent
    pub z: Option<FLT_1D>,
    /// Interpolation coefficients for the component along the height axis, to be used for a high precision evaluation of the physical quantity with finite elements, provided per element in the grid subset (first dimension).
    /// Units: as_parent
    pub z_coefficients: Option<FLT_2D>,
}

/// Identifier values on a generic grid (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridIdentifier {
    /// Index of the grid used to represent this quantity
    pub grid_index: Option<INT_0D>,
    /// Index of the grid subset the data is provided on. Corresponds to the index used in the grid subset definition: grid_subset(:)/identifier/index
    pub grid_subset_index: Option<INT_0D>,
    /// Identifier values, one value is provided per element in the grid subset. If the size of the child arrays is 1, their value applies to all elements of the subset.
    pub identifiers: IdentifierDynamicAos31d,
}

/// Vector values on a generic grid (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridVector {
    /// Index of the grid used to represent this quantity
    pub grid_index: Option<INT_0D>,
    /// Index of the grid subset the data is provided on. Corresponds to the index used in the grid subset definition: grid_subset(:)/identifier/index
    pub grid_subset_index: Option<INT_0D>,
    /// List of vector components, one list per element in the grid subset. First dimension: element index. Second dimension: vector component index.
    /// Units: as_parent
    pub values: Option<FLT_2D>,
    /// Interpolation coefficients, to be used for a high precision evaluation of the physical quantity with finite elements, provided per element in the grid subset (first dimension). Second dimension: vector component index. Third dimension: coefficient index
    /// Units: as_parent
    pub coefficients: Option<FLT_3D>,
}

/// Identifier value (single value per subset) on a generic grid (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridIdentifierSingle {
    /// Index of the grid used to represent this quantity
    pub grid_index: Option<INT_0D>,
    /// Index of the grid subset the data is provided on. Corresponds to the index used in the grid subset definition: grid_subset(:)/identifier/index
    pub grid_subset_index: Option<INT_0D>,
    /// Identifier value for the grid subset
    pub identifier: IdentifierDynamicAos3,
}

/// Generic grid (being itself the root of a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridAos3Root {
    /// Grid identifier
    pub identifier: IdentifierDynamicAos3,
    /// Path of the grid, including the IDS name, in case of implicit reference to a grid_ggd node described in another IDS. To be filled only if the grid is not described explicitly in this grid_ggd structure. Example syntax: #wall:2/description_ggd(1)/grid_ggd, means that the grid is located in the wall IDS, occurrence 2, with relative path description_ggd(1)/grid_ggd, using Fortran index convention (here : first index of the array)
    pub path: Option<STR_0D>,
    /// Set of grid spaces
    pub space: Vec<GenericGridDynamicSpace>,
    /// Grid subsets
    pub grid_subset: Vec<GenericGridDynamicGridSubset>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Structure describing the reference temperature for which static data are given
#[derive(Debug, Clone, Default)]
pub struct TemperatureReference {
    /// Description of how the reference temperature is defined : for which object, at which location, ...
    pub description: Option<STR_0D>,
    /// Reference temperature
    /// Units: K
    pub data: Option<FLT_0D>,
}

/// Generic decription of the code-specific parameters for the code that has produced this IDS
#[derive(Debug, Clone, Default)]
pub struct Code {
    /// Name of software generating IDS
    pub name: Option<STR_0D>,
    /// Short description of the software (type, purpose)
    pub description: Option<STR_0D>,
    /// Unique commit reference of software
    pub commit: Option<STR_0D>,
    /// Unique version (tag) of software
    pub version: Option<STR_0D>,
    /// URL of software repository
    pub repository: Option<STR_0D>,
    /// List of the code specific parameters in XML format
    pub parameters: Option<STR_0D>,
    /// Output flag : 0 means the run is successful, other values mean some difficulty has been encountered, the exact meaning is then code specific. Negative values mean the result shall not be used.
    pub output_flag: Option<INT_1D>,
    /// List of external libraries used by the code that has produced this IDS
    pub library: Vec<Library>,
}

/// 2D vessel unit description
#[derive(Debug, Clone, Default)]
pub struct Vessel2dUnit {
    /// Short string identifier (unique for a given device)
    pub name: Option<STR_0D>,
    /// Description, e.g. “channel viewing the upper divertor”
    pub description: Option<STR_0D>,
    /// Annular representation of a layer by two contours, inner and outer. Alternatively, the layer can be described by a centreline and thickness.
    pub annular: Vessel2dAnnular,
    /// Set of block elements
    pub element: Vec<Vessel2dElement>,
    /// Set of materials in this unit
    pub material: Vec<IdentifierStatic>,
    /// Fraction of the volume of each material in this unit
    pub material_volume_fraction: Option<FLT_1D>,
}

/// Standard type for identifiers (1D arrays for each node), dynamic within type 3 array of structures (index on time). The three fields: name, index and description are all representations of the same information. Associated with each application of this identifier-type, there should be a translation table defining the three fields for all objects to be identified.
#[derive(Debug, Clone, Default)]
pub struct IdentifierDynamicAos31d {
    /// Short string identifiers
    pub names: Option<STR_1D>,
    /// Integer identifiers (enumeration index within a list). Private identifier values must be indicated by a negative index.
    pub indices: Option<INT_1D>,
    /// Verbose description
    pub descriptions: Option<STR_1D>,
}

/// Generic grid space (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicSpace {
    /// Space identifier
    pub identifier: IdentifierDynamicAos3,
    /// Type of space geometry (0: standard, 1:Fourier, >1: Fourier with periodicity)
    pub geometry_type: IdentifierDynamicAos3,
    /// Type of coordinates describing the physical space, for every coordinate of the space. The size of this node therefore defines the dimension of the space.
    pub coordinates_type: Vec<IdentifierDynamicAos3>,
    /// Definition of the space objects for every dimension (from one to the dimension of the highest-dimensional objects). The index correspond to 1=nodes, 2=edges, 3=faces, 4=cells/volumes, .... For every index, a collection of objects of that dimension is described.
    pub objects_per_dimension: Vec<GenericGridDynamicSpaceDimension>,
}

/// Generic grid grid_subset (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicGridSubset {
    /// Grid subset identifier
    pub identifier: IdentifierDynamicAos3,
    /// Space dimension of the grid subset elements, using the convention 1=nodes, 2=edges, 3=faces, 4=cells/volumes
    pub dimension: Option<INT_0D>,
    /// Set of elements defining the grid subset. An element is defined by a combination of objects from potentially all spaces
    pub element: Vec<GenericGridDynamicGridSubsetElement>,
    /// Set of bases for the grid subset. For each base, the structure describes the projection of the base vectors on the canonical frame of the grid.
    pub base: Vec<GenericGridDynamicGridSubsetMetric>,
    /// Metric of the canonical frame onto Cartesian coordinates
    pub metric: GenericGridDynamicGridSubsetMetric,
}

/// Library used by the code that has produced this IDS
#[derive(Debug, Clone, Default)]
pub struct Library {
    /// Name of software
    pub name: Option<STR_0D>,
    /// Short description of the software (type, purpose)
    pub description: Option<STR_0D>,
    /// Unique commit reference of software
    pub commit: Option<STR_0D>,
    /// Unique version (tag) of software
    pub version: Option<STR_0D>,
    /// URL of software repository
    pub repository: Option<STR_0D>,
    /// List of the code specific parameters in XML format
    pub parameters: Option<STR_0D>,
}

/// 2D vessel annular description
#[derive(Debug, Clone, Default)]
pub struct Vessel2dAnnular {
    /// Inner vessel outline. Repeat the first point in case of a closed contour
    pub outline_inner: Rz1dStatic,
    /// Outer vessel outline. Repeat the first point in case of a closed contour
    pub outline_outer: Rz1dStatic,
    /// Centreline, i.e. middle of the vessel layer as a series of point. Repeat the first point in case of a closed contour
    pub centreline: Rz1dStatic,
    /// Thickness of the vessel layer  in the perpendicular direction to the centreline. Thickness(i) is the thickness of the layer between centreline/r(i),z(i) and centreline/r(i+1),z(i+1), so its size is equal to the length of centreline/r-1 if the thickness is varying. If the thickness is constant for all points, allocate this node to size 1 to store a single value.
    /// Units: m
    pub thickness: Option<FLT_1D>,
    /// Resistivity of the vessel unit
    /// Units: ohm.m
    pub resistivity: Option<FLT_0D>,
}

/// 2D vessel block element description
#[derive(Debug, Clone, Default)]
pub struct Vessel2dElement {
    /// Name of the block element
    pub name: Option<STR_0D>,
    /// Outline of the block element. Repeat the first point in case of a closed contour
    pub outline: Rz1dStatic,
    /// Thickness of this element evaluated at the midplane (whenever relevant)
    /// Units: m
    pub midplane_thickness: Option<FLT_0D>,
    /// Resistivity of the block element
    /// Units: ohm.m
    pub resistivity: Option<FLT_0D>,
    /// Resistance of the block element
    /// Units: ohm
    pub resistance: Option<FLT_0D>,
}

/// Generic grid, list of dimensions within a space (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicSpaceDimension {
    /// Set of objects for a given dimension
    pub object: Vec<GenericGridDynamicSpaceDimensionObject>,
    /// Content of the ../object/geometry node for this dimension
    pub geometry_content: IdentifierDynamicAos3,
}

/// Generic grid, element part of a grid_subset (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicGridSubsetElement {
    /// Set of objects defining the element
    pub object: Vec<GenericGridDynamicGridSubsetElementObject>,
}

/// Generic grid, metric description for a given grid_subset and base (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicGridSubsetMetric {
    /// Metric Jacobian
    /// Units: mixed
    pub jacobian: Option<FLT_1D>,
    /// Covariant metric tensor, given on each element of the subgrid (first dimension)
    /// Units: mixed
    pub tensor_covariant: Option<FLT_3D>,
    /// Contravariant metric tensor, given on each element of the subgrid (first dimension)
    /// Units: mixed
    pub tensor_contravariant: Option<FLT_3D>,
}

/// Generic grid, list of objects of a given dimension within a space (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicSpaceDimensionObject {
    /// Set of  (n-1)-dimensional objects defining the boundary of this n-dimensional object
    pub boundary: Vec<GenericGridDynamicSpaceDimensionObjectBoundary>,
    /// Geometry data associated with the object, its detailed content is defined by ../../geometry_content. Its dimension depends on the type of object, geometry and coordinate considered.
    /// Units: mixed
    pub geometry: Option<FLT_1D>,
    /// List of nodes forming this object (indices to objects_per_dimension(1)%object(:) in Fortran notation)
    pub nodes: Option<INT_1D>,
    /// Measure of the space object, i.e. physical size (length for 1d, area for 2d, volume for 3d objects,...)
    /// Units: m^dimension
    pub measure: Option<FLT_0D>,
    /// 2D geometry data associated with the object. Its dimension depends on the type of object, geometry and coordinate considered. Typically, the first dimension represents the object coordinates, while the second dimension would represent the values of the various degrees of freedom of the finite element attached to the object.
    /// Units: mixed
    pub geometry_2d: Option<FLT_2D>,
}

/// Generic grid, object part of an element part of a grid_subset (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicGridSubsetElementObject {
    /// Index of the space from which that object is taken
    pub space: Option<INT_0D>,
    /// Dimension of the object - using the convention  1=nodes, 2=edges, 3=faces, 4=cells/volumes
    pub dimension: Option<INT_0D>,
    /// Object index
    pub index: Option<INT_0D>,
}

/// Generic grid, description of an object boundary and its neighbours (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamicSpaceDimensionObjectBoundary {
    /// Index of this (n-1)-dimensional boundary object
    pub index: Option<INT_0D>,
    /// List of indices of the n-dimensional objects adjacent to the given n-dimensional object. An object can possibly have multiple neighbours on a boundary
    pub neighbours: Option<INT_1D>,
}

// ============================================================================
// Root IDS Structure
// ============================================================================

/// Description of the torus wall and its interaction with the plasma
#[derive(Debug, Clone, Default)]
pub struct Wall {
    /// Reference temperature for which the machine description data is given in this IDS
    pub temperature_reference: TemperatureReference,
    /// First wall surface area
    /// Units: m^2
    pub first_wall_surface_area: Option<FLT_0D>,
    /// Peak power flux on the first wall (including divertors)
    /// Units: W.m^-2
    pub first_wall_power_flux_peak: SignalFlt1d,
    /// Peak power flux on the first wall (excluding the divertors surface)
    /// Units: W.m^-2
    pub first_wall_power_flux_peak_outside_divertors: SignalFlt1d,
    /// Volume available to gas or plasma enclosed by the first wall contour
    /// Units: m^3
    pub first_wall_enclosed_volume: Option<FLT_0D>,
    /// Simple 0D description of plasma-wall interaction
    pub global_quantities: WallGlobalQuantitites,
    /// Set of 2D wall descriptions, for each type of possible physics or engineering configurations necessary (gas tight vs wall with ports and holes, coarse vs fine representation, single contour limiter, disjoint gapped plasma facing components, ...). A simplified description of the toroidal extension of the 2D contours is also provided by using the phi_extensions nodes.
    pub description_2d: Vec<Wall2d>,
    /// Set of 3D wall descriptions, described using the GGD, for each type of possible physics or engineering configurations necessary (gas tight vs wall with ports and holes, coarse vs fine representation, ...).
    pub description_ggd: Vec<WallDescriptionGgd>,
    pub code: Code,
}

// ============================================================================
// View, Accessor, and Accumulator Types
// ============================================================================

// --- PlasmaCompositionNeutralElementConstant View Types ---

/// View over multiple PlasmaCompositionNeutralElementConstant with field accumulation
pub struct PlasmaCompositionNeutralElementConstantSliceView<'a> {
    data: &'a [PlasmaCompositionNeutralElementConstant],
    pub a: Accumulator<'a, PlasmaCompositionNeutralElementConstant, FLT_0D>,
    pub z_n: Accumulator<'a, PlasmaCompositionNeutralElementConstant, INT_0D>,
    pub atoms_n: Accumulator<'a, PlasmaCompositionNeutralElementConstant, INT_0D>,
}

impl<'a> PlasmaCompositionNeutralElementConstantSliceView<'a> {
    pub fn new(data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self {
        Self {
            data,
            a: Accumulator::new(data, |item: &PlasmaCompositionNeutralElementConstant| item.a, "a"),
            z_n: Accumulator::new(data, |item: &PlasmaCompositionNeutralElementConstant| item.z_n, "z_n"),
            atoms_n: Accumulator::new(data, |item: &PlasmaCompositionNeutralElementConstant| item.atoms_n, "atoms_n"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PlasmaCompositionNeutralElementConstant> {
        self.data.iter()
    }
}

/// Mutable view over multiple PlasmaCompositionNeutralElementConstant
pub struct PlasmaCompositionNeutralElementConstantSliceViewMut<'a> {
    data: &'a mut [PlasmaCompositionNeutralElementConstant],
}

impl<'a> PlasmaCompositionNeutralElementConstantSliceViewMut<'a> {
    pub fn new(data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut PlasmaCompositionNeutralElementConstant> {
        self.data.iter_mut()
    }
}

/// Index trait for PlasmaCompositionNeutralElementConstant - enables .field(0) and .field(0..2) syntax
pub trait PlasmaCompositionNeutralElementConstantIndex<'a> {
    type Output;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output;
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for usize {
    type Output = &'a PlasmaCompositionNeutralElementConstant;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        &data[self]
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::Range<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::RangeTo<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantIndex<'a> for std::ops::RangeFull {
    type Output = PlasmaCompositionNeutralElementConstantSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceView::new(data)
    }
}

/// Mutable index trait for PlasmaCompositionNeutralElementConstant - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait PlasmaCompositionNeutralElementConstantMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output;
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for usize {
    type Output = &'a mut PlasmaCompositionNeutralElementConstant;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::Range<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementConstantMutIndex<'a> for std::ops::RangeFull {
    type Output = PlasmaCompositionNeutralElementConstantSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElementConstant]) -> Self::Output {
        PlasmaCompositionNeutralElementConstantSliceViewMut::new(data)
    }
}

// --- WallGlobalQuantititesNeutralOrigin View Types ---

/// View over multiple WallGlobalQuantititesNeutralOrigin with field accumulation
pub struct WallGlobalQuantititesNeutralOriginSliceView<'a> {
    data: &'a [WallGlobalQuantititesNeutralOrigin],
    pub name: StringAccumulator<'a, WallGlobalQuantititesNeutralOrigin>,
}

impl<'a> WallGlobalQuantititesNeutralOriginSliceView<'a> {
    pub fn new(data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallGlobalQuantititesNeutralOrigin| item.name.clone(), "name"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallGlobalQuantititesNeutralOrigin> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallGlobalQuantititesNeutralOrigin
pub struct WallGlobalQuantititesNeutralOriginSliceViewMut<'a> {
    data: &'a mut [WallGlobalQuantititesNeutralOrigin],
}

impl<'a> WallGlobalQuantititesNeutralOriginSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallGlobalQuantititesNeutralOrigin> {
        self.data.iter_mut()
    }
}

/// Index trait for WallGlobalQuantititesNeutralOrigin - enables .field(0) and .field(0..2) syntax
pub trait WallGlobalQuantititesNeutralOriginIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output;
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for usize {
    type Output = &'a WallGlobalQuantititesNeutralOrigin;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::Range<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginIndex<'a> for std::ops::RangeFull {
    type Output = WallGlobalQuantititesNeutralOriginSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceView::new(data)
    }
}

/// Mutable index trait for WallGlobalQuantititesNeutralOrigin - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallGlobalQuantititesNeutralOriginMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output;
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for usize {
    type Output = &'a mut WallGlobalQuantititesNeutralOrigin;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralOriginMutIndex<'a> for std::ops::RangeFull {
    type Output = WallGlobalQuantititesNeutralOriginSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutralOrigin]) -> Self::Output {
        WallGlobalQuantititesNeutralOriginSliceViewMut::new(data)
    }
}

// --- WallGlobalQuantititesNeutral View Types ---

/// View over multiple WallGlobalQuantititesNeutral with field accumulation
pub struct WallGlobalQuantititesNeutralSliceView<'a> {
    data: &'a [WallGlobalQuantititesNeutral],
    pub name: StringAccumulator<'a, WallGlobalQuantititesNeutral>,
}

impl<'a> WallGlobalQuantititesNeutralSliceView<'a> {
    pub fn new(data: &'a [WallGlobalQuantititesNeutral]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallGlobalQuantititesNeutral| item.name.clone(), "name"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallGlobalQuantititesNeutral> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallGlobalQuantititesNeutral
pub struct WallGlobalQuantititesNeutralSliceViewMut<'a> {
    data: &'a mut [WallGlobalQuantititesNeutral],
}

impl<'a> WallGlobalQuantititesNeutralSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallGlobalQuantititesNeutral]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallGlobalQuantititesNeutral> {
        self.data.iter_mut()
    }
}

/// Index trait for WallGlobalQuantititesNeutral - enables .field(0) and .field(0..2) syntax
pub trait WallGlobalQuantititesNeutralIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output;
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for usize {
    type Output = &'a WallGlobalQuantititesNeutral;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::Range<usize> {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralIndex<'a> for std::ops::RangeFull {
    type Output = WallGlobalQuantititesNeutralSliceView<'a>;
    fn get(self, data: &'a [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceView::new(data)
    }
}

/// Mutable index trait for WallGlobalQuantititesNeutral - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallGlobalQuantititesNeutralMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output;
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for usize {
    type Output = &'a mut WallGlobalQuantititesNeutral;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallGlobalQuantititesNeutralMutIndex<'a> for std::ops::RangeFull {
    type Output = WallGlobalQuantititesNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallGlobalQuantititesNeutral]) -> Self::Output {
        WallGlobalQuantititesNeutralSliceViewMut::new(data)
    }
}

// --- Wall2dLimiterUnit View Types ---

/// View over `component_type` (IdentifierStatic) across multiple Wall2dLimiterUnit
pub struct Wall2dLimiterUnitComponentTypeView<'a> {
    pub name: StringAccumulator<'a, Wall2dLimiterUnit>,
    pub index: Accumulator<'a, Wall2dLimiterUnit, INT_0D>,
    pub description: StringAccumulator<'a, Wall2dLimiterUnit>,
}

impl<'a> Wall2dLimiterUnitComponentTypeView<'a> {
    pub fn new(data: &'a [Wall2dLimiterUnit]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &Wall2dLimiterUnit| item.component_type.name.clone(), "component_type.name"),
            index: Accumulator::new(data, |item: &Wall2dLimiterUnit| item.component_type.index, "component_type.index"),
            description: StringAccumulator::new(
                data,
                |item: &Wall2dLimiterUnit| item.component_type.description.clone(),
                "component_type.description",
            ),
        }
    }
}

/// View over `outline` (Rz1dStatic) across multiple Wall2dLimiterUnit
pub struct Wall2dLimiterUnitOutlineView<'a> {
    _phantom: std::marker::PhantomData<&'a Wall2dLimiterUnit>,
}

impl<'a> Wall2dLimiterUnitOutlineView<'a> {
    pub fn new(_data: &'a [Wall2dLimiterUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `midplane_power_density_nuclear` (SignalFlt1d) across multiple Wall2dLimiterUnit
pub struct Wall2dLimiterUnitMidplanePowerDensityNuclearView<'a> {
    _phantom: std::marker::PhantomData<&'a Wall2dLimiterUnit>,
}

impl<'a> Wall2dLimiterUnitMidplanePowerDensityNuclearView<'a> {
    pub fn new(_data: &'a [Wall2dLimiterUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `power_nuclear` (SignalFlt1d) across multiple Wall2dLimiterUnit
pub struct Wall2dLimiterUnitPowerNuclearView<'a> {
    _phantom: std::marker::PhantomData<&'a Wall2dLimiterUnit>,
}

impl<'a> Wall2dLimiterUnitPowerNuclearView<'a> {
    pub fn new(_data: &'a [Wall2dLimiterUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over multiple Wall2dLimiterUnit with field accumulation
pub struct Wall2dLimiterUnitSliceView<'a> {
    data: &'a [Wall2dLimiterUnit],
    pub name: StringAccumulator<'a, Wall2dLimiterUnit>,
    pub description: StringAccumulator<'a, Wall2dLimiterUnit>,
    pub component_type: Wall2dLimiterUnitComponentTypeView<'a>,
    pub outline: Wall2dLimiterUnitOutlineView<'a>,
    pub resistivity: Accumulator<'a, Wall2dLimiterUnit, FLT_0D>,
    pub midplane_thickness: Accumulator<'a, Wall2dLimiterUnit, FLT_0D>,
    pub midplane_power_density_nuclear: Wall2dLimiterUnitMidplanePowerDensityNuclearView<'a>,
    pub power_nuclear: Wall2dLimiterUnitPowerNuclearView<'a>,
}

impl<'a> Wall2dLimiterUnitSliceView<'a> {
    pub fn new(data: &'a [Wall2dLimiterUnit]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &Wall2dLimiterUnit| item.name.clone(), "name"),
            description: StringAccumulator::new(data, |item: &Wall2dLimiterUnit| item.description.clone(), "description"),
            component_type: Wall2dLimiterUnitComponentTypeView::new(data),
            outline: Wall2dLimiterUnitOutlineView::new(data),
            resistivity: Accumulator::new(data, |item: &Wall2dLimiterUnit| item.resistivity, "resistivity"),
            midplane_thickness: Accumulator::new(data, |item: &Wall2dLimiterUnit| item.midplane_thickness, "midplane_thickness"),
            midplane_power_density_nuclear: Wall2dLimiterUnitMidplanePowerDensityNuclearView::new(data),
            power_nuclear: Wall2dLimiterUnitPowerNuclearView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Wall2dLimiterUnit> {
        self.data.iter()
    }
}

/// Mutable view over multiple Wall2dLimiterUnit
pub struct Wall2dLimiterUnitSliceViewMut<'a> {
    data: &'a mut [Wall2dLimiterUnit],
}

impl<'a> Wall2dLimiterUnitSliceViewMut<'a> {
    pub fn new(data: &'a mut [Wall2dLimiterUnit]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Wall2dLimiterUnit> {
        self.data.iter_mut()
    }
}

/// Index trait for Wall2dLimiterUnit - enables .field(0) and .field(0..2) syntax
pub trait Wall2dLimiterUnitIndex<'a> {
    type Output;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output;
}

impl<'a> Wall2dLimiterUnitIndex<'a> for usize {
    type Output = &'a Wall2dLimiterUnit;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dLimiterUnitIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dLimiterUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceView::new(data)
    }
}

/// Mutable index trait for Wall2dLimiterUnit - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Wall2dLimiterUnitMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output;
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for usize {
    type Output = &'a mut Wall2dLimiterUnit;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dLimiterUnitMutIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dLimiterUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dLimiterUnit]) -> Self::Output {
        Wall2dLimiterUnitSliceViewMut::new(data)
    }
}

// --- Rz1dDynamicAosTime View Types ---

/// View over multiple Rz1dDynamicAosTime with field accumulation
pub struct Rz1dDynamicAosTimeSliceView<'a> {
    data: &'a [Rz1dDynamicAosTime],
    pub time: Accumulator<'a, Rz1dDynamicAosTime, FLT_0D>,
}

impl<'a> Rz1dDynamicAosTimeSliceView<'a> {
    pub fn new(data: &'a [Rz1dDynamicAosTime]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &Rz1dDynamicAosTime| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Rz1dDynamicAosTime> {
        self.data.iter()
    }
}

/// Mutable view over multiple Rz1dDynamicAosTime
pub struct Rz1dDynamicAosTimeSliceViewMut<'a> {
    data: &'a mut [Rz1dDynamicAosTime],
}

impl<'a> Rz1dDynamicAosTimeSliceViewMut<'a> {
    pub fn new(data: &'a mut [Rz1dDynamicAosTime]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Rz1dDynamicAosTime> {
        self.data.iter_mut()
    }
}

/// Index trait for Rz1dDynamicAosTime - enables .field(0) and .field(0..2) syntax
pub trait Rz1dDynamicAosTimeIndex<'a> {
    type Output;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output;
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for usize {
    type Output = &'a Rz1dDynamicAosTime;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::Range<usize> {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeIndex<'a> for std::ops::RangeFull {
    type Output = Rz1dDynamicAosTimeSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceView::new(data)
    }
}

/// Mutable index trait for Rz1dDynamicAosTime - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Rz1dDynamicAosTimeMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output;
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for usize {
    type Output = &'a mut Rz1dDynamicAosTime;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::Range<usize> {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosTimeMutIndex<'a> for std::ops::RangeFull {
    type Output = Rz1dDynamicAosTimeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAosTime]) -> Self::Output {
        Rz1dDynamicAosTimeSliceViewMut::new(data)
    }
}

// --- Wall2dMobileUnit View Types ---

/// View over multiple Wall2dMobileUnit with field accumulation
pub struct Wall2dMobileUnitSliceView<'a> {
    data: &'a [Wall2dMobileUnit],
    pub name: StringAccumulator<'a, Wall2dMobileUnit>,
    pub resistivity: Accumulator<'a, Wall2dMobileUnit, FLT_0D>,
}

impl<'a> Wall2dMobileUnitSliceView<'a> {
    pub fn new(data: &'a [Wall2dMobileUnit]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &Wall2dMobileUnit| item.name.clone(), "name"),
            resistivity: Accumulator::new(data, |item: &Wall2dMobileUnit| item.resistivity, "resistivity"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Wall2dMobileUnit> {
        self.data.iter()
    }
}

/// Mutable view over multiple Wall2dMobileUnit
pub struct Wall2dMobileUnitSliceViewMut<'a> {
    data: &'a mut [Wall2dMobileUnit],
}

impl<'a> Wall2dMobileUnitSliceViewMut<'a> {
    pub fn new(data: &'a mut [Wall2dMobileUnit]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Wall2dMobileUnit> {
        self.data.iter_mut()
    }
}

/// Index trait for Wall2dMobileUnit - enables .field(0) and .field(0..2) syntax
pub trait Wall2dMobileUnitIndex<'a> {
    type Output;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output;
}

impl<'a> Wall2dMobileUnitIndex<'a> for usize {
    type Output = &'a Wall2dMobileUnit;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(&data[self])
    }
}

impl<'a> Wall2dMobileUnitIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dMobileUnitSliceView<'a>;
    fn get(self, data: &'a [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceView::new(data)
    }
}

/// Mutable index trait for Wall2dMobileUnit - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Wall2dMobileUnitMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output;
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for usize {
    type Output = &'a mut Wall2dMobileUnit;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMobileUnitMutIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dMobileUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2dMobileUnit]) -> Self::Output {
        Wall2dMobileUnitSliceViewMut::new(data)
    }
}

// --- GenericGridScalar View Types ---

/// View over multiple GenericGridScalar with field accumulation
pub struct GenericGridScalarSliceView<'a> {
    data: &'a [GenericGridScalar],
    pub grid_index: Accumulator<'a, GenericGridScalar, INT_0D>,
    pub grid_subset_index: Accumulator<'a, GenericGridScalar, INT_0D>,
}

impl<'a> GenericGridScalarSliceView<'a> {
    pub fn new(data: &'a [GenericGridScalar]) -> Self {
        Self {
            data,
            grid_index: Accumulator::new(data, |item: &GenericGridScalar| item.grid_index, "grid_index"),
            grid_subset_index: Accumulator::new(data, |item: &GenericGridScalar| item.grid_subset_index, "grid_subset_index"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridScalar> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridScalar
pub struct GenericGridScalarSliceViewMut<'a> {
    data: &'a mut [GenericGridScalar],
}

impl<'a> GenericGridScalarSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridScalar]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridScalar> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridScalar - enables .field(0) and .field(0..2) syntax
pub trait GenericGridScalarIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output;
}

impl<'a> GenericGridScalarIndex<'a> for usize {
    type Output = &'a GenericGridScalar;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(&data[self])
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(&data[self])
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(&data[self])
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(&data[self])
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(&data[self])
    }
}

impl<'a> GenericGridScalarIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridScalarSliceView<'a>;
    fn get(self, data: &'a [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridScalar - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridScalarMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output;
}

impl<'a> GenericGridScalarMutIndex<'a> for usize {
    type Output = &'a mut GenericGridScalar;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridScalarMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridScalarSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridScalar]) -> Self::Output {
        GenericGridScalarSliceViewMut::new(data)
    }
}

// --- PlasmaCompositionNeutralElement View Types ---

/// View over multiple PlasmaCompositionNeutralElement with field accumulation
pub struct PlasmaCompositionNeutralElementSliceView<'a> {
    data: &'a [PlasmaCompositionNeutralElement],
    pub a: Accumulator<'a, PlasmaCompositionNeutralElement, FLT_0D>,
    pub z_n: Accumulator<'a, PlasmaCompositionNeutralElement, INT_0D>,
    pub atoms_n: Accumulator<'a, PlasmaCompositionNeutralElement, INT_0D>,
}

impl<'a> PlasmaCompositionNeutralElementSliceView<'a> {
    pub fn new(data: &'a [PlasmaCompositionNeutralElement]) -> Self {
        Self {
            data,
            a: Accumulator::new(data, |item: &PlasmaCompositionNeutralElement| item.a, "a"),
            z_n: Accumulator::new(data, |item: &PlasmaCompositionNeutralElement| item.z_n, "z_n"),
            atoms_n: Accumulator::new(data, |item: &PlasmaCompositionNeutralElement| item.atoms_n, "atoms_n"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PlasmaCompositionNeutralElement> {
        self.data.iter()
    }
}

/// Mutable view over multiple PlasmaCompositionNeutralElement
pub struct PlasmaCompositionNeutralElementSliceViewMut<'a> {
    data: &'a mut [PlasmaCompositionNeutralElement],
}

impl<'a> PlasmaCompositionNeutralElementSliceViewMut<'a> {
    pub fn new(data: &'a mut [PlasmaCompositionNeutralElement]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut PlasmaCompositionNeutralElement> {
        self.data.iter_mut()
    }
}

/// Index trait for PlasmaCompositionNeutralElement - enables .field(0) and .field(0..2) syntax
pub trait PlasmaCompositionNeutralElementIndex<'a> {
    type Output;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output;
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for usize {
    type Output = &'a PlasmaCompositionNeutralElement;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        &data[self]
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::Range<usize> {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::RangeTo<usize> {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(&data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementIndex<'a> for std::ops::RangeFull {
    type Output = PlasmaCompositionNeutralElementSliceView<'a>;
    fn get(self, data: &'a [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceView::new(data)
    }
}

/// Mutable index trait for PlasmaCompositionNeutralElement - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait PlasmaCompositionNeutralElementMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output;
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for usize {
    type Output = &'a mut PlasmaCompositionNeutralElement;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::Range<usize> {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> PlasmaCompositionNeutralElementMutIndex<'a> for std::ops::RangeFull {
    type Output = PlasmaCompositionNeutralElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [PlasmaCompositionNeutralElement]) -> Self::Output {
        PlasmaCompositionNeutralElementSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdEnergyNeutralState View Types ---

/// View over `neutral_type` (IdentifierDynamicAos3) across multiple WallDescriptionGgdEnergyNeutralState
pub struct WallDescriptionGgdEnergyNeutralStateNeutralTypeView<'a> {
    pub name: StringAccumulator<'a, WallDescriptionGgdEnergyNeutralState>,
    pub index: Accumulator<'a, WallDescriptionGgdEnergyNeutralState, INT_0D>,
    pub description: StringAccumulator<'a, WallDescriptionGgdEnergyNeutralState>,
}

impl<'a> WallDescriptionGgdEnergyNeutralStateNeutralTypeView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyNeutralState| item.neutral_type.name.clone(),
                "neutral_type.name",
            ),
            index: Accumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyNeutralState| item.neutral_type.index,
                "neutral_type.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyNeutralState| item.neutral_type.description.clone(),
                "neutral_type.description",
            ),
        }
    }
}

/// View over multiple WallDescriptionGgdEnergyNeutralState with field accumulation
pub struct WallDescriptionGgdEnergyNeutralStateSliceView<'a> {
    data: &'a [WallDescriptionGgdEnergyNeutralState],
    pub name: StringAccumulator<'a, WallDescriptionGgdEnergyNeutralState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdEnergyNeutralState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdEnergyNeutralState>,
    pub neutral_type: WallDescriptionGgdEnergyNeutralStateNeutralTypeView<'a>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdEnergyNeutralState>,
}

impl<'a> WallDescriptionGgdEnergyNeutralStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdEnergyNeutralState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(data, |item: &WallDescriptionGgdEnergyNeutralState| item.vibrational_level, "vibrational_level"),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyNeutralState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            neutral_type: WallDescriptionGgdEnergyNeutralStateNeutralTypeView::new(data),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyNeutralState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdEnergyNeutralState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdEnergyNeutralState
pub struct WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdEnergyNeutralState],
}

impl<'a> WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdEnergyNeutralState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdEnergyNeutralState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdEnergyNeutralStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdEnergyNeutralState;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdEnergyNeutralState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdEnergyNeutralStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdEnergyNeutralState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutralState]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdEnergyIonState View Types ---

/// View over multiple WallDescriptionGgdEnergyIonState with field accumulation
pub struct WallDescriptionGgdEnergyIonStateSliceView<'a> {
    data: &'a [WallDescriptionGgdEnergyIonState],
    pub z_min: Accumulator<'a, WallDescriptionGgdEnergyIonState, FLT_0D>,
    pub z_max: Accumulator<'a, WallDescriptionGgdEnergyIonState, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdEnergyIonState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdEnergyIonState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdEnergyIonState>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdEnergyIonState>,
}

impl<'a> WallDescriptionGgdEnergyIonStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdEnergyIonState]) -> Self {
        Self {
            data,
            z_min: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIonState| item.z_min, "z_min"),
            z_max: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIonState| item.z_max, "z_max"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdEnergyIonState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIonState| item.vibrational_level, "vibrational_level"),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyIonState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdEnergyIonState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdEnergyIonState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdEnergyIonState
pub struct WallDescriptionGgdEnergyIonStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdEnergyIonState],
}

impl<'a> WallDescriptionGgdEnergyIonStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdEnergyIonState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdEnergyIonState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdEnergyIonStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdEnergyIonState;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdEnergyIonState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdEnergyIonStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdEnergyIonState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIonState]) -> Self::Output {
        WallDescriptionGgdEnergyIonStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdEnergyIon View Types ---

/// View over multiple WallDescriptionGgdEnergyIon with field accumulation
pub struct WallDescriptionGgdEnergyIonSliceView<'a> {
    data: &'a [WallDescriptionGgdEnergyIon],
    pub z_ion: Accumulator<'a, WallDescriptionGgdEnergyIon, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdEnergyIon>,
    pub neutral_index: Accumulator<'a, WallDescriptionGgdEnergyIon, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdEnergyIon, INT_0D>,
}

impl<'a> WallDescriptionGgdEnergyIonSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdEnergyIon]) -> Self {
        Self {
            data,
            z_ion: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIon| item.z_ion, "z_ion"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdEnergyIon| item.name.clone(), "name"),
            neutral_index: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIon| item.neutral_index, "neutral_index"),
            multiple_states_flag: Accumulator::new(data, |item: &WallDescriptionGgdEnergyIon| item.multiple_states_flag, "multiple_states_flag"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdEnergyIon> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdEnergyIon
pub struct WallDescriptionGgdEnergyIonSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdEnergyIon],
}

impl<'a> WallDescriptionGgdEnergyIonSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdEnergyIon> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdEnergyIon - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdEnergyIonIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdEnergyIon;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdEnergyIon - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdEnergyIonMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdEnergyIon;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyIonMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyIon]) -> Self::Output {
        WallDescriptionGgdEnergyIonSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdEnergyNeutral View Types ---

/// View over multiple WallDescriptionGgdEnergyNeutral with field accumulation
pub struct WallDescriptionGgdEnergyNeutralSliceView<'a> {
    data: &'a [WallDescriptionGgdEnergyNeutral],
    pub name: StringAccumulator<'a, WallDescriptionGgdEnergyNeutral>,
    pub ion_index: Accumulator<'a, WallDescriptionGgdEnergyNeutral, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdEnergyNeutral, INT_0D>,
}

impl<'a> WallDescriptionGgdEnergyNeutralSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdEnergyNeutral| item.name.clone(), "name"),
            ion_index: Accumulator::new(data, |item: &WallDescriptionGgdEnergyNeutral| item.ion_index, "ion_index"),
            multiple_states_flag: Accumulator::new(data, |item: &WallDescriptionGgdEnergyNeutral| item.multiple_states_flag, "multiple_states_flag"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdEnergyNeutral> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdEnergyNeutral
pub struct WallDescriptionGgdEnergyNeutralSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdEnergyNeutral],
}

impl<'a> WallDescriptionGgdEnergyNeutralSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdEnergyNeutral> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdEnergyNeutral - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdEnergyNeutralIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdEnergyNeutral;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdEnergyNeutral - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdEnergyNeutralMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdEnergyNeutral;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdEnergyNeutralMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdEnergyNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdEnergyNeutral]) -> Self::Output {
        WallDescriptionGgdEnergyNeutralSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdParticleNeutralState View Types ---

/// View over `neutral_type` (IdentifierDynamicAos3) across multiple WallDescriptionGgdParticleNeutralState
pub struct WallDescriptionGgdParticleNeutralStateNeutralTypeView<'a> {
    pub name: StringAccumulator<'a, WallDescriptionGgdParticleNeutralState>,
    pub index: Accumulator<'a, WallDescriptionGgdParticleNeutralState, INT_0D>,
    pub description: StringAccumulator<'a, WallDescriptionGgdParticleNeutralState>,
}

impl<'a> WallDescriptionGgdParticleNeutralStateNeutralTypeView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.neutral_type.name.clone(),
                "neutral_type.name",
            ),
            index: Accumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.neutral_type.index,
                "neutral_type.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.neutral_type.description.clone(),
                "neutral_type.description",
            ),
        }
    }
}

/// View over multiple WallDescriptionGgdParticleNeutralState with field accumulation
pub struct WallDescriptionGgdParticleNeutralStateSliceView<'a> {
    data: &'a [WallDescriptionGgdParticleNeutralState],
    pub name: StringAccumulator<'a, WallDescriptionGgdParticleNeutralState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdParticleNeutralState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdParticleNeutralState>,
    pub neutral_type: WallDescriptionGgdParticleNeutralStateNeutralTypeView<'a>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdParticleNeutralState>,
}

impl<'a> WallDescriptionGgdParticleNeutralStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdParticleNeutralState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.vibrational_level,
                "vibrational_level",
            ),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            neutral_type: WallDescriptionGgdParticleNeutralStateNeutralTypeView::new(data),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutralState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdParticleNeutralState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdParticleNeutralState
pub struct WallDescriptionGgdParticleNeutralStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdParticleNeutralState],
}

impl<'a> WallDescriptionGgdParticleNeutralStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdParticleNeutralState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdParticleNeutralState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdParticleNeutralStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdParticleNeutralState;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdParticleNeutralState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdParticleNeutralStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdParticleNeutralState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutralState]) -> Self::Output {
        WallDescriptionGgdParticleNeutralStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdParticleIonState View Types ---

/// View over multiple WallDescriptionGgdParticleIonState with field accumulation
pub struct WallDescriptionGgdParticleIonStateSliceView<'a> {
    data: &'a [WallDescriptionGgdParticleIonState],
    pub z_min: Accumulator<'a, WallDescriptionGgdParticleIonState, FLT_0D>,
    pub z_max: Accumulator<'a, WallDescriptionGgdParticleIonState, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdParticleIonState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdParticleIonState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdParticleIonState>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdParticleIonState>,
}

impl<'a> WallDescriptionGgdParticleIonStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdParticleIonState]) -> Self {
        Self {
            data,
            z_min: Accumulator::new(data, |item: &WallDescriptionGgdParticleIonState| item.z_min, "z_min"),
            z_max: Accumulator::new(data, |item: &WallDescriptionGgdParticleIonState| item.z_max, "z_max"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdParticleIonState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(data, |item: &WallDescriptionGgdParticleIonState| item.vibrational_level, "vibrational_level"),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleIonState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdParticleIonState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdParticleIonState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdParticleIonState
pub struct WallDescriptionGgdParticleIonStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdParticleIonState],
}

impl<'a> WallDescriptionGgdParticleIonStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdParticleIonState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdParticleIonState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdParticleIonStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdParticleIonState;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdParticleIonState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdParticleIonStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdParticleIonState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIonState]) -> Self::Output {
        WallDescriptionGgdParticleIonStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdParticleIon View Types ---

/// View over multiple WallDescriptionGgdParticleIon with field accumulation
pub struct WallDescriptionGgdParticleIonSliceView<'a> {
    data: &'a [WallDescriptionGgdParticleIon],
    pub z_ion: Accumulator<'a, WallDescriptionGgdParticleIon, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdParticleIon>,
    pub neutral_index: Accumulator<'a, WallDescriptionGgdParticleIon, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdParticleIon, INT_0D>,
}

impl<'a> WallDescriptionGgdParticleIonSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdParticleIon]) -> Self {
        Self {
            data,
            z_ion: Accumulator::new(data, |item: &WallDescriptionGgdParticleIon| item.z_ion, "z_ion"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdParticleIon| item.name.clone(), "name"),
            neutral_index: Accumulator::new(data, |item: &WallDescriptionGgdParticleIon| item.neutral_index, "neutral_index"),
            multiple_states_flag: Accumulator::new(data, |item: &WallDescriptionGgdParticleIon| item.multiple_states_flag, "multiple_states_flag"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdParticleIon> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdParticleIon
pub struct WallDescriptionGgdParticleIonSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdParticleIon],
}

impl<'a> WallDescriptionGgdParticleIonSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdParticleIon]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdParticleIon> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdParticleIon - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdParticleIonIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdParticleIon;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdParticleIon - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdParticleIonMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdParticleIon;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleIonMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleIon]) -> Self::Output {
        WallDescriptionGgdParticleIonSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdParticleNeutral View Types ---

/// View over multiple WallDescriptionGgdParticleNeutral with field accumulation
pub struct WallDescriptionGgdParticleNeutralSliceView<'a> {
    data: &'a [WallDescriptionGgdParticleNeutral],
    pub name: StringAccumulator<'a, WallDescriptionGgdParticleNeutral>,
    pub ion_index: Accumulator<'a, WallDescriptionGgdParticleNeutral, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdParticleNeutral, INT_0D>,
}

impl<'a> WallDescriptionGgdParticleNeutralSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdParticleNeutral]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdParticleNeutral| item.name.clone(), "name"),
            ion_index: Accumulator::new(data, |item: &WallDescriptionGgdParticleNeutral| item.ion_index, "ion_index"),
            multiple_states_flag: Accumulator::new(
                data,
                |item: &WallDescriptionGgdParticleNeutral| item.multiple_states_flag,
                "multiple_states_flag",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdParticleNeutral> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdParticleNeutral
pub struct WallDescriptionGgdParticleNeutralSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdParticleNeutral],
}

impl<'a> WallDescriptionGgdParticleNeutralSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdParticleNeutral> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdParticleNeutral - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdParticleNeutralIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdParticleNeutral;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdParticleNeutral - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdParticleNeutralMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdParticleNeutral;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdParticleNeutralMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdParticleNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdParticleNeutral]) -> Self::Output {
        WallDescriptionGgdParticleNeutralSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdRecyclingNeutralState View Types ---

/// View over `neutral_type` (IdentifierDynamicAos3) across multiple WallDescriptionGgdRecyclingNeutralState
pub struct WallDescriptionGgdRecyclingNeutralStateNeutralTypeView<'a> {
    pub name: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutralState>,
    pub index: Accumulator<'a, WallDescriptionGgdRecyclingNeutralState, INT_0D>,
    pub description: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutralState>,
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateNeutralTypeView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.neutral_type.name.clone(),
                "neutral_type.name",
            ),
            index: Accumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.neutral_type.index,
                "neutral_type.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.neutral_type.description.clone(),
                "neutral_type.description",
            ),
        }
    }
}

/// View over multiple WallDescriptionGgdRecyclingNeutralState with field accumulation
pub struct WallDescriptionGgdRecyclingNeutralStateSliceView<'a> {
    data: &'a [WallDescriptionGgdRecyclingNeutralState],
    pub name: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutralState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdRecyclingNeutralState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutralState>,
    pub neutral_type: WallDescriptionGgdRecyclingNeutralStateNeutralTypeView<'a>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutralState>,
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdRecyclingNeutralState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.vibrational_level,
                "vibrational_level",
            ),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            neutral_type: WallDescriptionGgdRecyclingNeutralStateNeutralTypeView::new(data),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutralState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdRecyclingNeutralState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdRecyclingNeutralState
pub struct WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdRecyclingNeutralState],
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdRecyclingNeutralState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdRecyclingNeutralState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdRecyclingNeutralStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdRecyclingNeutralState;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdRecyclingNeutralState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdRecyclingNeutralState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingNeutralStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutralState]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdRecyclingIonState View Types ---

/// View over multiple WallDescriptionGgdRecyclingIonState with field accumulation
pub struct WallDescriptionGgdRecyclingIonStateSliceView<'a> {
    data: &'a [WallDescriptionGgdRecyclingIonState],
    pub z_min: Accumulator<'a, WallDescriptionGgdRecyclingIonState, FLT_0D>,
    pub z_max: Accumulator<'a, WallDescriptionGgdRecyclingIonState, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdRecyclingIonState>,
    pub vibrational_level: Accumulator<'a, WallDescriptionGgdRecyclingIonState, FLT_0D>,
    pub vibrational_mode: StringAccumulator<'a, WallDescriptionGgdRecyclingIonState>,
    pub electron_configuration: StringAccumulator<'a, WallDescriptionGgdRecyclingIonState>,
}

impl<'a> WallDescriptionGgdRecyclingIonStateSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self {
        Self {
            data,
            z_min: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIonState| item.z_min, "z_min"),
            z_max: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIonState| item.z_max, "z_max"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdRecyclingIonState| item.name.clone(), "name"),
            vibrational_level: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIonState| item.vibrational_level, "vibrational_level"),
            vibrational_mode: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingIonState| item.vibrational_mode.clone(),
                "vibrational_mode",
            ),
            electron_configuration: StringAccumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingIonState| item.electron_configuration.clone(),
                "electron_configuration",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdRecyclingIonState> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdRecyclingIonState
pub struct WallDescriptionGgdRecyclingIonStateSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdRecyclingIonState],
}

impl<'a> WallDescriptionGgdRecyclingIonStateSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdRecyclingIonState> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdRecyclingIonState - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdRecyclingIonStateIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdRecyclingIonState;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingIonStateSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdRecyclingIonState - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdRecyclingIonStateMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdRecyclingIonState;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonStateMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingIonStateSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIonState]) -> Self::Output {
        WallDescriptionGgdRecyclingIonStateSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdRecyclingIon View Types ---

/// View over multiple WallDescriptionGgdRecyclingIon with field accumulation
pub struct WallDescriptionGgdRecyclingIonSliceView<'a> {
    data: &'a [WallDescriptionGgdRecyclingIon],
    pub z_ion: Accumulator<'a, WallDescriptionGgdRecyclingIon, FLT_0D>,
    pub name: StringAccumulator<'a, WallDescriptionGgdRecyclingIon>,
    pub neutral_index: Accumulator<'a, WallDescriptionGgdRecyclingIon, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdRecyclingIon, INT_0D>,
}

impl<'a> WallDescriptionGgdRecyclingIonSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdRecyclingIon]) -> Self {
        Self {
            data,
            z_ion: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIon| item.z_ion, "z_ion"),
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdRecyclingIon| item.name.clone(), "name"),
            neutral_index: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIon| item.neutral_index, "neutral_index"),
            multiple_states_flag: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingIon| item.multiple_states_flag, "multiple_states_flag"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdRecyclingIon> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdRecyclingIon
pub struct WallDescriptionGgdRecyclingIonSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdRecyclingIon],
}

impl<'a> WallDescriptionGgdRecyclingIonSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdRecyclingIon> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdRecyclingIon - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdRecyclingIonIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdRecyclingIon;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingIonSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdRecyclingIon - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdRecyclingIonMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdRecyclingIon;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingIonMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingIonSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingIon]) -> Self::Output {
        WallDescriptionGgdRecyclingIonSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdRecyclingNeutral View Types ---

/// View over multiple WallDescriptionGgdRecyclingNeutral with field accumulation
pub struct WallDescriptionGgdRecyclingNeutralSliceView<'a> {
    data: &'a [WallDescriptionGgdRecyclingNeutral],
    pub name: StringAccumulator<'a, WallDescriptionGgdRecyclingNeutral>,
    pub ion_index: Accumulator<'a, WallDescriptionGgdRecyclingNeutral, INT_0D>,
    pub multiple_states_flag: Accumulator<'a, WallDescriptionGgdRecyclingNeutral, INT_0D>,
}

impl<'a> WallDescriptionGgdRecyclingNeutralSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &WallDescriptionGgdRecyclingNeutral| item.name.clone(), "name"),
            ion_index: Accumulator::new(data, |item: &WallDescriptionGgdRecyclingNeutral| item.ion_index, "ion_index"),
            multiple_states_flag: Accumulator::new(
                data,
                |item: &WallDescriptionGgdRecyclingNeutral| item.multiple_states_flag,
                "multiple_states_flag",
            ),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdRecyclingNeutral> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdRecyclingNeutral
pub struct WallDescriptionGgdRecyclingNeutralSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdRecyclingNeutral],
}

impl<'a> WallDescriptionGgdRecyclingNeutralSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdRecyclingNeutral> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdRecyclingNeutral - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdRecyclingNeutralIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdRecyclingNeutral;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingNeutralSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdRecyclingNeutral - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdRecyclingNeutralMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output;
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdRecyclingNeutral;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdRecyclingNeutralMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdRecyclingNeutralSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdRecyclingNeutral]) -> Self::Output {
        WallDescriptionGgdRecyclingNeutralSliceViewMut::new(data)
    }
}

// --- GenericGridVectorComponentsRphiz View Types ---

/// View over multiple GenericGridVectorComponentsRphiz with field accumulation
pub struct GenericGridVectorComponentsRphizSliceView<'a> {
    data: &'a [GenericGridVectorComponentsRphiz],
    pub grid_index: Accumulator<'a, GenericGridVectorComponentsRphiz, INT_0D>,
    pub grid_subset_index: Accumulator<'a, GenericGridVectorComponentsRphiz, INT_0D>,
}

impl<'a> GenericGridVectorComponentsRphizSliceView<'a> {
    pub fn new(data: &'a [GenericGridVectorComponentsRphiz]) -> Self {
        Self {
            data,
            grid_index: Accumulator::new(data, |item: &GenericGridVectorComponentsRphiz| item.grid_index, "grid_index"),
            grid_subset_index: Accumulator::new(data, |item: &GenericGridVectorComponentsRphiz| item.grid_subset_index, "grid_subset_index"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridVectorComponentsRphiz> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridVectorComponentsRphiz
pub struct GenericGridVectorComponentsRphizSliceViewMut<'a> {
    data: &'a mut [GenericGridVectorComponentsRphiz],
}

impl<'a> GenericGridVectorComponentsRphizSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridVectorComponentsRphiz> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridVectorComponentsRphiz - enables .field(0) and .field(0..2) syntax
pub trait GenericGridVectorComponentsRphizIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output;
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for usize {
    type Output = &'a GenericGridVectorComponentsRphiz;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridVectorComponentsRphizSliceView<'a>;
    fn get(self, data: &'a [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridVectorComponentsRphiz - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridVectorComponentsRphizMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output;
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for usize {
    type Output = &'a mut GenericGridVectorComponentsRphiz;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorComponentsRphizMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridVectorComponentsRphizSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVectorComponentsRphiz]) -> Self::Output {
        GenericGridVectorComponentsRphizSliceViewMut::new(data)
    }
}

// --- GenericGridIdentifier View Types ---

/// View over `identifiers` (IdentifierDynamicAos31d) across multiple GenericGridIdentifier
pub struct GenericGridIdentifierIdentifiersView<'a> {
    _phantom: std::marker::PhantomData<&'a GenericGridIdentifier>,
}

impl<'a> GenericGridIdentifierIdentifiersView<'a> {
    pub fn new(_data: &'a [GenericGridIdentifier]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over multiple GenericGridIdentifier with field accumulation
pub struct GenericGridIdentifierSliceView<'a> {
    data: &'a [GenericGridIdentifier],
    pub grid_index: Accumulator<'a, GenericGridIdentifier, INT_0D>,
    pub grid_subset_index: Accumulator<'a, GenericGridIdentifier, INT_0D>,
    pub identifiers: GenericGridIdentifierIdentifiersView<'a>,
}

impl<'a> GenericGridIdentifierSliceView<'a> {
    pub fn new(data: &'a [GenericGridIdentifier]) -> Self {
        Self {
            data,
            grid_index: Accumulator::new(data, |item: &GenericGridIdentifier| item.grid_index, "grid_index"),
            grid_subset_index: Accumulator::new(data, |item: &GenericGridIdentifier| item.grid_subset_index, "grid_subset_index"),
            identifiers: GenericGridIdentifierIdentifiersView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridIdentifier> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridIdentifier
pub struct GenericGridIdentifierSliceViewMut<'a> {
    data: &'a mut [GenericGridIdentifier],
}

impl<'a> GenericGridIdentifierSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridIdentifier]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridIdentifier> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridIdentifier - enables .field(0) and .field(0..2) syntax
pub trait GenericGridIdentifierIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output;
}

impl<'a> GenericGridIdentifierIndex<'a> for usize {
    type Output = &'a GenericGridIdentifier;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridIdentifierSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridIdentifier - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridIdentifierMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output;
}

impl<'a> GenericGridIdentifierMutIndex<'a> for usize {
    type Output = &'a mut GenericGridIdentifier;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridIdentifierSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifier]) -> Self::Output {
        GenericGridIdentifierSliceViewMut::new(data)
    }
}

// --- GenericGridVector View Types ---

/// View over multiple GenericGridVector with field accumulation
pub struct GenericGridVectorSliceView<'a> {
    data: &'a [GenericGridVector],
    pub grid_index: Accumulator<'a, GenericGridVector, INT_0D>,
    pub grid_subset_index: Accumulator<'a, GenericGridVector, INT_0D>,
}

impl<'a> GenericGridVectorSliceView<'a> {
    pub fn new(data: &'a [GenericGridVector]) -> Self {
        Self {
            data,
            grid_index: Accumulator::new(data, |item: &GenericGridVector| item.grid_index, "grid_index"),
            grid_subset_index: Accumulator::new(data, |item: &GenericGridVector| item.grid_subset_index, "grid_subset_index"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridVector> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridVector
pub struct GenericGridVectorSliceViewMut<'a> {
    data: &'a mut [GenericGridVector],
}

impl<'a> GenericGridVectorSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridVector]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridVector> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridVector - enables .field(0) and .field(0..2) syntax
pub trait GenericGridVectorIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output;
}

impl<'a> GenericGridVectorIndex<'a> for usize {
    type Output = &'a GenericGridVector;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(&data[self])
    }
}

impl<'a> GenericGridVectorIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridVectorSliceView<'a>;
    fn get(self, data: &'a [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridVector - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridVectorMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output;
}

impl<'a> GenericGridVectorMutIndex<'a> for usize {
    type Output = &'a mut GenericGridVector;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridVectorMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridVectorSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridVector]) -> Self::Output {
        GenericGridVectorSliceViewMut::new(data)
    }
}

// --- GenericGridIdentifierSingle View Types ---

/// View over `identifier` (IdentifierDynamicAos3) across multiple GenericGridIdentifierSingle
pub struct GenericGridIdentifierSingleIdentifierView<'a> {
    pub name: StringAccumulator<'a, GenericGridIdentifierSingle>,
    pub index: Accumulator<'a, GenericGridIdentifierSingle, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridIdentifierSingle>,
}

impl<'a> GenericGridIdentifierSingleIdentifierView<'a> {
    pub fn new(data: &'a [GenericGridIdentifierSingle]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridIdentifierSingle| item.identifier.name.clone(), "identifier.name"),
            index: Accumulator::new(data, |item: &GenericGridIdentifierSingle| item.identifier.index, "identifier.index"),
            description: StringAccumulator::new(
                data,
                |item: &GenericGridIdentifierSingle| item.identifier.description.clone(),
                "identifier.description",
            ),
        }
    }
}

/// View over multiple GenericGridIdentifierSingle with field accumulation
pub struct GenericGridIdentifierSingleSliceView<'a> {
    data: &'a [GenericGridIdentifierSingle],
    pub grid_index: Accumulator<'a, GenericGridIdentifierSingle, INT_0D>,
    pub grid_subset_index: Accumulator<'a, GenericGridIdentifierSingle, INT_0D>,
    pub identifier: GenericGridIdentifierSingleIdentifierView<'a>,
}

impl<'a> GenericGridIdentifierSingleSliceView<'a> {
    pub fn new(data: &'a [GenericGridIdentifierSingle]) -> Self {
        Self {
            data,
            grid_index: Accumulator::new(data, |item: &GenericGridIdentifierSingle| item.grid_index, "grid_index"),
            grid_subset_index: Accumulator::new(data, |item: &GenericGridIdentifierSingle| item.grid_subset_index, "grid_subset_index"),
            identifier: GenericGridIdentifierSingleIdentifierView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridIdentifierSingle> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridIdentifierSingle
pub struct GenericGridIdentifierSingleSliceViewMut<'a> {
    data: &'a mut [GenericGridIdentifierSingle],
}

impl<'a> GenericGridIdentifierSingleSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridIdentifierSingle]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridIdentifierSingle> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridIdentifierSingle - enables .field(0) and .field(0..2) syntax
pub trait GenericGridIdentifierSingleIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output;
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for usize {
    type Output = &'a GenericGridIdentifierSingle;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(&data[self])
    }
}

impl<'a> GenericGridIdentifierSingleIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridIdentifierSingleSliceView<'a>;
    fn get(self, data: &'a [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridIdentifierSingle - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridIdentifierSingleMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output;
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for usize {
    type Output = &'a mut GenericGridIdentifierSingle;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridIdentifierSingleMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridIdentifierSingleSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridIdentifierSingle]) -> Self::Output {
        GenericGridIdentifierSingleSliceViewMut::new(data)
    }
}

// --- GenericGridAos3Root View Types ---

/// View over `identifier` (IdentifierDynamicAos3) across multiple GenericGridAos3Root
pub struct GenericGridAos3RootIdentifierView<'a> {
    pub name: StringAccumulator<'a, GenericGridAos3Root>,
    pub index: Accumulator<'a, GenericGridAos3Root, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridAos3Root>,
}

impl<'a> GenericGridAos3RootIdentifierView<'a> {
    pub fn new(data: &'a [GenericGridAos3Root]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridAos3Root| item.identifier.name.clone(), "identifier.name"),
            index: Accumulator::new(data, |item: &GenericGridAos3Root| item.identifier.index, "identifier.index"),
            description: StringAccumulator::new(data, |item: &GenericGridAos3Root| item.identifier.description.clone(), "identifier.description"),
        }
    }
}

/// View over multiple GenericGridAos3Root with field accumulation
pub struct GenericGridAos3RootSliceView<'a> {
    data: &'a [GenericGridAos3Root],
    pub identifier: GenericGridAos3RootIdentifierView<'a>,
    pub path: StringAccumulator<'a, GenericGridAos3Root>,
    pub time: Accumulator<'a, GenericGridAos3Root, FLT_0D>,
}

impl<'a> GenericGridAos3RootSliceView<'a> {
    pub fn new(data: &'a [GenericGridAos3Root]) -> Self {
        Self {
            data,
            identifier: GenericGridAos3RootIdentifierView::new(data),
            path: StringAccumulator::new(data, |item: &GenericGridAos3Root| item.path.clone(), "path"),
            time: Accumulator::new(data, |item: &GenericGridAos3Root| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridAos3Root> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridAos3Root
pub struct GenericGridAos3RootSliceViewMut<'a> {
    data: &'a mut [GenericGridAos3Root],
}

impl<'a> GenericGridAos3RootSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridAos3Root]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridAos3Root> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridAos3Root - enables .field(0) and .field(0..2) syntax
pub trait GenericGridAos3RootIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output;
}

impl<'a> GenericGridAos3RootIndex<'a> for usize {
    type Output = &'a GenericGridAos3Root;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(&data[self])
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(&data[self])
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(&data[self])
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(&data[self])
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(&data[self])
    }
}

impl<'a> GenericGridAos3RootIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridAos3RootSliceView<'a>;
    fn get(self, data: &'a [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridAos3Root - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridAos3RootMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output;
}

impl<'a> GenericGridAos3RootMutIndex<'a> for usize {
    type Output = &'a mut GenericGridAos3Root;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridAos3RootMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridAos3RootSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridAos3Root]) -> Self::Output {
        GenericGridAos3RootSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdMaterial View Types ---

/// View over multiple WallDescriptionGgdMaterial with field accumulation
pub struct WallDescriptionGgdMaterialSliceView<'a> {
    data: &'a [WallDescriptionGgdMaterial],
    pub time: Accumulator<'a, WallDescriptionGgdMaterial, FLT_0D>,
}

impl<'a> WallDescriptionGgdMaterialSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdMaterial]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &WallDescriptionGgdMaterial| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdMaterial> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdMaterial
pub struct WallDescriptionGgdMaterialSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdMaterial],
}

impl<'a> WallDescriptionGgdMaterialSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdMaterial]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdMaterial> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdMaterial - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdMaterialIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output;
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdMaterial;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdMaterialSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdMaterial - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdMaterialMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output;
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdMaterial;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMaterialMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdMaterialSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdMaterial]) -> Self::Output {
        WallDescriptionGgdMaterialSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdComponent View Types ---

/// View over multiple WallDescriptionGgdComponent with field accumulation
pub struct WallDescriptionGgdComponentSliceView<'a> {
    data: &'a [WallDescriptionGgdComponent],
    pub time: Accumulator<'a, WallDescriptionGgdComponent, FLT_0D>,
}

impl<'a> WallDescriptionGgdComponentSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdComponent]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &WallDescriptionGgdComponent| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdComponent> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdComponent
pub struct WallDescriptionGgdComponentSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdComponent],
}

impl<'a> WallDescriptionGgdComponentSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdComponent]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdComponent> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdComponent - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdComponentIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output;
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdComponent;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdComponentIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdComponentSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdComponent - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdComponentMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output;
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdComponent;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdComponentMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdComponentSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdComponent]) -> Self::Output {
        WallDescriptionGgdComponentSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdThickness View Types ---

/// View over multiple WallDescriptionGgdThickness with field accumulation
pub struct WallDescriptionGgdThicknessSliceView<'a> {
    data: &'a [WallDescriptionGgdThickness],
    pub time: Accumulator<'a, WallDescriptionGgdThickness, FLT_0D>,
}

impl<'a> WallDescriptionGgdThicknessSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdThickness]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &WallDescriptionGgdThickness| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdThickness> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdThickness
pub struct WallDescriptionGgdThicknessSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdThickness],
}

impl<'a> WallDescriptionGgdThicknessSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdThickness]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdThickness> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdThickness - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdThicknessIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output;
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdThickness;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdThicknessSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdThickness - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdThicknessMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output;
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdThickness;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdThicknessMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdThicknessSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdThickness]) -> Self::Output {
        WallDescriptionGgdThicknessSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdBrdf View Types ---

/// View over multiple WallDescriptionGgdBrdf with field accumulation
pub struct WallDescriptionGgdBrdfSliceView<'a> {
    data: &'a [WallDescriptionGgdBrdf],
    pub time: Accumulator<'a, WallDescriptionGgdBrdf, FLT_0D>,
}

impl<'a> WallDescriptionGgdBrdfSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdBrdf]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &WallDescriptionGgdBrdf| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdBrdf> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdBrdf
pub struct WallDescriptionGgdBrdfSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdBrdf],
}

impl<'a> WallDescriptionGgdBrdfSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdBrdf]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdBrdf> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdBrdf - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdBrdfIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output;
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdBrdf;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdBrdfSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdBrdf - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdBrdfMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output;
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdBrdf;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdBrdfMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdBrdfSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdBrdf]) -> Self::Output {
        WallDescriptionGgdBrdfSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgdGgd View Types ---

/// View over `recycling` (WallDescriptionGgdRecycling) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdRecyclingView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdRecyclingView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `particle_fluxes.electrons` (WallDescriptionGgdParticleEl) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdParticleFluxesElectronsView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdParticleFluxesElectronsView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `particle_fluxes` (WallDescriptionGgdParticle) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdParticleFluxesView<'a> {
    pub electrons: WallDescriptionGgdGgdParticleFluxesElectronsView<'a>,
}

impl<'a> WallDescriptionGgdGgdParticleFluxesView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            electrons: WallDescriptionGgdGgdParticleFluxesElectronsView::new(data),
        }
    }
}

/// View over `energy_fluxes.radiation` (WallDescriptionGgdEnergySimple) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesRadiationView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesRadiationView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `energy_fluxes.current` (WallDescriptionGgdEnergySimple) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesCurrentView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesCurrentView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `energy_fluxes.recombination` (WallDescriptionGgdRecombination) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesRecombinationView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesRecombinationView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `energy_fluxes.kinetic.electrons` (WallDescriptionGgdEnergySimple) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesKineticElectronsView<'a> {
    _phantom: std::marker::PhantomData<&'a WallDescriptionGgdGgd>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesKineticElectronsView<'a> {
    pub fn new(_data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `energy_fluxes.kinetic` (WallDescriptionGgdKinetic) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesKineticView<'a> {
    pub electrons: WallDescriptionGgdGgdEnergyFluxesKineticElectronsView<'a>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesKineticView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            electrons: WallDescriptionGgdGgdEnergyFluxesKineticElectronsView::new(data),
        }
    }
}

/// View over `energy_fluxes` (WallDescriptionGgdEnergy) across multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdEnergyFluxesView<'a> {
    pub radiation: WallDescriptionGgdGgdEnergyFluxesRadiationView<'a>,
    pub current: WallDescriptionGgdGgdEnergyFluxesCurrentView<'a>,
    pub recombination: WallDescriptionGgdGgdEnergyFluxesRecombinationView<'a>,
    pub kinetic: WallDescriptionGgdGgdEnergyFluxesKineticView<'a>,
}

impl<'a> WallDescriptionGgdGgdEnergyFluxesView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            radiation: WallDescriptionGgdGgdEnergyFluxesRadiationView::new(data),
            current: WallDescriptionGgdGgdEnergyFluxesCurrentView::new(data),
            recombination: WallDescriptionGgdGgdEnergyFluxesRecombinationView::new(data),
            kinetic: WallDescriptionGgdGgdEnergyFluxesKineticView::new(data),
        }
    }
}

/// View over multiple WallDescriptionGgdGgd with field accumulation
pub struct WallDescriptionGgdGgdSliceView<'a> {
    data: &'a [WallDescriptionGgdGgd],
    pub recycling: WallDescriptionGgdGgdRecyclingView<'a>,
    pub particle_fluxes: WallDescriptionGgdGgdParticleFluxesView<'a>,
    pub energy_fluxes: WallDescriptionGgdGgdEnergyFluxesView<'a>,
    pub time: Accumulator<'a, WallDescriptionGgdGgd, FLT_0D>,
}

impl<'a> WallDescriptionGgdGgdSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgdGgd]) -> Self {
        Self {
            data,
            recycling: WallDescriptionGgdGgdRecyclingView::new(data),
            particle_fluxes: WallDescriptionGgdGgdParticleFluxesView::new(data),
            energy_fluxes: WallDescriptionGgdGgdEnergyFluxesView::new(data),
            time: Accumulator::new(data, |item: &WallDescriptionGgdGgd| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgdGgd> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgdGgd
pub struct WallDescriptionGgdGgdSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgdGgd],
}

impl<'a> WallDescriptionGgdGgdSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgdGgd]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgdGgd> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgdGgd - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdGgdIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output;
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for usize {
    type Output = &'a WallDescriptionGgdGgd;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdGgdIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgdGgd - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdGgdMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output;
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgdGgd;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdGgdMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgdGgd]) -> Self::Output {
        WallDescriptionGgdGgdSliceViewMut::new(data)
    }
}

// --- Vessel2dUnit View Types ---

/// View over `annular.outline_inner` (Rz1dStatic) across multiple Vessel2dUnit
pub struct Vessel2dUnitAnnularOutlineInnerView<'a> {
    _phantom: std::marker::PhantomData<&'a Vessel2dUnit>,
}

impl<'a> Vessel2dUnitAnnularOutlineInnerView<'a> {
    pub fn new(_data: &'a [Vessel2dUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `annular.outline_outer` (Rz1dStatic) across multiple Vessel2dUnit
pub struct Vessel2dUnitAnnularOutlineOuterView<'a> {
    _phantom: std::marker::PhantomData<&'a Vessel2dUnit>,
}

impl<'a> Vessel2dUnitAnnularOutlineOuterView<'a> {
    pub fn new(_data: &'a [Vessel2dUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `annular.centreline` (Rz1dStatic) across multiple Vessel2dUnit
pub struct Vessel2dUnitAnnularCentrelineView<'a> {
    _phantom: std::marker::PhantomData<&'a Vessel2dUnit>,
}

impl<'a> Vessel2dUnitAnnularCentrelineView<'a> {
    pub fn new(_data: &'a [Vessel2dUnit]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `annular` (Vessel2dAnnular) across multiple Vessel2dUnit
pub struct Vessel2dUnitAnnularView<'a> {
    pub outline_inner: Vessel2dUnitAnnularOutlineInnerView<'a>,
    pub outline_outer: Vessel2dUnitAnnularOutlineOuterView<'a>,
    pub centreline: Vessel2dUnitAnnularCentrelineView<'a>,
    pub resistivity: Accumulator<'a, Vessel2dUnit, FLT_0D>,
}

impl<'a> Vessel2dUnitAnnularView<'a> {
    pub fn new(data: &'a [Vessel2dUnit]) -> Self {
        Self {
            outline_inner: Vessel2dUnitAnnularOutlineInnerView::new(data),
            outline_outer: Vessel2dUnitAnnularOutlineOuterView::new(data),
            centreline: Vessel2dUnitAnnularCentrelineView::new(data),
            resistivity: Accumulator::new(data, |item: &Vessel2dUnit| item.annular.resistivity, "annular.resistivity"),
        }
    }
}

/// View over multiple Vessel2dUnit with field accumulation
pub struct Vessel2dUnitSliceView<'a> {
    data: &'a [Vessel2dUnit],
    pub name: StringAccumulator<'a, Vessel2dUnit>,
    pub description: StringAccumulator<'a, Vessel2dUnit>,
    pub annular: Vessel2dUnitAnnularView<'a>,
}

impl<'a> Vessel2dUnitSliceView<'a> {
    pub fn new(data: &'a [Vessel2dUnit]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &Vessel2dUnit| item.name.clone(), "name"),
            description: StringAccumulator::new(data, |item: &Vessel2dUnit| item.description.clone(), "description"),
            annular: Vessel2dUnitAnnularView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vessel2dUnit> {
        self.data.iter()
    }
}

/// Mutable view over multiple Vessel2dUnit
pub struct Vessel2dUnitSliceViewMut<'a> {
    data: &'a mut [Vessel2dUnit],
}

impl<'a> Vessel2dUnitSliceViewMut<'a> {
    pub fn new(data: &'a mut [Vessel2dUnit]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vessel2dUnit> {
        self.data.iter_mut()
    }
}

/// Index trait for Vessel2dUnit - enables .field(0) and .field(0..2) syntax
pub trait Vessel2dUnitIndex<'a> {
    type Output;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output;
}

impl<'a> Vessel2dUnitIndex<'a> for usize {
    type Output = &'a Vessel2dUnit;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::Range<usize> {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dUnitIndex<'a> for std::ops::RangeFull {
    type Output = Vessel2dUnitSliceView<'a>;
    fn get(self, data: &'a [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceView::new(data)
    }
}

/// Mutable index trait for Vessel2dUnit - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Vessel2dUnitMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output;
}

impl<'a> Vessel2dUnitMutIndex<'a> for usize {
    type Output = &'a mut Vessel2dUnit;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::Range<usize> {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dUnitMutIndex<'a> for std::ops::RangeFull {
    type Output = Vessel2dUnitSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dUnit]) -> Self::Output {
        Vessel2dUnitSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicSpace View Types ---

/// View over `identifier` (IdentifierDynamicAos3) across multiple GenericGridDynamicSpace
pub struct GenericGridDynamicSpaceIdentifierView<'a> {
    pub name: StringAccumulator<'a, GenericGridDynamicSpace>,
    pub index: Accumulator<'a, GenericGridDynamicSpace, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridDynamicSpace>,
}

impl<'a> GenericGridDynamicSpaceIdentifierView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpace]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridDynamicSpace| item.identifier.name.clone(), "identifier.name"),
            index: Accumulator::new(data, |item: &GenericGridDynamicSpace| item.identifier.index, "identifier.index"),
            description: StringAccumulator::new(
                data,
                |item: &GenericGridDynamicSpace| item.identifier.description.clone(),
                "identifier.description",
            ),
        }
    }
}

/// View over `geometry_type` (IdentifierDynamicAos3) across multiple GenericGridDynamicSpace
pub struct GenericGridDynamicSpaceGeometryTypeView<'a> {
    pub name: StringAccumulator<'a, GenericGridDynamicSpace>,
    pub index: Accumulator<'a, GenericGridDynamicSpace, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridDynamicSpace>,
}

impl<'a> GenericGridDynamicSpaceGeometryTypeView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpace]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridDynamicSpace| item.geometry_type.name.clone(), "geometry_type.name"),
            index: Accumulator::new(data, |item: &GenericGridDynamicSpace| item.geometry_type.index, "geometry_type.index"),
            description: StringAccumulator::new(
                data,
                |item: &GenericGridDynamicSpace| item.geometry_type.description.clone(),
                "geometry_type.description",
            ),
        }
    }
}

/// View over multiple GenericGridDynamicSpace with field accumulation
pub struct GenericGridDynamicSpaceSliceView<'a> {
    data: &'a [GenericGridDynamicSpace],
    pub identifier: GenericGridDynamicSpaceIdentifierView<'a>,
    pub geometry_type: GenericGridDynamicSpaceGeometryTypeView<'a>,
}

impl<'a> GenericGridDynamicSpaceSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpace]) -> Self {
        Self {
            data,
            identifier: GenericGridDynamicSpaceIdentifierView::new(data),
            geometry_type: GenericGridDynamicSpaceGeometryTypeView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicSpace> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicSpace
pub struct GenericGridDynamicSpaceSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicSpace],
}

impl<'a> GenericGridDynamicSpaceSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicSpace]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicSpace> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicSpace - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicSpaceIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for usize {
    type Output = &'a GenericGridDynamicSpace;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicSpace - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicSpaceMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicSpace;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpace]) -> Self::Output {
        GenericGridDynamicSpaceSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicGridSubset View Types ---

/// View over `identifier` (IdentifierDynamicAos3) across multiple GenericGridDynamicGridSubset
pub struct GenericGridDynamicGridSubsetIdentifierView<'a> {
    pub name: StringAccumulator<'a, GenericGridDynamicGridSubset>,
    pub index: Accumulator<'a, GenericGridDynamicGridSubset, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridDynamicGridSubset>,
}

impl<'a> GenericGridDynamicGridSubsetIdentifierView<'a> {
    pub fn new(data: &'a [GenericGridDynamicGridSubset]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridDynamicGridSubset| item.identifier.name.clone(), "identifier.name"),
            index: Accumulator::new(data, |item: &GenericGridDynamicGridSubset| item.identifier.index, "identifier.index"),
            description: StringAccumulator::new(
                data,
                |item: &GenericGridDynamicGridSubset| item.identifier.description.clone(),
                "identifier.description",
            ),
        }
    }
}

/// View over `metric` (GenericGridDynamicGridSubsetMetric) across multiple GenericGridDynamicGridSubset
pub struct GenericGridDynamicGridSubsetMetricView<'a> {
    _phantom: std::marker::PhantomData<&'a GenericGridDynamicGridSubset>,
}

impl<'a> GenericGridDynamicGridSubsetMetricView<'a> {
    pub fn new(_data: &'a [GenericGridDynamicGridSubset]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over multiple GenericGridDynamicGridSubset with field accumulation
pub struct GenericGridDynamicGridSubsetSliceView<'a> {
    data: &'a [GenericGridDynamicGridSubset],
    pub identifier: GenericGridDynamicGridSubsetIdentifierView<'a>,
    pub dimension: Accumulator<'a, GenericGridDynamicGridSubset, INT_0D>,
    pub metric: GenericGridDynamicGridSubsetMetricView<'a>,
}

impl<'a> GenericGridDynamicGridSubsetSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicGridSubset]) -> Self {
        Self {
            data,
            identifier: GenericGridDynamicGridSubsetIdentifierView::new(data),
            dimension: Accumulator::new(data, |item: &GenericGridDynamicGridSubset| item.dimension, "dimension"),
            metric: GenericGridDynamicGridSubsetMetricView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicGridSubset> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicGridSubset
pub struct GenericGridDynamicGridSubsetSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicGridSubset],
}

impl<'a> GenericGridDynamicGridSubsetSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicGridSubset]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicGridSubset> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicGridSubset - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicGridSubsetIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for usize {
    type Output = &'a GenericGridDynamicGridSubset;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicGridSubset - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicGridSubsetMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicGridSubset;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubset]) -> Self::Output {
        GenericGridDynamicGridSubsetSliceViewMut::new(data)
    }
}

// --- Library View Types ---

/// View over multiple Library with field accumulation
pub struct LibrarySliceView<'a> {
    data: &'a [Library],
    pub name: StringAccumulator<'a, Library>,
    pub description: StringAccumulator<'a, Library>,
    pub commit: StringAccumulator<'a, Library>,
    pub version: StringAccumulator<'a, Library>,
    pub repository: StringAccumulator<'a, Library>,
    pub parameters: StringAccumulator<'a, Library>,
}

impl<'a> LibrarySliceView<'a> {
    pub fn new(data: &'a [Library]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &Library| item.name.clone(), "name"),
            description: StringAccumulator::new(data, |item: &Library| item.description.clone(), "description"),
            commit: StringAccumulator::new(data, |item: &Library| item.commit.clone(), "commit"),
            version: StringAccumulator::new(data, |item: &Library| item.version.clone(), "version"),
            repository: StringAccumulator::new(data, |item: &Library| item.repository.clone(), "repository"),
            parameters: StringAccumulator::new(data, |item: &Library| item.parameters.clone(), "parameters"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Library> {
        self.data.iter()
    }
}

/// Mutable view over multiple Library
pub struct LibrarySliceViewMut<'a> {
    data: &'a mut [Library],
}

impl<'a> LibrarySliceViewMut<'a> {
    pub fn new(data: &'a mut [Library]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Library> {
        self.data.iter_mut()
    }
}

/// Index trait for Library - enables .field(0) and .field(0..2) syntax
pub trait LibraryIndex<'a> {
    type Output;
    fn get(self, data: &'a [Library]) -> Self::Output;
}

impl<'a> LibraryIndex<'a> for usize {
    type Output = &'a Library;
    fn get(self, data: &'a [Library]) -> Self::Output {
        &data[self]
    }
}

impl<'a> LibraryIndex<'a> for std::ops::Range<usize> {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(&data[self])
    }
}

impl<'a> LibraryIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(&data[self])
    }
}

impl<'a> LibraryIndex<'a> for std::ops::RangeTo<usize> {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(&data[self])
    }
}

impl<'a> LibraryIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(&data[self])
    }
}

impl<'a> LibraryIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(&data[self])
    }
}

impl<'a> LibraryIndex<'a> for std::ops::RangeFull {
    type Output = LibrarySliceView<'a>;
    fn get(self, data: &'a [Library]) -> Self::Output {
        LibrarySliceView::new(data)
    }
}

/// Mutable index trait for Library - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait LibraryMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output;
}

impl<'a> LibraryMutIndex<'a> for usize {
    type Output = &'a mut Library;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::Range<usize> {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> LibraryMutIndex<'a> for std::ops::RangeFull {
    type Output = LibrarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Library]) -> Self::Output {
        LibrarySliceViewMut::new(data)
    }
}

// --- Vessel2dElement View Types ---

/// View over `outline` (Rz1dStatic) across multiple Vessel2dElement
pub struct Vessel2dElementOutlineView<'a> {
    _phantom: std::marker::PhantomData<&'a Vessel2dElement>,
}

impl<'a> Vessel2dElementOutlineView<'a> {
    pub fn new(_data: &'a [Vessel2dElement]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over multiple Vessel2dElement with field accumulation
pub struct Vessel2dElementSliceView<'a> {
    data: &'a [Vessel2dElement],
    pub name: StringAccumulator<'a, Vessel2dElement>,
    pub outline: Vessel2dElementOutlineView<'a>,
    pub midplane_thickness: Accumulator<'a, Vessel2dElement, FLT_0D>,
    pub resistivity: Accumulator<'a, Vessel2dElement, FLT_0D>,
    pub resistance: Accumulator<'a, Vessel2dElement, FLT_0D>,
}

impl<'a> Vessel2dElementSliceView<'a> {
    pub fn new(data: &'a [Vessel2dElement]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &Vessel2dElement| item.name.clone(), "name"),
            outline: Vessel2dElementOutlineView::new(data),
            midplane_thickness: Accumulator::new(data, |item: &Vessel2dElement| item.midplane_thickness, "midplane_thickness"),
            resistivity: Accumulator::new(data, |item: &Vessel2dElement| item.resistivity, "resistivity"),
            resistance: Accumulator::new(data, |item: &Vessel2dElement| item.resistance, "resistance"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vessel2dElement> {
        self.data.iter()
    }
}

/// Mutable view over multiple Vessel2dElement
pub struct Vessel2dElementSliceViewMut<'a> {
    data: &'a mut [Vessel2dElement],
}

impl<'a> Vessel2dElementSliceViewMut<'a> {
    pub fn new(data: &'a mut [Vessel2dElement]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vessel2dElement> {
        self.data.iter_mut()
    }
}

/// Index trait for Vessel2dElement - enables .field(0) and .field(0..2) syntax
pub trait Vessel2dElementIndex<'a> {
    type Output;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output;
}

impl<'a> Vessel2dElementIndex<'a> for usize {
    type Output = &'a Vessel2dElement;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::Range<usize> {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(&data[self])
    }
}

impl<'a> Vessel2dElementIndex<'a> for std::ops::RangeFull {
    type Output = Vessel2dElementSliceView<'a>;
    fn get(self, data: &'a [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceView::new(data)
    }
}

/// Mutable index trait for Vessel2dElement - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Vessel2dElementMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output;
}

impl<'a> Vessel2dElementMutIndex<'a> for usize {
    type Output = &'a mut Vessel2dElement;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::Range<usize> {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Vessel2dElementMutIndex<'a> for std::ops::RangeFull {
    type Output = Vessel2dElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Vessel2dElement]) -> Self::Output {
        Vessel2dElementSliceViewMut::new(data)
    }
}

// --- IdentifierStatic View Types ---

/// View over multiple IdentifierStatic with field accumulation
pub struct IdentifierStaticSliceView<'a> {
    data: &'a [IdentifierStatic],
    pub name: StringAccumulator<'a, IdentifierStatic>,
    pub index: Accumulator<'a, IdentifierStatic, INT_0D>,
    pub description: StringAccumulator<'a, IdentifierStatic>,
}

impl<'a> IdentifierStaticSliceView<'a> {
    pub fn new(data: &'a [IdentifierStatic]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &IdentifierStatic| item.name.clone(), "name"),
            index: Accumulator::new(data, |item: &IdentifierStatic| item.index, "index"),
            description: StringAccumulator::new(data, |item: &IdentifierStatic| item.description.clone(), "description"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &IdentifierStatic> {
        self.data.iter()
    }
}

/// Mutable view over multiple IdentifierStatic
pub struct IdentifierStaticSliceViewMut<'a> {
    data: &'a mut [IdentifierStatic],
}

impl<'a> IdentifierStaticSliceViewMut<'a> {
    pub fn new(data: &'a mut [IdentifierStatic]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut IdentifierStatic> {
        self.data.iter_mut()
    }
}

/// Index trait for IdentifierStatic - enables .field(0) and .field(0..2) syntax
pub trait IdentifierStaticIndex<'a> {
    type Output;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output;
}

impl<'a> IdentifierStaticIndex<'a> for usize {
    type Output = &'a IdentifierStatic;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        &data[self]
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::Range<usize> {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(&data[self])
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(&data[self])
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::RangeTo<usize> {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(&data[self])
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(&data[self])
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(&data[self])
    }
}

impl<'a> IdentifierStaticIndex<'a> for std::ops::RangeFull {
    type Output = IdentifierStaticSliceView<'a>;
    fn get(self, data: &'a [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceView::new(data)
    }
}

/// Mutable index trait for IdentifierStatic - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait IdentifierStaticMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output;
}

impl<'a> IdentifierStaticMutIndex<'a> for usize {
    type Output = &'a mut IdentifierStatic;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::Range<usize> {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierStaticMutIndex<'a> for std::ops::RangeFull {
    type Output = IdentifierStaticSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierStatic]) -> Self::Output {
        IdentifierStaticSliceViewMut::new(data)
    }
}

// --- IdentifierDynamicAos3 View Types ---

/// View over multiple IdentifierDynamicAos3 with field accumulation
pub struct IdentifierDynamicAos3SliceView<'a> {
    data: &'a [IdentifierDynamicAos3],
    pub name: StringAccumulator<'a, IdentifierDynamicAos3>,
    pub index: Accumulator<'a, IdentifierDynamicAos3, INT_0D>,
    pub description: StringAccumulator<'a, IdentifierDynamicAos3>,
}

impl<'a> IdentifierDynamicAos3SliceView<'a> {
    pub fn new(data: &'a [IdentifierDynamicAos3]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &IdentifierDynamicAos3| item.name.clone(), "name"),
            index: Accumulator::new(data, |item: &IdentifierDynamicAos3| item.index, "index"),
            description: StringAccumulator::new(data, |item: &IdentifierDynamicAos3| item.description.clone(), "description"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &IdentifierDynamicAos3> {
        self.data.iter()
    }
}

/// Mutable view over multiple IdentifierDynamicAos3
pub struct IdentifierDynamicAos3SliceViewMut<'a> {
    data: &'a mut [IdentifierDynamicAos3],
}

impl<'a> IdentifierDynamicAos3SliceViewMut<'a> {
    pub fn new(data: &'a mut [IdentifierDynamicAos3]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut IdentifierDynamicAos3> {
        self.data.iter_mut()
    }
}

/// Index trait for IdentifierDynamicAos3 - enables .field(0) and .field(0..2) syntax
pub trait IdentifierDynamicAos3Index<'a> {
    type Output;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output;
}

impl<'a> IdentifierDynamicAos3Index<'a> for usize {
    type Output = &'a IdentifierDynamicAos3;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        &data[self]
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::Range<usize> {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(&data[self])
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::RangeFrom<usize> {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(&data[self])
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::RangeTo<usize> {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(&data[self])
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::RangeInclusive<usize> {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(&data[self])
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::RangeToInclusive<usize> {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(&data[self])
    }
}

impl<'a> IdentifierDynamicAos3Index<'a> for std::ops::RangeFull {
    type Output = IdentifierDynamicAos3SliceView<'a>;
    fn get(self, data: &'a [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceView::new(data)
    }
}

/// Mutable index trait for IdentifierDynamicAos3 - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait IdentifierDynamicAos3MutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output;
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for usize {
    type Output = &'a mut IdentifierDynamicAos3;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::Range<usize> {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(&mut data[self])
    }
}

impl<'a> IdentifierDynamicAos3MutIndex<'a> for std::ops::RangeFull {
    type Output = IdentifierDynamicAos3SliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [IdentifierDynamicAos3]) -> Self::Output {
        IdentifierDynamicAos3SliceViewMut::new(data)
    }
}

// --- GenericGridDynamicSpaceDimension View Types ---

/// View over `geometry_content` (IdentifierDynamicAos3) across multiple GenericGridDynamicSpaceDimension
pub struct GenericGridDynamicSpaceDimensionGeometryContentView<'a> {
    pub name: StringAccumulator<'a, GenericGridDynamicSpaceDimension>,
    pub index: Accumulator<'a, GenericGridDynamicSpaceDimension, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridDynamicSpaceDimension>,
}

impl<'a> GenericGridDynamicSpaceDimensionGeometryContentView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpaceDimension]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &GenericGridDynamicSpaceDimension| item.geometry_content.name.clone(),
                "geometry_content.name",
            ),
            index: Accumulator::new(
                data,
                |item: &GenericGridDynamicSpaceDimension| item.geometry_content.index,
                "geometry_content.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &GenericGridDynamicSpaceDimension| item.geometry_content.description.clone(),
                "geometry_content.description",
            ),
        }
    }
}

/// View over multiple GenericGridDynamicSpaceDimension with field accumulation
pub struct GenericGridDynamicSpaceDimensionSliceView<'a> {
    data: &'a [GenericGridDynamicSpaceDimension],
    pub geometry_content: GenericGridDynamicSpaceDimensionGeometryContentView<'a>,
}

impl<'a> GenericGridDynamicSpaceDimensionSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpaceDimension]) -> Self {
        Self {
            data,
            geometry_content: GenericGridDynamicSpaceDimensionGeometryContentView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicSpaceDimension> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicSpaceDimension
pub struct GenericGridDynamicSpaceDimensionSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicSpaceDimension],
}

impl<'a> GenericGridDynamicSpaceDimensionSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicSpaceDimension> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicSpaceDimension - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for usize {
    type Output = &'a GenericGridDynamicSpaceDimension;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicSpaceDimension - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicSpaceDimension;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimension]) -> Self::Output {
        GenericGridDynamicSpaceDimensionSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicGridSubsetElement View Types ---

/// View over multiple GenericGridDynamicGridSubsetElement with field accumulation
pub struct GenericGridDynamicGridSubsetElementSliceView<'a> {
    data: &'a [GenericGridDynamicGridSubsetElement],
}

impl<'a> GenericGridDynamicGridSubsetElementSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicGridSubsetElement]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicGridSubsetElement> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicGridSubsetElement
pub struct GenericGridDynamicGridSubsetElementSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicGridSubsetElement],
}

impl<'a> GenericGridDynamicGridSubsetElementSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicGridSubsetElement> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicGridSubsetElement - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicGridSubsetElementIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for usize {
    type Output = &'a GenericGridDynamicGridSubsetElement;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetElementSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicGridSubsetElement - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicGridSubsetElementMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicGridSubsetElement;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetElementSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElement]) -> Self::Output {
        GenericGridDynamicGridSubsetElementSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicGridSubsetMetric View Types ---

/// View over multiple GenericGridDynamicGridSubsetMetric with field accumulation
pub struct GenericGridDynamicGridSubsetMetricSliceView<'a> {
    data: &'a [GenericGridDynamicGridSubsetMetric],
}

impl<'a> GenericGridDynamicGridSubsetMetricSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicGridSubsetMetric> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicGridSubsetMetric
pub struct GenericGridDynamicGridSubsetMetricSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicGridSubsetMetric],
}

impl<'a> GenericGridDynamicGridSubsetMetricSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicGridSubsetMetric> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicGridSubsetMetric - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicGridSubsetMetricIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for usize {
    type Output = &'a GenericGridDynamicGridSubsetMetric;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetMetricSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicGridSubsetMetric - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicGridSubsetMetricMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicGridSubsetMetric;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetMetricMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetMetricSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetMetric]) -> Self::Output {
        GenericGridDynamicGridSubsetMetricSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicSpaceDimensionObject View Types ---

/// View over multiple GenericGridDynamicSpaceDimensionObject with field accumulation
pub struct GenericGridDynamicSpaceDimensionObjectSliceView<'a> {
    data: &'a [GenericGridDynamicSpaceDimensionObject],
    pub measure: Accumulator<'a, GenericGridDynamicSpaceDimensionObject, FLT_0D>,
}

impl<'a> GenericGridDynamicSpaceDimensionObjectSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self {
        Self {
            data,
            measure: Accumulator::new(data, |item: &GenericGridDynamicSpaceDimensionObject| item.measure, "measure"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicSpaceDimensionObject> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicSpaceDimensionObject
pub struct GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicSpaceDimensionObject],
}

impl<'a> GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicSpaceDimensionObject> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicSpaceDimensionObject - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionObjectIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for usize {
    type Output = &'a GenericGridDynamicSpaceDimensionObject;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicSpaceDimensionObject - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionObjectMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicSpaceDimensionObject;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObject]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicGridSubsetElementObject View Types ---

/// View over multiple GenericGridDynamicGridSubsetElementObject with field accumulation
pub struct GenericGridDynamicGridSubsetElementObjectSliceView<'a> {
    data: &'a [GenericGridDynamicGridSubsetElementObject],
    pub space: Accumulator<'a, GenericGridDynamicGridSubsetElementObject, INT_0D>,
    pub dimension: Accumulator<'a, GenericGridDynamicGridSubsetElementObject, INT_0D>,
    pub index: Accumulator<'a, GenericGridDynamicGridSubsetElementObject, INT_0D>,
}

impl<'a> GenericGridDynamicGridSubsetElementObjectSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self {
        Self {
            data,
            space: Accumulator::new(data, |item: &GenericGridDynamicGridSubsetElementObject| item.space, "space"),
            dimension: Accumulator::new(data, |item: &GenericGridDynamicGridSubsetElementObject| item.dimension, "dimension"),
            index: Accumulator::new(data, |item: &GenericGridDynamicGridSubsetElementObject| item.index, "index"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicGridSubsetElementObject> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicGridSubsetElementObject
pub struct GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicGridSubsetElementObject],
}

impl<'a> GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicGridSubsetElementObject> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicGridSubsetElementObject - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicGridSubsetElementObjectIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for usize {
    type Output = &'a GenericGridDynamicGridSubsetElementObject;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicGridSubsetElementObject - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicGridSubsetElementObjectMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output;
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicGridSubsetElementObject;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicGridSubsetElementObjectMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicGridSubsetElementObjectSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicGridSubsetElementObject]) -> Self::Output {
        GenericGridDynamicGridSubsetElementObjectSliceViewMut::new(data)
    }
}

// --- GenericGridDynamicSpaceDimensionObjectBoundary View Types ---

/// View over multiple GenericGridDynamicSpaceDimensionObjectBoundary with field accumulation
pub struct GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a> {
    data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary],
    pub index: Accumulator<'a, GenericGridDynamicSpaceDimensionObjectBoundary, INT_0D>,
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self {
        Self {
            data,
            index: Accumulator::new(data, |item: &GenericGridDynamicSpaceDimensionObjectBoundary| item.index, "index"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamicSpaceDimensionObjectBoundary> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamicSpaceDimensionObjectBoundary
pub struct GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a> {
    data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary],
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamicSpaceDimensionObjectBoundary> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamicSpaceDimensionObjectBoundary - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for usize {
    type Output = &'a GenericGridDynamicSpaceDimensionObjectBoundary;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamicSpaceDimensionObjectBoundary - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output;
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamicSpaceDimensionObjectBoundary;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamicSpaceDimensionObjectBoundary]) -> Self::Output {
        GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut::new(data)
    }
}

// --- Wall2d View Types ---

/// View over `type` (IdentifierStatic) across multiple Wall2d
pub struct Wall2dTypeView<'a> {
    pub name: StringAccumulator<'a, Wall2d>,
    pub index: Accumulator<'a, Wall2d, INT_0D>,
    pub description: StringAccumulator<'a, Wall2d>,
}

impl<'a> Wall2dTypeView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &Wall2d| item.r#type.name.clone(), "type.name"),
            index: Accumulator::new(data, |item: &Wall2d| item.r#type.index, "type.index"),
            description: StringAccumulator::new(data, |item: &Wall2d| item.r#type.description.clone(), "type.description"),
        }
    }
}

/// View over `limiter.type` (IdentifierStatic) across multiple Wall2d
pub struct Wall2dLimiterTypeView<'a> {
    pub name: StringAccumulator<'a, Wall2d>,
    pub index: Accumulator<'a, Wall2d, INT_0D>,
    pub description: StringAccumulator<'a, Wall2d>,
}

impl<'a> Wall2dLimiterTypeView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &Wall2d| item.limiter.r#type.name.clone(), "limiter.type.name"),
            index: Accumulator::new(data, |item: &Wall2d| item.limiter.r#type.index, "limiter.type.index"),
            description: StringAccumulator::new(data, |item: &Wall2d| item.limiter.r#type.description.clone(), "limiter.type.description"),
        }
    }
}

/// View over `limiter` (Wall2dLimiter) across multiple Wall2d
pub struct Wall2dLimiterView<'a> {
    pub r#type: Wall2dLimiterTypeView<'a>,
}

impl<'a> Wall2dLimiterView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            r#type: Wall2dLimiterTypeView::new(data),
        }
    }
}

/// View over `mobile.type` (IdentifierStatic) across multiple Wall2d
pub struct Wall2dMobileTypeView<'a> {
    pub name: StringAccumulator<'a, Wall2d>,
    pub index: Accumulator<'a, Wall2d, INT_0D>,
    pub description: StringAccumulator<'a, Wall2d>,
}

impl<'a> Wall2dMobileTypeView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &Wall2d| item.mobile.r#type.name.clone(), "mobile.type.name"),
            index: Accumulator::new(data, |item: &Wall2d| item.mobile.r#type.index, "mobile.type.index"),
            description: StringAccumulator::new(data, |item: &Wall2d| item.mobile.r#type.description.clone(), "mobile.type.description"),
        }
    }
}

/// View over `mobile` (Wall2dMobile) across multiple Wall2d
pub struct Wall2dMobileView<'a> {
    pub r#type: Wall2dMobileTypeView<'a>,
}

impl<'a> Wall2dMobileView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            r#type: Wall2dMobileTypeView::new(data),
        }
    }
}

/// View over `vessel.type` (IdentifierStatic) across multiple Wall2d
pub struct Wall2dVesselTypeView<'a> {
    pub name: StringAccumulator<'a, Wall2d>,
    pub index: Accumulator<'a, Wall2d, INT_0D>,
    pub description: StringAccumulator<'a, Wall2d>,
}

impl<'a> Wall2dVesselTypeView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &Wall2d| item.vessel.r#type.name.clone(), "vessel.type.name"),
            index: Accumulator::new(data, |item: &Wall2d| item.vessel.r#type.index, "vessel.type.index"),
            description: StringAccumulator::new(data, |item: &Wall2d| item.vessel.r#type.description.clone(), "vessel.type.description"),
        }
    }
}

/// View over `vessel` (Vessel2d) across multiple Wall2d
pub struct Wall2dVesselView<'a> {
    pub r#type: Wall2dVesselTypeView<'a>,
}

impl<'a> Wall2dVesselView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            r#type: Wall2dVesselTypeView::new(data),
        }
    }
}

/// View over multiple Wall2d with field accumulation
pub struct Wall2dSliceView<'a> {
    data: &'a [Wall2d],
    pub r#type: Wall2dTypeView<'a>,
    pub limiter: Wall2dLimiterView<'a>,
    pub mobile: Wall2dMobileView<'a>,
    pub vessel: Wall2dVesselView<'a>,
}

impl<'a> Wall2dSliceView<'a> {
    pub fn new(data: &'a [Wall2d]) -> Self {
        Self {
            data,
            r#type: Wall2dTypeView::new(data),
            limiter: Wall2dLimiterView::new(data),
            mobile: Wall2dMobileView::new(data),
            vessel: Wall2dVesselView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Wall2d> {
        self.data.iter()
    }
}

/// Mutable view over multiple Wall2d
pub struct Wall2dSliceViewMut<'a> {
    data: &'a mut [Wall2d],
}

impl<'a> Wall2dSliceViewMut<'a> {
    pub fn new(data: &'a mut [Wall2d]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Wall2d> {
        self.data.iter_mut()
    }
}

/// Index trait for Wall2d - enables .field(0) and .field(0..2) syntax
pub trait Wall2dIndex<'a> {
    type Output;
    fn get(self, data: &'a [Wall2d]) -> Self::Output;
}

impl<'a> Wall2dIndex<'a> for usize {
    type Output = &'a Wall2d;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(&data[self])
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(&data[self])
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(&data[self])
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(&data[self])
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(&data[self])
    }
}

impl<'a> Wall2dIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dSliceView<'a>;
    fn get(self, data: &'a [Wall2d]) -> Self::Output {
        Wall2dSliceView::new(data)
    }
}

/// Mutable index trait for Wall2d - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Wall2dMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output;
}

impl<'a> Wall2dMutIndex<'a> for usize {
    type Output = &'a mut Wall2d;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::Range<usize> {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Wall2dMutIndex<'a> for std::ops::RangeFull {
    type Output = Wall2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Wall2d]) -> Self::Output {
        Wall2dSliceViewMut::new(data)
    }
}

// --- WallDescriptionGgd View Types ---

/// View over `type` (IdentifierStatic) across multiple WallDescriptionGgd
pub struct WallDescriptionGgdTypeView<'a> {
    pub name: StringAccumulator<'a, WallDescriptionGgd>,
    pub index: Accumulator<'a, WallDescriptionGgd, INT_0D>,
    pub description: StringAccumulator<'a, WallDescriptionGgd>,
}

impl<'a> WallDescriptionGgdTypeView<'a> {
    pub fn new(data: &'a [WallDescriptionGgd]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &WallDescriptionGgd| item.r#type.name.clone(), "type.name"),
            index: Accumulator::new(data, |item: &WallDescriptionGgd| item.r#type.index, "type.index"),
            description: StringAccumulator::new(data, |item: &WallDescriptionGgd| item.r#type.description.clone(), "type.description"),
        }
    }
}

/// View over multiple WallDescriptionGgd with field accumulation
pub struct WallDescriptionGgdSliceView<'a> {
    data: &'a [WallDescriptionGgd],
    pub r#type: WallDescriptionGgdTypeView<'a>,
}

impl<'a> WallDescriptionGgdSliceView<'a> {
    pub fn new(data: &'a [WallDescriptionGgd]) -> Self {
        Self {
            data,
            r#type: WallDescriptionGgdTypeView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WallDescriptionGgd> {
        self.data.iter()
    }
}

/// Mutable view over multiple WallDescriptionGgd
pub struct WallDescriptionGgdSliceViewMut<'a> {
    data: &'a mut [WallDescriptionGgd],
}

impl<'a> WallDescriptionGgdSliceViewMut<'a> {
    pub fn new(data: &'a mut [WallDescriptionGgd]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WallDescriptionGgd> {
        self.data.iter_mut()
    }
}

/// Index trait for WallDescriptionGgd - enables .field(0) and .field(0..2) syntax
pub trait WallDescriptionGgdIndex<'a> {
    type Output;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output;
}

impl<'a> WallDescriptionGgdIndex<'a> for usize {
    type Output = &'a WallDescriptionGgd;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        &data[self]
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(&data[self])
    }
}

impl<'a> WallDescriptionGgdIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdSliceView<'a>;
    fn get(self, data: &'a [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceView::new(data)
    }
}

/// Mutable index trait for WallDescriptionGgd - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait WallDescriptionGgdMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output;
}

impl<'a> WallDescriptionGgdMutIndex<'a> for usize {
    type Output = &'a mut WallDescriptionGgd;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::Range<usize> {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> WallDescriptionGgdMutIndex<'a> for std::ops::RangeFull {
    type Output = WallDescriptionGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [WallDescriptionGgd]) -> Self::Output {
        WallDescriptionGgdSliceViewMut::new(data)
    }
}

// ============================================================================
// Struct Impl Blocks for Vec Field Access
// ============================================================================

impl WallGlobalQuantititesNeutralOrigin {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElementConstant`, `.element(0..2)` returns `PlasmaCompositionNeutralElementConstantSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementConstantIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElementConstant`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementConstantSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementConstantMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallGlobalQuantititesNeutral {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElementConstant`, `.element(0..2)` returns `PlasmaCompositionNeutralElementConstantSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementConstantIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElementConstant`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementConstantSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementConstantMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallGlobalQuantititesNeutral {
    /// Access incident_species - use index for single element or range for slice view
    /// e.g. `.incident_species(0)` returns `&WallGlobalQuantititesNeutralOrigin`, `.incident_species(0..2)` returns `WallGlobalQuantititesNeutralOriginSliceView`
    pub fn incident_species<'a, I: WallGlobalQuantititesNeutralOriginIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident_species)
    }

    /// Access incident_species mutably - use index for single element or range for slice view
    /// e.g. `.incident_species_mut(0)` returns `&mut WallGlobalQuantititesNeutralOrigin`, `.incident_species_mut(0..2)` returns `WallGlobalQuantititesNeutralOriginSliceViewMut`
    pub fn incident_species_mut<'a, I: WallGlobalQuantititesNeutralOriginMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident_species)
    }

    /// Get the number of incident_species elements
    pub fn incident_species_len(&self) -> usize {
        self.incident_species.len()
    }
}

impl WallGlobalQuantitites {
    /// Access neutral - use index for single element or range for slice view
    /// e.g. `.neutral(0)` returns `&WallGlobalQuantititesNeutral`, `.neutral(0..2)` returns `WallGlobalQuantititesNeutralSliceView`
    pub fn neutral<'a, I: WallGlobalQuantititesNeutralIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.neutral)
    }

    /// Access neutral mutably - use index for single element or range for slice view
    /// e.g. `.neutral_mut(0)` returns `&mut WallGlobalQuantititesNeutral`, `.neutral_mut(0..2)` returns `WallGlobalQuantititesNeutralSliceViewMut`
    pub fn neutral_mut<'a, I: WallGlobalQuantititesNeutralMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.neutral)
    }

    /// Get the number of neutral elements
    pub fn neutral_len(&self) -> usize {
        self.neutral.len()
    }
}

impl Wall2dLimiter {
    /// Access unit - use index for single element or range for slice view
    /// e.g. `.unit(0)` returns `&Wall2dLimiterUnit`, `.unit(0..2)` returns `Wall2dLimiterUnitSliceView`
    pub fn unit<'a, I: Wall2dLimiterUnitIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.unit)
    }

    /// Access unit mutably - use index for single element or range for slice view
    /// e.g. `.unit_mut(0)` returns `&mut Wall2dLimiterUnit`, `.unit_mut(0..2)` returns `Wall2dLimiterUnitSliceViewMut`
    pub fn unit_mut<'a, I: Wall2dLimiterUnitMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.unit)
    }

    /// Get the number of unit elements
    pub fn unit_len(&self) -> usize {
        self.unit.len()
    }
}

impl Wall2dMobileUnit {
    /// Access outline - use index for single element or range for slice view
    /// e.g. `.outline(0)` returns `&Rz1dDynamicAosTime`, `.outline(0..2)` returns `Rz1dDynamicAosTimeSliceView`
    pub fn outline<'a, I: Rz1dDynamicAosTimeIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.outline)
    }

    /// Access outline mutably - use index for single element or range for slice view
    /// e.g. `.outline_mut(0)` returns `&mut Rz1dDynamicAosTime`, `.outline_mut(0..2)` returns `Rz1dDynamicAosTimeSliceViewMut`
    pub fn outline_mut<'a, I: Rz1dDynamicAosTimeMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.outline)
    }

    /// Get the number of outline elements
    pub fn outline_len(&self) -> usize {
        self.outline.len()
    }
}

impl Wall2dMobile {
    /// Access unit - use index for single element or range for slice view
    /// e.g. `.unit(0)` returns `&Wall2dMobileUnit`, `.unit(0..2)` returns `Wall2dMobileUnitSliceView`
    pub fn unit<'a, I: Wall2dMobileUnitIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.unit)
    }

    /// Access unit mutably - use index for single element or range for slice view
    /// e.g. `.unit_mut(0)` returns `&mut Wall2dMobileUnit`, `.unit_mut(0..2)` returns `Wall2dMobileUnitSliceViewMut`
    pub fn unit_mut<'a, I: Wall2dMobileUnitMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.unit)
    }

    /// Get the number of unit elements
    pub fn unit_len(&self) -> usize {
        self.unit.len()
    }
}

impl WallDescriptionGgdEnergySimple {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdEnergySimple {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdEnergyNeutralState {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdEnergyNeutralState {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdEnergyNeutral {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdEnergyNeutral {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdEnergyNeutral {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdEnergyNeutral {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdEnergyNeutralState`, `.state(0..2)` returns `WallDescriptionGgdEnergyNeutralStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdEnergyNeutralStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdEnergyNeutralState`, `.state_mut(0..2)` returns `WallDescriptionGgdEnergyNeutralStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdEnergyNeutralStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdEnergyIonState {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdEnergyIonState {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdEnergyIon {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdEnergyIon {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdEnergyIon {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdEnergyIon {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdEnergyIonState`, `.state(0..2)` returns `WallDescriptionGgdEnergyIonStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdEnergyIonStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdEnergyIonState`, `.state_mut(0..2)` returns `WallDescriptionGgdEnergyIonStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdEnergyIonStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdKinetic {
    /// Access ion - use index for single element or range for slice view
    /// e.g. `.ion(0)` returns `&WallDescriptionGgdEnergyIon`, `.ion(0..2)` returns `WallDescriptionGgdEnergyIonSliceView`
    pub fn ion<'a, I: WallDescriptionGgdEnergyIonIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ion)
    }

    /// Access ion mutably - use index for single element or range for slice view
    /// e.g. `.ion_mut(0)` returns `&mut WallDescriptionGgdEnergyIon`, `.ion_mut(0..2)` returns `WallDescriptionGgdEnergyIonSliceViewMut`
    pub fn ion_mut<'a, I: WallDescriptionGgdEnergyIonMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ion)
    }

    /// Get the number of ion elements
    pub fn ion_len(&self) -> usize {
        self.ion.len()
    }
}

impl WallDescriptionGgdKinetic {
    /// Access neutral - use index for single element or range for slice view
    /// e.g. `.neutral(0)` returns `&WallDescriptionGgdEnergyNeutral`, `.neutral(0..2)` returns `WallDescriptionGgdEnergyNeutralSliceView`
    pub fn neutral<'a, I: WallDescriptionGgdEnergyNeutralIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.neutral)
    }

    /// Access neutral mutably - use index for single element or range for slice view
    /// e.g. `.neutral_mut(0)` returns `&mut WallDescriptionGgdEnergyNeutral`, `.neutral_mut(0..2)` returns `WallDescriptionGgdEnergyNeutralSliceViewMut`
    pub fn neutral_mut<'a, I: WallDescriptionGgdEnergyNeutralMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.neutral)
    }

    /// Get the number of neutral elements
    pub fn neutral_len(&self) -> usize {
        self.neutral.len()
    }
}

impl WallDescriptionGgdRecombination {
    /// Access ion - use index for single element or range for slice view
    /// e.g. `.ion(0)` returns `&WallDescriptionGgdEnergyIon`, `.ion(0..2)` returns `WallDescriptionGgdEnergyIonSliceView`
    pub fn ion<'a, I: WallDescriptionGgdEnergyIonIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ion)
    }

    /// Access ion mutably - use index for single element or range for slice view
    /// e.g. `.ion_mut(0)` returns `&mut WallDescriptionGgdEnergyIon`, `.ion_mut(0..2)` returns `WallDescriptionGgdEnergyIonSliceViewMut`
    pub fn ion_mut<'a, I: WallDescriptionGgdEnergyIonMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ion)
    }

    /// Get the number of ion elements
    pub fn ion_len(&self) -> usize {
        self.ion.len()
    }
}

impl WallDescriptionGgdRecombination {
    /// Access neutral - use index for single element or range for slice view
    /// e.g. `.neutral(0)` returns `&WallDescriptionGgdEnergyNeutral`, `.neutral(0..2)` returns `WallDescriptionGgdEnergyNeutralSliceView`
    pub fn neutral<'a, I: WallDescriptionGgdEnergyNeutralIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.neutral)
    }

    /// Access neutral mutably - use index for single element or range for slice view
    /// e.g. `.neutral_mut(0)` returns `&mut WallDescriptionGgdEnergyNeutral`, `.neutral_mut(0..2)` returns `WallDescriptionGgdEnergyNeutralSliceViewMut`
    pub fn neutral_mut<'a, I: WallDescriptionGgdEnergyNeutralMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.neutral)
    }

    /// Get the number of neutral elements
    pub fn neutral_len(&self) -> usize {
        self.neutral.len()
    }
}

impl WallDescriptionGgdParticleNeutralState {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdParticleNeutralState {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdParticleNeutral {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdParticleNeutral {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdParticleNeutral {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdParticleNeutral {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdParticleNeutralState`, `.state(0..2)` returns `WallDescriptionGgdParticleNeutralStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdParticleNeutralStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdParticleNeutralState`, `.state_mut(0..2)` returns `WallDescriptionGgdParticleNeutralStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdParticleNeutralStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdParticleIonState {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdParticleIonState {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdParticleIon {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdParticleIon {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdParticleIon {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdParticleIon {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdParticleIonState`, `.state(0..2)` returns `WallDescriptionGgdParticleIonStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdParticleIonStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdParticleIonState`, `.state_mut(0..2)` returns `WallDescriptionGgdParticleIonStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdParticleIonStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdParticleEl {
    /// Access incident - use index for single element or range for slice view
    /// e.g. `.incident(0)` returns `&GenericGridScalar`, `.incident(0..2)` returns `GenericGridScalarSliceView`
    pub fn incident<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.incident)
    }

    /// Access incident mutably - use index for single element or range for slice view
    /// e.g. `.incident_mut(0)` returns `&mut GenericGridScalar`, `.incident_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn incident_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.incident)
    }

    /// Get the number of incident elements
    pub fn incident_len(&self) -> usize {
        self.incident.len()
    }
}

impl WallDescriptionGgdParticleEl {
    /// Access emitted - use index for single element or range for slice view
    /// e.g. `.emitted(0)` returns `&GenericGridScalar`, `.emitted(0..2)` returns `GenericGridScalarSliceView`
    pub fn emitted<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.emitted)
    }

    /// Access emitted mutably - use index for single element or range for slice view
    /// e.g. `.emitted_mut(0)` returns `&mut GenericGridScalar`, `.emitted_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn emitted_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.emitted)
    }

    /// Get the number of emitted elements
    pub fn emitted_len(&self) -> usize {
        self.emitted.len()
    }
}

impl WallDescriptionGgdParticle {
    /// Access ion - use index for single element or range for slice view
    /// e.g. `.ion(0)` returns `&WallDescriptionGgdParticleIon`, `.ion(0..2)` returns `WallDescriptionGgdParticleIonSliceView`
    pub fn ion<'a, I: WallDescriptionGgdParticleIonIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ion)
    }

    /// Access ion mutably - use index for single element or range for slice view
    /// e.g. `.ion_mut(0)` returns `&mut WallDescriptionGgdParticleIon`, `.ion_mut(0..2)` returns `WallDescriptionGgdParticleIonSliceViewMut`
    pub fn ion_mut<'a, I: WallDescriptionGgdParticleIonMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ion)
    }

    /// Get the number of ion elements
    pub fn ion_len(&self) -> usize {
        self.ion.len()
    }
}

impl WallDescriptionGgdParticle {
    /// Access neutral - use index for single element or range for slice view
    /// e.g. `.neutral(0)` returns `&WallDescriptionGgdParticleNeutral`, `.neutral(0..2)` returns `WallDescriptionGgdParticleNeutralSliceView`
    pub fn neutral<'a, I: WallDescriptionGgdParticleNeutralIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.neutral)
    }

    /// Access neutral mutably - use index for single element or range for slice view
    /// e.g. `.neutral_mut(0)` returns `&mut WallDescriptionGgdParticleNeutral`, `.neutral_mut(0..2)` returns `WallDescriptionGgdParticleNeutralSliceViewMut`
    pub fn neutral_mut<'a, I: WallDescriptionGgdParticleNeutralMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.neutral)
    }

    /// Get the number of neutral elements
    pub fn neutral_len(&self) -> usize {
        self.neutral.len()
    }
}

impl WallDescriptionGgdRecyclingNeutralState {
    /// Access coefficient - use index for single element or range for slice view
    /// e.g. `.coefficient(0)` returns `&GenericGridScalar`, `.coefficient(0..2)` returns `GenericGridScalarSliceView`
    pub fn coefficient<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.coefficient)
    }

    /// Access coefficient mutably - use index for single element or range for slice view
    /// e.g. `.coefficient_mut(0)` returns `&mut GenericGridScalar`, `.coefficient_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn coefficient_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.coefficient)
    }

    /// Get the number of coefficient elements
    pub fn coefficient_len(&self) -> usize {
        self.coefficient.len()
    }
}

impl WallDescriptionGgdRecyclingNeutral {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdRecyclingNeutral {
    /// Access coefficient - use index for single element or range for slice view
    /// e.g. `.coefficient(0)` returns `&GenericGridScalar`, `.coefficient(0..2)` returns `GenericGridScalarSliceView`
    pub fn coefficient<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.coefficient)
    }

    /// Access coefficient mutably - use index for single element or range for slice view
    /// e.g. `.coefficient_mut(0)` returns `&mut GenericGridScalar`, `.coefficient_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn coefficient_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.coefficient)
    }

    /// Get the number of coefficient elements
    pub fn coefficient_len(&self) -> usize {
        self.coefficient.len()
    }
}

impl WallDescriptionGgdRecyclingNeutral {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdRecyclingNeutralState`, `.state(0..2)` returns `WallDescriptionGgdRecyclingNeutralStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdRecyclingNeutralStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdRecyclingNeutralState`, `.state_mut(0..2)` returns `WallDescriptionGgdRecyclingNeutralStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdRecyclingNeutralStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdRecyclingIonState {
    /// Access coefficient - use index for single element or range for slice view
    /// e.g. `.coefficient(0)` returns `&GenericGridScalar`, `.coefficient(0..2)` returns `GenericGridScalarSliceView`
    pub fn coefficient<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.coefficient)
    }

    /// Access coefficient mutably - use index for single element or range for slice view
    /// e.g. `.coefficient_mut(0)` returns `&mut GenericGridScalar`, `.coefficient_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn coefficient_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.coefficient)
    }

    /// Get the number of coefficient elements
    pub fn coefficient_len(&self) -> usize {
        self.coefficient.len()
    }
}

impl WallDescriptionGgdRecyclingIon {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&PlasmaCompositionNeutralElement`, `.element(0..2)` returns `PlasmaCompositionNeutralElementSliceView`
    pub fn element<'a, I: PlasmaCompositionNeutralElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut PlasmaCompositionNeutralElement`, `.element_mut(0..2)` returns `PlasmaCompositionNeutralElementSliceViewMut`
    pub fn element_mut<'a, I: PlasmaCompositionNeutralElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl WallDescriptionGgdRecyclingIon {
    /// Access coefficient - use index for single element or range for slice view
    /// e.g. `.coefficient(0)` returns `&GenericGridScalar`, `.coefficient(0..2)` returns `GenericGridScalarSliceView`
    pub fn coefficient<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.coefficient)
    }

    /// Access coefficient mutably - use index for single element or range for slice view
    /// e.g. `.coefficient_mut(0)` returns `&mut GenericGridScalar`, `.coefficient_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn coefficient_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.coefficient)
    }

    /// Get the number of coefficient elements
    pub fn coefficient_len(&self) -> usize {
        self.coefficient.len()
    }
}

impl WallDescriptionGgdRecyclingIon {
    /// Access state - use index for single element or range for slice view
    /// e.g. `.state(0)` returns `&WallDescriptionGgdRecyclingIonState`, `.state(0..2)` returns `WallDescriptionGgdRecyclingIonStateSliceView`
    pub fn state<'a, I: WallDescriptionGgdRecyclingIonStateIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.state)
    }

    /// Access state mutably - use index for single element or range for slice view
    /// e.g. `.state_mut(0)` returns `&mut WallDescriptionGgdRecyclingIonState`, `.state_mut(0..2)` returns `WallDescriptionGgdRecyclingIonStateSliceViewMut`
    pub fn state_mut<'a, I: WallDescriptionGgdRecyclingIonStateMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.state)
    }

    /// Get the number of state elements
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}

impl WallDescriptionGgdRecycling {
    /// Access ion - use index for single element or range for slice view
    /// e.g. `.ion(0)` returns `&WallDescriptionGgdRecyclingIon`, `.ion(0..2)` returns `WallDescriptionGgdRecyclingIonSliceView`
    pub fn ion<'a, I: WallDescriptionGgdRecyclingIonIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ion)
    }

    /// Access ion mutably - use index for single element or range for slice view
    /// e.g. `.ion_mut(0)` returns `&mut WallDescriptionGgdRecyclingIon`, `.ion_mut(0..2)` returns `WallDescriptionGgdRecyclingIonSliceViewMut`
    pub fn ion_mut<'a, I: WallDescriptionGgdRecyclingIonMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ion)
    }

    /// Get the number of ion elements
    pub fn ion_len(&self) -> usize {
        self.ion.len()
    }
}

impl WallDescriptionGgdRecycling {
    /// Access neutral - use index for single element or range for slice view
    /// e.g. `.neutral(0)` returns `&WallDescriptionGgdRecyclingNeutral`, `.neutral(0..2)` returns `WallDescriptionGgdRecyclingNeutralSliceView`
    pub fn neutral<'a, I: WallDescriptionGgdRecyclingNeutralIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.neutral)
    }

    /// Access neutral mutably - use index for single element or range for slice view
    /// e.g. `.neutral_mut(0)` returns `&mut WallDescriptionGgdRecyclingNeutral`, `.neutral_mut(0..2)` returns `WallDescriptionGgdRecyclingNeutralSliceViewMut`
    pub fn neutral_mut<'a, I: WallDescriptionGgdRecyclingNeutralMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.neutral)
    }

    /// Get the number of neutral elements
    pub fn neutral_len(&self) -> usize {
        self.neutral.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access power_density - use index for single element or range for slice view
    /// e.g. `.power_density(0)` returns `&GenericGridScalar`, `.power_density(0..2)` returns `GenericGridScalarSliceView`
    pub fn power_density<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.power_density)
    }

    /// Access power_density mutably - use index for single element or range for slice view
    /// e.g. `.power_density_mut(0)` returns `&mut GenericGridScalar`, `.power_density_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn power_density_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.power_density)
    }

    /// Get the number of power_density elements
    pub fn power_density_len(&self) -> usize {
        self.power_density.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access temperature - use index for single element or range for slice view
    /// e.g. `.temperature(0)` returns `&GenericGridScalar`, `.temperature(0..2)` returns `GenericGridScalarSliceView`
    pub fn temperature<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.temperature)
    }

    /// Access temperature mutably - use index for single element or range for slice view
    /// e.g. `.temperature_mut(0)` returns `&mut GenericGridScalar`, `.temperature_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn temperature_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.temperature)
    }

    /// Get the number of temperature elements
    pub fn temperature_len(&self) -> usize {
        self.temperature.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access v_biasing - use index for single element or range for slice view
    /// e.g. `.v_biasing(0)` returns `&GenericGridScalar`, `.v_biasing(0..2)` returns `GenericGridScalarSliceView`
    pub fn v_biasing<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.v_biasing)
    }

    /// Access v_biasing mutably - use index for single element or range for slice view
    /// e.g. `.v_biasing_mut(0)` returns `&mut GenericGridScalar`, `.v_biasing_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn v_biasing_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.v_biasing)
    }

    /// Get the number of v_biasing elements
    pub fn v_biasing_len(&self) -> usize {
        self.v_biasing.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access j_total - use index for single element or range for slice view
    /// e.g. `.j_total(0)` returns `&GenericGridVectorComponentsRphiz`, `.j_total(0..2)` returns `GenericGridVectorComponentsRphizSliceView`
    pub fn j_total<'a, I: GenericGridVectorComponentsRphizIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.j_total)
    }

    /// Access j_total mutably - use index for single element or range for slice view
    /// e.g. `.j_total_mut(0)` returns `&mut GenericGridVectorComponentsRphiz`, `.j_total_mut(0..2)` returns `GenericGridVectorComponentsRphizSliceViewMut`
    pub fn j_total_mut<'a, I: GenericGridVectorComponentsRphizMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.j_total)
    }

    /// Get the number of j_total elements
    pub fn j_total_len(&self) -> usize {
        self.j_total.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access b_field - use index for single element or range for slice view
    /// e.g. `.b_field(0)` returns `&GenericGridVectorComponentsRphiz`, `.b_field(0..2)` returns `GenericGridVectorComponentsRphizSliceView`
    pub fn b_field<'a, I: GenericGridVectorComponentsRphizIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.b_field)
    }

    /// Access b_field mutably - use index for single element or range for slice view
    /// e.g. `.b_field_mut(0)` returns `&mut GenericGridVectorComponentsRphiz`, `.b_field_mut(0..2)` returns `GenericGridVectorComponentsRphizSliceViewMut`
    pub fn b_field_mut<'a, I: GenericGridVectorComponentsRphizMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.b_field)
    }

    /// Get the number of b_field elements
    pub fn b_field_len(&self) -> usize {
        self.b_field.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access em_force_density - use index for single element or range for slice view
    /// e.g. `.em_force_density(0)` returns `&GenericGridVectorComponentsRphiz`, `.em_force_density(0..2)` returns `GenericGridVectorComponentsRphizSliceView`
    pub fn em_force_density<'a, I: GenericGridVectorComponentsRphizIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.em_force_density)
    }

    /// Access em_force_density mutably - use index for single element or range for slice view
    /// e.g. `.em_force_density_mut(0)` returns `&mut GenericGridVectorComponentsRphiz`, `.em_force_density_mut(0..2)` returns `GenericGridVectorComponentsRphizSliceViewMut`
    pub fn em_force_density_mut<'a, I: GenericGridVectorComponentsRphizMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.em_force_density)
    }

    /// Get the number of em_force_density elements
    pub fn em_force_density_len(&self) -> usize {
        self.em_force_density.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access e_field - use index for single element or range for slice view
    /// e.g. `.e_field(0)` returns `&GenericGridVectorComponentsRphiz`, `.e_field(0..2)` returns `GenericGridVectorComponentsRphizSliceView`
    pub fn e_field<'a, I: GenericGridVectorComponentsRphizIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.e_field)
    }

    /// Access e_field mutably - use index for single element or range for slice view
    /// e.g. `.e_field_mut(0)` returns `&mut GenericGridVectorComponentsRphiz`, `.e_field_mut(0..2)` returns `GenericGridVectorComponentsRphizSliceViewMut`
    pub fn e_field_mut<'a, I: GenericGridVectorComponentsRphizMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.e_field)
    }

    /// Get the number of e_field elements
    pub fn e_field_len(&self) -> usize {
        self.e_field.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access a_field - use index for single element or range for slice view
    /// e.g. `.a_field(0)` returns `&GenericGridVectorComponentsRphiz`, `.a_field(0..2)` returns `GenericGridVectorComponentsRphizSliceView`
    pub fn a_field<'a, I: GenericGridVectorComponentsRphizIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.a_field)
    }

    /// Access a_field mutably - use index for single element or range for slice view
    /// e.g. `.a_field_mut(0)` returns `&mut GenericGridVectorComponentsRphiz`, `.a_field_mut(0..2)` returns `GenericGridVectorComponentsRphizSliceViewMut`
    pub fn a_field_mut<'a, I: GenericGridVectorComponentsRphizMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.a_field)
    }

    /// Get the number of a_field elements
    pub fn a_field_len(&self) -> usize {
        self.a_field.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access psi - use index for single element or range for slice view
    /// e.g. `.psi(0)` returns `&GenericGridScalar`, `.psi(0..2)` returns `GenericGridScalarSliceView`
    pub fn psi<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.psi)
    }

    /// Access psi mutably - use index for single element or range for slice view
    /// e.g. `.psi_mut(0)` returns `&mut GenericGridScalar`, `.psi_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn psi_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.psi)
    }

    /// Get the number of psi elements
    pub fn psi_len(&self) -> usize {
        self.psi.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access phi_potential - use index for single element or range for slice view
    /// e.g. `.phi_potential(0)` returns `&GenericGridScalar`, `.phi_potential(0..2)` returns `GenericGridScalarSliceView`
    pub fn phi_potential<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.phi_potential)
    }

    /// Access phi_potential mutably - use index for single element or range for slice view
    /// e.g. `.phi_potential_mut(0)` returns `&mut GenericGridScalar`, `.phi_potential_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn phi_potential_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.phi_potential)
    }

    /// Get the number of phi_potential elements
    pub fn phi_potential_len(&self) -> usize {
        self.phi_potential.len()
    }
}

impl WallDescriptionGgdGgd {
    /// Access resistivity - use index for single element or range for slice view
    /// e.g. `.resistivity(0)` returns `&GenericGridScalar`, `.resistivity(0..2)` returns `GenericGridScalarSliceView`
    pub fn resistivity<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.resistivity)
    }

    /// Access resistivity mutably - use index for single element or range for slice view
    /// e.g. `.resistivity_mut(0)` returns `&mut GenericGridScalar`, `.resistivity_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn resistivity_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.resistivity)
    }

    /// Get the number of resistivity elements
    pub fn resistivity_len(&self) -> usize {
        self.resistivity.len()
    }
}

impl WallDescriptionGgdThickness {
    /// Access grid_subset - use index for single element or range for slice view
    /// e.g. `.grid_subset(0)` returns `&GenericGridScalar`, `.grid_subset(0..2)` returns `GenericGridScalarSliceView`
    pub fn grid_subset<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grid_subset)
    }

    /// Access grid_subset mutably - use index for single element or range for slice view
    /// e.g. `.grid_subset_mut(0)` returns `&mut GenericGridScalar`, `.grid_subset_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn grid_subset_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grid_subset)
    }

    /// Get the number of grid_subset elements
    pub fn grid_subset_len(&self) -> usize {
        self.grid_subset.len()
    }
}

impl WallDescriptionGgdBrdf {
    /// Access r#type - use index for single element or range for slice view
    /// e.g. `.r#type(0)` returns `&GenericGridIdentifier`, `.r#type(0..2)` returns `GenericGridIdentifierSliceView`
    pub fn r#type<'a, I: GenericGridIdentifierIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.r#type)
    }

    /// Access r#type mutably - use index for single element or range for slice view
    /// e.g. `.r#type_mut(0)` returns `&mut GenericGridIdentifier`, `.r#type_mut(0..2)` returns `GenericGridIdentifierSliceViewMut`
    pub fn r#type_mut<'a, I: GenericGridIdentifierMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.r#type)
    }

    /// Get the number of r#type elements
    pub fn r#type_len(&self) -> usize {
        self.r#type.len()
    }
}

impl WallDescriptionGgdBrdf {
    /// Access parameters - use index for single element or range for slice view
    /// e.g. `.parameters(0)` returns `&GenericGridVector`, `.parameters(0..2)` returns `GenericGridVectorSliceView`
    pub fn parameters<'a, I: GenericGridVectorIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.parameters)
    }

    /// Access parameters mutably - use index for single element or range for slice view
    /// e.g. `.parameters_mut(0)` returns `&mut GenericGridVector`, `.parameters_mut(0..2)` returns `GenericGridVectorSliceViewMut`
    pub fn parameters_mut<'a, I: GenericGridVectorMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.parameters)
    }

    /// Get the number of parameters elements
    pub fn parameters_len(&self) -> usize {
        self.parameters.len()
    }
}

impl WallDescriptionGgdMaterial {
    /// Access grid_subset - use index for single element or range for slice view
    /// e.g. `.grid_subset(0)` returns `&GenericGridIdentifier`, `.grid_subset(0..2)` returns `GenericGridIdentifierSliceView`
    pub fn grid_subset<'a, I: GenericGridIdentifierIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grid_subset)
    }

    /// Access grid_subset mutably - use index for single element or range for slice view
    /// e.g. `.grid_subset_mut(0)` returns `&mut GenericGridIdentifier`, `.grid_subset_mut(0..2)` returns `GenericGridIdentifierSliceViewMut`
    pub fn grid_subset_mut<'a, I: GenericGridIdentifierMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grid_subset)
    }

    /// Get the number of grid_subset elements
    pub fn grid_subset_len(&self) -> usize {
        self.grid_subset.len()
    }
}

impl WallDescriptionGgdComponent {
    /// Access r#type - use index for single element or range for slice view
    /// e.g. `.r#type(0)` returns `&GenericGridIdentifierSingle`, `.r#type(0..2)` returns `GenericGridIdentifierSingleSliceView`
    pub fn r#type<'a, I: GenericGridIdentifierSingleIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.r#type)
    }

    /// Access r#type mutably - use index for single element or range for slice view
    /// e.g. `.r#type_mut(0)` returns `&mut GenericGridIdentifierSingle`, `.r#type_mut(0..2)` returns `GenericGridIdentifierSingleSliceViewMut`
    pub fn r#type_mut<'a, I: GenericGridIdentifierSingleMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.r#type)
    }

    /// Get the number of r#type elements
    pub fn r#type_len(&self) -> usize {
        self.r#type.len()
    }
}

impl WallDescriptionGgd {
    /// Access grid_ggd - use index for single element or range for slice view
    /// e.g. `.grid_ggd(0)` returns `&GenericGridAos3Root`, `.grid_ggd(0..2)` returns `GenericGridAos3RootSliceView`
    pub fn grid_ggd<'a, I: GenericGridAos3RootIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grid_ggd)
    }

    /// Access grid_ggd mutably - use index for single element or range for slice view
    /// e.g. `.grid_ggd_mut(0)` returns `&mut GenericGridAos3Root`, `.grid_ggd_mut(0..2)` returns `GenericGridAos3RootSliceViewMut`
    pub fn grid_ggd_mut<'a, I: GenericGridAos3RootMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grid_ggd)
    }

    /// Get the number of grid_ggd elements
    pub fn grid_ggd_len(&self) -> usize {
        self.grid_ggd.len()
    }
}

impl WallDescriptionGgd {
    /// Access material - use index for single element or range for slice view
    /// e.g. `.material(0)` returns `&WallDescriptionGgdMaterial`, `.material(0..2)` returns `WallDescriptionGgdMaterialSliceView`
    pub fn material<'a, I: WallDescriptionGgdMaterialIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.material)
    }

    /// Access material mutably - use index for single element or range for slice view
    /// e.g. `.material_mut(0)` returns `&mut WallDescriptionGgdMaterial`, `.material_mut(0..2)` returns `WallDescriptionGgdMaterialSliceViewMut`
    pub fn material_mut<'a, I: WallDescriptionGgdMaterialMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.material)
    }

    /// Get the number of material elements
    pub fn material_len(&self) -> usize {
        self.material.len()
    }
}

impl WallDescriptionGgd {
    /// Access component - use index for single element or range for slice view
    /// e.g. `.component(0)` returns `&WallDescriptionGgdComponent`, `.component(0..2)` returns `WallDescriptionGgdComponentSliceView`
    pub fn component<'a, I: WallDescriptionGgdComponentIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.component)
    }

    /// Access component mutably - use index for single element or range for slice view
    /// e.g. `.component_mut(0)` returns `&mut WallDescriptionGgdComponent`, `.component_mut(0..2)` returns `WallDescriptionGgdComponentSliceViewMut`
    pub fn component_mut<'a, I: WallDescriptionGgdComponentMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.component)
    }

    /// Get the number of component elements
    pub fn component_len(&self) -> usize {
        self.component.len()
    }
}

impl WallDescriptionGgd {
    /// Access thickness - use index for single element or range for slice view
    /// e.g. `.thickness(0)` returns `&WallDescriptionGgdThickness`, `.thickness(0..2)` returns `WallDescriptionGgdThicknessSliceView`
    pub fn thickness<'a, I: WallDescriptionGgdThicknessIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.thickness)
    }

    /// Access thickness mutably - use index for single element or range for slice view
    /// e.g. `.thickness_mut(0)` returns `&mut WallDescriptionGgdThickness`, `.thickness_mut(0..2)` returns `WallDescriptionGgdThicknessSliceViewMut`
    pub fn thickness_mut<'a, I: WallDescriptionGgdThicknessMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.thickness)
    }

    /// Get the number of thickness elements
    pub fn thickness_len(&self) -> usize {
        self.thickness.len()
    }
}

impl WallDescriptionGgd {
    /// Access brdf - use index for single element or range for slice view
    /// e.g. `.brdf(0)` returns `&WallDescriptionGgdBrdf`, `.brdf(0..2)` returns `WallDescriptionGgdBrdfSliceView`
    pub fn brdf<'a, I: WallDescriptionGgdBrdfIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.brdf)
    }

    /// Access brdf mutably - use index for single element or range for slice view
    /// e.g. `.brdf_mut(0)` returns `&mut WallDescriptionGgdBrdf`, `.brdf_mut(0..2)` returns `WallDescriptionGgdBrdfSliceViewMut`
    pub fn brdf_mut<'a, I: WallDescriptionGgdBrdfMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.brdf)
    }

    /// Get the number of brdf elements
    pub fn brdf_len(&self) -> usize {
        self.brdf.len()
    }
}

impl WallDescriptionGgd {
    /// Access ggd - use index for single element or range for slice view
    /// e.g. `.ggd(0)` returns `&WallDescriptionGgdGgd`, `.ggd(0..2)` returns `WallDescriptionGgdGgdSliceView`
    pub fn ggd<'a, I: WallDescriptionGgdGgdIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ggd)
    }

    /// Access ggd mutably - use index for single element or range for slice view
    /// e.g. `.ggd_mut(0)` returns `&mut WallDescriptionGgdGgd`, `.ggd_mut(0..2)` returns `WallDescriptionGgdGgdSliceViewMut`
    pub fn ggd_mut<'a, I: WallDescriptionGgdGgdMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ggd)
    }

    /// Get the number of ggd elements
    pub fn ggd_len(&self) -> usize {
        self.ggd.len()
    }
}

impl Vessel2d {
    /// Access unit - use index for single element or range for slice view
    /// e.g. `.unit(0)` returns `&Vessel2dUnit`, `.unit(0..2)` returns `Vessel2dUnitSliceView`
    pub fn unit<'a, I: Vessel2dUnitIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.unit)
    }

    /// Access unit mutably - use index for single element or range for slice view
    /// e.g. `.unit_mut(0)` returns `&mut Vessel2dUnit`, `.unit_mut(0..2)` returns `Vessel2dUnitSliceViewMut`
    pub fn unit_mut<'a, I: Vessel2dUnitMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.unit)
    }

    /// Get the number of unit elements
    pub fn unit_len(&self) -> usize {
        self.unit.len()
    }
}

impl GenericGridAos3Root {
    /// Access space - use index for single element or range for slice view
    /// e.g. `.space(0)` returns `&GenericGridDynamicSpace`, `.space(0..2)` returns `GenericGridDynamicSpaceSliceView`
    pub fn space<'a, I: GenericGridDynamicSpaceIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.space)
    }

    /// Access space mutably - use index for single element or range for slice view
    /// e.g. `.space_mut(0)` returns `&mut GenericGridDynamicSpace`, `.space_mut(0..2)` returns `GenericGridDynamicSpaceSliceViewMut`
    pub fn space_mut<'a, I: GenericGridDynamicSpaceMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.space)
    }

    /// Get the number of space elements
    pub fn space_len(&self) -> usize {
        self.space.len()
    }
}

impl GenericGridAos3Root {
    /// Access grid_subset - use index for single element or range for slice view
    /// e.g. `.grid_subset(0)` returns `&GenericGridDynamicGridSubset`, `.grid_subset(0..2)` returns `GenericGridDynamicGridSubsetSliceView`
    pub fn grid_subset<'a, I: GenericGridDynamicGridSubsetIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grid_subset)
    }

    /// Access grid_subset mutably - use index for single element or range for slice view
    /// e.g. `.grid_subset_mut(0)` returns `&mut GenericGridDynamicGridSubset`, `.grid_subset_mut(0..2)` returns `GenericGridDynamicGridSubsetSliceViewMut`
    pub fn grid_subset_mut<'a, I: GenericGridDynamicGridSubsetMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grid_subset)
    }

    /// Get the number of grid_subset elements
    pub fn grid_subset_len(&self) -> usize {
        self.grid_subset.len()
    }
}

impl Code {
    /// Access library - use index for single element or range for slice view
    /// e.g. `.library(0)` returns `&Library`, `.library(0..2)` returns `LibrarySliceView`
    pub fn library<'a, I: LibraryIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.library)
    }

    /// Access library mutably - use index for single element or range for slice view
    /// e.g. `.library_mut(0)` returns `&mut Library`, `.library_mut(0..2)` returns `LibrarySliceViewMut`
    pub fn library_mut<'a, I: LibraryMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.library)
    }

    /// Get the number of library elements
    pub fn library_len(&self) -> usize {
        self.library.len()
    }
}

impl Vessel2dUnit {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&Vessel2dElement`, `.element(0..2)` returns `Vessel2dElementSliceView`
    pub fn element<'a, I: Vessel2dElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut Vessel2dElement`, `.element_mut(0..2)` returns `Vessel2dElementSliceViewMut`
    pub fn element_mut<'a, I: Vessel2dElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl Vessel2dUnit {
    /// Access material - use index for single element or range for slice view
    /// e.g. `.material(0)` returns `&IdentifierStatic`, `.material(0..2)` returns `IdentifierStaticSliceView`
    pub fn material<'a, I: IdentifierStaticIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.material)
    }

    /// Access material mutably - use index for single element or range for slice view
    /// e.g. `.material_mut(0)` returns `&mut IdentifierStatic`, `.material_mut(0..2)` returns `IdentifierStaticSliceViewMut`
    pub fn material_mut<'a, I: IdentifierStaticMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.material)
    }

    /// Get the number of material elements
    pub fn material_len(&self) -> usize {
        self.material.len()
    }
}

impl GenericGridDynamicSpace {
    /// Access coordinates_type - use index for single element or range for slice view
    /// e.g. `.coordinates_type(0)` returns `&IdentifierDynamicAos3`, `.coordinates_type(0..2)` returns `IdentifierDynamicAos3SliceView`
    pub fn coordinates_type<'a, I: IdentifierDynamicAos3Index<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.coordinates_type)
    }

    /// Access coordinates_type mutably - use index for single element or range for slice view
    /// e.g. `.coordinates_type_mut(0)` returns `&mut IdentifierDynamicAos3`, `.coordinates_type_mut(0..2)` returns `IdentifierDynamicAos3SliceViewMut`
    pub fn coordinates_type_mut<'a, I: IdentifierDynamicAos3MutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.coordinates_type)
    }

    /// Get the number of coordinates_type elements
    pub fn coordinates_type_len(&self) -> usize {
        self.coordinates_type.len()
    }
}

impl GenericGridDynamicSpace {
    /// Access objects_per_dimension - use index for single element or range for slice view
    /// e.g. `.objects_per_dimension(0)` returns `&GenericGridDynamicSpaceDimension`, `.objects_per_dimension(0..2)` returns `GenericGridDynamicSpaceDimensionSliceView`
    pub fn objects_per_dimension<'a, I: GenericGridDynamicSpaceDimensionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.objects_per_dimension)
    }

    /// Access objects_per_dimension mutably - use index for single element or range for slice view
    /// e.g. `.objects_per_dimension_mut(0)` returns `&mut GenericGridDynamicSpaceDimension`, `.objects_per_dimension_mut(0..2)` returns `GenericGridDynamicSpaceDimensionSliceViewMut`
    pub fn objects_per_dimension_mut<'a, I: GenericGridDynamicSpaceDimensionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.objects_per_dimension)
    }

    /// Get the number of objects_per_dimension elements
    pub fn objects_per_dimension_len(&self) -> usize {
        self.objects_per_dimension.len()
    }
}

impl GenericGridDynamicGridSubset {
    /// Access element - use index for single element or range for slice view
    /// e.g. `.element(0)` returns `&GenericGridDynamicGridSubsetElement`, `.element(0..2)` returns `GenericGridDynamicGridSubsetElementSliceView`
    pub fn element<'a, I: GenericGridDynamicGridSubsetElementIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.element)
    }

    /// Access element mutably - use index for single element or range for slice view
    /// e.g. `.element_mut(0)` returns `&mut GenericGridDynamicGridSubsetElement`, `.element_mut(0..2)` returns `GenericGridDynamicGridSubsetElementSliceViewMut`
    pub fn element_mut<'a, I: GenericGridDynamicGridSubsetElementMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.element)
    }

    /// Get the number of element elements
    pub fn element_len(&self) -> usize {
        self.element.len()
    }
}

impl GenericGridDynamicGridSubset {
    /// Access base - use index for single element or range for slice view
    /// e.g. `.base(0)` returns `&GenericGridDynamicGridSubsetMetric`, `.base(0..2)` returns `GenericGridDynamicGridSubsetMetricSliceView`
    pub fn base<'a, I: GenericGridDynamicGridSubsetMetricIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.base)
    }

    /// Access base mutably - use index for single element or range for slice view
    /// e.g. `.base_mut(0)` returns `&mut GenericGridDynamicGridSubsetMetric`, `.base_mut(0..2)` returns `GenericGridDynamicGridSubsetMetricSliceViewMut`
    pub fn base_mut<'a, I: GenericGridDynamicGridSubsetMetricMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.base)
    }

    /// Get the number of base elements
    pub fn base_len(&self) -> usize {
        self.base.len()
    }
}

impl GenericGridDynamicSpaceDimension {
    /// Access object - use index for single element or range for slice view
    /// e.g. `.object(0)` returns `&GenericGridDynamicSpaceDimensionObject`, `.object(0..2)` returns `GenericGridDynamicSpaceDimensionObjectSliceView`
    pub fn object<'a, I: GenericGridDynamicSpaceDimensionObjectIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.object)
    }

    /// Access object mutably - use index for single element or range for slice view
    /// e.g. `.object_mut(0)` returns `&mut GenericGridDynamicSpaceDimensionObject`, `.object_mut(0..2)` returns `GenericGridDynamicSpaceDimensionObjectSliceViewMut`
    pub fn object_mut<'a, I: GenericGridDynamicSpaceDimensionObjectMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.object)
    }

    /// Get the number of object elements
    pub fn object_len(&self) -> usize {
        self.object.len()
    }
}

impl GenericGridDynamicGridSubsetElement {
    /// Access object - use index for single element or range for slice view
    /// e.g. `.object(0)` returns `&GenericGridDynamicGridSubsetElementObject`, `.object(0..2)` returns `GenericGridDynamicGridSubsetElementObjectSliceView`
    pub fn object<'a, I: GenericGridDynamicGridSubsetElementObjectIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.object)
    }

    /// Access object mutably - use index for single element or range for slice view
    /// e.g. `.object_mut(0)` returns `&mut GenericGridDynamicGridSubsetElementObject`, `.object_mut(0..2)` returns `GenericGridDynamicGridSubsetElementObjectSliceViewMut`
    pub fn object_mut<'a, I: GenericGridDynamicGridSubsetElementObjectMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.object)
    }

    /// Get the number of object elements
    pub fn object_len(&self) -> usize {
        self.object.len()
    }
}

impl GenericGridDynamicSpaceDimensionObject {
    /// Access boundary - use index for single element or range for slice view
    /// e.g. `.boundary(0)` returns `&GenericGridDynamicSpaceDimensionObjectBoundary`, `.boundary(0..2)` returns `GenericGridDynamicSpaceDimensionObjectBoundarySliceView`
    pub fn boundary<'a, I: GenericGridDynamicSpaceDimensionObjectBoundaryIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.boundary)
    }

    /// Access boundary mutably - use index for single element or range for slice view
    /// e.g. `.boundary_mut(0)` returns `&mut GenericGridDynamicSpaceDimensionObjectBoundary`, `.boundary_mut(0..2)` returns `GenericGridDynamicSpaceDimensionObjectBoundarySliceViewMut`
    pub fn boundary_mut<'a, I: GenericGridDynamicSpaceDimensionObjectBoundaryMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.boundary)
    }

    /// Get the number of boundary elements
    pub fn boundary_len(&self) -> usize {
        self.boundary.len()
    }
}

impl Wall {
    /// Access description_2d - use index for single element or range for slice view
    /// e.g. `.description_2d(0)` returns `&Wall2d`, `.description_2d(0..2)` returns `Wall2dSliceView`
    pub fn description_2d<'a, I: Wall2dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.description_2d)
    }

    /// Access description_2d mutably - use index for single element or range for slice view
    /// e.g. `.description_2d_mut(0)` returns `&mut Wall2d`, `.description_2d_mut(0..2)` returns `Wall2dSliceViewMut`
    pub fn description_2d_mut<'a, I: Wall2dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.description_2d)
    }

    /// Get the number of description_2d elements
    pub fn description_2d_len(&self) -> usize {
        self.description_2d.len()
    }
}

impl Wall {
    /// Access description_ggd - use index for single element or range for slice view
    /// e.g. `.description_ggd(0)` returns `&WallDescriptionGgd`, `.description_ggd(0..2)` returns `WallDescriptionGgdSliceView`
    pub fn description_ggd<'a, I: WallDescriptionGgdIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.description_ggd)
    }

    /// Access description_ggd mutably - use index for single element or range for slice view
    /// e.g. `.description_ggd_mut(0)` returns `&mut WallDescriptionGgd`, `.description_ggd_mut(0..2)` returns `WallDescriptionGgdSliceViewMut`
    pub fn description_ggd_mut<'a, I: WallDescriptionGgdMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.description_ggd)
    }

    /// Get the number of description_ggd elements
    pub fn description_ggd_len(&self) -> usize {
        self.description_ggd.len()
    }
}
