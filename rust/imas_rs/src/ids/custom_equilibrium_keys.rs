//! Custom (non-IMAS) keys added to the equilibrium IDS.
//!
//! This file is **not** compiled: it is deliberately absent from `mod.rs`. It is read by
//! `../../imas_updater/build_ids.py`, which splices the fields below into the matching
//! generated structs in `equilibrium.rs`. It is written in ordinary Rust syntax so that it
//! reads exactly like the generated file, and so editors can still parse it.
//!
//! Rust cannot add a field to a struct from a different file, so this splice at generation
//! time is what lets the keys sit flat alongside the IMAS ones (`profiles_2d.d_psi_d_r`,
//! not `profiles_2d.custom.d_psi_d_r`).
//!
//! To add a key: add the field to the struct below, matching the generated struct name
//! exactly, then re-run `build_ids.py`. A `pub struct` whose name is *not* a generated struct
//! declares a new nested structure instead (e.g. `EquilibriumBoundaryBounding`); it must be
//! referenced by some key's type, which is what still catches a mistyped struct name. Only the field name, type and `///` comments are
//! read; everything else here is ignored. Base types are written bare here (`FLT_2D`) and
//! come out wrapped (`Option<FLT_2D>`), exactly as the data dictionary's own leaves do, so
//! that an unset key is distinguishable from an empty one.
//!
//! Note that these keys have no IMAS counterpart, so they cannot be written to an IMAS
//! backend. Anything written out over the standard data dictionary must skip them.

use crate::dd_base_types::{FLT_0D, FLT_1D, FLT_2D, INT_0D, STR_0D};

/// Profiles along the horizontal line through the middle of the grid.
///
/// The row used is `floor(n_z / 2)`, which is the grid mid-row rather than the magnetic axis, so
/// this is a cut through the grid rather than through the plasma.
pub struct EquilibriumProfilesRMidplane {
    /// Major radius of each point along the mid-plane, which is the grid's own radial axis
    /// Units: m
    pub r: FLT_1D,
    /// Plasma pressure along the mid-plane. Zero outside the plasma boundary
    /// Units: Pa
    pub pressure: FLT_1D,
}

pub struct EquilibriumProfiles1d {
    /// Normalised poloidal flux radius, `sqrt(psi_norm)`. This is the poloidal counterpart of the
    /// data dictionary's `rho_tor_norm`, which the data dictionary itself does not define
    /// Units: dimensionless
    pub rho_pol: FLT_1D,
}

pub struct EquilibriumProfiles2d {
    /// Radial derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-1
    pub d_psi_d_r: FLT_2D,
    /// Vertical derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-1
    pub d_psi_d_z: FLT_2D,
    /// Second radial derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_r2: FLT_2D,
    /// Mixed second derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_r_d_z: FLT_2D,
    /// Second vertical derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_z2: FLT_2D,
    /// Mask for the grid in the poloidal plane
    /// Units: dimensionless
    pub mask: FLT_2D,
    /// Normalised poloidal flux on the grid, 0 at the magnetic axis and 1 at the plasma boundary
    pub psi_norm: FLT_2D,
    /// Contribution to the poloidal flux from the PF coils alone
    /// Units: Wb
    pub psi_coils: FLT_2D,
    /// Plasma pressure on the grid in the poloidal plane. Zero outside the plasma boundary
    /// Units: Pa
    pub pressure: FLT_2D,
    /// Vertical derivative of the vertical magnetic field on the grid in the poloidal plane. Used
    /// by the vertical stability control
    /// Units: T.m^-1
    pub d_b_field_z_d_z: FLT_2D,
}

pub struct EquilibriumTimeSlice {
    /// Source functions which parameterise the plasma current profile
    pub source_functions: EquilibriumSourceFunctions,
    /// Profiles along the horizontal line through the middle of the grid
    pub profiles_r_midplane: EquilibriumProfilesRMidplane,
    /// Scrape-off layer: the open field lines outside the last closed flux surface
    pub sol: EquilibriumSol,
    /// Fitted degrees of freedom of the passive structure currents
    /// Units: A
    pub passive_dof_values: FLT_1D,
}

pub struct EquilibriumGlobalQuantities {
    /// Current flowing in the central rod of the toroidal field coil. It sets the vacuum
    /// toroidal field function `f_vac = mu_0 * i_rod / (2 * pi)`, which the diamagnetic loop
    /// constraint is written against. Signed: a negative value is a reversed toroidal field
    /// Units: A
    pub i_rod: FLT_0D,
    /// Poloidal beta normalised to the flux-surface-averaged poloidal field:
    /// `2 * mu_0 * <p> / <<b_p ** 2>>`, where `<x>` is the volume average and `<<x>>` the
    /// flux-surface average
    pub beta_pol_1: FLT_0D,
    /// Poloidal beta normalised to the magnetic axis major radius:
    /// `4 * int(p dV) / (mu_0 * ip ** 2 * magnetic_axis/r)`
    pub beta_pol_2: FLT_0D,
    /// Poloidal beta normalised to the geometric major radius:
    /// `4 * int(p dV) / (mu_0 * ip ** 2 * boundary/geometric_axis/r)`
    ///
    /// This is the closest of the three to the data dictionary's own `beta_pol`, which uses the
    /// vacuum toroidal field reference radius `vacuum_toroidal_field/r0` instead
    pub beta_pol_3: FLT_0D,
    /// Vacuum toroidal magnetic field at the plasma geometric axis,
    /// `mu_0 * i_rod / (2 * pi * boundary/geometric_axis/r)`. Distinct from
    /// `vacuum_toroidal_field/b0`, which is evaluated at the fixed machine reference radius
    /// `vacuum_toroidal_field/r0` rather than following the plasma
    /// Units: T
    pub bt_vac_at_r_geo: FLT_0D,
    /// Internal inductance normalised to the flux-surface-averaged poloidal field:
    /// `<b_p ** 2> / <<b_p ** 2>>`, where `<x>` is the volume average and `<<x>>` the
    /// flux-surface average
    pub li_1: FLT_0D,
    /// Internal inductance normalised to the magnetic axis major radius:
    /// `2 * int(b_p ** 2 dV) / (mu_0 ** 2 * ip ** 2 * magnetic_axis/r)`
    ///
    /// The data dictionary's own `li_3` is the same quantity normalised to
    /// `boundary/geometric_axis/r` instead
    pub li_2: FLT_0D,
    /// Plain sum of `profiles_2d/pressure` over every grid cell. Not an integral: the cells are
    /// not weighted by their volume, so this is a diagnostic of the 2D pressure rather than a
    /// physical quantity. Kept because GSFit has always reported it as `global/p`
    /// Units: Pa
    pub pressure_2d_sum: FLT_0D,
    /// Loop voltage at the plasma boundary, `-d(boundary/psi)/d(time)`, by finite differences over
    /// the reconstruction times. Distinct from the data dictionary's `v_external`, which
    /// differentiates `psi_external_average` instead
    /// Units: V
    pub v_loop: FLT_0D,
}

pub struct EquilibriumBoundary {
    /// Point which defines the plasma boundary: the limiter point when the plasma is limited, or
    /// the X-point when it is diverted
    pub bounding: EquilibriumBoundaryBounding,
}

/// Point which defines the plasma boundary
pub struct EquilibriumBoundaryBounding {
    /// Major radius of the point which defines the plasma boundary
    /// Units: m
    pub r: FLT_0D,
    /// Height of the point which defines the plasma boundary
    /// Units: m
    pub z: FLT_0D,
}

pub struct EquilibriumConvergence {
    /// Vertical shift applied by the vertical feedback controller. Unset until the first inverse
    /// solve has run, and 0 while the vertical feedback is switched off
    /// Units: m
    pub delta_z: FLT_0D,
}

/// The IDS root. `greens` hangs off it rather than off `time_slice`, because the Greens tables
/// are geometry and so are the same for every time-slice.
pub struct Equilibrium {
    /// Greens tables: the poloidal flux and its derivatives per ampere in each current source
    pub greens: EquilibriumGreens,
}

pub struct Code {
    /// Maximum number of iterations the convergence loop is allowed to run for
    pub iterations_n_max: INT_0D,
    /// Minimum number of iterations before the convergence test is allowed to pass
    pub iterations_n_min: INT_0D,
    /// Number of initial iterations for which the vertical feedback is switched off
    pub iterations_n_no_vertical_feedback: INT_0D,
    /// Value of convergence/grad_shafranov_deviation_value below which the solution is taken as
    /// converged
    /// Units: mixed
    pub grad_shafranov_deviation_value_tolerance: FLT_0D,
    /// Initial guess for the plasma, used to seed the first iteration
    pub initial_guess: EquilibriumCodeInitialGuess,
}

/// Initial guess for the plasma, used to seed the first iteration.
///
/// The seed current distribution is an ellipse centred on (`cur_r`, `cur_z`) whose radial
/// semi-axis is `minor_radius` and whose vertical semi-axis is `minor_radius * elongation`,
/// carrying a total current of `ip`.
pub struct EquilibriumCodeInitialGuess {
    /// Initial total plasma current
    /// Units: A
    pub ip: FLT_0D,
    /// Radial centre of the initial current distribution
    /// Units: m
    pub cur_r: FLT_0D,
    /// Vertical centre of the initial current distribution
    /// Units: m
    pub cur_z: FLT_0D,
    /// Radial semi-axis of the initial current distribution
    /// Units: m
    pub minor_radius: FLT_0D,
    /// Elongation of the initial current distribution
    pub elongation: FLT_0D,
}

/// The grid each 2D profile is defined on
pub struct EquilibriumProfiles2dGrid {
    /// Area of one grid cell in the poloidal plane, `d_dim1 * d_dim2`. A single value, because
    /// the grid is rectangular and uniformly spaced in both directions
    /// Units: m^2
    pub d_area: FLT_0D,
}

/// Scrape-off layer, for a diverted plasma.
///
/// Traced from the active X-point outwards along the separatrix, one leg on each side of the
/// machine. Both legs are empty when the plasma is limited rather than diverted.
///
/// Stored per time-slice, so each leg is its own length. The `DataTree` equivalent packs every
/// time into one `(n_time, n_points)` array padded with NaN, and needs a separate `n` to say how
/// much of each row is real; here the array of structures over time does that job.
pub struct EquilibriumSol {
    /// High field side leg, on the inboard side of the machine
    pub hfs: EquilibriumSolLeg,
    /// Low field side leg, on the outboard side of the machine
    pub lfs: EquilibriumSolLeg,
}

/// One leg of the scrape-off layer
pub struct EquilibriumSolLeg {
    /// The field line itself, traced from the X-point to where it meets the wall
    pub contour: EquilibriumSolContour,
    /// Where the leg meets the wall, which is the last point of the contour
    pub strike_point: EquilibriumSolStrikePoint,
}

/// The path of one scrape-off layer leg through the poloidal plane
pub struct EquilibriumSolContour {
    /// Major radius of each point along the leg
    /// Units: m
    pub r: FLT_1D,
    /// Height of each point along the leg
    /// Units: m
    pub z: FLT_1D,
}

/// Where a scrape-off layer leg meets the wall
pub struct EquilibriumSolStrikePoint {
    /// Major radius of the strike point
    /// Units: m
    pub r: FLT_0D,
    /// Height of the strike point
    /// Units: m
    pub z: FLT_0D,
}

/// Source functions which parameterise the plasma current profile
pub struct EquilibriumSourceFunctions {
    /// The p' source function, dp/dpsi
    pub p_prime: EquilibriumSourceFunction,
    /// The FF' source function, F dF/dpsi
    pub ff_prime: EquilibriumSourceFunction,
}

/// A source function, as fitted by the equilibrium reconstruction
pub struct EquilibriumSourceFunction {
    /// Fitted degrees of freedom of the source function, in the basis the source function defines
    pub coefficients: FLT_1D,
}

/// Greens tables: the poloidal flux and its derivatives at every point of the plasma grid, per
/// ampere flowing in each current source.
///
/// These are geometry only - they depend on where the conductors and the grid are, never on the
/// plasma - so they are calculated once, before the first time-slice is solved, and are shared by
/// every time-slice. That is why they hang off the IDS root rather than off `time_slice`.
pub struct EquilibriumGreens {
    /// Active poloidal field coils, one entry per coil
    pub pf_active: Vec<EquilibriumGreensPfActive>,
    /// Passive conductors, one entry per conductor. Each carries its own degrees of freedom,
    /// because a passive is represented by a set of current distributions (e.g. eigenmodes)
    /// rather than by a single current
    pub pf_passive: Vec<EquilibriumGreensPfPassive>,
    /// The plasma grid onto itself: the flux at each grid point due to a unit current at each
    /// other grid point. Toroidal symmetry makes this a function of the vertical *offset* between
    /// the two points, not of their absolute heights, which is why only one vertical index is
    /// stored
    pub grid_grid: EquilibriumGreensGridGrid,
}

/// Greens table for one active poloidal field coil, summed over the coil's filaments
pub struct EquilibriumGreensPfActive {
    /// Name of the coil, e.g. `"BVL"`
    pub name: STR_0D,
    /// Poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1
    pub psi: FLT_2D,
    /// Radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1
    pub br: FLT_2D,
    /// Vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1
    pub bz: FLT_2D,
    /// Vertical derivative of the radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1.m^-1
    pub d_br_d_z: FLT_2D,
    /// Vertical derivative of the vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1.m^-1
    pub d_bz_d_z: FLT_2D,
    /// Radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_r: FLT_2D,
    /// Vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_z: FLT_2D,
    /// Second radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r2: FLT_2D,
    /// Mixed second derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r_d_z: FLT_2D,
    /// Second vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_z2: FLT_2D,
    /// Third derivative of the poloidal flux, twice by r, once by z at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r2_d_z: FLT_2D,
    /// Third derivative of the poloidal flux, once by r, twice by z at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r_d_z2: FLT_2D,
    /// Third vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_z3: FLT_2D,
}

/// Greens tables for one passive conductor
pub struct EquilibriumGreensPfPassive {
    /// Name of the passive conductor, e.g. `"IVC"`
    pub name: STR_0D,
    /// Degrees of freedom of this conductor, one entry per current distribution
    pub dof: Vec<EquilibriumGreensPfPassiveDof>,
}

/// Greens table for one degree of freedom of one passive conductor, summed over the filaments
/// weighted by that degree of freedom's current distribution
pub struct EquilibriumGreensPfPassiveDof {
    /// Name of the degree of freedom, e.g. `"EIG_01"`
    pub name: STR_0D,
    /// Poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1
    pub psi: FLT_1D,
    /// Radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: T.A^-1
    pub br: FLT_1D,
    /// Vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: T.A^-1
    pub bz: FLT_1D,
    /// Vertical derivative of the radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: T.A^-1.m^-1
    pub d_br_d_z: FLT_1D,
    /// Vertical derivative of the vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: T.A^-1.m^-1
    pub d_bz_d_z: FLT_1D,
    /// Radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_r: FLT_1D,
    /// Vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_z: FLT_1D,
    /// Second radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r2: FLT_1D,
    /// Mixed second derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r_d_z: FLT_1D,
    /// Second vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_z2: FLT_1D,
    /// Third derivative of the poloidal flux, twice by r, once by z at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r2_d_z: FLT_1D,
    /// Third derivative of the poloidal flux, once by r, twice by z at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r_d_z2: FLT_1D,
    /// Third vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_z3: FLT_1D,
}

/// Greens tables for the plasma grid onto itself
pub struct EquilibriumGreensGridGrid {
    /// Poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1
    pub psi: FLT_2D,
    /// Radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: T.A^-1
    pub br: FLT_2D,
    /// Vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: T.A^-1
    pub bz: FLT_2D,
    /// Vertical derivative of the radial magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: T.A^-1.m^-1
    pub d_br_d_z: FLT_2D,
    /// Vertical derivative of the vertical magnetic field at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: T.A^-1.m^-1
    pub d_bz_d_z: FLT_2D,
    /// Radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_r: FLT_2D,
    /// Vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_z: FLT_2D,
    /// Second radial derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r2: FLT_2D,
    /// Mixed second derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r_d_z: FLT_2D,
    /// Second vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_z2: FLT_2D,
    /// Third derivative of the poloidal flux, twice by r, once by z at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r2_d_z: FLT_2D,
    /// Third derivative of the poloidal flux, once by r, twice by z at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r_d_z2: FLT_2D,
    /// Third vertical derivative of the poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z * n_r, n_r)`, which unflattens to `(i_offset_z, i_r, i_current_r)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_z3: FLT_2D,
}
