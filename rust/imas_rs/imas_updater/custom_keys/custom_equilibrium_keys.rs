//! Custom (non-IMAS) keys added to the equilibrium IDS.
//!
//! This file is **not** compiled: it sits outside `src/`, as an input to
//! `../build_ids.py`, which splices the fields below into the matching generated structs in
//! `../../src/ids/equilibrium.rs`. It is written in ordinary Rust syntax so that it reads
//! exactly like the generated file, and so editors can still parse it.
//!
//! Rust cannot add a field to a struct from a different file, so this splice at generation
//! time is what lets the keys sit flat alongside the IMAS ones (`profiles_2d.d_psi_d_r`,
//! not `profiles_2d.custom.d_psi_d_r`).
//!
//! To add a key: add the field to the struct below, matching the generated struct name
//! exactly, then re-run `../build_ids.py`. A `pub struct` whose name is *not* a generated struct
//! declares a new nested structure instead (e.g. `EquilibriumBoundaryBounding`); it must be
//! referenced by some key's type, which is what still catches a mistyped struct name. Only the field name, type and `///` comments are
//! read; everything else here is ignored. Base types are written bare (`FLT_2D`), exactly as
//! the data dictionary's own leaves are generated, and an unset key reads as the IMAS empty
//! value (NaN, `EMPTY_INT`, or empty), just like they do.
//!
//! Note that these keys have no IMAS counterpart, so they cannot be written to an IMAS
//! backend. Anything written out over the standard data dictionary must skip them.

use crate::dd_base_types::{FLT_0D, FLT_1D, FLT_2D, FLT_3D, INT_0D, STR_0D};

/// Profiles along the horizontal line through the middle of the grid.
///
/// The row used is `floor(n_z / 2)`, which is the grid mid-row rather than the magnetic axis, so
/// this is a cut through the grid rather than through the plasma.
pub struct EquilibriumProfiles1dRMidplane {
    /// Major radius of each point along the mid-plane, which is the grid's own radial axis
    /// Units: m
    pub r: FLT_1D,
    /// Derivative of the pressure with respect to the poloidal flux, along the mid-plane. Zero
    /// outside the plasma boundary
    /// Units: Pa.Wb^-1
    pub dpressure_dpsi: FLT_1D,
    /// Diamagnetic function `f = R * b_field_phi` along the mid-plane. The vacuum value outside the
    /// plasma boundary, where no poloidal current flows
    /// Units: T.m
    pub f: FLT_1D,
    /// Derivative of `f` with respect to the poloidal flux, multiplied by `f`, along the mid-plane.
    /// Zero outside the plasma boundary
    /// Units: T^2.m^2.Wb^-1
    pub f_df_dpsi: FLT_1D,
    /// Toroidal plasma current density along the mid-plane. Zero outside the plasma boundary.
    /// Unlike `profiles_1d/j_phi`, which is a flux-surface average, this is a cut through the
    /// solved current density
    /// Units: A.m^-2
    pub j_phi: FLT_1D,
    /// Plasma pressure along the mid-plane. Zero outside the plasma boundary
    /// Units: Pa
    pub pressure: FLT_1D,
    /// Safety factor along the mid-plane. NaN outside the plasma boundary, where there is no closed
    /// flux surface to define it
    /// Units: dimensionless
    pub q: FLT_1D,
}

/// Profiles along the horizontal line through the magnetic axis, on a grid refined in `R`.
///
/// The purpose of this group is to map a normalised flux back to a major radius in the scrape-off
/// layer, where a few millimetres matters. The solved grid is far too coarse for that directly:
/// with `d_r` of 12.5 mm, reading `R` off the solved grid is worth about 6 mm, so `R` here is
/// refined by `N_R_SUBDIVISIONS_PER_CELL` and `psi` is filled by bicubic interpolation.
///
/// Unlike `profiles_1d_r_midplane`, which cuts the grid's own middle row, this cuts the row
/// through the magnetic axis, the same height `global_quantities/delta_r_sep` and
/// `global_quantities/f_x` use. That height moves between time-slices, so only `r` is the same on
/// every slice.
pub struct EquilibriumProfiles1dRMidplaneH {
    /// Major radius of each point along the refined mid-plane. The solved grid's radial axis
    /// subdivided by `N_R_SUBDIVISIONS_PER_CELL`, so every solved grid point is still one of
    /// these, and the same on every time-slice
    /// Units: m
    pub r: FLT_1D,
    /// Poloidal flux along the refined mid-plane, bicubically interpolated from the solved grid.
    /// NaN on a time-slice which did not converge
    /// Units: Wb
    pub psi: FLT_1D,
    /// Normalised poloidal flux along the refined mid-plane, `(psi - psi_axis) / (psi_boundary -
    /// psi_axis)`, so 0 at the magnetic axis and 1 at the plasma boundary.
    ///
    /// Unlike `profiles_2d/psi_norm` this is **not** masked to the plasma, which is the whole
    /// point of it: it carries on rising past 1 through the scrape-off layer. It is not monotonic
    /// across the full radial span, having a minimum at the magnetic axis, so inverting it for `R`
    /// means picking the inboard or outboard branch first
    /// Units: dimensionless
    pub psi_norm: FLT_1D,
}

pub struct EquilibriumProfiles1d {
    /// Major radius of each flux surface where it crosses `z = 0`, on the inboard side.
    ///
    /// The data dictionary's `r_inboard` is measured at the height of the magnetic axis, which
    /// moves between time-slices. This one is measured on the machine's mid-plane instead, so that
    /// a series of time-slices is a series of radii at one fixed height.
    ///
    /// The two sides are split at the major radius of the magnetic axis, which is the point every
    /// flux surface encloses. A surface which does not reach `z = 0`, or which does not enclose
    /// that point, is NaN
    /// Units: m
    pub r_inboard_z_0: FLT_1D,
    /// Major radius of each flux surface where it crosses `z = 0`, on the outboard side. The
    /// outboard counterpart of `r_inboard_z_0`
    /// Units: m
    pub r_outboard_z_0: FLT_1D,
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
    pub profiles_1d_r_midplane: EquilibriumProfiles1dRMidplane,
    /// Profiles along the horizontal line through the magnetic axis, on a grid refined in `R`
    pub profiles_1d_r_midplane_h: EquilibriumProfiles1dRMidplaneH,
    /// Scrape-off layer: the open field lines outside the last closed flux surface
    pub sol: EquilibriumSol,
    /// Fitted degrees of freedom of the passive structure currents
    /// Units: A
    pub passive_dof_values: FLT_1D,
}

pub struct EquilibriumGlobalQuantities {
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
    /// Magnitude of the poloidal magnetic field at the outboard midplane,
    /// `sqrt(b_field_r ** 2 + b_field_z ** 2)`, where the plasma boundary crosses the height of
    /// the magnetic axis on the outboard side. This is the same point `delta_r_sep` and `f_x`
    /// use. Evaluated from the solver's bicubic interpolation of `psi`, not from the gridded
    /// `profiles_2d/b_field_r` and `.../b_field_z`. NaN when the boundary does not cross that
    /// height within the grid
    /// Units: T
    pub bp_omp: FLT_0D,
    /// Vacuum toroidal magnetic field at the plasma geometric axis,
    /// `vacuum_toroidal_field/r0 * b0 / boundary/geometric_axis/r`. Distinct from
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
    /// Radial separation of the two separatrices at the height of the magnetic axis, on the
    /// outboard side: `r_outboard(psi at the lower X-point) - r_outboard(psi at the upper
    /// X-point)`. So it is negative for a lower single null, positive for an upper single null,
    /// and passes through zero at a connected double null. NaN when the plasma is limited, or
    /// when only one X-point was found
    ///
    /// It is evaluated in flux rather than by differencing two traced contours: the difference in
    /// `psi` between the two X-points is mapped to the outboard midplane through
    /// `d(psi)/d(r)` there. `psi` is stationary at an X-point, so this is insensitive to the
    /// X-point positions at second order, which is what makes a sub-millimetre answer meaningful
    /// on a centimetre grid
    /// Units: m
    pub delta_r_sep: FLT_0D,
    /// Flux expansion from the outboard midplane to the outboard strike point on the active
    /// X-point's last closed flux surface, `(r_omp * b_p_omp) / (r_strike * b_p_strike)`. The
    /// outboard midplane is at the height of the magnetic axis. This excludes the additional
    /// expansion along the target caused by the field-line incidence angle
    /// Units: dimensionless
    pub f_x: FLT_0D,
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
    /// Number of times the flux was calculated on the grid from the current density, each followed
    /// by the search for the magnetic axis and the plasma boundary. This is where nearly all the
    /// time goes: a Picard iteration does it once, a Newton iteration several times. Recorded
    /// whether or not the solve converged
    pub flux_evaluations_n: INT_0D,
}

pub struct EquilibriumConstraints {
    /// Set of Rogowski coils, in the same order as `magnetics/rogowski_coil`. The data dictionary has no constraint for them
    /// Units: A
    pub rogowski_coil: Vec<EquilibriumConstraints0dIpLike>,
}

/// The data dictionary uses this type for `time_slice(itime)/constraints/pf_current(i1)` and `time_slice(itime)/constraints/ip`,
/// and the custom key `time_slice(itime)/constraints/rogowski_coil(i1)` uses it too.
pub struct EquilibriumConstraints0dIpLike {
    /// Name of what was measured, matching its name in the source IDS, e.g. `pf_active/coil(i1)/name` for
    /// `constraints/pf_current(i1)`, `magnetics/rogowski_coil(i1)/name` for `constraints/rogowski_coil(i1)`, and
    /// `magnetics/ip(0)/method_name` for `constraints/ip`
    pub name: STR_0D,
}

/// The data dictionary uses this type for `time_slice(itime)/constraints/b_field_pol_probe(i1)` only.
pub struct EquilibriumConstraints0dOneLike {
    /// Name of the poloidal field probe, matching `magnetics/b_field_pol_probe(i1)/name`, e.g. `"P101"`
    pub name: STR_0D,
}

/// The data dictionary uses this type for `time_slice(itime)/constraints/diamagnetic_flux` only.
pub struct EquilibriumConstraints0dB0Like {
    /// Name of the method the diamagnetic flux is calculated with, matching `magnetics/diamagnetic_flux(0)/method_name`,
    /// e.g. `"DIALOOP"`
    pub name: STR_0D,
}

/// The keys below are for `time_slice(itime)/constraints/pf_passive_current(i1)`, the current in `pf_passive/loop(i1)`.
/// GSFit does not measure that current: it fits it, as an amplitude for each of the loop's degrees of freedom, and the
/// regularisations are what constrain the fit.
///
/// The data dictionary also uses this type for `b_field_tor_vacuum_r`, `faraday_angle(i1)`, `flux_loop(i1)`,
/// `iron_core_segment(i1)/magnetization_r` and `.../magnetization_z`, `mse_polarization_angle(i1)` and `n_e_line(i1)`, so the
/// keys below are on all of those too. Of those, only `flux_loop(i1)/name` is filled; the rest are left empty.
pub struct EquilibriumConstraints0d {
    /// Name of what was measured or fitted, matching its name in the source IDS, e.g. `pf_passive/loop(i1)/name` for
    /// `constraints/pf_passive_current(i1)`, and `magnetics/flux_loop(i1)/name` for `constraints/flux_loop(i1)`
    pub name: STR_0D,
    /// The degrees of freedom the passive loop's current is fitted with, `pf_passive/loop(i1)/dof`, in the same order
    pub dof: Vec<EquilibriumConstraints0dDof>,
    /// The regularisations of the degrees of freedom, `pf_passive/loop(i1)/regularisations`: one row of coefficients for each
    /// regularisation, one coefficient for each `dof`; no rows for none.
    /// Shape = `(n_regularisations, n_dof)`
    pub regularisations: FLT_2D,
    /// The weight of each regularisation in the fit, `pf_passive/loop(i1)/regularisations_weight`, one for each row of
    /// `regularisations`
    /// Units: dimensionless
    pub regularisations_weight: FLT_1D,
}

/// One degree of freedom of a passive loop's current, a copy of `pf_passive/loop(i1)/dof(i2)`
pub struct EquilibriumConstraints0dDof {
    /// Name of the degree of freedom, `pf_passive/loop(i1)/dof(i2)/name`, e.g. `"eig_01"`
    pub name: STR_0D,
}

/// The IDS root. `greens` and `limiter` hang off it rather than off `time_slice`, because they are
/// geometry and so are the same for every time-slice.
pub struct Equilibrium {
    /// Greens tables: what each observer, the grid or a magnetic sensor, sees per ampere in each current source
    pub greens: EquilibriumGreens,
    /// The limiter: the immobile surfaces which bound the plasma. A copy of the `wall` IDS's
    /// `wall/description_2d(0)/limiter`, made by `Plasma.copy_from_wall_ids`, so that the solver and
    /// the post-processor read it from the equilibrium IDS rather than from the `wall` IDS
    pub limiter: EquilibriumLimiter,
}

pub struct Code {
    /// GSFit's settings, `code/parameters`, read into structs. The XML of `code/parameters` is
    /// `equilibrium.xml` with each file it links to written in its link's place, so each element
    /// is the key of the same name, e.g. `<grid><n_r>` is `code/parameters_parsed/grid/n_r`. Its
    /// structs are generated from GSFit's XML schema of the settings,
    /// `python/gsfit/settings_schema/xsd/equilibrium/equilibrium.xsd`, rather than declared here
    pub parameters_parsed: EquilibriumCodeParametersParsed,
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

/// Greens tables: what each observer sees, per ampere flowing in each current source.
///
/// These are geometry only - they depend on where the observers and the conductors are, never on the
/// plasma - so they are calculated once, before the first time-slice is solved, and are shared by
/// every time-slice. That is why they hang off the IDS root rather than off `time_slice`.
///
/// They are arranged observer first, then source: `greens/<observer>/<source>`. This is the order of the data
/// dictionary's `em_coupling` IDS, "Quantity_Sensor_Source". The observers are the (R, Z) grid and the magnetic
/// sensors, and every observer has the same three sources:
/// * `plasma` - the plasma current in each grid cell
/// * `pf_active` - the PF coils, in the order of `pf_active/coil`
/// * `pf_passive` - the passive conductors' degrees of freedom, in the order of `pf_passive/loop`, and then of
///   `loop(i1)/dof`
///
/// Each magnetic sensor has one entry, with the same name and in the same order as in `magnetics/<sensor>(i1)` and
/// `time_slice(itime)/constraints/<sensor>(i1)`, so a sensor's tables are found by the name of its constraint. Its
/// tables with the PF coils and the passive degrees of freedom are arrays over them, in the order of
/// `greens/grid/pf_active` and `greens/grid/pf_passive`, which name them.
pub struct EquilibriumGreens {
    /// Observer: the (R, Z) grid
    pub grid: EquilibriumGreensGrid,
    /// Observer: each poloidal field probe, in the order of `magnetics/b_field_pol_probe`
    pub b_field_pol_probe: Vec<EquilibriumGreensBFieldPolProbe>,
    /// Observer: each flux loop, in the order of `magnetics/flux_loop`
    pub flux_loop: Vec<EquilibriumGreensFluxLoop>,
    /// Observer: each Rogowski coil, in the order of `magnetics/rogowski_coil`
    pub rogowski_coil: Vec<EquilibriumGreensRogowskiCoil>,
}

/// Observer: the (R, Z) grid. The poloidal flux and its derivatives at every grid point, per ampere in each current source
pub struct EquilibriumGreensGrid {
    /// Source: the plasma current in each grid cell
    pub plasma: EquilibriumGreensGridPlasma,
    /// Source: each PF coil, in the order of `pf_active/coil`. Every sensor's `pf_active` is in this order too
    pub pf_active: Vec<EquilibriumGreensGridPfActive>,
    /// Source: each passive conductor, in the order of `pf_passive/loop`, with its degrees of freedom. Every sensor's
    /// `pf_passive`, and `time_slice(itime)/passive_dof_values`, are in this order too
    pub pf_passive: Vec<EquilibriumGreensGridPfPassive>,
}

/// The grid onto itself: the flux at each grid point due to a unit current in each grid cell. Toroidal symmetry makes this a
/// function of the vertical *offset* between the two points, not of their absolute heights, so it is only stored for a source
/// in the lowest row, and an observer `i_offset_z` rows above it
pub struct EquilibriumGreensGridPlasma {
    /// Poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1
    pub psi: FLT_3D,
    /// Radial derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_r: FLT_3D,
    /// Vertical derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-1
    pub d_psi_d_z: FLT_3D,
    /// Second radial derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r2: FLT_3D,
    /// Mixed second derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_r_d_z: FLT_3D,
    /// Second vertical derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-2
    pub d2_psi_d_z2: FLT_3D,
    /// Third derivative of the poloidal flux, twice by r, once by z, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r2_d_z: FLT_3D,
    /// Third derivative of the poloidal flux, once by r, twice by z, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_r_d_z2: FLT_3D,
    /// Third vertical derivative of the poloidal flux, per ampere in the source.
    /// Shape = `(n_z, n_r, n_r)`, which is `(i_offset_z, i_r, i_r_source)`
    /// Units: Wb.A^-1.m^-3
    pub d3_psi_d_z3: FLT_3D,
}

/// Source: one PF coil, summed over the coil's filaments
pub struct EquilibriumGreensGridPfActive {
    /// Name of the coil, matching `pf_active/coil(i1)/name`, e.g. `"BVL"`
    pub name: STR_0D,
    /// Poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1
    pub psi: FLT_2D,
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

/// Source: one passive conductor, with one entry for each of its degrees of freedom
pub struct EquilibriumGreensGridPfPassive {
    /// Name of the passive conductor, matching `pf_passive/loop(i1)/name`, e.g. `"IVC"`
    pub name: STR_0D,
    /// Degrees of freedom of this conductor, one entry per current distribution, in the order of `pf_passive/loop(i1)/dof`
    pub dof: Vec<EquilibriumGreensGridPfPassiveDof>,
}

/// Source: one degree of freedom of one passive conductor, summed over the filaments weighted by that degree of freedom's
/// current distribution
pub struct EquilibriumGreensGridPfPassiveDof {
    /// Name of the degree of freedom, matching `pf_passive/loop(i1)/dof(i2)/name`, e.g. `"eig_01"`
    pub name: STR_0D,
    /// Poloidal flux at every grid point, per ampere in the source.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1
    pub psi: FLT_2D,
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

/// Observer: one poloidal field probe, `magnetics/b_field_pol_probe(i1)`. Each table is the field the probe measures, along its
/// normal, per ampere in the source
pub struct EquilibriumGreensBFieldPolProbe {
    /// Name of the probe, matching `magnetics/b_field_pol_probe(i1)/name` and `time_slice(itime)/constraints/b_field_pol_probe(i1)/name`,
    /// e.g. `"P101"`
    pub name: STR_0D,
    /// Source: the plasma current in each grid cell.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1
    pub plasma: FLT_2D,
    /// Source: the plasma current in each grid cell. The derivative of `plasma` with respect to the probe's height, for the change
    /// in the field with the vertical shift `time_slice(itime)/convergence/delta_z`.
    /// Shape = `(n_z, n_r)`
    /// Units: T.A^-1.m^-1
    pub d_plasma_d_z: FLT_2D,
    /// Source: each PF coil, in the order of `greens/grid/pf_active`
    /// Units: T.A^-1
    pub pf_active: FLT_1D,
    /// Source: each passive degree of freedom, in the order of `greens/grid/pf_passive(:)/dof(:)`
    /// Units: T.A^-1
    pub pf_passive: FLT_1D,
}

/// Observer: one flux loop, `magnetics/flux_loop(i1)`. Each table is the poloidal flux through the loop, GSFit's `psi`, per ampere
/// in the source
pub struct EquilibriumGreensFluxLoop {
    /// Name of the flux loop, matching `magnetics/flux_loop(i1)/name` and `time_slice(itime)/constraints/flux_loop(i1)/name`, e.g.
    /// `"L001"`
    pub name: STR_0D,
    /// Source: the plasma current in each grid cell.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1
    pub plasma: FLT_2D,
    /// Source: the plasma current in each grid cell. The derivative of `plasma` with respect to the loop's height, for the change
    /// in the flux with the vertical shift `time_slice(itime)/convergence/delta_z`.
    /// Shape = `(n_z, n_r)`
    /// Units: Wb.A^-1.m^-1
    pub d_plasma_d_z: FLT_2D,
    /// Source: each PF coil, in the order of `greens/grid/pf_active`
    /// Units: Wb.A^-1
    pub pf_active: FLT_1D,
    /// Source: each passive degree of freedom, in the order of `greens/grid/pf_passive(:)/dof(:)`
    /// Units: Wb.A^-1
    pub pf_passive: FLT_1D,
}

/// Observer: one Rogowski coil, `magnetics/rogowski_coil(i1)`. Each table is the current the coil measures, per ampere in the
/// source: the current inside the coil's contour, less `integral(b . d_l) / mu_0` along each of its gaps, from the gap's start
/// to its end
pub struct EquilibriumGreensRogowskiCoil {
    /// Name of the Rogowski coil, matching `magnetics/rogowski_coil(i1)/name` and
    /// `time_slice(itime)/constraints/rogowski_coil(i1)/name`, e.g. `"INIVC000"`
    pub name: STR_0D,
    /// Source: the plasma current in each grid cell.
    /// Shape = `(n_z, n_r)`
    /// Units: A.A^-1
    pub plasma: FLT_2D,
    /// Source: the plasma current in each grid cell. The derivative of `plasma` with respect to the coil's height, for the change
    /// in the current with the vertical shift `time_slice(itime)/convergence/delta_z`. Only the gaps contribute to it.
    /// Shape = `(n_z, n_r)`
    /// Units: A.A^-1.m^-1
    pub d_plasma_d_z: FLT_2D,
    /// Source: each PF coil, in the order of `greens/grid/pf_active`
    /// Units: A.A^-1
    pub pf_active: FLT_1D,
    /// Source: each passive degree of freedom, in the order of `greens/grid/pf_passive(:)/dof(:)`
    /// Units: A.A^-1
    pub pf_passive: FLT_1D,
}

/// The limiter: the immobile surfaces which bound the plasma.
///
/// A copy of the `wall` IDS's `wall/description_2d(0)/limiter`, made by `Plasma.copy_from_wall_ids`,
/// with the same layout, so that `unit(i)` here is `unit(i)` there. The data dictionary has no place
/// for the limiter in the equilibrium IDS, but an equilibrium cannot be found without one.
///
/// Like `greens`, it is geometry, so it is the same for every time-slice. That is why it hangs off the
/// IDS root rather than off `time_slice`.
pub struct EquilibriumLimiter {
    /// Limiter units, in the same order as `wall/description_2d(0)/limiter/unit`. The order is part
    /// of the contract: `unit(0)` is the vacuum vessel contour, the closed polygon bounding the region
    /// the plasma is allowed to occupy, and only `unit(0)` is used as the vessel. `unit(1)`,
    /// `unit(2)`, ... are other plasma facing components, such as tiles which protrude inside the
    /// vessel contour. The points of every unit are candidate limit points, so a plasma may be
    /// limited on the vessel or on any of the tiles
    pub unit: Vec<EquilibriumLimiterUnit>,
}

/// One limiter unit, a copy of `wall/description_2d(0)/limiter/unit(i)`
pub struct EquilibriumLimiterUnit {
    /// Short identifier for the unit, e.g. `"vacuum_vessel"`
    pub name: STR_0D,
    /// Outline of the limiting surface
    pub outline: EquilibriumLimiterUnitOutline,
}

/// Outline of one limiter unit, a copy of `wall/description_2d(0)/limiter/unit(i)/outline`
pub struct EquilibriumLimiterUnitOutline {
    /// Major radius of each point along the outline
    /// Units: m
    pub r: FLT_1D,
    /// Height of each point along the outline
    /// Units: m
    pub z: FLT_1D,
}
