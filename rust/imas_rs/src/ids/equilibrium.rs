//! IMAS Equilibrium IDS
//!
//! This module defines the equilibrium Interface Data Structure (IDS)
//! Auto-generated from IMAS Data Dictionary XSD schema.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::dd_base_types::{Accumulator, FLT_0D, FLT_1D, FLT_2D, FLT_3D, FLT_4D, INT_0D, INT_1D, INT_2D, STR_0D, StringAccumulator};

// ============================================================================
// Complex Types
// ============================================================================

/// Node description for critical points and limiter points
#[derive(Debug, Clone, Default)]
pub struct EquilibriumContourTreeNode {
    /// Critical-point type of the poloidal flux: 0 = local minimum, 1 = saddle (X-point), 2 = local maximum, 3 = limiter point. Whether the magnetic axis is a minimum or maximum of psi depends on the sign of the plasma current. A limiter point represents a point on the first wall where the plasma boundary makes contact. Limiter point nodes are inserted by splitting an existing tree edge at the limiter point's psi level; they have degree 2 in the tree. A limiter point is valid only if it belongs to the same connected component as at least one O-point in the subgraph obtained by removing all saddle nodes (critical_type = 1). Limiter points store the last closed flux surface passing through the limiter point in their levelset
    pub critical_type: Option<INT_0D>,
    /// Identifies whether this node represents a confined plasma feature or a vacuum field feature. Only meaningful for O-point nodes (critical_type 0 or 2). Limiter points are always plasma features. X-points and saddle points do not require classification
    pub node_type: IdentifierDynamicAos3,
    /// Major radius
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Value of the poloidal flux at the node location. Whether psi increases or decreases from the magnetic axis outward depends on the sign of the plasma current. All ordering rules in the contour tree (node ordering, levelset segment ordering) are defined in terms of normalised poloidal flux, psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis), which always increases from 0 at the magnetic axis to 1 at the plasma boundary
    /// Units: Wb
    pub psi: Option<FLT_0D>,
    /// Poloidal flux contour segments at the node's psi value that are topologically connected to the node. Each element of the array of structures stores one distinct contour segment. Only segments whose contour passes through the node are included; disconnected contours that share the same psi value but are spatially remote must be excluded. For O-points (flux extrema), no contour passes through the extremum and this field must contain a single entry with empty r and z arrays. For limiter points, the levelset contains the last closed flux surface (LCFS) that passes through the limiter point. For X-points, the segments are ordered as follows: segment 0 is the last closed flux surface (LCFS), the closed contour bounding the confined plasma that passes through the X-point. Segments 1..N are divertor or scrape-off layer leg contours, ordered by their departure angle from the X-point measured counterclockwise from the outboard midplane (theta=0 at R_max, Z of the X-point) in the standard (R, Z) poloidal plane. A standard single-null divertor (SND) X-point has 2 segments: the LCFS and the divertor legs. A double-null divertor (DND) has 2 segments per X-point; the primary X-point (lower psi_norm, closer to the magnetic axis) is stored at a lower node index than the secondary, so their levelsets are unambiguously associated. An exact snowflake, where two X-points merge into a single higher-order null with 6 separatrix branches, produces 3 segments: the LCFS and 2 distinct divertor channel contours.
    pub levelset: Vec<Rz1dDynamicAos>,
}

/// A structure to store the location, value, and connectivity of poloidal flux critical points and limiter points
#[derive(Debug, Clone, Default)]
pub struct EquilibriumContourTree {
    /// Set of critical points and limiter points of the poloidal flux. Each node is defined by its critical type (see critical_type), node type (see node_type), and position within the poloidal plane. All ordering rules below are defined in terms of normalised poloidal flux, psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis), which is 0 at the magnetic axis and 1 at the plasma boundary. Nodes are partitioned into two groups: critical points (critical_type 0, 1, 2) occupy the leading indices, followed by limiter points (critical_type 3). Within each partition, nodes are ordered by ascending psi_norm; ties are broken by ascending R, then ascending Z. Node 0 must contain the primary O-point (magnetic axis), which has psi_norm = 0. Node 1 must contain the primary X-point, defined as the X-point with the lowest psi_norm value. For double-null divertor (DND) configurations, the secondary X-point has a higher psi_norm value and must be stored at a higher node index than the primary; the two psi_norm values are always distinct in practice. For limiter-bounded plasmas where no X-point defines the LCFS, node 1 should contain the X-point with the lowest psi_norm value (nearest to the plasma boundary), if any X-point exists. If no X-points exist, nodes are ordered by ascending psi_norm starting from node 1. For doublet or multi-region plasmas, each confined plasma region has its own O-point; additional O-points follow the same ascending psi_norm ordering. When limiter points are absent, only the critical point partition is present and the ordering is unchanged from earlier versions. The node_type identifier distinguishes O-points that represent confined plasma regions (node_type = plasma) from vacuum field extrema (node_type = vacuum).
    pub node: Vec<EquilibriumContourTreeNode>,
    /// Edges encode the Reeb graph (contour tree) of the poloidal flux by connecting topologically adjacent nodes. For each edge (1st dimension), the indices of the two connected nodes are listed (indices referring to the ../node array). An edge (i, j) indicates that sweeping the contour level continuously between the psi values at nodes i and j encounters no other critical point or limiter point. Limiter point nodes participate in edges like any other node; they are inserted by splitting an existing edge at the limiter point's psi level, replacing one edge with two. The number of edges equals n_nodes minus the number of connected components of the graph. For simply-connected 2D domains the graph is a tree with n_edges = n_nodes - 1. For doublet or multi-island configurations the graph may be a forest of disconnected trees. Each edge pair must be stored with the lower node index first (edges[k, 0] < edges[k, 1]).
    pub edges: Option<INT_2D>,
}

/// Gap for describing the plasma boundary
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGap {
    /// Short string identifier (unique for a given device)
    pub name: Option<STR_0D>,
    /// Description, e.g. mid-plane gap
    pub description: Option<STR_0D>,
    /// Major radius of the reference point
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height of the reference point
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Angle measured clockwise from radial cylindrical vector (grad R) to gap vector (pointing away from reference point)
    /// Units: rad
    pub angle: Option<FLT_0D>,
    /// Value of the gap, i.e. distance between the reference point and the separatrix along the gap direction
    /// Units: m
    pub value: Option<FLT_0D>,
}

/// Structure for list of R, Z positions (1D list of Npoints, dynamic within a type 3 array of structures (index on time)), with coordinates referring to profiles_1d/psi
#[derive(Debug, Clone, Default)]
pub struct EquilibriumProfiles1dRz1dDynamicAos {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_1D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_1D>,
}

/// Convergence details for the equilibrium calculation
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConvergence {
    /// Number of iterations carried out in the convergence loop
    pub iterations_n: Option<INT_0D>,
    /// Expression for calculating the residual deviation between the left and right hand side of the Grad Shafranov equation
    pub grad_shafranov_deviation_expression: IdentifierDynamicAos3,
    /// Value of the residual deviation between the left and right hand side of the Grad Shafranov equation, evaluated as per grad_shafranov_deviation_expression
    /// Units: mixed
    pub grad_shafranov_deviation_value: Option<FLT_0D>,
    /// Convergence result
    pub result: IdentifierDynamicAos3,
    /// Vertical shift applied by the vertical feedback controller. Unset until the first inverse
    /// solve has run, and 0 while the vertical feedback is switched off
    /// Units: m
    pub delta_z: Option<FLT_0D>,
}

/// Position and distance to the plasma boundary of the point of the first wall which is the closest to plasma boundary
#[derive(Debug, Clone, Default)]
pub struct EquilibriumBoundaryClosest {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Distance to the plasma boundary
    /// Units: m
    pub distance: Option<FLT_0D>,
}

/// Geometry of the plasma boundary
#[derive(Debug, Clone, Default)]
pub struct EquilibriumBoundary {
    /// 0 (limiter) or 1 (diverted)
    pub r#type: Option<INT_0D>,
    /// RZ outline of the plasma boundary
    pub outline: Rz1dDynamicAos,
    /// Value of the normalized poloidal flux at which the boundary is taken, the flux being normalized to its value at the separatrix (so psi_norm = 1 if the boundary is the separatrix)
    pub psi_norm: Option<FLT_0D>,
    /// Value of the poloidal flux at which the boundary is taken. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi: Option<FLT_0D>,
    /// RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)
    pub geometric_axis: Rz0dDynamicAos,
    /// Minor radius of the plasma boundary (defined as (Rmax-Rmin) / 2 of the boundary)
    /// Units: m
    pub minor_radius: Option<FLT_0D>,
    /// Elongation of the plasma boundary
    pub elongation: Option<FLT_0D>,
    /// Triangularity of the plasma boundary
    pub triangularity: Option<FLT_0D>,
    /// Upper triangularity of the plasma boundary
    pub triangularity_upper: Option<FLT_0D>,
    /// Lower triangularity of the plasma boundary
    pub triangularity_lower: Option<FLT_0D>,
    /// Upper inner squareness of the plasma boundary (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_upper_inner: Option<FLT_0D>,
    /// Upper outer squareness of the plasma boundary (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_upper_outer: Option<FLT_0D>,
    /// Lower inner squareness of the plasma boundary (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_lower_inner: Option<FLT_0D>,
    /// Lower outer squareness of the plasma boundary (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_lower_outer: Option<FLT_0D>,
    /// Position and distance to the plasma boundary of the point of the first wall which is the closest to plasma boundary
    pub closest_wall_point: EquilibriumBoundaryClosest,
    /// Outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen.
    pub dr_dz_zero_point: Rz0dDynamicAos,
    /// Set of gaps, defined by a reference point and a direction.
    pub gap: Vec<EquilibriumGap>,
    /// Toroidal flux coordinate at the selected plasma boundary
    /// Units: m
    pub rho_tor: Option<FLT_0D>,
    /// Toroidal flux at the selected plasma boundary. Positive when the toroidal magnetic field is counter-clockwise when viewed from above
    /// Units: Wb
    pub phi: Option<FLT_0D>,
    /// Toroidal flux at the selected plasma boundary generated by the plasma poloidal current. Positive when the toroidal magnetic field is counter-clockwise when viewed from above
    /// Units: Wb
    pub phi_poloidal_current: Option<FLT_0D>,
    /// Point which defines the plasma boundary: the limiter point when the plasma is limited, or
    /// the X-point when it is diverted
    pub bounding: EquilibriumBoundaryBounding,
}

/// R, Z, and vertical velocity of current centre, dynamic within a type 3 array of structure (index on time)
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGlobalQuantitiesCurrentCentre {
    /// Major radius of the current center, defined as integral over the poloidal cross section of (j_tor*r*dS) / Ip
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height of the current center, defined as integral over the poloidal cross section of (j_tor*z*dS) / Ip
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Vertical velocity of the current center
    /// Units: m.s^-1
    pub velocity_z: Option<FLT_0D>,
}

/// R, Z, and Btor at magnetic axis, dynamic within a type 3 array of structure (index on time)
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGlobalQuantitiesMagneticAxis {
    /// Major radius of the magnetic axis
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height of the magnetic axis
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Total toroidal magnetic field at the magnetic axis. Positive sign means counter-clockwise when viewed from above
    /// Units: T
    pub b_field_phi: Option<FLT_0D>,
}

/// Position and value of q_min
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGlobalQuantitiesQmin {
    /// Minimum q value. Positive when toroidal current and toroidal magnetic field are in the same direction
    pub value: Option<FLT_0D>,
    /// Minimum q position in normalized toroidal flux coordinate
    pub rho_tor_norm: Option<FLT_0D>,
    /// Minimum q position in normalised poloidal flux
    pub psi_norm: Option<FLT_0D>,
    /// Minimum q position in poloidal flux. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi: Option<FLT_0D>,
}

/// 0D parameters of the equilibrium
#[derive(Debug, Clone, Default)]
pub struct EqulibriumGlobalQuantities {
    /// Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]
    pub beta_pol: Option<FLT_0D>,
    /// Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2
    pub beta_tor: Option<FLT_0D>,
    /// Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]
    pub beta_tor_norm: Option<FLT_0D>,
    /// Plasma current (toroidal component). Positive sign means counter-clockwise when viewed from above.
    /// Units: A
    pub ip: Option<FLT_0D>,
    /// Internal inductance
    pub li_3: Option<FLT_0D>,
    /// Total plasma volume
    /// Units: m^3
    pub volume: Option<FLT_0D>,
    /// Area of the LCFS poloidal cross section
    /// Units: m^2
    pub area: Option<FLT_0D>,
    /// Surface area of the toroidal flux surface
    /// Units: m^2
    pub surface: Option<FLT_0D>,
    /// Poloidal length of the magnetic surface
    /// Units: m
    pub length_pol: Option<FLT_0D>,
    /// Poloidal flux at the magnetic axis. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi_magnetic_axis: Option<FLT_0D>,
    /// Magnetic axis position and toroidal field
    pub magnetic_axis: EquilibriumGlobalQuantitiesMagneticAxis,
    /// Position and vertical velocity of the current centre
    pub current_centre: EquilibriumGlobalQuantitiesCurrentCentre,
    /// q at the magnetic axis. Positive when toroidal current and toroidal magnetic field are in the same direction
    pub q_axis: Option<FLT_0D>,
    /// q at the 95% poloidal flux surface (only positive when toroidal current and magnetic field are in same direction)
    pub q_95: Option<FLT_0D>,
    /// Minimum q value and position
    pub q_min: EquilibriumGlobalQuantitiesQmin,
    /// Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar
    /// Units: J
    pub energy_mhd: Option<FLT_0D>,
    /// Average (over the plasma poloidal cross section) plasma poloidal magnetic flux produced by all external circuits (CS and PF coils, eddy currents, VS in-vessel coils), given by the following formula : int(psi_external.j_tor.dS) / Ip. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi_external_average: Option<FLT_0D>,
    /// External voltage, i.e. time derivative of psi_external_average (with a minus sign : - d_psi_external_average/d_time)
    /// Units: V
    pub v_external: Option<FLT_0D>,
    /// Plasma inductance 2 E_magnetic/Ip^2, where E_magnetic = 1/2 * int(psi.j_tor.dS) (integral over the plasma poloidal cross-section)
    /// Units: H
    pub plasma_inductance: Option<FLT_0D>,
    /// Plasma resistance = int(e_field.j.dV) / Ip^2
    /// Units: ohm
    pub plasma_resistance: Option<FLT_0D>,
    /// Major radius of the upper X-point
    /// Units: m
    pub xpt_upper_r: Option<FLT_0D>,
    /// Height of the upper X-point
    /// Units: m
    pub xpt_upper_z: Option<FLT_0D>,
    /// Major radius of the lower X-point
    /// Units: m
    pub xpt_lower_r: Option<FLT_0D>,
    /// Height of the lower X-point
    /// Units: m
    pub xpt_lower_z: Option<FLT_0D>,
}

/// R,Z position constraint
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraintsPurePosition {
    /// Measured or estimated position
    pub position_measured: Rz0dDynamicAos,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: m
    pub sigma: Option<FLT_0D>,
    /// Position estimated from the reconstructed equilibrium
    pub position_reconstructed: Rz0dDynamicAos,
    /// Squared error on the major radius normalized by the variance considered in the minimization process : chi_squared = weight^2 *(position_reconstructed/r - position_measured/r)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    /// Units: m^-2
    pub chi_squared_r: Option<FLT_0D>,
    /// Squared error on the altitude normalized by the variance considered in the minimization process : chi_squared = weight^2 *(position_reconstructed/z - position_measured/z)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    /// Units: m^-2
    pub chi_squared_z: Option<FLT_0D>,
}

/// Scalar constraint with R,Z,phi position
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints0dPosition {
    /// Measured value
    /// Units: as_parent
    pub measured: Option<FLT_0D>,
    /// Position at which this measurement is given
    pub position: Rphizpsirho0dDynamicAos3,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: as_parent
    pub sigma: Option<FLT_0D>,
    /// Value calculated from the reconstructed equilibrium
    /// Units: as_parent
    pub reconstructed: Option<FLT_0D>,
    /// Squared error normalized by the variance considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    /// Units: as_parent
    pub chi_squared: Option<FLT_0D>,
}

/// Scalar constraint with toroidal-field-like sign convention
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints0dB0Like {
    /// Measured value
    /// Units: as_parent
    pub measured: Option<FLT_0D>,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: as_parent
    pub sigma: Option<FLT_0D>,
    /// Value calculated from the reconstructed equilibrium
    /// Units: as_parent
    pub reconstructed: Option<FLT_0D>,
    /// Squared error normalized by the variance considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    pub chi_squared: Option<FLT_0D>,
}

/// Scalar constraint with plasma-current-like sign convention
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints0dIpLike {
    /// Measured value
    /// Units: as_parent
    pub measured: Option<FLT_0D>,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: as_parent
    pub sigma: Option<FLT_0D>,
    /// Value calculated from the reconstructed equilibrium
    /// Units: as_parent
    pub reconstructed: Option<FLT_0D>,
    /// Squared error normalized by the variance considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    pub chi_squared: Option<FLT_0D>,
}

/// Scalar constraint with geometry-dependent sign convention
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints0dOneLike {
    /// Measured value
    /// Units: as_parent
    pub measured: Option<FLT_0D>,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: as_parent
    pub sigma: Option<FLT_0D>,
    /// Value calculated from the reconstructed equilibrium
    /// Units: as_parent
    pub reconstructed: Option<FLT_0D>,
    /// Squared error normalized by the variance considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    pub chi_squared: Option<FLT_0D>,
}

/// Scalar constraint with no sign convention transformation
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints0d {
    /// Measured value
    /// Units: as_parent
    pub measured: Option<FLT_0D>,
    /// Path to the source data for this measurement in the IMAS data dictionary
    pub source: Option<STR_0D>,
    /// Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used
    /// Units: s
    pub time_measurement: Option<FLT_0D>,
    /// Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit
    pub exact: Option<INT_0D>,
    /// Weight given to the measurement
    pub weight: Option<FLT_0D>,
    /// Standard deviation of the measurement error
    /// Units: as_parent
    pub sigma: Option<FLT_0D>,
    /// Value calculated from the reconstructed equilibrium
    /// Units: as_parent
    pub reconstructed: Option<FLT_0D>,
    /// Squared error normalized by the variance considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error
    pub chi_squared: Option<FLT_0D>,
}

/// Magnetization constraints along R and Z axis
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraintsMagnetization {
    /// Magnetization M of the iron core segment along the major radius axis, assumed to be constant inside a given iron segment. Reminder : H = 1/mu0 * B - mur * M;
    /// Units: T
    pub magnetization_r: EquilibriumConstraints0d,
    /// Magnetization M of the iron core segment along the vertical axis, assumed to be constant inside a given iron segment. Reminder : H = 1/mu0 * B - mur * M;
    /// Units: T
    pub magnetization_z: EquilibriumConstraints0d,
}

/// Measurements to constrain the equilibrium, output values and accuracy of the fit
#[derive(Debug, Clone, Default)]
pub struct EquilibriumConstraints {
    /// Vacuum field times major radius in the toroidal field magnet. Positive sign means counter-clockwise when viewed from above
    /// Units: T.m
    pub b_field_tor_vacuum_r: EquilibriumConstraints0d,
    /// Set of poloidal field probes
    /// Units: T
    pub b_field_pol_probe: Vec<EquilibriumConstraints0dOneLike>,
    /// Diamagnetic flux
    /// Units: Wb
    pub diamagnetic_flux: EquilibriumConstraints0dB0Like,
    /// Set of faraday angles
    /// Units: rad
    pub faraday_angle: Vec<EquilibriumConstraints0d>,
    /// Set of MSE polarization angles
    /// Units: rad
    pub mse_polarization_angle: Vec<EquilibriumConstraints0d>,
    /// Set of flux loops
    /// Units: Wb
    pub flux_loop: Vec<EquilibriumConstraints0d>,
    /// Plasma current. Positive sign means counter-clockwise when viewed from above
    /// Units: A
    pub ip: EquilibriumConstraints0dIpLike,
    /// Magnetization M of a set of iron core segments
    /// Units: T
    pub iron_core_segment: Vec<EquilibriumConstraintsMagnetization>,
    /// Set of local density measurements
    /// Units: m^-3
    pub n_e: Vec<EquilibriumConstraints0dPosition>,
    /// Set of line integrated density measurements
    /// Units: m^-2
    pub n_e_line: Vec<EquilibriumConstraints0d>,
    /// Current in a set of poloidal field coils
    /// Units: A
    pub pf_current: Vec<EquilibriumConstraints0dIpLike>,
    /// Current in a set of axisymmetric passive conductors
    /// Units: A
    pub pf_passive_current: Vec<EquilibriumConstraints0d>,
    /// Set of total pressure estimates
    /// Units: Pa
    pub pressure: Vec<EquilibriumConstraints0dPosition>,
    /// Set of rotational pressure estimates. The rotational pressure is defined as R0^2*rho*omega^2 / 2, where omega is the toroidal rotation frequency, rho=ne(R0,psi)*m, and m is the plasma equivalent mass.
    /// Units: Pa
    pub pressure_rotational: Vec<EquilibriumConstraints0dPosition>,
    /// Set of safety factor estimates at various positions
    pub q: Vec<EquilibriumConstraints0dPosition>,
    /// Set of flux-surface averaged toroidal current density approximations at various positions  (= average(j_tor/R) / average(1/R))
    /// Units: A.m^-2
    pub j_phi: Vec<EquilibriumConstraints0dPosition>,
    /// Set of flux-surface averaged parallel current density approximations at various positions (= average(j.B) / B0, where B0 = /vacuum_toroidal_field/b0)
    /// Units: A.m^-2
    pub j_parallel: Vec<EquilibriumConstraints0dPosition>,
    /// Array of X-points, for each of them the RZ position is given
    pub x_point: Vec<EquilibriumConstraintsPurePosition>,
    /// Array of strike points, for each of them the RZ position is given
    pub strike_point: Vec<EquilibriumConstraintsPurePosition>,
    /// Sum of the chi_squared of all constraints used for the equilibrium reconstruction, divided by the number of degrees of freedom of the identification model
    pub chi_squared_reduced: Option<FLT_0D>,
    /// Number of degrees of freedom of the identification model
    pub freedom_degrees_n: Option<INT_0D>,
    /// Number of constraints used (i.e. having a non-zero weight)
    pub constraints_n: Option<INT_0D>,
}

/// Equilibrium profiles (1D radial grid) as a function of the poloidal flux
#[derive(Debug, Clone, Default)]
pub struct EquilibriumProfiles1d {
    /// Poloidal flux. Integral of magnetic field passing through a contour defined by the intersection of a flux surface passing through the point of interest and a Z=constant plane. If the integration surface is flat, the surface normal vector is in the increasing vertical coordinate direction, Z, namely upwards.
    /// Units: Wb
    pub psi: Option<FLT_1D>,
    /// Normalised poloidal flux, namely (psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))
    pub psi_norm: Option<FLT_1D>,
    /// Toroidal flux. Positive when the toroidal magnetic field is counter-clockwise when viewed from above
    /// Units: Wb
    pub phi: Option<FLT_1D>,
    /// Pressure
    /// Units: Pa
    pub pressure: Option<FLT_1D>,
    /// Diamagnetic function (F=R B_Phi). Positive when the toroidal field is counter-clockwise when viewed from above
    /// Units: T.m
    pub f: Option<FLT_1D>,
    /// Derivative of pressure w.r.t. psi. Sign depends on the poloidal flux sign convention
    /// Units: Pa.Wb^-1
    pub dpressure_dpsi: Option<FLT_1D>,
    /// Derivative of F w.r.t. Psi, multiplied with F. Sign depends on the poloidal flux sign convention
    /// Units: T^2.m^2.Wb^-1
    pub f_df_dpsi: Option<FLT_1D>,
    /// Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R). Positive sign means counter-clockwise when viewed from above
    /// Units: A.m^-2
    pub j_phi: Option<FLT_1D>,
    /// Flux surface averaged approximation to parallel current density = average(j.B) / B0, where B0 = /vacuum_toroidal_field/b0. Sign is positive when the scalar product j.B is in the same direction as B0 (the signed vacuum toroidal field at R0)
    /// Units: A.m^-2
    pub j_parallel: Option<FLT_1D>,
    /// Safety factor (only positive when toroidal current and magnetic field are in same direction)
    pub q: Option<FLT_1D>,
    /// Magnetic shear, defined as rho_tor/q . dq/drho_tor
    pub magnetic_shear: Option<FLT_1D>,
    /// Radial coordinate (major radius) on the inboard side of the magnetic axis
    /// Units: m
    pub r_inboard: Option<FLT_1D>,
    /// Radial coordinate (major radius) on the outboard side of the magnetic axis
    /// Units: m
    pub r_outboard: Option<FLT_1D>,
    /// Toroidal flux coordinate = sqrt(phi/(pi*b0)), where the toroidal flux, phi, corresponds to time_slice/profiles_1d/phi, the toroidal magnetic field, b0, corresponds to that stored in vacuum_toroidal_field/b0 and pi can be found in the IMAS constants
    /// Units: m
    pub rho_tor: Option<FLT_1D>,
    /// Normalized toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation). Namely (rho_tor(rho)-rho_tor(magnetic_axis)) / (rho_tor(boundary)-rho_tor(magnetic_axis))
    pub rho_tor_norm: Option<FLT_1D>,
    /// Derivative of Psi with respect to Rho_Tor. Sign follows the poloidal flux convention
    /// Units: Wb.m^-1
    pub dpsi_drho_tor: Option<FLT_1D>,
    /// RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
    pub geometric_axis: EquilibriumProfiles1dRz1dDynamicAos,
    /// Elongation
    pub elongation: Option<FLT_1D>,
    /// Triangularity
    pub triangularity: Option<FLT_1D>,
    /// Upper triangularity
    pub triangularity_upper: Option<FLT_1D>,
    /// Lower triangularity
    pub triangularity_lower: Option<FLT_1D>,
    /// Upper inner squareness (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_upper_inner: Option<FLT_1D>,
    /// Upper outer squareness (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_upper_outer: Option<FLT_1D>,
    /// Lower inner squareness (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_lower_inner: Option<FLT_1D>,
    /// Lower outer squareness (definition from T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009)
    pub squareness_lower_outer: Option<FLT_1D>,
    /// Volume enclosed in the flux surface
    /// Units: m^3
    pub volume: Option<FLT_1D>,
    /// Normalized square root of enclosed volume (radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)
    pub rho_volume_norm: Option<FLT_1D>,
    /// Radial derivative of the volume enclosed in the flux surface with respect to Psi. Sign depends on the poloidal flux sign convention
    /// Units: m^3.Wb^-1
    pub dvolume_dpsi: Option<FLT_1D>,
    /// Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor
    /// Units: m^2
    pub dvolume_drho_tor: Option<FLT_1D>,
    /// Cross-sectional area of the flux surface
    /// Units: m^2
    pub area: Option<FLT_1D>,
    /// Radial derivative of the cross-sectional area of the flux surface with respect to psi. Sign depends on the poloidal flux sign convention
    /// Units: m^2.Wb^-1
    pub darea_dpsi: Option<FLT_1D>,
    /// Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor
    /// Units: m
    pub darea_drho_tor: Option<FLT_1D>,
    /// Surface area of the toroidal flux surface
    /// Units: m^2
    pub surface: Option<FLT_1D>,
    /// Trapped particle fraction
    pub trapped_fraction: Option<FLT_1D>,
    /// Flux surface averaged 1/R^2
    /// Units: m^-2
    pub gm1: Option<FLT_1D>,
    /// Flux surface averaged |grad_rho_tor|^2/R^2
    /// Units: m^-2
    pub gm2: Option<FLT_1D>,
    /// Flux surface averaged |grad_rho_tor|^2
    pub gm3: Option<FLT_1D>,
    /// Flux surface averaged 1/B^2
    /// Units: T^-2
    pub gm4: Option<FLT_1D>,
    /// Flux surface averaged B^2
    /// Units: T^2
    pub gm5: Option<FLT_1D>,
    /// Flux surface averaged |grad_rho_tor|^2/B^2
    /// Units: T^-2
    pub gm6: Option<FLT_1D>,
    /// Flux surface averaged |grad_rho_tor|
    pub gm7: Option<FLT_1D>,
    /// Flux surface averaged R
    /// Units: m
    pub gm8: Option<FLT_1D>,
    /// Flux surface averaged 1/R
    /// Units: m^-1
    pub gm9: Option<FLT_1D>,
    /// Flux surface averaged modulus of B (always positive, irrespective of the sign convention for the B-field direction).
    /// Units: T
    pub b_field_average: Option<FLT_1D>,
    /// Minimum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction)
    /// Units: T
    pub b_field_min: Option<FLT_1D>,
    /// Maximum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction)
    /// Units: T
    pub b_field_max: Option<FLT_1D>,
    /// Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]
    pub beta_pol: Option<FLT_1D>,
    /// Mass density
    /// Units: kg.m^-3
    pub mass_density: Option<FLT_1D>,
}

/// Equilibrium 2D profiles in the poloidal plane
#[derive(Debug, Clone, Default)]
pub struct EquilibriumProfiles2d {
    /// Type of profiles (distinguishes contribution from plasma, vaccum fields and total fields)
    pub r#type: IdentifierDynamicAos3,
    /// Selection of one of a set of grid types
    pub grid_type: IdentifierDynamicAos3,
    /// Definition of the 2D grid (the content of dim1 and dim2 is defined by the selected grid_type)
    pub grid: EquilibriumProfiles2dGrid,
    /// Values of the major radius on the grid
    /// Units: m
    pub r: Option<FLT_2D>,
    /// Values of the Height on the grid
    /// Units: m
    pub z: Option<FLT_2D>,
    /// Values of the poloidal flux at the grid in the poloidal plane. The poloidal flux is integral of magnetic field passing through a contour defined by the intersection of a flux surface passing through the point of interest and a Z=constant plane. If the integration surface is flat, the surface normal vector is in the increasing vertical coordinate direction, Z, namely upwards.
    /// Units: Wb
    pub psi: Option<FLT_2D>,
    /// Values of poloidal angle on the grid. The poloidal angle is centered on the magnetic axis and oriented such that (grad rho_tor_norm, grad theta, grad phi) form a right-handed set where grad rho_tor_norm points away from the magnetic axis.
    /// Units: rad
    pub theta: Option<FLT_2D>,
    /// Toroidal flux. Positive when the toroidal magnetic field is counter-clockwise when viewed from above
    /// Units: Wb
    pub phi: Option<FLT_2D>,
    /// Toroidal plasma current density. Positive sign means counter-clockwise when viewed from above
    /// Units: A.m^-2
    pub j_phi: Option<FLT_2D>,
    /// Defined as (j.B)/B0 where j and B are the current density and magnetic field vectors and B0 is the (signed) vacuum toroidal magnetic field strength at the geometric reference point (R0,Z0). It is formally not the component of the plasma current density parallel to the magnetic field
    /// Units: A.m^-2
    pub j_parallel: Option<FLT_2D>,
    /// R component of the poloidal magnetic field
    /// Units: T
    pub b_field_r: Option<FLT_2D>,
    /// Toroidal component of the magnetic field. Positive sign means counter-clockwise when viewed from above
    /// Units: T
    pub b_field_phi: Option<FLT_2D>,
    /// Z component of the magnetic field
    /// Units: T
    pub b_field_z: Option<FLT_2D>,
    /// Radial derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-1
    pub d_psi_d_r: Option<FLT_2D>,
    /// Vertical derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-1
    pub d_psi_d_z: Option<FLT_2D>,
    /// Second radial derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_r2: Option<FLT_2D>,
    /// Mixed second derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_r_d_z: Option<FLT_2D>,
    /// Second vertical derivative of the poloidal flux on the grid in the poloidal plane
    /// Units: Wb.m^-2
    pub d2_psi_d_z2: Option<FLT_2D>,
    /// Mask for the grid in the poloidal plane
    /// Units: dimensionless
    pub mask: Option<FLT_2D>,
    /// Normalised poloidal flux on the grid, 0 at the magnetic axis and 1 at the plasma boundary
    pub psi_norm: Option<FLT_2D>,
    /// Contribution to the poloidal flux from the PF coils alone
    /// Units: Wb
    pub psi_coils: Option<FLT_2D>,
}

/// Equilibrium ggd representation
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGgd {
    /// Values of the major radius on various grid subsets
    /// Units: m
    pub r: Vec<GenericGridScalar>,
    /// Values of the Height on various grid subsets
    /// Units: m
    pub z: Vec<GenericGridScalar>,
    /// Values of the poloidal flux, given on various grid subsets. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi: Vec<GenericGridScalar>,
    /// Values of the toroidal flux, given on various grid subsets. Positive sign means counter-clockwise when viewed from above
    /// Units: Wb
    pub phi: Vec<GenericGridScalar>,
    /// Values of the poloidal angle, given on various grid subsets. The poloidal angle is centered on the magnetic axis and oriented such that (grad rho_tor_norm, grad theta, grad phi) form a right-handed set where grad rho_tor_norm points away from the magnetic axis.
    /// Units: rad
    pub theta: Vec<GenericGridScalar>,
    /// Toroidal plasma current density, given on various grid subsets
    /// Units: A.m^-2
    pub j_phi: Vec<GenericGridScalar>,
    /// Parallel (to magnetic field) plasma current density, given on various grid subsets
    /// Units: A.m^-2
    pub j_parallel: Vec<GenericGridScalar>,
    /// R component of the poloidal magnetic field, given on various grid subsets
    /// Units: T
    pub b_field_r: Vec<GenericGridScalar>,
    /// Toroidal component of the magnetic field, given on various grid subsets
    /// Units: T
    pub b_field_phi: Vec<GenericGridScalar>,
    /// Z component of the magnetic field, given on various grid subsets
    /// Units: T
    pub b_field_z: Vec<GenericGridScalar>,
}

/// Multiple GGDs provided at a given time slice
#[derive(Debug, Clone, Default)]
pub struct EquilibriumGgdArray {
    /// Set of GGD grids for describing the equilibrium, at a given time slice
    pub grid: Vec<GenericGridDynamic>,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
}

/// Equilibrium at a given time slice
#[derive(Debug, Clone, Default)]
pub struct EquilibriumTimeSlice {
    /// Description of the plasma boundary. The boundary can be either the real separatrix (provided by a free boundary equilibrium solver) or the 0.99x psi_norm flux surface provided by a fixed boundary equilibrium
    pub boundary: EquilibriumBoundary,
    /// Contour tree (Reeb graph) of the poloidal flux, encoding the topological structure of the equilibrium through its critical points and limiter points. Nodes represent critical points of the poloidal flux (O-points, X-points) and limiter points where the plasma boundary touches the first wall. Edges connect nodes that are topologically adjacent in psi_norm space, meaning no other node lies between them. The tree captures the nesting structure of flux surfaces, the connectivity between plasma regions, the separatrix topology, and the locations of limiter contacts. Normalised poloidal flux, psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis), defines all ordering conventions.
    pub contour_tree: EquilibriumContourTree,
    /// In case of equilibrium reconstruction under constraints, measurements used to constrain the equilibrium, reconstructed values and accuracy of the fit. The names of the child nodes correspond to the following definition: the solver aims at minimizing a cost function defined as : J=1/2*sum_i [ weight_i^2 (reconstructed_i - measured_i)^2 / sigma_i^2 ]. in which sigma_i is the standard deviation of the measurement error (to be found in the IDS of the measurement)
    pub constraints: EquilibriumConstraints,
    /// 0D parameters of the equilibrium
    pub global_quantities: EqulibriumGlobalQuantities,
    /// Equilibrium profiles (1D radial grid) as a function of the poloidal flux
    pub profiles_1d: EquilibriumProfiles1d,
    /// Equilibrium 2D profiles in the poloidal plane. Multiple 2D representations of the equilibrium can be stored here.
    pub profiles_2d: Vec<EquilibriumProfiles2d>,
    /// Set of equilibrium representations using the generic grid description
    pub ggd: Vec<EquilibriumGgd>,
    /// Flux surface coordinate system on a square grid of flux and poloidal angle
    pub coordinate_system: EquilibriumCoordinateSystem,
    /// Convergence details
    pub convergence: EquilibriumConvergence,
    /// Time
    /// Units: s
    pub time: Option<FLT_0D>,
    /// Fitted degrees of freedom of the p' source function
    pub p_prime_dof_values: Option<FLT_1D>,
    /// Fitted degrees of freedom of the FF' source function
    pub ff_prime_dof_values: Option<FLT_1D>,
    /// Fitted degrees of freedom of the passive structure currents
    /// Units: A
    pub passive_dof_values: Option<FLT_1D>,
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

/// Structure for list of R, Z positions (1D list of Npoints, dynamic within a type 3 array of structures (index on time))
#[derive(Debug, Clone, Default)]
pub struct Rz1dDynamicAos {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_1D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_1D>,
}

/// Structure for scalar R, Z positions, dynamic within a type 3 array of structures (index on time)
#[derive(Debug, Clone, Default)]
pub struct Rz0dDynamicAos {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_0D>,
}

/// Structure for R, Z, Phi, psi, rho_tor positions (0D, dynamic within a type 3 array of structures (index on time))
#[derive(Debug, Clone, Default)]
pub struct Rphizpsirho0dDynamicAos3 {
    /// Major radius
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Toroidal angle (oriented counter-clockwise when viewed from above)
    /// Units: rad
    pub phi: Option<FLT_0D>,
    /// Height
    /// Units: m
    pub z: Option<FLT_0D>,
    /// Normalized toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS)
    pub rho_tor_norm: Option<FLT_0D>,
    /// Poloidal magnetic flux. For a positive plasma current (counter-clockwise when viewed from above), increases from the magnetic axis to the boundary
    /// Units: Wb
    pub psi: Option<FLT_0D>,
}

/// Definition of the 2D grid
#[derive(Debug, Clone, Default)]
pub struct EquilibriumProfiles2dGrid {
    /// First dimension values
    /// Units: mixed
    pub dim1: Option<FLT_1D>,
    /// Second dimension values
    /// Units: mixed
    pub dim2: Option<FLT_1D>,
    /// Elementary plasma volume of plasma enclosed in the cell formed by the nodes [dim1(i) dim2(j)], [dim1(i+1) dim2(j)], [dim1(i) dim2(j+1)] and [dim1(i+1) dim2(j+1)]
    /// Units: m^3
    pub volume_element: Option<FLT_2D>,
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

/// Generic grid (dynamic within a type 3 AoS)
#[derive(Debug, Clone, Default)]
pub struct GenericGridDynamic {
    /// Grid identifier
    pub identifier: IdentifierDynamicAos3,
    /// Path of the grid, including the IDS name, in case of implicit reference to a grid_ggd node described in another IDS. To be filled only if the grid is not described explicitly in this grid_ggd structure. Example syntax: #wall:2/description_ggd(1)/grid_ggd, means that the grid is located in the wall IDS, occurrence 2, with relative path description_ggd(1)/grid_ggd, using Fortran index convention (here : first index of the array)
    pub path: Option<STR_0D>,
    /// Set of grid spaces
    pub space: Vec<GenericGridDynamicSpace>,
    /// Grid subsets
    pub grid_subset: Vec<GenericGridDynamicGridSubset>,
}

/// Flux surface coordinate system on a square grid of flux and poloidal angle
#[derive(Debug, Clone, Default)]
pub struct EquilibriumCoordinateSystem {
    /// Type of coordinate system
    pub grid_type: IdentifierDynamicAos3,
    /// Definition of the 2D grid
    pub grid: EquilibriumProfiles2dGrid,
    /// Values of the major radius on the grid
    /// Units: m
    pub r: Option<FLT_2D>,
    /// Values of the Height on the grid
    /// Units: m
    pub z: Option<FLT_2D>,
    /// Absolute value of the jacobian of the coordinate system
    /// Units: mixed
    pub jacobian: Option<FLT_2D>,
    /// Covariant metric tensor on every point of the grid described by grid_type
    /// Units: mixed
    pub tensor_covariant: Option<FLT_4D>,
    /// Contravariant metric tensor on every point of the grid described by grid_type
    /// Units: mixed
    pub tensor_contravariant: Option<FLT_4D>,
}

/// Characteristics of the vacuum toroidal field. Time coordinate at the root of the IDS
#[derive(Debug, Clone, Default)]
pub struct BTorVacuum1 {
    /// Reference major radius where the vacuum toroidal magnetic field is given (usually a fixed position such as the middle of the vessel at the equatorial midplane)
    /// Units: m
    pub r0: Option<FLT_0D>,
    /// Vacuum toroidal field at R0 [T]; Positive sign means counter-clockwise when viewed from above. The product R0B0 must be consistent with the b_tor_vacuum_r field of the tf IDS.
    /// Units: T
    pub b0: Option<FLT_1D>,
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
    pub library: Library,
    /// Maximum number of iterations the convergence loop is allowed to run for
    pub iterations_n_max: Option<INT_0D>,
    /// Minimum number of iterations before the convergence test is allowed to pass
    pub iterations_n_min: Option<INT_0D>,
    /// Number of initial iterations for which the vertical feedback is switched off
    pub iterations_n_no_vertical_feedback: Option<INT_0D>,
    /// Value of convergence/grad_shafranov_deviation_value below which the solution is taken as
    /// converged
    /// Units: mixed
    pub grad_shafranov_deviation_value_tolerance: Option<FLT_0D>,
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

/// Custom (non-IMAS) structure, declared in custom_equilibrium_keys.rs
#[derive(Debug, Clone, Default)]
pub struct EquilibriumBoundaryBounding {
    /// Major radius of the point which defines the plasma boundary
    /// Units: m
    pub r: Option<FLT_0D>,
    /// Height of the point which defines the plasma boundary
    /// Units: m
    pub z: Option<FLT_0D>,
}

// ============================================================================
// Root IDS Structure
// ============================================================================

/// Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
#[derive(Debug, Clone, Default)]
pub struct Equilibrium {
    /// Characteristics of the vacuum toroidal field (used in rho_tor definition and in the normalization of current densities)
    pub vacuum_toroidal_field: BTorVacuum1,
    /// Grids (using the Generic Grid Description), for various time slices. The timebase of this array of structure must be a subset of the time_slice timebase
    pub grids_ggd: Vec<EquilibriumGgdArray>,
    /// Set of equilibria at various time slices
    pub time_slice: Vec<EquilibriumTimeSlice>,
    pub code: Code,
}

// ============================================================================
// Equilibrium Constructors
// ============================================================================

impl Equilibrium {
    /// Create a `Equilibrium` pre-populated with `n_time` default (empty) time slices.
    ///
    /// Every leaf field in each slice is unset (`None`), ready to be filled in,
    /// e.g. via `time_slice.par_iter_mut()`.
    pub fn with_size(n_time: usize) -> Self {
        let mut ids = Self::default();
        ids.time_slice = (0..n_time).map(|_| EquilibriumTimeSlice::default()).collect();
        ids
    }

    /// Create a `Equilibrium` with one time slice per entry in `time`,
    /// setting each slice's `time` field. All other leaf fields are unset (`None`).
    pub fn with_time(time: &FLT_1D) -> Self {
        let mut ids = Self::with_size(time.len());
        for (slice, &t) in ids.time_slice.iter_mut().zip(time.iter()) {
            slice.time = Some(t);
        }
        ids
    }
}

// ============================================================================
// View, Accessor, and Accumulator Types
// ============================================================================

// --- Rz1dDynamicAos View Types ---

/// View over multiple Rz1dDynamicAos with field accumulation
pub struct Rz1dDynamicAosSliceView<'a> {
    data: &'a [Rz1dDynamicAos],
}

impl<'a> Rz1dDynamicAosSliceView<'a> {
    pub fn new(data: &'a [Rz1dDynamicAos]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Rz1dDynamicAos> {
        self.data.iter()
    }
}

/// Mutable view over multiple Rz1dDynamicAos
pub struct Rz1dDynamicAosSliceViewMut<'a> {
    data: &'a mut [Rz1dDynamicAos],
}

impl<'a> Rz1dDynamicAosSliceViewMut<'a> {
    pub fn new(data: &'a mut [Rz1dDynamicAos]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Rz1dDynamicAos> {
        self.data.iter_mut()
    }
}

/// Index trait for Rz1dDynamicAos - enables .field(0) and .field(0..2) syntax
pub trait Rz1dDynamicAosIndex<'a> {
    type Output;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output;
}

impl<'a> Rz1dDynamicAosIndex<'a> for usize {
    type Output = &'a Rz1dDynamicAos;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        &data[self]
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::Range<usize> {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(&data[self])
    }
}

impl<'a> Rz1dDynamicAosIndex<'a> for std::ops::RangeFull {
    type Output = Rz1dDynamicAosSliceView<'a>;
    fn get(self, data: &'a [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceView::new(data)
    }
}

/// Mutable index trait for Rz1dDynamicAos - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait Rz1dDynamicAosMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output;
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for usize {
    type Output = &'a mut Rz1dDynamicAos;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::Range<usize> {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(&mut data[self])
    }
}

impl<'a> Rz1dDynamicAosMutIndex<'a> for std::ops::RangeFull {
    type Output = Rz1dDynamicAosSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [Rz1dDynamicAos]) -> Self::Output {
        Rz1dDynamicAosSliceViewMut::new(data)
    }
}

// --- EquilibriumContourTreeNode View Types ---

/// View over `node_type` (IdentifierDynamicAos3) across multiple EquilibriumContourTreeNode
pub struct EquilibriumContourTreeNodeNodeTypeView<'a> {
    pub name: StringAccumulator<'a, EquilibriumContourTreeNode>,
    pub index: Accumulator<'a, EquilibriumContourTreeNode, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumContourTreeNode>,
}

impl<'a> EquilibriumContourTreeNodeNodeTypeView<'a> {
    pub fn new(data: &'a [EquilibriumContourTreeNode]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &EquilibriumContourTreeNode| item.node_type.name.clone(), "node_type.name"),
            index: Accumulator::new(data, |item: &EquilibriumContourTreeNode| item.node_type.index, "node_type.index"),
            description: StringAccumulator::new(
                data,
                |item: &EquilibriumContourTreeNode| item.node_type.description.clone(),
                "node_type.description",
            ),
        }
    }
}

/// View over multiple EquilibriumContourTreeNode with field accumulation
pub struct EquilibriumContourTreeNodeSliceView<'a> {
    data: &'a [EquilibriumContourTreeNode],
    pub critical_type: Accumulator<'a, EquilibriumContourTreeNode, INT_0D>,
    pub node_type: EquilibriumContourTreeNodeNodeTypeView<'a>,
    pub r: Accumulator<'a, EquilibriumContourTreeNode, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumContourTreeNode, FLT_0D>,
    pub psi: Accumulator<'a, EquilibriumContourTreeNode, FLT_0D>,
}

impl<'a> EquilibriumContourTreeNodeSliceView<'a> {
    pub fn new(data: &'a [EquilibriumContourTreeNode]) -> Self {
        Self {
            data,
            critical_type: Accumulator::new(data, |item: &EquilibriumContourTreeNode| item.critical_type, "critical_type"),
            node_type: EquilibriumContourTreeNodeNodeTypeView::new(data),
            r: Accumulator::new(data, |item: &EquilibriumContourTreeNode| item.r, "r"),
            z: Accumulator::new(data, |item: &EquilibriumContourTreeNode| item.z, "z"),
            psi: Accumulator::new(data, |item: &EquilibriumContourTreeNode| item.psi, "psi"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumContourTreeNode> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumContourTreeNode
pub struct EquilibriumContourTreeNodeSliceViewMut<'a> {
    data: &'a mut [EquilibriumContourTreeNode],
}

impl<'a> EquilibriumContourTreeNodeSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumContourTreeNode]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumContourTreeNode> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumContourTreeNode - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumContourTreeNodeIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output;
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for usize {
    type Output = &'a EquilibriumContourTreeNode;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumContourTreeNodeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumContourTreeNode - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumContourTreeNodeMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output;
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumContourTreeNode;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumContourTreeNodeMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumContourTreeNodeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumContourTreeNode]) -> Self::Output {
        EquilibriumContourTreeNodeSliceViewMut::new(data)
    }
}

// --- EquilibriumGap View Types ---

/// View over multiple EquilibriumGap with field accumulation
pub struct EquilibriumGapSliceView<'a> {
    data: &'a [EquilibriumGap],
    pub name: StringAccumulator<'a, EquilibriumGap>,
    pub description: StringAccumulator<'a, EquilibriumGap>,
    pub r: Accumulator<'a, EquilibriumGap, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumGap, FLT_0D>,
    pub angle: Accumulator<'a, EquilibriumGap, FLT_0D>,
    pub value: Accumulator<'a, EquilibriumGap, FLT_0D>,
}

impl<'a> EquilibriumGapSliceView<'a> {
    pub fn new(data: &'a [EquilibriumGap]) -> Self {
        Self {
            data,
            name: StringAccumulator::new(data, |item: &EquilibriumGap| item.name.clone(), "name"),
            description: StringAccumulator::new(data, |item: &EquilibriumGap| item.description.clone(), "description"),
            r: Accumulator::new(data, |item: &EquilibriumGap| item.r, "r"),
            z: Accumulator::new(data, |item: &EquilibriumGap| item.z, "z"),
            angle: Accumulator::new(data, |item: &EquilibriumGap| item.angle, "angle"),
            value: Accumulator::new(data, |item: &EquilibriumGap| item.value, "value"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumGap> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumGap
pub struct EquilibriumGapSliceViewMut<'a> {
    data: &'a mut [EquilibriumGap],
}

impl<'a> EquilibriumGapSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumGap]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumGap> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumGap - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumGapIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output;
}

impl<'a> EquilibriumGapIndex<'a> for usize {
    type Output = &'a EquilibriumGap;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGapIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGapSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumGap - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumGapMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output;
}

impl<'a> EquilibriumGapMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumGap;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGapMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGapSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGap]) -> Self::Output {
        EquilibriumGapSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraints0dOneLike View Types ---

/// View over multiple EquilibriumConstraints0dOneLike with field accumulation
pub struct EquilibriumConstraints0dOneLikeSliceView<'a> {
    data: &'a [EquilibriumConstraints0dOneLike],
    pub measured: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumConstraints0dOneLike>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraints0dOneLike, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraints0dOneLike, FLT_0D>,
}

impl<'a> EquilibriumConstraints0dOneLikeSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraints0dOneLike]) -> Self {
        Self {
            data,
            measured: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.measured, "measured"),
            source: StringAccumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.source.clone(), "source"),
            time_measurement: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.time_measurement, "time_measurement"),
            exact: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.exact, "exact"),
            weight: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.weight, "weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.sigma, "sigma"),
            reconstructed: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.reconstructed, "reconstructed"),
            chi_squared: Accumulator::new(data, |item: &EquilibriumConstraints0dOneLike| item.chi_squared, "chi_squared"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraints0dOneLike> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraints0dOneLike
pub struct EquilibriumConstraints0dOneLikeSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraints0dOneLike],
}

impl<'a> EquilibriumConstraints0dOneLikeSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraints0dOneLike> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraints0dOneLike - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraints0dOneLikeIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for usize {
    type Output = &'a EquilibriumConstraints0dOneLike;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dOneLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraints0dOneLike - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraints0dOneLikeMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraints0dOneLike;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dOneLikeMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dOneLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dOneLike]) -> Self::Output {
        EquilibriumConstraints0dOneLikeSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraints0d View Types ---

/// View over multiple EquilibriumConstraints0d with field accumulation
pub struct EquilibriumConstraints0dSliceView<'a> {
    data: &'a [EquilibriumConstraints0d],
    pub measured: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumConstraints0d>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraints0d, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraints0d, FLT_0D>,
}

impl<'a> EquilibriumConstraints0dSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraints0d]) -> Self {
        Self {
            data,
            measured: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.measured, "measured"),
            source: StringAccumulator::new(data, |item: &EquilibriumConstraints0d| item.source.clone(), "source"),
            time_measurement: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.time_measurement, "time_measurement"),
            exact: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.exact, "exact"),
            weight: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.weight, "weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.sigma, "sigma"),
            reconstructed: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.reconstructed, "reconstructed"),
            chi_squared: Accumulator::new(data, |item: &EquilibriumConstraints0d| item.chi_squared, "chi_squared"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraints0d> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraints0d
pub struct EquilibriumConstraints0dSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraints0d],
}

impl<'a> EquilibriumConstraints0dSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraints0d]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraints0d> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraints0d - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraints0dIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dIndex<'a> for usize {
    type Output = &'a EquilibriumConstraints0d;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraints0d - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraints0dMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraints0d;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0d]) -> Self::Output {
        EquilibriumConstraints0dSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraintsMagnetization View Types ---

/// View over `magnetization_r` (EquilibriumConstraints0d) across multiple EquilibriumConstraintsMagnetization
pub struct EquilibriumConstraintsMagnetizationMagnetizationRView<'a> {
    pub measured: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumConstraintsMagnetization>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraintsMagnetization, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
}

impl<'a> EquilibriumConstraintsMagnetizationMagnetizationRView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsMagnetization]) -> Self {
        Self {
            measured: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.measured,
                "magnetization_r.measured",
            ),
            source: StringAccumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.source.clone(),
                "magnetization_r.source",
            ),
            time_measurement: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.time_measurement,
                "magnetization_r.time_measurement",
            ),
            exact: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.exact,
                "magnetization_r.exact",
            ),
            weight: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.weight,
                "magnetization_r.weight",
            ),
            sigma: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.sigma,
                "magnetization_r.sigma",
            ),
            reconstructed: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.reconstructed,
                "magnetization_r.reconstructed",
            ),
            chi_squared: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_r.chi_squared,
                "magnetization_r.chi_squared",
            ),
        }
    }
}

/// View over `magnetization_z` (EquilibriumConstraints0d) across multiple EquilibriumConstraintsMagnetization
pub struct EquilibriumConstraintsMagnetizationMagnetizationZView<'a> {
    pub measured: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumConstraintsMagnetization>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraintsMagnetization, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraintsMagnetization, FLT_0D>,
}

impl<'a> EquilibriumConstraintsMagnetizationMagnetizationZView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsMagnetization]) -> Self {
        Self {
            measured: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.measured,
                "magnetization_z.measured",
            ),
            source: StringAccumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.source.clone(),
                "magnetization_z.source",
            ),
            time_measurement: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.time_measurement,
                "magnetization_z.time_measurement",
            ),
            exact: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.exact,
                "magnetization_z.exact",
            ),
            weight: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.weight,
                "magnetization_z.weight",
            ),
            sigma: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.sigma,
                "magnetization_z.sigma",
            ),
            reconstructed: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.reconstructed,
                "magnetization_z.reconstructed",
            ),
            chi_squared: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsMagnetization| item.magnetization_z.chi_squared,
                "magnetization_z.chi_squared",
            ),
        }
    }
}

/// View over multiple EquilibriumConstraintsMagnetization with field accumulation
pub struct EquilibriumConstraintsMagnetizationSliceView<'a> {
    data: &'a [EquilibriumConstraintsMagnetization],
    pub magnetization_r: EquilibriumConstraintsMagnetizationMagnetizationRView<'a>,
    pub magnetization_z: EquilibriumConstraintsMagnetizationMagnetizationZView<'a>,
}

impl<'a> EquilibriumConstraintsMagnetizationSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsMagnetization]) -> Self {
        Self {
            data,
            magnetization_r: EquilibriumConstraintsMagnetizationMagnetizationRView::new(data),
            magnetization_z: EquilibriumConstraintsMagnetizationMagnetizationZView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraintsMagnetization> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraintsMagnetization
pub struct EquilibriumConstraintsMagnetizationSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraintsMagnetization],
}

impl<'a> EquilibriumConstraintsMagnetizationSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraintsMagnetization> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraintsMagnetization - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraintsMagnetizationIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output;
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for usize {
    type Output = &'a EquilibriumConstraintsMagnetization;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraintsMagnetizationSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraintsMagnetization - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraintsMagnetizationMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output;
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraintsMagnetization;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsMagnetizationMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraintsMagnetizationSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsMagnetization]) -> Self::Output {
        EquilibriumConstraintsMagnetizationSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraints0dPosition View Types ---

/// View over `position` (Rphizpsirho0dDynamicAos3) across multiple EquilibriumConstraints0dPosition
pub struct EquilibriumConstraints0dPositionPositionView<'a> {
    pub r: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub phi: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub rho_tor_norm: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub psi: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
}

impl<'a> EquilibriumConstraints0dPositionPositionView<'a> {
    pub fn new(data: &'a [EquilibriumConstraints0dPosition]) -> Self {
        Self {
            r: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.position.r, "position.r"),
            phi: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.position.phi, "position.phi"),
            z: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.position.z, "position.z"),
            rho_tor_norm: Accumulator::new(
                data,
                |item: &EquilibriumConstraints0dPosition| item.position.rho_tor_norm,
                "position.rho_tor_norm",
            ),
            psi: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.position.psi, "position.psi"),
        }
    }
}

/// View over multiple EquilibriumConstraints0dPosition with field accumulation
pub struct EquilibriumConstraints0dPositionSliceView<'a> {
    data: &'a [EquilibriumConstraints0dPosition],
    pub measured: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub position: EquilibriumConstraints0dPositionPositionView<'a>,
    pub source: StringAccumulator<'a, EquilibriumConstraints0dPosition>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraints0dPosition, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraints0dPosition, FLT_0D>,
}

impl<'a> EquilibriumConstraints0dPositionSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraints0dPosition]) -> Self {
        Self {
            data,
            measured: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.measured, "measured"),
            position: EquilibriumConstraints0dPositionPositionView::new(data),
            source: StringAccumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.source.clone(), "source"),
            time_measurement: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.time_measurement, "time_measurement"),
            exact: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.exact, "exact"),
            weight: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.weight, "weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.sigma, "sigma"),
            reconstructed: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.reconstructed, "reconstructed"),
            chi_squared: Accumulator::new(data, |item: &EquilibriumConstraints0dPosition| item.chi_squared, "chi_squared"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraints0dPosition> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraints0dPosition
pub struct EquilibriumConstraints0dPositionSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraints0dPosition],
}

impl<'a> EquilibriumConstraints0dPositionSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraints0dPosition]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraints0dPosition> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraints0dPosition - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraints0dPositionIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for usize {
    type Output = &'a EquilibriumConstraints0dPosition;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dPositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraints0dPosition - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraints0dPositionMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraints0dPosition;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dPositionMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dPositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dPosition]) -> Self::Output {
        EquilibriumConstraints0dPositionSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraints0dIpLike View Types ---

/// View over multiple EquilibriumConstraints0dIpLike with field accumulation
pub struct EquilibriumConstraints0dIpLikeSliceView<'a> {
    data: &'a [EquilibriumConstraints0dIpLike],
    pub measured: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumConstraints0dIpLike>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraints0dIpLike, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumConstraints0dIpLike, FLT_0D>,
}

impl<'a> EquilibriumConstraints0dIpLikeSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraints0dIpLike]) -> Self {
        Self {
            data,
            measured: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.measured, "measured"),
            source: StringAccumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.source.clone(), "source"),
            time_measurement: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.time_measurement, "time_measurement"),
            exact: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.exact, "exact"),
            weight: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.weight, "weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.sigma, "sigma"),
            reconstructed: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.reconstructed, "reconstructed"),
            chi_squared: Accumulator::new(data, |item: &EquilibriumConstraints0dIpLike| item.chi_squared, "chi_squared"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraints0dIpLike> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraints0dIpLike
pub struct EquilibriumConstraints0dIpLikeSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraints0dIpLike],
}

impl<'a> EquilibriumConstraints0dIpLikeSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraints0dIpLike> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraints0dIpLike - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraints0dIpLikeIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for usize {
    type Output = &'a EquilibriumConstraints0dIpLike;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dIpLikeSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraints0dIpLike - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraints0dIpLikeMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output;
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraints0dIpLike;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraints0dIpLikeMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraints0dIpLikeSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraints0dIpLike]) -> Self::Output {
        EquilibriumConstraints0dIpLikeSliceViewMut::new(data)
    }
}

// --- EquilibriumConstraintsPurePosition View Types ---

/// View over `position_measured` (Rz0dDynamicAos) across multiple EquilibriumConstraintsPurePosition
pub struct EquilibriumConstraintsPurePositionPositionMeasuredView<'a> {
    pub r: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
}

impl<'a> EquilibriumConstraintsPurePositionPositionMeasuredView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsPurePosition]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsPurePosition| item.position_measured.r,
                "position_measured.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsPurePosition| item.position_measured.z,
                "position_measured.z",
            ),
        }
    }
}

/// View over `position_reconstructed` (Rz0dDynamicAos) across multiple EquilibriumConstraintsPurePosition
pub struct EquilibriumConstraintsPurePositionPositionReconstructedView<'a> {
    pub r: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
}

impl<'a> EquilibriumConstraintsPurePositionPositionReconstructedView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsPurePosition]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsPurePosition| item.position_reconstructed.r,
                "position_reconstructed.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumConstraintsPurePosition| item.position_reconstructed.z,
                "position_reconstructed.z",
            ),
        }
    }
}

/// View over multiple EquilibriumConstraintsPurePosition with field accumulation
pub struct EquilibriumConstraintsPurePositionSliceView<'a> {
    data: &'a [EquilibriumConstraintsPurePosition],
    pub position_measured: EquilibriumConstraintsPurePositionPositionMeasuredView<'a>,
    pub source: StringAccumulator<'a, EquilibriumConstraintsPurePosition>,
    pub time_measurement: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumConstraintsPurePosition, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub position_reconstructed: EquilibriumConstraintsPurePositionPositionReconstructedView<'a>,
    pub chi_squared_r: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
    pub chi_squared_z: Accumulator<'a, EquilibriumConstraintsPurePosition, FLT_0D>,
}

impl<'a> EquilibriumConstraintsPurePositionSliceView<'a> {
    pub fn new(data: &'a [EquilibriumConstraintsPurePosition]) -> Self {
        Self {
            data,
            position_measured: EquilibriumConstraintsPurePositionPositionMeasuredView::new(data),
            source: StringAccumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.source.clone(), "source"),
            time_measurement: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.time_measurement, "time_measurement"),
            exact: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.exact, "exact"),
            weight: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.weight, "weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.sigma, "sigma"),
            position_reconstructed: EquilibriumConstraintsPurePositionPositionReconstructedView::new(data),
            chi_squared_r: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.chi_squared_r, "chi_squared_r"),
            chi_squared_z: Accumulator::new(data, |item: &EquilibriumConstraintsPurePosition| item.chi_squared_z, "chi_squared_z"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumConstraintsPurePosition> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumConstraintsPurePosition
pub struct EquilibriumConstraintsPurePositionSliceViewMut<'a> {
    data: &'a mut [EquilibriumConstraintsPurePosition],
}

impl<'a> EquilibriumConstraintsPurePositionSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumConstraintsPurePosition> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumConstraintsPurePosition - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumConstraintsPurePositionIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output;
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for usize {
    type Output = &'a EquilibriumConstraintsPurePosition;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraintsPurePositionSliceView<'a>;
    fn get(self, data: &'a [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumConstraintsPurePosition - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumConstraintsPurePositionMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output;
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumConstraintsPurePosition;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumConstraintsPurePositionMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumConstraintsPurePositionSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumConstraintsPurePosition]) -> Self::Output {
        EquilibriumConstraintsPurePositionSliceViewMut::new(data)
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

// --- GenericGridDynamic View Types ---

/// View over `identifier` (IdentifierDynamicAos3) across multiple GenericGridDynamic
pub struct GenericGridDynamicIdentifierView<'a> {
    pub name: StringAccumulator<'a, GenericGridDynamic>,
    pub index: Accumulator<'a, GenericGridDynamic, INT_0D>,
    pub description: StringAccumulator<'a, GenericGridDynamic>,
}

impl<'a> GenericGridDynamicIdentifierView<'a> {
    pub fn new(data: &'a [GenericGridDynamic]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &GenericGridDynamic| item.identifier.name.clone(), "identifier.name"),
            index: Accumulator::new(data, |item: &GenericGridDynamic| item.identifier.index, "identifier.index"),
            description: StringAccumulator::new(data, |item: &GenericGridDynamic| item.identifier.description.clone(), "identifier.description"),
        }
    }
}

/// View over multiple GenericGridDynamic with field accumulation
pub struct GenericGridDynamicSliceView<'a> {
    data: &'a [GenericGridDynamic],
    pub identifier: GenericGridDynamicIdentifierView<'a>,
    pub path: StringAccumulator<'a, GenericGridDynamic>,
}

impl<'a> GenericGridDynamicSliceView<'a> {
    pub fn new(data: &'a [GenericGridDynamic]) -> Self {
        Self {
            data,
            identifier: GenericGridDynamicIdentifierView::new(data),
            path: StringAccumulator::new(data, |item: &GenericGridDynamic| item.path.clone(), "path"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenericGridDynamic> {
        self.data.iter()
    }
}

/// Mutable view over multiple GenericGridDynamic
pub struct GenericGridDynamicSliceViewMut<'a> {
    data: &'a mut [GenericGridDynamic],
}

impl<'a> GenericGridDynamicSliceViewMut<'a> {
    pub fn new(data: &'a mut [GenericGridDynamic]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenericGridDynamic> {
        self.data.iter_mut()
    }
}

/// Index trait for GenericGridDynamic - enables .field(0) and .field(0..2) syntax
pub trait GenericGridDynamicIndex<'a> {
    type Output;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output;
}

impl<'a> GenericGridDynamicIndex<'a> for usize {
    type Output = &'a GenericGridDynamic;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        &data[self]
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(&data[self])
    }
}

impl<'a> GenericGridDynamicIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSliceView<'a>;
    fn get(self, data: &'a [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceView::new(data)
    }
}

/// Mutable index trait for GenericGridDynamic - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait GenericGridDynamicMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output;
}

impl<'a> GenericGridDynamicMutIndex<'a> for usize {
    type Output = &'a mut GenericGridDynamic;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::Range<usize> {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(&mut data[self])
    }
}

impl<'a> GenericGridDynamicMutIndex<'a> for std::ops::RangeFull {
    type Output = GenericGridDynamicSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [GenericGridDynamic]) -> Self::Output {
        GenericGridDynamicSliceViewMut::new(data)
    }
}

// --- EquilibriumProfiles2d View Types ---

/// View over `type` (IdentifierDynamicAos3) across multiple EquilibriumProfiles2d
pub struct EquilibriumProfiles2dTypeView<'a> {
    pub name: StringAccumulator<'a, EquilibriumProfiles2d>,
    pub index: Accumulator<'a, EquilibriumProfiles2d, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumProfiles2d>,
}

impl<'a> EquilibriumProfiles2dTypeView<'a> {
    pub fn new(data: &'a [EquilibriumProfiles2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &EquilibriumProfiles2d| item.r#type.name.clone(), "type.name"),
            index: Accumulator::new(data, |item: &EquilibriumProfiles2d| item.r#type.index, "type.index"),
            description: StringAccumulator::new(data, |item: &EquilibriumProfiles2d| item.r#type.description.clone(), "type.description"),
        }
    }
}

/// View over `grid_type` (IdentifierDynamicAos3) across multiple EquilibriumProfiles2d
pub struct EquilibriumProfiles2dGridTypeView<'a> {
    pub name: StringAccumulator<'a, EquilibriumProfiles2d>,
    pub index: Accumulator<'a, EquilibriumProfiles2d, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumProfiles2d>,
}

impl<'a> EquilibriumProfiles2dGridTypeView<'a> {
    pub fn new(data: &'a [EquilibriumProfiles2d]) -> Self {
        Self {
            name: StringAccumulator::new(data, |item: &EquilibriumProfiles2d| item.grid_type.name.clone(), "grid_type.name"),
            index: Accumulator::new(data, |item: &EquilibriumProfiles2d| item.grid_type.index, "grid_type.index"),
            description: StringAccumulator::new(data, |item: &EquilibriumProfiles2d| item.grid_type.description.clone(), "grid_type.description"),
        }
    }
}

/// View over `grid` (EquilibriumProfiles2dGrid) across multiple EquilibriumProfiles2d
pub struct EquilibriumProfiles2dGridView<'a> {
    _phantom: std::marker::PhantomData<&'a EquilibriumProfiles2d>,
}

impl<'a> EquilibriumProfiles2dGridView<'a> {
    pub fn new(_data: &'a [EquilibriumProfiles2d]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over multiple EquilibriumProfiles2d with field accumulation
pub struct EquilibriumProfiles2dSliceView<'a> {
    data: &'a [EquilibriumProfiles2d],
    pub r#type: EquilibriumProfiles2dTypeView<'a>,
    pub grid_type: EquilibriumProfiles2dGridTypeView<'a>,
    pub grid: EquilibriumProfiles2dGridView<'a>,
}

impl<'a> EquilibriumProfiles2dSliceView<'a> {
    pub fn new(data: &'a [EquilibriumProfiles2d]) -> Self {
        Self {
            data,
            r#type: EquilibriumProfiles2dTypeView::new(data),
            grid_type: EquilibriumProfiles2dGridTypeView::new(data),
            grid: EquilibriumProfiles2dGridView::new(data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumProfiles2d> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumProfiles2d
pub struct EquilibriumProfiles2dSliceViewMut<'a> {
    data: &'a mut [EquilibriumProfiles2d],
}

impl<'a> EquilibriumProfiles2dSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumProfiles2d]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumProfiles2d> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumProfiles2d - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumProfiles2dIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output;
}

impl<'a> EquilibriumProfiles2dIndex<'a> for usize {
    type Output = &'a EquilibriumProfiles2d;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumProfiles2dIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumProfiles2dSliceView<'a>;
    fn get(self, data: &'a [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumProfiles2d - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumProfiles2dMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output;
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumProfiles2d;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumProfiles2dMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumProfiles2dSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumProfiles2d]) -> Self::Output {
        EquilibriumProfiles2dSliceViewMut::new(data)
    }
}

// --- EquilibriumGgd View Types ---

/// View over multiple EquilibriumGgd with field accumulation
pub struct EquilibriumGgdSliceView<'a> {
    data: &'a [EquilibriumGgd],
}

impl<'a> EquilibriumGgdSliceView<'a> {
    pub fn new(data: &'a [EquilibriumGgd]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumGgd> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumGgd
pub struct EquilibriumGgdSliceViewMut<'a> {
    data: &'a mut [EquilibriumGgd],
}

impl<'a> EquilibriumGgdSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumGgd]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumGgd> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumGgd - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumGgdIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output;
}

impl<'a> EquilibriumGgdIndex<'a> for usize {
    type Output = &'a EquilibriumGgd;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGgdSliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumGgd - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumGgdMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output;
}

impl<'a> EquilibriumGgdMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumGgd;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGgdSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgd]) -> Self::Output {
        EquilibriumGgdSliceViewMut::new(data)
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

// --- EquilibriumGgdArray View Types ---

/// View over multiple EquilibriumGgdArray with field accumulation
pub struct EquilibriumGgdArraySliceView<'a> {
    data: &'a [EquilibriumGgdArray],
    pub time: Accumulator<'a, EquilibriumGgdArray, FLT_0D>,
}

impl<'a> EquilibriumGgdArraySliceView<'a> {
    pub fn new(data: &'a [EquilibriumGgdArray]) -> Self {
        Self {
            data,
            time: Accumulator::new(data, |item: &EquilibriumGgdArray| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumGgdArray> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumGgdArray
pub struct EquilibriumGgdArraySliceViewMut<'a> {
    data: &'a mut [EquilibriumGgdArray],
}

impl<'a> EquilibriumGgdArraySliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumGgdArray]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumGgdArray> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumGgdArray - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumGgdArrayIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output;
}

impl<'a> EquilibriumGgdArrayIndex<'a> for usize {
    type Output = &'a EquilibriumGgdArray;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(&data[self])
    }
}

impl<'a> EquilibriumGgdArrayIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGgdArraySliceView<'a>;
    fn get(self, data: &'a [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumGgdArray - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumGgdArrayMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output;
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumGgdArray;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumGgdArrayMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumGgdArraySliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumGgdArray]) -> Self::Output {
        EquilibriumGgdArraySliceViewMut::new(data)
    }
}

// --- EquilibriumTimeSlice View Types ---

/// View over `boundary.outline` (Rz1dDynamicAos) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryOutlineView<'a> {
    _phantom: std::marker::PhantomData<&'a EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceBoundaryOutlineView<'a> {
    pub fn new(_data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `boundary.geometric_axis` (Rz0dDynamicAos) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryGeometricAxisView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceBoundaryGeometricAxisView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.geometric_axis.r, "boundary.geometric_axis.r"),
            z: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.geometric_axis.z, "boundary.geometric_axis.z"),
        }
    }
}

/// View over `boundary.closest_wall_point` (EquilibriumBoundaryClosest) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryClosestWallPointView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub distance: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceBoundaryClosestWallPointView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.closest_wall_point.r,
                "boundary.closest_wall_point.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.closest_wall_point.z,
                "boundary.closest_wall_point.z",
            ),
            distance: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.closest_wall_point.distance,
                "boundary.closest_wall_point.distance",
            ),
        }
    }
}

/// View over `boundary.dr_dz_zero_point` (Rz0dDynamicAos) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryDrDzZeroPointView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceBoundaryDrDzZeroPointView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.dr_dz_zero_point.r,
                "boundary.dr_dz_zero_point.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.dr_dz_zero_point.z,
                "boundary.dr_dz_zero_point.z",
            ),
        }
    }
}

/// View over `boundary.bounding` (EquilibriumBoundaryBounding) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryBoundingView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceBoundaryBoundingView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.bounding.r, "boundary.bounding.r"),
            z: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.bounding.z, "boundary.bounding.z"),
        }
    }
}

/// View over `boundary` (EquilibriumBoundary) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceBoundaryView<'a> {
    pub r#type: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub outline: EquilibriumTimeSliceBoundaryOutlineView<'a>,
    pub psi_norm: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub psi: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub geometric_axis: EquilibriumTimeSliceBoundaryGeometricAxisView<'a>,
    pub minor_radius: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub elongation: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub triangularity: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub triangularity_upper: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub triangularity_lower: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub squareness_upper_inner: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub squareness_upper_outer: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub squareness_lower_inner: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub squareness_lower_outer: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub closest_wall_point: EquilibriumTimeSliceBoundaryClosestWallPointView<'a>,
    pub dr_dz_zero_point: EquilibriumTimeSliceBoundaryDrDzZeroPointView<'a>,
    pub rho_tor: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub phi: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub phi_poloidal_current: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub bounding: EquilibriumTimeSliceBoundaryBoundingView<'a>,
}

impl<'a> EquilibriumTimeSliceBoundaryView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r#type: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.r#type, "boundary.type"),
            outline: EquilibriumTimeSliceBoundaryOutlineView::new(data),
            psi_norm: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.psi_norm, "boundary.psi_norm"),
            psi: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.psi, "boundary.psi"),
            geometric_axis: EquilibriumTimeSliceBoundaryGeometricAxisView::new(data),
            minor_radius: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.minor_radius, "boundary.minor_radius"),
            elongation: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.elongation, "boundary.elongation"),
            triangularity: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.triangularity, "boundary.triangularity"),
            triangularity_upper: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.triangularity_upper,
                "boundary.triangularity_upper",
            ),
            triangularity_lower: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.triangularity_lower,
                "boundary.triangularity_lower",
            ),
            squareness_upper_inner: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.squareness_upper_inner,
                "boundary.squareness_upper_inner",
            ),
            squareness_upper_outer: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.squareness_upper_outer,
                "boundary.squareness_upper_outer",
            ),
            squareness_lower_inner: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.squareness_lower_inner,
                "boundary.squareness_lower_inner",
            ),
            squareness_lower_outer: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.squareness_lower_outer,
                "boundary.squareness_lower_outer",
            ),
            closest_wall_point: EquilibriumTimeSliceBoundaryClosestWallPointView::new(data),
            dr_dz_zero_point: EquilibriumTimeSliceBoundaryDrDzZeroPointView::new(data),
            rho_tor: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.rho_tor, "boundary.rho_tor"),
            phi: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.boundary.phi, "boundary.phi"),
            phi_poloidal_current: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.boundary.phi_poloidal_current,
                "boundary.phi_poloidal_current",
            ),
            bounding: EquilibriumTimeSliceBoundaryBoundingView::new(data),
        }
    }
}

/// View over `contour_tree` (EquilibriumContourTree) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceContourTreeView<'a> {
    _phantom: std::marker::PhantomData<&'a EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceContourTreeView<'a> {
    pub fn new(_data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `constraints.b_field_tor_vacuum_r` (EquilibriumConstraints0d) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConstraintsBFieldTorVacuumRView<'a> {
    pub measured: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub time_measurement: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceConstraintsBFieldTorVacuumRView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            measured: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.measured,
                "constraints.b_field_tor_vacuum_r.measured",
            ),
            source: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.source.clone(),
                "constraints.b_field_tor_vacuum_r.source",
            ),
            time_measurement: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.time_measurement,
                "constraints.b_field_tor_vacuum_r.time_measurement",
            ),
            exact: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.exact,
                "constraints.b_field_tor_vacuum_r.exact",
            ),
            weight: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.weight,
                "constraints.b_field_tor_vacuum_r.weight",
            ),
            sigma: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.sigma,
                "constraints.b_field_tor_vacuum_r.sigma",
            ),
            reconstructed: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.reconstructed,
                "constraints.b_field_tor_vacuum_r.reconstructed",
            ),
            chi_squared: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.b_field_tor_vacuum_r.chi_squared,
                "constraints.b_field_tor_vacuum_r.chi_squared",
            ),
        }
    }
}

/// View over `constraints.diamagnetic_flux` (EquilibriumConstraints0dB0Like) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConstraintsDiamagneticFluxView<'a> {
    pub measured: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub time_measurement: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceConstraintsDiamagneticFluxView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            measured: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.measured,
                "constraints.diamagnetic_flux.measured",
            ),
            source: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.source.clone(),
                "constraints.diamagnetic_flux.source",
            ),
            time_measurement: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.time_measurement,
                "constraints.diamagnetic_flux.time_measurement",
            ),
            exact: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.exact,
                "constraints.diamagnetic_flux.exact",
            ),
            weight: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.weight,
                "constraints.diamagnetic_flux.weight",
            ),
            sigma: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.sigma,
                "constraints.diamagnetic_flux.sigma",
            ),
            reconstructed: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.reconstructed,
                "constraints.diamagnetic_flux.reconstructed",
            ),
            chi_squared: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.diamagnetic_flux.chi_squared,
                "constraints.diamagnetic_flux.chi_squared",
            ),
        }
    }
}

/// View over `constraints.ip` (EquilibriumConstraints0dIpLike) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConstraintsIpView<'a> {
    pub measured: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub source: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub time_measurement: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub exact: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub weight: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub sigma: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub reconstructed: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub chi_squared: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceConstraintsIpView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            measured: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.ip.measured, "constraints.ip.measured"),
            source: StringAccumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.ip.source.clone(), "constraints.ip.source"),
            time_measurement: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.ip.time_measurement,
                "constraints.ip.time_measurement",
            ),
            exact: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.ip.exact, "constraints.ip.exact"),
            weight: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.ip.weight, "constraints.ip.weight"),
            sigma: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.ip.sigma, "constraints.ip.sigma"),
            reconstructed: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.ip.reconstructed,
                "constraints.ip.reconstructed",
            ),
            chi_squared: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.ip.chi_squared,
                "constraints.ip.chi_squared",
            ),
        }
    }
}

/// View over `constraints` (EquilibriumConstraints) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConstraintsView<'a> {
    pub b_field_tor_vacuum_r: EquilibriumTimeSliceConstraintsBFieldTorVacuumRView<'a>,
    pub diamagnetic_flux: EquilibriumTimeSliceConstraintsDiamagneticFluxView<'a>,
    pub ip: EquilibriumTimeSliceConstraintsIpView<'a>,
    pub chi_squared_reduced: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub freedom_degrees_n: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub constraints_n: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
}

impl<'a> EquilibriumTimeSliceConstraintsView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            b_field_tor_vacuum_r: EquilibriumTimeSliceConstraintsBFieldTorVacuumRView::new(data),
            diamagnetic_flux: EquilibriumTimeSliceConstraintsDiamagneticFluxView::new(data),
            ip: EquilibriumTimeSliceConstraintsIpView::new(data),
            chi_squared_reduced: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.chi_squared_reduced,
                "constraints.chi_squared_reduced",
            ),
            freedom_degrees_n: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.constraints.freedom_degrees_n,
                "constraints.freedom_degrees_n",
            ),
            constraints_n: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.constraints.constraints_n, "constraints.constraints_n"),
        }
    }
}

/// View over `global_quantities.magnetic_axis` (EquilibriumGlobalQuantitiesMagneticAxis) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceGlobalQuantitiesMagneticAxisView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub b_field_phi: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceGlobalQuantitiesMagneticAxisView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.magnetic_axis.r,
                "global_quantities.magnetic_axis.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.magnetic_axis.z,
                "global_quantities.magnetic_axis.z",
            ),
            b_field_phi: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.magnetic_axis.b_field_phi,
                "global_quantities.magnetic_axis.b_field_phi",
            ),
        }
    }
}

/// View over `global_quantities.current_centre` (EquilibriumGlobalQuantitiesCurrentCentre) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceGlobalQuantitiesCurrentCentreView<'a> {
    pub r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub velocity_z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceGlobalQuantitiesCurrentCentreView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.current_centre.r,
                "global_quantities.current_centre.r",
            ),
            z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.current_centre.z,
                "global_quantities.current_centre.z",
            ),
            velocity_z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.current_centre.velocity_z,
                "global_quantities.current_centre.velocity_z",
            ),
        }
    }
}

/// View over `global_quantities.q_min` (EquilibriumGlobalQuantitiesQmin) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceGlobalQuantitiesQMinView<'a> {
    pub value: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub rho_tor_norm: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub psi_norm: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub psi: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceGlobalQuantitiesQMinView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            value: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.q_min.value,
                "global_quantities.q_min.value",
            ),
            rho_tor_norm: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.q_min.rho_tor_norm,
                "global_quantities.q_min.rho_tor_norm",
            ),
            psi_norm: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.q_min.psi_norm,
                "global_quantities.q_min.psi_norm",
            ),
            psi: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.q_min.psi,
                "global_quantities.q_min.psi",
            ),
        }
    }
}

/// View over `global_quantities` (EqulibriumGlobalQuantities) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceGlobalQuantitiesView<'a> {
    pub beta_pol: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub beta_tor: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub beta_tor_norm: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub ip: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub li_3: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub volume: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub area: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub surface: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub length_pol: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub psi_magnetic_axis: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub magnetic_axis: EquilibriumTimeSliceGlobalQuantitiesMagneticAxisView<'a>,
    pub current_centre: EquilibriumTimeSliceGlobalQuantitiesCurrentCentreView<'a>,
    pub q_axis: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub q_95: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub q_min: EquilibriumTimeSliceGlobalQuantitiesQMinView<'a>,
    pub energy_mhd: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub psi_external_average: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub v_external: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub plasma_inductance: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub plasma_resistance: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub xpt_upper_r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub xpt_upper_z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub xpt_lower_r: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub xpt_lower_z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceGlobalQuantitiesView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            beta_pol: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.beta_pol,
                "global_quantities.beta_pol",
            ),
            beta_tor: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.beta_tor,
                "global_quantities.beta_tor",
            ),
            beta_tor_norm: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.beta_tor_norm,
                "global_quantities.beta_tor_norm",
            ),
            ip: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.ip, "global_quantities.ip"),
            li_3: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.li_3, "global_quantities.li_3"),
            volume: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.volume, "global_quantities.volume"),
            area: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.area, "global_quantities.area"),
            surface: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.surface, "global_quantities.surface"),
            length_pol: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.length_pol,
                "global_quantities.length_pol",
            ),
            psi_magnetic_axis: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.psi_magnetic_axis,
                "global_quantities.psi_magnetic_axis",
            ),
            magnetic_axis: EquilibriumTimeSliceGlobalQuantitiesMagneticAxisView::new(data),
            current_centre: EquilibriumTimeSliceGlobalQuantitiesCurrentCentreView::new(data),
            q_axis: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.q_axis, "global_quantities.q_axis"),
            q_95: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.global_quantities.q_95, "global_quantities.q_95"),
            q_min: EquilibriumTimeSliceGlobalQuantitiesQMinView::new(data),
            energy_mhd: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.energy_mhd,
                "global_quantities.energy_mhd",
            ),
            psi_external_average: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.psi_external_average,
                "global_quantities.psi_external_average",
            ),
            v_external: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.v_external,
                "global_quantities.v_external",
            ),
            plasma_inductance: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.plasma_inductance,
                "global_quantities.plasma_inductance",
            ),
            plasma_resistance: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.plasma_resistance,
                "global_quantities.plasma_resistance",
            ),
            xpt_upper_r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.xpt_upper_r,
                "global_quantities.xpt_upper_r",
            ),
            xpt_upper_z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.xpt_upper_z,
                "global_quantities.xpt_upper_z",
            ),
            xpt_lower_r: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.xpt_lower_r,
                "global_quantities.xpt_lower_r",
            ),
            xpt_lower_z: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.global_quantities.xpt_lower_z,
                "global_quantities.xpt_lower_z",
            ),
        }
    }
}

/// View over `profiles_1d.geometric_axis` (EquilibriumProfiles1dRz1dDynamicAos) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceProfiles1dGeometricAxisView<'a> {
    _phantom: std::marker::PhantomData<&'a EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceProfiles1dGeometricAxisView<'a> {
    pub fn new(_data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `profiles_1d` (EquilibriumProfiles1d) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceProfiles1dView<'a> {
    pub geometric_axis: EquilibriumTimeSliceProfiles1dGeometricAxisView<'a>,
}

impl<'a> EquilibriumTimeSliceProfiles1dView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            geometric_axis: EquilibriumTimeSliceProfiles1dGeometricAxisView::new(data),
        }
    }
}

/// View over `coordinate_system.grid_type` (IdentifierDynamicAos3) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceCoordinateSystemGridTypeView<'a> {
    pub name: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub index: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceCoordinateSystemGridTypeView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.coordinate_system.grid_type.name.clone(),
                "coordinate_system.grid_type.name",
            ),
            index: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.coordinate_system.grid_type.index,
                "coordinate_system.grid_type.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.coordinate_system.grid_type.description.clone(),
                "coordinate_system.grid_type.description",
            ),
        }
    }
}

/// View over `coordinate_system.grid` (EquilibriumProfiles2dGrid) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceCoordinateSystemGridView<'a> {
    _phantom: std::marker::PhantomData<&'a EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceCoordinateSystemGridView<'a> {
    pub fn new(_data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// View over `coordinate_system` (EquilibriumCoordinateSystem) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceCoordinateSystemView<'a> {
    pub grid_type: EquilibriumTimeSliceCoordinateSystemGridTypeView<'a>,
    pub grid: EquilibriumTimeSliceCoordinateSystemGridView<'a>,
}

impl<'a> EquilibriumTimeSliceCoordinateSystemView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            grid_type: EquilibriumTimeSliceCoordinateSystemGridTypeView::new(data),
            grid: EquilibriumTimeSliceCoordinateSystemGridView::new(data),
        }
    }
}

/// View over `convergence.grad_shafranov_deviation_expression` (IdentifierDynamicAos3) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConvergenceGradShafranovDeviationExpressionView<'a> {
    pub name: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub index: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceConvergenceGradShafranovDeviationExpressionView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.grad_shafranov_deviation_expression.name.clone(),
                "convergence.grad_shafranov_deviation_expression.name",
            ),
            index: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.grad_shafranov_deviation_expression.index,
                "convergence.grad_shafranov_deviation_expression.index",
            ),
            description: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.grad_shafranov_deviation_expression.description.clone(),
                "convergence.grad_shafranov_deviation_expression.description",
            ),
        }
    }
}

/// View over `convergence.result` (IdentifierDynamicAos3) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConvergenceResultView<'a> {
    pub name: StringAccumulator<'a, EquilibriumTimeSlice>,
    pub index: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub description: StringAccumulator<'a, EquilibriumTimeSlice>,
}

impl<'a> EquilibriumTimeSliceConvergenceResultView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            name: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.result.name.clone(),
                "convergence.result.name",
            ),
            index: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.convergence.result.index, "convergence.result.index"),
            description: StringAccumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.result.description.clone(),
                "convergence.result.description",
            ),
        }
    }
}

/// View over `convergence` (EquilibriumConvergence) across multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceConvergenceView<'a> {
    pub iterations_n: Accumulator<'a, EquilibriumTimeSlice, INT_0D>,
    pub grad_shafranov_deviation_expression: EquilibriumTimeSliceConvergenceGradShafranovDeviationExpressionView<'a>,
    pub grad_shafranov_deviation_value: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
    pub result: EquilibriumTimeSliceConvergenceResultView<'a>,
    pub delta_z: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceConvergenceView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            iterations_n: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.convergence.iterations_n, "convergence.iterations_n"),
            grad_shafranov_deviation_expression: EquilibriumTimeSliceConvergenceGradShafranovDeviationExpressionView::new(data),
            grad_shafranov_deviation_value: Accumulator::new(
                data,
                |item: &EquilibriumTimeSlice| item.convergence.grad_shafranov_deviation_value,
                "convergence.grad_shafranov_deviation_value",
            ),
            result: EquilibriumTimeSliceConvergenceResultView::new(data),
            delta_z: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.convergence.delta_z, "convergence.delta_z"),
        }
    }
}

/// View over multiple EquilibriumTimeSlice with field accumulation
pub struct EquilibriumTimeSliceSliceView<'a> {
    data: &'a [EquilibriumTimeSlice],
    pub boundary: EquilibriumTimeSliceBoundaryView<'a>,
    pub contour_tree: EquilibriumTimeSliceContourTreeView<'a>,
    pub constraints: EquilibriumTimeSliceConstraintsView<'a>,
    pub global_quantities: EquilibriumTimeSliceGlobalQuantitiesView<'a>,
    pub profiles_1d: EquilibriumTimeSliceProfiles1dView<'a>,
    pub coordinate_system: EquilibriumTimeSliceCoordinateSystemView<'a>,
    pub convergence: EquilibriumTimeSliceConvergenceView<'a>,
    pub time: Accumulator<'a, EquilibriumTimeSlice, FLT_0D>,
}

impl<'a> EquilibriumTimeSliceSliceView<'a> {
    pub fn new(data: &'a [EquilibriumTimeSlice]) -> Self {
        Self {
            data,
            boundary: EquilibriumTimeSliceBoundaryView::new(data),
            contour_tree: EquilibriumTimeSliceContourTreeView::new(data),
            constraints: EquilibriumTimeSliceConstraintsView::new(data),
            global_quantities: EquilibriumTimeSliceGlobalQuantitiesView::new(data),
            profiles_1d: EquilibriumTimeSliceProfiles1dView::new(data),
            coordinate_system: EquilibriumTimeSliceCoordinateSystemView::new(data),
            convergence: EquilibriumTimeSliceConvergenceView::new(data),
            time: Accumulator::new(data, |item: &EquilibriumTimeSlice| item.time, "time"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquilibriumTimeSlice> {
        self.data.iter()
    }
}

/// Mutable view over multiple EquilibriumTimeSlice
pub struct EquilibriumTimeSliceSliceViewMut<'a> {
    data: &'a mut [EquilibriumTimeSlice],
}

impl<'a> EquilibriumTimeSliceSliceViewMut<'a> {
    pub fn new(data: &'a mut [EquilibriumTimeSlice]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EquilibriumTimeSlice> {
        self.data.iter_mut()
    }
}

/// Index trait for EquilibriumTimeSlice - enables .field(0) and .field(0..2) syntax
pub trait EquilibriumTimeSliceIndex<'a> {
    type Output;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output;
}

impl<'a> EquilibriumTimeSliceIndex<'a> for usize {
    type Output = &'a EquilibriumTimeSlice;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        &data[self]
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(&data[self])
    }
}

impl<'a> EquilibriumTimeSliceIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumTimeSliceSliceView<'a>;
    fn get(self, data: &'a [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceView::new(data)
    }
}

/// Mutable index trait for EquilibriumTimeSlice - enables .field_mut(0) and .field_mut(0..2) syntax
pub trait EquilibriumTimeSliceMutIndex<'a> {
    type Output;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output;
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for usize {
    type Output = &'a mut EquilibriumTimeSlice;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        &mut data[self]
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::Range<usize> {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::RangeFrom<usize> {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::RangeTo<usize> {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::RangeInclusive<usize> {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::RangeToInclusive<usize> {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(&mut data[self])
    }
}

impl<'a> EquilibriumTimeSliceMutIndex<'a> for std::ops::RangeFull {
    type Output = EquilibriumTimeSliceSliceViewMut<'a>;
    fn get_mut(self, data: &'a mut [EquilibriumTimeSlice]) -> Self::Output {
        EquilibriumTimeSliceSliceViewMut::new(data)
    }
}

// ============================================================================
// Struct Impl Blocks for Vec Field Access
// ============================================================================

impl EquilibriumContourTreeNode {
    /// Access levelset - use index for single element or range for slice view
    /// e.g. `.levelset(0)` returns `&Rz1dDynamicAos`, `.levelset(0..2)` returns `Rz1dDynamicAosSliceView`
    pub fn levelset<'a, I: Rz1dDynamicAosIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.levelset)
    }

    /// Access levelset mutably - use index for single element or range for slice view
    /// e.g. `.levelset_mut(0)` returns `&mut Rz1dDynamicAos`, `.levelset_mut(0..2)` returns `Rz1dDynamicAosSliceViewMut`
    pub fn levelset_mut<'a, I: Rz1dDynamicAosMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.levelset)
    }

    /// Get the number of levelset elements
    pub fn levelset_len(&self) -> usize {
        self.levelset.len()
    }
}

impl EquilibriumContourTree {
    /// Access node - use index for single element or range for slice view
    /// e.g. `.node(0)` returns `&EquilibriumContourTreeNode`, `.node(0..2)` returns `EquilibriumContourTreeNodeSliceView`
    pub fn node<'a, I: EquilibriumContourTreeNodeIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.node)
    }

    /// Access node mutably - use index for single element or range for slice view
    /// e.g. `.node_mut(0)` returns `&mut EquilibriumContourTreeNode`, `.node_mut(0..2)` returns `EquilibriumContourTreeNodeSliceViewMut`
    pub fn node_mut<'a, I: EquilibriumContourTreeNodeMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.node)
    }

    /// Get the number of node elements
    pub fn node_len(&self) -> usize {
        self.node.len()
    }
}

impl EquilibriumBoundary {
    /// Access gap - use index for single element or range for slice view
    /// e.g. `.gap(0)` returns `&EquilibriumGap`, `.gap(0..2)` returns `EquilibriumGapSliceView`
    pub fn gap<'a, I: EquilibriumGapIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.gap)
    }

    /// Access gap mutably - use index for single element or range for slice view
    /// e.g. `.gap_mut(0)` returns `&mut EquilibriumGap`, `.gap_mut(0..2)` returns `EquilibriumGapSliceViewMut`
    pub fn gap_mut<'a, I: EquilibriumGapMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.gap)
    }

    /// Get the number of gap elements
    pub fn gap_len(&self) -> usize {
        self.gap.len()
    }
}

impl EquilibriumConstraints {
    /// Access b_field_pol_probe - use index for single element or range for slice view
    /// e.g. `.b_field_pol_probe(0)` returns `&EquilibriumConstraints0dOneLike`, `.b_field_pol_probe(0..2)` returns `EquilibriumConstraints0dOneLikeSliceView`
    pub fn b_field_pol_probe<'a, I: EquilibriumConstraints0dOneLikeIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.b_field_pol_probe)
    }

    /// Access b_field_pol_probe mutably - use index for single element or range for slice view
    /// e.g. `.b_field_pol_probe_mut(0)` returns `&mut EquilibriumConstraints0dOneLike`, `.b_field_pol_probe_mut(0..2)` returns `EquilibriumConstraints0dOneLikeSliceViewMut`
    pub fn b_field_pol_probe_mut<'a, I: EquilibriumConstraints0dOneLikeMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.b_field_pol_probe)
    }

    /// Get the number of b_field_pol_probe elements
    pub fn b_field_pol_probe_len(&self) -> usize {
        self.b_field_pol_probe.len()
    }
}

impl EquilibriumConstraints {
    /// Access faraday_angle - use index for single element or range for slice view
    /// e.g. `.faraday_angle(0)` returns `&EquilibriumConstraints0d`, `.faraday_angle(0..2)` returns `EquilibriumConstraints0dSliceView`
    pub fn faraday_angle<'a, I: EquilibriumConstraints0dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.faraday_angle)
    }

    /// Access faraday_angle mutably - use index for single element or range for slice view
    /// e.g. `.faraday_angle_mut(0)` returns `&mut EquilibriumConstraints0d`, `.faraday_angle_mut(0..2)` returns `EquilibriumConstraints0dSliceViewMut`
    pub fn faraday_angle_mut<'a, I: EquilibriumConstraints0dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.faraday_angle)
    }

    /// Get the number of faraday_angle elements
    pub fn faraday_angle_len(&self) -> usize {
        self.faraday_angle.len()
    }
}

impl EquilibriumConstraints {
    /// Access mse_polarization_angle - use index for single element or range for slice view
    /// e.g. `.mse_polarization_angle(0)` returns `&EquilibriumConstraints0d`, `.mse_polarization_angle(0..2)` returns `EquilibriumConstraints0dSliceView`
    pub fn mse_polarization_angle<'a, I: EquilibriumConstraints0dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.mse_polarization_angle)
    }

    /// Access mse_polarization_angle mutably - use index for single element or range for slice view
    /// e.g. `.mse_polarization_angle_mut(0)` returns `&mut EquilibriumConstraints0d`, `.mse_polarization_angle_mut(0..2)` returns `EquilibriumConstraints0dSliceViewMut`
    pub fn mse_polarization_angle_mut<'a, I: EquilibriumConstraints0dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.mse_polarization_angle)
    }

    /// Get the number of mse_polarization_angle elements
    pub fn mse_polarization_angle_len(&self) -> usize {
        self.mse_polarization_angle.len()
    }
}

impl EquilibriumConstraints {
    /// Access flux_loop - use index for single element or range for slice view
    /// e.g. `.flux_loop(0)` returns `&EquilibriumConstraints0d`, `.flux_loop(0..2)` returns `EquilibriumConstraints0dSliceView`
    pub fn flux_loop<'a, I: EquilibriumConstraints0dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.flux_loop)
    }

    /// Access flux_loop mutably - use index for single element or range for slice view
    /// e.g. `.flux_loop_mut(0)` returns `&mut EquilibriumConstraints0d`, `.flux_loop_mut(0..2)` returns `EquilibriumConstraints0dSliceViewMut`
    pub fn flux_loop_mut<'a, I: EquilibriumConstraints0dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.flux_loop)
    }

    /// Get the number of flux_loop elements
    pub fn flux_loop_len(&self) -> usize {
        self.flux_loop.len()
    }
}

impl EquilibriumConstraints {
    /// Access iron_core_segment - use index for single element or range for slice view
    /// e.g. `.iron_core_segment(0)` returns `&EquilibriumConstraintsMagnetization`, `.iron_core_segment(0..2)` returns `EquilibriumConstraintsMagnetizationSliceView`
    pub fn iron_core_segment<'a, I: EquilibriumConstraintsMagnetizationIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.iron_core_segment)
    }

    /// Access iron_core_segment mutably - use index for single element or range for slice view
    /// e.g. `.iron_core_segment_mut(0)` returns `&mut EquilibriumConstraintsMagnetization`, `.iron_core_segment_mut(0..2)` returns `EquilibriumConstraintsMagnetizationSliceViewMut`
    pub fn iron_core_segment_mut<'a, I: EquilibriumConstraintsMagnetizationMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.iron_core_segment)
    }

    /// Get the number of iron_core_segment elements
    pub fn iron_core_segment_len(&self) -> usize {
        self.iron_core_segment.len()
    }
}

impl EquilibriumConstraints {
    /// Access n_e - use index for single element or range for slice view
    /// e.g. `.n_e(0)` returns `&EquilibriumConstraints0dPosition`, `.n_e(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn n_e<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.n_e)
    }

    /// Access n_e mutably - use index for single element or range for slice view
    /// e.g. `.n_e_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.n_e_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn n_e_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.n_e)
    }

    /// Get the number of n_e elements
    pub fn n_e_len(&self) -> usize {
        self.n_e.len()
    }
}

impl EquilibriumConstraints {
    /// Access n_e_line - use index for single element or range for slice view
    /// e.g. `.n_e_line(0)` returns `&EquilibriumConstraints0d`, `.n_e_line(0..2)` returns `EquilibriumConstraints0dSliceView`
    pub fn n_e_line<'a, I: EquilibriumConstraints0dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.n_e_line)
    }

    /// Access n_e_line mutably - use index for single element or range for slice view
    /// e.g. `.n_e_line_mut(0)` returns `&mut EquilibriumConstraints0d`, `.n_e_line_mut(0..2)` returns `EquilibriumConstraints0dSliceViewMut`
    pub fn n_e_line_mut<'a, I: EquilibriumConstraints0dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.n_e_line)
    }

    /// Get the number of n_e_line elements
    pub fn n_e_line_len(&self) -> usize {
        self.n_e_line.len()
    }
}

impl EquilibriumConstraints {
    /// Access pf_current - use index for single element or range for slice view
    /// e.g. `.pf_current(0)` returns `&EquilibriumConstraints0dIpLike`, `.pf_current(0..2)` returns `EquilibriumConstraints0dIpLikeSliceView`
    pub fn pf_current<'a, I: EquilibriumConstraints0dIpLikeIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.pf_current)
    }

    /// Access pf_current mutably - use index for single element or range for slice view
    /// e.g. `.pf_current_mut(0)` returns `&mut EquilibriumConstraints0dIpLike`, `.pf_current_mut(0..2)` returns `EquilibriumConstraints0dIpLikeSliceViewMut`
    pub fn pf_current_mut<'a, I: EquilibriumConstraints0dIpLikeMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.pf_current)
    }

    /// Get the number of pf_current elements
    pub fn pf_current_len(&self) -> usize {
        self.pf_current.len()
    }
}

impl EquilibriumConstraints {
    /// Access pf_passive_current - use index for single element or range for slice view
    /// e.g. `.pf_passive_current(0)` returns `&EquilibriumConstraints0d`, `.pf_passive_current(0..2)` returns `EquilibriumConstraints0dSliceView`
    pub fn pf_passive_current<'a, I: EquilibriumConstraints0dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.pf_passive_current)
    }

    /// Access pf_passive_current mutably - use index for single element or range for slice view
    /// e.g. `.pf_passive_current_mut(0)` returns `&mut EquilibriumConstraints0d`, `.pf_passive_current_mut(0..2)` returns `EquilibriumConstraints0dSliceViewMut`
    pub fn pf_passive_current_mut<'a, I: EquilibriumConstraints0dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.pf_passive_current)
    }

    /// Get the number of pf_passive_current elements
    pub fn pf_passive_current_len(&self) -> usize {
        self.pf_passive_current.len()
    }
}

impl EquilibriumConstraints {
    /// Access pressure - use index for single element or range for slice view
    /// e.g. `.pressure(0)` returns `&EquilibriumConstraints0dPosition`, `.pressure(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn pressure<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.pressure)
    }

    /// Access pressure mutably - use index for single element or range for slice view
    /// e.g. `.pressure_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.pressure_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn pressure_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.pressure)
    }

    /// Get the number of pressure elements
    pub fn pressure_len(&self) -> usize {
        self.pressure.len()
    }
}

impl EquilibriumConstraints {
    /// Access pressure_rotational - use index for single element or range for slice view
    /// e.g. `.pressure_rotational(0)` returns `&EquilibriumConstraints0dPosition`, `.pressure_rotational(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn pressure_rotational<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.pressure_rotational)
    }

    /// Access pressure_rotational mutably - use index for single element or range for slice view
    /// e.g. `.pressure_rotational_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.pressure_rotational_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn pressure_rotational_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.pressure_rotational)
    }

    /// Get the number of pressure_rotational elements
    pub fn pressure_rotational_len(&self) -> usize {
        self.pressure_rotational.len()
    }
}

impl EquilibriumConstraints {
    /// Access q - use index for single element or range for slice view
    /// e.g. `.q(0)` returns `&EquilibriumConstraints0dPosition`, `.q(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn q<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.q)
    }

    /// Access q mutably - use index for single element or range for slice view
    /// e.g. `.q_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.q_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn q_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.q)
    }

    /// Get the number of q elements
    pub fn q_len(&self) -> usize {
        self.q.len()
    }
}

impl EquilibriumConstraints {
    /// Access j_phi - use index for single element or range for slice view
    /// e.g. `.j_phi(0)` returns `&EquilibriumConstraints0dPosition`, `.j_phi(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn j_phi<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.j_phi)
    }

    /// Access j_phi mutably - use index for single element or range for slice view
    /// e.g. `.j_phi_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.j_phi_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn j_phi_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.j_phi)
    }

    /// Get the number of j_phi elements
    pub fn j_phi_len(&self) -> usize {
        self.j_phi.len()
    }
}

impl EquilibriumConstraints {
    /// Access j_parallel - use index for single element or range for slice view
    /// e.g. `.j_parallel(0)` returns `&EquilibriumConstraints0dPosition`, `.j_parallel(0..2)` returns `EquilibriumConstraints0dPositionSliceView`
    pub fn j_parallel<'a, I: EquilibriumConstraints0dPositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.j_parallel)
    }

    /// Access j_parallel mutably - use index for single element or range for slice view
    /// e.g. `.j_parallel_mut(0)` returns `&mut EquilibriumConstraints0dPosition`, `.j_parallel_mut(0..2)` returns `EquilibriumConstraints0dPositionSliceViewMut`
    pub fn j_parallel_mut<'a, I: EquilibriumConstraints0dPositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.j_parallel)
    }

    /// Get the number of j_parallel elements
    pub fn j_parallel_len(&self) -> usize {
        self.j_parallel.len()
    }
}

impl EquilibriumConstraints {
    /// Access x_point - use index for single element or range for slice view
    /// e.g. `.x_point(0)` returns `&EquilibriumConstraintsPurePosition`, `.x_point(0..2)` returns `EquilibriumConstraintsPurePositionSliceView`
    pub fn x_point<'a, I: EquilibriumConstraintsPurePositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.x_point)
    }

    /// Access x_point mutably - use index for single element or range for slice view
    /// e.g. `.x_point_mut(0)` returns `&mut EquilibriumConstraintsPurePosition`, `.x_point_mut(0..2)` returns `EquilibriumConstraintsPurePositionSliceViewMut`
    pub fn x_point_mut<'a, I: EquilibriumConstraintsPurePositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.x_point)
    }

    /// Get the number of x_point elements
    pub fn x_point_len(&self) -> usize {
        self.x_point.len()
    }
}

impl EquilibriumConstraints {
    /// Access strike_point - use index for single element or range for slice view
    /// e.g. `.strike_point(0)` returns `&EquilibriumConstraintsPurePosition`, `.strike_point(0..2)` returns `EquilibriumConstraintsPurePositionSliceView`
    pub fn strike_point<'a, I: EquilibriumConstraintsPurePositionIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.strike_point)
    }

    /// Access strike_point mutably - use index for single element or range for slice view
    /// e.g. `.strike_point_mut(0)` returns `&mut EquilibriumConstraintsPurePosition`, `.strike_point_mut(0..2)` returns `EquilibriumConstraintsPurePositionSliceViewMut`
    pub fn strike_point_mut<'a, I: EquilibriumConstraintsPurePositionMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.strike_point)
    }

    /// Get the number of strike_point elements
    pub fn strike_point_len(&self) -> usize {
        self.strike_point.len()
    }
}

impl EquilibriumGgd {
    /// Access r - use index for single element or range for slice view
    /// e.g. `.r(0)` returns `&GenericGridScalar`, `.r(0..2)` returns `GenericGridScalarSliceView`
    pub fn r<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.r)
    }

    /// Access r mutably - use index for single element or range for slice view
    /// e.g. `.r_mut(0)` returns `&mut GenericGridScalar`, `.r_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn r_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.r)
    }

    /// Get the number of r elements
    pub fn r_len(&self) -> usize {
        self.r.len()
    }
}

impl EquilibriumGgd {
    /// Access z - use index for single element or range for slice view
    /// e.g. `.z(0)` returns `&GenericGridScalar`, `.z(0..2)` returns `GenericGridScalarSliceView`
    pub fn z<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.z)
    }

    /// Access z mutably - use index for single element or range for slice view
    /// e.g. `.z_mut(0)` returns `&mut GenericGridScalar`, `.z_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn z_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.z)
    }

    /// Get the number of z elements
    pub fn z_len(&self) -> usize {
        self.z.len()
    }
}

impl EquilibriumGgd {
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

impl EquilibriumGgd {
    /// Access phi - use index for single element or range for slice view
    /// e.g. `.phi(0)` returns `&GenericGridScalar`, `.phi(0..2)` returns `GenericGridScalarSliceView`
    pub fn phi<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.phi)
    }

    /// Access phi mutably - use index for single element or range for slice view
    /// e.g. `.phi_mut(0)` returns `&mut GenericGridScalar`, `.phi_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn phi_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.phi)
    }

    /// Get the number of phi elements
    pub fn phi_len(&self) -> usize {
        self.phi.len()
    }
}

impl EquilibriumGgd {
    /// Access theta - use index for single element or range for slice view
    /// e.g. `.theta(0)` returns `&GenericGridScalar`, `.theta(0..2)` returns `GenericGridScalarSliceView`
    pub fn theta<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.theta)
    }

    /// Access theta mutably - use index for single element or range for slice view
    /// e.g. `.theta_mut(0)` returns `&mut GenericGridScalar`, `.theta_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn theta_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.theta)
    }

    /// Get the number of theta elements
    pub fn theta_len(&self) -> usize {
        self.theta.len()
    }
}

impl EquilibriumGgd {
    /// Access j_phi - use index for single element or range for slice view
    /// e.g. `.j_phi(0)` returns `&GenericGridScalar`, `.j_phi(0..2)` returns `GenericGridScalarSliceView`
    pub fn j_phi<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.j_phi)
    }

    /// Access j_phi mutably - use index for single element or range for slice view
    /// e.g. `.j_phi_mut(0)` returns `&mut GenericGridScalar`, `.j_phi_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn j_phi_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.j_phi)
    }

    /// Get the number of j_phi elements
    pub fn j_phi_len(&self) -> usize {
        self.j_phi.len()
    }
}

impl EquilibriumGgd {
    /// Access j_parallel - use index for single element or range for slice view
    /// e.g. `.j_parallel(0)` returns `&GenericGridScalar`, `.j_parallel(0..2)` returns `GenericGridScalarSliceView`
    pub fn j_parallel<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.j_parallel)
    }

    /// Access j_parallel mutably - use index for single element or range for slice view
    /// e.g. `.j_parallel_mut(0)` returns `&mut GenericGridScalar`, `.j_parallel_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn j_parallel_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.j_parallel)
    }

    /// Get the number of j_parallel elements
    pub fn j_parallel_len(&self) -> usize {
        self.j_parallel.len()
    }
}

impl EquilibriumGgd {
    /// Access b_field_r - use index for single element or range for slice view
    /// e.g. `.b_field_r(0)` returns `&GenericGridScalar`, `.b_field_r(0..2)` returns `GenericGridScalarSliceView`
    pub fn b_field_r<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.b_field_r)
    }

    /// Access b_field_r mutably - use index for single element or range for slice view
    /// e.g. `.b_field_r_mut(0)` returns `&mut GenericGridScalar`, `.b_field_r_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn b_field_r_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.b_field_r)
    }

    /// Get the number of b_field_r elements
    pub fn b_field_r_len(&self) -> usize {
        self.b_field_r.len()
    }
}

impl EquilibriumGgd {
    /// Access b_field_phi - use index for single element or range for slice view
    /// e.g. `.b_field_phi(0)` returns `&GenericGridScalar`, `.b_field_phi(0..2)` returns `GenericGridScalarSliceView`
    pub fn b_field_phi<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.b_field_phi)
    }

    /// Access b_field_phi mutably - use index for single element or range for slice view
    /// e.g. `.b_field_phi_mut(0)` returns `&mut GenericGridScalar`, `.b_field_phi_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn b_field_phi_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.b_field_phi)
    }

    /// Get the number of b_field_phi elements
    pub fn b_field_phi_len(&self) -> usize {
        self.b_field_phi.len()
    }
}

impl EquilibriumGgd {
    /// Access b_field_z - use index for single element or range for slice view
    /// e.g. `.b_field_z(0)` returns `&GenericGridScalar`, `.b_field_z(0..2)` returns `GenericGridScalarSliceView`
    pub fn b_field_z<'a, I: GenericGridScalarIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.b_field_z)
    }

    /// Access b_field_z mutably - use index for single element or range for slice view
    /// e.g. `.b_field_z_mut(0)` returns `&mut GenericGridScalar`, `.b_field_z_mut(0..2)` returns `GenericGridScalarSliceViewMut`
    pub fn b_field_z_mut<'a, I: GenericGridScalarMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.b_field_z)
    }

    /// Get the number of b_field_z elements
    pub fn b_field_z_len(&self) -> usize {
        self.b_field_z.len()
    }
}

impl EquilibriumGgdArray {
    /// Access grid - use index for single element or range for slice view
    /// e.g. `.grid(0)` returns `&GenericGridDynamic`, `.grid(0..2)` returns `GenericGridDynamicSliceView`
    pub fn grid<'a, I: GenericGridDynamicIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grid)
    }

    /// Access grid mutably - use index for single element or range for slice view
    /// e.g. `.grid_mut(0)` returns `&mut GenericGridDynamic`, `.grid_mut(0..2)` returns `GenericGridDynamicSliceViewMut`
    pub fn grid_mut<'a, I: GenericGridDynamicMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grid)
    }

    /// Get the number of grid elements
    pub fn grid_len(&self) -> usize {
        self.grid.len()
    }
}

impl EquilibriumTimeSlice {
    /// Access profiles_2d - use index for single element or range for slice view
    /// e.g. `.profiles_2d(0)` returns `&EquilibriumProfiles2d`, `.profiles_2d(0..2)` returns `EquilibriumProfiles2dSliceView`
    pub fn profiles_2d<'a, I: EquilibriumProfiles2dIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.profiles_2d)
    }

    /// Access profiles_2d mutably - use index for single element or range for slice view
    /// e.g. `.profiles_2d_mut(0)` returns `&mut EquilibriumProfiles2d`, `.profiles_2d_mut(0..2)` returns `EquilibriumProfiles2dSliceViewMut`
    pub fn profiles_2d_mut<'a, I: EquilibriumProfiles2dMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.profiles_2d)
    }

    /// Get the number of profiles_2d elements
    pub fn profiles_2d_len(&self) -> usize {
        self.profiles_2d.len()
    }
}

impl EquilibriumTimeSlice {
    /// Access ggd - use index for single element or range for slice view
    /// e.g. `.ggd(0)` returns `&EquilibriumGgd`, `.ggd(0..2)` returns `EquilibriumGgdSliceView`
    pub fn ggd<'a, I: EquilibriumGgdIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.ggd)
    }

    /// Access ggd mutably - use index for single element or range for slice view
    /// e.g. `.ggd_mut(0)` returns `&mut EquilibriumGgd`, `.ggd_mut(0..2)` returns `EquilibriumGgdSliceViewMut`
    pub fn ggd_mut<'a, I: EquilibriumGgdMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.ggd)
    }

    /// Get the number of ggd elements
    pub fn ggd_len(&self) -> usize {
        self.ggd.len()
    }
}

impl GenericGridDynamic {
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

impl GenericGridDynamic {
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

impl Equilibrium {
    /// Access grids_ggd - use index for single element or range for slice view
    /// e.g. `.grids_ggd(0)` returns `&EquilibriumGgdArray`, `.grids_ggd(0..2)` returns `EquilibriumGgdArraySliceView`
    pub fn grids_ggd<'a, I: EquilibriumGgdArrayIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.grids_ggd)
    }

    /// Access grids_ggd mutably - use index for single element or range for slice view
    /// e.g. `.grids_ggd_mut(0)` returns `&mut EquilibriumGgdArray`, `.grids_ggd_mut(0..2)` returns `EquilibriumGgdArraySliceViewMut`
    pub fn grids_ggd_mut<'a, I: EquilibriumGgdArrayMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.grids_ggd)
    }

    /// Get the number of grids_ggd elements
    pub fn grids_ggd_len(&self) -> usize {
        self.grids_ggd.len()
    }
}

impl Equilibrium {
    /// Access time_slice - use index for single element or range for slice view
    /// e.g. `.time_slice(0)` returns `&EquilibriumTimeSlice`, `.time_slice(0..2)` returns `EquilibriumTimeSliceSliceView`
    pub fn time_slice<'a, I: EquilibriumTimeSliceIndex<'a>>(&'a self, index: I) -> I::Output {
        index.get(&self.time_slice)
    }

    /// Access time_slice mutably - use index for single element or range for slice view
    /// e.g. `.time_slice_mut(0)` returns `&mut EquilibriumTimeSlice`, `.time_slice_mut(0..2)` returns `EquilibriumTimeSliceSliceViewMut`
    pub fn time_slice_mut<'a, I: EquilibriumTimeSliceMutIndex<'a>>(&'a mut self, index: I) -> I::Output {
        index.get_mut(&mut self.time_slice)
    }

    /// Get the number of time_slice elements
    pub fn time_slice_len(&self) -> usize {
        self.time_slice.len()
    }
}
