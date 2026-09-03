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
}

pub struct EqulibriumGlobalQuantities {
    /// Major radius of the upper X-point
    /// Units: m
    pub xpt_upper_r: FLT_0D,
    /// Height of the upper X-point
    /// Units: m
    pub xpt_upper_z: FLT_0D,
    /// Major radius of the lower X-point
    /// Units: m
    pub xpt_lower_r: FLT_0D,
    /// Height of the lower X-point
    /// Units: m
    pub xpt_lower_z: FLT_0D,
}

pub struct EquilibriumTimeSlice {
    /// Fitted degrees of freedom of the p' source function
    pub p_prime_dof_values: FLT_1D,
    /// Fitted degrees of freedom of the FF' source function
    pub ff_prime_dof_values: FLT_1D,
    /// Fitted degrees of freedom of the passive structure currents
    /// Units: A
    pub passive_dof_values: FLT_1D,
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
}
