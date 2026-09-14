//! Grad-Shafranov solver writing into the IMAS `Equilibrium` IDS.
//!
//! `EquilibriumSolver` solves a single time-slice, mutating the `time_slice`
//! Equilibrium structure during iterations.

use super::Error;
use crate::plasma_geometry;
use crate::plasma_geometry::BoundaryContour;
use crate::plasma_geometry::MagneticAxis;
use crate::plasma_geometry::StationaryPoint;
use crate::plasma_geometry::bicubic_interpolator::BicubicInterpolator;
use crate::plasma_geometry::find_boundary;
use crate::plasma_geometry::find_magnetic_axis;
use crate::plasma_geometry::find_stationary_points_using_winding_number;
use crate::sensors::{SensorsDynamic, SensorsStatic};
use crate::source_functions::SourceFunctionTraits;
use crate::wall::{limiter_points, vacuum_vessel_outline};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{SolveLstsq, Svd as FaerSvd};
use faer::mat::MatRef;
use faer::{Accum, Par};
use imas_rs::EMPTY_INT;
use imas_rs::ids::wall::Wall as WallIds;
use imas_rs::{
    Code, EquilibriumContourTreeNode, EquilibriumGreens, EquilibriumGreensPfActive, EquilibriumGreensPfPassiveDof, EquilibriumProfiles2d,
    EquilibriumProfiles2dGrid, EquilibriumTimeSlice,
};
use ndarray::Axis;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, concatenate, s};
use ndarray_stats::QuantileExt;
use std::f64::consts::PI;
use std::sync::Arc;

const MU_0: f64 = physical_constants::VACUUM_MAG_PERMEABILITY;

/// Convergence status values, from the data dictionary's `equilibrium_convergence_status`
/// enumeration, stored in `equilibrium/time_slice/convergence/result`
const CONVERGENCE_STATUS_CONVERGED: i32 = 1;
const CONVERGENCE_STATUS_UNCONVERGED: i32 = 10;
const CONVERGENCE_STATUS_FATAL_ERROR: i32 = 20;

/// The radial position of every grid node, in the order the 2D fields are flattened.
///
/// The 2D fields have shape `[n_z, n_r]` and are flattened row-major, so for a rectangular
/// grid this is `dim1` tiled once per `dim2` entry.
///
/// It is built from the grid rather than by flattening `profiles_2d/r`, because the data
/// dictionary says of the position arrays that "in case of rectangular grids, the position
/// arrays should not be filled since they are redundant with grid/dim1 and dim2" - so this
/// keeps working when those are dropped.
fn flatten_grid_r(r: &Array1<f64>, n_z: usize) -> Array1<f64> {
    let n_r: usize = r.len();
    let mut flat_r: Array1<f64> = Array1::from_elem(n_r * n_z, f64::NAN);

    for i_z in 0..n_z {
        for i_r in 0..n_r {
            flat_r[i_z * n_r + i_r] = r[i_r];
        }
    }

    flat_r
}

#[cfg(test)]
mod tests {
    use super::flatten_grid_r;
    use ndarray::{Array1, Array2, ArrayView2, MeshIndex, meshgrid};

    /// `flatten_grid_r` rebuilds the flattened radial mesh from `grid/dim1` alone, so it has to
    /// agree with the mesh `Plasma::new` actually builds. If the meshgrid convention or the
    /// flattening order ever changes, this catches it rather than the solver quietly using the
    /// wrong radius in `j_2d * r` integrals.
    #[test]
    fn flattened_grid_r_matches_the_meshgrid() {
        let r: Array1<f64> = Array1::linspace(0.2, 1.0, 7);
        let z: Array1<f64> = Array1::linspace(-1.5, 1.5, 11);

        let (_mesh_z_view, mesh_r_view): (ArrayView2<f64>, ArrayView2<f64>) = meshgrid((&z, &r), MeshIndex::IJ);
        let mesh_r: Array2<f64> = mesh_r_view.to_owned();
        let expected: Array1<f64> = mesh_r.flatten().to_owned();

        let flat_r: Array1<f64> = flatten_grid_r(&r, z.len());

        assert_eq!(flat_r.len(), r.len() * z.len());
        assert_eq!(flat_r, expected);
    }
}

/// Scratch buffers for `calculate_psi_and_derivatives`.
///
/// This type holds no state. It exists only so that the two buffers are allocated **once per
/// time-slice** rather than on every Picard iteration: they are `(n_z, n_z * n_r)`, which is 8.3
/// million elements - 66 MB each - at [n_r, n_z] = [81, 321], so allocating them per iteration
/// costs an `mmap`/`munmap` pair and ~16,000 page faults each time round the loop.
///
/// `calculate_psi_and_derivatives` overwrites every element of both before it reads any of them,
/// so whatever they hold on entry is discarded, and nothing is carried from one iteration to the
/// next. Deleting this type and allocating inside the function would be correct, only slower.
///
/// The caller fills them with `NaN` rather than zeros, so that an element which the **first**
/// iteration fails to write propagates a visible `NaN` into `psi` rather than silently
/// contributing nothing. On later iterations they hold the previous iteration's values, so this
/// catches a gap in the fill loop only the first time round.
///
/// The read-only counterpart is [`PsiAndDerivativesGreens`]: both are built by the caller and
/// handed in, for the same reason.
pub struct PsiAndDerivativesTemporaryStorage {
    /// Current sources gathered for the even kernels; shape = (n_z, n_z * n_r)
    w_even: Array2<f64>,
    /// Current sources gathered for the odd kernels; shape = (n_z, n_z * n_r)
    w_odd: Array2<f64>,
}

/// Greens tables reorganised for `calculate_psi_and_derivatives`.
///
/// Precomputed **once per time-slice** (the tables do not change between Picard iterations),
/// while `calculate_psi_and_derivatives` is called **every** iteration.
///
/// The expensive part of `calculate_psi_and_derivatives` is the plasma grid-to-grid convolution:
/// ```text
///     field[(i_z, i_r)] = sum_{i_cur_z, i_cur_r} g[(|i_z - i_cur_z|, i_r, i_cur_r)] * j_2d[(i_cur_z, i_cur_r)] * d_area
/// ```
/// Because the Greens table only depends on the **vertical offset** `i_offset_z = |i_z - i_cur_z|`,
/// the convolution can be reorganised into a single matrix multiplication:
/// ```text
///     field = w @ g_plasma_by_offset
/// ```
/// where row `i_z` of `w` gathers, for each `(i_offset_z, i_cur_r)`, the (at most two) current
/// sources which see grid point `i_z` at that offset: `j_2d[(i_z - i_offset_z, i_cur_r)]` and
/// `j_2d[(i_z + i_offset_z, i_cur_r)]`. Kernels which are **even** in `z - z_current_source`
/// (`psi`, `d_psi_d_r`, `d2_psi_d_r2`, `d2_psi_d_z2`, `d3_psi_d_r_d_z2`) take the sum of the two
/// sources; kernels which are **odd** (`d_psi_d_z`, `d2_psi_d_r_d_z`, `d3_psi_d_r2_d_z`,
/// `d3_psi_d_z3`) take the difference. `w` is filled fresh each iteration (it depends on `j_2d`)
/// into the buffers of [`PsiAndDerivativesTemporaryStorage`]; the GEMM is done with `faer`.
///
/// The even kernels and the odd kernels are each concatenated column-wise, so the plasma
/// contribution to all nine fields costs exactly two GEMMs.
pub struct PsiAndDerivativesGreens {
    /// Plasma grid-to-grid, even kernels, concatenated column-wise in the order
    /// [`psi`, `d_psi_d_r`, `d2_psi_d_r2`, `d2_psi_d_z2`, `d3_psi_d_r_d_z2`];
    /// rows = (i_offset_z * n_r + i_cur_r); shape = (n_z * n_r, 5 * n_r)
    g_even_plasma_by_offset: Array2<f64>,
    /// Plasma grid-to-grid, odd kernels, concatenated column-wise in the order
    /// [`d_psi_d_z`, `d2_psi_d_r_d_z`, `d3_psi_d_r2_d_z`, `d3_psi_d_z3`];
    /// rows = (i_offset_z * n_r + i_cur_r); shape = (n_z * n_r, 4 * n_r)
    g_odd_plasma_by_offset: Array2<f64>,
    /// PF coils; each shape = (n_z * n_r, n_pf)
    g_d_psi_d_r_coils_matrix: Array2<f64>,
    g_d_psi_d_z_coils_matrix: Array2<f64>,
    g_d2_psi_d_r2_coils_matrix: Array2<f64>,
    g_d2_psi_d_r_d_z_coils_matrix: Array2<f64>,
    g_d2_psi_d_z2_coils_matrix: Array2<f64>,
    g_d3_psi_d_r2_d_z_coils_matrix: Array2<f64>,
    g_d3_psi_d_r_d_z2_coils_matrix: Array2<f64>,
    g_d3_psi_d_z3_coils_matrix: Array2<f64>,
    /// Passives; each shape = (n_z * n_r, n_passive_dof)
    g_psi_passives_matrix: Array2<f64>,
    g_d_psi_d_r_passives_matrix: Array2<f64>,
    g_d_psi_d_z_passives_matrix: Array2<f64>,
    g_d2_psi_d_r2_passives_matrix: Array2<f64>,
    g_d2_psi_d_r_d_z_passives_matrix: Array2<f64>,
    g_d2_psi_d_z2_passives_matrix: Array2<f64>,
    g_d3_psi_d_r2_d_z_passives_matrix: Array2<f64>,
    g_d3_psi_d_r_d_z2_passives_matrix: Array2<f64>,
    g_d3_psi_d_z3_passives_matrix: Array2<f64>,
}

impl PsiAndDerivativesGreens {
    /// Reorganise the Greens tables held on the equilibrium IDS into the matrix shapes the
    /// per-iteration GEMMs want.
    ///
    /// `n_r` and `n_z` are recovered from the grid-to-grid table's own shape rather than read from
    /// anywhere else, so this cannot be handed a table which disagrees with the grid it was
    /// calculated on.
    pub fn new(greens: &EquilibriumGreens) -> Self {
        let grid_grid_psi: &Array2<f64> = &greens.grid_grid.psi;
        let (n_offset_z_times_n_r, n_r): (usize, usize) = grid_grid_psi.dim();
        if n_r == 0 || n_offset_z_times_n_r % n_r != 0 {
            panic!("PsiAndDerivativesGreens: `greens/grid_grid/psi` has shape ({n_offset_z_times_n_r}, {n_r}), which is not (n_z * n_r, n_r)");
        }
        let n_z: usize = n_offset_z_times_n_r / n_r;

        // Plasma grid-to-grid tables; stored shape = (n_z * n_r, n_r), which unflattens to
        // (i_offset_z, i_r, i_cur_r). Permute to (i_offset_z, i_cur_r, i_r) and re-flatten so
        // that rows = (i_offset_z, i_cur_r) match the columns of `w`, and columns = i_r
        let permute_to_by_offset = |g_flat: &Array2<f64>| -> Array2<f64> {
            let g_3d: Array3<f64> = g_flat.to_shape((n_z, n_r, n_r)).unwrap().to_owned();
            let g_3d_permuted: Array3<f64> = g_3d.permuted_axes([0, 2, 1]);
            let g_by_offset: Array2<f64> = g_3d_permuted.as_standard_layout().to_shape((n_z * n_r, n_r)).unwrap().to_owned();
            g_by_offset
        };

        let grid_grid = |table: &Array2<f64>, key: &str| -> Array2<f64> {
            if table.is_empty() {
                panic!("PsiAndDerivativesGreens: `greens/grid_grid/{key}` unset");
            }
            permute_to_by_offset(table)
        };
        let g_psi_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.psi, "psi");
        let g_d_psi_d_r_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d_psi_d_r, "d_psi_d_r");
        let g_d_psi_d_z_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d_psi_d_z, "d_psi_d_z");
        let g_d2_psi_d_r2_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d2_psi_d_r2, "d2_psi_d_r2");
        let g_d2_psi_d_r_d_z_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d2_psi_d_r_d_z, "d2_psi_d_r_d_z");
        let g_d2_psi_d_z2_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d2_psi_d_z2, "d2_psi_d_z2");
        let g_d3_psi_d_r2_d_z_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d3_psi_d_r2_d_z, "d3_psi_d_r2_d_z");
        let g_d3_psi_d_r_d_z2_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d3_psi_d_r_d_z2, "d3_psi_d_r_d_z2");
        let g_d3_psi_d_z3_plasma_by_offset: Array2<f64> = grid_grid(&greens.grid_grid.d3_psi_d_z3, "d3_psi_d_z3");

        // Concatenate per parity, so each parity is a single GEMM
        // (`as_standard_layout` because `concatenate` does not guarantee a C-contiguous result)
        let g_even_plasma_by_offset: Array2<f64> = concatenate(
            Axis(1),
            &[
                g_psi_plasma_by_offset.view(),
                g_d_psi_d_r_plasma_by_offset.view(),
                g_d2_psi_d_r2_plasma_by_offset.view(),
                g_d2_psi_d_z2_plasma_by_offset.view(),
                g_d3_psi_d_r_d_z2_plasma_by_offset.view(),
            ],
        )
        .unwrap()
        .as_standard_layout()
        .to_owned();
        let g_odd_plasma_by_offset: Array2<f64> = concatenate(
            Axis(1),
            &[
                g_d_psi_d_z_plasma_by_offset.view(),
                g_d2_psi_d_r_d_z_plasma_by_offset.view(),
                g_d3_psi_d_r2_d_z_plasma_by_offset.view(),
                g_d3_psi_d_z3_plasma_by_offset.view(),
            ],
        )
        .unwrap()
        .as_standard_layout()
        .to_owned();

        // PF coils: each table is (n_z, n_r), gathered into one column per coil. The coils keep the
        // order they sit in on the IDS, which is the order the measured currents are in
        let n_pf: usize = greens.pf_active.len();
        let coils_matrix = |select: fn(&EquilibriumGreensPfActive) -> &Array2<f64>, key: &str| -> Array2<f64> {
            let mut g_coils: Array2<f64> = Array2::from_elem((n_z * n_r, n_pf), f64::NAN);
            for i_pf in 0..n_pf {
                let coil: &EquilibriumGreensPfActive = &greens.pf_active[i_pf];
                let table: &Array2<f64> = select(coil);
                if table.is_empty() {
                    panic!("PsiAndDerivativesGreens: `greens/pf_active({i_pf})/{key}` unset");
                }
                let table_flat: Array1<f64> = table
                    .to_shape(n_z * n_r)
                    .unwrap_or_else(|_| panic!("PsiAndDerivativesGreens: `greens/pf_active({i_pf})/{key}` is not (n_z, n_r)"))
                    .to_owned();
                g_coils.slice_mut(s![.., i_pf]).assign(&table_flat);
            }
            g_coils
        };
        let g_d_psi_d_r_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d_psi_d_r, "d_psi_d_r");
        let g_d_psi_d_z_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d_psi_d_z, "d_psi_d_z");
        let g_d2_psi_d_r2_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d2_psi_d_r2, "d2_psi_d_r2");
        let g_d2_psi_d_r_d_z_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d2_psi_d_r_d_z, "d2_psi_d_r_d_z");
        let g_d2_psi_d_z2_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d2_psi_d_z2, "d2_psi_d_z2");
        let g_d3_psi_d_r2_d_z_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d3_psi_d_r2_d_z, "d3_psi_d_r2_d_z");
        let g_d3_psi_d_r_d_z2_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d3_psi_d_r_d_z2, "d3_psi_d_r_d_z2");
        let g_d3_psi_d_z3_coils_matrix: Array2<f64> = coils_matrix(|coil| &coil.d3_psi_d_z3, "d3_psi_d_z3");

        // Passives: each table is already (n_z * n_r), gathered into one column per degree of
        // freedom. The degrees of freedom are laid out conductor by conductor, in IDS order, which
        // is the order `passive_dof_values` and the regularisation matrices are in
        let n_passives: usize = greens.pf_passive.len();
        let mut n_passive_dof: usize = 0;
        for i_passive in 0..n_passives {
            n_passive_dof += greens.pf_passive[i_passive].dof.len();
        }
        let passives_matrix = |select: fn(&EquilibriumGreensPfPassiveDof) -> &Array1<f64>, key: &str| -> Array2<f64> {
            let mut g_passives: Array2<f64> = Array2::from_elem((n_z * n_r, n_passive_dof), f64::NAN);
            let mut i_dof_total: usize = 0;
            for i_passive in 0..n_passives {
                let n_dof_this_passive: usize = greens.pf_passive[i_passive].dof.len();
                for i_dof in 0..n_dof_this_passive {
                    let dof: &EquilibriumGreensPfPassiveDof = &greens.pf_passive[i_passive].dof[i_dof];
                    let table: &Array1<f64> = select(dof);
                    if table.is_empty() {
                        panic!("PsiAndDerivativesGreens: `greens/pf_passive({i_passive})/dof({i_dof})/{key}` unset");
                    }
                    g_passives.slice_mut(s![.., i_dof_total]).assign(table);
                    i_dof_total += 1;
                }
            }
            g_passives
        };
        let g_psi_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.psi, "psi");
        let g_d_psi_d_r_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d_psi_d_r, "d_psi_d_r");
        let g_d_psi_d_z_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d_psi_d_z, "d_psi_d_z");
        let g_d2_psi_d_r2_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d2_psi_d_r2, "d2_psi_d_r2");
        let g_d2_psi_d_r_d_z_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d2_psi_d_r_d_z, "d2_psi_d_r_d_z");
        let g_d2_psi_d_z2_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d2_psi_d_z2, "d2_psi_d_z2");
        let g_d3_psi_d_r2_d_z_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d3_psi_d_r2_d_z, "d3_psi_d_r2_d_z");
        let g_d3_psi_d_r_d_z2_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d3_psi_d_r_d_z2, "d3_psi_d_r_d_z2");
        let g_d3_psi_d_z3_passives_matrix: Array2<f64> = passives_matrix(|dof| &dof.d3_psi_d_z3, "d3_psi_d_z3");

        Self {
            g_even_plasma_by_offset,
            g_odd_plasma_by_offset,
            g_d_psi_d_r_coils_matrix,
            g_d_psi_d_z_coils_matrix,
            g_d2_psi_d_r2_coils_matrix,
            g_d2_psi_d_r_d_z_coils_matrix,
            g_d2_psi_d_z2_coils_matrix,
            g_d3_psi_d_r2_d_z_coils_matrix,
            g_d3_psi_d_r_d_z2_coils_matrix,
            g_d3_psi_d_z3_coils_matrix,
            g_psi_passives_matrix,
            g_d_psi_d_r_passives_matrix,
            g_d_psi_d_z_passives_matrix,
            g_d2_psi_d_r2_passives_matrix,
            g_d2_psi_d_r_d_z_passives_matrix,
            g_d2_psi_d_z2_passives_matrix,
            g_d3_psi_d_r2_d_z_passives_matrix,
            g_d3_psi_d_r_d_z2_passives_matrix,
            g_d3_psi_d_z3_passives_matrix,
        }
    }
}

/// Grad-Shafranov solution, at single time-slice
pub struct EquilibriumSolver<'a> {
    // Object inputs
    /// The IDS time-slice being solved. Results live here as they are calculated, rather than
    /// being accumulated on this struct and copied over at the end
    time_slice: &'a mut EquilibriumTimeSlice,
    /// Description of the code, including the settings it was run with. These are the same for
    /// every time-slice, so they sit on the IDS itself (`equilibrium.code`) rather than inside
    /// `time_slice`
    equilibrium_code: &'a Code,
    /// Greens tables, shared by every time-slice because they are geometry only
    greens_tables: &'a EquilibriumGreens,
    /// The same Greens tables, reorganised into the shapes `calculate_psi_and_derivatives` wants.
    /// Built once by the caller and shared, because the reorganisation depends only on the
    /// geometry - see `PsiAndDerivativesGreens`
    psi_and_derivatives_greens: &'a PsiAndDerivativesGreens,
    /// The initial current-density guess, or the reason it could not be built. It is the same for
    /// every time-slice, so the caller builds it once; the `Result` is carried in so that a bad
    /// initial guess still fails every slice in the way it did when each built its own
    initial_j_2d: &'a Result<Array2<f64>, String>,
    /// `vacuum_toroidal_field/r0` and `vacuum_toroidal_field/b0` at this time-slice. `b0` is
    /// indexed by time on the IDS, so the caller hands in this slice's value rather than the array
    vacuum_toroidal_field_r0: f64,
    vacuum_toroidal_field_b0: f64,
    /// The machine's wall. The limiter points and the vacuum vessel contour are read from
    /// `wall/description_2d(0)/limiter`; see `crate::wall::Wall` for the unit ordering
    wall: &'a WallIds,
    coils_dynamic: &'a SensorsDynamic,
    bp_probes_static: &'a SensorsStatic,
    bp_probes_dynamic: &'a SensorsDynamic,
    flux_loops_static: &'a SensorsStatic,
    flux_loops_dynamic: &'a SensorsDynamic,
    dialoop_static: &'a SensorsStatic,
    dialoop_dynamic: &'a SensorsDynamic,
    rogowski_coils_static: &'a SensorsStatic,
    rogowski_coils_dynamic: &'a SensorsDynamic,
    isoflux_static: &'a SensorsStatic,
    isoflux_dynamic: &'a SensorsDynamic,
    isoflux_boundary_static: &'a SensorsStatic,
    isoflux_boundary_dynamic: &'a SensorsDynamic,
    pressure_sensors_static: &'a SensorsStatic,
    pressure_sensors_dynamic: &'a SensorsDynamic,
    magnetic_axis_static: &'a SensorsStatic,
    magnetic_axis_dynamic: &'a SensorsDynamic,
    // Results
    pub passive_dof_values: Array1<f64>,
    pub p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync>,
    passive_regularisations: Array2<f64>,
    passive_regularisations_weight: Array1<f64>,
}

impl<'a> EquilibriumSolver<'a> {
    pub fn new(
        time_slice: &'a mut EquilibriumTimeSlice,
        equilibrium_code: &'a Code,
        greens_tables: &'a EquilibriumGreens,
        wall: &'a WallIds,
        vacuum_toroidal_field_r0: f64,
        vacuum_toroidal_field_b0: f64,
        inputs: &GradShafranovInputs<'a>,
    ) -> Self {
        // The solver writes `psi` straight into the IDS, so `profiles_2d` has to exist before the
        // first iteration. GSFit solves on a single rectangular (R, Z) grid, so there is exactly
        // one entry in this array of structures. It is only allocated when absent:
        // `Plasma::initialise_equilibrium_ids` has normally already put the grid there, and
        // replacing the entry would discard it
        if time_slice.profiles_2d.is_empty() {
            time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];
        }

        EquilibriumSolver {
            // Object inputs
            time_slice,
            equilibrium_code,
            greens_tables,
            psi_and_derivatives_greens: inputs.psi_and_derivatives_greens,
            initial_j_2d: inputs.initial_j_2d,
            vacuum_toroidal_field_r0,
            vacuum_toroidal_field_b0,
            wall,
            coils_dynamic: inputs.coils_dynamic,
            bp_probes_static: inputs.bp_probes_static,
            bp_probes_dynamic: inputs.bp_probes_dynamic,
            flux_loops_static: inputs.flux_loops_static,
            flux_loops_dynamic: inputs.flux_loops_dynamic,
            dialoop_static: inputs.dialoop_static,
            dialoop_dynamic: inputs.dialoop_dynamic,
            rogowski_coils_static: inputs.rogowski_coils_static,
            rogowski_coils_dynamic: inputs.rogowski_coils_dynamic,
            isoflux_static: inputs.isoflux_static,
            isoflux_dynamic: inputs.isoflux_dynamic,
            isoflux_boundary_static: inputs.isoflux_boundary_static,
            isoflux_boundary_dynamic: inputs.isoflux_boundary_dynamic,
            pressure_sensors_static: inputs.pressure_sensors_static,
            pressure_sensors_dynamic: inputs.pressure_sensors_dynamic,
            magnetic_axis_static: inputs.magnetic_axis_static,
            magnetic_axis_dynamic: inputs.magnetic_axis_dynamic,
            // Results
            passive_dof_values: Array1::zeros(0),
            p_prime_source_function: inputs.p_prime_source_function.clone(),
            ff_prime_source_function: inputs.ff_prime_source_function.clone(),
            passive_regularisations: inputs.passive_regularisations.to_owned(),
            passive_regularisations_weight: inputs.passive_regularisations_weight.to_owned(),
        }
    }

    /// If the solver fails to converge, this function will set the solution to NAN values (but with the correct shape).
    fn set_to_failed_time_slice(&mut self, error: Error) {
        // Classified against the data dictionary's convergence status enumeration. Listed
        // variant by variant rather than with a catch-all, so that a new `Error` fails to
        // compile until it has been classified
        let (result_name, result_index): (&str, i32) = match &error {
            Error::MaxIterReached => ("unconverged", CONVERGENCE_STATUS_UNCONVERGED),
            Error::InvalidInitialCurrent(_) | Error::NoBoundaryFound { .. } | Error::NoMagneticAxisFound | Error::NoStationaryPointsFound => {
                ("fatal_error", CONVERGENCE_STATUS_FATAL_ERROR)
            }
        };
        println!("{:?}", error);
        self.time_slice.convergence.result.name = result_name.to_string();
        self.time_slice.convergence.result.index = result_index;
        self.time_slice.convergence.result.description = format!("{:?}", error);

        self.time_slice.convergence.grad_shafranov_deviation_value = f64::NAN;
        self.time_slice.source_functions.ff_prime.coefficients *= f64::NAN;
        self.time_slice.source_functions.p_prime.coefficients *= f64::NAN;
        self.passive_dof_values *= f64::NAN;
        self.time_slice.profiles_2d[0].psi *= f64::NAN;
        self.time_slice.profiles_2d[0].d_psi_d_r *= f64::NAN;
        self.time_slice.profiles_2d[0].d_psi_d_z *= f64::NAN;
        self.time_slice.profiles_2d[0].d2_psi_d_r2 *= f64::NAN;
        self.time_slice.profiles_2d[0].d2_psi_d_r_d_z *= f64::NAN;
        self.time_slice.profiles_2d[0].d2_psi_d_z2 *= f64::NAN;
        self.time_slice.profiles_2d[0].psi_norm *= f64::NAN;
        self.time_slice.profiles_2d[0].j_phi *= f64::NAN;
        self.time_slice.profiles_2d[0].mask *= f64::NAN;
        self.time_slice.profiles_2d[0].psi_coils *= f64::NAN;
        self.time_slice.boundary.psi = f64::NAN;
        self.time_slice.global_quantities.psi_magnetic_axis = f64::NAN;
        self.time_slice.global_quantities.ip = f64::NAN;
        self.time_slice.boundary.bounding.r = f64::NAN;
        self.time_slice.boundary.bounding.z = f64::NAN;
        self.time_slice.convergence.delta_z = f64::NAN;
        self.time_slice.convergence.iterations_n = EMPTY_INT;
        self.time_slice.global_quantities.magnetic_axis.r = f64::NAN;
        self.time_slice.global_quantities.magnetic_axis.z = f64::NAN;
        // A boundary which does not exist is neither limited nor diverted, and IMAS spells that out
        // with its reserved value, the integer counterpart of the `NaN` every float above is set to
        self.time_slice.boundary.r#type = EMPTY_INT;
    }

    /// Solve the inverse Grad-Shafranov problem
    pub fn solve(&mut self) {
        let p_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.p_prime_source_function.clone();
        let ff_prime_source_function: Arc<dyn SourceFunctionTraits + Send + Sync> = self.ff_prime_source_function.clone();

        // Unpack objects
        let coils_dynamic: &SensorsDynamic = self.coils_dynamic;

        // Get sensors
        let bp_probes_static: &SensorsStatic = self.bp_probes_static;
        let bp_probes_dynamic: &SensorsDynamic = self.bp_probes_dynamic;
        let flux_loops_static: &SensorsStatic = self.flux_loops_static;
        let flux_loops_dynamic: &SensorsDynamic = self.flux_loops_dynamic;
        let dialoop_static: &SensorsStatic = self.dialoop_static;
        let dialoop_dynamic: &SensorsDynamic = self.dialoop_dynamic;
        let rogowski_coils_static: &SensorsStatic = self.rogowski_coils_static;
        let rogowski_coils_dynamic: &SensorsDynamic = self.rogowski_coils_dynamic;
        let isoflux_static: &SensorsStatic = self.isoflux_static;
        let isoflux_dynamic: &SensorsDynamic = self.isoflux_dynamic;
        let isoflux_boundary_static: &SensorsStatic = self.isoflux_boundary_static;
        let pressure_sensors_static: &SensorsStatic = self.pressure_sensors_static;
        let pressure_sensors_dynamic: &SensorsDynamic = self.pressure_sensors_dynamic;
        let isoflux_boundary_dynamic: &SensorsDynamic = self.isoflux_boundary_dynamic;
        let magnetic_axis_static: &SensorsStatic = self.magnetic_axis_static;
        let magnetic_axis_dynamic: &SensorsDynamic = self.magnetic_axis_dynamic;

        // Plasma grid. The grid never changes while the solver runs, so it is read out of the IDS
        // rather than copied out of it; this borrow ends here, and `r` and `z` are taken again
        // inside the iteration loop, off the same `&mut profiles_2d[0]` as `psi` and its derivatives
        let grid: &EquilibriumProfiles2dGrid = &self.time_slice.profiles_2d[0].grid;
        let d_area: f64 = grid.d_area;
        let flat_r: Array1<f64> = flatten_grid_r(&grid.dim1, grid.dim2.len());
        // Limiter, from the `wall` IDS. `limit_pts` gathers every limiter unit, `vessel` is
        // `unit(0)` alone
        let (limit_pts_r, limit_pts_z): (Array1<f64>, Array1<f64>) = limiter_points(self.wall).unwrap();
        let (vessel_r, vessel_z): (Array1<f64>, Array1<f64>) = vacuum_vessel_outline(self.wall).unwrap();

        // Degrees of freedom
        let passives_shape: &[usize] = bp_probes_static.greens_with_passives.shape();
        let n_passive_dof: usize = passives_shape[0];
        let n_p_prime_dof: usize = p_prime_source_function.source_function_n_dof();
        let n_ff_prime_dof: usize = ff_prime_source_function.source_function_n_dof();
        // Solver settings, supplied through `equilibrium.code`
        let n_iter_max: usize = self.equilibrium_code.numerics.iterations.n_max as usize;
        let n_iter_min: usize = self.equilibrium_code.numerics.iterations.n_min as usize;
        let n_iter_no_vertical_feedback: usize = self.equilibrium_code.numerics.iterations.n_no_vertical_feedback as usize;
        let gs_error_tolerance: f64 = self.equilibrium_code.numerics.grad_shafranov_deviation_tolerance;

        // Constraints
        let n_bp: usize = bp_probes_dynamic.measured.len();
        let n_fl: usize = flux_loops_dynamic.measured.len();
        let n_dialoop: usize = dialoop_dynamic.measured.len();
        let n_rog: usize = rogowski_coils_dynamic.measured.len();
        let n_isoflux: usize = isoflux_dynamic.measured.len();
        let n_isoflux_boundary: usize = isoflux_boundary_dynamic.measured.len();
        let n_pressure_sensors: usize = pressure_sensors_dynamic.measured.len();
        let n_magnetic_axis_constraints: usize = magnetic_axis_dynamic.measured.len();
        let n_p_prime_regularisation: usize = p_prime_source_function.source_function_regularisation().shape()[0];
        let n_ff_prime_regularisation: usize = ff_prime_source_function.source_function_regularisation().shape()[0];
        let n_passive_regularisation: usize = self.passive_regularisations.shape()[0];
        let n_delta_z_regularisation: usize = 0; // initially set to 0 because we don't have previous iteration
        let n_constraints: usize = n_bp
            + n_fl
            + n_dialoop
            + n_rog
            + n_isoflux
            + n_isoflux_boundary
            + n_pressure_sensors
            + n_magnetic_axis_constraints
            + n_p_prime_regularisation
            + n_ff_prime_regularisation
            + n_passive_regularisation
            + n_delta_z_regularisation;

        // Magnetic sensor's Greens tables
        let greens_bp_probes_grid: &Array2<f64> = &bp_probes_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_bp_probes_dz: &Array2<f64> = &bp_probes_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_bp_probes_pf: &Array2<f64> = &bp_probes_static.greens_with_pf; // shape = [n_pf, n_sensors]
        let greens_bp_probes_passives: &Array2<f64> = &bp_probes_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        let greens_flux_loops_grid: &Array2<f64> = &flux_loops_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_flux_loops_dz: &Array2<f64> = &flux_loops_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_flux_loops_pf: &Array2<f64> = &flux_loops_static.greens_with_pf; // shape = [n_pf, n_sensors]
        let greens_flux_loops_passives: &Array2<f64> = &flux_loops_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        let greens_rogowski_coils_grid: &Array2<f64> = &rogowski_coils_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_rogowski_coils_dz: &Array2<f64> = &rogowski_coils_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_rogowski_coils_pf: &Array2<f64> = &rogowski_coils_static.greens_with_pf; // shape = [n_z*n_r, n_sensors]
        let greens_rogowski_coils_passives: &Array2<f64> = &rogowski_coils_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_grid: &Array2<f64> = &isoflux_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_dz: &Array2<f64> = &isoflux_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_pf: &Array2<f64> = &isoflux_static.greens_with_pf; // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_passives: &Array2<f64> = &isoflux_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        let greens_isoflux_boundary_grid: &Array2<f64> = &isoflux_boundary_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_isoflux_boundary_dz: &Array2<f64> = &isoflux_boundary_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_boundary_pf: &Array2<f64> = &isoflux_boundary_static.greens_with_pf; // shape = [n_z*n_r, n_sensors]
        let greens_isoflux_boundary_passives: &Array2<f64> = &isoflux_boundary_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        let greens_magnetic_axis_grid: &Array2<f64> = &magnetic_axis_static.greens_with_grid; // shape = [n_z*n_r, n_sensors]
        let greens_d_magnetic_axis_dz: &Array2<f64> = &magnetic_axis_static.greens_d_sensor_dz; // shape = [n_z*n_r, n_sensors]
        let greens_magnetic_axis_pf: &Array2<f64> = &magnetic_axis_static.greens_with_pf; // shape = [n_z*n_r, n_sensors]
        let greens_magnetic_axis_passives: &Array2<f64> = &magnetic_axis_static.greens_with_passives; // shape = [n_passive_dof, n_sensors]

        // pf_coil_currents
        let pf_coil_currents: &Array1<f64> = &coils_dynamic.measured;

        // TODO: IDEA- change the normalisation so that it does represent current. But this won't work for the IVC eigenvalues
        self.passive_dof_values = Array1::zeros(n_passive_dof);

        // Initialise the plasma.
        // Note: the initial seed is the same for all time-slices
        if let Err(reason) = self.initialise_plasma_with_quadratic_current_density() {
            self.set_to_failed_time_slice(Error::InvalidInitialCurrent(reason));
            return;
        }

        // Some variables we want to track between iterations
        let mut dof_values_previous: Array1<f64> = Array1::zeros(n_p_prime_dof + n_ff_prime_dof + n_passive_dof + 1);
        let mut psi_a_previous: f64 = 0.0; // needed to calculate gs-error

        // The reorganised Greens tables for `calculate_psi_and_derivatives`. They depend only on
        // the geometry, so they are built once by the caller and shared by every time-slice
        let psi_and_derivatives_greens: &PsiAndDerivativesGreens = self.psi_and_derivatives_greens;

        // Scratch space for `calculate_psi_and_derivatives`. Allocated out here purely so that it
        // is not reallocated on every iteration - it holds no state; see the type. Filled with
        // `NaN` rather than zeros so that an element the first iteration fails to write shows up as
        // `NaN` in `psi` instead of silently contributing nothing
        let n_r: usize = self.equilibrium_code.grid.n_r as usize;
        let n_z: usize = self.equilibrium_code.grid.n_z as usize;
        let mut temporary_storage: PsiAndDerivativesTemporaryStorage = PsiAndDerivativesTemporaryStorage {
            w_even: Array2::from_elem((n_z, n_z * n_r), f64::NAN),
            w_odd: Array2::from_elem((n_z, n_z * n_r), f64::NAN),
        };

        // Iteration loop
        'iteration_loop: for i_iter in 0..n_iter_max {
            // println!("");
            // println!("Iteration {i_iter}");
            // Updates `psi` and all of its derivatives (including the `delta_z` vertical stability correction);
            // timing: 350ms, with [n_r, n_z]=[81, 321]
            self.calculate_psi_and_derivatives(psi_and_derivatives_greens, &mut temporary_storage);

            // Construct pointers to the grid, to psi and to psi's derivatives, for convenience.
            // These borrow out of the IDS rather than copying out of it, so nothing is cloned each
            // iteration. The names match the parameters of the functions they are passed into.
            //
            // The single `&mut` to `profiles_2d[0]` matters: borrowing each field off it lets the
            // solver hold `psi` while it writes `mask` and `psi_norm` further down, because those
            // are disjoint fields of one struct. Writing `self.time_slice.profiles_2d[0]` out in
            // full at each site would not compile - the compiler cannot tell two index expressions
            // refer to the same element, so it treats the borrows as overlapping.
            let profiles_2d: &mut EquilibriumProfiles2d = &mut self.time_slice.profiles_2d[0];
            let psi_2d: &Array2<f64> = &profiles_2d.psi;
            let d_psi_d_r_2d: &Array2<f64> = &profiles_2d.d_psi_d_r;
            let d_psi_d_z_2d: &Array2<f64> = &profiles_2d.d_psi_d_z;
            let d2_psi_d_r2_2d: &Array2<f64> = &profiles_2d.d2_psi_d_r2;
            let d2_psi_d_r_d_z_2d: &Array2<f64> = &profiles_2d.d2_psi_d_r_d_z;
            let d2_psi_d_z2_2d: &Array2<f64> = &profiles_2d.d2_psi_d_z2;
            let r: &Array1<f64> = &profiles_2d.grid.dim1;
            let z: &Array1<f64> = &profiles_2d.grid.dim2;
            // `j_phi` is the previous iteration's current density: `calculate_psi_and_derivatives`
            // above reads it but never writes it
            let j_2d: &Array2<f64> = &profiles_2d.j_phi;

            // Grid spacing
            let d_r: f64 = r[1] - r[0];
            let d_z: f64 = z[1] - z[0];

            // Find stationary points in `psi` (magnetic axis and x-points)
            let stationary_points: Vec<StationaryPoint> = find_stationary_points_using_winding_number(
                r.view(),
                z.view(),
                psi_2d.view(),
                d_psi_d_r_2d.view(),
                d_psi_d_z_2d.view(),
                d2_psi_d_r2_2d.view(),
                d2_psi_d_r_d_z_2d.view(),
                d2_psi_d_z2_2d.view(),
            );
            // At a minimum we should have found the magnetic axis
            if stationary_points.is_empty() {
                // Set time-slice to failed

                // Store error state
                self.set_to_failed_time_slice(Error::NoStationaryPointsFound);

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }

            // Store the stationary points in the IDS
            self.time_slice.contour_tree.node = Self::contour_tree_nodes(&stationary_points);

            // Find the magnetic axis (o-point).
            // The search starts from the magnetic axis found on the previous iteration, which is
            // still what the IDS holds at this point; it is overwritten a few lines below.
            let mag_r_previous: f64 = self.time_slice.global_quantities.magnetic_axis.r;
            let mag_z_previous: f64 = self.time_slice.global_quantities.magnetic_axis.z;
            let magnetic_axis_or_error: Result<MagneticAxis, String> =
                find_magnetic_axis(&stationary_points, mag_r_previous, mag_z_previous, &vessel_r, &vessel_z);
            // Test if we have found the magnetic axis
            if magnetic_axis_or_error.is_err() {
                // Set time-slice to failed

                // Store error state
                self.set_to_failed_time_slice(Error::NoMagneticAxisFound);

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }
            // Unwrap and get results out of `magnetic_axis_or_error`
            let magnetic_axis: MagneticAxis = magnetic_axis_or_error.unwrap();
            let mag_r: f64 = magnetic_axis.r;
            let mag_z: f64 = magnetic_axis.z;
            let psi_a: f64 = magnetic_axis.psi;
            self.time_slice.global_quantities.magnetic_axis.r = mag_r;
            self.time_slice.global_quantities.magnetic_axis.z = mag_z;
            self.time_slice.global_quantities.psi_magnetic_axis = psi_a;

            // Find boundary
            let plasma_boundary_or_error: Result<BoundaryContour, plasma_geometry::Error> = find_boundary(
                r,
                z,
                psi_2d,
                d_psi_d_r_2d,
                d_psi_d_z_2d,
                d2_psi_d_r_d_z_2d,
                &stationary_points,
                &limit_pts_r,
                &limit_pts_z,
                &vessel_r,
                &vessel_z,
                mag_r,
                mag_z,
            );
            // Test if we have found a plasma boundary
            if plasma_boundary_or_error.is_err() {
                // Extract the reasons for no boundary found
                let plasma_boundary_error: plasma_geometry::Error = plasma_boundary_or_error.err().unwrap();
                let (no_xpt_reason, no_limit_point_reason) = match plasma_boundary_error {
                    plasma_geometry::Error::NoBoundaryFound {
                        no_xpt_reason,
                        no_limit_point_reason,
                    } => (no_xpt_reason, no_limit_point_reason),
                };
                // Set time-slice to failed, storing the reason in this module's own Error enum
                self.set_to_failed_time_slice(Error::NoBoundaryFound {
                    no_xpt_reason,
                    no_limit_point_reason,
                });

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }
            // Unwrap and store the plasma boundary
            let plasma_boundary: BoundaryContour = plasma_boundary_or_error.unwrap();
            profiles_2d.mask = plasma_boundary.mask.unwrap();
            self.time_slice.boundary.psi = plasma_boundary.bounding_psi;
            self.time_slice.boundary.bounding.r = plasma_boundary.bounding_r;
            self.time_slice.boundary.bounding.z = plasma_boundary.bounding_z;
            let mask: &Array2<f64> = &profiles_2d.mask;
            let psi_b: f64 = self.time_slice.boundary.psi;
            // "type" is a Rust key word, so we need to use the "raw identifier" = `r#` to access it.
            self.time_slice.boundary.r#type = plasma_boundary.xpt_diverted as i32;

            // Calculate psi_norm_2d
            profiles_2d.psi_norm = mask * (psi_2d - psi_a) / (psi_b - psi_a);
            let psi_norm_2d: &Array2<f64> = &profiles_2d.psi_norm;

            // Calculate GS error
            let gs_error_calculated: f64 = Self::calculate_gs_error(psi_a, psi_b, psi_a_previous);
            self.time_slice.convergence.grad_shafranov_deviation_value = gs_error_calculated;
            psi_a_previous = psi_a; // needed to calculate gs-error in next iteration

            // Check for convergence
            if gs_error_calculated < gs_error_tolerance && i_iter > n_iter_min {
                self.time_slice.convergence.iterations_n = i_iter as i32;
                self.time_slice.convergence.result.name = "converged".to_string();
                self.time_slice.convergence.result.index = CONVERGENCE_STATUS_CONVERGED;
                break 'iteration_loop; // Exit the iteration loop
            }

            // Check if we have reached the maximum number of iterations
            if i_iter == n_iter_max - 1 {
                // Set time-slice to failed
                self.set_to_failed_time_slice(Error::MaxIterReached);

                // Exit iteration loop for this time-slice
                break 'iteration_loop;
            }

            // Flatten variables
            let mask_flat: Array1<f64> = mask.flatten().to_owned();
            let psi_norm_flat: Array1<f64> = psi_norm_2d.flatten().to_owned();
            let j_2d_flat: Array1<f64> = j_2d.flatten().to_owned();

            let n_vertical_stabilisation: usize;
            if i_iter > n_iter_no_vertical_feedback {
                n_vertical_stabilisation = 1;
            } else {
                n_vertical_stabilisation = 0;
            }

            let n_dof: usize = n_p_prime_dof + n_ff_prime_dof + n_passive_dof + n_vertical_stabilisation;
            // Create the fitting matrix
            let mut fitting_matrix: Array2<f64> = Array2::zeros((n_constraints, n_dof));
            let mut constraint_weights: Array1<f64> = Array1::zeros(n_constraints);
            let mut constraint_values_from_coils: Array1<f64> = Array1::zeros(n_constraints);
            let mut s_measured: Array1<f64> = Array1::zeros(n_constraints);

            // Counter for the constraints
            let mut i_constraint: usize = 0;

            // Add bp_probes to fitting matrix
            for i_sensor in 0..n_bp {
                // j = 2.0 * pi * r * p_prime + 2.0 * pi * ff_prime / (mu_0 * r)

                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_bp_probes_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_bp_probes_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // j_2d is not consistent with mask. This inconsistency is how the plasma can "move" from iteration to iteration
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_bp_probes_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_bp_probes_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = bp_probes_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = bp_probes_static.fit_settings_weight[i_sensor] / bp_probes_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add flux_loops to fitting matrix
            for i_sensor in 0..n_fl {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_flux_loops_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_flux_loops_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_flux_loops_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                    //  * &mask_flat
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_flux_loops_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                // s_measured[i_constraint] = flux_loops_rs.all.psi.measured[i_sensor];
                s_measured[i_constraint] = flux_loops_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    2.0 * PI * flux_loops_static.fit_settings_weight[i_sensor] / flux_loops_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add dialoop (diamagnetic flux loop) to the fitting matrix.
            //
            // The diamagnetic loop responds to the toroidal flux function `f` (the poloidal-current
            // function), which depends only on the ff' source function — NOT on the toroidal
            // currents. The Green's tables relate toroidal currents to psi/B_p, so the dialoop uses
            // NO Green's functions, and no p', passive, coil or vertical-stabilisation terms.
            //
            // The diamagnetic flux is (Moret Eq. 41):
            //     Phi_t = integral( (f - f_vac) / R ) dA          (over the plasma mask)
            // where, as in `epp_bt_2d`, `f` is reconstructed from the ff' source function:
            //     f = sqrt( f_vac^2 + 2*(psi_b - psi_a)*G ),   G = sum_i ff'_dof[i]*ff'_integral_i(psi_norm)
            // and f_vac = R0*B_phi0 = MU_0*i_rod/(2*PI).
            //
            // Linearising for small diamagnetism (|f - f_vac| << |f_vac|):
            //     f - f_vac ~= (psi_b - psi_a) * G / f_vac
            // so the response is linear in the ff' degrees of freedom:
            //     T[i] = ((psi_b - psi_a) / f_vac) * dA * sum_grid [ mask * ff'_integral_i(psi_norm) / R ]
            //
            // Note on sign: this linearisation divides by the *signed* f_vac, so it already
            // preserves the correct sign for a negative TF rod current. Expanding the exact
            // f = sign(f_vac)*sqrt(f_vac^2 + 2*(psi_b-psi_a)*G) for small G gives
            // f - f_vac ~= (psi_b - psi_a)*G / f_vac, matching the term below without a separate
            // sign() factor.
            let i_rod: f64 = 2.0 * PI * self.vacuum_toroidal_field_r0 * self.vacuum_toroidal_field_b0 / MU_0;
            let f_vac: f64 = MU_0 * i_rod / (2.0 * PI);
            let d_psi: f64 = psi_b - psi_a;
            for i_sensor in 0..n_dialoop {
                // ff_prime degrees of freedom only (no p', no passives, no coils, no Green's)
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    let ff_prime_integral: Array1<f64> = ff_prime_source_function.source_function_integral_single_dof(&psi_norm_flat, i_ff_prime_dof);
                    let integrand: Array1<f64> = &mask_flat * &ff_prime_integral / &flat_r;
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = (d_psi / f_vac) * d_area * integrand.sum();
                }

                // Store sensor value
                s_measured[i_constraint] = dialoop_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = dialoop_static.fit_settings_weight[i_sensor] / dialoop_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add rogowski_coils to fitting matrix
            for i_sensor in 0..n_rog {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_rogowski_coils_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_rogowski_coils_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_rogowski_coils_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_rogowski_coils_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = rogowski_coils_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    rogowski_coils_static.fit_settings_weight[i_sensor] / rogowski_coils_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux to fitting matrix
            for i_sensor in 0..n_isoflux {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_isoflux_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_isoflux_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = isoflux_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] = isoflux_static.fit_settings_weight[i_sensor] / isoflux_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add isoflux_boundary to fitting matrix
            for i_sensor in 0..n_isoflux_boundary {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_isoflux_boundary_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] =
                        greens_isoflux_boundary_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                // TODO: check vertical stability for isoflux_boundary!!!!
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        0.0 * d_area * (&greens_d_isoflux_boundary_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_isoflux_boundary_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = psi_a;

                // Store weights
                constraint_weights[i_constraint] =
                    isoflux_boundary_static.fit_settings_weight[i_sensor] / isoflux_boundary_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add pressure_sensors to fitting matrix
            // d(psi)/d(psi_norm)
            let d_psi_d_psi_norm: f64 = 1.0 / (psi_b - psi_a);

            'loop_over_pressure_sensors: for i_sensor in 0..n_pressure_sensors {
                // Find the value of psi_norm at the location of the pressure sensor
                let sensor_r: f64 = pressure_sensors_static.geometry_r[i_sensor];
                let sensor_z: f64 = pressure_sensors_static.geometry_z[i_sensor];

                // Find the nearest grid point to the sensor location
                let i_r_nearest: usize = (r - sensor_r).abs().argmin().unwrap();
                let i_z_nearest: usize = (z - sensor_z).abs().argmin().unwrap();

                // Find the four corner grid points surrounding the pressure sensor
                let i_r_nearest_left: usize;
                let i_r_nearest_right: usize;
                let i_z_nearest_lower: usize;
                let i_z_nearest_upper: usize;
                if pressure_sensors_static.geometry_r[i_sensor] > r[i_r_nearest] {
                    i_r_nearest_left = i_r_nearest;
                    i_r_nearest_right = i_r_nearest + 1;
                } else {
                    i_r_nearest_left = i_r_nearest - 1;
                    i_r_nearest_right = i_r_nearest;
                }
                if pressure_sensors_static.geometry_z[i_sensor] > z[i_z_nearest] {
                    i_z_nearest_lower = i_z_nearest;
                    i_z_nearest_upper = i_z_nearest + 1;
                } else {
                    i_z_nearest_lower = i_z_nearest - 1;
                    i_z_nearest_upper = i_z_nearest;
                }

                // Gather psi and its gradients at the four corner grid points surrounding the magnetic axis
                let f: ArrayView2<f64> = psi_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d_f_d_r: ArrayView2<f64> = d_psi_d_r_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d_f_d_z: ArrayView2<f64> = d_psi_d_z_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);
                let d2_f_d_r_d_z: ArrayView2<f64> = d2_psi_d_r_d_z_2d.slice(s![i_z_nearest_lower..=i_z_nearest_upper, i_r_nearest_left..=i_r_nearest_right]);

                // Create a bicubic interpolator
                let bicubic_interpolator: BicubicInterpolator = BicubicInterpolator::new(d_r, d_z, f, d_f_d_r, d_f_d_z, d2_f_d_r_d_z);

                // Find psi at the pressure sensor
                let x: f64 = (pressure_sensors_static.geometry_r[i_sensor] - r[i_r_nearest_left]) / d_r;
                let y: f64 = (pressure_sensors_static.geometry_z[i_sensor] - z[i_z_nearest_lower]) / d_z;
                let psi_at_sensor: f64 = bicubic_interpolator.interpolate(x, y);

                let psi_norm_at_sensor: f64 = (psi_at_sensor - psi_a) / (psi_b - psi_a);
                if !(0.0..=1.0).contains(&psi_norm_at_sensor) {
                    println!(
                        "Warning: pressure sensor {} is outside of the plasma boundary (psi_norm = {})",
                        i_sensor, psi_norm_at_sensor
                    );
                    // Skip to the next sensor
                    continue 'loop_over_pressure_sensors;
                }

                let psi_norm_from_sensor_to_boundary: Array1<f64> = Array1::from_vec(vec![psi_norm_at_sensor, 1.0]);

                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    // Indefinitive integral of p_prime = pressure
                    let indefinite_integral_p_prime: Array1<f64> =
                        p_prime_source_function.source_function_integral_single_dof(&psi_norm_from_sensor_to_boundary, i_p_prime_dof);

                    // The constant of integration is zero pressure at the boundary; or this can be thought of as a definite integral from the sensor to the boundary
                    // let definite_integral_p_prime: f64 = indefinite_integral_p_prime[1] - indefinite_integral_p_prime[0];
                    let definite_integral_p_prime: f64 = indefinite_integral_p_prime[0] - indefinite_integral_p_prime[1];

                    // Add to fitting matrix
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = definite_integral_p_prime / d_psi_d_psi_norm;
                }

                // Vertical stability (not for pressure sensors)
                // TODO: should there be vertical stability for pressure sensors? I don't think so?

                // Store sensor values
                s_measured[i_constraint] = pressure_sensors_dynamic.measured[i_sensor];

                // Store weights
                constraint_weights[i_constraint] =
                    pressure_sensors_static.fit_settings_weight[i_sensor] / pressure_sensors_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add magnetic_axis to fitting matrix
            for i_sensor in 0..n_magnetic_axis_constraints {
                // p_prime degrees of freedom
                for i_p_prime_dof in 0..n_p_prime_dof {
                    fitting_matrix[(i_constraint, i_p_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_magnetic_axis_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * p_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_p_prime_dof)
                            * &flat_r)
                            .sum();
                }

                // ff_prime degrees of freedom
                for i_ff_prime_dof in 0..n_ff_prime_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + i_ff_prime_dof)] = 2.0
                        * PI
                        * d_area
                        * (&greens_magnetic_axis_grid.slice(s![.., i_sensor])
                            * &mask_flat
                            * ff_prime_source_function.source_function_value_single_dof(&psi_norm_flat, i_ff_prime_dof)
                            / (MU_0 * &flat_r))
                            .sum();
                }

                // Add passive degrees of freedom
                for i_passive_dof in 0..n_passive_dof {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + i_passive_dof)] = greens_magnetic_axis_passives[(i_passive_dof, i_sensor)];
                }

                // Vertical stability (using previous iteration)
                if i_iter > n_iter_no_vertical_feedback {
                    fitting_matrix[(i_constraint, n_p_prime_dof + n_ff_prime_dof + n_passive_dof)] =
                        d_area * (&greens_d_magnetic_axis_dz.slice(s![.., i_sensor]) * &j_2d_flat).sum();
                }

                // PF coil component
                let tmp: Array1<f64> = &greens_magnetic_axis_pf.slice(s![.., i_sensor]) * pf_coil_currents;
                constraint_values_from_coils[i_constraint] = tmp.sum();

                // Store sensor values
                s_measured[i_constraint] = 0.0; // Magnetic axis value is always zero

                // Store weights
                constraint_weights[i_constraint] =
                    magnetic_axis_static.fit_settings_weight[i_sensor] / magnetic_axis_static.fit_settings_expected_value[i_sensor];

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Pressure sensor:
            // 1.) Find where the pressure sensors are located in `psi_norm`
            // 2.) Calculate the "sensor" measurement matrix:
            //     `pressure[psi_norm] = pressure_int_dof_01 * d(psi)/d(psi_norm) + pressure_int_dof_02 * d(psi)/d(psi_norm) + ... = measured_pressure`
            //     where `pressure_int_dof_xx` = integral from LCFS to psi_norm of basis function xx

            // Add p_prime_regularisation to fitting matrix
            let p_prime_regularisation: Array2<f64> = p_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_p_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, 0..n_p_prime_dof])
                    .assign(&p_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Add ff_prime_regularisation to fitting matrix
            let ff_prime_regularisation: Array2<f64> = ff_prime_source_function.source_function_regularisation(); // shape = [n_regularisation, n_dof]
            for i_regularisation in 0..n_ff_prime_regularisation {
                // Add regularisation to fitting matrix
                fitting_matrix
                    .slice_mut(s![i_constraint, n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof])
                    .assign(&ff_prime_regularisation.slice(s![i_regularisation, ..]));
                // Store weights
                constraint_weights[i_constraint] = 1.0;
                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // // Add passive regularisation to the fitting matrix
            let regularisation_scaling: f64 = 0.001 * PI; // This regularisation_scaling factor need improving and explaining!

            let passive_regularisations: &Array2<f64> = &self.passive_regularisations;
            let passive_regularisations_weight: &Array1<f64> = &self.passive_regularisations_weight;
            for i_regularisation in 0..n_passive_regularisation {
                let passive_regularisation: ArrayView1<f64> = passive_regularisations.slice(s![i_regularisation, ..]);

                // Add passive degrees of freedom
                fitting_matrix
                    .slice_mut(s![
                        i_constraint,
                        n_p_prime_dof + n_ff_prime_dof..=n_p_prime_dof + n_ff_prime_dof + n_passive_dof - 1
                    ])
                    .assign(&passive_regularisation);

                // Add weight
                constraint_weights[i_constraint] = passive_regularisations_weight[i_regularisation] * regularisation_scaling;

                // Setup indexer for next sensor or constraint
                i_constraint += 1;
            }

            // Solve for the least squares problem for the source function coefficients, passive currents, and vertical stability
            let a: Array2<f64> = Array2::from_diag(&constraint_weights).dot(&fitting_matrix); // matrix-matrix multiplication
            let b: Array1<f64> = &constraint_weights * &s_measured - &constraint_weights * &constraint_values_from_coils;

            fn l2_norm(v: ArrayView1<f64>) -> f64 {
                // Sum of squares of the elements in the vector
                let sum_of_squares: f64 = v.iter().map(|&x| x * x).sum();
                // Take the square root to get the L2 norm
                sum_of_squares.sqrt()
            }

            // Preconditioner
            let n_cols: usize = a.ncols();

            // Compute the L2 norm for each column and fill the diagonal of D
            let mut d: Array2<f64> = Array2::zeros((n_cols, n_cols)); // Initialize a square matrix D with zeros
            for i in 0..n_cols {
                let norm: f64 = l2_norm(a.column(i));
                // Fill the diagonal of D with the inverse of the norm, or 0.0 if the norm is zero
                if norm > 0.0 {
                    d[(i, i)] = 1.0 / norm;
                } else {
                    println!("Warning: norm of column {} is zero, setting preconditioner to zero.", i);
                    d[(i, i)] = 0.0;
                }
            }

            let a_preconditioned: Array2<f64> = a.dot(&d);

            // SVD-based least squares solve using faer (equivalent to LAPACK dgelss)
            let (m_usize, n_usize) = a_preconditioned.dim();
            let a_faer: faer::Mat<f64> = faer::Mat::from_fn(m_usize, n_usize, |i, j| a_preconditioned[(i, j)]);
            let b_faer: faer::Mat<f64> = faer::Mat::from_fn(m_usize, 1, |i, _| b[i]);

            let svd: FaerSvd<f64> = FaerSvd::new_thin(a_faer.as_ref()).unwrap();
            let x_faer: faer::Mat<f64> = svd.solve_lstsq(b_faer.as_ref());

            let mut d_new_vec: Vec<f64> = Vec::with_capacity(n_dof);
            for i_dof in 0..n_dof {
                d_new_vec.push(x_faer[(i_dof, 0)]);
            }
            let d_new: Array1<f64> = Array1::from_vec(d_new_vec);

            let mut dof_values: Array1<f64> = d.dot(&d_new); // `d` is the preconditioning matrix

            // // Could add Anderson mixing here??????????????
            // if i_iter > 3 {
            //     dof_values = 0.6 * &dof_values + 0.4 * &dof_values_previous;
            // }
            // let dof_values_old: Array1<f64> = dof_values.clone();

            // Compute the condition number from SVD singular values
            let s_col = svd.S().column_vector();
            let mut s: Vec<f64> = Vec::with_capacity(s_col.nrows());
            for i_singular_value in 0..s_col.nrows() {
                s.push(s_col[i_singular_value]);
            }
            if let (Some(&sigma_max), Some(&sigma_min)) = (s.first(), s.iter().filter(|&&x| x > 0.0).last()) {
                let _condition_number: f64 = sigma_max / sigma_min;
            } else {
                println!("Matrix is rank-deficient or singular, condition number is undefined.");
            }

            // // Add Anderson mixing. Will this help???  // NO: Anderson mixing does not seem to help!!
            // if i_iter > 3 {
            //     dof_values = 0.3 * &dof_values + 0.7 * &dof_values_previous;
            // }

            if i_iter > n_iter_no_vertical_feedback {
                dof_values_previous = dof_values.clone();
            }

            // Extract p_prime
            let p_prime_dof_values: Array1<f64> = dof_values.slice(s![0..n_p_prime_dof]).to_owned();
            self.time_slice.source_functions.p_prime.coefficients = p_prime_dof_values;

            // Extract ff_prime
            let ff_prime_dof_values: Array1<f64> = dof_values.slice(s![n_p_prime_dof..n_p_prime_dof + n_ff_prime_dof]).to_owned();
            self.time_slice.source_functions.ff_prime.coefficients = ff_prime_dof_values;

            // Extract passive currents
            let passive_dof_values: Array1<f64> = dof_values
                .slice(s![n_p_prime_dof + n_ff_prime_dof..n_p_prime_dof + n_ff_prime_dof + n_passive_dof])
                .to_owned();
            self.passive_dof_values = passive_dof_values;

            // Extract vertical stability
            let delta_z: f64;
            if i_iter > n_iter_no_vertical_feedback {
                delta_z = dof_values.last().unwrap().to_owned();
            } else {
                delta_z = 0.0;
            }
            self.time_slice.convergence.delta_z = delta_z;

            // Calculate j_2d
            self.calculate_j();
            let j_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].j_phi;

            // Total plasma current
            // TODO: do we actually need to calculate Ip at every iteration?
            let i_2d: Array2<f64> = j_2d * d_area;
            let ip: f64 = i_2d.sum();
            self.time_slice.global_quantities.ip = ip;

            // // Write the time-slice to numpy files for debugging
            // self._write_time_slice_to_file(i_iter);
        }
    }

    /// Calculate the poloidal flux, psi, in the 2d (r, z) grid.
    ///
    /// 1. Calculate the "unshifted" flux and the required derivatives (9 fields):
    ///     * `psi_unshifted`
    ///     * `d_psi_d_r_unshifted`
    ///     * `d_psi_d_z_unshifted`
    ///     * `d2_psi_d_r2_unshifted`
    ///     * `d2_psi_d_r_d_z_unshifted`
    ///     * `d2_psi_d_z2_unshifted`
    ///     * `d3_psi_d_r2_d_z_unshifted`
    ///     * `d3_psi_d_r_d_z2_unshifted`
    ///     * `d3_psi_d_z3_unshifted`
    /// 2. Apply the vertical stability correction (resulting in 6 fields):
    ///     * `psi = psi_unshifted + delta_z * d_psi_d_z_unshifted`
    ///     * `d_psi_d_r = d_psi_d_r_unshifted + delta_z * d2_psi_d_r_d_z_unshifted`
    ///     * `d_psi_d_z = d_psi_d_z_unshifted + delta_z * d2_psi_d_z2_unshifted`
    ///     * `d2_psi_d_r2 = d2_psi_d_r2_unshifted + delta_z * d3_psi_d_r2_d_z_unshifted`
    ///     * `d2_psi_d_r_d_z = d2_psi_d_r_d_z_unshifted + delta_z * d3_psi_d_r_d_z2_unshifted`
    ///     * `d2_psi_d_z2 = d2_psi_d_z2_unshifted + delta_z * d3_psi_d_z3_unshifted`
    /// These are the derivatives we require for the bicubic interpolation to find the x-point and magnetic axis
    /// 3. Store the shifted flux and derivatives in the class.
    ///
    /// Only `psi` and its derivatives are used; `br` and `bz` do not appear.
    ///
    /// The plasma contribution is calculated with two GEMMs; see `PsiAndDerivativesGreens` for the
    /// reorganisation of the convolution over current sources.
    /// `temporary_storage` is owned by the caller only so that its buffers are not reallocated on
    /// every iteration; it carries nothing between calls. See
    /// [`PsiAndDerivativesTemporaryStorage`].
    pub fn calculate_psi_and_derivatives(&mut self, greens_tables: &PsiAndDerivativesGreens, temporary_storage: &mut PsiAndDerivativesTemporaryStorage) {
        // Unpack from self
        let n_r: usize = self.equilibrium_code.grid.n_r as usize;
        let n_z: usize = self.equilibrium_code.grid.n_z as usize;
        let d_area: f64 = self.time_slice.profiles_2d[0].grid.d_area;
        let j_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].j_phi;
        let pf_coil_currents: &Array1<f64> = &self.coils_dynamic.measured;
        let passive_dof_values: &Array1<f64> = &self.passive_dof_values;
        // NaN until the first inverse solve has run, which is when the vertical shift starts being
        // applied
        let delta_z: f64 = self.time_slice.convergence.delta_z;

        // ====================================================================
        // Part 1: the "unshifted" flux and derivatives
        // ====================================================================

        // Helper: contract a Greens matrix (n_z * n_r, n_dof) with a dof vector and reshape to (n_z, n_r)
        let contract = |g_matrix: &Array2<f64>, dof_values: &Array1<f64>| -> Array2<f64> {
            return g_matrix.dot(dof_values).to_shape((n_z, n_r)).unwrap().to_owned();
        };

        // PF coils
        // `psi` is precomputed (the PF currents are fixed within a time-slice);
        // the other fields are the Greens tables contracted with the PF currents
        let psi_2d_coils: &Array2<f64> = &self.time_slice.profiles_2d[0].psi_coils;
        let d_psi_d_r_2d_coils: Array2<f64> = contract(&greens_tables.g_d_psi_d_r_coils_matrix, pf_coil_currents);
        let d_psi_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d_psi_d_z_coils_matrix, pf_coil_currents);
        let d2_psi_d_r2_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r2_coils_matrix, pf_coil_currents);
        let d2_psi_d_r_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r_d_z_coils_matrix, pf_coil_currents);
        let d2_psi_d_z2_2d_coils: Array2<f64> = contract(&greens_tables.g_d2_psi_d_z2_coils_matrix, pf_coil_currents);
        let d3_psi_d_r2_d_z_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r2_d_z_coils_matrix, pf_coil_currents);
        let d3_psi_d_r_d_z2_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r_d_z2_coils_matrix, pf_coil_currents);
        let d3_psi_d_z3_2d_coils: Array2<f64> = contract(&greens_tables.g_d3_psi_d_z3_coils_matrix, pf_coil_currents);

        // Passives: the Greens tables contracted with the passive degrees of freedom
        let psi_2d_passives: Array2<f64> = contract(&greens_tables.g_psi_passives_matrix, passive_dof_values);
        let d_psi_d_r_2d_passives: Array2<f64> = contract(&greens_tables.g_d_psi_d_r_passives_matrix, passive_dof_values);
        let d_psi_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d_psi_d_z_passives_matrix, passive_dof_values);
        let d2_psi_d_r2_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r2_passives_matrix, passive_dof_values);
        let d2_psi_d_r_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_r_d_z_passives_matrix, passive_dof_values);
        let d2_psi_d_z2_2d_passives: Array2<f64> = contract(&greens_tables.g_d2_psi_d_z2_passives_matrix, passive_dof_values);
        let d3_psi_d_r2_d_z_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r2_d_z_passives_matrix, passive_dof_values);
        let d3_psi_d_r_d_z2_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_r_d_z2_passives_matrix, passive_dof_values);
        let d3_psi_d_z3_2d_passives: Array2<f64> = contract(&greens_tables.g_d3_psi_d_z3_passives_matrix, passive_dof_values);

        // Plasma: two GEMMs over the reorganised tables (see `PsiAndDerivativesGreens`). The two
        // scratch buffers gather the current sources by (vertical offset, source radius):
        //     w_even[(i_z, i_offset_z * n_r + i_cur_r)] = d_area * (j_below + j_above)
        //     w_odd[(i_z, i_offset_z * n_r + i_cur_r)]  = d_area * (j_below - j_above)
        // where `j_below = j_2d[(i_z - i_offset_z, i_cur_r)]` (a source below the grid point) and
        // `j_above = j_2d[(i_z + i_offset_z, i_cur_r)]` (a source at or above the grid point).
        // The odd kernels (`d_psi_d_z`, `d2_psi_d_r_d_z`) change sign with the source side:
        // sources with `i_z <= i_cur_z` enter with -1
        //
        // Each element is assigned rather than accumulated into, so that the scratch buffers never
        // have to be zeroed. They arrive holding the previous iteration's values, and every element
        // of both is overwritten here before any of it is read
        for i_z in 0..n_z {
            for i_offset_z in 0..n_z {
                let i_column_start: usize = i_offset_z * n_r;

                // Source below the grid point: i_cur_z = i_z - i_offset_z (excluding i_offset_z = 0)
                let i_cur_z_below: Option<usize> = if i_offset_z > 0 && i_z >= i_offset_z { Some(i_z - i_offset_z) } else { None };

                // Source at or above the grid point: i_cur_z = i_z + i_offset_z (including i_offset_z = 0)
                let i_cur_z_above: Option<usize> = if i_z + i_offset_z < n_z { Some(i_z + i_offset_z) } else { None };

                for i_cur_r in 0..n_r {
                    let j_below: f64 = match i_cur_z_below {
                        Some(i_cur_z) => d_area * j_2d[(i_cur_z, i_cur_r)],
                        None => 0.0,
                    };
                    let j_above: f64 = match i_cur_z_above {
                        Some(i_cur_z) => d_area * j_2d[(i_cur_z, i_cur_r)],
                        None => 0.0,
                    };
                    // The leading `0.0 +` is not redundant. It reproduces the zero-initialised
                    // accumulator these two lines used to start from, which matters because
                    // `j_phi` holds `-0.0` outside the mask (a negative `p_prime` times a zero
                    // mask). `0.0 + -0.0` is `+0.0`, so dropping it would flip the sign of those
                    // zeros and stop this being a bit-for-bit no-op
                    temporary_storage.w_even[(i_z, i_column_start + i_cur_r)] = 0.0 + j_below + j_above;
                    temporary_storage.w_odd[(i_z, i_column_start + i_cur_r)] = 0.0 + j_below - j_above;
                }
            }
        }

        // GEMMs, using `faer`. `Par::rayon(0)` - the whole pool - unconditionally. This runs
        // inside the caller's parallel loop over time-slices, so it looks like nested parallelism
        // worth avoiding, but it is not: measured with `Par::Seq` at 8, 16 and 32 threads it is
        // slightly *slower* at every count (e.g. 8 threads 120.3 s vs 124.8 s over the solve),
        // because with the pool already saturated by the outer loop faer runs the GEMM inline. At
        // a single time-slice the outer loop provides no parallelism and this is the only thing
        // keeping the cores busy. Unconditional is right for both regimes.
        let mut plasma_even: faer::Mat<f64> = faer::Mat::zeros(n_z, 5 * n_r);
        matmul(
            plasma_even.as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(temporary_storage.w_even.as_slice().unwrap(), n_z, n_z * n_r),
            MatRef::from_row_major_slice(greens_tables.g_even_plasma_by_offset.as_slice().unwrap(), n_z * n_r, 5 * n_r),
            1.0,
            Par::rayon(0),
        );
        let mut plasma_odd: faer::Mat<f64> = faer::Mat::zeros(n_z, 4 * n_r);
        matmul(
            plasma_odd.as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(temporary_storage.w_odd.as_slice().unwrap(), n_z, n_z * n_r),
            MatRef::from_row_major_slice(greens_tables.g_odd_plasma_by_offset.as_slice().unwrap(), n_z * n_r, 4 * n_r),
            1.0,
            Par::rayon(0),
        );

        // Assemble the unshifted fields (coils + passives + plasma).
        // The `plasma_even` / `plasma_odd` column blocks follow the concatenation order
        // documented on `PsiAndDerivativesGreens`
        let mut psi_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_r_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d_psi_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_r_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d2_psi_d_z2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_r2_d_z_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_r_d_z2_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        let mut d3_psi_d_z3_2d_unshifted: Array2<f64> = Array2::from_elem((n_z, n_r), f64::NAN);
        for i_z in 0..n_z {
            for i_r in 0..n_r {
                psi_2d_unshifted[(i_z, i_r)] = psi_2d_coils[(i_z, i_r)] + psi_2d_passives[(i_z, i_r)] + plasma_even[(i_z, i_r)];
                d_psi_d_r_2d_unshifted[(i_z, i_r)] = d_psi_d_r_2d_coils[(i_z, i_r)] + d_psi_d_r_2d_passives[(i_z, i_r)] + plasma_even[(i_z, n_r + i_r)];
                d2_psi_d_r2_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_r2_2d_coils[(i_z, i_r)] + d2_psi_d_r2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 2 * n_r + i_r)];
                d2_psi_d_z2_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_z2_2d_coils[(i_z, i_r)] + d2_psi_d_z2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 3 * n_r + i_r)];
                d3_psi_d_r_d_z2_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_r_d_z2_2d_coils[(i_z, i_r)] + d3_psi_d_r_d_z2_2d_passives[(i_z, i_r)] + plasma_even[(i_z, 4 * n_r + i_r)];
                d_psi_d_z_2d_unshifted[(i_z, i_r)] = d_psi_d_z_2d_coils[(i_z, i_r)] + d_psi_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, i_r)];
                d2_psi_d_r_d_z_2d_unshifted[(i_z, i_r)] =
                    d2_psi_d_r_d_z_2d_coils[(i_z, i_r)] + d2_psi_d_r_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, n_r + i_r)];
                d3_psi_d_r2_d_z_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_r2_d_z_2d_coils[(i_z, i_r)] + d3_psi_d_r2_d_z_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, 2 * n_r + i_r)];
                d3_psi_d_z3_2d_unshifted[(i_z, i_r)] =
                    d3_psi_d_z3_2d_coils[(i_z, i_r)] + d3_psi_d_z3_2d_passives[(i_z, i_r)] + plasma_odd[(i_z, 3 * n_r + i_r)];
            }
        }

        // ====================================================================
        // Part 2: apply the vertical stability correction
        // ====================================================================
        // `delta_z` is unset before the first inverse solve (and 0.0 while the vertical feedback is off)
        let psi_2d: Array2<f64>;
        let d_psi_d_r_2d: Array2<f64>;
        let d_psi_d_z_2d: Array2<f64>;
        let d2_psi_d_r2_2d: Array2<f64>;
        let d2_psi_d_r_d_z_2d: Array2<f64>;
        let d2_psi_d_z2_2d: Array2<f64>;
        if delta_z.is_nan() {
            psi_2d = psi_2d_unshifted;
            d_psi_d_r_2d = d_psi_d_r_2d_unshifted;
            d_psi_d_z_2d = d_psi_d_z_2d_unshifted;
            d2_psi_d_r2_2d = d2_psi_d_r2_2d_unshifted;
            d2_psi_d_r_d_z_2d = d2_psi_d_r_d_z_2d_unshifted;
            d2_psi_d_z2_2d = d2_psi_d_z2_2d_unshifted;
        } else {
            psi_2d = psi_2d_unshifted + delta_z * &d_psi_d_z_2d_unshifted;
            d_psi_d_r_2d = d_psi_d_r_2d_unshifted + delta_z * &d2_psi_d_r_d_z_2d_unshifted;
            d_psi_d_z_2d = d_psi_d_z_2d_unshifted + delta_z * &d2_psi_d_z2_2d_unshifted;
            d2_psi_d_r2_2d = d2_psi_d_r2_2d_unshifted + delta_z * &d3_psi_d_r2_d_z_2d_unshifted;
            d2_psi_d_r_d_z_2d = d2_psi_d_r_d_z_2d_unshifted + delta_z * &d3_psi_d_r_d_z2_2d_unshifted;
            d2_psi_d_z2_2d = d2_psi_d_z2_2d_unshifted + delta_z * &d3_psi_d_z3_2d_unshifted;
        }

        // ====================================================================
        // Part 3: store the shifted flux and derivatives in the class
        // ====================================================================
        self.time_slice.profiles_2d[0].psi = psi_2d;
        self.time_slice.profiles_2d[0].d_psi_d_r = d_psi_d_r_2d;
        self.time_slice.profiles_2d[0].d_psi_d_z = d_psi_d_z_2d;
        self.time_slice.profiles_2d[0].d2_psi_d_r2 = d2_psi_d_r2_2d;
        self.time_slice.profiles_2d[0].d2_psi_d_r_d_z = d2_psi_d_r_d_z_2d;
        self.time_slice.profiles_2d[0].d2_psi_d_z2 = d2_psi_d_z2_2d;
    }

    fn calculate_j(&mut self) {
        // Unpack from self
        let mesh_r: &Array2<f64> = &self.time_slice.profiles_2d[0].r;
        let psi_norm_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].psi_norm;
        let n_r: usize = self.equilibrium_code.grid.n_r as usize;
        let n_z: usize = self.equilibrium_code.grid.n_z as usize;
        let mask: &Array2<f64> = &self.time_slice.profiles_2d[0].mask;
        let p_prime_source_function: &Arc<dyn SourceFunctionTraits + Send + Sync> = &self.p_prime_source_function;
        let ff_prime_source_function: &Arc<dyn SourceFunctionTraits + Send + Sync> = &self.ff_prime_source_function;

        // Calculate profiles
        let psi_norm_flat: Array1<f64> = psi_norm_2d.flatten().to_owned();

        let p_prime_dof_values: &Array1<f64> = &self.time_slice.source_functions.p_prime.coefficients;
        let ff_prime_dof_values: &Array1<f64> = &self.time_slice.source_functions.ff_prime.coefficients;

        let p_prime_2d: Array2<f64> = p_prime_source_function
            .source_function_value(&psi_norm_flat, p_prime_dof_values)
            .to_shape((n_z, n_r))
            .unwrap()
            .to_owned();
        let j_2d_p_prime: Array2<f64> = 2.0 * PI * mesh_r * p_prime_2d * mask;

        let ff_prime_2d: Array2<f64> = ff_prime_source_function
            .source_function_value(&psi_norm_flat, ff_prime_dof_values)
            .to_shape((n_z, n_r))
            .unwrap()
            .to_owned();
        let j_2d_ff_prime: Array2<f64> = 2.0 * PI * ff_prime_2d * mask / (MU_0 * mesh_r);

        // Calculate j_2d
        let j_2d: Array2<f64> = j_2d_p_prime + j_2d_ff_prime;
        self.time_slice.profiles_2d[0].j_phi = j_2d;
    }

    /// Set this time-slice's starting `j_phi`, `psi_coils` and magnetic-axis position.
    ///
    /// The current-density seed is not built here: every input to it is shared between
    /// time-slices, so `grad_shafranov_solver` builds it once and this takes a borrow.
    /// `psi_coils` *is* per-time-slice, because it depends on this slice's measured PF currents.
    pub fn initialise_plasma_with_quadratic_current_density(&mut self) -> Result<(), String> {
        // Unpack objects
        let initial_guess_cur_r: f64 = self.equilibrium_code.initial_guess.cur_r;
        let initial_guess_cur_z: f64 = self.equilibrium_code.initial_guess.cur_z;
        let coils_dynamic: &SensorsDynamic = self.coils_dynamic;

        // Extract stuff from Coils
        let pf_currents: &Array1<f64> = &coils_dynamic.measured;

        // Flux from the poloidal field coils, from the Greens tables on the IDS. The coils are
        // summed in IDS order, which is the order the measured currents are in
        let n_pf: usize = self.greens_tables.pf_active.len();
        let n_r: usize = self.equilibrium_code.grid.n_r as usize;
        let n_z: usize = self.equilibrium_code.grid.n_z as usize;

        let mut psi_2d_coils: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_pf in 0..n_pf {
            let g_psi_coil: &Array2<f64> = &self.greens_tables.pf_active[i_pf].psi;
            if g_psi_coil.is_empty() {
                return Err(format!("equilibrium_solve: `greens/pf_active({i_pf})/psi` unset"));
            }
            psi_2d_coils = psi_2d_coils + g_psi_coil * pf_currents[i_pf];
        }

        // The seed, built once by the caller. Cloned because each time-slice owns the `j_phi` it
        // then overwrites on every iteration
        let j_2d: Array2<f64> = match self.initial_j_2d {
            Ok(initial_j_2d) => initial_j_2d.clone(),
            Err(reason) => return Err(reason.clone()),
        };

        // Store in self
        self.time_slice.profiles_2d[0].j_phi = j_2d;
        self.time_slice.profiles_2d[0].psi_coils = psi_2d_coils;
        self.time_slice.global_quantities.magnetic_axis.r = initial_guess_cur_r;
        self.time_slice.global_quantities.magnetic_axis.z = initial_guess_cur_z;
        Ok(())
    }

    /// Calculate the Grad-Shafranov "error"
    /// In the Picard iteration we change the solution by the error,
    /// so what we are doing here is checking to see how much the solutions
    /// is changing by
    /// Takes its inputs as arguments rather than reading them off `&mut self`, so that the caller
    /// can hold a borrow of `self.time_slice` across the call. A `&mut self` method borrows the
    /// whole struct, which would conflict with the `profiles_2d_*` pointers in `solve`.
    fn calculate_gs_error(psi_a: f64, psi_b: f64, psi_a_previous: f64) -> f64 {
        // Calculate the "error", in the same way EFIT does (called `cerror`)
        // Note, while this might "look" like a convergence test, it is in fact very similar
        // to a residule, since at each iteration the solution changes by the residule
        let gs_error_calculated: f64 = (psi_a - psi_a_previous).abs() / (psi_b - psi_a).abs();

        gs_error_calculated
    }

    /// Calculate the Grad Shafranov error by calcuating the LHS and RHS
    /// on the 2D (r, z) grid and seeing the difference = LHS - RHS.
    ///
    /// **This function is only used for development**
    fn _calculate_gs_error_numerical(&mut self) {
        let psi_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].psi;
        let r: &Array1<f64> = &self.time_slice.profiles_2d[0].grid.dim1;
        let z: &Array1<f64> = &self.time_slice.profiles_2d[0].grid.dim2;

        // Define some variables
        let d_r: f64 = r[1] - r[0];
        let d_z: f64 = z[1] - z[0];
        let n_r: usize = self.equilibrium_code.grid.n_r as usize;
        let n_z: usize = self.equilibrium_code.grid.n_z as usize;

        // Laplacian(psi)
        let mut laplacian_psi: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 1..(n_r - 1) {
            for i_z in 1..(n_z - 1) {
                let d2_psi_dz2: f64 = (psi_2d[(i_z + 1, i_r)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z - 1, i_r)]) / (d_z * d_z);
                let d2_psi_dr2: f64 = (psi_2d[(i_z, i_r + 1)] - 2.0 * psi_2d[(i_z, i_r)] + psi_2d[(i_z, i_r - 1)]) / (d_r * d_r);
                let r_d_psi_dr: f64 = (1.0 / r[i_r]) * (psi_2d[(i_z, i_r + 1)] - psi_2d[(i_z, i_r - 1)]) / (2.0 * d_r);

                laplacian_psi[(i_z, i_r)] = d2_psi_dr2 - r_d_psi_dr + d2_psi_dz2;
            }
        }
        let mask: &Array2<f64> = &self.time_slice.profiles_2d[0].mask;
        laplacian_psi *= mask;

        // RHS of Grad-Shafranov equation
        // Eq. 3 in "Tokamak equilibrium reconstruction code LIUQE and its real time implementation", 2015
        let j_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].j_phi;
        let mut gs_rhs: Array2<f64> = Array2::zeros((n_z, n_r));
        for i_r in 0..n_r {
            let tmp: Array1<f64> = -2.0 * PI * MU_0 * r[i_r] * j_2d.slice(s![.., i_r]).to_owned();
            gs_rhs.slice_mut(s![.., i_r]).assign(&tmp);
        }

        // Calculate the residual
        // Note - there is high residual at the boundary
        // Perhaps we should make the mask larger??
        let residual_2d: Array2<f64> = laplacian_psi - gs_rhs;
        println!("{:?}", residual_2d);
    }

    /// Writes the current time slice to numpy files for debugging
    ///
    /// **This function is only used for development**
    fn _write_time_slice_to_file(&self, i_iter: usize) {
        use std::path::Path;

        // Equivalent to `mkdir -p tmp`
        std::fs::create_dir_all("tmp").unwrap();

        let psi_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].psi;
        let d_psi_d_r_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].d_psi_d_r;
        let d_psi_d_z_2d: &Array2<f64> = &self.time_slice.profiles_2d[0].d_psi_d_z;
        let psi_b: f64 = self.time_slice.boundary.psi;
        let bounding_r: f64 = self.time_slice.boundary.bounding.r;
        let bounding_z: f64 = self.time_slice.boundary.bounding.z;
        let mag_r: f64 = self.time_slice.global_quantities.magnetic_axis.r;
        let mag_z: f64 = self.time_slice.global_quantities.magnetic_axis.z;

        // Filename has two leading zeros, e.g. i_iter=000, i_iter=001, ...
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_psi_2d.npy", i_iter)), psi_2d);
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_d_psi_d_r_2d.npy", i_iter)), d_psi_d_r_2d);
        npy_reader_and_writer::write_npy_2d(Path::new(&format!("tmp/i_iter={:03}_d_psi_d_z_2d.npy", i_iter)), d_psi_d_z_2d);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_psi_b.npy", i_iter)), psi_b);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_bounding_r.npy", i_iter)), bounding_r);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_bounding_z.npy", i_iter)), bounding_z);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_mag_r.npy", i_iter)), mag_r);
        npy_reader_and_writer::write_npy_0d(Path::new(&format!("tmp/i_iter={:03}_mag_z.npy", i_iter)), mag_z);
    }
    /// Convert the stationary points found in `psi` into contour-tree nodes.
    ///
    /// Classified by the second-derivative test: a negative determinant is a saddle (X-point); a
    /// positive determinant is a minimum when the trace is positive and a maximum when it is
    /// negative. The data dictionary node carries only the position, `psi` and the classification,
    /// so the Hessian and grid-index fields of `StationaryPoint` are not stored.
    fn contour_tree_nodes(stationary_points: &[StationaryPoint]) -> Vec<EquilibriumContourTreeNode> {
        let n_stationary_point: usize = stationary_points.len();
        let mut nodes: Vec<EquilibriumContourTreeNode> = Vec::with_capacity(n_stationary_point);
        for i_stationary_point in 0..n_stationary_point {
            let stationary_point: &StationaryPoint = &stationary_points[i_stationary_point];
            let critical_type: i32 = if stationary_point.hessian_determinant < 0.0 {
                1
            } else if stationary_point.hessian_trace > 0.0 {
                0
            } else {
                2
            };
            nodes.push(EquilibriumContourTreeNode {
                critical_type,
                r: stationary_point.r,
                z: stationary_point.z,
                psi: stationary_point.psi,
                ..Default::default()
            });
        }
        nodes
    }

    /// Copy the solution into an IMAS `EquilibriumTimeSlice`.
    ///
    /// Keys with no counterpart in the data dictionary are custom keys, declared by hand in
    /// `imas_rs/imas_updater/custom_keys/custom_equilibrium_keys.rs`.
    pub fn write_to_time_slice(&mut self) {
        // TODO: this should go into:
        // `time_slice(itime)/constraints/pf_passive_current(i1)/reconstructed`
        // but the IMAS data structure needs a bit of thought, since our passive degrees of freedom are the
        // eigenmodes! Perhaps we store each filament's current individually?
        // Degrees of freedom
        self.time_slice.passive_dof_values = self.passive_dof_values.to_owned();
    }
}

/// The data dictionary's `code/output_flag` for one time-slice.
///
/// 0 when the slice is usable; negative when it is not - "Negative values mean the result shall
/// not be used". The magnitude is the convergence status, so a consumer can tell an unconverged
/// slice from a fatal error without reading `convergence/result`.
///
/// `output_flag` is `INT_1D` indexed by time and lives on the IDS rather than the time-slice, so it
/// is assembled by the caller once every slice has been solved.
pub fn output_flag(time_slice: &EquilibriumTimeSlice) -> i32 {
    let convergence_status: i32 = time_slice.convergence.result.index;
    if convergence_status == CONVERGENCE_STATUS_CONVERGED {
        return 0;
    }
    if convergence_status == EMPTY_INT {
        // Never written, so the slice was never solved at all
        return -CONVERGENCE_STATUS_FATAL_ERROR;
    }
    -convergence_status
}

/// Everything the Grad-Shafranov solve needs which is not already in the `Equilibrium` IDS.
///
/// The sensors and the source functions are solver *inputs*, and the source functions are
/// behaviour (`Arc<dyn SourceFunctionTraits>`) rather than data, so neither can live in `imas_rs`
/// - it would have to depend on `gsfit_rs`, which already depends on it. They are passed in here
/// instead.
///
/// Everything here is already down-selected to a single time-slice by the caller, so the solver
/// is never handed a time index; it cannot read the wrong slice, and it does not know which slice
/// it is solving.
pub struct GradShafranovInputs<'a> {
    /// The Greens tables reorganised for `calculate_psi_and_derivatives`. Built once, before the
    /// parallel loop over time-slices, because it depends only on the geometry
    pub psi_and_derivatives_greens: &'a PsiAndDerivativesGreens,
    /// The initial current-density guess, built once before the loop over time-slices, or the
    /// reason it could not be built
    pub initial_j_2d: &'a Result<Array2<f64>, String>,
    pub coils_dynamic: &'a SensorsDynamic,
    pub bp_probes_static: &'a SensorsStatic,
    pub bp_probes_dynamic: &'a SensorsDynamic,
    pub flux_loops_static: &'a SensorsStatic,
    pub flux_loops_dynamic: &'a SensorsDynamic,
    pub dialoop_static: &'a SensorsStatic,
    pub dialoop_dynamic: &'a SensorsDynamic,
    pub rogowski_coils_static: &'a SensorsStatic,
    pub rogowski_coils_dynamic: &'a SensorsDynamic,
    pub isoflux_static: &'a SensorsStatic,
    pub isoflux_dynamic: &'a SensorsDynamic,
    pub isoflux_boundary_static: &'a SensorsStatic,
    pub isoflux_boundary_dynamic: &'a SensorsDynamic,
    pub pressure_sensors_static: &'a SensorsStatic,
    pub pressure_sensors_dynamic: &'a SensorsDynamic,
    pub magnetic_axis_static: &'a SensorsStatic,
    pub magnetic_axis_dynamic: &'a SensorsDynamic,
    pub p_prime_source_function: &'a Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub ff_prime_source_function: &'a Arc<dyn SourceFunctionTraits + Send + Sync>,
    pub passive_regularisations: &'a Array2<f64>,
    pub passive_regularisations_weight: &'a Array1<f64>,
}
