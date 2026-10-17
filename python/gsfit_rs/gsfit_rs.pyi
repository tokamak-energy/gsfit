from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .dd_support import ParametersInput
from .imas import Equilibrium
from .imas import Magnetics as MagneticsIds
from .imas import Path
from .imas import PfActive as PfActiveIds
from .imas import PfPassive as PfPassiveIds
from .imas import PulseSchedule as PulseScheduleIds
from .imas import Tf as TfIds
from .imas import Wall as WallIds

_T = TypeVar("_T")

class DataTreeAccessor:
    """Base class providing common data tree access methods for all gsfit_rs classes."""
    def get_f64(self, keys: list[str]) -> float:
        """
        Get single f64 value.

        :param keys: The path of keys to access the data, e.g. `["level1", "level2", "data_f64_value"]`.
        :return: f64 value
        """
        ...
    def get_array1(self, keys: list[str]) -> npt.NDArray[np.float64]:
        """
        Get 1D f64 numpy array.

        :param keys: The path of keys to access the data, e.g. `["level1", "level2", "data_f64_1d_array"]`.
                 Wildcards are supported, e.g. `["level1", "*", "data_f64_value"]`.
        :return: 1D numpy array of float64 values
        """
        ...
    def get_array2(self, keys: list[str]) -> npt.NDArray[np.float64]:
        """
        Get 2D f64 numpy array.

        :param keys: The path of keys to access the data, e.g. ["level1", "level2", "data_f64_2d_array"].
                Wildcards are supported, e.g. `["level1", "*", "data_f64_2d_array"]` or `["*", "*", "data_f64_value"]`.
                With wildcards the indexing order is from right to left.
        :return: 2D numpy array of float64 values
        """
        ...
    def get_array3(self, keys: list[str]) -> npt.NDArray[np.float64]:
        """
        Get 3D f64 numpy array.

        :param keys: The path of keys to access the data, e.g. ["level1", "level2", "data_f64_3d_array"].
                Wildcards are supported, e.g. `["level1", "*", "data_f64_3d_array"]` or `["*", "*", "data_f64_value"]`.
                With wildcards the indexing order is from right to left.
        :return: 3D numpy array of float64 values
        """
        ...
    def get_bool(self, keys: list[str]) -> bool: ...
    def get_usize(self, keys: list[str]) -> int: ...
    def get_vec_bool(self, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(self, keys: list[str]) -> list[int]: ...
    def keys(self, key_path: list[str] | None = None) -> list[str]: ...
    def pop(
        self,
        keys: list[str],
    ) -> None:
        """
        Remove from the data structure

        :param keys: Path to remove data structure
        """
        ...
    def print_keys(self) -> None: ...

def solve_grad_shafranov(
    plasma: Plasma,
    bp_probes: BpProbes,
    flux_loops: FluxLoops,
    rogowski_coils: RogowskiCoils,
    isoflux: Isoflux,
    isoflux_boundary: IsofluxBoundary,
    pressure_sensors: Pressure,
    stationary_point: StationaryPoint,
    dialoop: Dialoop,
) -> None:
    """
    :param plasma: Plasma object, note this is mutated and contains the solution. Its equilibrium IDS supplies the limiter,
        the vacuum toroidal field, the PF coil currents, the passive loops' regularisations and any gaps, so
        `Plasma.copy_from_wall_ids`, `Plasma.copy_from_tf_ids`, `Plasma.copy_from_pf_active_ids` and `Plasma.copy_from_pf_passive_ids`
        have to have been called first. `Plasma.copy_from_pulse_schedule_ids` is optional: without it the equilibrium has no gaps.
        The passive loops have to be the ones the Greens tables, `greens/grid/pf_passive`, were calculated with, by `Plasma.greens_with_pf_passive`.
        The fitted passive degrees of freedom are written to every time-slice's `passive_dof_values`, in the order of `greens/grid/pf_passive`
    :param bp_probes: BpProbes object, note this is mutated and contains the solution
    :param flux_loops: FluxLoops object, note this is mutated and contains the solution
    :param rogowski_coils: RogowskiCoils object, note this is mutated and contains the solution
    :param isoflux: Isoflux object, note this is mutated and contains the solution
    :param isoflux_boundary: IsofluxBoundary object, note this is mutated and contains the solution
    :param pressure_sensors: Pressure object, note this is mutated and contains the solution
    :param stationary_point: StationaryPoint object, note this is mutated and contains the solution
    :param dialoop: Dialoop object, note this is mutated and contains the solution

    The times to reconstruct are read from `plasma`, which was built with one equilibrium
    time-slice per time.
    """
    ...

def solve_circuit_equations(
    pf_active: PfActive,
    pf_passive: PfPassive,
    times_to_solve: npt.NDArray[np.float64],
    adaptive_time_stepping: bool,
) -> tuple[PfActive, PfPassive]:
    """
    Solves the circuit equations

    Note: the adaptive time-stepping can produce a lot of simulated time-points,
    so `adaptive_time_stepping = False` should be used to avoid unwieldly large outputs

    :param pf_active: The PF coils, each with its current or its voltage prescribed (`coil(i1)/prescribed`)
    :param pf_passive: The passive conductors, whose filaments, `loop(i1)/element(i2)`, carry eddy currents
    :param times_to_solve: Times to solve the circuit equations at [second]
    :param adaptive_time_stepping: The calculation always uses adaptive time stepping, this flag interpolates results onto `times_to_solve`
    :return: A copy of `pf_active`, whose `coil(i1)/current` is the simulated current, on the simulated timebase; and a copy of
        `pf_passive`, whose `loop(i1)/element(i2)/current` is the simulated current, on the simulated timebase, `loop(i1)/time`
    """
    ...

def greens_py(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    :param r: (by convention) Sensor radial positions, must be `> 0.0` [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions, must be `> 0.0` [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]

    Note: the inputs are symmetrical
    """
    ...

def greens_d_psi_d_r(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    First derivative of the Green's function with respect to `r`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d_psi_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    First derivative of the Green's function with respect to `z`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_r2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Second derivative of the Green's function with respect to `r`.

    When a sensor coincides with a current source, the source's `d_r` and `d_z`
    are used to evaluate the analytic self-term
    (see `documentation/jump_condition_dbr_dz.md`).

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_r_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Mixed second derivative of the Green's function with respect to `r` and `z`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: Radial widths [metre]
    :param d_z: Vertical heights [metre]
    """
    ...

def greens_d2_psi_d_z2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Second derivative of the Green's function with respect to `z`.

    When a sensor coincides with a current source, the source's `d_r` and `d_z`
    are used to evaluate the analytic self-term
    (see `documentation/jump_condition_dbr_dz.md`); if they are omitted the
    self-term is zero.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_z3(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `z` three times.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_r2_d_z(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `r` twice and `z`.

    Computed from the homogeneous Grad-Shafranov equation:
    `d3_psi_d_r2_d_z = d2_psi_d_r_d_z / r - d3_psi_d_z3`.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

def greens_d3_psi_d_r_d_z2(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Third derivative of the Green's function with respect to `r` and `z` twice.

    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]
    """
    ...

class PfPassive:
    """The machine's passive conductors, e.g. the vacuum vessel, the passive plates and the PF coil cases, stored as an IMAS
    `pf_passive` IDS.

    For each passive conductor, `loop(i1)`: `name`; `resistivity`; one `element(i2)` per filament, a parallelogram
    (`geometry/oblique/r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta`) of one turn (`turns_with_sign` = 1), with its `area`;
    and the custom keys `current_distribution_type`, `dof`, `regularisations` and `regularisations_weight`.

    The data dictionary describes a parallelogram from one of its corners, the reference point `oblique/r` and `z`. One side,
    `length_alpha` long, leaves it at angle `alpha` counter-clockwise from the R axis; the other, `length_beta` long, leaves it at angle
    `beta` counter-clockwise from the Z axis.

    The passive currents are fitted, as the amplitudes of a few current distributions, `dof(i2)`, for each loop.
    `current_distribution_type` is an identifier, saying which: `constant_current_density` (`index` 1), one degree of freedom with the same
    current density in every element; or `eig` (`index` 2), the loop's `n_dof` slowest decaying eigenmodes. The degrees of freedom of every
    loop, in the order of `loop` and then of `dof`, are the order the reconstruction fits them in.

    `Plasma.copy_from_pf_passive_ids` copies each loop's degrees of freedom and regularisations onto every time-slice of the equilibrium
    IDS, as `constraints/pf_passive_current(i1)`, which is where the solver reads the regularisations from.

    Read it back through `pf_passive_ids` and a path from `gsfit_rs.imas.pf_passive_paths`.

    The passive conductors are constructed with the settings they are built from, which are kept in `parameters_input`,
    and whose XML is also written to `pf_passive/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PfPassive:
        """
        Construct empty passive conductors, ready for `add_loop` to be called.

        :param parameters_input: The settings the passive conductors are built from, e.g. which run of which database they
            are read from. Their XML is written to `pf_passive/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the passive conductors were built from. This is a copy: changing it does not change the passive conductors."""
        ...
    def add_loop(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        length_alpha: npt.NDArray[np.float64],
        length_beta: npt.NDArray[np.float64],
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
        resistivity: float,
        current_distribution_type: str,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
        regularisations_weight: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a passive conductor to `pf_passive/loop`.

        :param name: short identifier for the passive conductor, unique within the passive conductors, e.g. `"IVC"`
        :param r: major radius of each filament's reference point, the corner which `alpha` and `beta` are measured from [metre]
        :param z: height of each filament's reference point [metre]
        :param length_alpha: length of each filament's side inclined at `alpha` [metre]
        :param length_beta: length of each filament's side inclined at `beta` [metre]
        :param alpha: inclination of each filament's first side, counter-clockwise from the R axis [radian]
        :param beta: inclination of each filament's second side, counter-clockwise from the Z axis [radian]
        :param resistivity: resistivity of the conductor, the same for every filament [ohm * metre]
        :param current_distribution_type: `"constant_current_density"` or `"eig"`
        :param n_dof: number of degrees of freedom: 1 for `"constant_current_density"`, the number of eigenmodes for `"eig"`
        :param regularisations: the regularisations of the degrees of freedom, shape = [n_regularisations, n_dof]; no rows for none
        :param regularisations_weight: the weight of each regularisation, shape = [n_regularisations] [dimensionless]

        Each filament becomes one `element(i2)`: a parallelogram of one turn, with a positive `r` and positive lengths.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pf_passive_paths`, straight out of the pf_passive IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PfPassive.get`.
        """
        ...
    @property
    def pf_passive_ids(self) -> PfPassiveIds:
        """A copy of the whole pf_passive IDS, read with `gsfit_rs.imas.pf_passive_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Plasma(DataTreeAccessor):
    def __new__(
        cls,
        n_r: int,
        n_z: int,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        psi_norm: npt.NDArray[np.float64],
        p_prime_source_function: "EfitPolynomial" | "TensionedCubicBSpline",
        ff_prime_source_function: "EfitPolynomial" | "TensionedCubicBSpline",
        initial_guess_ip: float,
        initial_guess_cur_r: float,
        initial_guess_cur_z: float,
        initial_guess_minor_radius: float,
        initial_guess_elongation: float,
        n_iter_max: int,
        n_iter_min: int,
        grad_shafranov_deviation_tolerance: float,
        nonlinear_solver_method: str,
        picard_n_iter_no_vertical_feedback: int,
        picard_apply_anderson_mixing: bool,
        picard_anderson_n_history: int,
        picard_anderson_mixing: float,
        newton_krylov_picard_handover: float,
        newton_krylov_n_krylov_max: int,
        newton_krylov_krylov_tolerance: float,
        newton_krylov_finite_difference_step: float,
        newton_krylov_verbose: bool,
        newton_picard_n_basis_max: int,
        newton_picard_contraction_threshold: float,
        newton_picard_finite_difference_step: float,
        newton_picard_verbose: bool,
        times_to_reconstruct: npt.NDArray[np.float64],
    ) -> Plasma:
        """
        :param n_r: Number of radial poitns [dimensionless]
        :param n_z: Number of vertical poitns [dimensionless]
        :param r_min: Minimum radius [metre]
        :param r_max: Maximum radius [metre]
        :param z_min: Minimum vertical position [metre]
        :param z_max: Maximum vertical position [metre]
        :param psi_norm: 1D array for `psi_norm`, which should go from [0.0, 1.0] [dimensionless]
        :param p_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
        :param ff_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
        :param initial_guess_ip: Initial plasma current [ampere]
        :param initial_guess_cur_r: Radial centre of the initial current distribution [metre]
        :param initial_guess_cur_z: Vertical centre of the initial current distribution [metre]
        :param initial_guess_minor_radius: Radial semi-axis of the initial current distribution [metre]
        :param initial_guess_elongation: Elongation of the initial current distribution [dimensionless]
        :param n_iter_max: Maximum number of nonlinear solver iterations
        :param n_iter_min: Minimum number of nonlinear solver iterations before the convergence test may pass
        :param grad_shafranov_deviation_tolerance: Grad-Shafranov deviation below which the solution is taken as converged
        :param nonlinear_solver_method: The nonlinear solver: "picard", "newton_krylov" or "newton_picard"
        :param picard_n_iter_no_vertical_feedback: Number of initial Picard iterations with the vertical feedback switched off
        :param picard_apply_anderson_mixing: Whether the Picard iterations use Anderson mixing: each new state is the combination of the last few iterates which best cancels their residuals
        :param picard_anderson_n_history: Maximum number of previous iterations Anderson mixing combines
        :param picard_anderson_mixing: Fraction of the (Anderson-mixed) residual taken at each iteration; 1 for all of it [dimensionless]
        :param newton_krylov_picard_handover: Grad-Shafranov deviation below which Newton-Krylov hands over from Picard to Newton iterations
        :param newton_krylov_n_krylov_max: Maximum number of Krylov directions per Newton iteration; each costs one Picard iteration's work
        :param newton_krylov_krylov_tolerance: The Krylov solve stops once the linearised residual is this fraction of the residual [dimensionless]
        :param newton_krylov_finite_difference_step: Jacobian-vector product step, as a multiple of the size of the residual [dimensionless]
        :param newton_krylov_verbose: Whether to print the progress of each Newton iteration
        :param newton_picard_n_basis_max: Maximum number of directions in which Newton-Picard uses Newton's method; each costs one Picard iteration's work to add
        :param newton_picard_contraction_threshold: A direction is added when the residual outside them shrinks by less than this factor per iteration [dimensionless]
        :param newton_picard_finite_difference_step: Jacobian-vector product step, as a multiple of the size of the residual [dimensionless]
        :param newton_picard_verbose: Whether to print the progress of each Newton-Picard iteration
        :param times_to_reconstruct: Times the equilibrium will be solved at; one equilibrium time-slice is allocated per time [second]
        """
        ...
    def copy_from_wall_ids(self, wall: Wall) -> None:
        """
        Copy the limiter from the wall IDS into the equilibrium IDS.

        Copies `wall/description_2d(0)/limiter/unit(i)` into `equilibrium/limiter/unit(i)`, in the same order, so `unit(0)` is
        still the vacuum vessel contour. The solver and the post-processor read the limiter from the equilibrium IDS, so this has
        to be called before `solve_grad_shafranov`.

        Any limiter already in the equilibrium IDS is replaced, so calling this twice does not repeat the units.

        :param wall: The wall, holding the limiter
        :raises ValueError: When the wall has no limiter units
        """
        ...
    def copy_from_tf_ids(self, tf: Tf) -> None:
        """
        Copy the vacuum toroidal field from the tf IDS into the equilibrium IDS.

        Fills `equilibrium/vacuum_toroidal_field/r0` from `tf/r0`, and `equilibrium/vacuum_toroidal_field/b0` from
        `tf/b_field_phi_vacuum_r`. That is on the experimental timebase, so it is interpolated onto the reconstruction times, the
        times of the time-slices `Plasma` was built with, and divided by `r0`. The solver and the post-processor read the vacuum
        toroidal field from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`.

        Any vacuum toroidal field already in the equilibrium IDS is replaced.

        :param tf: The toroidal field
        :raises ValueError: When `tf/r0` or `tf/b_field_phi_vacuum_r` is unset, or when `tf/b_field_phi_vacuum_r` cannot be
            interpolated onto the reconstruction times, e.g. because its timebase does not cover them
        """
        ...
    def copy_from_pf_active_ids(self, pf_active: PfActive) -> None:
        """
        Copy the PF coil currents from the pf_active IDS onto every time-slice of the equilibrium IDS.

        Fills, for every coil `pf_active/coil(i1)`, in the same order, on every time-slice:
        `equilibrium/time_slice(itime)/constraints/pf_current(i1)/name`, a custom key, from `pf_active/coil(i1)/name`;
        `.../measured`, `pf_active/coil(i1)/current` interpolated onto the time-slice's time [ampere];
        `.../source`, the path `pf_active/coil(i1)/current`; and `.../time_measurement`, the time-slice's own time [second].

        The solver reads the PF coil currents from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`.
        The solver multiplies them by the Greens tables, `greens/grid/pf_active(i1)`, so the coils have to be in the same order as
        those, which the solver checks by name.

        Any PF coil currents already on a time-slice are replaced, so calling this twice does not repeat them.

        :param pf_active: The PF coils, holding their currents
        :raises ValueError: When `pf_active/coil` is empty, or when a coil's current cannot be interpolated onto the
            reconstruction times, e.g. because its timebase does not cover them
        """
        ...
    def copy_from_pf_passive_ids(self, pf_passive: PfPassive) -> None:
        """
        Copy the passive loops' degrees of freedom, and their regularisations, from the pf_passive IDS onto every time-slice of the
        equilibrium IDS.

        Fills, for every loop `pf_passive/loop(i1)`, in the same order, on every time-slice, these custom keys:
        `equilibrium/time_slice(itime)/constraints/pf_passive_current(i1)/name`, from `pf_passive/loop(i1)/name`;
        `.../dof(i2)/name`, from `pf_passive/loop(i1)/dof(i2)/name`; `.../regularisations`, from `pf_passive/loop(i1)/regularisations`,
        shape = (n_regularisations, n_dof); and `.../regularisations_weight`, from `pf_passive/loop(i1)/regularisations_weight` [dimensionless].

        `pf_passive_current(i1)` is where the data dictionary puts the current in `pf_passive/loop(i1)`. GSFit does not measure that
        current: it fits it, as an amplitude for each of the loop's degrees of freedom, and the regularisations are what constrain the
        fit. So `measured`, `source` and `time_measurement` are left unset.

        The solver reads the regularisations from the equilibrium IDS, so this has to be called before `solve_grad_shafranov`. It fits
        the amplitudes with the Greens tables, `greens/grid/pf_passive(i1)/dof(i2)`, so the loops and their degrees of freedom have to be in
        the same order as those, which the solver checks by name.

        A pf_passive IDS with no loops is allowed: the equilibrium then has no passive conductors. Any passive loops already on a
        time-slice are replaced, so calling this twice does not repeat them.

        :param pf_passive: The passive conductors, holding their degrees of freedom and regularisations
        """
        ...
    def copy_from_pulse_schedule_ids(self, pulse_schedule: PulseSchedule) -> None:
        """
        Copy the gap definitions from the pulse_schedule IDS onto every time-slice of the equilibrium IDS.

        Fills `equilibrium/time_slice(itime)/boundary/gap(i)/name`, `.../description`, `.../r`, `.../z` and `.../angle` from
        `pulse_schedule/position_control/gap(i)`, in the same order, on every time-slice. `.../value` is left unset, for the
        equilibrium post-processor to calculate. A time-slice which does not converge keeps the definitions, with its values unset.

        The pulse schedule is optional: without this call, or with a pulse schedule which has no gaps, the equilibrium has no gaps.
        Any gaps already on a time-slice are replaced, so calling this twice does not repeat them.

        :param pulse_schedule: The pulse schedule, holding the gap definitions
        """
        ...
    def copy_from_magnetics_ids(self, magnetics: Magnetics) -> None:
        """
        Copy the magnetic sensors' measurements from the magnetics IDS onto every time-slice of the equilibrium IDS.

        Each measurement is on the experimental timebase, so it is interpolated onto the time-slice's time. On every time-slice, for
        every sensor, in the same order as in the magnetics IDS: `magnetics/b_field_pol_probe(i1)/field` fills
        `equilibrium/time_slice(itime)/constraints/b_field_pol_probe(i1)` [tesla]; `magnetics/flux_loop(i1)/flux` fills
        `.../constraints/flux_loop(i1)` [weber]; `magnetics/rogowski_coil(i1)/current` fills `.../constraints/rogowski_coil(i1)`, a custom
        key, as the data dictionary has no constraint for Rogowski coils [ampere]; `magnetics/diamagnetic_flux(0)/data` fills
        `.../constraints/diamagnetic_flux` [weber]; and `magnetics/ip(0)/data` fills `.../constraints/ip` [ampere].

        Each constraint gets `name`, a custom key: the sensor's `name`, or the `method_name` for `diamagnetic_flux` and `ip`; `measured`,
        the measurement interpolated onto the time-slice's time; `source`, the path the measurement was read from; and `time_measurement`,
        the time-slice's own time [second]. `diamagnetic_flux` and `ip` take the magnetics IDS's first, generally recommended, method,
        and are left unset when it has none. Which sensors are fitted, and their weights, are not set here.

        Any magnetic constraints already on a time-slice are replaced, so calling this twice does not repeat them.

        :param magnetics: The magnetic sensors, holding their measurements
        :raises ValueError: When a measurement cannot be interpolated onto the reconstruction times, e.g. because its timebase does not
            cover them
        """
        ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_magnetics(self, magnetics: Magnetics, pf_active: PfActive, pf_passive: PfPassive) -> None:
        """
        Calculate the magnetic sensors' Greens tables: what each sensor measures, per ampere in each current source.

        Fills, for every sensor in the magnetics IDS, with the same name and in the same order, `greens/b_field_pol_probe(i1)`,
        `greens/flux_loop(i1)` and `greens/rogowski_coil(i1)` on the equilibrium IDS, each with the sensor's tables with: `plasma`, the
        plasma current in each grid cell, shape = (n_z, n_r); `d_plasma_d_z`, the same differentiated with respect to the sensor's
        height, shape = (n_z, n_r); `pf_active`, each PF coil, in the order of `greens/grid/pf_active`; and `pf_passive`, each passive
        degree of freedom, in the order of `greens/grid/pf_passive(:)/dof(:)`.

        `greens/grid/pf_active` and `greens/grid/pf_passive` name the PF coils and the passive degrees of freedom, so
        `Plasma.greens_with_pf_active` and `Plasma.greens_with_pf_passive` have to be called first, with the same `pf_active` and
        `pf_passive`. Any sensors' Greens tables already on the IDS are replaced, so calling this twice does not repeat them.

        :param magnetics: The magnetic sensors, holding where they are
        :param pf_active: The PF coils, holding their filaments
        :param pf_passive: The passive conductors, holding their filaments and degrees of freedom
        :raises ValueError: When the PF coils, or the passive loops and their degrees of freedom, are not the ones
            `greens/grid/pf_active` and `greens/grid/pf_passive` were calculated with, in the same order
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.equilibrium_paths`, straight out of the equilibrium IDS.

        Empty until the Grad-Shafranov solver has run.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Equilibrium.get`.
        """
        ...
    @property
    def equilibrium_ids(self) -> Equilibrium:
        """A copy of the whole equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Tf:
    """The machine's toroidal field, stored as an IMAS `tf` IDS.

    Two nodes are filled: `tf/r0`, the reference major radius, and
    `tf/b_field_phi_vacuum_r`, the vacuum field times major radius on the experimental
    timebase.

    `b_field_phi_vacuum_r` is the vacuum poloidal-current function
    `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`, so the rod current is not stored
    separately: `solve_grad_shafranov` recovers it as `i_rod = 2 * pi * f_vac / mu_0`.

    It is signed: positive means counter-clockwise viewed from above.

    Read it back through `tf_ids` and a path from `gsfit_rs.imas.tf_paths`.

    The toroidal field is constructed with the settings it is built from, which are kept in `parameters_input`,
    and whose XML is also written to `tf/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Tf:
        """
        Construct an empty toroidal field, ready for `set_r0` and `set_b_field_phi_vacuum_r`.

        :param parameters_input: The settings the toroidal field is built from, e.g. the reference major radius, and which
            run of which database the field is read from. Their XML is written to `tf/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the toroidal field was built from. This is a copy: changing it does not change the toroidal field."""
        ...
    def set_r0(self, r0: float) -> None:
        """
        Set `tf/r0`, the reference major radius.

        :param r0: reference major radius the vacuum toroidal field is quoted at [metre]

        `Plasma.copy_from_tf_ids` copies this onto `equilibrium/vacuum_toroidal_field/r0`, so that
        the two IDSs cannot disagree.
        """
        ...
    def set_b_field_phi_vacuum_r(
        self,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Set `tf/b_field_phi_vacuum_r`, the vacuum field times major radius.

        :param time: the experimental timebase [second]
        :param data: vacuum toroidal field times major radius [tesla * metre]

        Store the **experimental** signal, not one interpolated onto the reconstruction
        times: `Plasma.copy_from_tf_ids` interpolates it itself.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.tf_paths`, straight out of the tf IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Tf.get`.
        """
        ...
    @property
    def tf_ids(self) -> TfIds:
        """A copy of the whole tf IDS, read with `gsfit_rs.imas.tf_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Wall:
    """The machine's wall, stored as an IMAS `wall` IDS.

    Only the limiter is filled so far:
    `wall/description_2d(0)/limiter/unit(i)/outline/r` and `.../z`.

    `Plasma.copy_from_wall_ids` copies the units into the equilibrium IDS, as
    `equilibrium/limiter/unit(i)`, in the same order, and the solver reads them from there by
    position. So the order units are added in is part of the contract: `unit(0)` is the vacuum
    vessel contour, and the solver uses that one, and only that one, as the region the plasma
    is allowed to occupy. Every unit contributes candidate limit points.

    Read it back through `wall_ids` and a path from `gsfit_rs.imas.wall_paths`.

    The wall is constructed with the settings it is built from, which are kept in `parameters_input`,
    and whose XML is also written to `wall/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Wall:
        """
        Construct an empty wall, ready for `add_limiter_unit` to be called.

        :param parameters_input: The settings the wall is built from, e.g. which run of which database the
            limiter is read from. Their XML is written to `wall/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the wall was built from. This is a copy: changing it does not change the wall."""
        ...
    def add_limiter_unit(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a limiter unit to `wall/description_2d(0)/limiter/unit`.

        :param name: short identifier for the unit, e.g. `"vacuum_vessel"`
        :param r: outline radial points [metre]
        :param z: outline vertical points [metre]

        The **first** unit added is the vacuum vessel contour.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.wall_paths`, straight out of the wall IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Wall.get`.
        """
        ...
    @property
    def wall_ids(self) -> WallIds:
        """A copy of the whole wall IDS, read with `gsfit_rs.imas.wall_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class PulseSchedule:
    """The machine's pulse schedule, stored as an IMAS `pulse_schedule` IDS.

    Only the gap definitions are filled so far:
    `pulse_schedule/position_control/gap(i)/name`, `.../r`, `.../z` and `.../angle`.

    A gap is a reference point and a direction. `Plasma.copy_from_pulse_schedule_ids` copies the
    definitions onto every time-slice's `equilibrium/time_slice(itime)/boundary/gap`, and
    `solve_grad_shafranov` calculates each value: the distance from the reference point to the
    plasma boundary along that direction.

    The angle is stored in the equilibrium IDS's convention, clockwise from `grad(R)` in the usual
    plot with `R` to the right and `Z` upwards, so the gap points along
    `(cos(angle), -sin(angle))`. A database reader holding a counter-clockwise angle must convert it.

    Read it back through `pulse_schedule_ids` and a path from `gsfit_rs.imas.pulse_schedule_paths`.

    The pulse schedule is constructed with the settings it is built from, which are kept in `parameters_input`,
    and whose XML is also written to `pulse_schedule/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PulseSchedule:
        """
        Construct an empty pulse schedule, with no gaps, ready for `add_gap` to be called.

        :param parameters_input: The settings the pulse schedule is built from, e.g. which run of which database the
            gaps are read from. Their XML is written to `pulse_schedule/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the pulse schedule was built from. This is a copy: changing it does not change the pulse schedule."""
        ...
    def add_gap(
        self,
        name: str,
        r: float,
        z: float,
        angle: float,
    ) -> None:
        """
        Append a gap to `pulse_schedule/position_control/gap`.

        :param name: short identifier for the gap, unique within the pulse schedule, e.g. `"IMGAP"`
        :param r: major radius of the reference point [metre]
        :param z: height of the reference point [metre]
        :param angle: direction the gap is measured in, clockwise from `grad(R)` [radian]
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pulse_schedule_paths`, straight out of the pulse_schedule IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PulseSchedule.get`.
        """
        ...
    @property
    def pulse_schedule_ids(self) -> PulseScheduleIds:
        """A copy of the whole pulse_schedule IDS, read with `gsfit_rs.imas.pulse_schedule_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class PfActive:
    """The machine's active poloidal field (PF) coils, stored as an IMAS `pf_active` IDS.

    For each coil, `coil(i1)`: `name`; one `element(i2)` per filament, a rectangle
    (`geometry/rectangle/r`, `z`, `width` and `height`) of one turn (`turns_with_sign` = 1);
    `inductance`, the self-inductance calculated from the filaments; `current`, per turn, on the
    experimental timebase; `voltage`, only when the coil's voltage is prescribed; and `prescribed`,
    a custom key.

    `coil(i1)/prescribed` is an identifier, saying which of the coil's signals is given: `current`
    (`index` 1), set by `add_pf_coil`, or `voltage` (`index` 2), set by `set_pf_voltage_prescribed`,
    when the current is calculated.

    `coil(i1)/current` is signed: positive means counter-clockwise viewed from above.

    The circuits and supplies are left empty: a power supply which feeds several coils in series is
    described by giving each coil its current.

    `Plasma.copy_from_pf_active_ids` copies each coil's current onto every time-slice of the equilibrium
    IDS, as `constraints/pf_current(i1)`, interpolated onto the reconstruction times.

    Read it back through `pf_active_ids` and a path from `gsfit_rs.imas.pf_active_paths`.

    The PF coils are constructed with the settings they are built from, which are kept in `parameters_input`,
    and whose XML is also written to `pf_active/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> PfActive:
        """
        Construct empty PF coils, ready for `add_pf_coil` to be called.

        :param parameters_input: The settings the PF coils are built from, e.g. which run of which database the
            coils are read from. Their XML is written to `pf_active/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the PF coils were built from. This is a copy: changing it does not change the PF coils."""
        ...
    def add_pf_coil(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        width: npt.NDArray[np.float64],
        height: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a coil to `pf_active/coil`, with its current prescribed.

        :param name: short identifier for the coil, unique within the PF coils, e.g. `"BVL"`
        :param r: major radius of each filament's centre [metre]
        :param z: height of each filament's centre [metre]
        :param width: radial full width of each filament [metre]
        :param height: vertical full height of each filament [metre]
        :param time: the experimental timebase [second]
        :param current: the current per turn [ampere]

        Each filament becomes one `element(i2)`: a rectangle of one turn, with a positive `r`, `width` and `height`.
        The coil's self-inductance, `coil(i1)/inductance`, is calculated from them.
        Store the **experimental** current, not one interpolated onto the reconstruction times:
        `Plasma.copy_from_pf_active_ids` interpolates it itself. A single time holds the current at that time only.
        """
        ...
    def set_pf_voltage_prescribed(
        self,
        name: str,
        time: npt.NDArray[np.float64],
        voltage: npt.NDArray[np.float64],
    ) -> None:
        """
        Prescribe a coil's voltage, rather than its current.

        :param name: the coil, as named in `add_pf_coil`
        :param time: the experimental timebase [second]
        :param voltage: voltage on the coil terminals [volt]

        Writes `coil(i1)/voltage`, and sets `coil(i1)/prescribed` to `voltage`. The coil's `current` is kept.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.pf_active_paths`, straight out of the pf_active IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.PfActive.get`.
        """
        ...
    @property
    def pf_active_ids(self) -> PfActiveIds:
        """A copy of the whole pf_active IDS, read with `gsfit_rs.imas.pf_active_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Magnetics:
    """The machine's magnetic sensors, stored as an IMAS `magnetics` IDS.

    For each poloidal field probe, `b_field_pol_probe(i1)`: `name`; `position/r` and `z`, the probe's centre; `poloidal_angle`,
    the direction of the probe's normal; and `field`, the measured field along the normal, on the experimental timebase.

    For each flux loop, `flux_loop(i1)`: `name`; `type` = 1, `poloidal_flux`, a loop all the way around the machine; `position(0)/r`
    and `z`, the loop's one point in the poloidal plane; and `flux`, the measured poloidal flux, on the experimental timebase.

    For each Rogowski coil, `rogowski_coil(i1)`: `name`; `position(i2)/r` and `z`, the coil's contour; `current`, the measured
    current, on the experimental timebase; and `gaps(i2)/name`, `r_start`, `r_end`, `z_start` and `z_end`, a custom key, the stretches of
    the contour which have no winding.

    For the diamagnetic flux, `diamagnetic_flux(i1)`, and the plasma current, `ip(i1)`, one entry for each method of calculating
    them: `method_name`, `data` and `time`.

    For now, `b_field_pol_probe(i1)/poloidal_angle` keeps GSFit's present definition, COCOS 13, as MDSplus stores it: counter-clockwise
    from `grad(R)`, in the usual plot with `R` to the right and `Z` upwards, so the probe's normal is
    `cos(poloidal_angle) * grad(R) + sin(poloidal_angle) * grad(Z)`. It is the same angle as `BpProbes`'s `angle_pol`. This is not the
    data dictionary's COCOS 17, which measures it clockwise, within [0, 2 * pi]. `toroidal_angle` is left empty.

    `flux_loop(i1)/flux` is GSFit's `psi`, the total poloidal flux, with no factor of 2 * pi. `ip(i1)/data` and
    `diamagnetic_flux(i1)/data` are positive counter-clockwise viewed from above.

    Read it back through `magnetics_ids` and a path from `gsfit_rs.imas.magnetics_paths`.

    The magnetic sensors are constructed with the settings they are built from, which are kept in `parameters_input`, and whose XML is
    also written to `magnetics/code/parameters`.
    """

    def __new__(cls, parameters_input: ParametersInput) -> Magnetics:
        """
        Construct empty magnetic sensors, ready for `add_b_field_pol_probe`, `add_flux_loop`, `add_rogowski_coil`,
        `add_diamagnetic_flux` and `add_ip` to be called.

        :param parameters_input: The settings the magnetic sensors are built from, e.g. which run of which database the sensors are
            read from. Their XML is written to `magnetics/code/parameters`
        """
        ...
    @property
    def parameters_input(self) -> ParametersInput:
        """The settings the magnetic sensors were built from. This is a copy: changing it does not change the magnetic sensors."""
        ...
    def add_b_field_pol_probe(
        self,
        name: str,
        r: float,
        z: float,
        poloidal_angle: float,
        time: npt.NDArray[np.float64],
        field: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a poloidal field probe to `magnetics/b_field_pol_probe`.

        :param name: short identifier for the probe, unique within the poloidal field probes, e.g. `"P101"`
        :param r: major radius of the probe's centre [metre]
        :param z: height of the probe's centre [metre]
        :param poloidal_angle: angle of the probe's normal, counter-clockwise from `grad(R)`: GSFit's COCOS 13, not the data dictionary's COCOS 17 [radian]
        :param time: the experimental timebase [second]
        :param field: the measured field along the probe's normal [tesla]
        """
        ...
    def add_flux_loop(
        self,
        name: str,
        r: float,
        z: float,
        time: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a flux loop to `magnetics/flux_loop`.

        :param name: short identifier for the flux loop, unique within the flux loops, e.g. `"L001"`
        :param r: major radius of the loop [metre]
        :param z: height of the loop [metre]
        :param time: the experimental timebase [second]
        :param flux: the measured poloidal flux, GSFit's `psi` [weber]

        The loop goes all the way around the machine, so it is a `poloidal_flux` loop with one `position`.
        """
        ...
    def add_rogowski_coil(
        self,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        gaps_name: list[str] = ...,
        gaps_r_start: npt.NDArray[np.float64] | None = None,
        gaps_r_end: npt.NDArray[np.float64] | None = None,
        gaps_z_start: npt.NDArray[np.float64] | None = None,
        gaps_z_end: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Append a Rogowski coil to `magnetics/rogowski_coil`.

        :param name: short identifier for the Rogowski coil, unique within the Rogowski coils, e.g. `"INIVC000"`
        :param r: major radius of each point of the coil's contour, at least 3 points [metre]
        :param z: height of each point of the coil's contour [metre]
        :param time: the experimental timebase [second]
        :param current: the measured current [ampere]
        :param gaps_name: (optional) short identifier for each gap in the coil's winding, unique within the coil
        :param gaps_r_start: (optional) major radius of the start of each gap [metre]
        :param gaps_r_end: (optional) major radius of the end of each gap [metre]
        :param gaps_z_start: (optional) height of the start of each gap [metre]
        :param gaps_z_end: (optional) height of the end of each gap [metre]

        Each gap is the straight line from its start to its end; the documentation of `rogowski_coil(i1)/gaps(i2)/r_start` says which
        way it runs. A coil with no gaps leaves the gap arguments out.
        """
        ...
    def add_diamagnetic_flux(
        self,
        method_name: str,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a diamagnetic flux to `magnetics/diamagnetic_flux`, one for each method of calculating it.

        :param method_name: name of the method the diamagnetic flux is calculated with, unique within the diamagnetic fluxes, e.g. `"DIALOOP"`
        :param time: the experimental timebase [second]
        :param data: the diamagnetic flux, positive counter-clockwise viewed from above [weber]

        The data dictionary asks for the generally recommended method to be added first.
        """
        ...
    def add_ip(
        self,
        method_name: str,
        time: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
    ) -> None:
        """
        Append a plasma current to `magnetics/ip`, one for each method of calculating it.

        :param method_name: name of the method the plasma current is calculated with, unique within the plasma currents
        :param time: the experimental timebase [second]
        :param data: the plasma current, positive counter-clockwise viewed from above [ampere]

        The data dictionary asks for the generally recommended method to be added first.
        """
        ...
    def get(self, path: Path[_T]) -> _T:
        """Read the data at `path`, from `gsfit_rs.imas.magnetics_paths`, straight out of the magnetics IDS.

        This is how the data is read: a path holds no data, so the IDS is only borrowed for the
        read, never copied. The shape of the result follows the shape of the index; see
        `gsfit_rs.imas.Magnetics.get`.
        """
        ...
    @property
    def magnetics_ids(self) -> MagneticsIds:
        """A copy of the whole magnetics IDS, read with `gsfit_rs.imas.magnetics_paths`.

        Read the data with `get` instead: this copies the IDS on every access. It is for when a
        detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
        """
        ...

class Symmetry:
    """Up-down symmetry of the equilibrium configuration.

    `Symmetric` selects the 7-term cos(h_n y) expansion (limiter or double-null);
    `Asymmetric` adds the sin(h_n y) family for the 12-term single-null solve.
    """

    Symmetric: Symmetry
    Asymmetric: Symmetry

class Configuration:
    """Plasma boundary topology and the shape parameters it requires.

    Each variant carries exactly its own shape inputs; up-down symmetry
    (7- vs 12-term expansion) is implied by the variant.
    """
    @staticmethod
    def SymmetricLimited(kappa: float, delta: float) -> Configuration:
        """Up-down symmetric smooth limiter (7-term).

        :param kappa: Elongation [dimensionless]
        :param delta: Triangularity [dimensionless]
        """
        ...
    @staticmethod
    def AntisymmetricLimited(kappa_upper: float, delta_upper: float, kappa_lower: float, delta_lower: float) -> Configuration:
        """Up-down asymmetric smooth limiter (12-term). NOT YET IMPLEMENTED.

        :param kappa_upper: Upper elongation [dimensionless]
        :param delta_upper: Upper triangularity [dimensionless]
        :param kappa_lower: Lower elongation [dimensionless]
        :param delta_lower: Lower triangularity [dimensionless]
        """
        ...
    @staticmethod
    def DoubleNull(kappa_x: float, delta_x: float) -> Configuration:
        """Up-down symmetric double-null divertor (7-term).

        :param kappa_x: X-point elongation [dimensionless]
        :param delta_x: X-point triangularity [dimensionless]
        """
        ...
    @staticmethod
    def SingleNull(kappa: float, delta: float, kappa_x: float, delta_x: float) -> Configuration:
        """Up-down asymmetric lower single-null divertor (12-term). The upper half is a smooth Miller
        profile (`kappa`, `delta`) and the lower half has the X-point (`kappa_x`, `delta_x`).

        :param kappa: Upper (smooth) elongation [dimensionless]
        :param delta: Upper (smooth) triangularity [dimensionless]
        :param kappa_x: X-point elongation [dimensionless]
        :param delta_x: X-point triangularity [dimensionless]
        """
        ...

class GuazzottoFreidberg:
    """Guazzotto-Freidberg analytic Grad-Shafranov equilibrium.

    Solves the analytic equilibrium and produces a `PfActive` object whose PF-coil
    currents reproduce the vacuum field consistent with the solution, i.e. a
    self-consistent free-boundary test case for GSFit to reconstruct.

    Flux follows the IMAS convention, the total poloidal flux `psi = 2 * pi * R * A_phi` [weber],
    so `psi_0` and the equilibrium IDS are `2 * pi` times GF's per-radian values.

    Reference: J. Plasma Phys. 87 (2021) 905870303.
    """
    def __new__(
        cls,
        configuration: Configuration,
        eps: float,
        nu: float,
        r_geo: float,
        bt_vac_at_r_geo: float,
        p_axis: float,
        n_r: int,
        n_z: int,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        n_radial_expansion: int,
    ) -> GuazzottoFreidberg:
        """Construct the equilibrium; it is not solved until `solve` is called.

        :param configuration: Boundary topology + shape (SymmetricLimited, AntisymmetricLimited, DoubleNull, or SingleNull)
        :param eps: Inverse aspect ratio, a / r_geo [dimensionless]
        :param nu: Profile parameter, approximately the poloidal beta [dimensionless]
        :param r_geo: Geometric major radius [metre]
        :param bt_vac_at_r_geo: Vacuum toroidal field at r_geo [tesla]
        :param p_axis: Pressure on the magnetic axis [pascal]
        :param n_r: Number of radial grid points [dimensionless]
        :param n_z: Number of vertical grid points [dimensionless]
        :param r_min: Minimum radius [metre]
        :param r_max: Maximum radius [metre]
        :param z_min: Minimum vertical position [metre]
        :param z_max: Maximum vertical position [metre]
        :param n_radial_expansion: Number of terms in the C_n / S_n series (the paper's M) [dimensionless]
        """
        ...
    def solve(
        self,
    ) -> None:
        """Solve the eigenvalue `alpha` and evaluate the analytic flux and plasma mask over the (R, Z) grid."""
        ...
    def get_pf_active(
        self,
        coil_regularisation_weight: float,
        control_points: npt.NDArray[np.float64],
        wall: Wall,
    ) -> PfActive:
        """
        Fit the PF-coil currents on the (R, Z) grid boundary so that, with the plasma's own field, they reproduce
        the analytic flux at `control_points`. Then store the free-boundary equilibrium in `equilibrium_ids` and
        post-process it. Requires `solve` first.

        The IDS holds the total flux, plasma plus coils, which is valid in the vacuum as well as inside the plasma.
        For a diverted plasma the magnetic axis, the X-points, the boundary flux and the plasma mask are found in
        that flux, as the Grad-Shafranov solver finds them; for a limited one they are the analytic ones.

        :param coil_regularisation_weight: Tikhonov weight on the coil-current magnitude [dimensionless]
        :param control_points: (n_control_point, 2) array of (r, z) points where the flux is matched [metre]
        :param wall: The wall: its limiter is copied into `equilibrium_ids`, where its limiter and vacuum vessel bound a
            diverted plasma, and the post-processor traces the scrape-off layer legs up to it
        :return: One single-filament PF coil per boundary node, carrying the fitted current, at the single time 0
        """
        ...
    def get_sensor_values(
        self,
        bp_probes: BpProbes,
        flux_loops: FluxLoops,
        pf_active: PfActive,
    ) -> None:
        """
        Fill each BP probe (tesla) and flux loop (weber) `experimental` value with the field the analytic plasma
        current plus `pf_active` produce at that sensor. Mutates `bp_probes` and `flux_loops` in place. Requires `solve` first.

        Each coil filament carries its coil's current at the first time of its experimental timebase.

        :param bp_probes: BpProbes with geometry already set; their measurements are overwritten
        :param flux_loops: FluxLoops with geometry already set; their measurements are overwritten
        :param pf_active: The PF coils, typically the ones `get_pf_active` returned
        """
        ...
    def model_surface(
        self,
        n_point_per_half: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        GF's model surface, the shape the analytic boundary is matched to at the matching points (the red curve in
        GF's figures). Does not need `solve`.

        :param n_point_per_half: Number of points on each of the upper and lower halves; must be even
        :return: `(r, z)`, the closed outline, anticlockwise from the outer midplane; the last point repeats the first,
            so each is `2 * n_point_per_half + 1` long [metre]
        """
        ...
    @property
    def configuration(self) -> Configuration:
        """Boundary topology and its shape parameters"""
        ...
    @property
    def eps(self) -> float:
        """Inverse aspect ratio, a / r_geo [dimensionless]"""
        ...
    @property
    def nu(self) -> float:
        """Profile parameter, approximately the poloidal beta [dimensionless]"""
        ...
    @property
    def r_geo(self) -> float:
        """Geometric major radius [metre]"""
        ...
    @property
    def bt_vac_at_r_geo(self) -> float:
        """Vacuum toroidal field at r_geo [tesla]"""
        ...
    @property
    def p_axis(self) -> float:
        """Pressure normalisation, the pressure where psi_hat = 1 [pascal]"""
        ...
    @property
    def n_r(self) -> int:
        """Number of radial grid points [dimensionless]"""
        ...
    @property
    def n_z(self) -> int:
        """Number of vertical grid points [dimensionless]"""
        ...
    @property
    def r_min(self) -> float:
        """Minimum radius of the grid [metre]"""
        ...
    @property
    def r_max(self) -> float:
        """Maximum radius of the grid [metre]"""
        ...
    @property
    def z_min(self) -> float:
        """Minimum vertical position of the grid [metre]"""
        ...
    @property
    def z_max(self) -> float:
        """Maximum vertical position of the grid [metre]"""
        ...
    @property
    def n_radial_expansion(self) -> int:
        """Number of terms in the C_n / S_n series (the paper's M) [dimensionless]"""
        ...
    @property
    def alpha(self) -> float:
        """The eigenvalue; NaN until `solve` has run [dimensionless]"""
        ...
    @property
    def beta_0(self) -> float:
        """Beta at psi_hat = 1, 2 * mu_0 * p_axis / bt_vac_at_r_geo ** 2 (GF Eq. 6.4); NaN until `solve` has run [dimensionless]"""
        ...
    @property
    def bt_diamagnetic_shift(self) -> float:
        """Diamagnetic change in the toroidal field (GF Eq. 6.4); NaN until `solve` has run [tesla]"""
        ...
    @property
    def psi_0(self) -> float:
        """Flux normalisation, psi = psi_0 * psi_hat: GF's Psi_0 (Eq. 6.5) times 2 * pi; NaN until `solve` has run [weber]"""
        ...
    @property
    def psi_analytic(self) -> npt.NDArray[np.float64]:
        """The analytic flux, psi_0 * psi_hat, on every grid cell, shape (n_z, n_r) [weber].

        Outside the model box the truncated series is unphysical and can diverge; use `mask` to select the plasma.
        """
        ...
    @property
    def mask(self) -> npt.NDArray[np.float64]:
        """1 inside the analytic plasma and 0 elsewhere, shape (n_z, n_r) [dimensionless]"""
        ...
    @property
    def equilibrium_ids(self) -> Equilibrium:
        """The free-boundary equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`. Empty until `get_pf_active` has run.

        The IDS is copied into the returned object, so it is a snapshot: changes made on the Rust side afterwards
        are not seen by it.
        """
        ...

class BpProbes(DataTreeAccessor):
    def __new__(
        cls,
    ) -> BpProbes: ...
    def add_sensor(
        cls,
        name: str,
        geometry_angle_pol: float,
        geometry_r: float,
        geometry_z: float,
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None:
        """
        :param name: Name of the sensor
        :param geometry_angle_pol: Poloidal angle of the sensor geometry [radian]
        :param geometry_r: Radial position of the sensor geometry [metre]
        :param geometry_z: Vertical position of the sensor geometry [metre]
        :param fit_settings_comment: Comment for the fit settings, used for debugging
        :param fit_settings_expected_value: Expected value for the fit settings, used for normalisation [tesla]
        :param fit_settings_include: Whether to include this sensor in the fit [bool]
        :param fit_settings_weight: Weight for the sensor [dimensionless]
        :param time: Time array [second]
        :param measured: Measured values [tesla]
        """
        ...
    def calculate_sensor_values(
        cls,
        plasma: Plasma,
    ) -> None:
        """
        Calculate the sensor values from the PF coil currents and the fitted passive degrees of freedom on the plasma's
        equilibrium IDS, and the plasma.
        Mutates self

        :param plasma: Plasma object
        """
        ...
    def calculate_sensor_values_vacuum(
        cls,
        pf_active: PfActive,
        pf_passive: PfPassive,
    ) -> None:
        """
        Calculate the sensor values from the PF coils and the passive conductors.
        Mutates self

        :param pf_active: The PF coils, whose currents share one timebase, e.g. the ones `solve_circuit_equations` returned
        :param pf_passive: The passive conductors, whose filaments' currents, `loop(i1)/element(i2)/current`, are on the PF coils' timebase,
            e.g. the ones `solve_circuit_equations` returned
        """
        ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

class FluxLoops(DataTreeAccessor):
    def __new__(
        cls,
    ) -> FluxLoops: ...
    def add_sensor(
        cls,
        name: str,
        geometry_r: float,
        geometry_z: float,
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None:
        """
        :param name: Name of the sensor
        :param geometry_r: Radial position of the sensor geometry [metre]
        :param geometry_z: Vertical position of the sensor geometry [metre]
        :param fit_settings_comment: Comment for the fit settings, used for debugging
        :param fit_settings_expected_value: Expected value for the fit settings, used for normalisation [weber]
        :param fit_settings_include: Whether to include this sensor in the fit [bool]
        :param fit_settings_weight: Weight for the sensor [dimensionless]
        :param time: Time vector [second]
        :param measured: Measured values [weber]
        """
        ...
    def calculate_sensor_values(
        cls,
        plasma: Plasma,
    ) -> None: ...
    def calculate_sensor_values_vacuum(
        cls,
        pf_active: PfActive,
        pf_passive: PfPassive,
    ) -> None:
        """
        Calculate the sensor values from the PF coils and the passive conductors.
        Mutates self

        :param pf_active: The PF coils, whose currents share one timebase, e.g. the ones `solve_circuit_equations` returned
        :param pf_passive: The passive conductors, whose filaments' currents, `loop(i1)/element(i2)/current`, are on the PF coils' timebase,
            e.g. the ones `solve_circuit_equations` returned
        """
        ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

class RogowskiCoils(DataTreeAccessor):
    def __new__(
        cls,
    ) -> RogowskiCoils: ...
    def add_sensor(
        cls,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
        gaps_r: npt.NDArray[np.float64],
        gaps_z: npt.NDArray[np.float64],
        gaps_d_r: npt.NDArray[np.float64],
        gaps_d_z: npt.NDArray[np.float64],
        gaps_name: list[str],
    ) -> None:
        """
        :param name: Name of the sensor
        :param r: 1D array containing the radial positions of the sensor geometry [metre]
        :param z: 1D array containing the vertical positions of the sensor geometry [metre]
        :param fit_settings_comment: Comment for the fit settings, used for debugging
        :param fit_settings_expected_value: Expected value for the fit settings, used for normalisation [ampere]
        :param fit_settings_include: Whether to include this sensor in the fit [bool]
        :param fit_settings_weight: Weight for the sensor [dimensionless]
        :param time: Time vector [second]
        :param measured: Measured values [ampere]
        :param gaps_r: A 1D array containing the radial positions of the gaps [metre]
        :param gaps_z: A 1D array containing the vertical positions of the gaps [metre]
        :param gaps_d_r: A 1D array containing the radial widths of the gaps [metre]
        :param gaps_d_z: A 1D array containing the vertical heights of the gaps [metre]
        :param gaps_name: A list of the names of the gaps
        """
        ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...
    def calculate_sensor_values(
        cls,
        plasma: Plasma,
    ) -> None: ...

class Isoflux(DataTreeAccessor):
    def __new__(
        cls,
    ) -> Isoflux: ...
    def add_sensor(
        cls,
        name: str,
        fit_settings_comment: str,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        location_1_r: npt.NDArray[np.float64],
        location_1_z: npt.NDArray[np.float64],
        location_2_r: npt.NDArray[np.float64],
        location_2_z: npt.NDArray[np.float64],
        times_to_reconstruct: npt.NDArray[np.float64],
    ) -> None: ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     plasma: "Plasma",
    # ) -> None: ...

class IsofluxBoundary(DataTreeAccessor):
    def __new__(
        cls,
    ) -> IsofluxBoundary: ...
    def add_sensor(
        cls,
        name: str,
        fit_settings_comment: str,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        location_1_r: npt.NDArray[np.float64],
        location_1_z: npt.NDArray[np.float64],
        times_to_reconstruct: npt.NDArray[np.float64],
    ) -> None: ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     plasma: "Plasma",
    # ) -> None: ...

class StationaryPoint:
    def __new__(
        cls,
    ) -> StationaryPoint: ...
    def add_sensor(
        cls,
        name: str,
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        mag_axis_r: npt.NDArray[np.float64],
        mag_axis_z: npt.NDArray[np.float64],
        times_to_reconstruct: npt.NDArray[np.float64],
    ) -> None: ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

class Pressure(DataTreeAccessor):
    def __new__(
        cls,
    ) -> Pressure: ...
    def add_sensor(
        cls,
        name: str,
        geometry_r: float,
        geometry_z: float,
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None: ...
    def greens_with_pf_active(
        cls,
        pf_active: PfActive,
    ) -> None: ...
    def greens_with_pf_passive(
        cls,
        pf_passive: PfPassive,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

class Dialoop(DataTreeAccessor):
    def __new__(
        cls,
    ) -> Dialoop: ...
    def add_sensor(
        cls,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        fit_settings_comment: str,
        fit_settings_expected_value: float,
        fit_settings_include: bool,
        fit_settings_weight: float,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None: ...
    def calculate_sensor_values(
        cls,
        plasma: "Plasma",
    ) -> None: ...

class EfitPolynomial(DataTreeAccessor):
    def __new__(
        cls,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
        exact: bool = False,
        coefficients: npt.NDArray[np.float64] | None = None,
    ) -> EfitPolynomial:
        """
        :param n_dof: Number of degrees of freedom
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]; not used when `exact=True`
        :param exact: When `True` the coefficients are fixed to `coefficients` rather than fitted. With both `p_prime` and `ff_prime` exact, this is a forward solve:
            their shapes are fixed, and if there is a constraint besides the magnetic axis (normally the plasma current from a Rogowski coil) they share one fitted amplitude,
            which holds the plasma radially. The coefficients written to the equilibrium are the given ones times this amplitude
        :param coefficients: A 1D array of size [n_dof] with the fixed coefficients; required when `exact=True`
        """
        ...

class TensionedCubicBSpline(DataTreeAccessor):
    def __new__(
        cls,
        regularisations: npt.NDArray[np.float64],
        interior_knots: npt.NDArray[np.float64],
        interval_tensions: npt.NDArray[np.float64],
    ) -> TensionedCubicBSpline:
        """
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]
        :param interior_knots: A 1D array of size [n_interior_knots] with the interior knots [dimensionless]
        :param interval_tensions: A 1D array of size [n_intervals] with the interval tensions [dimensionless]
        """
        ...
