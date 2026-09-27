from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .imas import Equilibrium
from .imas import Magnetics as MagneticsIds
from .imas import Path
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
    wall: Wall,
    tf: Tf,
    coils: Coils,
    passives: Passives,
    bp_probes: BpProbes,
    flux_loops: FluxLoops,
    rogowski_coils: RogowskiCoils,
    isoflux: Isoflux,
    isoflux_boundary: IsofluxBoundary,
    pressure_sensors: Pressure,
    stationary_point: StationaryPoint,
    dialoop: Dialoop,
    pulse_schedule: PulseSchedule | None = None,
) -> None:
    """
    :param plasma: Plasma object, note this is mutated and contains the solution
    :param wall: Wall object, supplying the limiter points and the vacuum vessel contour
    :param tf: Tf object, supplying the reference major radius and the vacuum toroidal field
    :param coils: Coils object
    :param passives: Passives object, note this is mutated and contains the solution
    :param bp_probes: BpProbes object, note this is mutated and contains the solution
    :param flux_loops: FluxLoops object, note this is mutated and contains the solution
    :param rogowski_coils: RogowskiCoils object, note this is mutated and contains the solution
    :param isoflux: Isoflux object, note this is mutated and contains the solution
    :param isoflux_boundary: IsofluxBoundary object, note this is mutated and contains the solution
    :param pressure_sensors: Pressure object, note this is mutated and contains the solution
    :param stationary_point: StationaryPoint object, note this is mutated and contains the solution
    :param dialoop: Dialoop object, note this is mutated and contains the solution
    :param pulse_schedule: (optional) PulseSchedule object, supplying the gap definitions. Its gaps are copied onto every
        time-slice's `boundary/gap` and their values calculated. `None`, the default, gives an equilibrium with no gaps

    The times to reconstruct are read from `plasma`, which was built with one equilibrium
    time-slice per time.
    """
    ...

def solve_circuit_equations(
    coils: Coils,
    passives: Passives,
    times_to_solve: npt.NDArray[np.float64],
    adaptive_time_stepping: bool,
) -> None:
    """
    Solves the circuit equations

    Note: the adaptive time-stepping can produce a lot of simulated time-points,
    so `adaptive_time_stepping = False` should be used to avoid unwieldly large outputs

    :param coils: Coils data structure containing either current or voltage waveforms, note this is mutated and contains the solution
    :param passives: Passives object, note this is mutated and contains the solution
    :param times_to_solve: Times to solve the circuit equations at [second]
    :param adaptive_time_stepping: The calculation always uses adaptive time stepping, this flag interpolates results onto `times_to_solve`
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

class Coils(DataTreeAccessor):
    """Coils class to hold PF and TF coils data"""
    def __new__(
        cls,
    ) -> Coils: ...
    def add_pf_coil(
        cls,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        d_r: npt.NDArray[np.float64],
        d_z: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None:
        """
        :param name: PF coil name
        :param r: PF coil radial positions [metre]
        :param z: PF coil vertical positions [metre]
        :param d_r: PF coil radial widths (note, area = d_r * d_z) [metre]
        :param d_z: PF coil vertical heights [metre]
        :param time: Experimental time [second]
        :param measured: Experimental PF coil current [ampere]
        :param angle1: Angle of the PF coil from the vertical ("DIII-D" parallelogram type) [radians]
        :param angle2: Angle of the PF coil from the horizontal ("DIII-D" parallelogram type) [radians]
        """
        ...
    def greens_with_self(
        cls,
    ) -> None:
        """
        Calculate the coil self and mutual inductance matrices, [henry]
        """
        ...
    def calculate_coil_resistance(
        cls,
    ) -> None:
        """
        Calculate the coil resistance matrix, a diagonal matrix, [ohm]
        """
        ...
    def set_pf_voltage_controlled(
        cls,
        name: str,
        time: npt.NDArray[np.float64],
        measured_voltage: npt.NDArray[np.float64],
    ) -> None:
        """
        :param name: PF coil name
        :param time: Experimental time [second]
        :param measured_voltage: Experimental PF coil voltage [volt]
        """
        ...

class Passives(DataTreeAccessor):
    """Contains the toroidally conducting strucutres, such as the vacuum vessel and passive plates.
    `passives` also contains the degrees of freedom allowed for the passive conductors
    """
    def __new__(
        cls,
    ) -> Passives: ...
    def add_passive(
        cls,
        name: str,
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        d_r: npt.NDArray[np.float64],
        d_z: npt.NDArray[np.float64],
        angle_1: npt.NDArray[np.float64],
        angle_2: npt.NDArray[np.float64],
        resistivity: float,
        current_distribution_type: str,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
        regularisations_weight: npt.NDArray[np.float64],
    ) -> None:
        """
        :param name: Passive name
        :param r: Radial centroid location for each filament [metre]
        :param z: Vertical centroid location for each filament [metre]
        :param d_r: Radial the widths of the filaments (note, area = d_r * d_z) [metre]
        :param d_z: Radial the heights of the filaments [metre]
        :param angle_1: Angle of the filament from the vertical ("DIII-D" parallelogram type) [radians]
        :param angle_2: Angle of the filament from the horizontal ("DIII-D" parallelogram type) [radians]
        :param resistivity: Resistivity of this passive (same for all filaments) [ohm * metre]
        :param current_distribution_type: "constant_current_density" or "eig"
        :param n_dof: number of degrees of freedom; if current_distribution_type=="constant_current_density", then n_dof=1; if current_distribution_type=="eig", then n_dof is the number of eigenvalues
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values
        :param regularisations_weight: A 1D array of size [n_regularisations] which contains the regularisation weights
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
    ) -> None: ...
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
    """

    def __new__(cls) -> Tf:
        """Construct an empty toroidal field, ready for `set_r0` and `set_b_field_phi_vacuum_r`."""
        ...
    def set_r0(self, r0: float) -> None:
        """
        Set `tf/r0`, the reference major radius.

        :param r0: reference major radius the vacuum toroidal field is quoted at [metre]

        The solver copies this onto `equilibrium/vacuum_toroidal_field/r0`, so that the two
        IDSs cannot disagree.
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
        times: `solve_grad_shafranov` interpolates it itself.
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

    The order units are added in is part of the contract: `unit(0)` is the vacuum vessel
    contour, and the solver uses that one, and only that one, as the region the plasma is
    allowed to occupy. Every unit contributes candidate limit points.

    Read it back through `wall_ids` and a path from `gsfit_rs.imas.wall_paths`.
    """

    def __new__(cls) -> Wall:
        """Construct an empty wall, ready for `add_limiter_unit` to be called."""
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

    A gap is a reference point and a direction. `solve_grad_shafranov` copies the definitions onto
    every time-slice's `equilibrium/time_slice(itime)/boundary/gap`, and calculates each value: the
    distance from the reference point to the plasma boundary along that direction.

    The angle is stored in the equilibrium IDS's convention, clockwise from `grad(R)` in the usual
    plot with `R` to the right and `Z` upwards, so the gap points along
    `(cos(angle), -sin(angle))`. A database reader holding a counter-clockwise angle must convert it.

    Read it back through `pulse_schedule_ids` and a path from `gsfit_rs.imas.pulse_schedule_paths`.
    """

    def __new__(cls) -> PulseSchedule:
        """Construct an empty pulse schedule, with no gaps, ready for `add_gap` to be called."""
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

class Magnetics:
    """The machine's magnetic sensors, stored as an IMAS `magnetics` IDS.

    Nothing is filled yet: this is constructed empty.

    Read it back through `magnetics_ids` and a path from `gsfit_rs.imas.magnetics_paths`.
    """

    def __new__(cls) -> Magnetics:
        """Construct an empty set of magnetic sensors."""
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

    Solves the analytic equilibrium and produces a `Coils` object whose PF-coil
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
    def get_coils(
        self,
        coil_regularisation_weight: float,
        control_points: npt.NDArray[np.float64],
        wall: Wall,
    ) -> Coils:
        """
        Fit the PF-coil currents on the (R, Z) grid boundary so that, with the plasma's own field, they reproduce
        the analytic flux at `control_points`. Then store the free-boundary equilibrium in `equilibrium_ids` and
        post-process it. Requires `solve` first.

        The IDS holds the total flux, plasma plus coils, which is valid in the vacuum as well as inside the plasma.
        For a diverted plasma the magnetic axis, the X-points, the boundary flux and the plasma mask are found in
        that flux, as the Grad-Shafranov solver finds them; for a limited one they are the analytic ones.

        :param coil_regularisation_weight: Tikhonov weight on the coil-current magnitude [dimensionless]
        :param control_points: (n_control_point, 2) array of (r, z) points where the flux is matched [metre]
        :param wall: The wall: its limiter and vacuum vessel bound a diverted plasma, and the post-processor traces
            the scrape-off layer legs up to it
        :return: One single-filament PF coil per boundary node, carrying the fitted current
        """
        ...
    def get_sensor_values(
        self,
        bp_probes: BpProbes,
        flux_loops: FluxLoops,
        coils: Coils,
    ) -> None:
        """
        Fill each BP probe (tesla) and flux loop (weber) `experimental` value with the field the analytic plasma
        current plus `coils` produce at that sensor. Mutates `bp_probes` and `flux_loops` in place. Requires `solve` first.

        Each coil filament carries its coil's current at the first time of its experimental timebase.

        :param bp_probes: BpProbes with geometry already set; their measurements are overwritten
        :param flux_loops: FluxLoops with geometry already set; their measurements are overwritten
        :param coils: The PF coils, typically the ones `get_coils` returned
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
        """The free-boundary equilibrium IDS, read with `gsfit_rs.imas.equilibrium_paths`. Empty until `get_coils` has run.

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
        coils: Coils,
        passives: Passives,
        plasma: Plasma,
    ) -> None:
        """
        Calculate the sensor values from the coils, passives and plasma.
        Mutates self

        :param coils: Coils object
        :param passives: Passives object
        :param plasma: Plasma object
        """
        ...
    def calculate_sensor_values_vacuum(
        cls,
        coils: Coils,
        passives: Passives,
    ) -> None:
        """
        Calculate the sensor values from the coils and passives.
        Mutates self

        :param coils: Coils object
        :param passives: Passives object
        """
        ...
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
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
        coils: Coils,
        passives: Passives,
        plasma: Plasma,
    ) -> None: ...
    def calculate_sensor_values_vacuum(
        cls,
        coils: Coils,
        passives: Passives,
    ) -> None:
        """
        Calculate the sensor values from the coils and passives.
        Mutates self

        :param coils: Coils object
        :param passives: Passives object
        """
        ...
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...
    def calculate_sensor_values(
        cls,
        coils: Coils,
        passives: Passives,
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
    ) -> None: ...
    def greens_with_plasma(
        cls,
        plasma: Plasma,
    ) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
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
    def greens_with_coils(
        cls,
        coils: Coils,
    ) -> None: ...
    def greens_with_passives(
        cls,
        passives: Passives,
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
