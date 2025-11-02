import numpy as np
import numpy.typing as npt

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
    def print_keys(self) -> None: ...

def solve_grad_shafranov(
    plasma: Plasma,
    coils: Coils,
    passives: Passives,
    bp_probes: BpProbes,
    flux_loops: FluxLoops,
    rogowski_coils: RogowskiCoils,
    isoflux: Isoflux,
    isoflux_boundary: IsofluxBoundary,
    magnetic_axis: MagneticAxis,
    times_to_reconstruct: npt.NDArray[np.float64],
    n_iter_max: int,
    n_iter_min: int,
    n_iter_no_vertical_feedback: int,
    gs_error: float,
    use_anderson_mixing: bool,
    anderson_mixing_from_previous_iter: float,
) -> None:
    """
    :param plasma: Plasma object, note this is mutated and contains the solution
    :param coils: Coils object
    :param passives: Passives object, note this is mutated and contains the solution
    :param bp_probes: BpProbes object, note this is mutated and contains the solution
    :param flux_loops: FluxLoops object, note this is mutated and contains the solution
    :param rogowski_coils: RogowskiCoils object, note this is mutated and contains the solution
    :param isoflux: Isoflux object, note this is mutated and contains the solution
    :param isoflux_boundary: IsofluxBoundary object, note this is mutated and contains the solution
    :param times_to_reconstruct: Times to reconstruct [second]
    :param n_iter_max: Maximum number of iterations
    :param n_iter_min: Minimum number of iterations
    :param n_iter_no_vertical_feedback: Number of iterations without vertical feedback
    :param gs_error: GS error
    :param use_anderson_mixing: Whether to use Anderson mixing
    :param anderson_mixing_from_previous_iter: Anderson mixing factor from the previous iteration [dimensionless]
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
    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]

    Note: the inputs are symmetrical
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
        :param d_r: PF coil radial widths [metre]
        :param d_z: PF coil vertical heights [metre]
        :param time: Experimental time [second]
        :param measured: Experimental PF coil current [ampere]
        :param angle1: Angle of the PF coil from the vertical ("DIII-D" parallelogram type) [radians]
        :param angle2: Angle of the PF coil from the horizontal ("DIII-D" parallelogram type) [radians]
        """
        ...
    def add_tf_coil(
        cls,
        time: npt.NDArray[np.float64],
        measured: npt.NDArray[np.float64],
    ) -> None:
        """
        :param time: Experimental time [second]
        :param measured: Experimental "rod" current [ampere]
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
        :param r: A 1D array containing the radial centroid location for each filament [metre]
        :param z: A 1D array containing the vertical centroid location for each filament [metre]
        :param d_r: A 1D array containing the radial the width of the filament from the centroid to either side of the filament, total_width=2.0*d_r [metre]
        :param d_z: A 1D array containing the radial the height of the filament from the centroid to either side of the filament, total_height=2.0*d_z [metre]
        :param angle_1: A 1D array containing the angle of the filament from the vertical ("DIII-D" parallelogram type) [radians]
        :param angle_2: A 1D array containing the angle of the filament from the horizontal ("DIII-D" parallelogram type) [radians]
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
        psi_n: npt.NDArray[np.float64],
        limit_pts_r: npt.NDArray[np.float64],
        limit_pts_z: npt.NDArray[np.float64],
        vessel_r: npt.NDArray[np.float64],
        vessel_z: npt.NDArray[np.float64],
        p_prime_source_function: "EfitPolynomial" | "LiuqePolynomial",
        ff_prime_source_function: "EfitPolynomial" | "LiuqePolynomial",
        initial_ip: float,
        initial_cur_r: float,
        initial_cur_z: float,
    ) -> Plasma:
        """
        :param n_r: Number of radial poitns [dimensionless]
        :param n_z: Number of vertical poitns [dimensionless]
        :param r_min: Minimum radius [metre]
        :param r_max: Maximum radius [metre]
        :param z_min: Minimum vertical position [metre]
        :param z_max: Maximum vertical position [metre]
        :param psi_n: 1D array for `psi_n`, which should go from [0.0, 1.0] [dimensionless]
        :param limit_pts_r: the limiter surfaces [metre]
        :param limit_pts_z: the limiter surfaces [metre]
        :param vessel_r: the vacuum vessel chamber, where the plasma can exist [metre]
        :param vessel_z: the vacuum vessel chamber, where the plasma can exist [metre]
        :param p_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
        :param ff_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
        :param initial_ip: Initial plasma current [ampere]
        :param initial_r: Initial major radius of the magnetic axis [metre]
        :param initial_z: Initial vertical position of the magnetic axis [metre]
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

class MagneticAxis:
    def __new__(
        cls,
    ) -> MagneticAxis: ...
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
    # def greens_with_coils(
    #     cls,
    #     coils: "Coils",
    # ) -> None: ...
    # def greens_with_passives(
    #     cls,
    #     coils: "Passives",
    # ) -> None: ...
    # def greens_with_plasma(
    #     cls,
    #     plasma: "Plasma",
    # ) -> None: ...
    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class Dialoop(DataTreeAccessor):
    def __new__(
        cls,
    ) -> Dialoop: ...
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
    # def greens_with_coils(
    #     cls,
    #     coils: "Coils",
    # ) -> None: ...
    # def greens_with_passives(
    #     cls,
    #     coils: "Passives",
    # ) -> None: ...
    # def greens_with_plasma(
    #     cls,
    #     plasma: "Plasma",
    # ) -> None: ...
    # def calculate_sensor_values(
    #     cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class EfitPolynomial(DataTreeAccessor):
    def __new__(
        cls,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
    ) -> EfitPolynomial:
        """
        :param n_dof: Number of degrees of freedom
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]
        """
        ...

class LiuqePolynomial(DataTreeAccessor):
    def __new__(
        cls,
        n_dof: int,
        regularisations: npt.NDArray[np.float64],
    ) -> LiuqePolynomial:
        """
        :param n_dof: Number of degrees of freedom
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values [dimensionless]
        """
        ...
