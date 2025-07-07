import numpy as np
import numpy.typing as npt

def solve_inverse_problem(
    plasma: Plasma,
    coils: Coils,
    passives: Passives,
    bp_probes: BpProbes,
    flux_loops: FluxLoops,
    rogowski_coils: RogowskiCoils,
    isoflux: Isoflux,
    isoflux_boundary: IsofluxBoundary,
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
) -> npt.NDArray[np.float64]:
    """
    :param r: (by convention) Sensor radial positions [meter]
    :param z: (by convention) Sensor vertical positions [meter]
    :param r_prime: (by convention) Current source radial positions [meter]
    :param z_prime: (by convention) Current source vertical positions [meter]

    Note: the inputs are symmetrical
    """
    ...

def greens_magnetic_field_py(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param r: Sensor radial positions [meter]
    :param z: Sensor vertical positions [meter]
    :param r_prime: Current source radial positions [meter]
    :param z_prime: Current source vertical positions [meter]

    Note: the inputs are not symmetrical!
    i.e. you can't change (r, z) for (r_prime, z_prime)
    """
    ...

def mutual_inductance_finite_size_to_finite_size(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64],
    d_z: npt.NDArray[np.float64],
    angle1: npt.NDArray[np.float64],
    angle2: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r_prime: npt.NDArray[np.float64],
    d_z_prime: npt.NDArray[np.float64],
    angle1_prime: npt.NDArray[np.float64],
    angle2_prime: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    :param r: Sensor radial positions [meter]
    :param z: Sensor vertical positions [meter]
    :param r_prime: Current source radial positions [meter]
    :param z_prime: Current source vertical positions [meter]

    Note: the inputs are not symmetrical!
    i.e. you can't change (r, z) for (r_prime, z_prime)
    """
    ...

class Coils:
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
        :param r: PF coil radial positions [meter]
        :param z: PF coil vertical positions [meter]
        :param d_r: PF coil radial widths [meter]
        :param d_z: PF coil vertical heights [meter]
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
    def print_keys(cls) -> None: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...

class Passives:
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
        :param r: A 1D array containing the radial centroid location for each filament [meter]
        :param z: A 1D array containing the vertical centroid location for each filament [meter]
        :param d_r: A 1D array containing the radial the width of the filament from the centroid to either side of the filament, total_width=2.0*d_r [meter]
        :param d_z: A 1D array containing the radial the height of the filament from the centroid to either side of the filament, total_height=2.0*d_z [meter]
        :param angle_1: A 1D array containing the angle of the filament from the vertical ("DIII-D" parallelogram type) [radians]
        :param angle_2: A 1D array containing the angle of the filament from the horizontal ("DIII-D" parallelogram type) [radians]
        :param resistivity: Resistivity of this passive (same for all filaments) [ohm * meter]
        :param current_distribution_type: "constant_current_density" or "eig"
        :param n_dof: number of degrees of freedom; if current_distribution_type=="constant_current_density", then n_dof=1; if current_distribution_type=="eig", then n_dof is the number of eigenvalues
        :param regularisations: A 2D array of size [n_regularisations, n_dof] with the regularisation values
        :param regularisations_weight: A 1D array of size [n_regularisations] which contains the regularisation weights
        """
        ...
    def print_keys(cls) -> None: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...

class Plasma:
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
    ) -> Plasma:
        """
        :param n_r: Number of radial poitns [dimensionless]
        :param n_z: Number of vertical poitns [dimensionless]
        :param r_min: Minimum radius [meter]
        :param r_max: Maximum radius [meter]
        :param z_min: Minimum vertical position [meter]
        :param z_max: Maximum vertical position [meter]
        :param psi_n: 1D array for `psi_n`, which should go from [0.0, 1.0] [dimensionless]
        :param limit_pts_r: the limiter surfaces [meter]
        :param limit_pts_z: the limiter surfaces [meter]
        :param vessel_r: the vacuum vessel chamber, where the plasma can exist [meter]
        :param vessel_z: the vacuum vessel chamber, where the plasma can exist [meter]
        :param p_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
        :param ff_prime_source_function: `p_prime` source function, needs to be constructed from `gsfit_rs.<source_function_name>`
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

class BpProbes:
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
        :param geometry_r: Radial position of the sensor geometry [meter]
        :param geometry_z: Vertical position of the sensor geometry [meter]
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

class FluxLoops:
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
        :param geometry_r: Radial position of the sensor geometry [meter]
        :param geometry_z: Vertical position of the sensor geometry [meter]
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

class RogowskiCoils:
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
        :param r: 1D array containing the radial positions of the sensor geometry [meter]
        :param z: 1D array containing the vertical positions of the sensor geometry [meter]
        :param fit_settings_comment: Comment for the fit settings, used for debugging
        :param fit_settings_expected_value: Expected value for the fit settings, used for normalisation [ampere]
        :param fit_settings_include: Whether to include this sensor in the fit [bool]
        :param fit_settings_weight: Weight for the sensor [dimensionless]
        :param time: Time vector [second]
        :param measured: Measured values [ampere]
        :param gaps_r: A 1D array containing the radial positions of the gaps [meter]
        :param gaps_z: A 1D array containing the vertical positions of the gaps [meter]
        :param gaps_d_r: A 1D array containing the radial widths of the gaps [meter]
        :param gaps_d_z: A 1D array containing the vertical heights of the gaps [meter]
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

class Isoflux:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class IsofluxBoundary:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class Pressure:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...
    # def calculate_sensor_values(
    # cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class Dialoop:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...
    # def calculate_sensor_values(
    #     cls,
    #     coils: "Coils",
    #     passives: "Passives",
    #     plasma: "Plasma",
    # ) -> None: ...

class EfitPolynomial:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...

class LiuqePolynomial:
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
    def get_array1(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array2(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_array3(cls, keys: list[str]) -> npt.NDArray[np.float64]: ...
    def get_bool(cls, keys: list[str]) -> bool: ...
    def get_f64(cls, keys: list[str]) -> float: ...
    def get_usize(cls, keys: list[str]) -> int: ...
    def get_vec_bool(cls, keys: list[str]) -> list[bool]: ...
    def get_vec_usize(cls, keys: list[str]) -> list[int]: ...
    def keys(cls, key_path: list[str] | None = None) -> list[str]: ...
    def print_keys(cls) -> None: ...
