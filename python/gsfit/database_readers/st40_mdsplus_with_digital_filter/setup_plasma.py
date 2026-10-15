import dataclasses
from typing import TYPE_CHECKING

import gsfit_rs
import numpy as np
import numpy.typing as npt
from gsfit_rs import Plasma

from ...settings_schema.python.equilibrium.equilibrium import EquilibriumSettings
from ...settings_schema.python.equilibrium.source_function_ff_prime import SourceFunctionFfPrime
from ...settings_schema.python.equilibrium.source_function_p_prime import SourceFunctionPPrime
from ...tensioned_cubic_splines_regularisations import make_tensioned_cubic_b_spline_regularisations

if TYPE_CHECKING:
    from . import DatabaseReader


def make_source_function(
    source_function_settings: SourceFunctionPPrime | SourceFunctionFfPrime,
    settings_file_name: str,
) -> gsfit_rs.EfitPolynomial | gsfit_rs.TensionedCubicBSpline:
    """
    Make a source function, p' or ff', from its settings.

    :param source_function_settings: The source function's settings, read from `equilibrium/source_function_p_prime.xml` or
        `equilibrium/source_function_ff_prime.xml`
    :param settings_file_name: The name of the file the settings were read from, e.g. "source_function_p_prime.xml", for the error messages
    :return: The source function
    """

    match source_function_settings.method:
        case "efit_polynomial":
            efit_polynomial_settings = source_function_settings.efit_polynomial
            if efit_polynomial_settings is None:
                raise ValueError(f"`{settings_file_name}` has `method` efit_polynomial, but no `efit_polynomial` element")
            n_dof: int = efit_polynomial_settings.n_dof
            regularisations: npt.NDArray[np.float64] = np.array(efit_polynomial_settings.regularisations, dtype=np.float64)
            # No regularisations is a matrix with no rows, which the settings give as shape (0, 0).
            # The source function takes shape (n_regularisations, n_dof), so it needs shape (0, n_dof)
            if regularisations.shape[0] == 0:
                regularisations = np.zeros((0, n_dof), dtype=np.float64)
            # `exact` fixes the coefficients to `coefficients`, rather than fitting them. Both are absent when the coefficients are fitted
            exact: bool = efit_polynomial_settings.exact if efit_polynomial_settings.exact is not None else False
            coefficients: list[float] = efit_polynomial_settings.coefficients if efit_polynomial_settings.coefficients is not None else []
            return gsfit_rs.EfitPolynomial(
                n_dof,
                regularisations,
                exact=exact,
                coefficients=np.array(coefficients, dtype=np.float64),
            )
        case "tensioned_cubic_b_spline":
            tensioned_cubic_b_spline_settings = source_function_settings.tensioned_cubic_b_spline
            if tensioned_cubic_b_spline_settings is None:
                raise ValueError(f"`{settings_file_name}` has `method` tensioned_cubic_b_spline, but no `tensioned_cubic_b_spline` element")
            interior_knots: npt.NDArray[np.float64] = np.array(tensioned_cubic_b_spline_settings.interior_knots, dtype=np.float64)
            interval_tensions: npt.NDArray[np.float64] = np.array(tensioned_cubic_b_spline_settings.interval_tensions, dtype=np.float64)
            n_dof = len(interior_knots) + 4
            # `make_cubic_spline_regularisations` is false when absent
            if tensioned_cubic_b_spline_settings.make_cubic_spline_regularisations:
                regularisation_builder = tensioned_cubic_b_spline_settings.regularisation_builder
                if regularisation_builder is None:
                    raise ValueError(f"`{settings_file_name}` has `make_cubic_spline_regularisations` true, but no `regularisation_builder` element")
                # Build the regularisations from settings (see `tensioned_cubic_splines_regularisations.py`).
                # It takes them as a dictionary, whose keys are the names of the `regularisation_builder` element's fields
                regularisations = make_tensioned_cubic_b_spline_regularisations(
                    interior_knots,
                    interval_tensions,
                    dataclasses.asdict(regularisation_builder),
                )
            else:
                if tensioned_cubic_b_spline_settings.regularisations is None:
                    regularisations = np.zeros((0, n_dof), dtype=np.float64)
                else:
                    regularisations = np.array(tensioned_cubic_b_spline_settings.regularisations, dtype=np.float64)
                # No regularisations is a matrix with no rows, which the settings give as shape (0, 0).
                # The source function takes shape (n_regularisations, n_dof), so it needs shape (0, n_dof)
                if regularisations.shape[0] == 0:
                    regularisations = np.zeros((0, n_dof), dtype=np.float64)
            return gsfit_rs.TensionedCubicBSpline(regularisations, interior_knots, interval_tensions)
        case _:
            raise ValueError(f"`{settings_file_name}` has an unknown `method`: {source_function_settings.method}")


def setup_plasma(
    self: "DatabaseReader",
    pulseNo: int,
    settings: EquilibriumSettings,
    times_to_reconstruct: npt.NDArray[np.float64],
) -> Plasma:
    """
    This method initialises the Rust `Plasma` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `EquilibriumSettings` object, read from `equilibrium/equilibrium.xml` and the files it links to, which holds the grid, the
        initial guess, the numerical settings of the Grad-Shafranov solve, and the source functions
    :param times_to_reconstruct: Times the equilibrium will be solved at [second]

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initial plasma conditions
    initial_guess_ip = settings.initial_guess.ip
    initial_guess_cur_r = settings.initial_guess.cur_r
    initial_guess_cur_z = settings.initial_guess.cur_z
    initial_guess_minor_radius = settings.initial_guess.minor_radius
    initial_guess_elongation = settings.initial_guess.elongation

    # Numerical settings the Grad-Shafranov solve is run with
    n_iter_max = settings.numerics.n_iter_max
    n_iter_min = settings.numerics.n_iter_min
    grad_shafranov_deviation_tolerance = settings.numerics.grad_shafranov_deviation_tolerance
    nonlinear_solver_settings = settings.numerics.nonlinear_solver
    nonlinear_solver_method = nonlinear_solver_settings.method
    picard_n_iter_no_vertical_feedback = nonlinear_solver_settings.picard.n_iter_no_vertical_feedback
    picard_apply_anderson_mixing = nonlinear_solver_settings.picard.apply_anderson_mixing
    picard_anderson_n_history = nonlinear_solver_settings.picard.anderson_n_history
    picard_anderson_mixing = nonlinear_solver_settings.picard.anderson_mixing
    newton_krylov_picard_handover = nonlinear_solver_settings.newton_krylov.picard_handover
    newton_krylov_n_krylov_max = nonlinear_solver_settings.newton_krylov.n_krylov_max
    newton_krylov_krylov_tolerance = nonlinear_solver_settings.newton_krylov.krylov_tolerance
    newton_krylov_finite_difference_step = nonlinear_solver_settings.newton_krylov.finite_difference_step
    newton_krylov_verbose = nonlinear_solver_settings.newton_krylov.verbose
    newton_picard_n_basis_max = nonlinear_solver_settings.newton_picard.n_basis_max
    newton_picard_contraction_threshold = nonlinear_solver_settings.newton_picard.contraction_threshold
    newton_picard_finite_difference_step = nonlinear_solver_settings.newton_picard.finite_difference_step
    newton_picard_verbose = nonlinear_solver_settings.newton_picard.verbose

    # Set the source functions
    p_prime_source_function = make_source_function(settings.source_function_p_prime, "source_function_p_prime.xml")
    ff_prime_source_function = make_source_function(settings.source_function_ff_prime, "source_function_ff_prime.xml")

    # Grid size and shape
    n_r = settings.grid.n_r
    n_z = settings.grid.n_z
    r_min = settings.grid.r_min
    r_max = settings.grid.r_max
    z_min = settings.grid.z_min
    z_max = settings.grid.z_max

    # Normalised poloidal flux grid
    n_psi_n = settings.grid.n_psi_n
    psi_n = np.linspace(0.0, 1.0, n_psi_n).astype(np.float64)

    # Initialise the Plasma Rust class
    plasma = Plasma(
        n_r,
        n_z,
        r_min,
        r_max,
        z_min,
        z_max,
        psi_n,  # BUXTON: perhaps better to send in `n_psi_n`
        p_prime_source_function,
        ff_prime_source_function,
        initial_guess_ip,
        initial_guess_cur_r,
        initial_guess_cur_z,
        initial_guess_minor_radius,
        initial_guess_elongation,
        n_iter_max,
        n_iter_min,
        grad_shafranov_deviation_tolerance,
        nonlinear_solver_method,
        picard_n_iter_no_vertical_feedback,
        picard_apply_anderson_mixing,
        picard_anderson_n_history,
        picard_anderson_mixing,
        newton_krylov_picard_handover,
        newton_krylov_n_krylov_max,
        newton_krylov_krylov_tolerance,
        newton_krylov_finite_difference_step,
        newton_krylov_verbose,
        newton_picard_n_basis_max,
        newton_picard_contraction_threshold,
        newton_picard_finite_difference_step,
        newton_picard_verbose,
        times_to_reconstruct,
    )

    return plasma
