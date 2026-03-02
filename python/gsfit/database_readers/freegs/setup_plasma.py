import typing
from typing import TYPE_CHECKING

import freegs  # type: ignore
import gsfit_rs
import numpy as np
import numpy.typing as npt
from gsfit_rs import Plasma

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGS


def setup_plasma(
    self: "DatabaseReaderFreeGS",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
) -> Plasma:
    """
    This method initialises the Rust `Plasma` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Measured time vector
    :param freegsnke_eqs: List of FreeGS equilibrium objects, one for each time-slice

    **This method is specific to FreeGS.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initial plasma conditions
    initial_ip = settings["GSFIT_code_settings.json"]["initial_guess"]["ip"]
    initial_cur_r = settings["GSFIT_code_settings.json"]["initial_guess"]["r_cur"]
    initial_cur_z = settings["GSFIT_code_settings.json"]["initial_guess"]["z_cur"]

    if settings["source_function_p_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        p_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_p_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_p_prime.json"]["liuqe_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["liuqe_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        p_prime_source_function = gsfit_rs.LiuqePolynomial(n_dof, regularisations)
    elif settings["source_function_p_prime.json"]["method"] == "tensioned_cubic_b_spline":
        regularisations = np.array(settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        interior_knots = np.array(settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["interior_knots"])
        n_dof = len(interior_knots) + 4
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        interval_tensions = np.array(settings["source_function_p_prime.json"]["tensioned_cubic_b_spline"]["interval_tensions"])
        p_prime_source_function = gsfit_rs.TensionedCubicBSpline(regularisations, interior_knots, interval_tensions)
    else:
        raise ValueError(f"Unknown method for p_prime source function: {settings['source_function_p_prime.json']['method']}")

    if settings["source_function_ff_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        ff_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_ff_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["liuqe_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["liuqe_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        ff_prime_source_function = gsfit_rs.LiuqePolynomial(n_dof, regularisations)
    elif settings["source_function_ff_prime.json"]["method"] == "tensioned_cubic_b_spline":
        regularisations = np.array(settings["source_function_ff_prime.json"]["tensioned_cubic_b_spline"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        interior_knots = np.array(settings["source_function_ff_prime.json"]["tensioned_cubic_b_spline"]["interior_knots"])
        n_dof = len(interior_knots) + 4
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        interval_tensions = np.array(settings["source_function_ff_prime.json"]["tensioned_cubic_b_spline"]["interval_tensions"])
        ff_prime_source_function = gsfit_rs.TensionedCubicBSpline(regularisations, interior_knots, interval_tensions)
    else:
        raise ValueError(f"Unknown method for ff_prime source function: {settings['source_function_ff_prime.json']['method']}")

    # Grid size and shape
    n_r = settings["GSFIT_code_settings.json"]["grid"]["n_r"]
    n_z = settings["GSFIT_code_settings.json"]["grid"]["n_z"]
    r_min = settings["GSFIT_code_settings.json"]["grid"]["r_min"]
    r_max = settings["GSFIT_code_settings.json"]["grid"]["r_max"]
    z_min = settings["GSFIT_code_settings.json"]["grid"]["z_min"]
    z_max = settings["GSFIT_code_settings.json"]["grid"]["z_max"]

    # Normalised poloidal flux grid
    n_psi_n = settings["GSFIT_code_settings.json"]["n_psi_n"]
    psi_n = np.linspace(0.0, 1.0, n_psi_n).astype(np.float64)

    # Limiter
    limit_pts_r = freegs_eqs[0].tokamak.wall.R
    limit_pts_z = freegs_eqs[0].tokamak.wall.Z

    # Vacuum vessel where the plasma is allowed to be
    vessel_r = limit_pts_r
    vessel_z = limit_pts_z

    # Initialise the Plasma Rust class
    plasma = Plasma(
        n_r,
        n_z,
        r_min,
        r_max,
        z_min,
        z_max,
        psi_n,  # BUXTON: perhaps better to send in `n_psi_n`
        limit_pts_r,
        limit_pts_z,
        vessel_r,
        vessel_z,
        p_prime_source_function,
        ff_prime_source_function,
        initial_ip,
        initial_cur_r,
        initial_cur_z,
    )

    return plasma
