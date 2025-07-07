import typing
from typing import TYPE_CHECKING

import freegsnke  # type: ignore
import gsfit_rs
import numpy as np
import numpy.typing as npt
from gsfit_rs import Plasma

if TYPE_CHECKING:
    from . import DatabaseReaderFreeGSNKE


def setup_plasma(
    self: "DatabaseReaderFreeGSNKE",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[freegsnke.machine_update.Machine],
) -> Plasma:
    """
    This method initialises the Rust `Plasma` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to FreeGSNKE.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Set the source functions types
    p_prime_source_function: gsfit_rs.EfitPolynomial | gsfit_rs.LiuqePolynomial
    ff_prime_source_function: gsfit_rs.EfitPolynomial | gsfit_rs.LiuqePolynomial

    # p_prime source function
    p_prime_source_function: gsfit_rs.EfitPolynomial | gsfit_rs.LiuqePolynomial
    if settings["source_function_p_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        p_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_p_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        p_prime_source_function = gsfit_rs.LiuqePolynomial(n_dof, regularisations)
    else:
        raise ValueError(f"Unknown method for p_prime source function: {settings['source_function_p_prime.json']['method']}")

    # ff_prime source function
    if settings["source_function_ff_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        ff_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_ff_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        ff_prime_source_function = gsfit_rs.LiuqePolynomial(n_dof, regularisations)

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
    freegsnke_tokamak = freegsnke_eqs[0].tokamak  # assume static data does not change in time
    limit_pts_r = np.array(freegsnke_tokamak.limiter.R)
    limit_pts_z = np.array(freegsnke_tokamak.limiter.Z)

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
    )

    return plasma
