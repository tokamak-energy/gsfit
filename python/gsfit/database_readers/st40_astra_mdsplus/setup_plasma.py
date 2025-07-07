import typing
from typing import TYPE_CHECKING

import gsfit_rs
import numpy as np
import numpy.typing as npt
from gsfit_rs import Plasma
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_plasma(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Plasma:
    """
    This method initialises the Rust `Plasma` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Set the source functions types
    p_prime_source_function: gsfit_rs.EfitPolynomial | gsfit_rs.LiuqePolynomial
    ff_prime_source_function: gsfit_rs.EfitPolynomial | gsfit_rs.LiuqePolynomial

    if settings["source_function_p_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"])
        p_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_p_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_p_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_p_prime.json"]["efit_polynomial"]["regularizations"])
        p_prime_source_function = gsfit_rs.LiuqePolynomial(n_dof, regularisations)
    else:
        raise ValueError(f"Unknown method for p_prime source function: {settings['source_function_p_prime.json']['method']}")

    if settings["source_function_ff_prime.json"]["method"] == "efit_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"])
        ff_prime_source_function = gsfit_rs.EfitPolynomial(n_dof, regularisations)
    elif settings["source_function_ff_prime.json"]["method"] == "liuqe_polynomial":
        n_dof = settings["source_function_ff_prime.json"]["efit_polynomial"]["n_dof"]
        regularisations = np.array(settings["source_function_ff_prime.json"]["efit_polynomial"]["regularizations"])
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
    elmag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["elmag_run_name"]
    elmag = GetData(12050, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)
    limit_pts_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.R"))
    limit_pts_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.Z"))

    # Vacuum vessel where the plasma is allowed to be
    vessel_r = limit_pts_r
    vessel_z = limit_pts_z

    # Add lower MC tiles
    limit_pts_r = np.append(limit_pts_r, 0.7103)
    limit_pts_z = np.append(limit_pts_z, -0.3131)
    # Add upper MC tiles
    limit_pts_r = np.append(limit_pts_r, 0.7103)
    limit_pts_z = np.append(limit_pts_z, 0.3031)

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
