import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import EfitPolynomial
from gsfit_rs import Plasma
from gsfit_rs import TensionedCubicBSpline

from .mock_get_data import MockGetData

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_plasma(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Plasma:
    """
    This method initialises the Rust `Plasma` class.

    :param pulseNo: Pulse number, used to select which mocked MDSplus tree to read
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to a mock of ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initial plasma conditions
    initial_ip = settings["GSFIT_code_settings.json"]["initial_guess"]["ip"]
    initial_cur_r = settings["GSFIT_code_settings.json"]["initial_guess"]["r_cur"]
    initial_cur_z = settings["GSFIT_code_settings.json"]["initial_guess"]["z_cur"]

    # Set the source functions types
    p_prime_source_function = build_source_function(settings["source_function_p_prime.json"])
    ff_prime_source_function = build_source_function(settings["source_function_ff_prime.json"])

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
    elmag = MockGetData.from_workflow(settings, pulseNo, "elmag")
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
        psi_n,
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


def build_source_function(source_settings: dict[str, typing.Any]) -> EfitPolynomial | TensionedCubicBSpline:
    """
    Construct the Rust source function (`p_prime` or `ff_prime`) requested by the settings.

    :param source_settings: Contents of either `source_function_p_prime.json` or `source_function_ff_prime.json`
    """

    if source_settings["method"] == "efit_polynomial":
        n_dof = source_settings["efit_polynomial"]["n_dof"]
        regularisations = np.array(source_settings["efit_polynomial"]["regularizations"])
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        return EfitPolynomial(n_dof, regularisations)

    if source_settings["method"] == "tensioned_cubic_b_spline":
        regularisations = np.array(source_settings["tensioned_cubic_b_spline"]["regularizations"])
        interior_knots = np.array(source_settings["tensioned_cubic_b_spline"]["interior_knots"])
        n_dof = len(interior_knots) + 4
        # If `regularisations` is [[]] in the json file, will be interpreted by numpy as having size (1, 0).
        # Which would be interpreted as (n_regularisations, n_dof). So it would cause an error
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        interval_tensions = np.array(source_settings["tensioned_cubic_b_spline"]["interval_tensions"])
        return TensionedCubicBSpline(regularisations, interior_knots, interval_tensions)

    raise ValueError(f"Unknown source function method: {source_settings['method']}")
