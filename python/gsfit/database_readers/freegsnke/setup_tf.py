import typing
from typing import TYPE_CHECKING

from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
import numpy as np
import numpy.typing as npt
from gsfit_rs import Tf
from scipy.constants import mu_0

if TYPE_CHECKING:
    from . import DatabaseReader

# There is no toroidal field circuit in the synthetic data, so a constant rod current stands in
SYNTHETIC_ROD_CURRENT = 1.0e6  # [ampere]


def setup_tf(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    time: npt.NDArray[np.float64],
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
) -> Tf:
    """
    This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param time: Times the equilibrium will be solved at [second]
    :param freegsnke_eqs: The forward-solved equilibria the synthetic measurements come from

    **This method is specific to synthetic data from FreeGSNKE.**

    Two nodes are filled:
    1. `tf/r0`, the reference major radius the vacuum toroidal field is quoted at
    2. `tf/b_field_phi_vacuum_r`, the vacuum field times major radius, on the **experimental**
       timebase. `solve_grad_shafranov` interpolates it onto the reconstruction times itself, so
       do not interpolate it here.

    `b_field_phi_vacuum_r` is signed: positive means counter-clockwise viewed from above.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Tf Rust class
    tf = Tf()

    tf.set_r0(settings["GSFIT_code_settings.json"]["vacuum_toroidal_field_reference_radius"])

    # `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`, the poloidal-current function in vacuum
    f_vac = mu_0 * SYNTHETIC_ROD_CURRENT / (2.0 * np.pi)

    # Constant in time, so two points are enough to interpolate from
    tf.set_b_field_phi_vacuum_r(
        time=np.array([0.0, 1.0]),
        data=np.array([f_vac, f_vac]),
    )

    return tf
