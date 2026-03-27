import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Passives

from .astra_passives_reader import astra_passives_reader

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_passives(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Passives:
    """
    This method initialises the Rust `Passives` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ASTRA simulations using fires.dat.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Passives Rust class
    passives = Passives()

    # Read passive data from fires.dat
    passives_data = astra_passives_reader()

    for passive_name, data in passives_data.items():
        if passive_name == "ERINGT" or passive_name == "ERINGB":  #  or passive_name == "MCTCASE" or passive_name == "MCBCASE"
            continue  # skip to the next passive

        if passive_name == "GASBFLB" or passive_name == "GASBFLT":
            # In our set-up we have the GASBFL's as part of the IVC
            continue

        if passive_name == "IVC":
            current_distribution_type = "eig"
            n_dof = settings["passive_dof_regularisation.json"]["IVC"]["n_dof"]
            regularisations = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations"])
            regularisations_weight = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations_weight"])

            # Add the GASBFL's to the IVC
            r = np.concatenate((data["r"], passives_data["GASBFLB"]["r"], passives_data["GASBFLT"]["r"]))
            z = np.concatenate((data["z"], passives_data["GASBFLB"]["z"], passives_data["GASBFLT"]["z"]))
            d_r = np.concatenate((data["d_r"], passives_data["GASBFLB"]["d_r"], passives_data["GASBFLT"]["d_r"]))
            d_z = np.concatenate((data["d_z"], passives_data["GASBFLB"]["d_z"], passives_data["GASBFLT"]["d_z"]))
            angle_1 = np.concatenate((data["angle_1"], passives_data["GASBFLB"]["angle_1"], passives_data["GASBFLT"]["angle_1"]))
            angle_2 = np.concatenate((data["angle_2"], passives_data["GASBFLB"]["angle_2"], passives_data["GASBFLT"]["angle_2"]))

            passives.add_passive(
                name=passive_name,
                r=r,
                z=z,
                d_r=d_r,
                d_z=d_z,
                angle_1=angle_1,
                angle_2=angle_2,
                resistivity=data["resistivity"],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )
        elif passive_name == "OVC":
            current_distribution_type = "constant_current_density"
            n_dof = 1
            regularisations = np.array([[1.0]])
            regularisations_weight = np.array([0.001])  # original: 0.1

            passives.add_passive(
                name=passive_name,
                r=data["r"],
                z=data["z"],
                d_r=data["d_r"],
                d_z=data["d_z"],
                angle_1=data["angle_1"],
                angle_2=data["angle_2"],
                resistivity=data["resistivity"],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )
        else:
            current_distribution_type = "constant_current_density"
            n_dof = 1
            regularisations = np.empty((0, 0))
            regularisations_weight = np.empty(0)

            passives.add_passive(
                name=passive_name,
                r=data["r"],
                z=data["z"],
                d_r=data["d_r"],
                d_z=data["d_z"],
                angle_1=data["angle_1"],
                angle_2=data["angle_2"],
                resistivity=data["resistivity"],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )

    return passives
