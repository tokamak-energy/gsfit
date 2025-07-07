import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Passives
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_passives(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Passives:
    """
    This method initialises the Rust `Passives` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Passives Rust class
    passives = Passives()

    elmag_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["elmag_run_name"]
    # FIXME: using a fixed shot is not a good idea!
    elmag = GetData(12050, f"ELMAG#{elmag_run_name}", is_fail_quiet=False)

    vessel_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.R"))
    vessel_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.Z"))
    vessel_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DR"))
    vessel_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DZ"))
    vessel_angle_1 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE1"))
    vessel_angle_2 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE2"))
    vessel_resistivity = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.RESISTIVITY"))
    vessel_fillaments_to_passives = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.FILS2PASSIVE"))
    [n_filaments, n_passives] = vessel_fillaments_to_passives.shape
    passive_names = elmag.get("VESSEL.PASSIVE_NAME")

    for i_passive in range(0, n_passives):
        passive_name = passive_names[i_passive]
        i_filaments = vessel_fillaments_to_passives[:, i_passive].astype(int) == True

        if passive_name == "IVC":
            current_distribution_type = "eig"
            n_dof = settings["passive_dof_regularisation.json"]["IVC"]["n_dof"]
            regularisations = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations"])
            regularisations_weight = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations_weight"])

            # BUXTON: temporary fix to use PFIT eigenvalues
            from MDSplus import Connection  # type: ignore

            conn = Connection("smaug")
            conn.openTree("elmag", 206)
            tmp_vessel_r = conn.get("\\ELMAG::TOP.VESSEL:R").data()[0:480]
            tmp_vessel_z = conn.get("\\ELMAG::TOP.VESSEL:Z").data()[0:480]
            tmp_vessel_d_r = conn.get("\\ELMAG::TOP.VESSEL:DR").data()[0:480]
            tmp_vessel_d_z = conn.get("\\ELMAG::TOP.VESSEL:DZ").data()[0:480]
            tmp_vessel_angle_1 = 0.0 * tmp_vessel_r
            tmp_vessel_angle_2 = 0.0 * tmp_vessel_r

            passives.add_passive(
                name=passive_name,
                r=tmp_vessel_r,
                z=tmp_vessel_z,
                d_r=tmp_vessel_d_r,
                d_z=tmp_vessel_d_z,
                angle_1=tmp_vessel_angle_1,
                angle_2=tmp_vessel_angle_2,
                resistivity=vessel_resistivity[i_passive],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )
        elif passive_name == "OVC":
            current_distribution_type = "constant_current_density"
            n_dof = 1
            regularisations = np.array([[1.0]])
            regularisations_weight = np.array([0.1])

            passives.add_passive(
                name=passive_name,
                r=vessel_r[i_filaments],
                z=vessel_z[i_filaments],
                d_r=vessel_d_r[i_filaments],
                d_z=vessel_d_z[i_filaments],
                angle_1=vessel_angle_1[i_filaments],
                angle_2=vessel_angle_2[i_filaments],
                resistivity=vessel_resistivity[i_passive],
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
                r=vessel_r[i_filaments],
                z=vessel_z[i_filaments],
                d_r=vessel_d_r[i_filaments],
                d_z=vessel_d_z[i_filaments],
                angle_1=vessel_angle_1[i_filaments],
                angle_2=vessel_angle_2[i_filaments],
                resistivity=vessel_resistivity[i_passive],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )

    return passives
