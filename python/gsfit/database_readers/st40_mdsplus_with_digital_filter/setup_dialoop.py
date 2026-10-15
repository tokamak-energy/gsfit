import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Dialoop
from st40_database import GetData

from ...settings_schema.python.equilibrium.constraints_diamagnetic_flux import ConstraintsDiamagneticFlux as ConstraintsDiamagneticFluxSettings
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_dialoop(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
    settings_dialoop: ConstraintsDiamagneticFluxSettings,
) -> Dialoop:
    """
    This method initialises the Rust `Dialoop` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object, read from `magnetics/magnetics.xml`, whose `workflow` says which DIALOOP run to read the diamagnetic
        flux from, and which MAG run to read the loop's geometry from
    :param settings_dialoop: `ConstraintsDiamagneticFlux` object, read from `equilibrium/constraints_diamagnetic_flux.xml`, which holds the
        diamagnetic loop's fitting settings. The loop is not included in the fit when it is not listed there

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Dialoop Rust class
    dialoop = Dialoop()

    # The DIALOOP and MAG runs to read, from the `workflow.xml` which `magnetics.xml` links to
    workflow = settings.workflow
    if workflow is None or workflow.dialoop is None or workflow.mag is None:
        raise ValueError(
            "`magnetics.xml` must link to a `workflow.xml` holding `dialoop` and `mag`, which say which DIALOOP run to read the diamagnetic flux from, "
            "and which MAG run to read the loop's geometry from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    dialoop_settings = workflow.dialoop
    if dialoop_settings.pulseNo is None:
        dialoop_pulseNo = pulseNo
    else:
        dialoop_pulseNo = dialoop_settings.pulseNo
    mag_settings = workflow.mag
    if mag_settings.pulseNo is None:
        mag_pulseNo = pulseNo
    else:
        mag_pulseNo = mag_settings.pulseNo

    # We use a single diamagnetic flux loop
    sensor_name = "DIALOOP"

    # Its fitting settings, from `constraints_diamagnetic_flux.xml`, which lists at most one loop
    sensor_fit_settings = settings_dialoop.diamagnetic_flux
    if sensor_fit_settings is not None and sensor_fit_settings.name == sensor_name:
        fit_settings_comment = sensor_fit_settings.comment
        fit_settings_expected_value = sensor_fit_settings.expected_value
        fit_settings_include = sensor_fit_settings.include
        fit_settings_weight = sensor_fit_settings.weight
    else:
        fit_settings_comment = ""
        fit_settings_expected_value = np.nan
        fit_settings_include = False
        fit_settings_weight = np.nan

    try:
        # Read in the diamagnetic flux from MDSplus
        dialoop_data = GetData(dialoop_pulseNo, f"DIALOOP#{dialoop_settings.run_name}", is_fail_quiet=False)

        # Get geometry (path of integration) directly via mdsthin from the MAG tree
        # (the geometry is stored in MAG, not in the DIALOOP tree)
        conn = mdsthin.Connection("smaug")
        conn.openTree("MAG", mag_pulseNo)
        path_r = conn.get(f"\\MAG::TOP.{mag_settings.run_name}.DIALOOP.L000:R_PATH").data().astype(np.float64)
        path_z = conn.get(f"\\MAG::TOP.{mag_settings.run_name}.DIALOOP.L000:Z_PATH").data().astype(np.float64)

        # Measured signal: \DIALOOP::TOP.<run_name>.GLOBAL:PHI_DIA
        time = typing.cast(npt.NDArray[np.float64], dialoop_data.get("TIME")).astype(np.float64)
        measured = typing.cast(npt.NDArray[np.float64], dialoop_data.get("GLOBAL.PHI_DIA")).astype(np.float64)

        # Add the sensor to the Rust class
        dialoop.add_sensor(
            name=sensor_name,
            r=path_r,
            z=path_z,
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )
    except Exception as e:
        print(f"Error setting up DIALOOP for pulse {pulseNo}: {e}")

    return dialoop
