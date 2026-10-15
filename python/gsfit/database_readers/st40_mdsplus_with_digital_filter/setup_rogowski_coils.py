import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import RogowskiCoils
from st40_database import GetData

from ...settings_schema.python.equilibrium.constraints_rogowski_coil import ConstraintsRogowskiCoil as ConstraintsRogowskiCoilSettings
from ...settings_schema.python.equilibrium.constraints_rogowski_coil import RogowskiCoil
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_rogowski_coils(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
    settings_rogowski_coils: ConstraintsRogowskiCoilSettings,
) -> RogowskiCoils:
    """
    This method initialises the Rust `RogowskiCoils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object, read from `magnetics/magnetics.xml`, whose `workflow` says which MAG run to read the Rogowski coils from
    :param settings_rogowski_coils: `ConstraintsRogowskiCoil` object, read from `equilibrium/constraints_rogowski_coil.xml`, which holds each
        Rogowski coil's fitting settings. A Rogowski coil which is not listed there is not included in the fit

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the RogowskiCoils Rust class
    rogowski_coils = RogowskiCoils()

    # The MAG run to read, from the `workflow.xml` which `magnetics.xml` links to
    workflow = settings.workflow
    if workflow is None or workflow.mag is None:
        raise ValueError("`magnetics.xml` must link to a `workflow.xml` holding `mag`, which says which MAG run to read the Rogowski coils from")
    # A nil `pulseNo` reads the pulse being reconstructed
    mag_settings = workflow.mag
    if mag_settings.pulseNo is None:
        mag_pulseNo = pulseNo
    else:
        mag_pulseNo = mag_settings.pulseNo

    # Read in mag data from MDSplus
    mag = GetData(mag_pulseNo, f"MAG#{mag_settings.run_name}", is_fail_quiet=False)

    # The fitting settings of each Rogowski coil listed in `constraints_rogowski_coil.xml`, by name
    sensors_fit_settings: dict[str, RogowskiCoil] = {}
    for sensor_fit_settings in settings_rogowski_coils.rogowski_coil:
        sensors_fit_settings[sensor_fit_settings.name] = sensor_fit_settings

    names_long = typing.cast(list[str], mag.get("ROG.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "ROG_", "")
    paths_r = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.R_PATH")).astype(np.float64)
    paths_z = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.Z_PATH")).astype(np.float64)

    n_sensors = len(sensors_names)
    for i_sensor in range(n_sensors):
        sensor_name = sensors_names[i_sensor]
        path_r = paths_r[i_sensor, :]
        path_z = paths_z[i_sensor, :]

        # Remove nan's
        # This is because in MDSplus "ALL" does not allow jagged arrays
        path_r = path_r[~np.isnan(path_r)]
        path_z = path_z[~np.isnan(path_z)]

        # Don't store the "fake" Rogowski coils (e.g. the MC supports)
        if len(path_r) > 4:
            if sensor_name in sensors_fit_settings:
                sensor_fit_settings = sensors_fit_settings[sensor_name]
                fit_settings_comment = sensor_fit_settings.comment
                fit_settings_expected_value = sensor_fit_settings.expected_value
                fit_settings_include = sensor_fit_settings.include
                fit_settings_weight = sensor_fit_settings.weight
            else:
                fit_settings_comment = ""
                fit_settings_expected_value = np.nan
                fit_settings_include = False
                fit_settings_weight = np.nan

            # Measured signals
            time = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)
            measured = typing.cast(npt.NDArray[np.float32], mag.get(f"ROG.{sensor_name}.I_PROCESSED")).astype(np.float64)

            # By default we don't have any gaps
            gaps_r = np.array([])
            gaps_z = np.array([])
            gaps_d_r = np.array([])
            gaps_d_z = np.array([])
            gaps_name = []

            # Only INIVC000 has gaps
            if sensor_name == "INIVC000":
                import mdsthin

                conn = mdsthin.Connection("smaug")
                conn.openTree("st40", 11010605)
                gaps_r = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:R").data().astype(np.float64)
                gaps_z = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:Z").data().astype(np.float64)
                gaps_d_r = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DR").data().astype(np.float64)
                gaps_d_z = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DZ").data().astype(np.float64)
                gaps_name = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:NAME").data()
                gaps_name = [str(gap_name).replace(" ", "")[2:-1] for gap_name in gaps_name]

            # Add Rogowski coil to the Rust class
            rogowski_coils.add_sensor(
                sensor_name,
                path_r,
                path_z,
                fit_settings_comment,
                fit_settings_expected_value,
                fit_settings_include,
                fit_settings_weight,
                time,
                measured,
                gaps_r=gaps_r,
                gaps_z=gaps_z,
                gaps_d_r=gaps_d_r,
                gaps_d_z=gaps_d_z,
                gaps_name=gaps_name,
            )

    return rogowski_coils
