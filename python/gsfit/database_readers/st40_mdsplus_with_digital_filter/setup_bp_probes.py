import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes
from st40_database import GetData

from ...settings_schema.python.equilibrium.constraints_b_field_pol_probe import BFieldPolProbe
from ...settings_schema.python.equilibrium.constraints_b_field_pol_probe import ConstraintsBFieldPolProbe as ConstraintsBFieldPolProbeSettings
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_bp_probes(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
    settings_bp_probes: ConstraintsBFieldPolProbeSettings,
) -> BpProbes:
    """
    This method initialises the Rust `BpProbes` class (Mirnov coils).

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object, read from `magnetics/magnetics.xml`, whose `workflow` says which MAG run to read the probes from
    :param settings_bp_probes: `ConstraintsBFieldPolProbe` object, read from `equilibrium/constraints_b_field_pol_probe.xml`, which holds each probe's
        fitting settings. A probe which is not listed there is not included in the fit

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the BpProbes Rust class
    bp_probes = BpProbes()

    # The MAG run to read, from the `workflow.xml` which `magnetics.xml` links to
    workflow = settings.workflow
    if workflow is None or "MAG" not in workflow.tree:
        raise ValueError(
            "`magnetics.xml` must link to a `workflow.xml` holding the MAG `tree`, which says which MAG run to read the poloidal field probes from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    mag_settings = workflow.tree["MAG"]
    if mag_settings.pulseNo is None:
        mag_pulseNo = pulseNo
    else:
        mag_pulseNo = mag_settings.pulseNo

    # read in mag data from MDSplus
    mag = GetData(mag_pulseNo, f"MAG#{mag_settings.run_name}", is_fail_quiet=False)

    # The fitting settings of each probe listed in `constraints_b_field_pol_probe.xml`, by name
    sensors_fit_settings: dict[str, BFieldPolProbe] = {}
    for sensor_fit_settings in settings_bp_probes.b_field_pol_probe:
        sensors_fit_settings[sensor_fit_settings.name] = sensor_fit_settings

    # Bp-probes
    names_long = typing.cast(list[str], mag.get("BPPROBE.ALL.NAMES"))
    sensors_names = np.char.replace(names_long, "BPPROBE_", "P")
    sensors_r = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.R"))
    sensors_z = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.Z"))
    sensors_angle_pol = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.THETA"))

    n_sensors = len(sensors_names)
    for i_sensor in range(n_sensors):
        sensor_name = sensors_names[i_sensor]

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

        # Measured values
        time = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)
        measured = typing.cast(npt.NDArray[np.float32], mag.get(f"BPPROBE.{sensor_name.upper()}.B_PROCESSED")).astype(np.float64)

        # Add the sensor to the Rust class
        bp_probes.add_sensor(
            name=sensor_name,
            geometry_angle_pol=sensors_angle_pol[i_sensor],
            geometry_r=sensors_r[i_sensor],
            geometry_z=sensors_z[i_sensor],
            fit_settings_comment=fit_settings_comment,
            fit_settings_expected_value=fit_settings_expected_value,
            fit_settings_include=fit_settings_include,
            fit_settings_weight=fit_settings_weight,
            time=time,
            measured=measured,
        )

    return bp_probes
