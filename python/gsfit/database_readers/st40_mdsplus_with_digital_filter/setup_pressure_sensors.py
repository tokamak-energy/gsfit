from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Pressure

from ...settings_schema.python.equilibrium.constraints_pressure import ConstraintsPressure as ConstraintsPressureSettings
from ...settings_schema.python.thomson_scattering.thomson_scattering import ThomsonScatteringSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def _read_good_time_slices(
    ppts_pulseNo: int,
    ppts_run_name: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Find the "good" time-slices for the pressure sensors using the `BAD_MA` flag in the PPTS tree.

    :param ppts_pulseNo: Pulse number to read the PPTS tree from
    :param ppts_run_name: The PPTS run to read, e.g. "BEST"

    :return: A tuple of `(good_indices, good_times)` where `good_indices` are the indices into the PPTS
        time vector where the magnetics are considered "good" (i.e. `BAD_MA` is `False`), and `good_times`
        are the corresponding times [second].

    **This method is specific to ST40's experimental MDSplus database.**
    """

    with mdsthin.Connection("smaug") as conn:
        conn.openTree("PPTS", ppts_pulseNo)
        time_vector = np.asarray(conn.get(f"\\PPTS::TOP.{ppts_run_name}:TIME").data(), dtype=np.float64)
        bad_ma = np.asarray(conn.get(f"\\PPTS::TOP.{ppts_run_name}.GLOBAL:BAD_MA").data(), dtype=bool)

    good_indices = np.where(~bad_ma)[0].astype(np.int64)
    good_times = time_vector[good_indices]

    return good_indices, good_times


def get_good_pressure_sensor_times(
    ppts_pulseNo: int,
    ppts_run_name: str,
) -> npt.NDArray[np.float64]:
    """
    Return the times [second] at which the pressure sensors (Thomson scattering) are considered "good".

    This is for `Gsfit.setup_timeslices` when `timeslices.method == "good_pressure_sensors"`, so that
    GSFit only reconstructs the time-slices where good pressure data exists.

    :param ppts_pulseNo: Pulse number to read the PPTS tree from
    :param ppts_run_name: The PPTS run to read, e.g. "BEST", from the active database reader's settings

    :return: The "good" times [second].

    **This method is specific to ST40's experimental MDSplus database.**
    """

    _, good_times = _read_good_time_slices(ppts_pulseNo, ppts_run_name)

    return good_times


def make_pressure_sensors(
    ppts_pulseNo: int,
    ppts_run_name: str,
    ts_pulseNo: int,
    ts_run_name: str,
    settings_pressure_sensors: ConstraintsPressureSettings,
) -> Pressure:
    """
    Make the Rust `Pressure` class using ST40's Thomson scattering (TS) data, when the pressure sensors are included.

    It is shared by the `st40_mdsplus` and `st40_mdsplus_with_digital_filter` readers, which each say which PPTS and TS runs to read, from their own
    settings. They only call it when `include` is true in `constraints_pressure.xml`, as the runs are only needed then.

    :param ppts_pulseNo: Pulse number to read the PPTS tree from
    :param ppts_run_name: The PPTS run to read the good time-slices from, e.g. "BEST"
    :param ts_pulseNo: Pulse number to read the TS tree from
    :param ts_run_name: The TS run to read the geometry and electron pressure from, e.g. "BEST"
    :param settings_pressure_sensors: `ConstraintsPressure` object, read from `equilibrium/constraints_pressure.xml`, which holds the pressure sensors'
        fitting settings

    The measured pressure is the electron pressure from the TS `PROFILES` node, scaled by
    `pressure_multiplier` (typically 2.0, to convert electron pressure into total pressure).
    Measurements below `minimum_pressure` are set to NaN so that GSFit ignores them.

    The sensors are added on the "good" time-base (where the `BAD_MA` flag in the PPTS tree is `False`).
    To reconstruct only these time-slices, set `method` to "good_pressure_sensors" in `timeslices.xml`.

    **This method is specific to ST40's experimental MDSplus database.**
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    pressure_settings = settings_pressure_sensors.pressure

    # `constraints_pressure.xml` must give these when the pressure sensors are included
    sensor_name_prefix = pressure_settings.sensor_name_prefix
    fit_settings_expected_value = pressure_settings.expected_value
    fit_settings_weight = pressure_settings.weight
    thomson_scattering_settings = pressure_settings.thomson_scattering
    if sensor_name_prefix is None or fit_settings_expected_value is None or fit_settings_weight is None or thomson_scattering_settings is None:
        raise ValueError(
            "`constraints_pressure.xml` includes the pressure sensors, so needs `sensor_name_prefix`, `expected_value`, `weight` and `thomson_scattering`"
        )
    # The comment is free text, which the fit does not use
    fit_settings_comment = pressure_settings.comment if pressure_settings.comment is not None else ""

    # Find the "good" time-slices using the `BAD_MA` flag in the PPTS tree
    good_indices, good_times = _read_good_time_slices(ppts_pulseNo, ppts_run_name)

    # Read the Thomson scattering geometry and electron pressure profiles
    pressure_multiplier = thomson_scattering_settings.pressure_multiplier
    minimum_pressure = thomson_scattering_settings.minimum_pressure

    with mdsthin.Connection("smaug") as conn:
        conn.openTree("TS", ts_pulseNo)
        sensors_geometry_r = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:R").data(), dtype=np.float64)
        sensors_geometry_z = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:Z").data(), dtype=np.float64)
        # electron_pressure has shape = [n_time, n_sensors]
        electron_pressure = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:PE").data(), dtype=np.float64)

    n_sensors = len(sensors_geometry_r)
    for i_sensor in range(n_sensors):
        # Convert electron pressure into total pressure at the "good" time-slices
        measured = pressure_multiplier * electron_pressure[good_indices, i_sensor]
        measured = measured.copy()

        # Ignore (set to NaN) any non-physical / below-threshold measurements
        measured[measured <= minimum_pressure] = np.nan

        # Only add the sensor if it has at least one valid measurement
        if np.any(~np.isnan(measured)):
            pressure.add_sensor(
                name=f"{sensor_name_prefix}{i_sensor + 1:02d}",
                geometry_r=sensors_geometry_r[i_sensor],
                geometry_z=sensors_geometry_z[i_sensor],
                fit_settings_comment=fit_settings_comment,
                fit_settings_expected_value=fit_settings_expected_value,
                fit_settings_include=True,
                fit_settings_weight=fit_settings_weight,
                time=good_times,
                measured=measured,
            )

    return pressure


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: ThomsonScatteringSettings | None,
    settings_pressure_sensors: ConstraintsPressureSettings,
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class using ST40's Thomson scattering (TS) data; see `make_pressure_sensors`.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `ThomsonScatteringSettings` object, read from `thomson_scattering/thomson_scattering.xml`, whose `workflow` says which PPTS and TS
        runs to read; None when `GSFIT_code_settings.xml` does not link one. Only needed when the pressure sensors are included
    :param settings_pressure_sensors: `ConstraintsPressure` object, read from `equilibrium/constraints_pressure.xml`, which says whether the pressure
        sensors are included, and holds their fitting settings

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Pressure sensors are opt-in, so the Thomson scattering is only read, and its runs only needed, when they are included
    if not settings_pressure_sensors.pressure.include:
        return Pressure()

    # The PPTS and TS runs to read, from the `workflow.xml` which `thomson_scattering.xml` links to
    workflow = settings.workflow if settings is not None else None
    if workflow is None or "PPTS" not in workflow.tree or "TS" not in workflow.tree:
        raise ValueError(
            "`constraints_pressure.xml` includes the pressure sensors, so `thomson_scattering.xml` must link to a `workflow.xml` holding the PPTS and TS "
            "`tree`s, which say which PPTS and TS runs to read"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    ppts_settings = workflow.tree["PPTS"]
    if ppts_settings.pulseNo is None:
        ppts_pulseNo = pulseNo
    else:
        ppts_pulseNo = ppts_settings.pulseNo
    ts_settings = workflow.tree["TS"]
    if ts_settings.pulseNo is None:
        ts_pulseNo = pulseNo
    else:
        ts_pulseNo = ts_settings.pulseNo

    return make_pressure_sensors(
        ppts_pulseNo=ppts_pulseNo,
        ppts_run_name=ppts_settings.run_name,
        ts_pulseNo=ts_pulseNo,
        ts_run_name=ts_settings.run_name,
        settings_pressure_sensors=settings_pressure_sensors,
    )
