import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Pressure

if TYPE_CHECKING:
    from . import DatabaseReader


def _get_workflow_settings(
    settings: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """
    Return the `workflow` section for the active `database_reader` method.

    The pressure sensor set-up is shared between the `st40_mdsplus` and `st40_mdsplus_with_digital_filter`
    readers, so the method-specific `workflow` section (which holds the PPTS and TS `run_name`s) is looked
    up from the active `method` rather than being hard-coded.

    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    """

    database_reader_settings: dict[str, typing.Any] = settings["GSFIT_code_settings.json"]["database_reader"]
    method: str = database_reader_settings["method"]
    workflow: dict[str, typing.Any] = database_reader_settings[method]["workflow"]

    return workflow


def _read_good_time_slices(
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Find the "good" time-slices for the pressure sensors using the `BAD_MA` flag in the PPTS tree.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    :return: A tuple of `(good_indices, good_times)` where `good_indices` are the indices into the PPTS
        time vector where the magnetics are considered "good" (i.e. `BAD_MA` is `False`), and `good_times`
        are the corresponding times [second].

    The PPTS `run_name` is read from the active method's `workflow` section in `GSFIT_code_settings.json`.

    **This method is specific to ST40's experimental MDSplus database.**
    """

    ppts_run_name: str = _get_workflow_settings(settings)["ppts"]["run_name"]

    with mdsthin.Connection("smaug") as conn:
        conn.openTree("PPTS", pulseNo)
        time_vector = np.asarray(conn.get(f"\\PPTS::TOP.{ppts_run_name}:TIME").data(), dtype=np.float64)
        bad_ma = np.asarray(conn.get(f"\\PPTS::TOP.{ppts_run_name}.GLOBAL:BAD_MA").data(), dtype=bool)

    good_indices = np.where(~bad_ma)[0].astype(np.int64)
    good_times = time_vector[good_indices]

    return good_indices, good_times


def get_good_pressure_sensor_times(
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> npt.NDArray[np.float64]:
    """
    Return the times [second] at which the pressure sensors (Thomson scattering) are considered "good".

    This is used by `Gsfit.setup_timeslices` when `timeslices.method == "good_pressure_sensors"`, so that
    GSFit only reconstructs the time-slices where good pressure data exists.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    :return: The "good" times [second].

    **This method is specific to ST40's experimental MDSplus database.**
    """

    _, good_times = _read_good_time_slices(pulseNo, settings)

    return good_times


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class using ST40's Thomson scattering (TS) data.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    The pressure sensors are only added when `sensor_weights_pressure.json["include"]` is `True`.
    The measured pressure is the electron pressure from the TS `PROFILES` node, scaled by
    `pressure_multiplier` (typically 2.0, to convert electron pressure into total pressure).
    Measurements below `minimum_pressure` are set to NaN so that GSFit ignores them.

    The PPTS and TS `run_name`s are read from the active method's `workflow` section in
    `GSFIT_code_settings.json` (the MDSplus node paths are built from these run names).

    The sensors are added on the "good" time-base (where the `BAD_MA` flag in the PPTS tree is `False`).
    To reconstruct only these time-slices, set `timeslices.method == "good_pressure_sensors"` in
    `GSFIT_code_settings.json`.

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Pressure Rust class
    pressure = Pressure()

    pressure_settings = settings["sensor_weights_pressure.json"]

    # Pressure sensors are opt-in
    if not pressure_settings.get("include", False):
        return pressure

    # Find the "good" time-slices using the `BAD_MA` flag in the PPTS tree
    good_indices, good_times = _read_good_time_slices(pulseNo, settings)

    # Read the Thomson scattering geometry and electron pressure profiles
    thomson_scattering_settings = pressure_settings["thomson_scattering"]
    pressure_multiplier = thomson_scattering_settings["pressure_multiplier"]
    minimum_pressure = thomson_scattering_settings["minimum_pressure"]

    # The TS `run_name` is read from the active method's workflow (e.g. "BEST")
    ts_run_name: str = _get_workflow_settings(settings)["ts"]["run_name"]

    with mdsthin.Connection("smaug") as conn:
        conn.openTree("TS", pulseNo)
        sensors_geometry_r = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:R").data(), dtype=np.float64)
        sensors_geometry_z = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:Z").data(), dtype=np.float64)
        # electron_pressure has shape = [n_time, n_sensors]
        electron_pressure = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:PE").data(), dtype=np.float64)

    # Fit settings, shared by all Thomson scattering channels
    sensor_name_prefix = pressure_settings["sensor_name_prefix"]
    fit_settings_comment = pressure_settings["fit_settings"]["comment"]
    fit_settings_expected_value = pressure_settings["fit_settings"]["expected_value"]
    fit_settings_weight = pressure_settings["fit_settings"]["weight"]

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
