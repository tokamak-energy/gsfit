from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import CoreProfiles

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.core_profiles.core_profiles import CoreProfilesSettings
from ...settings_schema.python.workflow import Workflow

if TYPE_CHECKING:
    from . import DatabaseReader


def _read_good_time_slices(
    ppts_pulseNo: int,
    ppts_run_name: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Find the "good" time-slices of the Thomson scattering using the `BAD_MA` flag in the PPTS tree.

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
    Return the times [second] at which the Thomson scattering, which the pressure constraints are made from, is considered "good".

    This is for `Gsfit.setup_timeslices` when `timeslices.method == "good_pressure_sensors"`, so that
    GSFit only reconstructs the time-slices where good pressure data exists.

    :param ppts_pulseNo: Pulse number to read the PPTS tree from
    :param ppts_run_name: The PPTS run to read, e.g. "BEST", from the active database reader's settings

    :return: The "good" times [second].

    **This method is specific to ST40's experimental MDSplus database.**
    """

    _, good_times = _read_good_time_slices(ppts_pulseNo, ppts_run_name)

    return good_times


def setup_core_profiles(
    self: "DatabaseReader",
    pulseNo: int,
    settings: CoreProfilesSettings,
) -> CoreProfiles:
    """
    This method initialises the Rust `CoreProfiles` class, which holds an IMAS `core_profiles` IDS, from ST40's Thomson scattering (TS).

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `CoreProfilesSettings` object, read from `core_profiles/core_profiles.xml`, whose `workflow` says which PPTS and TS runs
        to read, and whose `thomson_scattering` says how the electron pressure is turned into the thermal pressure

    **This method is specific to ST40's experimental MDSplus database, with a digital filter.**

    `core_profiles/profiles_2d(itime)` is filled at each TS channel's position, the irregular (R, Z) grid `irregular_rz_na`, one time per
    "good" time-slice, where the `BAD_MA` flag in the PPTS tree is `False`:
    1. `electrons/pressure`, the electron pressure from the TS `PROFILES` node, `n_e * T_e`
    2. `pressure_thermal`, the electron pressure scaled by `pressure_multiplier`, typically 2.0, which is `2 * n_e * T_e`: the thermal
       pressure when the ions have the electrons' density and temperature. A thermal pressure at or below `minimum_pressure` is not physical,
       so it is NaN, which the equilibrium does not fit.

    Both are on the **experimental** timebase. `Equilibrium.copy_from_core_profiles_ids` makes the pressure constraints from
    `pressure_thermal`, interpolated onto the reconstruction times itself, so do not interpolate here. To reconstruct only the good
    time-slices, set `method` to "good_pressure_sensors" in `timeslices.xml`.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The PPTS and TS runs to read, from the `workflow.xml` which `core_profiles.xml` links to
    workflow = settings.workflow
    if workflow is None or "PPTS" not in workflow.tree or "TS" not in workflow.tree:
        raise ValueError(
            "`core_profiles.xml` must link to a `workflow.xml` holding the PPTS and TS `tree`s, which say which PPTS and TS runs to read the "
            "good time-slices and the Thomson scattering from"
        )
    thomson_scattering_settings = settings.thomson_scattering
    if thomson_scattering_settings is None:
        raise ValueError(
            "`core_profiles.xml` needs `thomson_scattering`, which says how the Thomson scattering's electron pressure is turned into the thermal pressure"
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

    # Serialise the settings to be stored inside the IDS: `core_profiles.xml` holding only the settings which built the kinetic profiles
    settings_xml = serialise_python_dataclass_to_imas_parameters_input(
        settings=CoreProfilesSettings(
            workflow=Workflow(tree=KeyedList([ppts_settings, ts_settings])),
            thomson_scattering=thomson_scattering_settings,
        ),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the CoreProfiles Rust class
    core_profiles = CoreProfiles(settings_xml)

    # Find the "good" time-slices using the `BAD_MA` flag in the PPTS tree
    good_indices, good_times = _read_good_time_slices(ppts_pulseNo, ppts_settings.run_name)

    # Read the Thomson scattering channels' positions and electron pressure
    with mdsthin.Connection("smaug") as conn:
        conn.openTree("TS", ts_pulseNo)
        channel_r = np.asarray(conn.get(f"\\TS::TOP.{ts_settings.run_name}:R").data(), dtype=np.float64)
        channel_z = np.asarray(conn.get(f"\\TS::TOP.{ts_settings.run_name}:Z").data(), dtype=np.float64)
        # shape = [n_time, n_channel]
        electron_pressure_all_times = np.asarray(conn.get(f"\\TS::TOP.{ts_settings.run_name}.PROFILES:PE").data(), dtype=np.float64)

    # The good time-slices only. shape = [n_time, n_channel]
    electrons_pressure = electron_pressure_all_times[good_indices, :]

    # The thermal pressure, 2 * n_e * T_e when the ions have the electrons' density and temperature; not physical at or below the minimum
    pressure_thermal = thomson_scattering_settings.pressure_multiplier * electrons_pressure
    pressure_thermal[pressure_thermal <= thomson_scattering_settings.minimum_pressure] = np.nan

    core_profiles.set_profiles_2d_irregular_rz(
        time=good_times,
        r=channel_r,
        z=channel_z,
        electrons_pressure=electrons_pressure,
        pressure_thermal=pressure_thermal,
    )

    return core_profiles
