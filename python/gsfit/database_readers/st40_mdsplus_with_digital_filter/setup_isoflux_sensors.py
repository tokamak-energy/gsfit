"""
Build ST40 isoflux sensors from Thomson-scattering (TS) profiles by pairing high- and low-field-side radii at
configured fractions of the selected profile's peak.

This assumes the selected quantity (`TE`, `NE` or `PE`) is approximately constant on a flux surface. `TE` is
the default; selecting another quantity can provide a useful sensitivity check. Channels with relative
uncertainty above `max_relative_error` are discarded, but the remaining TS uncertainties do not individually
weight the isoflux constraints.
"""

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Isoflux

from gsfit.isoflux_pairs import find_isoflux_pairs_for_slice as find_isoflux_pairs_for_slice

if TYPE_CHECKING:
    from . import DatabaseReader

# TS `PROFILES` nodes which may be paired on
TS_PROFILE_NODES: tuple[str, ...] = ("TE", "NE", "PE")


def _get_workflow_settings(
    settings: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """
    Return the `workflow` section for the active `database_reader` method.

    The isoflux sensor set-up is shared between the `st40_mdsplus` and `st40_mdsplus_with_digital_filter`
    readers, so the method-specific `workflow` section (which holds the TS `run_name`) is looked up from the
    active `method` rather than being hard-coded.

    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    """

    database_reader_settings: dict[str, typing.Any] = settings["GSFIT_code_settings.json"]["database_reader"]
    method: str = database_reader_settings["method"]
    workflow: dict[str, typing.Any] = database_reader_settings[method]["workflow"]

    return workflow


def build_isoflux_pairs(
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Read the Thomson scattering profiles and build one HFS / LFS pair per (time, level).

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param times_to_reconstruct: Times the equilibrium will be solved at [second]

    :return: Dictionary of arrays, all of shape `[n_time, n_levels]` except `time` (`[n_time]`) and
        `level_fractions` (`[n_levels]`): `location_1_r` / `location_1_z` (HFS) and `location_2_r` /
        `location_2_z` (LFS). Pairs which could not be found are NaN, which `Isoflux.add_sensor` turns into a
        `False` in that sensor's time-dependent "include" flag.

    **This method is specific to ST40's experimental MDSplus database.**
    """

    isoflux_settings = settings["sensor_weights_isoflux.json"]
    thomson_scattering_settings = isoflux_settings["thomson_scattering"]

    quantity: str = thomson_scattering_settings["quantity"]
    if quantity not in TS_PROFILE_NODES:
        raise ValueError(f"sensor_weights_isoflux.json: thomson_scattering.quantity must be one of {TS_PROFILE_NODES}, got {quantity!r}")

    level_fractions = thomson_scattering_settings["level_fractions"]
    max_relative_error = thomson_scattering_settings["max_relative_error"]
    max_gap = thomson_scattering_settings["max_gap"]
    min_points_per_branch = thomson_scattering_settings["min_points_per_branch"]
    max_time_offset = thomson_scattering_settings["max_time_offset"]
    r_limits = thomson_scattering_settings["r_limits"]

    # The TS `run_name` is read from the active method's workflow (e.g. "BEST")
    ts_run_name: str = _get_workflow_settings(settings)["ts"]["run_name"]

    with mdsthin.Connection("smaug") as conn:
        conn.openTree("TS", pulseNo)
        sensors_geometry_r = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:R").data(), dtype=np.float64)
        sensors_geometry_z = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:Z").data(), dtype=np.float64)
        thomson_scattering_time = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:TIME").data(), dtype=np.float64)
        # `measured` and `measured_error` have shape = [n_time, n_sensors]
        measured = np.atleast_2d(np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:{quantity}").data(), dtype=np.float64))
        measured_error = np.atleast_2d(np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:{quantity}_ERR").data(), dtype=np.float64))

    # The TS chord sits a few mm off the midplane, so use its actual height rather than assuming Z = 0
    thomson_scattering_z = float(np.nanmedian(sensors_geometry_z))

    n_time = len(times_to_reconstruct)
    n_levels = len(level_fractions)
    location_1_r: npt.NDArray[np.float64] = np.full((n_time, n_levels), np.nan)
    location_2_r: npt.NDArray[np.float64] = np.full((n_time, n_levels), np.nan)

    for i_time in range(n_time):
        # Use the nearest TS time-slice, but only if it is close enough to be describing the same plasma
        i_thomson_scattering = int(np.argmin(np.abs(thomson_scattering_time - times_to_reconstruct[i_time])))
        if abs(thomson_scattering_time[i_thomson_scattering] - times_to_reconstruct[i_time]) > max_time_offset:
            continue

        pairs = find_isoflux_pairs_for_slice(
            r=sensors_geometry_r,
            value=measured[i_thomson_scattering],
            error=measured_error[i_thomson_scattering],
            level_fractions=level_fractions,
            max_relative_error=max_relative_error,
            max_gap=max_gap,
            min_points_per_branch=min_points_per_branch,
            r_limits=r_limits,
        )
        location_1_r[i_time] = pairs[:, 0]
        location_2_r[i_time] = pairs[:, 1]

    return {
        "time": np.asarray(times_to_reconstruct, dtype=np.float64),
        "level_fractions": np.asarray(level_fractions, dtype=np.float64),
        "location_1_r": location_1_r,
        "location_1_z": np.full_like(location_1_r, thomson_scattering_z),
        "location_2_r": location_2_r,
        "location_2_z": np.full_like(location_2_r, thomson_scattering_z),
    }


def setup_isoflux_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
    times_to_reconstruct: npt.NDArray[np.float64],
) -> Isoflux:
    """
    This method initialises the Rust `Isoflux` class using ST40's Thomson scattering (TS) data.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    :param times_to_reconstruct: Times the equilibrium will be solved at [second]

    The isoflux sensors are only added when `sensor_weights_isoflux.json["include"]` is `True`. One sensor is
    created per entry in `thomson_scattering.level_fractions`; see this module's docstring for how the pairs
    are found and for why `thomson_scattering.quantity` defaults to "TE".

    The TS `run_name` is read from the active method's `workflow` section in `GSFIT_code_settings.json`.

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Isoflux Rust class
    isoflux = Isoflux()

    isoflux_settings = settings["sensor_weights_isoflux.json"]

    # Isoflux sensors are opt-in
    if not isoflux_settings.get("include", False):
        return isoflux

    times_to_reconstruct = np.asarray(times_to_reconstruct, dtype=np.float64)
    pairs = build_isoflux_pairs(pulseNo, settings, times_to_reconstruct)

    sensor_name_prefix = isoflux_settings["sensor_name_prefix"]
    fit_settings_comment = isoflux_settings["fit_settings"]["comment"]
    fit_settings_weight = isoflux_settings["fit_settings"]["weight"]
    quantity = isoflux_settings["thomson_scattering"]["quantity"]

    n_levels = len(pairs["level_fractions"])
    for i_level in range(n_levels):
        location_1_r = pairs["location_1_r"][:, i_level]
        location_2_r = pairs["location_2_r"][:, i_level]

        # A level which never produced a pair would be excluded on every time-slice anyway
        if not np.any(np.isfinite(location_1_r) & np.isfinite(location_2_r)):
            continue

        # `time` is `times_to_reconstruct` itself, so `Isoflux.add_sensor` interpolates onto points it already
        # has exactly. That matters: it keeps a NaN confined to its own time-slice instead of letting it
        # spread into the neighbouring ones.
        isoflux.add_sensor(
            name=f"{sensor_name_prefix}{i_level + 1:02d}",
            fit_settings_comment=f"{fit_settings_comment}; TS {quantity} = {pairs['level_fractions'][i_level]:.2f} x peak",
            fit_settings_include=True,
            fit_settings_weight=fit_settings_weight,
            time=pairs["time"],
            location_1_r=location_1_r,
            location_1_z=pairs["location_1_z"][:, i_level],
            location_2_r=location_2_r,
            location_2_z=pairs["location_2_z"][:, i_level],
            times_to_reconstruct=times_to_reconstruct,
        )

    return isoflux
