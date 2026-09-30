"""
Thomson scattering (TS) derived isoflux sensors for ST40.

An isoflux sensor states that two points lie on the same flux surface::

    psi(r_1, z_1) - psi(r_2, z_2) = 0

which is linear in every degree of freedom of the fit and has a right-hand side of exactly zero. Unlike a
pressure sensor it needs neither an absolute calibration nor an assumed ion contribution, because the
constraint is invariant under scaling the measured profile by any constant.

The pairs come from the TS profile itself: if the measured quantity is a flux function, then the two radii
at which the profile crosses a given level lie on the same flux surface. Cutting the profile at several
levels gives several pairs per time-slice.

Which quantity to pair on is set by `thomson_scattering.quantity` ("TE", "NE" or "PE"). "TE" is the default,
despite being the noisiest of the three:

* Force balance gives `B . grad p = 0` only for the *total* pressure and only for a static plasma. With
  toroidal rotation the momentum equation picks up a centrifugal term, so the total pressure is no longer
  exactly a flux function.
* Parallel force balance at constant `T_s` integrates to `n_s ~ exp[(m_s Omega^2 R^2 / 2 - e_s Phi) / T_s]`,
  i.e. rotation pushes density outboard on a flux surface. "NE", and hence "PE", inherit that asymmetry.
* "TE" does not: fast parallel heat conduction equilibrates the temperature along a field line whatever the
  centrifugal force does.

The character of the error matters more than its size. The centrifugal term is anti-symmetric between the
high-field side (HFS) and the low-field side (LFS), so it biases every pair the same way and averaging over
pairs does not reduce it. The TS random error is larger on "TE", but it is random, so it averages down.
Pairing on "NE" as well and comparing is a useful cross-check.
"""

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Isoflux

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


def _radius_at_level(
    branch_r: npt.NDArray[np.float64],
    branch_value: npt.NDArray[np.float64],
    value_target: float,
    max_gap: float,
) -> float:
    """
    Find the radius at which a monotonic branch of the profile reaches `value_target`.

    :param branch_r: Radii of the branch [metre], ordered so that `branch_value` is non-decreasing. The radii
        themselves may ascend (HFS branch) or descend (LFS branch).
    :param branch_value: The branch's values, non-decreasing.
    :param value_target: The level to solve for.
    :param max_gap: Reject the level if the two channels bracketing it are further apart than this [metre],
        rather than interpolating across a hole in the data.

    :return: The interpolated radius [metre], or NaN if the level is out of range or unsupported by the data.
    """

    if branch_value.size < 2:
        return float("nan")

    # Out of range; extrapolating here would invent a crossing
    if value_target < branch_value[0] or value_target > branch_value[-1]:
        return float("nan")

    i_upper = int(np.clip(np.searchsorted(branch_value, value_target, side="left"), 1, branch_value.size - 1))
    value_lower, value_upper = branch_value[i_upper - 1], branch_value[i_upper]
    r_lower, r_upper = branch_r[i_upper - 1], branch_r[i_upper]

    # Refuse to interpolate across a hole in the data, e.g. one left behind by the uncertainty filter
    if abs(r_upper - r_lower) > max_gap:
        return float("nan")

    # A flat run, left behind by forcing the branch to be monotonic; the midpoint is the best guess
    if value_upper == value_lower:
        return float(0.5 * (r_lower + r_upper))

    return float(r_lower + (r_upper - r_lower) * (value_target - value_lower) / (value_upper - value_lower))


def find_isoflux_pairs_for_slice(
    r: npt.NDArray[np.float64],
    value: npt.NDArray[np.float64],
    error: npt.NDArray[np.float64],
    level_fractions: typing.Sequence[float],
    max_relative_error: float,
    max_gap: float,
    min_points_per_branch: int,
    r_limits: typing.Sequence[float],
) -> npt.NDArray[np.float64]:
    """
    Find the HFS / LFS radius pair at each level, for a single time-slice.

    The profile is filtered, split at its peak into an HFS branch and an LFS branch, and each branch is then
    forced to decrease monotonically away from the peak. Forcing monotonicity is shape-preserving: it can only
    flatten measured structure, never invent any. It is what makes the level crossing single-valued, and it
    encodes the assumption that the profile is not hollow.

    :param r: Radii of the TS channels [metre]
    :param value: The measured profile
    :param error: The uncertainty on `value`
    :param level_fractions: Levels to cut the profile at, as fractions of its peak value
    :param max_relative_error: Drop channels whose relative uncertainty exceeds this
    :param max_gap: Refuse to interpolate a level across a radial gap wider than this [metre]
    :param min_points_per_branch: Minimum usable channels needed on each branch before any pair is taken
    :param r_limits: Discard pairs falling outside this radial range [metre]

    :return: Array of shape `[n_levels, 2]` holding `(r_hfs, r_lfs)` per level. Levels for which no supported
        pair exists are NaN.
    """

    pairs: npt.NDArray[np.float64] = np.full((len(level_fractions), 2), np.nan)

    usable = np.isfinite(r) & np.isfinite(value) & np.isfinite(error) & (value > 0.0) & (error > 0.0)
    usable &= error <= max_relative_error * value
    if np.count_nonzero(usable) < 2 * min_points_per_branch:
        return pairs

    order = np.argsort(r[usable])
    r_sorted = r[usable][order]
    value_sorted = value[usable][order]

    i_peak = int(np.argmax(value_sorted))
    value_peak = float(value_sorted[i_peak])

    # Split at the peak. Both branches keep the peak itself, so each spans the full range from its edge value
    # up to the peak value.
    hfs_r, hfs_value = r_sorted[: i_peak + 1], value_sorted[: i_peak + 1]
    lfs_r, lfs_value = r_sorted[i_peak:], value_sorted[i_peak:]
    if hfs_r.size < min_points_per_branch or lfs_r.size < min_points_per_branch:
        return pairs

    # Accumulating the running minimum outwards can only push values down towards the edge
    hfs_value_monotonic = np.minimum.accumulate(hfs_value[::-1])[::-1]
    lfs_value_monotonic = np.minimum.accumulate(lfs_value)

    for i_level, fraction in enumerate(level_fractions):
        value_target = fraction * value_peak

        # Both branches are passed value-ascending, as `_radius_at_level` requires
        r_hfs = _radius_at_level(hfs_r, hfs_value_monotonic, value_target, max_gap)
        r_lfs = _radius_at_level(lfs_r[::-1], lfs_value_monotonic[::-1], value_target, max_gap)
        if not (np.isfinite(r_hfs) and np.isfinite(r_lfs) and r_lfs > r_hfs):
            continue

        # A flat-cored profile ("NE" especially) puts its level crossings far out in the scrape-off layer,
        # where the two points are no longer meaningfully on the same closed flux surface
        if r_hfs < r_limits[0] or r_lfs > r_limits[1]:
            continue

        pairs[i_level] = (r_hfs, r_lfs)

    return pairs


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
        time_thomson_scattering = np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}:TIME").data(), dtype=np.float64)
        # `measured` and `measured_error` have shape = [n_time, n_sensors]
        measured = np.atleast_2d(np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:{quantity}").data(), dtype=np.float64))
        measured_error = np.atleast_2d(np.asarray(conn.get(f"\\TS::TOP.{ts_run_name}.PROFILES:{quantity}_ERR").data(), dtype=np.float64))

    # The TS chord sits a few mm off the midplane, so use its actual height rather than assuming Z = 0
    z_thomson_scattering = float(np.nanmedian(sensors_geometry_z))

    n_time = len(times_to_reconstruct)
    n_levels = len(level_fractions)
    location_1_r: npt.NDArray[np.float64] = np.full((n_time, n_levels), np.nan)
    location_2_r: npt.NDArray[np.float64] = np.full((n_time, n_levels), np.nan)

    for i_time in range(n_time):
        # Use the nearest TS time-slice, but only if it is close enough to be describing the same plasma
        i_thomson_scattering = int(np.argmin(np.abs(time_thomson_scattering - times_to_reconstruct[i_time])))
        if abs(time_thomson_scattering[i_thomson_scattering] - times_to_reconstruct[i_time]) > max_time_offset:
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
        "location_1_z": np.full_like(location_1_r, z_thomson_scattering),
        "location_2_r": location_2_r,
        "location_2_z": np.full_like(location_2_r, z_thomson_scattering),
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
