"""Database-independent pairing of Thomson-scattering profile level crossings."""

import typing

import numpy as np
import numpy.typing as npt


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
    if np.count_nonzero(usable) < 2 * min_points_per_branch - 1:
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
