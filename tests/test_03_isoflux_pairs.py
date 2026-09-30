"""
Tests for the Thomson scattering derived isoflux pair-finding.

These exercise `find_isoflux_pairs_for_slice` only, which is pure: it takes a profile and returns the
high-field-side / low-field-side radius pairs. No database connection is needed.
"""

import numpy as np
import numpy.typing as npt
import pytest
from gsfit.database_readers.st40_mdsplus_with_digital_filter.setup_isoflux_sensors import find_isoflux_pairs_for_slice

# A symmetric, peaked profile: a pair cut at any level should straddle the peak symmetrically
R_AXIS = 0.5
R_WIDTH = 0.2
LEVEL_FRACTIONS = [0.25, 0.5, 0.75]


def _parabolic_profile(n_channels: int = 21) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return `(r, value, error)` for a symmetric parabolic profile peaking at `R_AXIS`."""

    r = np.linspace(R_AXIS - R_WIDTH, R_AXIS + R_WIDTH, n_channels)
    value = 1000.0 * (1.0 - ((r - R_AXIS) / R_WIDTH) ** 2) + 1.0
    error = 0.01 * value

    return r, value, error


def _find(
    r: npt.NDArray[np.float64],
    value: npt.NDArray[np.float64],
    error: npt.NDArray[np.float64],
    max_relative_error: float = 0.1,
    max_gap: float = 0.08,
    min_points_per_branch: int = 3,
    r_limits: tuple[float, float] = (0.1, 0.9),
) -> npt.NDArray[np.float64]:
    """Call `find_isoflux_pairs_for_slice` with the test defaults."""

    return find_isoflux_pairs_for_slice(
        r=r,
        value=value,
        error=error,
        level_fractions=LEVEL_FRACTIONS,
        max_relative_error=max_relative_error,
        max_gap=max_gap,
        min_points_per_branch=min_points_per_branch,
        r_limits=r_limits,
    )


def test_pairs_straddle_the_peak_symmetrically() -> None:
    """A symmetric profile must give pairs whose midpoint is the peak, at the right width."""

    r, value, error = _parabolic_profile()
    pairs = _find(r, value, error)

    assert np.all(np.isfinite(pairs)), "every level lies inside a full parabola, so every level should pair"

    for i_level, fraction in enumerate(LEVEL_FRACTIONS):
        r_hfs, r_lfs = pairs[i_level]
        assert r_hfs < R_AXIS < r_lfs

        # The midpoint of the pair is the magnetic axis of a symmetric profile
        assert 0.5 * (r_hfs + r_lfs) == pytest.approx(R_AXIS, abs=1.0e-3)

        # A parabola reaches `fraction` of its peak at `R_WIDTH * sqrt(1 - fraction)` from the peak
        assert 0.5 * (r_lfs - r_hfs) == pytest.approx(R_WIDTH * np.sqrt(1.0 - fraction), abs=5.0e-3)


def test_ordering_of_the_channels_does_not_matter() -> None:
    """The channels are sorted internally, so shuffling the input must not change the result."""

    r, value, error = _parabolic_profile()
    pairs_sorted = _find(r, value, error)

    shuffle = np.random.default_rng(0).permutation(r.size)
    pairs_shuffled = _find(r[shuffle], value[shuffle], error[shuffle])

    np.testing.assert_allclose(pairs_sorted, pairs_shuffled)


def test_noisy_channels_are_dropped() -> None:
    """A channel whose relative uncertainty is too large must not be used to place a crossing."""

    r, value, error = _parabolic_profile()

    # Spoil one whole side of the profile
    error[r > R_AXIS] = value[r > R_AXIS]

    pairs = _find(r, value, error)

    assert np.all(np.isnan(pairs)), "with one branch entirely filtered out, no pair can be formed"


def test_a_hole_in_the_data_is_not_interpolated_across() -> None:
    """A level falling inside a gap wider than `max_gap` must be rejected rather than guessed at."""

    r, value, error = _parabolic_profile()

    # Delete a block of low-field-side channels, leaving a hole much wider than `max_gap`
    keep = ~((r > R_AXIS + 0.02) & (r < R_AXIS + 0.16))

    pairs_narrow_gap = _find(r[keep], value[keep], error[keep], max_gap=0.02)
    pairs_wide_gap = _find(r[keep], value[keep], error[keep], max_gap=1.0)

    assert np.all(np.isnan(pairs_narrow_gap)), "every level's LFS crossing lies inside the hole"
    assert np.any(np.isfinite(pairs_wide_gap)), "the same data pairs up once the gap guard is relaxed"


def test_a_hollow_profile_is_flattened_not_double_valued() -> None:
    """The monotonic filter must keep one crossing per branch even when the core dips."""

    r, value, error = _parabolic_profile()

    # Put a dip in the core, so the raw profile crosses some levels four times rather than twice
    value[np.abs(r - R_AXIS) < 0.05] *= 0.5
    error = 0.01 * value

    pairs = _find(r, value, error)
    finite = np.isfinite(pairs).all(axis=1)

    assert np.any(finite), "a hollow profile should still produce pairs"
    for i_level in np.flatnonzero(finite):
        r_hfs, r_lfs = pairs[i_level]
        assert r_hfs < r_lfs


def test_pairs_outside_the_radial_limits_are_discarded() -> None:
    """A pair straying outside `r_limits` is no longer meaningfully on a closed flux surface."""

    r, value, error = _parabolic_profile()

    pairs = _find(r, value, error, r_limits=(R_AXIS - 0.05, R_AXIS + 0.05))

    assert np.all(np.isnan(pairs)), "every crossing lies outside the artificially narrow limits"


def test_too_few_channels_gives_no_pairs() -> None:
    """Below `min_points_per_branch` on either branch, no pair should be taken."""

    r, value, error = _parabolic_profile(n_channels=5)

    pairs = _find(r, value, error, min_points_per_branch=10)

    assert np.all(np.isnan(pairs))
