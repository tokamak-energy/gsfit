"""Tests for selecting Thomson-scattering time-slices for isoflux pairing."""

import numpy as np
import numpy.typing as npt
import pytest

from gsfit.database_readers.st40_mdsplus_with_digital_filter.setup_isoflux_sensors import _nearest_valid_ts_time_index


def test_nearest_valid_ts_time_ignores_nonfinite_offsets() -> None:
    time_thomson_scattering = np.array([0.9, np.nan, 1.02, np.inf])

    actual = _nearest_valid_ts_time_index(time_thomson_scattering, 1.0, 0.05)

    assert actual == 2


@pytest.mark.parametrize(
    ("time_thomson_scattering", "time_to_reconstruct", "max_time_offset"),
    [
        (np.array([np.nan, np.inf, -np.inf]), 1.0, 0.05),
        (np.array([0.9, 1.1]), np.nan, 0.05),
        (np.array([0.9, 1.1]), 1.0, 0.05),
    ],
)
def test_no_valid_ts_time_returns_none(
    time_thomson_scattering: npt.NDArray[np.float64],
    time_to_reconstruct: float,
    max_time_offset: float,
) -> None:
    actual = _nearest_valid_ts_time_index(time_thomson_scattering, time_to_reconstruct, max_time_offset)

    assert actual is None
