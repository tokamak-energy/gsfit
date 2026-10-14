"""Tests for selecting Thomson-scattering time-slices for isoflux pairing."""

import importlib

import numpy as np
import numpy.typing as npt
import pytest

from gsfit.database_readers.st40_mdsplus_with_digital_filter.setup_isoflux_sensors import _nearest_valid_ts_time_index

isoflux_reader = importlib.import_module("gsfit.database_readers.st40_mdsplus_with_digital_filter.setup_isoflux_sensors")


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


def test_database_reader_wires_selected_profiles_to_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNode:
        def __init__(self, data: npt.NDArray[np.float64]) -> None:
            self._data = data

        def data(self) -> npt.NDArray[np.float64]:
            return self._data

    class FakeConnection:
        def __init__(self, host: str) -> None:
            assert host == "smaug"
            self.nodes = {
                "\\TS::TOP.BEST:R": np.array([0.3, 0.4, 0.6, 0.7]),
                "\\TS::TOP.BEST:Z": np.array([0.01, 0.02, 0.03, 0.04]),
                "\\TS::TOP.BEST:TIME": np.array([0.99, np.nan, 1.03, np.inf]),
                "\\TS::TOP.BEST.PROFILES:NE": np.array(
                    [
                        [10.0, 11.0, 12.0, 13.0],
                        [20.0, 21.0, 22.0, 23.0],
                        [30.0, 31.0, 32.0, 33.0],
                        [40.0, 41.0, 42.0, 43.0],
                    ]
                ),
                "\\TS::TOP.BEST.PROFILES:NE_ERR": np.array(
                    [
                        [1.0, 1.1, 1.2, 1.3],
                        [2.0, 2.1, 2.2, 2.3],
                        [3.0, 3.1, 3.2, 3.3],
                        [4.0, 4.1, 4.2, 4.3],
                    ]
                ),
            }
            self.opened_tree: tuple[str, int] | None = None

        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def openTree(self, tree: str, pulse_number: int) -> None:
            self.opened_tree = (tree, pulse_number)

        def get(self, path: str) -> FakeNode:
            return FakeNode(self.nodes[path])

    class FakeIsoflux:
        def __init__(self) -> None:
            self.sensors: list[dict[str, object]] = []

        def add_sensor(self, **kwargs: object) -> None:
            self.sensors.append(kwargs)

    connection_instances: list[FakeConnection] = []
    fake_sensors: list[FakeIsoflux] = []
    finder_calls: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []

    def make_connection(host: str) -> FakeConnection:
        connection = FakeConnection(host)
        connection_instances.append(connection)
        return connection

    def make_isoflux() -> FakeIsoflux:
        sensors = FakeIsoflux()
        fake_sensors.append(sensors)
        return sensors

    def find_pairs(**kwargs: object) -> npt.NDArray[np.float64]:
        profile_value = kwargs["value"]
        profile_error = kwargs["error"]
        assert isinstance(profile_value, np.ndarray)
        assert isinstance(profile_error, np.ndarray)
        finder_calls.append((profile_value.copy(), profile_error.copy()))

        if profile_value[0] == 10.0:
            return np.array([[0.31, 0.69], [np.nan, np.nan], [np.nan, np.nan]])
        return np.array([[0.32, 0.68], [0.42, 0.58], [np.nan, np.nan]])

    monkeypatch.setattr(isoflux_reader.mdsthin, "Connection", make_connection)
    monkeypatch.setattr(isoflux_reader, "Isoflux", make_isoflux)
    monkeypatch.setattr(isoflux_reader, "find_isoflux_pairs_for_slice", find_pairs)

    settings = {
        "GSFIT_code_settings.json": {
            "database_reader": {
                "method": "st40_mdsplus_with_digital_filter",
                "st40_mdsplus_with_digital_filter": {"workflow": {"ts": {"run_name": "BEST"}}},
            }
        },
        "sensor_weights_isoflux.json": {
            "include": True,
            "sensor_name_prefix": "ISOFLUX",
            "fit_settings": {"comment": "TS isoflux", "weight": 100.0},
            "thomson_scattering": {
                "quantity": "NE",
                "level_fractions": [0.3, 0.6, 0.9],
                "max_relative_error": 0.1,
                "max_gap": 0.08,
                "min_points_per_branch": 2,
                "max_time_offset": 0.05,
                "limits_r": [0.25, 0.8],
            },
        },
    }
    times_to_reconstruct = np.array([1.0, 1.03, 1.5])

    actual = isoflux_reader.setup_isoflux_sensors(None, 12345, settings, times_to_reconstruct)

    assert actual is fake_sensors[0]
    assert connection_instances[0].opened_tree == ("TS", 12345)
    assert len(finder_calls) == 2
    np.testing.assert_array_equal(finder_calls[0][0], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(finder_calls[0][1], [1.0, 1.1, 1.2, 1.3])
    np.testing.assert_array_equal(finder_calls[1][0], [30.0, 31.0, 32.0, 33.0])
    np.testing.assert_array_equal(finder_calls[1][1], [3.0, 3.1, 3.2, 3.3])

    sensors = fake_sensors[0].sensors
    assert [sensor["name"] for sensor in sensors] == ["ISOFLUX01", "ISOFLUX02"]
    np.testing.assert_array_equal(sensors[0]["location_1_r"], [0.31, 0.32, np.nan])
    np.testing.assert_array_equal(sensors[0]["location_2_r"], [0.69, 0.68, np.nan])
    np.testing.assert_array_equal(sensors[1]["location_1_r"], [np.nan, 0.42, np.nan])
    np.testing.assert_array_equal(sensors[1]["location_2_r"], [np.nan, 0.58, np.nan])
    np.testing.assert_array_equal(sensors[0]["times_to_reconstruct"], times_to_reconstruct)
    assert sensors[0]["fit_settings_comment"] == "TS isoflux; TS NE = 0.30 x peak"
    assert sensors[1]["fit_settings_comment"] == "TS isoflux; TS NE = 0.60 x peak"
