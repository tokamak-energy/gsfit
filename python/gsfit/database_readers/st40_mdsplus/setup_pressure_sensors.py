import typing
from typing import TYPE_CHECKING

from gsfit_rs import Pressure

# The Thomson scattering (TS) tree reading and the PPTS `BAD_MA` "good time-slice" logic live in the
# `st40_mdsplus_with_digital_filter` reader. It is reused here so that the pressure-sensor set-up is
# defined in a single place (it depends only on `sensor_weights_pressure.json`, not on the digital
# filter), which keeps both ST40 readers consistent and avoids duplicated boiler-plate.
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import get_good_pressure_sensor_times as get_good_pressure_sensor_times
from ..st40_mdsplus_with_digital_filter.setup_pressure_sensors import setup_pressure_sensors as _setup_pressure_sensors

if TYPE_CHECKING:
    from ..st40_mdsplus_with_digital_filter import DatabaseReader


def setup_pressure_sensors(
    self: "DatabaseReader",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Pressure:
    """
    This method initialises the Rust `Pressure` class using ST40's Thomson scattering (TS) data.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    return _setup_pressure_sensors(self, pulseNo, settings)
