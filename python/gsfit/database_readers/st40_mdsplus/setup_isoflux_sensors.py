import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs import Isoflux

# The Thomson scattering (TS) tree reading and the isoflux pair-finding live in the
# `st40_mdsplus_with_digital_filter` reader. It is reused here so that the isoflux-sensor set-up is defined in
# a single place (it depends only on `sensor_weights_isoflux.json`, not on the digital filter), which keeps
# both ST40 readers consistent and avoids duplicated boiler-plate.
from ..st40_mdsplus_with_digital_filter.setup_isoflux_sensors import build_isoflux_pairs as build_isoflux_pairs
from ..st40_mdsplus_with_digital_filter.setup_isoflux_sensors import find_isoflux_pairs_for_slice as find_isoflux_pairs_for_slice
from ..st40_mdsplus_with_digital_filter.setup_isoflux_sensors import setup_isoflux_sensors as _setup_isoflux_sensors

if TYPE_CHECKING:
    from ..st40_mdsplus_with_digital_filter import DatabaseReader


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

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    return _setup_isoflux_sensors(self, pulseNo, settings, times_to_reconstruct)
