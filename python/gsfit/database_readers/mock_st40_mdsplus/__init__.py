"""
`mock_st40_mdsplus` database reader.

Rebuilds the GSFit Rust objects from a mock of ST40's MDSplus database, frozen to a
single time-slice, instead of a live database. This lets a reconstruction run in CI
with no database, network, or `st40_database` dependency.

The build logic mirrors the `st40_mdsplus` reader, but every `GetData(...).get()` is
replaced by a lookup into the mock via `MockGetData`.

Each (tree, pulse) is mocked by a single `.npz`, mirroring how MDSplus itself stores
one tree per shot (`gsfit_66014427.tree`, `gsfit_66014427.datafile`, ...):
    `mdsplus_mock_mag_12050.npz`

Each archive member is named for the bare MDSplus node path, e.g. `BPPROBE.P101.B`,
because the tree and pulse are already carried by the filename. Capture provenance is
stored inside the same archive under `__metadata__`, so it cannot drift away from the
data it describes. The files are produced by `investigation/capture_snapshot.py`.

Which tree, pulse and run each read maps to comes from the `workflow` settings, exactly
as it does for the live `st40_mdsplus` reader:
    settings["GSFIT_code_settings.json"]["database_reader"]["mock_st40_mdsplus"]
        ["mock_dir"]   directory holding the `.npz` files
        ["workflow"]   {workflow_name: {tree_name, pulseNo, run_name, usage}}
A `pulseNo` of `None` means "use the shot's pulseNo"; the machine-description reads pin
their own fixed pulse. See `tests/test_02_delta_z_shift_greater_than_d_z` for an example.
"""

import typing

from gsfit_rs import BpProbes
from gsfit_rs import Coils
from gsfit_rs import Dialoop
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Passives
from gsfit_rs import Plasma
from gsfit_rs import Pressure
from gsfit_rs import RogowskiCoils
from gsfit_rs import StationaryPoint
from gsfit_rs import Wall

from ..interface import DatabaseReaderProtocol
from .setup_bp_probes import setup_bp_probes
from .setup_coils import setup_coils
from .setup_dialoop import setup_dialoop
from .setup_flux_loops import setup_flux_loops
from .setup_isoflux_boundary_sensors import setup_isoflux_boundary_sensors
from .setup_isoflux_sensors import setup_isoflux_sensors
from .setup_passives import setup_passives
from .setup_plasma import setup_plasma
from .setup_pressure_sensors import setup_pressure_sensors
from .setup_rogowski_coils import setup_rogowski_coils
from .setup_stationary_point_sensors import setup_stationary_point_sensors
from .setup_wall import setup_wall


class DatabaseReader(DatabaseReaderProtocol):
    """
    This class inherits from the DatabaseReaderProtocol, which defines the inputs and outputs to the class methods.
    The methods in this class are used to initialise the Rust implementations.
    Here we are binding the methods to the class.

    See `python/gsfit/database_readers/interface.py` for the interface definitions.
    """

    def setup_bp_probes(self, *args: typing.Any, **kwargs: typing.Any) -> BpProbes:
        return setup_bp_probes(self, *args, **kwargs)

    def setup_coils(self, *args: typing.Any, **kwargs: typing.Any) -> Coils:
        return setup_coils(self, *args, **kwargs)

    def setup_dialoop(self, *args: typing.Any, **kwargs: typing.Any) -> Dialoop:
        return setup_dialoop(self, *args, **kwargs)

    def setup_flux_loops(self, *args: typing.Any, **kwargs: typing.Any) -> FluxLoops:
        return setup_flux_loops(self, *args, **kwargs)

    def setup_isoflux_boundary_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> IsofluxBoundary:
        return setup_isoflux_boundary_sensors(self, *args, **kwargs)

    def setup_isoflux_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> Isoflux:
        return setup_isoflux_sensors(self, *args, **kwargs)

    def setup_stationary_point_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> StationaryPoint:
        return setup_stationary_point_sensors(self, *args, **kwargs)

    def setup_wall(self, *args: typing.Any, **kwargs: typing.Any) -> Wall:
        return setup_wall(self, *args, **kwargs)

    def setup_passives(self, *args: typing.Any, **kwargs: typing.Any) -> Passives:
        return setup_passives(self, *args, **kwargs)

    def setup_plasma(self, *args: typing.Any, **kwargs: typing.Any) -> Plasma:
        return setup_plasma(self, *args, **kwargs)

    def setup_pressure_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> Pressure:
        return setup_pressure_sensors(self, *args, **kwargs)

    def setup_rogowski_coils(self, *args: typing.Any, **kwargs: typing.Any) -> RogowskiCoils:
        return setup_rogowski_coils(self, *args, **kwargs)
