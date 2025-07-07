import typing

from gsfit_rs import BpProbes
from gsfit_rs import Coils
from gsfit_rs import Dialoop
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Passives
from gsfit_rs import Plasma
from gsfit_rs import RogowskiCoils

from ..interface import DatabaseReaderProtocol
from .setup_bp_probes import setup_bp_probes
from .setup_coils import setup_coils
from .setup_dialoop import setup_dialoop
from .setup_flux_loops import setup_flux_loops
from .setup_isoflux_boundary_sensors import setup_isoflux_boundary_sensors
from .setup_isoflux_sensors import setup_isoflux_sensors
from .setup_passives import setup_passives
from .setup_plasma import setup_plasma
from .setup_rogowski_coils import setup_rogowski_coils


class DatabaseReaderSt40MDSplus(DatabaseReaderProtocol):
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

    def setup_passives(self, *args: typing.Any, **kwargs: typing.Any) -> Passives:
        return setup_passives(self, *args, **kwargs)

    def setup_plasma(self, *args: typing.Any, **kwargs: typing.Any) -> Plasma:
        return setup_plasma(self, *args, **kwargs)

    def setup_rogowski_coils(self, *args: typing.Any, **kwargs: typing.Any) -> RogowskiCoils:
        return setup_rogowski_coils(self, *args, **kwargs)
