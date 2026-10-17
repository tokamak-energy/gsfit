import typing

from gsfit_rs import Equilibrium
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Magnetics
from gsfit_rs import PfActive
from gsfit_rs import PfPassive
from gsfit_rs import Pressure
from gsfit_rs import PulseSchedule
from gsfit_rs import StationaryPoint
from gsfit_rs import Tf
from gsfit_rs import Wall

from ..interface import DatabaseReaderProtocol
from .setup_equilibrium import setup_equilibrium
from .setup_isoflux_boundary_sensors import setup_isoflux_boundary_sensors
from .setup_isoflux_sensors import setup_isoflux_sensors
from .setup_magnetics import setup_magnetics
from .setup_pf_active import setup_pf_active
from .setup_pf_passive import setup_pf_passive
from .setup_pressure_sensors import setup_pressure_sensors
from .setup_pulse_schedule import setup_pulse_schedule
from .setup_stationary_point_sensors import setup_stationary_point_sensors
from .setup_tf import setup_tf
from .setup_wall import setup_wall


class DatabaseReader(DatabaseReaderProtocol):
    """
    This class inherits from the DatabaseReaderProtocol, which defines the inputs and outputs to the class methods.
    The methods in this class are used to initialise the Rust implementations.
    Here we are binding the methods to the class.

    See `python/gsfit/database_readers/interface.py` for the interface definitions.
    """

    def setup_pf_active(self, *args: typing.Any, **kwargs: typing.Any) -> PfActive:
        return setup_pf_active(self, *args, **kwargs)

    def setup_equilibrium(self, *args: typing.Any, **kwargs: typing.Any) -> Equilibrium:
        return setup_equilibrium(self, *args, **kwargs)

    def setup_isoflux_boundary_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> IsofluxBoundary:
        return setup_isoflux_boundary_sensors(self, *args, **kwargs)

    def setup_isoflux_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> Isoflux:
        return setup_isoflux_sensors(self, *args, **kwargs)

    def setup_magnetics(self, *args: typing.Any, **kwargs: typing.Any) -> Magnetics:
        return setup_magnetics(self, *args, **kwargs)

    def setup_stationary_point_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> StationaryPoint:
        return setup_stationary_point_sensors(self, *args, **kwargs)

    def setup_tf(self, *args: typing.Any, **kwargs: typing.Any) -> Tf:
        return setup_tf(self, *args, **kwargs)

    def setup_wall(self, *args: typing.Any, **kwargs: typing.Any) -> Wall:
        return setup_wall(self, *args, **kwargs)

    def setup_pf_passive(self, *args: typing.Any, **kwargs: typing.Any) -> PfPassive:
        return setup_pf_passive(self, *args, **kwargs)

    def setup_pressure_sensors(self, *args: typing.Any, **kwargs: typing.Any) -> Pressure:
        return setup_pressure_sensors(self, *args, **kwargs)

    def setup_pulse_schedule(self, *args: typing.Any, **kwargs: typing.Any) -> PulseSchedule:
        return setup_pulse_schedule(self, *args, **kwargs)
