import typing

from gsfit_rs import CoreProfiles
from gsfit_rs import Equilibrium
from gsfit_rs import Magnetics
from gsfit_rs import PfActive
from gsfit_rs import PfPassive
from gsfit_rs import PulseSchedule
from gsfit_rs import Tf
from gsfit_rs import Wall

from ..interface import DatabaseReaderProtocol
from .setup_core_profiles import setup_core_profiles
from .setup_equilibrium import setup_equilibrium
from .setup_magnetics import setup_magnetics
from .setup_pf_active import setup_pf_active
from .setup_pf_passive import setup_pf_passive
from .setup_pulse_schedule import setup_pulse_schedule
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

    def setup_magnetics(self, *args: typing.Any, **kwargs: typing.Any) -> Magnetics:
        return setup_magnetics(self, *args, **kwargs)

    def setup_tf(self, *args: typing.Any, **kwargs: typing.Any) -> Tf:
        return setup_tf(self, *args, **kwargs)

    def setup_wall(self, *args: typing.Any, **kwargs: typing.Any) -> Wall:
        return setup_wall(self, *args, **kwargs)

    def setup_pf_passive(self, *args: typing.Any, **kwargs: typing.Any) -> PfPassive:
        return setup_pf_passive(self, *args, **kwargs)

    def setup_core_profiles(self, *args: typing.Any, **kwargs: typing.Any) -> CoreProfiles:
        return setup_core_profiles(self, *args, **kwargs)

    def setup_pulse_schedule(self, *args: typing.Any, **kwargs: typing.Any) -> PulseSchedule:
        return setup_pulse_schedule(self, *args, **kwargs)
