from typing import TYPE_CHECKING

from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database

if TYPE_CHECKING:
    from ...gsfit import Gsfit


class DatabaseWriterTokamakEnergyMDSplusNew(DatabaseWriterProtocol):
    def map_results_to_database(self, gsfit_controller: "Gsfit") -> None:
        return map_results_to_database(self, gsfit_controller)
