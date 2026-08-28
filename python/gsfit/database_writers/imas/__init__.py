from typing import TYPE_CHECKING

from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database

if TYPE_CHECKING:
    from imas.ids_toplevel import IDSToplevel

    from ...gsfit import Gsfit


class DatabaseWriterIMAS(DatabaseWriterProtocol):
    def map_results_to_database(self, gsfit_controller: "Gsfit") -> "IDSToplevel":
        return map_results_to_database(self, gsfit_controller)
