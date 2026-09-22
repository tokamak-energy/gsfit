from typing import TYPE_CHECKING

from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database

if TYPE_CHECKING:
    from imas.ids_toplevel import IDSToplevel

    from ...gsfit import Gsfit

__all__ = ["DatabaseWriterIMAS"]


class DatabaseWriterIMAS(DatabaseWriterProtocol):
    """
    Builds an IMAS `equilibrium` IDS from the GSFit results, using the official `imas-python`
    package. Nothing is written to a database: the IDS is handed back, and
    `Gsfit.write_results_to_database` keeps it on `gsfit_controller.equilibrium_ids`.
    """

    def map_results_to_database(self, gsfit_controller: "Gsfit") -> "IDSToplevel":
        return map_results_to_database(self, gsfit_controller)

    def get_workflow_names(self, gsfit_controller: "Gsfit") -> list[str]:
        """This writer stores no `INPUT.WORKFLOW`, so there are no workflow nodes to create"""

        return []
