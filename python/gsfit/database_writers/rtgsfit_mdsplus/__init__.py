from typing import TYPE_CHECKING

from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database

if TYPE_CHECKING:
    from ...gsfit import Gsfit


class DatabaseWriterRTGSFitMDSplus(DatabaseWriterProtocol):
    map_results_to_database = map_results_to_database

    def get_workflow_names(self, gsfit_controller: "Gsfit") -> list[str]:
        """This writer does not store `INPUT.WORKFLOW`, so there are no workflow nodes to create"""

        return []
