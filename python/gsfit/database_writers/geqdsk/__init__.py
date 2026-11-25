from typing import TYPE_CHECKING

from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database

if TYPE_CHECKING:
    from ...gsfit import Gsfit

__all__ = ["DatabaseWriterGeqdsk"]


class DatabaseWriterGeqdsk(DatabaseWriterProtocol):
    """
    Writes G EQDSK files, EFIT's equilibrium file format, from GSFit's Equilibrium IDS.
    One file per time-slice, written to the current working directory.
    """

    writes_to_mdsplus: bool = False

    def map_results_to_database(self, gsfit_controller: "Gsfit") -> None:
        map_results_to_database(self, gsfit_controller)

    def get_workflow_names(self, gsfit_controller: "Gsfit") -> list[str]:
        """This writer stores no `INPUT.WORKFLOW`, so there are no workflow nodes to create"""

        return []
