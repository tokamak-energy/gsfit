from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from ..gsfit import Gsfit


class DatabaseWriterProtocol(Protocol):
    def map_results_to_database(self, gsfit_controller: "Gsfit") -> None:
        """
        Map the results to MDSplus structure.
        self.results is a NestedDict type which has a 1:1 mapping to the MDSplus tree.
        """
        ...

    def get_workflow_names(self, gsfit_controller: "Gsfit") -> list[str]:
        """
        The names of the input codes which this writer will store under `INPUT.WORKFLOW`,
        or an empty list if it stores none.

        GSFit needs these before it has any results, so that the MDSplus nodes can be created
        while the Grad-Shafranov equation is still being solved.
        """
        ...
