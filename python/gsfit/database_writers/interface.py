from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from ..gsfit import Gsfit


class DatabaseWriterProtocol(Protocol):
    def map_results_to_database(self, gsfit_controller: "Gsfit") -> Any:
        """
        Map the results to MDSplus structure.
        self.results is a NestedDict type which has a 1:1 mapping to the MDSplus tree.

        The MDSplus writers fill `gsfit_controller.results` in place and return `None`. A writer
        which builds an object instead returns it, and `Gsfit.write_results_to_database` keeps it:
        the `imas` writer returns a populated IMAS `equilibrium` IDS.
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
