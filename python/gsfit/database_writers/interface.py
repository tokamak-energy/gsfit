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
