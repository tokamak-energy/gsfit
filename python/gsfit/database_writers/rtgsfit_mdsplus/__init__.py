from ..interface import DatabaseWriterProtocol
from .map_results_to_database import map_results_to_database


class DatabaseWriterRTGSFitMDSplus(DatabaseWriterProtocol):
    map_results_to_database = map_results_to_database
