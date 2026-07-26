from .interface import DatabaseReaderProtocol


def get_database_reader(method: str) -> DatabaseReaderProtocol:
    """
    Returns a "database_reader" class for accessing different databases.

    :param method: selcts which database_reader to use. `method` is specified in the `GSFIT_code_settings.json` input file.

    See `python/gsfit/database_readers/interface.py` for the interface (methods) definitions.
    """

    match method:
        case "st40_mdsplus":
            from .st40_mdsplus import DatabaseReader as St40MdsplusDatabaseReader

            return St40MdsplusDatabaseReader()
        case "st40_mdsplus_with_digital_filter":
            from .st40_mdsplus_with_digital_filter import DatabaseReader as St40MdsplusWithDigitalFilterDatabaseReader

            return St40MdsplusWithDigitalFilterDatabaseReader()
        case "st40_astra_mdsplus":
            from .st40_astra_mdsplus import DatabaseReader as St40AstraMdsplusDatabaseReader

            return St40AstraMdsplusDatabaseReader()
        case "st40_spider_mdsplus":
            from .st40_spider_mdsplus import DatabaseReader as St40SpiderMdsplusDatabaseReader

            return St40SpiderMdsplusDatabaseReader()
        case "freegs":
            from .freegs import DatabaseReader as FreegsDatabaseReader

            return FreegsDatabaseReader()
        case "freegsnke":
            from .freegsnke import DatabaseReader as FreegsNkeDatabaseReader

            return FreegsNkeDatabaseReader()
        case _:
            raise ValueError(f"Unknown database reader method: {method}")
