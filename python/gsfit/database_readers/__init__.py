from .interface import DatabaseReaderProtocol


def get_database_reader(method: str) -> DatabaseReaderProtocol:
    """
    Returns a "database_reader" class for accessing different databases.

    :param method: selcts which database_reader to use. `method` is specified in the `GSFIT_code_settings.json` input file.

    See `python/gsfit/database_readers/interface.py` for the interface (methods) definitions.
    """

    if method == "st40_mdsplus":
        from .st40_mdsplus import DatabaseReaderSt40MDSplus

        return DatabaseReaderSt40MDSplus()
    elif method == "st40_astra_mdsplus":
        from .st40_astra_mdsplus import DatabaseReaderST40AstraMDSplus

        return DatabaseReaderST40AstraMDSplus()
    elif method == "freegs":
        from .freegs import DatabaseReaderFreeGS

        return DatabaseReaderFreeGS()
    elif method == "freegsnke":
        from .freegsnke import DatabaseReaderFreeGSNKE

        return DatabaseReaderFreeGSNKE()
    else:
        raise ValueError(f"Unknown database reader method: {method}")
