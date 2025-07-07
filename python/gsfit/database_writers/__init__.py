from .interface import DatabaseWriterProtocol


def get_database_writer(method: str) -> DatabaseWriterProtocol:
    """
    Returns a `DatabaseWriter` instance for accessing different databases.

    :param method: selects which DatabaseWriter to use. `method` is specified in the `GSFIT_code_settings.json` input file.

    Lazy imports of the DatabaseWriter*** classes is deliberate.
    This allows GSFit to function without having every database_writer module installed.
    This is useful because not every institution will make their database interface public.

    Typically, the `gsfit_controller` object is passed into the `DatabaseWriter` class.
    `gsfit_controller.results` is intended to be a 1:1 mapping of nested dictionaries to the database structure
    (in Tokamak Energy's case this is MDSplus).
    """

    if method == "tokamak_energy_mdsplus":
        from .tokamak_energy_mdsplus import DatabaseWriterTokamakEnergyMDSplus

        return DatabaseWriterTokamakEnergyMDSplus()
    elif method == "rtgsfit_mdsplus":
        from .rtgsfit_mdsplus import DatabaseWriterRTGSFitMDSplus

        return DatabaseWriterRTGSFitMDSplus()
    else:
        raise ValueError(f"Unknown database writer method: {method}")
