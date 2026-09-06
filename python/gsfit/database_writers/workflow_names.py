import typing


def workflow_names_from_settings(settings: dict[str, typing.Any]) -> list[str]:
    """
    The names of the input codes ("workflow") which the selected `database_reader` reads from,
    for example `["elmag", "mag", "psu2coil"]`.

    These are the names which `map_results_to_database` stores under `INPUT.WORKFLOW`, and which
    the MDSplus `INPUT.WORKFLOW` nodes have to be created for. They come from the settings alone,
    so they are known before GSFit has read a single byte from the database.

    :param settings: Dictionary containing the JSON settings read from the `settings` directory
    """

    database_reader_method = settings["GSFIT_code_settings.json"]["database_reader"]["method"]
    database_reader_settings = settings["GSFIT_code_settings.json"]["database_reader"][database_reader_method]

    # Not every database_reader reads from other codes, e.g. `freegs` and `freegsnke` do not
    return list(database_reader_settings.get("workflow", {}).keys())
