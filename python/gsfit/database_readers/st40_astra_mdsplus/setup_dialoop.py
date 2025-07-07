import typing
from typing import TYPE_CHECKING

import numpy as np
from gsfit_rs import Dialoop
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_dialoop(
    self: "DatabaseReaderST40AstraMDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Dialoop:
    """
    This method initialises the Rust `Dialoop` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Dialoop Rust class
    dialoop = Dialoop()

    # TODO: implement the method

    return dialoop
