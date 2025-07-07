import typing
from typing import TYPE_CHECKING

import numpy as np
from gsfit_rs import Dialoop
from st40_database import GetData

if TYPE_CHECKING:
    from . import DatabaseReaderSt40MDSplus


def setup_dialoop(
    self: "DatabaseReaderSt40MDSplus",
    pulseNo: int,
    settings: dict[str, typing.Any],
) -> Dialoop:
    """
    This method initialises the Rust `Dialoop` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's experimental MDSplus database.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Dialoop Rust class
    dialoop = Dialoop()

    # TODO: implement the method

    # dialoop.add_sensor(
    #     name="DIALOOP_001",
    #     fit_settings_comment="",
    #     fit_settings_expected_value=0.0,
    #     fit_settings_include=True,
    #     fit_settings_weight=1.0,
    #     time=np.array([0.0]),
    #     measured=np.array([0.0]),
    # )

    return dialoop
