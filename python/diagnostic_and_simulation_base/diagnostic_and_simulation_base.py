import json
import logging
import sys
import textwrap
import typing
from importlib import resources as resources_py  # BUXTON: this is deprecated and needs removing
from pathlib import Path
from time import time as time_py

import f90nml  # type: ignore

from . import version_storage
from .nested_dictionary import NestedDict
from .utility import logging as util_logging
from .utility.make_settings_json import make_settings_json


class DiagnosticAndSimulationBase:
    """Class for creating Diagnostic Analysis codes, such as: `Gas`, `Efit`, `Ppts`, `Zeff_pi`"""

    log_records: list[logging.LogRecord] = []

    @property
    def log_string(self) -> str:
        return util_logging.format_logs(self.log_records)

    def __init__(
        self,
        pulseNo: int,
        run_name: str,
        run_description: str = "Standard run with default settings",
        settings_path: str = "default",
        write_to_mds: bool = True,
        pulseNo_write: int | None = None,
        analysis_name: str | None = None,
        link_run_to_best: bool = False,
    ) -> None:
        """Class constructor

        :param pulseNo: pulse number
        :param run_name: run_name to save to MDSplus
        :param run_description: help string for MDSplus Tree
        :param settings_path: location where code inputs are stored
        :param write_to_mds: flag to turn on / off writing to MDSplus
        :param pulseNo_write: pulse number in which data is to be written, if different from the current pulse
        :param analysis_name: for example "EFITP" is analysis using the "EFIT code"
        :param link_run_to_best: BEST will be linked to the run
        """
        self.analysis_time_start = time_py()

        # code_name: is the package name from "pip list" in upper case (note, this includes "_")
        code_name = self.__class__.__module__.split(".")[0].upper()  # BUXTON: I don't like this!!

        # analysis_name
        if analysis_name is None:
            analysis_name = code_name

        # Create a logger with the module name
        # Get the logging_level
        logging_level = util_logging.logger.getEffectiveLevel()

        # Create logger
        self.logger = logging.getLogger(analysis_name)
        # Don't propagate messages to the root logger; `mdsthin` uses the root logger and this avoids double logging
        self.logger.propagate = False
        # Set the logging level
        self.logger.setLevel(level=logging_level)

        # Check if the logger already has handlers to avoid adding duplicates
        if not self.logger.handlers:
            # Create and add handler to emit the logs to "stadnard out",
            # i.e. printing to the terminal (standard behaviour)
            handler_to_standard_out = logging.StreamHandler(sys.stdout)
            handler_to_standard_out.setFormatter(util_logging.CustomFormatter())
            self.logger.addHandler(handler_to_standard_out)

            # Create and add handler to emit the log "records" into `self.log_records`
            handler_to_list = util_logging.EmitLogToListHandler(self.log_records)
            self.logger.addHandler(handler_to_list)

        # Store inputs
        self.pulseNo = pulseNo
        self.code_name = code_name
        self.analysis_name = analysis_name
        self.run_name = run_name.upper()
        self.run_description = run_description
        self.write_to_mds = write_to_mds
        self.link_run_to_best = link_run_to_best

        # Determine which pulseNo to write to and store in class object
        if pulseNo_write is None:
            pulseNo_write = pulseNo
        self.pulseNo_write = pulseNo_write

        # Test if "settings_path" is a directory.
        # If directory doesn't exist, then treat as "relative path"
        if not (Path(settings_path).is_dir()):
            python_module = self.code_name.lower()
            settings_path = f"{resources_py.files(python_module)}/settings/{settings_path}"

        # Test if settings directory exists
        if not Path(settings_path).is_dir():
            raise FileNotFoundError("settings directory not found")

        # Store the resolved settings path
        self.settings_path = settings_path

        # Create empty settings dictionary
        self.settings = {}  # type: dict[str, typing.Any]

        self._load_settings_from_files()

        # Create results dictionary, with the stuff we already know
        # Important: results is a 1:1 mapping to MDSplus data-strucutre
        # TO-DO: Change this to a pre-populated dictionary from the *.csv file
        self.results = NestedDict()
        self.results["CODE_VERSION"]["COMPUTER"] = version_storage.__computer__
        self.results["CODE_VERSION"]["DATETIME"] = version_storage.__datetime__
        self.results["CODE_VERSION"]["GIT_ID"] = version_storage.__git_short_hash__
        self.results["CODE_VERSION"]["LIBRARY"] = version_storage.__python_library__
        self.results["CODE_VERSION"]["PYTHON"] = version_storage.__python__
        self.results["CODE_VERSION"]["USER"] = version_storage.__user__
        self.results["CODE_VERSION"]["VERSION"] = version_storage.__version__

    def __repr__(self) -> str:
        """Print to screen"""
        string_output = ""
        string_output += "╔═════════════════════════════════════════════════════════════════════════════╗\n"
        string_output += f"║ {f' <{self.__class__.__name__}>':<75} ║\n"
        string_output += f"║ {f' {version_storage.__version__}':<75} ║\n"
        string_output += f"║ {' ':<75} ║\n"
        string_output += f"║ {' pulseNo = ' + f'{self.pulseNo:_}':<75} ║\n"
        string_output += f"║ {' pulseNo_write = ' + f'{self.pulseNo_write:_}':<75} ║\n"
        string_output += f"║ {' run_name = ' + str(self.run_name):<75} ║\n"
        string_output += f"║ {' run_description = ' + str(self.run_description):<75} ║\n"
        string_output += f"║ {' settings_path = ...':<75} ║\n"
        wrapped_settings_path = textwrap.wrap(self.settings_path, width=75)
        for i, settings_path in enumerate(wrapped_settings_path):
            string_output += f"║  {str(settings_path):<75}║\n"
        string_output += "╚═════════════════════════════════════════════════════════════════════════════╝"
        return string_output

    def _load_settings_from_files(self) -> None:
        """Look in the "settings_path" directory and load all settings files.
        Will recursively load *.json and *.nml files

        TODO: add *.csv reader #SUNDAR - is ths even needed?
        """

        settings_path = self.settings_path

        # Load *.json settings, including sub-directories
        for file in Path(settings_path).glob("**/*.json"):
            with open(file, "r") as file_id:
                step_name = f'Loading settings from: "{file.name}"'
                try:
                    relative_path = str(file.relative_to(settings_path))
                    self.settings[relative_path] = json.load(file_id)
                    self.logger.info(msg=step_name)
                except Exception as exception_obj:
                    self.logger.exception(msg=step_name)
                    raise exception_obj

        # Load *.nml (namelist) settings, including sub-directories
        for file in Path(settings_path).glob("**/*.nml"):
            with open(file, "r") as file_id:
                step_name = f"Loading settings from {file.name}"
                try:
                    relative_path = str(file.relative_to(settings_path))
                    self.settings[relative_path] = f90nml.read(file_id)
                    self.logger.info(msg=step_name)
                except Exception as exception_obj:
                    self.logger.exception(msg=step_name)
                    raise exception_obj

    def _write_to_mds(self) -> None:
        """Writes data to MDSplus"""

        # Lazy loading of `standard_utility` because it's specific to Tokamak Energy.
        import standard_utility as util  # type: ignore

        # Add settings files to results.
        # We do this right at the end, as they can be programatially changed, e.g. for scans
        self.results["INPUT"]["SETTINGS"] = make_settings_json(data=self.settings, json_indent=2)

        # Create MDSplus nodes
        mdsplus_settings_file = f"{self.analysis_name}_mdsplus_settings.json"
        if mdsplus_settings_file in self.settings:
            pulseNo_cal = self.settings[mdsplus_settings_file].get("calibration", {}).get("pulse", None)
        else:
            pulseNo_cal = None

        try:
            workflow = list(self.results["INPUT"]["WORKFLOW"].keys())
        except KeyError:
            workflow = None

        util.create_script_nodes(
            script_name=self.analysis_name,
            pulseNo_write=self.pulseNo_write,
            pulseNo_cal=pulseNo_cal,
            run_name=self.run_name,
            run_info=self.run_description,
            workflows=workflow,
            link_best=self.link_run_to_best,
        )

        # Write to MDSplus
        util.write_script_data(
            script_name=self.analysis_name,
            pulseNo_write=self.pulseNo_write,
            data_to_write=self.results.to_dictionary(),
            pulseNo_cal=pulseNo_cal,
            run_name=self.run_name,
            run_description=self.run_description,
        )

    def run_with_log(
        self,
        operator: typing.Callable[..., typing.Any],
        message: str | None,
        args: dict[typing.Any, typing.Any] | None = None,
    ) -> None:
        if message is None:
            step_name = ""
        else:
            step_name = message

        try:
            if args is None:
                ret_data = operator()
            else:
                ret_data = operator(**args)
            self.logger.info(msg=step_name)
            return ret_data
        except Exception as exception_obj:
            self.logger.exception(msg=step_name)
            raise exception_obj
